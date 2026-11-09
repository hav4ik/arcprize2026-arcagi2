"""Experimental axial (column, row, message-region) RoPE for packed Qwen3 SFT.

Adapted conceptually from transductARC, not a drop-in reproduction: retain
Qwen's split-half rotary pairs and ARC's 16-token chat format. Attention masks,
labels and ordinary position_ids are deliberately untouched. This first
integration supports the modern, shared-rotary Qwen3 training/evaluation path;
legacy Unsloth cached-table forwards and autoregressive generation fail closed.
"""

from contextvars import ContextVar
from functools import wraps

import torch

from pre_sft_rope_3d import CONFIG_KEY, rope_3d_config


POSITION_KEY = "arc_rope_positions"


def document_positions_3d(input_ids):
    """Return causal, prefix-derived xyz positions for one complete ARC chat.

    Message m: three header tokens at (0,0,2m), grid cells at (x,y,2m+1),
    im_end at (0,0,2m+2). The next header shares that boundary region. A row
    separator occupies (row_width, row_index, 2m+1). Never use output dimensions
    or future tokens to choose a cell's position. No coordinate is clamped.
    """
    tokens = [int(token) for token in input_ids]
    positions = []
    cursor, message = 0, 0
    while cursor < len(tokens):
        role = 11 if message % 2 == 0 else 12
        if tokens[cursor:cursor + 3] != [14, role, 10]:
            raise ValueError(f"3D RoPE expected ARC16 chat header at offset {cursor}")
        positions.extend([(0, 0, 2 * message)] * 3)
        cursor += 3
        x, y, cells = 0, 0, 0
        width = None
        while cursor < len(tokens) and tokens[cursor] != 15:
            token = tokens[cursor]
            if 0 <= token <= 9:
                positions.append((x, y, 2 * message + 1))
                x += 1
                cells += 1
            elif token == 10 and x > 0:
                if width is not None and x != width:
                    raise ValueError("3D RoPE requires rectangular ARC grids")
                width = x
                positions.append((x, y, 2 * message + 1))
                x, y = 0, y + 1
            else:
                raise ValueError(f"Invalid ARC grid token {token} at offset {cursor}")
            cursor += 1
        if cursor == len(tokens) or cells == 0:
            raise ValueError("3D RoPE requires complete, nonempty ARC chat messages")
        if x and width is not None and x != width:
            raise ValueError("3D RoPE requires rectangular ARC grids")
        positions.append((0, 0, 2 * message + 2))
        message += 1
        cursor += 1
    if not positions:
        raise ValueError("3D RoPE cannot encode an empty document")
    return positions


def packed_positions_3d(input_ids, seq_lengths):
    tokens = [int(token) for token in input_ids]
    lengths = [int(length) for length in seq_lengths]
    if not lengths or min(lengths) <= 0 or sum(lengths) != len(tokens):
        raise ValueError("3D RoPE packed lengths must cover every token exactly")
    result, offset = [], 0
    for length in lengths:
        result.extend(document_positions_3d(tokens[offset:offset + length]))
        offset += length
    return result


def enable_rope_3d_collator(collator):
    """Attach xyz after native collation; keep masks and packing IDs byte-identical."""
    if getattr(collator, "_arc_rope_3d", False):
        return
    if not getattr(collator, "padding_free", False):
        raise RuntimeError("3D RoPE currently requires padding-free packed SFT")
    original = collator.torch_call

    def torch_call(examples):
        batch = original(examples)
        tokens, lengths, coordinates = [], [], []
        for example in examples:
            ids = [int(token) for token in example["input_ids"]]
            if "seq_lengths" not in example:
                raise RuntimeError("3D RoPE requires authoritative packed seq_lengths")
            document_lengths = [int(length) for length in example["seq_lengths"]]
            tokens.extend(ids)
            lengths.extend(document_lengths)
            coordinates.extend(packed_positions_3d(ids, document_lengths))
        actual = batch["input_ids"]
        if actual.ndim != 2 or actual.shape[0] != 1 or not torch.equal(
            actual, torch.tensor([tokens], device=actual.device, dtype=actual.dtype)
        ):
            raise RuntimeError("3D RoPE: padding-free collation changed token order")
        for key, expected in (
            ("packed_seq_lengths", lengths),
            ("position_ids", [i for length in lengths for i in range(length)]),
        ):
            found = batch.get(key)
            if found is None or not torch.equal(
                found.reshape(-1), torch.tensor(expected, device=found.device, dtype=found.dtype)
            ):
                raise RuntimeError(f"3D RoPE: mismatched/missing {key}")
        batch[POSITION_KEY] = torch.tensor([coordinates], dtype=torch.long, device=actual.device)
        return batch

    collator.torch_call = torch_call
    collator._arc_rope_3d = True


class ArcRotaryEmbedding3D(torch.nn.Module):
    """FP32 axial tables with Qwen's existing split-half rotation convention.

    This replaces the native rotary module, not an additional rotation. Only
    ``base`` controls these frequencies; the old model.config.rope_theta is
    inactive while this module is installed.

    Coordinates travel separately from the 1D packing/mask IDs. A scoped,
    per-forward context delivers already-built tables to the shared rotary
    call without rewriting version-specific Transformers/Unsloth model loops.
    Tables are returned as ordinary tensors and explicitly captured by decoder
    layer checkpointing; no mutable cross-batch cache is used for backward.
    """

    def __init__(self, head_dim, *, base=10000.0):
        super().__init__()
        if head_dim < 6 or head_dim % 2:
            raise ValueError("3D RoPE requires an even head_dim >= 6")
        rope_3d_config(base)  # Also validate direct (non-CLI) construction.
        pairs = head_dim // 2
        self.axis_pairs = (pairs // 3, pairs // 3, pairs - 2 * (pairs // 3))
        self.head_dim = head_dim
        self.base = float(base)
        self._tables = ContextVar(f"arc_rope_3d_{id(self)}", default=None)

    @torch.no_grad()
    def compute_cos_sin(self, xyz):
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.dtype != torch.long:
            raise ValueError("ARC xyz must be int64 [batch, sequence, 3]")
        # Do not store float buffers: Module.to(bfloat16) must never round the
        # frequency constants. Generate the tiny vectors directly in FP32.
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            angles = torch.cat([
                xyz[..., axis:axis + 1].float() * (
                    self.base ** (-torch.arange(pairs, device=xyz.device, dtype=torch.float32) / pairs)
                )
                for axis, pairs in enumerate(self.axis_pairs)
            ], dim=-1)
            angles = torch.cat((angles, angles), dim=-1)
            return angles.cos(), angles.sin()

    @torch.compiler.disable(recursive=False)
    def forward(self, x, position_ids=None, **kwargs):
        context = self._tables.get()
        if context is None:
            raise RuntimeError("3D RoPE needs the pre-SFT xyz forward adapter; generation is not supported")
        cos, sin = context[0]
        if x.shape[:2] != cos.shape[:2] or position_ids is None or position_ids.shape != cos.shape[:2]:
            raise RuntimeError("3D RoPE coordinates do not match the current packed forward")
        context[1] += 1
        # Match the native Qwen rotary API's output dtype. In particular do not
        # turn BF16 Q/K into FP32 while leaving V in BF16 in an eager eval path.
        return cos.to(device=x.device, dtype=x.dtype), sin.to(device=x.device, dtype=x.dtype)


def install_rope_3d(model, config):
    """Install after trainer/Delta-MH preparation; no parameter names change.

    Current native Qwen3 builds rotary tables once before its layer loop (and
    passes them into checkpointed layers). The older Unsloth cache-table path
    is explicitly unsupported instead of silently using the wrong positions.
    """
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("--rope-3d currently supports dense Qwen3 only, not Qwen3.5/GDN")
    backbone = model.model
    if getattr(model, "_arc_rope_3d_installed", False):
        raise RuntimeError("3D RoPE is already installed")
    if hasattr(backbone.rotary_emb, "extend_rope_embedding"):
        raise RuntimeError("3D RoPE requires the modern shared-rotary Qwen3 path, not legacy Unsloth")
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    rotary = ArcRotaryEmbedding3D(head_dim, base=config["base"])
    original = model.forward

    @wraps(original)
    @torch.compiler.disable(recursive=False)
    def forward(*forward_args, arc_rope_positions=None, **kwargs):
        if arc_rope_positions is None:
            raise RuntimeError("3D RoPE requires arc_rope_positions; use the 3D pre-SFT collator")
        if kwargs.get("use_cache", getattr(model.config, "use_cache", False)) or any(
            kwargs.get(key) is not None for key in ("past_key_values", "past_key_value")
        ):
            raise RuntimeError("3D RoPE supports full SFT/evaluation forwards only; KV decoding is not implemented")
        context = [rotary.compute_cos_sin(arc_rope_positions), 0]
        token = rotary._tables.set(context)
        try:
            result = original(*forward_args, **kwargs)
            if context[1] == 0:
                raise RuntimeError("The model bypassed its shared 3D rotary module")
            return result
        finally:
            rotary._tables.reset(token)

    backbone.rotary_emb = rotary
    model.forward = forward
    model._arc_rope_3d_installed = True
    setattr(model.config, CONFIG_KEY, dict(config))
    return rotary
