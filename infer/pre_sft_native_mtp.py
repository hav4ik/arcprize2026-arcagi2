"""Native Qwen3.5 MTP artifact and training support (imports torch lazily).

The original predictor is one full-attention decoder, not independent linear
heads. It consumes [norm(embed(x[t+1])), norm(h[t])] and predicts x[t+2].
Embedding and output weights are shared with the backbone, never duplicated.
Reference: vLLM model_executor/models/qwen3_5_mtp.py.
"""

from collections import defaultdict
import json
from pathlib import Path
import struct

NATIVE_TYPE = "qwen35_native"


def native_mtp_config(loss_weight=0.3, *, num_predictions=2):
    import math
    if type(num_predictions) is not int or not 2 <= num_predictions <= 16:
        raise ValueError("Native --mtp-num-predictions must be between 2 and 16")
    if not math.isfinite(loss_weight) or loss_weight < 0:
        raise ValueError("Native MTP loss weight must be finite and non-negative")
    contract = {
        "version": 1 if num_predictions == 2 else 2, "type": NATIVE_TYPE,
        "num_predictions": num_predictions,
        "num_hidden_layers": 1, "loss_weight": float(loss_weight),
        "mask": "same_document_contiguous_supervised_targets_no_eos_crossing",
        "initialization": "original_qwen35_mtp", "eval_loss": "next_token_only",
        "weight_storage": "hf_state_dict", "shared_embedding_and_head": True,
    }
    if num_predictions > 2:
        contract.update(unroll="shared_native_teacher_forcing", detach_recurrent_hidden=False,
                        auxiliary_reduction="mean_per_depth_ce")
    return contract


def hf_tensor_inventory(path):
    """Read only safetensors headers; safe before torch/CUDA initialization."""
    root = Path(path)
    index = root / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    filenames = sorted(set(mapping.values())) if mapping else ["model.safetensors"]
    result = {}
    for filename in filenames:
        file = root / filename
        if file.resolve().parent != root.resolve():
            raise ValueError(f"Unsafe model shard path: {filename}")
        with file.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 100_000_000:
                raise ValueError(f"Invalid safetensors header: {file}")
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in result or (mapping is not None and mapping.get(name) != filename):
                raise ValueError(f"Inconsistent model shard index for {name}")
            result[name] = {**entry, "file": filename}
    if mapping is not None and set(result) != set(mapping):
        raise ValueError("Model shard index has missing tensors")
    return result


def native_shapes(config):
    h, d = int(config["hidden_size"]), int(config["head_dim"])
    q, kv, intermediate = int(config["num_attention_heads"]), int(config["num_key_value_heads"]), int(config["intermediate_size"])
    return {
        "fc.weight": [h, 2 * h],
        "pre_fc_norm_embedding.weight": [h], "pre_fc_norm_hidden.weight": [h], "norm.weight": [h],
        "layers.0.input_layernorm.weight": [h], "layers.0.post_attention_layernorm.weight": [h],
        "layers.0.self_attn.q_proj.weight": [2 * q * d, h],
        "layers.0.self_attn.k_proj.weight": [kv * d, h],
        "layers.0.self_attn.v_proj.weight": [kv * d, h],
        "layers.0.self_attn.o_proj.weight": [h, q * d],
        "layers.0.self_attn.q_norm.weight": [d], "layers.0.self_attn.k_norm.weight": [d],
        "layers.0.mlp.gate_proj.weight": [intermediate, h],
        "layers.0.mlp.up_proj.weight": [intermediate, h],
        "layers.0.mlp.down_proj.weight": [h, intermediate],
    }


def validate_native_inventory(path):
    config = json.loads((Path(path) / "config.json").read_text())
    text = config.get("text_config", config)
    if text.get("mtp_num_hidden_layers") != 1 or text.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("Native MTP requires exactly one layer and shared embeddings")
    inventory = hf_tensor_inventory(path)
    mtp = {key.removeprefix("mtp."): value for key, value in inventory.items() if key.startswith("mtp.")}
    expected = native_shapes(text)
    if mtp.keys() != expected.keys() or any(mtp[k]["shape"] != shape for k, shape in expected.items()):
        raise ValueError(f"Missing or incompatible native MTP tensors: {path}")
    if any(entry["dtype"] not in {"BF16", "F32"} for entry in mtp.values()):
        raise ValueError("Native MTP weights must be BF16 or FP32")
    return mtp


def load_native_state(path):
    from safetensors import safe_open
    inventory = validate_native_inventory(path)
    by_file = defaultdict(list)
    for name, entry in inventory.items():
        by_file[entry["file"]].append(name)
    state = {}
    for filename, names in by_file.items():
        with safe_open(Path(path) / filename, framework="pt", device="cpu") as reader:
            for name in names:
                state[name] = reader.get_tensor("mtp." + name)
    return state


def build_native_module(config, *, device="meta", dtype=None):
    """Small, version-independent implementation matching native HF key layout.

    CUDA uses FlashAttention's packed causal kernel. CPU SDPA is a correctness
    reference, not a production fallback. Qwen3.5 norms are zero-centered.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    h, d = config.hidden_size, config.head_dim
    qh, kvh = config.num_attention_heads, config.num_key_value_heads
    if getattr(config, "attention_bias", False) or getattr(config, "attention_dropout", 0.0):
        raise ValueError("Native MTP currently requires bias-free, dropout-free Qwen3.5")
    dtype = dtype or torch.bfloat16

    def linear(in_features, out_features):
        return nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)

    class Norm(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(size, device=device, dtype=dtype))

        def forward(self, x):
            y = x.float()
            return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                    * (1.0 + self.weight.float())).to(x.dtype)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj = linear(h, 2 * qh * d), linear(h, kvh * d)
            self.v_proj, self.o_proj = linear(h, kvh * d), linear(qh * d, h)
            self.q_norm, self.k_norm = Norm(d), Norm(d)

        def forward(self, x, cos, sin, cu, lengths):
            n = x.shape[1]
            q, gate = self.q_proj(x).view(n, qh, 2 * d).chunk(2, dim=-1)
            q = self.q_norm(q)
            k = self.k_norm(self.k_proj(x).view(n, kvh, d))
            v = self.v_proj(x).view(n, kvh, d)
            rd = cos.shape[-1]

            def rotate(y):
                rot, tail = y[..., :rd], y[..., rd:]
                first, second = rot.chunk(2, dim=-1)
                rot = rot * cos[0, :, None] + torch.cat((-second, first), -1) * sin[0, :, None]
                return torch.cat((rot, tail), -1)

            q, k = rotate(q), rotate(k)
            if x.is_cuda:
                from flash_attn import flash_attn_varlen_func
                y = flash_attn_varlen_func(q, k, v, cu, cu, max(lengths), max(lengths),
                                          dropout_p=0.0, softmax_scale=d ** -0.5, causal=True)
            else:
                pieces, start = [], 0
                for length in lengths:
                    qq, kk, vv = [t[start:start + length].transpose(0, 1).unsqueeze(0) for t in (q, k, v)]
                    pieces.append(F.scaled_dot_product_attention(qq, kk, vv, is_causal=True,
                                  enable_gqa=True).squeeze(0).transpose(0, 1))
                    start += length
                y = torch.cat(pieces, 0)
            return self.o_proj((y * gate.sigmoid()).reshape(1, n, qh * d))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj = linear(h, config.intermediate_size), linear(h, config.intermediate_size)
            self.down_proj = linear(config.intermediate_size, h)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attention(), MLP()
            self.input_layernorm, self.post_attention_layernorm = Norm(h), Norm(h)

        def forward(self, x, cos, sin, cu, lengths):
            x = x + self.self_attn(self.input_layernorm(x), cos, sin, cu, lengths)
            return x + self.mlp(self.post_attention_layernorm(x))

    class NativeMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = linear(2 * h, h)
            self.layers = nn.ModuleList([Layer()])
            self.norm, self.pre_fc_norm_embedding, self.pre_fc_norm_hidden = Norm(h), Norm(h), Norm(h)

        def forward(self, hidden, embedding, cos, sin, cu, lengths):
            x = self.fc(torch.cat((self.pre_fc_norm_embedding(embedding), self.pre_fc_norm_hidden(hidden)), -1))
            return self.norm(self.layers[0](x, cos, sin, cu, lengths))

    return NativeMTP()


def install_native_mtp(model, config, source_path):
    import torch
    if getattr(model.config, "model_type", None) != "qwen3_5_text":
        raise ValueError("Native Qwen3.5 MTP requires a text-only Qwen3.5 model")
    if hasattr(model, "mtp") or source_path is None:
        raise ValueError("Native MTP is already attached or no pretrained source was supplied")
    weight = model.get_output_embeddings().weight
    if weight.shape[0] > 256:
        raise ValueError("Native MTP training currently requires the small ARC vocabulary")
    module = build_native_module(model.config, dtype=weight.dtype)
    state = load_native_state(source_path)
    module.load_state_dict(state, strict=True, assign=True)
    module.to(device=weight.device, dtype=weight.dtype)
    module.train(model.training)
    model.mtp = module
    model.config.arc_mtp = dict(config)
    model.config.mtp_num_hidden_layers = 1
    model.config.mtp_use_dedicated_embeddings = False
    return {"type": NATIVE_TYPE, "heads": 1, "parameters": sum(p.numel() for p in module.parameters()),
            "dtype": str(weight.dtype), "restored": True, "source": str(source_path),
            "auxiliary_offsets": list(range(2, config["num_predictions"] + 1)),
            "unroll_steps": config["num_predictions"] - 1}


def native_auxiliary_loss(model, decoder, hidden, labels, *, input_ids, packed_seq_lengths=None,
                          position_ids=None, eos_token_ids=(), checkpoint=True, num_predictions=2):
    """Recurrently reuse ONE native block for offsets 2..num_predictions.

    At depth d, pass the previous depth's hidden state (never detached) and
    the ground-truth embedding at t+d; predict t+d+1. Drop one more source
    position per document at each depth. Attention/teacher forcing never
    cross packed boundaries, and supervision cannot cross prompts or EOS.
    Each depth's CE has its own valid-token denominator; average the depths.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as checkpoint_fn
    from pre_sft_mtp import future_targets

    if type(num_predictions) is not int or not 2 <= num_predictions <= 16:
        raise ValueError("Native --mtp-num-predictions must be between 2 and 16")
    if (input_ids is None or labels.ndim != 2 or labels.shape[0] != 1
            or hidden.shape[:2] != labels.shape or input_ids.shape != labels.shape):
        raise ValueError("Native MTP requires one padding-free packed row with aligned hidden states/IDs/labels")
    lengths = (packed_seq_lengths.tolist() if torch.is_tensor(packed_seq_lengths)
               else list(packed_seq_lengths)) if packed_seq_lengths is not None else [labels.shape[1]]
    if lengths and isinstance(lengths[0], (list, tuple)):
        if len(lengths) != 1:
            raise ValueError("Native MTP expects a single packed row")
        lengths = lengths[0]
    lengths = [int(n) for n in lengths]
    if not lengths or any(n <= 0 for n in lengths) or sum(lengths) != labels.shape[1]:
        raise ValueError("Native MTP requires exact packed lengths (no untracked padding)")
    source_indices = torch.arange(labels.shape[1], device=hidden.device)
    current_hidden, current_lengths = hidden, lengths
    losses, counts = [], []
    for depth in range(1, num_predictions):
        keep, positions, shifted_lengths, start = [], [], [], 0
        for n in current_lengths:
            if n > 1:
                keep.extend(range(start, start + n - 1))
                positions.extend(range(depth, depth + n - 1))
                shifted_lengths.append(n - 1)
            start += n
        if not keep:
            # Extremely short documents: keep zero losses connected to the
            # graph, but never invent a wraparound token or mix documents.
            zero = current_hidden.sum() * 0 + sum(p.sum() * 0 for p in model.mtp.parameters())
            for _ in range(depth, num_predictions):
                losses.append(zero)
                counts.append(torch.zeros((), device=hidden.device, dtype=torch.long))
            break
        keep = torch.tensor(keep, device=hidden.device, dtype=torch.long)
        source_indices = source_indices.index_select(0, keep)
        source_hidden = current_hidden.index_select(1, keep)
        next_ids = input_ids.index_select(1, source_indices + depth)
        embeddings = model.get_input_embeddings()(next_ids)
        if position_ids is None:
            pos = torch.tensor(positions, device=hidden.device).reshape(1, -1)
        else:
            pos = position_ids.index_select(-1, source_indices + depth)
        if pos.ndim == 2:
            pos = pos.unsqueeze(0).expand(3, -1, -1)
        if pos.shape != (3, 1, source_indices.numel()):
            raise ValueError("Unexpected native MTP RoPE position shape")
        cos, sin = decoder.rotary_emb(source_hidden, pos)
        cu = torch.tensor([0, *shifted_lengths], device=hidden.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        args = (source_hidden, embeddings, cos, sin, cu, shifted_lengths)
        if checkpoint and torch.is_grad_enabled():
            current_hidden = checkpoint_fn(model.mtp, *args, use_reentrant=False)
        else:
            current_hidden = model.mtp(*args)
        targets = future_targets(labels, depth + 1, packed_seq_lengths=lengths,
                                 input_ids=input_ids, eos_token_ids=eos_token_ids)
        targets = F.pad(targets, (0, labels.shape[1] - targets.shape[1]), value=-100).index_select(1, source_indices)
        count = targets.ne(-100).sum()
        logits = model.get_output_embeddings()(current_hidden).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                               ignore_index=-100, reduction="sum") / count.clamp(min=1)
        losses.append(loss)
        counts.append(count)
        current_lengths = shifted_lengths
    return torch.stack(losses).mean(), torch.stack(losses), torch.stack(counts)
