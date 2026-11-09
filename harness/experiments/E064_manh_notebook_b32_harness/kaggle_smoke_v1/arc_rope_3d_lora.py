"""Checkpoint-aware 3D RoPE for Unsloth Qwen3 task LoRA and KV/DFS inference."""

from functools import wraps
import os

import torch

from arc_rope_3d import ArcRotaryEmbedding3D
from arc_rope_3d_positions import advance_positions
from pre_sft_rope_3d import CONFIG_KEY, read_rope_3d_config, rope_3d_config


class SharedArcRoPETables(tuple):
    """Forward-local cast cache, keyed by actual layer dtype/device."""

    def __new__(cls, tables):
        obj = super().__new__(cls, tables)
        obj._arc_cast_cache = {}
        return obj


class ArcRopeCache(tuple):
    """Legacy KV tuple plus immutable per-row coordinate cursors.

    Each DFS child gets a new cursor. Reusing a parent cache (or selecting it
    twice) never advances the parent's grid coordinates.
    """

    def __new__(cls, layers, states, sequence_length):
        obj = super().__new__(cls, layers)
        obj.arc_states = tuple(states)
        obj.arc_sequence_length = int(sequence_length)
        return obj

    def select(self, indices):
        rows = indices.detach().cpu().tolist()
        layers = tuple(
            tuple(t.index_select(0, indices.to(t.device)) for t in layer)
            for layer in self
        )
        return type(self)(layers, [self.arc_states[i] for i in rows], self.arc_sequence_length)


def install_lora_rope_3d(model, config):
    """Bind the inner causal LM before PEFT attachment; no trainable state added.

    Only the explicit-tables Unsloth Qwen3 fast path is accepted. Its regular
    and one-token forwards both consume the same axial tables. Training uses
    ordinary tensor kwargs so reentrant/smart checkpointing can replay them.
    """
    if config != rope_3d_config(config["base"]):
        raise ValueError("Unsupported ARC 3D RoPE checkpoint contract")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("ARC 3D RoPE LoRA currently supports dense Qwen3 only")
    if int(model.config.vocab_size) != 16:
        raise ValueError("ARC 3D RoPE LoRA requires the cut ARC16 tokenizer/model")
    if getattr(model, "_arc_lora_rope_3d", None) is not None:
        if model._arc_lora_rope_3d != config:
            raise ValueError("Cannot change 3D RoPE after loading the model")
        return
    rotary = ArcRotaryEmbedding3D(model.config.head_dim, base=config["base"])
    share_tables = os.environ.get("ARC_ROPE3D_SHARED_TABLES", "1") != "0"
    original = model.forward

    @wraps(original)
    @torch.compiler.disable(recursive=False)
    def forward(input_ids=None, *args, **kwargs):
        # ARC task forwards pass the remaining fields by keyword. Do not
        # silently miss a positional cache/mask with a changing HF signature.
        if args:
            raise TypeError("ARC 3D RoPE: pass cache/mask/labels by keyword")
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("ARC 3D RoPE requires input_ids [batch, sequence]")
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("ARC 3D RoPE needs actual token IDs, not inputs_embeds")
        batch, length = input_ids.shape
        past = kwargs.get("past_key_values")
        if past is None:
            states, offset = [None] * batch, 0
        else:
            if not isinstance(past, ArcRopeCache):
                raise RuntimeError("3D RoPE KV cache lost its grid-coordinate state")
            states, offset = past.arc_states, past.arc_sequence_length
            if len(states) != batch or int(past[0][0].shape[-2]) != offset:
                raise RuntimeError("3D RoPE KV cache/cursor lengths disagree")
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                kwargs["position_ids"] = torch.arange(
                    offset, offset + length, device=input_ids.device
                ).expand(batch, -1)
            elif not torch.equal(
                position_ids,
                torch.arange(offset, offset + length, device=position_ids.device).expand(batch, -1),
            ):
                raise RuntimeError("3D RoPE cached positions disagree with the KV prefix")
        coordinates, next_states = [], []
        for row, state in zip(input_ids.detach().cpu().tolist(), states):
            xyz, next_state = advance_positions(row, state, allow_padding=True)
            coordinates.append(xyz)
            next_states.append(next_state)
        xyz = torch.tensor(coordinates, dtype=torch.long, device=input_ids.device)
        if "arc_rope_cos_sin" in kwargs:
            raise ValueError("ARC 3D RoPE tables are derived from token IDs, not caller overrides")
        tables = rotary.compute_cos_sin(xyz)
        if share_tables:
            # Retain the original FP32 tables; the fork resolves the actual
            # activation dtype lazily, avoiding double rounding under autocast.
            tables = SharedArcRoPETables(tables)
        kwargs["arc_rope_cos_sin"] = tables
        result = original(input_ids, **kwargs)
        cache = getattr(result, "past_key_values", None)
        if cache is not None:
            if not isinstance(cache, (tuple, list)):
                raise RuntimeError("ARC task 3D RoPE requires Unsloth's legacy tuple KV cache")
            result.past_key_values = ArcRopeCache(cache, next_states, offset + length)
        elif not hasattr(result, "past_key_values") and kwargs.get("use_cache", False):
            raise RuntimeError("ARC 3D RoPE cached inference requires return_dict=True")
        return result

    setattr(model.config, CONFIG_KEY, dict(config))
    model.forward = forward
    model._arc_lora_rope_3d = dict(config)


def load_checkpoint_rope_3d(model, model_path):
    """No-op for ordinary checkpoints; require the matching fork when enabled."""
    config = read_rope_3d_config(model_path)
    if config is None:
        return None
    from unsloth.models import qwen3

    if not getattr(qwen3, "UNSLOTH_SUPPORTS_ARC_ROPE_3D", False):
        raise RuntimeError(
            "3D RoPE LoRA needs the updated nguyen599/unsloth fork (explicit "
            "Qwen3 ARC tables). Update Unsloth before loading this checkpoint."
        )
    install_lora_rope_3d(model, config)
    return config
