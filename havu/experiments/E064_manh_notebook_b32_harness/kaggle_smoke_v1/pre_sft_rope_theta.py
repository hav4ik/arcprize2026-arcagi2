"""Opt-in native 1D Qwen3 RoPE theta; no changes to positions or attention.

Configuration helpers remain torch-free until called after rank isolation.
Save the standard HF fields, so ordinary inference also reloads this theta.
"""

import copy
import json
import math
from pathlib import Path


class UnsupportedNativeRoPE(ValueError):
    """An existing non-default RoPE path which this override does not handle."""


def validate_rope_theta(theta):
    if isinstance(theta, bool) or not math.isfinite(float(theta)) or float(theta) <= 1:
        raise ValueError("--rope-theta must be finite and greater than 1")
    return float(theta)


def native_rope_settings(config):
    """Read legacy Transformers 4.x or flat Transformers 5.x RoPE fields."""
    parameters = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if not isinstance(parameters, dict) or any(isinstance(v, dict) for v in parameters.values()):
        raise UnsupportedNativeRoPE("--rope-theta requires a flat native Qwen3 RoPE configuration")
    rope_type = parameters.get("rope_type", parameters.get("type", "default"))
    if rope_type != "default" or parameters.get("partial_rotary_factor", 1.0) != 1.0:
        raise UnsupportedNativeRoPE("--rope-theta currently requires default, full-dimension 1D RoPE")
    theta = parameters.get("rope_theta", config.get("rope_theta"))
    if theta is None:
        return None
    theta = validate_rope_theta(theta)
    legacy_theta = config.get("rope_theta")
    if legacy_theta is not None and theta != validate_rope_theta(legacy_theta):
        raise ValueError("Conflicting rope_theta and rope_parameters in model config")
    return theta


def resolve_rope_theta(model_path, resume_path=None, *, theta=None, rope_3d=False, golden_gate=False):
    """None leaves the native loader untouched; resume restores saved theta.

    A changed theta is an architecture change, not an exact optimizer/data
    continuation. Use a checkpoint as model_path for a weights-only experiment.
    """
    explicitly_requested = theta is not None
    if explicitly_requested:
        theta = validate_rope_theta(theta)
        if rope_3d or golden_gate:
            raise ValueError("--rope-theta cannot be combined with 3D or Golden-Gate RoPE")
    if rope_3d or golden_gate or (theta is None and resume_path is None):
        return None
    source = json.loads((Path(model_path) / "config.json").read_text())
    if source.get("model_type") != "qwen3":
        if theta is not None:
            raise ValueError("--rope-theta currently supports dense Qwen3 only")
        return None
    saved = json.loads((Path(resume_path) / "config.json").read_text()) if resume_path else source
    if saved.get("arc_rope_3d") is not None:
        raise ValueError("Native --rope-theta cannot replace saved 3D RoPE")
    try:
        source_theta = native_rope_settings(source)
        saved_theta = native_rope_settings(saved)
    except UnsupportedNativeRoPE:
        if theta is not None:
            raise
        # Existing scaled/native paths without this flag are not modified.
        return None
    if resume_path:
        if theta is not None and theta != saved_theta:
            raise ValueError(
                "Cannot change --rope-theta during full checkpoint resume. Use "
                "the checkpoint as --model-path without --resume-from-checkpoint "
                "to start a weights-only architecture experiment."
            )
        theta = saved_theta if theta is None else theta
        if theta is None or (not explicitly_requested and theta == source_theta):
            return None
    return theta


def set_config_rope_theta(config, theta):
    """Synchronize old/new HF fields without changing context length/scaling."""
    theta = validate_rope_theta(theta)
    if config.model_type != "qwen3" or getattr(config, "arc_rope_3d", None) is not None:
        raise ValueError("--rope-theta requires native dense Qwen3, not an axial model")
    native_rope_settings(config.to_dict())
    config.rope_theta = theta  # Transformers 4.x / older Unsloth.
    if getattr(config, "rope_parameters", None) is not None:
        config.rope_parameters = {**config.rope_parameters, "rope_theta": theta}
    if isinstance(getattr(config, "rope_scaling", None), dict):
        config.rope_scaling = {**config.rope_scaling, "rope_theta": theta}
    return config


def rope_theta_model_load_kwargs(model_path, theta):
    if theta is None:
        return {}
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return {"config": set_config_rope_theta(config, theta)}


def audit_native_rope_theta(model, theta):
    """Check real runtime frequencies, not just the serialized config value."""
    import torch

    theta = validate_rope_theta(theta)
    if native_rope_settings(model.config.to_dict()) != theta:
        raise RuntimeError("The model loader did not retain the requested native RoPE theta")
    rotary = getattr(model.model, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    if inv_freq is None or hasattr(rotary, "extend_rope_embedding"):
        raise RuntimeError("--rope-theta requires modern shared Qwen3 rotary with auditable inv_freq")
    dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    expected = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=inv_freq.device) / dim)
    if inv_freq.dtype != torch.float32 or inv_freq.shape != expected.shape:
        raise RuntimeError("Native RoPE frequency shape/dtype does not match dense Qwen3")
    torch.testing.assert_close(inv_freq, expected, rtol=1e-6, atol=1e-9)
    original = getattr(rotary, "original_inv_freq", None)
    if original is not None:
        torch.testing.assert_close(original, expected, rtol=1e-6, atol=1e-9)
    return {"theta": theta, "rotary_class": type(rotary).__name__, "frequency_pairs": inv_freq.numel()}


def reset_native_rope_theta_for_probe(model, theta):
    """Evaluation-only sweep helper: rebuild native buffers, keep all weights.

    Training applies config BEFORE loading instead. Do not hot-swap theta in
    the middle of an optimizer run, or reuse an existing generation KV cache.
    """
    rotary = model.model.rotary_emb
    if not hasattr(rotary, "inv_freq") or hasattr(rotary, "extend_rope_embedding"):
        raise RuntimeError("Theta sweep requires modern native shared Qwen3 rotary")
    config = set_config_rope_theta(copy.deepcopy(model.config), theta)
    replacement = type(rotary)(config=config).to(device=rotary.inv_freq.device)
    if hasattr(replacement, "original_inv_freq"):
        # Some older HF releases store this as a plain attribute, not a
        # registered buffer, so Module.to(device) does not migrate it.
        replacement.original_inv_freq = replacement.inv_freq.clone()
    replacement.train(rotary.training)
    # Maintain one authoritative config reference for the model/backbone.
    model.config = model.model.config = config
    model.model.rotary_emb = replacement
    return audit_native_rope_theta(model, theta)
