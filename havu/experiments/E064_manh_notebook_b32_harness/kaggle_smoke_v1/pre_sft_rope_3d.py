"""CPU-only configuration/checkpoint contract for opt-in ARC axial RoPE.

Keep torch imports out of this module: pre_sft isolates CUDA devices before
importing its training runtime. The actual rotary module lives in arc_rope_3d.
"""

import json
import math
from pathlib import Path


SIDECAR = "arc_rope_3d.json"
CONFIG_KEY = "arc_rope_3d"


def rope_3d_config(base=10000.0):
    if not math.isfinite(base) or base <= 1:
        raise ValueError("--rope-3d-base must be finite and greater than 1")
    return {
        "version": 1,
        "type": "arc_axial_3d",
        "base": float(base),
        "axes": ["column", "row", "message_region"],
        "pair_partition": "thirds_remainder_z",
        "pair_layout": "qwen_split_half",
        "positions": "arc16_chat_regions_v1",
        "reset_per_document": True,
    }


def read_rope_3d_config(path):
    """Read either portable config metadata or the redundant checkpoint sidecar."""
    root = Path(path)
    model_config = root / "config.json"
    embedded = None
    if model_config.is_file():
        embedded = json.loads(model_config.read_text()).get(CONFIG_KEY)
    sidecar_path = root / SIDECAR
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else None
    if embedded is not None and sidecar is not None and embedded != sidecar:
        raise ValueError(f"3D RoPE config/sidecar disagree in {root}")
    config = embedded if embedded is not None else sidecar
    if config is not None:
        if not isinstance(config, dict) or config != rope_3d_config(config.get("base", 10000.0)):
            raise ValueError(f"Unsupported or incomplete 3D RoPE metadata in {root}")
    return config


def resolve_rope_3d_config(model_path, resume_path=None, *, enabled=None, base=None):
    """Auto-restore saved architecture; never silently change it on full resume.

    A 1D model can be a weights-only starting point for a NEW 3D experiment.
    Full optimizer/data-state resume must use the exact saved RoPE contract.
    """
    model_config = read_rope_3d_config(model_path)
    resume_config = read_rope_3d_config(resume_path) if resume_path else None
    if resume_path and model_config is not None and model_config != resume_config:
        raise ValueError("Model-path and resume checkpoint have incompatible 3D RoPE settings")
    saved = resume_config if resume_path else model_config
    if enabled is False and saved is not None:
        raise ValueError("Cannot disable 3D RoPE on a checkpoint trained with it")
    use_3d = bool(saved) if enabled is None else enabled
    if base is not None and not use_3d:
        raise ValueError("--rope-3d-base requires --rope-3d or a saved 3D RoPE model")
    requested = rope_3d_config(
        base if base is not None else (saved["base"] if saved else 10000.0)
    ) if use_3d else None
    if saved is not None and requested != saved:
        raise ValueError("Requested 3D RoPE settings differ from the saved model")
    if resume_path and requested != resume_config:
        raise ValueError(
            "Cannot change 1D/3D RoPE during full checkpoint resume. For a new "
            "architecture experiment, use the checkpoint as --model-path, "
            "omit --resume-from-checkpoint, and use a new output directory."
        )
    return requested


def write_rope_3d_sidecar(path, config):
    if config is None:
        return
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    target = root / SIDECAR
    temporary = root / (SIDECAR + ".tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)

