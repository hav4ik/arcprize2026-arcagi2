"""PiSSA initialization lifecycle for independent BF16 ARC task adapters.

The retained base is W_res, not W_original. Reusing it must attach a plain
adapter and restore A0/B0, never run another SVD on W_res. Only immutable CPU
factors are retained; no old wrapper, optimizer, or GPU tensor is kept alive.
Serving either converts (B @ A - B0 @ A0) against the original checkpoint,
or loads the exact shared W_res artifact and serves the rank-r B @ A directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import uuid

import torch

from run_config import is_pissa_init


_RESIDUAL_MARKER = "_arc_pissa_residual_generation"


def prepare_pissa_targets(peft_params):
    """Use full tiny-vocabulary training, never SVD on the 16-row head."""
    params = dict(peft_params)
    targets = params.get("target_modules")
    if not isinstance(targets, (list, tuple, set)):
        raise ValueError("ARC PiSSA requires an explicit projection target list")
    saved = list(params.get("modules_to_save") or ())
    params["target_modules"] = []
    for name in targets:
        if name in {"embed_tokens", "lm_head"}:
            if name not in saved:
                saved.append(name)
        else:
            params["target_modules"].append(name)
    params["modules_to_save"] = saved or None
    return params


def validate_pissa_base(model, peft_params):
    """Check every targeted weight before PEFT can mutate any of them."""
    if getattr(model.config, "quantization_config", None):
        raise ValueError("ARC PiSSA requires an unquantized base checkpoint")
    targets = peft_params["target_modules"]
    rank = int(peft_params["r"])
    found = set()
    weights = set()
    for name, module in model.named_modules():
        matching = [t for t in targets if name == t or name.endswith('.' + t)]
        if not matching:
            continue
        weight = getattr(module, "weight", None)
        if (
            not isinstance(module, torch.nn.Linear)
            or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.dtype not in {torch.float32, torch.float16, torch.bfloat16}
            or weight.device.type == "meta"
        ):
            raise ValueError(f"PiSSA target {name!r} is not an ordinary floating-point Linear")
        if min(weight.shape) < rank:
            raise ValueError(f"PiSSA rank {rank} exceeds target {name!r} shape {tuple(weight.shape)}")
        if id(weight) in weights:
            raise ValueError(f"PiSSA cannot mutate shared projection weight at {name!r}")
        weights.add(id(weight))
        found.update(matching)
    if not weights or set(targets) != found:
        raise ValueError(f"PiSSA projection targets not found: {sorted(set(targets) - found)}")


def _projection_adapters(model, adapter_name):
    for name, module in model.named_modules():
        a = getattr(module, "lora_A", None)
        b = getattr(module, "lora_B", None)
        if a is None or b is None or adapter_name not in a:
            continue
        if adapter_name not in b:
            raise ValueError(f"PiSSA projection {name!r} is missing LoRA B")
        yield name, module, a[adapter_name].weight, b[adapter_name].weight


@dataclass
class PiSSAInitialization:
    weights: dict[str, torch.Tensor]
    scalings: dict[str, float]
    initialization: str
    residual_generation: str

    @property
    def tensor_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.weights.values())

    @classmethod
    @torch.no_grad()
    def from_shared_checkpoint(cls, model, model_path, adapter_name="default"):
        """Attach saved A0/B0 to a loaded W_res; never run SVD a second time."""
        from safetensors.torch import load_file
        from arc_pissa_shared import (
            INITIAL_ADAPTER_DIR, SHARED_CONFIG_KEY, file_sha256,
            read_shared_manifest,
        )
        from pathlib import Path

        manifest = read_shared_manifest(model_path)
        identity = manifest["base_id"]
        if getattr(model.config, SHARED_CONFIG_KEY, None) != identity:
            raise ValueError("Loaded model is not the prepared PiSSA residual base")
        config = model.peft_config[adapter_name]
        settings = manifest["settings"]
        if (
            config.init_lora_weights is not True or config.r != settings["rank"]
            or config.lora_alpha != settings["alpha"]
            or bool(config.use_rslora) != (settings["method"] == "rslora")
            or config.rank_pattern or config.alpha_pattern
        ):
            raise ValueError("Shared PiSSA attachment settings mismatch; attach a plain adapter without SVD")
        relative = f"{INITIAL_ADAPTER_DIR}/adapter_model.safetensors"
        path = Path(model_path) / relative
        if file_sha256(path) != manifest["files"][relative]["sha256"]:
            raise ValueError("Shared PiSSA initial factor checksum mismatch")
        tensors = load_file(path, device="cpu")
        scale = settings["alpha"] / (settings["rank"]**0.5 if config.use_rslora else settings["rank"])
        scalings = {f"base_model.model.{name}": scale for name in manifest["projections"]}
        snapshot = cls(tensors, scalings, settings["initialization"], identity)
        bases = []
        seen = set()
        for name, module, a, b in _projection_adapters(model, adapter_name):
            for side, target in (("A", a), ("B", b)):
                key = f"{name}.lora_{side}.weight"
                source = tensors.get(key)
                if source is None or source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
                    raise ValueError(f"Shared PiSSA factor shape/dtype/value mismatch: {key}")
                seen.add(key)
            bases.append(module.get_base_layer())
        if seen != tensors.keys():
            raise ValueError("Shared PiSSA attachment has a different projection set")
        for base in bases:
            setattr(base, _RESIDUAL_MARKER, identity)
        snapshot.restore(model, adapter_name)
        return snapshot

    @classmethod
    @torch.no_grad()
    def capture(cls, model, adapter_name="default"):
        config = model.peft_config[adapter_name]
        initialization = config.init_lora_weights
        if not is_pissa_init(initialization):
            raise ValueError("Cannot snapshot PiSSA factors from a non-PiSSA adapter")
        if config.rank_pattern or config.alpha_pattern or getattr(config, "use_dora", False):
            raise ValueError("ARC PiSSA requires uniform rank/alpha without DoRA")
        tensors, scalings, bases = {}, {}, []
        for name, module, a, b in _projection_adapters(model, adapter_name):
            if a.shape[0] != config.r or b.shape[1] != config.r:
                raise ValueError(f"PiSSA projection {name!r} has inconsistent rank")
            base = module.get_base_layer()
            if base.weight.requires_grad or base.weight.dtype not in {
                torch.float32, torch.float16, torch.bfloat16,
            }:
                raise ValueError(f"PiSSA residual base {name!r} must be frozen and unquantized")
            for side, tensor in (("A", a), ("B", b)):
                value = tensor.detach().to(device="cpu", copy=True).contiguous()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite PiSSA initialization at {name}.lora_{side}")
                tensors[f"{name}.lora_{side}.weight"] = value
            scale = float(module.scaling[adapter_name])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid PiSSA scaling at {name!r}")
            scalings[name] = scale
            bases.append(base)
        if not tensors:
            raise ValueError("No PiSSA projection factors were found")
        generation = uuid.uuid4().hex
        for base in bases:
            setattr(base, _RESIDUAL_MARKER, generation)
        return cls(tensors, scalings, initialization, generation)

    @torch.no_grad()
    def restore(self, model, adapter_name="default"):
        """Restore task-zero factors after plain PEFT reattachment to W_res."""
        config = model.peft_config[adapter_name]
        if config.init_lora_weights is not True:
            raise ValueError("PiSSA reuse must attach with init_lora_weights=True, not rerun SVD")
        copies = []
        seen = set()
        for name, module, a, b in _projection_adapters(model, adapter_name):
            base = module.get_base_layer()
            if getattr(base, _RESIDUAL_MARKER, None) != self.residual_generation:
                raise ValueError(f"PiSSA reuse does not reference its original frozen residual at {name!r}")
            if base.weight.requires_grad:
                raise ValueError(f"PiSSA residual unexpectedly trainable at {name!r}")
            if float(module.scaling[adapter_name]) != self.scalings.get(name):
                raise ValueError(f"PiSSA scaling changed at {name!r}")
            for side, target in (("A", a), ("B", b)):
                key = f"{name}.lora_{side}.weight"
                source = self.weights.get(key)
                if source is None or source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(f"PiSSA factor shape/dtype changed at {key!r}")
                copies.append((target, source))
                seen.add(key)
        if seen != self.weights.keys():
            raise ValueError("PiSSA reuse projection set does not match its initialization")
        # Validate the entire attachment before modifying any factors.
        for target, source in copies:
            target.copy_(source)
            target.grad = None
        config.init_lora_weights = self.initialization
        return len(self.scalings)
