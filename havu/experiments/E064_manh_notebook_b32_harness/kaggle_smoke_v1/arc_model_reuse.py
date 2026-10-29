"""Helpers for safely reusing a frozen base model between ARC LoRA tasks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import gc
from typing import Any

import torch


_worker_gc_owner: ContextVar[dict[str, Any] | None] = ContextVar(
    "arc_worker_gc_owner", default=None,
)


@contextmanager
def worker_gc_scope():
    """Keep GC freezing local to one worker invocation, including error exits.

    In-place adapter reuse freezes the long-lived training object graph to
    speed up per-task collections. Frozen model/PEFT reference cycles survive
    both gc.collect() and CUDA empty_cache(), so thaw our pool before returning
    the GPU to an inference service. The caller can then collect the now-dead
    worker frame and release allocator blocks. This process-wide optimization
    is for the dedicated training worker, not a shared notebook interpreter.
    """
    if _worker_gc_owner.get() is not None:
        # Nested helpers share the outer lane's lease; only its exit thaws it.
        yield
        return
    owner = {"frozen": False, "rank": None}
    token = _worker_gc_owner.set(owner)
    try:
        yield
    finally:
        try:
            if owner["frozen"]:
                count = gc.get_freeze_count()
                gc.unfreeze()
                print(
                    f"[Rank {owner['rank']}] gc.unfreeze: objects={count}; "
                    "training lane ended, model cycles can be collected",
                    flush=True,
                )
        finally:
            _worker_gc_owner.reset(token)


def freeze_worker_gc(rank: int) -> bool:
    """Freeze once per dedicated training lane, paired with scoped unfreeze."""
    owner = _worker_gc_owner.get()
    if owner is None:
        raise RuntimeError("freeze_worker_gc requires worker_gc_scope")
    if owner["frozen"]:
        return False
    gc.collect()
    gc.freeze()
    owner.update(frozen=True, rank=rank)
    return True


@torch.no_grad()
def restore_fp8_base_before_unload(model: torch.nn.Module) -> tuple[int, int]:
    """Restore pristine BF16, never dequantized FP8, before PEFT reattachment.

    Frozen-FP8 conversion must have used ``preserve_bf16=True``. Validate all
    converted projections before mutating any of them: missing snapshots must
    fail rather than silently reuse rounded weights or incompatible forwards.
    The snapshots remain on CPU across tasks and are not optimizer state.

    Returns ``(restored_projection_count, pristine_cpu_bytes)``. A model whose
    native-inference transition already restored BF16 needs no further work.
    """
    converted = []
    seen = set()
    for name, module in model.named_modules():
        base_layer = getattr(module, "base_layer", module)
        if id(base_layer) in seen:
            continue
        seen.add(id(base_layer))
        weight = getattr(base_layer, "weight", None)
        if not hasattr(weight, "_unsloth_te_fp8_matmul"):
            continue
        original = getattr(base_layer, "_unsloth_bf16_base_weight_cpu", None)
        if (
            not isinstance(original, torch.Tensor)
            or original.device.type != "cpu"
            or original.dtype != torch.bfloat16
            or original.shape != weight.shape
            or weight.requires_grad
        ):
            raise RuntimeError(
                "FP8 base reuse requires an exact frozen BF16 CPU snapshot "
                f"for every converted projection; invalid/missing at {name!r}. "
                "Start the worker with ARC_REUSE_BASE_MODEL=1 so FP8 conversion "
                "preserves the original weights; do not reuse dequantized FP8."
            )
        converted.append((base_layer, original))

    if not converted:
        return 0, 0

    from unsloth.kernels.fast_lora import restore_model_lora_base_weights_bf16

    restored = restore_model_lora_base_weights_bf16(model)
    if restored < len(converted) or any(
        hasattr(layer.weight, "_unsloth_te_fp8_matmul")
        or bool(getattr(layer, "_unsloth_fp8_frozen_linear", False))
        or layer.weight.dtype != torch.bfloat16
        or layer.weight.requires_grad
        for layer, _ in converted
    ):
        raise RuntimeError(
            "FP8 base restoration did not restore every frozen BF16 projection; "
            "refusing to reattach PEFT to a partially quantized model"
        )
    return len(converted), sum(t.numel() * t.element_size() for _, t in converted)


def _active_adapter_names(module: Any) -> list[str]:
    """Return the active adapter names exposed by a PEFT auxiliary wrapper."""

    names = getattr(module, "active_adapters", None)
    if names is None:
        names = getattr(module, "active_adapter", None)
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    return [str(name) for name in names]


@torch.no_grad()
def restore_modules_to_save_before_unload(model: torch.nn.Module) -> tuple[int, int]:
    """Restore PEFT ``modules_to_save`` copies from their frozen originals.

    PEFT's ``ModulesToSaveWrapper.unload_and_optionally_merge_module`` returns
    the *active trainable copy*, even when the caller uses ``unload()`` rather
    than ``merge_and_unload()``.  Reusing that returned model would therefore
    carry task-specific full-module updates (ARC trains ``embed_tokens`` this
    way) into the next supposedly independent LoRA task.

    Copying the untouched ``original_module`` state back into each active copy
    immediately before ``unload()`` makes the object PEFT returns an exact
    frozen-base equivalent while preserving its device placement and hooks.

    Returns:
        ``(module_count, byte_count)`` for restored state tensors.
    """

    restored_modules = 0
    restored_bytes = 0
    seen: set[int] = set()

    for module_name, module in model.named_modules():
        if id(module) in seen:
            continue
        seen.add(id(module))

        modules_to_save = getattr(module, "modules_to_save", None)
        original_module = getattr(module, "original_module", None)
        if modules_to_save is None or original_module is None:
            continue

        active_names = _active_adapter_names(module)
        if len(active_names) != 1:
            raise RuntimeError(
                "Expected exactly one active PEFT modules_to_save adapter "
                f"at {module_name or '<root>'}, found {active_names!r}"
            )
        adapter_name = active_names[0]
        if adapter_name not in modules_to_save:
            raise RuntimeError(
                f"Active PEFT adapter {adapter_name!r} is absent from "
                f"modules_to_save at {module_name or '<root>'}"
            )

        active_module = modules_to_save[adapter_name]
        original_state = original_module.state_dict()
        active_state = active_module.state_dict()
        if original_state.keys() != active_state.keys():
            missing = sorted(set(original_state) - set(active_state))
            unexpected = sorted(set(active_state) - set(original_state))
            raise RuntimeError(
                "PEFT modules_to_save state does not match its frozen "
                f"original at {module_name or '<root>'}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        for state_name, source in original_state.items():
            destination = active_state[state_name]
            if source.shape != destination.shape:
                raise RuntimeError(
                    "PEFT modules_to_save tensor shape does not match its "
                    f"frozen original at {module_name}.{state_name}: "
                    f"active={tuple(destination.shape)}, "
                    f"original={tuple(source.shape)}"
                )
            destination.copy_(source, non_blocking=False)
            restored_bytes += destination.numel() * destination.element_size()

        restored_modules += 1

    return restored_modules, restored_bytes


def _capture_rng_state() -> dict[str, Any]:
    import random

    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    import random

    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import numpy as np

    if left.keys() != right.keys() or left["python"] != right["python"]:
        return False
    if not all(
        np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        for a, b in zip(left["numpy"], right["numpy"])
    ):
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    return all(
        torch.equal(a, b) for a, b in zip(left.get("cuda", ()), right.get("cuda", ()))
    )


def _bit_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


# Module tables keyed by hook handle id; the core tables hold the module tree.
_HOOK_TABLES = (
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)
# ``training`` follows model.train()/eval(), which the trainer sets model-wide.
_CORE_TABLES = (
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
    "training",
)


def _wrapper_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Modules a PEFT unload + re-attach would recreate: the root and PEFT types."""
    return {
        name: module
        for name, module in model.named_modules()
        if not name or type(module).__module__.split(".", 1)[0] == "peft"
    }


def _capture_module_state(module: torch.nn.Module):
    attributes = {
        key: (value.copy() if type(value) in (list, dict, set) else value)
        for key, value in module.__dict__.items()
        if key not in _CORE_TABLES and key not in _HOOK_TABLES
    }
    hooks = {
        table: frozenset(getattr(module, table))
        for table in _HOOK_TABLES
        if hasattr(module, table)
    }
    return attributes, hooks


def _restore_module_state(module: torch.nn.Module, state) -> int:
    """Drop attributes/hooks added after the attach and rebind changed values."""
    attributes, hooks = state
    changed = 0
    for key in list(module.__dict__):
        if key in _CORE_TABLES or key in _HOOK_TABLES:
            continue
        if key not in attributes:
            del module.__dict__[key]
            changed += 1
    for key, value in attributes.items():
        current = module.__dict__.get(key, _MISSING)
        if current is value:
            continue
        if type(value) in (list, dict, set):
            if type(current) is type(value) and current == value:
                continue
            value = value.copy()
        elif current is not _MISSING and type(current) is type(value) and (
            isinstance(value, (bool, int, float, str, bytes, tuple, frozenset))
            and current == value
        ):
            continue
        module.__dict__[key] = value
        changed += 1
    for table, keep in hooks.items():
        entries = getattr(module, table)
        for handle_id in [h for h in entries if h not in keep]:
            del entries[handle_id]
            changed += 1
    return changed


_MISSING = object()


class InitialAdapterSnapshot:
    """Bit-exact initial trainable state of a freshly attached task adapter.

    ``get_peft_model`` seeds every RNG with a fixed ``random_state`` before
    creating LoRA factors, so each task's fresh adapter is the same tensor
    state.  Restoring that state (and the RNG streams as they were right
    after the attach) in place reproduces a fresh attach without PEFT
    unload/re-wrap or re-running CPU Kaiming initialization.  All-zero
    tensors (``lora_B``) are stored as a flag; the rest go to pinned CPU.
    """

    def __init__(self, entries, rng_state, module_states=None):
        self.entries = entries
        self.rng_state = rng_state
        self.module_states = module_states or {}
        self.restored_wrapper_changes = 0
        self.tensor_bytes = sum(
            cpu.numel() * cpu.element_size()
            for _, _, _, cpu in entries
            if cpu is not None
        )

    @classmethod
    @torch.no_grad()
    def capture(cls, model: torch.nn.Module) -> "InitialAdapterSnapshot":
        entries = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            cpu = None
            if bool(_bit_view(parameter).any()):
                cpu = torch.empty(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device="cpu",
                    pin_memory=parameter.is_cuda,
                )
                cpu.copy_(parameter.detach())
            entries.append((name, tuple(parameter.shape), parameter.dtype, cpu))
        if not entries:
            raise RuntimeError("fresh task adapter has no trainable parameters")
        module_states = {
            name: _capture_module_state(module)
            for name, module in _wrapper_modules(model).items()
        }
        return cls(entries, _capture_rng_state(), module_states)

    def _trainable(self, model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        params = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        expected = [name for name, _, _, _ in self.entries]
        if list(params) != expected:
            missing = sorted(set(expected) - set(params))
            unexpected = sorted(set(params) - set(expected))
            raise RuntimeError(
                "trainable parameters differ from the initial adapter snapshot: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for name, shape, dtype, _ in self.entries:
            parameter = params[name]
            if tuple(parameter.shape) != shape or parameter.dtype != dtype:
                raise RuntimeError(
                    f"trainable parameter {name!r} changed layout since the "
                    f"initial adapter snapshot: {tuple(parameter.shape)}/"
                    f"{parameter.dtype} != {shape}/{dtype}"
                )
        return params

    def _restore_wrappers(self, model: torch.nn.Module) -> int:
        """Return PEFT wrapper modules to their post-attach attributes/hooks.

        Training setup registers hooks and per-task flags on these modules
        (e.g. input-require-grad hooks, the gradient-checkpointing percentage
        read by the trainer). A re-attach would start from fresh wrappers, so
        anything added since the attach is dropped here.
        """
        modules = _wrapper_modules(model)
        if modules.keys() != self.module_states.keys():
            missing = sorted(self.module_states.keys() - modules.keys())
            unexpected = sorted(modules.keys() - self.module_states.keys())
            raise RuntimeError(
                "adapter wrapper modules differ from the initial snapshot: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        return sum(
            _restore_module_state(modules[name], state)
            for name, state in self.module_states.items()
        )

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> tuple[int, int]:
        """Reset trainable tensors, wrapper state and RNG to the post-attach state."""
        params = self._trainable(model)
        self.restored_wrapper_changes = self._restore_wrappers(model)
        for name, _, _, cpu in self.entries:
            parameter = params[name]
            if parameter.grad is not None:
                raise RuntimeError(
                    f"refusing to reset {name!r} while it still holds a gradient"
                )
            if cpu is None:
                parameter.zero_()
            else:
                parameter.copy_(cpu, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _restore_rng_state(self.rng_state)
        return len(self.entries), self.tensor_bytes

    @torch.no_grad()
    def matches(self, model: torch.nn.Module) -> tuple[int, int, bool]:
        """Compare a freshly attached adapter with this snapshot, bit for bit.

        Returns ``(matching_tensors, total_tensors, rng_state_matches)``.
        """
        params = self._trainable(model)
        matching = 0
        for name, _, _, cpu in self.entries:
            current = _bit_view(params[name])
            if cpu is None:
                equal = not bool(current.any())
            else:
                equal = torch.equal(current.cpu(), _bit_view(cpu))
            matching += int(equal)
        return (
            matching,
            len(self.entries),
            _rng_state_equal(_capture_rng_state(), self.rng_state),
        )
