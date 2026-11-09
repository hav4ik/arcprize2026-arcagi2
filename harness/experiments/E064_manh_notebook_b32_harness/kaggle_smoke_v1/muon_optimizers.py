"""Shared conventional and quantized Muon optimizer integration.

The 8/32-bit classes derive from the pinned bitsandbytes feature branch;
MuonBF16 is an ARC extension in the vendored offline overlay. This module
keeps that dependency lazy and combines matrix-only Muon with the elementwise
optimizer needed by norms, embeddings, language heads, and routing parameters.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
import importlib
import sys
from typing import Any

import torch

from normuon_offload import make_state_offloaded_optimizer_class


BITSANDBYTES_MUON_COMMIT = "4779028943707ea6aced4669effc9ebfba51981c"
ARC_PER_HEAD_MUON_VERSION = 1
MUON_ALGORITHMS = frozenset({"muon", "normuon"})


def require_bitsandbytes_muon(momentum_bits: int):
    """Return the requested Muon class or raise an actionable pinned error."""
    if momentum_bits not in {8, 16, 32}:
        raise ValueError(f"Muon momentum bits must be 8, 16 (BF16), or 32; got {momentum_bits}")
    class_name = "MuonBF16" if momentum_bits == 16 else f"Muon{momentum_bits}bit"
    try:
        import bitsandbytes as bitsandbytes_module
        from bitsandbytes import optim as bnb_optim
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"{class_name} requires bitsandbytes plus the ARC offline Muon "
            f"overlay (upstream base commit {BITSANDBYTES_MUON_COMMIT})."
        ) from error
    # Prefer the concrete module over attributes cached by
    # ``bitsandbytes.optim``. The offline overlay replaces and reloads this
    # module without rewriting the package initializer, whose old class object
    # may otherwise survive in a long-lived notebook process.
    try:
        muon_module = importlib.import_module("bitsandbytes.optim.muon")
    except ImportError:
        muon_module = None
    optimizer_class = (
        None if muon_module is None else getattr(muon_module, class_name, None)
    )
    if optimizer_class is None:
        optimizer_class = getattr(bnb_optim, class_name, None)
    if optimizer_class is None:
        raise RuntimeError(
            f"Installed bitsandbytes {getattr(bitsandbytes_module, '__version__', 'unknown')} "
            f"does not provide bitsandbytes.optim.muon.{class_name}. Copy the "
            "complete ARC third_party/bitsandbytes_muon directory and run "
            "ensure_bitsandbytes_muon() before optimizer construction "
            f"(upstream base commit {BITSANDBYTES_MUON_COMMIT})."
        )
    if momentum_bits == 16 and getattr(optimizer_class, "bf16_muon_version", None) != 1:
        raise RuntimeError("MuonBF16 requires the current ARC offline bitsandbytes Muon overlay")
    return bitsandbytes_module, optimizer_class


def _clean_parameter_group(group: Mapping[str, Any]) -> dict[str, Any]:
    """Keep optimizer overrides that affect matrix-update semantics."""
    cleaned = {"params": list(group["params"])}
    for name in ("lr", "weight_decay", "num_heads", "group_name"):
        if name in group:
            cleaned[name] = group[name]
    return cleaned


def split_muon_parameter_groups(
    parameter_groups: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split annotated Dion-style groups into matrix and scalar optimizers."""
    matrix_groups = []
    scalar_groups = []
    parameter_ids: set[int] = set()
    for group in parameter_groups:
        parameters = list(group.get("params", ()))
        if not parameters:
            continue
        duplicates = [
            parameter for parameter in parameters if id(parameter) in parameter_ids
        ]
        if duplicates:
            raise ValueError("Muon parameter groups contain duplicate parameters")
        parameter_ids.update(id(parameter) for parameter in parameters)
        algorithm = group.get("algorithm")
        if algorithm in MUON_ALGORITHMS:
            if any(parameter.ndim < 2 for parameter in parameters):
                raise ValueError(
                    "Muon matrix groups cannot contain parameters with ndim < 2"
                )
            cleaned = _clean_parameter_group(group)
            num_heads = cleaned.get("num_heads")
            if num_heads is not None:
                if isinstance(num_heads, bool) or not isinstance(num_heads, int):
                    raise TypeError(
                        f"Muon num_heads must be an integer; got {num_heads!r}"
                    )
                if num_heads < 2:
                    raise ValueError(
                        f"Muon num_heads must be at least 2; got {num_heads}"
                    )
                invalid_shapes = [
                    tuple(parameter.shape)
                    for parameter in parameters
                    if parameter.ndim != 2 or parameter.shape[0] % num_heads
                ]
                if invalid_shapes:
                    raise ValueError(
                        f"Muon num_heads={num_heads} does not evenly split "
                        f"2-D output rows for shapes {invalid_shapes}"
                    )
            matrix_groups.append(cleaned)
        elif algorithm == "adamw":
            scalar_groups.append(_clean_parameter_group(group))
        else:
            raise ValueError(
                "Muon composite groups must declare algorithm='muon', "
                f"'normuon', or 'adamw'; got {algorithm!r}"
            )
    if not matrix_groups:
        raise RuntimeError("Muon found no eligible matrix parameter groups")
    return matrix_groups, scalar_groups


class CompositeOptimizer(torch.optim.Optimizer):
    """Expose multiple disjoint child optimizers as one Trainer optimizer.

    The wrapper intentionally does not call ``Optimizer.__init__`` because it
    must expose the exact child parameter-group dictionaries.  Accelerate's
    FSDP2 preparation replaces each group's parameter list in place; shared
    dictionaries ensure the child optimizers receive those DTensors too.
    """

    def __init__(self, optimizers: Mapping[str, torch.optim.Optimizer]):
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one child")
        self.optimizers = OrderedDict(optimizers)
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

    @property
    def param_groups(self):
        return [
            group
            for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]

    @param_groups.setter
    def param_groups(self, _value):
        # Optimizer.__setstate__ may assign this during generic framework
        # bookkeeping. Child state_dict/load_state_dict remains authoritative.
        if hasattr(self, "optimizers"):
            raise AttributeError(
                "CompositeOptimizer parameter groups are owned by its children"
            )

    @property
    def state(self):
        combined = {}
        for optimizer in self.optimizers.values():
            combined.update(optimizer.state)
        return combined

    @property
    def defaults(self):
        return {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers.values():
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "composite_optimizer_version": 1,
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("composite_optimizer_version") != 1:
            raise ValueError("Unsupported CompositeOptimizer state_dict")
        child_states = state_dict.get("optimizers", {})
        if tuple(child_states) != tuple(self.optimizers):
            raise ValueError(
                "CompositeOptimizer children do not match checkpoint: "
                f"expected {tuple(self.optimizers)}, got {tuple(child_states)}"
            )
        # Validate every child's precision before mutating either child. In
        # particular, legacy optimizers must not silently discard FP32 masters
        # or stochastic-rounding counters from a newer checkpoint.
        from bf16_optimizer_precision import PRECISION_KEY, validate_precision_load
        for name, optimizer in self.optimizers.items():
            writeback = getattr(optimizer, "writeback", None)
            validate_precision_load(
                child_states[name].get(PRECISION_KEY),
                writeback.metadata() if writeback is not None else None,
                allow_legacy_migration=getattr(optimizer, "allow_precision_migration", False),
            )
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(child_states[name])

    def add_param_group(self, _param_group):
        raise NotImplementedError(
            "Add groups to the matrix/scalar child before constructing the composite"
        )


def create_muon_composite_optimizer(
    parameter_groups,
    *,
    momentum_bits: int,
    lr: float,
    momentum: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    nesterov: bool,
    adjust_lr: str | None,
    ns_chunk_size: int,
    min_8bit_size: int = 4096,
    scalar_optimizer: str = "adamw_torch",
    state_offload: bool = False,
    state_offload_pin_memory: bool = True,
    log_prefix: str = "[Trainer]",
    muon_update_precision: str = "bf16",
    adamw_update_precision: str = "bf16",
    optimizer_rounding_seed: int = 0,
    allow_precision_migration: bool = False,
):
    """Build matrix Muon plus a disjoint AdamW scalar/special child."""
    if ns_chunk_size <= 0:
        raise ValueError("Muon ns_chunk_size must be positive")
    if min_8bit_size <= 0:
        raise ValueError("Muon min_8bit_size must be positive")
    if scalar_optimizer not in {"adamw_torch", "adamw_8bit"}:
        raise ValueError("Muon scalar optimizer must be adamw_torch or adamw_8bit")
    bitsandbytes_module, muon_class = require_bitsandbytes_muon(momentum_bits)
    matrix_groups, scalar_groups = split_muon_parameter_groups(parameter_groups)
    from bf16_optimizer_precision import UPDATE_PRECISIONS, PrecisionAdamW, precision_muon_class
    if muon_update_precision not in ("bf16", "stochastic") or adamw_update_precision not in UPDATE_PRECISIONS:
        raise ValueError("invalid optimizer update precision")
    if adamw_update_precision != "bf16" and scalar_optimizer != "adamw_torch":
        raise ValueError("AdamW update precision requires adamw_torch scalar optimizer")
    matrix_kwargs = {}
    if muon_update_precision != "bf16":
        muon_class = precision_muon_class(muon_class)
        matrix_kwargs = dict(update_precision=muon_update_precision,
                             rounding_seed=optimizer_rounding_seed,
                             allow_precision_migration=allow_precision_migration)
    if any(group.get("num_heads") is not None for group in matrix_groups):
        installed_version = getattr(muon_class, "per_head_muon_version", None)
        if installed_version != ARC_PER_HEAD_MUON_VERSION:
            raise RuntimeError(
                "Per-head Muon requires the ARC bitsandbytes Muon overlay "
                f"version {ARC_PER_HEAD_MUON_VERSION}; installed optimizer "
                f"reports {installed_version!r}. Re-run the offline overlay "
                "before constructing the optimizer."
            )
    matrix_optimizer = muon_class(
        matrix_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        adjust_lr=adjust_lr,
        ns_chunk_size=ns_chunk_size,
        min_8bit_size=min_8bit_size,
        is_paged=False,
        **matrix_kwargs,
    )
    children: OrderedDict[str, torch.optim.Optimizer] = OrderedDict(
        (("muon", matrix_optimizer),)
    )
    if scalar_groups:
        if scalar_optimizer == "adamw_torch":
            scalar_class = torch.optim.AdamW
            scalar_kwargs = {}
            if adamw_update_precision != "bf16":
                scalar_class = PrecisionAdamW
                scalar_kwargs = dict(update_precision=adamw_update_precision,
                                     rounding_seed=optimizer_rounding_seed ^ 0x4144414D,
                                     allow_precision_migration=allow_precision_migration)
        else:
            scalar_class = bitsandbytes_module.optim.AdamW8bit
            scalar_kwargs = {"min_8bit_size": min_8bit_size}
        children["scalar"] = scalar_class(
            scalar_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **scalar_kwargs,
        )
    composite_class = CompositeOptimizer
    if state_offload:
        composite_class = make_state_offloaded_optimizer_class(
            torch,
            CompositeOptimizer,
            pin_memory=state_offload_pin_memory,
            manager_attribute="_arc_muon_state_offloader",
            optimizer_label=("MuonBF16 optimizer" if momentum_bits == 16 else f"Muon{momentum_bits}bit optimizer"),
            pin_memory_option="--no-muon-state-offload-pin-memory",
        )
    optimizer = composite_class(children)
    matrix_numel = sum(
        parameter.numel() for group in matrix_groups for parameter in group["params"]
    )
    scalar_numel = sum(
        parameter.numel() for group in scalar_groups for parameter in group["params"]
    )
    extra_precision_bytes = (
        sum(parameter.numel() * (12 - 2 * parameter.element_size())
            for group in scalar_groups for parameter in group["params"])
        if adamw_update_precision == "fp32-master" else 0
    )
    print(
        f"{log_prefix} bitsandbytes Muon initialized: "
        f"version={getattr(bitsandbytes_module, '__version__', 'unknown')}, "
        f"momentum_bits={momentum_bits}, matrix_values={matrix_numel:,}, "
        f"momentum_dtype={'bfloat16' if momentum_bits == 16 else 'float32' if momentum_bits == 32 else 'uint8-blockwise'}, "
        f"scalar_values={scalar_numel:,}, scalar_optimizer={scalar_optimizer}, "
        f"ns_chunk_size={ns_chunk_size}, min_8bit_size={min_8bit_size}, "
        f"adjust_lr={adjust_lr}, state_offload={state_offload}, "
        f"state_offload_pin_memory={state_offload_pin_memory}, "
        f"muon_update_precision={muon_update_precision}, "
        f"adamw_update_precision={adamw_update_precision}, "
        f"extra_precision_state_mib={extra_precision_bytes / 2**20:.2f}",
        file=sys.__stdout__,
        flush=True,
    )
    return optimizer
