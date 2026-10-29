"""Opt-in BF16 weight writeback policies, independent of model/kernel precision.

The default optimizers are unchanged. FP32 masters live in optimizer.state so
the existing state offloader and checkpoint writer retain their exact bits.
Stochastic rounding uses a private counter-derived RNG, never torch's global
RNG. CUDA and CPU use distinct versioned RNG implementations; resume must keep
the implementation, seed and counter. This module supports ordinary one-device
parameters only, not DTensor or distributed training.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


UPDATE_PRECISIONS = ("bf16", "fp32-master", "stochastic")
PRECISION_KEY = "arc_weight_update_precision"
MASTER_KEY = "arc_fp32_master"
VERSION = 1


def _seed_for_call(seed: int, counter: int) -> int:
    # SplitMix64 expands a checkpointed seed/counter pair into a per-write seed.
    mask = (1 << 64) - 1
    value = (seed + (counter + 1) * 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def validate_precision_load(saved, current, *, allow_legacy_migration=False):
    """Fail before mutation; permission only admits legacy BF16 -> a new mode."""
    if saved is None and current is None:
        return
    if saved is None:
        if allow_legacy_migration:
            return
        raise ValueError("optimizer precision change requires --allow-resume-optimizer-precision-change")
    if not isinstance(saved, Mapping) or saved.get("version") != VERSION:
        raise ValueError("invalid optimizer precision checkpoint metadata")
    if current is None or any(
        saved.get(key) != value for key, value in current.items() if key != "counter"
    ):
        raise ValueError("optimizer precision/rounding seed or RNG implementation differs from checkpoint")
    counter = saved.get("counter")
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
        raise ValueError("invalid stochastic-rounding counter in checkpoint")


def _parameters(optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]


def _saved_parameter_states(optimizer, saved):
    groups = saved["param_groups"]
    if len(groups) != len(optimizer.param_groups) or any(
        len(old["params"]) != len(new["params"])
        for old, new in zip(groups, optimizer.param_groups)
    ):
        raise ValueError("optimizer parameter groups do not match checkpoint")
    for old, new in zip(groups, optimizer.param_groups):
        for key, p in zip(old["params"], new["params"]):
            yield p, saved["state"].get(key, {})


class WeightWriteback:
    """Candidate arithmetic is FP32; only its final store rounds to BF16."""

    def __init__(self, params, mode, *, seed=0, kind="muon"):
        if mode not in UPDATE_PRECISIONS[1:]:
            raise ValueError("precision writeback requires fp32-master or stochastic")
        params = list(params)
        if not params:
            raise ValueError("precision writeback requires parameters")
        devices = {p.device for p in params}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("optimizer precision supports one CPU/CUDA device only")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("optimizer precision currently supports single-process training only")
        for p in params:
            if hasattr(p, "to_local") or not p.is_contiguous():
                raise ValueError("optimizer precision requires contiguous, non-DTensor parameters")
            if p.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("optimizer precision requires BF16 or FP32 model weights")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("optimizer rounding seed must be in [0, 2**63)")
        device = next(iter(devices))
        self.mode, self.seed, self.kind, self.counter = mode, seed, kind, 0
        self.rng = ("triton-philox-v1" if device.type == "cuda" else "torch-cpu-v1")

    def metadata(self):
        return {"version": VERSION, "kind": self.kind, "mode": self.mode,
                "seed": self.seed if self.mode == "stochastic" else None,
                "rng": self.rng if self.mode == "stochastic" else None,
                "counter": self.counter}

    def load(self, metadata, *, allow_legacy_migration=False):
        validate_precision_load(metadata, self.metadata(), allow_legacy_migration=allow_legacy_migration)
        self.counter = 0 if metadata is None else metadata["counter"]

    @torch.no_grad()
    def apply(self, p, update, state, *, scale: float, decay: float):
        if update.shape != p.shape:
            raise ValueError("weight and update shapes differ")
        if not math.isfinite(scale) or not math.isfinite(decay):
            raise ValueError("nonfinite optimizer scale or decay")
        master = None
        if self.mode == "fp32-master":
            master = state.get(MASTER_KEY)
            if master is None:
                master = state[MASTER_KEY] = p.detach().float().clone()
            if master.dtype != torch.float32 or master.shape != p.shape or master.device != p.device:
                raise ValueError("FP32 master has invalid dtype, shape or device")
        call_seed = _seed_for_call(self.seed, self.counter)
        if p.is_cuda:
            from bf16_optimizer_precision_kernels import writeback_cuda
            writeback_cuda(p, update.contiguous(), master, scale, decay, call_seed,
                           stochastic=self.mode == "stochastic")
            # External Triton writes must invalidate parameter-version caches.
            torch.autograd.graph.increment_version(p)
        else:
            generator = torch.Generator(device="cpu").manual_seed(call_seed)
            flat_p, flat_u = p.view(-1), update.reshape(-1)
            flat_master = master.view(-1) if master is not None else None
            for start in range(0, p.numel(), 1 << 20):
                target = flat_p[start:start + (1 << 20)]
                source = target if flat_master is None else flat_master[start:start + (1 << 20)]
                candidate = source.float().clone().mul_(decay).add_(
                    flat_u[start:start + (1 << 20)].float(), alpha=-scale)
                if flat_master is not None:
                    source.copy_(candidate)
                if self.mode == "stochastic" and p.dtype == torch.bfloat16:
                    bits = candidate.view(torch.int32)
                    random = torch.randint(0, 65536, bits.shape, dtype=torch.int32, generator=generator)
                    rounded = ((bits + random) & -65536).view(torch.float32)
                    # Preserve NaN/Inf rather than turning their bit patterns into finite values.
                    candidate = torch.where(torch.isfinite(candidate), rounded, candidate)
                target.copy_(candidate)
        self.counter += 1


class PrecisionAdamW(torch.optim.AdamW):
    """AdamW with FP32 masters/moments or stochastic BF16 weight stores.

Stochastic mode retains BF16 moments for BF16 weights, matching legacy storage;
it evaluates the final direction/candidate in FP32. FP32-master mode is the
high-precision reference. The model's Parameter objects are never replaced.
"""

    def __init__(self, params, *, update_precision="fp32-master", rounding_seed=0,
                 allow_precision_migration=False, **kwargs):
        super().__init__(params, **kwargs)
        if any(g.get(name) for g in self.param_groups
               for name in ("amsgrad", "maximize", "capturable", "differentiable", "fused")):
            raise ValueError("PrecisionAdamW does not support AMSGrad/maximize/capturable/differentiable/fused")
        self.writeback = WeightWriteback(_parameters(self), update_precision, seed=rounding_seed, kind="adamw")
        self.allow_precision_migration = bool(allow_precision_migration)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, decay = float(group["lr"]), 1.0 - float(group["lr"]) * group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("PrecisionAdamW does not support sparse gradients")
                state = self.state[p]
                dtype = torch.float32 if self.writeback.mode == "fp32-master" else p.dtype
                if not state:
                    state["step"] = torch.tensor(0.0, device="cpu")
                    state["exp_avg"] = torch.zeros_like(p, dtype=dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype)
                state["step"] += 1
                step = int(state["step"].item())
                grad = p.grad.to(dtype)
                mean, variance = state["exp_avg"], state["exp_avg_sq"]
                if mean.dtype != dtype or variance.dtype != dtype:
                    raise ValueError("AdamW moment dtype does not match precision policy")
                mean.lerp_(grad, 1.0 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denominator = variance.float().sqrt().div_(math.sqrt(1.0 - beta2**step)).add_(group["eps"])
                direction = mean.float() / denominator
                self.writeback.apply(p, direction, state, scale=lr / (1.0 - beta1**step), decay=decay)
        return loss

    def state_dict(self):
        result = super().state_dict()
        result[PRECISION_KEY] = self.writeback.metadata()
        return result

    def load_state_dict(self, saved):
        metadata = saved.get(PRECISION_KEY)
        validate_precision_load(metadata, self.writeback.metadata(),
                                allow_legacy_migration=self.allow_precision_migration)
        pairs = list(_saved_parameter_states(self, saved))
        for p, old in pairs:
            for key in ("exp_avg", "exp_avg_sq", MASTER_KEY):
                if key in old and (not torch.is_tensor(old[key]) or old[key].shape != p.shape):
                    raise ValueError(f"invalid AdamW checkpoint tensor {key}")
            if metadata is not None and old and self.writeback.mode == "fp32-master":
                if any(key not in old or old[key].dtype != torch.float32
                       for key in ("exp_avg", "exp_avg_sq", MASTER_KEY)):
                    raise ValueError("FP32 AdamW checkpoint lost its master or moment precision")
        result = super().load_state_dict(saved)
        for p, old in pairs:
            if not old:
                continue
            state = self.state[p]
            state["step"] = torch.as_tensor(old["step"], dtype=torch.float32, device="cpu").clone()
            dtype = torch.float32 if self.writeback.mode == "fp32-master" else p.dtype
            # Optimizer.load_state_dict normally downcasts these to p.dtype!
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = old[key].to(device=p.device, dtype=dtype).clone()
            if self.writeback.mode == "fp32-master":
                state[MASTER_KEY] = (old[MASTER_KEY] if MASTER_KEY in old else p.detach()).to(
                    device=p.device, dtype=torch.float32).clone()
        self.writeback.load(metadata, allow_legacy_migration=self.allow_precision_migration)
        return result


def precision_muon_class(base_class):
    """Wrap the audited Muon writeback hook, leaving momentum and NS unchanged."""
    if getattr(base_class, "parameter_update_version", None) != VERSION:
        raise RuntimeError("Muon update precision requires the current ARC bitsandbytes Muon overlay")

    class PrecisionMuon(base_class):
        def __init__(self, *args, update_precision, rounding_seed=0,
                     allow_precision_migration=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.writeback = WeightWriteback(_parameters(self), update_precision, seed=rounding_seed, kind="muon")
            self.allow_precision_migration = bool(allow_precision_migration)
            self._arc_parameter_update = self._precision_update

        def _precision_update(self, parameter, update, *, scale, decay):
            self.writeback.apply(parameter, update, self.state[parameter], scale=scale, decay=decay)

        def state_dict(self):
            result = super().state_dict()
            result[PRECISION_KEY] = self.writeback.metadata()
            return result

        def load_state_dict(self, saved, **kwargs):
            metadata = saved.get(PRECISION_KEY)
            validate_precision_load(metadata, self.writeback.metadata(),
                                    allow_legacy_migration=self.allow_precision_migration)
            pairs = list(_saved_parameter_states(self, saved))
            for p, old in pairs:
                master = old.get(MASTER_KEY)
                if master is not None and (master.dtype != torch.float32 or master.shape != p.shape):
                    raise ValueError("invalid FP32 Muon master checkpoint tensor")
                if metadata is not None and old and self.writeback.mode == "fp32-master" and master is None:
                    raise ValueError("FP32 Muon checkpoint is missing its master weights")
            result = super().load_state_dict(saved, **kwargs)
            for p, old in pairs:
                if old and self.writeback.mode == "fp32-master":
                    self.state[p][MASTER_KEY] = (old[MASTER_KEY] if MASTER_KEY in old else p.detach()).to(
                        device=p.device, dtype=torch.float32).clone()
            self.writeback.load(metadata, allow_legacy_migration=self.allow_precision_migration)
            return result

    return PrecisionMuon
