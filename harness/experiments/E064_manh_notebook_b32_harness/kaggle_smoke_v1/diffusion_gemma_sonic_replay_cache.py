"""Experimental reentrant MoE-intermediate cache; no production default.

Retain upstream SonicMoE forward intermediates for checkpoint replay instead
of making the whole encoder layer live. Backward still uses the existing
canonical BF16 gradient sinks and original reentrant graph boundaries.
Only single-GPU, first-order full-BF16 SFT with fused experts is supported.
"""

from contextvars import ContextVar
from types import MethodType

import torch

_ACTIVE = ContextVar("arc_sonic_replay_cache", default=None)


class _RecordedContext:
    def __init__(self, real):
        object.__setattr__(self, "real", real)
        object.__setattr__(self, "attributes", {})
        object.__setattr__(self, "saved", ())
        object.__setattr__(self, "non_differentiable", ())
        object.__setattr__(self, "materialize", None)

    def __setattr__(self, name, value):
        self.attributes[name] = value
        setattr(self.real, name, value)

    def __getattr__(self, name):
        return getattr(self.real, name)

    def save_for_backward(self, *tensors):
        object.__setattr__(self, "saved", tensors)
        self.real.save_for_backward(*tensors)

    def mark_non_differentiable(self, *tensors):
        object.__setattr__(self, "non_differentiable", tensors)
        self.real.mark_non_differentiable(*tensors)

    def set_materialize_grads(self, value):
        object.__setattr__(self, "materialize", value)
        self.real.set_materialize_grads(value)

    def replay(self, ctx):
        for name, value in self.attributes.items():
            setattr(ctx, name, value)
        ctx.save_for_backward(*self.saved)
        if self.non_differentiable:
            ctx.mark_non_differentiable(*self.non_differentiable)
        if self.materialize is not None:
            ctx.set_materialize_grads(self.materialize)


def _cached_forward(key, original, ctx, *args):
    active = _ACTIVE.get()
    if active is None:
        return original(ctx, *args)
    phase, entries = active
    if phase == "record":
        if key in entries:
            raise RuntimeError(f"Sonic replay cache recorded {key} twice")
        recorded = _RecordedContext(ctx)
        output = original(recorded, *args)
        # Do not retain the discarded no-grad autograd context itself.
        object.__setattr__(recorded, "real", None)
        entries[key] = (recorded, output)
        return output
    recorded, output = entries.pop(key)
    recorded.replay(ctx)
    return output


def _make_installer(*, enabled=None, verify_inputs=False):
    from diffusion_gemma_bf16_fusions import (
        _FusedSonicDown, _FusedSonicUp, fused_sonic_experts,
    )

    enabled = [True] if enabled is None else enabled
    counters = {"record": 0, "replay": 0, "input_checks": 0, "layers": 0}
    for name, cls in (("up", _FusedSonicUp), ("down", _FusedSonicDown)):
        if getattr(cls, "_arc_replay_cache_wrapped", False):
            continue  # Other models use the unchanged path while _ACTIVE is None.
        original = cls.forward

        def forward(ctx, *args, _name=name, _original=original):
            return _cached_forward(_name, _original, ctx, *args)

        cls.forward = staticmethod(forward)
        cls._arc_replay_cache_wrapped = True

    def setup(model, *, implementation):
        if implementation != "reentrant":
            raise ValueError("Sonic replay cache requires reentrant encoder checkpoints")
        for layer in model.model.encoder.language_model.layers:
            expert = layer.experts
            if hasattr(expert, "_arc_replay_cache_state"):
                continue
            if not hasattr(expert, "_arc_bf16_expert_grad_accum"):
                raise ValueError("Sonic replay cache requires BF16 fused expert gradients")
            state = {"entries": None, "inputs": None, "versions": None}
            expert._arc_replay_cache_state = state
            original = expert.forward

            def cached(self, x, indices, scores, _layer=layer, _state=state, _original=original):
                if (not enabled[0] or not self.training
                        or not getattr(_layer, "_arc_diffusion_gemma_checkpoint_enabled", False)):
                    return _original(x, indices, scores)
                parameters = (self.gate_up_proj, self.down_proj)
                versions = tuple(p._version for p in parameters)
                if not torch.is_grad_enabled():
                    if _state["entries"] is not None:
                        raise RuntimeError("unconsumed Sonic replay cache before next forward")
                    entries = {}
                    _state.update(entries=entries, versions=versions,
                                  inputs=(x.detach(), indices.detach(), scores.detach()))
                    phase = "record"
                else:
                    entries = _state["entries"]
                    if entries is None or _state["versions"] != versions:
                        raise RuntimeError("missing or stale Sonic checkpoint intermediates")
                    for actual, saved in zip((x, indices, scores), _state["inputs"], strict=True):
                        if (actual.shape, actual.dtype, actual.device) != (saved.shape, saved.dtype, saved.device):
                            raise RuntimeError("Sonic replay input metadata changed")
                        if verify_inputs and not torch.equal(actual, saved):
                            raise RuntimeError("Sonic checkpoint replay input values changed")
                    if verify_inputs:
                        counters["input_checks"] += 1
                    phase = "replay"
                token = _ACTIVE.set((phase, entries))
                try:
                    result = fused_sonic_experts(self, x, indices, scores)
                    if phase == "record":
                        # Up's post-GEGLU activation feeds only Down.forward.
                        # Replay skips that GEMM too, and Down.backward uses
                        # the preactivation h instead. Do not retain ~3 GiB of
                        # dead auxiliary activations over 24 layers at 12K.
                        up_context, (_, h) = entries["up"]
                        object.__setattr__(up_context, "non_differentiable", ())
                        entries["up"] = (up_context, (None, h))
                    if phase == "replay" and entries:
                        raise RuntimeError("Sonic checkpoint replay left unused intermediates")
                    counters[phase] += 1
                    return result
                except BaseException:
                    _state.update(entries=None, versions=None, inputs=None)
                    raise
                finally:
                    _ACTIVE.reset(token)
                    if phase == "replay":
                        _state.update(entries=None, versions=None, inputs=None)

            expert.forward = MethodType(cached, expert)
            counters["layers"] += 1
        model._arc_sonic_replay_cache_stats = counters
        return counters

    setup.counters = counters
    return setup


def install_model_sonic_replay_cache(model, *, implementation, verify_inputs=False):
    """Install once after ordinary encoder checkpointing and BF16 expert fusions."""
    if hasattr(model, "_arc_sonic_replay_cache_stats"):
        raise RuntimeError("Sonic replay cache already installed on this model")
    return _make_installer(verify_inputs=verify_inputs)(model, implementation=implementation)


def install_sonic_replay_cache(recipe, *, enabled=None, verify_inputs=False):
    """Benchmark adapter: enabled can be switched during a gradient replay audit."""
    installer = _make_installer(enabled=enabled, verify_inputs=verify_inputs)
    original_setup = recipe.install_encoder_checkpointing

    def setup(model, *args, **kwargs):
        indices = original_setup(model, *args, **kwargs)
        installer(model, implementation=kwargs.get("implementation"))
        return indices

    recipe.install_encoder_checkpointing = setup
    return installer.counters
