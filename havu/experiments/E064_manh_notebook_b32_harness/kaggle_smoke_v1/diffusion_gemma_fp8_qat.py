"""Weight-only E4M3 QAT for dense and routed-expert full SFT.

This simulates FP8 weight quantization, not FP8 GEMM execution or W8A8 QAT.
Canonical Parameters/checkpoints and activation/gradient compute stay BF16.
Routed experts use per-expert scales, never one scale for an entire layer.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

QAT_SCALING_MODES = ("tensor", "block128")


def fake_quantize_weight_reference(weight, scaling="tensor"):
    """Readable independent oracle; 2-D blocks use power-of-two FP32 scales."""
    if weight.ndim not in (2, 3) or scaling not in QAT_SCALING_MODES:
        raise ValueError("FP8 QAT expects a 2-D/3-D weight and tensor/block128 scaling")
    original_shape = weight.shape
    rows, columns = weight.shape[-2:]
    x = weight.detach().float().reshape(-1, rows, columns)
    if scaling == "block128":
        x = F.pad(x, (0, -columns % 128, 0, -rows % 128))
        x = x.reshape(x.shape[0], x.shape[1] // 128, 128, x.shape[2] // 128, 128)
        x = x.permute(0, 1, 3, 2, 4)
    amax = x.abs().amax(dim=(-2, -1), keepdim=True)
    scale = torch.where(amax > 0, 448.0 / amax, 1.0)
    if scaling == "block128":
        scale = torch.exp2(torch.floor(torch.log2(scale)))
    inverse = scale.reciprocal()
    result = (x * scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * inverse
    if scaling == "block128":
        result = result.permute(0, 1, 3, 2, 4).reshape(
            -1, ((rows + 127) // 128) * 128, ((columns + 127) // 128) * 128
        )[:, :rows, :columns]
    return result.reshape(original_shape).to(weight.dtype)


def _quantize_dequantize(weight, scaling):
    if weight.ndim not in (2, 3) or scaling not in QAT_SCALING_MODES:
        raise ValueError("FP8 QAT expects a 2-D/3-D weight and tensor/block128 scaling")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("FP8 QAT master weights must be BF16 or FP32")
    if weight.is_cuda:
        from diffusion_gemma_fp8_qat_kernels import quantize_dequantize_weight

        return quantize_dequantize_weight(weight, scaling)
    return fake_quantize_weight_reference(weight, scaling)


class _WeightFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scaling):
        return _quantize_dequantize(weight, scaling)

    @staticmethod
    def backward(ctx, gradient):
        # Identity STE; no rounding/scale derivative is applied to the master.
        return gradient, None


def fake_quantize_weight(weight, scaling="tensor"):
    return _WeightFakeQuant.apply(weight, scaling)


class _QATLinear(torch.autograd.Function):
    """Regenerate fake weights for dX instead of saving another BF16 model."""

    @staticmethod
    def forward(ctx, x, weight, scaling):
        ctx.save_for_backward(x, weight)
        ctx.scaling = scaling
        return F.linear(x, _quantize_dequantize(weight, scaling))

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = torch.matmul(dy, _quantize_dequantize(weight, ctx.scaling))
        if ctx.needs_input_grad[1]:
            dw = dy.reshape(-1, dy.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        return dx, dw, None


def _qat_linear_forward(self, x):
    return _QATLinear.apply(x, self.weight, self._arc_fp8_qat_scaling)


def qat_experts_forward(module, hidden_states, indices, scores, weight_forward):
    """Recompute each MoE during backward to bound fake-weight memory.

    No mutable parameter replacement: shared encoder/decoder modules retain the
    same master Parameters throughout GA, checkpoint replay, and serialization.
    Explicit master arguments make expert gradients work even if input has no
    gradient. No RNG is consumed by this deterministic QAT/recompute boundary.
    """
    scaling = module._arc_fp8_qat_scaling

    def forward(h, gate_up, down, routing_indices, routing_scores):
        return weight_forward(
            module, h, routing_indices, routing_scores,
            fake_quantize_weight(gate_up, scaling),
            fake_quantize_weight(down, scaling),
        )

    args = (
        hidden_states, module.gate_up_proj, module.down_proj, indices, scores
    )
    if not torch.is_grad_enabled():
        return forward(*args)
    return checkpoint(forward, *args, use_reentrant=True, preserve_rng_state=False)


def install_diffusion_gemma_fp8_qat(model, *, scaling="tensor", module_filter):
    """Install weight-only QAT without changing parameter or state-dict names."""
    if scaling not in QAT_SCALING_MODES:
        raise ValueError(f"unknown FP8 QAT scaling: {scaling}")
    before = {name: id(p) for name, p in model.named_parameters()}
    state_keys = set(model.state_dict())
    covered = {}
    dense = []
    experts = []
    for name, module in model.named_modules():
        if not module_filter(module, name):
            continue
        if hasattr(module, "_arc_fp8_qat_scaling"):
            raise RuntimeError(f"FP8 QAT already installed at {name}")
        if isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                raise ValueError("DiffusionGemma FP8 QAT expects bias-free projections")
            module.forward = MethodType(_qat_linear_forward, module)
            weights = [module.weight]
            dense.append(name)
        else:
            weights = [module.gate_up_proj, module.down_proj]
            if any(w.ndim != 3 for w in weights):
                raise ValueError("routed FP8 QAT expects packed 3-D expert parameters")
            experts.append(name)
        module._arc_fp8_qat_scaling = scaling
        for weight in weights:
            if not weight.requires_grad or weight.dtype != torch.bfloat16:
                raise TypeError("full FP8 QAT requires trainable BF16 master weights")
            covered[id(weight)] = weight.numel()
    if not dense or not experts:
        raise RuntimeError("full FP8 QAT must cover both dense and routed-expert weights")
    if before != {name: id(p) for name, p in model.named_parameters()}:
        raise RuntimeError("FP8 QAT changed canonical model Parameters")
    if state_keys != set(model.state_dict()):
        raise RuntimeError("FP8 QAT changed portable checkpoint keys")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "mode": "fp8-weight-only-qat-bf16-gemms",
        "master_parameter_dtype": "torch.bfloat16",
        "weight_gradient_dtype": "torch.bfloat16-identity-ste",
        "quantization_dtype": "torch.float8_e4m3fn",
        "scaling": scaling,
        "scale_scope": "per-expert-matrix" if scaling == "tensor" else "per-128x128-weight-block",
        "covered_parameter_tensors": len(covered),
        "covered_numel": sum(covered.values()),
        "covered_fraction": sum(covered.values()) / total,
        "trainable_numel": total,
        "dense_modules": dense,
        "expert_modules": experts,
        "dense_compute": "torch.bfloat16-with-e4m3-fake-weights",
        "grouped_expert_compute": "sonicmoe-bfloat16-with-e4m3-fake-weights",
        "activations_quantized": False,
        "expert_recompute": True,
        "portable_state_dict_tensors": len(state_keys),
    }
