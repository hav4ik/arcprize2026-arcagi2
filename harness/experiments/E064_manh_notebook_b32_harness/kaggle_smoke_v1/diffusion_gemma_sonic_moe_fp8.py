"""Opt-in SM120 E4M3 expert training with SonicMoE routing and QuACK GEMMs.

Adapted from the SonicMoE FP8 approach at bdepyzy/sonic-moe, commit 1630585,
and our existing ARC QuACK backward. SonicMoE is Apache-2.0; its license is
bundled in third_party/sonic_moe. ARC changes: per-expert FP32 scales, correct
interleaved tanh-GeGLU, canonical weight layouts, no stale version-only cache,
and the SM120 plain-FP8 load fix. All six expert GEMMs use FP8 operands;
master Parameters, accumulated gradients, and checkpoints remain BF16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quantize_e4m3(x, *, per_expert=False):
    x = x.detach().contiguous()
    if x.dtype != torch.bfloat16 or x.ndim not in (2, 3):
        raise TypeError("SonicMoE FP8 expects BF16 2-D activations or 3-D expert weights")
    if per_expert != (x.ndim == 3):
        raise ValueError("packed expert tensors must use independent per-expert scales")
    maxima = x.abs().amax(dim=(-2, -1)).float()
    scales = maxima.clamp_min(1e-12) / 448.0
    if x.is_cuda:
        from diffusion_gemma_fp8_qat_kernels import quantize_e4m3_tensor

        return quantize_e4m3_tensor(x, scales), scales
    shape = (-1, 1, 1) if per_expert else ()
    q = (x.float() / scales.reshape(shape)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scales


def _ops():
    from quack.gemm_interface import gemm, gemm_dgated

    return gemm, gemm_dgated


def _descale(x, scale):
    if scale.ndim:
        scale = scale[:, None]
    return (x.float() * scale).to(torch.bfloat16)


class _SonicFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, indices, scores, gate_up, down):
        from diffusion_gemma_sonic_moe import _load_arc_sonicmoe_kernel

        sonic = _load_arc_sonicmoe_kernel().functional
        gemm, _ = _ops()
        tokens, hidden_dim = hidden.shape
        top_k = indices.shape[-1]
        experts = gate_up.shape[0]
        pairs = tokens * top_k
        device = hidden.device
        token_ids = torch.arange(tokens, device=device, dtype=torch.int32).repeat_interleave(top_k)
        expert_ids = indices.reshape(-1).int()
        scores = scores.reshape(-1).float()
        scatter = torch.empty(pairs, device=device, dtype=torch.int32)
        reverse = torch.empty_like(scatter)
        gather = torch.empty_like(scatter)
        frequencies = torch.empty(experts, device=device, dtype=torch.int32)
        offsets = torch.empty(experts + 1, device=device, dtype=torch.int32)
        token_offsets = torch.empty(tokens + 1, device=device, dtype=torch.int32)
        sonic.general_routing_router_metadata_triton(
            token_ids, expert_ids, tokens, experts, frequencies, offsets,
            gather, scatter, reverse, token_offsets,
        )
        sorted_experts = expert_ids[scatter]
        sorted_scores = scores[scatter]
        qx, sx = quantize_e4m3(hidden)
        qw1, sw1 = quantize_e4m3(gate_up, per_expert=True)
        # Preserve column-major B here: its noncontiguous axis is the
        # concatenated gate/up axis. Output PreAct is always interleaved.
        raw_h = gemm(
            qx, qw1.transpose(1, 2), out_dtype=torch.bfloat16,
            cu_seqlens_m=offsets, A_idx=gather, concat_layout=("B",), tuned=False,
        )
        h = _descale(raw_h, sx * sw1[sorted_experts])
        a = (F.gelu(h[:, ::2].float(), approximate="tanh") * h[:, 1::2].float()).to(hidden.dtype)
        qa, sa = quantize_e4m3(a)
        qw2, sw2 = quantize_e4m3(down, per_expert=True)
        y = _descale(gemm(
            qa, qw2.transpose(1, 2), out_dtype=torch.bfloat16,
            cu_seqlens_m=offsets, tuned=False,
        ), sa * sw2[sorted_experts])
        output = torch.empty_like(hidden)
        sonic._router_forward(
            y=y, o=output, topk_scores=scores, s_reverse_scatter_idx=reverse,
            num_activated_expert_per_token_offset=token_offsets,
            varlen_K_max=experts, H=hidden_dim, is_varlen_K=True,
        )
        ctx.sonic = sonic
        ctx.shape = (tokens, hidden_dim, experts, top_k)
        ctx.save_for_backward(qx, sx, gate_up, down, h, sorted_experts, sorted_scores,
                              gather, scatter, reverse, offsets, token_offsets)
        return output

    @staticmethod
    def backward(ctx, dy):
        gemm, gemm_dgated = _ops()
        (qx, sx, gate_up, down, h, sorted_experts, sorted_scores,
         gather, scatter, reverse, offsets, token_offsets) = ctx.saved_tensors
        tokens, hidden_dim, experts, top_k = ctx.shape
        qdy, sdy = quantize_e4m3(dy)
        qw2, sw2 = quantize_e4m3(down, per_expert=True)
        dh, a_prime, ds = gemm_dgated(
            qdy, qw2, h, activation="geglu",
            out_dtype=torch.bfloat16, postact_dtype=torch.bfloat16,
            colvec_scale=sorted_scores, colvec_reduce=True,
            cu_seqlens_m=offsets, A_idx=gather, dynamic_scheduler=False, tuned=False,
        )
        dh = _descale(dh, sdy * sw2[sorted_experts])
        ds = ds * (sdy * sw2[sorted_experts])
        qap, sap = quantize_e4m3(a_prime)
        dw2 = _descale(gemm(
            qdy.T, qap, out_dtype=torch.bfloat16, cu_seqlens_k=offsets,
            A_idx=gather, dynamic_scheduler=False, tuned=False,
        ), sdy * sap)
        qdh, sdh = quantize_e4m3(dh)
        qw1, sw1 = quantize_e4m3(gate_up, per_expert=True)
        dx_routed = _descale(gemm(
            qdh, qw1, out_dtype=torch.bfloat16, cu_seqlens_m=offsets,
            concat_layout=("B",), dynamic_scheduler=False, tuned=False,
        ), sdh * sw1[sorted_experts])
        # Native [E, 2I, H] output; concat remaps its noncontiguous row axis.
        dw1 = _descale(gemm(
            qdh.T, qx[gather], out_dtype=torch.bfloat16, cu_seqlens_k=offsets,
            concat_layout=("out",), dynamic_scheduler=False, tuned=False,
        ), sdh * sx)
        dx = torch.empty(tokens, hidden_dim, device=dy.device, dtype=dy.dtype)
        ctx.sonic._token_broadcast_backward(
            dx_reduced=dx, dx_expanded=dx_routed, s_reverse_scatter_idx=reverse,
            num_activated_expert_per_token_offset=token_offsets,
            varlen_K_max=experts, H=hidden_dim, is_varlen_K=True,
        )
        original_ds = torch.empty_like(ds)
        original_ds[scatter] = ds
        return dx, None, original_ds.reshape(tokens, top_k), dw1, dw2


def sonic_fp8_experts_forward(module, hidden_states, indices, scores):
    if (not module.has_gate or not module.is_concatenated or module.is_transposed
            or module.has_bias):
        raise ValueError("FP8 SonicMoE requires native bias-free concatenated gated experts")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("FP8 SonicMoE expects BF16 residual activations")
    return _SonicFP8.apply(hidden_states, indices, scores, module.gate_up_proj, module.down_proj)
