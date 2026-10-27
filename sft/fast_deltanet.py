"""Fast Triton + torch.compile Gated DeltaNet acceleration for Qwen3.8-27B SFT on B200."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5
import triton
import triton.language as tl


@triton.jit
def _chunk_recurrent_fwd_kernel(
    q_ptr,  # [BH, C, 64, K] float32
    k_ptr,  # [BH, C, 64, K] float32
    v_ptr,  # [BH, C, 64, V] float32
    k_cum_ptr,  # [BH, C, 64, K] float32
    attn_ptr,  # [BH, C, 64, 64] float32
    g_ptr,  # [BH, C, 64] float32
    out_ptr,  # [BH, C, 64, V] float32
    states_ptr,  # [BH, C, K, V] float32 (saved pre-chunk states for exact backward)
    num_chunks,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BV: tl.constexpr,
):
    """Fused forward recurrence across all 64-token chunks keeping state S [K=128, BV=16] in B200 SRAM."""
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_t = tl.arange(0, 64)
    offs_k = tl.arange(0, K_DIM)
    offs_v = pid_v * BV + tl.arange(0, BV)

    s = tl.zeros([K_DIM, BV], dtype=tl.float32)

    bh_q_base = q_ptr + pid_bh * num_chunks * 64 * K_DIM
    bh_k_base = k_ptr + pid_bh * num_chunks * 64 * K_DIM
    bh_v_base = v_ptr + pid_bh * num_chunks * 64 * V_DIM
    bh_kcum_base = k_cum_ptr + pid_bh * num_chunks * 64 * K_DIM
    bh_attn_base = attn_ptr + pid_bh * num_chunks * 64 * 64
    bh_g_base = g_ptr + pid_bh * num_chunks * 64
    bh_out_base = out_ptr + pid_bh * num_chunks * 64 * V_DIM
    bh_state_base = states_ptr + pid_bh * num_chunks * K_DIM * V_DIM

    for c in range(0, num_chunks):
        # Save state S_c before chunk c for O(1) per-chunk backward
        st_ptrs = bh_state_base + c * K_DIM * V_DIM + offs_k[:, None] * V_DIM + offs_v[None, :]
        tl.store(st_ptrs, s)

        q_c = tl.load(bh_q_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :])
        k_c = tl.load(bh_k_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :])
        v_c = tl.load(bh_v_base + c * 64 * V_DIM + offs_t[:, None] * V_DIM + offs_v[None, :])
        kcum_c = tl.load(bh_kcum_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :])
        attn_c = tl.load(bh_attn_base + c * 64 * 64 + offs_t[:, None] * 64 + offs_t[None, :])
        g_c = tl.load(bh_g_base + c * 64 + offs_t)
        g_last = tl.load(bh_g_base + c * 64 + 63)

        # v_new = v_c - kcum_c @ s
        v_prime = tl.dot(kcum_c, s)
        v_new = v_c - v_prime

        # attn_inter = (q_c * exp(g_c)[:, None]) @ s
        q_scaled = q_c * tl.exp(g_c)[:, None]
        attn_inter = tl.dot(q_scaled, s)

        # core_attn_out = attn_inter + attn_c @ v_new
        out_c = attn_inter + tl.dot(attn_c, v_new)
        out_ptrs = bh_out_base + c * 64 * V_DIM + offs_t[:, None] * V_DIM + offs_v[None, :]
        tl.store(out_ptrs, out_c)

        # Update state: s = s * exp(g_last) + (k_c * exp(g_last - g_c)[:, None])^T @ v_new
        decay_k = tl.exp(g_last - g_c)
        k_decayed_t = tl.trans(k_c * decay_k[:, None])
        s = s * tl.exp(g_last) + tl.dot(k_decayed_t, v_new)


@triton.jit
def _chunk_recurrent_bwd_kernel(
    q_ptr,  # [BH, C, 64, K] float32
    k_ptr,  # [BH, C, 64, K] float32
    v_ptr,  # [BH, C, 64, V] float32
    k_cum_ptr,  # [BH, C, 64, K] float32
    attn_ptr,  # [BH, C, 64, 64] float32
    g_ptr,  # [BH, C, 64] float32
    states_ptr,  # [BH, C, K, V] float32
    dout_ptr,  # [BH, C, 64, V] float32
    dq_part_ptr,  # [BH, NV, C, 64, K] float32
    dk_part_ptr,  # [BH, NV, C, 64, K] float32
    dv_ptr,  # [BH, C, 64, V] float32
    dkcum_part_ptr,  # [BH, NV, C, 64, K] float32
    dattn_part_ptr,  # [BH, NV, C, 64, 64] float32
    dg_part_ptr,  # [BH, NV, C, 64] float32
    num_chunks,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    """Fused reverse-mode autodiff across all 64-token chunks keeping dS [K=128, BV=32] in B200 SRAM."""
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_t = tl.arange(0, 64)
    offs_k = tl.arange(0, K_DIM)
    offs_v = pid_v * BV + tl.arange(0, BV)

    ds = tl.zeros([K_DIM, BV], dtype=tl.float32)

    bh_q_base = q_ptr + pid_bh * num_chunks * 64 * K_DIM
    bh_k_base = k_ptr + pid_bh * num_chunks * 64 * K_DIM
    bh_v_base = v_ptr + pid_bh * num_chunks * 64 * V_DIM
    bh_kcum_base = k_cum_ptr + pid_bh * num_chunks * 64 * K_DIM
    bh_attn_base = attn_ptr + pid_bh * num_chunks * 64 * 64
    bh_g_base = g_ptr + pid_bh * num_chunks * 64
    bh_state_base = states_ptr + pid_bh * num_chunks * K_DIM * V_DIM
    bh_dout_base = dout_ptr + pid_bh * num_chunks * 64 * V_DIM
    bh_dv_base = dv_ptr + pid_bh * num_chunks * 64 * V_DIM

    part_idx = pid_bh * NV + pid_v
    bh_dq_base = dq_part_ptr + part_idx * num_chunks * 64 * K_DIM
    bh_dk_base = dk_part_ptr + part_idx * num_chunks * 64 * K_DIM
    bh_dkcum_base = dkcum_part_ptr + part_idx * num_chunks * 64 * K_DIM
    bh_dattn_base = dattn_part_ptr + part_idx * num_chunks * 64 * 64
    bh_dg_base = dg_part_ptr + part_idx * num_chunks * 64

    for rev_c in range(0, num_chunks):
        c = num_chunks - 1 - rev_c

        q_c = tl.load(bh_q_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :])
        k_c = tl.load(bh_k_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :])
        v_c = tl.load(bh_v_base + c * 64 * V_DIM + offs_t[:, None] * V_DIM + offs_v[None, :])
        kcum_c = tl.load(bh_kcum_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :])
        attn_c = tl.load(bh_attn_base + c * 64 * 64 + offs_t[:, None] * 64 + offs_t[None, :])
        g_c = tl.load(bh_g_base + c * 64 + offs_t)
        g_last = tl.load(bh_g_base + c * 64 + 63)
        s_c = tl.load(bh_state_base + c * K_DIM * V_DIM + offs_k[:, None] * V_DIM + offs_v[None, :])
        dout_c = tl.load(bh_dout_base + c * 64 * V_DIM + offs_t[:, None] * V_DIM + offs_v[None, :])

        exp_g = tl.exp(g_c)
        exp_glast = tl.exp(g_last)
        decay_k = tl.exp(g_last - g_c)
        k_tilde = k_c * decay_k[:, None]

        # Recompute v_new = v_c - kcum_c @ s_c
        v_new = v_c - tl.dot(kcum_c, s_c)

        # d_attn_c = dout_c @ v_new^T
        d_attn_c = tl.dot(dout_c, tl.trans(v_new))
        tl.store(
            bh_dattn_base + c * 64 * 64 + offs_t[:, None] * 64 + offs_t[None, :],
            d_attn_c,
        )

        # d_v_new = attn_c^T @ dout_c + k_tilde @ ds
        d_v_new = tl.dot(tl.trans(attn_c), dout_c) + tl.dot(k_tilde, ds)
        tl.store(
            bh_dv_base + c * 64 * V_DIM + offs_t[:, None] * V_DIM + offs_v[None, :],
            d_v_new,
        )

        # d_kcum_c = -d_v_new @ s_c^T
        s_c_t = tl.trans(s_c)
        d_kcum_c = -tl.dot(d_v_new, s_c_t)
        tl.store(
            bh_dkcum_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :],
            d_kcum_c,
        )

        # d_q_scaled = dout_c @ s_c^T -> d_q_c = d_q_scaled * exp_g[:, None]
        d_q_scaled = tl.dot(dout_c, s_c_t)
        d_q_c = d_q_scaled * exp_g[:, None]
        tl.store(
            bh_dq_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :],
            d_q_c,
        )

        # d_k_tilde = v_new @ ds^T -> d_k_c = d_k_tilde * decay_k[:, None]
        d_k_tilde = tl.dot(v_new, tl.trans(ds))
        d_k_c = d_k_tilde * decay_k[:, None]
        tl.store(
            bh_dk_base + c * 64 * K_DIM + offs_t[:, None] * K_DIM + offs_k[None, :],
            d_k_c,
        )

        # Gradients w.r.t. g_c (including g_last at index 63)
        dg_from_q = tl.sum(d_q_scaled * q_c * exp_g[:, None], axis=1)
        dg_from_k = tl.sum(d_k_tilde * k_tilde, axis=1)
        dg_vec = dg_from_q - dg_from_k

        # Additional scalar contribution to g_last (index 63):
        # dL/dg_last += sum(ds * s_c * exp_glast) + sum(dg_from_k)
        dg_last_extra = tl.sum(ds * s_c) * exp_glast + tl.sum(dg_from_k, axis=0)
        is_last = offs_t == 63
        dg_vec = tl.where(is_last, dg_vec + dg_last_extra, dg_vec)
        tl.store(bh_dg_base + c * 64 + offs_t, dg_vec)

        # Propagate ds to chunk c - 1:
        # ds_prev = ds * exp_glast + (q_c * exp_g[:, None])^T @ dout_c - kcum_c^T @ d_v_new
        q_scaled = q_c * exp_g[:, None]
        ds = (
            ds * exp_glast
            + tl.dot(tl.trans(q_scaled), dout_c)
            - tl.dot(tl.trans(kcum_c), d_v_new)
        )


class _TritonChunkGatedDeltaRecurrence(torch.autograd.Function):
    """Autograd function wrapping the Triton forward & backward inter-chunk recurrence."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cumdecay: torch.Tensor,
        attn_inter_mask: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        # Shapes:
        # query, key, k_cumdecay: [B, H, C, 64, K=128]
        # value: [B, H, C, 64, V=128]
        # attn_inter_mask: [B, H, C, 64, 64]
        # g: [B, H, C, 64]
        b, h, num_chunks, chunk_len, k_dim = query.shape
        v_dim = value.shape[-1]
        assert chunk_len == 64 and k_dim == 128 and v_dim == 128

        bh = b * h
        bv = 16
        nv = v_dim // bv

        q_flat = query.reshape(bh, num_chunks, 64, k_dim).to(torch.float32).contiguous()
        k_flat = key.reshape(bh, num_chunks, 64, k_dim).to(torch.float32).contiguous()
        v_flat = value.reshape(bh, num_chunks, 64, v_dim).to(torch.float32).contiguous()
        kcum_flat = k_cumdecay.reshape(bh, num_chunks, 64, k_dim).to(torch.float32).contiguous()
        attn_flat = attn_inter_mask.reshape(bh, num_chunks, 64, 64).to(torch.float32).contiguous()
        g_flat = g.reshape(bh, num_chunks, 64).to(torch.float32).contiguous()

        out_flat = torch.empty_like(v_flat)
        states = torch.empty((bh, num_chunks, k_dim, v_dim), dtype=torch.float32, device=query.device)

        grid = (bh, nv)
        _chunk_recurrent_fwd_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            kcum_flat,
            attn_flat,
            g_flat,
            out_flat,
            states,
            num_chunks,
            K_DIM=k_dim,
            V_DIM=v_dim,
            BV=bv,
            num_warps=4,
            num_stages=1,
        )

        ctx.save_for_backward(q_flat, k_flat, v_flat, kcum_flat, attn_flat, g_flat, states)
        ctx.shapes = (b, h, num_chunks, k_dim, v_dim, bv, nv)
        return out_flat.view(b, h, num_chunks, 64, v_dim)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q_flat, k_flat, v_flat, kcum_flat, attn_flat, g_flat, states = ctx.saved_tensors
        b, h, num_chunks, k_dim, v_dim, bv, nv = ctx.shapes
        bh = b * h

        dout_flat = grad_out.reshape(bh, num_chunks, 64, v_dim).to(torch.float32).contiguous()
        dq_part = torch.empty((bh, nv, num_chunks, 64, k_dim), dtype=torch.float32, device=q_flat.device)
        dk_part = torch.empty((bh, nv, num_chunks, 64, k_dim), dtype=torch.float32, device=q_flat.device)
        dv_flat = torch.empty((bh, num_chunks, 64, v_dim), dtype=torch.float32, device=q_flat.device)
        dkcum_part = torch.empty((bh, nv, num_chunks, 64, k_dim), dtype=torch.float32, device=q_flat.device)
        dattn_part = torch.empty((bh, nv, num_chunks, 64, 64), dtype=torch.float32, device=q_flat.device)
        dg_part = torch.empty((bh, nv, num_chunks, 64), dtype=torch.float32, device=q_flat.device)

        grid = (bh, nv)
        _chunk_recurrent_bwd_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            kcum_flat,
            attn_flat,
            g_flat,
            states,
            dout_flat,
            dq_part,
            dk_part,
            dv_flat,
            dkcum_part,
            dattn_part,
            dg_part,
            num_chunks,
            K_DIM=k_dim,
            V_DIM=v_dim,
            BV=bv,
            NV=nv,
            num_warps=4,
            num_stages=1,
        )

        dq = dq_part.sum(dim=1).view(b, h, num_chunks, 64, k_dim)
        dk = dk_part.sum(dim=1).view(b, h, num_chunks, 64, k_dim)
        dv = dv_flat.view(b, h, num_chunks, 64, v_dim)
        dkcum = dkcum_part.sum(dim=1).view(b, h, num_chunks, 64, k_dim)
        dattn = dattn_part.sum(dim=1).view(b, h, num_chunks, 64, 64)
        dg = dg_part.sum(dim=1).view(b, h, num_chunks, 64)
        return dq, dk, dv, dkcum, dattn, dg


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def fast_triton_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Drop-in replacement for `torch_chunk_gated_delta_rule` using vectorized triangular solve + Triton recurrence."""
    del kwargs
    initial_dtype = query.dtype
    with torch.amp.autocast(device_type=query.device.type, enabled=False):
        if use_qk_l2norm_in_kernel:
            query = _l2norm(query, dim=-1, eps=1e-6)
            key = _l2norm(key, dim=-1, eps=1e-6)

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        if pad_size > 0:
            query = F.pad(query, (0, 0, 0, pad_size))
            key = F.pad(key, (0, 0, 0, pad_size))
            value = F.pad(value, (0, 0, 0, pad_size))
            beta = F.pad(beta, (0, pad_size))
            g = F.pad(g, (0, pad_size))

        scale = 1.0 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask_triu0 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
        )
        mask_triu1 = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
        )

        # Intra-chunk cumulative decay and lower-triangular solve (I + L)^{-1}
        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        strict_lower = ((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_triu0, 0.0)
        eye = torch.eye(chunk_size, dtype=strict_lower.dtype, device=strict_lower.device)
        # Solve (I + strict_lower) @ X = I in a single batched cuBLAS call instead of a 63-step Python loop!
        attn_inv = torch.linalg.solve_triangular(eye + strict_lower, eye, upper=False, unitriangular=True)

        value_intra = attn_inv @ v_beta
        k_cumdecay = attn_inv @ (k_beta * g.exp().unsqueeze(-1))
        attn_qk = ((query @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_triu1, 0.0)

        if (
            _USE_TRITON_RECURRENCE
            and query.is_cuda
            and chunk_size == 64
            and k_head_dim == 128
            and v_head_dim == 128
            and initial_state is None
            and not output_final_state
        ):
            core_attn_out = _TritonChunkGatedDeltaRecurrence.apply(
                query, key, value_intra, k_cumdecay, attn_qk, g
            )
            last_recurrent_state = None
        else:
            # Verified PyTorch autograd inter-chunk fallback (still uses vectorized cuBLAS solve_triangular above)
            last_recurrent_state = (
                torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value_intra.dtype, device=value_intra.device)
                if initial_state is None
                else initial_state.to(value_intra)
            )
            chunk_outputs = []
            num_chunks = query.shape[2]
            for i in range(num_chunks):
                q_i, k_i, v_i = query[:, :, i], key[:, :, i], value_intra[:, :, i]
                attn_i = attn_qk[:, :, i]
                v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
                v_new = v_i - v_prime
                attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
                chunk_outputs.append(attn_inter + attn_i @ v_new)
                last_recurrent_state = (
                    last_recurrent_state * g[:, :, i, -1, None, None].exp()
                    + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                )
            core_attn_out = torch.stack(chunk_outputs, dim=2)
            if not output_final_state:
                last_recurrent_state = None

        core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


_USE_TRITON_RECURRENCE: bool = True
_ORIG_TORCH_CHUNK_GATED_DELTA_RULE = modeling_qwen3_5.torch_chunk_gated_delta_rule


def set_use_triton_recurrence(enabled: bool) -> None:
    """Enables or disables the Triton inter-chunk recurrence kernel (falling back to vectorized solve + PyTorch recurrence)."""
    global _USE_TRITON_RECURRENCE
    _USE_TRITON_RECURRENCE = bool(enabled)


def patch_qwen35_fast_kernels(model: Optional[torch.nn.Module] = None) -> int:
    """Monkey-patches `modeling_qwen3_5` and any instantiated `Qwen3_5GatedDeltaNet` modules to use the fast kernel."""
    modeling_qwen3_5.torch_chunk_gated_delta_rule = fast_triton_chunk_gated_delta_rule
    patched = 0
    if model is not None:
        for mod in model.modules():
            if mod.__class__.__name__ == "Qwen3_5GatedDeltaNet" or hasattr(
                mod, "torch_chunk_gated_delta_rule"
            ):
                mod.torch_chunk_gated_delta_rule = fast_triton_chunk_gated_delta_rule
                patched += 1
    return patched


def validate_deltanet_backward_all_params(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    cos_sim_threshold: float = 0.999,
    rel_l2_threshold: float = 0.02,
) -> dict[str, dict[str, float | bool]]:
    """Validates custom DeltaNet forward & backward across EVERY trainable parameter of Qwen3_5GatedDeltaNet.

    Tests all 9 parameter tensors (including newly trainable full-SFT params: `in_proj_b`, `in_proj_a`,
    `A_log`, `dt_bias`, `conv1d`, `norm`, as well as `in_proj_qkv`, `in_proj_z`, `out_proj`) with
    `head_k_dim=128, head_v_dim=128`. Automatically falls back to the verified PyTorch recurrence if
    any parameter gradient fails validation.
    """
    import copy
    from transformers.models.qwen3_5 import configuration_qwen3_5

    def _run_validation_once() -> tuple[bool, dict[str, dict[str, float | bool]]]:
        torch.manual_seed(20260405)
        cfg = configuration_qwen3_5.Qwen3_5TextConfig(
            hidden_size=256,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            rms_norm_eps=1e-6,
        )

        ref_layer = modeling_qwen3_5.Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(device=device, dtype=dtype)
        ref_layer.torch_chunk_gated_delta_rule = _ORIG_TORCH_CHUNK_GATED_DELTA_RULE
        fast_layer = copy.deepcopy(ref_layer).to(device=device, dtype=dtype)
        fast_layer.torch_chunk_gated_delta_rule = fast_triton_chunk_gated_delta_rule

        x_ref = torch.randn(1, 144, 256, device=device, dtype=dtype).requires_grad_(True)
        x_fast = x_ref.detach().clone().requires_grad_(True)
        grad_out = torch.randn(1, 144, 256, device=device, dtype=torch.float32)

        prev_fn = modeling_qwen3_5.torch_chunk_gated_delta_rule
        try:
            modeling_qwen3_5.torch_chunk_gated_delta_rule = _ORIG_TORCH_CHUNK_GATED_DELTA_RULE
            out_ref = ref_layer(x_ref)
            (out_ref.float() * grad_out).sum().backward()

            modeling_qwen3_5.torch_chunk_gated_delta_rule = fast_triton_chunk_gated_delta_rule
            out_fast = fast_layer(x_fast)
            (out_fast.float() * grad_out).sum().backward()
        finally:
            modeling_qwen3_5.torch_chunk_gated_delta_rule = prev_fn

        report_inner: dict[str, dict[str, float | bool]] = {}
        all_passed_inner = True
        ref_params = dict(ref_layer.named_parameters())
        fast_params = dict(fast_layer.named_parameters())

        for name, p_ref in ref_params.items():
            p_fast = fast_params[name]
            g_ref = p_ref.grad
            g_fast = p_fast.grad
            if g_ref is None or g_fast is None:
                report_inner[name] = {"has_grad": False, "finite": False, "passed": False}
                all_passed_inner = False
                continue
            g_ref_f = g_ref.detach().float().reshape(-1)
            g_fast_f = g_fast.detach().float().reshape(-1)
            is_finite = bool(torch.isfinite(g_fast_f).all().item() and torch.isfinite(g_ref_f).all().item())
            cos_sim = float(
                F.cosine_similarity(g_ref_f.unsqueeze(0), g_fast_f.unsqueeze(0), eps=1e-12).item()
            )
            rel_l2 = float(
                (torch.linalg.vector_norm(g_ref_f - g_fast_f) / (torch.linalg.vector_norm(g_ref_f) + 1e-12)).item()
            )
            passed = is_finite and (cos_sim >= cos_sim_threshold) and (rel_l2 <= rel_l2_threshold)
            if not passed:
                all_passed_inner = False
            report_inner[name] = {
                "has_grad": True,
                "finite": is_finite,
                "cos_sim": round(cos_sim, 6),
                "rel_l2": round(rel_l2, 6),
                "passed": passed,
                "used_triton_recurrence": _USE_TRITON_RECURRENCE,
            }
        return all_passed_inner, report_inner

    try:
        all_passed, report = _run_validation_once()
    except Exception:
        all_passed = False
        report = {}

    if not all_passed and _USE_TRITON_RECURRENCE:
        set_use_triton_recurrence(False)
        all_passed, report = _run_validation_once()

    return report

