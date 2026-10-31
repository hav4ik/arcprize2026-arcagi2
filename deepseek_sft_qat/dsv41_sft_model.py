"""Pruned DeepSeek-V4.1-Flash-1017 (`Tool E128`) Full-SFT Model & 2-Bit GPTQ/Humming Exporter.

Implements:
  1. Exact de-quantization of `dsv41_e128_2bit` (`model-00001/00002.safetensors` FP8_E4M3 + UE8M0
     block-32 backbone weights and `gptq/layer_00..39.safetensors` 2-bit Humming `uint2` routed
     experts) into BF16 / FP32 for training.
  2. Differentiable full-sequence teacher-forced PyTorch forward pass matching DeepSeek-V4.1:
     - Single-Pass mHC (`hc_mixes` with 20-iteration Sinkhorn-Knopp in FP32, `hc_pre`, `hc_post`)
     - CSA2 Compressor (`ratio=2` learned softmax pooling on L2/8/14, `ratio=1` on L20)
     - CSA2 Two-Level Indexer (top-k compressed KV selection + candidate block filtering at L20)
     - Differentiable Causal SWA (`window_size=128`) + Compressed KV Sparse Attention with learnable
       `attn_sink`, YaRN RoPE (`rope_head_dim=64`), and grouped `wo_a`/`wo_b` projections
     - Compact Engram Conditional Memory (`layers.{1,14}`) with sharded FP8/row_map lookup and
       trainable `wkv`, `q_weight`, `k_weight`
     - Top-6 Routed MoE (`128` experts/layer sharded across ranks, `sqrtsoftplus` scoring,
       `route_scale=1.5`, clamped SwiGLU `limit=10.0`, differentiable `dist.nn.all_reduce`) +
       trainable Shared Expert + optional routed-expert LoRA (`routed_expert_lora_r`) that merges
       into the expert weights and re-packs to 2-bit Humming format (`q, s, z`).
  3. 2-Bit Checkpoint Export (`export_dsv41_e128_2bit_checkpoint`) at every SFT milestone
     (`0.25`, `0.50`, `0.75`, and `1.00` epoch), writing:
     - `model-00001-of-00002.safetensors` & `model-00002-of-00002.safetensors` (with updated
       `FP8_E4M3` + `UE8M0` block-32 and `BF16`/`F32` backbone tensors)
     - `gptq/layer_00..39.safetensors` (with 2-bit Humming `uint2` `q, s, z` expert tensors)
     - All config, tokenizer, manifest, and compact Engram files for drop-in vLLM/Humming serving.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent import futures
from dataclasses import dataclass
import functools
import io
import json
import math
import os
import shutil
import tempfile
from typing import Any

import numpy as np
import safetensors
from safetensors.torch import save_file
import torch
from torch import nn
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_F
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_ckpt

from google3.experimental.users.arazdai.kaggle.sft import export_and_quantize
from google3.pyglib import gfile


FP8_BLOCK_SIZE = 32
FP8_MAX = 448.0
# Position-based tie-break added to indexer scores before top-k. Far below float32 resolution of
# any non-zero score (ulp(1e-6) ~ 1e-13), so it only orders exact ties (e.g. ReLU-zeroed scores)
# deterministically and identically across the single-stream and batched decode paths.
_TOPK_TIE_EPS = 1e-30
FP8_AMAX_FLOOR = 1e-4
FP4_MAX = 6.0
SHIFTS_U2 = torch.arange(0, 32, 2, dtype=torch.int32)
FP4_E2M1_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


# ==============================================================================
# 1. Exact De-Quantization, Re-Quantization & QAT STE Primitives (Havu + Nemotron)
# ==============================================================================


def e8m0_to_f32_torch(scale_u8: torch.Tensor) -> torch.Tensor:
    """Converts uint8 / float8_e8m0fnu exponent bytes to float32 2^(e - 127)."""
    if scale_u8.dtype != torch.uint8:
        scale_u8 = scale_u8.view(torch.uint8)
    return torch.exp2(scale_u8.to(torch.int32).to(torch.float32) - 127.0)


def _fast_round_scale_e8m0(amax: torch.Tensor, max_val: float = FP8_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact IEEE-754 power-of-two ceiling scale matching `inference/kernel.py:fast_round_scale`."""
    prod = (amax.clamp_min(FP8_AMAX_FLOOR) * (1.0 / max_val)).to(torch.float32)
    bits = prod.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man_nonzero = ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exp_ceil = (exp + man_nonzero).clamp(0, 254)
    scale_u8 = exp_ceil.to(torch.uint8)
    scale_f32 = torch.exp2(exp_ceil.to(torch.float32) - 127.0)
    return scale_u8, scale_f32


def dequantize_fp8_block32_torch(
    w_fp8: torch.Tensor,
    w_scale_u8: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """De-quantizes `[out, in]` `float8_e4m3fn` weight + 1D `[out, in//32]` or 2D `[out//32, in//32]` `UE8M0` scale."""
    out_dim, in_dim = w_fp8.shape
    b_in = in_dim // block_size
    s_f32 = e8m0_to_f32_torch(w_scale_u8)
    if s_f32.shape == (out_dim, b_in):
        w_f32 = w_fp8.to(torch.float32).view(out_dim, b_in, block_size)
        return (w_f32 * s_f32.unsqueeze(-1)).reshape(out_dim, in_dim).to(out_dtype)
    b_out = out_dim // block_size
    w_f32 = w_fp8.to(torch.float32).view(b_out, block_size, b_in, block_size)
    return (w_f32 * s_f32.view(b_out, 1, b_in, 1)).reshape(out_dim, in_dim).to(out_dtype)


def quantize_fp8_block32_torch(
    w: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    scale_2d: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantizes `[out, in]` BF16/FP32 weight to `float8_e4m3fn` + `UE8M0` scale (1D `[out, in//32]` or 2D `[out//32, in//32]`)."""
    out_dim, in_dim = w.shape
    if in_dim % block_size != 0:
        raise ValueError(
            f"Input dim {in_dim} of shape {(out_dim, in_dim)} not divisible by block_size={block_size}"
        )
    b_in = in_dim // block_size
    if scale_2d and out_dim % block_size == 0:
        b_out = out_dim // block_size
        w_f32 = w.detach().to(torch.float32).view(b_out, block_size, b_in, block_size)
        amax = w_f32.abs().amax(dim=(1, 3))
        scale_u8, scale_f32 = _fast_round_scale_e8m0(amax, FP8_MAX)
        q_f32 = (w_f32 / scale_f32.view(b_out, 1, b_in, 1)).clamp(-FP8_MAX, FP8_MAX)
        w_fp8 = q_f32.reshape(out_dim, in_dim).to(torch.float8_e4m3fn)
        scale_e8m0 = scale_u8.view(torch.float8_e8m0fnu)
        return w_fp8, scale_e8m0

    w_f32 = w.detach().to(torch.float32).view(out_dim, b_in, block_size)
    amax = w_f32.abs().amax(dim=-1)
    scale_u8, scale_f32 = _fast_round_scale_e8m0(amax, FP8_MAX)
    q_f32 = (w_f32 / scale_f32.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    w_fp8 = q_f32.reshape(out_dim, in_dim).to(torch.float8_e4m3fn)
    scale_e8m0 = scale_u8.view(torch.float8_e8m0fnu)
    return w_fp8, scale_e8m0


def _fake_quant_fp8_block32_nograd_eager(
    w: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    use_2d: bool = True,
) -> torch.Tensor:
    out_dim, in_dim = w.shape
    b_in = in_dim // block_size
    if use_2d:
        b_out = out_dim // block_size
        w_f32 = w.detach().to(torch.float32).view(b_out, block_size, b_in, block_size)
        amax = w_f32.abs().amax(dim=(1, 3), keepdim=True)
        _, scale_f32 = _fast_round_scale_e8m0(amax, FP8_MAX)
        q_fp8 = (w_f32 / scale_f32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return (q_fp8.to(torch.float32) * scale_f32).reshape(out_dim, in_dim).to(w.dtype)
    w_f32 = w.detach().to(torch.float32).view(out_dim, b_in, block_size)
    amax = w_f32.abs().amax(dim=-1, keepdim=True)
    _, scale_f32 = _fast_round_scale_e8m0(amax, FP8_MAX)
    q_fp8 = (w_f32 / scale_f32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q_fp8.to(torch.float32) * scale_f32).reshape(out_dim, in_dim).to(w.dtype)


def _act_quant_fp8_nograd_eager(
    x: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    x_f32 = x.detach().to(torch.float32).reshape(-1, last_dim // block_size, block_size)
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    _, scale_f32 = _fast_round_scale_e8m0(amax, FP8_MAX)
    q_fp8 = (x_f32 / scale_f32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q_fp8.to(torch.float32) * scale_f32).reshape(orig_shape).to(x.dtype)


_COMPILED_QAT_FNS: dict[str, Any] = {}
_COMPILE_QAT_DISABLED = False


def _get_compiled_qat_fn(name: str, fn: Any) -> Any:
    global _COMPILE_QAT_DISABLED
    if _COMPILE_QAT_DISABLED or os.environ.get("DSV41_COMPILE_QAT", "1") in ("0", "false", "False"):
        return None
    compiled = _COMPILED_QAT_FNS.get(name)
    if compiled is None:
        try:
            compiled = torch.compile(fn, dynamic=True)
            _COMPILED_QAT_FNS[name] = compiled
        except Exception:  # pylint: disable=broad-exception-caught
            _COMPILE_QAT_DISABLED = True
            return None
    return compiled


def fake_quant_fp8_block32_ste(
    w: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    scale_2d: bool = True,
) -> torch.Tensor:
    """Applies Straight-Through Estimator (STE) FP8_E4M3 + UE8M0 block-32 fake-quantization to weight `w`."""
    if w.shape[-1] % block_size != 0:
        return w
    use_2d = scale_2d and (w.shape[0] % block_size == 0)
    with torch.no_grad():
        w_deq: torch.Tensor | None = None
        if w.is_cuda:
            c_fn = _get_compiled_qat_fn("fake_quant_fp8", _fake_quant_fp8_block32_nograd_eager)
            if c_fn is not None:
                try:
                    w_deq = c_fn(w, block_size, use_2d)
                except Exception:  # pylint: disable=broad-exception-caught
                    global _COMPILE_QAT_DISABLED
                    _COMPILE_QAT_DISABLED = True
                    w_deq = None
        if w_deq is None:
            w_deq = _fake_quant_fp8_block32_nograd_eager(w, block_size, use_2d)
    if not torch.is_grad_enabled() or not w.requires_grad:
        return w_deq
    return w + (w_deq - w).detach()


def act_quant_fp8_ste(
    x: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Applies Havu's dynamic in-place FP8_E4M3 + UE8M0 block-32 activation quantization (`kernel.py:act_quant`) with STE."""
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    if last_dim == 0 or last_dim % block_size != 0:
        return x
    with torch.no_grad():
        x_deq: torch.Tensor | None = None
        if x.is_cuda:
            c_fn = _get_compiled_qat_fn("act_quant_fp8", _act_quant_fp8_nograd_eager)
            if c_fn is not None:
                try:
                    x_deq = c_fn(x, block_size)
                except Exception:  # pylint: disable=broad-exception-caught
                    global _COMPILE_QAT_DISABLED
                    _COMPILE_QAT_DISABLED = True
                    x_deq = None
        if x_deq is None:
            x_deq = _act_quant_fp8_nograd_eager(x, block_size)
    if not x.requires_grad:
        return x_deq
    return x + (x_deq - x).detach()


def fp4_act_quant_ste(
    x: torch.Tensor,
    block_size: int = 32,
    use_e8m0: bool = True,
    scale_fmt: str | None = None,
) -> torch.Tensor:
    """Applies Havu's dynamic E2M1 FP4 activation quantization (`kernel.py:fp4_act_quant_kernel`) with STE.

    Args:
      x: activation tensor whose last dimension is divisible by `block_size`.
      block_size: `32` for Indexer `q`/`k` (`UE8M0` scale), `16` for `compress_kv` (`FP8_E4M3` scale).
      use_e8m0: if True, rounds block scale to `UE8M0` power-of-two ceiling; else casts `amax / 6.0` to `float8_e4m3fn`.
      scale_fmt: optional `"ue8m0"` or `"e4m3"` string overriding `use_e8m0`.
    """
    if scale_fmt is not None:
        use_e8m0 = scale_fmt.lower() == "ue8m0"
    orig_shape = x.shape
    last_dim = orig_shape[-1]
    if last_dim == 0 or last_dim % block_size != 0:
        return x
    with torch.no_grad():
        x_f32 = x.detach().to(torch.float32).reshape(-1, last_dim // block_size, block_size)
        amax = x_f32.abs().amax(dim=-1, keepdim=True)
        if use_e8m0:
            _, scale_f32 = _fast_round_scale_e8m0(amax, FP4_MAX)
        else:
            scale_fp8 = (amax.clamp_min(FP8_AMAX_FLOOR) * (1.0 / FP4_MAX)).to(torch.float8_e4m3fn)
            scale_f32 = scale_fp8.to(torch.float32).clamp_min(1e-12)
        scaled = (x_f32 / scale_f32).clamp(-FP4_MAX, FP4_MAX)
        sign = torch.sign(scaled)
        abs_v = scaled.abs()
        # Exact nearest E2M1 grid `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}` midpoints:
        # 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 (with round-to-even at half-integers)
        q_abs = torch.where(
            abs_v <= 2.0,
            torch.round(abs_v * 2.0) * 0.5,
            torch.where(
                abs_v < 5.0,
                torch.round(abs_v),
                torch.full_like(abs_v, 6.0),
            ),
        )
        x_deq = (sign * q_abs * scale_f32).reshape(orig_shape).to(x.dtype)
    if not x.requires_grad:
        return x_deq
    return x + (x_deq - x).detach()


def unpack_uint2_torch(packed_int32: torch.Tensor) -> torch.Tensor:
    """Unpacks `int32 [..., M]` -> `uint8 [..., M * 16]` with LSB-first 2-bit values in `{0, 1, 2, 3}`."""
    shifts = SHIFTS_U2.to(packed_int32.device)
    return (
        ((packed_int32.unsqueeze(-1) >> shifts) & 3)
        .to(torch.uint8)
        .reshape(*packed_int32.shape[:-1], packed_int32.shape[-1] * 16)
    )


def pack_uint2_torch(codes_u8: torch.Tensor) -> torch.Tensor:
    """Packs `uint8 [..., M * 16]` with values in `{0, 1, 2, 3}` into `int32 [..., M]` LSB-first."""
    if codes_u8.shape[-1] % 16 != 0:
        raise ValueError(f"Last dim {codes_u8.shape[-1]} must be divisible by 16")
    shifts = SHIFTS_U2.to(codes_u8.device)
    grouped = (codes_u8.to(torch.int32) & 3).reshape(
        *codes_u8.shape[:-1], codes_u8.shape[-1] // 16, 16
    )
    packed = torch.zeros(
        grouped.shape[:-1], dtype=torch.int32, device=codes_u8.device
    )
    for b in range(16):
        packed = packed | (grouped[..., b] << shifts[b])
    return packed


def dequant_gptq_expert_tensor(
    q: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    group: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """De-quantizes a 2-bit Humming GPTQ expert matrix `(q, s, z)` into exact `BF16 [N, K]`.

    Args:
      q: `int32 [N, K // 16]` (16 `uint2` codes packed LSB-first along K).
      s: `bfloat16 [N, K // group]` (per-group scale).
      z: `int32 [N // 16, K // group]` (16 `uint2` zero-points packed LSB-first along N!).
      group: `128` for `w1`/`w3`, `64` for `w2`.
      out_dtype: target floating-point dtype (`torch.bfloat16` or `torch.float32`).
    """
    q_u2 = unpack_uint2_torch(q).to(torch.float32)  # [N, K]
    z_u2 = (
        unpack_uint2_torch(z.T.contiguous())
        .T.contiguous()
        .to(torch.float32)
    )  # [N, K // group]
    s_f32 = s.to(torch.float32)  # [N, K // group]
    n_dim, k_dim = q_u2.shape
    num_groups = k_dim // group
    w_f32 = (
        q_u2.view(n_dim, num_groups, group) - z_u2.unsqueeze(-1)
    ) * s_f32.unsqueeze(-1)
    return w_f32.reshape(n_dim, k_dim).to(out_dtype)


def quantize_uint2_humming_locked_z(
    w: torch.Tensor,
    group: int,
    z_u2: torch.Tensor,
    s_anchor: torch.Tensor,
    q_locked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantizes `[N, K]` weight into 2-bit Humming `(q_u2, s_bf16, z_u2)` preserving Havu's GPTQ `z_u2` and scale anchor.

    At `Delta W = 0` and `s_anchor = s_orig`, this is an exact identity on `(q_orig, s_orig, z_orig)`
    with zero GPTQ Hessian degradation.
    """
    n_dim, k_dim = w.shape
    num_groups = k_dim // group
    s_bf16 = s_anchor.clamp_min(1e-6).to(torch.bfloat16)
    if q_locked is not None:
        return q_locked.to(torch.uint8), s_bf16, z_u2.to(torch.uint8)
    wg = w.detach().to(torch.float32).view(n_dim, num_groups, group)
    s_f32 = s_bf16.to(torch.float32).unsqueeze(-1)
    z_f32 = z_u2.to(torch.float32).unsqueeze(-1)
    q_u2 = (
        torch.round(wg / s_f32 + z_f32)
        .clamp(0.0, 3.0)
        .to(torch.uint8)
        .reshape(n_dim, k_dim)
    )
    return q_u2, s_bf16, z_u2.to(torch.uint8)


def quantize_uint2_humming_expert_tensor(
    w: torch.Tensor,
    group: int,
    z_u2: torch.Tensor | None = None,
    s_anchor: torch.Tensor | None = None,
    q_locked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantizes `[N, K]` BF16/FP32 expert weight into packed 2-bit Humming `(q, s, z)` format.

    If `z_u2` and `s_anchor` are provided (from Havu's GPTQ checkpoint + QAT scale parameter),
    preserves `z_u2` and quantizes around `s_anchor` so that at `Delta W = 0`, `(q, s, z)` is
    100% bit-identical to Havu's Cholesky-GPTQ checkpoint. Otherwise falls back to L2 grid search.
    """
    n_dim, k_dim = w.shape
    if n_dim % 16 != 0 or k_dim % group != 0 or k_dim % 16 != 0:
        raise ValueError(
            f"Invalid expert shape {(n_dim, k_dim)} for group={group}"
        )
    if z_u2 is not None and s_anchor is not None:
        q_u2, s_bf16, z_out_u2 = quantize_uint2_humming_locked_z(
            w, group=group, z_u2=z_u2, s_anchor=s_anchor, q_locked=q_locked
        )
        q_packed = pack_uint2_torch(q_u2)
        z_packed = pack_uint2_torch(z_out_u2.T.contiguous()).T.contiguous()
        return q_packed, s_bf16, z_packed

    num_groups = k_dim // group
    wg = w.detach().to(torch.float32).view(n_dim, num_groups, group)
    w_min = wg.amin(dim=-1)
    w_max = wg.amax(dim=-1)

    best_err = torch.full(
        (n_dim, num_groups), float("inf"), dtype=torch.float32, device=w.device
    )
    best_q = torch.zeros(
        (n_dim, num_groups, group), dtype=torch.uint8, device=w.device
    )
    best_s = torch.ones(
        (n_dim, num_groups), dtype=torch.bfloat16, device=w.device
    )
    best_z = torch.zeros(
        (n_dim, num_groups), dtype=torch.uint8, device=w.device
    )

    base_s = ((w_max - w_min) / 3.0).clamp_min(1e-6)
    for z_val in range(4):
        for shrink in (1.0, 0.95, 0.90, 0.85):
            if z_val == 0:
                s_cand = (w_max.clamp_min(0.0) / 3.0).clamp_min(1e-6) * shrink
            elif z_val == 3:
                s_cand = ((-w_min).clamp_min(0.0) / 3.0).clamp_min(1e-6) * shrink
            else:
                s_cand = torch.maximum(
                    (-w_min) / float(z_val),
                    w_max / float(3 - z_val),
                ).clamp_min(1e-6) * shrink
            s_cand = torch.where(base_s < 1e-5, base_s, s_cand).to(torch.bfloat16)
            s_f32 = s_cand.to(torch.float32)
            q_cand = (
                torch.round(wg / s_f32.unsqueeze(-1) + float(z_val))
                .clamp(0.0, 3.0)
                .to(torch.uint8)
            )
            deq = (q_cand.to(torch.float32) - float(z_val)) * s_f32.unsqueeze(-1)
            err = (deq - wg).square().sum(dim=-1)
            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_s = torch.where(improved, s_cand, best_s)
            best_z = torch.where(
                improved,
                torch.full_like(best_z, z_val),
                best_z,
            )
            best_q = torch.where(improved.unsqueeze(-1), q_cand, best_q)

    q_packed = pack_uint2_torch(best_q.reshape(n_dim, k_dim))
    z_packed = pack_uint2_torch(best_z.T.contiguous()).T.contiguous()
    return q_packed, best_s, z_packed


# ==============================================================================
# 2. Pruned DeepSeek-V4.1 (`Tool E128`) Configuration & Differentiable Modules
# ==============================================================================


def _is_prime_int(n: int) -> bool:
    """Fast deterministic primality check for Engram prime bucket moduli (`n ~ 1.6e7`)."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Smallest prime above `start` not in `seen_primes` (`inference/engram.py:find_next_prime`)."""
    candidate = start + 1
    while not _is_prime_int(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map_from_file(tokenizer_json_path: str) -> tuple[list[int], int]:
    """Exact replica of `inference/engram.py:build_compressed_token_map` using `tokenizers.Tokenizer`."""
    from tokenizers import Regex, Tokenizer, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    if tokenizer_json_path.startswith("/cns/"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_tok = tf.name
        try:
            with gfile.Open(tokenizer_json_path, "rb") as f_in, open(tmp_tok, "wb") as f_out:
                f_out.write(f_in.read())
            backend = Tokenizer.from_file(tmp_tok)
        finally:
            if os.path.exists(tmp_tok):
                os.remove(tmp_tok)
    else:
        backend = Tokenizer.from_file(tokenizer_json_path)

    vocab_len = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * vocab_len
    for token_id in range(vocab_len):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """Exact replica of `inference/engram.py:compute_hash_multipliers`."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // max(1, tokenizer_vocab_size)) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.int64))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Exact prime bucket layout of DeepSeek-V4.1's n-gram hash tables (`inference/engram.py:EngramLayout`)."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_cfg(cls, cfg: Dsv41E128Config) -> EngramLayout | None:
        layer_ids = tuple(cfg.engram_layer_ids)
        if not layer_ids or cfg.engram_n_heads <= 0 or cfg.engram_max_ngram_size <= 1:
            return None
        max_ngram_size, n_heads = cfg.engram_max_ngram_size, cfg.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], cfg.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(cfg.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=cfg.engram_head_dim,
        )

    def head_sizes_for_layer(self, layer_hash_index: int) -> tuple[int, ...]:
        return tuple(p for per_ngram in self.primes[layer_hash_index] for p in per_ngram)


class NgramHashStateModule(nn.Module):
    """Stateless & KV-cached n-gram hash computer matching `inference/engram.py:NgramHashState`."""

    DEAD = -1

    def __init__(
        self,
        cfg: Dsv41E128Config,
        layout: EngramLayout,
        token_map: list[int] | None = None,
    ):
        super().__init__()
        self.layout = layout
        if token_map is None:
            token_map = [i % max(1, cfg.engram_compressed_vocab_size) for i in range(cfg.vocab_size)]
            vocab_size = cfg.engram_compressed_vocab_size
        else:
            vocab_size = len(set(token_map))
        self.pad_id = int(token_map[min(cfg.engram_pad_id, len(token_map) - 1)])
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer(
            "primes", torch.tensor(layout.primes, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "offsets", torch.tensor(np.array(offsets), dtype=torch.int64), persistent=False
        )
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False
        )

    def set_token_map(self, token_map: list[int], pad_token_id: int = 2) -> None:
        device = self.token_map.device
        self.pad_id = int(token_map[min(pad_token_id, len(token_map) - 1)])
        self.token_map = torch.tensor(token_map, dtype=torch.int64, device=device)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_compressed: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes `[B, L, n_engram_layers, n_hash_cols]` int64 hash IDs for `input_ids [B, L]`.

        If `prefix_compressed` (`[B, P]` of compressed token IDs before `input_ids`) is given,
        uses the last `max_ngram_size - 1` tokens of `prefix_compressed` for seamless KV-cached decode.
        `token_mask` (`[B, L]` bool, False inside image spans) marks positions that take no part
        in any n-gram: they are cached as `DEAD`, so look-back stops at them (reference
        `NgramHashState`).
        """
        batch, seqlen = input_ids.shape
        safe_ids = input_ids.clamp(0, self.token_map.numel() - 1)
        compressed = self.token_map[safe_ids]  # [B, L]
        if token_mask is not None:
            compressed = torch.where(
                token_mask.to(torch.bool), compressed, torch.full_like(compressed, self.DEAD)
            )
        if prefix_compressed is not None and prefix_compressed.size(1) > 0:
            full_comp = torch.cat([prefix_compressed, compressed], dim=1)
            start_pos = prefix_compressed.size(1)
        else:
            full_comp = compressed
            start_pos = 0

        positions = torch.arange(
            start_pos, start_pos + seqlen, device=input_ids.device, dtype=torch.int64
        ).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = full_comp.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens_t = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

        products = tokens_t.unsqueeze(2) * self.multipliers  # [B, L, n_engram_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


@dataclass
class Dsv41E128Config:
    """Configuration for pruned DeepSeek-V4.1-Flash-1017 (`Tool E128`)."""

    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (
        0, 0,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_compact_kept: tuple[tuple[int, ...], ...] = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    max_seq_len: int = 65536
    routed_expert_lora_r: int = 0
    routed_expert_lora_alpha: float = 16.0
    trainable_routed_experts_dense: bool = False
    enable_qat: bool = False
    qat_trainable_expert_scales: bool = True
    qat_lock_gptq_codes: bool = False
    load_compact_engram: bool = True
    # Vision tower (DeepSeek-V4.1 VL). `vision_n_layers == 0` disables the image path entirely;
    # the released E128 image checkpoints carry `vision.*`, `aligner.*`, `image_{start,end,newline}`.
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 544 * 544
    # Raw id of `<｜deepseek_image｜>`; every position of an image span carries it in `input_ids`.
    image_token_id: int = 129264

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> Dsv41E128Config:
        tc = raw.get("text_config", raw)
        rope = tc.get("rope_scaling") or {}
        n_layers = int(tc.get("n_layers", tc.get("num_hidden_layers", 40)))
        comp_ratios = tuple(int(x) for x in tc.get("compress_ratios", cls.compress_ratios))[:n_layers]
        compact_kept_raw = tc.get("engram_compact_kept") or ()
        compact_kept = tuple(tuple(int(v) for v in row) for row in compact_kept_raw)
        return cls(
            vocab_size=int(tc.get("vocab_size", cls.vocab_size)),
            dim=int(tc.get("dim", tc.get("hidden_size", cls.dim))),
            moe_inter_dim=int(tc.get("moe_inter_dim", tc.get("moe_intermediate_size", cls.moe_inter_dim))),
            n_layers=n_layers,
            n_heads=int(tc.get("n_heads", tc.get("num_attention_heads", cls.n_heads))),
            n_routed_experts=int(tc.get("n_routed_experts", cls.n_routed_experts)),
            n_shared_experts=int(tc.get("n_shared_experts", cls.n_shared_experts)),
            n_activated_experts=int(tc.get("n_activated_experts", tc.get("num_experts_per_tok", cls.n_activated_experts))),
            score_func=str(tc.get("score_func", tc.get("scoring_func", cls.score_func))),
            route_scale=float(tc.get("route_scale", tc.get("routed_scaling_factor", cls.route_scale))),
            swiglu_limit=float(tc.get("swiglu_limit", cls.swiglu_limit)),
            q_lora_rank=int(tc.get("q_lora_rank", cls.q_lora_rank)),
            head_dim=int(tc.get("head_dim", cls.head_dim)),
            rope_head_dim=int(tc.get("rope_head_dim", tc.get("qk_rope_head_dim", cls.rope_head_dim))),
            norm_eps=float(tc.get("norm_eps", tc.get("rms_norm_eps", cls.norm_eps))),
            o_groups=int(tc.get("o_groups", cls.o_groups)),
            o_lora_rank=int(tc.get("o_lora_rank", cls.o_lora_rank)),
            window_size=int(tc.get("window_size", tc.get("sliding_window", cls.window_size))),
            compress_ratios=comp_ratios,
            kv_source_layers=tuple(int(x) for x in tc.get("kv_source_layers", tc.get("kv_source_layer_ids", cls.kv_source_layers))),
            index_source_layers=tuple(int(x) for x in tc.get("index_source_layers", tc.get("index_source_layer_ids", cls.index_source_layers))),
            original_seq_len=int(tc.get("original_seq_len", rope.get("original_max_position_embeddings", cls.original_seq_len))),
            rope_theta=float(tc.get("rope_theta", cls.rope_theta)),
            compress_rope_theta=float(tc.get("compress_rope_theta", cls.compress_rope_theta)),
            rope_factor=float(tc.get("rope_factor", rope.get("factor", cls.rope_factor))),
            beta_fast=int(tc.get("beta_fast", rope.get("beta_fast", cls.beta_fast))),
            beta_slow=int(tc.get("beta_slow", rope.get("beta_slow", cls.beta_slow))),
            index_n_heads=int(tc.get("index_n_heads", cls.index_n_heads)),
            index_head_dim=int(tc.get("index_head_dim", cls.index_head_dim)),
            index_topk=int(tc.get("index_topk", cls.index_topk)),
            candidate_source_layer=int(tc.get("candidate_source_layer", tc.get("candidate_source_layer_id", cls.candidate_source_layer))),
            candidate_topk_blocks=int(tc.get("candidate_topk_blocks", cls.candidate_topk_blocks)),
            candidate_block_size=int(tc.get("candidate_block_size", cls.candidate_block_size)),
            hc_mult=int(tc.get("hc_mult", cls.hc_mult)),
            hc_sinkhorn_iters=int(tc.get("hc_sinkhorn_iters", cls.hc_sinkhorn_iters)),
            hc_eps=float(tc.get("hc_eps", cls.hc_eps)),
            engram_layer_ids=tuple(int(x) for x in tc.get("engram_layer_ids", cls.engram_layer_ids)),
            engram_num_embeddings=tuple(int(x) for x in tc.get("engram_num_embeddings", cls.engram_num_embeddings)),
            engram_compact_kept=compact_kept,
            engram_max_ngram_size=int(tc.get("engram_max_ngram_size", cls.engram_max_ngram_size)),
            engram_vocab_size=int(tc.get("engram_vocab_size", cls.engram_vocab_size)),
            engram_n_heads=int(tc.get("engram_n_heads", cls.engram_n_heads)),
            engram_head_dim=int(tc.get("engram_head_dim", cls.engram_head_dim)),
            engram_pad_id=int(tc.get("engram_pad_id", tc.get("engram_pad_token_id", cls.engram_pad_id))),
            engram_compressed_vocab_size=int(tc.get("engram_compressed_vocab_size", cls.engram_compressed_vocab_size)),
            **cls._vision_kwargs_from_json(raw),
        )

    @classmethod
    def _vision_kwargs_from_json(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Vision-tower fields from `vision_config`, unless the checkpoint is marked text-only.

        The E043 text-only export keeps a `vision_config` block in `config.json` but ships no
        `vision.*` tensors (`e043_pruning.text_only == true`), so vision stays disabled for it.
        """
        pruning = raw.get("e043_pruning") or {}
        vc = raw.get("vision_config") or {}
        if not vc or bool(pruning.get("text_only", False)):
            return {"vision_n_layers": 0}
        tc = raw.get("text_config", raw)
        # `inference/config.json` style keys first, then HF `vision_config` keys.
        return {
            "vision_n_layers": int(tc.get("vision_n_layers", vc.get("num_hidden_layers", 32))),
            "vision_dim": int(tc.get("vision_dim", vc.get("hidden_size", cls.vision_dim))),
            "vision_n_heads": int(tc.get("vision_n_heads", vc.get("num_attention_heads", cls.vision_n_heads))),
            "vision_inter_dim": int(tc.get("vision_inter_dim", vc.get("intermediate_size", cls.vision_inter_dim))),
            "vision_patch_size": int(tc.get("vision_patch_size", vc.get("patch_size", cls.vision_patch_size))),
            "vision_rope_theta": float(tc.get("vision_rope_theta", vc.get("rope_theta", cls.vision_rope_theta))),
            "vision_downsample_ratio": int(tc.get("vision_downsample_ratio", vc.get("downsample_ratio", cls.vision_downsample_ratio))),
            "vision_max_n_token": int(tc.get("vision_max_n_token", vc.get("max_image_tokens", cls.vision_max_n_token))),
            "vision_min_pixels": int(tc.get("vision_min_pixels", vc.get("min_pixels", cls.vision_min_pixels))),
            "image_token_id": int(raw.get("image_token_id", tc.get("image_token_id", cls.image_token_id))),
        }

    @classmethod
    def tiny_test_config(cls) -> Dsv41E128Config:
        """Compact 4-layer config exercising pure SWA, ratio=2 Compressor/Indexer, ratio=1 Candidate Source, and MoE."""
        return cls(
            vocab_size=256,
            dim=64,
            moe_inter_dim=64,
            n_layers=4,
            n_heads=4,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
            q_lora_rank=32,
            head_dim=32,
            rope_head_dim=16,
            o_groups=2,
            o_lora_rank=16,
            window_size=8,
            compress_ratios=(0, 2, 1, 1),
            kv_source_layers=(1, 2),
            index_source_layers=(1, 2, 3),
            original_seq_len=128,
            index_n_heads=4,
            index_head_dim=16,
            index_topk=4,
            candidate_source_layer=2,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            hc_mult=2,
            hc_sinkhorn_iters=5,
            engram_layer_ids=(1,),
            engram_num_embeddings=(96,),
            engram_compact_kept=((16, 16, 16, 16, 16, 16),),
            engram_max_ngram_size=4,
            engram_vocab_size=10,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_pad_id=2,
            engram_compressed_vocab_size=128,
            max_seq_len=128,
            routed_expert_lora_r=4,
            routed_expert_lora_alpha=8.0,
        )


class _SymmetricAllReduce(torch.autograd.Function):
    """Performs an autograd-friendly symmetric SUM all_reduce without FutureWarnings."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        ctx.group = group
        out = x.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        g = grad_output.contiguous().clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.group)
        return g, None


class _EPAllGather(torch.autograd.Function):
    """Gathers `[L_max, ...]` across all EP ranks into `[W, L_max, ...]`; backward is `reduce_scatter(SUM)`."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        x_contig = x.contiguous()
        out = torch.empty(
            (world_size, *x_contig.shape),
            dtype=x_contig.dtype,
            device=x_contig.device,
        )
        dist.all_gather_into_tensor(out, x_contig, group=group)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        group = ctx.group
        g_contig = grad_output.contiguous()
        grad_in = torch.empty(
            g_contig.shape[1:],
            dtype=g_contig.dtype,
            device=g_contig.device,
        )
        dist.reduce_scatter_tensor(
            grad_in, g_contig, op=dist.ReduceOp.SUM, group=group
        )
        return grad_in, None


class _EPReduceScatter(torch.autograd.Function):
    """Reduce-scatters `[W, L_max, D]` across all EP ranks into `[L_max, D]`; backward is `all_gather`."""

    @staticmethod
    def forward(
        ctx: Any, y_all: torch.Tensor, group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        ctx.group = group
        y_contig = y_all.contiguous()
        out = torch.empty(
            y_contig.shape[1:],
            dtype=y_contig.dtype,
            device=y_contig.device,
        )
        dist.reduce_scatter_tensor(
            out, y_contig, op=dist.ReduceOp.SUM, group=group
        )
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        group = ctx.group
        world_size = dist.get_world_size(group)
        g_contig = grad_output.contiguous()
        grad_all = torch.empty(
            (world_size, *g_contig.shape),
            dtype=g_contig.dtype,
            device=g_contig.device,
        )
        dist.all_gather_into_tensor(grad_all, g_contig, group=group)
        return grad_all, None


def _differentiable_all_reduce(
    x: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Performs an autograd-friendly all_reduce across distributed EP ranks."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        return _SymmetricAllReduce.apply(x, group)
    return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-20):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.square().mean(dim=-1, keepdim=True)
        normed = xf * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


def precompute_freqs_cis_torch(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes `(cos, sin)` YaRN rotary tables of shape `[seqlen, dim // 2]` in float32."""
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / float(dim))
    )
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return (
                dim
                * math.log(original_seq_len / (rotations * 2.0 * math.pi))
                / (2.0 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / max(high - low, 1e-3)
        ).clamp(0.0, 1.0)
        smooth = 1.0 - ramp
        freqs = freqs / factor * (1.0 - smooth) + freqs * smooth

    angles = torch.outer(
        torch.arange(seqlen, dtype=torch.float32, device=device), freqs
    )
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb_differentiable(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Differentiably rotates trailing `rope_dim` channels of `x` (`[B, S, D]` or `[B, S, H, D]`)."""
    nope = x[..., :-rope_dim]
    tail = x[..., -rope_dim:].float()
    evens = tail[..., 0::2]
    odds = tail[..., 1::2]
    if x.ndim == 3:
        c = cos.unsqueeze(0)
        s = sin.unsqueeze(0)
    else:
        c = cos.unsqueeze(0).unsqueeze(2)
        s = sin.unsqueeze(0).unsqueeze(2)
    if inverse:
        s = -s
    rot_evens = evens * c - odds * s
    rot_odds = evens * s + odds * c
    rot_tail = torch.stack([rot_evens, rot_odds], dim=-1).flatten(-2).to(x.dtype)
    return torch.cat([nope, rot_tail], dim=-1)


def hc_split_sinkhorn_differentiable(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable PyTorch implementation of `kernel.py:hc_split_sinkhorn_kernel`."""
    m = mixes.float()
    sc = hc_scale.float()
    hb = hc_base.float()
    pre = torch.sigmoid(m[..., :hc_mult] * sc[0] + hb[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        m[..., hc_mult : 2 * hc_mult] * sc[1] + hb[hc_mult : 2 * hc_mult]
    )
    comb_logits = (
        m[..., 2 * hc_mult :] * sc[2] + hb[2 * hc_mult :]
    ).unflatten(-1, (hc_mult, hc_mult))
    comb = F.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


_SINKHORN_GRAPH_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_SINKHORN_GRAPH_DISABLED = False


def _hc_graph_enabled() -> bool:
    if _SINKHORN_GRAPH_DISABLED:
        return False
    return os.environ.get("DSV41_HC_CUDA_GRAPH", "1") not in ("0", "false", "False")


@torch.no_grad()
def _sinkhorn_graphed(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """CUDA-graph replay of `hc_split_sinkhorn_differentiable` for inference.

    The eager Sinkhorn loop issues ~6 tiny kernels per iteration (x20 iterations, x2 calls per
    layer) and dominated the decode step in profiling (launch-bound). The graph replays the
    identical fp32 arithmetic from static buffers; one graph is captured per
    `(batch, hc_mult, iters, eps, device)` and self-checked against the eager result.
    Returns None (caller falls back to eager) if capture is unavailable.
    """
    global _SINKHORN_GRAPH_DISABLED
    key = (
        tuple(mixes.shape),
        mixes.dtype,
        hc_mult,
        sinkhorn_iters,
        float(eps),
        mixes.device.index,
    )
    entry = _SINKHORN_GRAPH_CACHE.get(key)
    try:
        if entry is None:
            static_in = mixes.detach().clone()
            static_scale = hc_scale.detach().float().clone()
            static_base = hc_base.detach().float().clone()
            # Warm up on a side stream (allocator state), then capture.
            side = torch.cuda.Stream(device=mixes.device)
            side.wait_stream(torch.cuda.current_stream(mixes.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    hc_split_sinkhorn_differentiable(
                        static_in, static_scale, static_base, hc_mult, sinkhorn_iters, eps
                    )
            torch.cuda.current_stream(mixes.device).wait_stream(side)
            torch.cuda.synchronize(mixes.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = hc_split_sinkhorn_differentiable(
                    static_in, static_scale, static_base, hc_mult, sinkhorn_iters, eps
                )
            # Self-check replay against eager on the captured input
            graph.replay()
            torch.cuda.synchronize(mixes.device)
            ref = hc_split_sinkhorn_differentiable(
                static_in, static_scale, static_base, hc_mult, sinkhorn_iters, eps
            )
            for o, r in zip(outs, ref):
                if not torch.allclose(o, r, atol=1e-5, rtol=1e-5):
                    raise RuntimeError("sinkhorn CUDA graph self-check mismatch")
            entry = {
                "graph": graph,
                "in": static_in,
                "scale": static_scale,
                "base": static_base,
                "outs": outs,
            }
            _SINKHORN_GRAPH_CACHE[key] = entry
            if len(_SINKHORN_GRAPH_CACHE) == 1:
                print("[dsv41] Sinkhorn CUDA-graph path enabled", flush=True)
        entry["in"].copy_(mixes)
        entry["scale"].copy_(hc_scale)
        entry["base"].copy_(hc_base)
        entry["graph"].replay()
        pre, post, comb = entry["outs"]
        return pre.clone(), post.clone(), comb.clone()
    except Exception as e:  # pylint: disable=broad-exception-caught
        _SINKHORN_GRAPH_DISABLED = True
        _SINKHORN_GRAPH_CACHE.clear()
        print(f"[dsv41] Sinkhorn CUDA-graph path disabled ({type(e).__name__}: {e})", flush=True)
        return None


class DifferentiableCompressor(nn.Module):
    """Differentiable CSA2 Compressor (`ratio=1` or `ratio=2` softmax pooling)."""

    def __init__(self, cfg: Dsv41E128Config, layer_id: int):
        super().__init__()
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.head_dim = cfg.head_dim
        self.enable_qat = cfg.enable_qat
        self.teacher_mode = False
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        w_dtype = torch.float32 if self.compress_ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False, dtype=w_dtype)
        self.wgate = (
            nn.Linear(cfg.dim, cfg.head_dim, bias=False, dtype=torch.float32)
            if self.compress_ratio > 1
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        if ratio == 1:
            use_qat = (
                self.enable_qat
                and not self.teacher_mode
                and self.wkv.weight.requires_grad
            )
            x_in = act_quant_fp8_ste(x) if use_qat else x
            return self.norm(self.wkv(x_in))
        if seqlen < ratio:
            return None
        cutoff = seqlen - (seqlen % ratio)
        xf = x[:, :cutoff].float()
        kv = self.wkv(xf).unflatten(1, (-1, ratio))
        score = self.wgate(xf).unflatten(1, (-1, ratio))
        pooled = (kv * F.softmax(score, dim=2)).sum(dim=2)
        return self.norm(pooled.to(x.dtype))


def select_candidate_blocks_torch(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level-1 block selection matching `model.py:select_candidate_blocks`."""
    width = logits.size(-1)
    pad_w = (-width) % block_size
    scores = F.pad(logits, (0, pad_w), value=-torch.inf) if pad_w else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    block_ids = torch.arange(num_blocks, device=logits.device)
    scores = scores.masked_fill(block_ids == last, torch.inf)
    k = min(topk_blocks, num_blocks)
    top = scores.topk(k, dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class DifferentiableIndexer(nn.Module):
    """CSA2 Two-Level Indexer selecting `index_topk` compressed KV positions per query."""

    def __init__(self, cfg: Dsv41E128Config, layer_id: int):
        super().__init__()
        self.owns_k = layer_id in cfg.kv_source_layers
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == cfg.candidate_source_layer
        self.uses_candidates = 0 <= cfg.candidate_source_layer < layer_id
        self.candidate_topk_blocks = cfg.candidate_topk_blocks
        self.candidate_block_size = cfg.candidate_block_size
        self.dim = cfg.dim
        self.n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.index_topk = cfg.index_topk
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.enable_qat = cfg.enable_qat
        self.teacher_mode = False

        self.wq_b = nn.Linear(
            cfg.q_lora_rank, self.n_heads * self.index_head_dim, bias=False, dtype=torch.bfloat16
        )
        self.weights_proj = nn.Linear(
            cfg.dim, self.n_heads, bias=False, dtype=torch.bfloat16
        )
        if self.owns_k:
            self.wk = nn.Linear(
                cfg.head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16
            )
            self.k_norm = RMSNorm(self.index_head_dim, cfg.norm_eps)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        offset: int,
        shared_state: dict[str, Any],
        cos_tbl: torch.Tensor,
        sin_tbl: torch.Tensor,
        query_chunk_size: int = 1024,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        total_len = start_pos + seqlen
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        comp_total = total_len // ratio
        if comp_total == 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)

        use_qat = (
            self.enable_qat
            and not self.teacher_mode
            and self.wq_b.weight.requires_grad
        )
        if self.owns_k and latent is not None and latent.size(1) > 0:
            prev_comp = shared_state.get("index_k")
            prev_comp_len = 0 if (prev_comp is None or start_pos == 0) else prev_comp.size(1)
            new_comp_len = latent.size(1)
            pos_idx = torch.arange(
                prev_comp_len * ratio,
                (prev_comp_len + new_comp_len) * ratio,
                ratio,
                device=x.device,
            )
            k = self.k_norm(self.wk(latent))
            k = apply_rotary_emb_differentiable(
                k, cos_tbl[pos_idx], sin_tbl[pos_idx], rd
            )
            if use_qat:
                k = fp4_act_quant_ste(k, block_size=FP8_BLOCK_SIZE, scale_fmt="ue8m0")
            if prev_comp is not None and start_pos > 0:
                shared_state["index_k"] = torch.cat([prev_comp, k], dim=1)
            else:
                shared_state["index_k"] = k

        qr_in = act_quant_fp8_ste(qr) if use_qat else qr
        q = self.wq_b(qr_in).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = apply_rotary_emb_differentiable(
            q, cos_tbl[start_pos:total_len], sin_tbl[start_pos:total_len], rd
        )
        if use_qat:
            q = fp4_act_quant_ste(q, block_size=FP8_BLOCK_SIZE, scale_fmt="ue8m0")
        index_k_f = shared_state["index_k"][:, :comp_total].float()
        weights = self.weights_proj(x).float() * (
            self.softmax_scale * (self.n_heads**-0.5)
        )

        topk = min(self.index_topk, comp_total)
        t_arange = torch.arange(comp_total, device=x.device).view(1, 1, comp_total)

        block_size = self.candidate_block_size
        pad_w = (-comp_total) % block_size
        num_blocks = (comp_total + pad_w) // block_size
        cand_k = min(self.candidate_topk_blocks, num_blocks)
        block_ids = torch.arange(num_blocks, device=x.device)

        cand_blocks_all = (
            shared_state.get("candidates")
            if self.uses_candidates
            else None
        )
        new_cand_blocks_chunks: list[torch.Tensor] = []
        idxs_chunks: list[torch.Tensor] = []

        for s_start in range(0, seqlen, query_chunk_size):
            s_end = min(s_start + query_chunk_size, seqlen)
            q_c = q[:, s_start:s_end].float()
            w_c = weights[:, s_start:s_end].unsqueeze(-1)
            raw_c = torch.einsum("bchd,btd->bcht", q_c, index_k_f)
            raw_c.relu_().mul_(w_c)
            score_c = raw_c.sum(dim=2)
            del raw_c

            comp_lens_c = (
                torch.arange(
                    start_pos + s_start + 1,
                    start_pos + s_end + 1,
                    device=x.device,
                )
                // ratio
            ).view(1, s_end - s_start, 1)
            score_c.masked_fill_(t_arange >= comp_lens_c, -torch.inf)
            # Deterministic tie-break for exact ties (ReLU-zeroed scores): prefer more recent positions.
            # 1e-30 is far below float32 resolution of any non-zero score, so only exact ties are affected.
            score_c.add_(t_arange.float() * _TOPK_TIE_EPS)

            if self.is_candidate_source:
                scores_b = (
                    F.pad(score_c, (0, pad_w), value=-torch.inf)
                    if pad_w
                    else score_c
                )
                scores_b = scores_b.unflatten(-1, (-1, block_size)).amax(dim=-1)
                last_b = (comp_lens_c - 1) // block_size
                scores_b = scores_b.masked_fill(block_ids == last_b, torch.inf)
                scores_b = scores_b + block_ids.float() * _TOPK_TIE_EPS
                top_b = scores_b.topk(cand_k, dim=-1)
                cand_idx_c = torch.where(
                    top_b.values > -torch.inf,
                    top_b.indices,
                    torch.full_like(top_b.indices, -1),
                ).to(torch.int32)
                new_cand_blocks_chunks.append(cand_idx_c)
            elif cand_blocks_all is not None:
                cand_idx_c = cand_blocks_all[:, s_start:s_end]
                valid_b = cand_idx_c >= 0
                safe_b = cand_idx_c.clamp_min(0).long()
                keep_b = torch.zeros(
                    (bsz, s_end - s_start, num_blocks),
                    dtype=torch.bool,
                    device=x.device,
                ).scatter_(-1, safe_b, valid_b)
                keep_mask_c = keep_b.repeat_interleave(block_size, dim=-1)[
                    ..., :comp_total
                ]
                score_c.masked_fill_(~keep_mask_c, -torch.inf)

            top_c = score_c.topk(topk, dim=-1, sorted=False)
            idxs_c = torch.where(
                (top_c.values > -torch.inf) & (top_c.indices < comp_lens_c),
                top_c.indices + offset,
                torch.full_like(top_c.indices, -1),
            ).to(torch.int32)
            idxs_chunks.append(idxs_c)

        if self.is_candidate_source:
            shared_state["candidates"] = torch.cat(
                new_cand_blocks_chunks, dim=1
            )

        return torch.cat(idxs_chunks, dim=1)

    @torch.no_grad()
    def forward_batched_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        index_k: torch.Tensor | None,
        comp_len: torch.Tensor,
        comp_total: int,
        shared: dict[str, Any],
        cos_tbl: torch.Tensor,
        sin_tbl: torch.Tensor,
        pos_t: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised 1-token indexer over `B` streams with per-stream compressed lengths.

        `index_k: [B, cap, d]` is the padded batched index-key buffer, `comp_len: [B]` the
        number of valid compressed positions per stream and `comp_total = max(comp_len)`.
        Returns raw compressed slot indices `[B, 1, topk]` (`-1` = invalid), i.e. WITHOUT the
        sliding-window offset used by the single-stream path.
        """
        bsz = x.size(0)
        rd = self.rope_head_dim
        if comp_total <= 0 or index_k is None:
            return torch.empty(bsz, 1, 0, dtype=torch.int32, device=x.device)
        use_qat = (
            self.enable_qat
            and not self.teacher_mode
            and self.wq_b.weight.requires_grad
        )

        qr_in = act_quant_fp8_ste(qr) if use_qat else qr
        q = self.wq_b(qr_in).unflatten(-1, (self.n_heads, self.index_head_dim))
        cos_b = cos_tbl[pos_t].unsqueeze(1)
        sin_b = sin_tbl[pos_t].unsqueeze(1)
        q = _apply_rope_at_positions(q, cos_b, sin_b, rd)
        if use_qat:
            q = fp4_act_quant_ste(q, block_size=FP8_BLOCK_SIZE, scale_fmt="ue8m0")
        ik_f = index_k[:, :comp_total].float()  # [B, T, d]
        weights = self.weights_proj(x).float() * (
            self.softmax_scale * (self.n_heads**-0.5)
        )  # [B, 1, Hi]

        raw = torch.einsum("bchd,btd->bcht", q.float(), ik_f)
        raw.relu_().mul_(weights.unsqueeze(-1))
        score = raw.sum(dim=2)  # [B, 1, T]
        del raw
        t_ar = torch.arange(comp_total, device=x.device).view(1, 1, comp_total)
        cl = comp_len.view(bsz, 1, 1)
        score.masked_fill_(t_ar >= cl, -torch.inf)
        # Same deterministic exact-tie break as the single-stream path
        score.add_(t_ar.float() * _TOPK_TIE_EPS)

        topk = min(self.index_topk, comp_total)
        block_size = self.candidate_block_size
        pad_w = (-comp_total) % block_size
        num_blocks = (comp_total + pad_w) // block_size
        cand_k = min(self.candidate_topk_blocks, num_blocks)
        block_ids = torch.arange(num_blocks, device=x.device)

        if self.is_candidate_source:
            scores_b = F.pad(score, (0, pad_w), value=-torch.inf) if pad_w else score
            scores_b = scores_b.unflatten(-1, (-1, block_size)).amax(dim=-1)
            last_b = (cl - 1) // block_size
            scores_b = scores_b.masked_fill(block_ids == last_b, torch.inf)
            scores_b = scores_b + block_ids.float() * _TOPK_TIE_EPS
            top_b = scores_b.topk(cand_k, dim=-1)
            shared["cand"] = torch.where(
                top_b.values > -torch.inf,
                top_b.indices,
                torch.full_like(top_b.indices, -1),
            ).to(torch.int32)
        elif self.uses_candidates and shared.get("cand") is not None:
            cand = shared["cand"]
            valid_b = cand >= 0
            safe_b = cand.clamp_min(0).long()
            keep_b = torch.zeros(
                (bsz, 1, num_blocks), dtype=torch.bool, device=x.device
            ).scatter_(-1, safe_b, valid_b)
            keep_mask = keep_b.repeat_interleave(block_size, dim=-1)[..., :comp_total]
            score.masked_fill_(~keep_mask, -torch.inf)

        top = score.topk(topk, dim=-1, sorted=False)
        return torch.where(
            (top.values > -torch.inf) & (top.indices < cl),
            top.indices,
            torch.full_like(top.indices, -1),
        ).to(torch.int32)


def _sparse_attn_single_chunk(
    q_chunk: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    idx_chunk: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Computes sparse attention for a single query chunk `[B, C, H, D]`."""
    bsz, _, n_heads, _ = q_chunk.shape
    num_kv = kv.size(1)
    sink = attn_sink.float().view(1, 1, n_heads, 1)
    valid = idx_chunk >= 0
    safe_idx = idx_chunk.clamp(0, max(0, num_kv - 1)).long()

    b_idx = torch.arange(bsz, device=q_chunk.device).view(bsz, 1, 1)
    kv_g = kv[b_idx, safe_idx].to(q_chunk.dtype)  # [B, C, K, D]

    scores = (
        torch.einsum("bchd,bckd->bchk", q_chunk.float(), kv_g.float())
        * softmax_scale
    )
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    scores_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    probs = torch.exp(scores - scores_max)
    sum_exp = probs.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max)
    weights = (probs / sum_exp).to(q_chunk.dtype)
    return torch.einsum("bchk,bckd->bchd", weights.float(), kv_g.float()).to(
        q_chunk.dtype
    )


def differentiable_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    query_chunk_size: int = 512,
) -> torch.Tensor:
    """Memory-efficient differentiable sparse attention over gathered `topk_idxs` with `attn_sink`.

    Processes queries in chunks of `query_chunk_size` and checkpoints each chunk when
    gradients are enabled so only a single chunk (`< 0.8 GiB` instead of `56.6 GiB`
    at `S = 36,864`) is materialized in autograd memory at any point during backward.
    """
    _, seqlen, _, _ = q.shape
    use_chunk_ckpt = torch.is_grad_enabled() and (
        q.requires_grad or kv.requires_grad or attn_sink.requires_grad
    )
    outputs = []

    for start in range(0, seqlen, query_chunk_size):
        end = min(start + query_chunk_size, seqlen)
        q_chunk = q[:, start:end]
        idx_chunk = topk_idxs[:, start:end]
        if use_chunk_ckpt:
            out_chunk = torch_ckpt(
                lambda qc, kvc, sc, ic=idx_chunk: _sparse_attn_single_chunk(
                    qc, kvc, sc, ic, softmax_scale
                ),
                q_chunk,
                kv,
                attn_sink,
                use_reentrant=True,
            )
        else:
            out_chunk = _sparse_attn_single_chunk(
                q_chunk, kv, attn_sink, idx_chunk, softmax_scale
            )
        outputs.append(out_chunk)

    return torch.cat(outputs, dim=1)


def _apply_rope_at_positions(
    x: torch.Tensor,
    cos_b: torch.Tensor,
    sin_b: torch.Tensor,
    rope_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotates trailing `rope_dim` channels of `x` (`[B, 1, D]` or `[B, 1, H, D]`) with per-stream tables.

    `cos_b` / `sin_b` are `[B, 1, rope_dim // 2]` (one position per stream), matching
    `apply_rotary_emb_differentiable` numerically.
    """
    nope = x[..., :-rope_dim]
    tail = x[..., -rope_dim:].float()
    evens = tail[..., 0::2]
    odds = tail[..., 1::2]
    if x.ndim == 3:
        c, s = cos_b, sin_b
    else:
        c, s = cos_b.unsqueeze(2), sin_b.unsqueeze(2)
    if inverse:
        s = -s
    rot_evens = evens * c - odds * s
    rot_odds = evens * s + odds * c
    rot_tail = torch.stack([rot_evens, rot_odds], dim=-1).flatten(-2).to(x.dtype)
    return torch.cat([nope, rot_tail], dim=-1)


def _sparse_attn_two_source(
    q: torch.Tensor,
    swa_kv: torch.Tensor,
    win_valid: torch.Tensor,
    ckv: torch.Tensor | None,
    comp_idxs: torch.Tensor | None,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Batched 1-token sparse attention over the sliding window plus gathered compressed KV.

    `q: [B, 1, H, D]`, `swa_kv: [B, W, D]` (right-aligned), `win_valid: [B, W]`,
    `ckv: [B, cap, D]` padded compressed KV and `comp_idxs: [B, 1, K]` raw slots (`-1` = invalid).
    Numerically identical to `_sparse_attn_single_chunk` over `cat([window, compressed])`.
    """
    bsz, _, n_heads, _ = q.shape
    kv_all = swa_kv.unsqueeze(1)  # [B, 1, W, D]
    valid = win_valid.unsqueeze(1)  # [B, 1, W]
    if ckv is not None and comp_idxs is not None and comp_idxs.size(-1) > 0:
        cvalid = comp_idxs >= 0
        safe = comp_idxs.clamp(0, max(0, ckv.size(1) - 1)).long()
        b_idx = torch.arange(bsz, device=q.device).view(bsz, 1, 1)
        kv_c = ckv[b_idx, safe]  # [B, 1, K, D]
        kv_all = torch.cat([kv_all, kv_c.to(kv_all.dtype)], dim=2)
        valid = torch.cat([valid, cvalid], dim=2)
    sink = attn_sink.float().view(1, 1, n_heads, 1)
    scores = torch.einsum("bchd,bckd->bchk", q.float(), kv_all.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    scores_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    probs = torch.exp(scores - scores_max)
    sum_exp = probs.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max)
    weights = (probs / sum_exp).to(q.dtype)
    return torch.einsum("bchk,bckd->bchd", weights.float(), kv_all.float()).to(q.dtype)


class DifferentiableAttention(nn.Module):
    """Differentiable DeepSeek-V4.1 SWA + CSA2 Latent Attention."""

    def __init__(self, layer_id: int, cfg: Dsv41E128Config):
        super().__init__()
        self.layer_id = layer_id
        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.n_groups = cfg.o_groups
        self.window_size = cfg.window_size
        self.compress_ratio = cfg.compress_ratios[layer_id]
        self.softmax_scale = cfg.head_dim**-0.5
        self.enable_qat = cfg.enable_qat
        self.teacher_mode = False

        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.q_norm = RMSNorm(self.q_lora_rank, cfg.norm_eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=torch.bfloat16)
        self.kv_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo_b = nn.Linear(
            self.n_groups * self.o_lora_rank, self.dim, bias=False, dtype=torch.bfloat16
        )

        self.is_kv_source = layer_id in cfg.kv_source_layers
        self.is_index_source = layer_id in cfg.index_source_layers
        self.compressor = (
            DifferentiableCompressor(cfg, layer_id) if self.is_kv_source else None
        )
        self.indexer = (
            DifferentiableIndexer(cfg, layer_id) if self.is_index_source else None
        )

    def _window_topk_idxs(
        self,
        bsz: int,
        seqlen: int,
        device: torch.device,
        start_pos: int = 0,
        swa_buf_len: int | None = None,
    ) -> torch.Tensor:
        win = self.window_size
        total_len = start_pos + seqlen
        buf_len = total_len if swa_buf_len is None else swa_buf_len
        w = min(buf_len, win)
        # Global position of each query in [start_pos, total_len - 1]
        global_end = torch.arange(
            start_pos, total_len, device=device, dtype=torch.int32
        ).unsqueeze(1)
        # Offset into swa_kv buffer (whose last position corresponds to global_end = total_len - 1)
        buf_shift = total_len - buf_len
        local_end = global_end - buf_shift
        idxs = (local_end - win + 1).clamp_min(0) + torch.arange(
            w, device=device, dtype=torch.int32
        ).unsqueeze(0)
        idxs = torch.where((idxs > local_end) | (idxs < 0), -1, idxs)
        return idxs.unsqueeze(0).expand(bsz, seqlen, w)

    def forward(
        self,
        x: torch.Tensor,
        shared_state: dict[str, Any],
        cos_swa: torch.Tensor,
        sin_swa: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
        layer_cache: dict[str, Any] | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        if layer_cache is not None or (not self.wq_a.weight.requires_grad and not x.requires_grad):
            with torch.no_grad():
                return self._forward_impl(
                    x,
                    shared_state,
                    cos_swa,
                    sin_swa,
                    cos_comp,
                    sin_comp,
                    layer_cache=layer_cache,
                    start_pos=start_pos,
                )
        return self._forward_impl(
            x,
            shared_state,
            cos_swa,
            sin_swa,
            cos_comp,
            sin_comp,
            layer_cache=layer_cache,
            start_pos=start_pos,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        shared_state: dict[str, Any],
        cos_swa: torch.Tensor,
        sin_swa: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
        layer_cache: dict[str, Any] | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        total_len = start_pos + seqlen
        ratio = self.compress_ratio
        cos_tbl = cos_comp if ratio > 0 else cos_swa
        sin_tbl = sin_comp if ratio > 0 else sin_swa
        rd = self.rope_head_dim
        use_qat = (
            self.enable_qat
            and not self.teacher_mode
            and self.wq_a.weight.requires_grad
        )

        x_in = act_quant_fp8_ste(x) if use_qat else x
        qr = self.q_norm(self.wq_a(x_in))
        qr_in = act_quant_fp8_ste(qr) if use_qat else qr
        q = self.wq_b(qr_in).unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rotary_emb_differentiable(
            q, cos_tbl[start_pos:total_len], sin_tbl[start_pos:total_len], rd
        )

        kv_new = self.kv_norm(self.wkv(x_in))
        kv_new = apply_rotary_emb_differentiable(
            kv_new, cos_tbl[start_pos:total_len], sin_tbl[start_pos:total_len], rd
        )
        if use_qat:
            kv_new = act_quant_fp8_ste(kv_new)

        if layer_cache is not None:
            prev_swa = layer_cache.get("swa_kv")
            if prev_swa is not None and start_pos > 0:
                keep_prev = max(0, self.window_size - 1)
                if prev_swa.size(1) > keep_prev:
                    prev_swa = prev_swa[:, -keep_prev:]
                kv = torch.cat([prev_swa, kv_new], dim=1)
            else:
                kv = kv_new
            # Keep only trailing window_size tokens in cache for subsequent decode steps
            layer_cache["swa_kv"] = kv[:, -self.window_size:]
        else:
            kv = kv_new

        swa_buf_len = kv.size(1)
        win_idxs = self._window_topk_idxs(
            bsz, seqlen, x.device, start_pos=start_pos, swa_buf_len=swa_buf_len
        )

        if ratio > 0:
            latent = None
            if self.is_kv_source:
                assert self.compressor is not None
                if layer_cache is not None and ratio > 1:
                    prev_pending = layer_cache.get("pending_x")
                    x_comp_in = (
                        torch.cat([prev_pending, x], dim=1)
                        if (prev_pending is not None and start_pos > 0)
                        else x
                    )
                    rem = x_comp_in.size(1) % ratio
                    layer_cache["pending_x"] = (
                        x_comp_in[:, -rem:] if rem > 0 else x_comp_in[:, :0]
                    )
                    latent = self.compressor(x_comp_in)
                else:
                    latent = self.compressor(x)
                if layer_cache is not None and start_pos > 0:
                    shared_state["compress_kv"] = layer_cache.get("compress_kv")
                    shared_state["index_k"] = layer_cache.get("index_k")

            if self.is_index_source:
                assert self.indexer is not None
                comp_idxs = self.indexer(
                    x,
                    qr,
                    latent,
                    swa_buf_len,
                    shared_state,
                    cos_tbl,
                    sin_tbl,
                    start_pos=start_pos,
                )
                shared_state["topk_idxs"] = comp_idxs
            else:
                comp_idxs = shared_state["topk_idxs"]

            if self.is_kv_source:
                prev_ckv = (
                    layer_cache.get("compress_kv")
                    if (layer_cache is not None and start_pos > 0)
                    else None
                )
                prev_ckv_len = 0 if prev_ckv is None else prev_ckv.size(1)
                if latent is not None and latent.size(1) > 0:
                    new_ckv_len = latent.size(1)
                    pos_idx = torch.arange(
                        prev_ckv_len * ratio,
                        (prev_ckv_len + new_ckv_len) * ratio,
                        ratio,
                        device=x.device,
                    )
                    rot_lat = apply_rotary_emb_differentiable(
                        latent, cos_tbl[pos_idx], sin_tbl[pos_idx], rd
                    )
                    if use_qat:
                        rot_lat = fp4_act_quant_ste(
                            rot_lat, block_size=16, scale_fmt="e4m3"
                        )
                    if prev_ckv is not None:
                        shared_state["compress_kv"] = torch.cat(
                            [prev_ckv, rot_lat], dim=1
                        )
                    else:
                        shared_state["compress_kv"] = rot_lat
                elif prev_ckv is not None:
                    shared_state["compress_kv"] = prev_ckv
                else:
                    shared_state["compress_kv"] = kv[:, :0]

                if layer_cache is not None:
                    layer_cache["compress_kv"] = shared_state["compress_kv"]
                    layer_cache["index_k"] = shared_state.get("index_k")

            compress_kv = shared_state["compress_kv"]
            if compress_kv is not None and compress_kv.size(1) > 0:
                full_kv = torch.cat([kv, compress_kv], dim=1)
                topk_idxs = torch.cat([win_idxs, comp_idxs], dim=-1)
            else:
                full_kv = kv
                topk_idxs = win_idxs
        else:
            full_kv = kv
            topk_idxs = win_idxs

        o = differentiable_sparse_attn(
            q, full_kv, self.attn_sink, topk_idxs, self.softmax_scale
        )
        o = apply_rotary_emb_differentiable(
            o,
            cos_tbl[start_pos:total_len],
            sin_tbl[start_pos:total_len],
            rd,
            inverse=True,
        )
        if use_qat:
            o = act_quant_fp8_ste(o.flatten(2)).view(bsz, seqlen, self.n_groups, -1)
        else:
            o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a_w = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o_proj = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a_w.float()).to(x.dtype)
        o_proj_flat = o_proj.flatten(2)
        if use_qat:
            o_proj_flat = act_quant_fp8_ste(o_proj_flat)
        return self.wo_b(o_proj_flat)

    # ------------------------------------------------------------------
    # Fully vectorised batched 1-token decode across `B` streams.
    #
    # State layout (per layer, built by `begin_batched_state` from the per-stream
    # KV caches and written back by `end_batched_state`):
    #   "swa"    : [B, W, hd]        right-aligned sliding-window KV (newest at W-1)
    #   "ckv"    : [B, cap, hd]      padded compressed KV   (kv-source layers only)
    #   "ikey"   : [B, cap, idx_hd]  padded index keys      (kv-source layers owning k)
    #   "pend_x" : [B, ratio-1, dim] pending (not yet compressed) tokens (ratio > 1)
    # Per-stream validity is derived deterministically from the token positions:
    #   comp_len_b = (pos_b + 1) // ratio   and   win_valid_b = min(pos_b + 1, W).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def begin_batched_state(
        self,
        layer_caches: list[dict[str, Any]],
        positions: list[int],
        cap: int,
        device: torch.device,
        max_steps: int = 64,
    ) -> dict[str, Any]:
        """Builds the padded batched decode state for this layer from `B` per-stream caches.

        Also allocates per-step history buffers (`kv_hist`, and `x_hist` for ratio>1 kv-source
        layers) so `end_batched_state` can roll a stream back to any earlier step of the session
        (streams that hit EOS / a stop string mid-chunk keep an exact prefix cache).
        """
        bsz = len(layer_caches)
        w_size = self.window_size
        dtype = self.wkv.weight.dtype
        swa = torch.zeros((bsz, w_size, self.head_dim), dtype=dtype, device=device)
        for b, lc in enumerate(layer_caches):
            prev = lc.get("swa_kv")
            if prev is not None and positions[b] > 0 and prev.size(1) > 0:
                v = min(int(prev.size(1)), w_size)
                swa[b, w_size - v :] = prev[0, -v:].to(dtype)
        state: dict[str, Any] = {
            "swa": swa,
            "cap": cap,
            "kv_hist": torch.zeros(
                (bsz, max(1, max_steps), self.head_dim), dtype=dtype, device=device
            ),
            "x_hist": None,
        }

        ratio = self.compress_ratio
        if ratio > 1 and self.is_kv_source:
            state["x_hist"] = torch.zeros(
                (bsz, max(1, max_steps), self.dim), dtype=dtype, device=device
            )
        if ratio > 0 and self.is_kv_source:
            ckv = torch.zeros((bsz, cap, self.head_dim), dtype=dtype, device=device)
            owns_k = self.indexer is not None and self.indexer.owns_k
            ikey = (
                torch.zeros(
                    (bsz, cap, self.indexer.index_head_dim), dtype=dtype, device=device
                )
                if owns_k
                else None
            )
            pend_x = (
                torch.zeros((bsz, ratio - 1, self.dim), dtype=dtype, device=device)
                if ratio > 1
                else None
            )
            for b, lc in enumerate(layer_caches):
                if positions[b] <= 0:
                    continue
                n_c = positions[b] // ratio
                prev_ckv = lc.get("compress_kv")
                if prev_ckv is not None and n_c > 0:
                    assert prev_ckv.size(1) >= n_c, (prev_ckv.shape, n_c)
                    ckv[b, :n_c] = prev_ckv[0, :n_c].to(dtype)
                if ikey is not None:
                    prev_ik = lc.get("index_k")
                    if prev_ik is not None and n_c > 0:
                        assert prev_ik.size(1) >= n_c, (prev_ik.shape, n_c)
                        ikey[b, :n_c] = prev_ik[0, :n_c].to(dtype)
                if pend_x is not None:
                    r = positions[b] % ratio
                    prev_px = lc.get("pending_x")
                    if r > 0:
                        assert prev_px is not None and prev_px.size(1) == r, (
                            None if prev_px is None else prev_px.shape,
                            r,
                        )
                        pend_x[b, :r] = prev_px[0].to(dtype)
            state.update({"ckv": ckv, "ikey": ikey, "pend_x": pend_x})
        return state

    @torch.no_grad()
    def grow_batched_state(self, state: dict[str, Any], new_cap: int) -> None:
        if new_cap <= int(state["cap"]):
            return
        extra = new_cap - int(state["cap"])
        for key in ("ckv", "ikey"):
            buf = state.get(key)
            if buf is not None:
                pad = torch.zeros(
                    (buf.size(0), extra, buf.size(2)), dtype=buf.dtype, device=buf.device
                )
                state[key] = torch.cat([buf, pad], dim=1)
        state["cap"] = new_cap

    @torch.no_grad()
    def ensure_history_capacity(self, state: dict[str, Any], steps: int) -> None:
        for key in ("kv_hist", "x_hist"):
            buf = state.get(key)
            if buf is not None and buf.size(1) < steps:
                pad = torch.zeros(
                    (buf.size(0), steps + 32 - buf.size(1), buf.size(2)),
                    dtype=buf.dtype,
                    device=buf.device,
                )
                state[key] = torch.cat([buf, pad], dim=1)

    @torch.no_grad()
    def end_batched_state(
        self,
        state: dict[str, Any],
        layer_caches: list[dict[str, Any]],
        positions: list[int],
        begin_positions: list[int] | None = None,
        valid_steps: list[int] | None = None,
    ) -> None:
        """Writes the batched state back into the per-stream caches.

        `positions` are the new total lengths after all session steps. If `valid_steps[b]` is
        smaller than the number of steps run for stream `b`, that stream is rolled back exactly to
        `begin_positions[b] + valid_steps[b]` tokens using the per-step history buffers (the
        per-stream cache tensors are untouched until this call, so they serve as the pre-session
        snapshot).
        """
        w_size = self.window_size
        ratio = self.compress_ratio
        swa = state["swa"]
        for b, lc in enumerate(layer_caches):
            steps_run = (
                positions[b] - begin_positions[b] if begin_positions is not None else None
            )
            k = valid_steps[b] if valid_steps is not None else None
            rollback = (
                begin_positions is not None
                and k is not None
                and steps_run is not None
                and 0 <= k < steps_run
            )
            if not rollback:
                v = min(positions[b], w_size)
                lc["swa_kv"] = swa[b : b + 1, w_size - v :].clone()
                if ratio > 0 and self.is_kv_source:
                    n_c = positions[b] // ratio
                    lc["compress_kv"] = state["ckv"][b : b + 1, :n_c].clone()
                    if state.get("ikey") is not None:
                        lc["index_k"] = state["ikey"][b : b + 1, :n_c].clone()
                    if state.get("pend_x") is not None:
                        r = positions[b] % ratio
                        lc["pending_x"] = state["pend_x"][b : b + 1, :r].clone()
                continue

            # ---- exact rollback to begin_positions[b] + k ----
            assert begin_positions is not None and k is not None
            new_pos = begin_positions[b] + k
            orig_swa = lc.get("swa_kv")
            parts = []
            if orig_swa is not None and begin_positions[b] > 0 and orig_swa.size(1) > 0:
                parts.append(orig_swa[0].to(swa.dtype))
            if k > 0:
                parts.append(state["kv_hist"][b, :k])
            if parts:
                win = torch.cat(parts, dim=0)[-w_size:]
            else:
                win = swa.new_zeros((0, swa.size(2)))
            lc["swa_kv"] = win.unsqueeze(0).clone()
            if ratio > 0 and self.is_kv_source:
                n_c = new_pos // ratio
                lc["compress_kv"] = state["ckv"][b : b + 1, :n_c].clone()
                if state.get("ikey") is not None:
                    lc["index_k"] = state["ikey"][b : b + 1, :n_c].clone()
                if state.get("pend_x") is not None:
                    r = new_pos % ratio
                    x_hist = state["x_hist"]
                    assert x_hist is not None
                    orig_pend = lc.get("pending_x")
                    pend_parts = []
                    if orig_pend is not None and begin_positions[b] > 0 and orig_pend.size(1) > 0:
                        pend_parts.append(orig_pend[0].to(x_hist.dtype))
                    if k > 0:
                        pend_parts.append(x_hist[b, :k])
                    if r > 0:
                        pend_all = torch.cat(pend_parts, dim=0)
                        assert pend_all.size(0) >= r, (pend_all.shape, r)
                        lc["pending_x"] = pend_all[-r:].unsqueeze(0).clone()
                    else:
                        lc["pending_x"] = x_hist.new_zeros((1, 0, x_hist.size(2)))

    @torch.no_grad()
    def forward_batched_decode(
        self,
        x: torch.Tensor,
        state: dict[str, Any],
        shared: dict[str, Any],
        cos_swa: torch.Tensor,
        sin_swa: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
        pos_t: torch.Tensor,
        win_valid: torch.Tensor,
        max_pos: int,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """Vectorised batched 1-token decode (`x: [B, 1, dim]`, `pos_t: [B]` new-token positions).

        No per-stream Python loops: every op runs once per layer over the full batch.
        `shared` carries the batched compressed-KV / index-key / top-k state between the
        kv-source, index-source and consumer layers of one decode step.
        """
        bsz, seqlen, _ = x.shape
        assert seqlen == 1
        ratio = self.compress_ratio
        cos_tbl = cos_comp if ratio > 0 else cos_swa
        sin_tbl = sin_comp if ratio > 0 else sin_swa
        rd = self.rope_head_dim
        use_qat = (
            self.enable_qat
            and not self.teacher_mode
            and self.wq_a.weight.requires_grad
        )
        dev = x.device
        b_ar = torch.arange(bsz, device=dev)

        x_in = act_quant_fp8_ste(x) if use_qat else x
        qr = self.q_norm(self.wq_a(x_in))
        qr_in = act_quant_fp8_ste(qr) if use_qat else qr
        q = self.wq_b(qr_in).unflatten(-1, (self.n_heads, self.head_dim))
        kv_new = self.kv_norm(self.wkv(x_in))

        cos_b = cos_tbl[pos_t].unsqueeze(1)  # [B, 1, rd // 2]
        sin_b = sin_tbl[pos_t].unsqueeze(1)
        q = _apply_rope_at_positions(q, cos_b, sin_b, rd)
        kv_new = _apply_rope_at_positions(kv_new, cos_b, sin_b, rd)
        if use_qat:
            kv_new = act_quant_fp8_ste(kv_new)

        # Sliding window: shift left by one, newest token at slot W-1
        swa = torch.cat([state["swa"][:, 1:], kv_new.to(state["swa"].dtype)], dim=1)
        state["swa"] = swa
        # Per-step history for exact mid-session rollback (see `end_batched_state`)
        kv_hist = state.get("kv_hist")
        if kv_hist is not None:
            kv_hist[:, step_idx] = kv_new[:, 0].to(kv_hist.dtype)
        x_hist = state.get("x_hist")
        if x_hist is not None:
            x_hist[:, step_idx] = x[:, 0].to(x_hist.dtype)

        comp_idxs: torch.Tensor | None = None
        ckv: torch.Tensor | None = None
        if ratio > 0:
            if self.is_kv_source:
                assert self.compressor is not None
                if ratio == 1:
                    latent = self.compressor(x)
                    slot = pos_t
                else:
                    pend_x = state["pend_x"]
                    x_full = torch.cat([pend_x, x.to(pend_x.dtype)], dim=1)  # [B, ratio, dim]
                    latent = self.compressor(x_full)
                    p_cnt = pos_t % ratio
                    commit = p_cnt == (ratio - 1)
                    nxt = (pos_t + 1) // ratio
                    slot = torch.where(commit, nxt - 1, nxt)
                    # Non-committing streams keep the new token pending (chronological order)
                    pend_x[b_ar, p_cnt.clamp(max=ratio - 2)] = x[:, 0].to(pend_x.dtype)
                assert latent is not None and latent.size(1) == 1
                cpos = slot * ratio
                cos_c = cos_tbl[cpos].unsqueeze(1)
                sin_c = sin_tbl[cpos].unsqueeze(1)
                rot_lat = _apply_rope_at_positions(latent, cos_c, sin_c, rd)
                if use_qat:
                    rot_lat = fp4_act_quant_ste(rot_lat, block_size=16, scale_fmt="e4m3")
                ckv_buf = state["ckv"]
                # Committing streams write their new latent; others write a scratch value into
                # their next (still invalid, masked) slot which is overwritten when they commit.
                ckv_buf[b_ar, slot] = rot_lat[:, 0].to(ckv_buf.dtype)
                ikey_buf = state.get("ikey")
                if ikey_buf is not None:
                    assert self.indexer is not None
                    k = self.indexer.k_norm(self.indexer.wk(latent))
                    k = _apply_rope_at_positions(k, cos_c, sin_c, rd)
                    if use_qat:
                        k = fp4_act_quant_ste(k, block_size=FP8_BLOCK_SIZE, scale_fmt="ue8m0")
                    ikey_buf[b_ar, slot] = k[:, 0].to(ikey_buf.dtype)
                shared["ckv"] = ckv_buf
                shared["ikey"] = ikey_buf
                shared["comp_len"] = (pos_t + 1) // ratio
                shared["comp_total"] = (max_pos + 1) // ratio

            if self.is_index_source:
                assert self.indexer is not None
                comp_idxs = self.indexer.forward_batched_decode(
                    x,
                    qr,
                    shared.get("ikey"),
                    shared["comp_len"],
                    int(shared["comp_total"]),
                    shared,
                    cos_tbl,
                    sin_tbl,
                    pos_t,
                )
                shared["comp_idxs"] = comp_idxs
            else:
                comp_idxs = shared.get("comp_idxs")
            ckv = shared.get("ckv")

        o = _sparse_attn_two_source(
            q, swa, win_valid, ckv, comp_idxs, self.attn_sink, self.softmax_scale
        )
        o = _apply_rope_at_positions(o, cos_b, sin_b, rd, inverse=True)
        if use_qat:
            o = act_quant_fp8_ste(o.flatten(2)).view(bsz, 1, self.n_groups, -1)
        else:
            o = o.view(bsz, 1, self.n_groups, -1)
        wo_a_w = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o_proj = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a_w.float()).to(x.dtype)
        o_proj_flat = o_proj.flatten(2)
        if use_qat:
            o_proj_flat = act_quant_fp8_ste(o_proj_flat)
        return self.wo_b(o_proj_flat)


@torch.no_grad()
def _compute_humming_master_and_quant_nograd(
    weight: torch.Tensor,
    s_orig: torch.Tensor,
    z_orig: torch.Tensor,
    qat_scale: torch.Tensor | None,
    lora_A: torch.Tensor | None,
    lora_B: torch.Tensor | None,
    scaling: float,
    group: int,
    lock_gptq_codes: bool = False,
    d_orig_cached: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes `(d_orig_f32, w_master_f32, w_quant_f32, q_u2, s_bf16)` under `no_grad` with locked `z_orig`."""
    z0 = z_orig.detach().float().unsqueeze(-1)
    if d_orig_cached is not None and d_orig_cached.numel() > 0:
        d_orig = d_orig_cached.detach().float()
    else:
        w_init_g = weight.detach().float().unflatten(1, (-1, group))
        s0 = s_orig.detach().float().unsqueeze(-1)
        d_orig = torch.where(
            s0 > 0,
            torch.round(w_init_g / s0.clamp_min(1e-12)),
            torch.zeros_like(w_init_g),
        ).clamp(-z0, 3.0 - z0)

    s_cur = (
        qat_scale.detach().float().clamp_min(1e-7)
        if qat_scale is not None
        else s_orig.detach().float().clamp_min(1e-7)
    )
    w_scaled = (d_orig * s_cur.unsqueeze(-1)).flatten(1)
    has_lora = lora_A is not None and lora_B is not None
    if has_lora and lora_A is not None and lora_B is not None:
        if lora_A.is_cuda:
            delta = (
                lora_B.detach().to(torch.bfloat16)
                @ lora_A.detach().to(torch.bfloat16)
            ).float() * scaling
        else:
            delta = (lora_B.detach().float() @ lora_A.detach().float()) * scaling
        w_master = w_scaled + delta
    else:
        w_master = w_scaled

    s_bf16 = (
        qat_scale.detach().to(torch.bfloat16)
        if qat_scale is not None
        else s_orig.detach()
    )
    s_q = s_bf16.float().unsqueeze(-1)
    if lock_gptq_codes and not has_lora:
        q_u2 = (d_orig + z0).round().clamp(0.0, 3.0)
    else:
        w_m_g = w_master.unflatten(1, (-1, group))
        q_u2 = torch.where(
            s_q > 0,
            torch.round(w_m_g / s_q.clamp_min(1e-12) + z0).clamp(0.0, 3.0),
            z0,
        )
    w_quant = ((q_u2 - z0) * s_q).flatten(1)
    return d_orig, w_master, w_quant, q_u2.to(torch.uint8), s_bf16


@torch.no_grad()
def _compute_humming_quant_weight_bf16_nograd(
    weight: torch.Tensor,
    s_orig: torch.Tensor,
    z_orig: torch.Tensor,
    qat_scale: torch.Tensor | None,
    lora_A: torch.Tensor | None,
    lora_B: torch.Tensor | None,
    scaling: float,
    group: int,
    lock_gptq_codes: bool = False,
    d_orig_cached: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Single-buffer in-place computation of `w_quant` (`[N, K]`, `out_dtype`) under `no_grad`."""
    z0 = z_orig.detach().float().unsqueeze(-1)
    if d_orig_cached is not None and d_orig_cached.numel() > 0:
        d_orig_src: torch.Tensor = d_orig_cached.detach()
    else:
        w_init_g = weight.detach().float().unflatten(1, (-1, group))
        s0 = s_orig.detach().float().unsqueeze(-1)
        d_orig_src = torch.where(
            s0 > 0,
            torch.round(w_init_g / s0.clamp_min(1e-12)),
            torch.zeros_like(w_init_g),
        ).clamp(-z0, 3.0 - z0)

    s_bf16 = (
        qat_scale.detach().to(torch.bfloat16)
        if qat_scale is not None
        else s_orig.detach()
    )
    s_q = s_bf16.float().unsqueeze(-1)
    has_lora = lora_A is not None and lora_B is not None
    if lock_gptq_codes and not has_lora:
        return (d_orig_src.float().mul_(s_q)).flatten(1).to(out_dtype)

    s_cur = (
        qat_scale.detach().float().clamp_min(1e-7)
        if qat_scale is not None
        else s_orig.detach().float().clamp_min(1e-7)
    )
    buf_g = d_orig_src.float().mul_(s_cur.unsqueeze(-1))
    buf = buf_g.flatten(1)
    if has_lora and lora_A is not None and lora_B is not None:
        if lora_A.is_cuda:
            delta = (
                lora_B.detach().to(torch.bfloat16)
                @ lora_A.detach().to(torch.bfloat16)
            ).float()
        else:
            delta = lora_B.detach().float() @ lora_A.detach().float()
        buf.add_(delta, alpha=scaling)
    buf_g = buf.unflatten(1, (-1, group))
    s_q_safe = s_q.clamp_min(1e-12)
    buf_g.div_(s_q_safe).add_(z0).round_().clamp_(0.0, 3.0)
    buf_g = torch.where(s_q > 0, buf_g, z0)
    buf_g.sub_(z0).mul_(s_q)
    return buf_g.flatten(1).to(out_dtype)


class _HummingQATLinearFn(torch.autograd.Function):
    """Zero-weight-storage STE autograd Function for 2-bit Humming QAT + FP32 LoRA + FP32 per-group scales."""

    @staticmethod
    def forward(
        ctx: Any,
        x_in: torch.Tensor,
        weight: torch.Tensor,
        s_orig: torch.Tensor,
        z_orig: torch.Tensor,
        d_orig_cached: torch.Tensor,
        qat_scale: torch.Tensor | None,
        lora_A: torch.Tensor | None,
        lora_B: torch.Tensor | None,
        scaling: float,
        group: int,
        lock_gptq_codes: bool,
    ) -> torch.Tensor:
        w_quant_bf16 = _compute_humming_quant_weight_bf16_nograd(
            weight=weight,
            s_orig=s_orig,
            z_orig=z_orig,
            qat_scale=qat_scale,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            group=group,
            lock_gptq_codes=lock_gptq_codes,
            d_orig_cached=d_orig_cached,
            out_dtype=x_in.dtype,
        )
        ctx.save_for_backward(x_in, qat_scale, lora_A, lora_B)
        ctx.w_quant_bf16 = w_quant_bf16
        ctx.weight = weight
        ctx.s_orig = s_orig
        ctx.z_orig = z_orig
        ctx.d_orig_cached = d_orig_cached
        ctx.scaling = scaling
        ctx.group = group
        ctx.lock_gptq_codes = lock_gptq_codes
        return F.linear(x_in, w_quant_bf16)

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        x_in, qat_scale, lora_A, lora_B = ctx.saved_tensors
        w_quant_bf16: torch.Tensor | None = getattr(ctx, "w_quant_bf16", None)
        ctx.w_quant_bf16 = None
        weight: torch.Tensor = ctx.weight
        s_orig: torch.Tensor = ctx.s_orig
        z_orig: torch.Tensor = ctx.z_orig
        d_orig_cached: torch.Tensor = ctx.d_orig_cached
        scaling: float = ctx.scaling
        group: int = ctx.group
        lock_gptq_codes: bool = ctx.lock_gptq_codes

        orig_shape = x_in.shape
        x_2d = x_in.reshape(-1, orig_shape[-1])
        go_2d = grad_out.reshape(-1, grad_out.shape[-1])

        need_x = ctx.needs_input_grad[0]
        need_s = qat_scale is not None and ctx.needs_input_grad[5]
        need_a = lora_A is not None and ctx.needs_input_grad[6]
        need_b = lora_B is not None and ctx.needs_input_grad[7]

        grad_x: torch.Tensor | None = None
        if need_x:
            if w_quant_bf16 is None:
                w_quant_bf16 = _compute_humming_quant_weight_bf16_nograd(
                    weight=weight,
                    s_orig=s_orig,
                    z_orig=z_orig,
                    qat_scale=qat_scale,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    group=group,
                    lock_gptq_codes=lock_gptq_codes,
                    d_orig_cached=d_orig_cached,
                    out_dtype=x_in.dtype,
                )
            grad_x = (go_2d.to(x_in.dtype) @ w_quant_bf16).reshape(orig_shape)
        del w_quant_bf16

        use_cuda_tc = x_2d.is_cuda
        x_bf = x_2d.to(torch.bfloat16) if use_cuda_tc else None
        go_bf = go_2d.to(torch.bfloat16) if use_cuda_tc else None

        grad_s: torch.Tensor | None = None
        if need_s and qat_scale is not None:
            if d_orig_cached is not None and d_orig_cached.numel() > 0:
                d_orig_rel: torch.Tensor = d_orig_cached.detach()
            else:
                d_orig_rel = _compute_humming_master_and_quant_nograd(
                    weight=weight,
                    s_orig=s_orig,
                    z_orig=z_orig,
                    qat_scale=qat_scale,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    group=group,
                    lock_gptq_codes=lock_gptq_codes,
                )[0]
            if use_cuda_tc and x_bf is not None and go_bf is not None:
                grad_w = (go_bf.T @ x_bf).float()
            else:
                grad_w = go_2d.float().T @ x_2d.float()
            raw_grad_s = (grad_w.unflatten(1, (-1, group)) * d_orig_rel).sum(dim=-1)
            grad_s = torch.where(
                qat_scale.detach().float() >= 1e-7,
                raw_grad_s,
                torch.zeros_like(raw_grad_s),
            ).to(qat_scale.dtype)
            del grad_w

        grad_a: torch.Tensor | None = None
        grad_b: torch.Tensor | None = None
        if (need_a or need_b) and lora_A is not None and lora_B is not None:
            if use_cuda_tc and x_bf is not None and go_bf is not None:
                if need_b:
                    xa = x_bf @ lora_A.detach().to(torch.bfloat16).T
                    grad_b = ((go_bf.T @ xa).float() * scaling).to(lora_B.dtype)
                if need_a:
                    gob = go_bf @ lora_B.detach().to(torch.bfloat16)
                    grad_a = ((gob.T @ x_bf).float() * scaling).to(lora_A.dtype)
            else:
                x_f = x_2d.float()
                go_f = go_2d.float()
                if need_b:
                    xa = x_f @ lora_A.detach().float().T
                    grad_b = ((go_f.T @ xa) * scaling).to(lora_B.dtype)
                if need_a:
                    gob = go_f @ lora_B.detach().float()
                    grad_a = ((gob.T @ x_f) * scaling).to(lora_A.dtype)

        return (
            grad_x,
            None,
            None,
            None,
            None,
            grad_s,
            grad_a,
            grad_b,
            None,
            None,
            None,
        )


class LoRALinear(nn.Module):
    """2-Bit Humming QAT + FP32 LoRA Linear for Routed MoE Experts.

    Retains Havu's Cholesky-GPTQ zero-points `z_orig` (`uint8 [N, K // group]`) and
    initial scales `s_orig` (`bfloat16 [N, K // group]`), and trains:
    1. FP32 per-group scale `qat_scale` (`float32 [N, K // group]`)
    2. FP32 low-rank LoRA adapters `lora_A`, `lora_B`
    with locked-`z_orig` 2-bit Humming Straight-Through Estimation (STE) in `forward()`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        group: int = 128,
        enable_qat: bool = False,
        trainable_scale: bool = True,
        lock_gptq_codes: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.scaling = lora_alpha / float(max(1, lora_r))
        self.group = group
        self.enable_qat = enable_qat
        self.trainable_scale = trainable_scale and enable_qat
        self.lock_gptq_codes = lock_gptq_codes
        self.teacher_mode = False

        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        n_groups = in_features // group
        self.register_buffer(
            "z_orig",
            torch.ones(out_features, n_groups, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "s_orig",
            torch.ones(out_features, n_groups, dtype=torch.bfloat16) * 0.05,
            persistent=False,
        )
        self.register_buffer(
            "d_orig",
            torch.empty(0, dtype=torch.int8),
            persistent=False,
        )
        self.has_gptq_state = False

        if self.trainable_scale:
            self.qat_scale = nn.Parameter(
                torch.ones(out_features, n_groups, dtype=torch.float32) * 0.05
            )
        else:
            self.register_parameter("qat_scale", None)

        lora_dtype = torch.float32 if enable_qat else torch.bfloat16
        if lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(lora_r, in_features, dtype=lora_dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, lora_r, dtype=lora_dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @torch.no_grad()
    def _refresh_d_orig_cache(self) -> None:
        """Caches the locked GPTQ integer codes `d_orig` (int8 `[N, K]`) for the QAT STE forward.

        The cache is only materialized when `enable_qat=True`: it costs a full `[N, K]` int8 tensor per
        projection (~22.6 GB per EP rank for 640 routed experts x 3 projections of 5120x2304), and the
        non-QAT forward never reads it. Export (`_compute_master_and_quant_weight`) recomputes it
        transiently from an empty cache with identical results.
        """
        if not self.enable_qat:
            self.d_orig = torch.empty(0, dtype=torch.int8, device=self.weight.device)
            return
        w_init_g = self.weight.detach().float().unflatten(1, (-1, self.group))
        s0 = self.s_orig.detach().float().unsqueeze(-1)
        z0 = self.z_orig.detach().float().unsqueeze(-1)
        d0 = (
            torch.where(
                s0 > 0,
                torch.round(w_init_g / s0.clamp_min(1e-12)),
                torch.zeros_like(w_init_g),
            )
            .clamp(-z0, 3.0 - z0)
            .to(torch.int8)
        )
        self.d_orig = d0

    @torch.no_grad()
    def set_gptq_state(
        self,
        w_bf16: torch.Tensor,
        s: torch.Tensor,
        z_packed: torch.Tensor,
    ) -> None:
        """Initializes `weight`, `s_orig`, `z_orig`, `d_orig` (QAT only), and `qat_scale` from Havu's 2-bit GPTQ tensors."""
        if self.weight.data.shape != w_bf16.shape:
            self.weight.data = torch.empty_like(w_bf16)
        self.weight.data.copy_(w_bf16)
        if self.s_orig.shape != s.shape:
            self.s_orig = torch.empty_like(s, dtype=torch.bfloat16)
        self.s_orig.copy_(s.to(torch.bfloat16))
        z_u2 = (
            unpack_uint2_torch(z_packed.T.contiguous())
            .T.contiguous()
            .to(torch.uint8)
        )
        if self.z_orig.shape != z_u2.shape:
            self.z_orig = torch.empty_like(z_u2, dtype=torch.uint8)
        self.z_orig.copy_(z_u2)
        if self.qat_scale is not None:
            self.qat_scale.data.copy_(s.float())
        self._refresh_d_orig_cache()
        self.has_gptq_state = True

    @torch.no_grad()
    def _ensure_gptq_state(self) -> None:
        if not self.has_gptq_state:
            _, s_p, z_p = quantize_uint2_humming_expert_tensor(
                self.weight, group=self.group
            )
            self.s_orig.copy_(s_p)
            self.z_orig.copy_(
                unpack_uint2_torch(z_p.T.contiguous())
                .T.contiguous()
                .to(torch.uint8)
            )
            if self.qat_scale is not None:
                self.qat_scale.data.copy_(s_p.float())
            self._refresh_d_orig_cache()
            self.has_gptq_state = True
        elif self.enable_qat and self.d_orig.numel() == 0:
            self._refresh_d_orig_cache()

    def _compute_master_and_quant_weight(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes `(w_master_f32, w_quant_f32, q_u2, s_bf16, z_u2)` with locked `z_orig`."""
        self._ensure_gptq_state()
        _, w_master, w_quant, q_u2, s_bf16 = (
            _compute_humming_master_and_quant_nograd(
                weight=self.weight,
                s_orig=self.s_orig,
                z_orig=self.z_orig,
                qat_scale=self.qat_scale,
                lora_A=self.lora_A if self.lora_r > 0 else None,
                lora_B=self.lora_B if self.lora_r > 0 else None,
                scaling=self.scaling,
                group=self.group,
                lock_gptq_codes=self.lock_gptq_codes,
                d_orig_cached=self.d_orig,
            )
        )
        return w_master, w_quant, q_u2, s_bf16, self.z_orig

    @torch.no_grad()
    def export_humming_2bit(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exports `(q_packed, s_bf16, z_packed)` using locked `z_orig` when GPTQ state is present."""
        q_packed, s_bf16, z_packed, _ = self.export_humming_2bit_with_stats()
        return q_packed, s_bf16, z_packed

    @torch.no_grad()
    def export_humming_2bit_with_stats(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        """`export_humming_2bit` plus on-disk delta counters vs the loaded GPTQ state.

        Stats (`codes_total`, `codes_changed`, `scales_total`, `scales_changed`) count 2-bit codes
        that differ from the original GPTQ codes and BF16 group scales that differ from `s_orig`
        -- i.e. what a byte diff of the exported `gptq/` shard against the base would show.
        """
        if self.has_gptq_state:
            self._ensure_gptq_state()
            d_orig, _, _, q_u2, s_bf16 = _compute_humming_master_and_quant_nograd(
                weight=self.weight,
                s_orig=self.s_orig,
                z_orig=self.z_orig,
                qat_scale=self.qat_scale,
                lora_A=self.lora_A if self.lora_r > 0 else None,
                lora_B=self.lora_B if self.lora_r > 0 else None,
                scaling=self.scaling,
                group=self.group,
                lock_gptq_codes=self.lock_gptq_codes,
                d_orig_cached=self.d_orig,
            )
            z0 = self.z_orig.detach().float().unsqueeze(-1)
            q_orig_u2 = (d_orig + z0).round().clamp(0.0, 3.0).to(torch.uint8)
            stats = {
                "codes_total": int(q_u2.numel()),
                "codes_changed": int((q_u2 != q_orig_u2).sum().item()),
                "scales_total": int(s_bf16.numel()),
                "scales_changed": int((s_bf16 != self.s_orig).sum().item()),
            }
            q_packed = pack_uint2_torch(q_u2.flatten(1))
            z_packed = pack_uint2_torch(self.z_orig.T.contiguous()).T.contiguous()
            return q_packed, s_bf16, z_packed, stats
        q_packed, s_bf16, z_packed = quantize_uint2_humming_expert_tensor(
            self.merged_weight(), group=self.group
        )
        return q_packed, s_bf16, z_packed, {
            "codes_total": int(self.weight.numel()),
            "codes_changed": -1,
            "scales_total": int(s_bf16.numel()),
            "scales_changed": -1,
        }

    def merged_weight(self) -> torch.Tensor:
        if self.enable_qat and self.has_gptq_state:
            _, w_quant, _, _, _ = self._compute_master_and_quant_weight()
            return w_quant.to(torch.bfloat16)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            if self.lora_A.is_cuda:
                delta = (
                    self.lora_B.detach().to(torch.bfloat16)
                    @ self.lora_A.detach().to(torch.bfloat16)
                ).float() * self.scaling
            else:
                delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            return (self.weight.float() + delta).to(torch.bfloat16)
        return self.weight

    def forward(
        self, x: torch.Tensor, prequant_input: bool = False
    ) -> torch.Tensor:
        if self.teacher_mode:
            return F.linear(x, self.weight.to(x.dtype))
        if self.enable_qat:
            self._ensure_gptq_state()
            x_in = x if prequant_input else act_quant_fp8_ste(x)
            if not torch.is_grad_enabled():
                w_quant_bf16 = _compute_humming_quant_weight_bf16_nograd(
                    weight=self.weight,
                    s_orig=self.s_orig,
                    z_orig=self.z_orig,
                    qat_scale=self.qat_scale,
                    lora_A=self.lora_A if self.lora_r > 0 else None,
                    lora_B=self.lora_B if self.lora_r > 0 else None,
                    scaling=self.scaling,
                    group=self.group,
                    lock_gptq_codes=self.lock_gptq_codes,
                    d_orig_cached=self.d_orig,
                    out_dtype=x_in.dtype,
                )
                return F.linear(x_in, w_quant_bf16)
            return _HummingQATLinearFn.apply(
                x_in,
                self.weight,
                self.s_orig,
                self.z_orig,
                self.d_orig,
                self.qat_scale,
                self.lora_A if self.lora_r > 0 else None,
                self.lora_B if self.lora_r > 0 else None,
                self.scaling,
                self.group,
                self.lock_gptq_codes,
            )
        out = F.linear(x, self.weight)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (
                F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
                * self.scaling
            )
            out = out + lora_out.to(out.dtype)
        return out


class SwiGLUExpert(nn.Module):
    """SwiGLU FFN expert with native `swiglu_limit` clamping and QAT STE."""

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 10.0,
        trainable_dense: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        enable_qat: bool = False,
        qat_trainable_scales: bool = True,
        qat_lock_gptq_codes: bool = False,
    ):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.trainable_dense = trainable_dense
        self.enable_qat = enable_qat
        self.teacher_mode = False
        if trainable_dense:
            w_dtype = torch.float32 if enable_qat else torch.bfloat16
            self.w1 = nn.Linear(dim, inter_dim, bias=False, dtype=w_dtype)
            self.w2 = nn.Linear(inter_dim, dim, bias=False, dtype=w_dtype)
            self.w3 = nn.Linear(dim, inter_dim, bias=False, dtype=w_dtype)
            self.register_buffer("w1_init", torch.empty(0, dtype=torch.bfloat16), persistent=False)
            self.register_buffer("w2_init", torch.empty(0, dtype=torch.bfloat16), persistent=False)
            self.register_buffer("w3_init", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        else:
            self.w1 = LoRALinear(
                dim,
                inter_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                group=128 if dim >= 128 else max(16, dim),
                enable_qat=enable_qat,
                trainable_scale=qat_trainable_scales,
                lock_gptq_codes=qat_lock_gptq_codes,
            )
            self.w2 = LoRALinear(
                inter_dim,
                dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                group=64 if inter_dim >= 64 else max(16, inter_dim),
                enable_qat=enable_qat,
                trainable_scale=qat_trainable_scales,
                lock_gptq_codes=qat_lock_gptq_codes,
            )
            self.w3 = LoRALinear(
                dim,
                inter_dim,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                group=128 if dim >= 128 else max(16, dim),
                enable_qat=enable_qat,
                trainable_scale=qat_trainable_scales,
                lock_gptq_codes=qat_lock_gptq_codes,
            )

    @torch.no_grad()
    def snapshot_teacher_weights(self) -> None:
        if self.trainable_dense:
            self.w1_init = self.w1.weight.detach().to(dtype=torch.bfloat16).clone()
            self.w2_init = self.w2.weight.detach().to(dtype=torch.bfloat16).clone()
            self.w3_init = self.w3.weight.detach().to(dtype=torch.bfloat16).clone()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None = None,
        prequant_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        if self.trainable_dense:
            if self.teacher_mode and self.w1_init.numel() > 0:
                w1_t = self.w1_init.to(device=x.device, dtype=orig_dtype, non_blocking=True)
                w3_t = self.w3_init.to(device=x.device, dtype=orig_dtype, non_blocking=True)
                w2_t = self.w2_init.to(device=x.device, dtype=orig_dtype, non_blocking=True)
                gate = F.linear(x, w1_t).float()
                up = F.linear(x, w3_t).float()
                if self.swiglu_limit > 0:
                    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                    gate = torch.clamp(gate, max=self.swiglu_limit)
                act = F.silu(gate) * up
                if weights is not None:
                    act = act * weights.float()
                return F.linear(act.to(orig_dtype), w2_t)

            if self.enable_qat and not self.teacher_mode:
                x_in = act_quant_fp8_ste(x)
                if prequant_weights is not None:
                    w1_q, w2_q, w3_q = prequant_weights
                    w1_q = w1_q.to(orig_dtype)
                    w2_q = w2_q.to(orig_dtype)
                    w3_q = w3_q.to(orig_dtype)
                else:
                    w1_q = fake_quant_fp8_block32_ste(self.w1.weight, scale_2d=True).to(orig_dtype)
                    w3_q = fake_quant_fp8_block32_ste(self.w3.weight, scale_2d=True).to(orig_dtype)
                    w2_q = fake_quant_fp8_block32_ste(self.w2.weight, scale_2d=True).to(orig_dtype)
                gate = F.linear(x_in, w1_q).float()
                up = F.linear(x_in, w3_q).float()
                if self.swiglu_limit > 0:
                    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                    gate = torch.clamp(gate, max=self.swiglu_limit)
                act = F.silu(gate) * up
                if weights is not None:
                    act = act * weights.float()
                act_in = act_quant_fp8_ste(act.to(orig_dtype))
                return F.linear(act_in, w2_q)

            gate = F.linear(x, self.w1.weight.to(orig_dtype)).float()
            up = F.linear(x, self.w3.weight.to(orig_dtype)).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            act = F.silu(gate) * up
            if weights is not None:
                act = act * weights.float()
            return F.linear(act.to(orig_dtype), self.w2.weight.to(orig_dtype))

        if self.enable_qat and not self.teacher_mode:
            x_in = act_quant_fp8_ste(x)
            gate = self.w1(x_in, prequant_input=True).float()
            up = self.w3(x_in, prequant_input=True).float()
        else:
            gate = self.w1(x).float()
            up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        act = F.silu(gate) * up
        if weights is not None:
            act = act * weights.float()
        return self.w2(act.to(orig_dtype))


class DifferentiableGate(nn.Module):
    """DeepSeek-V4.1 128-expert router gate (`sqrtsoftplus`, `route_scale=1.5`)."""

    def __init__(self, cfg: Dsv41E128Config):
        super().__init__()
        self.n_routed_experts = cfg.n_routed_experts
        self.topk = cfg.n_activated_experts
        self.score_func = cfg.score_func
        self.gate_temp = cfg.gate_temp
        self.norm_topk_prob = cfg.norm_topk_prob
        self.route_scale = cfg.route_scale
        self.weight = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, cfg.dim, dtype=torch.bfloat16)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "bias_vl",
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`x: [n, dim]`; `image_mask: [n]` bool selects the VL routing bias (`bias_vl`) per row."""
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).clamp_min(1e-12).sqrt()
        bias = self.bias
        if image_mask is not None and torch.any(image_mask):
            bias = torch.where(
                image_mask.to(torch.bool).unsqueeze(-1), self.bias_vl.float(), self.bias.float()
            )
        indices = (scores + bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class DifferentiableMoE(nn.Module):
    """Sharded 128-Expert MoE + Shared Expert with differentiable all-reduce."""

    def __init__(
        self,
        layer_id: int,
        cfg: Dsv41E128Config,
        ep_rank: int = 0,
        ep_world_size: int = 1,
        ep_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = cfg.dim
        self.n_routed_experts = cfg.n_routed_experts
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        self.ep_group = ep_group
        self.n_local_experts = cfg.n_routed_experts // ep_world_size
        self.experts_start_idx = ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = DifferentiableGate(cfg)
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(
                    cfg.dim,
                    cfg.moe_inter_dim,
                    swiglu_limit=cfg.swiglu_limit,
                    trainable_dense=cfg.trainable_routed_experts_dense,
                    lora_r=cfg.routed_expert_lora_r,
                    lora_alpha=cfg.routed_expert_lora_alpha,
                    enable_qat=cfg.enable_qat,
                    qat_trainable_scales=cfg.qat_trainable_expert_scales,
                    qat_lock_gptq_codes=cfg.qat_lock_gptq_codes,
                )
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(cfg.n_routed_experts)
            ]
        )
        self.shared_experts = SwiGLUExpert(
            cfg.dim,
            cfg.moe_inter_dim,
            swiglu_limit=cfg.swiglu_limit,
            trainable_dense=True,
            enable_qat=cfg.enable_qat,
        )
        self.swiglu_limit = cfg.swiglu_limit
        self._stacked_w13: torch.Tensor | None = None
        self._stacked_w2: torch.Tensor | None = None
        self._stacked_qat_mode: bool | None = None
        self._shared_prequant_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _clear_stacked_eval_weights(self) -> None:
        self._stacked_w13 = None
        self._stacked_w2 = None
        self._stacked_qat_mode = None
        self._shared_prequant_cache = None

    def _can_use_stacked_eval_path(self) -> bool:
        """True only for pure-inference expert configs where unstacked weights can be freed.

        The stacked path keeps one contiguous copy of all local experts; it is only
        memory-neutral when the per-expert parameters can be released afterwards
        (no dense training, no LoRA, no trainable QAT scales, grad disabled).
        Trainer-side validation (``model.eval()`` between steps) therefore keeps
        using the per-expert loop and never duplicates expert memory.
        """
        if self.training or torch.is_grad_enabled() or self.n_local_experts <= 0:
            return False
        first_exp = self.experts[self.experts_start_idx]
        if first_exp is None or first_exp.trainable_dense or first_exp.teacher_mode:
            return False
        return (
            getattr(first_exp.w1, "lora_r", 0) == 0
            and not getattr(first_exp.w1, "trainable_scale", False)
        )

    @torch.no_grad()
    def _ensure_stacked_eval_weights(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, bool]:
        first_exp = self.experts[self.experts_start_idx]
        assert first_exp is not None
        use_qat = bool(first_exp.enable_qat and not first_exp.teacher_mode)
        if (
            self._stacked_w13 is not None
            and self._stacked_w2 is not None
            and self._stacked_w13.dtype == dtype
            and self._stacked_qat_mode == use_qat
        ):
            return self._stacked_w13, self._stacked_w2, use_qat

        w13_list: list[torch.Tensor] = []
        w2_list: list[torch.Tensor] = []
        for i in range(self.experts_start_idx, self.experts_end_idx):
            exp = self.experts[i]
            assert exp is not None
            if exp.trainable_dense:
                if exp.teacher_mode and exp.w1_init.numel() > 0:
                    w1 = exp.w1_init.to(dtype)
                    w3 = exp.w3_init.to(dtype)
                    w2 = exp.w2_init.to(dtype)
                elif use_qat:
                    w1 = fake_quant_fp8_block32_ste(exp.w1.weight, scale_2d=True).to(dtype)
                    w3 = fake_quant_fp8_block32_ste(exp.w3.weight, scale_2d=True).to(dtype)
                    w2 = fake_quant_fp8_block32_ste(exp.w2.weight, scale_2d=True).to(dtype)
                else:
                    w1 = exp.w1.weight.to(dtype)
                    w3 = exp.w3.weight.to(dtype)
                    w2 = exp.w2.weight.to(dtype)
            else:
                if exp.teacher_mode:
                    w1 = exp.w1.weight.to(dtype)
                    w3 = exp.w3.weight.to(dtype)
                    w2 = exp.w2.weight.to(dtype)
                else:
                    w1 = exp.w1.merged_weight().to(dtype)
                    w3 = exp.w3.merged_weight().to(dtype)
                    w2 = exp.w2.merged_weight().to(dtype)
            # NOTE: `w1/w2/w3` may alias the live nn.Parameter objects (e.g. `merged_weight()`
            # returns `self.weight` when QAT and LoRA are disabled and `.to(dtype)` is a no-op),
            # so nothing may be freed until `torch.stack` below has materialised real copies.
            w13_list.append(torch.cat([w1, w3], dim=0))
            w2_list.append(w2)

        stacked_w13 = torch.stack(w13_list, dim=0).contiguous()
        stacked_w2 = torch.stack(w2_list, dim=0).contiguous()
        del w13_list, w2_list, w1, w2, w3
        assert stacked_w13.dim() == 3 and stacked_w2.dim() == 3
        assert stacked_w13.size(0) == self.n_local_experts
        assert stacked_w2.shape[1:] == (self.dim, stacked_w13.size(1) // 2), stacked_w2.shape

        if self._can_use_stacked_eval_path():
            # Pure inference: release the now-redundant per-expert parameters so the
            # stacked copies are memory-neutral.
            for i in range(self.experts_start_idx, self.experts_end_idx):
                exp = self.experts[i]
                assert exp is not None
                for proj in (exp.w1, exp.w2, exp.w3):
                    dev = stacked_w13.device
                    proj.weight.data = torch.empty(0, dtype=dtype, device=dev)
                    if hasattr(proj, "d_orig"):
                        proj.d_orig = torch.empty(0, dtype=torch.int8, device=dev)
                    if hasattr(proj, "s_orig"):
                        proj.s_orig = torch.empty(0, dtype=torch.bfloat16, device=dev)
                    if hasattr(proj, "z_orig"):
                        proj.z_orig = torch.empty(0, dtype=torch.uint8, device=dev)

        self._stacked_w13 = stacked_w13
        self._stacked_w2 = stacked_w2
        self._stacked_qat_mode = use_qat
        return self._stacked_w13, self._stacked_w2, use_qat

    @torch.no_grad()
    def _moe_eval_stacked_forward(
        self,
        x_flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-sync evaluation forward for routed experts (eliminates 16 GPU->CPU syncs per layer)."""
        l_tok = x_flat.size(0)
        if l_tok == 0 or self.n_local_experts <= 0:
            return torch.zeros_like(x_flat, dtype=torch.float32)

        w13_stack, w2_stack, use_qat = self._ensure_stacked_eval_weights(x_flat.dtype)
        x_in = act_quant_fp8_ste(x_flat) if use_qat else x_flat

        if l_tok <= 64:
            # Small-batch / decode fast path: 2 batched GEMMs over all 16 local experts with zero host syncs
            h13 = torch.einsum("ld,end->len", x_in, w13_stack).float()
            gate, up = h13.chunk(2, dim=-1)
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            act = (F.silu(gate) * up).to(x_flat.dtype)
            if use_qat:
                act = act_quant_fp8_ste(act)
            out_all = torch.einsum("lei,edi->led", act, w2_stack).float()

            local_idx = indices - self.experts_start_idx
            owned = (local_idx >= 0) & (local_idx < self.n_local_experts)
            safe_local = local_idx.clamp(0, max(0, self.n_local_experts - 1)).long()
            g_idx = safe_local.unsqueeze(-1).expand(-1, -1, self.dim)
            picked = torch.gather(out_all, 1, g_idx)
            w_masked = torch.where(owned, weights.float(), 0.0).unsqueeze(-1)
            return (picked * w_masked).sum(dim=1)

        # Prefill path (`l_tok > 64`): single GPU->CPU sync of active local expert counts
        local_idx = indices - self.experts_start_idx
        owned = (local_idx >= 0) & (local_idx < self.n_local_experts)
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        for local_e in range(self.n_local_experts):
            mask = owned & (local_idx == local_e)
            idx, top = torch.where(mask)
            if idx.numel() == 0:
                continue
            x_e = x_in[idx]
            w_e = weights[idx, top].unsqueeze(-1).float()
            h13_e = F.linear(x_e, w13_stack[local_e]).float()
            gate_e, up_e = h13_e.chunk(2, dim=-1)
            if self.swiglu_limit > 0:
                up_e = torch.clamp(up_e, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate_e = torch.clamp(gate_e, max=self.swiglu_limit)
            act_e = F.silu(gate_e) * up_e * w_e
            act_in = act_quant_fp8_ste(act_e.to(x_flat.dtype)) if use_qat else act_e.to(x_flat.dtype)
            out_e = F.linear(act_in, w2_stack[local_e]).float()
            y = y.index_add(0, idx, out_e)
        return y

    def _gate_forward(
        self, x_flat: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.gate.weight.requires_grad and not self.gate.bias.requires_grad:
            with torch.no_grad():
                return self.gate(x_flat, image_mask)
        return self.gate(x_flat, image_mask)

    def _moe_body_forward(
        self,
        x_flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        ep_l_max: int,
        ep_replicated_input: bool,
        shared_prequant: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        l_local = x_flat.size(0)
        use_dp_ep = (
            self.ep_world_size > 1
            and not ep_replicated_input
            and dist.is_available()
            and dist.is_initialized()
        )
        if use_dp_ep:
            l_max = max(ep_l_max, l_local)
            if l_local < l_max:
                pad_len = l_max - l_local
                x_pad = F.pad(x_flat, (0, 0, 0, pad_len), value=0.0)
                w_pad = F.pad(weights, (0, 0, 0, pad_len), value=0.0)
                idx_pad = F.pad(indices, (0, 0, 0, pad_len), value=-1)
            else:
                x_pad = x_flat
                w_pad = weights
                idx_pad = indices

            x_all = _EPAllGather.apply(x_pad, self.ep_group).view(-1, self.dim)
            w_all = _EPAllGather.apply(w_pad, self.ep_group).view(-1, weights.size(-1))
            idx_all_buf = torch.empty(
                (self.ep_world_size, *idx_pad.shape),
                dtype=idx_pad.dtype,
                device=idx_pad.device,
            )
            dist.all_gather_into_tensor(
                idx_all_buf, idx_pad.contiguous(), group=self.ep_group
            )
            idx_all = idx_all_buf.view(-1, indices.size(-1))

            y_all = torch.zeros_like(x_all)
            ckpt_expert = bool(self.training and torch.is_grad_enabled())
            expert_chunk = 16384
            for i in range(self.experts_start_idx, self.experts_end_idx):
                idx, top = torch.where(idx_all == i)
                if idx.numel() == 0:
                    continue
                expert = self.experts[i]
                assert expert is not None
                for s in range(0, int(idx.numel()), expert_chunk):
                    idx_s = idx[s : s + expert_chunk]
                    top_s = top[s : s + expert_chunk]
                    x_e = x_all[idx_s]
                    w_e = w_all[idx_s, top_s].unsqueeze(-1)
                    if ckpt_expert and x_e.size(0) >= 512:
                        out_e = torch_ckpt(expert, x_e, w_e, use_reentrant=False)
                    else:
                        out_e = expert(x_e, w_e)
                    y_all.index_add_(0, idx_s, out_e.to(y_all.dtype))

            if self.training and torch.is_grad_enabled() and x_all.requires_grad and not y_all.requires_grad:
                y_all = y_all + x_all.sum() * 0.0
            y_pad = _EPReduceScatter.apply(
                y_all.view(self.ep_world_size, l_max, self.dim),
                self.ep_group,
            )
            y = y_pad[:l_local].float()
        else:
            first_exp = self.experts[self.experts_start_idx]
            unstacked_released = (
                self._stacked_w13 is not None
                and first_exp is not None
                and first_exp.w1.weight.numel() == 0
            )
            if self._can_use_stacked_eval_path() or unstacked_released:
                y = self._moe_eval_stacked_forward(x_flat, weights, indices)
            else:
                self._clear_stacked_eval_weights()
                y = torch.zeros_like(x_flat, dtype=torch.float32)
                ckpt_expert = bool(self.training and torch.is_grad_enabled())
                expert_chunk = 16384
                for i in range(self.experts_start_idx, self.experts_end_idx):
                    idx, top = torch.where(indices == i)
                    if idx.numel() == 0:
                        continue
                    expert = self.experts[i]
                    assert expert is not None
                    for s in range(0, int(idx.numel()), expert_chunk):
                        idx_s = idx[s : s + expert_chunk]
                        top_s = top[s : s + expert_chunk]
                        x_e = x_flat[idx_s]
                        w_e = weights[idx_s, top_s].unsqueeze(-1)
                        if ckpt_expert and x_e.size(0) >= 512:
                            out_e = torch_ckpt(expert, x_e, w_e, use_reentrant=False)
                        else:
                            out_e = expert(x_e, w_e)
                        y.index_add_(0, idx_s, out_e.float())
            if self.ep_world_size > 1:
                if not self.training and not torch.is_grad_enabled():
                    dist.all_reduce(y, group=self.ep_group)
                else:
                    y = _differentiable_all_reduce(y, group=self.ep_group)

        if self.training and torch.is_grad_enabled():
            grad_anchor = torch.zeros((), dtype=torch.float32, device=x_flat.device)
            for i in range(self.experts_start_idx, self.experts_end_idx):
                expert = self.experts[i]
                if expert is None:
                    continue
                if expert.trainable_dense:
                    for proj in (expert.w1, expert.w2, expert.w3):
                        if proj.weight.requires_grad:
                            grad_anchor = grad_anchor + proj.weight.sum() * 0.0
                else:
                    for proj in (expert.w1, expert.w2, expert.w3):
                        if proj.lora_A is not None and proj.lora_A.requires_grad:
                            grad_anchor = grad_anchor + proj.lora_A.sum() * 0.0
                        if proj.lora_B is not None and proj.lora_B.requires_grad:
                            grad_anchor = grad_anchor + proj.lora_B.sum() * 0.0
                        if proj.qat_scale is not None and proj.qat_scale.requires_grad:
                            grad_anchor = grad_anchor + proj.qat_scale.sum() * 0.0
            y = y + grad_anchor

        if self.training and torch.is_grad_enabled() and x_flat.size(0) >= 512:
            shared_out = torch_ckpt(
                self.shared_experts,
                x_flat,
                None,
                shared_prequant,
                use_reentrant=False,
            )
        else:
            shared_out = self.shared_experts(
                x_flat, prequant_weights=shared_prequant
            )
        y = y + shared_out.float()
        return y.to(x_flat.dtype)

    def forward(
        self,
        x: torch.Tensor,
        ep_l_max: int = 0,
        ep_replicated_input: bool = False,
        use_moe_ckpt: bool = False,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.dim)
        mask_flat = image_mask.reshape(-1) if image_mask is not None else None
        weights, indices = self._gate_forward(x_flat, mask_flat)
        shared_prequant: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if (
            self.shared_experts.trainable_dense
            and self.shared_experts.enable_qat
            and not self.shared_experts.teacher_mode
        ):
            if self._can_use_stacked_eval_path():
                if self._shared_prequant_cache is None:
                    self._shared_prequant_cache = (
                        fake_quant_fp8_block32_ste(
                            self.shared_experts.w1.weight, scale_2d=True
                        ),
                        fake_quant_fp8_block32_ste(
                            self.shared_experts.w2.weight, scale_2d=True
                        ),
                        fake_quant_fp8_block32_ste(
                            self.shared_experts.w3.weight, scale_2d=True
                        ),
                    )
                shared_prequant = self._shared_prequant_cache
            else:
                self._shared_prequant_cache = None
                shared_prequant = (
                    fake_quant_fp8_block32_ste(
                        self.shared_experts.w1.weight, scale_2d=True
                    ),
                    fake_quant_fp8_block32_ste(
                        self.shared_experts.w2.weight, scale_2d=True
                    ),
                    fake_quant_fp8_block32_ste(
                        self.shared_experts.w3.weight, scale_2d=True
                    ),
                )
        if use_moe_ckpt and self.training and x_flat.requires_grad:
            if shared_prequant is not None:
                w1_pq, w2_pq, w3_pq = shared_prequant
                y = torch_ckpt(
                    lambda xf, wf, idxf, w1q, w2q, w3q: self._moe_body_forward(
                        xf,
                        wf,
                        idxf,
                        ep_l_max=ep_l_max,
                        ep_replicated_input=ep_replicated_input,
                        shared_prequant=(w1q, w2q, w3q),
                    ),
                    x_flat,
                    weights,
                    indices,
                    w1_pq,
                    w2_pq,
                    w3_pq,
                    use_reentrant=True,
                )
            else:
                y = torch_ckpt(
                    lambda xf, wf, idxf: self._moe_body_forward(
                        xf,
                        wf,
                        idxf,
                        ep_l_max=ep_l_max,
                        ep_replicated_input=ep_replicated_input,
                    ),
                    x_flat,
                    weights,
                    indices,
                    use_reentrant=True,
                )
        else:
            y = self._moe_body_forward(
                x_flat,
                weights,
                indices,
                ep_l_max=ep_l_max,
                ep_replicated_input=ep_replicated_input,
                shared_prequant=shared_prequant,
            )
        return y.view(orig_shape)


class CompactEngramModule(nn.Module):
    """Differentiable Engram Conditional Memory (`wkv`, `q_weight`, `k_weight`) with compact FP8 table."""

    def __init__(
        self,
        cfg: Dsv41E128Config,
        layer_id: int,
        ep_rank: int = 0,
        ep_world_size: int = 1,
        ep_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = cfg.engram_layer_ids.index(layer_id)
        self.dim = cfg.dim
        self.hc_mult = cfg.hc_mult
        self.eps = cfg.norm_eps
        self.head_dim = cfg.engram_head_dim
        self.n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        self.ep_group = ep_group
        self.enable_qat = cfg.enable_qat
        self.teacher_mode = False

        layout = EngramLayout.from_cfg(cfg)
        assert layout is not None
        if ep_world_size > 1 and self.n_hash_cols % ep_world_size == 0:
            self.part_n_hash_cols = self.n_hash_cols // ep_world_size
            self.head_start = ep_rank * self.part_n_hash_cols
        else:
            self.part_n_hash_cols = self.n_hash_cols
            self.head_start = 0

        head_sizes = layout.head_sizes_for_layer(self.layer_hash_index)
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(
            head_sizes[: self.head_start + self.part_n_hash_cols]
        )
        self.part_vocab_size = self.vocab_end_idx - self.vocab_start_idx

        if cfg.engram_compact_kept and self.layer_hash_index < len(cfg.engram_compact_kept):
            kept = cfg.engram_compact_kept[self.layer_hash_index]
            self.compact_start = sum(kept[: self.head_start])
            self.table_rows = sum(
                kept[self.head_start : self.head_start + self.part_n_hash_cols]
            )
            self.total_compact_rows = sum(kept)
        else:
            self.compact_start = self.vocab_start_idx
            self.table_rows = self.part_vocab_size
            self.total_compact_rows = sum(head_sizes)

        self.wkv = nn.Linear(
            self.n_hash_cols * self.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            dtype=torch.bfloat16,
        )
        self.q_weight = nn.Parameter(
            torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16)
        )
        self.k_weight = nn.Parameter(
            torch.ones(self.hc_mult, self.dim, dtype=torch.bfloat16)
        )

        self.has_compact_table = False
        self.wkv.weight.requires_grad_(False)
        self.q_weight.requires_grad_(False)
        self.k_weight.requires_grad_(False)
        self.register_buffer("table_weight", torch.empty(0), persistent=False)
        self.register_buffer("table_scale", torch.empty(0), persistent=False)
        self.register_buffer("row_map", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _lookup_local_slice(self, hash_slice: torch.Tensor) -> torch.Tensor:
        """Looks up `[..., cols]` hash IDs against this rank's local compact FP8 table."""
        owned = (hash_slice >= self.vocab_start_idx) & (
            hash_slice < self.vocab_end_idx
        )
        local_orig = torch.where(owned, hash_slice - self.vocab_start_idx, 0)
        if self.row_map.numel() > 0:
            mapped = self.row_map[local_orig]
            owned = owned & (mapped >= 0)
            local_compact = torch.where(owned, mapped, 0).long()
        else:
            local_compact = local_orig.long()

        vals_fp8 = self.table_weight[local_compact]
        scales_u8 = self.table_scale[local_compact]
        scales_f32 = e8m0_to_f32_torch(scales_u8)
        vals = (
            vals_fp8.float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
            * scales_f32.unsqueeze(-1)
        ).flatten(-2).to(torch.bfloat16)
        return vals.masked_fill(~owned.unsqueeze(-1), 0)

    @torch.no_grad()
    def _lookup_embeddings(
        self,
        hash_ids: torch.Tensor | None,
        bsz: int,
        seqlen: int,
        device: torch.device,
        ep_l_max: int = 0,
        ep_replicated_input: bool = True,
    ) -> torch.Tensor:
        if (
            not self.has_compact_table
            or hash_ids is None
            or self.table_weight.numel() == 0
        ):
            return torch.zeros(
                bsz,
                seqlen,
                self.n_hash_cols * self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )

        is_sharded = (
            self.ep_world_size > 1
            and self.part_n_hash_cols < self.n_hash_cols
            and dist.is_available()
            and dist.is_initialized()
        )
        if not is_sharded:
            vals = self._lookup_local_slice(hash_ids)
            return vals.flatten(-2)

        if ep_replicated_input:
            vals = self._lookup_local_slice(hash_ids)
            dist.all_reduce(vals, group=self.ep_group)
            return vals.flatten(-2)

        # DP+EP training mode (`ep_replicated_input=False`): each rank has its own sequence `hash_ids [B, S, 24]`.
        # Gather all ranks' hash_ids (`[W, L_max, 24]`), look up this rank's 3 owned columns `[head_start : head_end]`
        # for all W ranks (`[W, L_max, 3, 256]`), and exchange via `all_to_all_single` in memory-bounded chunks.
        l_local = bsz * seqlen
        l_max = max(ep_l_max, l_local)
        h_flat = hash_ids.view(l_local, self.n_hash_cols)
        if l_local < l_max:
            h_pad = F.pad(h_flat, (0, 0, 0, l_max - l_local), value=0)
        else:
            h_pad = h_flat

        w_size = self.ep_world_size
        h_all = torch.empty(
            (w_size, l_max, self.n_hash_cols),
            dtype=h_pad.dtype,
            device=device,
        )
        dist.all_gather_into_tensor(
            h_all, h_pad.contiguous(), group=self.ep_group
        )

        c_start = self.head_start
        c_end = c_start + self.part_n_hash_cols
        h_owned = h_all[:, :, c_start:c_end]  # [W, L_max, 3]

        out_cols = torch.empty(
            (w_size, l_local, self.part_n_hash_cols, self.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        chunk_tok = 4096
        for t0 in range(0, l_max, chunk_tok):
            t1 = min(t0 + chunk_tok, l_max)
            tc = t1 - t0
            send_part = self._lookup_local_slice(h_owned[:, t0:t1]).contiguous()  # [W, tc, 3, 256]
            recv_part = torch.empty_like(send_part)
            dist.all_to_all_single(recv_part, send_part, group=self.ep_group)
            if t0 < l_local:
                valid_tc = min(tc, l_local - t0)
                out_cols[:, t0 : t0 + valid_tc] = recv_part[:, :valid_tc]

        # Reassemble `[l_local, W, 3, 256]` -> `[B, S, 24 * 256]`
        merged = (
            out_cols.permute(1, 0, 2, 3)
            .contiguous()
            .view(bsz, seqlen, self.n_hash_cols * self.head_dim)
        )
        return merged

    def _compute_delta(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        ep_l_max: int = 0,
        ep_replicated_input: bool = True,
    ) -> torch.Tensor:
        bsz, seqlen, _, _ = x.shape
        embed_flat = self._lookup_embeddings(
            hash_ids,
            bsz,
            seqlen,
            x.device,
            ep_l_max=ep_l_max,
            ep_replicated_input=ep_replicated_input,
        )
        if self.enable_qat and not self.teacher_mode:
            embed_flat = act_quant_fp8_ste(embed_flat)
        kv = self.wkv(embed_flat)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(dim=-1) + self.eps) * torch.rsqrt(
            key.square().mean(dim=-1) + self.eps
        )
        dot = (h * weight * key).sum(dim=-1) * rstd * (self.dim**-0.5)
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
        )
        return (gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor | None = None,
        ep_l_max: int = 0,
        ep_replicated_input: bool = True,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`token_mask` (`[B, L]` bool, False inside image spans) zeroes the Engram delta there."""
        if (
            not self.has_compact_table
            or hash_ids is None
            or self.table_weight.numel() == 0
        ):
            return x
        engram_trainable = (
            self.wkv.weight.requires_grad
            or self.q_weight.requires_grad
            or self.k_weight.requires_grad
        )
        if not engram_trainable:
            with torch.no_grad():
                delta = self._compute_delta(
                    x.detach(),
                    hash_ids,
                    ep_l_max=ep_l_max,
                    ep_replicated_input=ep_replicated_input,
                )
        else:
            delta = self._compute_delta(
                x,
                hash_ids,
                ep_l_max=ep_l_max,
                ep_replicated_input=ep_replicated_input,
            )
        if token_mask is not None:
            delta = delta.masked_fill(~token_mask.to(torch.bool)[:, :, None, None], 0)
        return x + delta


class DifferentiableBlock(nn.Module):
    """DeepSeek-V4.1 Transformer Block with Single-Pass mHC, Attention, MoE, and optional Engram."""

    def __init__(
        self,
        layer_id: int,
        cfg: Dsv41E128Config,
        ep_rank: int = 0,
        ep_world_size: int = 1,
        ep_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = cfg.norm_eps
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps

        self.engram = (
            CompactEngramModule(
                cfg,
                layer_id,
                ep_rank=ep_rank,
                ep_world_size=ep_world_size,
                ep_group=ep_group,
            )
            if layer_id in cfg.engram_layer_ids
            else None
        )
        self.attn = DifferentiableAttention(layer_id, cfg)
        self.ffn = DifferentiableMoE(
            layer_id,
            cfg,
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
            ep_group=ep_group,
        )
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.dim
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def _hc_mixes_impl(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, hc_fn.float()) * rsqrt
        if (
            not torch.is_grad_enabled()
            and mixes.is_cuda
            and mixes.size(1) == 1
            and _hc_graph_enabled()
        ):
            out = _sinkhorn_graphed(
                mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            )
            if out is not None:
                return out
        return hc_split_sinkhorn_differentiable(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not hc_fn.requires_grad
            and not hc_scale.requires_grad
            and not hc_base.requires_grad
        ):
            with torch.no_grad():
                return self._hc_mixes_impl(x.detach(), hc_fn, hc_scale, hc_base)
        return self._hc_mixes_impl(x, hc_fn, hc_scale, hc_base)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsi,bsid->bsd", pre_mix.to(x.dtype), x)

    def hc_post(
        self,
        sub_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bsj,bsd->bsjd", post.to(sub_out.dtype), sub_out
        ) + torch.einsum(
            "bsij,bsid->bsjd", comb.to(residual.dtype), residual
        )

    def forward(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        shared_state: dict[str, Any],
        cos_swa: torch.Tensor,
        sin_swa: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
        engram_hash_ids: torch.Tensor | None = None,
        ep_l_max: int = 0,
        ep_replicated_input: bool = False,
        layer_cache: dict[str, Any] | None = None,
        start_pos: int = 0,
        image_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`image_mask`: `[B, S]` bool, True inside image spans (VL routing bias, no Engram)."""
        if self.engram is not None:
            x = self.engram(
                x,
                engram_hash_ids,
                ep_l_max=ep_l_max,
                ep_replicated_input=ep_replicated_input,
                token_mask=None if image_mask is None else ~image_mask,
            )

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        h_attn = self.attn_norm(self.hc_pre(x, pre_mix))
        if (
            self.training
            and layer_cache is None
            and torch.is_grad_enabled()
            and h_attn.requires_grad
        ):
            ckv_in = shared_state.get("compress_kv")
            ik_in = shared_state.get("index_k")
            tk_in = shared_state.get("topk_idxs")
            cd_in = shared_state.get("candidates")

            def _run_attn(
                ha: torch.Tensor,
                ckv: torch.Tensor | None,
                ik: torch.Tensor | None,
                tk: torch.Tensor | None,
                cd: torch.Tensor | None,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]:
                st = {
                    "compress_kv": ckv,
                    "index_k": ik,
                    "topk_idxs": tk,
                    "candidates": cd,
                }
                out = self.attn(
                    ha,
                    st,
                    cos_swa,
                    sin_swa,
                    cos_comp,
                    sin_comp,
                    layer_cache=None,
                    start_pos=start_pos,
                )
                return (
                    out,
                    st["compress_kv"],
                    st["index_k"],
                    st["topk_idxs"],
                    st["candidates"],
                )

            (
                h_attn,
                shared_state["compress_kv"],
                shared_state["index_k"],
                shared_state["topk_idxs"],
                shared_state["candidates"],
            ) = torch_ckpt(
                _run_attn,
                h_attn,
                ckv_in,
                ik_in,
                tk_in,
                cd_in,
                use_reentrant=True,
            )
        else:
            h_attn = self.attn(
                h_attn,
                shared_state,
                cos_swa,
                sin_swa,
                cos_comp,
                sin_comp,
                layer_cache=layer_cache,
                start_pos=start_pos,
            )
        x = self.hc_post(h_attn, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        h_ffn = self.ffn_norm(self.hc_pre(x, attn_pre))
        h_ffn = self.ffn(
            h_ffn,
            ep_l_max=ep_l_max,
            ep_replicated_input=ep_replicated_input,
            use_moe_ckpt=False,
            image_mask=image_mask,
        )
        x = self.hc_post(h_ffn, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    @torch.no_grad()
    def forward_batched_decode(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        attn_state: dict[str, Any],
        shared: dict[str, Any],
        cos_swa: torch.Tensor,
        sin_swa: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
        engram_hash_ids: torch.Tensor | None,
        pos_t: torch.Tensor,
        win_valid: torch.Tensor,
        max_pos: int,
        ep_replicated_input: bool = True,
        step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched 1-token decode across `B` sequences (`x: [B, 1, hc_mult, dim]`), batching all dense projections and MoE experts."""
        bsz = x.size(0)
        if self.engram is not None:
            x = self.engram(
                x,
                engram_hash_ids,
                ep_l_max=bsz,
                ep_replicated_input=ep_replicated_input,
            )

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        h_attn_in = self.attn_norm(self.hc_pre(x, pre_mix))
        h_attn = self.attn.forward_batched_decode(
            h_attn_in,
            attn_state,
            shared,
            cos_swa,
            sin_swa,
            cos_comp,
            sin_comp,
            pos_t=pos_t,
            win_valid=win_valid,
            max_pos=max_pos,
            step_idx=step_idx,
        )
        x = self.hc_post(h_attn, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        h_ffn = self.ffn_norm(self.hc_pre(x, attn_pre))
        h_ffn = self.ffn(
            h_ffn,
            ep_l_max=bsz,
            ep_replicated_input=ep_replicated_input,
            use_moe_ckpt=False,
        )
        x = self.hc_post(h_ffn, residual, ffn_post, ffn_comb)
        return x, ffn_pre


# ==============================================================================
# Vision tower (DeepSeek-V4.1 VL): exact port of `inference/vision.py`
# ==============================================================================


@functools.lru_cache(maxsize=16)
def _vision_cos_sin_cpu(n_h: int, n_w: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D RoPE tables `[n_h * n_w, 1, dim]` (reference `get_vision_cos_sin`)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _vision_apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    """FP32 RMSNorm with `eps=1e-6` (reference `vision.RMSNorm`)."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * xf).to(dtype)


class _CastLinear(nn.Linear):
    """`nn.Linear` whose parameters may be FP32 masters while the matmul runs in the input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class VisionPatchEmbed(nn.Module):

    def __init__(self, cfg: Dsv41E128Config, dtype: torch.dtype):
        super().__init__()
        self.proj = _CastLinear(3 * cfg.vision_patch_size**2, cfg.vision_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    """Full bidirectional attention over one image's patches with 2-D RoPE."""

    def __init__(self, cfg: Dsv41E128Config, dtype: torch.dtype):
        super().__init__()
        self.n_heads = cfg.vision_n_heads
        self.head_dim = cfg.vision_dim // cfg.vision_n_heads
        self.wqkv = _CastLinear(cfg.vision_dim, 3 * cfg.vision_dim, dtype=dtype)
        self.wo = _CastLinear(cfg.vision_dim, cfg.vision_dim, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = _vision_apply_rotary(q, cos, sin)
        k = _vision_apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):

    def __init__(self, cfg: Dsv41E128Config, dtype: torch.dtype):
        super().__init__()
        self.w1 = _CastLinear(cfg.vision_dim, 2 * cfg.vision_inter_dim, bias=False, dtype=dtype)
        self.w2 = _CastLinear(cfg.vision_inter_dim, cfg.vision_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):

    def __init__(self, cfg: Dsv41E128Config, dtype: torch.dtype):
        super().__init__()
        self.norm1 = VisionRMSNorm(cfg.vision_dim, dtype=dtype)
        self.attn = VisionAttention(cfg, dtype)
        self.norm2 = VisionRMSNorm(cfg.vision_dim, dtype=dtype)
        self.mlp = VisionMLP(cfg, dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """DeepSeek ViT (`vision.*` tensors): patch embed -> `vision_n_layers` blocks -> RMSNorm."""

    def __init__(self, cfg: Dsv41E128Config, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.rope_dim = cfg.vision_dim // cfg.vision_n_heads // 2
        self.rope_theta = cfg.vision_rope_theta
        self.patch_embed = VisionPatchEmbed(cfg, dtype)
        self.blocks = nn.ModuleList([VisionBlock(cfg, dtype) for _ in range(cfg.vision_n_layers)])
        self.norm = VisionRMSNorm(cfg.vision_dim, dtype=dtype)
        self.gradient_checkpointing = True

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """`patches: [n_h * n_w, 3, p, p]` -> `[n_h * n_w, vision_dim]`."""
        any_trainable = patches.requires_grad or any(p.requires_grad for p in self.parameters())
        if not any_trainable:
            with torch.no_grad():
                x = self.patch_embed(patches)
                cos, sin = _vision_cos_sin_cpu(n_h, n_w, self.rope_dim, self.rope_theta)
                cos = cos.to(x.device)
                sin = sin.to(x.device)
                for block in self.blocks:
                    x = block(x, cos, sin)
                return self.norm(x)
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin_cpu(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt and x.requires_grad:
                x = torch_ckpt(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return self.norm(x)


class VisionAligner(nn.Module):
    """3x3 patch-merge + 2-layer GELU MLP into the LLM width (`aligner.*` tensors)."""

    def __init__(self, cfg: Dsv41E128Config, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.downsample_ratio = cfg.vision_downsample_ratio
        in_dim = cfg.vision_dim * self.downsample_ratio**2
        self.w1 = _CastLinear(in_dim, cfg.dim, dtype=dtype)
        self.w2 = _CastLinear(cfg.dim, cfg.dim, dtype=dtype)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """`x: [n_h * n_w, vision_dim]` -> `[ceil(n_h/r) * ceil(n_w/r), dim]` in reading order."""
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


@dataclass
class ImageInput:
    """One image for the forward pass: ViT patches (`[n_vit_h * n_vit_w, 3, p, p]`) and its patch grid."""

    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int


class Dsv41SftModel(nn.Module):
    """Full-Sequence Differentiable & KV-Cached Pruned DeepSeek-V4.1 (`Tool E128`) Model for QAT SFT & Eval."""

    def __init__(
        self,
        cfg: Dsv41E128Config,
        ep_rank: int = 0,
        ep_world_size: int = 1,
        gradient_checkpointing: bool = True,
        grad_ckpt_stride: int = 1,
        ep_replicated_input: bool = False,
        ep_group: dist.ProcessGroup | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        self.ep_group = ep_group
        self.gradient_checkpointing = gradient_checkpointing
        self.grad_ckpt_stride = max(1, grad_ckpt_stride)
        self.ep_replicated_input = ep_replicated_input
        self.teacher_mode = False

        engram_layout = EngramLayout.from_cfg(cfg)
        self.engram_hash = (
            NgramHashStateModule(cfg, engram_layout)
            if engram_layout is not None
            else None
        )

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(
            [
                DifferentiableBlock(
                    i,
                    cfg,
                    ep_rank=ep_rank,
                    ep_world_size=ep_world_size,
                    ep_group=ep_group,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False, dtype=torch.bfloat16)

        cos_swa, sin_swa = precompute_freqs_cis_torch(
            cfg.rope_head_dim,
            cfg.max_seq_len,
            0,
            cfg.rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        )
        cos_comp, sin_comp = precompute_freqs_cis_torch(
            cfg.rope_head_dim,
            cfg.max_seq_len,
            cfg.original_seq_len,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        )
        self.register_buffer("cos_swa", cos_swa, persistent=False)
        self.register_buffer("sin_swa", sin_swa, persistent=False)
        self.register_buffer("cos_comp", cos_comp, persistent=False)
        self.register_buffer("sin_comp", sin_comp, persistent=False)

        # Vision tower (names match the released checkpoint keys `vision.*`, `aligner.*`, `image_*`).
        # FP32 master weights under QAT so `lr ~ 1e-5` updates are not lost to BF16 rounding; the
        # matmuls run in the activation dtype either way and the export casts back to BF16.
        self.vision: VisionTransformer | None = None
        self.aligner: VisionAligner | None = None
        if cfg.vision_enabled:
            v_dtype = torch.float32 if cfg.enable_qat else torch.bfloat16
            self.vision = VisionTransformer(cfg, dtype=v_dtype)
            self.aligner = VisionAligner(cfg, dtype=v_dtype)
            self.image_start = nn.Parameter(torch.zeros(cfg.dim, dtype=v_dtype))
            self.image_end = nn.Parameter(torch.zeros(cfg.dim, dtype=v_dtype))
            self.image_newline = nn.Parameter(torch.zeros(cfg.dim, dtype=v_dtype))
        self._vision_teacher_state: dict[str, torch.Tensor] | None = None
        self._vision_teacher_vis_state: dict[str, torch.Tensor] | None = None
        self._vision_teacher_al_state: dict[str, torch.Tensor] | None = None

    def set_teacher_mode(self, enabled: bool) -> None:
        """Toggles frozen base teacher mode across all submodules (for Nemotron QAD forward-KL)."""
        self.teacher_mode = enabled
        for mod in self.modules():
            if hasattr(mod, "teacher_mode"):
                mod.teacher_mode = enabled

    def snapshot_teacher_weights(self) -> None:
        """Snapshots initial shared_experts BF16 weights across all layers for Nemotron QAD teacher pass."""
        for blk in self.layers:
            if blk.ffn.shared_experts is not None:
                blk.ffn.shared_experts.snapshot_teacher_weights()
        if self.vision is not None:
            # On-device BF16 copy (~110 MB) of the released vision weights; `encode_images` swaps it in under teacher mode.
            self._vision_teacher_state = {
                k: v.detach().to(dtype=torch.bfloat16).clone()
                for k, v in self.named_parameters()
                if k.startswith(("vision.", "aligner.", "image_"))
            }
            self._vision_teacher_vis_state = {
                k[len("vision."):]: v
                for k, v in self._vision_teacher_state.items()
                if k.startswith("vision.")
            }
            self._vision_teacher_al_state = {
                k[len("aligner."):]: v
                for k, v in self._vision_teacher_state.items()
                if k.startswith("aligner.")
            }

    # ---------------------------------------------------------------- images

    def vision_parameter_names(self) -> list[str]:
        return [n for n, _ in self.named_parameters() if n.startswith(("vision.", "aligner.", "image_"))]

    def encode_images(self, images: Sequence[ImageInput]) -> list[torch.Tensor]:
        """Runs ViT + aligner per image; returns `[n_llm_h * n_llm_w, dim]` rows per image (reading order)."""
        assert self.vision is not None and self.aligner is not None, "model has no vision tower"
        act_dtype = self.embed.weight.dtype
        if self.teacher_mode and self._vision_teacher_state is not None:
            dev = self.embed.weight.device
            if self._vision_teacher_vis_state is not None and self._vision_teacher_al_state is not None:
                vis_state = self._vision_teacher_vis_state
                al_state = self._vision_teacher_al_state
            else:
                state = {k: v.to(dev, non_blocking=True) for k, v in self._vision_teacher_state.items()}
                vis_state = {k[len("vision."):]: v for k, v in state.items() if k.startswith("vision.")}
                al_state = {k[len("aligner."):]: v for k, v in state.items() if k.startswith("aligner.")}
            outs = []
            for img in images:
                feats = torch.func.functional_call(
                    self.vision, vis_state, (img.patches.to(dev, act_dtype), img.n_vit_h, img.n_vit_w)
                )
                outs.append(
                    torch.func.functional_call(self.aligner, al_state, (feats, img.n_vit_h, img.n_vit_w))
                )
            return outs
        outs = []
        for img in images:
            patches = img.patches.to(self.embed.weight.device, act_dtype, non_blocking=True)
            feats = self.vision(patches, img.n_vit_h, img.n_vit_w)
            outs.append(self.aligner(feats, img.n_vit_h, img.n_vit_w))
        return outs

    def _delimiter_embeddings(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.teacher_mode and self._vision_teacher_state is not None:
            dev = self.embed.weight.device
            st = self._vision_teacher_state
            return st["image_start"].to(dev), st["image_end"].to(dev), st["image_newline"].to(dev)
        return self.image_start, self.image_end, self.image_newline

    def merge_image_embeddings(
        self,
        h: torch.Tensor,
        token_types: torch.Tensor,
        images: Sequence[Sequence[ImageInput]],
    ) -> torch.Tensor:
        """Overwrites image-span positions of `h [B, S, dim]` with ViT/aligner rows and delimiter embeddings.

        `token_types [B, S]`: -1 text, 0 START, 1 IMAGE, 2 NEW_LINE, 3 END (`dsv41_image_tokens`).
        `images[b]`: the images of sequence `b` in prompt order. IMAGE slots take the aligner rows in
        reading order (reference `merge_image_embeddings`).
        """
        if self.vision is None:
            raise ValueError("Received image spans but the model has no vision tower.")
        if token_types.shape != h.shape[:2]:
            raise ValueError(f"token_types {tuple(token_types.shape)} vs hidden {tuple(h.shape[:2])}")
        h = h.clone()
        start_e, end_e, nl_e = self._delimiter_embeddings()
        for b in range(h.size(0)):
            types_b = token_types[b]
            imgs_b = images[b] if b < len(images) else ()
            is_img = types_b == 1
            n_slots = int(is_img.sum().item())
            if n_slots == 0 and not imgs_b:
                continue
            embeds = self.encode_images(imgs_b)
            rows = torch.cat(embeds, dim=0) if embeds else h.new_zeros((0, h.size(-1)))
            if rows.size(0) != n_slots:
                raise ValueError(
                    f"sequence {b}: {n_slots} IMAGE slots but {rows.size(0)} aligner rows from "
                    f"{len(imgs_b)} images"
                )
            h[b, types_b == 0] = start_e.to(h.dtype)
            h[b, types_b == 3] = end_e.to(h.dtype)
            h[b, types_b == 2] = nl_e.to(h.dtype)
            if n_slots:
                h[b, is_img] = rows.to(h.dtype)
        return h

    def create_kv_cache(self) -> dict[str, Any]:
        """Creates an empty incremental KV cache dict for `forward_hidden_cached`."""
        return {"seqlen": 0, "layers": [{} for _ in range(len(self.layers))]}

    def ensure_rope_capacity(self, required_len: int) -> None:
        if required_len <= self.cos_swa.size(0):
            return
        new_len = max(required_len, int(self.cos_swa.size(0) * 2))
        device = self.cos_swa.device
        cos_swa, sin_swa = precompute_freqs_cis_torch(
            self.cfg.rope_head_dim,
            new_len,
            0,
            self.cfg.rope_theta,
            self.cfg.rope_factor,
            self.cfg.beta_fast,
            self.cfg.beta_slow,
            device=device,
        )
        cos_comp, sin_comp = precompute_freqs_cis_torch(
            self.cfg.rope_head_dim,
            new_len,
            self.cfg.original_seq_len,
            self.cfg.compress_rope_theta,
            self.cfg.rope_factor,
            self.cfg.beta_fast,
            self.cfg.beta_slow,
            device=device,
        )
        self.cos_swa = cos_swa
        self.sin_swa = sin_swa
        self.cos_comp = cos_comp
        self.sin_comp = sin_comp

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        token_types: torch.Tensor | None = None,
        images: Sequence[Sequence[ImageInput]] | None = None,
    ) -> torch.Tensor:
        """Runs embed -> 40 mHC blocks -> final RMSNorm and returns `[B, S, dim]` hidden states.

        `token_types [B, S]` (int, -1 = text, >= 0 inside image spans) and `images[b]` (the
        sequence's `ImageInput`s in prompt order) enable the VL path: image spans get ViT/aligner
        embeddings, route with `gate.bias_vl`, and take no part in Engram n-grams. Attention stays
        causal everywhere (DeepSeek's reference; the `is_mm_prefix_lm` bidirectional variant is a bug).
        """
        bsz, seqlen = input_ids.shape
        self.ensure_rope_capacity(seqlen)
        ep_l_max = bsz * seqlen
        if (
            self.ep_world_size > 1
            and not self.ep_replicated_input
            and dist.is_available()
            and dist.is_initialized()
        ):
            l_t = torch.tensor([ep_l_max], dtype=torch.int64, device=input_ids.device)
            dist.all_reduce(l_t, op=dist.ReduceOp.MAX, group=self.ep_group)
            ep_l_max = int(l_t.item())

        image_mask: torch.Tensor | None = None
        if token_types is not None:
            token_types = token_types.to(input_ids.device)
            if bool((token_types >= 0).any()):
                image_mask = token_types >= 0

        engram_hashes: torch.Tensor | None = None
        if self.engram_hash is not None and any(
            blk.engram is not None and blk.engram.has_compact_table
            for blk in self.layers
        ):
            engram_hashes = self.engram_hash(
                input_ids, token_mask=None if image_mask is None else ~image_mask
            )

        h = self.embed(input_ids)
        if image_mask is not None:
            h = self.merge_image_embeddings(h, token_types, images or [])
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        if self.training and torch.is_grad_enabled() and not h.requires_grad:
            h = h.requires_grad_(True)
        pre_mix = h.new_zeros(bsz, seqlen, self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0

        shared_state: dict[str, Any] = {
            "compress_kv": None,
            "index_k": None,
            "topk_idxs": None,
            "candidates": None,
        }
        cos_s = self.cos_swa[:seqlen]
        sin_s = self.sin_swa[:seqlen]
        cos_c = self.cos_comp[:seqlen]
        sin_c = self.sin_comp[:seqlen]
        ep_rep = self.ep_replicated_input

        for i, block in enumerate(self.layers):
            layer_engram_ids = (
                engram_hashes[:, :, block.engram.layer_hash_index]
                if (engram_hashes is not None and block.engram is not None)
                else None
            )
            use_ckpt = (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and (i % self.grad_ckpt_stride == 0)
            )
            if use_ckpt:
                in_comp_kv = shared_state["compress_kv"]
                in_index_k = shared_state["index_k"]
                in_topk = shared_state["topk_idxs"]
                in_cand = shared_state["candidates"]

                def _run_block(
                    x_in: torch.Tensor,
                    pm_in: torch.Tensor,
                    ckv: torch.Tensor | None,
                    ik: torch.Tensor | None,
                    tk: torch.Tensor | None,
                    cd: torch.Tensor | None,
                    blk: DifferentiableBlock = block,
                    eh_ids: torch.Tensor | None = layer_engram_ids,
                    l_max_val: int = ep_l_max,
                    rep_val: bool = ep_rep,
                    img_mask: torch.Tensor | None = image_mask,
                ) -> tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor | None,
                    torch.Tensor | None,
                    torch.Tensor | None,
                    torch.Tensor | None,
                ]:
                    local_shared = {
                        "compress_kv": ckv,
                        "index_k": ik,
                        "topk_idxs": tk,
                        "candidates": cd,
                    }
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda t: t, lambda t: t
                    ):
                        x_out, pm_out = blk(
                            x_in,
                            pm_in,
                            local_shared,
                            cos_s,
                            sin_s,
                            cos_c,
                            sin_c,
                            eh_ids,
                            ep_l_max=l_max_val,
                            ep_replicated_input=rep_val,
                            image_mask=img_mask,
                        )
                    return (
                        x_out,
                        pm_out,
                        local_shared["compress_kv"],
                        local_shared["index_k"],
                        local_shared["topk_idxs"],
                        local_shared["candidates"],
                    )

                offload_ckpt_boundary = bool(
                    h.is_cuda
                    and (
                        (seqlen >= 20480 and (i % 4 != 3))
                        or (8192 <= seqlen < 20480 and (i % 2 == 0))
                    )
                )
                if offload_ckpt_boundary:
                    with torch.autograd.graph.save_on_cpu(pin_memory=False):
                        (
                            h,
                            pre_mix,
                            shared_state["compress_kv"],
                            shared_state["index_k"],
                            shared_state["topk_idxs"],
                            shared_state["candidates"],
                        ) = torch_ckpt(
                            _run_block,
                            h,
                            pre_mix,
                            in_comp_kv,
                            in_index_k,
                            in_topk,
                            in_cand,
                            use_reentrant=True,
                        )
                else:
                    (
                        h,
                        pre_mix,
                        shared_state["compress_kv"],
                        shared_state["index_k"],
                        shared_state["topk_idxs"],
                        shared_state["candidates"],
                    ) = torch_ckpt(
                        _run_block,
                        h,
                        pre_mix,
                        in_comp_kv,
                        in_index_k,
                        in_topk,
                        in_cand,
                        use_reentrant=True,
                    )
            else:
                h, pre_mix = block(
                    h,
                    pre_mix,
                    shared_state,
                    cos_s,
                    sin_s,
                    cos_c,
                    sin_c,
                    layer_engram_ids,
                    ep_l_max=ep_l_max,
                    ep_replicated_input=ep_rep,
                    image_mask=image_mask,
                )

        h_final = self.layers[-1].hc_pre(h, pre_mix)
        return self.norm(h_final)

    @torch.no_grad()
    def embed_with_images(
        self,
        input_ids: torch.Tensor,
        token_types: torch.Tensor | None,
        images: Sequence[ImageInput] | None,
    ) -> torch.Tensor:
        """Token embeddings `[1, S, dim]` with image spans already replaced (eval prefill helper).

        Call once per prompt (the ViT runs once per image) and feed slices of the result to
        `forward_hidden_cached(..., inputs_embeds=...)` so chunked prefill never splits an image.
        """
        h = self.embed(input_ids)
        if token_types is not None and images:
            h = self.merge_image_embeddings(h, token_types.to(input_ids.device), [list(images)])
        return h

    @torch.no_grad()
    def forward_hidden_cached(
        self,
        new_input_ids: torch.Tensor,
        kv_cache: dict[str, Any],
        token_types: torch.Tensor | None = None,
        images: Sequence[ImageInput] | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs incremental KV-cached prefill or 1-token decode for fast Candidate B32 evaluation.

        `kv_cache` is a mutable dict initialized as `{}` on a new sequence or prefix reuse.

        VL prefill (image checkpoints): `token_types [B, S]` (-1 text, 0..3 image-span types) for
        this chunk enables `gate.bias_vl` routing on image rows and keeps image tokens out of the
        Engram n-grams (the cached tail stores them as DEAD, exactly like the reference
        `NgramHashState`). Image embeddings come either from `images` (the chunk must contain the
        complete spans) or from `inputs_embeds [B, S, dim]` precomputed with `embed_with_images`
        (safe for chunked prefill). Decode steps are text-only and need none of these.
        """
        bsz, seqlen = new_input_ids.shape
        start_pos = int(kv_cache.get("seqlen", 0))
        total_len = start_pos + seqlen
        self.ensure_rope_capacity(total_len)

        image_mask: torch.Tensor | None = None
        if token_types is not None:
            token_types = token_types.to(new_input_ids.device)
            if token_types.shape != new_input_ids.shape:
                raise ValueError(
                    f"token_types {tuple(token_types.shape)} vs input_ids {tuple(new_input_ids.shape)}"
                )
            if bool((token_types >= 0).any()):
                image_mask = token_types >= 0

        engram_hashes: torch.Tensor | None = None
        if self.engram_hash is not None and any(
            blk.engram is not None and blk.engram.has_compact_table
            for blk in self.layers
        ):
            prev_comp_tok = kv_cache.get("engram_tail")
            engram_hashes = self.engram_hash(
                new_input_ids,
                prefix_compressed=prev_comp_tok,
                token_mask=None if image_mask is None else ~image_mask,
            )
            safe_ids = new_input_ids.clamp(0, self.engram_hash.token_map.numel() - 1)
            new_comp = self.engram_hash.token_map[safe_ids]
            if image_mask is not None:
                new_comp = torch.where(
                    image_mask, torch.full_like(new_comp, self.engram_hash.DEAD), new_comp
                )
            full_comp = (
                torch.cat([prev_comp_tok, new_comp], dim=1)
                if (prev_comp_tok is not None and start_pos > 0)
                else new_comp
            )
            keep_n = max(1, self.cfg.engram_max_ngram_size - 1)
            kv_cache["engram_tail"] = full_comp[:, -keep_n:]

        layer_caches: list[dict[str, Any]] = kv_cache.setdefault(
            "layers", [{} for _ in range(len(self.layers))]
        )

        if inputs_embeds is not None:
            if tuple(inputs_embeds.shape[:2]) != (bsz, seqlen):
                raise ValueError(
                    f"inputs_embeds {tuple(inputs_embeds.shape)} vs input_ids {tuple(new_input_ids.shape)}"
                )
            h = inputs_embeds.to(self.embed.weight.dtype)
        else:
            h = self.embed(new_input_ids)
            if image_mask is not None:
                if not images:
                    raise ValueError(
                        "token_types has image spans but neither `images` nor `inputs_embeds` was given"
                    )
                h = self.merge_image_embeddings(h, token_types, [list(images)])
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = h.new_zeros(bsz, seqlen, self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0

        shared_state: dict[str, Any] = {
            "compress_kv": None,
            "index_k": None,
            "topk_idxs": None,
            "candidates": None,
        }
        cos_s = self.cos_swa[:total_len]
        sin_s = self.sin_swa[:total_len]
        cos_c = self.cos_comp[:total_len]
        sin_c = self.sin_comp[:total_len]
        ep_l_max = bsz * seqlen

        for i, block in enumerate(self.layers):
            layer_engram_ids = (
                engram_hashes[:, :, block.engram.layer_hash_index]
                if (engram_hashes is not None and block.engram is not None)
                else None
            )
            h, pre_mix = block(
                h,
                pre_mix,
                shared_state,
                cos_s,
                sin_s,
                cos_c,
                sin_c,
                layer_engram_ids,
                ep_l_max=ep_l_max,
                ep_replicated_input=True,
                layer_cache=layer_caches[i],
                start_pos=start_pos,
                image_mask=image_mask,
            )

        kv_cache["seqlen"] = total_len
        h_final = self.layers[-1].hc_pre(h, pre_mix)
        return self.norm(h_final)

    # ------------------------------------------------------------------
    # Batched multi-stream decode session API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def begin_batched_decode(
        self,
        kv_caches: list[dict[str, Any]],
        max_steps: int = 64,
    ) -> "BatchedDecodeSession":
        """Builds padded batched per-layer state for `B` prefilled streams (one-off cost per decode chunk)."""
        bsz = len(kv_caches)
        assert bsz >= 1
        positions = [int(c.get("seqlen", 0)) for c in kv_caches]
        assert all(p > 0 for p in positions), (
            "begin_batched_decode requires prefilled streams (seqlen > 0)",
            positions,
        )
        device = self.embed.weight.device
        max_pos = max(positions)
        all_layer_caches: list[list[dict[str, Any]]] = [
            c.setdefault("layers", [{} for _ in range(len(self.layers))])
            for c in kv_caches
        ]
        layer_states: list[dict[str, Any]] = []
        for i, block in enumerate(self.layers):
            ratio = block.attn.compress_ratio
            cap = ((max_pos + max_steps) // max(1, ratio)) + 2 if ratio > 0 else 0
            layer_states.append(
                block.attn.begin_batched_state(
                    [all_layer_caches[b][i] for b in range(bsz)],
                    positions,
                    cap=cap,
                    device=device,
                    max_steps=max_steps,
                )
            )
        session = BatchedDecodeSession(
            kv_caches=kv_caches,
            positions=positions,
            layer_states=layer_states,
            bsz=bsz,
            device=device,
        )
        session.begin_positions = list(positions)
        session.orig_engram_tails = [c.get("engram_tail") for c in kv_caches]
        session.tok_hist = torch.zeros(
            (bsz, max(1, max_steps)), dtype=torch.long, device=device
        )
        session.steps_run = 0
        return session

    @torch.no_grad()
    def step_batched_decode(
        self,
        new_input_ids: torch.Tensor,
        session: "BatchedDecodeSession",
    ) -> torch.Tensor:
        """Runs one fully batched decode step (`new_input_ids: [B, 1]`) on an open session."""
        assert not session.closed
        bsz, seqlen = new_input_ids.shape
        assert seqlen == 1 and bsz == session.bsz, (new_input_ids.shape, session.bsz)
        positions = session.positions
        max_pos = max(positions)
        self.ensure_rope_capacity(max_pos + 1)
        device = new_input_ids.device

        step_idx = session.steps_run
        # Grow compressed-KV / history capacity if a long session outlives the initial margin
        for i, block in enumerate(self.layers):
            st = session.layer_states[i]
            ratio = block.attn.compress_ratio
            if ratio > 0 and block.attn.is_kv_source:
                need = (max_pos + 1) // ratio + 2
                if need > int(st["cap"]):
                    block.attn.grow_batched_state(st, need + 64)
            block.attn.ensure_history_capacity(st, step_idx + 1)
        if session.tok_hist.size(1) < step_idx + 1:
            session.tok_hist = torch.cat(
                [
                    session.tok_hist,
                    torch.zeros(
                        (bsz, step_idx + 33 - session.tok_hist.size(1)),
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
        session.tok_hist[:, step_idx] = new_input_ids[:, 0]

        pos_t = torch.tensor(positions, dtype=torch.long, device=device)
        w_size = self.cfg.window_size
        valid_w = (pos_t + 1).clamp(max=w_size)
        win_valid = torch.arange(w_size, device=device).unsqueeze(0) >= (
            w_size - valid_w
        ).unsqueeze(1)  # [B, W]

        engram_hashes: torch.Tensor | None = None
        if self.engram_hash is not None and any(
            blk.engram is not None and blk.engram.has_compact_table
            for blk in self.layers
        ):
            keep_n = max(1, self.cfg.engram_max_ngram_size - 1)
            safe_all = new_input_ids.clamp(
                0, self.engram_hash.token_map.numel() - 1
            )
            new_comp_all = self.engram_hash.token_map[safe_all]
            kv_caches = session.kv_caches
            tails = [kv_caches[b].get("engram_tail") for b in range(bsz)]
            uniform_tail = all(
                positions[b] > 0 and tails[b] is not None and tails[b].size(1) == keep_n
                for b in range(bsz)
            )
            if uniform_tail:
                # Steady-state decode: one batched hash over `[B, keep_n]` prefixes (no per-stream loop)
                prefix_b = torch.cat(tails, dim=0)  # [B, keep_n]
                engram_hashes = self.engram_hash(
                    new_input_ids, prefix_compressed=prefix_b
                )
                full_comp_b = torch.cat([prefix_b, new_comp_all], dim=1)[:, -keep_n:]
                for b in range(bsz):
                    kv_caches[b]["engram_tail"] = full_comp_b[b : b + 1]
            else:
                hash_list: list[torch.Tensor] = []
                for b in range(bsz):
                    prev_comp_tok = tails[b]
                    sp = positions[b]
                    hb = self.engram_hash(
                        new_input_ids[b : b + 1],
                        prefix_compressed=prev_comp_tok if sp > 0 else None,
                    )
                    hash_list.append(hb)
                    new_comp_b = new_comp_all[b : b + 1]
                    full_comp = (
                        torch.cat([prev_comp_tok, new_comp_b], dim=1)
                        if (prev_comp_tok is not None and sp > 0)
                        else new_comp_b
                    )
                    kv_caches[b]["engram_tail"] = full_comp[:, -keep_n:]
                engram_hashes = torch.cat(hash_list, dim=0)

        h = self.embed(new_input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.cfg.hc_mult, 1)
        pre_mix = h.new_zeros(bsz, 1, self.cfg.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0

        shared: dict[str, Any] = {}
        cos_s = self.cos_swa[: max_pos + 1]
        sin_s = self.sin_swa[: max_pos + 1]
        cos_c = self.cos_comp[: max_pos + 1]
        sin_c = self.sin_comp[: max_pos + 1]

        for i, block in enumerate(self.layers):
            layer_engram_ids = (
                engram_hashes[:, :, block.engram.layer_hash_index]
                if (engram_hashes is not None and block.engram is not None)
                else None
            )
            h, pre_mix = block.forward_batched_decode(
                h,
                pre_mix,
                session.layer_states[i],
                shared,
                cos_s,
                sin_s,
                cos_c,
                sin_c,
                layer_engram_ids,
                pos_t=pos_t,
                win_valid=win_valid,
                max_pos=max_pos,
                ep_replicated_input=True,
                step_idx=step_idx,
            )

        for b in range(bsz):
            session.positions[b] = positions[b] + 1
        session.steps_run = step_idx + 1
        h_final = self.layers[-1].hc_pre(h, pre_mix)
        return self.norm(h_final)

    @torch.no_grad()
    def end_batched_decode(
        self,
        session: "BatchedDecodeSession",
        valid_steps: list[int] | None = None,
    ) -> None:
        """Writes the batched per-layer state back into the per-stream KV caches and closes the session.

        `valid_steps[b]` (optional) = number of this session's steps to keep for stream `b`;
        streams with fewer valid steps than steps run are rolled back exactly (their caches end
        at `begin_position + valid_steps[b]` tokens), e.g. after an EOS / stop string mid-chunk.
        """
        if session.closed:
            return
        bsz = session.bsz
        positions = session.positions
        begin_positions = session.begin_positions
        steps_run = session.steps_run
        if valid_steps is not None:
            assert len(valid_steps) == bsz
            valid_steps = [max(0, min(int(k), steps_run)) for k in valid_steps]
        for i, block in enumerate(self.layers):
            block.attn.end_batched_state(
                session.layer_states[i],
                [session.kv_caches[b]["layers"][i] for b in range(bsz)],
                positions,
                begin_positions=begin_positions,
                valid_steps=valid_steps,
            )
        keep_n = max(1, self.cfg.engram_max_ngram_size - 1)
        for b in range(bsz):
            k = valid_steps[b] if valid_steps is not None else steps_run
            if k < steps_run:
                new_pos = begin_positions[b] + k
                session.kv_caches[b]["seqlen"] = new_pos
                if self.engram_hash is not None and "engram_tail" in session.kv_caches[b]:
                    orig_tail = session.orig_engram_tails[b]
                    parts = []
                    if orig_tail is not None and begin_positions[b] > 0 and orig_tail.size(1) > 0:
                        parts.append(orig_tail[0])
                    if k > 0:
                        safe = session.tok_hist[b, :k].clamp(
                            0, self.engram_hash.token_map.numel() - 1
                        )
                        parts.append(self.engram_hash.token_map[safe])
                    if parts:
                        tail = torch.cat(parts, dim=0)[-keep_n:]
                        session.kv_caches[b]["engram_tail"] = tail.unsqueeze(0).clone()
                    else:
                        session.kv_caches[b].pop("engram_tail", None)
            else:
                session.kv_caches[b]["seqlen"] = positions[b]
        session.layer_states = []
        session.closed = True

    @torch.no_grad()
    def forward_hidden_cached_batch(
        self,
        new_input_ids: torch.Tensor,
        kv_caches: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Runs one batched 1-token decode across `B` active sequences (`new_input_ids: [B, 1]`).

        Convenience wrapper around `begin_batched_decode` / `step_batched_decode` /
        `end_batched_decode`; callers decoding many consecutive steps for the same streams
        should hold a session open across the steps to amortise the state build/write-back.
        """
        bsz, seqlen = new_input_ids.shape
        if bsz == 1:
            return self.forward_hidden_cached(new_input_ids, kv_caches[0])
        assert seqlen == 1 and len(kv_caches) == bsz
        session = self.begin_batched_decode(kv_caches, max_steps=1)
        try:
            return self.step_batched_decode(new_input_ids, session)
        finally:
            self.end_batched_decode(session)


class BatchedDecodeSession:
    """Open batched decode session: padded per-layer batched state for `B` streams."""

    def __init__(
        self,
        kv_caches: list[dict[str, Any]],
        positions: list[int],
        layer_states: list[dict[str, Any]],
        bsz: int,
        device: torch.device,
    ):
        self.kv_caches = kv_caches
        self.positions = positions
        self.layer_states = layer_states
        self.bsz = bsz
        self.device = device
        self.closed = False
        self.begin_positions: list[int] = list(positions)
        self.orig_engram_tails: list[torch.Tensor | None] = [None] * bsz
        self.tok_hist: torch.Tensor = torch.zeros((bsz, 1), dtype=torch.long, device=device)
        self.steps_run: int = 0


# ==============================================================================
# 3. Loading & De-Quantizing `dsv41_e128_2bit` + Exporting 2-Bit Checkpoints
# ==============================================================================


def _copy_cns_file_to_local(cns_path: str, local_dir: str) -> str:
    local_path = os.path.join(local_dir, os.path.basename(cns_path))
    if not os.path.exists(local_path):
        with gfile.Open(cns_path, "rb") as f_in, open(local_path, "wb") as f_out:
            while True:
                buf = f_in.read(32 << 20)
                if not buf:
                    break
                f_out.write(buf)
    return local_path


def _load_compact_engram_from_shards(
    engram_dir: str,
    model: Dsv41SftModel,
    device: torch.device,
    is_dist: bool,
    rank: int,
    world_size: int,
) -> int:
    """Loads sharded compact Engram tables (`model-engram-00001..00016.safetensors`) + `tokenizer.json`."""
    engram_blocks = [blk for blk in model.layers if blk.engram is not None]
    if not engram_blocks or not model.cfg.load_compact_engram:
        return 0

    tok_path = os.path.join(engram_dir, "tokenizer.json")
    if model.engram_hash is not None and gfile.Exists(tok_path):
        try:
            lookup, _ = build_compressed_token_map_from_file(tok_path)
            model.engram_hash.set_token_map(lookup, model.cfg.engram_pad_id)
        except Exception as e:  # pylint: disable=broad-exception-caught
            if rank == 0:
                print(f"[dsv41_sft_model] Warning: could not load tokenizer.json for Engram: {e}", flush=True)

    probe_shard = os.path.join(engram_dir, "model-engram-00001-of-00016.safetensors")
    if not gfile.Exists(probe_shard):
        return 0

    # Wave 1 (shards 1..9 -> layer 1, since shard 9 has layers.1.engram.embed_tokens.row_map),
    # Wave 2 (shards 9..16 -> layer 14)
    loaded_count = 0
    waves = [
        (0, [f"model-engram-{i:05d}-of-00016.safetensors" for i in range(1, 10)]),
        (1, [f"model-engram-{i:05d}-of-00016.safetensors" for i in range(9, 17)]),
    ]
    staged_root = "/tmp/dsv41_e128_engram_staged"
    if is_dist and engram_dir.startswith("/cns/"):
        if rank == 0:
            os.makedirs(staged_root, exist_ok=True)
        dist.barrier()

    for wave_idx, wave_files in waves:
        if wave_idx >= len(engram_blocks):
            break
        blk = engram_blocks[wave_idx]
        eng = blk.engram
        assert eng is not None
        lid = blk.layer_id
        prefix = f"layers.{lid}.engram."
        w_prefix = f"{prefix}embed.weight.rows_"
        s_key = f"{prefix}embed.scale"
        rm_key = f"{prefix}embed_tokens.row_map"

        eng.table_weight = torch.empty(
            (eng.table_rows, eng.head_dim), dtype=torch.float8_e4m3fn, device=device
        )
        eng.table_scale = torch.empty(
            (eng.table_rows, eng.head_dim // FP8_BLOCK_SIZE),
            dtype=torch.uint8,
            device=device,
        )
        eng.row_map = torch.empty(
            (eng.part_vocab_size,), dtype=torch.int32, device=device
        )

        if is_dist and engram_dir.startswith("/cns/"):
            my_files = wave_files[rank::world_size]

            def _stage_eng(fname: str) -> None:
                cns_src = os.path.join(engram_dir, fname)
                local_dst = os.path.join(staged_root, fname)
                tmp_dst = f"{local_dst}.tmp.{rank}"
                if not os.path.exists(local_dst) and gfile.Exists(cns_src):
                    with gfile.Open(cns_src, "rb") as f_in, open(tmp_dst, "wb") as f_out:
                        while True:
                            buf = f_in.read(32 << 20)
                            if not buf:
                                break
                            f_out.write(buf)
                    os.replace(tmp_dst, local_dst)

            with futures.ThreadPoolExecutor(max_workers=max(1, len(my_files))) as pool:
                list(pool.map(_stage_eng, my_files))
            dist.barrier()
            local_paths = [
                os.path.join(staged_root, fn)
                for fn in wave_files
                if os.path.exists(os.path.join(staged_root, fn))
            ]
        else:
            local_paths = [
                os.path.join(engram_dir, fn)
                for fn in wave_files
                if os.path.exists(os.path.join(engram_dir, fn))
            ]

        c_start = eng.compact_start
        c_end = c_start + eng.table_rows
        v_start = eng.vocab_start_idx
        v_end = eng.vocab_end_idx

        for lp in local_paths:
            with safetensors.safe_open(lp, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                if s_key in keys:
                    s_slice = f.get_slice(s_key)[c_start:c_end]
                    eng.table_scale.copy_(s_slice.view(torch.uint8).to(device))
                    loaded_count += 1
                if rm_key in keys:
                    rm_slice = f.get_slice(rm_key)[v_start:v_end]
                    rm_dev = rm_slice.to(device=device, dtype=torch.int32)
                    eng.row_map.copy_(
                        torch.where(rm_dev >= 0, rm_dev - c_start, rm_dev)
                    )
                    loaded_count += 1
                for k in keys:
                    if not k.startswith(w_prefix):
                        continue
                    tail = k[len(w_prefix) :]
                    w_slice_obj = f.get_slice(k)
                    if "_" in tail:
                        s_str, e_str = tail.split("_", 1)
                        r0, r1 = int(s_str), int(e_str)
                    else:
                        r0 = int(tail)
                        r1 = r0 + int(w_slice_obj.get_shape()[0])
                    ov_start = max(r0, c_start)
                    ov_end = min(r1, c_end)
                    if ov_start < ov_end:
                        w_slice = w_slice_obj[ov_start - r0 : ov_end - r0]
                        eng.table_weight[
                            ov_start - c_start : ov_end - c_start
                        ].copy_(w_slice.view(torch.float8_e4m3fn).to(device))
                        loaded_count += 1

        eng.has_compact_table = True
        if is_dist and engram_dir.startswith("/cns/"):
            dist.barrier()
            if rank == 0:
                for lp in local_paths:
                    # Keep shard 00009 after wave 0 since wave 1 needs it
                    if wave_idx == 0 and lp.endswith("00009-of-00016.safetensors"):
                        continue
                    if os.path.exists(lp):
                        os.remove(lp)
            dist.barrier()

    return loaded_count


REPAIR_SHARD_PREFIX = "model-repair-"


LEGACY_BACKBONE_SHARDS = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def _list_backbone_shards(model_dir: str) -> list[str]:
    """Returns the non-Engram, non-gptq backbone shard basenames of a checkpoint, in index order.

    Prefers `model.safetensors.index.json` (`weight_map` values), which covers both the legacy
    `model-0000N-of-00002.safetensors` layout and the 3-shard `model-0000N.safetensors` layout of
    the image checkpoints; Engram row chunks (`model-engram-*`) are loaded separately.
    """
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    shards: list[str] = []
    try:
        if gfile.Exists(idx_path):
            with gfile.Open(idx_path, "r") as f:
                weight_map = (json.load(f) or {}).get("weight_map") or {}
            shards = sorted(
                {
                    os.path.basename(v)
                    for v in weight_map.values()
                    if v.endswith(".safetensors") and "engram" not in os.path.basename(v)
                }
            )
    except Exception:  # pylint: disable=broad-exception-caught
        shards = []
    if not shards:
        shards = [s for s in LEGACY_BACKBONE_SHARDS if gfile.Exists(os.path.join(model_dir, s))]
    if not shards:
        shards = list(LEGACY_BACKBONE_SHARDS)
    return shards


def _list_repair_shards(model_dir: str) -> list[str]:
    """Returns sorted basenames of optional `model-repair-*.safetensors` shards in `model_dir`."""
    try:
        entries = gfile.ListDir(model_dir) if gfile.Exists(model_dir) else []
    except Exception:  # pylint: disable=broad-exception-caught
        entries = []
    return sorted(
        os.path.basename(e)
        for e in entries
        if os.path.basename(e).startswith(REPAIR_SHARD_PREFIX)
        and e.endswith(".safetensors")
    )


def _load_fp8_backbone_shards(
    local_shards: list[str],
    param_dict: dict[str, torch.Tensor],
    buf_dict: dict[str, torch.Tensor],
    device: torch.device,
    stats: dict[str, Any],
) -> None:
    """De-quantizes FP8 block-32 backbone tensors from `local_shards` into `param_dict` / `buf_dict`.

    The `.scale` tensor of an FP8 weight may live in a *different* shard than the weight itself
    (e.g. `layers.9.ffn.shared_experts.w{1,2,3}` in chankhavu's 2-shard export), so scales are
    resolved across all shards instead of only within the shard currently being read. An FP8
    tensor whose scale cannot be found anywhere is an error rather than a silent raw copy.
    """
    key_to_shard: dict[str, str] = {}
    for shard_path in local_shards:
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                key_to_shard[k] = shard_path

    handles: dict[str, Any] = {}
    try:
        for shard_path in local_shards:
            handles[shard_path] = safetensors.safe_open(
                shard_path, framework="pt", device=str(device)
            )
        for k in sorted(key_to_shard):
            if k.endswith(".scale"):
                continue
            target = param_dict.get(k)
            if target is None:
                target = buf_dict.get(k)
            if target is None:
                continue
            raw_t = handles[key_to_shard[k]].get_tensor(k)
            scale_key = k[:-7] + ".scale" if k.endswith(".weight") else ""
            scale_shard = key_to_shard.get(scale_key) if scale_key else None
            if scale_shard is not None:
                raw_s = handles[scale_shard].get_tensor(scale_key)
                deq = dequantize_fp8_block32_torch(raw_t, raw_s, out_dtype=target.dtype)
                target.data.copy_(deq)
                stats["fp8_dequantized_tensors"] += 1
            else:
                if raw_t.is_floating_point() and raw_t.element_size() == 1:
                    raise ValueError(
                        f"FP8 tensor {k!r} has no {scale_key!r} in any backbone shard; refusing to "
                        "copy raw FP8 codes without their block scales."
                    )
                target.data.copy_(raw_t.to(target.dtype))
                stats["dense_copied_tensors"] += 1
    finally:
        for h in handles.values():
            close = getattr(h, "__exit__", None)
            if close is not None:
                close(None, None, None)

    # Sanity guard against silently loading unscaled FP8 codes (e4m3 codes reach 448 while real
    # projection weights are O(0.01-1)). This is exactly how `layers.9.ffn.shared_experts.*`
    # was corrupted in earlier runs when its cross-shard `.scale` was not resolved.
    suspicious: list[tuple[str, float]] = []
    for k in sorted(key_to_shard):
        if not k.endswith(".weight"):
            continue
        t = param_dict.get(k)
        if t is None:
            t = buf_dict.get(k)
        if t is None or t.dim() != 2 or t.numel() < (1 << 20):
            continue
        amax = float(t.detach().abs().max())
        if amax > 64.0:
            suspicious.append((k, amax))
    if suspicious:
        raise ValueError(
            "Backbone load sanity check failed: projection weights with |w|max > 64 (unscaled FP8 "
            f"codes?): {suspicious[:8]}{' ...' if len(suspicious) > 8 else ''}"
        )
    stats["backbone_max_abs_checked_tensors"] = stats.get(
        "backbone_max_abs_checked_tensors", 0
    ) + sum(
        1
        for k in key_to_shard
        if k.endswith(".weight")
        and (param_dict.get(k) if param_dict.get(k) is not None else buf_dict.get(k)) is not None
    )


def load_dequantized_dsv41_e128(
    model_dir: str,
    model: Dsv41SftModel,
    device: torch.device,
    engram_model_dir: str | None = None,
) -> dict[str, Any]:
    """Loads `model-00001/00002.safetensors` (FP8 block-32) + `gptq/layer_00..39.safetensors` (2-bit Humming) + compact Engram into `model`."""
    stats = {
        "fp8_dequantized_tensors": 0,
        "dense_copied_tensors": 0,
        "gptq_2bit_experts_unpacked": 0,
        "engram_slices_loaded": 0,
    }
    param_dict = dict(model.named_parameters())
    buf_dict = dict(model.named_buffers())

    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    backbone_rel = _list_backbone_shards(model_dir)
    stats["backbone_shards"] = list(backbone_rel)
    # Optional repair shards (e.g. re-quantised `layers.9.ffn.shared_experts.*` weight+scale pairs
    # for legacy exports that dropped cross-shard scales). Loaded last so their keys override.
    backbone_rel += _list_repair_shards(model_dir)
    gptq_rel = [f"gptq/layer_{i:02d}.safetensors" for i in range(len(model.layers))]
    all_rel = backbone_rel + gptq_rel

    if is_dist and model_dir.startswith("/cns/"):
        local_rank = model.ep_rank if model.ep_world_size > 1 else (rank % max(1, world_size))
        local_world_size = model.ep_world_size if model.ep_world_size > 1 else world_size
        staged_root = "/tmp/dsv41_e128_staged"
        if local_rank == 0:
            os.makedirs(os.path.join(staged_root, "gptq"), exist_ok=True)
        dist.barrier()
        my_rel = all_rel[local_rank::local_world_size]

        def _stage_one(rel: str) -> None:
            cns_src = os.path.join(model_dir, rel)
            if not gfile.Exists(cns_src):
                raise FileNotFoundError(f"Missing checkpoint shard on CNS: {cns_src}")
            local_dst = os.path.join(staged_root, rel)
            tmp_dst = f"{local_dst}.tmp.{local_rank}"
            if not os.path.exists(local_dst):
                with gfile.Open(cns_src, "rb") as f_in, open(tmp_dst, "wb") as f_out:
                    while True:
                        buf = f_in.read(32 << 20)
                        if not buf:
                            break
                        f_out.write(buf)
                os.replace(tmp_dst, local_dst)

        with futures.ThreadPoolExecutor(max_workers=min(6, max(1, len(my_rel)))) as pool:
            list(pool.map(_stage_one, my_rel))
        dist.barrier()

        # 1. Load & de-quantize non-routed backbone from staged model-00001/00002.safetensors
        #    (scales are resolved across both shards; see `_load_fp8_backbone_shards`).
        _load_fp8_backbone_shards(
            [os.path.join(staged_root, s) for s in backbone_rel],
            param_dict,
            buf_dict,
            device,
            stats,
        )

        # 2. Unpack & de-quantize this rank's 2-bit routed experts from staged gptq/layer_00..39.safetensors
        for layer_idx, block in enumerate(model.layers):
            local_layer = os.path.join(staged_root, f"gptq/layer_{layer_idx:02d}.safetensors")
            moe = block.ffn
            moe._clear_stacked_eval_weights()
            if model.cfg.enable_qat:
                moe.shared_experts.snapshot_teacher_weights()
            with safetensors.safe_open(local_layer, framework="pt", device=str(device)) as f:
                for e_idx in range(moe.experts_start_idx, moe.experts_end_idx):
                    expert = moe.experts[e_idx]
                    assert expert is not None
                    for proj_name, default_group in (("w1", 128), ("w3", 128), ("w2", 64)):
                        q = f.get_tensor(f"e{e_idx}.{proj_name}.q")
                        s = f.get_tensor(f"e{e_idx}.{proj_name}.s")
                        z = f.get_tensor(f"e{e_idx}.{proj_name}.z")
                        proj_mod = getattr(expert, proj_name)
                        group = int(
                            getattr(
                                proj_mod,
                                "group",
                                (q.size(1) * 16) // max(1, s.size(1))
                                if s.dim() == 2 and s.size(1) > 0
                                else default_group,
                            )
                        )
                        w_bf16 = dequant_gptq_expert_tensor(
                            q, s, z, group=group, out_dtype=torch.bfloat16
                        )
                        if isinstance(proj_mod, LoRALinear):
                            proj_mod.set_gptq_state(w_bf16, s, z)
                        else:
                            if proj_mod.weight.data.shape != w_bf16.shape:
                                proj_mod.weight.data = torch.empty_like(
                                    w_bf16, dtype=proj_mod.weight.dtype
                                )
                            proj_mod.weight.data.copy_(w_bf16)
                    stats["gptq_2bit_experts_unpacked"] += 1

        dist.barrier()
        if local_rank == 0:
            for rel in all_rel:
                p = os.path.join(staged_root, rel)
                if os.path.exists(p):
                    os.remove(p)
        dist.barrier()

        eng_dir = _resolve_engram_dir(engram_model_dir or model_dir)
        stats["engram_dir"] = eng_dir
        stats["engram_slices_loaded"] = _load_compact_engram_from_shards(
            eng_dir, model, device, is_dist, local_rank, local_world_size
        )
        return stats

    with tempfile.TemporaryDirectory(prefix="dsv41_load_") as tmp_dir:
        # 1. Load & de-quantize non-routed backbone from model-00001/00002.safetensors. Both shards are
        #    staged before loading so FP8 scales can be resolved across shards (`_load_fp8_backbone_shards`).
        local_shards: list[str] = []
        for shard_name in backbone_rel:
            cns_shard = os.path.join(model_dir, shard_name)
            if not gfile.Exists(cns_shard):
                raise FileNotFoundError(f"Missing backbone shard: {cns_shard}")
            local_shards.append(_copy_cns_file_to_local(cns_shard, tmp_dir))
        _load_fp8_backbone_shards(local_shards, param_dict, buf_dict, device, stats)
        for local_shard in local_shards:
            os.remove(local_shard)

        # 2. Unpack & de-quantize this rank's 2-bit routed experts from gptq/layer_00..39.safetensors
        gptq_dir = os.path.join(model_dir, "gptq")
        for layer_idx, block in enumerate(model.layers):
            cns_layer = os.path.join(gptq_dir, f"layer_{layer_idx:02d}.safetensors")
            if not gfile.Exists(cns_layer):
                raise FileNotFoundError(f"Missing 2-bit expert shard: {cns_layer}")
            local_layer = _copy_cns_file_to_local(cns_layer, tmp_dir)
            moe = block.ffn
            moe._clear_stacked_eval_weights()
            if model.cfg.enable_qat:
                moe.shared_experts.snapshot_teacher_weights()
            with safetensors.safe_open(local_layer, framework="pt", device=str(device)) as f:
                for e_idx in range(moe.experts_start_idx, moe.experts_end_idx):
                    expert = moe.experts[e_idx]
                    assert expert is not None
                    for proj_name, default_group in (("w1", 128), ("w3", 128), ("w2", 64)):
                        q = f.get_tensor(f"e{e_idx}.{proj_name}.q")
                        s = f.get_tensor(f"e{e_idx}.{proj_name}.s")
                        z = f.get_tensor(f"e{e_idx}.{proj_name}.z")
                        proj_mod = getattr(expert, proj_name)
                        group = int(
                            getattr(
                                proj_mod,
                                "group",
                                (q.size(1) * 16) // max(1, s.size(1))
                                if s.dim() == 2 and s.size(1) > 0
                                else default_group,
                            )
                        )
                        w_bf16 = dequant_gptq_expert_tensor(
                            q, s, z, group=group, out_dtype=torch.bfloat16
                        )
                        if isinstance(proj_mod, LoRALinear):
                            proj_mod.set_gptq_state(w_bf16, s, z)
                        else:
                            if proj_mod.weight.data.shape != w_bf16.shape:
                                proj_mod.weight.data = torch.empty_like(
                                    w_bf16, dtype=proj_mod.weight.dtype
                                )
                            proj_mod.weight.data.copy_(w_bf16)
                    stats["gptq_2bit_experts_unpacked"] += 1
            os.remove(local_layer)

        eng_dir = _resolve_engram_dir(engram_model_dir or model_dir)
        stats["engram_dir"] = eng_dir
        stats["engram_slices_loaded"] = _load_compact_engram_from_shards(
            eng_dir, model, device, False, 0, 1
        )

    return stats


def _resolve_engram_dir(preferred_dir: str) -> str:
    """Returns `preferred_dir` if it holds the compact Engram shards, else the E043 default dir.

    The image checkpoints (`dsv41_img_e128_2bit_causal`) and every export derived from them ship no
    `model-engram-*.safetensors`; their `config.json` carries the identical `engram_compact_kept`
    table layout, so the E043 shards are reused when loading from CNS.
    """
    marker = "model-engram-00001-of-00016.safetensors"
    if not preferred_dir.startswith("/cns/"):
        return preferred_dir
    if gfile.Exists(os.path.join(preferred_dir, marker)):
        return preferred_dir
    default_base = "/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit"
    if gfile.Exists(os.path.join(default_base, marker)):
        return default_base
    return preferred_dir


@torch.no_grad()
def export_dsv41_e128_2bit_checkpoint(
    model: Dsv41SftModel,
    base_model_dir: str,
    export_cns_dir: str,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """Exports the trained `Dsv41SftModel` into a complete 2-bit Humming + FP8 block-32 checkpoint on CNS."""
    if rank == 0:
        export_and_quantize.ensure_cns_quota_dir(export_cns_dir)
        export_and_quantize.ensure_cns_quota_dir(
            os.path.join(export_cns_dir, "gptq")
        )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()

    exported_files: list[str] = []
    # On-disk delta counters (what a byte diff against the base checkpoint would show).
    delta: dict[str, int] = {
        "routed_codes_total": 0,
        "routed_codes_changed": 0,
        "routed_scales_total": 0,
        "routed_scales_changed": 0,
        "backbone_trainable_tensors": 0,
        "backbone_trainable_elems": 0,
        "backbone_changed_elems": 0,
    }
    # 1. Export 2-bit Humming routed experts (`gptq/layer_00..39.safetensors`) FIRST
    routed_experts_trained = (
        model.cfg.trainable_routed_experts_dense
        or model.cfg.routed_expert_lora_r > 0
        or (model.cfg.enable_qat and model.cfg.qat_trainable_expert_scales)
    )
    if not routed_experts_trained:
        for layer_idx in range(rank, model.cfg.n_layers, world_size):
            rel = f"gptq/layer_{layer_idx:02d}.safetensors"
            src_cns = os.path.join(base_model_dir, rel)
            dst_cns = os.path.join(export_cns_dir, rel)
            if gfile.Exists(src_cns) and not gfile.Exists(dst_cns):
                with gfile.Open(src_cns, "rb") as f_in, gfile.Open(
                    dst_cns, "wb"
                ) as f_out:
                    while True:
                        buf = f_in.read(32 << 20)
                        if not buf:
                            break
                        f_out.write(buf)
                exported_files.append(rel)
    else:
        ep_ws = max(1, model.ep_world_size)
        is_primary_ep_replica = rank < ep_ws
        with tempfile.TemporaryDirectory(
            prefix=f"dsv41_export_gptq_r{rank}_"
        ) as tmp_dir:
            pending_writes: list[tuple[int, dict[str, torch.Tensor]]] = []
            for layer_idx, block in enumerate(model.layers):
                moe = block.ffn
                local_expert_dict: dict[str, torch.Tensor] = {}
                if is_primary_ep_replica:
                    for e_idx in range(moe.experts_start_idx, moe.experts_end_idx):
                        expert = moe.experts[e_idx]
                        assert expert is not None
                        for proj_name, group in (("w1", 128), ("w3", 128), ("w2", 64)):
                            proj_mod = getattr(expert, proj_name)
                            if isinstance(proj_mod, LoRALinear):
                                q, s, z, p_stats = proj_mod.export_humming_2bit_with_stats()
                                delta["routed_codes_total"] += p_stats["codes_total"]
                                delta["routed_scales_total"] += p_stats["scales_total"]
                                if p_stats["codes_changed"] >= 0:
                                    delta["routed_codes_changed"] += p_stats["codes_changed"]
                                    delta["routed_scales_changed"] += p_stats["scales_changed"]
                            else:
                                w_full = (
                                    proj_mod.merged_weight()
                                    if hasattr(proj_mod, "merged_weight")
                                    else proj_mod.weight
                                )
                                q, s, z = quantize_uint2_humming_expert_tensor(
                                    w_full, group=group
                                )
                            local_expert_dict[f"e{e_idx}.{proj_name}.q"] = q.cpu()
                            local_expert_dict[f"e{e_idx}.{proj_name}.s"] = s.cpu()
                            local_expert_dict[f"e{e_idx}.{proj_name}.z"] = z.cpu()

                writer_rank = layer_idx % max(1, world_size)
                if world_size > 1 and dist.is_available() and dist.is_initialized():
                    gathered: list[dict[str, torch.Tensor]] = [
                        {} for _ in range(world_size)
                    ]
                    dist.all_gather_object(gathered, local_expert_dict)
                    if rank == writer_rank:
                        merged_dict: dict[str, torch.Tensor] = {}
                        for d in gathered:
                            merged_dict.update(d)
                        pending_writes.append((layer_idx, merged_dict))
                else:
                    pending_writes.append((layer_idx, local_expert_dict))

            def _write_layer_shard(item: tuple[int, dict[str, torch.Tensor]]) -> str:
                l_idx, m_dict = item
                local_out = os.path.join(tmp_dir, f"layer_{l_idx:02d}.safetensors")
                save_file(m_dict, local_out)
                dst_cns = os.path.join(
                    export_cns_dir, "gptq", f"layer_{l_idx:02d}.safetensors"
                )
                with open(local_out, "rb") as f_in, gfile.Open(
                    dst_cns, "wb"
                ) as f_out:
                    while True:
                        buf = f_in.read(32 << 20)
                        if not buf:
                            break
                        f_out.write(buf)
                os.remove(local_out)
                return f"gptq/layer_{l_idx:02d}.safetensors"

            if pending_writes:
                with futures.ThreadPoolExecutor(
                    max_workers=min(5, len(pending_writes))
                ) as pool:
                    exported_files.extend(
                        pool.map(_write_layer_shard, pending_writes)
                    )

    # 2. One rank per shard exports the updated non-routed backbone shards (2-shard legacy layout
    #    or the 3-shard image layout, discovered from the base index), preserving 2D `[out//32, in//32]`
    #    vs 1D `[out, in//32]` scale shapes from the original checkpoint. Trainable non-FP8 tensors
    #    (e.g. vision tower / aligner / `image_*` under FP32 masters) are written back in their
    #    original dtype; frozen tensors are copied byte-for-byte.
    backbone_shards = tuple(_list_backbone_shards(base_model_dir))
    my_backbone_shards = [
        s
        for idx, s in enumerate(backbone_shards)
        if rank == (idx % max(1, world_size))
    ]
    if my_backbone_shards or rank == 0:
        param_dict = dict(model.named_parameters())
        buf_dict = dict(model.named_buffers())
        with tempfile.TemporaryDirectory(
            prefix=f"dsv41_export_backbone_r{rank}_"
        ) as tmp_dir:
            # Stage *both* base shards: an FP8 weight's `.scale` may live in the other shard
            # (e.g. `layers.9.ffn.shared_experts.w{1,2,3}`), and must be read from there.
            local_by_shard: dict[str, str] = {}
            for shard_name in backbone_shards:
                src_cns = os.path.join(base_model_dir, shard_name)
                if gfile.Exists(src_cns):
                    local_by_shard[shard_name] = _copy_cns_file_to_local(src_cns, tmp_dir)
            key_to_shard: dict[str, str] = {}
            for shard_name, local_p in local_by_shard.items():
                with safetensors.safe_open(local_p, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        key_to_shard[k] = shard_name
            # Scales that get re-homed next to their weight in the exported shards.
            moved_scales: dict[str, str] = {
                s: key_to_shard[s[:-6] + ".weight"]
                for s, home in key_to_shard.items()
                if s.endswith(".scale")
                and (s[:-6] + ".weight") in key_to_shard
                and key_to_shard[s[:-6] + ".weight"] != home
            }

            for shard_name in my_backbone_shards:
                local_src = local_by_shard.get(shard_name)
                if local_src is None:
                    continue
                out_tensors: dict[str, torch.Tensor] = {}
                handles = {
                    name: safetensors.safe_open(p, framework="pt", device="cpu")
                    for name, p in local_by_shard.items()
                }
                try:
                    f = handles[shard_name]
                    orig_keys = set(f.keys())
                    for k in sorted(orig_keys):
                        if k.endswith(".scale"):
                            continue
                        scale_key = (
                            k[:-7] + ".scale" if k.endswith(".weight") else ""
                        )
                        scale_home = key_to_shard.get(scale_key) if scale_key else None
                        f_scale = handles[scale_home] if scale_home is not None else None
                        src_param = param_dict.get(k)
                        if src_param is None:
                            src_param = buf_dict.get(k)
                        if src_param is None or not getattr(
                            src_param, "requires_grad", False
                        ):
                            out_tensors[k] = f.get_tensor(k)
                            if f_scale is not None:
                                out_tensors[scale_key] = f_scale.get_tensor(scale_key)
                            continue
                        if f_scale is not None:
                            orig_s_shape = f_scale.get_slice(scale_key).get_shape()
                            is_2d_scale = (
                                len(orig_s_shape) == 2
                                and orig_s_shape[0]
                                == src_param.shape[0] // FP8_BLOCK_SIZE
                            )
                            w_fp8, s_e8m0 = quantize_fp8_block32_torch(
                                src_param.data, scale_2d=is_2d_scale
                            )
                            out_tensors[k] = w_fp8.cpu()
                            out_tensors[scale_key] = s_e8m0.cpu()
                            # Value-level delta vs the base (bit diffs over-count: a halved UE8M0
                            # scale re-codes a whole block without changing any value).
                            try:
                                base_deq = dequantize_fp8_block32_torch(
                                    f.get_tensor(k), f_scale.get_tensor(scale_key), out_dtype=torch.bfloat16
                                )
                                new_deq = dequantize_fp8_block32_torch(
                                    w_fp8.cpu(), s_e8m0.cpu(), out_dtype=torch.bfloat16
                                )
                                delta["backbone_changed_elems"] += int((base_deq != new_deq).sum().item())
                            except Exception:  # pylint: disable=broad-exception-caught
                                pass
                            delta["backbone_trainable_tensors"] += 1
                            delta["backbone_trainable_elems"] += int(src_param.numel())
                        else:
                            base_t = f.get_tensor(k)
                            orig_dtype = base_t.dtype
                            if orig_dtype.is_floating_point and torch.empty(
                                0, dtype=orig_dtype
                            ).element_size() == 1:
                                raise ValueError(
                                    f"Cannot export FP8 tensor {k!r}: no {scale_key!r} found in any "
                                    "base backbone shard."
                                )
                            new_t = src_param.data.to(orig_dtype).cpu()
                            out_tensors[k] = new_t
                            delta["backbone_trainable_tensors"] += 1
                            delta["backbone_trainable_elems"] += int(new_t.numel())
                            if base_t.shape == new_t.shape:
                                delta["backbone_changed_elems"] += int((base_t != new_t).sum().item())
                finally:
                    for h in handles.values():
                        close = getattr(h, "__exit__", None)
                        if close is not None:
                            close(None, None, None)

                local_out = os.path.join(tmp_dir, f"out_{shard_name}")
                save_file(out_tensors, local_out)
                dst_cns = os.path.join(export_cns_dir, shard_name)
                with open(local_out, "rb") as f_in, gfile.Open(
                    dst_cns, "wb"
                ) as f_out:
                    while True:
                        buf = f_in.read(32 << 20)
                        if not buf:
                            break
                        f_out.write(buf)
                exported_files.append(shard_name)
                os.remove(local_out)
            for local_p in local_by_shard.values():
                if os.path.exists(local_p):
                    os.remove(local_p)

            # Copy configs, tokenizer, masks, and scripts on Rank 0
            if rank == 0 and gfile.Exists(base_model_dir):
                for entry in sorted(gfile.ListDir(base_model_dir)):
                    if entry.endswith(".safetensors") or entry == "gptq":
                        continue
                    src_p = os.path.join(base_model_dir, entry)
                    dst_p = os.path.join(export_cns_dir, entry)
                    if gfile.IsDirectory(src_p):
                        continue
                    if entry == "model.safetensors.index.json" and moved_scales:
                        # Keep the index consistent with scales re-homed next to their weights.
                        with gfile.Open(src_p, "r") as f_in:
                            index_doc = json.load(f_in)
                        wm = index_doc.get("weight_map", {})
                        for s_key, new_home in moved_scales.items():
                            if s_key in wm:
                                wm[s_key] = new_home
                        with gfile.Open(dst_p, "w") as f_out:
                            json.dump(index_doc, f_out, indent=2)
                        exported_files.append(entry)
                        continue
                    with gfile.Open(src_p, "rb") as f_in, gfile.Open(
                        dst_p, "wb"
                    ) as f_out:
                        f_out.write(f_in.read())
                    exported_files.append(entry)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        keys = sorted(delta)
        dev = model.embed.weight.device
        t = torch.tensor([delta[k] for k in keys], dtype=torch.int64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        delta = dict(zip(keys, (int(v) for v in t.tolist())))
        dist.barrier()

    def _frac(num: int, den: int) -> float:
        return round(float(num) / float(den), 6) if den > 0 else 0.0

    return {
        "export_cns_dir": export_cns_dir,
        "routed_expert_lora_r": model.cfg.routed_expert_lora_r,
        "trainable_routed_experts_dense": model.cfg.trainable_routed_experts_dense,
        "enable_qat": model.cfg.enable_qat,
        "exported_files_count": len(exported_files),
        **delta,
        "routed_codes_changed_frac": _frac(delta["routed_codes_changed"], delta["routed_codes_total"]),
        "routed_scales_changed_frac": _frac(delta["routed_scales_changed"], delta["routed_scales_total"]),
        "backbone_changed_frac": _frac(delta["backbone_changed_elems"], delta["backbone_trainable_elems"]),
    }

