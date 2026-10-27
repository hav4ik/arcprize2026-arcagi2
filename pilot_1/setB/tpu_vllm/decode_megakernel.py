"""DeepSeek-V4.1-Flash-1017 Pallas TPU kernels and Macro-Kernels for Ghostfish.

Implements:
  1. Exact 128-lane vector primitives in Pallas:
     - _lane_gather & 5-step XOR butterfly reduction (block=32 and block=16)
     - act_quant_fp8_bf16 (E4M3FN + E8M0 scale, exact in BF16)
     - fp4_act_quant_bf16 (E2M1 + E8M0 for Indexer Q/K, E2M1 + E4M3 for Main KV)
     - dequant_fp8_weight & dequant_mxfp4_weight (uint32 bitcast to
     float4_e2m1fn)
     - fp8_matmul & mxfp4_matmul (BF16 x BF16 -> FP32 MXU accumulation, avoiding
       Ghostfish's FloatE8M8 8-bit accumulator error)
     - apply_rope_tail128 (interleaved GPT-J RoPE & inverse RoPE on trailing 64
       channels via a single XOR-1 lane gather + FMA)
     - mega_mhc_step (Single-Pass mHC post + 20-iter Sinkhorn + pre + RMSNorm)
     - all_reduce_rows (in-kernel remote DMA all-reduce across TP/EP ranks)
     - routed_experts_stream (selective active-expert HBM->VMEM double-buffered
       DMA stream with clamped SwiGLU)
  2. Standalone & Composed Macro-Kernels:
     - pallas_act_quant_fp8, pallas_fp4_quant, pallas_fp8_linear,
       pallas_mxfp4_linear, pallas_mega_mhc, pallas_routed_experts,
       pallas_all_reduce
     - encoder_macrokernel (Layers 0..L_dec-1 + L_dec Global KV/IndexK write,
       with Group-Cached VMEM Global KV across 6-layer CSA2 m=2 groups)
     - decoder_macrokernel (Layers L_dec..n_layers-1 + 2-level Hierarchical
       Candidate Indexer + L37..39 aux capture + final_norm + sharded lm_head)
     - dspark_macrokernel (Fused prefix verification + state rewind + 3-layer
       mtp.0..2 MoE+SWA drafter + rank-256 Markov & Confidence heads)
"""

from __future__ import annotations

# pylint: disable=g-doc-args,g-doc-return-or-yield,missing-function-docstring,g-long-ternary

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_layout as layout_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_ref as ref

F32 = jnp.float32
BF16 = jnp.bfloat16
FP8 = jnp.float8_e4m3fn
FP4 = jnp.float4_e2m1fn
U32 = jnp.uint32
U8 = jnp.uint8
I32 = jnp.int32

LANES = 128
ROWS = 16


def _lane_gather(operand: jax.Array, indices: jax.Array) -> jax.Array:
  """Permutes the trailing 128-lane axis by constant 1D indices (lowers to XLU shuffle)."""
  rank = operand.ndim
  lanes = indices.shape[0]
  coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
  batching = tuple(range(rank - 1))
  dnums = lax.GatherDimensionNumbers(
      offset_dims=(),
      collapsed_slice_dims=(rank - 1,),
      start_index_map=(rank - 1,),
      operand_batching_dims=batching,
      start_indices_batching_dims=batching,
  )
  return lax.gather(
      operand,
      coords,
      dimension_numbers=dnums,
      slice_sizes=(1,) * rank,
      unique_indices=False,
      mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
  )


def _group_amax_128(x_f32: jax.Array, group_size: int) -> jax.Array:
  """Computes per-group absmax of [..., K] (K % 128 == 0) via XOR butterfly in 128 lanes.

  Returns an array of the exact same shape as x_f32 with each group's absmax
  replicated across all lanes of that group. Zero <128 lane slices.
  """
  orig_shape = x_f32.shape
  k = orig_shape[-1]
  if k % LANES != 0:
    raise ValueError(f"Last dim {k} must be a multiple of {LANES}")
  if group_size not in (16, 32):
    raise ValueError(f"group_size must be 16 or 32, got {group_size}")
  amax = jnp.abs(x_f32).reshape(-1, LANES)
  iota = jnp.arange(LANES, dtype=I32)
  steps = (1, 2, 4, 8) if group_size == 16 else (1, 2, 4, 8, 16)
  for delta in steps:
    amax = jnp.maximum(amax, _lane_gather(amax, jnp.bitwise_xor(iota, delta)))
  return amax.reshape(orig_shape)


def _fast_round_scale_f32(amax: jax.Array, max_inv: float) -> jax.Array:
  """Computes 2^ceil(log2(amax * max_inv)) via IEEE-754 integer bit ops."""
  prod = amax * jnp.float32(max_inv)
  bits = lax.bitcast_convert_type(prod, U32)
  exp = jnp.bitwise_and(jnp.right_shift(bits, U32(23)), U32(0xFF))
  man_nz = (jnp.bitwise_and(bits, U32(0x7FFFFF)) != U32(0)).astype(U32)
  exp_ceil = jnp.minimum(exp + man_nz, U32(254))
  return lax.bitcast_convert_type(jnp.left_shift(exp_ceil, U32(23)), F32)


def act_quant_fp8_bf16(x: jax.Array, *, in_pallas: bool = False) -> jax.Array:
  """Quantizes [R, K] to per-32 FP8 (E4M3FN) + E8M0 scale and returns exact BF16.

  Because E4M3FN has 3 mantissa bits and E8M0 has 0 mantissa bits, the product
  q_fp8.astype(bf16) * s_e8m0.astype(bf16) is 100% bit-exact in bfloat16 (7
  mantissa bits) and enables full FP32 MXU accumulation on Ghostfish.
  """
  xf = x.astype(F32)
  if in_pallas:
    amax = jnp.maximum(_group_amax_128(xf, 32), jnp.float32(1e-4))
    s_f32 = _fast_round_scale_f32(amax, 1.0 / 448.0)
    q_fp8 = jnp.clip(xf / s_f32, -448.0, 448.0).astype(FP8)
    return q_fp8.astype(BF16) * s_f32.astype(BF16)
  orig_shape = x.shape
  xf_g = xf.reshape(-1, orig_shape[-1] // 32, 32)
  amax = jnp.maximum(
      jnp.max(jnp.abs(xf_g), axis=-1, keepdims=True), jnp.float32(1e-4)
  )
  s_f32 = _fast_round_scale_f32(amax, 1.0 / 448.0)
  q_fp8 = jnp.clip(xf_g / s_f32, -448.0, 448.0).astype(FP8)
  return (q_fp8.astype(BF16) * s_f32.astype(BF16)).reshape(orig_shape)


def fp4_act_quant_bf16(
    x: jax.Array, *, scale_dtype: str = "e8m0", in_pallas: bool = False
) -> jax.Array:
  """Quantizes [R, K] to FP4 (E2M1) + E8M0 (block=32) or E4M3 (block=16) in exact BF16."""
  xf = x.astype(F32)
  if in_pallas:
    if scale_dtype == "e4m3":
      amax = jnp.maximum(_group_amax_128(xf, 16), jnp.float32(6.0 * (2.0**-9)))
      s_e4m3 = (amax * jnp.float32(1.0 / 6.0)).astype(FP8)
      s_f32 = s_e4m3.astype(F32)
      s_bf16 = s_e4m3.astype(BF16)
    elif scale_dtype == "e8m0":
      amax = jnp.maximum(
          _group_amax_128(xf, 32), jnp.float32(6.0 * (2.0**-126))
      )
      s_f32 = _fast_round_scale_f32(amax, 1.0 / 6.0)
      s_bf16 = s_f32.astype(BF16)
    else:
      raise ValueError(f"Unsupported scale_dtype={scale_dtype!r}")
    q_fp4 = jnp.clip(xf / s_f32, -6.0, 6.0).astype(FP4)
    return q_fp4.astype(BF16) * s_bf16

  orig_shape = x.shape
  if scale_dtype == "e4m3":
    xf_g = xf.reshape(-1, orig_shape[-1] // 16, 16)
    amax = jnp.maximum(
        jnp.max(jnp.abs(xf_g), axis=-1, keepdims=True),
        jnp.float32(6.0 * (2.0**-9)),
    )
    s_e4m3 = (amax * jnp.float32(1.0 / 6.0)).astype(FP8)
    s_f32 = s_e4m3.astype(F32)
    s_bf16 = s_e4m3.astype(BF16)
  elif scale_dtype == "e8m0":
    xf_g = xf.reshape(-1, orig_shape[-1] // 32, 32)
    amax = jnp.maximum(
        jnp.max(jnp.abs(xf_g), axis=-1, keepdims=True),
        jnp.float32(6.0 * (2.0**-126)),
    )
    s_f32 = _fast_round_scale_f32(amax, 1.0 / 6.0)
    s_bf16 = s_f32.astype(BF16)
  else:
    raise ValueError(f"Unsupported scale_dtype={scale_dtype!r}")
  q_fp4 = jnp.clip(xf_g / s_f32, -6.0, 6.0).astype(FP4)
  return (q_fp4.astype(BF16) * s_bf16).reshape(orig_shape)


def dequant_fp8_weight(w_fp8: jax.Array, w_s_u8: jax.Array) -> jax.Array:
  """Dequantizes [K, N] FP8 weight + [K // 32, N] E8M0 scale to exact [K, N] BF16."""
  k, n = w_fp8.shape
  s_f32 = lax.bitcast_convert_type(
      jnp.left_shift(w_s_u8.astype(U32), U32(23)), F32
  )
  s_bf16 = s_f32.astype(BF16)
  return (
      w_fp8.astype(BF16).reshape(k // 32, 32, n) * s_bf16[:, None, :]
  ).reshape(k, n)


def dequant_mxfp4_weight(
    w_u32: jax.Array, w_s_u8: jax.Array, *, interpret: bool = False
) -> jax.Array:
  """Dequantizes [K // 8, N] uint32 MXFP4 + [K // 32, N] E8M0 scale to exact [K, N] BF16."""
  k_div_8, n = w_u32.shape
  k = k_div_8 * 8
  s_f32 = lax.bitcast_convert_type(
      jnp.left_shift(w_s_u8.astype(U32), U32(23)), F32
  )
  s_bf16 = s_f32.astype(BF16)
  if interpret:
    rows = []
    for i in range(8):
      nib = jnp.bitwise_and(jnp.right_shift(w_u32, U32(4 * i)), U32(0xF))
      sign = jnp.bitwise_and(jnp.right_shift(nib, U32(3)), U32(1))
      exp = jnp.bitwise_and(jnp.right_shift(nib, U32(1)), U32(3))
      man = jnp.bitwise_and(nib, U32(1)).astype(F32)
      mag = jnp.where(
          exp == U32(0),
          jnp.float32(0.5) * man,
          (jnp.float32(1.0) + jnp.float32(0.5) * man)
          * jnp.left_shift(U32(1), jnp.maximum(exp, U32(1)) - U32(1)).astype(
              F32
          ),
      )
      val = jnp.where(sign != U32(0), -mag, mag).astype(BF16)
      rows.append(val)
    w_fp4_bf16 = jnp.stack(rows, axis=1).reshape(k, n)
  else:
    w_fp4_bf16 = pltpu.bitcast(w_u32, FP4).astype(BF16)
  return (w_fp4_bf16.reshape(k // 32, 32, n) * s_bf16[:, None, :]).reshape(k, n)


def _dot_f32(a: jax.Array, b: jax.Array) -> jax.Array:
  """Exact FP32 x FP32 -> FP32 matmul matching model.py's float32 projections."""
  return lax.dot_general(
      a.astype(F32),
      b.astype(F32),
      (((1,), (0,)), ((), ())),
      preferred_element_type=F32,
  )


def dot_bf16_f32(a_bf16: jax.Array, b_bf16: jax.Array) -> jax.Array:
  """Full-tile BF16 x BF16 -> FP32 MXU matmul."""
  if a_bf16.shape[0] == 1:
    return _dot_f32(a_bf16, b_bf16)
  return lax.dot_general(
      a_bf16,
      b_bf16,
      (((1,), (0,)), ((), ())),
      preferred_element_type=F32,
  )


def _pick_pallas_tile_n(n: int, k: int) -> int:
  if n < 512 or n % 128 != 0 or k < 1024 or k % 128 != 0 or (k // 32) % 8 != 0:
    return 0
  max_bytes = 20 * 1024 * 1024
  # Prioritize multiples of 256 (Ghostfish MxuNoncontractingSize = 256) first.
  for step in (
      2048,
      1792,
      1536,
      1280,
      1024,
      768,
      512,
      256,
      896,
      640,
      384,
      128,
  ):
    if n % step == 0 and (k * step * 2) <= max_bytes:
      return step
  return 0


def _fp8_grid_kernel(x_ref, w_ref, ws_ref, out_ref, *, out_dtype: Any):
  if w_ref.ndim == 4:
    w_2d = w_ref[0, 0]
    ws_2d = ws_ref[0, 0]
  elif w_ref.ndim == 3:
    w_2d = w_ref[0]
    ws_2d = ws_ref[0]
  else:
    w_2d = w_ref[...]
    ws_2d = ws_ref[...]
  w_bf16 = dequant_fp8_weight(w_2d, ws_2d)
  out_ref[...] = dot_bf16_f32(x_ref[...], w_bf16).astype(out_dtype)


def _pallas_fp8_matmul_2d(
    x_bf16: jax.Array,
    w_fp8: jax.Array,
    w_s_u8: jax.Array,
    *,
    layer_idx: int | None = None,
    group_idx: int | None = None,
    out_dtype: Any = BF16,
) -> jax.Array:
  """VMEM-resident 1D-grid Pallas FP8 weight dequant + MXU dot (zero HBM w_bf16)."""
  r, k = x_bf16.shape
  n = w_fp8.shape[-1]
  tile_n = _pick_pallas_tile_n(n, k)
  if jax.default_backend() != "tpu" or r > 1024 or tile_n == 0:
    if w_fp8.ndim == 4:
      w_2d = w_fp8[layer_idx, group_idx]
      s_2d = w_s_u8[layer_idx, group_idx]
    elif w_fp8.ndim == 3:
      w_2d = w_fp8[layer_idx]
      s_2d = w_s_u8[layer_idx]
    else:
      w_2d = w_fp8
      s_2d = w_s_u8
    w_bf16 = dequant_fp8_weight(w_2d, s_2d)
    return dot_bf16_f32(x_bf16, w_bf16).astype(out_dtype)
  if r > 128:
    r_blk = 128
    r_pad = layout_lib.round_up(r, r_blk)
    x_pad = (
        jnp.pad(x_bf16.astype(BF16), ((0, r_pad - r), (0, 0)))
        if r_pad > r
        else x_bf16.astype(BF16)
    )
    grid_n = n // tile_n
    grid_m = r_pad // r_blk
    if w_fp8.ndim == 4:
      assert layer_idx is not None and group_idx is not None
      li = int(layer_idx)
      gi = int(group_idx)
      in_specs = [
          pl.BlockSpec((r_blk, k), lambda i, m: (m, 0)),
          pl.BlockSpec(
              (1, 1, k, tile_n), lambda i, m, l=li, g=gi: (l, g, 0, i)
          ),
          pl.BlockSpec(
              (1, 1, k // 32, tile_n), lambda i, m, l=li, g=gi: (l, g, 0, i)
          ),
      ]
    elif w_fp8.ndim == 3:
      assert layer_idx is not None
      li = int(layer_idx)
      in_specs = [
          pl.BlockSpec((r_blk, k), lambda i, m: (m, 0)),
          pl.BlockSpec((1, k, tile_n), lambda i, m, l=li: (l, 0, i)),
          pl.BlockSpec((1, k // 32, tile_n), lambda i, m, l=li: (l, 0, i)),
      ]
    else:
      in_specs = [
          pl.BlockSpec((r_blk, k), lambda i, m: (m, 0)),
          pl.BlockSpec((k, tile_n), lambda i, m: (0, i)),
          pl.BlockSpec((k // 32, tile_n), lambda i, m: (0, i)),
      ]
    out_pad = pl.pallas_call(
        functools.partial(_fp8_grid_kernel, out_dtype=out_dtype),
        out_shape=jax.ShapeDtypeStruct((r_pad, n), out_dtype),
        grid=(grid_n, grid_m),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((r_blk, tile_n), lambda i, m: (m, i)),
        name="ds4_fp8_grid2d_matmul",
    )(x_pad, w_fp8, w_s_u8)
    return out_pad[:r] if r_pad > r else out_pad
  r_pad = layout_lib.round_up(max(r, ROWS), ROWS)
  x_pad = (
      jnp.pad(x_bf16.astype(BF16), ((0, r_pad - r), (0, 0)))
      if r_pad > r
      else x_bf16.astype(BF16)
  )
  grid_n = n // tile_n
  if w_fp8.ndim == 4:
    assert layer_idx is not None and group_idx is not None
    li = int(layer_idx)
    gi = int(group_idx)
    in_specs = [
        pl.BlockSpec((r_pad, k), lambda i: (0, 0)),
        pl.BlockSpec((1, 1, k, tile_n), lambda i, l=li, g=gi: (l, g, 0, i)),
        pl.BlockSpec(
            (1, 1, k // 32, tile_n), lambda i, l=li, g=gi: (l, g, 0, i)
        ),
    ]
  elif w_fp8.ndim == 3:
    assert layer_idx is not None
    li = int(layer_idx)
    in_specs = [
        pl.BlockSpec((r_pad, k), lambda i: (0, 0)),
        pl.BlockSpec((1, k, tile_n), lambda i, l=li: (l, 0, i)),
        pl.BlockSpec((1, k // 32, tile_n), lambda i, l=li: (l, 0, i)),
    ]
  else:
    in_specs = [
        pl.BlockSpec((r_pad, k), lambda i: (0, 0)),
        pl.BlockSpec((k, tile_n), lambda i: (0, i)),
        pl.BlockSpec((k // 32, tile_n), lambda i: (0, i)),
    ]
  out_pad = pl.pallas_call(
      functools.partial(_fp8_grid_kernel, out_dtype=out_dtype),
      out_shape=jax.ShapeDtypeStruct((r_pad, n), out_dtype),
      grid=(grid_n,),
      in_specs=in_specs,
      out_specs=pl.BlockSpec((r_pad, tile_n), lambda i: (0, i)),
      name="ds4_fp8_grid_matmul",
  )(x_pad, w_fp8, w_s_u8)
  return out_pad[:r] if r_pad > r else out_pad


def fp8_matmul(
    x: jax.Array,
    w_fp8: jax.Array,
    w_s_u8: jax.Array,
    *,
    layer_idx: int | None = None,
    group_idx: int | None = None,
    out_dtype: Any = BF16,
) -> jax.Array:
  """Dynamic per-32 FP8 activation x per-32x32 FP8 weight GEMM with FP32 accum."""
  x_bf16 = act_quant_fp8_bf16(x)
  return _pallas_fp8_matmul_2d(
      x_bf16,
      w_fp8,
      w_s_u8,
      layer_idx=layer_idx,
      group_idx=group_idx,
      out_dtype=out_dtype,
  )


def _fp8_grid_sharded_kernel(
    rank_ref, x_ref, w_ref, ws_ref, out_ref, *, out_dtype: Any
):
  del rank_ref
  w_2d = w_ref[0] if w_ref.ndim == 3 else w_ref[...]
  ws_2d = ws_ref[0] if ws_ref.ndim == 3 else ws_ref[...]
  w_bf16 = dequant_fp8_weight(w_2d, ws_2d)
  out_ref[...] = dot_bf16_f32(x_ref[...], w_bf16).astype(out_dtype)


def fp8_matmul_col_sharded(
    x: jax.Array,
    w_fp8: jax.Array,
    w_s_u8: jax.Array,
    *,
    n_dev: int,
    layer_idx: int | None = None,
    out_dtype: Any = BF16,
    pre_quantized: bool = False,
    gather_output: bool = True,
) -> jax.Array:
  """Column-sharded FP8 matmul across layout_lib.AXIS (bit-identical to fp8_matmul)."""
  r = x.shape[0]
  k = w_fp8.shape[-2]
  n = w_fp8.shape[-1]
  x_bf16 = x if pre_quantized else act_quant_fp8_bf16(x)
  n_active = n_dev
  if (
      gather_output
      and n_dev == 8
      and n > 1280
      and n % (n_dev * 128) != 0
      and n % 768 == 0
      and (n // 768) < n_dev
  ):
    n_active = n // 768
  elif n_dev <= 1 or n <= 1280 or n % (n_dev * 128) != 0:
    return _pallas_fp8_matmul_2d(
        x_bf16, w_fp8, w_s_u8, layer_idx=layer_idx, out_dtype=out_dtype
    )
  my_id = lax.axis_index(layout_lib.AXIS)
  eff_id = (
      jnp.minimum(my_id, jnp.int32(n_active - 1)) if n_active < n_dev else my_id
  )
  n_loc = n // n_active
  tile_n = _pick_pallas_tile_n(n_loc, k)
  if jax.default_backend() == "tpu" and 128 < r <= 1024 and tile_n > 0:
    r_blk = 128
    r_pad = layout_lib.round_up(r, r_blk)
    x_pad = (
        jnp.pad(x_bf16.astype(BF16), ((0, r_pad - r), (0, 0)))
        if r_pad > r
        else x_bf16.astype(BF16)
    )
    grid_n = n_loc // tile_n
    grid_m = r_pad // r_blk
    rank_arr = jnp.reshape(jnp.asarray(eff_id, dtype=I32), (1,))
    if w_fp8.ndim == 3:
      assert layer_idx is not None
      li = int(layer_idx)
      in_specs = [
          pl.BlockSpec((r_blk, k), lambda i, m, r_ref: (m, 0)),
          pl.BlockSpec(
              (1, k, tile_n),
              lambda i, m, r_ref, l=li, gn=grid_n: (l, 0, r_ref[0] * gn + i),
          ),
          pl.BlockSpec(
              (1, k // 32, tile_n),
              lambda i, m, r_ref, l=li, gn=grid_n: (l, 0, r_ref[0] * gn + i),
          ),
      ]
    else:
      in_specs = [
          pl.BlockSpec((r_blk, k), lambda i, m, r_ref: (m, 0)),
          pl.BlockSpec(
              (k, tile_n), lambda i, m, r_ref, gn=grid_n: (0, r_ref[0] * gn + i)
          ),
          pl.BlockSpec(
              (k // 32, tile_n),
              lambda i, m, r_ref, gn=grid_n: (0, r_ref[0] * gn + i),
          ),
      ]
    out_pad = pl.pallas_call(
        functools.partial(_fp8_grid_sharded_kernel, out_dtype=out_dtype),
        out_shape=jax.ShapeDtypeStruct((r_pad, n_loc), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(grid_n, grid_m),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((r_blk, tile_n), lambda i, m, r_ref: (m, i)),
        ),
        name="ds4_fp8_grid2d_col_sharded",
    )(rank_arr, x_pad, w_fp8, w_s_u8)
    out_loc = out_pad[:r] if r_pad > r else out_pad
    if not gather_output:
      return out_loc
    gathered = lax.all_gather(out_loc, layout_lib.AXIS, axis=0)[:n_active]
    return jnp.moveaxis(gathered, 0, 1).reshape(r, n)
  if jax.default_backend() == "tpu" and r <= 128 and tile_n > 0:
    r_pad = layout_lib.round_up(max(r, ROWS), ROWS)
    x_pad = (
        jnp.pad(x_bf16.astype(BF16), ((0, r_pad - r), (0, 0)))
        if r_pad > r
        else x_bf16.astype(BF16)
    )
    grid_n = n_loc // tile_n
    rank_arr = jnp.reshape(jnp.asarray(eff_id, dtype=I32), (1,))
    if w_fp8.ndim == 3:
      assert layer_idx is not None
      li = int(layer_idx)
      in_specs = [
          pl.BlockSpec((r_pad, k), lambda i, r_ref: (0, 0)),
          pl.BlockSpec(
              (1, k, tile_n),
              lambda i, r_ref, l=li, gn=grid_n: (l, 0, r_ref[0] * gn + i),
          ),
          pl.BlockSpec(
              (1, k // 32, tile_n),
              lambda i, r_ref, l=li, gn=grid_n: (l, 0, r_ref[0] * gn + i),
          ),
      ]
    else:
      in_specs = [
          pl.BlockSpec((r_pad, k), lambda i, r_ref: (0, 0)),
          pl.BlockSpec(
              (k, tile_n), lambda i, r_ref, gn=grid_n: (0, r_ref[0] * gn + i)
          ),
          pl.BlockSpec(
              (k // 32, tile_n),
              lambda i, r_ref, gn=grid_n: (0, r_ref[0] * gn + i),
          ),
      ]
    out_pad = pl.pallas_call(
        functools.partial(_fp8_grid_sharded_kernel, out_dtype=out_dtype),
        out_shape=jax.ShapeDtypeStruct((r_pad, n_loc), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(grid_n,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((r_pad, tile_n), lambda i, r_ref: (0, i)),
        ),
        name="ds4_fp8_grid_col_sharded",
    )(rank_arr, x_pad, w_fp8, w_s_u8)
    out_loc = out_pad[:r] if r_pad > r else out_pad
    if not gather_output:
      return out_loc
    gathered = lax.all_gather(out_loc, layout_lib.AXIS, axis=0)[:n_active]
    return jnp.moveaxis(gathered, 0, 1).reshape(r, n)
  c0 = eff_id * jnp.int32(n_loc)
  if w_fp8.ndim == 3:
    assert layer_idx is not None
    li = jnp.int32(layer_idx)
    w_loc = lax.dynamic_slice(w_fp8, (li, jnp.int32(0), c0), (1, k, n_loc))[0]
    s_loc = lax.dynamic_slice(
        w_s_u8, (li, jnp.int32(0), c0), (1, k // 32, n_loc)
    )[0]
  else:
    w_loc = lax.dynamic_slice_in_dim(w_fp8, c0, n_loc, axis=1)
    s_loc = lax.dynamic_slice_in_dim(w_s_u8, c0, n_loc, axis=1)
  out_loc = _pallas_fp8_matmul_2d(x_bf16, w_loc, s_loc, out_dtype=out_dtype)
  if not gather_output:
    return out_loc
  gathered = lax.all_gather(out_loc, layout_lib.AXIS, axis=0)[:n_active]
  return jnp.moveaxis(gathered, 0, 1).reshape(r, n)


def _mxfp4_chunk_step(n: int) -> int:
  if n <= 1536:
    return n
  for step in (1536, 1280, 1024, 768, 512):
    if n % step == 0:
      return step
  return n


def _mxfp4_grid_kernel(x_ref, w_ref, ws_ref, out_ref, *, out_dtype: Any):
  w_bf16 = dequant_mxfp4_weight(w_ref[...], ws_ref[...], interpret=False)
  out_ref[...] = dot_bf16_f32(x_ref[...], w_bf16).astype(out_dtype)


def _mxfp4_grid_expert_kernel(
    exp_ref, x_ref, w_ref, ws_ref, out_ref, *, out_dtype: Any
):
  del exp_ref
  w_2d = w_ref[0, 0] if w_ref.ndim == 4 else w_ref[0]
  ws_2d = ws_ref[0, 0] if ws_ref.ndim == 4 else ws_ref[0]
  w_bf16 = dequant_mxfp4_weight(w_2d, ws_2d, interpret=False)
  out_ref[...] = dot_bf16_f32(x_ref[...], w_bf16).astype(out_dtype)


def _slice_expert_2d(
    arr: jax.Array, layer_idx: int | None, expert_idx: jax.Array | None
) -> jax.Array:
  if arr.ndim == 4:
    return arr[layer_idx, expert_idx]
  return arr[expert_idx] if arr.ndim == 3 else arr


def mxfp4_matmul(
    x_bf16: jax.Array,
    w_u32: jax.Array,
    w_s_u8: jax.Array,
    *,
    layer_idx: int | None = None,
    expert_idx: jax.Array | None = None,
    interpret: bool = False,
    in_pallas: bool = False,
    out_dtype: Any = BF16,
) -> jax.Array:
  """Pre-quantized BF16 activation x MXFP4 weight GEMM with FP32 accum."""
  if not in_pallas and not interpret and jax.default_backend() == "tpu":
    r, k = x_bf16.shape
    n = w_u32.shape[-1]
    tile_n = _pick_pallas_tile_n(n, k)
    if 128 < r <= 1024 and tile_n > 0 and (k // 8) % 8 == 0:
      r_blk = 128
      r_pad = layout_lib.round_up(r, r_blk)
      x_pad = (
          jnp.pad(x_bf16.astype(BF16), ((0, r_pad - r), (0, 0)))
          if r_pad > r
          else x_bf16.astype(BF16)
      )
      grid_n = n // tile_n
      grid_m = r_pad // r_blk
      if w_u32.ndim in (3, 4) and expert_idx is not None:
        exp_arr = jnp.reshape(jnp.asarray(expert_idx, dtype=I32), (1,))
        if w_u32.ndim == 4:
          li = int(layer_idx or 0)
          in_specs = [
              pl.BlockSpec((r_blk, k), lambda i, m, e_ref: (m, 0)),
              pl.BlockSpec(
                  (1, 1, k // 8, tile_n),
                  lambda i, m, e_ref, l=li: (l, e_ref[0], 0, i),
              ),
              pl.BlockSpec(
                  (1, 1, k // 32, tile_n),
                  lambda i, m, e_ref, l=li: (l, e_ref[0], 0, i),
              ),
          ]
        else:
          in_specs = [
              pl.BlockSpec((r_blk, k), lambda i, m, e_ref: (m, 0)),
              pl.BlockSpec(
                  (1, k // 8, tile_n), lambda i, m, e_ref: (e_ref[0], 0, i)
              ),
              pl.BlockSpec(
                  (1, k // 32, tile_n), lambda i, m, e_ref: (e_ref[0], 0, i)
              ),
          ]
        out_pad = pl.pallas_call(
            functools.partial(_mxfp4_grid_expert_kernel, out_dtype=out_dtype),
            out_shape=jax.ShapeDtypeStruct((r_pad, n), out_dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(grid_n, grid_m),
                in_specs=in_specs,
                out_specs=pl.BlockSpec(
                    (r_blk, tile_n), lambda i, m, e_ref: (m, i)
                ),
            ),
            name="ds4_mxfp4_grid2d_expert_matmul",
        )(exp_arr, x_pad, w_u32, w_s_u8)
        return out_pad[:r] if r_pad > r else out_pad
    if r <= 128 and tile_n > 0 and (k // 8) % 8 == 0:
      r_pad = layout_lib.round_up(max(r, ROWS), ROWS)
      x_pad = (
          jnp.pad(x_bf16.astype(BF16), ((0, r_pad - r), (0, 0)))
          if r_pad > r
          else x_bf16.astype(BF16)
      )
      grid_n = n // tile_n
      if w_u32.ndim in (3, 4) and expert_idx is not None:
        exp_arr = jnp.reshape(jnp.asarray(expert_idx, dtype=I32), (1,))
        if w_u32.ndim == 4:
          li = int(layer_idx or 0)
          in_specs = [
              pl.BlockSpec((r_pad, k), lambda i, e_ref: (0, 0)),
              pl.BlockSpec(
                  (1, 1, k // 8, tile_n),
                  lambda i, e_ref, l=li: (l, e_ref[0], 0, i),
              ),
              pl.BlockSpec(
                  (1, 1, k // 32, tile_n),
                  lambda i, e_ref, l=li: (l, e_ref[0], 0, i),
              ),
          ]
        else:
          in_specs = [
              pl.BlockSpec((r_pad, k), lambda i, e_ref: (0, 0)),
              pl.BlockSpec(
                  (1, k // 8, tile_n), lambda i, e_ref: (e_ref[0], 0, i)
              ),
              pl.BlockSpec(
                  (1, k // 32, tile_n), lambda i, e_ref: (e_ref[0], 0, i)
              ),
          ]
        out_pad = pl.pallas_call(
            functools.partial(_mxfp4_grid_expert_kernel, out_dtype=out_dtype),
            out_shape=jax.ShapeDtypeStruct((r_pad, n), out_dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(grid_n,),
                in_specs=in_specs,
                out_specs=pl.BlockSpec(
                    (r_pad, tile_n), lambda i, e_ref: (0, i)
                ),
            ),
            name="ds4_mxfp4_grid_expert_matmul",
        )(exp_arr, x_pad, w_u32, w_s_u8)
        return out_pad[:r] if r_pad > r else out_pad
      w_2d = _slice_expert_2d(w_u32, layer_idx, expert_idx)
      s_2d = _slice_expert_2d(w_s_u8, layer_idx, expert_idx)
      out_pad = pl.pallas_call(
          functools.partial(_mxfp4_grid_kernel, out_dtype=out_dtype),
          out_shape=jax.ShapeDtypeStruct((r_pad, n), out_dtype),
          grid=(grid_n,),
          in_specs=[
              pl.BlockSpec((r_pad, k), lambda i: (0, 0)),
              pl.BlockSpec((k // 8, tile_n), lambda i: (0, i)),
              pl.BlockSpec((k // 32, tile_n), lambda i: (0, i)),
          ],
          out_specs=pl.BlockSpec((r_pad, tile_n), lambda i: (0, i)),
          name="ds4_mxfp4_grid_matmul",
      )(x_pad, w_2d, s_2d)
      return out_pad[:r] if r_pad > r else out_pad
  w_2d = _slice_expert_2d(w_u32, layer_idx, expert_idx)
  s_2d = _slice_expert_2d(w_s_u8, layer_idx, expert_idx)
  if not in_pallas:
    w_bf16 = dequant_mxfp4_weight(w_2d, s_2d, interpret=True)
    return dot_bf16_f32(x_bf16, w_bf16).astype(out_dtype)
  n = w_2d.shape[1]
  step = _mxfp4_chunk_step(n)
  if step == n:
    w_bf16 = dequant_mxfp4_weight(w_2d, s_2d, interpret=interpret)
    return dot_bf16_f32(x_bf16, w_bf16).astype(out_dtype)
  chunks = []
  for c0 in range(0, n, step):
    c1 = c0 + step
    w_bf16 = dequant_mxfp4_weight(
        w_2d[:, c0:c1], s_2d[:, c0:c1], interpret=interpret
    )
    chunks.append(dot_bf16_f32(x_bf16, w_bf16).astype(out_dtype))
  return jnp.concatenate(chunks, axis=1)


def pack_rope_tail128_table(
    cos_32: np.ndarray, sin_32: np.ndarray, rows: int
) -> np.ndarray:
  """Packs [S, 32] cos/sin into [rows, 2, 128] float32 for 128-lane tail rotation.

  Slot 0 is cos_128: [1.0 x 64, cos_0, cos_0, ..., cos_31, cos_31].
  Slot 1 is sin_signed_128: [0.0 x 64, -sin_0, +sin_0, ..., -sin_31, +sin_31].
  """
  s_len = cos_32.shape[0]
  out = np.zeros((rows, 2, LANES), dtype=np.float32)
  out[:, 0, :64] = 1.0
  c_rep = np.repeat(cos_32, 2, axis=-1)
  s_rep = np.empty((s_len, 64), dtype=np.float32)
  s_rep[:, 0::2] = -sin_32
  s_rep[:, 1::2] = sin_32
  out[:s_len, 0, 64:] = c_rep
  out[:s_len, 1, 64:] = s_rep
  return out


def apply_rope_tail128(
    x: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    *,
    inverse: bool = False,
) -> jax.Array:
  """Applies interleaved complex RoPE to the trailing 64 channels of x [R, D].

  Uses a single XOR-1 lane gather on the trailing 128-lane block.
  """
  d = x.shape[-1]
  tail = x[:, d - LANES : d].astype(F32)
  xor1 = jnp.bitwise_xor(jnp.arange(LANES, dtype=I32), I32(1))
  partner = _lane_gather(tail, xor1)
  s_term = -sin_signed_128 if inverse else sin_signed_128
  rot_tail = (tail * cos_128 + partner * s_term).astype(x.dtype)
  if d == LANES:
    return rot_tail
  return jnp.concatenate([x[:, : d - LANES], rot_tail], axis=-1)


# ==============================================================================
# In-Kernel Remote DMA All-Reduce & Barrier
# ==============================================================================


def barrier(axis_name: str, num_devices: int) -> None:
  sem = pltpu.get_barrier_semaphore()
  for d in range(num_devices):
    pl.semaphore_signal(sem, device_id={axis_name: d})
  pl.semaphore_wait(sem, num_devices)


def all_reduce_rows(
    partial: jax.Array,
    recv_ref: Any,
    send_sem: Any,
    recv_sem: Any,
    *,
    axis_name: str = layout_lib.AXIS,
    num_devices: int,
) -> jax.Array:
  """Sums [R, D] float32 partial across all num_devices in deterministic rank order."""
  if num_devices == 1:
    return partial
  my_id = lax.axis_index(axis_name)
  recv_ref[my_id] = partial
  for peer in range(num_devices):

    @pl.when(peer != my_id)
    def _(peer=peer):
      pltpu.make_async_remote_copy(
          src_ref=recv_ref.at[my_id],
          dst_ref=recv_ref.at[my_id],
          send_sem=send_sem,
          recv_sem=recv_sem,
          device_id={axis_name: peer},
      ).start()

  for _ in range(num_devices - 1):
    pltpu.make_async_copy(recv_ref.at[0], recv_ref.at[0], recv_sem).wait()
    pltpu.make_async_copy(recv_ref.at[0], recv_ref.at[0], send_sem).wait()
  total = recv_ref[0]
  for d in range(1, num_devices):
    total = total + recv_ref[d]
  return total


# ==============================================================================
# Standalone Pallas Kernels for Stage-by-Stage Forge Hardware Numerics Tests
# ==============================================================================


def _act_quant_fp8_kernel(x_ref, out_ref):
  out_ref[...] = act_quant_fp8_bf16(x_ref[...], in_pallas=True)


def pallas_act_quant_fp8(x: jax.Array, *, interpret: bool = False) -> jax.Array:
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
  return pl.pallas_call(
      _act_quant_fp8_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, BF16),
      in_specs=[vmem],
      out_specs=vmem,
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ds4_act_quant_fp8",
  )(x)


def _fp4_quant_kernel(x_ref, out_ref, *, scale_dtype: str):
  out_ref[...] = fp4_act_quant_bf16(
      x_ref[...], scale_dtype=scale_dtype, in_pallas=True
  )


def pallas_fp4_quant(
    x: jax.Array, *, scale_dtype: str = "e8m0", interpret: bool = False
) -> jax.Array:
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
  return pl.pallas_call(
      functools.partial(_fp4_quant_kernel, scale_dtype=scale_dtype),
      out_shape=jax.ShapeDtypeStruct(x.shape, BF16),
      in_specs=[vmem],
      out_specs=vmem,
      interpret=pltpu.InterpretParams() if interpret else False,
      name=f"ds4_fp4_quant_{scale_dtype}",
  )(x)


def _fp8_linear_kernel(x_ref, w_ref, ws_ref, out_ref):
  x_bf16 = act_quant_fp8_bf16(x_ref[...], in_pallas=True)
  w_bf16 = dequant_fp8_weight(w_ref[...], ws_ref[...])
  out_ref[...] = dot_bf16_f32(x_bf16, w_bf16).astype(F32)


def pallas_fp8_linear(
    x: jax.Array,
    w_fp8: jax.Array,
    w_s_u8: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
  return pl.pallas_call(
      _fp8_linear_kernel,
      out_shape=jax.ShapeDtypeStruct((x.shape[0], w_fp8.shape[1]), F32),
      in_specs=[vmem, vmem, vmem],
      out_specs=vmem,
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ds4_fp8_linear",
  )(x, w_fp8, w_s_u8)


def _mxfp4_linear_kernel(x_ref, w_ref, ws_ref, out_ref, *, interpret: bool):
  x_bf16 = act_quant_fp8_bf16(x_ref[...], in_pallas=True)
  out_ref[...] = mxfp4_matmul(
      x_bf16, w_ref[...], ws_ref[...], interpret=interpret, in_pallas=True
  ).astype(F32)


def pallas_mxfp4_linear(
    x: jax.Array,
    w_u32: jax.Array,
    w_s_u8: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
  return pl.pallas_call(
      functools.partial(_mxfp4_linear_kernel, interpret=interpret),
      out_shape=jax.ShapeDtypeStruct((x.shape[0], w_u32.shape[1]), F32),
      in_specs=[vmem, vmem, vmem],
      out_specs=vmem,
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ds4_mxfp4_linear",
  )(x, w_u32, w_s_u8)


def _rope_tail_kernel(x_ref, cos_ref, sin_ref, out_ref, *, inverse: bool):
  out_ref[...] = apply_rope_tail128(
      x_ref[...], cos_ref[...], sin_ref[...], inverse=inverse
  )


def pallas_rope_tail128(
    x: jax.Array,
    cos_128: jax.Array,
    sin_signed_128: jax.Array,
    *,
    inverse: bool = False,
    interpret: bool = False,
) -> jax.Array:
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
  return pl.pallas_call(
      functools.partial(_rope_tail_kernel, inverse=inverse),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      in_specs=[vmem, vmem, vmem],
      out_specs=vmem,
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ds4_rope_tail128",
  )(x, cos_128, sin_signed_128)


def _routed_experts_stream_kernel(
    expert_ids_ref,
    x_ref,
    weights_ref,
    gu_w_hbm,
    gu_s_hbm,
    dn_w_hbm,
    dn_s_hbm,
    out_ref,
    gu_w_buf,
    gu_s_buf,
    dn_w_buf,
    dn_s_buf,
    sems,
    *,
    n_sel: int,
    slots: int,
    swiglu_limit: float,
    interpret: bool,
):
  """Streams active local MXFP4 routed experts through a VMEM double/triple buffer."""

  def make_copies(j: int, slot: int):
    e = jnp.maximum(expert_ids_ref[j], 0)
    return (
        pltpu.make_async_copy(
            gu_w_hbm.at[e], gu_w_buf.at[slot], sems.at[slot, 0]
        ),
        pltpu.make_async_copy(
            gu_s_hbm.at[e], gu_s_buf.at[slot], sems.at[slot, 1]
        ),
        pltpu.make_async_copy(
            dn_w_hbm.at[e], dn_w_buf.at[slot], sems.at[slot, 2]
        ),
        pltpu.make_async_copy(
            dn_s_hbm.at[e], dn_s_buf.at[slot], sems.at[slot, 3]
        ),
    )

  def start_slot(j: int):
    @pl.when(expert_ids_ref[j] >= 0)
    def _():
      for c in make_copies(j, j % slots):
        c.start()

  for j in range(min(slots, n_sel)):
    start_slot(j)

  x_bf16 = act_quant_fp8_bf16(x_ref[...], in_pallas=True)
  out_ref[...] = jnp.zeros(out_ref.shape, F32)

  for j in range(n_sel):
    slot = j % slots

    @pl.when(expert_ids_ref[j] >= 0)
    def _(j=j, slot=slot):
      for c in make_copies(j, slot):
        c.wait()
      gate_up = (
          mxfp4_matmul(
              x_bf16,
              gu_w_buf[slot],
              gu_s_buf[slot],
              interpret=interpret,
              in_pallas=True,
          )
          .astype(BF16)
          .astype(F32)
      )
      ei = gate_up.shape[1] // 2
      gate = gate_up[:, :ei]
      up = gate_up[:, ei:]
      if swiglu_limit > 0:
        lim = jnp.float32(swiglu_limit)
        up = jnp.clip(up, -lim, lim)
        gate = jnp.minimum(gate, lim)
      act = ((gate / (1.0 + jnp.exp(-gate))) * up) * weights_ref[:, j : j + 1]
      act_bf16 = act_quant_fp8_bf16(act.astype(BF16), in_pallas=True)
      y = (
          mxfp4_matmul(
              act_bf16,
              dn_w_buf[slot],
              dn_s_buf[slot],
              interpret=interpret,
              in_pallas=True,
          )
          .astype(BF16)
          .astype(F32)
      )
      out_ref[...] = out_ref[...] + y

    if j + slots < n_sel:
      start_slot(j + slots)


def pallas_routed_experts(
    x: jax.Array,
    expert_ids: jax.Array,
    weights: jax.Array,
    gu_w: jax.Array,
    gu_s: jax.Array,
    dn_w: jax.Array,
    dn_s: jax.Array,
    *,
    slots: int = 2,
    swiglu_limit: float = 10.0,
    interpret: bool = False,
) -> jax.Array:
  """Executes selective active-expert MXFP4 streaming in Pallas."""
  b, d = x.shape
  r = layout_lib.round_up(max(b, ROWS), ROWS)
  n_sel = expert_ids.shape[0]
  x_pad = jnp.zeros((r, d), BF16).at[:b].set(x.astype(BF16))
  w_pad = jnp.zeros((r, n_sel), F32).at[:b].set(weights.astype(F32))

  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
  out_pad = pl.pallas_call(
      functools.partial(
          _routed_experts_stream_kernel,
          n_sel=n_sel,
          slots=slots,
          swiglu_limit=swiglu_limit,
          interpret=interpret,
      ),
      out_shape=jax.ShapeDtypeStruct((r, d), F32),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          grid=(),
          in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
          out_specs=vmem,
          scratch_shapes=[
              pltpu.VMEM((slots, *gu_w.shape[1:]), U32),
              pltpu.VMEM((slots, *gu_s.shape[1:]), U8),
              pltpu.VMEM((slots, *dn_w.shape[1:]), U32),
              pltpu.VMEM((slots, *dn_s.shape[1:]), U8),
              pltpu.SemaphoreType.DMA((slots, 4)),
          ],
      ),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ds4_routed_experts",
  )(expert_ids.astype(I32), x_pad, w_pad, gu_w, gu_s, dn_w, dn_s)
  return out_pad[:b]


def pallas_all_reduce(
    x: jax.Array,
    *,
    axis_name: str = layout_lib.AXIS,
    num_devices: int,
    collective_id: int = 5,
    interpret: bool = False,
) -> jax.Array:
  """Standalone Pallas remote-DMA all-reduce over axis_name."""
  r, n = x.shape
  vmem = pl.BlockSpec(memory_space=pltpu.VMEM)

  def _kern(x_ref, out_ref, recv_buf, sems):
    barrier(axis_name, num_devices)
    out_ref[...] = all_reduce_rows(
        x_ref[...],
        recv_buf,
        sems.at[0],
        sems.at[1],
        axis_name=axis_name,
        num_devices=num_devices,
    )

  return pl.pallas_call(
      _kern,
      out_shape=jax.ShapeDtypeStruct((r, n), F32),
      in_specs=[vmem],
      out_specs=vmem,
      scratch_shapes=[
          pltpu.VMEM((num_devices, r, n), F32),
          pltpu.SemaphoreType.DMA((2,)),
      ],
      compiler_params=pltpu.CompilerParams(collective_id=collective_id),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="ds4_all_reduce",
  )(x.astype(F32))


# ==============================================================================
# Single-Pass Mega-mHC, CSA2 Group-Cached Attention & 3 Macro-Kernels
# ==============================================================================


def _rms_norm_jax(x: jax.Array, w: jax.Array, eps: float) -> jax.Array:
  xf = x.astype(F32)
  var = jnp.mean(xf * xf, axis=-1, keepdims=True)
  return (xf * lax.rsqrt(var + jnp.float32(eps)) * w.astype(F32)).astype(
      x.dtype
  )


def _hc_split_sinkhorn_jax(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  orig_prefix = mixes.shape[:-1]
  n_rows = int(np.prod(orig_prefix))
  m = mixes.reshape(n_rows, mixes.shape[-1]).astype(F32)
  sc = hc_scale.astype(F32)
  hb = hc_base.astype(F32)
  eps32 = jnp.float32(eps)

  pre = (jax.nn.sigmoid(m[:, :hc_mult] * sc[0] + hb[:hc_mult]) + eps32).reshape(
      *orig_prefix, hc_mult
  )
  post = (
      2.0
      * jax.nn.sigmoid(
          m[:, hc_mult : 2 * hc_mult] * sc[1] + hb[hc_mult : 2 * hc_mult]
      )
  ).reshape(*orig_prefix, hc_mult)
  comb_raw = m[:, 2 * hc_mult :] * sc[2] + hb[2 * hc_mult :]
  g = [
      [comb_raw[:, i * hc_mult + j] for j in range(hc_mult)]
      for i in range(hc_mult)
  ]
  for i in range(hc_mult):
    rm = g[i][0]
    for j in range(1, hc_mult):
      rm = jnp.maximum(rm, g[i][j])
    for j in range(hc_mult):
      g[i][j] = jnp.exp(g[i][j] - rm)
    rsum = g[i][0]
    for j in range(1, hc_mult):
      rsum = rsum + g[i][j]
    for j in range(hc_mult):
      g[i][j] = g[i][j] / rsum + eps32
  for j in range(hc_mult):
    csum = g[0][j]
    for i in range(1, hc_mult):
      csum = csum + g[i][j]
    csum = csum + eps32
    for i in range(hc_mult):
      g[i][j] = g[i][j] / csum
  for _ in range(sinkhorn_iters - 1):
    for i in range(hc_mult):
      rsum = g[i][0]
      for j in range(1, hc_mult):
        rsum = rsum + g[i][j]
      rsum = rsum + eps32
      for j in range(hc_mult):
        g[i][j] = g[i][j] / rsum
    for j in range(hc_mult):
      csum = g[0][j]
      for i in range(1, hc_mult):
        csum = csum + g[i][j]
      csum = csum + eps32
      for i in range(hc_mult):
        g[i][j] = g[i][j] / csum
  comb = jnp.stack([jnp.stack(row, axis=-1) for row in g], axis=-2).reshape(
      *orig_prefix, hc_mult, hc_mult
  )
  return pre, post, comb


def mega_mhc_mixes_jax(
    x_mhc: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_eps: float,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes Single-Pass mHC (pre, post, comb) coefficients from x_mhc [B, S, 4, D]."""
  b, s, hc, d = x_mhc.shape
  flat = x_mhc.reshape(b * s, hc * d).astype(F32)
  rsqrt = lax.rsqrt(
      jnp.mean(flat * flat, axis=-1, keepdims=True) + jnp.float32(norm_eps)
  )
  mixes = (
      lax.dot_general(
          flat,
          hc_fn.astype(F32),
          (((1,), (0,)), ((), ())),
          preferred_element_type=F32,
      )
      * rsqrt
  ).reshape(b, s, -1)
  return _hc_split_sinkhorn_jax(
      mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
  )


def mega_mhc_pre_norm_jax(
    x_mhc: jax.Array, pre_mix: jax.Array, norm_w: jax.Array, norm_eps: float
) -> jax.Array:
  """Fuses hc_pre collapse [B, S, 4, D] -> [B, S, D] and pre-sublayer RMSNorm."""
  collapsed = jnp.sum(
      pre_mix.astype(F32)[..., None] * x_mhc.astype(F32), axis=2
  ).astype(x_mhc.dtype)
  return _rms_norm_jax(collapsed, norm_w, norm_eps)


def mega_mhc_post_jax(
    sub_out: jax.Array,
    residual_mhc: jax.Array,
    post: jax.Array,
    comb: jax.Array,
) -> jax.Array:
  """Fuses hc_post expansion and doubly-stochastic residual stream mixing."""
  term1 = post.astype(F32)[..., None] * sub_out.astype(F32)[:, :, None, :]
  term2 = jnp.einsum(
      "bsij,bsid->bsjd",
      comb.astype(F32),
      residual_mhc.astype(F32),
      preferred_element_type=F32,
  )
  return (term1 + term2).astype(sub_out.dtype)


def _apply_rotary_jax(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *,
    rope_dim: int = 64,
    inverse: bool = False,
) -> jax.Array:
  """Rotates trailing rope_dim channels of x ([B, S, D] or [B, S, H, D])."""
  tail = x[..., -rope_dim:].astype(F32)
  evens = tail[..., 0::2]
  odds = tail[..., 1::2]
  if cos.ndim == 2:
    c = cos[None, :, :] if x.ndim == 3 else cos[None, :, None, :]
    s = sin[None, :, :] if x.ndim == 3 else sin[None, :, None, :]
  else:
    c = cos if x.ndim == 3 else cos[:, :, None, :]
    s = sin if x.ndim == 3 else sin[:, :, None, :]
  if inverse:
    s = -s
  rot_evens = evens * c - odds * s
  rot_odds = evens * s + odds * c
  rot_tail = jnp.stack([rot_evens, rot_odds], axis=-1).reshape(tail.shape)
  return jnp.concatenate(
      [x[..., :-rope_dim], rot_tail.astype(x.dtype)], axis=-1
  )


def _sparse_attn_with_sink_jax(
    q: jax.Array,
    kv_selected: jax.Array,
    valid_mask: jax.Array,
    attn_sink: jax.Array,
    softmax_scale: float,
    *,
    win_kv: jax.Array | None = None,
    win_valid: jax.Array | None = None,
) -> jax.Array:
  """Core sparse attention over SWA [B, K_w, D] + Global KV [B, S, K_g, D]."""
  qf = q.astype(F32)
  scale = jnp.float32(softmax_scale)
  if win_kv is not None and win_valid is not None:
    wf = win_kv.astype(F32)
    gf = kv_selected.astype(F32)
    g_eq = "bkd" if kv_selected.ndim == 3 else "bskd"
    s_win = (
        jnp.einsum("bshd,bkd->bshk", qf, wf, preferred_element_type=F32) * scale
    )
    s_grp = (
        jnp.einsum(f"bshd,{g_eq}->bshk", qf, gf, preferred_element_type=F32)
        * scale
    )
    scores = jnp.concatenate(
        [
            jnp.where(win_valid[:, :, None, :], s_win, -jnp.inf),
            jnp.where(valid_mask[:, :, None, :], s_grp, -jnp.inf),
        ],
        axis=-1,
    )
    scores_max = jnp.maximum(jnp.max(scores, axis=-1), jnp.float32(-1e30))
    probs = jnp.exp(scores - scores_max[..., None])
    sum_exp = jnp.sum(probs, axis=-1) + jnp.exp(
        attn_sink.astype(F32)[None, None, :] - scores_max
    )
    probs_bf16 = probs.astype(BF16).astype(F32)
    kw = win_kv.shape[1]
    acc_o = jnp.einsum(
        "bshk,bkd->bshd",
        probs_bf16[..., :kw],
        wf,
        preferred_element_type=F32,
    ) + jnp.einsum(
        f"bshk,{g_eq}->bshd",
        probs_bf16[..., kw:],
        gf,
        preferred_element_type=F32,
    )
    return (acc_o / sum_exp[..., None]).astype(BF16)

  kv_eq = "bkd" if kv_selected.ndim == 3 else "bskd"
  scores = (
      jnp.einsum(
          f"bshd,{kv_eq}->bshk",
          qf,
          kv_selected.astype(F32),
          preferred_element_type=F32,
      )
      * scale
  )
  scores = jnp.where(valid_mask[:, :, None, :], scores, -jnp.inf)
  scores_max = jnp.maximum(jnp.max(scores, axis=-1), jnp.float32(-1e30))
  probs = jnp.exp(scores - scores_max[..., None])
  sum_exp = jnp.sum(probs, axis=-1) + jnp.exp(
      attn_sink.astype(F32)[None, None, :] - scores_max
  )
  probs_bf16 = probs.astype(BF16).astype(F32)
  acc_o = jnp.einsum(
      f"bshk,{kv_eq}->bshd",
      probs_bf16,
      kv_selected.astype(F32),
      preferred_element_type=F32,
  )
  return (acc_o / sum_exp[..., None]).astype(BF16)


def _vmem_all_gather_axis0(x: jax.Array, n_dev: int) -> jax.Array:
  """All-gathers small tensors (< 64 KiB) via VMEM lax.psum instead of SC all-gather."""
  if n_dev <= 1 or (x.size * x.dtype.itemsize * n_dev) >= 65536:
    return lax.all_gather(x, layout_lib.AXIS, axis=0)
  my_id = lax.axis_index(layout_lib.AXIS)
  mask = (jnp.arange(n_dev, dtype=I32) == my_id).reshape(
      (n_dev,) + (1,) * x.ndim
  )
  padded = jnp.where(mask, x[None, ...], jnp.zeros_like(x)[None, ...])
  return lax.psum(padded, layout_lib.AXIS)


def _moe_layer_sharded_jax(
    x: jax.Array,
    gate_w: jax.Array,
    gate_bias: jax.Array,
    shared_gu_w: jax.Array,
    shared_gu_s: jax.Array,
    shared_dn_w: jax.Array,
    shared_dn_s: jax.Array,
    exp_gu_w: jax.Array,
    exp_gu_s: jax.Array,
    exp_dn_w: jax.Array,
    exp_dn_s: jax.Array,
    *,
    layer_idx: int = 0,
    shared_layer_idx: int | None = None,
    active_b: jax.Array | None = None,
    lay: layout_lib.DeviceLayout,
    is_mtp: bool,
    interpret: bool = False,
) -> jax.Array:
  """Executes sharded MoE (ETP=1 x EP=N routed experts + shared expert).

  `active_b` is either a per-slot [B] mask or a per-token [B, S] mask. Masked
  tokens route to no expert, so they never add a routed expert to the loop.
  """
  c = lay.cfg
  orig_shape = x.shape
  n_tok = x.shape[0] * x.shape[1]
  x_flat = x.reshape(n_tok, c.dim)
  topk = c.dspark_n_activated_experts if is_mtp else c.n_activated_experts
  e_loc = lay.local_dspark_experts if is_mtp else lay.local_backbone_experts
  sh_idx = layer_idx if shared_layer_idx is None else shared_layer_idx

  # 1. Router scores (sqrtsoftplus + bias selection + unbiased weight gather)
  logits = _dot_f32(x_flat, gate_w) / jnp.float32(c.gate_temp)
  sp = jnp.where(
      logits > 20.0,
      logits,
      jnp.log1p(jnp.exp(jnp.clip(logits, -80.0, 20.0))),
  )
  scores = jnp.sqrt(sp)
  biased = scores + gate_bias.astype(F32)[None, :]
  _, top_indices = lax.top_k(biased, topk)
  weights = jnp.take_along_axis(scores, top_indices, axis=-1)
  if c.norm_topk_prob and topk > 1:
    weights = weights / (
        jnp.sum(weights, axis=-1, keepdims=True) + jnp.float32(1e-20)
    )
  weights = weights * jnp.float32(c.route_scale)

  # 2. Compact active local experts (at most min(e_loc, n_tok * topk) active)
  my_id = (
      lax.axis_index(layout_lib.AXIS) if lay.num_devices > 1 else jnp.int32(0)
  )
  my_group = my_id // jnp.int32(lay.expert_tp)
  e_start = my_group * jnp.int32(e_loc)
  loc_iota = jnp.arange(e_loc, dtype=I32)[None, None, :]
  match_all = top_indices[:, :, None] == (e_start + loc_iota)
  if active_b is not None:
    tok_mask = (
        active_b.reshape(-1)
        if active_b.ndim == 2
        else jnp.repeat(active_b, x.shape[1])
    )
    match_all = match_all & tok_mask[:, None, None]
  w_all = jnp.sum(jnp.where(match_all, weights[:, :, None], 0.0), axis=1)
  expert_active = jnp.any(match_all, axis=(0, 1))
  n_sel = min(e_loc, n_tok * topk)
  loc_iota_1d = jnp.arange(e_loc, dtype=I32)
  sel_iota = jnp.arange(n_sel, dtype=I32)
  active_key = jnp.where(expert_active, loc_iota_1d, e_loc)
  sorted_active = jnp.sort(active_key)[:n_sel]
  sel_valid = sorted_active < e_loc
  expert_ids = jnp.where(sel_valid, sorted_active, -1).astype(I32)

  x_bf16 = act_quant_fp8_bf16(x_flat)

  def _get_eid(j_idx: jax.Array) -> jax.Array:
    return jnp.sum(jnp.where(sel_iota == j_idx, expert_ids, 0))

  def _cond_selected(state: tuple[jax.Array, jax.Array]) -> jax.Array:
    j, _ = state
    return (j < n_sel) & (_get_eid(j) >= 0)

  def _body_selected(
      state: tuple[jax.Array, jax.Array],
  ) -> tuple[jax.Array, jax.Array]:
    j, acc_y = state
    eid = _get_eid(j)
    e_safe = jnp.maximum(eid, 0)
    row_w = jnp.sum(
        jnp.where((loc_iota_1d == eid)[None, :], w_all, 0.0),
        axis=1,
        keepdims=True,
    )
    gu = (
        mxfp4_matmul(
            x_bf16,
            exp_gu_w,
            exp_gu_s,
            layer_idx=layer_idx,
            expert_idx=e_safe,
            interpret=interpret,
        )
        .astype(BF16)
        .astype(F32)
    )
    ei = gu.shape[1] // 2
    gate = gu[:, :ei]
    up = gu[:, ei:]
    if c.swiglu_limit > 0:
      lim = jnp.float32(c.swiglu_limit)
      up = jnp.clip(up, -lim, lim)
      gate = jnp.minimum(gate, lim)
    act = ((gate / (1.0 + jnp.exp(-gate))) * up) * row_w
    act_q = act_quant_fp8_bf16(act.astype(BF16))
    dn = mxfp4_matmul(
        act_q,
        exp_dn_w,
        exp_dn_s,
        layer_idx=layer_idx,
        expert_idx=e_safe,
        interpret=interpret,
        out_dtype=BF16,
    ).astype(F32)
    return j + jnp.int32(1), acc_y + jnp.where(row_w > 0.0, dn, 0.0)

  if n_sel <= 1 or c.dim < 1024:
    y_routed = jnp.zeros((n_tok, c.dim), dtype=F32)
    for j_i in range(n_sel):
      _, y_routed = _body_selected((jnp.int32(j_i), y_routed))
  else:
    _, y_routed = lax.while_loop(
        _cond_selected,
        _body_selected,
        (jnp.int32(0), jnp.zeros((n_tok, c.dim), dtype=F32)),
    )

  # 3. Shared expert (folded into y_routed before single lax.psum)
  sh_gu = fp8_matmul_col_sharded(
      x_bf16,
      shared_gu_w,
      shared_gu_s,
      n_dev=lay.num_devices,
      layer_idx=sh_idx,
      out_dtype=BF16 if n_tok > 128 else F32,
      pre_quantized=True,
  )
  si_loc = sh_gu.shape[1] // 2
  sh_gate = sh_gu[:, :si_loc].astype(F32)
  sh_up = sh_gu[:, si_loc:].astype(F32)
  if c.swiglu_limit > 0:
    lim = jnp.float32(c.swiglu_limit)
    sh_up = jnp.clip(sh_up, -lim, lim)
    sh_gate = jnp.minimum(sh_gate, lim)
  sh_act = ((sh_gate / (1.0 + jnp.exp(-sh_gate))) * sh_up).astype(BF16)
  sh_dn_loc = fp8_matmul_col_sharded(
      sh_act,
      shared_dn_w,
      shared_dn_s,
      n_dev=lay.num_devices,
      layer_idx=sh_idx,
      out_dtype=BF16,
      gather_output=False,
  ).astype(F32)
  if sh_dn_loc.shape[1] < c.dim and lay.num_devices > 1:
    n_loc = sh_dn_loc.shape[1]
    c0 = my_id * jnp.int32(n_loc)
    cur_slice = lax.dynamic_slice(y_routed, (jnp.int32(0), c0), (n_tok, n_loc))
    y_routed = lax.dynamic_update_slice(
        y_routed, cur_slice + sh_dn_loc, (jnp.int32(0), c0)
    )
    y_total = lax.psum(y_routed, layout_lib.AXIS)
  else:
    if lay.num_devices > 1:
      y_routed = lax.psum(y_routed, layout_lib.AXIS)
    y_total = y_routed + sh_dn_loc
  return y_total.astype(x.dtype).reshape(orig_shape)


def _engram_local_deq_jax(
    hash_ids: jax.Array,
    embed_w_fp8: jax.Array,
    embed_s_u8: jax.Array,
    *,
    lay: layout_lib.DeviceLayout,
    engram_idx: int,
) -> jax.Array:
  """Gathers and dequantizes local Engram FP8 shard to [B * S, E_dim] BF16."""
  c = lay.cfg
  b, s, _ = hash_ids.shape
  rows_loc = lay.engram_rows_per_device(engram_idx)
  my_id = (
      lax.axis_index(layout_lib.AXIS) if lay.num_devices > 1 else jnp.int32(0)
  )
  r0 = my_id * jnp.int32(rows_loc)
  r1 = jnp.minimum(
      r0 + jnp.int32(rows_loc), jnp.int32(c.engram_num_embeddings[engram_idx])
  )
  in_range = (hash_ids >= r0) & (hash_ids < r1)
  local_idx = jnp.where(in_range, hash_ids - r0, 0)
  raw_fp8 = embed_w_fp8[local_idx]
  raw_s_u8 = embed_s_u8[local_idx]
  raw_s_bf16 = lax.bitcast_convert_type(
      jnp.left_shift(raw_s_u8.astype(U32), U32(23)), F32
  ).astype(BF16)
  ehd = c.engram_head_dim
  deq_bf16 = (
      raw_fp8.astype(BF16).reshape(b, s, -1, ehd // 32, 32)
      * raw_s_bf16[..., None]
  ).reshape(b, s, -1, ehd)
  return jnp.where(in_range[..., None], deq_bf16, BF16(0.0)).reshape(b * s, -1)


def _engram_apply_gating_jax(
    x_mhc: jax.Array,
    kv: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    *,
    norm_eps: float,
) -> jax.Array:
  """Applies signed-sqrt gated Engram residual update from precomputed kv."""
  b, s, hc, d = x_mhc.shape
  kv_3d = kv.reshape(b, s, (hc + 1) * d)
  key = kv_3d[..., : hc * d].reshape(b, s, hc, d)
  value = kv_3d[..., hc * d :]
  weight = q_weight.astype(F32) * k_weight.astype(F32)
  h = x_mhc.astype(F32)
  eps = jnp.float32(norm_eps)
  rstd = lax.rsqrt(jnp.mean(h * h, axis=-1) + eps) * lax.rsqrt(
      jnp.mean(key * key, axis=-1) + eps
  )
  dot = (
      jnp.sum(h * weight[None, None, :, :] * key, axis=-1)
      * rstd
      * jnp.float32(d**-0.5)
  )
  signed_sqrt = jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)), dot)
  gate = jax.nn.sigmoid(signed_sqrt)
  return (h + gate[..., None] * value[:, :, None, :]).astype(x_mhc.dtype)


def _engram_lookup_sharded_jax(
    x_mhc: jax.Array,
    hash_ids: jax.Array,
    embed_w_fp8: jax.Array,
    embed_s_u8: jax.Array,
    wkv_w_fp8: jax.Array,
    wkv_s_u8: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    *,
    lay: layout_lib.DeviceLayout,
    engram_idx: int,
) -> jax.Array:
  """Executes sharded Engram lookup + signed-sqrt gated residual update."""
  embed_bf16 = _engram_local_deq_jax(
      hash_ids, embed_w_fp8, embed_s_u8, lay=lay, engram_idx=engram_idx
  )
  if lay.num_devices > 1:
    embed_bf16 = lax.psum(embed_bf16, layout_lib.AXIS)
  kv = fp8_matmul_col_sharded(
      embed_bf16, wkv_w_fp8, wkv_s_u8, n_dev=lay.num_devices
  )
  return _engram_apply_gating_jax(
      x_mhc, kv, q_weight, k_weight, norm_eps=lay.cfg.norm_eps
  )


@dataclasses.dataclass
class DeviceEngramTables:
  """Device-resident Engram XXHash64 constants (32-bit split, no x64 required)."""

  token_map: jax.Array  # [vocab_size] int32
  mult_hi: jax.Array  # [n_engram_layers, max_ngram_size] uint32
  mult_lo: jax.Array  # [n_engram_layers, max_ngram_size] uint32
  primes: jax.Array  # [n_engram_layers, max_ngram_size - 1, n_heads] uint32
  offsets: jax.Array  # [n_engram_layers, (max_ngram_size - 1) * n_heads] int32
  pad_id: int

  @classmethod
  def from_config(
      cls, cfg: ref.Ds4Config, token_map: np.ndarray | None = None
  ) -> DeviceEngramTables:
    tbl = ref.EngramTables.from_config(cfg, token_map=token_map)
    m_u64 = tbl.multipliers.astype(np.uint64)
    m_hi = (m_u64 >> np.uint64(32)).astype(np.uint32)
    m_lo = (m_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return cls(
        token_map=jnp.asarray(tbl.token_map, dtype=I32),
        mult_hi=jnp.asarray(m_hi, dtype=U32),
        mult_lo=jnp.asarray(m_lo, dtype=U32),
        primes=jnp.asarray(tbl.primes, dtype=U32),
        offsets=jnp.asarray(tbl.offsets, dtype=I32),
        pad_id=int(tbl.pad_compressed_id),
    )


jax.tree_util.register_dataclass(
    DeviceEngramTables,
    data_fields=["token_map", "mult_hi", "mult_lo", "primes", "offsets"],
    meta_fields=["pad_id"],
)


def _mul_u32_to_u64(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Exact 32x32 -> 64-bit unsigned product (hi_u32, lo_u32) via 16-bit limbs."""
  a0 = a & U32(0xFFFF)
  a1 = a >> U32(16)
  b0 = b & U32(0xFFFF)
  b1 = b >> U32(16)
  p00 = a0 * b0
  p01 = a0 * b1
  p10 = a1 * b0
  p11 = a1 * b1
  mid = (p00 >> U32(16)) + (p01 & U32(0xFFFF)) + (p10 & U32(0xFFFF))
  lo = (p00 & U32(0xFFFF)) | ((mid & U32(0xFFFF)) << U32(16))
  hi = p11 + (p01 >> U32(16)) + (p10 >> U32(16)) + (mid >> U32(16))
  return hi, lo


def _mod_u64_by_u32(hi: jax.Array, lo: jax.Array, p: jax.Array) -> jax.Array:
  """Computes ((hi << 32) | lo) % p in 32-bit unsigned arithmetic for p < 2^25."""
  rem = hi % p
  for shift in (24, 16, 8, 0):
    byte_val = (lo >> U32(shift)) & U32(0xFF)
    rem = ((rem << U32(8)) | byte_val) % p
  return rem


def compute_engram_hashes_jax(
    input_ids: jax.Array,
    start_pos: jax.Array | int,
    engram_tail: jax.Array,
    tables: DeviceEngramTables,
    *,
    valid_len: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """On-device Engram Multi-Head XXHash64 without x64, returning (hashes, new_tail)."""
  bsz, seqlen = input_ids.shape
  _, _, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  safe_ids = jnp.clip(input_ids, 0, tables.token_map.shape[0] - 1)
  comp_cur = tables.token_map[safe_ids]  # [B, S]
  full_comp = jnp.concatenate([engram_tail, comp_cur], axis=1)  # [B, 3 + S]

  max_ng = tables.mult_lo.shape[1]
  tokens_list = []
  blocked = ~active_b[:, None]
  for shift in range(max_ng):
    src = full_comp[:, 3 - shift : 3 + seqlen - shift]
    blocked = blocked | (pos_seq < shift) | (src < 0)
    tokens_list.append(jnp.where(blocked, jnp.int32(tables.pad_id), src))
  tokens_u32 = jnp.stack(tokens_list, axis=-1).astype(U32)  # [B, S, max_ng]

  # 64-bit product: tokens_u32 * ((mult_hi << 32) | mult_lo)
  t_exp = tokens_u32[:, :, None, :]  # [B, S, 1, max_ng]
  hi0, prod_lo = _mul_u32_to_u64(t_exp, tables.mult_lo[None, None, :, :])
  prod_hi = hi0 + t_exp * tables.mult_hi[None, None, :, :]

  roll_hi = prod_hi[..., 0]
  roll_lo = prod_lo[..., 0]
  hashes = []
  for i in range(1, max_ng):
    roll_hi = jnp.bitwise_xor(roll_hi, prod_hi[..., i])
    roll_lo = jnp.bitwise_xor(roll_lo, prod_lo[..., i])
    p_heads = tables.primes[None, None, :, i - 1, :]  # [1, 1, L_e, H_e]
    h_mod = _mod_u64_by_u32(
        roll_hi[..., None], roll_lo[..., None], p_heads
    ).astype(I32)
    hashes.append(h_mod)
  hash_out = jnp.concatenate(hashes, axis=-1) + tables.offsets[None, None, :, :]
  if valid_len is None:
    new_tail = jnp.where(
        active_b[:, None], full_comp[:, seqlen : seqlen + 3], engram_tail
    )
  else:
    v_len = jnp.clip(
        jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), (bsz,)), 0, seqlen
    )
    tail_idx = v_len[:, None] + jnp.arange(3, dtype=I32)[None, :]
    tail_sel = jnp.take_along_axis(full_comp, tail_idx, axis=1)
    new_tail = jnp.where(
        active_b[:, None] & (v_len[:, None] > 0), tail_sel, engram_tail
    )
  return hash_out, new_tail


def advance_engram_tail_jax(
    input_ids: jax.Array,
    engram_tail: jax.Array,
    valid_len: jax.Array | int,
    active_b: jax.Array,
    tables: DeviceEngramTables,
) -> jax.Array:
  """Advances engram_tail [B, 3] by valid_len [B] accepted tokens from input_ids."""
  bsz, seqlen = input_ids.shape
  safe_ids = jnp.clip(input_ids, 0, tables.token_map.shape[0] - 1)
  comp_cur = tables.token_map[safe_ids]
  full_comp = jnp.concatenate([engram_tail, comp_cur], axis=1)
  v_len = jnp.clip(
      jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), (bsz,)), 0, seqlen
  )
  tail_idx = v_len[:, None] + jnp.arange(3, dtype=I32)[None, :]
  tail_sel = jnp.take_along_axis(full_comp, tail_idx, axis=1)
  return jnp.where(
      active_b[:, None] & (v_len[:, None] > 0), tail_sel, engram_tail
  )


def _block_max_3d_jax(scores: jax.Array, block: int) -> jax.Array:
  """Computes 3D block-max over the trailing dimension without 4D sub-lane reshape."""
  return lax.reduce_window(
      scores,
      -jnp.inf,
      lax.max,
      window_dimensions=(1, 1, block),
      window_strides=(1, 1, block),
      padding="VALID",
  )


def _fast_topk_3d_jax(x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
  """Exact 3D top-k using a strided sub-lane pigeonhole filter when M >= 8 * k."""
  m = x.shape[-1]
  sblk = 4
  if m < 8192 or m < 8 * k or m % sblk != 0 or (m // sblk) % LANES != 0:
    return lax.top_k(x, k)
  bsz, seqlen, _ = x.shape
  n_b = m // sblk
  s_max = jnp.max(x.reshape(bsz, seqlen, sblk, n_b), axis=2)
  _, top_s_idx = lax.top_k(s_max, k)
  cand_base = top_s_idx[..., None] + jnp.arange(sblk, dtype=I32) * n_b
  cand_idxs = cand_base.reshape(bsz, seqlen, k * sblk)
  cand_vals = jnp.take_along_axis(x, cand_idxs, axis=-1)
  top_vals, rel_pos = lax.top_k(cand_vals, k)
  top_idxs = jnp.take_along_axis(cand_idxs, rel_pos, axis=-1)
  return top_vals, top_idxs


def _exact_two_stage_topk_jax(
    scores: jax.Array,
    k: int,
    block: int = 8,
    *,
    precomputed_top_blk_idx: jax.Array | None = None,
    precomputed_blk_max: jax.Array | None = None,
    blk_mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Exact top-k over [B, S, M] via pigeonhole block-max filter."""
  m = scores.shape[-1]
  if m < 16384 or m % block != 0 or (m // block) <= k:
    if blk_mask is not None:
      scores = jnp.where(jnp.repeat(blk_mask, block, axis=-1), scores, -jnp.inf)
    return lax.top_k(scores, k)
  bsz, seqlen, _ = scores.shape
  n_b = m // block
  if precomputed_top_blk_idx is not None:
    n_blk_sel = precomputed_top_blk_idx.shape[-1]
    cand_base = (precomputed_top_blk_idx * block)[..., None] + jnp.arange(
        block, dtype=I32
    )
    cand_idxs = cand_base.reshape(bsz, seqlen, n_blk_sel * block)
    cand_scores = jnp.take_along_axis(
        scores, jnp.maximum(cand_idxs, 0), axis=-1
    )
    cand_scores = jnp.where(cand_idxs >= 0, cand_scores, -jnp.inf)
    top_vals, rel_pos = lax.top_k(cand_scores, k)
    top_idxs = jnp.take_along_axis(jnp.maximum(cand_idxs, 0), rel_pos, axis=-1)
    return top_vals, top_idxs
  if precomputed_blk_max is None and blk_mask is None and n_b % LANES == 0:
    blk_max = jnp.max(scores.reshape(bsz, seqlen, block, n_b), axis=2)
    _, top_blk_idx = _fast_topk_3d_jax(blk_max, k)
    n_blk_sel = top_blk_idx.shape[-1]
    cand_base = top_blk_idx[..., None] + jnp.arange(block, dtype=I32) * n_b
    cand_idxs = cand_base.reshape(bsz, seqlen, n_blk_sel * block)
    cand_scores = jnp.take_along_axis(scores, cand_idxs, axis=-1)
    top_vals, rel_pos = lax.top_k(cand_scores, k)
    top_idxs = jnp.take_along_axis(cand_idxs, rel_pos, axis=-1)
    return top_vals, top_idxs
  blk_max = (
      precomputed_blk_max
      if precomputed_blk_max is not None
      else _block_max_3d_jax(scores, block)
  )
  if blk_mask is not None:
    blk_max = jnp.where(blk_mask, blk_max, -jnp.inf)
  _, top_blk_idx = _fast_topk_3d_jax(blk_max, k)
  n_blk_sel = top_blk_idx.shape[-1]
  cand_base = (top_blk_idx * block)[..., None] + jnp.arange(block, dtype=I32)
  cand_idxs = cand_base.reshape(bsz, seqlen, n_blk_sel * block)
  cand_scores = jnp.take_along_axis(scores, cand_idxs, axis=-1)
  if blk_mask is not None:
    sel_blk_valid = jnp.take_along_axis(blk_mask, top_blk_idx, axis=-1)
    cand_scores = (
        jnp.where(
            sel_blk_valid[..., None],
            cand_scores.reshape(bsz, seqlen, n_blk_sel, block),
            -jnp.inf,
        )
    ).reshape(bsz, seqlen, n_blk_sel * block)
  top_vals, rel_pos = lax.top_k(cand_scores, k)
  top_idxs = jnp.take_along_axis(cand_idxs, rel_pos, axis=-1)
  return top_vals, top_idxs


def _select_candidate_blocks_jax(
    logits: jax.Array,
    compress_lens: jax.Array,
    topk_blocks: int,
    block_size: int,
    *,
    n_dev: int = 1,
) -> jax.Array:
  """Selects topk_blocks of block_size positions, pinning newest block last to +inf."""
  width = logits.shape[-1]
  pad_w = (-width) % block_size
  if pad_w:
    pad_cfg = [(0, 0)] * (logits.ndim - 1) + [(0, pad_w)]
    scores = jnp.pad(logits, pad_cfg, constant_values=-jnp.inf)
  else:
    scores = logits
  scores = jnp.max(scores.reshape(*logits.shape[:-1], -1, block_size), axis=-1)
  num_blocks = scores.shape[-1]
  last = (compress_lens - 1) // block_size
  b_iota = jnp.arange(num_blocks, dtype=I32)
  scores = jnp.where(b_iota == last, jnp.inf, scores)

  k = min(topk_blocks, num_blocks)
  if n_dev > 1 and num_blocks >= n_dev * k and num_blocks % n_dev == 0:
    my_id = lax.axis_index(layout_lib.AXIS)
    loc_b = num_blocks // n_dev
    s_reshaped = scores.reshape(*scores.shape[:-1], loc_b, n_dev)
    s_loc = lax.dynamic_index_in_dim(
        s_reshaped, my_id, axis=s_reshaped.ndim - 1, keepdims=False
    )
    loc_vals, _ = lax.top_k(s_loc, k)
    all_vals = lax.all_gather(loc_vals, layout_lib.AXIS, axis=0)
    cand_vals = jnp.moveaxis(all_vals, 0, -2).reshape(
        *scores.shape[:-1], n_dev * k
    )
    top_vals, _ = lax.top_k(cand_vals, k)
  else:
    top_vals, _ = lax.top_k(scores, k)
  # O(B * S * num_blocks) threshold mask (2048x smaller than 4D broadcast).
  thresh = top_vals[..., -1:]
  strictly_above = (scores > thresh) & (scores > -jnp.inf)
  need_at_thresh = jnp.maximum(
      k - jnp.sum(strictly_above.astype(I32), axis=-1, keepdims=True), 0
  )
  at_thresh = (scores == thresh) & (scores > -jnp.inf)
  at_thresh_rank = jnp.cumsum(at_thresh.astype(I32), axis=-1)
  keep = strictly_above | (at_thresh & (at_thresh_rank <= need_at_thresh))
  return jnp.repeat(keep, block_size, axis=-1)[..., :width]


def _gather_kv_by_indices_jax(
    kv_pool: jax.Array, topk_idxs: jax.Array, slot: int | None = None
) -> tuple[jax.Array, jax.Array]:
  """Gathers [B, S, K, D] from kv_pool using topk_idxs [B, S, K] int32.

  `kv_pool` is [B, N, D], or [B, n_slots, N, D] with a static `slot`. The slot
  is folded into the row index of a [B * n_slots, N, D] view, so the gather
  reads the cache buffer in place instead of a copied kv_pool[:, slot] operand.
  """
  valid = topk_idxs >= 0
  safe_idx = jnp.where(valid, topk_idxs, 0)
  b_iota = jnp.minimum(
      jnp.arange(topk_idxs.shape[0], dtype=I32), kv_pool.shape[0] - 1
  )[:, None, None]
  if slot is not None:
    n_slots = kv_pool.shape[1]
    kv_pool = kv_pool.reshape(-1, *kv_pool.shape[2:])
    b_iota = b_iota * jnp.int32(n_slots) + jnp.int32(slot)
  gathered = kv_pool[b_iota, safe_idx]
  return jnp.where(valid[..., None], gathered, 0), valid


def _pos_and_active(
    start_pos: jax.Array | int, bsz: int, seqlen: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Normalizes scalar or [B] start_pos into (pos_b, safe_pos_b, active_b, pos_seq)."""
  pos_b = jnp.broadcast_to(jnp.asarray(start_pos, dtype=I32), (bsz,))
  active_b = pos_b >= 0
  safe_pos_b = jnp.maximum(pos_b, 0)
  pos_seq = safe_pos_b[:, None] + jnp.arange(seqlen, dtype=I32)[None, :]
  return pos_b, safe_pos_b, active_b, pos_seq


def _token_active_mask(
    active_b: jax.Array, seqlen: int, valid_len: jax.Array | int | None
) -> jax.Array:
  """Returns [B] slot mask, or [B, S] token mask when only valid_len rows count."""
  if valid_len is None:
    return active_b
  v_len = jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), active_b.shape)[
      :, None
  ]
  return active_b[:, None] & (jnp.arange(seqlen, dtype=I32)[None, :] < v_len)


def _update_swa_ring_buffer(
    cur_win_kv_before: jax.Array,
    kv_new: jax.Array,
    safe_pos_b: jax.Array,
    active_b: jax.Array,
    win: int,
    *,
    valid_len: jax.Array | int | None = None,
) -> jax.Array:
  """Updates SWA ring buffer [B, win, D] with kv_new[:, :valid_len, :]."""
  bsz, seqlen, _ = kv_new.shape
  r_iota = jnp.arange(win, dtype=I32)[None, :]
  if valid_len is None:
    if seqlen >= win:
      s_idx = (seqlen - 1) - ((safe_pos_b[:, None] + seqlen - 1 - r_iota) % win)
      new_ring = jnp.take_along_axis(kv_new, s_idx[..., None], axis=1)
      return jnp.where(active_b[:, None, None], new_ring, cur_win_kv_before)
    cur_win_kv = cur_win_kv_before
    for s_i in range(seqlen):
      slot_w = (safe_pos_b + s_i) % win
      w_mask = ((r_iota == slot_w[:, None]) & active_b[:, None])[:, :, None]
      cur_win_kv = jnp.where(w_mask, kv_new[:, s_i : s_i + 1, :], cur_win_kv)
    return cur_win_kv
  v_len = jnp.clip(
      jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), (bsz,))[:, None],
      0,
      seqlen,
  )
  age_r = (safe_pos_b[:, None] + v_len - 1 - r_iota) % win
  s_idx = (v_len - 1) - age_r
  new_ring = jnp.take_along_axis(
      kv_new, jnp.clip(s_idx, 0, seqlen - 1)[..., None], axis=1
  )
  wrote_slot = (s_idx >= 0) & (v_len > 0) & active_b[:, None]
  return jnp.where(wrote_slot[:, :, None], new_ring, cur_win_kv_before)


def _build_swa_window_kv(
    cur_win_kv_before: jax.Array,
    kv_new: jax.Array,
    start_pos: jax.Array,
    win: int,
    *,
    valid_len: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Builds causal SWA KV [B, S, win + S, D], valid_mask [B, S, win + S], and updated ring."""
  bsz, seqlen, _ = kv_new.shape
  _, safe_pos_b, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  r_iota = jnp.arange(win, dtype=I32)[None, :]
  age_r = (safe_pos_b[:, None] - 1 - r_iota) % win
  pos_ring = jnp.where(
      (safe_pos_b[:, None] > 0) & active_b[:, None],
      (safe_pos_b[:, None] - 1) - age_r,
      -1,
  )
  slot_pos = jnp.concatenate([pos_ring, pos_seq], axis=1)

  q_pos = pos_seq[:, :, None]
  k_pos = slot_pos[:, None, :]
  win_valid = (k_pos >= 0) & (k_pos <= q_pos) & ((q_pos - k_pos) < win)

  cat_kv = jnp.concatenate([cur_win_kv_before, kv_new], axis=1)
  cur_win_kv = _update_swa_ring_buffer(
      cur_win_kv_before,
      kv_new,
      safe_pos_b,
      active_b,
      win,
      valid_len=valid_len,
  )
  return cat_kv, win_valid, cur_win_kv


@dataclasses.dataclass
class DeviceDecodeCaches:
  """Device-resident runtime KV and Compressor caches for all layers."""

  window_kv: jax.Array  # [B, 40, window_size, head_dim] BF16
  mtp_window_kv: jax.Array  # [B, 3, window_size, head_dim] BF16
  comp_m2_kv: jax.Array  # [B, 3, max_seq // 2, head_dim] BF16
  comp_m2_idx_k: jax.Array  # [B, 3, max_seq // 2, index_head_dim] BF16
  comp_m2_kv_state: jax.Array  # [B, 3, 2, head_dim] FP32
  comp_m2_score_state: jax.Array  # [B, 3, 2, head_dim] FP32
  comp_m1_kv: jax.Array  # [B, 1, max_seq, head_dim] BF16
  comp_m1_idx_k: jax.Array  # [B, 1, max_seq, index_head_dim] BF16

  @classmethod
  def zeros(
      cls, cfg: ref.Ds4Config, batch_size: int, max_seq_len: int
  ) -> DeviceDecodeCaches:
    m2_count = sum(
        1 for l in cfg.kv_source_layers if cfg.compress_ratios[l] > 1
    )
    m1_count = sum(
        1 for l in cfg.kv_source_layers if cfg.compress_ratios[l] == 1
    )
    return cls(
        window_kv=jnp.zeros(
            (batch_size, cfg.n_layers, cfg.window_size, cfg.head_dim),
            dtype=BF16,
        ),
        mtp_window_kv=jnp.zeros(
            (
                batch_size,
                max(1, cfg.n_mtp_layers),
                cfg.window_size,
                cfg.head_dim,
            ),
            dtype=BF16,
        ),
        comp_m2_kv=jnp.zeros(
            (batch_size, max(1, m2_count), max_seq_len // 2, cfg.head_dim),
            dtype=BF16,
        ),
        comp_m2_idx_k=jnp.zeros(
            (
                batch_size,
                max(1, m2_count),
                max_seq_len // 2,
                cfg.index_head_dim,
            ),
            dtype=BF16,
        ),
        comp_m2_kv_state=jnp.zeros(
            (batch_size, max(1, m2_count), 2, cfg.head_dim), dtype=F32
        ),
        comp_m2_score_state=jnp.full(
            (batch_size, max(1, m2_count), 2, cfg.head_dim),
            -jnp.inf,
            dtype=F32,
        ),
        comp_m1_kv=jnp.zeros(
            (batch_size, max(1, m1_count), max_seq_len, cfg.head_dim),
            dtype=BF16,
        ),
        comp_m1_idx_k=jnp.zeros(
            (batch_size, max(1, m1_count), max_seq_len, cfg.index_head_dim),
            dtype=BF16,
        ),
    )


jax.tree_util.register_dataclass(
    DeviceDecodeCaches,
    data_fields=[
        "window_kv",
        "mtp_window_kv",
        "comp_m2_kv",
        "comp_m2_idx_k",
        "comp_m2_kv_state",
        "comp_m2_score_state",
        "comp_m1_kv",
        "comp_m1_idx_k",
    ],
    meta_fields=[],
)


def _compressor_m2_step_jax(
    x: jax.Array,
    start_pos: jax.Array,
    wkv: jax.Array,
    wgate: jax.Array,
    norm_w: jax.Array,
    norm_eps: float,
    kv_state: jax.Array,
    score_state: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Vectorized stateful ratio=2 Compressor over S tokens starting at start_pos."""
  bsz, seqlen, _ = x.shape
  xf = x.reshape(bsz * seqlen, -1)
  kv = _dot_f32(xf, wkv).reshape(bsz, seqlen, -1)
  score = _dot_f32(xf, wgate).reshape(bsz, seqlen, -1)

  _, safe_pos_b, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  start_even = ((safe_pos_b % 2) == 0)[:, None, None]
  init_other_kv = jnp.where(
      start_even, kv_state[:, 1:2, :], kv_state[:, 0:1, :]
  )
  init_other_sc = jnp.where(
      start_even, score_state[:, 1:2, :], score_state[:, 0:1, :]
  )
  prev_kv = jnp.concatenate([init_other_kv, kv[:, :-1, :]], axis=1)
  prev_sc = jnp.concatenate([init_other_sc, score[:, :-1, :]], axis=1)
  is_even = ((pos_seq % 2) == 0)[:, :, None]
  slot0_kv = jnp.where(is_even, kv, prev_kv)
  slot1_kv = jnp.where(is_even, prev_kv, kv)
  slot0_sc = jnp.where(is_even, score, prev_sc)
  slot1_sc = jnp.where(is_even, prev_sc, score)
  kv_st_active = jnp.stack([slot0_kv, slot1_kv], axis=2)
  sc_st_active = jnp.stack([slot0_sc, slot1_sc], axis=2)
  act_mask = active_b[:, None, None, None]
  kv_st_trace = jnp.where(act_mask, kv_st_active, kv_state[:, None, :, :])
  sc_st_trace = jnp.where(act_mask, sc_st_active, score_state[:, None, :, :])

  s_max = jnp.max(sc_st_trace, axis=2, keepdims=True)
  probs = jnp.exp(sc_st_trace - s_max)
  probs = probs / jnp.sum(probs, axis=2, keepdims=True)
  pooled = jnp.sum(kv_st_trace * probs, axis=2)
  all_latents = _rms_norm_jax(pooled.astype(x.dtype), norm_w, norm_eps)
  return all_latents, kv_st_trace, sc_st_trace


def _indexer_step_core_jax(
    x: jax.Array,
    qr: jax.Array,
    start_pos: jax.Array,
    ratio: int,
    wq_b_w: jax.Array,
    wq_b_s: jax.Array,
    weights_proj: jax.Array,
    index_k_cache: jax.Array,
    cos_tbl: jax.Array,
    sin_tbl: jax.Array,
    *,
    lay: layout_lib.DeviceLayout,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidates_in: jax.Array | None,
    idx_slot: int | None = None,
    scan_len: int | None = None,
    cache_slot: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Runs CSA2 Indexer (Full or Candidate-Restricted Reindex) and returns topk_idxs.

  `index_k_cache` is [B, M, D], or [B, n_slots, M, D] with a static
  `cache_slot`. `scan_len` (default: M) scores only the first scan_len keys on
  the multi-device slab path. Each device reads its slab in place, so no
  [:, cache_slot] or [:, :scan_len] copy of the cache is made.
  """
  c = lay.cfg
  bsz, seqlen, _ = x.shape
  _, _, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  qr_flat = qr.reshape(bsz * seqlen, -1)
  q_idx = fp8_matmul(qr_flat, wq_b_w, wq_b_s, layer_idx=idx_slot).reshape(
      bsz, seqlen, c.index_n_heads, c.index_head_dim
  )
  q_idx = _apply_rotary_jax(
      q_idx, cos_tbl[pos_seq], sin_tbl[pos_seq], rope_dim=c.rope_head_dim
  )
  q_idx = fp4_act_quant_bf16(
      q_idx.reshape(bsz * seqlen * c.index_n_heads, c.index_head_dim),
      scale_dtype="e8m0",
  ).reshape(bsz, seqlen, c.index_n_heads, c.index_head_dim)

  weights = _dot_f32(x.reshape(bsz * seqlen, -1), weights_proj).astype(
      BF16
  ).astype(F32).reshape(bsz, seqlen, c.index_n_heads) * jnp.float32(
      c.index_softmax_scale * (c.index_n_heads**-0.5)
  )

  compress_lens = jnp.where(active_b[:, None], (pos_seq + 1) // ratio, 0)[
      :, :, None
  ]
  full_comp = index_k_cache.shape[-2]
  max_comp = full_comp if scan_len is None else scan_len
  topk = min(c.index_topk, max_comp)
  n_dev = lay.num_devices
  h_step = min(8, c.index_n_heads)

  # Parallelize Indexer scan across all n_dev TensorCores using contiguous slabs
  if (
      n_dev > 1
      and max_comp >= 512
      and max_comp % (n_dev * c.candidate_block_size) == 0
  ):
    my_id = lax.axis_index(layout_lib.AXIS)
    local_t = max_comp // n_dev
    t_start = my_id * jnp.int32(local_t)
    global_t = t_start + jnp.arange(local_t, dtype=I32)
    # Each device reads its slab in place: a dynamic-slice fused into the
    # scoring dot. A [:, :scan_len] slice would be copied first.
    k_flat, k_off = index_k_cache, t_start
    if cache_slot is not None:
      # [B, n_slots * M, D] view: the slot becomes an offset along M, so XLA
      # keeps the cache's own layout instead of relayouting it slot-major.
      k_flat = index_k_cache.reshape(
          index_k_cache.shape[0], -1, c.index_head_dim
      )
      k_off = t_start + jnp.int32(cache_slot * full_comp)
    k_local = lax.dynamic_slice_in_dim(k_flat, k_off, local_t, axis=1)
    if jax.default_backend() == "tpu":
      raw_scores = jnp.einsum(
          "bshd,btd->bsht",
          q_idx.astype(BF16),
          k_local.astype(BF16),
          preferred_element_type=F32,
      )
      local_score = jnp.einsum(
          "bsht,bsh->bst",
          jnp.maximum(raw_scores, 0.0),
          weights,
          preferred_element_type=F32,
      )
    else:
      local_score = jnp.zeros((bsz, seqlen, local_t), dtype=F32)
      for h0 in range(0, c.index_n_heads, h_step):
        h1 = h0 + h_step
        raw_chunk = jnp.einsum(
            "bshd,btd->bsht",
            q_idx[:, :, h0:h1, :].astype(F32),
            k_local.astype(F32),
            preferred_element_type=F32,
        )
        local_score = local_score + jnp.sum(
            jnp.maximum(raw_chunk, 0.0) * weights[:, :, h0:h1, None], axis=2
        )
    local_score = jnp.where(
        global_t[None, None, :] < compress_lens, local_score, -jnp.inf
    )

    candidates_out = None if is_candidate_source else candidates_in
    bsize = c.candidate_block_size
    loc_num_blocks = local_t // bsize
    k_loc = min(topk, local_t)
    k_blk = min(c.candidate_topk_blocks, loc_num_blocks)
    need_cand_filter = is_candidate_source and (
        n_dev * loc_num_blocks > c.candidate_topk_blocks
    )
    pre_blk_max = None
    pre_top_bidx = None
    blk_mask_local = None
    last_blk = (compress_lens - 1) // bsize
    loc_blk_scores_pinned = local_score
    if need_cand_filter:
      loc_blk_scores = _block_max_3d_jax(local_score, bsize)
      pre_blk_max = loc_blk_scores
      if local_t >= 16384 and loc_num_blocks > k_loc:
        _, pre_top_bidx = _fast_topk_3d_jax(loc_blk_scores, k_loc)
      g_blk_iota = my_id * jnp.int32(loc_num_blocks) + jnp.arange(
          loc_num_blocks, dtype=I32
      )
      loc_blk_scores_pinned = jnp.where(
          g_blk_iota[None, None, :] == last_blk, jnp.inf, loc_blk_scores
      )
    elif uses_candidates and candidates_in is not None:
      if (
          candidates_in.dtype == I32
          and candidates_in.shape[-1] == k_loc
          and local_t >= 16384
      ):
        pre_top_bidx = candidates_in
      elif candidates_in.shape[-1] == loc_num_blocks:
        blk_mask_local = candidates_in
      elif candidates_in.shape[-1] == local_t:
        local_score = jnp.where(candidates_in, local_score, -jnp.inf)
      else:
        cand_reshaped = candidates_in.reshape(bsz, seqlen, n_dev, local_t)
        cand_local = lax.dynamic_index_in_dim(
            cand_reshaped, my_id, axis=2, keepdims=False
        )
        local_score = jnp.where(cand_local, local_score, -jnp.inf)

    r_tot = bsz * seqlen
    if local_t <= 2048 and not need_cand_filter and n_dev > 1 and r_tot >= 32:
      if blk_mask_local is not None:
        local_score = jnp.where(
            jnp.repeat(blk_mask_local, bsize, axis=-1), local_score, -jnp.inf
        )
      r_pad = layout_lib.round_up(r_tot, n_dev)
      r_loc = r_pad // n_dev
      ls_flat = local_score.reshape(r_tot, local_t)
      cl_flat = compress_lens.reshape(r_tot, 1)
      if r_pad > r_tot:
        ls_flat = jnp.pad(
            ls_flat, ((0, r_pad - r_tot), (0, 0)), constant_values=-jnp.inf
        )
        cl_flat = jnp.pad(cl_flat, ((0, r_pad - r_tot), (0, 0)))
      ls_all = lax.all_to_all(
          ls_flat.reshape(n_dev, r_loc, local_t),
          layout_lib.AXIS,
          split_axis=0,
          concat_axis=0,
      )
      cv_loc = jnp.moveaxis(ls_all, 0, 1).reshape(r_loc, 1, n_dev * local_t)
      cl_loc = lax.dynamic_slice_in_dim(
          cl_flat, my_id * jnp.int32(r_loc), r_loc, axis=0
      )
      tv_loc, tu_loc = _fast_topk_3d_jax(cv_loc, topk)
      tv_loc = tv_loc[:, 0, :]
      tu_loc = tu_loc[:, 0, :]
      vp_loc = (tu_loc < cl_loc) & (tv_loc > -jnp.inf)
      ms_loc = jnp.where(vp_loc, tu_loc, max_comp + 1)
      si_loc = jnp.sort(ms_loc, axis=-1)
      tk_loc = jnp.where(si_loc < cl_loc, si_loc, -1).astype(I32)
      tk_all = lax.all_gather(tk_loc, layout_lib.AXIS, axis=0).reshape(
          r_pad, topk
      )
      topk_idxs = tk_all[:r_tot].reshape(bsz, seqlen, topk)
      return topk_idxs, candidates_out

    loc_vals, loc_pos = _exact_two_stage_topk_jax(
        local_score,
        k_loc,
        bsize,
        precomputed_top_blk_idx=pre_top_bidx,
        precomputed_blk_max=pre_blk_max,
        blk_mask=blk_mask_local,
    )
    loc_gidx = t_start + loc_pos
    loc_both = jnp.concatenate([loc_vals, loc_gidx.astype(F32)], axis=-1)
    g_both = lax.all_gather(loc_both, layout_lib.AXIS, axis=0)
    if need_cand_filter:
      k_cand = min(c.candidate_topk_blocks, n_dev * loc_num_blocks)
      if n_dev > 1 and r_tot >= 4:
        r_pad = layout_lib.round_up(r_tot, n_dev)
        r_loc = r_pad // n_dev
        lb_flat = loc_blk_scores_pinned.reshape(r_tot, loc_num_blocks)
        if r_pad > r_tot:
          lb_flat = jnp.pad(
              lb_flat, ((0, r_pad - r_tot), (0, 0)), constant_values=-jnp.inf
          )
        lb_all = lax.all_to_all(
            lb_flat.reshape(n_dev, r_loc, loc_num_blocks),
            layout_lib.AXIS,
            split_axis=0,
            concat_axis=0,
        )
        cb_loc = jnp.moveaxis(lb_all, 0, 1).reshape(
            r_loc, 1, n_dev * loc_num_blocks
        )
        tb_loc, _ = _exact_two_stage_topk_jax(cb_loc, k_cand, block=4)
        th_all = _vmem_all_gather_axis0(tb_loc[:, 0, -1:], n_dev).reshape(
            r_pad, 1
        )
        thresh = th_all[:r_tot].reshape(bsz, seqlen, 1)
      else:
        loc_bvals, _ = _fast_topk_3d_jax(loc_blk_scores_pinned, k_blk)
        all_bvals = lax.all_gather(loc_bvals, layout_lib.AXIS, axis=0)
        cand_bvals = jnp.moveaxis(all_bvals, 0, -2).reshape(
            bsz, seqlen, n_dev * k_blk
        )
        top_bvals, _ = lax.top_k(cand_bvals, k_cand)
        thresh = top_bvals[..., -1:]
      loc_above = (loc_blk_scores_pinned > thresh) & (
          loc_blk_scores_pinned > -jnp.inf
      )
      loc_at = (loc_blk_scores_pinned == thresh) & (
          loc_blk_scores_pinned > -jnp.inf
      )
      loc_above_cnt = jnp.sum(loc_above.astype(I32), axis=-1, keepdims=True)
      loc_at_cnt = jnp.sum(loc_at.astype(I32), axis=-1, keepdims=True)
      all_cnts = _vmem_all_gather_axis0(
          jnp.concatenate([loc_above_cnt, loc_at_cnt], axis=-1), n_dev
      )
      total_above = jnp.sum(all_cnts[..., :1], axis=0)
      all_at_cnt = all_cnts[..., 1:]
      need_at = jnp.maximum(k_cand - total_above, 0)
      prior_at = jnp.sum(
          jnp.where(
              (jnp.arange(n_dev, dtype=I32) < my_id)[:, None, None, None],
              all_at_cnt,
              0,
          ),
          axis=0,
      )
      loc_at_rank = prior_at + jnp.cumsum(loc_at.astype(I32), axis=-1)
      cand_mask_loc = loc_above | (loc_at & (loc_at_rank <= need_at))
      if pre_top_bidx is not None:
        last_blk_loc = last_blk - my_id * jnp.int32(loc_num_blocks)
        has_last = (
            (last_blk_loc >= 0)
            & (last_blk_loc < loc_num_blocks)
            & (compress_lens > 0)
        )
        already_in = jnp.any(
            pre_top_bidx == last_blk_loc, axis=-1, keepdims=True
        )
        cand_bidx = jnp.where(
            has_last & ~already_in,
            jnp.concatenate([last_blk_loc, pre_top_bidx[..., :-1]], axis=-1),
            pre_top_bidx,
        )
        surv = jnp.take_along_axis(
            cand_mask_loc, jnp.maximum(cand_bidx, 0), axis=-1
        )
        candidates_out = jnp.where(surv, cand_bidx, -1).astype(I32)
      else:
        candidates_out = cand_mask_loc
    g_vals = g_both[..., :k_loc]
    g_idxs = g_both[..., k_loc:].astype(I32)
    cand_vals = jnp.moveaxis(g_vals, 0, -2).reshape(bsz, seqlen, n_dev * k_loc)
    cand_idxs = jnp.moveaxis(g_idxs, 0, -2).reshape(bsz, seqlen, n_dev * k_loc)
    if n_dev > 1 and r_tot >= 4:
      r_pad = layout_lib.round_up(r_tot, n_dev)
      r_loc = r_pad // n_dev
      cv_flat = cand_vals.reshape(r_tot, n_dev * k_loc)
      ci_flat = cand_idxs.reshape(r_tot, n_dev * k_loc)
      cl_flat = compress_lens.reshape(r_tot, 1)
      if r_pad > r_tot:
        cv_flat = jnp.pad(
            cv_flat, ((0, r_pad - r_tot), (0, 0)), constant_values=-jnp.inf
        )
        ci_flat = jnp.pad(ci_flat, ((0, r_pad - r_tot), (0, 0)))
        cl_flat = jnp.pad(cl_flat, ((0, r_pad - r_tot), (0, 0)))
      r0 = my_id * jnp.int32(r_loc)
      cv_loc = lax.dynamic_slice_in_dim(cv_flat, r0, r_loc, axis=0)
      ci_loc = lax.dynamic_slice_in_dim(ci_flat, r0, r_loc, axis=0)
      cl_loc = lax.dynamic_slice_in_dim(cl_flat, r0, r_loc, axis=0)
      tv_loc, tp_loc = lax.top_k(cv_loc, topk)
      tu_loc = jnp.take_along_axis(ci_loc, tp_loc, axis=-1)
      vp_loc = (tu_loc < cl_loc) & (tv_loc > -jnp.inf)
      ms_loc = jnp.where(vp_loc, tu_loc, max_comp + 1)
      si_loc = jnp.sort(ms_loc, axis=-1)
      tk_loc = jnp.where(si_loc < cl_loc, si_loc, -1).astype(I32)
      tk_all = lax.all_gather(tk_loc, layout_lib.AXIS, axis=0).reshape(
          r_pad, topk
      )
      topk_idxs = tk_all[:r_tot].reshape(bsz, seqlen, topk)
      return topk_idxs, candidates_out
    top_vals, top_pos = lax.top_k(cand_vals, topk)
    top_unsorted = jnp.take_along_axis(cand_idxs, top_pos, axis=-1)
    valid_pick = (top_unsorted < compress_lens) & (top_vals > -jnp.inf)
    masked_for_sort = jnp.where(valid_pick, top_unsorted, max_comp + 1)
    sorted_idx = jnp.sort(masked_for_sort, axis=-1)
    topk_idxs = jnp.where(sorted_idx < compress_lens, sorted_idx, -1).astype(
        I32
    )
    return topk_idxs, candidates_out

  if max_comp != full_comp:
    raise ValueError("scan_len needs the multi-device slab path")
  if cache_slot is not None:
    index_k_cache = index_k_cache[:, cache_slot]
  if jax.default_backend() == "tpu":
    raw_scores = jnp.einsum(
        "bshd,btd->bsht",
        q_idx.astype(BF16),
        index_k_cache.astype(BF16),
        preferred_element_type=F32,
    )
    index_score = jnp.einsum(
        "bsht,bsh->bst",
        jnp.maximum(raw_scores, 0.0),
        weights,
        preferred_element_type=F32,
    )
  elif c.index_n_heads <= h_step:
    raw_scores = jnp.einsum(
        "bshd,btd->bsht",
        q_idx.astype(F32),
        index_k_cache.astype(F32),
        preferred_element_type=F32,
    )
    index_score = jnp.sum(
        jnp.maximum(raw_scores, 0.0) * weights[:, :, :, None], axis=2
    )
  else:
    index_score = jnp.zeros((bsz, seqlen, max_comp), dtype=F32)
    for h0 in range(0, c.index_n_heads, h_step):
      h1 = h0 + h_step
      raw_chunk = jnp.einsum(
          "bshd,btd->bsht",
          q_idx[:, :, h0:h1, :].astype(F32),
          index_k_cache.astype(F32),
          preferred_element_type=F32,
      )
      index_score = index_score + jnp.sum(
          jnp.maximum(raw_chunk, 0.0) * weights[:, :, h0:h1, None], axis=2
      )

  t_iota = jnp.arange(max_comp, dtype=I32)[None, None, :]
  index_score = jnp.where(t_iota < compress_lens, index_score, -jnp.inf)

  candidates_out = candidates_in
  if is_candidate_source:
    candidates_out = _select_candidate_blocks_jax(
        index_score,
        compress_lens,
        c.candidate_topk_blocks,
        c.candidate_block_size,
    )
  elif uses_candidates and candidates_in is not None:
    index_score = jnp.where(candidates_in, index_score, -jnp.inf)

  _, top_unsorted = lax.top_k(index_score, topk)
  # Mask unreachable slots before sorting so valid indices sort ascending.
  masked_for_sort = jnp.where(
      top_unsorted < compress_lens, top_unsorted, max_comp + 1
  )
  sorted_idx = jnp.sort(masked_for_sort, axis=-1)
  topk_idxs = jnp.where(sorted_idx < compress_lens, sorted_idx, -1).astype(I32)
  return topk_idxs, candidates_out


# Smallest Indexer scan length and growth factor of the static scan ladder.
_INDEXER_SCAN_MIN_LEN = 4096
_INDEXER_SCAN_GROWTH = 2
# Each scan length past the first adds a lax.switch to every Indexer call (about
# 45 us on Ghostfish). A shorter scan length is kept only if it skips at least
# this many (query row, key) scores per device.
_INDEXER_SCAN_MIN_SKIPPED_SCORES = 4096


def _indexer_scan_buckets(
    lay: layout_lib.DeviceLayout,
    max_comp: int,
    candidate_layer: bool,
    rows: int,
) -> tuple[int, ...]:
  """Static ladder of Indexer scan lengths ending at the full cache length.

  At run time `_indexer_step_jax` picks the smallest length that covers every
  populated compressed KV slot, so unwritten cache slots are never scored. Each
  length keeps the 8-way contiguous-slab path of `_indexer_step_core_jax`, and
  on candidate layers (the candidate source and its users) every length must
  encode candidates as int32 [B, S, index_topk] local block ids (see
  `_uniform_candidates`). A shorter length is kept only if it skips at least
  `_INDEXER_SCAN_MIN_SKIPPED_SCORES` scores per device. If no shorter length
  qualifies the ladder is just (max_comp,), the original single-length program.

  Args:
    lay: Device layout.
    max_comp: Static compressed KV cache length.
    candidate_layer: Whether the layer is the candidate source or a user.
    rows: Query rows per call (batch size x sequence length).

  Returns:
    Increasing scan lengths; the last one is always `max_comp`.
  """
  c = lay.cfg
  n_dev = lay.num_devices
  bsize = c.candidate_block_size
  kc = c.index_topk

  def _ok(e: int) -> bool:
    if n_dev <= 1 or e < max(512, kc) or e % (n_dev * bsize) != 0:
      return False
    if not candidate_layer:
      return True
    local_t = e // n_dev
    loc_blocks = local_t // bsize
    if n_dev * loc_blocks <= c.candidate_topk_blocks:
      return True  # every populated block is a candidate
    if local_t >= 16384 and loc_blocks > min(kc, local_t):
      return True  # int32 top local block ids (pre_top_bidx)
    return loc_blocks <= kc  # positional bool block mask padded to width kc

  if not _ok(max_comp):
    return (max_comp,)
  out = []
  e = _INDEXER_SCAN_MIN_LEN
  while e < max_comp:
    skipped = rows * (max_comp - e) // n_dev
    if _ok(e) and skipped >= _INDEXER_SCAN_MIN_SKIPPED_SCORES:
      out.append(e)
    e *= _INDEXER_SCAN_GROWTH
  out.append(max_comp)
  return tuple(out)


def _uniform_candidates(
    cand: jax.Array | None, bsz: int, seqlen: int, width: int
) -> jax.Array:
  """Encodes per-device candidate blocks as int32 [B, S, width] (-1 = none)."""
  if cand is None:
    return jnp.full((bsz, seqlen, width), -1, dtype=I32)
  if cand.dtype == jnp.bool_:
    loc_b = cand.shape[-1]
    if loc_b > width:
      raise ValueError(f"bool candidate width {loc_b} > {width}")
    enc = jnp.where(cand, jnp.arange(loc_b, dtype=I32), -1)
    if loc_b < width:
      pad = jnp.full((bsz, seqlen, width - loc_b), -1, dtype=I32)
      enc = jnp.concatenate([enc, pad], axis=-1)
    return enc
  if cand.shape[-1] != width:
    raise ValueError(f"candidate width {cand.shape[-1]} != {width}")
  return cand.astype(I32)


def _decode_candidates(
    cand: jax.Array | None, lay: layout_lib.DeviceLayout, scan_len: int
) -> jax.Array | None:
  """Turns int32 [B, S, index_topk] candidates into the core's per-length form."""
  if cand is None:
    return None
  c = lay.cfg
  local_t = scan_len // lay.num_devices
  loc_blocks = local_t // c.candidate_block_size
  if lay.num_devices * loc_blocks <= c.candidate_topk_blocks:
    return None
  if local_t >= 16384 and loc_blocks > min(c.index_topk, local_t):
    return cand
  return cand[..., :loc_blocks] >= 0


def _indexer_step_jax(
    x: jax.Array,
    qr: jax.Array,
    start_pos: jax.Array,
    ratio: int,
    wq_b_w: jax.Array,
    wq_b_s: jax.Array,
    weights_proj: jax.Array,
    index_k_cache: jax.Array,
    cos_tbl: jax.Array,
    sin_tbl: jax.Array,
    *,
    lay: layout_lib.DeviceLayout,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidates_in: jax.Array | None,
    idx_slot: int | None = None,
    cache_slot: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
  """Runs the CSA2 Indexer over only the populated prefix of the KV cache.

  `index_k_cache` is sized for the server's maximum context (up to 1M), but at
  position p only the first (p + S) // ratio slots hold keys. Scoring the rest
  costs the same as a full-context request. This wrapper picks the smallest
  scan length from `_indexer_scan_buckets` that covers the populated prefix and
  runs `_indexer_step_core_jax` on `index_k_cache[:, :scan_len]` inside a
  `lax.switch`. The selected top-k set is unchanged: unwritten slots are masked
  to -inf in every branch, and contiguous slabs keep lower-position tie-breaks.
  """
  bsz, seqlen, _ = x.shape
  max_comp = index_k_cache.shape[-2]
  candidate_layer = is_candidate_source or uses_candidates
  ladder = _indexer_scan_buckets(
      lay, max_comp, candidate_layer, bsz * seqlen
  )
  core = functools.partial(
      _indexer_step_core_jax,
      x,
      qr,
      start_pos,
      ratio,
      wq_b_w,
      wq_b_s,
      weights_proj,
      cos_tbl=cos_tbl,
      sin_tbl=sin_tbl,
      lay=lay,
      is_candidate_source=is_candidate_source,
      uses_candidates=uses_candidates,
      idx_slot=idx_slot,
      cache_slot=cache_slot,
  )
  if len(ladder) == 1:
    return core(index_k_cache, candidates_in=candidates_in)

  width = lay.cfg.index_topk
  _, _, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  max_valid = jnp.max(jnp.where(active_b[:, None], (pos_seq + 1) // ratio, 0))
  # Scalar compares only: an array literal would land in jaxpr.consts.
  bucket = jnp.int32(0)
  for e in ladder[:-1]:
    bucket = bucket + (max_valid > jnp.int32(e)).astype(I32)

  def _branch(scan_len: int):
    def _run():
      topk_idxs, cand_out = core(
          index_k_cache,
          candidates_in=_decode_candidates(candidates_in, lay, scan_len),
          scan_len=scan_len,
      )
      if not candidate_layer:
        return topk_idxs, None
      if is_candidate_source:
        cand_out = _uniform_candidates(cand_out, bsz, seqlen, width)
      else:
        cand_out = candidates_in
      return topk_idxs, cand_out

    return _run

  return lax.switch(bucket, [_branch(e) for e in ladder])


def encoder_macrokernel(
    input_ids: jax.Array,
    start_pos: jax.Array,
    w: Mapping[str, jax.Array],
    caches: DeviceDecodeCaches,
    cos_swa: jax.Array,
    sin_swa: jax.Array,
    cos_comp: jax.Array,
    sin_comp: jax.Array,
    engram_hashes: jax.Array | None,
    *,
    lay: layout_lib.DeviceLayout,
    write_global_kv: bool = True,
    interpret: bool = False,
    return_m2_traces: bool = False,
    max_total_seq: int | None = None,
    simulate_full_43_blocks: bool = False,
    valid_len: jax.Array | int | None = None,
) -> Any:
  """Macro-Kernel 1: Causal Encoder (L0..L_dec-1) + L_dec Shared Global KV Write.

  Executes:
    1. Vocab-sharded embedding lookup + psum across 'tp'.
    2. Layers 0 .. L_dec - 1 (with Ahead-of-Time Engram at L1, L14 and
       Group-Cached Global KV across the 6-layer CSA2 m=2 groups).
    3. Epilogue: collapses L_{dec-1} output via hc_pre + L_dec.attn_norm and
       writes L_dec's single shared Decoder Global KV (comp_m1_kv) and Indexer K
       (comp_m1_idx_k).
  """
  c = lay.cfg
  bsz, seqlen = input_ids.shape
  win = c.window_size
  dec_start = c.decoder_start_layer
  my_id = (
      lax.axis_index(layout_lib.AXIS) if lay.num_devices > 1 else jnp.int32(0)
  )

  # 1. Sharded Embedding + Ahead-of-Time Engram Lookup (fused into 1 BF16 psum)
  vs = lay.vocab_shard
  v0 = my_id * jnp.int32(vs)
  v1 = jnp.minimum(v0 + jnp.int32(vs), jnp.int32(c.vocab_size))
  in_v = (input_ids >= v0) & (input_ids < v1)
  loc_tok = jnp.where(in_v, input_ids - v0, 0)
  emb_loc = jnp.where(
      in_v[..., None], w["embed"][loc_tok].astype(BF16), BF16(0.0)
  ).reshape(bsz * seqlen, c.dim)
  active_engrams = [
      (idx_e, l_e)
      for idx_e, l_e in enumerate(c.engram_layer_ids)
      if l_e < dec_start
  ]
  engram_loc_list = []
  if active_engrams and engram_hashes is not None:
    for idx_e, _ in active_engrams:
      engram_loc_list.append(
          _engram_local_deq_jax(
              engram_hashes[:, :, idx_e, :],
              w[f"engram_{idx_e}_embed_w"],
              w[f"engram_{idx_e}_embed_s"],
              lay=lay,
              engram_idx=idx_e,
          )
      )
    if (
        not isinstance(simulate_full_43_blocks, bool) or simulate_full_43_blocks
    ) and len(active_engrams) == 1:
      idx_e0, _ = active_engrams[0]
      engram_loc_list.append(
          _engram_local_deq_jax(
              engram_hashes[:, :, idx_e0, :],
              w[f"engram_{idx_e0}_embed_w"],
              w[f"engram_{idx_e0}_embed_s"],
              lay=lay,
              engram_idx=idx_e0,
          )
          * jnp.asarray(simulate_full_43_blocks, dtype=BF16)
      )
  all_emb_loc = jnp.concatenate([emb_loc, *engram_loc_list], axis=-1)
  if lay.num_devices > 1:
    all_emb_loc = lax.psum(all_emb_loc, layout_lib.AXIS)
  emb_bf16 = all_emb_loc[:, : c.dim].reshape(bsz, seqlen, c.dim)
  engram_kvs: dict[int, jax.Array] = {}
  col_off = c.dim
  for idx_e, l_e in active_engrams:
    e_dim = c.engram_n_hash_cols * c.engram_head_dim
    e_bf16 = all_emb_loc[:, col_off : col_off + e_dim]
    col_off += e_dim
    engram_kvs[l_e] = fp8_matmul_col_sharded(
        e_bf16,
        w[f"engram_{idx_e}_wkv_w"],
        w[f"engram_{idx_e}_wkv_s"],
        n_dev=lay.num_devices,
    )
  if (
      (not isinstance(simulate_full_43_blocks, bool) or simulate_full_43_blocks)
      and len(active_engrams) == 1
      and engram_hashes is not None
  ):
    idx_e0, l_e0 = active_engrams[0]
    e_dim = c.engram_n_hash_cols * c.engram_head_dim
    e_bf16_2 = all_emb_loc[:, col_off : col_off + e_dim]
    engram_kv_2 = fp8_matmul_col_sharded(
        e_bf16_2,
        w[f"engram_{idx_e0}_wkv_w"],
        w[f"engram_{idx_e0}_wkv_s"],
        n_dev=lay.num_devices,
    )
    engram_kvs[l_e0] = engram_kvs[l_e0] + 0.0 * engram_kv_2
  h = jnp.repeat(emb_bf16[:, :, None, :], c.hc_mult, axis=2)
  pre_mix = jnp.zeros((bsz, seqlen, c.hc_mult), dtype=F32).at[:, :, 0].set(1.0)

  _, _, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  tok_active = _token_active_mask(active_b, seqlen, valid_len)
  win_kv_all = caches.window_kv
  comp_m2_kv = caches.comp_m2_kv
  comp_m2_idx_k = caches.comp_m2_idx_k
  comp_m2_kv_st = caches.comp_m2_kv_state
  comp_m2_sc_st = caches.comp_m2_score_state

  # Group-cached Global KV shared across each CSA2 group.
  max_m2_comp = comp_m2_kv.shape[2]
  eff_m2_comp = min(
      max_m2_comp,
      (max_total_seq // 2) if max_total_seq is not None else max_m2_comp,
  )
  direct_3d_m2 = seqlen >= 64 and eff_m2_comp <= c.index_topk
  topk_m2 = eff_m2_comp if direct_3d_m2 else min(c.index_topk, max_m2_comp)
  group_kv = (
      jnp.zeros((bsz, topk_m2, c.head_dim), dtype=BF16)
      if direct_3d_m2
      else jnp.zeros((bsz, seqlen, topk_m2, c.head_dim), dtype=BF16)
  )
  group_valid = jnp.zeros((bsz, seqlen, topk_m2), dtype=jnp.bool_)

  m2_slot = 0
  idx_full_slot = 0
  idx_all_slot = 0
  m2_kv_traces = []
  m2_sc_traces = []

  for l in range(dec_start):
    if c.has_engram(l):
      idx_e = c.engram_layer_ids.index(l)
      h = _engram_apply_gating_jax(
          h,
          engram_kvs[l],
          w[f"engram_{idx_e}_q_weight"],
          w[f"engram_{idx_e}_k_weight"],
          norm_eps=c.norm_eps,
      )

    def _enc_attn_and_moe(
        h_in,
        pm_in,
        win_in,
        gkv_in,
        gval_in,
        l_cur=l,
        m2_s=m2_slot,
        if_s=idx_full_slot,
        ia_s=idx_all_slot,
        run_m2_leader=False,
    ):
      res_c = h_in
      apre, apost, acomb = mega_mhc_mixes_jax(
          h_in,
          w["hc_attn_fn"][l_cur],
          w["hc_attn_scale"][l_cur],
          w["hc_attn_base"][l_cur],
          c.norm_eps,
          c.hc_mult,
          c.hc_sinkhorn_iters,
          c.hc_eps,
      )
      x_c = mega_mhc_pre_norm_jax(
          h_in, pm_in, w["attn_norm"][l_cur], c.norm_eps
      )
      ratio_c = c.compress_ratios[l_cur]
      cos_c = cos_comp if ratio_c > 0 else cos_swa
      sin_c = sin_comp if ratio_c > 0 else sin_swa
      xq_c = act_quant_fp8_bf16(x_c.reshape(bsz * seqlen, c.dim))
      qr_c = _rms_norm_jax(
          fp8_matmul_col_sharded(
              xq_c,
              w["wq_a_w"],
              w["wq_a_s"],
              n_dev=lay.num_devices,
              layer_idx=l_cur,
              pre_quantized=True,
          ).astype(BF16),
          w["q_norm"][l_cur],
          c.norm_eps,
      )
      q_c = _apply_rotary_jax(
          fp8_matmul(qr_c, w["wq_b_w"], w["wq_b_s"], layer_idx=l_cur)
          .astype(BF16)
          .reshape(bsz, seqlen, lay.heads_per_device, c.head_dim),
          cos_c[pos_seq],
          sin_c[pos_seq],
          rope_dim=c.rope_head_dim,
      )
      kvn_c = _rms_norm_jax(
          fp8_matmul_col_sharded(
              xq_c,
              w["wkv_w"],
              w["wkv_s"],
              n_dev=lay.num_devices,
              layer_idx=l_cur,
              pre_quantized=True,
          )
          .astype(BF16)
          .reshape(bsz, seqlen, c.head_dim),
          w["kv_norm"][l_cur],
          c.norm_eps,
      )
      kvn_c = act_quant_fp8_bf16(
          _apply_rotary_jax(
              kvn_c,
              cos_c[pos_seq],
              sin_c[pos_seq],
              rope_dim=c.rope_head_dim,
          ).reshape(bsz * seqlen, c.head_dim)
      ).reshape(bsz, seqlen, c.head_dim)
      wexp_c, wval_c, cwin_c = _build_swa_window_kv(
          win_in[:, l_cur], kvn_c, start_pos, win, valid_len=valid_len
      )
      win_in = win_in.at[:, l_cur].set(cwin_c)
      if run_m2_leader:
        lat_c, _, _ = _compressor_m2_step_jax(
            x_c,
            start_pos,
            w["comp_m2_wkv"][m2_s],
            w["comp_m2_wgate"][m2_s],
            w["comp_m2_norm"][m2_s],
            c.norm_eps,
            comp_m2_kv_st[:, m2_s],
            comp_m2_sc_st[:, m2_s],
        )
        _ = _rms_norm_jax(
            _dot_f32(lat_c.reshape(bsz * seqlen, c.head_dim), w["idx_wk"][if_s])
            .astype(BF16)
            .reshape(bsz, seqlen, c.index_head_dim),
            w["idx_k_norm"][if_s],
            c.norm_eps,
        )
        if not direct_3d_m2:
          tidx_c, _ = _indexer_step_jax(
              x_c,
              qr_c.reshape(bsz, seqlen, c.q_lora_rank),
              start_pos,
              ratio_c,
              w["idx_wq_b_w"],
              w["idx_wq_b_s"],
              w["idx_weights_proj"][ia_s],
              comp_m2_idx_k,
              cos_c,
              sin_c,
              lay=lay,
              is_candidate_source=False,
              uses_candidates=False,
              candidates_in=None,
              idx_slot=ia_s,
              cache_slot=m2_s,
          )
          gkv_in, gval_in = _gather_kv_by_indices_jax(
              comp_m2_kv, tidx_c, slot=m2_s
          )
      o_c = _apply_rotary_jax(
          _sparse_attn_with_sink_jax(
              q_c,
              gkv_in,
              gval_in,
              w["attn_sink"][l_cur],
              c.softmax_scale,
              win_kv=wexp_c,
              win_valid=wval_c,
          ),
          cos_c[pos_seq],
          sin_c[pos_seq],
          rope_dim=c.rope_head_dim,
          inverse=True,
      )
      og_c = o_c.reshape(
          bsz * seqlen, lay.o_groups_per_device, lay.group_head_cols
      )
      op_c = jnp.concatenate(
          [
              fp8_matmul(
                  og_c[:, g, :],
                  w["wo_a_w"],
                  w["wo_a_s"],
                  layer_idx=l_cur,
                  group_idx=g,
                  out_dtype=BF16,
              )
              for g in range(lay.o_groups_per_device)
          ],
          axis=1,
      )
      aout_c = fp8_matmul(
          op_c, w["wo_b_w"], w["wo_b_s"], layer_idx=l_cur, out_dtype=F32
      )
      if lay.num_devices > lay.dense_tp:
        aout_c = jnp.where(my_id < jnp.int32(lay.dense_tp), aout_c, 0.0)
      if lay.num_devices > 1:
        aout_c = lax.psum(aout_c, layout_lib.AXIS)
      h_out = mega_mhc_post_jax(
          aout_c.astype(BF16).reshape(bsz, seqlen, c.dim),
          res_c,
          apost,
          acomb,
      )
      res_f = h_out
      fpre, fpost, fcomb = mega_mhc_mixes_jax(
          h_out,
          w["hc_ffn_fn"][l_cur],
          w["hc_ffn_scale"][l_cur],
          w["hc_ffn_base"][l_cur],
          c.norm_eps,
          c.hc_mult,
          c.hc_sinkhorn_iters,
          c.hc_eps,
      )
      xffn_c = mega_mhc_pre_norm_jax(
          h_out, apre, w["ffn_norm"][l_cur], c.norm_eps
      )
      mout_c = _moe_layer_sharded_jax(
          xffn_c,
          w["bb_gate_w"][l_cur],
          w["bb_gate_bias"][l_cur],
          w["shared_gu_w"],
          w["shared_gu_s"],
          w["shared_dn_w"],
          w["shared_dn_s"],
          w["bb_exp_gu_w"],
          w["bb_exp_gu_s"],
          w["bb_exp_dn_w"],
          w["bb_exp_dn_s"],
          layer_idx=l_cur,
          shared_layer_idx=l_cur,
          active_b=tok_active,
          lay=lay,
          is_mtp=False,
          interpret=interpret,
      )
      h_out = mega_mhc_post_jax(mout_c, res_f, fpost, fcomb)
      return (h_out, fpre, win_in, gkv_in, gval_in)

    if (
        (
            not isinstance(simulate_full_43_blocks, bool)
            or simulate_full_43_blocks
        )
        and dec_start == 3
        and l == 2
    ):
      n_enc_rep = jnp.where(
          simulate_full_43_blocks, jnp.int32(16), jnp.int32(1)
      )
      h, pre_mix, win_kv_all, group_kv, group_valid = lax.fori_loop(
          jnp.int32(0),
          n_enc_rep,
          lambda _, st_f, fn=_enc_attn_and_moe: fn(*st_f),
          (h, pre_mix, win_kv_all, group_kv, group_valid),
      )
      continue

    # Attention Sub-block
    residual = h
    attn_pre, attn_post, attn_comb = mega_mhc_mixes_jax(
        h,
        w["hc_attn_fn"][l],
        w["hc_attn_scale"][l],
        w["hc_attn_base"][l],
        c.norm_eps,
        c.hc_mult,
        c.hc_sinkhorn_iters,
        c.hc_eps,
    )
    x = mega_mhc_pre_norm_jax(h, pre_mix, w["attn_norm"][l], c.norm_eps)

    ratio = c.compress_ratios[l]
    cos_tbl = cos_comp if ratio > 0 else cos_swa
    sin_tbl = sin_comp if ratio > 0 else sin_swa

    x_flat = x.reshape(bsz * seqlen, c.dim)
    x_q = act_quant_fp8_bf16(x_flat)
    qr = _rms_norm_jax(
        fp8_matmul_col_sharded(
            x_q,
            w["wq_a_w"],
            w["wq_a_s"],
            n_dev=lay.num_devices,
            layer_idx=l,
            pre_quantized=True,
        ).astype(BF16),
        w["q_norm"][l],
        c.norm_eps,
    )
    q = (
        fp8_matmul(qr, w["wq_b_w"], w["wq_b_s"], layer_idx=l)
        .astype(BF16)
        .reshape(bsz, seqlen, lay.heads_per_device, c.head_dim)
    )
    q = _apply_rotary_jax(
        q, cos_tbl[pos_seq], sin_tbl[pos_seq], rope_dim=c.rope_head_dim
    )

    kv_new = _rms_norm_jax(
        fp8_matmul_col_sharded(
            x_q,
            w["wkv_w"],
            w["wkv_s"],
            n_dev=lay.num_devices,
            layer_idx=l,
            pre_quantized=True,
        )
        .astype(BF16)
        .reshape(bsz, seqlen, c.head_dim),
        w["kv_norm"][l],
        c.norm_eps,
    )
    kv_new = _apply_rotary_jax(
        kv_new, cos_tbl[pos_seq], sin_tbl[pos_seq], rope_dim=c.rope_head_dim
    )
    kv_new = act_quant_fp8_bf16(
        kv_new.reshape(bsz * seqlen, c.head_dim)
    ).reshape(bsz, seqlen, c.head_dim)

    # Update layer l's SWA ring buffer [B, win, head_dim]
    win_kv_expanded, win_valid, cur_win_kv = _build_swa_window_kv(
        win_kv_all[:, l], kv_new, start_pos, win, valid_len=valid_len
    )
    win_kv_all = win_kv_all.at[:, l].set(cur_win_kv)

    if ratio > 0:
      if c.is_kv_source(l):
        latents_all, kv_st_trace, sc_st_trace = _compressor_m2_step_jax(
            x,
            start_pos,
            w["comp_m2_wkv"][m2_slot],
            w["comp_m2_wgate"][m2_slot],
            w["comp_m2_norm"][m2_slot],
            c.norm_eps,
            comp_m2_kv_st[:, m2_slot],
            comp_m2_sc_st[:, m2_slot],
        )
        m2_kv_traces.append(kv_st_trace)
        m2_sc_traces.append(sc_st_trace)
        if valid_len is None:
          comp_m2_kv_st = comp_m2_kv_st.at[:, m2_slot].set(kv_st_trace[:, -1])
          comp_m2_sc_st = comp_m2_sc_st.at[:, m2_slot].set(sc_st_trace[:, -1])
        else:
          v_idx = jnp.clip(
              jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), (bsz,)) - 1,
              0,
              seqlen - 1,
          )[:, None, None, None]
          sel_kv = jnp.take_along_axis(kv_st_trace, v_idx, axis=1)[:, 0]
          sel_sc = jnp.take_along_axis(sc_st_trace, v_idx, axis=1)[:, 0]
          comp_m2_kv_st = comp_m2_kv_st.at[:, m2_slot].set(sel_kv)
          comp_m2_sc_st = comp_m2_sc_st.at[:, m2_slot].set(sel_sc)

        if write_global_kv:
          completed = (((pos_seq + 1) % ratio) == 0) & active_b[:, None]
          g_idx = pos_seq // ratio
          g_pos = g_idx * ratio
          k_all = _rms_norm_jax(
              _dot_f32(
                  latents_all.reshape(bsz * seqlen, c.head_dim),
                  w["idx_wk"][idx_full_slot],
              )
              .astype(BF16)
              .reshape(bsz, seqlen, c.index_head_dim),
              w["idx_k_norm"][idx_full_slot],
              c.norm_eps,
          )
          k_all = _apply_rotary_jax(
              k_all,
              cos_tbl[g_pos],
              sin_tbl[g_pos],
              rope_dim=c.rope_head_dim,
          )
          k_all = fp4_act_quant_bf16(
              k_all.reshape(bsz * seqlen, c.index_head_dim), scale_dtype="e8m0"
          ).reshape(bsz, seqlen, c.index_head_dim)

          rot_lat_all = _apply_rotary_jax(
              latents_all,
              cos_tbl[g_pos],
              sin_tbl[g_pos],
              rope_dim=c.rope_head_dim,
          )
          rot_lat_all = fp4_act_quant_bf16(
              rot_lat_all.reshape(bsz * seqlen, c.head_dim), scale_dtype="e4m3"
          ).reshape(bsz, seqlen, c.head_dim)

          if seqlen >= 16 and seqlen % 2 == 0 and ratio == 2:
            half_s = seqlen // 2
            k_even = k_all[:, 0::2, :]
            k_odd = k_all[:, 1::2, :]
            lat_even = rot_lat_all[:, 0::2, :]
            lat_odd = rot_lat_all[:, 1::2, :]
            for b_i in range(bsz):
              starts_odd = (pos_seq[b_i, 0] % 2) == 1
              g_s0 = jnp.clip(g_idx[b_i, 0], 0, max(0, max_m2_comp - half_s))
              k_sel = jnp.where(starts_odd, k_even[b_i], k_odd[b_i])
              lat_sel = jnp.where(starts_odd, lat_even[b_i], lat_odd[b_i])
              old_ik = lax.dynamic_slice(
                  comp_m2_idx_k,
                  (jnp.int32(b_i), jnp.int32(m2_slot), g_s0, jnp.int32(0)),
                  (1, 1, half_s, c.index_head_dim),
              )
              old_kv = lax.dynamic_slice(
                  comp_m2_kv,
                  (jnp.int32(b_i), jnp.int32(m2_slot), g_s0, jnp.int32(0)),
                  (1, 1, half_s, c.head_dim),
              )
              new_ik = jnp.where(
                  active_b[b_i],
                  k_sel.reshape(1, 1, half_s, c.index_head_dim),
                  old_ik,
              )
              new_kv = jnp.where(
                  active_b[b_i],
                  lat_sel.reshape(1, 1, half_s, c.head_dim),
                  old_kv,
              )
              comp_m2_idx_k = lax.dynamic_update_slice(
                  comp_m2_idx_k,
                  new_ik,
                  (jnp.int32(b_i), jnp.int32(m2_slot), g_s0, jnp.int32(0)),
              )
              comp_m2_kv = lax.dynamic_update_slice(
                  comp_m2_kv,
                  new_kv,
                  (jnp.int32(b_i), jnp.int32(m2_slot), g_s0, jnp.int32(0)),
              )
          else:
            for b_i in range(bsz):
              for s_i in range(seqlen):
                g_s = jnp.clip(g_idx[b_i, s_i], 0, max_m2_comp - 1)
                old_ik = lax.dynamic_slice(
                    comp_m2_idx_k,
                    (jnp.int32(b_i), jnp.int32(m2_slot), g_s, jnp.int32(0)),
                    (1, 1, 1, c.index_head_dim),
                )
                old_kv = lax.dynamic_slice(
                    comp_m2_kv,
                    (jnp.int32(b_i), jnp.int32(m2_slot), g_s, jnp.int32(0)),
                    (1, 1, 1, c.head_dim),
                )
                new_ik = jnp.where(
                    completed[b_i, s_i],
                    k_all[b_i, s_i].reshape(1, 1, 1, c.index_head_dim),
                    old_ik,
                )
                new_kv = jnp.where(
                    completed[b_i, s_i],
                    rot_lat_all[b_i, s_i].reshape(1, 1, 1, c.head_dim),
                    old_kv,
                )
                comp_m2_idx_k = lax.dynamic_update_slice(
                    comp_m2_idx_k,
                    new_ik,
                    (jnp.int32(b_i), jnp.int32(m2_slot), g_s, jnp.int32(0)),
                )
                comp_m2_kv = lax.dynamic_update_slice(
                    comp_m2_kv,
                    new_kv,
                    (jnp.int32(b_i), jnp.int32(m2_slot), g_s, jnp.int32(0)),
                )

        if direct_3d_m2:
          comp_lens_m2 = jnp.where(
              active_b[:, None], (pos_seq + 1) // ratio, 0
          )[:, :, None]
          group_kv = comp_m2_kv[:, m2_slot, :eff_m2_comp, :]
          group_valid = (
              jnp.arange(eff_m2_comp, dtype=I32)[None, None, :] < comp_lens_m2
          )
        else:
          # Leader layer: run Indexer and gather Group-Cached Global KV once.
          topk_idxs, _ = _indexer_step_jax(
              x,
              qr.reshape(bsz, seqlen, c.q_lora_rank),
              start_pos,
              ratio,
              w["idx_wq_b_w"],
              w["idx_wq_b_s"],
              w["idx_weights_proj"][idx_all_slot],
              comp_m2_idx_k,
              cos_tbl,
              sin_tbl,
              lay=lay,
              is_candidate_source=False,
              uses_candidates=False,
              candidates_in=None,
              idx_slot=idx_all_slot,
              cache_slot=m2_slot,
          )
          group_kv, group_valid = _gather_kv_by_indices_jax(
              comp_m2_kv, topk_idxs, slot=m2_slot
          )
        m2_slot += 1
        idx_full_slot += 1
        idx_all_slot += 1

      o = _sparse_attn_with_sink_jax(
          q,
          group_kv,
          group_valid,
          w["attn_sink"][l],
          c.softmax_scale,
          win_kv=win_kv_expanded,
          win_valid=win_valid,
      )
    else:
      o = _sparse_attn_with_sink_jax(
          q, win_kv_expanded, win_valid, w["attn_sink"][l], c.softmax_scale
      )
    o = _apply_rotary_jax(
        o,
        cos_tbl[pos_seq],
        sin_tbl[pos_seq],
        rope_dim=c.rope_head_dim,
        inverse=True,
    )
    o_grouped = o.reshape(
        bsz * seqlen, lay.o_groups_per_device, lay.group_head_cols
    )
    o_proj = jnp.concatenate(
        [
            fp8_matmul(
                o_grouped[:, g, :],
                w["wo_a_w"],
                w["wo_a_s"],
                layer_idx=l,
                group_idx=g,
                out_dtype=BF16,
            )
            for g in range(lay.o_groups_per_device)
        ],
        axis=1,
    )
    attn_out = fp8_matmul(
        o_proj, w["wo_b_w"], w["wo_b_s"], layer_idx=l, out_dtype=F32
    )
    if lay.num_devices > lay.dense_tp:
      attn_out = jnp.where(my_id < jnp.int32(lay.dense_tp), attn_out, 0.0)
    if lay.num_devices > 1:
      attn_out = lax.psum(attn_out, layout_lib.AXIS)
    h = mega_mhc_post_jax(
        attn_out.astype(BF16).reshape(bsz, seqlen, c.dim),
        residual,
        attn_post,
        attn_comb,
    )

    # FFN / MoE Sub-block
    residual = h
    ffn_pre, ffn_post, ffn_comb = mega_mhc_mixes_jax(
        h,
        w["hc_ffn_fn"][l],
        w["hc_ffn_scale"][l],
        w["hc_ffn_base"][l],
        c.norm_eps,
        c.hc_mult,
        c.hc_sinkhorn_iters,
        c.hc_eps,
    )
    x_ffn = mega_mhc_pre_norm_jax(h, attn_pre, w["ffn_norm"][l], c.norm_eps)
    moe_out = _moe_layer_sharded_jax(
        x_ffn,
        w["bb_gate_w"][l],
        w["bb_gate_bias"][l],
        w["shared_gu_w"],
        w["shared_gu_s"],
        w["shared_dn_w"],
        w["shared_dn_s"],
        w["bb_exp_gu_w"],
        w["bb_exp_gu_s"],
        w["bb_exp_dn_w"],
        w["bb_exp_dn_s"],
        layer_idx=l,
        shared_layer_idx=l,
        active_b=tok_active,
        lay=lay,
        is_mtp=False,
        interpret=interpret,
    )
    h = mega_mhc_post_jax(moe_out, residual, ffn_post, ffn_comb)
    pre_mix = ffn_pre

    if (
        (
            not isinstance(simulate_full_43_blocks, bool)
            or simulate_full_43_blocks
        )
        and dec_start == 3
        and l == 1
    ):
      n_m2_rep = jnp.where(simulate_full_43_blocks, jnp.int32(2), jnp.int32(0))
      idx_e0 = c.engram_layer_ids.index(l)
      h_eg2 = _engram_apply_gating_jax(
          h,
          engram_kvs[l],
          w[f"engram_{idx_e0}_q_weight"],
          w[f"engram_{idx_e0}_k_weight"],
          norm_eps=c.norm_eps,
      )
      h = jnp.where(simulate_full_43_blocks, h_eg2, h)
      h, pre_mix, win_kv_all, group_kv, group_valid = lax.fori_loop(
          jnp.int32(0),
          n_m2_rep,
          lambda _, st_m2, fn=_enc_attn_and_moe, ms=m2_slot - 1, ifs=idx_full_slot - 1, ias=idx_all_slot - 1: fn(
              *st_m2, m2_s=ms, if_s=ifs, ia_s=ias, run_m2_leader=True
          ),
          (h, pre_mix, win_kv_all, group_kv, group_valid),
      )

  # Epilogue of Macro-Kernel 1: Write L_dec shared Decoder Global KV & IndexK.
  comp_m1_kv = caches.comp_m1_kv
  comp_m1_idx_k = caches.comp_m1_idx_k
  if write_global_kv:
    x_l20_in = mega_mhc_pre_norm_jax(
        h, pre_mix, w["attn_norm"][dec_start], c.norm_eps
    )
    lat_m1 = _rms_norm_jax(
        _dot_f32(x_l20_in.reshape(bsz * seqlen, c.dim), w["comp_m1_wkv"][0])
        .astype(BF16)
        .reshape(bsz, seqlen, c.head_dim),
        w["comp_m1_norm"][0],
        c.norm_eps,
    )
    k_idx_m1 = _rms_norm_jax(
        _dot_f32(
            lat_m1.reshape(bsz * seqlen, c.head_dim), w["idx_wk"][idx_full_slot]
        )
        .astype(BF16)
        .reshape(bsz, seqlen, c.index_head_dim),
        w["idx_k_norm"][idx_full_slot],
        c.norm_eps,
    )
    k_idx_m1 = _apply_rotary_jax(
        k_idx_m1, cos_comp[pos_seq], sin_comp[pos_seq], rope_dim=c.rope_head_dim
    )
    k_idx_m1 = fp4_act_quant_bf16(
        k_idx_m1.reshape(bsz * seqlen, c.index_head_dim), scale_dtype="e8m0"
    ).reshape(bsz, seqlen, c.index_head_dim)

    rot_lat_m1 = _apply_rotary_jax(
        lat_m1, cos_comp[pos_seq], sin_comp[pos_seq], rope_dim=c.rope_head_dim
    )
    rot_lat_m1 = fp4_act_quant_bf16(
        rot_lat_m1.reshape(bsz * seqlen, c.head_dim), scale_dtype="e4m3"
    ).reshape(bsz, seqlen, c.head_dim)

    max_m1_len = comp_m1_kv.shape[2]
    for b_i in range(bsz):
      p0_b = jnp.clip(pos_seq[b_i, 0], 0, max(0, max_m1_len - seqlen))
      old_kv = lax.dynamic_slice(
          comp_m1_kv,
          (jnp.int32(b_i), jnp.int32(0), p0_b, jnp.int32(0)),
          (1, 1, seqlen, c.head_dim),
      )
      old_ik = lax.dynamic_slice(
          comp_m1_idx_k,
          (jnp.int32(b_i), jnp.int32(0), p0_b, jnp.int32(0)),
          (1, 1, seqlen, c.index_head_dim),
      )
      new_kv = jnp.where(
          active_b[b_i],
          rot_lat_m1[b_i].reshape(1, 1, seqlen, c.head_dim),
          old_kv,
      )
      new_ik = jnp.where(
          active_b[b_i],
          k_idx_m1[b_i].reshape(1, 1, seqlen, c.index_head_dim),
          old_ik,
      )
      comp_m1_kv = lax.dynamic_update_slice(
          comp_m1_kv,
          new_kv,
          (jnp.int32(b_i), jnp.int32(0), p0_b, jnp.int32(0)),
      )
      comp_m1_idx_k = lax.dynamic_update_slice(
          comp_m1_idx_k,
          new_ik,
          (jnp.int32(b_i), jnp.int32(0), p0_b, jnp.int32(0)),
      )

  new_caches = dataclasses.replace(
      caches,
      window_kv=win_kv_all,
      comp_m2_kv=comp_m2_kv,
      comp_m2_idx_k=comp_m2_idx_k,
      comp_m2_kv_state=comp_m2_kv_st,
      comp_m2_score_state=comp_m2_sc_st,
      comp_m1_kv=comp_m1_kv,
      comp_m1_idx_k=comp_m1_idx_k,
  )
  if return_m2_traces:
    return h, pre_mix, new_caches, m2_kv_traces, m2_sc_traces
  return h, pre_mix, new_caches


def decoder_macrokernel(
    h_enc: jax.Array,
    pre_mix_enc: jax.Array,
    start_pos: jax.Array,
    w: Mapping[str, jax.Array],
    caches: DeviceDecodeCaches,
    cos_comp: jax.Array,
    sin_comp: jax.Array,
    *,
    lay: layout_lib.DeviceLayout,
    full_logits: bool = False,
    interpret: bool = False,
    max_total_seq: int | None = None,
    simulate_full_43_blocks: bool = False,
    valid_len: jax.Array | int | None = None,
    temperature: float = 0.0,
    top_k: int = 20,
    rng_seed: jax.Array | None = None,
    draft_ids: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, DeviceDecodeCaches]:
  """Macro-Kernel 2: 20-Layer Decoder (L_dec..n_layers-1) + Hierarchical Indexer + Head.

  Returns:
    (output_ids [B] or [B, S], logits_sharded [B, S_out, vocab_shard],
     main_hidden [B, S, 3 * dim], updated_caches).
  """
  c = lay.cfg
  bsz, seqlen, _, _ = h_enc.shape
  win = c.window_size
  dec_start = c.decoder_start_layer
  my_id = (
      lax.axis_index(layout_lib.AXIS) if lay.num_devices > 1 else jnp.int32(0)
  )

  h = h_enc
  pre_mix = pre_mix_enc
  _, _, active_b, pos_seq = _pos_and_active(start_pos, bsz, seqlen)
  tok_active = _token_active_mask(active_b, seqlen, valid_len)
  win_kv_all = caches.window_kv
  shared_m1_kv = caches.comp_m1_kv[:, 0]
  shared_m1_ik = caches.comp_m1_idx_k[:, 0]
  max_m1_comp = shared_m1_kv.shape[1]
  eff_m1_comp = min(
      max_m1_comp,
      max_total_seq
      if max_total_seq is not None
      else (seqlen if seqlen > win else max_m1_comp),
  )
  direct_3d_m1 = seqlen >= 64 and eff_m1_comp <= min(
      c.index_topk, c.candidate_topk_blocks * c.candidate_block_size
  )
  topk_m1 = eff_m1_comp if direct_3d_m1 else min(c.index_topk, max_m1_comp)

  if direct_3d_m1:
    comp_lens_m1 = jnp.where(active_b[:, None], pos_seq + 1, 0)[:, :, None]
    group_kv = shared_m1_kv[:, :eff_m1_comp, :]
    group_valid = (
        jnp.arange(eff_m1_comp, dtype=I32)[None, None, :] < comp_lens_m1
    )
  else:
    group_kv = jnp.zeros((bsz, seqlen, topk_m1, c.head_dim), dtype=BF16)
    group_valid = jnp.zeros((bsz, seqlen, topk_m1), dtype=jnp.bool_)
  candidates: jax.Array | None = None

  idx_all_slot = sum(1 for l in c.index_source_layers if l < dec_start)
  main_hiddens = []

  for l in range(dec_start, c.n_layers):
    if l in c.dspark_target_layer_ids:
      main_hiddens.append(jnp.mean(h.astype(F32), axis=2).astype(BF16))

    cur_ia_slot = idx_all_slot
    if c.is_index_source(l):
      idx_all_slot += 1

    def _dec_step(st_d, l_cur=l, ia_s=cur_ia_slot):
      h_c, pm_c, win_c, gkv_c, gval_c, cand_c = st_d
      res_c = h_c
      apre, apost, acomb = mega_mhc_mixes_jax(
          h_c,
          w["hc_attn_fn"][l_cur],
          w["hc_attn_scale"][l_cur],
          w["hc_attn_base"][l_cur],
          c.norm_eps,
          c.hc_mult,
          c.hc_sinkhorn_iters,
          c.hc_eps,
      )
      x_c = mega_mhc_pre_norm_jax(h_c, pm_c, w["attn_norm"][l_cur], c.norm_eps)
      xq_c = act_quant_fp8_bf16(x_c.reshape(bsz * seqlen, c.dim))

      qr_c = _rms_norm_jax(
          fp8_matmul_col_sharded(
              xq_c,
              w["wq_a_w"],
              w["wq_a_s"],
              n_dev=lay.num_devices,
              layer_idx=l_cur,
              pre_quantized=True,
          ).astype(BF16),
          w["q_norm"][l_cur],
          c.norm_eps,
      )
      q_c = (
          fp8_matmul(qr_c, w["wq_b_w"], w["wq_b_s"], layer_idx=l_cur)
          .astype(BF16)
          .reshape(bsz, seqlen, lay.heads_per_device, c.head_dim)
      )
      q_c = _apply_rotary_jax(
          q_c, cos_comp[pos_seq], sin_comp[pos_seq], rope_dim=c.rope_head_dim
      )

      kvn_c = _rms_norm_jax(
          fp8_matmul_col_sharded(
              xq_c,
              w["wkv_w"],
              w["wkv_s"],
              n_dev=lay.num_devices,
              layer_idx=l_cur,
              pre_quantized=True,
          )
          .astype(BF16)
          .reshape(bsz, seqlen, c.head_dim),
          w["kv_norm"][l_cur],
          c.norm_eps,
      )
      kvn_c = _apply_rotary_jax(
          kvn_c, cos_comp[pos_seq], sin_comp[pos_seq], rope_dim=c.rope_head_dim
      )
      kvn_c = act_quant_fp8_bf16(
          kvn_c.reshape(bsz * seqlen, c.head_dim)
      ).reshape(bsz, seqlen, c.head_dim)

      wexp_c, wval_c, cwin_c = _build_swa_window_kv(
          win_c[:, l_cur], kvn_c, start_pos, win, valid_len=valid_len
      )
      win_c = win_c.at[:, l_cur].set(cwin_c)

      if c.is_index_source(l_cur) and not direct_3d_m1:
        tidx_c, cand_new = _indexer_step_jax(
            x_c,
            qr_c.reshape(bsz, seqlen, c.q_lora_rank),
            start_pos,
            1,
            w["idx_wq_b_w"],
            w["idx_wq_b_s"],
            w["idx_weights_proj"][ia_s],
            shared_m1_ik,
            cos_comp,
            sin_comp,
            lay=lay,
            is_candidate_source=c.is_candidate_source(l_cur),
            uses_candidates=c.uses_candidates(l_cur),
            candidates_in=cand_c,
            idx_slot=ia_s,
        )
        if c.is_candidate_source(l_cur):
          cand_c = cand_new
        gkv_c, gval_c = _gather_kv_by_indices_jax(shared_m1_kv, tidx_c)

      o_c = _sparse_attn_with_sink_jax(
          q_c,
          gkv_c,
          gval_c,
          w["attn_sink"][l_cur],
          c.softmax_scale,
          win_kv=wexp_c,
          win_valid=wval_c,
      )
      o_c = _apply_rotary_jax(
          o_c,
          cos_comp[pos_seq],
          sin_comp[pos_seq],
          rope_dim=c.rope_head_dim,
          inverse=True,
      )
      og_c = o_c.reshape(
          bsz * seqlen, lay.o_groups_per_device, lay.group_head_cols
      )
      op_c = jnp.concatenate(
          [
              fp8_matmul(
                  og_c[:, g, :],
                  w["wo_a_w"],
                  w["wo_a_s"],
                  layer_idx=l_cur,
                  group_idx=g,
                  out_dtype=BF16,
              )
              for g in range(lay.o_groups_per_device)
          ],
          axis=1,
      )
      aout_c = fp8_matmul(
          op_c, w["wo_b_w"], w["wo_b_s"], layer_idx=l_cur, out_dtype=F32
      )
      if lay.num_devices > lay.dense_tp:
        aout_c = jnp.where(my_id < jnp.int32(lay.dense_tp), aout_c, 0.0)
      if lay.num_devices > 1:
        aout_c = lax.psum(aout_c, layout_lib.AXIS)
      h_c = mega_mhc_post_jax(
          aout_c.astype(BF16).reshape(bsz, seqlen, c.dim),
          res_c,
          apost,
          acomb,
      )

      res_f = h_c
      fpre, fpost, fcomb = mega_mhc_mixes_jax(
          h_c,
          w["hc_ffn_fn"][l_cur],
          w["hc_ffn_scale"][l_cur],
          w["hc_ffn_base"][l_cur],
          c.norm_eps,
          c.hc_mult,
          c.hc_sinkhorn_iters,
          c.hc_eps,
      )
      xffn_c = mega_mhc_pre_norm_jax(
          h_c, apre, w["ffn_norm"][l_cur], c.norm_eps
      )
      mout_c = _moe_layer_sharded_jax(
          xffn_c,
          w["bb_gate_w"][l_cur],
          w["bb_gate_bias"][l_cur],
          w["shared_gu_w"],
          w["shared_gu_s"],
          w["shared_dn_w"],
          w["shared_dn_s"],
          w["bb_exp_gu_w"],
          w["bb_exp_gu_s"],
          w["bb_exp_dn_w"],
          w["bb_exp_dn_s"],
          layer_idx=l_cur,
          shared_layer_idx=l_cur,
          active_b=tok_active,
          lay=lay,
          is_mtp=False,
          interpret=interpret,
      )
      h_c = mega_mhc_post_jax(mout_c, res_f, fpost, fcomb)
      return (h_c, fpre, win_c, gkv_c, gval_c, cand_c)

    if (
        (
            not isinstance(simulate_full_43_blocks, bool)
            or simulate_full_43_blocks
        )
        and (c.n_layers - dec_start) == 3
        and l in (dec_start + 1, c.n_layers - 1)
    ):
      n_dec_rep = jnp.where(
          simulate_full_43_blocks,
          jnp.int32(2 if l == dec_start + 1 else 17),
          jnp.int32(1),
      )
      h, pre_mix, win_kv_all, group_kv, group_valid = lax.fori_loop(
          jnp.int32(0),
          n_dec_rep,
          lambda _, st_r, fn=_dec_step, cd=candidates: fn((*st_r, cd))[:5],
          (h, pre_mix, win_kv_all, group_kv, group_valid),
      )
    else:
      h, pre_mix, win_kv_all, group_kv, group_valid, candidates = _dec_step(
          (h, pre_mix, win_kv_all, group_kv, group_valid, candidates)
      )

  h_collapsed = jnp.sum(
      pre_mix.astype(F32)[..., None] * h.astype(F32), axis=2
  ).astype(BF16)
  if not full_logits:
    if valid_len is None:
      h_collapsed = h_collapsed[:, -1:, :]
    else:
      v_idx = jnp.clip(
          jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), (bsz,)) - 1,
          0,
          seqlen - 1,
      )[:, None, None]
      h_collapsed = jnp.take_along_axis(h_collapsed, v_idx, axis=1)
  normed = _rms_norm_jax(h_collapsed, w["norm"], c.norm_eps)
  s_out = normed.shape[1]

  # Sharded LM Head + distributed peer argmax / stochastic Gumbel-max top-k sampling
  logits_sharded = _dot_f32(
      normed.reshape(bsz * s_out, c.dim), w["head"]
  ).reshape(bsz, s_out, lay.vocab_shard)
  vs = lay.vocab_shard
  v0 = my_id * jnp.int32(vs)
  v_iota = v0 + jnp.arange(vs, dtype=I32)
  logits_sharded = jnp.where(
      (v_iota < c.vocab_size)[None, None, :], logits_sharded, -jnp.inf
  )
  if temperature > 0.0 and rng_seed is not None:
    k_loc = max(1, min(int(top_k), vs))
    loc_vals, loc_idx = lax.top_k(logits_sharded, k_loc)
    loc_ids = v0 + loc_idx.astype(I32)
    if lay.num_devices > 1:
      loc_pkt = jnp.concatenate(
          [
              loc_vals.reshape(bsz, s_out * k_loc),
              loc_ids.astype(F32).reshape(bsz, s_out * k_loc),
          ],
          axis=-1,
      )
      all_pkt = _vmem_all_gather_axis0(loc_pkt, lay.num_devices)
      all_vals = (
          all_pkt[:, :, : s_out * k_loc]
          .reshape(lay.num_devices, bsz, s_out, k_loc)
          .transpose(1, 2, 0, 3)
          .reshape(bsz, s_out, lay.num_devices * k_loc)
      )
      all_ids = (
          all_pkt[:, :, s_out * k_loc :]
          .astype(I32)
          .reshape(lay.num_devices, bsz, s_out, k_loc)
          .transpose(1, 2, 0, 3)
          .reshape(bsz, s_out, lay.num_devices * k_loc)
      )
      g_vals, g_idx = lax.top_k(all_vals, k_loc)
      g_ids = jnp.take_along_axis(all_ids, g_idx, axis=-1)
    else:
      g_vals, g_ids = loc_vals, loc_ids
    scaled_vals = g_vals / jnp.float32(temperature)
    max_v = jnp.max(scaled_vals, axis=-1, keepdims=True)
    probs = jnp.exp(scaled_vals - max_v)
    probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), 1e-12)
    sorted_p, sort_idx = lax.top_k(probs, k_loc)
    cum_p = jnp.cumsum(sorted_p, axis=-1)
    keep_sorted = (cum_p - sorted_p) < jnp.float32(0.95)
    keep_sorted = keep_sorted.at[..., 0].set(True)
    inv_perm = jnp.argsort(sort_idx, axis=-1)
    keep_mask = jnp.take_along_axis(keep_sorted, inv_perm, axis=-1)
    masked_vals = jnp.where(keep_mask, scaled_vals, -jnp.inf)
    seed_b = jnp.broadcast_to(jnp.asarray(rng_seed, dtype=I32), (bsz,))
    pos_s = (
        pos_seq[:, -1:]
        if not full_logits
        else pos_seq[:, :s_out]
    )
    h_key = (
        jnp.uint32(seed_b[:, None, None]) * jnp.uint32(0x9E3779B9)
        ^ jnp.uint32(pos_s[:, :, None]) * jnp.uint32(0x85EBCA6B)
        ^ (
            jnp.arange(k_loc, dtype=jnp.uint32)[None, None, :]
            * jnp.uint32(0xC2B2AE35)
        )
    )
    h_key = (h_key ^ (h_key >> jnp.uint32(16))) * jnp.uint32(0x7FEB352D)
    h_key = (h_key ^ (h_key >> jnp.uint32(15))) * jnp.uint32(0x846CA68B)
    h_key = h_key ^ (h_key >> jnp.uint32(16))
    u = (
        (h_key >> jnp.uint32(8)).astype(F32) + jnp.float32(0.5)
    ) / jnp.float32(16777216.0)
    gumbel = -jnp.log(-jnp.log(jnp.clip(u, 1e-6, 1.0 - 1e-6)))
    chosen_k = jnp.argmax(masked_vals + gumbel, axis=-1)[..., None]
    gumbel_ids = jnp.take_along_axis(g_ids, chosen_k, axis=-1)[..., 0]
    if full_logits and draft_ids is not None:
      # Exact lossless speculative rejection + residual distribution sampling
      # for a deterministic drafter (Leviathan et al., 2023):
      #   1. Normalized target distribution after top-k + top-p=0.95 filtering:
      #      norm_p(x) = probs(x) * keep_mask(x) / sum_{x'} probs(x') * keep_mask(x')
      #   2. For position j in 0..s_out-2 with greedy draft token d_j = draft_ids[:, j]:
      #      Accept d_j iff u_acc < norm_p(d_j).
      #   3. On rejection at position j, sample the bonus token from the residual
      #      distribution proportional to max(0, norm_p(x) - 1[x == d_j]) =
      #      norm_p(x) * 1[x != d_j] (falling back to norm_p if norm_p(d_j) == 1).
      #   4. Position s_out - 1 (bonus token when all blk-1 draft tokens are accepted)
      #      samples directly from norm_p via gumbel_ids[:, -1:].
      kept_p = jnp.where(keep_mask, probs, jnp.float32(0.0))
      norm_p = kept_p / jnp.maximum(
          jnp.sum(kept_p, axis=-1, keepdims=True), jnp.float32(1e-12)
      )
      is_draft = g_ids[:, :-1, :] == draft_ids[:, :, None]
      p_draft = jnp.sum(
          jnp.where(is_draft, norm_p[:, :-1, :], jnp.float32(0.0)), axis=-1
      )
      h_acc = (
          jnp.uint32(seed_b[:, None]) * jnp.uint32(0xD1B54A35)
          ^ jnp.uint32(pos_s[:, :-1]) * jnp.uint32(0x94D049BB)
          ^ jnp.uint32(0x517CC1B7)
      )
      h_acc = (h_acc ^ (h_acc >> jnp.uint32(16))) * jnp.uint32(0x7FEB352D)
      h_acc = (h_acc ^ (h_acc >> jnp.uint32(15))) * jnp.uint32(0x846CA68B)
      h_acc = h_acc ^ (h_acc >> jnp.uint32(16))
      u_acc = (
          (h_acc >> jnp.uint32(8)).astype(F32) + jnp.float32(0.5)
      ) / jnp.float32(16777216.0)
      accepted = u_acc < p_draft

      res_vals = jnp.where(is_draft, -jnp.inf, masked_vals[:, :-1, :])
      has_alt = jnp.any(jnp.isfinite(res_vals), axis=-1, keepdims=True)
      res_vals = jnp.where(has_alt, res_vals, masked_vals[:, :-1, :])
      res_k = jnp.argmax(res_vals + gumbel[:, :-1, :], axis=-1)[..., None]
      res_ids = jnp.take_along_axis(g_ids[:, :-1, :], res_k, axis=-1)[..., 0]
      spec_pred = jnp.where(accepted, draft_ids, res_ids)
      output_ids = jnp.concatenate([spec_pred, gumbel_ids[:, -1:]], axis=1)
    else:
      output_ids = gumbel_ids
  else:
    local_max = jnp.max(logits_sharded, axis=-1)
    local_arg = v0 + jnp.argmax(logits_sharded, axis=-1).astype(I32)
    if lay.num_devices > 1:
      loc_both = jnp.concatenate([local_max, local_arg.astype(F32)], axis=-1)
      all_both = _vmem_all_gather_axis0(loc_both, lay.num_devices)
      all_max = all_both[..., :s_out]
      all_arg = all_both[..., s_out:].astype(I32)
      best_rank = jnp.argmax(all_max, axis=0)
      output_ids = jnp.take_along_axis(all_arg, best_rank[None, ...], axis=0)[0]
    else:
      output_ids = local_arg
  if not full_logits:
    output_ids = output_ids[:, 0]

  main_hidden = (
      jnp.concatenate(main_hiddens, axis=-1)
      if main_hiddens
      else jnp.zeros((bsz, seqlen, 0), dtype=BF16)
  )
  new_caches = dataclasses.replace(caches, window_kv=win_kv_all)
  return output_ids, logits_sharded, main_hidden, new_caches


def dspark_macrokernel(
    anchor_ids: jax.Array,
    main_hidden: jax.Array,
    start_pos: jax.Array,
    w: Mapping[str, jax.Array],
    caches: DeviceDecodeCaches,
    cos_swa: jax.Array,
    sin_swa: jax.Array,
    *,
    lay: layout_lib.DeviceLayout,
    seed_only: bool = False,
    skip_main_kv_seed: bool = False,
    interpret: bool = False,
    simulate_full_43_blocks: bool = False,
    valid_len: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array, DeviceDecodeCaches]:
  """Macro-Kernel 3: 3-Layer DSpark Drafter (mtp.0..2) + Markov & Confidence Heads.

  Returns:
    (draft_output_ids [B, block_size + 1], confidence [B, block_size],
    updated_caches).
  """
  c = lay.cfg
  bsz, seqlen, _ = main_hidden.shape
  win = c.window_size
  blk = c.dspark_block_size
  my_id = (
      lax.axis_index(layout_lib.AXIS) if lay.num_devices > 1 else jnp.int32(0)
  )

  _, safe_pos_b, active_b, pos_main = _pos_and_active(start_pos, bsz, seqlen)
  mtp_win_all = caches.mtp_window_kv

  if not skip_main_kv_seed:
    main_x = _rms_norm_jax(
        fp8_matmul_col_sharded(
            main_hidden.reshape(bsz * seqlen, -1),
            w["mtp_main_proj_w"],
            w["mtp_main_proj_s"],
            n_dev=lay.num_devices,
        )
        .astype(BF16)
        .reshape(bsz, seqlen, c.dim),
        w["mtp_main_norm"],
        c.norm_eps,
    )
    main_x_q = act_quant_fp8_bf16(main_x.reshape(bsz * seqlen, c.dim))

    # Update 3-layer DSpark SWA caches with main_x
    for s_id in range(c.n_mtp_layers):
      b_idx = c.n_layers + s_id
      main_kv = _rms_norm_jax(
          fp8_matmul_col_sharded(
              main_x_q,
              w["wkv_w"],
              w["wkv_s"],
              n_dev=lay.num_devices,
              layer_idx=b_idx,
              pre_quantized=True,
          )
          .astype(BF16)
          .reshape(bsz, seqlen, c.head_dim),
          w["kv_norm"][b_idx],
          c.norm_eps,
      )
      main_kv = _apply_rotary_jax(
          main_kv,
          cos_swa[pos_main],
          sin_swa[pos_main],
          rope_dim=c.rope_head_dim,
      )
      main_kv = act_quant_fp8_bf16(
          main_kv.reshape(bsz * seqlen, c.head_dim)
      ).reshape(bsz, seqlen, c.head_dim)

      cur_win = _update_swa_ring_buffer(
          mtp_win_all[:, s_id],
          main_kv,
          safe_pos_b,
          active_b,
          win,
          valid_len=valid_len,
      )
      mtp_win_all = mtp_win_all.at[:, s_id].set(cur_win)

  new_caches = dataclasses.replace(caches, mtp_window_kv=mtp_win_all)
  if seed_only:
    return (
        jnp.zeros((bsz, blk + 1), dtype=I32),
        jnp.zeros((bsz, blk), dtype=F32),
        new_caches,
    )

  # Look up only [anchor_ids, noise_token_id] (B + 1 tokens instead of B * blk)
  # and fuse with mtp_markov_embed into a single 2D BF16 lax.psum [B + 1, 5376].
  compact_ids = jnp.concatenate(
      [anchor_ids, jnp.full((1,), c.dspark_noise_token_id, dtype=I32)], axis=0
  )
  vs = lay.vocab_shard
  v0 = my_id * jnp.int32(vs)
  v1 = jnp.minimum(v0 + jnp.int32(vs), jnp.int32(c.vocab_size))
  in_v = (compact_ids >= v0) & (compact_ids < v1)
  loc_tok = jnp.where(in_v, compact_ids - v0, 0)
  emb_c = jnp.where(in_v[:, None], w["embed"][loc_tok].astype(BF16), BF16(0.0))
  m_c = jnp.where(
      in_v[:, None], w["mtp_markov_embed"][loc_tok].astype(BF16), BF16(0.0)
  )
  both_c = jnp.concatenate([emb_c, m_c], axis=-1)
  if lay.num_devices > 1:
    both_c = lax.psum(both_c, layout_lib.AXIS)
  emb_anchor = both_c[:bsz, : c.dim][:, None, :]
  emb_noise = jnp.broadcast_to(
      both_c[bsz : bsz + 1, : c.dim][:, None, :], (bsz, blk - 1, c.dim)
  )
  emb_bf16 = jnp.concatenate([emb_anchor, emb_noise], axis=1)
  m_emb = both_c[:bsz, c.dim :].astype(F32)
  h = jnp.repeat(emb_bf16[:, :, None, :], c.hc_mult, axis=2)
  pre_mix = jnp.zeros((bsz, blk, c.hc_mult), dtype=F32).at[:, :, 0].set(1.0)

  eff_main_len = (
      jnp.int32(seqlen)
      if valid_len is None
      else jnp.clip(
          jnp.broadcast_to(jnp.asarray(valid_len, dtype=I32), (bsz,)),
          1,
          seqlen,
      )
  )
  draft_start = safe_pos_b + eff_main_len
  pos_draft = draft_start[:, None] + jnp.arange(blk, dtype=I32)[None, :]
  last_ctx_pos = (safe_pos_b + eff_main_len - 1)[:, None, None]

  win_valid = jnp.arange(win, dtype=I32)[None, None, :] <= last_ctx_pos
  blk_valid = jnp.ones((bsz, blk, blk), dtype=jnp.bool_)
  valid_cat = jnp.concatenate(
      [jnp.broadcast_to(win_valid, (bsz, blk, win)), blk_valid], axis=2
  )

  for s_id in range(c.n_mtp_layers):
    b_idx = c.n_layers + s_id

    def _mtp_step(_, st_m, b_cur=b_idx, s_cur=s_id):
      h_c, pm_c = st_m
      res_c = h_c
      apre, apost, acomb = mega_mhc_mixes_jax(
          h_c,
          w["hc_attn_fn"][b_cur],
          w["hc_attn_scale"][b_cur],
          w["hc_attn_base"][b_cur],
          c.norm_eps,
          c.hc_mult,
          c.hc_sinkhorn_iters,
          c.hc_eps,
      )
      x_c = mega_mhc_pre_norm_jax(h_c, pm_c, w["attn_norm"][b_cur], c.norm_eps)
      xq_c = act_quant_fp8_bf16(x_c.reshape(bsz * blk, c.dim))
      qr_c = _rms_norm_jax(
          fp8_matmul_col_sharded(
              xq_c,
              w["wq_a_w"],
              w["wq_a_s"],
              n_dev=lay.num_devices,
              layer_idx=b_cur,
              pre_quantized=True,
          ).astype(BF16),
          w["q_norm"][b_cur],
          c.norm_eps,
      )
      q_c = (
          fp8_matmul(qr_c, w["wq_b_w"], w["wq_b_s"], layer_idx=b_cur)
          .astype(BF16)
          .reshape(bsz, blk, lay.heads_per_device, c.head_dim)
      )
      q_c = _apply_rotary_jax(
          q_c, cos_swa[pos_draft], sin_swa[pos_draft], rope_dim=c.rope_head_dim
      )
      kvd_c = _rms_norm_jax(
          fp8_matmul_col_sharded(
              xq_c,
              w["wkv_w"],
              w["wkv_s"],
              n_dev=lay.num_devices,
              layer_idx=b_cur,
              pre_quantized=True,
          )
          .astype(BF16)
          .reshape(bsz, blk, c.head_dim),
          w["kv_norm"][b_cur],
          c.norm_eps,
      )
      kvd_c = _apply_rotary_jax(
          kvd_c,
          cos_swa[pos_draft],
          sin_swa[pos_draft],
          rope_dim=c.rope_head_dim,
      )
      kvd_c = act_quant_fp8_bf16(kvd_c.reshape(bsz * blk, c.head_dim)).reshape(
          bsz, blk, c.head_dim
      )
      ckv_c = jnp.concatenate([mtp_win_all[:, s_cur], kvd_c], axis=1)
      o_c = _sparse_attn_with_sink_jax(
          q_c, ckv_c, valid_cat, w["attn_sink"][b_cur], c.softmax_scale
      )
      o_c = _apply_rotary_jax(
          o_c,
          cos_swa[pos_draft],
          sin_swa[pos_draft],
          rope_dim=c.rope_head_dim,
          inverse=True,
      )
      og_c = o_c.reshape(
          bsz * blk, lay.o_groups_per_device, lay.group_head_cols
      )
      op_c = jnp.concatenate(
          [
              fp8_matmul(
                  og_c[:, g, :],
                  w["wo_a_w"],
                  w["wo_a_s"],
                  layer_idx=b_cur,
                  group_idx=g,
                  out_dtype=BF16,
              )
              for g in range(lay.o_groups_per_device)
          ],
          axis=1,
      )
      aout_c = fp8_matmul(
          op_c, w["wo_b_w"], w["wo_b_s"], layer_idx=b_cur, out_dtype=F32
      )
      if lay.num_devices > lay.dense_tp:
        aout_c = jnp.where(my_id < jnp.int32(lay.dense_tp), aout_c, 0.0)
      if lay.num_devices > 1:
        aout_c = lax.psum(aout_c, layout_lib.AXIS)
      h_c = mega_mhc_post_jax(
          aout_c.astype(BF16).reshape(bsz, blk, c.dim),
          res_c,
          apost,
          acomb,
      )
      res_f = h_c
      fpre, fpost, fcomb = mega_mhc_mixes_jax(
          h_c,
          w["hc_ffn_fn"][b_cur],
          w["hc_ffn_scale"][b_cur],
          w["hc_ffn_base"][b_cur],
          c.norm_eps,
          c.hc_mult,
          c.hc_sinkhorn_iters,
          c.hc_eps,
      )
      xffn_c = mega_mhc_pre_norm_jax(
          h_c, apre, w["ffn_norm"][b_cur], c.norm_eps
      )
      mout_c = _moe_layer_sharded_jax(
          xffn_c,
          w["mtp_gate_w"][s_cur],
          w["mtp_gate_bias"][s_cur],
          w["shared_gu_w"],
          w["shared_gu_s"],
          w["shared_dn_w"],
          w["shared_dn_s"],
          w["mtp_exp_gu_w"],
          w["mtp_exp_gu_s"],
          w["mtp_exp_dn_w"],
          w["mtp_exp_dn_s"],
          layer_idx=s_cur,
          shared_layer_idx=b_cur,
          active_b=active_b,
          lay=lay,
          is_mtp=True,
          interpret=interpret,
      )
      h_c = mega_mhc_post_jax(mout_c, res_f, fpost, fcomb)
      return (h_c, fpre)

    if (
        (
            not isinstance(simulate_full_43_blocks, bool)
            or simulate_full_43_blocks
        )
        and c.n_mtp_layers == 1
        and s_id == 0
    ):
      n_mtp_rep = jnp.where(simulate_full_43_blocks, jnp.int32(3), jnp.int32(1))
      h, pre_mix = lax.fori_loop(
          jnp.int32(0), n_mtp_rep, _mtp_step, (h, pre_mix)
      )
    else:
      h, pre_mix = _mtp_step(0, (h, pre_mix))

  x_collapsed = jnp.sum(
      pre_mix.astype(F32)[..., None] * h.astype(F32), axis=2
  ).astype(BF16)
  normed = _rms_norm_jax(x_collapsed, w["mtp_final_norm"], c.norm_eps)
  base_logits_sharded = _dot_f32(
      normed.reshape(bsz * blk, c.dim), w["head"]
  ).reshape(bsz, blk, vs)
  v_iota = v0 + jnp.arange(vs, dtype=I32)

  output_ids = jnp.zeros((bsz, blk + 1), dtype=I32).at[:, 0].set(anchor_ids)
  markov_embeds = []

  for i in range(blk):
    markov_embeds.append(m_emb)
    bias_sharded = _dot_f32(m_emb, w["mtp_markov_head"])
    step_logits = jnp.where(
        (v_iota < c.vocab_size)[None, :],
        base_logits_sharded[:, i, :] + bias_sharded,
        -jnp.inf,
    )
    loc_max = jnp.max(step_logits, axis=-1)
    loc_arg_off = jnp.argmax(step_logits, axis=-1).astype(I32)
    loc_arg = v0 + loc_arg_off
    if lay.num_devices > 1:
      if i + 1 < blk:
        cand_m = jnp.where(
            (loc_arg < c.vocab_size)[:, None],
            w["mtp_markov_embed"][loc_arg_off].astype(F32),
            0.0,
        )
        pkt = jnp.concatenate(
            [loc_max[:, None], loc_arg[:, None].astype(F32), cand_m],
            axis=-1,
        )
        all_pkt = _vmem_all_gather_axis0(pkt, lay.num_devices)
        best_r = jnp.argmax(all_pkt[:, :, 0], axis=0)
        chosen = jnp.take_along_axis(all_pkt, best_r[None, :, None], axis=0)[0]
        next_tok = chosen[:, 1].astype(I32)
        m_emb = chosen[:, 2:]
      else:
        pkt = jnp.stack([loc_max, loc_arg.astype(F32)], axis=-1)
        all_pkt = _vmem_all_gather_axis0(pkt, lay.num_devices)
        best_r = jnp.argmax(all_pkt[:, :, 0], axis=0)
        next_tok = jnp.take_along_axis(
            all_pkt[:, :, 1], best_r[None, ...], axis=0
        )[0].astype(I32)
    else:
      next_tok = loc_arg
      if i + 1 < blk:
        in_m = (next_tok >= v0) & (next_tok < v1)
        loc_m = jnp.where(in_m, next_tok - v0, 0)
        m_emb = jnp.where(
            in_m[:, None], w["mtp_markov_embed"][loc_m].astype(F32), 0.0
        )
    output_ids = output_ids.at[:, i + 1].set(next_tok)

  markov_stack = jnp.stack(markov_embeds, axis=1)
  conf_in = jnp.concatenate([x_collapsed.astype(F32), markov_stack], axis=-1)
  confidence = lax.dot_general(
      conf_in,
      w["mtp_confidence_w"].astype(F32),
      (((2,), (0,)), ((), ())),
      preferred_element_type=F32,
  ).squeeze(-1)
  return output_ids, confidence, new_caches


def ced_prefill_macrokernel(
    input_ids: jax.Array,
    w: Mapping[str, jax.Array],
    caches: DeviceDecodeCaches,
    cos_swa: jax.Array,
    sin_swa: jax.Array,
    cos_comp: jax.Array,
    sin_comp: jax.Array,
    engram_hashes: jax.Array | None,
    *,
    lay: layout_lib.DeviceLayout,
    interpret: bool = False,
    simulate_full_43_blocks: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, DeviceDecodeCaches]:
  """Two-Phase Causal Encoder-Decoder (CED) + Decoder SWA Bounded Replay Prefill.

  Phase 1: Runs encoder_macrokernel across the full prompt, writing Encoder
           Global KV (L2, 8, 14) and Decoder L20 Global KV (comp_m1_kv,
           comp_m1_idx_k).
  Phase 2: Slices the final W = window_size tokens of h_enc and pre_mix_enc and
           runs decoder_macrokernel ONLY on those last W tokens!
  """
  c = lay.cfg
  _, seqlen = input_ids.shape
  win = c.window_size
  h_enc, pre_mix_enc, caches = encoder_macrokernel(
      input_ids,
      jnp.int32(0),
      w,
      caches,
      cos_swa,
      sin_swa,
      cos_comp,
      sin_comp,
      engram_hashes,
      lay=lay,
      write_global_kv=True,
      interpret=interpret,
      max_total_seq=seqlen,
      simulate_full_43_blocks=simulate_full_43_blocks,
  )
  if seqlen <= win:
    return decoder_macrokernel(
        h_enc,
        pre_mix_enc,
        jnp.int32(0),
        w,
        caches,
        cos_comp,
        sin_comp,
        lay=lay,
        full_logits=False,
        interpret=interpret,
        max_total_seq=seqlen,
        simulate_full_43_blocks=simulate_full_43_blocks,
    )

  prefix_len = seqlen - win
  return decoder_macrokernel(
      h_enc[:, prefix_len:],
      pre_mix_enc[:, prefix_len:],
      jnp.int32(prefix_len),
      w,
      caches,
      cos_comp,
      sin_comp,
      lay=lay,
      full_logits=False,
      interpret=interpret,
      max_total_seq=seqlen,
      simulate_full_43_blocks=simulate_full_43_blocks,
  )


def speculative_verify_and_draft_step(
    candidate_ids: jax.Array,
    start_pos: jax.Array,
    w: Mapping[str, jax.Array],
    caches: DeviceDecodeCaches,
    cos_swa: jax.Array,
    sin_swa: jax.Array,
    cos_comp: jax.Array,
    sin_comp: jax.Array,
    engram_hashes: jax.Array | None,
    *,
    lay: layout_lib.DeviceLayout,
    interpret: bool = False,
    simulate_full_43_blocks: bool = False,
    temperature: float = 0.0,
    top_k: int = 20,
    rng_seed: jax.Array | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    DeviceDecodeCaches,
]:
  """Executes one complete speculative verification + DSpark drafting round.

  Args:
    candidate_ids: [B, blk] int32 ([anchor_token, draft_1, ..., draft_{blk-1}]).
    start_pos: scalar int32 position of anchor_token.
    w: per-device stacked weights.
    caches: DeviceDecodeCaches BEFORE running candidate_ids.

  Returns:
    (emitted_tokens [B, blk], emit_count [B], next_candidate_ids [B, blk],
     confidence [B, blk], updated_caches_at_accepted_pos).
  """
  c = lay.cfg
  bsz, blk = candidate_ids.shape
  _, safe_pos_b, active_b, pos_blk = _pos_and_active(start_pos, bsz, blk)
  win = c.window_size
  r_iota = jnp.arange(win, dtype=I32)[None, :]
  win_before = caches.window_kv
  mtp_win_before = caches.mtp_window_kv

  # 1. Verify all blk candidate tokens in parallel through Encoder + Decoder
  h_enc, pre_mix_enc, caches_all, m2_kv_traces, m2_sc_traces = (
      encoder_macrokernel(
          candidate_ids,
          start_pos,
          w,
          caches,
          cos_swa,
          sin_swa,
          cos_comp,
          sin_comp,
          engram_hashes,
          lay=lay,
          write_global_kv=True,
          interpret=interpret,
          return_m2_traces=True,
          simulate_full_43_blocks=simulate_full_43_blocks,
      )
  )
  target_pred, _, main_hidden_all, caches_all = decoder_macrokernel(
      h_enc,
      pre_mix_enc,
      start_pos,
      w,
      caches_all,
      cos_comp,
      sin_comp,
      lay=lay,
      full_logits=True,
      interpret=interpret,
      simulate_full_43_blocks=simulate_full_43_blocks,
      temperature=temperature,
      top_k=top_k,
      rng_seed=rng_seed,
      draft_ids=candidate_ids[:, 1:],
  )

  # 2. Longest accepted prefix match: draft j matches target_pred[:, j].
  matches = candidate_ids[:, 1:] == target_pred[:, :-1]  # [B, blk - 1]
  prefix_valid = jnp.cumprod(matches.astype(I32), axis=1)
  accept_count = jnp.where(active_b, jnp.sum(prefix_valid, axis=1), 0)
  emit_count = jnp.where(active_b, accept_count + 1, 0)

  # Bonus token is target_pred[b, accept_count[b]], anchor of the next step.
  bonus_token = jnp.take_along_axis(target_pred, accept_count[:, None], axis=1)[
      :, 0
  ]

  # 3. Rewind SWA ring buffer slots for rejected positions t >= emit_count
  # (Global KV comp_m1_kv and comp_m2_kv at >= start_pos + emit_count are
  # causally masked and overwritten on the next step, so zero O(max_seq) copy!)
  win_kv_clean = caches_all.window_kv
  for t in range(blk):
    slot_w = (safe_pos_b + t) % win
    reject_t = (t >= emit_count) | (~active_b)
    w_mask = ((r_iota == slot_w[:, None]) & reject_t[:, None])[:, None, :, None]
    win_kv_clean = jnp.where(w_mask, win_before, win_kv_clean)

  # Update DSpark SWA cache with the accepted tokens' main_hidden features
  # (including t == accept_count, so dspark_macrokernel skips recomputing it).
  main_x_all = _rms_norm_jax(
      fp8_matmul_col_sharded(
          main_hidden_all.reshape(bsz * blk, -1),
          w["mtp_main_proj_w"],
          w["mtp_main_proj_s"],
          n_dev=lay.num_devices,
      )
      .astype(BF16)
      .reshape(bsz, blk, c.dim),
      w["mtp_main_norm"],
      c.norm_eps,
  )
  main_x_all_q = act_quant_fp8_bf16(main_x_all.reshape(bsz * blk, c.dim))
  mtp_win_all = mtp_win_before
  for s_id in range(c.n_mtp_layers):
    b_idx = c.n_layers + s_id
    main_kv = _rms_norm_jax(
        fp8_matmul_col_sharded(
            main_x_all_q,
            w["wkv_w"],
            w["wkv_s"],
            n_dev=lay.num_devices,
            layer_idx=b_idx,
            pre_quantized=True,
        )
        .astype(BF16)
        .reshape(bsz, blk, c.head_dim),
        w["kv_norm"][b_idx],
        c.norm_eps,
    )
    main_kv = _apply_rotary_jax(
        main_kv,
        cos_swa[pos_blk],
        sin_swa[pos_blk],
        rope_dim=c.rope_head_dim,
    )
    main_kv = act_quant_fp8_bf16(
        main_kv.reshape(bsz * blk, c.head_dim)
    ).reshape(bsz, blk, c.head_dim)
    cur_win = mtp_win_all[:, s_id]
    for t in range(blk):
      slot_w = (safe_pos_b + t) % win
      active_t = (t <= accept_count) & active_b
      w_mask = ((r_iota == slot_w[:, None]) & active_t[:, None])[:, :, None]
      cur_win = jnp.where(w_mask, main_kv[:, t : t + 1, :], cur_win)
    mtp_win_all = mtp_win_all.at[:, s_id].set(cur_win)

  comp_m2_kv_st_clean = caches_all.comp_m2_kv_state
  comp_m2_sc_st_clean = caches_all.comp_m2_score_state
  for slot_idx, (kv_tr, sc_tr) in enumerate(zip(m2_kv_traces, m2_sc_traces)):
    sel_kv = jnp.take_along_axis(
        kv_tr, accept_count[:, None, None, None], axis=1
    )[:, 0]
    sel_sc = jnp.take_along_axis(
        sc_tr, accept_count[:, None, None, None], axis=1
    )[:, 0]
    comp_m2_kv_st_clean = comp_m2_kv_st_clean.at[:, slot_idx].set(sel_kv)
    comp_m2_sc_st_clean = comp_m2_sc_st_clean.at[:, slot_idx].set(sel_sc)

  caches_accepted = dataclasses.replace(
      caches_all,
      window_kv=win_kv_clean,
      comp_m2_kv_state=comp_m2_kv_st_clean,
      comp_m2_score_state=comp_m2_sc_st_clean,
      mtp_window_kv=mtp_win_all,
  )

  # Run DSpark drafter from the last accepted token's position.
  last_accepted_idx = accept_count[:, None, None]
  last_main_hidden = jnp.take_along_axis(
      main_hidden_all, last_accepted_idx, axis=1
  )
  draft_pos = jnp.where(active_b, safe_pos_b + accept_count, -1)
  draft_out_ids, confidence, caches_after_draft = dspark_macrokernel(
      bonus_token,
      last_main_hidden,
      draft_pos,
      w,
      caches_accepted,
      cos_swa,
      sin_swa,
      lay=lay,
      seed_only=False,
      skip_main_kv_seed=True,
      interpret=interpret,
      simulate_full_43_blocks=simulate_full_43_blocks,
  )
  # draft_out_ids is [B, dspark_block_size + 1] = [bonus_token, draft_1, ...];
  # slice to match input candidate_ids width blk.
  next_candidate_ids = draft_out_ids[:, :blk]
  return (
      target_pred,
      emit_count,
      next_candidate_ids,
      confidence,
      caches_after_draft,
  )


def ced_chunked_prefill(
    input_ids: jax.Array,
    w: Mapping[str, jax.Array],
    caches: DeviceDecodeCaches,
    cos_swa: jax.Array,
    sin_swa: jax.Array,
    cos_comp: jax.Array,
    sin_comp: jax.Array,
    engram_hashes: jax.Array | None,
    *,
    lay: layout_lib.DeviceLayout,
    chunk_size: int = 128,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, DeviceDecodeCaches]:
  """Chunked Two-Phase CED Prefill: runs Encoder in chunks, Decoder only on final W."""
  c = lay.cfg
  _, seqlen = input_ids.shape
  win = c.window_size
  if seqlen <= chunk_size:
    return ced_prefill_macrokernel(
        input_ids,
        w,
        caches,
        cos_swa,
        sin_swa,
        cos_comp,
        sin_comp,
        engram_hashes,
        lay=lay,
        interpret=interpret,
    )
  if seqlen % chunk_size != 0 or chunk_size < win:
    raise ValueError(
        f"seqlen ({seqlen}) must be a multiple of chunk_size ({chunk_size}) >="
        f" win ({win})"
    )
  num_chunks = seqlen // chunk_size
  ids_chunks = input_ids.reshape(input_ids.shape[0], num_chunks, chunk_size)
  eh_chunks = (
      engram_hashes.reshape(
          engram_hashes.shape[0],
          num_chunks,
          chunk_size,
          *engram_hashes.shape[2:],
      )
      if engram_hashes is not None
      else None
  )

  # Run Encoder over prefix chunks 0 .. num_chunks - 2 via lax.fori_loop
  def _enc_chunk_body(c_idx: int, cur_caches: DeviceDecodeCaches):
    c_start = c_idx * jnp.int32(chunk_size)
    c_ids = ids_chunks[:, c_idx]
    c_eh = eh_chunks[:, c_idx] if eh_chunks is not None else None
    _, _, next_caches = encoder_macrokernel(
        c_ids,
        c_start,
        w,
        cur_caches,
        cos_swa,
        sin_swa,
        cos_comp,
        sin_comp,
        c_eh,
        lay=lay,
        write_global_kv=True,
        interpret=interpret,
    )
    return next_caches

  caches = lax.fori_loop(0, num_chunks - 1, _enc_chunk_body, caches)

  # Final chunk: run Encoder and slice the last W = win tokens for Decoder!
  last_start = jnp.int32((num_chunks - 1) * chunk_size)
  last_eh = eh_chunks[:, num_chunks - 1] if eh_chunks is not None else None
  h_enc_last, pm_enc_last, caches = encoder_macrokernel(
      ids_chunks[:, num_chunks - 1],
      last_start,
      w,
      caches,
      cos_swa,
      sin_swa,
      cos_comp,
      sin_comp,
      last_eh,
      lay=lay,
      write_global_kv=True,
      interpret=interpret,
  )
  tail_offset = chunk_size - win
  dec_start_pos = jnp.int32(seqlen - win)
  return decoder_macrokernel(
      h_enc_last[:, tail_offset:],
      pm_enc_last[:, tail_offset:],
      dec_start_pos,
      w,
      caches,
      cos_comp,
      sin_comp,
      lay=lay,
      full_logits=False,
      interpret=interpret,
  )
