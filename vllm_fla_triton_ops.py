"""Triton kernels for Qwen3.5 / GatedDeltaNet (flash_linear_attention ops) on H100.

Provides all Triton kernels required by //third_party/py/vllm's
QwenGatedDeltaNetAttention on CUDA without requiring external flash_linear_attention
packages or removed Triton 3.4+ tl.make_block_ptr APIs:
  - chunk_gated_delta_rule (register-resident Triton prefill kernel with [N, HV, V, K] state)
  - fused_post_conv_prep
  - fused_sigmoid_gating_delta_rule_update
  - fused_recurrent_gated_delta_rule_packed_decode
  - l2norm_fwd
  - rmsnorm_fn / layernorm_fn
  - prepare_lens / prepare_chunk_indices / prepare_chunk_offsets
  - FLA_CHUNK_SIZE = 64
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

import einops
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

FLA_CHUNK_SIZE = 64


# ============================================================================
# Utilities & Index Helpers
# ============================================================================
def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
  """Simple LRU cache decorator for tensor metadata helpers."""
  cache_entries: list[tuple[tuple[Any, ...], dict[str, Any], torch.Tensor]] = []

  @functools.wraps(fn)
  def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
    for c_args, c_kwargs, res in cache_entries:
      if len(args) == len(c_args) and kwargs.keys() == c_kwargs.keys():
        if all(
            (a is ca if isinstance(a, torch.Tensor) else a == ca)
            for a, ca in zip(args, c_args)
        ) and all(
            (
                kwargs[k] is c_kwargs[k]
                if isinstance(kwargs[k], torch.Tensor)
                else kwargs[k] == c_kwargs[k]
            )
            for k in kwargs
        ):
          return res
    res = fn(*args, **kwargs)
    if len(cache_entries) >= 8:
      cache_entries.pop(0)
    cache_entries.append((args, kwargs, res))
    return res

  return wrapper


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Ensures tensor inputs are contiguous and sets CUDA device context."""

  @functools.wraps(fn)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    contiguous_args = tuple(
        i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args
    )
    contiguous_kwargs = {
        k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
        for k, v in kwargs.items()
    }
    tensor = None
    for arg in args:
      if isinstance(arg, torch.Tensor):
        tensor = arg
        break
    if tensor is None:
      for v in kwargs.values():
        if isinstance(v, torch.Tensor):
          tensor = v
          break
    if tensor is not None and tensor.is_cuda:
      with torch.cuda.device(tensor.device.index or 0):
        return fn(*contiguous_args, **contiguous_kwargs)
    return fn(*contiguous_args, **contiguous_kwargs)

  return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
  return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
  indices = torch.cat([
      torch.arange(n, device=cu_seqlens.device)
      for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
  ])
  return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
  return torch.cat(
      [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
  ).cumsum(-1)


# ============================================================================
# L2 Norm
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
  i_t = tl.program_id(0)
  x += i_t * D
  y += i_t * D
  cols = tl.arange(0, BD)
  mask = cols < D
  b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
  b_var = tl.sum(b_x * b_x, axis=0)
  b_rstd = 1 / tl.sqrt(b_var + eps)
  b_y = b_x * b_rstd
  tl.store(y + cols, b_y, mask=mask)


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None
) -> torch.Tensor:
  x_shape_og = x.shape
  x = x.view(-1, x.shape[-1]).contiguous()
  if output_dtype is None:
    y = torch.empty_like(x)
  else:
    y = torch.empty_like(x, dtype=output_dtype)
  T, D = x.shape[0], x.shape[-1]
  BD = triton.next_power_of_2(D)
  l2norm_fwd_kernel1[(T,)](
      x,
      y,
      D=D,
      BD=BD,
      eps=eps,
  )
  return y.view(x_shape_og)


# ============================================================================
# Gated Delta Rule Prefill Forward (Register-Resident Triton Kernel)
# ============================================================================
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _gated_delta_rule_prefill_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
  i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
  i_n, i_hv = i_nh // HV, i_nh % HV
  i_h = i_hv // (HV // H)

  if IS_VARLEN:
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    all_tok = T
    seq_len = eos - bos
  else:
    bos = i_n * T
    eos = bos + T
    all_tok = B * T
    seq_len = T

  if seq_len <= 0:
    return

  o_k = i_k * BK + tl.arange(0, BK)
  o_v = i_v * BV + tl.arange(0, BV)
  mask_k = o_k < K
  mask_v = o_v < V
  mask_h = mask_v[:, None] & mask_k[None, :]

  b_h = tl.zeros([BV, BK], dtype=tl.float32)
  if USE_INITIAL_STATE:
    p_h0 = h0 + (i_n * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

  p_q = q + (bos * H + i_h) * K + o_k
  p_k = k + (bos * H + i_h) * K + o_k
  p_v = v + (bos * HV + i_hv) * V + o_v
  p_g = g + bos * HV + i_hv
  p_beta = beta + bos * HV + i_hv
  p_o = o + ((i_k * all_tok + bos) * HV + i_hv) * V + o_v

  for _ in range(0, seq_len):
    b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
    b_g = tl.load(p_g).to(tl.float32)
    b_beta = tl.load(p_beta).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
      b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
      b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_q += H * K
    p_k += H * K
    p_v += HV * V
    p_g += HV
    p_beta += HV
    p_o += HV * V

  if STORE_FINAL_STATE:
    p_ht = ht + (i_n * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
  """Executes GatedDeltaNet prefill on GPU using register-resident Triton kernel."""
  del chunk_indices, chunk_offsets
  if head_first:
    q, k, v, beta, g = map(
        lambda x: einops.rearrange(x, "b h t ... -> b t h ..."),
        (q, k, v, beta, g),
    )
  if scale is None:
    scale = k.shape[-1] ** -0.5

  q_c = q.contiguous()
  k_c = k.contiguous()
  v_c = v.contiguous()
  g_c = g.contiguous()
  beta_c = beta.contiguous()

  B, T, H, K = k_c.shape
  HV, V = v_c.shape[-2], v_c.shape[-1]
  N = B if cu_seqlens is None else len(cu_seqlens) - 1

  BK = triton.next_power_of_2(K)
  BV = min(triton.next_power_of_2(V), 32)
  NK = triton.cdiv(K, BK)
  NV = triton.cdiv(V, BV)

  o = q_c.new_empty(NK, *v_c.shape)
  h0 = initial_state.contiguous() if initial_state is not None else None
  ht = (
      q_c.new_empty(N, HV, V, K, dtype=torch.float32)
      if output_final_state
      else None
  )

  grid = (NK, NV, N * HV)
  _gated_delta_rule_prefill_fwd_kernel[grid](
      q=q_c,
      k=k_c,
      v=v_c,
      g=g_c,
      beta=beta_c,
      o=o,
      h0=h0,
      ht=ht,
      cu_seqlens=cu_seqlens,
      scale=scale,
      T=T,
      B=B,
      H=H,
      HV=HV,
      K=K,
      V=V,
      BK=BK,
      BV=BV,
      USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
      num_warps=4,
      num_stages=3,
  )
  o_out = o.squeeze(0) if NK == 1 else o.sum(0).to(v_c.dtype)
  if head_first:
    o_out = einops.rearrange(o_out, "b t h ... -> b h t ...")
  if core_attn_out is not None:
    o_flat = o_out.squeeze(0).reshape(-1)
    co_flat = core_attn_out.reshape(-1)
    co_flat[: o_flat.numel()].copy_(o_flat)
  return o_out, ht


# ============================================================================
# Fused Post-Conv Prep
# ============================================================================
@triton.jit
def _fused_post_conv_prep_kernel(
    conv_output_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    g_out_ptr,
    beta_out_ptr,
    stride_conv_tok,
    stride_a_tok,
    stride_b_tok,
    stride_q_tok,
    stride_q_head,
    stride_k_tok,
    stride_k_head,
    stride_v_tok,
    stride_v_head,
    stride_g_tok,
    stride_beta_tok,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BVH: tl.constexpr,
    KEY_DIM: tl.constexpr,
    EPS: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
):
  tok_idx = tl.program_id(0)
  conv_base = conv_output_ptr + tok_idx * stride_conv_tok
  k_dim_offs = tl.arange(0, BK)
  k_mask = k_dim_offs < HEAD_K_DIM

  for kh in tl.static_range(NUM_K_HEADS):
    q_vals = tl.load(
        conv_base + kh * HEAD_K_DIM + k_dim_offs, mask=k_mask, other=0.0
    ).to(tl.float32)
    if APPLY_L2NORM:
      q_inv_norm = tl.rsqrt(tl.sum(q_vals * q_vals, axis=0) + EPS)
      q_vals = q_vals * q_inv_norm
    q_dst = q_out_ptr + tok_idx * stride_q_tok + kh * stride_q_head + k_dim_offs
    tl.store(q_dst, q_vals.to(q_out_ptr.dtype.element_ty), mask=k_mask)

    k_vals = tl.load(
        conv_base + KEY_DIM + kh * HEAD_K_DIM + k_dim_offs,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    if APPLY_L2NORM:
      k_inv_norm = tl.rsqrt(tl.sum(k_vals * k_vals, axis=0) + EPS)
      k_vals = k_vals * k_inv_norm
    k_dst = k_out_ptr + tok_idx * stride_k_tok + kh * stride_k_head + k_dim_offs
    tl.store(k_dst, k_vals.to(k_out_ptr.dtype.element_ty), mask=k_mask)

  v_dim_offs = tl.arange(0, BV)
  v_mask = v_dim_offs < HEAD_V_DIM
  v_conv_base = conv_base + 2 * KEY_DIM
  for vh in tl.static_range(NUM_V_HEADS):
    v_vals = tl.load(
        v_conv_base + vh * HEAD_V_DIM + v_dim_offs, mask=v_mask, other=0.0
    )
    v_dst = v_out_ptr + tok_idx * stride_v_tok + vh * stride_v_head + v_dim_offs
    tl.store(v_dst, v_vals, mask=v_mask)

  vh_offs = tl.arange(0, BVH)
  vh_mask = vh_offs < NUM_V_HEADS
  a_vals = tl.load(
      a_ptr + tok_idx * stride_a_tok + vh_offs, mask=vh_mask, other=0.0
  ).to(tl.float32)
  dt_bias_vals = tl.load(dt_bias_ptr + vh_offs, mask=vh_mask, other=0.0).to(
      tl.float32
  )
  A_log_vals = tl.load(A_log_ptr + vh_offs, mask=vh_mask, other=0.0).to(
      tl.float32
  )

  x = a_vals + dt_bias_vals
  bx = SOFTPLUS_BETA * x
  softplus_x = tl.where(
      bx <= SOFTPLUS_THRESHOLD,
      (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(bx)),
      x,
  )
  g_vals = -tl.exp(A_log_vals) * softplus_x
  if OUTPUT_G_EXP:
    g_vals = tl.exp(g_vals)
  tl.store(
      g_out_ptr + tok_idx * stride_g_tok + vh_offs,
      g_vals.to(g_out_ptr.dtype.element_ty),
      mask=vh_mask,
  )

  b_vals = tl.load(
      b_ptr + tok_idx * stride_b_tok + vh_offs, mask=vh_mask, other=0.0
  ).to(tl.float32)
  beta_vals = tl.sigmoid(b_vals)
  tl.store(
      beta_out_ptr + tok_idx * stride_beta_tok + vh_offs,
      beta_vals.to(beta_out_ptr.dtype.element_ty),
      mask=vh_mask,
  )


def fused_post_conv_prep(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    eps: float = 1e-6,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  num_tokens = conv_output.shape[0]
  num_v_heads = a.shape[-1]
  dtype = conv_output.dtype
  device = conv_output.device
  key_dim = num_k_heads * head_k_dim

  q_out = torch.empty(
      (num_tokens, num_k_heads, head_k_dim), dtype=dtype, device=device
  )
  k_out = torch.empty(
      (num_tokens, num_k_heads, head_k_dim), dtype=dtype, device=device
  )
  v_out = torch.empty(
      (num_tokens, num_v_heads, head_v_dim), dtype=dtype, device=device
  )
  g_out = torch.empty(
      (num_tokens, num_v_heads), dtype=torch.float32, device=device
  )
  beta_out = torch.empty((num_tokens, num_v_heads), dtype=dtype, device=device)

  if num_tokens == 0:
    return q_out, k_out, v_out, g_out, beta_out

  bk = triton.next_power_of_2(head_k_dim)
  bv = triton.next_power_of_2(head_v_dim)
  bvh = triton.next_power_of_2(num_v_heads)

  grid = (num_tokens,)
  _fused_post_conv_prep_kernel[grid](
      conv_output,
      a,
      b,
      A_log,
      dt_bias,
      q_out,
      k_out,
      v_out,
      g_out,
      beta_out,
      stride_conv_tok=conv_output.stride(0),
      stride_a_tok=a.stride(0),
      stride_b_tok=b.stride(0),
      stride_q_tok=q_out.stride(0),
      stride_q_head=q_out.stride(1),
      stride_k_tok=k_out.stride(0),
      stride_k_head=k_out.stride(1),
      stride_v_tok=v_out.stride(0),
      stride_v_head=v_out.stride(1),
      stride_g_tok=g_out.stride(0),
      stride_beta_tok=beta_out.stride(0),
      NUM_K_HEADS=num_k_heads,
      NUM_V_HEADS=num_v_heads,
      HEAD_K_DIM=head_k_dim,
      HEAD_V_DIM=head_v_dim,
      BK=bk,
      BV=bv,
      BVH=bvh,
      KEY_DIM=key_dim,
      EPS=eps,
      SOFTPLUS_BETA=softplus_beta,
      SOFTPLUS_THRESHOLD=softplus_threshold,
      APPLY_L2NORM=apply_l2norm,
      OUTPUT_G_EXP=output_g_exp,
  )
  return q_out, k_out, v_out, g_out, beta_out


# ============================================================================
# Fused Sigmoid Gating Delta Rule Update (Decode / Speculative Decode)
# ============================================================================
@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": (
            lambda args: args["ssm_state_indices"] is not None
        ),
        "IS_SPEC_DECODING": (
            lambda args: args["num_accepted_tokens"] is not None
        ),
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
  i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
  i_n, i_hv = i_nh // HV, i_nh % HV
  i_h = i_hv // (HV // H)
  if IS_VARLEN:
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    all_tok = T
    T = eos - bos
  else:
    bos, eos = i_n * T, i_n * T + T
    all_tok = B * T

  if T == 0:
    return

  o_k = i_k * BK + tl.arange(0, BK)
  o_v = i_v * BV + tl.arange(0, BV)

  p_q = q + (bos * H + i_h) * K + o_k
  p_k = k + (bos * H + i_h) * K + o_k
  p_v = v + (bos * HV + i_hv) * V + o_v

  p_A_log = A_log + i_hv
  if not IS_KDA:
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv
  else:
    p_a = a + (bos * HV + i_hv) * K + o_k
    p_dt_bias = dt_bias + i_hv * K + o_k

  p_b = b + bos * HV + i_hv
  p_o = o + ((i_k * all_tok + bos) * HV + i_hv) * V + o_v

  mask_k = o_k < K
  mask_v = o_v < V
  mask_h = mask_v[:, None] & mask_k[None, :]

  b_h = tl.zeros([BV, BK], dtype=tl.float32)
  if USE_INITIAL_STATE:
    if IS_CONTINUOUS_BATCHING:
      if IS_SPEC_DECODING:
        i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
      else:
        i_t = 0
      state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
          tl.int64
      )
      if state_idx < 0:
        return
      p_h0 = h0 + state_idx * stride_init_state_token
    else:
      p_h0 = h0 + bos * HV * V * K
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

  for i_t in range(0, T):
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    b_b = tl.load(p_b).to(tl.float32)

    x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
    b_beta = tl.sigmoid(b_b.to(tl.float32))

    if USE_QK_L2NORM_IN_KERNEL:
      b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
      b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
    b_q = b_q * scale
    if not IS_KDA:
      b_h *= tl.exp(b_g)
    else:
      b_h *= tl.exp(b_g[None, :])
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    if INPLACE_FINAL_STATE:
      if ssm_state_indices is not None:
        if stride_indices_tok == 0 or T > 1:
          final_state_idx = tl.load(
              ssm_state_indices + i_n * stride_indices_seq
          ).to(tl.int64)
        else:
          final_state_idx = tl.load(
              ssm_state_indices + i_n * stride_indices_seq + i_t
          ).to(tl.int64)
      else:
        final_state_idx = i_n
      if final_state_idx >= 0:
        p_ht = ht + final_state_idx * stride_final_state_token
        p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
    else:
      p_ht = ht + (bos + i_t) * stride_final_state_token
      p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
      tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

    p_q += H * K
    p_k += H * K
    p_o += HV * V
    p_v += HV * V
    p_b += HV
    p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  B, T, H, K, V = *k.shape, v.shape[-1]
  HV = v.shape[2]
  N = B if cu_seqlens is None else len(cu_seqlens) - 1
  BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
  NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
  assert NK == 1, "NK > 1 is not supported yet"
  num_stages = 3
  num_warps = 4

  if scale is None:
    scale = k.shape[-1] ** -0.5

  o = q.new_empty(NK, *v.shape)
  if inplace_final_state:
    final_state = initial_state
  else:
    final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

  stride_init_state_token = initial_state.stride(0)
  stride_final_state_token = final_state.stride(0)

  if ssm_state_indices is None:
    stride_indices_seq, stride_indices_tok = 1, 1
  elif ssm_state_indices.ndim == 1:
    stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 0
  else:
    stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

  grid = (NK, NV, N * HV)
  fused_sigmoid_gating_delta_rule_update_kernel[grid](
      A_log=A_log,
      a=a.contiguous(),
      b=b.contiguous(),
      dt_bias=dt_bias,
      beta=beta,
      threshold=threshold,
      q=q.contiguous(),
      k=k.contiguous(),
      v=v.contiguous(),
      o=o,
      h0=initial_state,
      ht=final_state,
      cu_seqlens=cu_seqlens,
      ssm_state_indices=ssm_state_indices,
      num_accepted_tokens=num_accepted_tokens,
      scale=scale,
      N=N,
      T=T,
      B=B,
      H=H,
      HV=HV,
      K=K,
      V=V,
      BK=BK,
      BV=BV,
      stride_init_state_token=stride_init_state_token,
      stride_final_state_token=stride_final_state_token,
      stride_indices_seq=stride_indices_seq,
      stride_indices_tok=stride_indices_tok,
      INPLACE_FINAL_STATE=inplace_final_state,
      USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
      IS_KDA=is_kda,
      num_warps=num_warps,
      num_stages=num_stages,
  )
  o = o.squeeze(0)
  return o, final_state


# ============================================================================
# Fused Recurrent Packed Decode
# ============================================================================
@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
  i_v, i_nh = tl.program_id(0), tl.program_id(1)
  i_n, i_hv = i_nh // HV, i_nh % HV
  i_h = i_hv // (HV // H)

  o_k = tl.arange(0, BK)
  o_v = i_v * BV + tl.arange(0, BV)
  mask_k = o_k < K
  mask_v = o_v < V
  mask_h = mask_v[:, None] & mask_k[None, :]

  state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
  if state_idx < 0:
    return

  key_dim = H * K
  p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
  p_q = p_mixed + i_h * K + o_k
  p_k = p_mixed + key_dim + i_h * K + o_k
  p_v = p_mixed + 2 * key_dim + i_hv * V + o_v

  p_a = a + i_n * stride_a_tok + i_hv
  p_b = b + i_n * stride_b_tok + i_hv
  p_A_log = A_log + i_hv
  p_dt_bias = dt_bias + i_hv
  p_o = o + (i_n * HV + i_hv) * V + o_v

  p_h0 = (
      h0
      + state_idx * stride_init_state_token
      + i_hv * V * K
      + o_v[:, None] * K
      + o_k[None, :]
  )
  b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

  b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
  b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
  b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
  b_a = tl.load(p_a).to(tl.float32)
  b_b = tl.load(p_b).to(tl.float32)
  b_A_log = tl.load(p_A_log).to(tl.float32)
  b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

  x = b_a + b_dt_bias
  softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
  b_g = -tl.exp(b_A_log) * softplus_x
  b_beta = tl.sigmoid(b_b)

  if USE_QK_L2NORM_IN_KERNEL:
    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
  b_q = b_q * scale

  b_h *= tl.exp(b_g)
  b_v -= tl.sum(b_h * b_k[None, :], 1)
  b_v *= b_beta
  b_h += b_v[:, None] * b_k[None, :]
  b_o = tl.sum(b_h * b_q[None, :], 1)

  tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

  p_ht = (
      ht
      + state_idx * stride_final_state_token
      + i_hv * V * K
      + o_v[:, None] * K
      + o_k[None, :]
  )
  tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  B = mixed_qkv.shape[0]
  HV, V, K = initial_state.shape[1], initial_state.shape[2], initial_state.shape[3]
  H = (mixed_qkv.shape[1] - HV * V) // (2 * K)

  BK = triton.next_power_of_2(K)
  BV = min(triton.next_power_of_2(V), 32)
  NV = triton.cdiv(V, BV)

  stride_indices_seq = ssm_state_indices.stride(0)
  grid = (NV, B * HV)
  fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
      mixed_qkv=mixed_qkv,
      a=a,
      b=b,
      A_log=A_log,
      dt_bias=dt_bias,
      o=out,
      h0=initial_state,
      ht=initial_state,
      ssm_state_indices=ssm_state_indices,
      scale=scale,
      stride_mixed_qkv_tok=mixed_qkv.stride(0),
      stride_a_tok=a.stride(0),
      stride_b_tok=b.stride(0),
      stride_init_state_token=initial_state.stride(0),
      stride_final_state_token=initial_state.stride(0),
      stride_indices_seq=stride_indices_seq,
      H=H,
      HV=HV,
      K=K,
      V=V,
      BK=BK,
      BV=BV,
      SOFTPLUS_THRESHOLD=20.0,
      USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
      num_warps=4,
      num_stages=3,
  )
  return out, initial_state


# ============================================================================
# LayerNorm / RMSNorm Guard (for FusedRMSNormGated)
# ============================================================================
def rmsnorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = True,
) -> torch.Tensor:
  """PyTorch/Torch-compiled implementation of gated RMSNorm/LayerNorm."""
  orig_dtype = x.dtype
  x_f = x.float()
  if not norm_before_gate and z is not None:
    x_f = x_f * F.silu(z.float())

  if group_size is not None and group_size < x_f.shape[-1]:
    shape_prefix = x_f.shape[:-1]
    d = x_f.shape[-1]
    ng = d // group_size
    x_g = x_f.view(*shape_prefix, ng, group_size)
    if is_rms_norm:
      rstd = torch.rsqrt(x_g.pow(2).mean(dim=-1, keepdim=True) + eps)
      x_norm = (x_g * rstd).view(*shape_prefix, d)
    else:
      mean = x_g.mean(dim=-1, keepdim=True)
      var = (x_g - mean).pow(2).mean(dim=-1, keepdim=True)
      x_norm = ((x_g - mean) * torch.rsqrt(var + eps)).view(*shape_prefix, d)
  else:
    if is_rms_norm:
      rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
      x_norm = x_f * rstd
    else:
      mean = x_f.mean(dim=-1, keepdim=True)
      var = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
      x_norm = (x_f - mean) * torch.rsqrt(var + eps)

  out = x_norm * weight.float()
  if bias is not None:
    out = out + bias.float()
  if norm_before_gate and z is not None:
    out = out * F.silu(z.float())
  return out.to(orig_dtype)


def layernorm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
) -> torch.Tensor:
  return rmsnorm_fn(
      x=x,
      weight=weight,
      bias=bias,
      z=z,
      eps=eps,
      group_size=group_size,
      norm_before_gate=norm_before_gate,
      is_rms_norm=False,
  )
