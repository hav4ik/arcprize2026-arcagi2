"""DeepSeek-V4.1-Flash-1017 exact NumPy/ml_dtypes reference implementation.

Matches the official reference equations in:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
  - inference/model.py (CED, CSA2, Single-Pass mHC, MoE, DSpark, Engram)
  - inference/kernel.py (act_quant, fp4_act_quant, sparse_attn,
  hc_split_sinkhorn)
  - inference/engram.py (EngramLayout, NgramHashState)
  - inference/convert.py (cast_e2m1fn_to_e4m3fn, checkpoint key mapping)

No silent fallbacks: any unsupported configuration or missing weight key raises
immediately.
"""

from __future__ import annotations

# pylint: disable=g-doc-args,g-doc-return-or-yield,missing-function-docstring

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
from typing import Any

import ml_dtypes
import numpy as np

BF16 = np.dtype(ml_dtypes.bfloat16)
FP8 = np.dtype(ml_dtypes.float8_e4m3fn)
FP4 = np.dtype(ml_dtypes.float4_e2m1fn)
F32 = np.dtype(np.float32)

FP8_MAX = np.float32(448.0)
FP8_MAX_INV = np.float32(1.0 / 448.0)
FP4_MAX = np.float32(6.0)
FP4_MAX_INV = np.float32(1.0 / 6.0)
ACT_AMAX_FLOOR = np.float32(1e-4)
KV_FP4_AMAX_FLOOR = np.float32(6.0 * (2.0**-9))
IDX_FP4_AMAX_FLOOR = np.float32(6.0 * (2.0**-126))
FP8_BLOCK = 32
FP4_BLOCK = 32
KV_FP4_BLOCK = 16

FP4_TABLE = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


@dataclasses.dataclass(frozen=True)
class Ds4Config:
  """DeepSeek-V4.1-Flash-1017 hyperparameters."""

  vocab_size: int = 129280
  dim: int = 5120
  moe_inter_dim: int = 2304
  n_layers: int = 40
  n_mtp_layers: int = 3
  n_heads: int = 64
  n_routed_experts: int = 384
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
      0,
      0,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
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
  engram_max_ngram_size: int = 4
  engram_vocab_size: int = 16000000
  engram_n_heads: int = 8
  engram_head_dim: int = 256
  engram_pad_id: int = 2
  engram_compressed_vocab_size: int = 99092
  dspark_block_size: int = 5
  dspark_noise_token_id: int = 128799
  dspark_target_layer_ids: tuple[int, ...] = (37, 38, 39)
  dspark_markov_rank: int = 256
  dspark_n_routed_experts: int = 128
  dspark_n_activated_experts: int = 3
  bos_token_id: int = 0
  eos_token_id: int = 1
  pad_token_id: int = 2
  image_token_id: int = 129264

  def __post_init__(self):
    total = self.n_layers + self.n_mtp_layers
    if len(self.compress_ratios) != total:
      raise ValueError(
          f"compress_ratios length {len(self.compress_ratios)} != {total}"
      )
    if self.n_heads % self.o_groups != 0:
      raise ValueError(
          f"n_heads {self.n_heads} not divisible by o_groups {self.o_groups}"
      )
    if self.head_dim % 128 != 0 or self.rope_head_dim != 64:
      raise ValueError(
          f"Invalid head_dim={self.head_dim},"
          f" rope_head_dim={self.rope_head_dim}"
      )

  @property
  def total_layers(self) -> int:
    return self.n_layers + self.n_mtp_layers

  @property
  def nope_head_dim(self) -> int:
    return self.head_dim - self.rope_head_dim

  @property
  def softmax_scale(self) -> float:
    return self.head_dim**-0.5

  @property
  def index_softmax_scale(self) -> float:
    return self.index_head_dim**-0.5

  @property
  def mix_hc(self) -> int:
    return (2 + self.hc_mult) * self.hc_mult

  @property
  def hc_dim(self) -> int:
    return self.hc_mult * self.dim

  @property
  def engram_n_hash_cols(self) -> int:
    return (self.engram_max_ngram_size - 1) * self.engram_n_heads

  @property
  def encoder_layers(self) -> tuple[int, ...]:
    """Layers in the Causal Encoder (compress_ratio in {0, 2} before L20)."""
    enc = [l for l in range(self.n_layers) if l < self.decoder_start_layer]
    return tuple(enc)

  @property
  def decoder_start_layer(self) -> int:
    """First Decoder layer (first layer with compress_ratio == 1, L20 in release)."""
    for l in range(self.n_layers):
      if self.compress_ratios[l] == 1:
        return l
    return self.n_layers // 2

  @property
  def decoder_layers(self) -> tuple[int, ...]:
    return tuple(range(self.decoder_start_layer, self.n_layers))

  def is_kv_source(self, layer_id: int) -> bool:
    return layer_id < self.n_layers and layer_id in self.kv_source_layers

  def is_index_source(self, layer_id: int) -> bool:
    return layer_id < self.n_layers and layer_id in self.index_source_layers

  def is_candidate_source(self, layer_id: int) -> bool:
    return layer_id == self.candidate_source_layer

  def uses_candidates(self, layer_id: int) -> bool:
    return 0 <= self.candidate_source_layer < layer_id < self.n_layers

  def has_engram(self, layer_id: int) -> bool:
    return layer_id in self.engram_layer_ids

  def get_moe_config(self, layer_id: int) -> tuple[int, int]:
    if layer_id < self.n_layers:
      return self.n_routed_experts, self.n_activated_experts
    return (
        self.dspark_n_routed_experts or self.n_routed_experts,
        self.dspark_n_activated_experts or self.n_activated_experts,
    )

  def kv_owner_of(self, layer_id: int) -> int | None:
    """Returns the kv_source_layer whose compress_kv_cache layer_id reads."""
    if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
      return None
    owners = [s for s in self.kv_source_layers if s <= layer_id]
    if not owners:
      raise ValueError(f"Layer {layer_id} has no preceding kv_source_layer")
    return owners[-1]

  def index_owner_of(self, layer_id: int) -> int | None:
    """Returns the index_source_layer whose topk_idxs layer_id reads."""
    if layer_id >= self.n_layers or self.compress_ratios[layer_id] == 0:
      return None
    owners = [s for s in self.index_source_layers if s <= layer_id]
    if not owners:
      raise ValueError(f"Layer {layer_id} has no preceding index_source_layer")
    return owners[-1]

  @classmethod
  def from_dict(cls, raw: Mapping[str, Any]) -> Ds4Config:
    """Parses either HF root config.json or inference/config.json."""
    tc = raw.get("text_config", raw)
    rope = tc.get("rope_scaling") or {}
    return cls(
        vocab_size=int(tc.get("vocab_size", cls.vocab_size)),
        dim=int(tc.get("dim", tc.get("hidden_size", cls.dim))),
        moe_inter_dim=int(
            tc.get(
                "moe_inter_dim",
                tc.get("moe_intermediate_size", cls.moe_inter_dim),
            )
        ),
        n_layers=int(
            tc.get("n_layers", tc.get("num_hidden_layers", cls.n_layers))
        ),
        n_mtp_layers=int(
            tc.get(
                "n_mtp_layers",
                tc.get("num_nextn_predict_layers", cls.n_mtp_layers),
            )
        ),
        n_heads=int(
            tc.get("n_heads", tc.get("num_attention_heads", cls.n_heads))
        ),
        n_routed_experts=int(tc.get("n_routed_experts", cls.n_routed_experts)),
        n_shared_experts=int(tc.get("n_shared_experts", cls.n_shared_experts)),
        n_activated_experts=int(
            tc.get(
                "n_activated_experts",
                tc.get("num_experts_per_tok", cls.n_activated_experts),
            )
        ),
        score_func=str(
            tc.get("score_func", tc.get("scoring_func", cls.score_func))
        ),
        route_scale=float(
            tc.get(
                "route_scale", tc.get("routed_scaling_factor", cls.route_scale)
            )
        ),
        swiglu_limit=float(tc.get("swiglu_limit", cls.swiglu_limit)),
        q_lora_rank=int(tc.get("q_lora_rank", cls.q_lora_rank)),
        head_dim=int(tc.get("head_dim", cls.head_dim)),
        rope_head_dim=int(
            tc.get(
                "rope_head_dim", tc.get("qk_rope_head_dim", cls.rope_head_dim)
            )
        ),
        norm_eps=float(
            tc.get("norm_eps", tc.get("rms_norm_eps", cls.norm_eps))
        ),
        o_groups=int(tc.get("o_groups", cls.o_groups)),
        o_lora_rank=int(tc.get("o_lora_rank", cls.o_lora_rank)),
        window_size=int(
            tc.get("window_size", tc.get("sliding_window", cls.window_size))
        ),
        compress_ratios=tuple(int(x) for x in tc["compress_ratios"]),
        kv_source_layers=tuple(
            int(x)
            for x in tc.get(
                "kv_source_layers", tc.get("kv_source_layer_ids", ())
            )
        ),
        index_source_layers=tuple(
            int(x)
            for x in tc.get(
                "index_source_layers", tc.get("index_source_layer_ids", ())
            )
        ),
        original_seq_len=int(
            tc.get(
                "original_seq_len",
                rope.get(
                    "original_max_position_embeddings", cls.original_seq_len
                ),
            )
        ),
        rope_theta=float(tc.get("rope_theta", cls.rope_theta)),
        compress_rope_theta=float(
            tc.get("compress_rope_theta", cls.compress_rope_theta)
        ),
        rope_factor=float(
            tc.get("rope_factor", rope.get("factor", cls.rope_factor))
        ),
        beta_fast=int(
            tc.get("beta_fast", rope.get("beta_fast", cls.beta_fast))
        ),
        beta_slow=int(
            tc.get("beta_slow", rope.get("beta_slow", cls.beta_slow))
        ),
        index_n_heads=int(tc.get("index_n_heads", cls.index_n_heads)),
        index_head_dim=int(tc.get("index_head_dim", cls.index_head_dim)),
        index_topk=int(tc.get("index_topk", cls.index_topk)),
        candidate_source_layer=int(
            tc.get(
                "candidate_source_layer",
                tc.get("candidate_source_layer_id", cls.candidate_source_layer),
            )
        ),
        candidate_topk_blocks=int(
            tc.get("candidate_topk_blocks", cls.candidate_topk_blocks)
        ),
        candidate_block_size=int(
            tc.get("candidate_block_size", cls.candidate_block_size)
        ),
        hc_mult=int(tc.get("hc_mult", cls.hc_mult)),
        hc_sinkhorn_iters=int(
            tc.get("hc_sinkhorn_iters", cls.hc_sinkhorn_iters)
        ),
        hc_eps=float(tc.get("hc_eps", cls.hc_eps)),
        engram_layer_ids=tuple(int(x) for x in tc.get("engram_layer_ids", ())),
        engram_num_embeddings=tuple(
            int(x) for x in tc.get("engram_num_embeddings", ())
        ),
        engram_max_ngram_size=int(
            tc.get("engram_max_ngram_size", cls.engram_max_ngram_size)
        ),
        engram_vocab_size=int(
            tc.get("engram_vocab_size", cls.engram_vocab_size)
        ),
        engram_n_heads=int(tc.get("engram_n_heads", cls.engram_n_heads)),
        engram_head_dim=int(tc.get("engram_head_dim", cls.engram_head_dim)),
        engram_pad_id=int(
            tc.get(
                "engram_pad_id",
                tc.get("engram_pad_token_id", cls.engram_pad_id),
            )
        ),
        engram_compressed_vocab_size=int(
            tc.get(
                "engram_compressed_vocab_size", cls.engram_compressed_vocab_size
            )
        ),
        dspark_block_size=int(
            tc.get("dspark_block_size", cls.dspark_block_size)
        ),
        dspark_noise_token_id=int(
            tc.get("dspark_noise_token_id", cls.dspark_noise_token_id)
        ),
        dspark_target_layer_ids=tuple(
            int(x) for x in tc.get("dspark_target_layer_ids", ())
        ),
        dspark_markov_rank=int(
            tc.get("dspark_markov_rank", cls.dspark_markov_rank)
        ),
        dspark_n_routed_experts=int(
            tc.get("dspark_n_routed_experts", cls.dspark_n_routed_experts)
        ),
        dspark_n_activated_experts=int(
            tc.get(
                "dspark_n_activated_experts",
                tc.get(
                    "dspark_num_experts_per_tok", cls.dspark_n_activated_experts
                ),
            )
        ),
    )

  @classmethod
  def from_json(cls, path: str) -> Ds4Config:
    with open(path, "r") as f:
      return cls.from_dict(json.load(f))

  @classmethod
  def release_standin_config(cls, n_dev: int) -> Ds4Config:
    """6-layer + 1-MTP config with 100% release per-layer dimensions."""
    o_groups = min(8, n_dev) if n_dev >= 2 else 1
    return cls(
        vocab_size=129280,
        dim=5120,
        moe_inter_dim=2304,
        n_layers=6,
        n_mtp_layers=1,
        n_heads=64,
        n_routed_experts=384,
        n_shared_experts=1,
        n_activated_experts=6,
        q_lora_rank=1280,
        head_dim=512,
        rope_head_dim=64,
        o_groups=o_groups,
        o_lora_rank=1024,
        window_size=128,
        compress_ratios=(0, 2, 2, 1, 1, 1, 0),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4),
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_source_layer=3,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=(1,),
        engram_num_embeddings=(65536,),
        engram_max_ngram_size=4,
        engram_vocab_size=65536,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_id=2,
        engram_compressed_vocab_size=99092,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=(3, 4, 5),
        dspark_markov_rank=256,
        dspark_n_routed_experts=128,
        dspark_n_activated_experts=3,
    )

  @classmethod
  def tiny_config(
      cls,
      *,
      dim: int = 256,
      moe_inter_dim: int = 256,
      n_heads: int = 8,
      o_groups: int = 2,
      o_lora_rank: int = 128,
      q_lora_rank: int = 128,
      n_routed_experts: int = 8,
      n_activated_experts: int = 2,
      dspark_n_routed_experts: int = 8,
      dspark_n_activated_experts: int = 2,
      vocab_size: int = 512,
      window_size: int = 16,
      index_n_heads: int = 8,
      index_topk: int = 8,
      candidate_topk_blocks: int = 4,
      candidate_block_size: int = 4,
  ) -> Ds4Config:
    """Compact 9-layer + 2-MTP config exercising every DeepSeek-V4.1 pathway.

    Layers:
      L0: Encoder Pure SWA (ratio=0)
      L1: Encoder Pure SWA + Engram (ratio=0)
      L2: Encoder CSA2 Full Mode (ratio=2) + Engram
      L3: Encoder CSA2 Reuse Mode (ratio=2, reuses L2)
      L4: Decoder CED Full Mode + Candidate Source (ratio=1)
      L5: Decoder CSA2 Reuse Mode (ratio=1, reuses L4)
      L6: Decoder CSA2 Reindex Mode (ratio=1, reuses L4 KV + candidate pool)
      L7: Decoder CSA2 Reuse Mode (ratio=1, reuses L4 KV + L6 indices)
      L8: Decoder CSA2 Reuse Mode (ratio=1, reuses L4 KV + L6 indices)
      L9..10 (mtp.0..1): DSpark Drafter (reading target layers L6, L7, L8)
    """
    return cls(
        vocab_size=vocab_size,
        dim=dim,
        moe_inter_dim=moe_inter_dim,
        n_layers=9,
        n_mtp_layers=2,
        n_heads=n_heads,
        n_routed_experts=n_routed_experts,
        n_shared_experts=1,
        n_activated_experts=n_activated_experts,
        q_lora_rank=q_lora_rank,
        head_dim=512,
        rope_head_dim=64,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        window_size=window_size,
        compress_ratios=(0, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0),
        kv_source_layers=(2, 4),
        index_source_layers=(2, 4, 6),
        index_n_heads=index_n_heads,
        index_head_dim=128,
        index_topk=index_topk,
        candidate_source_layer=4,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        engram_layer_ids=(1, 2),
        engram_num_embeddings=(2048, 2048),
        engram_max_ngram_size=4,
        engram_vocab_size=64,
        engram_n_heads=2,
        engram_head_dim=128,
        engram_pad_id=2,
        engram_compressed_vocab_size=vocab_size,
        dspark_block_size=5,
        dspark_noise_token_id=vocab_size - 1,
        dspark_target_layer_ids=(6, 7, 8),
        dspark_markov_rank=128,
        dspark_n_routed_experts=dspark_n_routed_experts,
        dspark_n_activated_experts=dspark_n_activated_experts,
    )


# ==============================================================================
# Quantization & Dequantization Primitives (kernel.py & convert.py)
# ==============================================================================


def fast_round_scale_e8m0(
    amax: np.ndarray, max_inv: float | np.floating[Any]
) -> tuple[np.ndarray, np.ndarray]:
  """Computes power-of-two scale 2^ceil(log2(amax * max_inv)) via IEEE-754 bits.

  Matches kernel.py:fast_round_scale bit-for-bit.

  Args:
    amax: float32 array of positive group maxima.
    max_inv: 1.0 / 448.0 for FP8 or 1.0 / 6.0 for FP4.

  Returns:
    (scale_f32, scale_u8_e8m0).
  """
  prod = (np.asarray(amax, dtype=np.float32) * np.float32(max_inv)).astype(
      np.float32
  )
  bits = prod.view(np.uint32)
  exp = (bits >> np.uint32(23)) & np.uint32(0xFF)
  man_nonzero = ((bits & np.uint32(0x7FFFFF)) != 0).astype(np.uint32)
  exp_ceil = np.clip(exp + man_nonzero, 0, 254).astype(np.uint32)
  scale_f32 = (exp_ceil << np.uint32(23)).view(np.float32)
  return scale_f32, exp_ceil.astype(np.uint8)


def e8m0_to_f32(scale_u8: np.ndarray) -> np.ndarray:
  """Converts uint8 E8M0 exponent bytes to float32 2^(e - 127)."""
  return (np.asarray(scale_u8, dtype=np.uint32) << np.uint32(23)).view(
      np.float32
  )


def act_quant_fp8(
    x: np.ndarray, block_size: int = FP8_BLOCK, inplace: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
  """Block-wise FP8 (E4M3FN) quantization with E8M0 power-of-two scales.

  Matches kernel.py:act_quant(..., block_size=32, scale_fmt='ue8m0').
  """
  orig_shape = x.shape
  n = orig_shape[-1]
  if n % block_size != 0:
    raise ValueError(f"Last dim {n} not divisible by block_size {block_size}")
  groups = np.asarray(x, dtype=np.float32).reshape(
      -1, n // block_size, block_size
  )
  amax = np.maximum(np.max(np.abs(groups), axis=-1), ACT_AMAX_FLOOR)
  s_f32, s_u8 = fast_round_scale_e8m0(amax, FP8_MAX_INV)
  q_fp8 = np.clip(groups / s_f32[:, :, None], -FP8_MAX, FP8_MAX).astype(FP8)
  if inplace:
    deq = (q_fp8.astype(np.float32) * s_f32[:, :, None]).astype(x.dtype)
    return deq.reshape(orig_shape)
  return (
      q_fp8.reshape(orig_shape),
      s_u8.reshape(*orig_shape[:-1], n // block_size),
  )


def _quantize_f32_to_e2m1_nibbles(val_f32: np.ndarray) -> np.ndarray:
  """Quantizes clamped [-6, 6] float32 values to 4-bit E2M1 nibbles (0..15)."""
  q_fp4 = np.clip(val_f32, -FP4_MAX, FP4_MAX).astype(FP4)
  return q_fp4.view(np.uint8) & np.uint8(0x0F)


def pack_e2m1_nibbles(nibbles: np.ndarray) -> np.ndarray:
  """Packs [..., K] uint8 nibbles (0..15) into [..., K // 2] uint8 bytes."""
  low = nibbles[..., 0::2] & np.uint8(0x0F)
  high = (nibbles[..., 1::2] & np.uint8(0x0F)) << np.uint8(4)
  return (low | high).astype(np.uint8)


def unpack_e2m1_bytes(packed_u8: np.ndarray) -> np.ndarray:
  """Unpacks [..., K // 2] uint8 bytes into [..., K] float32 E2M1 values."""
  u8 = np.asarray(packed_u8, dtype=np.uint8)
  low = u8 & np.uint8(0x0F)
  high = (u8 >> np.uint8(4)) & np.uint8(0x0F)
  vals = np.stack([FP4_TABLE[low], FP4_TABLE[high]], axis=-1)
  return vals.reshape(*u8.shape[:-1], u8.shape[-1] * 2)


def fp4_act_quant(
    x: np.ndarray,
    block_size: int = FP4_BLOCK,
    inplace: bool = False,
    scale_dtype: str = "e8m0",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
  """FP4 quantization with E8M0 scales (Indexer, block=32) or E4M3 scales (Main KV, block=16).

  Matches kernel.py:fp4_act_quant bit-for-bit.
  """
  orig_shape = x.shape
  n = orig_shape[-1]
  if n % block_size != 0:
    raise ValueError(f"Last dim {n} not divisible by block_size {block_size}")
  groups = np.asarray(x, dtype=np.float32).reshape(
      -1, n // block_size, block_size
  )
  amax = np.max(np.abs(groups), axis=-1)
  if scale_dtype == "e4m3":
    amax = np.maximum(amax, KV_FP4_AMAX_FLOOR)
    s_out = (amax / FP4_MAX).astype(FP8)
    s_f32 = s_out.astype(np.float32)
  elif scale_dtype == "e8m0":
    amax = np.maximum(amax, IDX_FP4_AMAX_FLOOR)
    s_f32, s_out = fast_round_scale_e8m0(amax, FP4_MAX_INV)
  else:
    raise ValueError(f"Unsupported scale_dtype={scale_dtype!r}")

  scaled = np.clip(groups / s_f32[:, :, None], -FP4_MAX, FP4_MAX)
  q_fp4 = scaled.astype(FP4)
  if inplace:
    deq = (q_fp4.astype(np.float32) * s_f32[:, :, None]).astype(x.dtype)
    return deq.reshape(orig_shape)
  nibbles = q_fp4.view(np.uint8) & np.uint8(0x0F)
  packed = pack_e2m1_nibbles(nibbles.reshape(*orig_shape))
  return packed, s_out.reshape(*orig_shape[:-1], n // block_size)


def dequantize_fp8_block(
    w_fp8: np.ndarray, w_scale_u8: np.ndarray, block_size: int = FP8_BLOCK
) -> np.ndarray:
  """Dequantizes [out, in] FP8 weight with [out//B, in//B] E8M0 scale to float32."""
  out_dim, in_dim = w_fp8.shape
  b_out = out_dim // block_size
  b_in = in_dim // block_size
  s_f32 = e8m0_to_f32(w_scale_u8).reshape(b_out, 1, b_in, 1)
  w_4d = w_fp8.astype(np.float32).reshape(b_out, block_size, b_in, block_size)
  return (w_4d * s_f32).reshape(out_dim, in_dim)


def dequantize_mxfp4_weight(
    w_packed_u8: np.ndarray, w_scale_u8: np.ndarray, block_size: int = FP4_BLOCK
) -> np.ndarray:
  """Dequantizes [out, in // 2] packed FP4 weight with [out, in // 32] E8M0 scale."""
  vals = unpack_e2m1_bytes(w_packed_u8)
  out_dim, in_dim = vals.shape
  s_f32 = e8m0_to_f32(w_scale_u8).reshape(out_dim, in_dim // block_size, 1)
  return (
      vals.reshape(out_dim, in_dim // block_size, block_size) * s_f32
  ).reshape(out_dim, in_dim)


def cast_e2m1fn_to_e4m3fn(
    w_packed_u8: np.ndarray, w_scale_u8: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Losslessly converts [out, in//2] E2M1 + [out, in//32] E8M0 to FP8 + [out//32, in//32] E8M0.

  Exact NumPy port of inference/convert.py:cast_e2m1fn_to_e4m3fn.
  """
  vals = unpack_e2m1_bytes(w_packed_u8)
  out_dim, in_dim = vals.shape
  b_out = out_dim // FP8_BLOCK
  b_in = in_dim // FP8_BLOCK
  # [b_out, b_in, 32, 32]
  x = vals.reshape(b_out, FP8_BLOCK, b_in, FP8_BLOCK).transpose(0, 2, 1, 3)
  s_f32 = (
      e8m0_to_f32(w_scale_u8)
      .reshape(b_out, FP8_BLOCK, b_in, 1)
      .transpose(0, 2, 1, 3)
  )
  # Max scale in each 32x32 block divided by 2^6 (since 6.0 * 2^6 = 384 <= 448)
  s_max = np.max(s_f32, axis=(2, 3), keepdims=True) / np.float32(64.0)
  offset = s_f32 / s_max
  w_fp8 = (
      (x * offset).transpose(0, 2, 1, 3).reshape(out_dim, in_dim).astype(FP8)
  )
  s_bits = s_max.squeeze((2, 3)).astype(np.float32).view(np.uint32)
  s_e8m0 = ((s_bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint8)
  return w_fp8, s_e8m0


def linear_fp8(
    x: np.ndarray, w_fp8: np.ndarray, w_scale_u8: np.ndarray
) -> np.ndarray:
  """Dynamic FP8 activation x FP8 weight GEMM with per-32 E8M0 scales."""
  orig_shape = x.shape
  q_a, s_a = act_quant_fp8(x, FP8_BLOCK, inplace=False)
  a_deq = (
      q_a.astype(np.float32).reshape(-1, orig_shape[-1] // FP8_BLOCK, FP8_BLOCK)
      * e8m0_to_f32(s_a).reshape(-1, orig_shape[-1] // FP8_BLOCK, 1)
  ).reshape(-1, orig_shape[-1])
  w_deq = dequantize_fp8_block(w_fp8, w_scale_u8, FP8_BLOCK)
  out = a_deq @ w_deq.T
  return out.astype(x.dtype).reshape(*orig_shape[:-1], w_fp8.shape[0])


def linear_fp4(
    x: np.ndarray, w_packed_u8: np.ndarray, w_scale_u8: np.ndarray
) -> np.ndarray:
  """Dynamic FP8 activation x FP4 weight GEMM with per-32 E8M0 scales."""
  orig_shape = x.shape
  q_a, s_a = act_quant_fp8(x, FP8_BLOCK, inplace=False)
  a_deq = (
      q_a.astype(np.float32).reshape(-1, orig_shape[-1] // FP8_BLOCK, FP8_BLOCK)
      * e8m0_to_f32(s_a).reshape(-1, orig_shape[-1] // FP8_BLOCK, 1)
  ).reshape(-1, orig_shape[-1])
  w_deq = dequantize_mxfp4_weight(w_packed_u8, w_scale_u8, FP4_BLOCK)
  out = a_deq @ w_deq.T
  return out.astype(x.dtype).reshape(*orig_shape[:-1], w_packed_u8.shape[0])


def linear_dense(x: np.ndarray, w: np.ndarray) -> np.ndarray:
  """Plain BF16/FP32 linear projection x @ w^T."""
  orig_shape = x.shape
  out = (
      np.asarray(x, dtype=np.float32).reshape(-1, orig_shape[-1])
      @ np.asarray(w, dtype=np.float32).T
  )
  return out.astype(x.dtype).reshape(*orig_shape[:-1], w.shape[0])


# ==============================================================================
# Normalization, RoPE, and Single-Pass mHC (model.py & kernel.py)
# ==============================================================================


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
  """RMSNorm in FP32, scaled by weight and cast back to x.dtype."""
  xf = np.asarray(x, dtype=np.float32)
  var = np.mean(xf * xf, axis=-1, keepdims=True)
  normed = xf / np.sqrt(var + np.float32(eps))
  return (normed * np.asarray(weight, dtype=np.float32)).astype(x.dtype)


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Precomputes YaRN rotary (cos, sin) tables of shape [seqlen, dim // 2] float32.

  Matches model.py:precompute_freqs_cis bit-for-bit.
  """
  freqs = 1.0 / (
      np.float32(base)
      ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
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
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3),
        0.0,
        1.0,
    )
    smooth = 1.0 - ramp
    freqs = freqs / np.float32(factor) * (1.0 - smooth) + freqs * smooth

  angles = np.outer(np.arange(seqlen, dtype=np.float32), freqs).astype(
      np.float32
  )
  return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def apply_rotary_emb(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    *,
    rope_dim: int = 64,
    inverse: bool = False,
) -> np.ndarray:
  """Rotates the trailing rope_dim channels of x as adjacent complex pairs.

  Args:
    x: [B, S, D] or [B, S, H, D].
    cos: [S, rope_dim // 2] float32.
    sin: [S, rope_dim // 2] float32.
    rope_dim: number of trailing channels rotated (default 64).
    inverse: if True, conjugates the rotation (sin -> -sin).

  Returns:
    Copy of x with x[..., -rope_dim:] rotated.
  """
  out = np.array(x, copy=True)
  tail = out[..., -rope_dim:].astype(np.float32)
  evens = tail[..., 0::2]
  odds = tail[..., 1::2]
  c = cos[None, :, :] if x.ndim == 3 else cos[None, :, None, :]
  s = sin[None, :, :] if x.ndim == 3 else sin[None, :, None, :]
  if inverse:
    s = -s
  rot_evens = evens * c - odds * s
  rot_odds = evens * s + odds * c
  rot_tail = np.stack([rot_evens, rot_odds], axis=-1).reshape(tail.shape)
  out[..., -rope_dim:] = rot_tail.astype(x.dtype)
  return out


def hc_split_sinkhorn(
    mixes: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes (pre, post, comb) mHC coefficients with Sinkhorn-Knopp balancing.

  Matches kernel.py:hc_split_sinkhorn_kernel bit-for-bit.
  """
  m = np.asarray(mixes, dtype=np.float32)
  sc = np.asarray(hc_scale, dtype=np.float32)
  hb = np.asarray(hc_base, dtype=np.float32)
  eps32 = np.float32(eps)

  pre_logits = m[..., :hc_mult] * sc[0] + hb[:hc_mult]
  pre = (1.0 / (1.0 + np.exp(-pre_logits))) + eps32

  post_logits = (
      m[..., hc_mult : 2 * hc_mult] * sc[1] + hb[hc_mult : 2 * hc_mult]
  )
  post = 2.0 / (1.0 + np.exp(-post_logits))

  comb_logits = (m[..., 2 * hc_mult :] * sc[2] + hb[2 * hc_mult :]).reshape(
      *m.shape[:-1], hc_mult, hc_mult
  )
  row_max = np.max(comb_logits, axis=-1, keepdims=True)
  comb = np.exp(comb_logits - row_max)
  comb = comb / np.sum(comb, axis=-1, keepdims=True) + eps32
  comb = comb / (np.sum(comb, axis=-2, keepdims=True) + eps32)
  for _ in range(sinkhorn_iters - 1):
    comb = comb / (np.sum(comb, axis=-1, keepdims=True) + eps32)
    comb = comb / (np.sum(comb, axis=-2, keepdims=True) + eps32)
  return (
      pre.astype(np.float32),
      post.astype(np.float32),
      comb.astype(np.float32),
  )


def hc_mixes(
    x_mhc: np.ndarray,
    hc_fn: np.ndarray,
    hc_scale: np.ndarray,
    hc_base: np.ndarray,
    norm_eps: float,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Projects flattened [B, S, hc_mult * dim] stream and runs Sinkhorn."""
  b, s, hc, d = x_mhc.shape
  flat = x_mhc.reshape(b, s, hc * d).astype(np.float32)
  rsqrt = 1.0 / np.sqrt(
      np.mean(flat * flat, axis=-1, keepdims=True) + np.float32(norm_eps)
  )
  mixes = (flat @ np.asarray(hc_fn, dtype=np.float32).T) * rsqrt
  return hc_split_sinkhorn(
      mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps
  )


def hc_pre(x_mhc: np.ndarray, pre_mix: np.ndarray) -> np.ndarray:
  """Collapses [B, S, hc_mult, dim] into [B, S, dim] using pre_mix [B, S, hc_mult]."""
  y = np.sum(
      np.asarray(pre_mix, dtype=np.float32)[..., None]
      * np.asarray(x_mhc, dtype=np.float32),
      axis=2,
  )
  return y.astype(x_mhc.dtype)


def hc_post(
    sub_out: np.ndarray,
    residual_mhc: np.ndarray,
    post: np.ndarray,
    comb: np.ndarray,
) -> np.ndarray:
  """Expands sub_out [B, S, dim] back to [B, S, hc_mult, dim] and mixes residual."""
  term1 = (
      np.asarray(post, dtype=np.float32)[..., None]
      * np.asarray(sub_out, dtype=np.float32)[:, :, None, :]
  )
  # sum_i comb[..., i, j] * residual[..., i, d]
  term2 = np.einsum(
      "bsij,bsid->bsjd",
      np.asarray(comb, dtype=np.float32),
      np.asarray(residual_mhc, dtype=np.float32),
  )
  return (term1 + term2).astype(sub_out.dtype)


def make_identity_pre_mix(
    batch: int, seqlen: int, hc_mult: int = 4
) -> np.ndarray:
  pre_mix = np.zeros((batch, seqlen, hc_mult), dtype=np.float32)
  pre_mix[:, :, 0] = 1.0
  return pre_mix


# ==============================================================================
# Engram Conditional Memory (engram.py & model.py)
# ==============================================================================


def _is_prime(n: int) -> bool:
  if n < 2:
    return False
  if n % 2 == 0:
    return n == 2
  r = int(math.isqrt(n))
  for d in range(3, r + 1, 2):
    if n % d == 0:
      return False
  return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
  candidate = start + 1
  while not _is_prime(candidate) or candidate in seen_primes:
    candidate += 1
  return candidate


def compute_hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int
) -> np.ndarray:
  """Computes [n_engram_layers, max_ngram_size] odd int64 hash multipliers."""
  max_long = np.iinfo(np.int64).max
  multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
  rows = []
  for layer_id in layer_ids:
    gen = np.random.default_rng(10007 * int(layer_id))
    vals = gen.integers(
        0, multiplier_bound, size=(max_ngram_size,), dtype=np.int64
    )
    rows.append(vals * 2 + 1)
  return np.stack(rows, axis=0)


@dataclasses.dataclass(frozen=True)
class EngramTables:
  """Precomputed deterministic N-gram hash tables for Engram layers."""

  layer_ids: tuple[int, ...]
  max_ngram_size: int
  n_heads: int
  head_dim: int
  pad_compressed_id: int
  token_map: np.ndarray  # [vocab_size] int64
  multipliers: np.ndarray  # [n_engram_layers, max_ngram_size] int64
  primes: np.ndarray  # [n_engram_layers, max_ngram_size - 1, n_heads] int64
  offsets: np.ndarray  # [n_engram_layers, (max_ngram_size - 1) * n_heads] int64

  @classmethod
  def from_config(
      cls, cfg: Ds4Config, token_map: np.ndarray | None = None
  ) -> EngramTables:
    if token_map is None:
      # Identity map mod engram_compressed_vocab_size when tokenizer is omitted.
      token_map = (
          np.arange(cfg.vocab_size, dtype=np.int64)
          % cfg.engram_compressed_vocab_size
      )
    else:
      token_map = np.asarray(token_map, dtype=np.int64)
    seen: set[int] = set()
    primes_list = []
    offsets_list = []
    for _ in cfg.engram_layer_ids:
      per_ngram = []
      flat_sizes = []
      for _ in range(cfg.engram_max_ngram_size - 1):
        sizes = []
        cur = cfg.engram_vocab_size - 1
        for _ in range(cfg.engram_n_heads):
          cur = find_next_prime(cur, seen)
          seen.add(cur)
          sizes.append(cur)
          flat_sizes.append(cur)
        per_ngram.append(sizes)
      primes_list.append(per_ngram)
      offsets_list.append(np.cumsum([0, *flat_sizes[:-1]], dtype=np.int64))
    multipliers = compute_hash_multipliers(
        cfg.engram_layer_ids,
        cfg.engram_max_ngram_size,
        cfg.engram_compressed_vocab_size,
    )
    return cls(
        layer_ids=cfg.engram_layer_ids,
        max_ngram_size=cfg.engram_max_ngram_size,
        n_heads=cfg.engram_n_heads,
        head_dim=cfg.engram_head_dim,
        pad_compressed_id=int(token_map[cfg.engram_pad_id]),
        token_map=token_map,
        multipliers=multipliers,
        primes=np.asarray(primes_list, dtype=np.int64),
        offsets=np.asarray(offsets_list, dtype=np.int64),
    )


def compute_engram_hashes(
    input_ids: np.ndarray,
    start_pos: int,
    compressed_history_cache: np.ndarray,
    tables: EngramTables,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Updates compressed_history_cache and returns [B, S, n_engram_layers, n_hash_cols] int64."""
  dead = -1
  batch, seqlen = input_ids.shape
  compressed = tables.token_map[input_ids]
  if token_mask is not None:
    compressed = np.where(token_mask, compressed, dead)
  compressed_history_cache[:batch, start_pos : start_pos + seqlen] = compressed

  positions = np.broadcast_to(
      np.arange(start_pos, start_pos + seqlen, dtype=np.int64)[None, :],
      (batch, seqlen),
  )
  tokens = []
  blocked = np.zeros((batch, seqlen), dtype=bool)
  for shift in range(tables.max_ngram_size):
    src_pos = np.maximum(positions - shift, 0)
    source = np.take_along_axis(
        compressed_history_cache[:batch], src_pos, axis=1
    )
    blocked = blocked | (positions < shift) | (source == dead)
    tokens.append(np.where(blocked, tables.pad_compressed_id, source))
  tokens_arr = np.stack(tokens, axis=-1)  # [B, S, max_ngram_size]

  # [B, S, n_engram_layers, max_ngram_size]
  products = tokens_arr[:, :, None, :] * tables.multipliers[None, None, :, :]
  rolling = products[..., 0]
  hashes = []
  for i in range(1, tables.max_ngram_size):
    rolling = np.bitwise_xor(rolling, products[..., i])
    # modulo prime for each head: [B, S, n_engram_layers, n_heads]
    h_mod = rolling[..., None] % tables.primes[None, None, :, i - 1, :]
    hashes.append(h_mod)
  return np.concatenate(hashes, axis=-1) + tables.offsets[None, None, :, :]


def engram_forward(
    x_mhc: np.ndarray,
    hash_ids: np.ndarray,
    embed_w_fp8: np.ndarray,
    embed_s_u8: np.ndarray,
    wkv_w_fp8: np.ndarray,
    wkv_s_u8: np.ndarray,
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    norm_eps: float,
    token_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Executes Engram conditional memory lookup and signed-sqrt gated residual add.

  Matches model.py:Engram.forward bit-for-bit.
  """
  b, s, hc_mult, dim = x_mhc.shape
  # Lookup FP8 rows and E8M0 scales: [B, S, n_hash_cols, head_dim]
  raw_fp8 = embed_w_fp8[hash_ids]
  raw_scale = e8m0_to_f32(embed_s_u8[hash_ids])
  head_dim = embed_w_fp8.shape[-1]
  embed_deq = (
      (
          raw_fp8.astype(np.float32).reshape(
              b, s, -1, head_dim // FP8_BLOCK, FP8_BLOCK
          )
          * raw_scale[..., None]
      )
      .reshape(b, s, -1)
      .astype(BF16)
  )

  kv = linear_fp8(embed_deq, wkv_w_fp8, wkv_s_u8)
  key = kv[..., : hc_mult * dim].astype(np.float32).reshape(b, s, hc_mult, dim)
  value = kv[..., hc_mult * dim :].astype(np.float32)

  weight = np.asarray(q_weight, dtype=np.float32) * np.asarray(
      k_weight, dtype=np.float32
  )
  h = np.asarray(x_mhc, dtype=np.float32)
  eps = np.float32(norm_eps)
  rstd = (1.0 / np.sqrt(np.mean(h * h, axis=-1) + eps)) * (
      1.0 / np.sqrt(np.mean(key * key, axis=-1) + eps)
  )
  dot = (
      np.sum(h * weight[None, None, :, :] * key, axis=-1)
      * rstd
      * np.float32(dim**-0.5)
  )
  signed_sqrt = np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)
  gate = 1.0 / (1.0 + np.exp(-signed_sqrt))
  if token_mask is not None:
    gate = np.where(token_mask[:, :, None], gate, 0.0)
  out = h + gate[..., None] * value[:, :, None, :]
  return out.astype(x_mhc.dtype)


# ==============================================================================
# CSA2 Compressor, Two-Level Indexer, Sparse Attention, and MoE
# ==============================================================================


@dataclasses.dataclass
class LayerCache:
  """Runtime caches for one layer."""

  window_kv: (
      np.ndarray
  )  # [B, window_size, head_dim] BF16 (FP8 in-place quantized)
  compress_kv: np.ndarray | None = (
      None  # [B, max_seq // ratio, head_dim] BF16 (FP4 in-place quantized)
  )
  index_k: np.ndarray | None = (
      None  # [B, max_seq // ratio, index_head_dim] BF16 (FP4 in-place)
  )
  comp_kv_state: np.ndarray | None = None  # [B, ratio, head_dim] FP32
  comp_score_state: np.ndarray | None = None  # [B, ratio, head_dim] FP32


def compressor_forward(
    x: np.ndarray,
    start_pos: int,
    ratio: int,
    wkv: np.ndarray,
    norm_w: np.ndarray,
    norm_eps: float,
    wgate: np.ndarray | None = None,
    kv_state: np.ndarray | None = None,
    score_state: np.ndarray | None = None,
) -> np.ndarray | None:
  """Pools compress_ratio consecutive tokens into one pre-RoPE KV latent.

  Matches model.py:Compressor.forward bit-for-bit, including stateful decode
  carry.
  """
  bsz, seqlen, _ = x.shape
  if ratio == 1:
    return rms_norm(linear_dense(x, wkv), norm_w, norm_eps)

  assert wgate is not None and kv_state is not None and score_state is not None
  xf = x.astype(np.float32)
  kv = linear_dense(xf, wkv.astype(np.float32))
  score = linear_dense(xf, wgate.astype(np.float32))

  if start_pos == 0:
    should_compress = seqlen >= ratio
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    if remainder:
      kv_state[:bsz, :remainder] = kv[:, cutoff:]
      score_state[:bsz, :remainder] = score[:, cutoff:]
      kv = kv[:, :cutoff]
      score = score[:, :cutoff]
    if not should_compress:
      return None
    kv = kv.reshape(bsz, -1, ratio, kv.shape[-1])
    score = score.reshape(bsz, -1, ratio, score.shape[-1])
    s_max = np.max(score, axis=2, keepdims=True)
    probs = np.exp(score - s_max)
    probs = probs / np.sum(probs, axis=2, keepdims=True)
    pooled = np.sum(kv * probs, axis=2)
    return rms_norm(pooled.astype(x.dtype), norm_w, norm_eps)

  # Single-token or multi-token decode / continuation from start_pos > 0:
  # Process each token t at pos = start_pos + t through the ratio-slot buffer.
  emitted = []
  for t in range(seqlen):
    pos = start_pos + t
    slot = pos % ratio
    kv_state[:bsz, slot] = kv[:, t]
    score_state[:bsz, slot] = score[:, t]
    if (pos + 1) % ratio == 0:
      s_max = np.max(score_state[:bsz], axis=1, keepdims=True)
      probs = np.exp(score_state[:bsz] - s_max)
      probs = probs / np.sum(probs, axis=1, keepdims=True)
      pooled = np.sum(kv_state[:bsz] * probs, axis=1, keepdims=True)
      emitted.append(pooled)
  if not emitted:
    return None
  cat = np.concatenate(emitted, axis=1)
  return rms_norm(cat.astype(x.dtype), norm_w, norm_eps)


def select_candidate_blocks(
    logits: np.ndarray,
    compress_lens: np.ndarray | int,
    topk_blocks: int,
    block_size: int,
) -> np.ndarray:
  """Selects topk_blocks of block_size positions, pinning newest block last to +inf.

  Matches model.py:select_candidate_blocks bit-for-bit.
  """
  width = logits.shape[-1]
  pad_w = (-width) % block_size
  if pad_w:
    pad_cfg = [(0, 0)] * (logits.ndim - 1) + [(0, pad_w)]
    scores = np.pad(logits, pad_cfg, mode="constant", constant_values=-np.inf)
  else:
    scores = logits.copy()
  scores = np.max(scores.reshape(*logits.shape[:-1], -1, block_size), axis=-1)
  num_blocks = scores.shape[-1]

  last = (np.asarray(compress_lens) - 1) // block_size
  block_arange = np.arange(num_blocks)
  scores = np.where(block_arange == last, np.inf, scores)

  k = min(topk_blocks, num_blocks)
  # Exact top-k selection along last axis
  part_idx = np.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
  part_vals = np.take_along_axis(scores, part_idx, axis=-1)
  keep = np.zeros_like(scores, dtype=bool)
  np.put_along_axis(keep, part_idx, part_vals > -np.inf, axis=-1)
  return np.repeat(keep, block_size, axis=-1)[..., :width]


def get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
) -> np.ndarray:
  """Returns [bsz, seqlen, min(seqlen, window_size) or window_size] int32 window indices."""
  if start_pos == 0:
    end = np.arange(seqlen, dtype=np.int32)[:, None]
    w = min(seqlen, window_size)
    idxs = (
        np.maximum(end - window_size + 1, 0)
        + np.arange(w, dtype=np.int32)[None, :]
    )
    idxs = np.where(idxs > end, -1, idxs)
    return np.broadcast_to(idxs[None, :, :], (bsz, seqlen, w)).copy()
  # For decode (seqlen == 1 or speculative block seqlen >= 1):
  rows = []
  for s in range(seqlen):
    p = start_pos + s
    oldest = p % window_size + 1
    row = np.concatenate(
        [
            np.arange(oldest, window_size, dtype=np.int32),
            np.arange(oldest, dtype=np.int32),
        ],
        axis=0,
    )
    row = np.where(row > p, -1, row)
    rows.append(row)
  stacked = np.stack(rows, axis=0)
  return np.broadcast_to(stacked[None, :, :], (bsz, seqlen, window_size)).copy()


def sparse_attn_forward(
    q: np.ndarray,
    kv: np.ndarray,
    attn_sink: np.ndarray,
    topk_idxs: np.ndarray,
    softmax_scale: float,
) -> np.ndarray:
  """Sparse attention over gathered topk_idxs with learnable attn_sink.

  Matches kernel.py:sparse_attn_kernel bit-for-bit (including -1e30 max init).

  Args:
    q: [B, M, H, D] BF16.
    kv: [B, N, D] BF16.
    attn_sink: [H] FP32.
    topk_idxs: [B, M, K] int32 (-1 for masked entries).
    softmax_scale: float (D ** -0.5).

  Returns:
    [B, M, H, D] BF16.
  """
  bsz, m_len, n_heads, d = q.shape
  out = np.zeros((bsz, m_len, n_heads, d), dtype=BF16)
  sink = np.asarray(attn_sink, dtype=np.float32)

  for b in range(bsz):
    for m in range(m_len):
      idxs = topk_idxs[b, m]
      valid = idxs != -1
      safe_idxs = np.where(valid, idxs, 0)
      kv_gathered = kv[b, safe_idxs].astype(np.float32)  # [K, D]
      q_m = q[b, m].astype(np.float32)  # [H, D]
      scores = (q_m @ kv_gathered.T) * np.float32(softmax_scale)  # [H, K]
      scores = np.where(valid[None, :], scores, -np.inf)
      scores_max = np.maximum(np.max(scores, axis=-1), np.float32(-1e30))  # [H]
      probs = np.exp(scores - scores_max[:, None])  # [H, K]
      sum_exp = np.sum(probs, axis=-1) + np.exp(sink - scores_max)  # [H]
      # kernel.py casts probs to BF16 before multiplying by kv_shared (BF16).
      probs_bf16 = probs.astype(BF16).astype(np.float32)
      acc_o = probs_bf16 @ kv[b, safe_idxs].astype(np.float32)  # [H, D]
      out[b, m] = (acc_o / sum_exp[:, None]).astype(BF16)
  return out


def moe_gate_forward(
    x: np.ndarray,
    gate_weight: np.ndarray,
    gate_bias: np.ndarray,
    *,
    topk: int,
    score_func: str = "sqrtsoftplus",
    gate_temp: float = 1.0,
    norm_topk_prob: bool = True,
    route_scale: float = 1.5,
    gate_bias_vl: np.ndarray | None = None,
    image_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes MoE routing weights [N, topk] FP32 and expert indices [N, topk] int32.

  Matches model.py:Gate.forward bit-for-bit.
  """
  xf = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
  wf = np.asarray(gate_weight, dtype=np.float32)
  logits = (xf @ wf.T) / np.float32(gate_temp)
  if score_func == "softmax":
    l_max = np.max(logits, axis=-1, keepdims=True)
    scores = np.exp(logits - l_max)
    scores = scores / np.sum(scores, axis=-1, keepdims=True)
  elif score_func == "sigmoid":
    scores = 1.0 / (1.0 + np.exp(-logits))
  elif score_func == "sqrtsoftplus":
    # log1p(exp(x)) in float32
    sp = np.where(
        logits > 20.0, logits, np.log1p(np.exp(np.clip(logits, -80.0, 20.0)))
    )
    scores = np.sqrt(sp)
  else:
    raise ValueError(f"Unsupported score_func={score_func!r}")

  bias = np.asarray(gate_bias, dtype=np.float32)[None, :]
  if image_mask is not None and gate_bias_vl is not None:
    mask_flat = image_mask.reshape(-1)[:, None]
    bias = np.where(
        mask_flat, np.asarray(gate_bias_vl, dtype=np.float32)[None, :], bias
    )

  biased = scores + bias
  # Exact descending top-k
  indices = np.argsort(-biased, axis=-1, kind="stable")[:, :topk].astype(
      np.int32
  )
  weights = np.take_along_axis(scores, indices, axis=-1)
  if norm_topk_prob and topk > 1:
    weights = weights / (
        np.sum(weights, axis=-1, keepdims=True) + np.float32(1e-20)
    )
  weights = (weights * np.float32(route_scale)).astype(np.float32)
  return weights, indices


def swiglu_clamped(
    gate: np.ndarray,
    up: np.ndarray,
    swiglu_limit: float = 10.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
  """Clamped SwiGLU activation: up in [-limit, limit], gate <= limit."""
  gf = np.asarray(gate, dtype=np.float32)
  uf = np.asarray(up, dtype=np.float32)
  if swiglu_limit > 0:
    lim = np.float32(swiglu_limit)
    uf = np.clip(uf, -lim, lim)
    gf = np.minimum(gf, lim)
  silu_g = gf / (1.0 + np.exp(-gf))
  act = silu_g * uf
  if weights is not None:
    act = act * np.asarray(weights, dtype=np.float32)
  return act.astype(BF16)


# ==============================================================================
# Full Layer & Model Forward Passes (Attention, Block, DSpark, CED Prefill)
# ==============================================================================


@dataclasses.dataclass
class SharedAttnState:
  """Cross-layer shared CSA2 state within a forward pass."""

  compress_kv: np.ndarray | None = None
  index_k: np.ndarray | None = None
  topk_idxs: np.ndarray | None = None
  candidates: np.ndarray | None = None


def indexer_forward(
    x: np.ndarray,
    qr: np.ndarray,
    latent: np.ndarray | None,
    start_pos: int,
    offset: int,
    layer_id: int,
    cfg: Ds4Config,
    params: Mapping[str, np.ndarray],
    cache: LayerCache,
    shared: SharedAttnState,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    *,
    write_global_kv: bool = True,
) -> np.ndarray:
  """Executes CSA2 Indexer (Full Mode or Candidate-Restricted Reindex Mode).

  Matches model.py:Indexer.forward bit-for-bit.
  """
  bsz, seqlen, _ = x.shape
  ratio = cfg.compress_ratios[layer_id]
  rd = cfg.rope_head_dim
  end_pos = start_pos + seqlen
  owns_k = cfg.is_kv_source(layer_id)
  prefix = f"layers.{layer_id}.attn.indexer."

  if owns_k and latent is not None and write_global_kv:
    assert cache.index_k is not None
    if start_pos == 0:
      pos_idx = np.arange(0, seqlen - (seqlen % ratio), ratio, dtype=np.int32)
    else:
      # Each emitted latent stands for the first token of its ratio-token group
      num_emitted = latent.shape[1]
      first_group_start = ((start_pos + ratio - 1) // ratio) * ratio - (
          ratio - 1
      )
      if ratio == 1:
        first_group_start = start_pos
      else:
        # First completed group in [start_pos, end_pos) started at
        # first_end - (ratio - 1).
        first_end = start_pos + ((ratio - (start_pos % ratio)) % ratio)
        if (start_pos + 1) % ratio == 0:
          first_end = start_pos
        else:
          first_end = start_pos + (ratio - 1 - (start_pos % ratio))
        first_group_start = first_end - (ratio - 1)
      pos_idx = (
          first_group_start + np.arange(num_emitted, dtype=np.int32) * ratio
      )

    k = rms_norm(
        linear_dense(latent, params[prefix + "wk.weight"]),
        params[prefix + "k_norm.weight"],
        cfg.norm_eps,
    )
    k = apply_rotary_emb(k, cos_table[pos_idx], sin_table[pos_idx], rope_dim=rd)
    k = np.asarray(
        fp4_act_quant(k, FP4_BLOCK, inplace=True, scale_dtype="e8m0")
    )
    write_slot = start_pos // ratio
    cache.index_k[:bsz, write_slot : write_slot + k.shape[1]] = k

  if owns_k:
    shared.index_k = cache.index_k

  q = linear_fp8(
      qr, params[prefix + "wq_b.weight"], params[prefix + "wq_b.scale"]
  ).reshape(bsz, seqlen, cfg.index_n_heads, cfg.index_head_dim)
  q = apply_rotary_emb(
      q,
      cos_table[start_pos:end_pos],
      sin_table[start_pos:end_pos],
      rope_dim=rd,
  )
  q = np.asarray(fp4_act_quant(q, FP4_BLOCK, inplace=True, scale_dtype="e8m0"))

  assert shared.index_k is not None
  comp_total = end_pos // ratio
  index_k = shared.index_k[:bsz, :comp_total]
  weights = linear_dense(x, params[prefix + "weights_proj.weight"]).astype(
      np.float32
  ) * np.float32(cfg.index_softmax_scale * (cfg.index_n_heads**-0.5))

  # [B, S, H, T]
  raw_scores = np.einsum(
      "bshd,btd->bsht",
      q.astype(np.float32),
      index_k.astype(np.float32),
  )
  index_score = np.sum(
      np.maximum(raw_scores, 0.0) * weights[:, :, :, None], axis=2
  )

  if start_pos == 0:
    compress_lens = (np.arange(1, seqlen + 1, dtype=np.int32) // ratio)[
        None, :, None
    ]
    t_arange = np.arange(comp_total, dtype=np.int32)[None, None, :]
    index_score = np.where(t_arange >= compress_lens, -np.inf, index_score)
  else:
    compress_lens = (
        np.arange(start_pos + 1, end_pos + 1, dtype=np.int32) // ratio
    )[None, :, None]
    t_arange = np.arange(comp_total, dtype=np.int32)[None, None, :]
    index_score = np.where(t_arange >= compress_lens, -np.inf, index_score)

  if cfg.is_candidate_source(layer_id):
    shared.candidates = select_candidate_blocks(
        index_score,
        compress_lens,
        cfg.candidate_topk_blocks,
        cfg.candidate_block_size,
    )
  elif cfg.uses_candidates(layer_id):
    assert shared.candidates is not None
    index_score = np.where(shared.candidates, index_score, -np.inf)

  topk = min(cfg.index_topk, comp_total)
  if topk == 0:
    return np.empty((bsz, seqlen, 0), dtype=np.int32)
  # Pick top-k by score, then sort indices in ascending position order
  top_unsorted = np.argsort(-index_score, axis=-1, kind="stable")[..., :topk]
  idxs = np.sort(top_unsorted, axis=-1)
  return np.where(idxs < compress_lens, idxs + offset, -1).astype(np.int32)


def dequant_in_bf16(
    vals: np.ndarray, scales: np.ndarray, block_size: int
) -> np.ndarray:
  """Multiplies quantized codes and block scales in bfloat16, returning float32."""
  v_bf16: Any = np.asarray(vals, dtype=BF16).reshape(
      vals.shape[0], vals.shape[1] // block_size, block_size
  )
  s_bf16: Any = np.asarray(scales, dtype=BF16)[:, :, None]
  return np.asarray(v_bf16 * s_bf16, dtype=np.float32).reshape(vals.shape)


def attention_forward(
    x: np.ndarray,
    start_pos: int,
    layer_id: int,
    cfg: Ds4Config,
    params: Mapping[str, np.ndarray],
    cache: LayerCache,
    shared: SharedAttnState,
    cos_swa: np.ndarray,
    sin_swa: np.ndarray,
    cos_comp: np.ndarray,
    sin_comp: np.ndarray,
    *,
    write_global_kv: bool = True,
    swa_window_floor: int = 0,
) -> np.ndarray:
  """Executes SWA + CSA2 Attention for layer_id."""
  bsz, seqlen, _ = x.shape
  ratio = cfg.compress_ratios[layer_id]
  cos_tbl = cos_comp if ratio > 0 else cos_swa
  sin_tbl = sin_comp if ratio > 0 else sin_swa
  end_pos = start_pos + seqlen
  rd = cfg.rope_head_dim
  win = cfg.window_size
  prefix = f"layers.{layer_id}.attn."

  qr = rms_norm(
      linear_fp8(
          x, params[prefix + "wq_a.weight"], params[prefix + "wq_a.scale"]
      ),
      params[prefix + "q_norm.weight"],
      cfg.norm_eps,
  )
  q = linear_fp8(
      qr, params[prefix + "wq_b.weight"], params[prefix + "wq_b.scale"]
  ).reshape(bsz, seqlen, cfg.n_heads, cfg.head_dim)
  q = apply_rotary_emb(
      q, cos_tbl[start_pos:end_pos], sin_tbl[start_pos:end_pos], rope_dim=rd
  )

  kv = rms_norm(
      linear_fp8(
          x, params[prefix + "wkv.weight"], params[prefix + "wkv.scale"]
      ),
      params[prefix + "kv_norm.weight"],
      cfg.norm_eps,
  )
  kv = apply_rotary_emb(
      kv, cos_tbl[start_pos:end_pos], sin_tbl[start_pos:end_pos], rope_dim=rd
  )
  kv = np.asarray(act_quant_fp8(kv, FP8_BLOCK, inplace=True))

  if start_pos == 0 and seqlen > 1:
    if seqlen <= win:
      cache.window_kv[:bsz, :seqlen] = kv
    else:
      cutoff = seqlen % win
      tail_kv = kv[:, -win:]
      cache.window_kv[:bsz, cutoff:win] = tail_kv[:, : win - cutoff]
      cache.window_kv[:bsz, :cutoff] = tail_kv[:, win - cutoff :]
    window_kv = kv
    win_idxs = get_window_topk_idxs(win, bsz, seqlen, 0)
    if swa_window_floor > 0:
      win_idxs = np.where(win_idxs < swa_window_floor, -1, win_idxs)
  else:
    for s in range(seqlen):
      cache.window_kv[:bsz, (start_pos + s) % win] = kv[:, s]
    window_kv = cache.window_kv[:bsz]
    win_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)

  if ratio > 0:
    comp_len = end_pos // ratio
    latent = None
    if cfg.is_kv_source(layer_id):
      c_pre = prefix + "compressor."
      wgate = params.get(c_pre + "wgate.weight")
      wkv_c = params[c_pre + "wkv.weight"]
      latent = compressor_forward(
          x,
          start_pos,
          ratio,
          wkv_c,
          params[c_pre + "norm.weight"],
          cfg.norm_eps,
          wgate=wgate,
          kv_state=cache.comp_kv_state,
          score_state=cache.comp_score_state,
      )
      shared.compress_kv = cache.compress_kv

    if cfg.is_index_source(layer_id):
      if comp_len == 0:
        comp_idxs = np.empty((bsz, seqlen, 0), dtype=np.int32)
      else:
        comp_idxs = indexer_forward(
            x,
            qr,
            latent,
            start_pos,
            window_kv.shape[1],
            layer_id,
            cfg,
            params,
            cache,
            shared,
            cos_tbl,
            sin_tbl,
            write_global_kv=write_global_kv,
        )
      shared.topk_idxs = comp_idxs
    else:
      assert shared.topk_idxs is not None
      comp_idxs = shared.topk_idxs

    if latent is not None and write_global_kv:
      assert cache.compress_kv is not None
      if start_pos == 0:
        pos_idx = np.arange(0, seqlen - (seqlen % ratio), ratio, dtype=np.int32)
      else:
        num_emitted = latent.shape[1]
        if ratio == 1:
          first_group_start = start_pos
        else:
          first_end = (
              start_pos
              if (start_pos + 1) % ratio == 0
              else start_pos + (ratio - 1 - (start_pos % ratio))
          )
          first_group_start = first_end - (ratio - 1)
        pos_idx = (
            first_group_start + np.arange(num_emitted, dtype=np.int32) * ratio
        )
      rot_lat = apply_rotary_emb(
          latent, cos_tbl[pos_idx], sin_tbl[pos_idx], rope_dim=rd
      )
      rot_lat = np.asarray(
          fp4_act_quant(rot_lat, KV_FP4_BLOCK, inplace=True, scale_dtype="e4m3")
      )
      write_slot = start_pos // ratio
      cache.compress_kv[:bsz, write_slot : write_slot + rot_lat.shape[1]] = (
          rot_lat
      )

    assert shared.compress_kv is not None
    compress_kv = shared.compress_kv[:bsz, :comp_len]
    full_kv = np.concatenate([window_kv, compress_kv], axis=1)
    topk_idxs = np.concatenate([win_idxs, comp_idxs], axis=-1)
  else:
    full_kv = window_kv
    topk_idxs = win_idxs

  o = sparse_attn_forward(
      q,
      full_kv,
      params[prefix + "attn_sink"],
      topk_idxs,
      cfg.softmax_scale,
  )
  o = apply_rotary_emb(
      o,
      cos_tbl[start_pos:end_pos],
      sin_tbl[start_pos:end_pos],
      rope_dim=rd,
      inverse=True,
  )
  o = o.reshape(bsz, seqlen, cfg.o_groups, -1)
  o = np.asarray(act_quant_fp8(o, FP8_BLOCK, inplace=True))

  wo_a_w = params[prefix + "wo_a.weight"]
  if (prefix + "wo_a.scale") in params:
    wo_a_bf16 = dequantize_fp8_block(
        wo_a_w, params[prefix + "wo_a.scale"], FP8_BLOCK
    ).astype(BF16)
  else:
    wo_a_bf16 = wo_a_w.astype(BF16)
  wo_a_grouped = wo_a_bf16.reshape(cfg.o_groups, cfg.o_lora_rank, -1).astype(
      np.float32
  )
  o_proj = np.einsum(
      "bsgd,grd->bsgr", o.astype(np.float32), wo_a_grouped
  ).astype(BF16)
  return linear_fp8(
      o_proj.reshape(bsz, seqlen, -1),
      params[prefix + "wo_b.weight"],
      params[prefix + "wo_b.scale"],
  )


def expert_forward(
    x: np.ndarray,
    prefix: str,
    params: Mapping[str, np.ndarray],
    swiglu_limit: float,
    weights: np.ndarray | None = None,
    *,
    is_shared: bool = False,
) -> np.ndarray:
  lin = linear_fp8 if is_shared else linear_fp4
  gate = lin(x, params[prefix + "w1.weight"], params[prefix + "w1.scale"])
  up = lin(x, params[prefix + "w3.weight"], params[prefix + "w3.scale"])
  act = swiglu_clamped(gate, up, swiglu_limit=swiglu_limit, weights=weights)
  return lin(act, params[prefix + "w2.weight"], params[prefix + "w2.scale"])


def moe_forward(
    x: np.ndarray,
    layer_id: int,
    cfg: Ds4Config,
    params: Mapping[str, np.ndarray],
    image_mask: np.ndarray | None = None,
    *,
    layer_prefix: str | None = None,
) -> np.ndarray:
  orig_shape = x.shape
  x_flat = x.reshape(-1, cfg.dim)
  n_routed, topk = cfg.get_moe_config(layer_id)
  if layer_prefix is None:
    layer_prefix = (
        f"layers.{layer_id}."
        if layer_id < cfg.n_layers
        else f"mtp.{layer_id - cfg.n_layers}."
    )
  ffn_pre = layer_prefix + "ffn."

  weights, indices = moe_gate_forward(
      x_flat,
      params[ffn_pre + "gate.weight"],
      params[ffn_pre + "gate.bias"],
      topk=topk,
      score_func=cfg.score_func,
      gate_temp=cfg.gate_temp,
      norm_topk_prob=cfg.norm_topk_prob,
      route_scale=cfg.route_scale,
      gate_bias_vl=params.get(ffn_pre + "gate.bias_vl"),
      image_mask=image_mask,
  )

  y = np.zeros_like(x_flat, dtype=np.float32)
  for e in range(n_routed):
    mask = indices == e
    if not np.any(mask):
      continue
    tok_idx, slot_idx = np.where(mask)
    exp_out = expert_forward(
        x_flat[tok_idx],
        f"{ffn_pre}experts.{e}.",
        params,
        cfg.swiglu_limit,
        weights=weights[tok_idx, slot_idx, None],
        is_shared=False,
    )
    np.add.at(y, tok_idx, exp_out.astype(np.float32))

  shared_out = expert_forward(
      x_flat,
      f"{ffn_pre}shared_experts.",
      params,
      cfg.swiglu_limit,
      weights=None,
      is_shared=True,
  )
  y = y + shared_out.astype(np.float32)
  return y.astype(x.dtype).reshape(orig_shape)


def block_forward(
    x_mhc: np.ndarray,
    start_pos: int,
    pre_mix: np.ndarray,
    layer_id: int,
    cfg: Ds4Config,
    params: Mapping[str, np.ndarray],
    cache: LayerCache,
    shared: SharedAttnState,
    cos_swa: np.ndarray,
    sin_swa: np.ndarray,
    cos_comp: np.ndarray,
    sin_comp: np.ndarray,
    *,
    image_mask: np.ndarray | None = None,
    write_global_kv: bool = True,
    swa_window_floor: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
  prefix = f"layers.{layer_id}."
  residual = x_mhc
  attn_pre, attn_post, attn_comb = hc_mixes(
      x_mhc,
      params[prefix + "hc_attn_fn"],
      params[prefix + "hc_attn_scale"],
      params[prefix + "hc_attn_base"],
      cfg.norm_eps,
      cfg.hc_mult,
      cfg.hc_sinkhorn_iters,
      cfg.hc_eps,
  )
  x = hc_pre(x_mhc, pre_mix)
  x = rms_norm(x, params[prefix + "attn_norm.weight"], cfg.norm_eps)
  x = attention_forward(
      x,
      start_pos,
      layer_id,
      cfg,
      params,
      cache,
      shared,
      cos_swa,
      sin_swa,
      cos_comp,
      sin_comp,
      write_global_kv=write_global_kv,
      swa_window_floor=swa_window_floor,
  )
  x_mhc = hc_post(x, residual, attn_post, attn_comb)

  residual = x_mhc
  ffn_pre, ffn_post, ffn_comb = hc_mixes(
      x_mhc,
      params[prefix + "hc_ffn_fn"],
      params[prefix + "hc_ffn_scale"],
      params[prefix + "hc_ffn_base"],
      cfg.norm_eps,
      cfg.hc_mult,
      cfg.hc_sinkhorn_iters,
      cfg.hc_eps,
  )
  x = hc_pre(x_mhc, attn_pre)
  x = rms_norm(x, params[prefix + "ffn_norm.weight"], cfg.norm_eps)
  x = moe_forward(x, layer_id, cfg, params, image_mask=image_mask)
  x_mhc = hc_post(x, residual, ffn_post, ffn_comb)
  return x_mhc, ffn_pre


def get_dspark_topk_idxs(
    window_size: int, bsz: int, block_size: int, start_pos: int
) -> np.ndarray:
  assert start_pos > 0
  ctx_part = np.arange(min(window_size, start_pos + 1), dtype=np.int32)
  blk_part = window_size + np.arange(block_size, dtype=np.int32)
  vec = np.concatenate([ctx_part, blk_part], axis=0)
  return np.broadcast_to(
      vec[None, None, :], (bsz, block_size, vec.shape[0])
  ).copy()


def dspark_attention_forward(
    x: np.ndarray,
    start_pos: int,
    main_x: np.ndarray,
    stage_id: int,
    cfg: Ds4Config,
    params: Mapping[str, np.ndarray],
    cache: LayerCache,
    cos_swa: np.ndarray,
    sin_swa: np.ndarray,
) -> np.ndarray:
  bsz, seqlen, _ = main_x.shape
  win = cfg.window_size
  rd = cfg.rope_head_dim
  prefix = f"mtp.{stage_id}.attn."

  main_kv = rms_norm(
      linear_fp8(
          main_x, params[prefix + "wkv.weight"], params[prefix + "wkv.scale"]
      ),
      params[prefix + "kv_norm.weight"],
      cfg.norm_eps,
  )
  main_kv = apply_rotary_emb(
      main_kv,
      cos_swa[start_pos : start_pos + seqlen],
      sin_swa[start_pos : start_pos + seqlen],
      rope_dim=rd,
  )
  main_kv = np.asarray(act_quant_fp8(main_kv, FP8_BLOCK, inplace=True))

  if start_pos == 0:
    if seqlen <= win:
      cache.window_kv[:bsz, :seqlen] = main_kv
    else:
      cutoff = seqlen % win
      tail_kv = main_kv[:, -win:]
      cache.window_kv[:bsz, cutoff:win] = tail_kv[:, : win - cutoff]
      cache.window_kv[:bsz, :cutoff] = tail_kv[:, win - cutoff :]
    return x

  for s in range(seqlen):
    cache.window_kv[:bsz, (start_pos + s) % win] = main_kv[:, s]

  _, block_size, _ = x.shape
  draft_start = start_pos + seqlen
  draft_end = draft_start + block_size
  cos_d = cos_swa[draft_start:draft_end]
  sin_d = sin_swa[draft_start:draft_end]

  qr = rms_norm(
      linear_fp8(
          x, params[prefix + "wq_a.weight"], params[prefix + "wq_a.scale"]
      ),
      params[prefix + "q_norm.weight"],
      cfg.norm_eps,
  )
  q = linear_fp8(
      qr, params[prefix + "wq_b.weight"], params[prefix + "wq_b.scale"]
  ).reshape(bsz, block_size, cfg.n_heads, cfg.head_dim)
  q = apply_rotary_emb(q, cos_d, sin_d, rope_dim=rd)

  kv = rms_norm(
      linear_fp8(
          x, params[prefix + "wkv.weight"], params[prefix + "wkv.scale"]
      ),
      params[prefix + "kv_norm.weight"],
      cfg.norm_eps,
  )
  kv = apply_rotary_emb(kv, cos_d, sin_d, rope_dim=rd)
  kv = np.asarray(act_quant_fp8(kv, FP8_BLOCK, inplace=True))

  topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos + seqlen - 1)
  cat_kv = np.concatenate([cache.window_kv[:bsz], kv], axis=1)
  o = sparse_attn_forward(
      q, cat_kv, params[prefix + "attn_sink"], topk_idxs, cfg.softmax_scale
  )
  o = apply_rotary_emb(o, cos_d, sin_d, rope_dim=rd, inverse=True)
  o = o.reshape(bsz, block_size, cfg.o_groups, -1)
  o = np.asarray(act_quant_fp8(o, FP8_BLOCK, inplace=True))

  wo_a_w = params[prefix + "wo_a.weight"]
  if (prefix + "wo_a.scale") in params:
    wo_a_bf16 = dequantize_fp8_block(
        wo_a_w, params[prefix + "wo_a.scale"], FP8_BLOCK
    ).astype(BF16)
  else:
    wo_a_bf16 = wo_a_w.astype(BF16)
  wo_a_grouped = wo_a_bf16.reshape(cfg.o_groups, cfg.o_lora_rank, -1).astype(
      np.float32
  )
  o_proj = np.einsum(
      "bsgd,grd->bsgr", o.astype(np.float32), wo_a_grouped
  ).astype(BF16)
  return linear_fp8(
      o_proj.reshape(bsz, block_size, -1),
      params[prefix + "wo_b.weight"],
      params[prefix + "wo_b.scale"],
  )


def dspark_forward_spec(
    input_ids: np.ndarray,
    main_hidden: np.ndarray,
    start_pos: int,
    cfg: Ds4Config,
    params: Mapping[str, np.ndarray],
    mtp_caches: Sequence[LayerCache],
    cos_swa: np.ndarray,
    sin_swa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  bsz = input_ids.shape[0]
  block_size = cfg.dspark_block_size
  main_x = rms_norm(
      linear_fp8(
          main_hidden,
          params["mtp.0.main_proj.weight"],
          params["mtp.0.main_proj.scale"],
      ),
      params["mtp.0.main_norm.weight"],
      cfg.norm_eps,
  )
  draft_input_ids = np.full(
      (bsz, block_size), cfg.dspark_noise_token_id, dtype=np.int32
  )
  draft_input_ids[:, 0] = input_ids
  x_emb = params["embed.weight"][draft_input_ids].astype(BF16)
  h = np.repeat(x_emb[:, :, None, :], cfg.hc_mult, axis=2)
  pre_mix = make_identity_pre_mix(bsz, block_size, cfg.hc_mult)

  for stage_id in range(cfg.n_mtp_layers):
    prefix = f"mtp.{stage_id}."
    layer_id = cfg.n_layers + stage_id
    if start_pos == 0:
      dspark_attention_forward(
          h[:, :, 0, :],
          0,
          main_x,
          stage_id,
          cfg,
          params,
          mtp_caches[stage_id],
          cos_swa,
          sin_swa,
      )
      continue

    residual = h
    attn_pre, attn_post, attn_comb = hc_mixes(
        h,
        params[prefix + "hc_attn_fn"],
        params[prefix + "hc_attn_scale"],
        params[prefix + "hc_attn_base"],
        cfg.norm_eps,
        cfg.hc_mult,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    )
    x = hc_pre(h, pre_mix)
    x = rms_norm(x, params[prefix + "attn_norm.weight"], cfg.norm_eps)
    x = dspark_attention_forward(
        x,
        start_pos,
        main_x,
        stage_id,
        cfg,
        params,
        mtp_caches[stage_id],
        cos_swa,
        sin_swa,
    )
    h = hc_post(x, residual, attn_post, attn_comb)

    residual = h
    ffn_pre, ffn_post, ffn_comb = hc_mixes(
        h,
        params[prefix + "hc_ffn_fn"],
        params[prefix + "hc_ffn_scale"],
        params[prefix + "hc_ffn_base"],
        cfg.norm_eps,
        cfg.hc_mult,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
    )
    x = hc_pre(h, attn_pre)
    x = rms_norm(x, params[prefix + "ffn_norm.weight"], cfg.norm_eps)
    x = moe_forward(
        x, layer_id, cfg, params, image_mask=None, layer_prefix=prefix
    )
    h = hc_post(x, residual, ffn_post, ffn_comb)
    pre_mix = ffn_pre

  if start_pos == 0:
    return None

  last_prefix = f"mtp.{cfg.n_mtp_layers - 1}."
  x_collapsed = hc_pre(h, pre_mix)
  normed = rms_norm(
      x_collapsed, params[last_prefix + "norm.weight"], cfg.norm_eps
  )
  logits = (
      normed.astype(np.float32) @ params["head.weight"].astype(np.float32).T
  )

  output_ids = np.empty((bsz, block_size + 1), dtype=np.int32)
  output_ids[:, 0] = input_ids
  markov_embeds = []
  mw_embed = params[last_prefix + "markov_head.embed.weight"].astype(np.float32)
  mw_head = params[last_prefix + "markov_head.head.weight"].astype(np.float32)
  for i in range(block_size):
    m_emb = mw_embed[output_ids[:, i]]
    logits_bias = m_emb @ mw_head.T
    logits[:, i] = logits[:, i] + logits_bias
    markov_embeds.append(m_emb)
    output_ids[:, i + 1] = np.argmax(logits[:, i], axis=-1).astype(np.int32)

  markov_stack = np.stack(markov_embeds, axis=1)
  conf_in = np.concatenate(
      [x_collapsed.astype(np.float32), markov_stack], axis=-1
  )
  conf_w = params[last_prefix + "confidence_head.proj.weight"].astype(
      np.float32
  )
  confidence = (conf_in @ conf_w.T).squeeze(-1)
  return output_ids, logits, confidence


@dataclasses.dataclass
class ReferenceState:
  """Complete runtime KV + Compressor + Engram state across all layers."""

  cfg: Ds4Config
  max_batch_size: int
  max_seq_len: int
  layer_caches: list[LayerCache]
  mtp_caches: list[LayerCache]
  engram_history: np.ndarray
  engram_tables: EngramTables | None
  cos_swa: np.ndarray
  sin_swa: np.ndarray
  cos_comp: np.ndarray
  sin_comp: np.ndarray

  @classmethod
  def create(
      cls,
      cfg: Ds4Config,
      max_batch_size: int = 4,
      max_seq_len: int = 512,
      token_map: np.ndarray | None = None,
  ) -> ReferenceState:
    cos_swa, sin_swa = precompute_freqs_cis(
        cfg.rope_head_dim,
        max_seq_len,
        0,
        cfg.rope_theta,
        cfg.rope_factor,
        cfg.beta_fast,
        cfg.beta_slow,
    )
    cos_comp, sin_comp = precompute_freqs_cis(
        cfg.rope_head_dim,
        max_seq_len,
        cfg.original_seq_len,
        cfg.compress_rope_theta,
        cfg.rope_factor,
        cfg.beta_fast,
        cfg.beta_slow,
    )
    layer_caches = []
    for l in range(cfg.n_layers):
      ratio = cfg.compress_ratios[l]
      win_kv = np.zeros(
          (max_batch_size, cfg.window_size, cfg.head_dim), dtype=BF16
      )
      comp_kv = None
      idx_k = None
      kv_st = None
      sc_st = None
      if cfg.is_kv_source(l):
        comp_kv = np.zeros(
            (max_batch_size, max_seq_len // ratio, cfg.head_dim), dtype=BF16
        )
        idx_k = np.zeros(
            (max_batch_size, max_seq_len // ratio, cfg.index_head_dim),
            dtype=BF16,
        )
        if ratio > 1:
          kv_st = np.zeros(
              (max_batch_size, ratio, cfg.head_dim), dtype=np.float32
          )
          sc_st = np.full(
              (max_batch_size, ratio, cfg.head_dim), -np.inf, dtype=np.float32
          )
      layer_caches.append(
          LayerCache(
              window_kv=win_kv,
              compress_kv=comp_kv,
              index_k=idx_k,
              comp_kv_state=kv_st,
              comp_score_state=sc_st,
          )
      )
    mtp_caches = [
        LayerCache(
            window_kv=np.zeros(
                (max_batch_size, cfg.window_size, cfg.head_dim), dtype=BF16
            )
        )
        for _ in range(cfg.n_mtp_layers)
    ]
    engram_tables = (
        EngramTables.from_config(cfg, token_map=token_map)
        if cfg.engram_layer_ids
        else None
    )
    engram_history = np.zeros((max_batch_size, max_seq_len), dtype=np.int64)
    return cls(
        cfg=cfg,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        layer_caches=layer_caches,
        mtp_caches=mtp_caches,
        engram_history=engram_history,
        engram_tables=engram_tables,
        cos_swa=cos_swa,
        sin_swa=sin_swa,
        cos_comp=cos_comp,
        sin_comp=sin_comp,
    )


def forward_backbone(
    input_ids: np.ndarray,
    start_pos: int,
    state: ReferenceState,
    params: Mapping[str, np.ndarray],
    *,
    full_logits: bool = False,
    write_global_kv: bool = True,
    swa_window_floor: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  cfg = state.cfg
  bsz, seqlen = input_ids.shape
  engram_hashes = None
  if state.engram_tables is not None:
    engram_hashes = compute_engram_hashes(
        input_ids, start_pos, state.engram_history, state.engram_tables
    )

  h_emb = params["embed.weight"][input_ids].astype(BF16)
  h = np.repeat(h_emb[:, :, None, :], cfg.hc_mult, axis=2)
  pre_mix = make_identity_pre_mix(bsz, seqlen, cfg.hc_mult)
  shared = SharedAttnState()
  main_hiddens = []

  for l in range(cfg.n_layers):
    if cfg.has_engram(l):
      assert engram_hashes is not None
      idx_e = cfg.engram_layer_ids.index(l)
      epre = f"layers.{l}.engram."
      h = engram_forward(
          h,
          engram_hashes[:, :, idx_e, :],
          params[epre + "embed.weight"],
          params[epre + "embed.scale"],
          params[epre + "wkv.weight"],
          params[epre + "wkv.scale"],
          params[epre + "q_weight"],
          params[epre + "k_weight"],
          cfg.norm_eps,
      )
    if l in cfg.dspark_target_layer_ids:
      main_hiddens.append(np.mean(h.astype(np.float32), axis=2).astype(BF16))
    h, pre_mix = block_forward(
        h,
        start_pos,
        pre_mix,
        l,
        cfg,
        params,
        state.layer_caches[l],
        shared,
        state.cos_swa,
        state.sin_swa,
        state.cos_comp,
        state.sin_comp,
        write_global_kv=write_global_kv,
        swa_window_floor=swa_window_floor,
    )

  h_out = hc_pre(h, pre_mix)
  if not full_logits:
    h_out = h_out[:, -1:, :]
  normed = rms_norm(h_out, params["norm.weight"], cfg.norm_eps)
  logits = (
      normed.astype(np.float32) @ params["head.weight"].astype(np.float32).T
  )
  if not full_logits:
    logits = logits[:, 0, :]
  output_ids = np.argmax(logits, axis=-1).astype(np.int32)
  main_hidden = np.concatenate(main_hiddens, axis=-1) if main_hiddens else None
  return output_ids, logits, main_hidden


def prefill_ced_bounded_replay(
    input_ids: np.ndarray,
    state: ReferenceState,
    params: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  cfg = state.cfg
  bsz, seqlen = input_ids.shape
  win = cfg.window_size
  dec_start = cfg.decoder_start_layer
  if seqlen <= win:
    return forward_backbone(input_ids, 0, state, params)

  prefix_len = seqlen - win
  engram_hashes = None
  if state.engram_tables is not None:
    engram_hashes = compute_engram_hashes(
        input_ids, 0, state.engram_history, state.engram_tables
    )

  h_emb = params["embed.weight"][input_ids].astype(BF16)
  h = np.repeat(h_emb[:, :, None, :], cfg.hc_mult, axis=2)
  pre_mix = make_identity_pre_mix(bsz, seqlen, cfg.hc_mult)
  shared = SharedAttnState()

  for l in range(dec_start):
    if cfg.has_engram(l):
      assert engram_hashes is not None
      idx_e = cfg.engram_layer_ids.index(l)
      epre = f"layers.{l}.engram."
      h = engram_forward(
          h,
          engram_hashes[:, :, idx_e, :],
          params[epre + "embed.weight"],
          params[epre + "embed.scale"],
          params[epre + "wkv.weight"],
          params[epre + "wkv.scale"],
          params[epre + "q_weight"],
          params[epre + "k_weight"],
          cfg.norm_eps,
      )
    h, pre_mix = block_forward(
        h,
        0,
        pre_mix,
        l,
        cfg,
        params,
        state.layer_caches[l],
        shared,
        state.cos_swa,
        state.sin_swa,
        state.cos_comp,
        state.sin_comp,
    )

  x_enc_prefix = rms_norm(
      hc_pre(h[:, :prefix_len], pre_mix[:, :prefix_len]),
      params[f"layers.{dec_start}.attn_norm.weight"],
      cfg.norm_eps,
  )
  c_pre = f"layers.{dec_start}.attn.compressor."
  lat_prefix = compressor_forward(
      x_enc_prefix,
      0,
      1,
      params[c_pre + "wkv.weight"],
      params[c_pre + "norm.weight"],
      cfg.norm_eps,
  )
  assert lat_prefix is not None
  ipre = f"layers.{dec_start}.attn.indexer."
  k_idx_prefix = rms_norm(
      linear_dense(lat_prefix, params[ipre + "wk.weight"]),
      params[ipre + "k_norm.weight"],
      cfg.norm_eps,
  )
  pos_prefix = np.arange(prefix_len, dtype=np.int32)
  k_idx_prefix = apply_rotary_emb(
      k_idx_prefix,
      state.cos_comp[pos_prefix],
      state.sin_comp[pos_prefix],
      rope_dim=cfg.rope_head_dim,
  )
  k_idx_prefix = np.asarray(
      fp4_act_quant(k_idx_prefix, FP4_BLOCK, inplace=True, scale_dtype="e8m0")
  )
  dec_cache = state.layer_caches[dec_start]
  assert dec_cache.index_k is not None and dec_cache.compress_kv is not None
  dec_cache.index_k[:bsz, :prefix_len] = k_idx_prefix

  rot_lat_prefix = apply_rotary_emb(
      lat_prefix,
      state.cos_comp[pos_prefix],
      state.sin_comp[pos_prefix],
      rope_dim=cfg.rope_head_dim,
  )
  rot_lat_prefix = np.asarray(
      fp4_act_quant(
          rot_lat_prefix, KV_FP4_BLOCK, inplace=True, scale_dtype="e4m3"
      )
  )
  dec_cache.compress_kv[:bsz, :prefix_len] = rot_lat_prefix

  h_tail = h[:, prefix_len:]
  pre_mix_tail = pre_mix[:, prefix_len:]
  main_hiddens = []
  for l in range(dec_start, cfg.n_layers):
    if l in cfg.dspark_target_layer_ids:
      main_hiddens.append(
          np.mean(h_tail.astype(np.float32), axis=2).astype(BF16)
      )
    h_tail, pre_mix_tail = block_forward(
        h_tail,
        prefix_len,
        pre_mix_tail,
        l,
        cfg,
        params,
        state.layer_caches[l],
        shared,
        state.cos_swa,
        state.sin_swa,
        state.cos_comp,
        state.sin_comp,
    )

  h_out = hc_pre(h_tail, pre_mix_tail)[:, -1:, :]
  normed = rms_norm(h_out, params["norm.weight"], cfg.norm_eps)
  logits = (
      normed.astype(np.float32) @ params["head.weight"].astype(np.float32).T
  )[:, 0, :]
  output_ids = np.argmax(logits, axis=-1).astype(np.int32)
  main_hidden = np.concatenate(main_hiddens, axis=-1) if main_hiddens else None
  return output_ids, logits, main_hidden


def make_random_params(cfg: Ds4Config, seed: int = 0) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  p: dict[str, np.ndarray] = {}

  def rand_bf16(shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(BF16)

  def rand_fp8(out_dim: int, in_dim: int) -> tuple[np.ndarray, np.ndarray]:
    raw = rng.standard_normal((out_dim, in_dim), dtype=np.float32) * 0.05
    w_q = np.clip(raw * 64.0, -FP8_MAX, FP8_MAX).astype(FP8)
    s_u8 = np.full(
        (out_dim // FP8_BLOCK, in_dim // FP8_BLOCK), 127 - 6, dtype=np.uint8
    )
    return w_q, s_u8

  def rand_fp4(out_dim: int, in_dim: int) -> tuple[np.ndarray, np.ndarray]:
    nibbles = rng.integers(0, 16, size=(out_dim, in_dim), dtype=np.uint8)
    packed = pack_e2m1_nibbles(nibbles)
    s_u8 = np.full((out_dim, in_dim // FP4_BLOCK), 127 - 4, dtype=np.uint8)
    return packed, s_u8

  p["embed.weight"] = rand_bf16((cfg.vocab_size, cfg.dim), 0.02)
  p["norm.weight"] = np.ones((cfg.dim,), dtype=BF16)
  p["head.weight"] = rand_bf16((cfg.vocab_size, cfg.dim), 0.02)

  for l in range(cfg.total_layers):
    is_mtp = l >= cfg.n_layers
    prefix = f"layers.{l}." if not is_mtp else f"mtp.{l - cfg.n_layers}."
    apre = prefix + "attn."
    p[prefix + "attn_norm.weight"] = np.ones((cfg.dim,), dtype=BF16)
    p[prefix + "ffn_norm.weight"] = np.ones((cfg.dim,), dtype=BF16)
    p[prefix + "hc_attn_fn"] = (
        rng.standard_normal((cfg.mix_hc, cfg.hc_dim), dtype=np.float32) * 0.01
    )
    p[prefix + "hc_ffn_fn"] = (
        rng.standard_normal((cfg.mix_hc, cfg.hc_dim), dtype=np.float32) * 0.01
    )
    p[prefix + "hc_attn_base"] = np.zeros((cfg.mix_hc,), dtype=np.float32)
    p[prefix + "hc_ffn_base"] = np.zeros((cfg.mix_hc,), dtype=np.float32)
    p[prefix + "hc_attn_scale"] = np.ones((3,), dtype=np.float32) * 0.1
    p[prefix + "hc_ffn_scale"] = np.ones((3,), dtype=np.float32) * 0.1

    p[apre + "attn_sink"] = np.zeros((cfg.n_heads,), dtype=np.float32)
    p[apre + "wq_a.weight"], p[apre + "wq_a.scale"] = rand_fp8(
        cfg.q_lora_rank, cfg.dim
    )
    p[apre + "q_norm.weight"] = np.ones((cfg.q_lora_rank,), dtype=BF16)
    p[apre + "wq_b.weight"], p[apre + "wq_b.scale"] = rand_fp8(
        cfg.n_heads * cfg.head_dim, cfg.q_lora_rank
    )
    p[apre + "wkv.weight"], p[apre + "wkv.scale"] = rand_fp8(
        cfg.head_dim, cfg.dim
    )
    p[apre + "kv_norm.weight"] = np.ones((cfg.head_dim,), dtype=BF16)
    p[apre + "wo_a.weight"], p[apre + "wo_a.scale"] = rand_fp8(
        cfg.o_groups * cfg.o_lora_rank,
        (cfg.n_heads * cfg.head_dim) // cfg.o_groups,
    )
    p[apre + "wo_b.weight"], p[apre + "wo_b.scale"] = rand_fp8(
        cfg.dim, cfg.o_groups * cfg.o_lora_rank
    )

    if cfg.is_kv_source(l):
      cpre = apre + "compressor."
      ratio = cfg.compress_ratios[l]
      p[cpre + "norm.weight"] = np.ones((cfg.head_dim,), dtype=BF16)
      wkv_dt = np.float32 if ratio > 1 else BF16
      p[cpre + "wkv.weight"] = (
          rng.standard_normal((cfg.head_dim, cfg.dim), dtype=np.float32) * 0.02
      ).astype(wkv_dt)
      if ratio > 1:
        p[cpre + "wgate.weight"] = (
            rng.standard_normal((cfg.head_dim, cfg.dim), dtype=np.float32)
            * 0.02
        )

    if cfg.is_index_source(l):
      ipre = apre + "indexer."
      p[ipre + "wq_b.weight"], p[ipre + "wq_b.scale"] = rand_fp8(
          cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank
      )
      p[ipre + "weights_proj.weight"] = rand_bf16(
          (cfg.index_n_heads, cfg.dim), 0.02
      )
      if cfg.is_kv_source(l):
        p[ipre + "wk.weight"] = rand_bf16(
            (cfg.index_head_dim, cfg.head_dim), 0.02
        )
        p[ipre + "k_norm.weight"] = np.ones((cfg.index_head_dim,), dtype=BF16)

    if not is_mtp and cfg.has_engram(l):
      idx_e = cfg.engram_layer_ids.index(l)
      n_emb = cfg.engram_num_embeddings[idx_e]
      epre = prefix + "engram."
      p[epre + "embed.weight"] = (
          rng.standard_normal((n_emb, cfg.engram_head_dim), dtype=np.float32)
          * 0.5
      ).astype(FP8)
      p[epre + "embed.scale"] = np.full(
          (n_emb, cfg.engram_head_dim // FP8_BLOCK), 127 - 4, dtype=np.uint8
      )
      p[epre + "wkv.weight"], p[epre + "wkv.scale"] = rand_fp8(
          cfg.dim * (cfg.hc_mult + 1),
          cfg.engram_n_hash_cols * cfg.engram_head_dim,
      )
      p[epre + "q_weight"] = np.ones((cfg.hc_mult, cfg.dim), dtype=BF16)
      p[epre + "k_weight"] = np.ones((cfg.hc_mult, cfg.dim), dtype=BF16)

    fpre = prefix + "ffn."
    n_routed, _ = cfg.get_moe_config(l)
    p[fpre + "gate.weight"] = rand_bf16((n_routed, cfg.dim), 0.02)
    p[fpre + "gate.bias"] = np.zeros((n_routed,), dtype=np.float32)
    p[fpre + "gate.bias_vl"] = np.zeros((n_routed,), dtype=np.float32)
    for proj, (o_d, i_d) in (
        ("w1", (cfg.moe_inter_dim, cfg.dim)),
        ("w2", (cfg.dim, cfg.moe_inter_dim)),
        ("w3", (cfg.moe_inter_dim, cfg.dim)),
    ):
      (
          p[f"{fpre}shared_experts.{proj}.weight"],
          p[f"{fpre}shared_experts.{proj}.scale"],
      ) = rand_fp8(o_d, i_d)
      for e in range(n_routed):
        (
            p[f"{fpre}experts.{e}.{proj}.weight"],
            p[f"{fpre}experts.{e}.{proj}.scale"],
        ) = rand_fp4(o_d, i_d)

    if is_mtp:
      stage_id = l - cfg.n_layers
      if stage_id == 0:
        p["mtp.0.main_proj.weight"], p["mtp.0.main_proj.scale"] = rand_fp8(
            cfg.dim, len(cfg.dspark_target_layer_ids) * cfg.dim
        )
        p["mtp.0.main_norm.weight"] = np.ones((cfg.dim,), dtype=BF16)
      if stage_id == cfg.n_mtp_layers - 1:
        p[prefix + "norm.weight"] = np.ones((cfg.dim,), dtype=BF16)
        p[prefix + "markov_head.embed.weight"] = rand_bf16(
            (cfg.vocab_size, cfg.dspark_markov_rank), 0.02
        )
        p[prefix + "markov_head.head.weight"] = rand_bf16(
            (cfg.vocab_size, cfg.dspark_markov_rank), 0.02
        )
        p[prefix + "confidence_head.proj.weight"] = rand_bf16(
            (1, cfg.dim + cfg.dspark_markov_rank), 0.02
        )

  return p
