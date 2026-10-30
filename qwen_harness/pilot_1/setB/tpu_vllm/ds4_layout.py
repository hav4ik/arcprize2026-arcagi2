"""Per-device weight layout and mesh topology for DeepSeek-V4.1-Flash-1017.

Defines how the 40-layer Causal Encoder-Decoder backbone, 2 Engram conditional
memory modules (L1, L14), and 3-layer DSpark speculative drafter (mtp.0..2) are
sharded across N TensorCore devices (N in {1, 2, 4, 8, 16, 32}):

  * Hybrid Sharding:
    - Dense Attention (wq_b, wo_a, wo_b, attn_sink) and Shared Expert (w1, w2,
    w3)
      shard across dense_tp = gcd(N, o_groups) devices (1, 2, 4, or 8; on N=16
      each 8-device host forms a complete dense_tp=8 replica).
    - Routed MoE Experts (384 in backbone, 128 in DSpark, moe_inter_dim=2304)
      shard across expert_ep = N // expert_tp groups on the expert axis and
      expert_tp in {1, 2} on the intermediate axis (1152 cols per chiplet at
      expert_tp=2, zero 128-lane padding waste).
    - Vocabulary (embed, head, markov_head) and Engram tables shard across all
      N devices.
    - Routed expert weights are packed into uint32 [K // 8, N] (8 E2M1 nibbles
      per uint32 word along K) with uint8 [K // 32, N] E8M0 scales.
"""

from __future__ import annotations

# pylint: disable=invalid-name

from collections.abc import Mapping
import dataclasses
import math
from typing import Any

import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_ref as ref

BF16 = ref.BF16
FP8 = ref.FP8
F32 = ref.F32
U8 = np.dtype(np.uint8)
U32 = np.dtype(np.uint32)
AXIS = "tp"


def cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
  return cdiv(a, b) * b


def host_key(device: Any) -> int:
  coords = getattr(device, "coords", None)
  if coords is not None and len(coords) >= 3:
    return int(coords[2])
  return int(getattr(device, "process_index", 0))


def chip_key(device: Any) -> tuple[int, ...]:
  coords = getattr(device, "coords", None)
  if coords is not None and len(coords) >= 3:
    return (int(coords[2]), int(coords[1]), int(coords[0]))
  return (int(getattr(device, "process_index", 0)), int(device.id) // 2)


def rank_sort_key(device: Any) -> tuple[int, ...]:
  """Orders TPU devices by (z, y, x, core) so chip pairs are (2k, 2k+1)."""
  coords = getattr(device, "coords", None)
  if coords is not None and len(coords) >= 3:
    core = (
        int(coords[3])
        if len(coords) >= 4
        else int(getattr(device, "core_on_chip", int(device.id) % 2))
    )
    return (int(coords[2]), int(coords[1]), int(coords[0]), core)
  return (int(getattr(device, "process_index", 0)), int(device.id))


def pack_expert_bytes(w_packed_u8: np.ndarray) -> np.ndarray:
  """Converts checkpoint [N, K // 2] uint8 FP4 bytes -> kernel [K // 8, N] uint32."""
  u8 = np.ascontiguousarray(w_packed_u8, dtype=np.uint8)
  if u8.ndim != 2 or u8.shape[1] % 4 != 0:
    raise ValueError(
        f"Expected 2D uint8 array with cols divisible by 4, got {u8.shape}"
    )
  return u8.view("<u4").T.copy()


def unpack_expert_u32(
    packed_u32: np.ndarray, scale_u8: np.ndarray
) -> np.ndarray:
  """Unpacks [K // 8, N] uint32 + [K // 32, N] uint8 E8M0 scale to [K, N] float32."""
  u32 = np.ascontiguousarray(packed_u32, dtype=np.uint32)
  # Transpose back to [N, K // 8] uint32 -> view as [N, K // 2] uint8
  u8_nk = np.ascontiguousarray(u32.T).view(np.uint8)
  vals_nk = ref.unpack_e2m1_bytes(u8_nk)  # [N, K]
  vals_kn = vals_nk.T  # [K, N]
  k_dim, n_dim = vals_kn.shape
  s_f32 = ref.e8m0_to_f32(scale_u8).reshape(k_dim // ref.FP4_BLOCK, 1, n_dim)
  return (
      vals_kn.reshape(k_dim // ref.FP4_BLOCK, ref.FP4_BLOCK, n_dim) * s_f32
  ).reshape(k_dim, n_dim)


@dataclasses.dataclass(frozen=True)
class DeviceLayout:
  """Per-device sharding specification for DeepSeek-V4.1-Flash-1017."""

  cfg: ref.Ds4Config
  num_devices: int
  expert_tp: int
  expert_ep: int
  dense_tp: int

  def __post_init__(self):
    c = self.cfg
    if self.num_devices < 1:
      raise ValueError(f"num_devices={self.num_devices} must be >= 1")
    if self.expert_tp * self.expert_ep != self.num_devices:
      raise ValueError(
          f"expert_tp ({self.expert_tp}) * expert_ep ({self.expert_ep}) != "
          f"num_devices ({self.num_devices})"
      )
    if c.moe_inter_dim % (self.expert_tp * ref.FP4_BLOCK) != 0:
      raise ValueError(
          f"moe_inter_dim ({c.moe_inter_dim}) not divisible by "
          f"expert_tp * 32 ({self.expert_tp * ref.FP4_BLOCK})"
      )
    if c.n_routed_experts % self.expert_ep != 0:
      raise ValueError(
          f"n_routed_experts ({c.n_routed_experts}) not divisible by "
          f"expert_ep ({self.expert_ep})"
      )
    if c.dspark_n_routed_experts % self.expert_ep != 0:
      raise ValueError(
          f"dspark_n_routed_experts ({c.dspark_n_routed_experts}) not divisible"
          f" by expert_ep ({self.expert_ep})"
      )
    if (
        self.num_devices % self.dense_tp != 0
        or c.o_groups % self.dense_tp != 0
        or c.n_heads % self.dense_tp != 0
    ):
      raise ValueError(
          f"Invalid dense_tp={self.dense_tp} for"
          f" num_devices={self.num_devices}, o_groups={c.o_groups}"
      )

  @property
  def heads_per_device(self) -> int:
    return self.cfg.n_heads // self.dense_tp

  @property
  def o_groups_per_device(self) -> int:
    return self.cfg.o_groups // self.dense_tp

  @property
  def o_lora_per_device(self) -> int:
    return self.o_groups_per_device * self.cfg.o_lora_rank

  @property
  def group_head_cols(self) -> int:
    return (self.cfg.n_heads // self.cfg.o_groups) * self.cfg.head_dim

  @property
  def shared_inter_valid_per_device(self) -> int:
    return self.cfg.moe_inter_dim

  @property
  def shared_inter_per_device(self) -> int:
    return round_up(self.shared_inter_valid_per_device, 128)

  @property
  def expert_inter_per_device(self) -> int:
    return self.cfg.moe_inter_dim // self.expert_tp

  @property
  def local_backbone_experts(self) -> int:
    return self.cfg.n_routed_experts // self.expert_ep

  @property
  def local_dspark_experts(self) -> int:
    return self.cfg.dspark_n_routed_experts // self.expert_ep

  @property
  def vocab_shard(self) -> int:
    return round_up(cdiv(self.cfg.vocab_size, self.num_devices), 128)

  def dense_rank(self, device_id: int) -> int:
    return device_id % self.dense_tp

  def contributes_dense(self, device_id: int) -> bool:
    """True when device_id is in the primary dense_tp replica group."""
    return device_id < self.dense_tp

  def expert_group(self, device_id: int) -> int:
    return device_id // self.expert_tp

  def expert_tp_rank(self, device_id: int) -> int:
    return device_id % self.expert_tp

  def backbone_experts_of(self, device_id: int) -> range:
    g = self.expert_group(device_id)
    e_loc = self.local_backbone_experts
    return range(g * e_loc, (g + 1) * e_loc)

  def dspark_experts_of(self, device_id: int) -> range:
    g = self.expert_group(device_id)
    e_loc = self.local_dspark_experts
    return range(g * e_loc, (g + 1) * e_loc)

  def vocab_range(self, device_id: int) -> tuple[int, int]:
    vs = self.vocab_shard
    start = min(device_id * vs, self.cfg.vocab_size)
    end = min((device_id + 1) * vs, self.cfg.vocab_size)
    return start, end

  def engram_rows_per_device(self, engram_idx: int) -> int:
    total_rows = self.cfg.engram_num_embeddings[engram_idx]
    return cdiv(total_rows, self.num_devices)

  def engram_range(self, engram_idx: int, device_id: int) -> tuple[int, int]:
    total_rows = self.cfg.engram_num_embeddings[engram_idx]
    shard = self.engram_rows_per_device(engram_idx)
    start = min(device_id * shard, total_rows)
    end = min((device_id + 1) * shard, total_rows)
    return start, end

  def abstract_device_arrays(
      self,
  ) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Returns per-device {array_name: (shape, dtype)} for all weight stacks."""
    c = self.cfg
    L = c.n_layers
    M = c.n_mtp_layers
    B_all = L + M
    d = c.dim
    ql = c.q_lora_rank
    hd = c.head_dim
    h_loc = self.heads_per_device
    og_loc = self.o_groups_per_device
    ol_loc = self.o_lora_per_device
    ghc = self.group_head_cols
    si_loc = self.shared_inter_per_device
    ei_loc = self.expert_inter_per_device
    eb_loc = self.local_backbone_experts
    ed_loc = self.local_dspark_experts
    vs = self.vocab_shard
    fb = ref.FP8_BLOCK
    pb = ref.FP4_BLOCK

    kv_m2_layers = sum(
        1 for l in c.kv_source_layers if c.compress_ratios[l] > 1
    )
    kv_m1_layers = sum(
        1 for l in c.kv_source_layers if c.compress_ratios[l] == 1
    )
    idx_full_layers = len(c.kv_source_layers)
    idx_all_layers = len(c.index_source_layers)

    specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {
        # Vocabulary & Final Norm
        "embed": ((vs, d), BF16),
        "norm": ((d,), BF16),
        "head": ((d, vs), BF16),
        # Stacked across all L + M = 43 blocks:
        # Norms & mHC
        "attn_norm": ((B_all, d), BF16),
        "ffn_norm": ((B_all, d), BF16),
        "hc_attn_fn": ((B_all, c.hc_dim, c.mix_hc), F32),
        "hc_ffn_fn": ((B_all, c.hc_dim, c.mix_hc), F32),
        "hc_attn_base": ((B_all, c.mix_hc), F32),
        "hc_ffn_base": ((B_all, c.mix_hc), F32),
        "hc_attn_scale": ((B_all, 3), F32),
        "hc_ffn_scale": ((B_all, 3), F32),
        # Core Attention Projections (transposed to [in, out])
        "attn_sink": ((B_all, h_loc), F32),
        "q_norm": ((B_all, ql), BF16),
        "kv_norm": ((B_all, hd), BF16),
        "wq_a_w": ((B_all, d, ql), FP8),
        "wq_a_s": ((B_all, d // fb, ql), U8),
        "wq_b_w": ((B_all, ql, h_loc * hd), FP8),
        "wq_b_s": ((B_all, ql // fb, h_loc * hd), U8),
        "wkv_w": ((B_all, d, hd), FP8),
        "wkv_s": ((B_all, d // fb, hd), U8),
        "wo_a_w": ((B_all, og_loc, ghc, c.o_lora_rank), FP8),
        "wo_a_s": ((B_all, og_loc, ghc // fb, c.o_lora_rank), U8),
        "wo_b_w": ((B_all, ol_loc, d), FP8),
        "wo_b_s": ((B_all, ol_loc // fb, d), U8),
        # Shared Experts (stacked across L + M blocks, fused [w1 | w3]).
        "shared_gu_w": ((B_all, d, 2 * si_loc), FP8),
        "shared_gu_s": ((B_all, d // fb, 2 * si_loc), U8),
        "shared_dn_w": ((B_all, si_loc, d), FP8),
        "shared_dn_s": ((B_all, si_loc // fb, d), U8),
        # Backbone Routed Experts (stacked across L = 40 layers)
        "bb_gate_w": ((L, d, c.n_routed_experts), BF16),
        "bb_gate_bias": ((L, c.n_routed_experts), F32),
        "bb_gate_bias_vl": ((L, c.n_routed_experts), F32),
        "bb_exp_gu_w": ((L, eb_loc, d // 8, 2 * ei_loc), U32),
        "bb_exp_gu_s": ((L, eb_loc, d // pb, 2 * ei_loc), U8),
        "bb_exp_dn_w": ((L, eb_loc, ei_loc // 8, d), U32),
        "bb_exp_dn_s": ((L, eb_loc, ei_loc // pb, d), U8),
        # CSA2 Compressors (m=2 on L2,8,14; m=1 on L20)
        "comp_m2_wkv": ((max(1, kv_m2_layers), d, hd), F32),
        "comp_m2_wgate": ((max(1, kv_m2_layers), d, hd), F32),
        "comp_m2_norm": ((max(1, kv_m2_layers), hd), BF16),
        "comp_m1_wkv": ((max(1, kv_m1_layers), d, hd), BF16),
        "comp_m1_norm": ((max(1, kv_m1_layers), hd), BF16),
        # CSA2 Indexers (wk/k_norm on kv_source; wq_b/weights_proj on index).
        "idx_wk": ((max(1, idx_full_layers), hd, c.index_head_dim), BF16),
        "idx_k_norm": ((max(1, idx_full_layers), c.index_head_dim), BF16),
        "idx_wq_b_w": (
            (max(1, idx_all_layers), ql, c.index_n_heads * c.index_head_dim),
            FP8,
        ),
        "idx_wq_b_s": (
            (
                max(1, idx_all_layers),
                ql // fb,
                c.index_n_heads * c.index_head_dim,
            ),
            U8,
        ),
        "idx_weights_proj": (
            (max(1, idx_all_layers), d, c.index_n_heads),
            BF16,
        ),
    }

    for idx_e in range(len(c.engram_layer_ids)):
      rows_loc = self.engram_rows_per_device(idx_e)
      ehd = c.engram_head_dim
      nhc = c.engram_n_hash_cols
      specs[f"engram_{idx_e}_embed_w"] = ((rows_loc, ehd), FP8)
      specs[f"engram_{idx_e}_embed_s"] = ((rows_loc, ehd // fb), U8)
      specs[f"engram_{idx_e}_wkv_w"] = (
          (nhc * ehd, d * (c.hc_mult + 1)),
          FP8,
      )
      specs[f"engram_{idx_e}_wkv_s"] = (
          ((nhc * ehd) // fb, d * (c.hc_mult + 1)),
          U8,
      )
      specs[f"engram_{idx_e}_q_weight"] = ((c.hc_mult, d), BF16)
      specs[f"engram_{idx_e}_k_weight"] = ((c.hc_mult, d), BF16)

    if M > 0:
      specs.update({
          "mtp_main_proj_w": ((len(c.dspark_target_layer_ids) * d, d), FP8),
          "mtp_main_proj_s": (
              ((len(c.dspark_target_layer_ids) * d) // fb, d),
              U8,
          ),
          "mtp_main_norm": ((d,), BF16),
          "mtp_final_norm": ((d,), BF16),
          "mtp_markov_embed": ((vs, c.dspark_markov_rank), BF16),
          "mtp_markov_head": ((c.dspark_markov_rank, vs), BF16),
          "mtp_confidence_w": ((d + c.dspark_markov_rank, 1), BF16),
          "mtp_gate_w": ((M, d, c.dspark_n_routed_experts), BF16),
          "mtp_gate_bias": ((M, c.dspark_n_routed_experts), F32),
          "mtp_gate_bias_vl": ((M, c.dspark_n_routed_experts), F32),
          "mtp_exp_gu_w": ((M, ed_loc, d // 8, 2 * ei_loc), U32),
          "mtp_exp_gu_s": ((M, ed_loc, d // pb, 2 * ei_loc), U8),
          "mtp_exp_dn_w": ((M, ed_loc, ei_loc // 8, d), U32),
          "mtp_exp_dn_s": ((M, ed_loc, ei_loc // pb, d), U8),
      })
    return specs


def build_layout(
    cfg: ref.Ds4Config,
    num_devices: int,
    *,
    expert_tp: int | None = None,
    expert_ep: int | None = None,
    dense_tp: int | None = None,
) -> DeviceLayout:
  """Constructs a validated DeviceLayout for num_devices."""
  if expert_tp is None:
    expert_tp = 1
  if expert_ep is None:
    expert_ep = num_devices // expert_tp
  if dense_tp is None:
    dense_tp = math.gcd(num_devices, cfg.o_groups)
  return DeviceLayout(
      cfg=cfg,
      num_devices=num_devices,
      expert_tp=expert_tp,
      expert_ep=expert_ep,
      dense_tp=dense_tp,
  )


def build_device_arrays(
    params: Mapping[str, np.ndarray],
    lay: DeviceLayout,
    device_id: int,
) -> dict[str, np.ndarray]:
  """Slices flat checkpoint params into one device's stacked HBM arrays."""
  c = lay.cfg
  L = c.n_layers
  M = c.n_mtp_layers
  B_all = L + M
  hd = c.head_dim
  fb = ref.FP8_BLOCK
  pb = ref.FP4_BLOCK
  dr = lay.dense_rank(device_id)
  etp_r = lay.expert_tp_rank(device_id)

  specs = lay.abstract_device_arrays()
  out = {name: np.zeros(sh, dtype=dt) for name, (sh, dt) in specs.items()}

  # 1. Vocab & Final Norm
  v0, v1 = lay.vocab_range(device_id)
  if v1 > v0:
    out["embed"][: v1 - v0] = params["embed.weight"][v0:v1]
    out["head"][:, : v1 - v0] = params["head.weight"][v0:v1].T
  out["norm"][...] = params["norm.weight"]

  h_loc = lay.heads_per_device
  h0, h1 = dr * h_loc, (dr + 1) * h_loc
  og_loc = lay.o_groups_per_device
  og0, og1 = dr * og_loc, (dr + 1) * og_loc
  ol_loc = lay.o_lora_per_device
  ol0, ol1 = dr * ol_loc, (dr + 1) * ol_loc
  si_val = lay.shared_inter_valid_per_device
  si_loc = lay.shared_inter_per_device
  si0, si1 = 0, si_val
  ei_loc = lay.expert_inter_per_device
  ei0, ei1 = etp_r * ei_loc, (etp_r + 1) * ei_loc

  m2_slot = 0
  m1_slot = 0
  idx_full_slot = 0
  idx_all_slot = 0

  for b in range(B_all):
    is_mtp = b >= L
    prefix = f"layers.{b}." if not is_mtp else f"mtp.{b - L}."
    apre = prefix + "attn."
    fpre = prefix + "ffn."

    out["attn_norm"][b] = params[prefix + "attn_norm.weight"]
    out["ffn_norm"][b] = params[prefix + "ffn_norm.weight"]
    out["hc_attn_fn"][b] = params[prefix + "hc_attn_fn"].T
    out["hc_ffn_fn"][b] = params[prefix + "hc_ffn_fn"].T
    out["hc_attn_base"][b] = params[prefix + "hc_attn_base"]
    out["hc_ffn_base"][b] = params[prefix + "hc_ffn_base"]
    out["hc_attn_scale"][b] = params[prefix + "hc_attn_scale"]
    out["hc_ffn_scale"][b] = params[prefix + "hc_ffn_scale"]

    out["attn_sink"][b] = params[apre + "attn_sink"][h0:h1]
    out["q_norm"][b] = params[apre + "q_norm.weight"]
    out["kv_norm"][b] = params[apre + "kv_norm.weight"]

    out["wq_a_w"][b] = params[apre + "wq_a.weight"].T
    out["wq_a_s"][b] = np.repeat(params[apre + "wq_a.scale"].T, fb, axis=1)
    out["wq_b_w"][b] = params[apre + "wq_b.weight"][h0 * hd : h1 * hd].T
    out["wq_b_s"][b] = np.repeat(
        params[apre + "wq_b.scale"][(h0 * hd) // fb : (h1 * hd) // fb].T,
        fb,
        axis=1,
    )
    out["wkv_w"][b] = params[apre + "wkv.weight"].T
    out["wkv_s"][b] = np.repeat(params[apre + "wkv.scale"].T, fb, axis=1)

    # wo_a: [o_groups * o_lora_rank, ghc] -> [og_loc, ghc, o_lora_rank]
    wo_a_w = params[apre + "wo_a.weight"].reshape(
        c.o_groups, c.o_lora_rank, lay.group_head_cols
    )[og0:og1]
    wo_a_s = params[apre + "wo_a.scale"].reshape(
        c.o_groups, c.o_lora_rank // fb, lay.group_head_cols // fb
    )[og0:og1]
    out["wo_a_w"][b] = np.transpose(wo_a_w, (0, 2, 1))
    out["wo_a_s"][b] = np.repeat(np.transpose(wo_a_s, (0, 2, 1)), fb, axis=2)

    out["wo_b_w"][b] = params[apre + "wo_b.weight"][:, ol0:ol1].T
    out["wo_b_s"][b] = np.repeat(
        params[apre + "wo_b.scale"][:, ol0 // fb : ol1 // fb].T, fb, axis=1
    )

    # Shared experts: fuse [w1 | w3] along output columns, padded to si_loc
    out["shared_gu_s"][b].fill(127)
    out["shared_dn_s"][b].fill(127)
    out["shared_gu_w"][b, :, :si_val] = params[
        fpre + "shared_experts.w1.weight"
    ][si0:si1].T
    out["shared_gu_w"][b, :, si_loc : si_loc + si_val] = params[
        fpre + "shared_experts.w3.weight"
    ][si0:si1].T
    out["shared_gu_s"][b, :, :si_val] = np.repeat(
        params[fpre + "shared_experts.w1.scale"][si0 // fb : si1 // fb].T,
        fb,
        axis=1,
    )
    out["shared_gu_s"][b, :, si_loc : si_loc + si_val] = np.repeat(
        params[fpre + "shared_experts.w3.scale"][si0 // fb : si1 // fb].T,
        fb,
        axis=1,
    )
    out["shared_dn_w"][b, :si_val, :] = params[
        fpre + "shared_experts.w2.weight"
    ][:, si0:si1].T
    out["shared_dn_s"][b, : si_val // fb, :] = np.repeat(
        params[fpre + "shared_experts.w2.scale"][:, si0 // fb : si1 // fb].T,
        fb,
        axis=1,
    )

    if not is_mtp:
      # Backbone routed experts
      out["bb_gate_w"][b] = params[fpre + "gate.weight"].T
      out["bb_gate_bias"][b] = params[fpre + "gate.bias"]
      out["bb_gate_bias_vl"][b] = params[fpre + "gate.bias_vl"]
      for loc_i, e in enumerate(lay.backbone_experts_of(device_id)):
        epre = f"{fpre}experts.{e}."
        w1_u32 = pack_expert_bytes(params[epre + "w1.weight"][ei0:ei1])
        w3_u32 = pack_expert_bytes(params[epre + "w3.weight"][ei0:ei1])
        s1_u8 = params[epre + "w1.scale"][ei0:ei1].T
        s3_u8 = params[epre + "w3.scale"][ei0:ei1].T
        out["bb_exp_gu_w"][b, loc_i] = np.concatenate([w1_u32, w3_u32], axis=1)
        out["bb_exp_gu_s"][b, loc_i] = np.concatenate([s1_u8, s3_u8], axis=1)
        # w2: [dim, moe_inter_dim // 2] -> slice cols [ei0 // 2 : ei1 // 2]
        out["bb_exp_dn_w"][b, loc_i] = pack_expert_bytes(
            params[epre + "w2.weight"][:, ei0 // 2 : ei1 // 2]
        )
        out["bb_exp_dn_s"][b, loc_i] = params[epre + "w2.scale"][
            :, ei0 // pb : ei1 // pb
        ].T

      if c.is_kv_source(b):
        cpre = apre + "compressor."
        if c.compress_ratios[b] > 1:
          out["comp_m2_wkv"][m2_slot] = (
              params[cpre + "wkv.weight"].astype(F32).T
          )
          out["comp_m2_wgate"][m2_slot] = (
              params[cpre + "wgate.weight"].astype(F32).T
          )
          out["comp_m2_norm"][m2_slot] = params[cpre + "norm.weight"]
          m2_slot += 1
        else:
          out["comp_m1_wkv"][m1_slot] = (
              params[cpre + "wkv.weight"].astype(BF16).T
          )
          out["comp_m1_norm"][m1_slot] = params[cpre + "norm.weight"]
          m1_slot += 1

        ipre = apre + "indexer."
        out["idx_wk"][idx_full_slot] = params[ipre + "wk.weight"].T
        out["idx_k_norm"][idx_full_slot] = params[ipre + "k_norm.weight"]
        idx_full_slot += 1

      if c.is_index_source(b):
        ipre = apre + "indexer."
        out["idx_wq_b_w"][idx_all_slot] = params[ipre + "wq_b.weight"].T
        out["idx_wq_b_s"][idx_all_slot] = np.repeat(
            params[ipre + "wq_b.scale"].T, fb, axis=1
        )
        out["idx_weights_proj"][idx_all_slot] = params[
            ipre + "weights_proj.weight"
        ].T
        idx_all_slot += 1
    else:
      s = b - L
      out["mtp_gate_w"][s] = params[fpre + "gate.weight"].T
      out["mtp_gate_bias"][s] = params[fpre + "gate.bias"]
      out["mtp_gate_bias_vl"][s] = params[fpre + "gate.bias_vl"]
      for loc_i, e in enumerate(lay.dspark_experts_of(device_id)):
        epre = f"{fpre}experts.{e}."
        w1_u32 = pack_expert_bytes(params[epre + "w1.weight"][ei0:ei1])
        w3_u32 = pack_expert_bytes(params[epre + "w3.weight"][ei0:ei1])
        s1_u8 = params[epre + "w1.scale"][ei0:ei1].T
        s3_u8 = params[epre + "w3.scale"][ei0:ei1].T
        out["mtp_exp_gu_w"][s, loc_i] = np.concatenate([w1_u32, w3_u32], axis=1)
        out["mtp_exp_gu_s"][s, loc_i] = np.concatenate([s1_u8, s3_u8], axis=1)
        out["mtp_exp_dn_w"][s, loc_i] = pack_expert_bytes(
            params[epre + "w2.weight"][:, ei0 // 2 : ei1 // 2]
        )
        out["mtp_exp_dn_s"][s, loc_i] = params[epre + "w2.scale"][
            :, ei0 // pb : ei1 // pb
        ].T

  # Engram layers
  for idx_e, lid in enumerate(c.engram_layer_ids):
    epre = f"layers.{lid}.engram."
    r0, r1 = lay.engram_range(idx_e, device_id)
    out[f"engram_{idx_e}_embed_s"].fill(127)  # pad scale = 1.0 (2^0)
    if r1 > r0:
      out[f"engram_{idx_e}_embed_w"][: r1 - r0] = params[epre + "embed.weight"][
          r0:r1
      ]
      out[f"engram_{idx_e}_embed_s"][: r1 - r0] = params[epre + "embed.scale"][
          r0:r1
      ]
    out[f"engram_{idx_e}_wkv_w"][...] = params[epre + "wkv.weight"].T
    out[f"engram_{idx_e}_wkv_s"][...] = np.repeat(
        params[epre + "wkv.scale"].T, fb, axis=1
    )
    out[f"engram_{idx_e}_q_weight"][...] = params[epre + "q_weight"]
    out[f"engram_{idx_e}_k_weight"][...] = params[epre + "k_weight"]

  if M > 0:
    out["mtp_main_proj_w"][...] = params["mtp.0.main_proj.weight"].T
    out["mtp_main_proj_s"][...] = np.repeat(
        params["mtp.0.main_proj.scale"].T, fb, axis=1
    )
    out["mtp_main_norm"][...] = params["mtp.0.main_norm.weight"]
    last_mtp = f"mtp.{M - 1}."
    out["mtp_final_norm"][...] = params[last_mtp + "norm.weight"]
    if v1 > v0:
      out["mtp_markov_embed"][: v1 - v0] = params[
          last_mtp + "markov_head.embed.weight"
      ][v0:v1]
      out["mtp_markov_head"][:, : v1 - v0] = params[
          last_mtp + "markov_head.head.weight"
      ][v0:v1].T
    out["mtp_confidence_w"][...] = params[
        last_mtp + "confidence_head.proj.weight"
    ].T

  return out
