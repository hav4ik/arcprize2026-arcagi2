"""Pure JAX engine programs for DeepSeek-V4.1-Flash-1017 serving.

Holds the `wiz_py` `prefill`, `generate`, and `insert` programs, the boxed
`Ds4GemaxState`, and the Flax weight holder, with no Jet Engine dependency, so
Forge hardware tests can time the exact serving programs. `ds4_engine` wraps
these in `Ds4GemaxEngine`.
"""

# pylint: disable=g-importing-member

from __future__ import annotations

from typing import Any

from flax import linen as nn
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import decode_megakernel as mk
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_layout as layout_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_ref as ref

DS4_PAD_ID = 2
DS4_EOS_IDS = (1, 2)
VOCAB_SIZE = 129280


@struct.dataclass
class Ds4GemaxState:
  """Boxed state dictionary carried by `Prefix.state` and `DecodeState.state`."""

  value: dict[str, Any]

  def pad_or_shrink_prefix(self, new_length: int, pad: bool) -> Ds4GemaxState:
    """Pads or truncates the m1 and m2 KV caches to `new_length` tokens."""
    new_length = int(new_length)
    m2_len = max(1, new_length // 2)

    def _resize(x: jax.Array, axis: int, target: int) -> jax.Array:
      cur = x.shape[axis]
      if cur == target:
        return x
      if not pad or target < cur:
        return lax.slice_in_dim(x, 0, target, axis=axis)
      widths = [(0, 0)] * x.ndim
      widths[axis] = (0, target - cur)
      return jnp.pad(x, widths)

    return Ds4GemaxState(
        value={
            **self.value,
            "comp_m2_kv": _resize(self.value["comp_m2_kv"], 2, m2_len),
            "comp_m2_idx_k": _resize(self.value["comp_m2_idx_k"], 2, m2_len),
            "comp_m1_kv": _resize(self.value["comp_m1_kv"], 2, new_length),
            "comp_m1_idx_k": _resize(
                self.value["comp_m1_idx_k"], 2, new_length
            ),
        }
    )

  def change_prefix_length(self, new_length: Any) -> Ds4GemaxState:
    old_pos = self.value["position"]
    new_pos = jnp.asarray(new_length, jnp.int32).reshape(old_pos.shape)
    return Ds4GemaxState(
        value={**self.value, "position": jnp.maximum(new_pos, 0)}
    )

  def change_prefix_length_by(self, delta: int) -> Ds4GemaxState:
    if delta == 0:
      return self
    return self.change_prefix_length(self.value["position"] + delta)

  def stack_along_batch(self, to_stack: list[Ds4GemaxState]) -> Ds4GemaxState:
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *to_stack)


@struct.dataclass
class Ds4StepOutput:
  """Output returned by the purified DS4 `generate` step."""

  emitted: jax.Array
  emit_count: jax.Array
  last_token: jax.Array


def aux_table_arrays(
    cfg: ref.Ds4Config, max_seq_len: int
) -> tuple[dict[str, np.ndarray], int]:
  """Precomputes host NumPy Engram & RoPE tables for `Ds4WeightsModule`."""
  tbl = ref.EngramTables.from_config(cfg)
  m_u64 = tbl.multipliers.view(np.uint64)
  m_hi = (m_u64 >> np.uint64(32)).astype(np.uint32)
  m_lo = (m_u64 & np.uint64(0xFFFFFFFF)).astype(np.uint32)
  cos_swa_np, sin_swa_np = ref.precompute_freqs_cis(
      cfg.rope_head_dim,
      max_seq_len,
      0,
      cfg.rope_theta,
      cfg.rope_factor,
      cfg.beta_fast,
      cfg.beta_slow,
  )
  cos_comp_np, sin_comp_np = ref.precompute_freqs_cis(
      cfg.rope_head_dim,
      max_seq_len,
      cfg.original_seq_len,
      cfg.compress_rope_theta,
      cfg.rope_factor,
      cfg.beta_fast,
      cfg.beta_slow,
  )
  aux = {
      "engram_token_map": np.asarray(tbl.token_map, dtype=np.int32),
      "engram_mult_hi": m_hi,
      "engram_mult_lo": m_lo,
      "engram_primes": np.asarray(tbl.primes, dtype=np.uint32),
      "engram_offsets": np.asarray(tbl.offsets, dtype=np.int32),
      "cos_swa": np.asarray(cos_swa_np, dtype=np.float32),
      "sin_swa": np.asarray(sin_swa_np, dtype=np.float32),
      "cos_comp": np.asarray(cos_comp_np, dtype=np.float32),
      "sin_comp": np.asarray(sin_comp_np, dtype=np.float32),
  }
  return aux, int(tbl.pad_compressed_id)


class Ds4WeightsModule(nn.Module):
  """Flax parameter holder for the 71 stacked DS4 device weight arrays + aux tables."""

  lay: layout_lib.DeviceLayout
  max_seq_len: int

  @nn.compact
  def __call__(self) -> dict[str, jax.Array]:
    n = self.lay.num_devices
    specs = dict(self.lay.abstract_device_arrays())
    aux_np, _ = aux_table_arrays(self.lay.cfg, self.max_seq_len)
    for k, arr in aux_np.items():
      specs[k] = (arr.shape, arr.dtype)
    out = {}
    for name, (sh, dt) in specs.items():
      out[name] = self.param(
          name,
          lambda rng, s=(n, *sh), d=dt: jnp.zeros(s, dtype=d),
      )
    return out


def make_ds4_engine_functions(
    mesh: Mesh,
    lay: layout_lib.DeviceLayout,
    *,
    batch_size: int,
    max_seq_len: int,
    steps_per_call: int = 1,
    interpret: bool = False,
):
  """Builds `(prefill, generate, insert, state_init, state_shard, prefix_state_shard)`."""
  cfg = lay.cfg
  rep_spec = P()
  verify_len = 1 + cfg.dspark_block_size
  _, engram_pad_id = aux_table_arrays(cfg, max_seq_len)

  def state_init(b_sz: int, *, is_prefill: bool = False) -> Ds4GemaxState:
    z = mk.DeviceDecodeCaches.zeros(cfg, b_sz, max_seq_len)
    pos = (
        jnp.zeros((b_sz,), dtype=jnp.int32)
        if is_prefill
        else jnp.full((b_sz,), -1, dtype=jnp.int32)
    )
    return Ds4GemaxState(
        value={
            "window_kv": z.window_kv,
            "mtp_window_kv": z.mtp_window_kv,
            "comp_m2_kv": z.comp_m2_kv,
            "comp_m2_idx_k": z.comp_m2_idx_k,
            "comp_m2_kv_state": z.comp_m2_kv_state,
            "comp_m2_score_state": z.comp_m2_score_state,
            "comp_m1_kv": z.comp_m1_kv,
            "comp_m1_idx_k": z.comp_m1_idx_k,
            "engram_tail": jnp.zeros((b_sz, 3), dtype=jnp.int32),
            "candidates": jnp.zeros((b_sz, verify_len), dtype=jnp.int32),
            "token": jnp.zeros((b_sz,), dtype=jnp.int32),
            "position": pos,
        }
    )

  def state_shard(block_meshes: Any = None) -> Ds4GemaxState:
    del block_meshes
    return Ds4GemaxState(
        value={
            "window_kv": rep_spec,
            "mtp_window_kv": rep_spec,
            "comp_m2_kv": rep_spec,
            "comp_m2_idx_k": rep_spec,
            "comp_m2_kv_state": rep_spec,
            "comp_m2_score_state": rep_spec,
            "comp_m1_kv": rep_spec,
            "comp_m1_idx_k": rep_spec,
            "engram_tail": rep_spec,
            "candidates": rep_spec,
            "token": rep_spec,
            "position": rep_spec,
        }
    )

  def prefix_state_shard(block_meshes: Any = None) -> Ds4GemaxState:
    return state_shard(block_meshes)

  def insert(
      state: Ds4GemaxState,
      prefix_state: Ds4GemaxState,
      slot: int | jax.Array,
      insert_context: Any = None,
  ) -> Ds4GemaxState:
    del insert_context
    dst = state.value
    src = prefix_state.value
    slot_i = jnp.asarray(slot, jnp.int32).reshape(())
    pos_0 = jnp.asarray(src["position"][0], jnp.int32).reshape(())

    def _copy_kv_prefix(
        dst_arr: jax.Array, src_arr: jax.Array, valid_pos: jax.Array
    ) -> jax.Array:
      max_len = min(src_arr.shape[2], dst_arr.shape[2])
      blk = 1024 if (max_len >= 1024 and max_len % 1024 == 0) else max_len
      if blk == max_len:
        return lax.dynamic_update_slice_in_dim(
            dst_arr, src_arr[:1, :, :max_len], slot_i, axis=0
        )
      n_blks = jnp.clip((valid_pos + blk - 1) // blk, 1, max_len // blk)

      def _blk_body(b_idx, acc_dst):
        start = b_idx * jnp.int32(blk)
        chunk = lax.dynamic_slice_in_dim(src_arr[:1], start, blk, axis=2)
        return lax.dynamic_update_slice(
            acc_dst, chunk, (slot_i, jnp.int32(0), start, jnp.int32(0))
        )

      return lax.fori_loop(jnp.int32(0), n_blks, _blk_body, dst_arr)

    updated = {}
    for k, dst_arr in dst.items():
      src_arr = src[k]
      if k in ("comp_m1_kv", "comp_m1_idx_k"):
        updated[k] = _copy_kv_prefix(dst_arr, src_arr, pos_0)
      elif k in ("comp_m2_kv", "comp_m2_idx_k"):
        updated[k] = _copy_kv_prefix(dst_arr, src_arr, pos_0 // jnp.int32(2))
      else:
        updated[k] = lax.dynamic_update_slice_in_dim(
            dst_arr, src_arr[:1], slot_i, axis=0
        )
    return Ds4GemaxState(value=updated)

  def prefill(
      prompt: tuple[jnp.ndarray, jnp.ndarray],
      state: Any = None,
      per_request_hyperparams: Any = None,
      history_length: Any = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray, Ds4GemaxState]:
    del per_request_hyperparams
    tokens, true_length = prompt
    weights = Ds4WeightsModule(lay=lay, max_seq_len=max_seq_len, name="ds4")()
    prev_state = (
        state.state
        if (state is not None and hasattr(state, "state"))
        else state
    )
    prompt_2d = tokens.reshape(1, -1).astype(jnp.int32)
    true_len = jnp.asarray(true_length, jnp.int32).reshape(())
    if prev_state is None:
      caches0 = mk.DeviceDecodeCaches.zeros(cfg, 1, max_seq_len)
      tail0 = jnp.zeros((1, 3), dtype=jnp.int32)
      start_pos = jnp.int32(0)
    else:
      ps_pad = prev_state.pad_or_shrink_prefix(max_seq_len, pad=True).value
      caches0 = mk.DeviceDecodeCaches(
          window_kv=ps_pad["window_kv"],
          mtp_window_kv=ps_pad["mtp_window_kv"],
          comp_m2_kv=ps_pad["comp_m2_kv"],
          comp_m2_idx_k=ps_pad["comp_m2_idx_k"],
          comp_m2_kv_state=ps_pad["comp_m2_kv_state"],
          comp_m2_score_state=ps_pad["comp_m2_score_state"],
          comp_m1_kv=ps_pad["comp_m1_kv"],
          comp_m1_idx_k=ps_pad["comp_m1_idx_k"],
      )
      tail0 = ps_pad["engram_tail"]
      start_pos = (
          jnp.asarray(history_length, jnp.int32).reshape(())
          if history_length is not None
          else jnp.asarray(ps_pad["position"][0], jnp.int32).reshape(())
      )

    def _per_dev(w_dev, c_in, tl_in, sp_in, p_in, t_len):
      w_d = jax.tree.map(lambda a: a[0], w_dev)
      cs_d = w_d["cos_swa"]
      ss_d = w_d["sin_swa"]
      cc_d = w_d["cos_comp"]
      sc_d = w_d["sin_comp"]
      engram_tbl = mk.DeviceEngramTables(
          token_map=w_d["engram_token_map"],
          mult_hi=w_d["engram_mult_hi"],
          mult_lo=w_d["engram_mult_lo"],
          primes=w_d["engram_primes"],
          offsets=w_d["engram_offsets"],
          pad_id=engram_pad_id,
      )
      seqlen = p_in.shape[1]
      win = cfg.window_size
      enc_chunk = (
          4 * win if (seqlen >= 4 * win and seqlen % (4 * win) == 0) else win
      )
      eff_t_len = jnp.clip(t_len, 1, seqlen)
      if seqlen <= win or seqlen % enc_chunk != 0:
        eh_d, tail_out = mk.compute_engram_hashes_jax(
            p_in, sp_in, tl_in, engram_tbl, valid_len=eff_t_len
        )
        h_enc, pm_enc, caches1 = mk.encoder_macrokernel(
            p_in,
            sp_in,
            w_d,
            c_in,
            cs_d,
            ss_d,
            cc_d,
            sc_d,
            eh_d,
            lay=lay,
            write_global_kv=True,
            interpret=interpret,
            valid_len=eff_t_len,
        )
        next_tok, _, main_h, caches1 = mk.decoder_macrokernel(
            h_enc,
            pm_enc,
            sp_in,
            w_d,
            caches1,
            cc_d,
            sc_d,
            lay=lay,
            full_logits=False,
            interpret=interpret,
            valid_len=eff_t_len,
        )
        dec_sp = sp_in
        dec_vlen = eff_t_len
      else:
        num_chunks = seqlen // enc_chunk
        ids_chunks = p_in.reshape(1, num_chunks, enc_chunk)
        n_full = (eff_t_len - 1) // jnp.int32(enc_chunk)
        rem = eff_t_len - n_full * jnp.int32(enc_chunk)
        h0 = jnp.zeros((1, enc_chunk, cfg.hc_mult, cfg.dim), dtype=jnp.bfloat16)
        pm0 = jnp.zeros((1, enc_chunk, cfg.hc_mult), dtype=jnp.float32)

        def _enc_chunk_body(c_idx, carry):
          cur_c, cur_tl, _, _, prev_h_c, prev_pm_c = carry
          c_start = sp_in + c_idx * jnp.int32(enc_chunk)
          c_ids = ids_chunks[:, c_idx]
          v_len_c = jnp.where(c_idx == n_full, rem, jnp.int32(enc_chunk))
          c_eh, next_tl = mk.compute_engram_hashes_jax(
              c_ids, c_start, cur_tl, engram_tbl, valid_len=v_len_c
          )
          h_c, pm_c, next_c = mk.encoder_macrokernel(
              c_ids,
              c_start,
              w_d,
              cur_c,
              cs_d,
              ss_d,
              cc_d,
              sc_d,
              c_eh,
              lay=lay,
              write_global_kv=True,
              interpret=interpret,
              valid_len=v_len_c,
          )
          return next_c, next_tl, prev_h_c, prev_pm_c, h_c, pm_c

        caches1, tail_out, prev_h, prev_pm, cur_h, cur_pm = lax.fori_loop(
            jnp.int32(0),
            n_full + jnp.int32(1),
            _enc_chunk_body,
            (c_in, tl_in, h0, pm0, h0, pm0),
        )
        cat_h = jnp.concatenate([prev_h, cur_h], axis=1)
        cat_pm = jnp.concatenate([prev_pm, cur_pm], axis=1)
        has_full_win = eff_t_len > jnp.int32(win)
        win_start = jnp.where(
            has_full_win,
            jnp.int32(enc_chunk) + rem - jnp.int32(win),
            jnp.int32(enc_chunk),
        )
        h_dec_in = lax.dynamic_slice_in_dim(cat_h, win_start, win, axis=1)
        pm_dec_in = lax.dynamic_slice_in_dim(cat_pm, win_start, win, axis=1)
        dec_sp = jnp.where(
            has_full_win, sp_in + eff_t_len - jnp.int32(win), sp_in
        )
        dec_vlen = jnp.where(has_full_win, jnp.int32(win), rem)
        next_tok, _, main_h, caches1 = mk.decoder_macrokernel(
            h_dec_in,
            pm_dec_in,
            dec_sp,
            w_d,
            caches1,
            cc_d,
            sc_d,
            lay=lay,
            full_logits=False,
            interpret=interpret,
            valid_len=dec_vlen,
        )

      end_pos = sp_in + eff_t_len
      draft_out, _, caches2 = mk.dspark_macrokernel(
          next_tok,
          main_h,
          dec_sp,
          w_d,
          caches1,
          cs_d,
          ss_d,
          lay=lay,
          seed_only=False,
          interpret=interpret,
          valid_len=dec_vlen,
      )
      return next_tok, draft_out[:, :verify_len], tail_out, end_pos, caches2

    next_tok, cand_out, tail_out, end_pos_out, caches_out = jax.shard_map(
        _per_dev,
        mesh=mesh,
        in_specs=(
            P(layout_lib.AXIS),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(), P(), P(), P(), P()),
        check_vma=False,
    )(weights, caches0, tail0, start_pos, prompt_2d, true_len)

    new_state = Ds4GemaxState(
        value={
            "window_kv": caches_out.window_kv,
            "mtp_window_kv": caches_out.mtp_window_kv,
            "comp_m2_kv": caches_out.comp_m2_kv,
            "comp_m2_idx_k": caches_out.comp_m2_idx_k,
            "comp_m2_kv_state": caches_out.comp_m2_kv_state,
            "comp_m2_score_state": caches_out.comp_m2_score_state,
            "comp_m1_kv": caches_out.comp_m1_kv,
            "comp_m1_idx_k": caches_out.comp_m1_idx_k,
            "engram_tail": tail_out,
            "candidates": cand_out,
            "token": next_tok.reshape((1,)).astype(jnp.int32),
            "position": jnp.reshape(
                jnp.asarray(end_pos_out, dtype=jnp.int32), (1,)
            ),
        }
    )
    scores = jnp.zeros((prompt_2d.shape[1],), dtype=jnp.float32)
    final_logit = jnp.zeros((1, cfg.vocab_size), dtype=jnp.float32)
    return scores, final_logit, new_state

  def generate(
      tokens: jnp.ndarray,
      state: Ds4GemaxState,
      generate_context: Any = None,
  ) -> tuple[Ds4StepOutput, Ds4GemaxState]:
    del generate_context
    weights = Ds4WeightsModule(lay=lay, max_seq_len=max_seq_len, name="ds4")()
    s = state.value
    tokens_1d = tokens.reshape((batch_size,)).astype(jnp.int32)
    is_active = (tokens_1d != DS4_PAD_ID) & (s["position"] >= 0)
    active_pos = jnp.where(is_active, s["position"], jnp.int32(-1))
    cand_in = (
        s["candidates"]
        .at[:, 0]
        .set(jnp.where(is_active, s["token"], jnp.int32(0)))
    )
    caches_in = mk.DeviceDecodeCaches(
        window_kv=s["window_kv"],
        mtp_window_kv=s["mtp_window_kv"],
        comp_m2_kv=s["comp_m2_kv"],
        comp_m2_idx_k=s["comp_m2_idx_k"],
        comp_m2_kv_state=s["comp_m2_kv_state"],
        comp_m2_score_state=s["comp_m2_score_state"],
        comp_m1_kv=s["comp_m1_kv"],
        comp_m1_idx_k=s["comp_m1_idx_k"],
    )

    def _per_dev(w_dev, c_in, cd_in, pos_in, tl_in):
      w_d = jax.tree.map(lambda a: a[0], w_dev)
      cs_d = w_d["cos_swa"]
      ss_d = w_d["sin_swa"]
      cc_d = w_d["cos_comp"]
      sc_d = w_d["sin_comp"]
      engram_tbl = mk.DeviceEngramTables(
          token_map=w_d["engram_token_map"],
          mult_hi=w_d["engram_mult_hi"],
          mult_lo=w_d["engram_mult_lo"],
          primes=w_d["engram_primes"],
          offsets=w_d["engram_offsets"],
          pad_id=engram_pad_id,
      )
      if steps_per_call <= 1:
        eh_d, _ = mk.compute_engram_hashes_jax(cd_in, pos_in, tl_in, engram_tbl)
        t_pred, emit_cnt, next_cand, _, c_out = (
            mk.speculative_verify_and_draft_step(
                cd_in,
                pos_in,
                w_d,
                c_in,
                cs_d,
                ss_d,
                cc_d,
                sc_d,
                eh_d,
                lay=lay,
                interpret=interpret,
            )
        )
        tl_next = mk.advance_engram_tail_jax(
            cd_in, tl_in, emit_cnt, pos_in >= 0, engram_tbl
        )
        emitted_bs = jnp.concatenate([cd_in[:, :1], t_pred[:, :-1]], axis=1)
        next_pos = jnp.where(pos_in >= 0, pos_in + emit_cnt, jnp.int32(-1))
        return emitted_bs, emit_cnt, next_cand, next_pos, tl_next, c_out

      step_toks0 = jnp.zeros(
          (batch_size, steps_per_call, verify_len), dtype=jnp.int32
      )
      step_cnts0 = jnp.zeros((batch_size, steps_per_call), dtype=jnp.int32)

      def _spec_step(s_idx, carry):
        cur_c, cur_cd, cur_pos, cur_tl, acc_toks, acc_cnts = carry
        eh_s, _ = mk.compute_engram_hashes_jax(
            cur_cd, cur_pos, cur_tl, engram_tbl
        )
        tp_s, ec_s, ncand_s, _, next_c = mk.speculative_verify_and_draft_step(
            cur_cd,
            cur_pos,
            w_d,
            cur_c,
            cs_d,
            ss_d,
            cc_d,
            sc_d,
            eh_s,
            lay=lay,
            interpret=interpret,
        )
        ntl_s = mk.advance_engram_tail_jax(
            cur_cd, cur_tl, ec_s, cur_pos >= 0, engram_tbl
        )
        em_s = jnp.concatenate([cur_cd[:, :1], tp_s[:, :-1]], axis=1)
        npos_s = jnp.where(cur_pos >= 0, cur_pos + ec_s, jnp.int32(-1))
        acc_toks = acc_toks.at[:, s_idx, :].set(em_s)
        acc_cnts = acc_cnts.at[:, s_idx].set(ec_s)
        return next_c, ncand_s, npos_s, ntl_s, acc_toks, acc_cnts

      c_out, next_cand, next_pos, tl_next, step_toks, step_cnts = lax.fori_loop(
          jnp.int32(0),
          jnp.int32(steps_per_call),
          _spec_step,
          (c_in, cd_in, pos_in, tl_in, step_toks0, step_cnts0),
      )
      valid_m = (
          jnp.arange(verify_len, dtype=jnp.int32)[None, None, :]
          < step_cnts[:, :, None]
      )
      flat_toks = step_toks.reshape(batch_size, steps_per_call * verify_len)
      flat_valid = valid_m.reshape(batch_size, steps_per_call * verify_len)
      perm = jnp.argsort(~flat_valid, axis=1)
      emitted_bs = jnp.take_along_axis(flat_toks, perm, axis=1)
      emit_cnt = jnp.sum(step_cnts, axis=1)
      return emitted_bs, emit_cnt, next_cand, next_pos, tl_next, c_out

    emitted_bs, emit_cnt, next_cand, next_pos, tl_next, caches_out = (
        jax.shard_map(
            _per_dev,
            mesh=mesh,
            in_specs=(
                P(layout_lib.AXIS),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=(P(), P(), P(), P(), P(), P()),
            check_vma=False,
        )(weights, caches_in, cand_in, active_pos, s["engram_tail"])
    )

    last_tok = next_cand[:, 0]
    new_state = Ds4GemaxState(
        value={
            "window_kv": caches_out.window_kv,
            "mtp_window_kv": caches_out.mtp_window_kv,
            "comp_m2_kv": caches_out.comp_m2_kv,
            "comp_m2_idx_k": caches_out.comp_m2_idx_k,
            "comp_m2_kv_state": caches_out.comp_m2_kv_state,
            "comp_m2_score_state": caches_out.comp_m2_score_state,
            "comp_m1_kv": caches_out.comp_m1_kv,
            "comp_m1_idx_k": caches_out.comp_m1_idx_k,
            "engram_tail": tl_next,
            "candidates": next_cand,
            "token": jnp.where(is_active, last_tok, s["token"]),
            "position": next_pos,
        }
    )
    out = Ds4StepOutput(
        emitted=emitted_bs,
        emit_count=emit_cnt,
        last_token=last_tok,
    )
    return out, new_state

  return prefill, generate, insert, state_init, state_shard, prefix_state_shard
