"""JIT + shard_map compilation wrappers for DeepSeek-V4.1-Flash-1017 on Ghostfish.

Wraps the 3 Macro-Kernels (`encoder_macrokernel`, `decoder_macrokernel`,
`dspark_macrokernel`), Two-Phase CED Bounded Replay prefill
(`ced_prefill_macrokernel`), and fused speculative verify-and-draft step
(`speculative_verify_and_draft_step`) into sharded `jax.jit` programs with
in-place KV cache buffer donation (`donate_argnums`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import decode_megakernel as mk
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_layout as layout_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_ref as ref


@dataclasses.dataclass(frozen=True)
class Ds4ShardedPrograms:
  """Compiled sharded programs and shardings for DeepSeek-V4.1-Flash-1017."""

  mesh: Mesh
  lay: layout_lib.DeviceLayout
  max_batch_size: int
  max_seq_len: int
  weight_shardings: dict[str, NamedSharding]
  cache_shardings: mk.DeviceDecodeCaches
  replicated_sharding: NamedSharding
  ced_prefill_fn: Any
  decode_step_fn: Any
  speculative_step_fn: Any


def make_weight_shardings(
    mesh: Mesh, lay: layout_lib.DeviceLayout
) -> dict[str, NamedSharding]:
  """Returns NamedShardings for per-device stacked weight arrays [N, ...]."""
  sh = NamedSharding(mesh, P(layout_lib.AXIS))
  return {name: sh for name in lay.abstract_device_arrays()}


def abstract_sharded_weights(
    mesh: Mesh, lay: layout_lib.DeviceLayout
) -> dict[str, jax.ShapeDtypeStruct]:
  """Returns global ShapeDtypeStructs [N, ...] for deviceless AOT compilation."""
  n = lay.num_devices
  sh = NamedSharding(mesh, P(layout_lib.AXIS))
  return {
      name: jax.ShapeDtypeStruct((n, *shape), dtype, sharding=sh)
      for name, (shape, dtype) in lay.abstract_device_arrays().items()
  }


def make_cache_shardings(mesh: Mesh) -> mk.DeviceDecodeCaches:
  """Returns replicated NamedShardings for DeviceDecodeCaches."""
  rep: Any = NamedSharding(mesh, P())
  return mk.DeviceDecodeCaches(
      window_kv=rep,
      mtp_window_kv=rep,
      comp_m2_kv=rep,
      comp_m2_idx_k=rep,
      comp_m2_kv_state=rep,
      comp_m2_score_state=rep,
      comp_m1_kv=rep,
      comp_m1_idx_k=rep,
  )


def abstract_sharded_caches(
    mesh: Mesh,
    cfg: ref.Ds4Config,
    batch_size: int,
    max_seq_len: int,
) -> mk.DeviceDecodeCaches:
  """Returns ShapeDtypeStructs for DeviceDecodeCaches on `mesh`."""
  rep = NamedSharding(mesh, P())
  z = jax.eval_shape(
      lambda: mk.DeviceDecodeCaches.zeros(cfg, batch_size, max_seq_len)
  )
  return jax.tree.map(
      lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=rep), z
  )


def build_sharded_programs(
    mesh: Mesh,
    cfg: ref.Ds4Config,
    *,
    max_batch_size: int = 1,
    max_seq_len: int = 4096,
    lay: layout_lib.DeviceLayout | None = None,
    interpret: bool = False,
) -> Ds4ShardedPrograms:
  """Builds sharded JIT callables with buffer donation on `caches`."""
  if lay is None:
    lay = layout_lib.build_layout(cfg, num_devices=mesh.size)
  rep = NamedSharding(mesh, P())
  w_shardings = make_weight_shardings(mesh, lay)
  c_shardings = make_cache_shardings(mesh)

  def _ced_prefill_impl(
      w_stacked,
      caches,
      input_ids,
      engram_hashes,
      cos_swa,
      sin_swa,
      cos_comp,
      sin_comp,
  ):
    def _per_device(w_dev, c_in, ids_in, h_in, cs, ss, cc, sc):
      w_d = jax.tree.map(lambda a: a[0], w_dev)
      next_tok, _, main_h, c_out = mk.ced_prefill_macrokernel(
          ids_in,
          w_d,
          c_in,
          cs,
          ss,
          cc,
          sc,
          h_in,
          lay=lay,
          interpret=interpret,
      )
      _, _, c_out = mk.dspark_macrokernel(
          next_tok,
          main_h,
          jnp.int32(0),
          w_d,
          c_out,
          cs,
          ss,
          lay=lay,
          seed_only=True,
          interpret=interpret,
      )
      return next_tok, c_out

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(
            P(layout_lib.AXIS),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(), P()),
        check_vma=False,
    )(
        w_stacked,
        caches,
        input_ids,
        engram_hashes,
        cos_swa,
        sin_swa,
        cos_comp,
        sin_comp,
    )

  def _decode_step_impl(
      w_stacked,
      caches,
      token_ids,
      start_pos,
      engram_hashes,
      cos_swa,
      sin_swa,
      cos_comp,
      sin_comp,
  ):
    def _per_device(w_dev, c_in, ids_in, sp_in, h_in, cs, ss, cc, sc):
      w_d = jax.tree.map(lambda a: a[0], w_dev)
      h_enc, pm_enc, c_mid = mk.encoder_macrokernel(
          ids_in,
          sp_in,
          w_d,
          c_in,
          cs,
          ss,
          cc,
          sc,
          h_in,
          lay=lay,
          write_global_kv=True,
          interpret=interpret,
      )
      out_tok, _, main_h, c_dec = mk.decoder_macrokernel(
          h_enc,
          pm_enc,
          sp_in,
          w_d,
          c_mid,
          cc,
          sc,
          lay=lay,
          full_logits=False,
          interpret=interpret,
      )
      draft_ids, draft_conf, c_out = mk.dspark_macrokernel(
          out_tok,
          main_h,
          sp_in,
          w_d,
          c_dec,
          cs,
          ss,
          lay=lay,
          seed_only=False,
          interpret=interpret,
      )
      return out_tok, draft_ids, draft_conf, c_out

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(
            P(layout_lib.AXIS),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(), P(), P(), P()),
        check_vma=False,
    )(
        w_stacked,
        caches,
        token_ids,
        start_pos,
        engram_hashes,
        cos_swa,
        sin_swa,
        cos_comp,
        sin_comp,
    )

  def _speculative_step_impl(
      w_stacked,
      caches,
      verify_tokens,
      start_pos,
      engram_hashes,
      cos_swa,
      sin_swa,
      cos_comp,
      sin_comp,
  ):
    def _per_device(w_dev, c_in, v_tok, sp_in, h_in, cs, ss, cc, sc):
      w_d = jax.tree.map(lambda a: a[0], w_dev)
      return mk.speculative_verify_and_draft_step(
          v_tok,
          sp_in,
          w_d,
          c_in,
          cs,
          ss,
          cc,
          sc,
          h_in,
          lay=lay,
          interpret=interpret,
      )

    return jax.shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(
            P(layout_lib.AXIS),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(), P(), P(), P(), P()),
        check_vma=False,
    )(
        w_stacked,
        caches,
        verify_tokens,
        start_pos,
        engram_hashes,
        cos_swa,
        sin_swa,
        cos_comp,
        sin_comp,
    )

  ced_prefill_fn = jax.jit(_ced_prefill_impl, donate_argnums=(1,))
  decode_step_fn = jax.jit(_decode_step_impl, donate_argnums=(1,))
  speculative_step_fn = jax.jit(_speculative_step_impl, donate_argnums=(1,))

  return Ds4ShardedPrograms(
      mesh=mesh,
      lay=lay,
      max_batch_size=max_batch_size,
      max_seq_len=max_seq_len,
      weight_shardings=w_shardings,
      cache_shardings=c_shardings,
      replicated_sharding=rep,
      ced_prefill_fn=ced_prefill_fn,
      decode_step_fn=decode_step_fn,
      speculative_step_fn=speculative_step_fn,
  )
