# SPDX-License-Identifier: Apache-2.0
"""CPU tests for overlay v2 ([dsv41-offload v2]): RequestOffloadState.update_offload_keys builds keys incrementally
(direct list indexing) instead of islice-ing block_hashes from index 0 on every call. The keys must be exactly the old
code's, call for call.

Run against a tree with overlay_v2 installed:
    PYTHONPATH=<patched vllm tree v2>:<this dir> python -m unittest -v test_dsv41_kv_offload_v2

  * randomized request histories on the real 9-group DSv4.1 layout (7 SWA checkpoint groups at 1 hash per 32-token
    chunk, the MLA group at 4 hashes per chunk, the skipped ring): prompts of any length, growth by 0, 1, partial and
    chunk-crossing token counts, repeated calls without growth, and calls through _update_req_states with the
    preempted/resumed flag; after every call the keys equal both the verbatim v1 implementation and a from-scratch
    rebuild;
  * end to end: the same multi-turn MiniEngine workload with the v1 and v2 key builders emits identical store and load
    jobs (request, GPU blocks, host blocks), with checkpoint and dense SWA policies.
"""

from __future__ import annotations

import random
import unittest
from itertools import islice
from types import SimpleNamespace

from test_dsv41_kv_offload import CKPT, MiniEngine, _conversations, _make_request, _make_scheduler, _tokens

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as sched_mod
from vllm.v1.kv_offload.base import make_offload_key

V2_IMPL = sched_mod.RequestOffloadState.update_offload_keys


def v1_update_offload_keys(self) -> None:
    """Verbatim copy of the v1 overlay's (and the fork's) update_offload_keys."""
    for group_config, group_state in zip(self.config.kv_group_configs, self.group_states):
        if group_config.skip:
            continue
        for req_block_hash in islice(
            self.req.block_hashes,
            group_config.hashes_per_chunk * len(group_state.offload_keys) + group_config.hashes_per_chunk - 1,
            None,
            group_config.hashes_per_chunk,
        ):
            group_state.offload_keys.append(make_offload_key(req_block_hash, group_config.group_idx))


def from_scratch(state) -> list[list]:
    return [
        []
        if gc.skip
        else [make_offload_key(h, gc.group_idx) for h in islice(state.req.block_hashes, gc.hashes_per_chunk - 1, None,
                                                                 gc.hashes_per_chunk)]
        for gc in state.config.kv_group_configs
    ]


def _resume_output(req_id: str, preempted: bool):
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[req_id], new_block_ids=[None], resumed_req_ids={req_id} if preempted else set()
        ),
        kv_cache_block_copies=None,
    )


class IncrementalKeysTest(unittest.TestCase):
    def test_v2_is_installed(self):
        import inspect

        src = inspect.getsource(V2_IMPL)
        self.assertIn("[dsv41-offload v2]", src)
        self.assertNotIn("islice", V2_IMPL.__code__.co_names)  # code, not the comment that explains it

    def test_randomized_histories_match_v1(self):
        n_calls = 0
        for seed in range(60):
            rng = random.Random(seed)
            sched, *_ = _make_scheduler(dict(CKPT) if seed % 2 else {"swa_store_policy": "all"})
            groups = sched.config.kv_group_configs
            self.assertEqual([g.hashes_per_chunk for g in groups if not g.skip], [1] * 7 + [4])
            plen = rng.choice([0, 1, 31, 32, 33, 127, 128, 129, rng.randrange(2, 6000)])
            req = _make_request(f"r{seed}", _tokens(seed + 1, max(plen, 1)), max_tokens=100000)
            sched.on_new_request(req)
            state = sched._req_status[req.request_id]
            twin = SimpleNamespace(
                req=req, config=state.config, group_states=[SimpleNamespace(offload_keys=[]) for _ in groups]
            )
            for step in range(rng.randrange(20, 120)):
                grow = rng.choice([0, 0, 1, 1, 2, 31, 32, 33, 64, 127, 128, 129, rng.randrange(1, 700)])
                if grow:
                    req.append_output_token_ids([rng.randrange(10, 50000) for _ in range(grow)])
                how = rng.random()
                if how < 0.25:
                    sched._update_req_states(_resume_output(req.request_id, preempted=rng.random() < 0.5))
                else:
                    state.update_offload_keys()
                    if how > 0.9:  # repeated call without growth
                        state.update_offload_keys()
                v1_update_offload_keys(twin)
                got = [list(gs.offload_keys) for gs in state.group_states]
                self.assertEqual(got, [list(gs.offload_keys) for gs in twin.group_states], (seed, step))
                self.assertEqual(got, from_scratch(state), (seed, step))
                n_calls += 1
        self.assertGreater(n_calls, 2000)

    def test_ring_group_never_gets_keys(self):
        sched, *_ = _make_scheduler(dict(CKPT))
        req = _make_request("ring", _tokens(3, 5000), max_tokens=1000)
        sched.on_new_request(req)
        state = sched._req_status["ring"]
        req.append_output_token_ids(list(range(10, 900)))
        state.update_offload_keys()
        for gc, gs in zip(state.config.kv_group_configs, state.group_states):
            if gc.skip:
                self.assertEqual(gs.offload_keys, [])
            else:
                self.assertGreater(len(gs.offload_keys), 0)


def _job_log(extra, seeds, turns, impl):
    """Store/load jobs emitted by a MiniEngine multi-turn workload with the given key builder."""
    saved = sched_mod.RequestOffloadState.update_offload_keys
    sched_mod.RequestOffloadState.update_offload_keys = impl
    try:
        eng = MiniEngine(extra, num_gpu_blocks=260)
        log = []
        orig = eng.sched.build_connector_meta

        def record(so):
            meta = orig(so)
            for kind, jobs in (("store", meta.store_jobs), ("load", meta.load_jobs)):
                for job in jobs.values():
                    log.append((kind, job.req_id, tuple(job.src_spec.block_ids), tuple(job.dst_spec.block_ids)))
            return meta

        eng.sched.build_connector_meta = record
        _conversations(eng, seeds, turns, prompt_len=700, out_len=300, tool_len=90)
        return log, eng.loads_checked
    finally:
        sched_mod.RequestOffloadState.update_offload_keys = saved


class EndToEndTest(unittest.TestCase):
    def _check(self, extra):
        for seeds in ([1, 2, 3], [7, 11]):
            v1_log, v1_loads = _job_log(extra, seeds, 4, v1_update_offload_keys)
            v2_log, v2_loads = _job_log(extra, seeds, 4, V2_IMPL)
            self.assertGreater(len(v1_log), 20)
            self.assertGreater(v1_loads, 0)  # host restores happened and passed the content check
            self.assertEqual(v1_log, v2_log)
            self.assertEqual(v1_loads, v2_loads)

    def test_checkpoint_policy_jobs_identical(self):
        self._check(dict(CKPT, align_hits=True, offload_prompt_only=False))

    def test_dense_policy_jobs_identical(self):
        self._check({"swa_store_policy": "all", "align_hits": True, "offload_prompt_only": False})


if __name__ == "__main__":
    unittest.main()
