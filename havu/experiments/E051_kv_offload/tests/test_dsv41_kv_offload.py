# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the DeepSeek-V4.1 KV CPU-offloading patch ([dsv41-offload]).

Run (CPU only, no GPU needed):
    PYTHONPATH=<patched vllm tree>:<this dir> python -m unittest -v test_dsv41_kv_offload

Covers:
  * config resolution on the real 9-group DSv4.1 layout (both E042 failures);
  * group / block mapping (scratch ring skipped, UniformType groups unwrapped);
  * hit validity: MLA prefix contiguous, every SWA group's window ending at the
    hit, hits aligned to 128, last prompt token recomputed;
  * a mini engine (real KVCacheManager + patched connector scheduler + real
    CPUOffloadingManager) with a per-slot model of GPU and host block contents,
    so a stale, misplaced or partial copy shows up as a content mismatch;
  * checkpoint store policy: host bytes, rolling discard, finished-request cap;
  * worker registration of the packed 113,920-byte pool on CPU tensors.
"""

from __future__ import annotations

import itertools
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

from dsv41_fixture import (
    POOL_BLOCK_BYTES,
    allocate_cpu_kv_caches,
    build_kv_cache_config,
    make_vllm_config,
)

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    scheduler as sched_mod,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, iter_layer_specs
from vllm.v1.kv_offload.base import LookupResult, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

init_none_hash(sha256)

TPH = 32  # tokens_per_hash = gcd(128, 32)
ALIGN = 128
SWA_GROUPS = tuple(range(7))
MLA_GROUP = 7
RING_GROUP = 8
CKPT = {"swa_store_policy": "checkpoint"}


def _group_kinds(kvc):
    kinds = []
    for g in kvc.kv_cache_groups:
        names = {type(s).__name__ for s in iter_layer_specs(g.kv_cache_spec)}
        assert len(names) == 1
        kinds.append(names.pop())
    return kinds


def _make_scheduler(extra=None, num_gpu_blocks=1024, cpu_gib=8):
    # The engine core hands the scheduler-side connector the scheduler view of
    # the config (UniformTypeKVCacheSpecs replaced by their first layer spec,
    # core.py:332); the worker side gets the wrapped per-worker config.
    kvc = generate_scheduler_kv_cache_config([build_kv_cache_config(num_gpu_blocks)])
    cfg = {"cpu_bytes_to_use": int(cpu_gib * (1 << 30))}
    cfg.update(extra or {})
    vc = make_vllm_config(cfg)
    oc = build_offloading_config(vc, kvc)
    spec = CPUOffloadingSpec(oc)
    return OffloadingConnectorScheduler(spec, vc, kvc), kvc, vc, spec


def _tokens(seed: int, n: int, start: int = 0) -> list[int]:
    return [(seed * 7919 + (start + i) * 104729) % 50000 + 10 for i in range(n)]


def _make_request(req_id, prompt, max_tokens=64):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=get_request_block_hasher(TPH, sha256),
    )


# ---------------------------------------------------------------------------
# 1. Layout and config resolution
# ---------------------------------------------------------------------------
class LayoutAndConfigTest(unittest.TestCase):
    def test_real_group_layout(self):
        kvc = build_kv_cache_config(64)
        self.assertEqual(
            _group_kinds(kvc),
            ["SlidingWindowMLASpec"] * 7 + ["MLAAttentionSpec", "CircularBufferSpec"],
        )
        self.assertEqual(
            [g.kv_cache_spec.block_size for g in kvc.kv_cache_groups],
            [32] * 7 + [128, 8],
        )
        self.assertEqual(
            [len(g.layer_names) for g in kvc.kv_cache_groups],
            [6, 6, 6, 6, 6, 5, 5, 8, 3],
        )
        self.assertEqual(
            kvc.kv_cache_groups[MLA_GROUP].kv_cache_spec.page_size_bytes,
            POOL_BLOCK_BYTES,
        )
        self.assertFalse(kvc.kv_cache_groups[RING_GROUP].kv_cache_spec.prefix_cacheable)

    def test_offloading_config_resolves(self):
        kvc = build_kv_cache_config(64)
        for pmu, tph in ((None, 32), (8, 8)):
            oc = build_offloading_config(make_vllm_config(prefix_match_unit=pmu), kvc)
            self.assertEqual(oc.cache.tokens_per_hash, tph)
            self.assertEqual(
                [g.tokens_per_block for g in oc.groups], [32] * 7 + [128, 8]
            )
            self.assertEqual(oc.worker_kv_bytes_per_block, POOL_BLOCK_BYTES)
            # Multi-group: no single-copy replicated layout (4 copies at TP4).
            self.assertFalse(oc.replicated_layout)

    def test_cpu_sizing(self):
        kvc = build_kv_cache_config(64)
        oc = build_offloading_config(
            make_vllm_config({"cpu_bytes_to_use": 28 << 30}), kvc
        )
        spec = CPUOffloadingSpec(oc)
        # One host slot = one 113,920-B block for each of the 4 TP ranks,
        # page-aligned: 455,680 -> 458,752 B.
        self.assertEqual(spec.kv_bytes_per_chunk, 458_752)
        self.assertEqual(spec.num_blocks, (28 << 30) // 458_752)

    def test_shared_region_knob(self):
        kvc = build_kv_cache_config(64)
        for flag in (False, True):
            oc = build_offloading_config(
                make_vllm_config({"shared_region": flag}), kvc
            )
            with mock.patch(
                "vllm.v1.kv_offload.cpu.spec.current_platform"
            ) as plat:
                plat.is_cuda_alike.return_value = True
                spec = CPUOffloadingSpec(oc)
                self.assertEqual(spec._uses_shared_region(), flag)

    def test_scheduler_group_configs(self):
        sched, kvc, _, _ = _make_scheduler()
        cfgs = sched.config.kv_group_configs
        for g in SWA_GROUPS:
            self.assertEqual(
                (cfgs[g].tokens_per_chunk, cfgs[g].hashes_per_chunk), (32, 1)
            )
            self.assertEqual(cfgs[g].sliding_window_size_in_chunks, 4)
            self.assertFalse(cfgs[g].skip)
        self.assertEqual(
            (cfgs[MLA_GROUP].tokens_per_chunk, cfgs[MLA_GROUP].hashes_per_chunk),
            (128, 4),
        )
        self.assertIsNone(cfgs[MLA_GROUP].sliding_window_size_in_chunks)
        self.assertTrue(cfgs[RING_GROUP].skip)
        self.assertNotIn(RING_GROUP, sched._lookup_groups)
        self.assertEqual(sched._lookup_groups[0], MLA_GROUP)
        # Default extra config keeps upstream behaviour.
        self.assertFalse(sched.config.swa_checkpoints)
        self.assertEqual(sched.config.hit_alignment_tokens, 1)

    def test_checkpoint_options(self):
        sched, *_ = _make_scheduler(CKPT)
        self.assertTrue(sched.config.swa_checkpoints)
        self.assertEqual(sched.config.hit_alignment_tokens, ALIGN)
        self.assertEqual(sched._swa_ckpt_groups, frozenset(SWA_GROUPS))
        sched2, *_ = _make_scheduler({"align_hits": True})
        self.assertFalse(sched2.config.swa_checkpoints)
        self.assertEqual(sched2.config.hit_alignment_tokens, ALIGN)
        with self.assertRaises(ValueError):
            _make_scheduler({"swa_store_policy": "sometimes"})
        with self.assertRaises(ValueError):
            _make_scheduler({"swa_store_policy": "checkpoint", "align_hits": False})
        with self.assertRaises(ValueError):
            _make_scheduler({"swa_store_policy": "checkpoint", "blocks_per_chunk": 2})

    def test_uniform_type_unwrap(self):
        kvc = build_kv_cache_config(64)
        get = sched_mod.get_sliding_window_size_in_chunks
        self.assertEqual(get(kvc.kv_cache_groups[0].kv_cache_spec, 32), 4)
        self.assertIsNone(get(kvc.kv_cache_groups[MLA_GROUP].kv_cache_spec, 128))

    def test_offload_keys_skip_ring(self):
        sched, *_ = _make_scheduler()
        req = _make_request("r", _tokens(1, 1000))
        sched.on_new_request(req)
        st = sched._req_status["r"]
        st.update_offload_keys()
        self.assertEqual(len(st.group_states[RING_GROUP].offload_keys), 0)
        self.assertEqual(len(st.group_states[MLA_GROUP].offload_keys), 1000 // 128)
        for g in SWA_GROUPS:
            self.assertEqual(len(st.group_states[g].offload_keys), 1000 // 32)
        # MLA chunk i is keyed by the hash at its last 32-token sub-block.
        self.assertEqual(
            st.group_states[MLA_GROUP].offload_keys[2],
            make_offload_key(req.block_hashes[11], MLA_GROUP),
        )


# ---------------------------------------------------------------------------
# 2. Hit-validity logic against a hand-filled host cache
# ---------------------------------------------------------------------------
class LookupTest(unittest.TestCase):
    def _setup(self, extra, prompt_len):
        sched, kvc, vc, spec = _make_scheduler(extra)
        req = _make_request("q", _tokens(3, prompt_len))
        sched.on_new_request(req)
        st = sched._req_status["q"]
        st.update_offload_keys()
        return sched, req, st

    def _put(self, sched, keys):
        out = sched.manager.prepare_store(keys, None)
        assert out is not None
        sched.manager.complete_store(out.keys_to_store, None)

    def _mla(self, st, upto):
        return st.group_states[MLA_GROUP].offload_keys[: upto // 128]

    def _window(self, st, boundary, groups=SWA_GROUPS):
        keys = []
        for g in groups:
            keys += st.group_states[g].offload_keys[boundary // 32 - 4 : boundary // 32]
        return keys

    def test_full_hit_needs_mla_prefix_and_all_windows(self):
        sched, req, st = self._setup(CKPT, 1100)
        self._put(sched, self._mla(st, 1024) + self._window(st, 1024))
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0), (1024, True))

    def test_missing_one_group_window_falls_back(self):
        sched, req, st = self._setup(CKPT, 1100)
        self._put(sched, self._mla(st, 1024) + self._window(st, 512))
        self._put(sched, self._window(st, 1024, groups=SWA_GROUPS[:-1]))
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], 512)

    def test_mla_gap_limits_hit(self):
        sched, req, st = self._setup(CKPT, 1100)
        mla = self._mla(st, 1024)
        self._put(sched, mla[:3] + mla[4:] + self._window(st, 1024))
        self._put(sched, self._window(st, 384))
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], 384)

    def test_unaligned_window_rejected_when_aligned(self):
        # A window ending at 1120 (= 35 chunks) is not a GPU-reachable resume
        # point: aligned mode refuses it, upstream mode accepts it.
        for extra, expected in ((CKPT, 0), (None, 1120)):
            sched, req, st = self._setup(extra, 1200)
            self._put(sched, self._mla(st, 1152) + self._window(st, 1120))
            self.assertEqual(
                sched.get_num_new_matched_tokens(req, 0)[0], expected, extra
            )

    def test_adjacent_windows_realign(self):
        # Windows at 1024 and 1152 form one 8-chunk run. With the bound at
        # 1100 the SWA scan would end at 1088; alignment re-checks at 1024.
        sched, req, st = self._setup(CKPT, 1101)
        self._put(
            sched,
            self._mla(st, 1152) + self._window(st, 1024) + self._window(st, 1152),
        )
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], 1024)

    def test_realign_reverifies_triggering_group(self):
        # Group 0 finds the run ending at 1120 and rounds the bound to 1024,
        # but group 0 itself has no window at 1024: it must be re-scanned and
        # the hit must fall back to the complete windows at 512.
        sched, req, st = self._setup(CKPT, 1200)
        self._put(
            sched,
            self._mla(st, 1152)
            + self._window(st, 1120)
            + self._window(st, 1024, groups=SWA_GROUPS[1:])
            + self._window(st, 512),
        )
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], 512)

    def test_last_token_recomputed(self):
        # Prompt of exactly 1024 tokens: the hit stops at 1023 -> 896.
        sched, req, st = self._setup(CKPT, 1024)
        self._put(
            sched,
            self._mla(st, 1024) + self._window(st, 1024) + self._window(st, 896),
        )
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], 896)

    def test_local_gpu_hit_prefix(self):
        sched, req, st = self._setup(CKPT, 1100)
        mla = self._mla(st, 1024)
        self._put(sched, mla[5:] + self._window(st, 1024))
        self.assertEqual(sched.get_num_new_matched_tokens(req, 640)[0], 1024 - 640)
        self.assertEqual(sched.get_num_new_matched_tokens(req, 0)[0], 0)


# ---------------------------------------------------------------------------
# 3. Mini engine: real KV manager + connector + byte-level content model
# ---------------------------------------------------------------------------
class MiniEngine:
    """Drives the patched scheduler-side connector the way vLLM's Scheduler
    does, and models GPU / host block contents slot by slot.

    A slot written for token t of a request holds (group, fingerprint(t)),
    where the fingerprint is a hash of tokens[0..t]. A correct restore leaves
    exactly the values a GPU prefix-cache hit would expose. Stores built at
    step s run at step s+1: before that step's forward if the job is flushed,
    after it otherwise (the worst case the real side stream allows).
    """

    def __init__(self, extra=None, num_gpu_blocks=400, cpu_blocks=None):
        cpu_gib = (cpu_blocks or 20000) * 458_752 / (1 << 30)
        self.sched, self.kvc, self.vc, self.spec = _make_scheduler(
            extra, num_gpu_blocks=num_gpu_blocks, cpu_gib=cpu_gib
        )
        self.kvm = KVCacheManager(
            self.kvc,
            max_model_len=32768,
            scheduler_block_size=ALIGN,
            hash_block_size=TPH,
            enable_caching=True,
        )
        self.world = self.vc.parallel_config.world_size
        self.bs = [g.kv_cache_spec.block_size for g in self.kvc.kv_cache_groups]
        self.gpu: dict[int, dict[int, tuple]] = defaultdict(dict)
        self.host: dict[int, dict[int, tuple]] = {}
        self.fp: dict[str, list[int]] = {}
        self.pending_stores: list = []  # (job_id, TransferJob) from last step
        self.running: dict[str, Request] = {}
        self.first_sched: set[str] = set()
        self.finished_last_step: set[str] = set()
        self.hits: list[tuple[str, int, int]] = []  # (req, local, external)
        self.loads_checked = 0
        self._pending_add = None
        self._add_blocks: dict[str, tuple] = {}

    # -- content model ------------------------------------------------------
    def _fingerprints(self, req):
        fps = self.fp.setdefault(req.request_id, [])
        toks = req.all_token_ids
        h = fps[-1] if fps else 0
        for t in toks[len(fps) :]:
            h = hash((h, t))
            fps.append(h)
        return fps

    def _write(self, req, start, end):
        fps = self._fingerprints(req)
        block_ids = self.kvm.get_block_ids(req.request_id)
        for g, ids in enumerate(block_ids):
            bs = self.bs[g]
            for t in range(start, end):
                if g == RING_GROUP:
                    blk, slot = ids[0], t % bs
                else:
                    blk, slot = ids[t // bs], t % bs
                assert blk != 0, (g, t)
                self.gpu[blk][slot] = (g, fps[t])

    def check_state(self, req, upto):
        """GPU blocks of `req` hold the right values for a resume at `upto`."""
        fps = self._fingerprints(req)
        block_ids = self.kvm.get_block_ids(req.request_id)
        for g, ids in enumerate(block_ids):
            if g == RING_GROUP:
                continue
            bs = self.bs[g]
            lo = 0 if g == MLA_GROUP else max(0, upto - 127)
            for t in range(lo, upto):
                got = self.gpu[ids[t // bs]].get(t % bs)
                assert got == (g, fps[t]), (req.request_id, g, t, upto)

    # -- transfers ----------------------------------------------------------
    def _copy(self, job, to_host):
        gpu_spec = job.src_spec if to_host else job.dst_spec
        cpu_spec = job.dst_spec if to_host else job.src_spec
        gids = list(gpu_spec.block_ids)
        cids = list(cpu_spec.block_ids)
        assert sum(gpu_spec.group_sizes) == len(gids) == len(cids)
        assert len(gpu_spec.group_sizes) == len(self.bs)
        assert gpu_spec.group_sizes[RING_GROUP] == 0
        for gb, cb in zip(gids, cids):
            if to_host:
                self.host[cb] = dict(self.gpu[gb])
            else:
                self.gpu[gb] = dict(self.host[cb])

    def _complete(self, job_ids):
        if job_ids:
            meta = OffloadingWorkerMetadata(
                completed_jobs={j: self.world for j in job_ids}
            )
            self.sched.update_connector_output(
                KVConnectorOutput(kv_connector_worker_meta=meta)
            )

    # -- engine step --------------------------------------------------------
    def add(self, req):
        self.sched.on_new_request(req)
        blocks, local, _ = self.kvm.get_computed_blocks(req)
        ext, is_async = self.sched.get_num_new_matched_tokens(req, local)
        assert ext is not None
        self.hits.append((req.request_id, local, ext))
        if ext:
            assert is_async and (local + ext) % ALIGN == 0
            new = self.kvm.allocate_slots(
                req,
                0,
                num_new_computed_tokens=local,
                new_computed_blocks=blocks,
                num_external_computed_tokens=ext,
                delay_cache_blocks=True,
            )
            assert new is not None
            self.sched.update_state_after_alloc(
                req, self.kvm.get_blocks(req.request_id), ext
            )
            req.num_computed_tokens = local + ext
            req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self._pending_add = (req, blocks, local)
            return ext
        req.num_computed_tokens = 0
        self._add_blocks[req.request_id] = (blocks, local)
        req.status = RequestStatus.WAITING
        self.running[req.request_id] = req
        return 0

    def step(self, chunk=512, max_new=None):
        so = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], new_block_ids=[], resumed_req_ids=set()
            ),
            num_scheduled_tokens={},
            finished_req_ids=set(self.finished_last_step),
            preempted_req_ids=set(),
            kv_cache_block_copies=None,
            kv_connector_block_state=None,
        )
        self.finished_last_step = set()
        plan = []
        for rid, req in list(self.running.items()):
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                continue
            if rid not in self.first_sched:
                blocks, local = self._add_blocks.pop(rid, (None, 0))
                n = min(chunk, req.num_tokens - req.num_computed_tokens - local)
                new = self.kvm.allocate_slots(
                    req,
                    n,
                    num_new_computed_tokens=local,
                    new_computed_blocks=blocks,
                )
                assert new is not None, "GPU pool too small for the test"
                req.num_computed_tokens += local
                so.scheduled_new_reqs.append(
                    SimpleNamespace(
                        req_id=rid, block_ids=self.kvm.get_block_ids(rid)
                    )
                )
                self.first_sched.add(rid)
            else:
                n = min(chunk, req.num_tokens - req.num_computed_tokens)
                new = self.kvm.allocate_slots(req, n)
                assert new is not None, "GPU pool too small for the test"
                so.scheduled_cached_reqs.req_ids.append(rid)
                so.scheduled_cached_reqs.new_block_ids.append(
                    new.get_block_ids(allow_none=True)
                )
            req.status = RequestStatus.RUNNING
            so.num_scheduled_tokens[rid] = n
            plan.append((req, n))

        meta = self.sched.build_connector_meta(so)
        flush = set(meta.jobs_to_flush or ())
        # Deferred stores of the previous step: flushed ones before forward.
        late = []
        done = []
        for job_id, job in self.pending_stores:
            if job_id in flush:
                self._copy(job, to_host=True)
                done.append(job_id)
            else:
                late.append((job_id, job))
        for job_id, job in meta.load_jobs.items():
            self._copy(job, to_host=False)
            done.append(job_id)
        # Forward.
        for req, n in plan:
            self._write(req, req.num_computed_tokens, req.num_computed_tokens + n)
        for job_id, job in late:
            self._copy(job, to_host=True)
            done.append(job_id)
        self.pending_stores = list(meta.store_jobs.items())
        self._complete(done)
        # Loads completed: the waiting request becomes schedulable.
        if self._pending_add is not None and meta.load_jobs:
            req, blocks, local = self._pending_add
            self.check_state(req, req.num_computed_tokens)
            self.loads_checked += 1
            self.kvm.cache_blocks(req, req.num_computed_tokens)
            req.status = RequestStatus.WAITING
            self.running[req.request_id] = req
            self._pending_add = None
        # Advance requests; sample one token when the prompt is done.
        for req, n in plan:
            req.num_computed_tokens += n
            if req.num_computed_tokens >= req.num_tokens:
                tok = hash((req.request_id, req.num_tokens)) % 50000 + 10
                req.append_output_token_ids(tok)
                if req.num_output_tokens >= req.max_tokens:
                    self.finish(req)

    def finish(self, req):
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        self.sched.request_finished(req)
        self.kvm.free(req)
        del self.running[req.request_id]
        self.finished_last_step.add(req.request_id)

    def run_until_done(self, max_steps=5000):
        for _ in range(max_steps):
            if not self.running and not self.finished_last_step:
                break
            self.step(chunk=512)
        # Drain deferred stores.
        self.step()
        self.step()

    def run_request(self, req):
        ext = self.add(req)
        if ext:
            self.step()  # load step
        self.run_until_done()
        return ext

    # -- host accounting ----------------------------------------------------
    def host_keys_by_group(self):
        policy = self.sched.manager._policy
        out = defaultdict(int)
        for key in policy.blocks:
            out[int.from_bytes(key[-4:], "big")] += 1
        return out


def _conversations(engine, seeds, turns, prompt_len=700, out_len=300, tool_len=90):
    """Interleaved multi-turn conversations: turn t of every conversation runs
    before turn t+1 of any, so idle conversations lose their GPU blocks. Each
    turn's prompt = the previous turn's tokens + a tool result."""
    tokens = {seed: _tokens(seed, prompt_len) for seed in seeds}
    reqs = []
    for t in range(turns):
        for seed in seeds:
            req = _make_request(f"c{seed}t{t}", tokens[seed], max_tokens=out_len)
            engine.run_request(req)
            reqs.append(req)
            tokens[seed] = list(req.all_token_ids) + _tokens(
                seed * 31 + t, tool_len, 10**6
            )
    return reqs


class MiniEngineTest(unittest.TestCase):
    def test_turn_resume_from_host_after_gpu_eviction(self):
        eng = MiniEngine(CKPT)
        req1 = _make_request("a", _tokens(5, 900), max_tokens=300)
        eng.run_request(req1)
        total = req1.num_tokens
        # Evict everything on the GPU (as a long tool call under pressure).
        self.assertTrue(eng.kvm.reset_prefix_cache())
        req2 = _make_request(
            "b", list(req1.all_token_ids) + _tokens(6, 150, 10**6), max_tokens=40
        )
        ext = eng.run_request(req2)
        resume = (total - 1) // ALIGN * ALIGN
        self.assertEqual(ext, resume)
        self.assertEqual(eng.loads_checked, 1)

    def test_prompt_end_checkpoint_serves_divergent_continuation(self):
        eng = MiniEngine(CKPT)
        prompt = _tokens(8, 1000)
        eng.run_request(_make_request("p1", prompt, max_tokens=300))
        eng.kvm.reset_prefix_cache()
        # Same prompt, different continuation (e.g. a re-rendered turn).
        ext = eng.run_request(
            _make_request("p2", prompt + _tokens(9, 200, 10**6), max_tokens=8)
        )
        self.assertEqual(ext, 999 // ALIGN * ALIGN)

    def test_many_conversations_under_gpu_pressure(self):
        # Small GPU pool: blocks are freed, evicted and reused constantly, so
        # any stale or misplaced store would surface as a content mismatch in
        # check_state on a later restore.
        eng = MiniEngine(CKPT, num_gpu_blocks=260)
        reqs = _conversations(eng, seeds=range(1, 6), turns=3)
        self.assertEqual(len(reqs), 15)
        # Every turn after the first resumes from the host (GPU copy evicted).
        self.assertGreaterEqual(eng.loads_checked, 8)
        for rid, local, ext in eng.hits:
            self.assertEqual((local + ext) % ALIGN, 0, rid)

    def test_retried_checkpoint_never_stores_reused_blocks(self):
        # The host refuses s1's stores (full) until its prompt-end window has
        # slid out on the GPU and a concurrent prefill (f1) has recycled those
        # blocks. The retried checkpoint must not copy recycled blocks; any
        # later restore must be content-exact (check_state).
        eng = MiniEngine(CKPT, num_gpu_blocks=260)
        real = eng.sched.manager.prepare_store
        state = {"fail": True, "refused": 0}

        def flaky(keys, ctx):
            if state["fail"] and ctx is not None and ctx.req_id == "s1":
                state["refused"] += 1
                return None
            return real(keys, ctx)

        eng.sched.manager.prepare_store = flaky
        prompt = _tokens(21, 1500)
        eng.add(_make_request("s1", prompt, max_tokens=300))
        eng.add(_make_request("f1", _tokens(23, 3000), max_tokens=4))
        for _ in range(16):
            eng.step(chunk=256)
        state["fail"] = False
        eng.run_until_done()
        self.assertGreater(state["refused"], 5)
        self.assertTrue(eng.kvm.reset_prefix_cache())
        ext = eng.run_request(
            _make_request("s2", prompt + _tokens(22, 50, 10**6), max_tokens=4)
        )
        self.assertEqual(ext % ALIGN, 0)
        self.assertEqual(eng.loads_checked, 1 if ext else 0)
        # The prompt-end window (1408) could not be stored intact.
        self.assertLess(ext, 1499 // ALIGN * ALIGN)

    def test_checkpoint_policy_host_footprint(self):
        footprints = {}
        for name, extra in (("all", None), ("checkpoint", CKPT)):
            eng = MiniEngine(extra)
            req = _make_request("f", _tokens(11, 2000), max_tokens=700)
            eng.run_request(req)
            by_group = eng.host_keys_by_group()
            self.assertNotIn(RING_GROUP, by_group)
            computed = req.num_tokens - 1
            self.assertEqual(by_group[MLA_GROUP], computed // 128)
            footprints[name] = sum(by_group[g] for g in SWA_GROUPS)
            if name == "all":
                self.assertEqual(by_group[0], computed // 32)
            else:
                # prompt-end window + final rolling window, 4 chunks each.
                for g in SWA_GROUPS:
                    self.assertLessEqual(by_group[g], 8, by_group)
                self.assertFalse(eng.sched._pending_discards)
        self.assertGreater(footprints["all"], 10 * footprints["checkpoint"])

    def test_finished_request_last_chunk_not_stored(self):
        # 1000 prompt + 280 output = 1280 tokens, a multiple of 128: the last
        # sampled token's KV was never computed, so chunk [1152, 1280) and the
        # SWA chunk [1248, 1280) must not be on the host.
        eng = MiniEngine(CKPT)
        req = _make_request("z", _tokens(12, 1000), max_tokens=280)
        eng.run_request(req)
        self.assertEqual(req.num_tokens, 1280)
        st_keys = eng.sched.manager._policy.blocks
        mla_last = make_offload_key(req.block_hashes[1280 // 32 - 1], MLA_GROUP)
        swa_last = make_offload_key(req.block_hashes[1280 // 32 - 1], 0)
        mla_prev = make_offload_key(req.block_hashes[1152 // 32 - 1], MLA_GROUP)
        self.assertNotIn(mla_last, st_keys)
        self.assertNotIn(swa_last, st_keys)
        self.assertIn(mla_prev, st_keys)

    def test_default_policy_still_restores(self):
        # Upstream store policy (dense SWA) on the same layout: also correct.
        eng = MiniEngine({"align_hits": True})
        req1 = _make_request("d1", _tokens(13, 800), max_tokens=200)
        eng.run_request(req1)
        eng.kvm.reset_prefix_cache()
        req2 = _make_request(
            "d2", list(req1.all_token_ids) + _tokens(14, 64, 10**6), max_tokens=8
        )
        self.assertEqual(eng.run_request(req2), (req1.num_tokens - 1) // 128 * 128)


# ---------------------------------------------------------------------------
# 4. CPU manager discard
# ---------------------------------------------------------------------------
class DiscardTest(unittest.TestCase):
    def test_discard_semantics(self):
        m = CPUOffloadingManager(num_blocks=8)
        keys = [make_offload_key(bytes([i]) * 8, 0) for i in range(4)]
        out = m.prepare_store(keys[:3], None)
        m.complete_store(keys[:2], None)  # keys[2] stays in flight
        m.prepare_load([keys[1]], None)  # keys[1] is being loaded
        pending = m.discard(keys)
        self.assertEqual(pending, [keys[2]])
        self.assertEqual(m.lookup(keys[0], None), LookupResult.MISS)
        self.assertEqual(m.lookup(keys[1], None), LookupResult.HIT)
        m.complete_store([keys[2]], None)
        self.assertEqual(m.discard([keys[2]]), [])
        self.assertEqual(m.lookup(keys[2], None), LookupResult.MISS)
        # Freed slots are reused.
        self.assertIsNotNone(m.prepare_store(keys[:1] + keys[3:], None))
        del out


# ---------------------------------------------------------------------------
# 5. Worker registration on CPU tensors (packed pool, scratch group present)
# ---------------------------------------------------------------------------
class WorkerRegistrationTest(unittest.TestCase):
    def test_packed_pool_single_region(self):
        kvc = build_kv_cache_config(16)
        raw, kv_caches = allocate_cpu_kv_caches(kvc)
        spec = mock.MagicMock()
        spec.replicated_layout = False
        worker = OffloadingConnectorWorker(spec, make_vllm_config(), kvc)
        worker.register_kv_caches(kv_caches)
        canonical = spec.get_worker.call_args[0][0]
        self.assertEqual(len(canonical.tensors), 1)
        t = canonical.tensors[0].tensor
        self.assertEqual(tuple(t.shape), (16, POOL_BLOCK_BYTES))
        self.assertEqual(t.data_ptr(), raw.data_ptr())
        self.assertEqual(len(canonical.group_data_refs), 9)
        for refs in canonical.group_data_refs:
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].page_size_bytes, POOL_BLOCK_BYTES)
        # Every layer's pages lie inside its block's 113,920-B row.
        for name, view in kv_caches.items():
            off = view.storage_offset() * view.element_size()
            self.assertLess(off % POOL_BLOCK_BYTES + 1, POOL_BLOCK_BYTES + 1, name)

    def test_ring_spec_is_circular_buffer(self):
        kvc = build_kv_cache_config(16)
        self.assertTrue(
            all(
                isinstance(s, CircularBufferSpec)
                for s in iter_layer_specs(kvc.kv_cache_groups[RING_GROUP].kv_cache_spec)
            )
        )


if __name__ == "__main__":
    unittest.main()
