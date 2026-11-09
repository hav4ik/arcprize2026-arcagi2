# SPDX-License-Identifier: Apache-2.0
"""Differential check: with default kv_connector_extra_config, the patched
offloading scheduler emits exactly the same load/store jobs as the unpatched
fork on models that worked before the patch.

Run once per tree and compare the printed digests:
    PYTHONPATH=<base tree>:<tests>    python diff_default_behavior.py
    PYTHONPATH=<patched tree>:<tests> python diff_default_behavior.py
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import torch

from dsv41_fixture import make_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

init_none_hash(sha256)
BS = 16


def make_config(kind: str, num_blocks: int = 96) -> KVCacheConfig:
    kw = dict(block_size=BS, num_kv_heads=2, head_size=8, dtype=torch.float16)
    groups = [KVCacheGroupSpec(["fa0", "fa1"], FullAttentionSpec(**kw))]
    if kind == "hybrid":
        groups.append(
            KVCacheGroupSpec(["sw0", "sw1"], SlidingWindowSpec(**kw, sliding_window=48))
        )
    page = groups[0].kv_cache_spec.page_size_bytes
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=page * num_blocks, layers=["fa0", "sw0"], layer_stride=0,
                          block_stride=page),
            KVCacheTensor(size=page * num_blocks, layers=["fa1", "sw1"], layer_stride=0,
                          block_stride=page),
        ],
        kv_cache_groups=groups,
    )


def run(kind: str, extra: dict) -> list:
    kvc = make_config(kind)
    vc = make_vllm_config({"cpu_bytes_to_use": 1 << 24, **extra}, tp=1)
    vc.cache_config.block_size = BS
    vc.model_config.use_mla = False
    sched = OffloadingConnectorScheduler(
        CPUOffloadingSpec(build_offloading_config(vc, kvc)), vc, kvc
    )
    kvm = KVCacheManager(kvc, max_model_len=4096, scheduler_block_size=BS,
                         hash_block_size=BS, enable_caching=True)
    log = []
    finished_prev: set[str] = set()
    first: set[str] = set()

    def emit(meta):
        for jid, job in sorted(meta.load_jobs.items()):
            log.append(("load", jid, list(map(int, job.src_spec.block_ids)),
                        list(map(int, job.dst_spec.block_ids)),
                        list(job.dst_spec.group_sizes)))
        for jid, job in sorted(meta.store_jobs.items()):
            log.append(("store", jid, list(map(int, job.src_spec.block_ids)),
                        list(map(int, job.dst_spec.block_ids)),
                        list(job.src_spec.group_sizes)))
        done = list(meta.load_jobs) + list(meta.store_jobs)
        if done:
            sched.update_connector_output(KVConnectorOutput(
                kv_connector_worker_meta=OffloadingWorkerMetadata(
                    completed_jobs={j: 1 for j in done})))

    def step(reqs, chunk):
        nonlocal finished_prev
        so = SimpleNamespace(
            scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(
                req_ids=[], new_block_ids=[], resumed_req_ids=set()),
            num_scheduled_tokens={}, finished_req_ids=set(finished_prev),
            preempted_req_ids=set(), kv_cache_block_copies=None,
            kv_connector_block_state=None)
        finished_prev = set()
        plan = []
        for req, blocks, local in reqs:
            n = min(chunk, req.num_tokens - req.num_computed_tokens - local)
            if req.request_id not in first:
                new = kvm.allocate_slots(req, n, num_new_computed_tokens=local,
                                         new_computed_blocks=blocks)
                req.num_computed_tokens += local
                so.scheduled_new_reqs.append(SimpleNamespace(
                    req_id=req.request_id, block_ids=kvm.get_block_ids(req.request_id)))
                first.add(req.request_id)
            else:
                new = kvm.allocate_slots(req, n)
                so.scheduled_cached_reqs.req_ids.append(req.request_id)
                so.scheduled_cached_reqs.new_block_ids.append(
                    new.get_block_ids(allow_none=True))
            assert new is not None
            req.status = RequestStatus.RUNNING
            so.num_scheduled_tokens[req.request_id] = n
            plan.append((req, n))
        emit(sched.build_connector_meta(so))
        for req, n in plan:
            req.num_computed_tokens += n
            if req.num_computed_tokens >= req.num_tokens:
                req.append_output_token_ids(7 + req.num_tokens % 97)
                if req.num_output_tokens >= req.max_tokens:
                    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                    sched.request_finished(req)
                    kvm.free(req)
                    finished_prev.add(req.request_id)

    prompts = [[(i * 31 + s) % 1000 for i in range(n)]
               for s, n in ((1, 200), (1, 260), (2, 150), (1, 200))]
    for idx, prompt in enumerate(prompts):
        req = Request(f"r{idx}", prompt, SamplingParams(max_tokens=40), None,
                      block_hasher=get_request_block_hasher(BS, sha256))
        sched.on_new_request(req)
        blocks, local, _ = kvm.get_computed_blocks(req)
        ext, _ = sched.get_num_new_matched_tokens(req, local)
        log.append(("hit", req.request_id, local, ext))
        if ext:
            kvm.allocate_slots(req, 0, num_new_computed_tokens=local,
                               new_computed_blocks=blocks,
                               num_external_computed_tokens=ext,
                               delay_cache_blocks=True)
            sched.update_state_after_alloc(req, kvm.get_blocks(req.request_id), ext)
            req.num_computed_tokens = local + ext
            step([], 64)
            kvm.cache_blocks(req, req.num_computed_tokens)
            blocks, local = None, 0
        state = [(req, blocks, local)]
        while req.request_id not in finished_prev:
            step(state, 64)
            state = [(req, None, 0)]
        step([], 64)
        if idx == 1:
            kvm.reset_prefix_cache()  # force host hits for the next requests
    return log


if __name__ == "__main__":
    out = {}
    for kind in ("full", "hybrid"):
        for extra in ({"offload_prompt_only": True}, {"offload_prompt_only": False}):
            log = run(kind, extra)
            key = f"{kind}/{json.dumps(extra)}"
            digest = hashlib.sha256(json.dumps(log).encode()).hexdigest()[:16]
            hits = [e for e in log if e[0] == "hit"]
            out[key] = (digest, len(log), hits)
    for k, v in out.items():
        print(k, v)
