"""CPU benchmark of the offloading connector's scheduler side under steady decode (E051 overhead hunt).

N requests with L-token prompts are prefilled in 2048-token chunks, then decode one token per request per step.
Every step runs the patched connector exactly as vLLM's scheduler does (build_connector_meta, then the workers' report
of completed jobs via update_connector_output) and times only those connector calls. Optional cProfile of the decode
steps. GPU-side costs (transfers, worker metadata aggregation) are not covered.

    PYTHONPATH=<patched vllm tree>:<this dir> python bench_scheduler.py --n 36 --len 40000 --steps 256 [--profile]
"""
import argparse
import cProfile
import pstats
import time
from types import SimpleNamespace

from test_dsv41_kv_offload import ALIGN, TPH, _make_request, _make_scheduler, _tokens

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=36)
    p.add_argument("--len", type=int, default=40000)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--policy", default="checkpoint")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    extra = {"offload_prompt_only": False, "swa_store_policy": a.policy, "eviction_policy": "lru"}
    blocks = a.n * (a.len // 128 + 64) * 2 + 4096
    sched, kvc, vc, spec = _make_scheduler(extra, num_gpu_blocks=blocks, cpu_gib=28)
    kvm = KVCacheManager(kvc, max_model_len=163840, scheduler_block_size=ALIGN, hash_block_size=TPH,
                         enable_caching=True)
    world = vc.parallel_config.world_size
    reqs = []
    for i in range(a.n):
        r = _make_request(f"r{i}", _tokens(i + 1, a.len), max_tokens=100000)
        sched.on_new_request(r)
        blk, local, _ = kvm.get_computed_blocks(r)
        sched.get_num_new_matched_tokens(r, local)
        r.num_computed_tokens = 0
        reqs.append(r)
    first = set()
    t_conn = []

    def step(chunk):
        so = SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()), num_scheduled_tokens={}, finished_req_ids=set(),
            preempted_req_ids=set(), kv_cache_block_copies=None, kv_connector_block_state=None)
        plan = []
        for r in reqs:
            n = min(chunk, r.num_tokens - r.num_computed_tokens) if r.num_computed_tokens < r.num_prompt_tokens else 1
            new = kvm.allocate_slots(r, n)
            assert new is not None, "pool too small"
            if r.request_id not in first:
                so.scheduled_new_reqs.append(SimpleNamespace(req_id=r.request_id,
                                                             block_ids=kvm.get_block_ids(r.request_id)))
                first.add(r.request_id)
            else:
                so.scheduled_cached_reqs.req_ids.append(r.request_id)
                so.scheduled_cached_reqs.new_block_ids.append(new.get_block_ids(allow_none=True))
            r.status = RequestStatus.RUNNING
            so.num_scheduled_tokens[r.request_id] = n
            plan.append((r, n))
        t0 = time.perf_counter()
        meta = sched.build_connector_meta(so)
        jobs = list(meta.store_jobs) + list(meta.load_jobs)
        if jobs:
            sched.update_connector_output(KVConnectorOutput(
                kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={j: world for j in jobs})))
        dt = time.perf_counter() - t0
        for r, n in plan:
            r.num_computed_tokens += n
            if r.num_computed_tokens >= r.num_tokens:
                r.append_output_token_ids(hash((r.request_id, r.num_tokens)) % 50000 + 10)
        return dt

    t0 = time.time()
    while any(r.num_computed_tokens < r.num_prompt_tokens for r in reqs):
        step(2048)
    print(f"prefill {a.n} x {a.len}: {time.time() - t0:.1f} s wall (incl. KV manager)")
    prof = cProfile.Profile() if a.profile else None
    if prof:
        prof.enable()
    for _ in range(a.steps):
        t_conn.append(step(1))
    if prof:
        prof.disable()
    t_conn.sort()
    mean = sum(t_conn) / len(t_conn)
    print(f"decode: connector ms/step mean {1000 * mean:.2f}, p50 {1000 * t_conn[len(t_conn) // 2]:.2f}, "
          f"p99 {1000 * t_conn[int(0.99 * len(t_conn))]:.2f}, max {1000 * t_conn[-1]:.2f} (n={a.n}, L={a.len})")
    if prof:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
