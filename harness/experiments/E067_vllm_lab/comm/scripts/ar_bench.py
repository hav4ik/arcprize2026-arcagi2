"""NCCL all-reduce latency / bus bandwidth, 2 ranks (torch.distributed).

Per size: eager back-to-back in-place calls (stream-serialised, so each call
waits for the previous one: the dependent latency a decode step sees) and the
same calls captured in a CUDA graph (no CPU launch cost, as vLLM decode runs).
Reported time = slowest rank's median over reps. busbw = algbw * 2(n-1)/n.
NCCL env (NCCL_PROTO, NCCL_P2P_DISABLE, ...) is taken from the environment.
Optionally also vLLM's custom all-reduce (--custom), which needs P2P.

Usage: python ar_bench.py --label default --out res.json [--custom]
"""
import argparse
import json
import os
import statistics
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DECODE = [8 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20]
PREFILL = [2 << 20, 4 << 20, 8 << 20, 16 << 20, 32 << 20, 64 << 20]


def bench_eager(fn, iters, reps=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / iters * 1e3)
    return statistics.median(ts)


def bench_graph(fn, per_graph, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(per_graph):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / per_graph * 1e3)
    del g
    return statistics.median(ts)


def worker(rank, world, init_file, args, q):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank,
                            world_size=world, device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    rows = []
    sizes = DECODE + PREFILL
    custom = None
    if args.custom:
        custom = make_custom(cpu, rank, args.force_custom)
    for n in sizes:
        numel = n // 2
        x = torch.randn(numel, dtype=torch.bfloat16, device="cuda") * 0.01
        iters = max(10, min(200, (64 << 20) // n))
        per_graph = max(4, min(50, (32 << 20) // n))
        dist.barrier(cpu)
        r = {"bytes": n}
        r["eager_us"] = bench_eager(lambda: dist.all_reduce(x), iters)
        dist.barrier(cpu)
        r["graph_us"] = bench_graph(lambda: dist.all_reduce(x), per_graph)
        if custom is not None and n <= custom.max_size:
            y = torch.empty_like(x)
            dist.barrier(cpu)
            try:
                r["custom_eager_us"] = bench_eager(lambda: custom.all_reduce(x, out=y, registered=False), iters)
            except Exception as e:  # noqa: BLE001
                r["custom_err"] = repr(e)[:200]
        # slowest rank
        allr = [None] * world
        dist.all_gather_object(allr, r, group=cpu)
        if rank == 0:
            m = {"bytes": n}
            for k in r:
                if k.endswith("_us"):
                    m[k] = round(max(a[k] for a in allr if k in a), 1)
                    m[k.replace("_us", "_busbw_GBps")] = round(n / m[k] / 1e3 * 2 * (world - 1) / world, 2)
                elif k != "bytes":
                    m[k] = r[k]
            rows.append(m)
            print(args.label, json.dumps(m), flush=True)
        del x
    # correctness: sum of rank-dependent values
    x = torch.full((4096,), float(rank + 1), dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(x)
    ok = bool((x == sum(range(1, world + 1))).all())
    if rank == 0:
        q.put({"label": args.label, "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
               "nccl": ".".join(map(str, torch.cuda.nccl.version())), "correct": ok, "rows": rows})
    dist.barrier(cpu)
    dist.destroy_process_group()


def make_custom(cpu_group, rank, force):
    """vLLM 0.20 custom all-reduce on a gloo group. force=True skips the P2P
    test to see what happens without P2P."""
    from vllm.distributed.device_communicators import custom_all_reduce as car
    if force:
        car._can_p2p = lambda r, w: True
    try:
        c = car.CustomAllreduce(group=cpu_group, device=torch.device("cuda", rank))
    except Exception as e:  # noqa: BLE001
        print(f"rank {rank}: custom all-reduce init raised {e!r}"[:300], flush=True)
        return None
    print(f"rank {rank}: custom all-reduce disabled={c.disabled}", flush=True)
    return None if c.disabled else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--custom", action="store_true")
    ap.add_argument("--force-custom", action="store_true")
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    fd, init_file = tempfile.mkstemp(prefix="arinit_", dir=os.environ.get("LAB_TMP", "/tmp"))
    os.close(fd)
    os.unlink(init_file)
    ps = [ctx.Process(target=worker, args=(r, 2, init_file, args, q)) for r in range(2)]
    for p in ps:
        p.start()
    import queue
    while True:  # give up as soon as a rank dies
        try:
            res = q.get(timeout=2)
            break
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in ps):
                for p in ps:
                    p.kill()
                raise SystemExit("a rank failed")
    for p in ps:
        p.join(timeout=120)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
