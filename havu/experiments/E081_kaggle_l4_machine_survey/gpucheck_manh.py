"""Per-GPU pinned host<->GPU copy speed and NCCL latency for every GPU pair."""

import itertools
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [528384, 4227072]  # bytes per rank: 516 KB, 4 MB
_port = [29500]


def copy_speed():
    for g in range(torch.cuda.device_count()):
        torch.cuda.set_device(g)
        h = torch.empty(64 << 20, dtype=torch.uint8).pin_memory()
        d = torch.empty_like(h, device=f"cuda:{g}")
        speed = []
        for fn in (lambda: d.copy_(h, non_blocking=True),
                   lambda: h.copy_(d, non_blocking=True)):
            fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            speed.append(10 * 64 / 1024 / (time.perf_counter() - t))
        print(f"gpu{g}  H2D {speed[0]:5.1f} GB/s  D2H {speed[1]:5.1f} GB/s",
              flush=True)
        del h, d
    torch.cuda.empty_cache()


def pair_worker(rank, gpus, port, q):
    torch.cuda.set_device(gpus[rank])
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2)
    peer = 1 - rank
    res = {}
    for size in SIZES:
        send = torch.ones(size, dtype=torch.uint8, device="cuda")
        recv = torch.empty(2 * size, dtype=torch.uint8, device="cuda")
        p2p = torch.empty(size, dtype=torch.uint8, device="cuda")

        def all_gather():
            dist.all_gather_into_tensor(recv, send)

        def send_recv():
            ops = [dist.P2POp(dist.isend, send, peer),
                   dist.P2POp(dist.irecv, p2p, peer)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()

        for name, fn in (("ag", all_gather), ("sr", send_recv)):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(50):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b) * 1000)
            res[f"{name}{size // 1024}K"] = sorted(times)[len(times) // 2]
    if rank == 0:
        q.put((tuple(gpus), res))
    dist.destroy_process_group()


def run_pairs(groups):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for gpus in groups:
        _port[0] += 1
        for rank in range(2):
            p = ctx.Process(target=pair_worker,
                            args=(rank, list(gpus), _port[0], q))
            p.start()
            procs.append(p)
    out = dict(q.get(timeout=300) for _ in groups)
    for p in procs:
        p.join()
    return out


def show(mode, out):
    for (a, b), r in sorted(out.items()):
        cols = "  ".join(f"{k} {v:7.0f}" for k, v in r.items())
        print(f"{mode:8s} {a}-{b}  {cols}", flush=True)


if __name__ == "__main__":
    n = torch.cuda.device_count()
    print(f"{n} GPUs, NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}")
    copy_speed()
    print("\nNCCL median us (ag = all_gather, sr = send/recv, per-rank bytes)")
    for pair in itertools.combinations(range(n), 2):
        show("alone", run_pairs([pair]))
    if n == 4:
        for split in ([(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]):
            show("together", run_pairs(split))

# !nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.width.max --format=csv
# !python3 -u /tmp/gpucheck.py 2>&1 | grep -v FutureWarning | grep -v 'return func'
