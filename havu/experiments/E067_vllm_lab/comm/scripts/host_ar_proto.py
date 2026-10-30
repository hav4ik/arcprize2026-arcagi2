"""Host all-reduce prototypes vs NCCL, 2 ranks, bf16.

Paths (one process per rank, the same message on each):
  nccl            torch.distributed NCCL (whatever NCCL_* env says)
  staged-spin     Python: cudaMemcpyAsync D2H into a shared pinned /dev/shm slot,
                  host waits for it (spinning on the event), flags in shared
                  memory, each rank sums half on the CPU (torch bf16 add = fp32
                  sum rounded once), flags, cudaMemcpyAsync H2D.
  staged-block    the same with a blocking-sync event (the thread sleeps while
                  the GPU works) and yielding flag polls: a "cold" host.
  kernel-warm     har_ext.cu: one GPU kernel stores straight into mapped host
                  memory and polls host flags; persistent C++ reducer threads
                  hot-spin (keep-warm); unit-pipelined; no host sync; graphable.
  kernel-cold     the same reducers sleeping poll_ns between polls.
Each path runs back to back (dependent calls on one stream) and, with --gap-us,
with a GPU busy kernel of that length before every call (decode-like: compute
between collectives; a cold host falls asleep during it). Graph timings for
nccl and kernel paths. Results are checked against NCCL (bitwise).

Usage: python host_ar_proto.py --out res.json [--gap-us 0,200] [--paths ...]
"""
import argparse
import ctypes
import json
import mmap
import os
import statistics
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [8 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20,
         2 << 20, 4 << 20, 8 << 20, 16 << 20, 32 << 20, 64 << 20]
MAX_BYTES = 64 << 20


def load_ext():
    from torch.utils.cpp_extension import load
    return load(name="har_ext", sources=[os.path.join(HERE, "har_ext.cu")],
                extra_cuda_cflags=["-O3", "-Xcompiler", "-O3,-march=native,-funroll-loops"],
                extra_cflags=["-O3", "-march=native"], verbose=False)


def open_segment(path, size, create):
    if create:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(fd, size)
    else:
        fd = os.open(path, os.O_RDWR)
    mm = mmap.mmap(fd, size)
    os.close(fd)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    return mm, addr


class Staged:
    """Python host all-reduce through a shared pinned segment."""

    def __init__(self, ext, path, rank, world, create, blocking):
        self.rank, self.world, self.blocking = rank, world, blocking
        self.flag_bytes = 4096
        size = self.flag_bytes + (world + 1) * MAX_BYTES
        self.mm, self.addr = open_segment(path, size, create)
        ext.register_host(self.addr, size)
        self.flags = np.frombuffer(self.mm, dtype=np.uint32, count=self.flag_bytes // 4)
        n = MAX_BYTES // 2
        self.slots = [torch.frombuffer(self.mm, dtype=torch.bfloat16, count=n,
                                       offset=self.flag_bytes + w * MAX_BYTES) for w in range(world)]
        self.res = torch.frombuffer(self.mm, dtype=torch.bfloat16, count=n,
                                    offset=self.flag_bytes + world * MAX_BYTES)
        assert self.slots[0].is_pinned(), "segment not seen as pinned"
        self.seq = 0
        self.ev = torch.cuda.Event(blocking=blocking)

    def _wait(self, idx, seq):
        f = self.flags
        if self.blocking:
            while f[idx] != seq:
                time.sleep(0)
        else:
            while f[idx] != seq:
                pass

    def all_reduce(self, x, out):
        self.seq += 1
        seq, r, w = self.seq, self.rank, self.world
        n = x.numel()
        self.slots[r][:n].copy_(x, non_blocking=True)
        self.ev.record()
        if self.blocking:
            self.ev.synchronize()
        else:
            while not self.ev.query():
                pass
        self.flags[16 * r] = seq
        for p in range(w):
            self._wait(16 * p, seq)
        lo, hi = n * r // w, n * (r + 1) // w
        acc = self.slots[0][lo:hi]
        torch.add(acc, self.slots[1][lo:hi], out=self.res[lo:hi])
        for p in range(2, w):
            self.res[lo:hi].add_(self.slots[p][lo:hi])
        self.flags[16 * (w + r)] = seq
        for p in range(w):
            self._wait(16 * (w + p), seq)
        out.copy_(self.res[:n], non_blocking=True)
        return out


class Kernel:
    """har_ext.cu zero-copy kernel + C++ reducer threads."""

    def __init__(self, ext, path, rank, world, create, unit, threads, poll_ns, writers, readers, block):
        self.ext = ext
        self.rank, self.world = rank, world
        self.unit, self.writers, self.readers, self.block = unit, writers, readers, block
        lay = ext.layout(MAX_BYTES, unit, world)
        size = lay[-1]
        self.mm, self.addr = open_segment(path, size, create)
        self.dev = ext.register_host(self.addr, size)
        self.ctr = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.threads, self.poll_ns = threads, poll_ns
        self.started = False

    def start(self):
        self.ext.start(self.addr, MAX_BYTES, self.unit, self.world, self.rank, self.threads, self.poll_ns)
        self.started = True

    def all_reduce(self, x, out):
        self.ext.all_reduce(x, out, self.dev, MAX_BYTES, self.unit, self.world, self.rank,
                            self.writers, self.readers, self.block, self.ctr)
        return out

    def close(self):
        if self.started:
            self.ext.stop()
            self.started = False


def time_loop(fn, iters, gap_cycles, reps=5):
    for _ in range(5):
        if gap_cycles:
            torch.cuda._sleep(gap_cycles)
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            if gap_cycles:
                torch.cuda._sleep(gap_cycles)
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / iters * 1e3)
    return statistics.median(ts)


def time_graph(fn, per_graph, reps=5):
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
    return statistics.median(ts)


def calibrate_sleep(us):
    cyc = 100000
    for _ in range(3):
        torch.cuda._sleep(cyc)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        torch.cuda._sleep(cyc)
    b.record()
    b.synchronize()
    per = a.elapsed_time(b) / 20 * 1e3
    return int(cyc * us / per)


def worker(rank, world, init_file, seg_dir, args, q):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    cpu = dist.new_group(backend="gloo")
    torch.set_num_threads(args.cpu_threads)
    ext = load_ext() if rank == 0 else None
    dist.barrier(cpu)
    if ext is None:
        ext = load_ext()
    paths = args.paths.split(",")
    objs = {}
    for name in paths:
        if name == "nccl":
            continue
        box = [os.path.join(seg_dir, f"seg_{name}_{os.getpid()}")]
        dist.broadcast_object_list(box, src=0, group=cpu)
        create = rank == 0
        if not create:
            dist.barrier(cpu)
        if name.startswith("staged"):
            o = Staged(ext, box[0], rank, world, create, blocking=name.endswith("block"))
        else:
            poll = 0 if name == "kernel-warm" else args.cold_poll_ns
            o = Kernel(ext, box[0], rank, world, create, args.unit_kb << 10, args.threads, poll,
                       args.writers, args.readers, args.block)
        if create:
            dist.barrier(cpu)
        dist.barrier(cpu)
        if rank == 0:
            os.unlink(box[0])
        objs[name] = o
    gaps = [int(g) for g in args.gap_us.split(",")]
    gap_cycles = {g: (calibrate_sleep(g) if g else 0) for g in gaps}
    rows = []
    for n in SIZES:
        numel = n // 2
        g0 = torch.Generator(device="cuda").manual_seed(1234 + rank)
        x = torch.randn(numel, generator=g0, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        ref = x.clone()
        dist.all_reduce(ref)
        out = torch.empty_like(x)
        iters = max(10, min(200, (32 << 20) // n))
        row = {"bytes": n}
        for name in paths:
            if name == "nccl":
                y = x.clone()
                fn = lambda: dist.all_reduce(y)  # noqa: E731
            else:
                o = objs[name]
                if isinstance(o, Kernel) and not o.started:
                    o.start()
                fn = lambda o=o: o.all_reduce(x, out)  # noqa: E731
                dist.barrier(cpu)
                fn()
                torch.cuda.synchronize()
                row[f"{name}_mismatch"] = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
            for gname, gc in gap_cycles.items():
                dist.barrier(cpu)
                t = time_loop(fn, iters if not gc else max(10, iters // 4), gc)
                if gc:
                    dist.barrier(cpu)
                    tg = time_loop(lambda: None, max(10, iters // 4), gc)
                    t -= tg
                row[f"{name}_gap{gname}_us"] = t
            if name == "nccl" or name.startswith("kernel"):
                dist.barrier(cpu)
                row[f"{name}_graph_us"] = time_graph(fn, max(4, min(50, (32 << 20) // n)))
        allr = [None] * world
        dist.all_gather_object(allr, row, group=cpu)
        if rank == 0:
            m = {"bytes": n}
            for k in row:
                if k == "bytes":
                    continue
                vals = [a[k] for a in allr]
                m[k] = round(max(vals), 1) if k.endswith("_us") else max(vals)
            rows.append(m)
            print(json.dumps(m), flush=True)
    for o in objs.values():
        if isinstance(o, Kernel):
            o.close()
    if rank == 0:
        q.put({"args": vars(args), "rows": rows,
               "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}})
    dist.barrier(cpu)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paths", default="nccl,staged-spin,staged-block,kernel-warm,kernel-cold")
    ap.add_argument("--gap-us", default="0,200")
    ap.add_argument("--cpu-threads", type=int, default=4, help="torch CPU threads (staged sums)")
    ap.add_argument("--threads", type=int, default=2, help="C++ reducer threads per rank")
    ap.add_argument("--unit-kb", type=int, default=16)
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--readers", type=int, default=12)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--cold-poll-ns", type=int, default=50000)
    args = ap.parse_args()
    tmp = os.environ.get("LAB_TMP", "/tmp")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    fd, init_file = tempfile.mkstemp(prefix="harinit_", dir=tmp)
    os.close(fd)
    os.unlink(init_file)
    ps = [ctx.Process(target=worker, args=(r, 2, init_file, "/dev/shm", args, q)) for r in range(2)]
    for p in ps:
        p.start()
    res = q.get(timeout=1500)
    for p in ps:
        p.join(timeout=120)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
