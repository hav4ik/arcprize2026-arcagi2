#!/usr/bin/env python3
"""E042: NCCL bf16 all-reduce latency/bandwidth on the 4 GPUs at TP-relevant message sizes.

Run: torchrun --nproc-per-node 4 nccl_allreduce_bench.py [--label X]  (NCCL_* env vars apply).
Sizes are tokens x 5120 bf16 (one TP all-reduce of a [T, 5120] activation): T = 1 .. 8192 (96-192: decode batches).
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", default=os.environ.get("NCCL_PROTO", "default"))
    p.add_argument("--out")
    a = p.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rows = []
    for tokens in (1, 8, 32, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192):
        x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="cuda")
        iters = 50 if tokens <= 256 else 20
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        nbytes = x.numel() * 2
        rows.append({"tokens": tokens, "MiB": round(nbytes / 2**20, 3), "ms": round(dt * 1e3, 3),
                     "algbw_GBps": round(nbytes / dt / 1e9, 2),
                     "busbw_GBps": round(nbytes / dt / 1e9 * 2 * 3 / 4, 2)})
    if rank == 0:
        for r in rows:
            print(f"{a.label:12s} T={r['tokens']:5d} {r['MiB']:8.3f} MiB {r['ms']:8.3f} ms "
                  f"algbw {r['algbw_GBps']:6.2f} GB/s busbw {r['busbw_GBps']:6.2f} GB/s", flush=True)
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"label": a.label, "env": {k: v for k, v in os.environ.items()
                                                     if k.startswith("NCCL_")}, "rows": rows}, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
