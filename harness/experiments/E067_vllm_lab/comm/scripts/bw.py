"""Host<->device copy bandwidth/latency on each GPU, alone and both at once.

Pinned host buffers, cudaMemcpyAsync via torch copy_(non_blocking=True), CUDA
events. Also GPU0->GPU1 copies (staged through host when P2P is unavailable).
Usage: python bw.py out.json
"""
import json
import sys

import torch

SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20]


def timed(fn, iters, devs):
    for _ in range(3):
        fn()
    for d in devs:
        torch.cuda.synchronize(d)
    starts = {d: torch.cuda.Event(enable_timing=True) for d in devs}
    ends = {d: torch.cuda.Event(enable_timing=True) for d in devs}
    for d in devs:
        with torch.cuda.device(d):
            starts[d].record()
    for _ in range(iters):
        fn()
    for d in devs:
        with torch.cuda.device(d):
            ends[d].record()
    for d in devs:
        torch.cuda.synchronize(d)
    return max(starts[d].elapsed_time(ends[d]) for d in devs) / iters * 1e3  # us


def main():
    out = {"p2p": {f"{i}->{j}": torch.cuda.can_device_access_peer(i, j)
                   for i in range(2) for j in range(2) if i != j}, "rows": []}
    for n in SIZES:
        iters = max(5, min(200, (256 << 20) // n))
        h = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        h2 = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        g = [torch.empty(n, dtype=torch.uint8, device=f"cuda:{i}") for i in range(2)]
        row = {"bytes": n}
        for i in range(2):
            row[f"h2d_gpu{i}_us"] = timed(lambda: g[i].copy_(h[i], non_blocking=True), iters, [i])
            row[f"d2h_gpu{i}_us"] = timed(lambda: h[i].copy_(g[i], non_blocking=True), iters, [i])

        def both_h2d():
            for i in range(2):
                with torch.cuda.device(i):
                    g[i].copy_(h[i], non_blocking=True)

        def both_d2h():
            for i in range(2):
                with torch.cuda.device(i):
                    h[i].copy_(g[i], non_blocking=True)

        def bidir():  # each GPU reads and writes host at once (2 streams)
            for i in range(2):
                with torch.cuda.device(i):
                    s = torch.cuda.current_stream()
                    h[i].copy_(g[i], non_blocking=True)
            for i in range(2):
                with torch.cuda.device(i):
                    g[i].copy_(h2[i], non_blocking=True)

        row["h2d_both_us"] = timed(both_h2d, iters, [0, 1])
        row["d2h_both_us"] = timed(both_d2h, iters, [0, 1])
        row["d2h_then_h2d_both_us"] = timed(bidir, iters, [0, 1])
        with torch.cuda.device(0):
            row["gpu0_to_gpu1_us"] = timed(lambda: g[1].copy_(g[0], non_blocking=True), iters, [0, 1])
        for k in list(row):
            if k.endswith("_us"):
                row[k.replace("_us", "_GBps")] = round(n / row[k] / 1e3, 2)
                row[k] = round(row[k], 1)
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
