#!/usr/bin/env python3
"""E042: sustained bf16 matmul TFLOPS on every GPU at once for ~60 s (thermal throttling check)."""
import sys, time
import torch
import torch.multiprocessing as mp


def work(dev, secs, q):
    torch.cuda.set_device(dev)
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda"); b = torch.randn_like(a)
    flop = 2 * 8192**3; t_end = time.time() + secs; marks = []
    while time.time() < t_end:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            a @ b
        torch.cuda.synchronize(); marks.append(20 * flop / (time.perf_counter() - t0) / 1e12)
    q.put((dev, marks[0], sum(marks[-5:]) / 5))


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    mp.set_start_method("spawn"); q = mp.Queue()
    ps = [mp.Process(target=work, args=(d, secs, q)) for d in range(torch.cuda.device_count())]
    [p.start() for p in ps]; [p.join() for p in ps]
    for dev, first, last in sorted(q.get() for _ in ps):
        print(f"GPU{dev}: bf16 matmul first {first:6.1f} TFLOPS, last-5 mean {last:6.1f} TFLOPS")
