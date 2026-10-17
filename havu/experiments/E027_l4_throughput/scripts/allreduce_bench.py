"""NCCL all-reduce timing across the 4 GPUs for tensor sizes TP4 inference uses per all-reduce
(bf16 [tokens, 5120]): decode steps (24-48 tokens) and prefill chunks (2048-8192 tokens).
A TP4 forward of Qwen3.8-27B does 2 all-reduces per layer x 64 layers = 128 per step.
usage: torchrun --nproc_per_node 4 allreduce_bench.py"""
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
for tokens in (24, 48, 512, 2048, 4096, 8192):
    x = torch.randn(tokens, 5120, dtype=torch.bfloat16, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    n = 50 if tokens <= 2048 else 20
    t0 = time.time()
    for _ in range(n):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    if r == 0:
        mb = x.numel() * 2 / 1e6
        print(f"tokens {tokens:5d}  {mb:7.1f} MB  {dt * 1e3:7.2f} ms per all-reduce  "
              f"bus {2 * 3 / 4 * mb / 1e3 / dt:6.1f} GB/s  x128 per step = {128 * dt:6.3f} s", flush=True)
dist.destroy_process_group()
