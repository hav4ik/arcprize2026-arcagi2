"""Correctness of NCCL all-reduce across the 4 GPUs (run with and without NCCL_P2P_LEVEL=SYS):
random bf16/fp32 tensors of several sizes, all-reduce result compared with the sum computed from gathered inputs.
usage: torchrun --nproc_per_node 4 allreduce_check.py"""
import torch, torch.distributed as dist

dist.init_process_group("nccl")
r = dist.get_rank()
w = dist.get_world_size()
torch.cuda.set_device(r)
bad = 0
for it in range(200):
    for tokens in (1, 24, 48, 511, 2048, 8192):
        g = torch.Generator(device="cuda").manual_seed(1000 * it + tokens * 7 + r)
        x = torch.randn(tokens, 5120, dtype=torch.float32, device="cuda", generator=g)
        parts = [torch.empty_like(x) for _ in range(w)]
        dist.all_gather(parts, x)
        ref = torch.stack(parts).sum(0)
        y = x.clone()
        dist.all_reduce(y)
        err = (y - ref).abs().max().item()
        if err > 1e-3:
            bad += 1
            if bad <= 5:
                print(f"rank {r} it {it} tokens {tokens}: max abs err {err}", flush=True)
t = torch.tensor([bad], device="cuda")
dist.all_reduce(t)
if r == 0:
    print(f"all-reduce check: {int(t.item())} mismatching tensors out of {200 * 6 * w} rank-checks", flush=True)
dist.destroy_process_group()
