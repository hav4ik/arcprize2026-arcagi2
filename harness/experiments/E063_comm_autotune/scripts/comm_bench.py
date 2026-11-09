"""E063: time every lossless collective path at decode/prefill sizes on this box.

Usage: python comm_bench.py OUT.json [world]   (spawns `world` ranks, one per GPU)
Paths: NCCL all-reduce, host all-reduce (Manh's VLLM_HOST_ALLREDUCE, lossless mode 0),
custom all-reduce (allow_pcie; only where P2P works), NCCL all-gather, host gather,
NCCL grouped send/recv all-to-all (DCP partials). NCCL's own transport is whatever
NCCL picked under this process's env (run it again with NCCL_* envs to compare).
Times are the slowest rank's median ms per call (comm_autotune.time_candidates + agree).
"""
import datetime, json, os, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp


def worker(rank, world, port, out_path):
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=300))
    from vllm.distributed.device_communicators.comm_autotune import agree, time_candidates
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.distributed.device_communicators.host_all_reduce import HostAllReduce
    from vllm.distributed.device_communicators.host_gather import HostGather
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    dev = torch.device("cuda", rank)
    g = dist.group.WORLD
    nccl = PyNcclCommunicator(g, device=dev)
    host = HostAllReduce(g, dev, max_bytes=24 << 20)
    custom = CustomAllreduce(g, dev, allow_pcie=True)
    hg = HostGather(g, dev, max_bytes=8 << 20)
    sizes = [1 << k for k in range(12, 25)]  # 4 KiB .. 16 MiB per rank

    ar = {"nccl_ar": (nccl.all_reduce, lambda x: True)}
    if not host.disabled:
        ar["host_ar"] = (host.all_reduce, host.should_use)
    if not custom.disabled:
        ar["custom_ar"] = (custom.custom_all_reduce, custom.should_custom_ar)
    t_ar = time_candidates(ar, sizes, lambda n: torch.randn(n // 2, dtype=torch.bfloat16, device=dev), g)

    def nccl_ag(x):
        o = x.new_empty((world * x.numel(),)); nccl.all_gather(o, x); return o

    def nccl_a2a(x):  # x: this rank's packed partials for every peer, world chunks
        chunks = x.view(world, -1); recv = torch.empty_like(chunks)
        nccl.group_start()
        for p in range(world):
            if p != rank:
                nccl.send(chunks[p], p); nccl.recv(recv[p], p)
        nccl.group_end()
        return recv

    ag = {"nccl_ag": (nccl_ag, lambda x: True), "nccl_a2a": (nccl_a2a, lambda x: x.numel() % world == 0)}
    if not hg.disabled:
        ag["host_ag"] = (lambda x: hg.all_gather(x, 0), hg.can_gather)
    t_ag = time_candidates(ag, sizes[:-2], lambda n: torch.empty(n, dtype=torch.uint8, device=dev), g)
    times = {**t_ar, **{k: v + [float("inf")] * 2 for k, v in t_ag.items()}}
    agree(times, g)
    if rank == 0:
        env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
        json.dump({"world": world, "sizes": sizes, "times_ms": times, "nccl_env": env,
                   "custom_ar": not custom.disabled, "host_ar": not host.disabled,
                   "host_gather": not hg.disabled, "nccl_version": nccl.nccl_version}, open(out_path, "w"), indent=1)
        print(f"{'KiB':>8} " + " ".join(f"{k:>10}" for k in times))
        for j, s in enumerate(sizes):
            print(f"{s/1024:8g} " + " ".join(f"{times[k][j]:10.4f}" for k in times))
    host.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    from vllm.utils.network_utils import get_open_port
    world = int(sys.argv[2]) if len(sys.argv) > 2 else torch.cuda.device_count()
    mp.spawn(worker, args=(world, get_open_port(), sys.argv[1]), nprocs=world)
