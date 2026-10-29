"""E063: does NCCL_PROTO apply per communicator within one process?
Creates a default PyNccl communicator, then one with NCCL_PROTO=LL set only during its init,
and times all-reduce / grouped send-recv on both at 0.5-4 MiB."""
import datetime, os, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp


def worker(rank, world, port):
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=300))
    from vllm.distributed.device_communicators.comm_autotune import agree, time_candidates
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    dev = torch.device("cuda", rank); g = dist.group.WORLD
    a = PyNcclCommunicator(g, device=dev)
    os.environ["NCCL_PROTO"] = "LL"
    b = PyNcclCommunicator(g, device=dev)
    del os.environ["NCCL_PROTO"]
    c = PyNcclCommunicator(g, device=dev)  # default again, created after

    def a2a(comm):
        def run(x):
            ch = x.view(world, -1); rv = torch.empty_like(ch)
            comm.group_start()
            for p in range(world):
                if p != rank:
                    comm.send(ch[p], p); comm.recv(rv[p], p)
            comm.group_end()
        return run
    cands = {"ar_def": (a.all_reduce, lambda x: True), "ar_ll": (b.all_reduce, lambda x: True),
             "ar_def2": (c.all_reduce, lambda x: True),
             "a2a_def": (a2a(a), lambda x: True), "a2a_ll": (a2a(b), lambda x: True)}
    sizes = [1 << k for k in (19, 20, 21, 22)]
    t = time_candidates(cands, sizes, lambda n: torch.randn(n // 2, dtype=torch.bfloat16, device=dev), g)
    agree(t, g)
    if rank == 0:
        for j, s in enumerate(sizes):
            print(f"{s>>10:6d} KiB " + " ".join(f"{k}={t[k][j]:.3f}" for k in t))
    dist.destroy_process_group()


if __name__ == "__main__":
    from vllm.utils.network_utils import get_open_port
    mp.spawn(worker, args=(4, get_open_port()), nprocs=4)
