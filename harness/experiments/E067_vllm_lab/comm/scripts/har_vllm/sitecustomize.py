"""Lab-only: route vLLM's TP all-reduces through har_ext.cu (zero-copy host
all-reduce) without touching the installed vLLM. Active only with HAR_ENABLE=1
and this directory on PYTHONPATH; patches CudaCommunicator right after vLLM
imports it, in every worker process.

Env: HAR_MAX_MB (default 64; larger messages stay on NCCL), HAR_THREADS (CPU
reducer threads per rank, 2), HAR_UNIT_KB (16), HAR_WRITERS / HAR_READERS (12),
HAR_POLL_NS (0 = hot-spinning reducers; >0 = sleep between polls, "cold").
"""
import importlib.abc
import importlib.util
import os
import sys

TARGET = "vllm.distributed.device_communicators.cuda_communicator"
SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _patch(mod):
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SCRIPTS)
    import host_ar_proto as hp

    max_bytes = int(float(os.environ.get("HAR_MAX_MB", "64")) * (1 << 20))
    hp.MAX_BYTES = max_bytes
    cls = mod.CudaCommunicator
    orig_init, orig_ar = cls.__init__, cls.all_reduce

    def new_init(self, cpu_group, device=None, device_group=None, unique_name="", *a, **kw):
        orig_init(self, cpu_group, device, device_group, unique_name, *a, **kw)
        self.har = None
        world = dist.get_world_size(cpu_group)
        if "tp" not in unique_name or world < 2:
            return
        rank = dist.get_rank(cpu_group)
        ext = hp.load_ext()
        box = [f"/dev/shm/har_vllm_{os.getpid()}"]
        dist.broadcast_object_list(box, src=dist.get_global_rank(cpu_group, 0), group=cpu_group)
        if rank != 0:
            dist.barrier(cpu_group)
        with torch.cuda.device(self.device):
            k = hp.Kernel(ext, box[0], rank, world, rank == 0,
                          int(os.environ.get("HAR_UNIT_KB", "16")) << 10,
                          int(os.environ.get("HAR_THREADS", "2")),
                          int(os.environ.get("HAR_POLL_NS", "0")),
                          int(os.environ.get("HAR_WRITERS", "12")),
                          int(os.environ.get("HAR_READERS", "12")), 256)
        if rank == 0:
            dist.barrier(cpu_group)
        dist.barrier(cpu_group)
        if rank == 0:
            os.unlink(box[0])
        k.start()
        self.har = k
        self.har_calls = 0
        print(f"[har] rank {rank}/{world} host all-reduce on for bf16 <= {max_bytes >> 20} MiB "
              f"(threads {k.threads}, poll_ns {k.poll_ns}, unit {k.unit >> 10} KiB)", flush=True)

    def new_ar(self, x):
        h = getattr(self, "har", None)
        if h is not None and x.dtype == torch.bfloat16 and x.is_contiguous():
            n = x.numel() * 2
            if 0 < n <= max_bytes and n % 16 == 0 and x.data_ptr() % 16 == 0:
                out = torch.empty_like(x)
                h.all_reduce(x, out)
                return out
        return orig_ar(self, x)

    cls.__init__ = new_init
    cls.all_reduce = new_ar


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        sys.meta_path.insert(0, self)
        if spec is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch(module)

        loader.exec_module = exec_module
        return spec


if os.environ.get("HAR_ENABLE") == "1":
    sys.meta_path.insert(0, _Finder())
