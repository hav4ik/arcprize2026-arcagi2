"""--gpu-ids auto-bench: the GPU order from a GPU link bench that runs once per server.

``python gpu_pair_bench.py OUT.json`` times pinned host<->GPU copies on each GPU and NCCL all-gather and send/recv
for every GPU pair, alone and, on 4 GPUs, beside the other pair of each split in lock step: vLLM's DCP pairs
exchange at the same moment each layer, while unsynced pairs collide only by chance (which split looked contended
flipped between runs). choose() runs it in its own process group before the server starts, while the GPUs are
idle, and caches the result in ARC_GPU_BENCH_PATH (default /kaggle/working/gpu_pair_bench.json) under the GPU
UUIDs, NCCL version and NCCL_* environment, so restarts and later runs on the same server reuse it; delete the
file to bench again. Any bench failure falls back to the prod command's order.
"""

from contextlib import suppress
from functools import partial
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

AUTO = "auto-bench"
SIZES = (528384, 4227072)  # bytes per rank
SPLITS = ([(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)])
# vLLM puts consecutive TP ranks in one DCP group, so 0,2,1,3 pairs GPUs 0-2 and 1-3: the fast pairs on every
# healthy 4x L4 server measured so far (0-1 and 2-3 are slow).
FALLBACK = {4: "0,2,1,3"}
TIE = 1.03


def cache_path():
    return Path(os.environ.get("ARC_GPU_BENCH_PATH", "/kaggle/working/gpu_pair_bench.json"))


def copy_speed():
    import torch
    speeds = []
    for g in range(torch.cuda.device_count()):
        torch.cuda.set_device(g)
        host = torch.empty(64 << 20, dtype=torch.uint8).pin_memory()
        device = torch.empty_like(host, device=f"cuda:{g}")
        speed = []
        for dst, src in ((device, host), (host, device)):
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            speed.append(10 * 64 / 1024 / (time.perf_counter() - start))
        print(f"gpu{g}  H2D {speed[0]:5.1f} GB/s  D2H {speed[1]:5.1f} GB/s", flush=True)
        speeds.append(speed)
        del host, device
    torch.cuda.empty_cache()
    return speeds


def send_recv(dist, ops):
    for work in dist.batch_isend_irecv(ops):
        work.wait()


def pair_worker(rank, gpus, port, barrier, queue):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(gpus[rank])
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    peer = 1 - rank
    times = {}
    for size in SIZES:
        send = torch.ones(size, dtype=torch.uint8, device="cuda")
        gathered = torch.empty(2 * size, dtype=torch.uint8, device="cuda")
        received = torch.empty(size, dtype=torch.uint8, device="cuda")
        ops = [dist.P2POp(dist.isend, send, peer), dist.P2POp(dist.irecv, received, peer)]
        for name, fn in (("ag", partial(dist.all_gather_into_tensor, gathered, send)),
                         ("sr", partial(send_recv, dist, ops))):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(50):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                barrier.wait(timeout=120)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            times[f"{name}{size // 1024}K"] = sorted(samples)[len(samples) // 2]
    if rank == 0:
        queue.put((f"{gpus[0]}-{gpus[1]}", times))
    dist.destroy_process_group()


def run_pairs(groups, mode):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    barrier = ctx.Barrier(2 * len(groups))
    procs = []
    for gpus in groups:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        procs += [ctx.Process(target=pair_worker, args=(rank, list(gpus), port, barrier, queue)) for rank in range(2)]
    for proc in procs:
        proc.start()
    try:
        out = dict(queue.get(timeout=180) for _ in groups)
    finally:
        for proc in procs:
            proc.join(60)
            if proc.is_alive():
                proc.kill()
    for pair, times in sorted(out.items()):
        print(f"{mode:8s} {pair}  " + "  ".join(f"{k} {v:6.0f}" for k, v in times.items()), flush=True)
    return out


def bench(out):
    import torch
    n = torch.cuda.device_count()
    result = dict(gpus=n, nccl=".".join(map(str, torch.cuda.nccl.version())), alone={}, together={})
    print(f"gpu auto-bench: {n} GPUs, NCCL {result['nccl']}", flush=True)
    result["copy"] = copy_speed()
    print("NCCL median us (ag = all_gather, sr = send/recv, bytes per rank)", flush=True)
    for pair in itertools.combinations(range(n), 2):
        result["alone"].update(run_pairs([pair], "alone"))
    for split in SPLITS if n == 4 else ():
        result["together"][",".join(f"{a}-{b}" for a, b in split)] = run_pairs(split, "together")
    Path(out).write_text(json.dumps(result) + "\n")


def nvidia_smi(query):
    return subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader"], text=True, timeout=30).split()


def bench_env():
    return {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}


def nccl_version():
    """The NCCL the bench and vLLM load: an upgrade on the same server changes how the pairs contend."""
    return subprocess.check_output([sys.executable, "-c", "import torch; print(*torch.cuda.nccl.version(), sep='.')"],
                                   env=bench_env(), stdin=subprocess.DEVNULL, text=True, timeout=300).strip()


def measure():
    env = bench_env()
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / "bench.json"
        process = subprocess.Popen([sys.executable, "-W", "ignore::FutureWarning", "-u", str(Path(__file__).resolve()),
                                    str(out)], env=env, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=900)
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if code:
            raise RuntimeError(f"bench exited with {code}")
        return json.loads(out.read_text())


def load(path):
    key = dict(uuids=nvidia_smi("--query-gpu=uuid"), nccl=nccl_version(),
               nccl_env={k: v for k, v in sorted(os.environ.items()) if k.startswith("NCCL_")})
    with suppress(OSError, ValueError, KeyError, TypeError):
        saved = json.loads(path.read_text())
        if saved["key"] == key:
            return saved["result"], "cached"
    busy = nvidia_smi("--query-compute-apps=pid")
    if busy:
        raise RuntimeError(f"GPU processes are active ({' '.join(busy)}), so the bench would be skewed")
    result = measure()
    for _ in range(30):
        if not nvidia_smi("--query-compute-apps=pid"):
            break
        time.sleep(1)
    if path.parent.is_dir():
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(dict(key=key, result=result), indent=1) + "\n")
        os.replace(temp, path)
    return result, "benched"


def slowdown(rows):
    """Each row's time over the fastest row's, averaged over the four transfers."""
    best = {m: min(r[m] for r in rows.values()) for m in next(iter(rows.values()))}
    return {k: sum(r[m] / best[m] for m in best) / len(best) for k, r in rows.items()}


def pick(result, tp, dcp, fallback):
    """The order and a note on why; ties within TIE keep the lowest GPUs, or the fallback for DCP pairs."""
    if tp == 1:
        speed = [min(c) for c in result["copy"]]
        gpu = next(g for g, s in enumerate(speed) if s * TIE >= max(speed))
        return str(gpu), "slower copy direction GB/s " + " ".join(f"{s:.1f}" for s in speed)
    if tp == 2:
        cost = slowdown(result["alone"])
        pair = min((p for p in cost if cost[p] <= TIE * min(cost.values())),
                   key=lambda p: tuple(map(int, p.split("-"))))
        return pair.replace("-", ","), "pair cost " + ", ".join(f"{p} {c:.2f}" for p, c in cost.items())
    if tp == 4 and dcp == 2 and result["together"]:
        cost = slowdown({(s, p): r for s, pairs in result["together"].items() for p, r in pairs.items()})
        split = {s: max(c for (t, _), c in cost.items() if t == s) for s in result["together"]}
        best = min(split, key=split.get)
        default = "{}-{},{}-{}".format(*fallback.split(","))
        if split.get(default, float("inf")) <= TIE * split[best]:
            best = default
        return best.replace("-", ","), "DCP pair cost " + ", ".join(
            f"{s.replace('-', ',')} {c:.2f}" for s, c in split.items())
    return fallback, "one group spans the GPUs, so their order does not matter"


def health_warnings(result):
    speed = [min(c) for c in result["copy"]]
    out = [f"GPU {g} host copies H2D {h:.1f} / D2H {d:.1f} GB/s vs {max(speed):.1f}: host all-reduce and KV offload "
           "slow down" for g, (h, d) in enumerate(result["copy"]) if min(h, d) < 0.6 * max(speed)]
    if result["alone"]:
        out += [f"NCCL pair {p} is {c:.1f}x the fastest pair" for p, c in slowdown(result["alone"]).items() if c > 2.5]
    return out


def choose(tp, dcp):
    """The --gpu-ids for TP size tp and DCP size dcp; prints the choice and any slow link."""
    fallback = FALLBACK.get(tp, ",".join(map(str, range(tp))))
    path = cache_path()
    try:
        result, source = load(path)
    except Exception as exc:
        print(f"WARNING gpu auto-bench failed ({exc}); using --gpu-ids {fallback}", flush=True)
        return fallback
    if result["gpus"] < tp:
        raise ValueError(f"TP {tp} needs {tp} GPUs; the bench found {result['gpus']}")
    ids, note = pick(result, tp, dcp, fallback)
    print(f"gpu auto-bench: --gpu-ids {ids} for TP {tp} DCP {dcp}; {note} ({source} {path})", flush=True)
    for line in health_warnings(result):
        print(f"WARNING gpu auto-bench: {line}", flush=True)
    return ids


if __name__ == "__main__":
    bench(sys.argv[1])
