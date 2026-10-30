#!/usr/bin/env python3
"""Emulate Kaggle's GPU headroom on a pod: hold a fixed amount of memory on every GPU for the whole run.

Why: vLLM sizes its allocator cap from the free memory it finds at startup, and Kaggle's L4s leave less of it
than our pods (on the B32-M audit ~0.38 GiB more non-torch memory on Kaggle, e.g. CUDA graphs 0.65 vs 0.34 GiB:
vLLM's "cap - reserved" was 0.67-0.73 GiB on Kaggle vs 1.16-1.22 on the pod). A pod replica without the hold has
~0.5 GiB more GPU headroom than Kaggle and can pass where Kaggle OOMs.

hold MIB   One CUDA context per GPU (libcudart via ctypes, no torch) plus one cudaMalloc each, sized so this
           process's nvidia-smi compute-app memory reaches MIB per GPU (fallback when nvidia-smi does not list our
           pid: the board-level memory.used delta of the context). The memory is touched (cudaMemset), logged, and
           held until the process is killed (it frees on exit). Start it BEFORE the server cell starts vLLM (e.g.
           run_notebook.py --on-cell-start <server cell>:"gpu_hold.py hold 350 ...") and leave it running all run.
           A solver that waits for nvidia-smi memory.used < 1500 MiB before restarting its server is not blocked.
calibrate  The next hold size from a pod server log and Kaggle's numbers: the hold moves vLLM's cap one-for-one
           (B32-M: 410 -> 350 MiB moved cap - reserved from 0.61-0.66 to 0.66-0.72 GiB vs Kaggle's 0.67-0.73).
           new = hold + mean over ranks (pod cap - reserved - Kaggle cap - reserved) x 1024 MiB.

CUDA ordinals are made equal to nvidia-smi indices (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES unset).

Usage: gpu_hold.py hold 350 --log gpu_hold.log [--pid-file gpu_hold.pid] [--gpus 0,1,2,3]
       gpu_hold.py calibrate --hold-mib 410 --pod-log server.log (--kaggle-log kaggle_server.log | --kaggle 0.67-0.73)"""
import argparse
import ctypes
import ctypes.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reli_common as rc  # noqa: E402


def load_cudart(path=None):
    for c in (path, os.environ.get("RELI_CUDART"), "/usr/local/cuda/lib64/libcudart.so",
              ctypes.util.find_library("cudart"), "libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        if c:
            try:
                return ctypes.CDLL(c)
            except OSError:
                pass
    raise SystemExit("libcudart not found: pass --cudart PATH or set RELI_CUDART")


def smi_used_by_pid(pid, nvidia_smi="nvidia-smi"):
    """{nvidia-smi index: MiB} of one pid's compute apps; {} when nvidia-smi does not list it."""
    gpus, apps, _ = rc.query_gpus(nvidia_smi)
    out = {}
    for p, idx, mib in apps:
        if p == pid and idx is not None and mib is not None:
            out[idx] = out.get(idx, 0) + int(mib)
    return out


def smi_board_used(nvidia_smi="nvidia-smi"):
    gpus, _, _ = rc.query_gpus(nvidia_smi)
    return {g["index"]: g["used_mib"] for g in gpus if g.get("used_mib") is not None}


def hold(cudart, target_mib, gpus=None, used_by_me=None, board_used=None, log=print, settle_s=1.0):
    """Create a context on each GPU and top it up to target_mib. Returns ({gpu: allocated MiB}, pointers, ctx)."""
    n = ctypes.c_int()
    if cudart.cudaGetDeviceCount(ctypes.byref(n)) != 0:
        raise SystemExit("cudaGetDeviceCount failed")
    gpus = list(range(n.value)) if gpus is None else gpus
    bad = [g for g in gpus if g >= n.value]
    if bad:
        raise SystemExit("GPUs %s not present (%d devices)" % (bad, n.value))
    before = board_used() if board_used else {}
    for d in gpus:
        if cudart.cudaSetDevice(d) != 0 or cudart.cudaFree(ctypes.c_void_p(0)) != 0:
            raise SystemExit("could not create a context on GPU %d" % d)
    time.sleep(settle_s)
    ctx = used_by_me() if used_by_me else {}
    how = "nvidia-smi compute apps"
    if not all(d in ctx for d in gpus) and board_used:
        after = board_used()
        ctx = {d: int(after.get(d, 0) - before.get(d, 0)) for d in gpus}
        how = "board memory.used delta"
    allocated, ptrs = {}, []
    for d in gpus:
        extra = max(0, target_mib - int(ctx.get(d, 0)))
        cudart.cudaSetDevice(d)
        if extra:
            p = ctypes.c_void_p()
            rcode = cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(extra << 20))
            if rcode != 0:
                raise SystemExit("cudaMalloc(%d MiB) on GPU %d failed: %d" % (extra, d, rcode))
            cudart.cudaMemset(p, 0, ctypes.c_size_t(extra << 20))
            ptrs.append(p)
        allocated[d] = extra
    log("HOLD pid %d: target %d MiB/GPU on GPUs %s; context %s MiB (%s); allocated %s MiB"
        % (os.getpid(), target_mib, gpus, ctx, how, allocated))
    return allocated, ptrs, ctx


def parse_caps(path):
    """{rank: cap - reserved} from the newest server instance's "Capped the PyTorch allocator" lines."""
    lg = rc.VllmLog(path)
    with open(path, errors="replace") as f:
        for line in f:
            lg.feed(rc.ANSI.sub("", line), time.time())
    return {r: round(c["cap"] - c["reserved"], 3) for r, c in lg.caps.items()}


def parse_kaggle(spec):
    """"0.67-0.73" -> {"*": 0.70}; "TP0=0.73,TP1=0.71" -> per rank; "0.70" -> {"*": 0.70}."""
    spec = spec.strip()
    if "=" in spec:
        return {k.strip(): float(v) for k, v in (x.split("=") for x in spec.split(","))}
    if "-" in spec[1:]:
        lo, hi = spec.split("-", 1)
        return {"*": (float(lo) + float(hi)) / 2}
    return {"*": float(spec)}


def suggest(hold_mib, pod, kaggle):
    """(new hold MiB rounded to 10, mean difference GiB, per-rank rows). Positive difference = the pod has more
    headroom than Kaggle = hold more."""
    rows, diffs = [], []
    for r in sorted(pod):
        k = kaggle.get(r, kaggle.get("*"))
        if k is None:
            continue
        diffs.append(pod[r] - k)
        rows.append((r, pod[r], k, pod[r] - k))
    if not diffs:
        raise SystemExit("no rank in common between the pod log %s and Kaggle's %s" % (sorted(pod), sorted(kaggle)))
    mean = sum(diffs) / len(diffs)
    return int(round((hold_mib + mean * 1024) / 10.0) * 10), mean, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hold")
    h.add_argument("mib", type=int, help="target MiB per GPU, context included")
    h.add_argument("--gpus", help="comma-separated nvidia-smi indices (default: all)")
    h.add_argument("--log", default="gpu_hold.log")
    h.add_argument("--pid-file", default="gpu_hold.pid")
    h.add_argument("--cudart")
    h.add_argument("--nvidia-smi", default="nvidia-smi")
    c = sub.add_parser("calibrate")
    c.add_argument("--hold-mib", type=int, required=True, help="the hold used for --pod-log's run (0 = none)")
    c.add_argument("--pod-log", required=True, help="the pod server log with the startup cap lines")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--kaggle-log", help="a Kaggle run's server log (e.g. the commit smoke's)")
    g.add_argument("--kaggle", help='Kaggle cap - reserved: "0.67-0.73", "0.70" or "TP0=0.73,TP1=0.71,..."')
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        pod = parse_caps(a.pod_log)
        if not pod:
            raise SystemExit("no 'Capped the PyTorch allocator' line in %s" % a.pod_log)
        kag = parse_caps(a.kaggle_log) if a.kaggle_log else parse_kaggle(a.kaggle)
        new, mean, rows = suggest(a.hold_mib, pod, kag)
        print("| rank | pod cap - reserved | Kaggle | pod - Kaggle |\n|---|---|---|---|")
        for r, p, k, d in rows:
            print("| %s | %.2f | %.2f | %+.2f |" % (r, p, k, d))
        print("mean pod - Kaggle %+.3f GiB -> hold %d MiB/GPU (was %d)%s" % (
            mean, new, a.hold_mib, "" if abs(mean) > 0.015 else "; within 0.015 GiB: keep the current hold"))
        return 0
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cudart = load_cudart(a.cudart)
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None

    def log(msg):
        with open(a.log, "a") as f:
            f.write("%s %s\n" % (rc.utc(time.time()), msg))
        print(msg, flush=True)
    with open(a.pid_file, "w") as f:
        f.write("%d\n" % os.getpid())
    hold(cudart, a.mib, gpus, lambda: smi_used_by_pid(os.getpid(), a.nvidia_smi),
         lambda: smi_board_used(a.nvidia_smi), log)
    time.sleep(1)
    log("HOLD now %s MiB per GPU (nvidia-smi compute apps of pid %d)"
        % (smi_used_by_pid(os.getpid(), a.nvidia_smi), os.getpid()))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
