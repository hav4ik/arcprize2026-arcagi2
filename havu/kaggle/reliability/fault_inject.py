#!/usr/bin/env python3
"""Fault injection for a replica run, with exact targeting. Never kills by name pattern.

Actions (each appends JSON lines to --log, default $RELI_FAULT_LOG or ./fault_inject.jsonl; give the same file to
sampler.py --fault-log so it samples every 5 s around the fault, and to report.py / `verify`):
  find                     list vLLM API servers with their process trees and the GPU compute apps
  kill-tp RANK             SIGKILL exactly one TP worker, the process titled "VLLM::Worker_TP<RANK>" (DCP / PP
                           suffixes such as "_DCP0" allowed) inside the server's process tree: a worker crash
  sigstop-engine           SIGSTOP the EngineCore: a hang (the API server stays up, no token is generated). The
                           supervisor's stall detection must find and restart it. --resume-after S sends SIGCONT
                           after S s if the process still exists; else resume it with sigcont-engine
  sigcont-engine           SIGCONT a stopped EngineCore
  gpu-oom-burst GPU        allocate all free memory (or --mib) on one GPU for --hold-s s from a separate process
                           (libcudart via ctypes, no torch), so vLLM's next allocation beyond its reserved pool fails
  cpu-stress               --procs N busy-loop processes for --seconds S (contention like a smaller Kaggle host)
  verify                   recovery checks for every action in --log (see recovery.py for the pass criteria)

Safety checks before any signal: exactly one process matches the exact title (the cmdline is that single token),
it is a descendant of the one vLLM API server (or of --server-pid), it is not this process or pid 1, it belongs to
this user (or we are root), and its cmdline is read again through a pidfd right before the signal (no pid reuse).
--dry-run prints the target and the plan and sends nothing. Refuses to run inside a Kaggle kernel.

Usage: fault_inject.py kill-tp 3 --log fault.jsonl [--dry-run]
       fault_inject.py verify --log fault.jsonl --samples mon.jsonl --logs-dir /kaggle/working/dsv41_logs"""
import argparse
import ctypes
import ctypes.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reli_common as rc  # noqa: E402

WORKER_TITLE = re.compile(r"VLLM::Worker((?:_[A-Z]+\d+)+)")
ENGINE_TITLE = re.compile(r"VLLM::EngineCore(?:_[A-Za-z]*\d+)?")


class Refused(Exception):
    pass


def worker_rank(title):
    """The TP rank of an exact worker title ("VLLM::Worker_TP3", "VLLM::Worker_TP0_DCP0"), else None."""
    m = WORKER_TITLE.fullmatch(title or "")
    if not m:
        return None
    parts = dict(re.findall(r"_([A-Z]+)(\d+)", m.group(1)))
    return int(parts["TP"]) if "TP" in parts else None


def api_servers(procs):
    return sorted(p for p, v in procs.items() if rc.is_api_server(v["argv"]))


def server_root(procs, server_pid=None):
    if server_pid:
        if server_pid <= 1:
            raise Refused("--server-pid must be the server's own pid, not %d" % server_pid)
        if server_pid not in procs:
            raise Refused("--server-pid %d does not exist" % server_pid)
        return server_pid
    s = api_servers(procs)
    if len(s) != 1:
        raise Refused("found %d vLLM API servers %s; pass --server-pid" % (len(s), s))
    return s[0]


def find_targets(procs, root, match):
    tree = set(rc.descendants(procs, root))
    return sorted(p for p in tree if match(rc.exact_title(procs[p]["argv"])))


def pick_one(procs, root, match, what):
    hits = find_targets(procs, root, match)
    if len(hits) != 1:
        raise Refused("expected exactly one %s under pid %d, found %d: %s"
                      % (what, root, len(hits), [(p, procs[p]["cmd"][:40]) for p in hits]))
    pid = hits[0]
    if pid in (1, os.getpid()):
        raise Refused("refusing to signal pid %d" % pid)
    uid = procs[pid].get("uid")
    if os.geteuid() != 0 and uid is not None and uid != os.geteuid():
        raise Refused("pid %d belongs to uid %s, not us" % (pid, uid))
    return pid


def chain(procs, pid):
    out = []
    while pid in procs and pid > 1 and len(out) < 12:
        out.append("%d:%s" % (pid, procs[pid]["cmd"][:30]))
        pid = procs[pid]["ppid"]
    return out


def send_exact(pid, title, sig, proc_root="/proc"):
    """Signal pid only if its cmdline is still exactly `title`, through a pidfd when available (Linux >= 5.3,
    Python >= 3.9), so a reused pid can never be hit."""
    fd = None
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        try:
            fd = os.pidfd_open(pid)
        except OSError:
            fd = None
    try:
        argv = rc.read_argv(pid, proc_root)
        if rc.exact_title(argv) != title:
            raise Refused("pid %d cmdline changed to %r before the signal" % (pid, argv[:3]))
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(pid, sig)
    finally:
        if fd is not None:
            os.close(fd)


class Log:
    def __init__(self, path):
        self.path = path

    def write(self, **e):
        e.setdefault("t", round(time.time(), 3))
        e["utc"] = rc.utc(e["t"])
        e["host"] = socket.gethostname()
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(e) + "\n")
        print(json.dumps(e), flush=True)
        return e


def signal_action(a, log, what, match, sig, action):
    if os.path.realpath(a.proc_root) != "/proc" and not a.dry_run:
        raise Refused("--proc-root %s is for dry runs only (its pids are not this host's)" % a.proc_root)
    procs = rc.scan_procs(a.proc_root)
    root = server_root(procs, a.server_pid)
    pid = pick_one(procs, root, match, what)
    title = rc.exact_title(procs[pid]["argv"])
    info = {"action": action, "pid": pid, "target": title, "server_pid": root, "signal": signal.Signals(sig).name,
            "chain": chain(procs, pid), "rss_gib": round(procs[pid]["rss"] / rc.GIB, 2),
            "vllm_pids": sorted(p for p in rc.descendants(procs, root) + [root]
                                if procs[p]["cmd"].startswith("VLLM::") or p == root)}
    if a.dry_run:
        log.write(phase="dry_run", dry_run=True, **info)
        return pid, title
    log.write(phase="about_to_signal", **info)
    t = time.time()
    send_exact(pid, title, sig, a.proc_root)
    log.write(phase="signaled", t=round(t, 3), **info)
    return pid, title


def cmd_find(a, log):
    procs = rc.scan_procs(a.proc_root)
    roots = [a.server_pid] if a.server_pid else api_servers(procs)
    if not roots:
        print("no vLLM API server found")
    for root in roots:
        print("server pid %d: %s" % (root, procs.get(root, {}).get("cmd", "?")[:120]))
        for p in sorted(rc.descendants(procs, root)):
            v = procs[p]
            title = rc.exact_title(v["argv"])
            tag = ("  <- kill-tp %d" % worker_rank(title)) if worker_rank(title) is not None else \
                ("  <- sigstop-engine" if ENGINE_TITLE.fullmatch(title or "") else "")
            print("  pid %-8d ppid %-8d state %s rss %6.1f GiB  %s%s"
                  % (p, v["ppid"], v["state"], v["rss"] / rc.GIB, v["cmd"][:60], tag))
    gpus, apps, err = rc.query_gpus(a.nvidia_smi)
    print("GPU compute apps (pid, gpu, MiB): %s%s" % (apps, (" (nvidia-smi: %s)" % err) if err else ""))
    return 0


def cmd_kill_tp(a, log):
    signal_action(a, log, "VLLM::Worker_TP%d process" % a.rank, lambda t: worker_rank(t) == a.rank,
                  signal.SIGKILL, "kill-tp")
    return 0


def cmd_sigstop(a, log):
    pid, title = signal_action(a, log, "VLLM::EngineCore process", lambda t: bool(ENGINE_TITLE.fullmatch(t or "")),
                               signal.SIGSTOP, "sigstop-engine")
    if a.dry_run:
        return 0
    if a.resume_after:
        time.sleep(a.resume_after)
        try:
            send_exact(pid, title, signal.SIGCONT, a.proc_root)
            log.write(phase="resumed", action="sigcont-engine", pid=pid, target=title)
        except (OSError, Refused) as exc:
            log.write(phase="resume_skipped", action="sigcont-engine", pid=pid, target=title,
                      reason="%s: %s (the supervisor killed it?)" % (type(exc).__name__, exc))
    else:
        print("EngineCore pid %d is stopped; resume with: %s sigcont-engine --server-pid %s"
              % (pid, sys.argv[0], a.server_pid or "<server pid>"))
    return 0


def cmd_sigcont(a, log):
    pid, title = signal_action(a, log, "VLLM::EngineCore process", lambda t: bool(ENGINE_TITLE.fullmatch(t or "")),
                               signal.SIGCONT, "sigcont-engine")
    return 0


# ---------------------------------------------------------------------------------------------- GPU OOM burst
def load_cudart(path=None):
    cands = [path, os.environ.get("RELI_CUDART"), "/usr/local/cuda/lib64/libcudart.so", ctypes.util.find_library("cudart"),
             "libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    errs = []
    for c in cands:
        if not c:
            continue
        try:
            return ctypes.CDLL(c)
        except OSError as exc:
            errs.append("%s: %s" % (c, exc))
    raise Refused("libcudart not found (--cudart PATH or RELI_CUDART): %s" % "; ".join(errs[-3:]))


def burst(cudart, mib=None, leave_mib=0, chunk_mib=256, hold_s=30.0, sleep=time.sleep):
    """Allocate device memory in chunks on the current device until `mib` MiB (or all but leave_mib) are taken or
    cudaMalloc fails; hold; free. Returns (allocated MiB, free MiB before)."""
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    assert cudart.cudaFree(ctypes.c_void_p(0)) == 0  # context
    assert cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
    free_mib = free.value >> 20
    want = mib if mib is not None else max(0, free_mib - leave_mib)
    ptrs, got = [], 0
    while got < want:
        n = min(chunk_mib, want - got)
        p = ctypes.c_void_p()
        if cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n << 20)) != 0:
            if n > 1 and chunk_mib > 1:
                chunk_mib = max(1, chunk_mib // 2)
                continue
            break
        cudart.cudaMemset(p, 0, ctypes.c_size_t(n << 20))
        ptrs.append(p)
        got += n
    sleep(hold_s)
    for p in ptrs:
        cudart.cudaFree(p)
    return got, free_mib


def cmd_burst_child(a, log):
    got, free = burst(load_cudart(a.cudart), a.mib, a.leave_mib, a.chunk_mib, a.hold_s)
    print(json.dumps({"allocated_mib": got, "free_mib_before": free}), flush=True)
    return 0


def cmd_gpu_oom(a, log):
    procs = rc.scan_procs(a.proc_root)
    gpus, apps, err = rc.query_gpus(a.nvidia_smi)
    g = next((x for x in gpus if x["index"] == a.gpu), None)
    if g is None:
        raise Refused("GPU %d not in nvidia-smi's list %s%s" % (a.gpu, [x["index"] for x in gpus],
                                                               (" (%s)" % err) if err else ""))
    root = None
    if not a.allow_any_gpu:
        root = server_root(procs, a.server_pid)
        workers = {p for p in rc.descendants(procs, root) if worker_rank(rc.exact_title(procs[p]["argv"])) is not None}
        on_gpu = [x[0] for x in apps if x[1] == a.gpu]
        if not (workers & set(on_gpu)):
            raise Refused("GPU %d hosts no vLLM worker of server %d (apps %s); --allow-any-gpu to override"
                          % (a.gpu, root, on_gpu))
    if os.path.realpath(a.proc_root) != "/proc" and not a.dry_run:
        raise Refused("--proc-root %s is for dry runs only" % a.proc_root)
    info = {"action": "gpu-oom-burst", "gpu": a.gpu, "target": "GPU %d" % a.gpu, "pid": None, "server_pid": root,
            "free_mib": g.get("free_mib"), "mib": a.mib, "leave_mib": a.leave_mib, "hold_s": a.hold_s}
    if a.dry_run:
        log.write(phase="dry_run", dry_run=True, **info)
        return 0
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(a.gpu))
    cmd = [sys.executable, os.path.abspath(__file__), "_burst", "--hold-s", str(a.hold_s), "--leave-mib",
           str(a.leave_mib), "--chunk-mib", str(a.chunk_mib)] + (["--mib", str(a.mib)] if a.mib else []) + \
        (["--cudart", a.cudart] if a.cudart else [])
    log.write(phase="about_to_signal", **info)
    t = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True, start_new_session=True)
    log.write(phase="signaled", t=round(t, 3), child_pid=p.pid, **info)
    out, _ = p.communicate()
    log.write(phase="done", action="gpu-oom-burst", gpu=a.gpu, rc=p.returncode, result=out.strip()[-300:])
    return 0 if p.returncode == 0 else 1


def cmd_cpu_stress(a, log):
    n = os.cpu_count() or 1
    if a.procs < 1 or a.procs > n:
        raise Refused("--procs must be 1..%d (this host's CPUs)" % n)
    if a.seconds <= 0 or (a.seconds > 4 * 3600 and not a.force):
        raise Refused("--seconds must be in (0, 14400] (--force for longer)")
    info = {"action": "cpu-stress", "target": "%d busy processes" % a.procs, "pid": None, "procs": a.procs,
            "seconds": a.seconds}
    if a.dry_run:
        log.write(phase="dry_run", dry_run=True, **info)
        return 0
    code = "import time\ne = time.time() + %f\nwhile time.time() < e:\n    pass\n" % a.seconds
    t = time.time()
    kids = [subprocess.Popen([sys.executable, "-c", code], start_new_session=True) for _ in range(a.procs)]
    log.write(phase="signaled", t=round(t, 3), child_pids=[k.pid for k in kids], **info)
    try:
        for k in kids:
            k.wait()
    except KeyboardInterrupt:
        for k in kids:
            k.kill()
    log.write(phase="done", action="cpu-stress", seconds=round(time.time() - t, 1))
    return 0


def cmd_verify(a, log):
    import recovery
    recs = rc.load_records(a.samples)
    hours, _ = rc.clock(recs)
    slog, att = recovery.default_paths(a.logs_dir)
    faults = recovery.load_faults(a.log)
    if not faults:
        print("no executed fault actions in %s" % a.log)
        return 1
    worst = 0
    for f in faults:
        res = recovery.analyze(f, recs, slog, att, a.window_s, a.max_restart_s, a.ram_margin_gib, a.ram_limit,
                               a.hang_limit_s)
        print(recovery.format_result(res, hours))
        worst = max(worst, {"PASS": 0, "WARN": 0, "FAIL": 1}[res["verdict"]])
    return worst


def build_args(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log", default=os.environ.get("RELI_FAULT_LOG", "fault_inject.jsonl"),
                        help="JSONL action log (default $RELI_FAULT_LOG or ./fault_inject.jsonl)")
    common.add_argument("--proc-root", default="/proc")
    common.add_argument("--nvidia-smi", default="nvidia-smi")
    common.add_argument("--server-pid", type=int,
                        help="root of the server's process tree (default: the one vLLM API server)")
    common.add_argument("--dry-run", action="store_true", help="print the target and plan; send nothing")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)
    add("find")
    k = add("kill-tp")
    k.add_argument("rank", type=int)
    s = add("sigstop-engine")
    s.add_argument("--resume-after", type=float, help="SIGCONT after this many seconds if still alive")
    add("sigcont-engine")
    g = add("gpu-oom-burst")
    g.add_argument("gpu", type=int, help="nvidia-smi index")
    for p in (g, add("_burst")):
        p.add_argument("--mib", type=int, help="allocate this much (default: all free minus --leave-mib)")
        p.add_argument("--leave-mib", type=int, default=0)
        p.add_argument("--chunk-mib", type=int, default=256)
        p.add_argument("--hold-s", type=float, default=30.0)
        p.add_argument("--cudart")
    g.add_argument("--allow-any-gpu", action="store_true", help="skip the 'hosts a vLLM worker' check")
    c = add("cpu-stress")
    c.add_argument("--procs", type=int, required=True)
    c.add_argument("--seconds", type=float, required=True)
    c.add_argument("--force", action="store_true")
    v = add("verify")
    v.add_argument("--samples", required=True)
    v.add_argument("--logs-dir", help="the solver's log dir (solver.log, attempts.jsonl)")
    v.add_argument("--window-s", type=float, default=3600)
    v.add_argument("--max-restart-s", type=float, default=900)
    v.add_argument("--hang-limit-s", type=float, default=2100,
                   help="sigstop: the supervisor's stall timeout (25 min) + margin")
    v.add_argument("--ram-margin-gib", type=float, default=10.0)
    v.add_argument("--ram-limit", type=float, default=rc.KAGGLE_RAM_LIMIT_GIB)
    return ap.parse_args(argv)


CMDS = {"find": cmd_find, "kill-tp": cmd_kill_tp, "sigstop-engine": cmd_sigstop, "sigcont-engine": cmd_sigcont,
        "gpu-oom-burst": cmd_gpu_oom, "_burst": cmd_burst_child, "cpu-stress": cmd_cpu_stress, "verify": cmd_verify}


def main(argv=None):
    a = build_args(argv)
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print("refused: this looks like a Kaggle kernel (KAGGLE_KERNEL_RUN_TYPE is set)", file=sys.stderr)
        return 2
    log = Log(None if a.cmd in ("find", "verify", "_burst") else a.log)
    try:
        return CMDS[a.cmd](a, log)
    except Refused as exc:
        print("refused: %s" % exc, file=sys.stderr)
        if a.cmd not in ("find", "verify", "_burst"):
            log.write(phase="refused", action=a.cmd, reason=str(exc), dry_run=a.dry_run)
        return 2


if __name__ == "__main__":
    sys.exit(main())
