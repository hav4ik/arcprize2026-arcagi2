#!/usr/bin/env python3
"""Read-only monitor for a Kaggle-candidate replica run: one JSON line per sample to --out.

Per sample (every --interval s, 60 by default; every --fast-interval s, 5 by default, while the server is down,
the supervisor is not "up", or within 15 min of a fault_inject.py action):
- host RAM: the container's cgroup (v1 memory.stat total_rss + total_shmem, or v2 anon + shmem) and the
  Kaggle-equivalent  kaggle_eq_gib = cgroup anon + shmem + 7 GiB  against Kaggle's 173 GiB memory.max (no swap).
  /proc/meminfo is kept for reference only: on pods it is HOST-wide (other tenants' anon memory included).
- processes: RSS / RssAnon / RssShmem by group (TP workers, EngineCore, API server, solver, tool sandboxes, ...);
  every vLLM process (pid, title, anon, shmem); the EngineCore count (two engines' RAM must never coexist).
- GPU: nvidia-smi per GPU and the compute apps (board free memory is NOT a pressure signal: vLLM caps its
  allocator just under it).
- vLLM: per rank the startup "Capped the PyTorch allocator at X GiB (Y GiB reserved)" and the newest
  "GPU memory: A allocated (peak P), R reserved (S split), F free, N allocator retries, K OOMs" line, so
  cap - peak, cap - reserved, retries and OOMs; the KV pool; OOM / engine-death / Traceback line counts.
  The per-rank lines need VLLM_GPU_MEMORY_LOG_INTERVAL (seconds) in the server's environment (logging only).
- solver: server deaths / ups, re-queues, retries (a try spent), finalized stop=error attempts, the Kaggle clock
  zero (t0) and the last "server:" metrics line; attempts.jsonl record / error-record counts; supervisor status.

Logs are read incrementally (only new bytes), so a 12 h run costs little. Never signals or writes anything but --out.

Usage: sampler.py --out mon.jsonl [--working /kaggle/working] [--interval 60] [--timeline nb.timeline] [--once]"""
import argparse
import os
import socket
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reli_common as rc  # noqa: E402

FAULT_FAST_S = 900


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="JSONL output (appended)")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--fast-interval", type=float, default=5.0)
    ap.add_argument("--working", default="/kaggle/working", help="the notebook's working dir")
    ap.add_argument("--logs-dir", help="solver log dir (default: <working>/dsv41_logs)")
    ap.add_argument("--server-log-glob", action="append",
                    help="vLLM server log glob(s), recursive ** allowed (default: <logs-dir>/**/server*.log and "
                         "<working>/**/server*.log)")
    ap.add_argument("--solver-log", help="default: <logs-dir>/solver.log")
    ap.add_argument("--attempts", help="default: <logs-dir>/attempts.jsonl")
    ap.add_argument("--status-glob", action="append",
                    help="supervisor status JSON glob(s) (default: <logs-dir>/**/*supervisor*.json)")
    ap.add_argument("--timeline", action="append", default=[],
                    help="runner timeline / log whose end line marks the notebook's end (repeatable)")
    ap.add_argument("--fault-log", default=os.environ.get("RELI_FAULT_LOG", ""),
                    help="fault_inject.py log: sample fast for 15 min after each action")
    ap.add_argument("--solver-pattern", default="solver_dsv41.py", help="cmdline substring of the solver")
    ap.add_argument("--supervisor-pattern", default="dsv41_vllm.py", help="cmdline substring of a supervisor")
    ap.add_argument("--overhead-gib", type=float, default=rc.KAGGLE_OVERHEAD_GIB,
                    help="Kaggle's container overhead added to the cgroup figure (default 7)")
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--cgroup-root", default="/sys/fs/cgroup")
    ap.add_argument("--shm", default="/dev/shm")
    ap.add_argument("--nvidia-smi", default="nvidia-smi")
    ap.add_argument("--once", action="store_true", help="write one sample and exit")
    ap.add_argument("--stop-on-notebook-end", action="store_true",
                    help="write a last sample and exit once a --timeline shows the notebook's end")
    a = ap.parse_args(argv)
    a.logs_dir = a.logs_dir or os.path.join(a.working, "dsv41_logs")
    a.server_log_glob = a.server_log_glob or [os.path.join(a.logs_dir, "**", "server*.log"),
                                              os.path.join(a.working, "**", "server*.log")]
    a.solver_log = a.solver_log or os.path.join(a.logs_dir, "solver.log")
    a.attempts = a.attempts or os.path.join(a.logs_dir, "attempts.jsonl")
    a.status_glob = a.status_glob or [os.path.join(a.logs_dir, "**", "*supervisor*.json")]
    return a


class Sampler:
    def __init__(self, a):
        self.a = a
        self.server = rc.ServerLogs(a.server_log_glob)
        self.solver = rc.SolverLog(a.solver_log)
        self.attempts = rc.AttemptsFile(a.attempts)

    def sample(self, now=None):
        a = self.a
        t = now or time.time()
        rec = {"kind": "sample", "schema": rc.SCHEMA, "t": round(t, 1), "utc": rc.hms(t)}
        mi = rc.read_meminfo(a.proc_root)
        cg = rc.read_cgroup_memory(a.cgroup_root, a.proc_root)
        keq, src = rc.kaggle_equivalent(cg, mi, a.overhead_gib)
        keq_mi, _ = rc.kaggle_equivalent({}, mi, a.overhead_gib)
        rec["kaggle_eq_gib"] = round(keq, 2) if keq is not None else None
        rec["kaggle_eq_source"] = src
        rec["kaggle_eq_meminfo_gib"] = round(keq_mi, 2) if keq_mi is not None else None
        rec["cgroup"] = ({"version": cg["version"], "anon_gib": round(cg["anon"] / rc.GIB, 2),
                          "shmem_gib": round(cg["shmem"] / rc.GIB, 2),
                          "limit_gib": round(cg["limit"] / rc.GIB, 1) if cg.get("limit") else None}
                         if cg.get("version") else {"version": None})
        rec["meminfo_gib"] = {k: round(v / rc.GIB, 2) for k, v in mi.items()}
        shm = rc.dev_shm_used(a.shm)
        rec["dev_shm_used_gib"] = round(shm / rc.GIB, 2) if shm is not None else None
        procs = rc.scan_procs(a.proc_root)
        groups = rc.classify(procs, a.solver_pattern, a.supervisor_pattern)
        rec["groups"] = rc.group_totals(procs, groups)
        rec["vllm_procs"] = sorted([pid, procs[pid]["cmd"][:40], round(procs[pid]["anon"] / rc.GIB, 2),
                                    round(procs[pid]["shmem"] / rc.GIB, 2)]
                                   for pid, g in groups.items() if g in ("tp_workers", "engine_core", "api_server"))
        rec["engines"] = sum(1 for g in groups.values() if g == "engine_core")
        sb = [procs[p]["rss"] for p, g in groups.items() if g == "sandboxes"]
        rec["sandboxes"] = {"n": len(sb), "total_gib": round(sum(sb) / rc.GIB, 2),
                            "max_gib": round(max(sb or [0]) / rc.GIB, 2)}
        gpus, apps, err = rc.query_gpus(a.nvidia_smi)
        rec["gpus"] = [{k: g[k] for k in ("index", "used_mib", "free_mib", "power_w", "sm_mhz", "temp_c")}
                       for g in gpus]
        rec["gpu_apps"] = apps
        if err:
            rec["gpu_error"] = err
        rec["vllm"] = self.server.update(t)
        rec["solver"] = self.solver.update(t)
        rec["attempts"] = self.attempts.update()
        rec["supervisor"] = rc.read_status(a.status_glob)
        rec["notebook_end"] = rc.notebook_done(a.timeline)
        rec["fast"] = self.fast(rec, t)
        return rec

    def fast(self, rec, t):
        if rec["solver"].get("down"):
            return True
        st = (rec.get("supervisor") or {}).get("state")
        if st not in (None, "up"):
            return True
        fl = self.a.fault_log
        try:
            if fl and os.path.exists(fl) and t - os.path.getmtime(fl) < FAULT_FAST_S:
                return True
        except OSError:
            pass
        return False


def main(argv=None):
    a = build_args(argv)
    s = Sampler(a)
    with open(a.out, "a") as f:
        f.write(json.dumps({"kind": "start", "t": round(time.time(), 1), "host": socket.gethostname(),
                            "cpus": os.cpu_count(), "pid": os.getpid(),
                            "config": {k: v for k, v in vars(a).items()}}) + "\n")
    while True:
        t0 = time.time()
        try:
            rec = s.sample(t0)
        except Exception as exc:  # keep sampling through anything unexpected
            rec = {"kind": "sample", "schema": rc.SCHEMA, "t": round(t0, 1), "utc": rc.hms(t0),
                   "sampler_error": "%s: %s" % (type(exc).__name__, exc)}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if a.once or (a.stop_on_notebook_end and rec.get("notebook_end")):
            return 0
        dt = a.fast_interval if rec.get("fast") else a.interval
        time.sleep(max(0.5, dt - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
