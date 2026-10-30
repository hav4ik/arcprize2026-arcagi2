"""B32-M v3 replica monitor (read-only), one jsonl record every <interval> s:
- /proc/meminfo fields;
- RSS (VmRSS, RssAnon, RssShmem) summed by process group: vllm_workers (VLLM::Worker*), engine_core (VLLM::EngineCore),
  api_server (vllm serve / vllm.entrypoints), solver (solver_dsv41.py), sandboxes (the harness's python -c executors),
  kernel (ipykernel_launcher), other;
- nvidia-smi per GPU (used/free MiB, power W, SM MHz, temp C) and the GPU compute apps;
- /dev/shm used; Kaggle-equivalent host RAM = AnonPages + Shmem + 7 GiB (Kaggle's cgroup limit 173.0 GiB, no swap).
  This pod's /proc/meminfo is host-wide (~12-13 GiB of AnonPages belong to no process of this container, also at idle),
  so kaggle_eq_gib uses the container's cgroup v1 memory.stat rss + shmem when present (kaggle_eq_meminfo_gib keeps the
  meminfo formula).
Every 5 s instead while the solver's newest server event is "server died" (until its "server up after"), and from
the moment the watcher creates /root/e066/b32mv3/fast.flag (cap - peak < 0.35 GiB on a rank, or an allocator retry). Per vLLM
process: pid, name, RssAnon, RssShmem; sandboxes: count, total and largest RSS; shm_engines = EngineCore processes.
Usage: mon_sampler.py <out.jsonl> [interval_s=60]"""
import json, os, subprocess, sys, time

out = sys.argv[1]
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 60
MI = ("MemTotal", "MemAvailable", "AnonPages", "Shmem", "Cached", "Mlocked", "PageTables", "AnonHugePages")
G = 1048576


def group(cmd):
    if cmd.startswith("VLLM::Worker"):
        return "vllm_workers"
    if cmd.startswith("VLLM::EngineCore"):
        return "engine_core"
    if "vllm.entrypoints" in cmd or ("/vllm " in cmd and " serve " in cmd) or cmd.endswith("/vllm serve"):
        return "api_server"
    if "solver_dsv41.py" in cmd:
        return "solver"
    if "import ast, contextlib" in cmd:
        return "sandboxes"
    if "ipykernel_launcher" in cmd:
        return "kernel"
    return "other"


def fast_mode():
    if os.path.exists("/root/e066/b32mv3/fast.flag"):  # the watcher sets it: cap - peak < 0.35 GiB or an allocator retry
        return True
    try:
        with open("/kaggle/working/dsv41_logs/solver.log", "rb") as f:
            f.seek(0, 2); f.seek(max(0, f.tell() - 200000))
            txt = f.read().decode(errors="replace")
        return txt.rfind("server died") > txt.rfind("server up after")
    except OSError:
        return False


def sample():
    t = time.time()
    mi = {}
    for l in open("/proc/meminfo"):
        k = l.split(":")[0]
        if k in MI:
            mi[k] = int(l.split()[1])
    groups, vllm, sbx = {}, [], []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
            st = {l.split(":")[0]: int(l.split()[1]) for l in open(f"/proc/{p}/status")
                  if l.startswith(("VmRSS", "RssAnon", "RssShmem"))}
        except (OSError, ValueError, IndexError):
            continue
        if not cmd or not st:
            continue
        gname = group(cmd)
        if gname in ("vllm_workers", "engine_core", "api_server"):
            vllm.append([int(p), cmd[:24], st.get("RssAnon", 0), st.get("RssShmem", 0)])
        elif gname == "sandboxes":
            sbx.append(st.get("VmRSS", 0))
        g = groups.setdefault(gname, {"n": 0, "rss_kb": 0, "anon_kb": 0, "shmem_kb": 0})
        g["n"] += 1
        g["rss_kb"] += st.get("VmRSS", 0)
        g["anon_kb"] += st.get("RssAnon", 0)
        g["shmem_kb"] += st.get("RssShmem", 0)
    gpu, apps = [], []
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,power.draw,clocks.sm,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        gpu = [[float(x) for x in l.split(",")] for l in q.strip().splitlines() if l.strip()]
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        apps = [l.strip() for l in a.strip().splitlines() if l.strip()]
    except Exception as exc:
        gpu = [f"err {exc}"]
    sv = os.statvfs("/dev/shm")
    shm_used_kb = (sv.f_blocks - sv.f_bfree) * sv.f_frsize // 1024
    keq_mi = (mi.get("AnonPages", 0) + mi.get("Shmem", 0)) / G + 7
    cg = {}
    try:
        for l in open("/sys/fs/cgroup/memory/memory.stat"):
            k, v = l.split()
            if k in ("rss", "shmem", "cache", "mapped_file", "total_rss", "total_shmem"):
                cg[k] = int(v)
    except OSError:
        pass
    keq = (cg["total_rss"] + cg["total_shmem"]) / 2**30 + 7 if "total_rss" in cg else keq_mi
    return {"t": round(t, 1), "utc": time.strftime("%H:%M:%S", time.gmtime(t)), "meminfo_kb": mi, "groups": groups,
            "gpu": gpu, "gpu_apps": apps, "vllm_procs": vllm,
            "sandboxes": {"n": len(sbx), "total_kb": sum(sbx), "max_kb": max(sbx or [0])},
            "engines": sum(1 for v in vllm if v[1].startswith("VLLM::EngineCore")), "fast": fast_mode(), "dev_shm_used_kb": shm_used_kb, "kaggle_eq_gib": round(keq, 2),
            "kaggle_eq_meminfo_gib": round(keq_mi, 2), "cgroup_v1_bytes": cg}


while True:
    t0 = time.time()
    try:
        rec = sample()
    except Exception as exc:
        rec = {"t": round(t0, 1), "error": f"{type(exc).__name__}: {exc}"}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(max(1.0, (5 if rec.get("fast") else dt) - (time.time() - t0)))
