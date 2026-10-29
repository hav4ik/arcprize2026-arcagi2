"""Host RAM / process / GPU sampler for the restart test, every <interval> s (read-only): /proc/meminfo, the RSS of all
VLLM processes and of the vllm serve leader(s), their pids, the GPU process list, and the supervisor status file(s).
Usage: restart_sampler.py <out.jsonl> [interval_s=5] [status glob=/kaggle/working/dsv41_logs/server/*status*]"""
import glob, json, os, subprocess, sys, time

out = sys.argv[1]
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 5
sglob = sys.argv[3] if len(sys.argv) > 3 else "/kaggle/working/dsv41_logs/server/*status*"
KEYS = ("MemAvailable", "AnonPages", "AnonHugePages", "Shmem", "Cached", "Mlocked", "PageTables")


def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k = l.split(":")[0]
        if k in KEYS:
            d[k] = int(l.split()[1])
    return d


def procs():
    v, total = [], 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
            rss = next((int(l.split()[1]) for l in open(f"/proc/{p}/status") if l.startswith("VmRSS")), 0)
        except OSError:
            continue
        total += rss
        if cmd.startswith("VLLM::") or "vllm.entrypoints.cli.main serve" in cmd:
            v.append((int(p), cmd[:40], rss))
    return v, total


while True:
    t = time.time()
    v, total = procs()
    try:
        g = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.split("\n")
        gpu = [x.strip() for x in g if x.strip()]
    except Exception as exc:
        gpu = [f"err {exc}"]
    st = {}
    for f in glob.glob(sglob):
        try:
            st[os.path.basename(f)] = open(f).read()[:400]
        except OSError:
            pass
    rec = {"t": round(t, 1), "utc": time.strftime("%H:%M:%S", time.gmtime(t)), "mem_kb": meminfo(),
           "all_rss_kb": total, "vllm_rss_kb": sum(r for _, _, r in v), "vllm": [[p, c, r] for p, c, r in v],
           "gpu": gpu, "status": st}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(max(0.0, dt - (time.time() - t)))
