"""Read-only host-memory probe of Manh's server (pod 3, notebook run): /proc/meminfo fields, per worker (TP0-3) and the
API server process: VmRSS, RssAnon, RssShmem, VmPTE (status) + AnonHugePages (smaps_rollup), and the server tree's
total RSS. Waits for the server's /v1/models, then samples every 600 s for 6 samples (+ one at server up).
Usage: memprobe.py <server pid> <out jsonl> [now: one sample, no wait]"""
import json, os, sys, time, urllib.request
root, out = int(sys.argv[1]), sys.argv[2]
MI = ("MemTotal", "MemAvailable", "AnonPages", "AnonHugePages", "ShmemHugePages", "Mlocked", "Unevictable", "Cached",
      "Shmem", "PageTables")


def kb(path, keys):
    d = {}
    try:
        for l in open(path):
            k = l.split(":")[0]
            if k in keys:
                d[k] = int(l.split()[1])
    except OSError:
        pass
    return d


def tree(pid):
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(pp, []).append(int(p))
            except OSError:
                pass
    out, st = [], [pid]
    while st:
        x = st.pop(); out.append(x); st += kids.get(x, [])
    return out


def name(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return "?"


def sample(tag):
    pids = tree(root)
    rec = {"t": time.strftime("%H:%M:%S", time.gmtime()), "tag": tag,
           "meminfo_kb": kb("/proc/meminfo", MI), "tree_n": len(pids),
           "tree_rss_kb": sum(kb(f"/proc/{p}/status", ("VmRSS",)).get("VmRSS", 0) for p in pids), "procs": {}}
    for p in pids:
        n = name(p)
        if n.startswith("VLLM::Worker") or p == root or n.startswith("VLLM::EngineCor"):
            st = kb(f"/proc/{p}/status", ("VmRSS", "RssAnon", "RssShmem", "RssFile", "VmPTE"))
            st["AnonHugePages"] = kb(f"/proc/{p}/smaps_rollup", ("AnonHugePages",)).get("AnonHugePages")
            rec["procs"][f"{n}:{p}"] = st
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if len(sys.argv) > 3:
    sample(sys.argv[3])
    sys.exit()
while True:
    try:
        with urllib.request.urlopen("http://127.0.0.1:18084/v1/models", timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    time.sleep(10)
sample("server_up")
for i in range(6):
    time.sleep(600)
    sample(f"+{(i + 1) * 10}min")
