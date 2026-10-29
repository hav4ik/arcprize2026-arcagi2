"""Read-only host-memory probe for Manh's vLLM server (use_thp comparison with pod 3). Writes JSON lines:
  {"kind": "vmstat", "label": "before_load" | "after_load", compact_*}  (before: at start; after: server ready)
  {"kind": "sample", "t": ..., "meminfo": {...}, "procs": [...], "tree_rss_kib": ...}  every <interval> s for <minutes>
  {"kind": "log", "engram": [...], "madv_collapse": [...], "model_loading": [...]}  (server.log lines, once ready)
Usage: mem_probe.py <out.jsonl> <server.log glob> <minutes> <interval_s> [--now]"""
import glob, json, os, re, sys, time
out, log_glob, minutes, interval = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
now = "--now" in sys.argv
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MEM = ("MemTotal", "MemAvailable", "AnonPages", "AnonHugePages", "Mlocked", "Unevictable", "Cached", "Shmem",
       "PageTables")
def w(rec):
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
def vmstat(label):
    v = {l.split()[0]: int(l.split()[1]) for l in open("/proc/vmstat") if l.startswith("compact_")}
    w({"kind": "vmstat", "label": label, "t": time.time(), **{k: v[k] for k in ("compact_stall", "compact_fail",
                                                                              "compact_success") if k in v}})
def kv(path, keys):
    d = {}
    try:
        for l in open(path):
            k = l.split(":")[0]
            if k in keys:
                d[k] = int(l.split()[1])
    except OSError:
        pass
    return d
def children():
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(pp, []).append(int(p))
            except OSError:
                pass
    return kids
def leader():
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                a = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if b"vllm.entrypoints.cli.main" in a and b"serve" in a:
                return int(p)
def tree(root):
    kids, todo, seen = children(), [root], []
    while todo:
        p = todo.pop(); seen.append(p); todo += kids.get(p, [])
    return seen
def logfile():
    hits = sorted(glob.glob(log_glob), key=os.path.getmtime)
    return hits[-1] if hits else None
T_START = time.time()
def ready():  # a server.log written after the probe started (not an earlier run's) that says the server is up
    f = logfile()
    return bool(f) and os.path.getmtime(f) > T_START and "Application startup complete" in open(f, errors="replace").read()
def sample():
    root = leader()
    procs, tot = [], 0
    for p in (tree(root) if root else []):
        st = kv(f"/proc/{p}/status", ("VmRSS", "RssAnon", "RssShmem", "VmPTE"))
        sm = kv(f"/proc/{p}/smaps_rollup", ("AnonHugePages",))
        try:
            name = open(f"/proc/{p}/comm").read().strip()
        except OSError:
            continue
        tot += st.get("VmRSS", 0)
        if st.get("VmRSS", 0) > 512 * 1024:
            procs.append({"pid": p, "name": name, **st, **sm})
    w({"kind": "sample", "t": time.time(), "meminfo": kv("/proc/meminfo", MEM), "procs": procs, "tree_rss_kib": tot,
       "leader": root})
if not now:
    vmstat("before_load")
    while not ready():
        time.sleep(10)
    vmstat("after_load")
txt = ANSI.sub("", open(logfile(), errors="replace").read()) if logfile() else ""
w({"kind": "log", "engram": re.findall(r".*Engram table offloaded.*", txt),
   "madv_collapse": re.findall(r".*(?:MADV_COLLAPSE|madv_collapse|collapse).*", txt, re.I)[:40],
   "model_loading": re.findall(r".*Model loading took.*", txt), "thp_config": re.findall(r".*use_thp=.*", txt)[:2]})
t0 = time.time()
while True:
    sample()
    if time.time() - t0 + interval > minutes * 60 + 1:
        break
    time.sleep(interval)
