"""One JSON line with the B32-M v3 replica's state for the local watcher (read-only): Kaggle-eq host RAM (cgroup v1
rss + shmem + 7), min board-free VRAM, per rank the newest vLLM "GPU memory:" line of the newest server log and that
log's allocator cap, server-log counts (OOM / engine-death patterns, ERROR, Traceback, HTTP 500), solver counts (server
died / server start / server up), attempt records (errors, APITimeoutError), the last metrics scrapes (generation
tokens, running, waiting), the sampler's last sample time, and whether the notebook ended."""
import glob, json, os, re, subprocess, time

W, D = "/kaggle/working/dsv41_logs", "/root/e066/b32mv3"
out = {"t": time.time()}
cg = {}
for l in open("/sys/fs/cgroup/memory/memory.stat"):
    k, v = l.split()
    cg[k] = int(v)
out["keq"] = round((cg["total_rss"] + cg["total_shmem"]) / 2**30 + 7, 2)
out["shmem"] = round(cg["total_shmem"] / 2**30, 2)
q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
out["minfree"] = min(int(x) for x in q.split())
logs = sorted(glob.glob(f"{W}/server*.log"), key=os.path.getmtime)
out["logs"] = [os.path.basename(x) for x in logs]
pat = dict(oom=r"out of memory|OutOfMemoryError", dead=r"EngineDeadError|EngineCore encountered|WorkerProc hit an exception|RPC call to execute_model timed out",
           error=r" ERROR ", tb=r"Traceback", h500=r'" 500 |HTTP/1.1" 500')
cnt = {k: 0 for k in pat}
for f in logs:
    r = subprocess.run(["grep", "-acE", "|".join(pat.values()), f], capture_output=True, text=True).stdout.strip()
    if r and int(r):
        for k, p in pat.items():
            cnt[k] += int(subprocess.run(["grep", "-acE", p, f], capture_output=True, text=True).stdout.strip() or 0)
out["srv"] = cnt
ranks = {}
if logs:
    f = logs[-1]
    caps = subprocess.run(["grep", "-a", "Capped the PyTorch allocator", f], capture_output=True, text=True).stdout
    with open(f, "rb") as fh:
        fh.seek(0, 2); fh.seek(max(0, fh.tell() - 400000)); tail = re.sub(r"\x1b\[[0-9;]*m", "", fh.read().decode(errors="replace"))
    for line in re.sub(r"\x1b\[[0-9;]*m", "", caps).splitlines():
        m = re.search(r"Worker_TP(\d)\S* pid=\d+.*Capped the PyTorch allocator at ([\d.]+) GiB", line)
        if m: ranks.setdefault(m[1], {})["cap"] = float(m[2])
    for line in tail.splitlines():
        m = re.search(r"Worker_TP(\d)\S* pid=\d+\).* (\d\d:\d\d:\d\d) .*GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved \(([\d.]+) GiB in split segments\), ([\d.]+) GiB free, (\d+) allocator retries, (\d+) OOMs", line)
        if m:
            ranks.setdefault(m[1], {}).update(at=m[2], alloc=float(m[3]), peak=float(m[4]), reserved=float(m[5]), split=float(m[6]),
                                               free=float(m[7]), retries=int(m[8]), ooms=int(m[9]))
out["ranks"] = ranks
sl = open(f"{W}/solver.log", errors="replace").read() if os.path.exists(f"{W}/solver.log") else ""
out["died"], out["starts"], out["ups"] = sl.count("server died"), sl.count("server start ("), sl.count("server up after")
n = e = to = 0
if os.path.exists(f"{W}/attempts.jsonl"):
    for l in open(f"{W}/attempts.jsonl"):
        n += 1
        if '"error"' in l:
            e += 1
            to += "APITimeoutError" in l or "timed out" in l
out["att"], out["att_err"], out["att_timeout"] = n, e, to
sc = []
try:
    with open(f"{D}/metrics_scrape.txt", "rb") as fh:
        fh.seek(0, 2); fh.seek(max(0, fh.tell() - 20000)); txt = fh.read().decode(errors="replace")
    cur = None
    for line in txt.splitlines():
        if line.startswith("T "): cur = {"t": int(line.split()[1])}; sc.append(cur)
        elif cur is not None and line.startswith("vllm:"):
            k, v = line.rsplit(" ", 1); k = k.replace("vllm:", "")
            if k in ("generation_tokens_total", "num_requests_running", "num_requests_waiting"):
                cur[k] = cur.get(k, 0) + float(v)
except OSError:
    pass
out["scrape"] = sc[-5:]
try:
    with open(f"{D}/mon.jsonl", "rb") as fh:
        fh.seek(0, 2); fh.seek(max(0, fh.tell() - 20000)); out["mon_t"] = json.loads(fh.read().decode().strip().splitlines()[-1])["t"]
except Exception:
    out["mon_t"] = 0
out["nb_end"] = open(f"{D}/run.log", errors="replace").read().count("nb end")
out["papermill"] = int(subprocess.run(["pgrep", "-fc", "papermill --kernel py313b32m"], capture_output=True, text=True).stdout.strip() or 0)
print(json.dumps(out))
