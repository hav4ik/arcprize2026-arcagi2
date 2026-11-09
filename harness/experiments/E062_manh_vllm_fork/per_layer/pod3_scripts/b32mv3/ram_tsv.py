"""ram.tsv for the dashboard: the sampler's Kaggle-eq host RAM. Normal stretches are thinned to 1 point per 10 min
(the first sample of each 10-min bucket, plus the newest). Every sample of an outage is kept, plus the one just before
and the one just after. An outage runs from the first OOM (server-log torch.OutOfMemoryError) or "server died" (solver
log) to the next "server up after" + 60 s. Kaggle-eq = cgroup v1 rss + shmem + 7 GiB; samples before 03:50 UTC (no
cgroup field) use the processes' summed RssAnon + Shmem + 7 (this pod's /proc/meminfo is host-wide).
Usage: ram_tsv.py <mon.jsonl> <T0 epoch>"""
import calendar, glob, json, re, subprocess, sys, time
T0 = float(sys.argv[2]); G = 1048576
day = time.strftime("%Y-%m-%d", time.gmtime(T0))
ts = lambda hms, d=day: calendar.timegm(time.strptime(f"{d} {hms}", "%Y-%m-%d %H:%M:%S"))
W = "/kaggle/working/dsv41_logs"
bad, ups = [], []
for f in glob.glob(f"{W}/server*.log"):
    out = subprocess.run(["grep", "-a", "OutOfMemoryError", f], capture_output=True, text=True).stdout
    for m in re.finditer(r"ERROR (\d\d)-(\d\d) (\d\d:\d\d:\d\d)", re.sub(r"\x1b\[[0-9;]*m", "", out)):
        bad.append(ts(m[3], f"{time.gmtime(T0).tm_year}-{m[1]}-{m[2]}"))
try:
    for line in open(f"{W}/solver.log", errors="replace"):
        m = re.match(r"\[(\d\d:\d\d:\d\d) \+([\d.]+)h\]", line)
        if not m:
            continue
        t = T0 + float(m[2]) * 3600  # the elapsed stamp carries the date; the clock time refines it
        t = min((ts(m[1]) + k * 86400 for k in (-1, 0, 1)), key=lambda x: abs(x - t))
        if "server died" in line: bad.append(t)
        elif "server up after" in line: ups.append(t)
except OSError:
    pass
wins = []
for b in sorted(bad):
    up = min([u for u in ups if u > b] or [time.time() + 1e9])
    if wins and b <= wins[-1][1]:
        continue
    wins.append((b, up + 60))
R = [json.loads(l) for l in open(sys.argv[1]) if '"meminfo_kb"' in l]
inw = lambda t: any(a <= t <= b for a, b in wins)
keep = set()
for i, r in enumerate(R):
    if inw(r["t"]):
        keep.update({i, max(0, i - 1), min(len(R) - 1, i + 1)})
seen = set()
for i, r in enumerate(R):
    k = int(r["t"] // 600)
    if k not in seen:
        seen.add(k); keep.add(i)
keep.add(len(R) - 1)
print("utc\tkaggle_clock_h\tkaggle_eq_gib")
for i in sorted(keep):
    r = R[i]; cg = r.get("cgroup_v1_bytes") or {}
    keq = (cg["total_rss"] + cg["total_shmem"]) / 2**30 + 7 if "total_rss" in cg else \
        (sum(g["anon_kb"] for g in r["groups"].values()) + r["meminfo_kb"]["Shmem"]) / G + 7
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(r['t']))}\t{(r['t'] - T0) / 3600:.3f}\t{keq:.2f}")
