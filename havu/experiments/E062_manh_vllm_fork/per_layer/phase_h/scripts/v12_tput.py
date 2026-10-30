"""Throughput per 30 min of Kaggle clock from the server log's 10 s "Engine 000" lines (all server logs of the run),
plus the solver's KV-gate lines and metrics.jsonl (host-restore share). Usage: v12_tput.py <dsv41_logs> <t0 epoch (notebook start)>"""
import calendar, glob, json, os, re, sys, time
L, T0 = sys.argv[1], float(sys.argv[2]); ANSI = re.compile(r"\x1b\[[0-9;]*m")
E = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers.py:\d+\] Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%(?:, External prefix cache hit rate: ([\d.]+)%)?")
yr = time.gmtime(T0).tm_year; rows = []
for f in glob.glob(f"{L}/**/server.log", recursive=True):
    for l in open(f, errors="replace"):
        m = E.search(ANSI.sub("", l))
        if m:
            mo, d, H, M, S = map(int, m.groups()[:5])
            t = calendar.timegm((yr, mo, d, H, M, S))
            rows.append((t, float(m.group(6)), float(m.group(7)), int(m.group(8)), int(m.group(9)), float(m.group(10)), float(m.group(12) or 0)))
rows.sort()
gate = []
for l in open(f"{L}/solver.log", errors="replace"):
    m = re.search(r"\[(\d\d):(\d\d):(\d\d) \+([\d.]+)h\] KV gate: (.*?); holding new attempts at (\d+) running", l)
    if m: gate.append((T0 + 0 + float(m.group(4)) * 3600, m.group(5), int(m.group(6))))
met = [json.loads(l) for l in open(f"{L}/metrics.jsonl")] if os.path.exists(f"{L}/metrics.jsonl") else []
print("| Kaggle clock (h) | mean gen tok/s | mean running | tok/s per stream | mean waiting | max waiting | mean KV % | ext. (host) prefix hit % | solver gate holds (in-flight at hold) |")
print("|---|---|---|---|---|---|---|---|---|")
b = {}
for r in rows:
    b.setdefault(int((r[0] - T0) // 1800), []).append(r)
for k in sorted(b):
    x = b[k]; n = len(x); g = sum(r[2] for r in x) / n; run = sum(r[3] for r in x) / n
    gh = [q for q in gate if T0 + k * 1800 <= q[0] < T0 + (k + 1) * 1800]
    print(f"| {k*0.5:.1f}-{k*0.5+0.5:.1f} | {g:.0f} | {run:.1f} | {g / run if run else 0:.1f} | {sum(r[4] for r in x)/n:.1f} | {max(r[4] for r in x)} | "
          f"{sum(r[5] for r in x)/n:.1f} | {sum(r[6] for r in x)/n:.1f} | {len(gh)} ({', '.join(str(q[2]) for q in gh[:6])}) |")
xs = [r for r in rows if r[0] >= T0 + 600 + (rows[0][0] - T0 if rows else 0)]
if xs:
    print(f"\nwhole run after the first 10 min of serving: mean gen {sum(r[2] for r in xs)/len(xs):.0f} tok/s, mean running {sum(r[3] for r in xs)/len(xs):.1f}, n={len(xs)} 10-s lines")
print("\nlast gate lines:"); [print(" ", time.strftime('%H:%M:%S', time.gmtime(q[0])), q[1], "->", q[2], "in flight") for q in gate[-6:]]
