"""Per bench step vLLM periodic-metrics summary. Usage: metrics_windows.py <server.log> <chain.log>"""
import sys, re, datetime as dt
pat = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) .*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
M = []
for line in open(sys.argv[1], errors="replace"):
    m = pat.search(line)
    if m:
        t = dt.datetime(2026, int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]))
        M.append((t, float(m[6]), float(m[7]), int(m[8]), int(m[9]), float(m[10])))
ev = []; day = dt.date(2026, 10, 21); prev = None
for line in open(sys.argv[2]):
    m = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (.*)", line)
    if not m: continue
    t = dt.datetime.combine(day, dt.time(int(m[1]), int(m[2]), int(m[3])))
    if prev and t < prev - dt.timedelta(hours=1): day += dt.timedelta(days=1); t += dt.timedelta(days=1)
    prev = t; ev.append((t, m[4].strip()))
steps = []
for i, (t, s) in enumerate(ev):
    m = re.match(r"([ABC]) (c\d+):", s)
    if m: steps.append((m[1] + " " + m[2], ev[i - 1][0], t))
print("step      | max run | max wait | max KV% | max prompt tok/s | max gen tok/s | pure-decode gen tok/s (prompt=0, wait=0, run=C)")
for name, a, b in steps:
    w = [x for x in M if a < x[0] <= b]
    if not w: print(name, "no metrics"); continue
    C = int(name.split("c")[1])
    pure = [x[2] for x in w if x[1] == 0 and x[4] == 0 and x[3] == C]
    print(f"{name:9s} | {max(x[3] for x in w):7d} | {max(x[4] for x in w):8d} | {max(x[5] for x in w):7.1f} | {max(x[1] for x in w):16.0f} | {max(x[2] for x in w):13.1f} | "
          + (f"{sum(pure)/len(pure):.1f} (n={len(pure)})" if pure else "-"))
