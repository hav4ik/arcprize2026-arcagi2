"""Per-sweep GPU stats from gpu.csv using chain.log timestamps. Usage: gpu_windows.py <run_dir>"""
import sys, re, datetime as dt, collections, statistics as st
R = sys.argv[1]
rows = []
for line in open(f"{R}/gpu.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 7: continue
    try:
        t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f")
        rows.append((t, int(p[1]), float(p[2].split()[0]), float(p[3].split()[0]), float(p[4]), float(p[5].split()[0]), float(p[6].split()[0])))
    except Exception:
        pass
day0 = rows[0][0].date()
ev = []
prev = None; add = 0
for line in open(f"{R}/chain.log"):
    m = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (.*)", line)
    if not m: continue
    t = dt.datetime.combine(day0, dt.time(int(m[1]), int(m[2]), int(m[3])))
    if prev and t + dt.timedelta(days=add) < prev - dt.timedelta(hours=1): add += 1
    t += dt.timedelta(days=add); prev = t
    ev.append((t, m[4]))
def find(prefix):
    for t, s in ev:
        if s.startswith(prefix): return t
start = ev[0][0]; up = find("server up"); A = find("sweep A"); B = find("sweep B"); C = find("sweep C"); end = find("CHAIN_DONE") or ev[-1][0]
wins = [("load/startup", start, up), ("sweep A", A, B), ("sweep B", B, C), ("sweep C", C, end)]
print("peak mem MiB per GPU (whole run):", {g: max(r[2] for r in rows if r[1] == g) for g in range(4)})
for name, a, b in wins:
    if not a or not b: print(name, "n/a"); continue
    w = [r for r in rows if a <= r[0] <= b]
    if not w: print(name, "no samples"); continue
    print(f"{name} {a:%H:%M:%S}-{b:%H:%M:%S} ({(b-a).total_seconds()/60:.1f} min)")
    for g in range(4):
        x = [r for r in w if r[1] == g]
        busy = [r for r in x if r[3] > 0]
        print(f"  gpu{g}: mem max {max(r[2] for r in x):.0f} MiB | util mean {st.mean(r[3] for r in x):.0f}% | "
              f"temp max {max(r[4] for r in x):.0f} mean {st.mean(r[4] for r in x):.1f} | "
              f"SM MHz max {max(r[5] for r in x):.0f} mean {st.mean(r[5] for r in x):.0f} "
              f"(busy-mean {st.mean(r[5] for r in busy) if busy else 0:.0f}, busy-min {min(r[5] for r in busy) if busy else 0:.0f}) | "
              f"power max {max(r[6] for r in x):.0f} W mean {st.mean(r[6] for r in x):.0f} W")
