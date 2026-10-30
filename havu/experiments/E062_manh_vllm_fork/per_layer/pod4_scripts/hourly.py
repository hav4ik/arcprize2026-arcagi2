#!/usr/bin/env python3
"""Short hourly summary from a pod 4 run's dashboard copy (data/): throughput per 30 min of Kaggle clock (tps.tsv),
per-rank VRAM margins over time per server instance (gpumem.log: cap - reserved, cap - peak, retries, OOMs), host RAM
(ram.tsv) and the status line. Usage: hourly.py <data dir>"""
import json, re, sys
from collections import defaultdict
D = sys.argv[1]
rows = [l.rstrip("\n").split("\t") for l in open(f"{D}/tps.tsv")][1:]
b = defaultdict(list)
for r in rows:
    h = float(r[1]); b[int(h * 2) / 2].append((float(r[2]), int(r[3]), int(r[4]), float(r[5]) if r[5] else None))
print("| Kaggle h | gen tok/s | running | waiting | KV % | minutes at 0 tok/s |")
print("|---|---|---|---|---|---|")
for k in sorted(b):
    v = b[k]; n = len(v)
    kv = [x[3] for x in v if x[3] is not None]
    print(f"| {k:.1f}-{k + 0.5:.1f} | {sum(x[0] for x in v) / n:.0f} | {sum(x[1] for x in v) / n:.1f} | "
          f"{sum(x[2] for x in v) / n:.1f} | {sum(kv) / len(kv) if kv else 0:.1f} | {sum(1 for x in v if x[0] == 0)} |")
# VRAM per instance (a new cap line from a new pid starts an instance)
caps, inst, series = {}, -1, []
pat_cap = re.compile(r"Worker_TP(\d)\S* pid=(\d+)\).* (\d\d:\d\d:\d\d) .*Capped the PyTorch allocator at ([\d.]+) GiB \(([\d.]+) GiB reserved\)")
pat_mem = re.compile(r"Worker_TP(\d)\S* pid=(\d+)\).* (\d\d:\d\d:\d\d) .*GPU memory: ([\d.]+) GiB allocated \(peak ([\d.]+) GiB\), ([\d.]+) GiB reserved.* (\d+) allocator retries, (\d+) OOMs")
for line in open(f"{D}/gpumem.log", errors="replace"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = pat_cap.search(line)
    if m:
        r, pid = m[1], m[2]
        if r in caps and caps[r][1] != pid:
            caps = {}
        if not caps:
            inst += 1
        caps[r] = (float(m[4]), pid)
        continue
    m = pat_mem.search(line)
    if m and m[1] in caps:
        series.append((inst, m[3], int(m[1]), caps[m[1]][0] - float(m[6]), caps[m[1]][0] - float(m[5]), int(m[7]), int(m[8]), float(m[5]), float(m[6])))
by = defaultdict(dict)
for s in series:
    by[(s[0], s[1][:4])][s[2]] = s  # one row per instance per 10 min (HH:M)
print("\n| server | UTC | cap - reserved TP0/1/2/3 | cap - peak TP0/1/2/3 | peak max | reserved max | retries | OOMs |")
print("|---|---|---|---|---|---|---|---|")
keys = sorted(by)
shown = [k for i, k in enumerate(keys) if i == 0 or k[0] != keys[i - 1][0] or i == len(keys) - 1 or i % 6 == 0]
for k in shown:
    d = by[k]; rs = [d.get(i) for i in range(4)]
    if not all(rs):
        continue
    print(f"| {k[0]} | {rs[0][1]} | {' / '.join(f'{x[3]:.2f}' for x in rs)} | {' / '.join(f'{x[4]:.2f}' for x in rs)} | "
          f"{max(x[7] for x in rs):.2f} | {max(x[8] for x in rs):.2f} | {max(x[5] for x in rs)} | {sum(x[6] for x in rs)} |")
ram = [l.rstrip("\n").split("\t") for l in open(f"{D}/ram.tsv")][1:]
if ram:
    print(f"\nKaggle-eq RAM: last {ram[-1][2]} GiB at {ram[-1][0]}; max {max(float(r[2]) for r in ram):.2f}")
print("status:", open(f"{D}/status.json").read().strip())
