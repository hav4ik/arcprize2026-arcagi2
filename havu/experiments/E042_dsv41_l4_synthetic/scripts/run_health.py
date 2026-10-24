#!/usr/bin/env python3
"""E042: per-run health from gpumon.sh (gpu.csv) and kvmon.sh (kvmon.log): GPU temperature and SM clock under load,
KV usage, waiting requests and preemptions (vLLM's evictions of running requests).
Usage: run_health.py <run dir> [<run dir> ...]"""

import re
import statistics
import sys
from pathlib import Path


def gpus(f):
    by = {}
    for line in open(f):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue
        try:
            i, util, temp, clk = int(p[1]), int(p[3].split()[0]), int(p[4]), int(p[5].split()[0])
        except ValueError:
            continue
        by.setdefault(i, []).append((util, temp, clk))
    out = []
    for i, rows in sorted(by.items()):
        busy = [r for r in rows if r[0] >= 90] or rows
        clks = sorted(r[2] for r in busy)
        out.append(f"GPU{i} max {max(r[1] for r in rows)} C, clock p50 {clks[len(clks) // 2]} / p5 "
                   f"{clks[len(clks) // 20]} MHz")
    return "; ".join(out)


def kv(f):
    rows = [dict(re.findall(r"(\w+)=([\d.e+-]+)", l)) for l in open(f)]
    rows = [r for r in rows if "kv_cache_usage_perc" in r]
    if not rows:
        return "no kvmon rows"
    u = [float(r["kv_cache_usage_perc"]) for r in rows]
    w = [float(r.get("num_requests_waiting", 0)) for r in rows]
    pre = float(rows[-1].get("num_preemptions_total", 0)) - float(rows[0].get("num_preemptions_total", 0))
    return (f"KV peak {max(u):.2f}, p50 {statistics.median(u):.2f}; waiting max {max(w):.0f}; "
            f"preemptions {pre:.0f} over {len(rows) * 30 / 60:.0f} min")


for d in map(Path, sys.argv[1:]):
    parts = []
    if (d / "gpu.csv").exists():
        parts.append(gpus(d / "gpu.csv"))
    if (d / "kvmon.log").exists():
        parts.append(kv(d / "kvmon.log"))
    print(f"{d.name}: " + " | ".join(parts))
