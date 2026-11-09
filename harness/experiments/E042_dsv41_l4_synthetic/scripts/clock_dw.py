#!/usr/bin/env python3
"""E042: clock-adjusted decode-bench A/B. The box's GPU2 throttles at 87 C and the slowest GPU gates every TP4 step,
so raw tok/s between runs mixes the kernel effect with thermal state (R3 vs R5: slowest-GPU clock 721 vs 1038 MHz).

For each run's runs/<name>/dw/ctx*_c*.json (decode_window with t0_unix), takes the slowest GPU's mean SM clock over
the point's prefill [0, prefill_all_first_s] and window [window_start_s, +window_s] from runs/<name>/gpu.csv, then per
point and phase fits log(tok/s) = a + b * log(clock) + d * [arm] over all runs (arm = run name without its trailing digits).
d is the clock-adjusted effect of each arm vs the base arm; b is the clock elasticity.

Run: clock_dw.py --base-arm r7_b 'r7_[bhe][0-9]*'
"""

import argparse
import glob
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np

RUNS = "/root/dsv41/runs"


def clocks(name):
    """{unix second: [sm MHz per gpu]} from gpu.csv (nvidia-smi timestamps, UTC on the box)."""
    out = {}
    for line in open(f"{RUNS}/{name}/gpu.csv"):
        f = [x.strip() for x in line.split(",")]
        t = datetime.strptime(f[0][:19], "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        out.setdefault(int(t), {})[int(f[1])] = float(f[5].split()[0])
    return {t: [g[i] for i in range(4)] for t, g in out.items() if len(g) == 4}


def slowest(clk, a, b):
    sel = [min(v) for t, v in clk.items() if a <= t <= b]
    return sum(sel) / len(sel) if sel else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-arm", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    names = sorted({os.path.basename(x) for r in a.runs for x in glob.glob(f"{RUNS}/{r}") if glob.glob(f"{x}/dw/*.json")})
    rows = []
    for n in names:
        clk = clocks(n)
        arm = re.sub(r"\d+$", "", n)
        for f in sorted(glob.glob(f"{RUNS}/{n}/dw/ctx*_c*.json")):
            d = json.load(open(f))
            if "t0_unix" not in d:
                continue
            t0 = d["t0_unix"]
            pt = f"{d['ctx']}x{d['conc']}"
            rows.append((n, arm, pt, "prefill", d["prefill_tok_s"], slowest(clk, t0, t0 + d["prefill_all_first_s"])))
            w0 = t0 + d["window_start_s"]
            rows.append((n, arm, pt, "decode", d["decode_tok_s"], slowest(clk, w0, w0 + d["window_s"])))
    print("| run | point | phase | tok/s | slowest-GPU MHz |\n|---|---|---|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[2]} | {r[3]} | {r[4]:.0f} | {r[5]:.0f} |")
    arms = sorted({r[1] for r in rows} - {a.base_arm})
    print("\n| point | phase | clock elasticity b | " + " | ".join(f"{x} vs {a.base_arm} (clock-adj.)" for x in arms)
          + " | " + " | ".join(f"{x} raw" for x in arms) + " |")
    print("|---|---|---:|" + "---:|" * (2 * len(arms)))
    for pt in sorted({r[2] for r in rows}, key=lambda s: tuple(map(int, s.split("x")))):
        for ph in ("decode", "prefill"):
            sub = [r for r in rows if r[2] == pt and r[3] == ph and not math.isnan(r[5])]
            y = np.log([r[4] for r in sub])
            X = np.column_stack([np.ones(len(sub)), np.log([r[5] for r in sub])] + [[r[1] == x for r in sub] for x in arms])
            coef, *_ = np.linalg.lstsq(X.astype(float), y, rcond=None)
            raw = []
            for x in arms:
                mx = np.mean([r[4] for r in sub if r[1] == x]); mb = np.mean([r[4] for r in sub if r[1] == a.base_arm])
                raw.append(f"{100 * (mx / mb - 1):+.1f}%")
            adj = [f"{100 * (math.exp(c) - 1):+.1f}%" for c in coef[2:]]
            print(f"| {pt} | {ph} | {coef[1]:.2f} | " + " | ".join(adj) + " | " + " | ".join(raw) + " |")


if __name__ == "__main__":
    main()
