#!/usr/bin/env python3
"""E042: markdown table of decode_window results (runs/<name>/dw/ctx<ctx>_c<C>.json) for the given runs.

Run: dw_table.py [--base name,name,...] run [run ...]   (glob patterns allowed, e.g. 'p12_*')
Each cell is decode tok/s / prefill tok/s; --base averages those runs into a 'base mean' row and adds % vs it.
"""

import argparse
import glob
import json
import os
import re

RUNS = "/root/dsv41/runs"


def load(name):
    out = {}
    for f in glob.glob(f"{RUNS}/{name}/dw/ctx*_c*.json"):
        ctx, c = map(int, re.findall(r"ctx(\d+)_c(\d+)", f)[0])
        d = json.load(open(f))
        out[(ctx, c)] = (d["decode_tok_s"], d["prefill_tok_s"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="")
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    names = []
    for r in a.runs:
        names += sorted(os.path.basename(x) for x in glob.glob(f"{RUNS}/{r}")) or [r]
    data = {n: load(n) for n in names}
    base = [b for b in a.base.split(",") if b]
    if base:
        pts = set.intersection(*(set(data.get(b) or load(b)) for b in base))
        bm = {pt: tuple(sum((data.get(b) or load(b))[pt][i] for b in base) / len(base) for i in (0, 1)) for pt in pts}
        data = {"base mean": bm, **data}
    points = sorted({pt for d in data.values() for pt in d}, key=lambda x: (x[0], x[1]))
    print("| run | " + " | ".join(f"{ctx // 1024}K x {c}" for ctx, c in points) + " |")
    print("|---|" + "---:|" * len(points))
    for n, d in data.items():
        cells = []
        for pt in points:
            if pt not in d:
                cells.append("")
                continue
            dec, pre = d[pt]
            cell = f"{dec:.0f} / {pre:.0f}"
            if base and n != "base mean" and pt in data["base mean"]:
                b = data["base mean"][pt]
                cell += f" ({100 * (dec / b[0] - 1):+.0f}% / {100 * (pre / b[1] - 1):+.0f}%)"
            cells.append(cell)
        print(f"| {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
