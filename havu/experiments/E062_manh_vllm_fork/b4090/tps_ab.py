#!/usr/bin/env python3
"""E062 4090: generation throughput of A/B windows aligned at server-up (not at the run's start: the fork's server
takes ~15 min to start here, B32's ~4 min), from vLLM's 10-s stats lines in each run's server.log.

    tps_ab.py RUN_DIR[=label] ... [--minutes M]

For each run: minutes from server-up to stop-new; mean gen tok/s, prompt tok/s, running, KV %, prefix-hit % over the
first M minutes after server-up (M = the shortest run's span by default) and per 10-min bin; attempts finished
(metrics.jsonl `done`) at the last sample before M.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: "
                  r"([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, "
                  r"Prefix cache hit rate: ([\d.]+)%")


def find(run, name):
    for p in Path(run).rglob(name):
        return p
    raise SystemExit(f"{name} not found under {run}")


def load(run):
    rows = []
    for line in open(find(run, "server.log"), errors="replace"):
        m = LINE.search(line)
        if m:
            t = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S")
            rows.append((t, *(float(x) for x in m.groups()[1:])))
    solver = open(find(run, "solver.log"), errors="replace").read()
    up = re.search(r"\[(\d\d:\d\d:\d\d) \+([\d.]+)h\] server up after (\d+)s", solver)
    stop = re.search(r"stop-new ([\d.]+)h", solver)
    up_h, stop_h = float(up.group(2)), float(stop.group(1))
    t_up = datetime.strptime(rows[0][0].strftime("%Y-%m-%d ") + up.group(1), "%Y-%m-%d %H:%M:%S")
    metrics = [json.loads(x) for x in open(find(run, "metrics.jsonl")) if x.strip()]
    return rows, t_up, up_h, stop_h, int(up.group(3)), metrics


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    minutes = None
    if "--minutes" in sys.argv:
        minutes = float(sys.argv[sys.argv.index("--minutes") + 1])
    runs = []
    for a in args:
        d, _, lab = a.partition("=")
        runs.append((lab or Path(d).name, *load(d)))
    spans = [(stop_h - up_h) * 60 for _, _, _, up_h, stop_h, _, _ in runs]
    M = minutes or min(spans)
    print(f"window: first {M:.0f} min after server-up (stop-new spans: "
          + ", ".join(f"{r[0]} {s:.0f} min" for r, s in zip(runs, spans)) + ")")
    print(f"{'run':16s} {'start s':>7s} {'gen tok/s':>9s} {'prompt':>7s} {'running':>7s} {'KV %':>5s} {'hit %':>5s} "
          f"{'done':>5s}   per 10 min gen tok/s")
    for lab, rows, t_up, up_h, stop_h, start_s, metrics in runs:
        sel = [r for r in rows if 0 <= (r[0] - t_up).total_seconds() < M * 60]
        if not sel:
            print(f"{lab:16s} no samples")
            continue
        mean = lambda i: sum(r[i] for r in sel) / len(sel)
        bins = []
        for b in range(int(M // 10) + (1 if M % 10 else 0)):
            bs = [r[2] for r in sel if b * 600 <= (r[0] - t_up).total_seconds() < (b + 1) * 600]
            bins.append(f"{sum(bs) / len(bs):.0f}" if bs else "-")
        done = [m.get("done", 0) for m in metrics if m["elapsed_h"] <= up_h + M / 60]
        print(f"{lab:16s} {start_s:7d} {mean(2):9.1f} {mean(1):7.1f} {mean(3):7.1f} {mean(5):5.1f} {mean(6):5.1f} "
              f"{(done[-1] if done else 0):5d}   {' '.join(bins)}")


if __name__ == "__main__":
    main()
