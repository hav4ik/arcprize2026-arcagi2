"""E062 L4: generation throughput of replica runs at the same pod hours, from each run's solver metrics.jsonl.

Usage: compare_tps.py RUN_DIR[=label] [RUN_DIR[=label] ...] [--upto H]
For each run: cumulative generated tokens, finished attempts ("done"), mean gen tok/s since server up, and per-hour
means of gen tok/s, KV usage, prefix-hit rate, host-restored tok/s and preemptions. Rows stop at the shortest run's
last sample unless --upto is given.
"""
import json
import sys
from pathlib import Path


def load(run):
    p = Path(run)
    for cand in (p / "out/dsv41_logs/metrics.jsonl", p / "metrics.jsonl"):
        if cand.exists():
            rows = [json.loads(line) for line in open(cand) if line.strip()]
            break
    else:
        raise SystemExit(f"no metrics.jsonl under {run}")
    up = None
    log = p / "out/dsv41_logs/solver.log"
    if log.exists():
        for line in open(log, errors="replace"):
            if "server up after" in line:
                up = float(line.split("+")[1].split("h]")[0])
                break
    return rows, up


def cum(rows):
    out, tot = [], 0.0
    for r in rows:
        tot += r["gen_tok_s"] * r["window_s"]
        out.append((r["elapsed_h"], tot, r.get("done", 0)))
    return out


def at(c, h):
    prev = [x for x in c if x[0] <= h]
    return prev[-1] if prev else (0.0, 0.0, 0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--upto", type=float)
    a = ap.parse_args()
    args, upto = a.runs, a.upto
    runs = []
    for x in args:
        path, _, label = x.partition("=")
        rows, up = load(path)
        runs.append((label or Path(path).name, rows, up, cum(rows)))
    end = upto if upto is not None else max(r[3][-1][0] for r in runs)
    print(f"| run | server up (h) | KV pool | last sample (h) |")
    print("|---|---|---|---|")
    for lab, rows, up, c in runs:
        print(f"| {lab} | {up} | | {c[-1][0]:.2f} |")
    marks = [h for h in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) if h <= end + 1e-9]
    if not marks or marks[-1] < end - 0.05:
        marks.append(round(end, 2))
    print("\n| pod h | " + " | ".join(f"{r[0]} gen tok (M) / done / mean tok/s since up" for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for h in marks:
        cells = []
        for lab, rows, up, c in runs:
            if c[-1][0] < h - 0.05:  # this run ended before h
                cells.append("-")
                continue
            e, tot, done = at(c, h)
            dt = (e - (up or 0.0)) * 3600
            cells.append(f"{tot / 1e6:.2f} / {done} / {tot / dt:.0f}" if dt > 0 else "-")
        print(f"| {h} | " + " | ".join(cells) + " |")
    print("\nPer-hour means (gen tok/s, KV, prefix hit, host tok/s, preemptions in the hour, running):")
    for lab, rows, up, c in runs:
        hours = {}
        for r in rows:
            if r["elapsed_h"] > end + 1e-9:
                break
            k = int(r["elapsed_h"])
            h = hours.setdefault(k, {"w": 0.0, "g": 0.0, "kv": 0.0, "ph": 0.0, "host": 0.0, "pre": 0.0, "run": 0.0})
            w = r["window_s"]
            h["w"] += w
            h["g"] += r["gen_tok_s"] * w
            h["kv"] += (r.get("kv") or 0) * w
            h["ph"] += (r.get("prefix_hit") or 0) * w
            h["host"] += (r.get("host_tok_s") or 0) * w
            h["pre"] += (r.get("preemptions") or 0)
            h["run"] += (r.get("srv_running") or 0) * w
        print(f"- {lab}: " + "; ".join(
            f"h{k}: {v['g'] / v['w']:.0f} tok/s, KV {v['kv'] / v['w']:.2f}, hit {v['ph'] / v['w']:.2f}, "
            f"host {v['host'] / v['w']:.0f}, pre {v['pre']:.0f}, run {v['run'] / v['w']:.0f}"
            for k, v in sorted(hours.items())))


if __name__ == "__main__":
    main()
