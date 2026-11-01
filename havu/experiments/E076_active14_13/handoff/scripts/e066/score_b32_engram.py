#!/usr/bin/env python3
"""E066 Engram A/B scoring: pulls the box's run files (no solutions there), scores them locally with ARC Prize scoring
(partial credit per test input), and compares with B32's references on the same tasks:
  - E052 arm A `new_ar` (B32's harness + auto-repair, same E128, FP8 Engram, 1 attempt; one record per task, multi-test
    preds as a list): pass@1 of attempt 0, 76.5 on all 120;
  - E046 run 7 `e043_e128t_final_all` (same E128, pre-auto-repair harness, 2 attempts): pass@1 73.7 / pass@2 84.6;
  - the same-box arms of this A/B (b32_fp4, b32_48fp4, b32_fp8) against each other once present.
  score_b32_engram.py [--pull] [--paired]
"""
import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/vu/Workspace/ARC-AGI-2")
LOCAL = ROOT / "experiments/E066_dsv41_image_pruning/results/b32_engram"
SOL = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
TASKS = sorted(SOL)
ARMS = ["b32_fp4", "b32_48fp4", "b32_fp8"]


def depth(x):
    d = 0
    while isinstance(x, list) and x:
        x, d = x[0], d + 1
    return d


def load_run(path):
    """(task, attempt) -> {pair: correct} for finished attempts (per-pair records or E052's per-task records)."""
    out = defaultdict(dict)
    if not Path(path).exists():
        return out
    latest = {}
    for line in open(path):
        r = json.loads(line)
        latest[(r["task_id"], r["pair"], r["attempt"])] = r
    for (t, p, a), r in latest.items():
        if a > 1 or t not in SOL or "error" in r:
            continue
        pred, outs = r.get("pred"), SOL[t]
        if pred is not None and depth(pred) == 3 and len(outs) > 1:  # one record holding every test input's answer
            for i, o in enumerate(outs):
                out[(t, a)][i] = i < len(pred) and pred[i] == o
        else:
            out[(t, a)][p] = pred == outs[p]
    return out


def done(d, t, a):
    return (t, a) in d and len(d[(t, a)]) == len(SOL[t])


def score(d, tasks, attempts=(0, 1)):
    p1 = p2 = 0.0
    for t in tasks:
        n = len(SOL[t])
        ok = lambda a, p: bool(d.get((t, a), {}).get(p))
        p1 += sum(ok(a, p) for a in attempts for p in range(n)) / (len(attempts) * n)
        p2 += sum(any(ok(a, p) for a in attempts) for p in range(n)) / n
    return 100 * p1 / max(1, len(tasks)), 100 * p2 / max(1, len(tasks))


def paired(x, y, tasks, attempts=(0, 1), k=0):
    dx = [score(x, [t], attempts)[k] - score(y, [t], attempts)[k] for t in tasks]
    rng = random.Random(0)
    bs = sorted(sum(dx[rng.randrange(len(dx))] for _ in dx) / len(dx) for _ in range(5000))
    return sum(dx) / len(dx), bs[125], bs[4875], sum(v > 0 for v in dx), sum(v < 0 for v in dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if a.pull:
        subprocess.run(["rsync", "-az", "-e", "ssh -o ConnectTimeout=30 -p 42304",
                        "root@23.16.123.156:/workspace/dsv4/results/", str(LOCAL) + "/"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    refs = {"E052 A (B32 harness, FP8, 1 att.)": load_run(ROOT / "experiments/E052_next_code_gate/runs/new_ar/out/dsv41_logs/attempts.jsonl"),
            "E046 run 7 (FP8, older harness)": load_run(ROOT / "evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_final_all.jsonl")}
    arms = {k: load_run(LOCAL / f"{k}.jsonl") for k in ARMS}
    for name, d in arms.items():
        if not d:
            continue
        fin = sum(done(d, t, x) for t in TASKS for x in (0, 1))
        both = [t for t in TASKS if done(d, t, 0) and done(d, t, 1)]
        a0 = [t for t in TASKS if done(d, t, 0)]
        print(f"== {name}: {fin}/240 attempts finished; {len(a0)} tasks with attempt 0, {len(both)} with both")
        if not a0:
            continue
        print(f"   pass@1 attempt 0 only, {len(a0)} tasks: {score(d, a0, (0,))[0]:.1f}")
        if both:
            p1, p2 = score(d, both)
            print(f"   pass@1 (both attempts) / pass@2, {len(both)} tasks: {p1:.1f} / {p2:.1f}")
        for rn, rd in refs.items():
            atts = (0,) if rn.startswith("E052") else (0, 1)
            sub = [t for t in (a0 if atts == (0,) else both) if all(done(rd, t, x) for x in atts)]
            if not sub:
                continue
            m, lo, hi, b, w = paired(d, rd, sub, atts)
            line = f"   vs {rn}, {len(sub)} tasks: pass@1 {score(d, sub, atts)[0]:.1f} vs {score(rd, sub, atts)[0]:.1f}, diff {m:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            if atts == (0, 1):
                m2, lo2, hi2, _, _ = paired(d, rd, sub, atts, k=1)
                line += f" | pass@2 {score(d, sub)[1]:.1f} vs {score(rd, sub)[1]:.1f}, diff {m2:+.1f} [{lo2:+.1f}, {hi2:+.1f}]"
            print(line + f" | tasks better/worse {b}/{w}")
        for on, od in arms.items():
            if on == name or not od or ARMS.index(on) < ARMS.index(name):
                continue
            sub = [t for t in TASKS if all(done(x, t, y) for x in (d, od) for y in (0, 1))]
            if len(sub) >= 5:
                m, lo, hi, b, w = paired(d, od, sub)
                m2, lo2, hi2, _, _ = paired(d, od, sub, k=1)
                print(f"   vs same-box {on}, {len(sub)} tasks: pass@1 diff {m:+.1f} [{lo:+.1f}, {hi:+.1f}] | pass@2 diff {m2:+.1f} [{lo2:+.1f}, {hi2:+.1f}] | better/worse {b}/{w}")


if __name__ == "__main__":
    main()
