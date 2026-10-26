#!/usr/bin/env python3
"""E042: per-attempt accuracy of Kaggle/pod eval-bench runs (solver attempts.jsonl) on the pairs they share, plus
E043's E128 records (8x box, per-pair) as the reference rate on the same pairs. A coarse check that a serving change
(e.g. the Humming dense kernel) does not cost accuracy; benches finish the smallest tasks first, so they overlap.
Usage: bench_accuracy.py <solutions.json> <name>=<attempts.jsonl> [<name>=<attempts.jsonl> ...]
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E043 = ROOT / "evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80.jsonl"
sol = json.load(open(sys.argv[1]))
runs = {}
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    recs = {}
    for r in map(json.loads, open(path)):
        if "task_id" in r and r.get("pred") is not None and not r.get("error"):
            if r.get("multi_test"):  # one record per pair (E046)
                for q, g in enumerate(r["pred"] if isinstance(r["pred"], list) else []):
                    recs[(r["task_id"], q, r["attempt"])] = g == sol[r["task_id"]][q]
            else:
                recs[(r["task_id"], r["pair"], r["attempt"])] = r["pred"] == sol[r["task_id"]][r["pair"]]
    runs[name] = recs
ref = collections.defaultdict(list)
for r in map(json.loads, open(E043)):
    if "task_id" in r:
        ref[(r["task_id"], r["pair"])].append(bool(r.get("correct")))
for name, recs in runs.items():
    pairs = {(t, p) for t, p, _ in recs}
    exp = [sum(ref[k]) / len(ref[k]) for k in pairs if ref.get(k)]
    print(f"{name:12s} {len(recs):3d} attempts on {len(pairs):3d} pairs: correct {sum(recs.values()):3d} "
          f"({100 * sum(recs.values()) / max(1, len(recs)):.1f}%); E043 pass@1 on these pairs "
          f"{100 * sum(exp) / max(1, len(exp)):.1f}%")
names = list(runs)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        common = set(runs[a]) & set(runs[b])
        if common:
            ca, cb = sum(runs[a][k] for k in common), sum(runs[b][k] for k in common)
            print(f"{a} vs {b}: {len(common)} shared attempts, correct {ca} vs {cb}")
