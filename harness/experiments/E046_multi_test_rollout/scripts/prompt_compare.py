#!/usr/bin/env python3
"""E046 --multi-prompt screen: per variant, on the same multi-test pairs: pass@1 / pass@2 per pair, no-answer
task-attempts, output and input (prompt, all turns) tokens and turns per task-attempt, cost. Per-pair baseline = E030 hdrmin.
Usage (repo root): prompt_compare.py [tag ...]   (default: e046_multi_hdrmin e046p_note e046p_lean e046p_early e046p_first)"""
import json, statistics as st, sys
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
BASE = "e030_hdrmin"

def load(tag):
    d = {}
    try:
        for r in map(json.loads, open(R + tag + ".jsonl")):
            if "task_id" in r:
                d[(r["task_id"], r["pair"], r["attempt"])] = r
    except FileNotFoundError:
        pass
    return d

tags = sys.argv[1:] or ["e046_multi_hdrmin", "e046p_note", "e046p_lean", "e046p_early", "e046p_first"]
runs = {t: load(t) for t in tags}
base = load(BASE)
# pairs of tasks that every run has finished (both attempts of pair 0, no API error)
def done(d):
    return {t for t, p, a in d if p == 0 and all((t, 0, b) in d and "error" not in d[t, 0, b] for b in (0, 1))}
tasks = set.intersection(*(done(d) for d in runs.values() if d))
pairs = sorted({(t, p) for d in runs.values() for t, p, a in d if t in tasks})
print(f"{len(tasks)} tasks, {len(pairs)} pairs finished in every run")
print(f"{'run':22} {'pass@1':>7} {'pass@2':>7} {'no-ans':>7} {'out/task-att':>13} {'median':>7} {'in/task-att':>12} {'turns':>6} {'$':>6} {'task pass@2':>11}")
for name, d in [("per-pair " + BASE, base)] + list(runs.items()):
    p1, p2, tw = [], [], {}
    for t, p in pairs:
        c = [bool(d.get((t, p, a), {}).get("correct")) for a in (0, 1)]
        p1.append(sum(c) / 2); p2.append(any(c)); tw.setdefault(t, []).append(any(c))
    out, turns, noans, cost, pr1 = [], [], 0, 0.0, []
    for t in tasks:
        for a in (0, 1):
            rs = [r for (tt, p, aa), r in d.items() if tt == t and aa == a and not r.get("multi_share")]
            out.append(sum(r["usage"]["completion"] for r in rs)); turns.append(sum(r["turns"] for r in rs))
            pr1.append(sum(r["usage"]["prompt"] for r in rs))
            cost += sum(r.get("est_cost_usd") or 0 for r in rs)
            r0 = d.get((t, 0, a))
            noans += r0 is not None and r0.get("pred") is None
    print(f"{name:22} {100*st.mean(p1):6.1f}% {100*st.mean(p2):6.1f}% {noans:7d} {st.mean(out)/1e3:12.1f}k {st.median(out)/1e3:6.1f}k"
          f" {st.mean(pr1)/1e3:11.0f}k {st.mean(turns):6.1f} {cost:6.2f} {100*st.mean(st.mean(v) for v in tw.values()):10.1f}%")
