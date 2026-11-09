#!/usr/bin/env python3
"""E049 (result tags e048or_*): does a free second guess (`alt_answer`, set inside attempt 0) recover pass@2 without a second rollout?
Per run and attempt sample: task-weighted pass@1 (main answer), free pass@2 (main or alt correct), how often alt is set
(pairs), how often a set alt is right, alt rescues (main wrong, alt right), output tokens / turns per task-attempt.
Baselines: fresh attempts with the same flags and no alt note (E046 e046p_note_all a0/a1, E047 OpenRouter ctrl a1..a3).
Usage (repo root): analyze.py [tag ...]"""
import json, statistics as st, sys
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
tags = sys.argv[1:] or ["e046p_note_all", "e047or_ctrl", "e048or_passive", "e048or_active"]

def load(tag):
    d = {}
    for r in map(json.loads, open(R + tag + ".jsonl")):
        if "task_id" in r and "error" not in r:
            d[r["task_id"], r["pair"], r["attempt"]] = r
    return d

print(f"{'run':16} {'att':>3} {'tasks':>5} {'pass@1':>7} {'free@2':>7} {'alt set':>7} {'alt ok':>6} {'rescue':>6}"
      f" {'out/att':>8} {'turns':>6}")
for tag in tags:
    d = load(tag)
    for k in sorted({a for t, p, a in d}):
        if tag.startswith("e047") and k == 0:
            continue  # seeded from e046p_note_all
        tasks = sorted({t for t, p, a in d if a == k and p == 0})
        p1, f2, out, turns, nset, nok, resc, n = [], [], [], [], 0, 0, 0, 0
        for t in tasks:
            ps = sorted({p for tt, p, a in d if tt == t and a == k})
            rs = [d[t, p, k] for p in ps]
            p1.append(st.mean(bool(r["correct"]) for r in rs))
            f2.append(st.mean(bool(r["correct"] or r.get("alt_correct")) for r in rs))
            for r in rs:
                n += 1
                if r.get("alt_pred") is not None:
                    nset += 1; nok += bool(r.get("alt_correct")); resc += (not r["correct"]) and bool(r.get("alt_correct"))
            rs = [r for r in rs if not r.get("multi_share")]
            out.append(sum(r["usage"]["completion"] for r in rs)); turns.append(sum(r["turns"] for r in rs))
        print(f"{tag:16} {k:3d} {len(tasks):5d} {100*st.mean(p1):6.1f}% {100*st.mean(f2):6.1f}% {nset:3d}/{n:<3d}"
              f" {nok:6d} {resc:6d} {st.mean(out)/1e3:7.1f}k {st.mean(turns):6.1f}")
