#!/usr/bin/env python3
"""E047: offline policy mixes. Attempt 0 is shared, so each task's later attempt can come from the run picked by
attempt 0's train check (pair 0): policy = (run if train check failed, run if passed). Reports pass@2 (task-weighted,
mean over the two samples) and later-attempt output tokens per task, plus the token share spent on train-failed tasks.
Usage (repo root): policy_mix.py"""
import json, statistics as st
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
runs = {}
for tag in ("ctrl", "verify", "verifycode", "hybrid"):
    d = {}
    try:
        for r in map(json.loads, open(f"{R}e047or_{tag}.jsonl")):
            if "task_id" in r and "error" not in r:
                d[r["task_id"], r["pair"], r["attempt"]] = r
    except FileNotFoundError:
        continue
    runs[tag] = d
base = runs["ctrl"]
tasks = sorted({t for t, p, a in base if a == 0 and p == 0})
ps = {t: sorted({p for tt, p, a in base if tt == t and a == 0}) for t in tasks}
tp = {t: bool(base[t, 0, 0].get("train_pass")) for t in tasks}
print(f"{sum(not v for v in tp.values())} of {len(tasks)} tasks fail attempt 0's train check")
print(f"{'fail -> / pass ->':28} {'pass@2':>7} {'tok/task':>9} {'tok fail-tasks':>15}")
for f in runs:
    for g in list(runs) + ["none"]:
        p2, tok, tokf = [], [], []
        for k in (1, 2):
            sc = tt = tf = 0
            for t in tasks:
                src = runs[f] if not tp[t] else (runs[g] if g != "none" else None)
                if src is not None and (t, 0, k) not in src:
                    src = runs["ctrl"]  # missing attempt (errors): fall back
                sc += st.mean(bool(base[t, p, 0]["correct"] or (src and src.get((t, p, k), {}).get("correct"))) for p in ps[t])
                if src:
                    u = sum(src[t, p, k]["usage"]["completion"] for p in ps[t]
                            if (t, p, k) in src and not src[t, p, k].get("multi_share"))
                    tt += u; tf += u if not tp[t] else 0
            p2.append(100 * sc / len(tasks)); tok.append(tt / len(tasks)); tokf.append(tf / len(tasks))
        print(f"{f + ' / ' + g:28} {st.mean(p2):6.1f}% {st.mean(tok)/1e3:8.1f}k {st.mean(tokf)/1e3:14.1f}k")
