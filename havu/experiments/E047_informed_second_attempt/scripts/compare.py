#!/usr/bin/env python3
"""E047: informed later attempts vs fresh ones. Attempt 0 is shared (E046 e046p_note_all); each later attempt k gives
one sample of pass@2 = (attempt 0, attempt k). Per run: task-weighted pass@2 and pass@1 of the later attempts (mean over
samples), output tokens / turns per task-attempt, how often the later attempt resubmits attempt 0's answer, and what
changed answers did (fixed = a0 wrong -> right, broke = a0 right -> wrong).
Usage (repo root): compare.py [tag ...]"""
import json, statistics as st, sys
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
tags = sys.argv[1:] or ["e046p_note_all", "e047or_ctrl", "e047or_verify", "e047or_verifycode", "e047_verify",
                        "e047_verifycode"]

def load(tag):
    d = {}
    try:
        for r in map(json.loads, open(R + tag + ".jsonl")):
            if "task_id" in r and "error" not in r:
                d[r["task_id"], r["pair"], r["attempt"]] = r
    except FileNotFoundError:
        pass
    return d

print(f"{'run':20} {'samples':>7} {'tasks':>5} {'pass@2':>7} {'pass@1 ak':>9} {'out/att':>8} {'turns':>6} {'same as a0':>10}"
      f" {'fixed':>5} {'broke':>5} {'$':>6}")
for tag in tags:
    d = load(tag)
    if not d:
        continue
    tasks = sorted({t for t, p, a in d if a == 0 and p == 0})
    ks = sorted({a for t, p, a in d if a > 0})
    for k in ks:
        done = [t for t in tasks if (t, 0, k) in d]
        if not done:
            continue
        p2, p1, out, turns, same, fixed, broke, cost, n = [], [], [], [], 0, 0, 0, 0.0, 0
        for t in done:
            ps = sorted({p for tt, p, a in d if tt == t and a == 0})
            p2.append(st.mean(bool(d[t, p, 0]["correct"] or d.get((t, p, k), {}).get("correct")) for p in ps))
            p1.append(st.mean(bool(d.get((t, p, k), {}).get("correct")) for p in ps))
            rs = [d[t, p, k] for p in ps if (t, p, k) in d and not d[t, p, k].get("multi_share")]
            out.append(sum(r["usage"]["completion"] for r in rs)); turns.append(sum(r["turns"] for r in rs))
            cost += sum(r.get("est_cost_usd") or 0 for r in rs)
            for p in ps:
                if (t, p, k) not in d:
                    continue
                a0, ak = d[t, p, 0], d[t, p, k]
                n += 1
                if ak["pred"] == a0["pred"]:
                    same += 1
                else:
                    fixed += (not a0["correct"]) and bool(ak["correct"])
                    broke += bool(a0["correct"]) and not ak["correct"]
        print(f"{tag:20} {'a' + str(k):>7} {len(done):5d} {100*st.mean(p2):6.1f}% {100*st.mean(p1):8.1f}% {st.mean(out)/1e3:7.1f}k"
              f" {st.mean(turns):6.1f} {100*same/n:9.0f}% {fixed:5d} {broke:5d} {cost:6.2f}")
