"""E065: image set (e065) vs text twin (e065t) vs the old E031/E036 text traces, paired on the same pairs.

Usage (repo root): .venv/bin/python experiments/E065_image_harness_rollouts/scripts/analyze.py
Per batch: pairs scored by all three (no API error), pass@1, train_pass, P(correct | train_pass), discordant pairs vs
the image set, output tokens per task (multi-test usage sits on pair 0) mean / p50 / p90, recorded $ per task.
"""
import json, os, statistics as st
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
SETS = {"image": "e065_{}", "text": "e065t_{}", "old": {"arc2_train": ["e031_arc2_train"], "nvarc_b0": ["e031_nvarc_b0"],
                                                       "nvarc_b1": ["e036_nvarc_b1"]}}

def load(tags):
    out = {}
    for t in tags:
        f = R + t + ".jsonl"
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if not r.get("error"):
                    out[(r["task_id"], r["pair"])] = r
    return out

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0

for b in ["arc2_train", "nvarc_b0", "nvarc_b1"]:
    d = {"image": load([SETS["image"].format(b)]), "text": load([SETS["text"].format(b)]), "old": load(SETS["old"][b])}
    common = set(d["image"]) & set(d["text"]) & set(d["old"])
    tasks = {k[0] for k in common}
    print(f"\n{b}: {len(common)} pairs in all three ({len(tasks)} tasks); rows image {len(d['image'])}, "
          f"text {len(d['text'])}, old {len(d['old'])}")
    print(f"{'set':6} {'pass@1':>7} {'train_p':>7} {'P(c|tp)':>7} {'c&tp':>5} {'+/- vs img':>10} "
          f"{'tok mean':>8} {'p50':>6} {'p90':>6} {'$/task':>7}")
    for s in ["image", "text", "old"]:
        rs = [d[s][k] for k in common]
        c = sum(bool(r["correct"]) for r in rs); tp = [r for r in rs if r.get("train_pass")]
        ctp = sum(bool(r["correct"]) for r in tp)
        better = sum(bool(d[s][k]["correct"]) and not d["image"][k]["correct"] for k in common)
        worse = sum(not d[s][k]["correct"] and bool(d["image"][k]["correct"]) for k in common)
        p0 = [d[s][k] for k in common if k[1] == 0]
        tok = [r["usage"]["completion"] for r in p0]; cost = [r.get("est_cost_usd") or 0 for r in p0]
        print(f"{s:6} {100*c/len(rs):6.1f}% {100*len(tp)/len(rs):6.1f}% {100*ctp/max(1,len(tp)):6.1f}% {ctp:5d} "
              f"{'' if s == 'image' else f'+{better}/-{worse}':>10} {st.mean(tok)/1e3:7.1f}k "
              f"{pct(tok,.5)/1e3:5.1f}k {pct(tok,.9)/1e3:5.1f}k {st.mean(cost):7.4f}")
