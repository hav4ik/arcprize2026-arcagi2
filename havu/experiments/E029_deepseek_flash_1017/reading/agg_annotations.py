"""Aggregate the E029 close-reading annotations (reading/*.jsonl). No API calls.
Usage: .venv/bin/python -I agg_annotations.py"""
import json, glob, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
rows, cmp_ = {}, {}
for f in sorted(glob.glob(str(HERE / "*.jsonl"))):
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l)
        k = (r["task"], r["pair"], r["model"])
        (cmp_ if r["model"] == "COMPARE" else rows)[k] = r
sel = {(s["task"], s["pair"]): s for s in json.load(open(HERE / "selection.json"))}
done = sorted({(t, p) for t, p, m in rows if (t, p, "OLD") in rows and (t, p, "NEW") in rows})
print(f"pairs with both traces annotated: {len(done)} / {len(sel)}; missing: {sorted(set(sel) - set(done))}")
G = ["SAVE", "GAIN", "WASTE", "REGRESS"]
num = lambda x: x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
def mean(xs):
    xs = [x for x in xs if x is not None]
    return (st.mean(xs), len(xs)) if xs else (None, 0)
def fmt(v):
    m, n = v
    return "-" if m is None else (f"{m/1000:.1f}k" if abs(m) >= 100 else f"{m:.1f}") + f" (n={n})"

def stats(R):
    out = {}
    out["traces"] = len(R)
    out["correct"] = sum(bool(r.get("correct")) for r in R)
    out["tokens"] = fmt(mean([num(r.get("tokens")) for r in R]))
    out["turn1"] = fmt(mean([num(r.get("turn1_tokens")) for r in R]))
    out["first_solve_tok"] = fmt(mean([num(r.get("first_solve_tok")) for r in R]))
    out["first_solve_never"] = sum(r.get("first_solve_tok") is None for r in R)
    out["train_pass_tok"] = fmt(mean([num(r.get("train_pass_tok")) for r in R]))
    out["train_pass_never"] = sum(r.get("train_pass_tok") is None for r in R)
    out["post_pass_tokens"] = fmt(mean([num((r.get("post_pass") or {}).get("tokens")) for r in R]))
    out["hand_reading"] = fmt(mean([num((r.get("perception") or {}).get("hand_reading_tokens_est")) for r in R]))
    out["perception_mode"] = dict(Counter((r.get("perception") or {}).get("mode") for r in R))
    out["n_rules"] = fmt(mean([num((r.get("hypotheses") or {}).get("n_rules_tried")) for r in R]))
    ip = [num((r.get("hypotheses") or {}).get("idea_tok")) for r in R]
    out["idea_tok"] = fmt(mean(ip))
    out["idea_pct"] = fmt(mean([100 * i / r["tokens"] if i is not None and r.get("tokens") else None for i, r in zip(ip, R)]))
    out["abandon"] = dict(Counter((r.get("hypotheses") or {}).get("abandon_style") for r in R))
    out["checks"] = fmt(mean([num((r.get("verification") or {}).get("n_checks")) for r in R]))
    out["redundant_checks"] = fmt(mean([num((r.get("verification") or {}).get("redundant_checks")) for r in R]))
    out["redundant_tok"] = fmt(mean([num((r.get("verification") or {}).get("redundant_tokens_est")) for r in R]))
    out["loops_present"] = sum(bool((r.get("loops") or {}).get("present")) for r in R)
    out["loop_tok"] = fmt(mean([num((r.get("loops") or {}).get("tokens_est")) or 0 for r in R]))
    fa = Counter()
    for r in R:
        a = (r.get("final_answer") or "").lower()
        fa["after full train pass" if "after full train pass" in a else "failing train pair" if "failing" in a
           else "session fallback" if "fallback" in a else "other"] += 1
    out["final_answer"] = dict(fa)
    w = defaultdict(lambda: [0, 0])
    for r in R:
        seen = set()
        for x in r.get("waste") or []:
            p = x.get("pattern", "other"); w[p][1] += num(x.get("tokens_est")) or 0
            if p not in seen: w[p][0] += 1; seen.add(p)
    out["waste (traces, mean tok/trace over all traces)"] = {p: f"{c} tr, {t/len(R)/1000:.1f}k" for p, (c, t) in sorted(w.items(), key=lambda kv: -kv[1][1])}
    out["waste_total_mean"] = f"{sum(t for c, t in w.values())/max(1,len(R))/1000:.1f}k"
    return out

def show(title, R):
    print(f"\n## {title}")
    for k, v in stats(R).items(): print(f"  {k}: {v}")

for m in ("OLD", "NEW"):
    show(f"{m} all", [rows[(t, p, m)] for t, p in done])
for g in G:
    for m in ("OLD", "NEW"):
        R = [rows[(t, p, m)] for t, p in done if sel[(t, p)]["group"] == g]
        if R: show(f"{m} {g}", R)
show("NEW correct only", [rows[(t, p, "NEW")] for t, p in done if rows[(t, p, "NEW")].get("correct")])
show("OLD correct only", [rows[(t, p, "OLD")] for t, p in done if rows[(t, p, "OLD")].get("correct")])
print("\n## failure causes")
for t, p in done:
    for m in ("OLD", "NEW"):
        r = rows[(t, p, m)]
        if not r.get("correct"): print(f"  {m} {t}_p{p} [{sel[(t,p)]['group']}]: {r.get('failure_cause')}")
