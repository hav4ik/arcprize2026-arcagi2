"""Offline simulation of a 'restart when no train-passing solve() by X tokens' policy, using the 5 attempts per pair
(3 sub80s40 + 2 sub96n) as the pool of fresh attempts. No API. Optimistic caveats in failure_analysis.md."""
import json, collections, statistics as st
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
rows = json.load(open(OUT / "signatures.json"))
idx = {tuple(r["key"]): r for r in rows}
pairs = collections.defaultdict(list)
for r in rows:
    pairs[(r["key"][1], r["key"][2])].append(r)
tasks = collections.defaultdict(set)
for r in rows:
    tasks[r["key"][1]].add(r["key"][2])
CAP = 80000

def score(fn):
    """task-weighted pass@1 and mean tokens of policy fn(attempt)->(p_correct, tokens)."""
    per_task = collections.defaultdict(list); toks = []
    for r in rows:
        p, t = fn(r)
        per_task[r["key"][1]].append((r["key"][2], p)); toks.append(t)
    acc = st.mean(st.mean(p for _, p in v) for v in per_task.values())
    return acc, st.mean(toks)

def base(r):
    return (r["outcome"] == "correct") * 1.0, r["tokens"]

def restart(X, need="full"):
    def fn(r):
        passed = r["first_full"] is not None and r["first_full"] <= X
        if r["tokens"] <= X or passed:
            return base(r)
        others = [o for o in pairs[(r["key"][1], r["key"][2])] if o["key"] != r["key"]]
        rem = CAP - X
        ps = [(o["outcome"] == "correct" and o["tokens"] <= rem) for o in others]
        ts = [X + min(o["tokens"], rem) for o in others]
        return st.mean(ps), st.mean(ts)
    return fn

a, t = score(base)
print(f"baseline (as run): pass@1 {a:.3f}  tok {t/1e3:.1f}k")
for X in (20000, 25000, 30000, 35000, 40000, 50000):
    a2, t2 = score(restart(X))
    print(f"restart at {X//1000}k if no probed train-passing fn (fresh attempt gets {CAP-X} tokens): pass@1 {a2:.3f} ({a2-a:+.3f})  tok {t2/1e3:.1f}k ({t2/t-1:+.1%})")
