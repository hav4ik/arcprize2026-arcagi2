"""Offline: how often does test_input fall outside the train inputs' feature ranges, and does that predict a wrong
answer? Features are cheap and LLM-free (candidate content for a 'test novelty report' on submit). No API."""
import json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/E017_reasoning_efficiency"
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"

def label8(m):
    seen, out = np.zeros_like(m, bool), []
    H, W = m.shape
    for y in range(H):
        for x in range(W):
            if m[y, x] and not seen[y, x]:
                st, cells = [(y, x)], []; seen[y, x] = True
                while st:
                    cy, cx = st.pop(); cells.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; st.append((ny, nx))
                out.append(cells)
    return out

def feats(g):
    a = np.array(g); vals, cnt = np.unique(a, return_counts=True); bg = vals[cnt.argmax()]
    comps, touch, sizes = 0, 0, []
    for c in vals:
        if c == bg: continue
        for cells in label8(a == c):
            comps += 1
            ys = [y for y, _ in cells]; xs = [x for _, x in cells]; sizes.append(len(cells))
            touch += bool(min(ys) == 0 or min(xs) == 0 or max(ys) == a.shape[0] - 1 or max(xs) == a.shape[1] - 1)
    return dict(h=a.shape[0], w=a.shape[1], colours=set(vals.tolist()) - {bg}, ncol=len(vals), comps=comps, touch=touch,
                maxsize=max(sizes, default=0), minsize=min(sizes, default=0))

def novelty(task, pair):
    tr = [feats(p["input"]) for p in task["train"]]; te = feats(task["test"][pair]["input"])
    fl = []
    allc = set().union(*[f["colours"] for f in tr])
    if te["colours"] - allc: fl.append("new_colour")
    for k in ("h", "w", "ncol", "comps", "touch", "maxsize", "minsize"):
        lo, hi = min(f[k] for f in tr), max(f[k] for f in tr)
        if te[k] < lo or te[k] > hi: fl.append(f"{k}_out_of_range")
    return fl

rows = [json.loads(l) for l in open(OUT / "attempts_index.jsonl")]
cache, stat, per = {}, collections.Counter(), collections.defaultdict(collections.Counter)
pairs_flag = {}
for r in rows:
    k = (r["task_id"], r["pair"])
    if k not in cache:
        t = json.loads((TASKS / f"{r['task_id']}.json").read_text()); cache[k] = novelty(t, r["pair"])
    fl = cache[k]
    o = r["outcome"]
    stat[(bool(fl), o)] += 1
    for f in fl: per[f][o] += 1
for nov in (True, False):
    n = sum(v for (a, _), v in stat.items() if a == nov)
    print("novel test" if nov else "in-range test", "attempts", n, {o: f"{stat[(nov, o)]/n:.0%}" for o in ("correct", "wrong", "none")})
print("pairs with any novelty flag:", sum(bool(v) for v in cache.values()), "of", len(cache))
for f, v in sorted(per.items(), key=lambda x: -sum(x[1].values())):
    n = sum(v.values()); print(f"  {f}: attempts {n}, correct {v['correct']/n:.0%}")
