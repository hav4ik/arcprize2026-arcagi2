"""E032 minimal vs xhigh (first attempts) on the same pairs: dev56 and the 93-pair set, plus the Kaggle-relevant
correct-within-29k count.  python3 min_vs_xhigh.py"""
import json, statistics as st
from pathlib import Path
R = Path("/home/vu/Workspace/ARC-AGI-2/evals/results")
def load(*tags):
    d = {}
    for t in tags:
        f = R / f"tools_persistent_grid_lean2sub_e032_low_e032_{t}.jsonl"
        if f.exists():
            d.update({(r["task_id"], r["pair"]): r for r in map(json.loads, open(f)) if r.get("attempt", 0) == 0})
    return d
ok = lambda r: r.get("correct") is True
tok = lambda r: r["usage"]["completion"]
tot = [0, 0, 0, 0, 0]
for name, xs, ms, n in (("dev56", ("dev56",), ("dev56_effmin",), 56), ("k93", ("k93a", "k93b"), ("k93_effmin",), 93)):
    x, m = load(*xs), load(*ms)
    k = sorted(set(x) & set(m))
    if not m:
        continue
    cx, cm = sum(ok(x[i]) for i in k), sum(ok(m[i]) for i in k)
    wx, wm = sum(ok(x[i]) and tok(x[i]) <= 29000 for i in k), sum(ok(m[i]) and tok(m[i]) <= 29000 for i in k)
    tot = [tot[0] + len(k), tot[1] + cx, tot[2] + cm, tot[3] + wx, tot[4] + wm]
    print(f"{name:5s} minimal done {len(m):2d}/{n} (correct {sum(map(ok, m.values()))}) | same {len(k)} pairs: xhigh {cx} vs minimal {cm}"
          f" | within 29k: {wx} vs {wm} | out tok mean {st.mean(tok(x[i]) for i in k)/1e3 if k else 0:.1f}k vs {st.mean(tok(m[i]) for i in k)/1e3 if k else 0:.1f}k"
          f" | cap {sum(x[i].get('stop_reason') == 'output_budget' for i in k)} vs {sum(m[i].get('stop_reason') == 'output_budget' for i in k)}")
print(f"TOTAL same {tot[0]} pairs: xhigh {tot[1]} vs minimal {tot[2]} | correct within 29k: xhigh {tot[3]} vs minimal {tot[4]}")
