"""dev56 effort-tier comparison: minimal (and medium) vs xhigh, per model, on the pairs each tier run has finished.

  python3 effort_compare.py [effmin|effmed]
"""
import json, statistics as st, sys
from pathlib import Path
R = Path("/home/vu/Workspace/ARC-AGI-2/evals/results")
tag = sys.argv[1] if len(sys.argv) > 1 else "effmin"
def load(m, t):
    f = R / f"tools_persistent_grid_lean2sub_{m}_low_e032_dev56{t}.jsonl"
    return {(r["task_id"], r["pair"]): r for r in map(json.loads, open(f)) if r.get("attempt", 0) == 0} if f.exists() else {}
ok = lambda r: r.get("correct") is True
tok = lambda r: r["usage"]["completion"]
runs = {(m, t): load(m, t) for m in ("e032", "base") for t in ("", "_" + tag)}
for m in ("e032", "base"):
    x, y = runs[(m, "")], runs[(m, "_" + tag)]
    k = sorted(set(x) & set(y))
    if not k:
        continue
    print(f"{m:5s} {tag} done {len(y):2d}/56 | on those pairs: xhigh {sum(ok(x[i]) for i in k):2d} vs {tag} {sum(ok(y[i]) for i in k):2d}"
          f" | out tok mean xhigh {st.mean(tok(x[i]) for i in k)/1e3:4.1f}k vs {st.mean(tok(y[i]) for i in k)/1e3:4.1f}k"
          f" | cap xhigh {sum(x[i].get('stop_reason') == 'output_budget' for i in k)} vs {sum(y[i].get('stop_reason') == 'output_budget' for i in k)}"
          f" | correct<=29k xhigh {sum(ok(x[i]) and tok(x[i]) <= 29000 for i in k)} vs {sum(ok(y[i]) and tok(y[i]) <= 29000 for i in k)}")
a, b = runs[("e032", "_" + tag)], runs[("base", "_" + tag)]
k = sorted(set(a) & set(b))
if k:
    print(f"{tag} head-to-head on {len(k)}: e032 {sum(ok(a[i]) for i in k)} vs base {sum(ok(b[i]) for i in k)}"
          f" | correct<=29k e032 {sum(ok(a[i]) and tok(a[i]) <= 29000 for i in k)} vs base {sum(ok(b[i]) and tok(b[i]) <= 29000 for i in k)}")
