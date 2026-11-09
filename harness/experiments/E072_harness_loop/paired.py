#!/usr/bin/env python3
"""paired.py ARM BASE [BASE2 ...] [--metric p2|p1]: paired per-task comparison of score.json files (runs/<label>/score.json).
Task value = mean over the task's pairs of (p2: either attempt right | p1: attempt_1 right); several BASE files are averaged per
task (a pooled baseline). Reports the paired difference in points over the shared tasks with a task-bootstrap 95% CI (10,000
resamples, seed 1), overall and on the dev / holdout halves. Promote only if the overall CI excludes 0 AND the holdout half (or a
second arm run) agrees in sign with its own CI lower bound > 0 / or the point estimate >= half of the overall one."""
import json, random, sys
from pathlib import Path
args = [a for a in sys.argv[1:] if not a.startswith("--")]; metric = "p1" if "--metric" in sys.argv and sys.argv[sys.argv.index("--metric") + 1] == "p1" else "p2"
args = [a for a in args if a not in ("p1", "p2")]
H = Path(__file__).resolve().parent
def load(p):
    p = Path(p); p = p if p.suffix == ".json" else H / "runs" / p / "score.json"; return json.load(open(p))
def vals(sc): return {t: sum(x[metric] for x in ps) / len(ps) for t, ps in sc["per_task"].items()}
arm = vals(load(args[0])); bases = [vals(load(b)) for b in args[1:]]
common = sorted(set(arm) & set.intersection(*[set(b) for b in bases]))
base = {t: sum(b[t] for b in bases) / len(bases) for t in common}
dev = set((H / "workloads/eval120_kaggle_dev_ids.txt").read_text().split()); hold = set((H / "workloads/eval120_kaggle_holdout_ids.txt").read_text().split())
def ci(ts):
    d = [arm[t] - base[t] for t in ts]; rng = random.Random(1); n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return 100 * sum(d) / n, 100 * bs[250], 100 * bs[9749]
for name, ts in (("all", common), ("dev", [t for t in common if t in dev]), ("holdout", [t for t in common if t in hold])):
    m, lo, hi = ci(ts); print(f"{name:8s} n={len(ts):3d} arm {100*sum(arm[t] for t in ts)/len(ts):5.1f} base {100*sum(base[t] for t in ts)/len(ts):5.1f}  diff {m:+5.1f} [{lo:+5.1f}, {hi:+5.1f}] ({metric})")
