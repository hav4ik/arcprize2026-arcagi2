#!/usr/bin/env python3
"""Pooled per-pair pass@1 / pass@2 of one config over several runs' attempts (E046 E128).
Usage: pooled.py <baseline.jsonl> <run.jsonl>:<attempts> [<run.jsonl>:<attempts> ...]   e.g. r3.jsonl:0,1,2,3 r4.jsonl:0,1
pass@2 is reported (a) per disjoint attempt pair (consecutive attempts within one run), and (b) as the unbiased
estimator 1 - C(n-c,2)/C(n,2) over all n attempts of a pair. Errored records count as wrong (and are listed)."""
import json, math, statistics, sys
def load(p):
    d = {}
    for r in map(json.loads, open(p)):
        if "task_id" in r:
            d[r["task_id"], r["pair"], r["attempt"]] = r
    return d
base = load(sys.argv[1])
pairs = sorted({(t, p) for t, p, a in base})
runs = []
for spec in sys.argv[2:]:
    path, atts = spec.rsplit(":", 1)
    runs.append((path, load(path), [int(x) for x in atts.split(",")]))
pairs = [k for k in pairs if all((k[0], k[1], a) in d for _, d, atts in runs for a in atts)]
ok = lambda r: bool(r.get("correct")) and not r.get("error")
n_err = sum(1 for _, d, atts in runs for k in pairs for a in atts if d[k[0], k[1], a].get("error"))
outcomes = {k: [ok(d[k[0], k[1], a]) for _, d, atts in runs for a in atts] for k in pairs}
p1 = statistics.mean(statistics.mean(v) for v in outcomes.values())
dis = []
for path, d, atts in runs:
    for i in range(0, len(atts) - 1, 2):
        dis.append(statistics.mean(ok(d[t, p, atts[i]]) or ok(d[t, p, atts[i + 1]]) for t, p in pairs))
def unb(v):
    n, c = len(v), sum(v)
    return 1 - math.comb(n - c, 2) / math.comb(n, 2)
p2u = statistics.mean(unb(v) for v in outcomes.values())
n_att = len(next(iter(outcomes.values())))
bp1 = statistics.mean(statistics.mean(ok(base[t, p, a]) for a in (0, 1)) for t, p in pairs)
bp2 = statistics.mean(any(ok(base[t, p, a]) for a in (0, 1)) for t, p in pairs)
print(f"{len(pairs)} pairs, {n_att} attempts each ({n_err} errored pair-records counted wrong)")
print(f"pooled multi: pass@1 {100*p1:.1f}% | pass@2 per disjoint pair {', '.join(f'{100*x:.1f}%' for x in dis)} "
      f"(mean {100*statistics.mean(dis):.1f}%) | unbiased pass@2 over {n_att} attempts {100*p2u:.1f}%")
print(f"per-pair baseline (2 attempts): pass@1 {100*bp1:.1f}%, pass@2 {100*bp2:.1f}%")
