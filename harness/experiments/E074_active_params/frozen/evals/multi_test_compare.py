#!/usr/bin/env python3
"""E046: one attempt per task (--multi-test) against per-pair attempts on the same multi-pair tasks.

Per pair: pass@1 (mean over attempts) and pass@2 (attempts 0 and 1). Per task: output tokens of one attempt index
(multi: the pair-0 record of the attempt; per-pair: the sum over the task's pairs), and turns.
Usage: multi_test_compare.py <multi.jsonl> <per_pair.jsonl>
"""
import collections
import json
import statistics
import sys


def load(path):
    recs = [r for r in map(json.loads, open(path)) if "task_id" in r]
    latest = {}
    for r in recs:
        latest[(r["task_id"], r["pair"], r["attempt"])] = r
    return latest


multi, base = load(sys.argv[1]), load(sys.argv[2])
tasks = sorted({t for t, p, a in multi if p == 0 and "error" not in multi[t, p, a]})
pairs = sorted({(t, p) for t, p, a in multi if t in tasks})
rows = []
for name, d in (("per-pair (baseline)", base), ("one attempt per task", multi)):
    p1, p2, tok, turns, n_done = [], [], [], [], 0
    for t, p in pairs:
        rs = [d.get((t, p, a)) for a in (0, 1)]
        rs = [r for r in rs if r is not None and "error" not in r]
        if len(rs) < 2:
            continue
        n_done += 1
        p1.append(statistics.mean(bool(r.get("correct")) for r in rs))
        p2.append(any(r.get("correct") for r in rs))
    for t in tasks:
        for a in (0, 1):
            rs = [r for (u, p, b), r in d.items() if u == t and b == a and "error" not in r]
            if rs and (name.startswith("one") or len(rs) == len({p for u, p in pairs if u == t})):
                tok.append(sum((r.get("usage") or {}).get("completion") or 0 for r in rs))
                turns.append(sum(r.get("turns") or 0 for r in rs if not r.get("multi_share")))
    rows.append((name, n_done, p1, p2, tok, turns))
print(f"{len(tasks)} tasks, {len(pairs)} pairs with a finished multi-test attempt")
for name, n, p1, p2, tok, turns in rows:
    print(f"{name:22s} pairs {n:3d}  pass@1 {100 * statistics.mean(p1):5.1f}%  pass@2 {100 * statistics.mean(p2):5.1f}%  "
          f"tokens/task/attempt {statistics.mean(tok) / 1000:5.1f}k (median {statistics.median(tok) / 1000:4.1f}k)  "
          f"turns/task/attempt {statistics.mean(turns):4.1f}")
# Paired per-pair outcomes
both = collections.Counter()
for t, p in pairs:
    a = [base.get((t, p, k)) for k in (0, 1)]
    b = [multi.get((t, p, k)) for k in (0, 1)]
    if all(x is not None and "error" not in x for x in a + b):
        both[(any(x.get("correct") for x in a), any(x.get("correct") for x in b))] += 1
print("pass@2 per pair (baseline, multi):", dict(both))
src = collections.Counter(r.get("answer_source") for (t, p, a), r in multi.items() if p == 0 and t in tasks)
print("multi answer sources:", dict(src))
