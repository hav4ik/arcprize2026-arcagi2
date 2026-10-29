"""Paired first-attempt accuracy: a notebook run's attempts.jsonl (x2t workload) vs an eval-set results jsonl (attempt 0,
`correct` per pair). Multi-test records of the run (one record, pred = list of grids) are split per pair. x2t ids map to
eval ids by the workload's idmap (`orig`); copies are d8 + colour-permuted versions of the same eval task, so they are
reported separately from the originals. McNemar: exact two-sided binomial on the discordant pairs.
Usage: paired_c1.py <attempts.jsonl> <x2t solutions.json> <x2t idmap.json> <eval results.jsonl> [eval solutions.json]"""
import json, sys
from math import comb

att, sols, idmap, ev = sys.argv[1:5]
sol, imap = json.load(open(sols)), json.load(open(idmap))


def grid(x):
    return json.loads(x) if isinstance(x, str) else x


run = {}  # (x2t id, pair) -> correct
for l in open(att):
    r = json.loads(l)
    if r.get("attempt") != 0 or r.get("error"):
        continue
    t, s = r["task_id"], sol.get(r["task_id"])
    if s is None:
        continue
    pr = grid(r.get("pred"))
    if r.get("multi_test"):
        for p in range(len(s)):
            run[(t, p)] = isinstance(pr, list) and len(pr) > p and pr[p] == s[p]
    else:
        run[(t, r.get("pair", 0))] = pr == s[r.get("pair", 0)]

evr = {}
for l in open(ev):
    r = json.loads(l)
    if r.get("attempt") == 0:
        evr[(r["task_id"], r["pair"])] = bool(r.get("correct"))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def report(name, keys):
    common = [(k, run[k], evr[(imap[k[0]]["orig"], k[1])]) for k in keys if (imap[k[0]]["orig"], k[1]) in evr]
    n = len(common)
    if not n:
        print(f"{name}: no common pairs"); return
    a, e = sum(x for _, x, _ in common), sum(y for _, _, y in common)
    b = sum(1 for _, x, y in common if x and not y)  # run right, eval wrong
    c = sum(1 for _, x, y in common if y and not x)
    tasks = sorted({k[0] for k, _, _ in common})
    print(f"{name}: n={n} pairs ({len(tasks)} tasks); run {a}/{n} = {100 * a / n:.1f}%, eval {e}/{n} = {100 * e / n:.1f}%; "
          f"both {sum(1 for _, x, y in common if x and y)}, neither {sum(1 for _, x, y in common if not x and not y)}, "
          f"run-only {b}, eval-only {c}; McNemar exact p = {mcnemar(b, c):.3f}")
    return [(k, x, y) for k, x, y in common if x != y]


orig = [k for k in run if imap[k[0]]["kind"] == "orig"]
copy = [k for k in run if imap[k[0]]["kind"] != "orig"]
print(f"run attempt-0 pairs scored: {len(run)} ({len(orig)} orig, {len(copy)} copies); eval attempt-0 pairs: {len(evr)}")
d1 = report("originals (same task)", orig)
d2 = report("copies -> eval original", copy)
d3 = report("all", list(run))
for k, x, y in (d3 or []):
    print(f"  discordant {k[0]} ({imap[k[0]]['kind']}, orig {imap[k[0]]['orig']}) pair {k[1]}: run {'OK' if x else 'x'}, eval {'OK' if y else 'x'}")
