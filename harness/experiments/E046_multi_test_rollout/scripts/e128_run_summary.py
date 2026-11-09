#!/usr/bin/env python3
"""E046 E128 multi-test run summary: finished attempts, per-pair accuracy vs the baseline's same (task, pair, attempt),
tokens/turns per task-attempt, answer sources, no-answer causes, token-exact totals.
Usage: analyze.py <multi.jsonl> <baseline.jsonl> [<other multi run for paired comparison>]"""
import collections, gzip, json, statistics, sys
def load(p):
    d = {}
    for r in map(json.loads, open(p)):
        if "task_id" in r:
            d[r["task_id"], r["pair"], r["attempt"]] = r
    return d
m, b = load(sys.argv[1]), load(sys.argv[2])
o = load(sys.argv[3]) if len(sys.argv) > 3 else None
done = sorted({(t, a) for t, p, a in m if p == 0})
mc, bc, oc, mt, bt, mturn, bturn = [], [], [], [], [], [], []
for t, a in done:
    ps = sorted(p for u, p, x in m if u == t and x == a)
    for p in ps:
        mc.append(bool(m[t, p, a].get("correct")))
        bc.append(bool(b[t, p, a].get("correct")))
        if o is not None and (t, p, a) in o:
            oc.append((bool(m[t, p, a].get("correct")), bool(o[t, p, a].get("correct"))))
    mt.append(m[t, 0, a]["usage"]["completion"]); mturn.append(m[t, 0, a]["turns"])
    bt.append(sum(b[t, p, a]["usage"]["completion"] for p in ps)); bturn.append(sum(b[t, p, a]["turns"] for p in ps))
print(f"{len(done)} finished task-attempts ({sum(a == 0 for t, a in done)} a0, {sum(a == 1 for t, a in done)} a1), "
      f"{len({t for t, a in done})} tasks")
print(f"per-pair correct: multi {sum(mc)}/{len(mc)} = {100*statistics.mean(mc):.1f}% vs baseline same attempts "
      f"{sum(bc)}/{len(bc)} = {100*statistics.mean(bc):.1f}%")
if oc:
    print(f"  vs other run's same attempts ({len(oc)} pairs): this {100*statistics.mean(x for x, y in oc):.1f}% other "
          f"{100*statistics.mean(y for x, y in oc):.1f}%")
print(f"tokens/task-attempt: multi mean {statistics.mean(mt)/1e3:.1f}k median {statistics.median(mt)/1e3:.1f}k vs baseline "
      f"{statistics.mean(bt)/1e3:.1f}k | turns {statistics.mean(mturn):.1f} vs {statistics.mean(bturn):.1f}")
p0 = [m[t, 0, a] for t, a in done]
print("answer sources:", dict(collections.Counter(r.get("answer_source") for r in p0)),
      "| stop reasons:", dict(collections.Counter(r.get("stop_reason") for r in p0)))
causes = collections.Counter()
for r in p0:
    if r.get("answer_source") in ("session", "session_solve"):
        causes[f"{r['stop_reason']} -> rescued by {r['answer_source']}"] += 1
    if r.get("answer_source"):
        continue
    why = "?"
    try:
        ev = json.load(gzip.open(r["trace_path"]))["events"]
        fb = [e for e in ev if e.get("fallback") == "session_answer"]
        out = fb[0]["output"] if fb else ""
        why = ("answer not set" if ("NameError" in out or "not defined" in out) else
               "answer invalid: " + out.split("got ")[-1][:40] if "not accepted" in out else out[:60].replace("\n", " "))
        fs = [e for e in ev if e.get("fallback") == "session_solve"]
        if fs:
            o2 = fs[0]["output"]
            why += ("; no solve defined" if "name 'solve' is not defined" in o2 else
                    "; fallback solve failed: " + o2.strip().splitlines()[-2][:60] if len(o2.strip().splitlines()) > 1 else "; solve failed")
    except Exception as exc:
        why = f"trace? {type(exc).__name__}"
    causes[f"NO ANSWER: {r['stop_reason']}, {why}"] += 1
for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
    print(f"   {v:3d}  {k}")
te = collections.Counter()
for r in p0:
    for k, v in (r.get("token_exact") or {}).items():
        te[k] += v
print("rescues", sum(r.get("rescues") or 0 for r in p0), "| errors", sum(1 for r in p0 if r.get("error")),
      "| train_pass", sum(bool(r.get("train_pass")) for r in p0), "| token_exact", dict(te) or None)
