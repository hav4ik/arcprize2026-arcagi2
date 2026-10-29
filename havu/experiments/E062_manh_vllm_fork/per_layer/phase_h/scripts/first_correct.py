"""First-attempt pairs done / correct in a run's attempts.jsonl (multi-test pred lists handled). Usage: first_correct.py <attempts.jsonl> <solutions.json>"""
import collections, json, sys
sol = json.load(open(sys.argv[2])); n = ok = 0; stops = collections.Counter(); tasks = set()
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("attempt") != 0 or r.get("error"):
        continue
    t, p = r["task_id"], int(r.get("pair", 0)); s = sol.get(t)
    if s is None:
        continue
    pr = r.get("pred"); n += 1; tasks.add(t)
    ok += (len(s) > 1 and isinstance(pr, list) and len(pr) > p and pr[p] == s[p]) or (len(s) == 1 and pr == s[0])
    if p == 0: stops[r.get("stop_reason")] += 1
print(f"first-attempt pairs done {n} correct {ok} ({ok / max(1, n):.3f}); tasks {len(tasks)}; stops {dict(stops)}")
