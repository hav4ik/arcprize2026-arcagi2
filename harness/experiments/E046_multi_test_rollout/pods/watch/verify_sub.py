"""Check a submission.json against the challenges: every task present, one row per test input, attempt_1/attempt_2 grids."""
import json, sys
sub, ch = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = []
for t, task in ch.items():
    rows = sub.get(t)
    if not isinstance(rows, list) or len(rows) != len(task["test"]):
        bad.append((t, "rows", None if rows is None else len(rows), len(task["test"]))); continue
    for i, r in enumerate(rows):
        for k in ("attempt_1", "attempt_2"):
            g = r.get(k) if isinstance(r, dict) else None
            if not (isinstance(g, list) and g and all(isinstance(x, list) and all(isinstance(v, int) for v in x) for x in g)):
                bad.append((t, i, k))
extra = set(sub) - set(ch)
print(f"tasks {len(sub)} (challenges {len(ch)}), extra {len(extra)}, problems {len(bad)}: {bad[:5]}")
