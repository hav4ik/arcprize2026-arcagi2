"""Validate submission.json against the challenge file, then cross-check it against the solver's attempts.jsonl.
Usage: validate.py <challenges.json> <submission.json> [<dsv41_logs dir>]
Format: JSON loads; same task ids; one row per test input; rows are exactly {attempt_1, attempt_2}; grids are lists of
1-30 equal-length lists of 1-30 ints (not bools) 0-9.
Semantics (fake server answers are fliplr / flipud / oracle-fliplr of the pair's own test input): each attempt must be
the test input (fallback), fliplr(input) or flipud(input) of THAT row; recomputed from attempts.jsonl with
train_check.ranked_answers + split_multi, the submission must match exactly."""
import json
import sys
from collections import Counter


def ok_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


def main():
    chal = json.load(open(sys.argv[1]))
    raw = open(sys.argv[2]).read()
    sub = json.loads(raw)
    errs = []
    if not isinstance(sub, dict):
        print("FAIL: not a dict")
        return 1
    missing, extra = set(chal) - set(sub), set(sub) - set(chal)
    if missing or extra:
        errs.append(f"task ids: {len(missing)} missing {sorted(missing)[:5]}, {len(extra)} extra {sorted(extra)[:5]}")
    kinds = Counter()
    misroute = []
    for t, task in chal.items():
        rows = sub.get(t)
        if not isinstance(rows, list) or len(rows) != len(task["test"]):
            errs.append(f"{t}: {len(rows) if isinstance(rows, list) else type(rows).__name__} rows for "
                        f"{len(task['test'])} test inputs")
            continue
        for q, (row, x) in enumerate(zip(rows, task["test"])):
            if not isinstance(row, dict) or set(row) != {"attempt_1", "attempt_2"}:
                errs.append(f"{t}[{q}]: keys {sorted(row) if isinstance(row, dict) else type(row).__name__}")
                continue
            for a in ("attempt_1", "attempt_2"):
                g = row[a]
                if not ok_grid(g):
                    errs.append(f"{t}[{q}].{a}: invalid grid {str(g)[:80]}")
                    continue
                inp = x["input"]
                lr = [list(r)[::-1] for r in inp]
                ud = [list(r) for r in inp[::-1]]
                if g == inp:
                    kinds[f"{a}:fallback"] += 1
                elif g == lr:
                    kinds[f"{a}:fliplr"] += 1
                elif g == ud:
                    kinds[f"{a}:flipud"] += 1
                else:
                    others = [p for p, y in enumerate(task["test"]) if p != q and g in (
                        y["input"], [list(r)[::-1] for r in y["input"]], [list(r) for r in y["input"][::-1]])]
                    misroute.append(f"{t}[{q}].{a}: not derived from this row's input (matches rows {others})")
    rows_total = sum(len(v["test"]) for v in chal.values())
    print(f"format: {len(chal)} tasks, {rows_total} rows; errors: {len(errs)}")
    for e in errs[:20]:
        print("  ERR", e)
    print("answer kinds:", dict(sorted(kinds.items())))
    print(f"misrouted / unexplained grids: {len(misroute)}")
    for e in misroute[:20]:
        print("  MIS", e)
    bad = bool(errs or misroute)
    if len(sys.argv) > 3:
        sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else ".")
        recs = {}
        n_lines = 0
        stops = Counter()
        errors = Counter()
        for line in open(f"{sys.argv[3]}/attempts.jsonl"):
            r = json.loads(line)
            n_lines += 1
            stops[r.get("stop_reason")] += 1
            if r.get("error"):
                errors[r["error"].split(":")[0][:60]] += 1
            n = r.get("multi_test")
            if n:
                preds = r["pred"] if isinstance(r.get("pred"), list) and len(r["pred"]) == n else [None] * n
                parts = [dict(r, pair=q, pred=preds[q]) for q in range(n)]
            else:
                parts = [r]
            for x in parts:
                if ok_grid(x.get("pred")):
                    recs.setdefault((x["task_id"], x["pair"]), []).append(x)
        diff = []
        answered = 0
        for t, task in chal.items():
            for q, x in enumerate(task["test"]):
                rr = recs.get((t, q), [])
                order = sorted(rr, key=lambda r: (r.get("train_pass") is not True, r["attempt"]))
                preds = []
                for r in order:
                    if r["pred"] not in preds:
                        preds.append(r["pred"])
                    if len(preds) == 2:
                        break
                answered += bool(preds)
                exp = {"attempt_1": preds[0] if preds else x["input"],
                       "attempt_2": preds[1] if len(preds) > 1 else x["input"]}
                if sub.get(t) and q < len(sub[t]) and sub[t][q] != exp:
                    diff.append(f"{t}[{q}]")
        print(f"attempts.jsonl: {n_lines} records; stop reasons {dict(stops)}")
        print(f"errors: {dict(errors)}")
        print(f"rows with a model answer: {answered}/{rows_total}; rows differing from the attempts.jsonl recompute: "
              f"{len(diff)} {diff[:10]}")
        bad = bad or bool(diff)
    print("RESULT:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
