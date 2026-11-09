"""Per-sample pass@1 statistics for runs with several independent attempts per pair.

For each results file: pass@1 (ARC Prize task-weighted) of each attempt index, mean and std across attempts,
mean output tokens per attempt, pass@2 over attempt pairs, and the pass@1 of a majority vote over the first k
attempts (ties -> earliest attempt; pairs with no parsed answer count as wrong).

    .venv/bin/python evals/sample_stats.py evals/results/<file>.jsonl [...]
    .venv/bin/python evals/sample_stats.py --e001d     # the no-tool baseline, same metrics
"""
import argparse
import ast
import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}


def tokens(r):
    u = r.get("usage") or {}
    if isinstance(u, str):
        u = ast.literal_eval(u)
    return u.get("completion", u.get("completion_tokens", 0)) or 0


def score(answer):
    """answer: dict (task, pair) -> list of candidate grids (pass@k uses all)."""
    total = 0.0
    for t, task in TASKS.items():
        n = len(task["test"])
        total += sum(task["test"][p]["output"] in answer.get((t, p), []) for p in range(n)) / n
    return total / len(TASKS)


def as_grid(pred):
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return None
    return pred if isinstance(pred, list) else None


def analyze(name, records):
    by_attempt = {}
    for r in records:
        by_attempt.setdefault(int(r["attempt"]), {})[(r["task_id"], int(r["pair"]))] = r
    atts = sorted(by_attempt)
    p1 = []
    for a in atts:
        ans = {k: [g] for k, r in by_attempt[a].items() if "error" not in r and (g := as_grid(r.get("pred"))) is not None}
        p1.append(score(ans))
    ok = [r for a in atts for r in by_attempt[a].values() if "error" not in r]
    errs = sum(1 for a in atts for r in by_attempt[a].values() if "error" in r)
    print(f"\n{name}: attempts {atts}, records {sum(len(v) for v in by_attempt.values())} (errors {errs})")
    print("  pass@1 per attempt: " + ", ".join(f"{x:.1%}" for x in p1)
          + (f"; mean {statistics.mean(p1):.1%} (sd {statistics.stdev(p1):.1%})" if len(p1) > 1 else ""))
    print(f"  output tokens per attempt (successful calls): mean {statistics.mean(tokens(r) for r in ok) / 1e3:.1f}k, "
          f"median {statistics.median(tokens(r) for r in ok) / 1e3:.1f}k")
    if len(atts) >= 2:
        p2 = []
        for a, b in combinations(atts, 2):
            ans = {}
            for x in (a, b):
                for k, r in by_attempt[x].items():
                    g = as_grid(r.get("pred"))
                    if "error" not in r and g is not None:
                        ans.setdefault(k, []).append(g)
            p2.append(score(ans))
        print(f"  pass@2 over attempt pairs: mean {statistics.mean(p2):.1%} (range {min(p2):.1%}-{max(p2):.1%})")
    for k in range(3, len(atts) + 1):
        ans = {}
        keys = {key for a in atts[:k] for key in by_attempt[a]}
        for key in keys:
            votes = Counter()
            first = {}
            for i, a in enumerate(atts[:k]):
                r = by_attempt[a].get(key)
                g = as_grid(r.get("pred")) if r and "error" not in r else None
                if g is not None:
                    s = json.dumps(g)
                    votes[s] += 1
                    first.setdefault(s, i)
            if votes:
                best = sorted(votes, key=lambda s: (-votes[s], first[s]))[0]
                ans[key] = [json.loads(best)]
        print(f"  majority vote over first {k} attempts (pass@1): {score(ans):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--e001d", action="store_true")
    args = ap.parse_args()
    if args.e001d:
        recs = [json.loads(l) for l in open(ROOT / "evals/results/arcprize_deepseek-flash_max.jsonl")]
        analyze("E001d (no tools, max effort)", recs)
    for f in args.files:
        latest = {}
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[(r["task_id"], r["pair"], r["attempt"])] = r
        analyze(Path(f).stem, list(latest.values()))


if __name__ == "__main__":
    main()
