"""Score a single-answer (pass@1) candidate built from several tool-use result files.

Per test pair, the candidate's answer is chosen without looking at the truth:
1. If any code-mode attempt had a submission accepted (transform passed every train pair), take the
   majority prediction among accepted submissions (ties -> earliest file in the list).
2. Otherwise, majority vote over all other parsed predictions (tool attempts' final grids, plus
   --fallback records such as E001d attempt 0), ties -> earliest source.
pass@2 uses the top-2 distinct answers of the same ranking. Scoring is ARC Prize task-weighted.

    .venv/bin/python evals/candidate_pass1.py results/a.jsonl results/b.jsonl \
        [--fallback results/arcprize_deepseek-flash_max.jsonl --fallback-attempts 0]
"""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def as_grid(pred):
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return None
    return pred if isinstance(pred, list) and pred and all(isinstance(r, list) for r in pred) else None


def output_tokens(rec):
    usage = rec.get("usage") or {}
    if isinstance(usage, str):
        usage = ast.literal_eval(usage)
    return usage.get("completion", usage.get("completion_tokens", 0)) or 0


def rank(cands):
    """cands: list of (priority, grid). Majority first, then earliest priority."""
    votes = defaultdict(lambda: [0, 10**9, None])
    for prio, grid in cands:
        key = json.dumps(grid)
        v = votes[key]
        v[0] += 1
        v[1] = min(v[1], prio)
        v[2] = grid
    return [v[2] for v in sorted(votes.values(), key=lambda v: (-v[0], v[1]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fallback", action="append", default=[])
    ap.add_argument("--fallback-attempts", default="0", help="attempt ids to use from fallback files")
    ap.add_argument("--tasks", help="restrict scoring to these comma-separated task ids")
    args = ap.parse_args()
    tasks = {p.stem: json.loads(p.read_text()) for p in sorted(TASKS.glob("*.json"))}
    if args.tasks:
        tasks = {t: tasks[t] for t in args.tasks.split(",")}

    accepted, others = defaultdict(list), defaultdict(list)
    out_tok, cost, n_calls = 0, 0.0, 0
    prio = 0
    for f in args.files:
        for r in load(f):
            prio += 1
            if r["task_id"] not in tasks:
                continue
            n_calls += 1
            out_tok += output_tokens(r)
            cost += r.get("est_cost_usd", 0)
            grid = as_grid(r.get("pred"))
            if "error" in r or grid is None:
                continue
            key = (r["task_id"], int(r["pair"]))
            (accepted if r.get("stop_reason") == "submitted" else others)[key].append((prio, grid))
    fb_attempts = set(args.fallback_attempts.split(","))
    for f in args.fallback:
        for r in load(f):
            prio += 1
            grid = as_grid(r.get("pred"))
            if r["task_id"] in tasks and str(r["attempt"]) in fb_attempts and grid is not None and "error" not in r:
                others[(r["task_id"], int(r["pair"]))].append((10**6 + prio, grid))

    p1 = p2 = 0.0
    n_acc = acc_right = n_pairs = n_none = 0
    for t, task in tasks.items():
        s1 = s2 = 0
        for p, pair in enumerate(task["test"]):
            n_pairs += 1
            truth = pair["output"]
            ranked = rank(accepted[(t, p)]) + [g for g in rank(others[(t, p)])]
            dedup = []
            for g in ranked:
                if g not in dedup:
                    dedup.append(g)
            if accepted[(t, p)]:
                n_acc += 1
                acc_right += dedup[0] == truth
            if not dedup:
                n_none += 1
            s1 += bool(dedup) and dedup[0] == truth
            s2 += truth in dedup[:2]
        p1 += s1 / len(task["test"])
        p2 += s2 / len(task["test"])
    n = len(tasks)
    print(f"tasks {n}, pairs {n_pairs}: pass@1 {p1 / n:.1%}  pass@2 {p2 / n:.1%}")
    print(f"pairs with an accepted submission: {n_acc} (top answer correct {acc_right}); no answer: {n_none}")
    print(f"tool calls counted: {n_calls}, output tokens {out_tok / 1e6:.2f}M "
          f"({out_tok / max(n_calls, 1) / 1e3:.1f}k/call), est. cost ${cost:.2f}")


if __name__ == "__main__":
    main()
