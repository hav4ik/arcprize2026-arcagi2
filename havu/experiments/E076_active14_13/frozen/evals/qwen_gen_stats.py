"""Stats for evals/run_qwen_generate.py results, and a paired comparison with Flash E001d.

Scoring is ARC Prize task-weighted (as evals/sample_stats.py): a task's score is the fraction of its pairs solved,
averaged over tasks. pass@1 = mean over attempt indices; pass@2 = per pair, either of attempts 0/1 correct.
Flash E001d has 3 attempts: pass@1 is the mean over all 3; pass@2 is shown for attempts 0+1 (used for the\npaired breakdown) and as the mean over the 3 attempt pairs (as evals/sample_stats.py).

    .venv/bin/python evals/qwen_gen_stats.py evals/results/qwen_nvfp4_gen_medium_48k.jsonl [--pilot]
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {p.stem: json.loads(p.read_text())
         for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}
FLASH = ROOT / "evals/results/arcprize_deepseek-flash_max.jsonl"


def load(path):
    latest = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return latest


def ok(r):
    if r is None or "error" in r:
        return False
    pred = r.get("pred")
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return False
    return pred == TASKS[r["task_id"]]["test"][int(r["pair"])]["output"]


def score(solved, tasks):
    """solved: set of (task, pair). Task-weighted."""
    return statistics.mean(sum((t, p) in solved for p in range(len(TASKS[t]["test"]))) / len(TASKS[t]["test"])
                           for t in tasks)


def pairs_of(tasks):
    return [(t, p) for t in tasks for p in range(len(TASKS[t]["test"]))]


def solved_by(recs, attempts, tasks):
    return {k for k in pairs_of(tasks) if any(ok(recs.get((*k, a))) for a in attempts)}


def report(name, recs, tasks, attempts, p2_attempts):
    p1 = [score(solved_by(recs, [a], tasks), tasks) for a in attempts]
    p2 = score(solved_by(recs, p2_attempts, tasks), tasks)
    print(f"{name}: pass@1 {statistics.mean(p1):.1%} (per attempt {', '.join(f'{x:.1%}' for x in p1)}), "
          f"pass@2 {p2:.1%}")
    return solved_by(recs, p2_attempts, tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--pilot", action="store_true", help="first 15 tasks only")
    args = ap.parse_args()
    q = load(args.file)
    tasks = sorted(TASKS)[:15] if args.pilot else sorted(TASKS)
    pairs = pairs_of(tasks)
    rs = [r for (t, p, a), r in q.items() if t in tasks]
    attempts = sorted({a for (_, _, a) in q})
    done = {(t, p) for (t, p, a) in q if t in tasks}
    print(f"{len(tasks)} tasks / {len(pairs)} pairs; records {len(rs)} (errors {sum('error' in r for r in rs)}); "
          f"pairs with records {len(done)}")
    if not rs:
        return
    good = [r for r in rs if "error" not in r]
    toks = [r["completion_tokens"] for r in good]
    forced = [r for r in good if r["forced_final"]]
    print(f"  answered {sum(r['answered'] for r in good) / len(good):.1%}; correct attempts "
          f"{sum(ok(r) for r in good)}/{len(good)}")
    print(f"  completion tokens mean {statistics.mean(toks) / 1e3:.1f}k, median {statistics.median(toks) / 1e3:.1f}k")
    print(f"  attempts with EOS-continuations {sum(r['continuations'] > 0 for r in good) / len(good):.1%} "
          f"(total continuations {sum(r['continuations'] for r in good)}); closed </think> naturally "
          f"{sum(r['closed_think'] for r in good) / len(good):.1%}")
    print(f"  forced final {len(forced) / len(good):.1%} ({len(forced)}; reasons "
          f"{sorted({r['force_reason'] for r in forced})}), accuracy when forced "
          f"{(sum(ok(r) for r in forced) / len(forced)) if forced else float('nan'):.1%}, answered when forced "
          f"{(sum(r['answered'] for r in forced) / len(forced)) if forced else float('nan'):.1%}")
    nf = [r for r in good if not r["forced_final"]]
    if nf:
        print(f"  accuracy when not forced {sum(ok(r) for r in nf) / len(nf):.1%} ({len(nf)})")
    print(f"  elapsed mean {statistics.mean(r['elapsed_s'] for r in good) / 60:.1f} min, "
          f"max {max(r['elapsed_s'] for r in good) / 60:.1f} min")

    complete = [t for t in tasks if all((t, p, a) in q for p in range(len(TASKS[t]["test"])) for a in attempts)]
    if len(complete) < len(tasks):
        print(f"  (comparison restricted to {len(complete)} fully finished tasks)")
    f = load(FLASH)
    sq = report("Qwen", q, complete, attempts, attempts[:2])
    sf = report("Flash E001d (pass@2 from attempts 0+1)", f, complete, [0, 1, 2], [0, 1])
    p2s = [score(solved_by(f, ab, complete), complete) for ab in ([0, 1], [0, 2], [1, 2])]
    print(f"  Flash pass@2 mean over the 3 attempt pairs {statistics.mean(p2s):.1%} "
          f"({', '.join(f'{x:.1%}' for x in p2s)})")
    pp = pairs_of(complete)
    both, oq, of = len(sq & sf), len(sq - sf), len(sf - sq)
    print(f"paired pass@2 (pairs): both {both}, only Qwen {oq}, only Flash {of}, neither {len(pp) - both - oq - of}"
          f" / {len(pp)}")
    print(f"  only Qwen: {sorted(sq - sf)}")


if __name__ == "__main__":
    main()
