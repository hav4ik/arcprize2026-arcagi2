"""E062 L4: health and accuracy of finished attempts in replica runs (attempts.jsonl + the workload's solutions).

Usage: attempt_stats.py RUN_DIR[=label] ... [--upto H] [--solutions F]
Per run, attempts finished by pod hour H (t_end_h): count, stop reasons, train_pass rate, token_exact totals
(turns / breaks / rerenders / resyncs / parse_fallbacks), rescues, auto-repairs, mean output tokens and turns, and
per-pair accuracy of the first attempts (attempt 0) against the solutions.
"""
import argparse
import collections
import json
from pathlib import Path

DEF_SOL = "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--upto", type=float)
    ap.add_argument("--solutions", default=DEF_SOL)
    a = ap.parse_args()
    sol = json.load(open(a.solutions))
    for x in a.runs:
        path, _, label = x.partition("=")
        f = Path(path) / "out/dsv41_logs/attempts.jsonl"
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if a.upto is not None:
            rows = [r for r in rows if r.get("t_end_h", 0) <= a.upto]
        te = collections.Counter()
        for r in rows:
            for k, v in (r.get("token_exact") or {}).items():
                te[k] += v
        a0 = [r for r in rows if r["attempt"] == 0]
        ok = sum(r.get("pred") == sol[r["task_id"]][r["pair"]] for r in a0)
        n = max(1, len(rows))
        print(f"{label or Path(path).name}: {len(rows)} attempt records (first attempts {len(a0)}, correct {ok} = "
              f"{ok / max(1, len(a0)):.1%}); stop {dict(collections.Counter(r['stop_reason'] for r in rows))}; "
              f"train_pass {sum(bool(r.get('train_pass')) for r in rows) / n:.1%}; token_exact {dict(te)}; "
              f"rescues {sum(r.get('rescues', 0) for r in rows)}; auto_repair applied "
              f"{sum((r.get('auto_repairs') or {}).get('applied', 0) for r in rows)}; "
              f"mean out {sum(r['usage']['completion'] for r in rows) / n / 1000:.1f}k, turns "
              f"{sum(r['turns'] for r in rows) / n:.1f}")


if __name__ == "__main__":
    main()
