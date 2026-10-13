"""E020: per-arm stats and paired differences on common pairs (attempt 0 unless --attempt), shards merged.

    .venv/bin/python experiments/E020_qwen_harness_80/compare_arms.py \
        x96p='evals/results/qwen_tools_e020_x96p_shard*.jsonl' x96pc8='evals/results/qwen_tools_e020_x96pc8_shard*.jsonl'

The first arm is the reference. Reference arms may also be no-tool files (correct / tokens fields are read the same
way). Paired CI: 95% cluster bootstrap over tasks, pair-weighted (Kaggle) and task-weighted (ARC Prize).
"""
import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_sim import paired_ci, tokens  # noqa: E402


def load(pattern, attempt):
    out = {}
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if int(r["attempt"]) == attempt and not ("error" in r and r.get("pred") is None):
                    out[(r["task_id"], int(r["pair"]))] = r
    return out


def tw(vals):
    tasks = {}
    for (t, _), v in vals.items():
        tasks.setdefault(t, []).append(v)
    return statistics.mean(statistics.mean(v) for v in tasks.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=glob")
    ap.add_argument("--attempt", type=int, default=0)
    args = ap.parse_args()
    arms = {}
    for spec in args.arms:
        name, pat = spec.split("=", 1)
        arms[name] = load(pat, args.attempt)
    keys = sorted(set.intersection(*(set(v) for v in arms.values())))
    print(f"{len(keys)} common pairs, {len({k[0] for k in keys})} tasks (attempt {args.attempt})\n")
    print("| arm | pass@1 pair-w | pass@1 task-w | tokens mean / median | correct-median tok | cap stops | "
          "train-pass rate | P(correct\\|tp) | think nudges / forced per attempt |")
    print("|---|---|---|---|---|---|---|---|---|")
    corr = {}
    for n, recs in arms.items():
        rs = [recs[k] for k in keys]
        corr[n] = {k: float(bool(recs[k]["correct"])) for k in keys}
        toks = [tokens(r) for r in rs]
        ct = [tokens(r) for r in rs if r["correct"]]
        stops = Counter(r.get("stop_reason") for r in rs)
        caps = stops["output_budget"] + stops["context"] + stops["unclosed_think"]
        tp = [r for r in rs if r.get("train_pass")]
        nud = [r.get("think_nudges", 0) for r in rs]
        frc = [r.get("think_forced", 0) for r in rs]
        print(f"| {n} | {statistics.mean(corr[n].values()):.1%} | {tw(corr[n]):.1%} | "
              f"{statistics.mean(toks) / 1000:.1f}k / {statistics.median(toks) / 1000:.1f}k | "
              f"{(statistics.median(ct) / 1000 if ct else 0):.1f}k | {caps / len(rs):.0%} | "
              f"{len(tp) / len(rs):.0%} | {(sum(r['correct'] for r in tp) / len(tp) if tp else float('nan')):.0%} | "
              f"{statistics.mean(nud):.1f} / {statistics.mean(frc):.2f} |")
    names = list(arms)
    print("\n| B - A | Δ pass@1 pair-w [95% CI] | Δ pass@1 task-w [95% CI] | B wins / A wins |")
    print("|---|---|---|---|")
    a = names[0]
    for b in names[1:]:
        d, lo, hi = paired_ci(corr[a], corr[b])
        dt, lot, hit = paired_ci(corr[a], corr[b], task_weighted=True)
        wins = sum(corr[b][k] > corr[a][k] for k in keys), sum(corr[a][k] > corr[b][k] for k in keys)
        print(f"| {b} - {a} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
              f"{100 * dt:+.1f} [{100 * lot:+.1f}, {100 * hit:+.1f}] | {wins[0]} / {wins[1]} |")


if __name__ == "__main__":
    main()
