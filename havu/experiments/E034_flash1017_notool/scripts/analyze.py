"""E034 analysis: no-tool ARC Prize replica on Flash-1017 vs E001d (old Flash), attempts 0 and 1.

ARC Prize scoring: a task's score = fraction of its test pairs solved (by attempt 0 for pass@1, by either of
attempts 0/1 for pass@2), averaged over the 120 tasks. Per (task, pair, attempt) the last non-error record is
used; missing or errored attempts count as wrong.

    .venv/bin/python experiments/E034_flash1017_notool/scripts/analyze.py [--peak-after ISO_UTC]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"
NEW = ROOT / "evals/results/arcprize_deepseek-flash_max_e034_flash1017.jsonl"
BASE = ROOT / "evals/results/arcprize_deepseek-flash_max.jsonl"
OUT_PRICE, IN_PRICE = 0.60, 0.14  # $/M, Flash off-peak list price (peak = 2x)


def load(path):
    recs = [json.loads(l) for l in open(path)]
    best = {}
    for d in recs:
        if d["attempt"] > 1:
            continue
        k = (d["task_id"], d["pair"], d["attempt"])
        if "error" not in d or k not in best:
            best[k] = d
    return recs, best


def task_scores(best, npairs, attempts):
    return {t: np.mean([any(best.get((t, p, a), {}).get("correct", False) for a in attempts)
                        for p in range(n)]) for t, n in npairs.items()}


def pair_rate(best, npairs, attempts):
    return np.mean([any(best.get((t, p, a), {}).get("correct", False) for a in attempts)
                    for t, n in npairs.items() for p in range(n)])


def metrics(best, npairs):
    s = {"pass@1 (a0)": task_scores(best, npairs, [0]),
         "pass@1 (a1)": task_scores(best, npairs, [1]),
         "pass@2": task_scores(best, npairs, [0, 1])}
    s["pass@1 (mean a0/a1)"] = {t: (s["pass@1 (a0)"][t] + s["pass@1 (a1)"][t]) / 2 for t in npairs}
    return s


def boot(delta, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(list(delta.values()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tslog", default=None, help="timestamped copy of the E034 log ('<ISO UTC> [i/n] t.p#a ...')"
                    " to price each call at 2x if it completed in a weekday peak window (01-04, 06-10 UTC)")
    args = ap.parse_args()
    npairs = {p.stem: len(json.load(open(p))["test"]) for p in sorted(TASKS.glob("*.json"))}
    out = {}
    for name, path in [("E034 (1017)", NEW), ("E001d (old)", BASE)]:
        recs, best = load(path)
        ok = [d for d in best.values() if "error" not in d]
        m = metrics(best, npairs)
        print(f"\n== {name}: {path.name}")
        print(f"  records {len(recs)} (attempts 0/1 keys {len(best)}/{2 * sum(npairs.values())}), "
              f"errored records {sum('error' in d for d in recs if d['attempt'] < 2)}, "
              f"keys still errored/missing {2 * sum(npairs.values()) - len(ok)}")
        for k, v in m.items():
            print(f"  {k:22s} task-weighted {np.mean(list(v.values())):6.1%}")
        print(f"  pair-weighted: pass@1 a0 {pair_rate(best, npairs, [0]):.1%}, "
              f"mean a0/a1 {(pair_rate(best, npairs, [0]) + pair_rate(best, npairs, [1])) / 2:.1%}, "
              f"pass@2 {pair_rate(best, npairs, [0, 1]):.1%}")
        toks = np.array([d["usage"]["completion_tokens"] for d in ok if d.get("usage")])
        ptoks = np.array([d["usage"].get("prompt_tokens", 0) for d in ok if d.get("usage")])
        print(f"  out tokens/call: mean {toks.mean():,.0f} median {np.median(toks):,.0f} "
              f"p90 {np.percentile(toks, 90):,.0f} max {toks.max():,}  (n={len(toks)})")
        print(f"  finish_reason: {dict(Counter(d.get('finish_reason') for d in ok))}, "
              f"unparsed {sum(d.get('pred') is None for d in ok)}")
        errs = Counter(d["error"].split(":")[0] for d in recs if "error" in d)
        print(f"  errors (all records): {dict(errs)}")
        # cost of all billed successful records (errored calls may also have been billed partially; unknown)
        allok = [d for d in recs if "error" not in d and d.get("usage") and d["attempt"] < 2]
        c_out = sum(d["usage"]["completion_tokens"] for d in allok) / 1e6 * OUT_PRICE
        c_in = sum(d["usage"].get("prompt_tokens", 0) for d in allok) / 1e6 * IN_PRICE
        print(f"  cost (off-peak list, attempts 0/1): out ${c_out:.2f} + in ${c_in:.2f} = ${c_out + c_in:.2f}")
        if args.tslog and name.startswith("E034"):
            import datetime as dt
            peak = {}
            for l in open(args.tslog):
                parts = l.split()
                if len(parts) > 2 and parts[1].startswith("[") and "#" in parts[2]:
                    ts = dt.datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                    tp, a = parts[2].rsplit("#", 1)
                    t, pr = tp.rsplit(".", 1)
                    peak[(t, int(pr), int(a))] = ts.weekday() < 5 and (1 <= ts.hour < 4 or 6 <= ts.hour < 10)
            mult = [2 if peak.get((d["task_id"], d["pair"], d["attempt"])) else 1 for d in allok]
            c_pk = sum(m_ * (d["usage"]["completion_tokens"] * OUT_PRICE + d["usage"].get("prompt_tokens", 0)
                             * IN_PRICE) / 1e6 for m_, d in zip(mult, allok))
            print(f"  cost with peak pricing by completion time: ${c_pk:.2f} ({sum(m_ == 2 for m_ in mult)} calls at peak)")
        out[name] = m
    print("\n== Paired E034 - E001d, bootstrap 95% CI over 120 tasks")
    for k in ["pass@2", "pass@1 (a0)", "pass@1 (mean a0/a1)"]:
        a, b = out["E034 (1017)"][k], out["E001d (old)"][k]
        mu, lo, hi = boot({t: a[t] - b[t] for t in npairs})
        wins = sum(a[t] > b[t] for t in npairs)
        loss = sum(a[t] < b[t] for t in npairs)
        print(f"  d{k:22s} {mu * 100:+5.1f} pp  CI [{lo * 100:+.1f}, {hi * 100:+.1f}]  tasks better {wins} / worse {loss}")


if __name__ == "__main__":
    main()
