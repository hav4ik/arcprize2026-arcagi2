#!/usr/bin/env python3
"""E060 scorer: one row per run on the 120 public eval tasks (167 pairs), plus a paired comparison with E048.

Per run (latest record per task/pair/attempt; a later successful record replaces an errored one; several files may
be merged with '+', e.g. run.jsonl+run_retry.jsonl):
- task-weighted pass@1 (Kaggle/ARC Prize metric): per pair, mean correctness over the attempts; per task, mean over
  its pairs; then mean over all 120 tasks. Missing or errored attempts count as wrong.
- pair pass@1 (mean over pairs and attempts), per-attempt task-weighted pass@1;
- pass@2 (task-weighted): top-2 distinct answers ranked by train_pass then attempt order
  (evals/train_check.py:ranked_answers), and oracle pass@k over all attempts;
- output tokens per attempt (mean / median / p90), stop reasons, answer sources, train_pass rate and
  P(correct | train_pass), seconds per attempt (mean / p90), wall time (first to last record).
--vs FILE[+FILE]: paired task-weighted difference (run minus FILE) on the pairs both cover, attempt 0 and all
attempts, with a 95% bootstrap CI over tasks.

    python score_e060.py RUN.jsonl [RUN2.jsonl+RUN2_retry.jsonl ...] [--vs E048.jsonl+E048_retry.jsonl]
"""
import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "evals/train_check.py").exists())  # repo or box copy
sys.path.insert(0, str(ROOT / "evals"))
from train_check import ranked_answers  # noqa: E402

TASK_DIR = ROOT / "data/arc-agi-2-github/data/evaluation"


def load_tasks():
    return {p.stem: json.loads(p.read_text()) for p in sorted(TASK_DIR.glob("*.json"))}


def load(spec):
    """Latest record per (task, pair, attempt) over the '+'-joined files; a success is never replaced by an error."""
    out, times = {}, []
    for f in spec.split("+"):
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r["task_id"], int(r["pair"]), int(r["attempt"]))
            if "error" in r and k in out and "error" not in out[k]:
                continue
            out[k] = r
            if r.get("finished_at"):
                times.append(r["finished_at"])
    return out, times


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def task_weighted(tasks, value):
    """value(task_id, pair) -> float in [0, 1]; mean over pairs per task, then over all tasks."""
    return st.mean(st.mean(value(t, p) for p in range(len(task["test"]))) for t, task in tasks.items())


def summarize(name, recs, tasks):
    ok = {k: r for k, r in recs.items() if "error" not in r}
    atts = sorted({k[2] for k in recs}) or [0]
    good = lambda t, p, a: bool(ok.get((t, p, a), {}).get("correct"))
    p1 = task_weighted(tasks, lambda t, p: st.mean(good(t, p, a) for a in atts))
    per_att = [task_weighted(tasks, lambda t, p, a=a: good(t, p, a)) for a in atts]
    n_pairs = sum(len(task["test"]) for task in tasks.values())
    pair_p1 = st.mean(good(t, p, a) for t, task in tasks.items() for p in range(len(task["test"])) for a in atts)

    def ranked(t, p, k):
        rs = [ok[(t, p, a)] for a in atts if (t, p, a) in ok]
        return any(r.get("correct") for r in ranked_answers(rs, k))
    p2 = task_weighted(tasks, lambda t, p: ranked(t, p, 2)) if len(atts) > 1 else None
    oracle = task_weighted(tasks, lambda t, p: any(good(t, p, a) for a in atts))
    toks = [r["usage"]["completion"] for r in ok.values() if not r.get("multi_share")]
    secs = [r.get("seconds", 0) for r in ok.values() if not r.get("multi_share")]
    stops = Counter(r.get("stop_reason") or "none" for r in ok.values())
    srcs = Counter(r.get("answer_source") or "none" for r in ok.values())
    tp = [r for r in ok.values() if r.get("train_pass") is True]
    tp_known = [r for r in ok.values() if r.get("train_pass") is not None]
    errors = sum("error" in r for r in recs.values())
    return {
        "run": name, "attempts": len(ok), "errors": errors, "expected": n_pairs * len(atts),
        "pass1_tw": p1, "pass1_per_attempt": per_att, "pair_pass1": pair_p1, "pass2_ranked": p2,
        "oracle": oracle, "k": len(atts),
        "tok_mean": st.mean(toks) if toks else 0, "tok_median": st.median(toks) if toks else 0,
        "tok_p90": pct(toks, 0.9) if toks else 0, "sec_mean": st.mean(secs) if secs else 0, "sec_p90": pct(secs, 0.9),
        "stops": dict(stops.most_common()), "sources": dict(srcs.most_common()),
        "train_pass_rate": len(tp) / len(ok) if ok else 0, "train_pass_known": len(tp_known),
        "p_correct_given_tp": (sum(bool(r.get("correct")) for r in tp) / len(tp)) if tp else float("nan"),
        "p_correct_given_not_tp": (sum(bool(r.get("correct")) for r in ok.values() if r.get("train_pass") is not True)
                                   / max(1, sum(r.get("train_pass") is not True for r in ok.values()))),
    }


def bootstrap(diff, n=10000, seed=0):
    rng = random.Random(seed)
    vals = list(diff.values())
    means = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired(recs, base, attempts):
    """Task-weighted difference on the pairs both files cover with every listed attempt successful."""
    ok = lambda R, t, p, a: (t, p, a) in R and "error" not in R[(t, p, a)]
    keys = sorted({(t, p) for (t, p, _) in recs} & {(t, p) for (t, p, _) in base})
    keys = [k for k in keys if all(ok(recs, *k, a) for a in attempts) and all(ok(base, *k, a) for a in attempts)]
    by_task = defaultdict(list)
    for t, p in keys:
        run = st.mean(bool(recs[(t, p, a)].get("correct")) for a in attempts)
        bas = st.mean(bool(base[(t, p, a)].get("correct")) for a in attempts)
        by_task[t].append((run, bas))
    rr = {t: st.mean(x for x, _ in v) for t, v in by_task.items()}
    bb = {t: st.mean(y for _, y in v) for t, v in by_task.items()}
    diff = {t: rr[t] - bb[t] for t in rr}
    lo, hi = bootstrap(diff) if diff else (float("nan"), float("nan"))
    both = Counter((bool(x), bool(y)) for v in by_task.values() for x, y in v) if len(attempts) == 1 else None
    return {"pairs": len(keys), "tasks": len(by_task), "run": st.mean(rr.values()) if rr else float("nan"),
            "base": st.mean(bb.values()) if bb else float("nan"),
            "diff": st.mean(diff.values()) if diff else float("nan"), "ci": (lo, hi), "both": both}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="results jsonl; join a run's files with '+'")
    ap.add_argument("--vs", help="baseline jsonl(s) joined with '+' (e.g. E048 main+retry)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tasks = load_tasks()
    base = load(args.vs)[0] if args.vs else None
    rows = []
    print("| run | k | attempts ok / expected (err) | pass@1 tw | per attempt | pair pass@1 | pass@2 ranked | oracle "
          "| out tok mean / med / p90 | s/attempt mean / p90 | train_pass | P(ok given tp) | stops |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in args.runs:
        recs, _ = load(spec)
        s = summarize(Path(spec.split("+")[0]).stem, recs, tasks)
        rows.append(s)
        print(f"| {s['run']} | {s['k']} | {s['attempts']} / {s['expected']} ({s['errors']}) | {s['pass1_tw']:.1%} | "
              + " / ".join(f"{x:.1%}" for x in s["pass1_per_attempt"]) + f" | {s['pair_pass1']:.1%} | "
              + (f"{s['pass2_ranked']:.1%}" if s["pass2_ranked"] is not None else "-") + f" | {s['oracle']:.1%} | "
              f"{s['tok_mean'] / 1e3:.1f}k / {s['tok_median'] / 1e3:.1f}k / {s['tok_p90'] / 1e3:.1f}k | "
              f"{s['sec_mean']:.0f} / {s['sec_p90']:.0f} | {s['train_pass_rate']:.0%} | {s['p_correct_given_tp']:.0%} | "
              + ", ".join(f"{k} {v}" for k, v in s["stops"].items()) + " |")
        if base is not None:
            for atts in ([0], sorted({k[2] for k in recs})):
                if len(atts) > 1 or atts == [0]:
                    d = paired(recs, base, atts)
                    extra = (f"; pairs both {d['both'].get((True, True), 0)}, only run {d['both'].get((True, False), 0)}, "
                             f"only base {d['both'].get((False, True), 0)}, neither {d['both'].get((False, False), 0)}"
                             if d["both"] else "")
                    print(f"|  paired vs baseline, attempts {atts}: {d['pairs']} pairs / {d['tasks']} tasks: run "
                          f"{d['run']:.1%} vs base {d['base']:.1%}, diff {d['diff'] * 100:+.1f} pp "
                          f"[{d['ci'][0] * 100:+.1f}, {d['ci'][1] * 100:+.1f}]{extra} |")
    if args.json:
        print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
