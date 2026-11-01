"""Paired comparison of tool-use runs against a multi-sample baseline (default: E004-GL) on the same pairs (E009).

For each results file, restricted to the pairs it covers (latest record per task/pair/attempt, errors dropped):
- pair accuracy and ARC Prize task-weighted pass@1 (mean over attempts), for the run and for the baseline's mean
  over its samples on exactly those pairs;
- the paired difference with a 95% bootstrap CI over tasks (10k resamples): per task, the run's mean success
  rate minus the baseline's mean success rate, task-weighted;
- output tokens per attempt (mean / median) vs the baseline on the same pairs, and the relative change;
- answer sources (submit / session / parse / lenient / none), accuracy per source, and pass@1 with lenient-parse
  answers counted as wrong;
- token distribution of correct rollouts (quartiles), useful for SFT budgeting.

    .venv/bin/python evals/paired_vs_baseline.py evals/results/<run>.jsonl [...] [--baseline <file>]
"""
import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals/results"
BASELINE = RESULTS / "tools_persistent_grid_lean2_deepseek-flash_low_full_a0.jsonl"
TASKS = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}


def latest(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return {k: r for k, r in out.items() if "error" not in r}


def by_pair(recs):
    pairs = defaultdict(list)
    for (t, p, _), r in recs.items():
        pairs[(t, p)].append(r)
    return pairs


def task_rates(pairs, keys, correct=lambda r: bool(r.get("correct"))):
    """Per task: mean over its covered pairs of the per-pair success rate (mean over attempts)."""
    tasks = defaultdict(list)
    for k in keys:
        tasks[k[0]].append(st.mean(correct(r) for r in pairs[k]))
    return {t: st.mean(v) for t, v in tasks.items()}


def tok(r):
    return r["usage"]["completion"]


def per_attempt_task_weighted(recs, keys):
    """ARC Prize pass@1 per attempt index over the covered pairs (task-weighted within the covered tasks)."""
    atts = sorted({a for (_, _, a) in recs})
    out = []
    for a in atts:
        tasks = defaultdict(list)
        for k in keys:
            r = recs.get((k[0], k[1], a))
            if r is not None:
                tasks[k[0]].append(bool(r.get("correct")))
        out.append(st.mean(st.mean(v) for v in tasks.values()) if tasks else float("nan"))
    return atts, out


def task_pass2(recs, keys, pairs_of_attempts=None):
    """Per task: pass@2 (a pair counts if either attempt of an attempt pair is correct), averaged over all
    attempt pairs (or the given ones), then over the task's covered pairs. Missing/errored attempts count wrong."""
    atts = sorted({a for (_, _, a) in recs})
    combos = pairs_of_attempts or [(a, b) for i, a in enumerate(atts) for b in atts[i + 1:]]
    out = defaultdict(list)
    for k in keys:
        vals = [bool(recs.get((k[0], k[1], a), {}).get("correct")) or bool(recs.get((k[0], k[1], b), {}).get("correct"))
                for a, b in combos]
        out[k[0]].append(st.mean(vals))
    return {t: st.mean(v) for t, v in out.items()}, combos


def bootstrap(diff, n=10000, seed=0):
    rng = random.Random(seed)
    vals = list(diff.values())
    means = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)], sum(m <= 0 for m in means) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--attempts", help="comma-separated attempt indices to keep in the runs (default: all)")
    ap.add_argument("--table", action="store_true", help="one markdown row per file instead of the full report")
    args = ap.parse_args()
    if args.table:
        print("| run | pairs | attempts | pass@1 run | baseline | diff [95% CI] | tok/attempt mean / median | vs baseline "
              "| correct-rollout tok | sources | est cost |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    base = by_pair(latest(args.baseline))
    for f in args.files:
        recs = latest(f)
        if args.attempts:
            keep = {int(a) for a in args.attempts.split(",")}
            recs = {k: r for k, r in recs.items() if k[2] in keep}
        run = by_pair(recs)
        keys = sorted(k for k in run if k in base)
        if not keys:
            print(f"\n{Path(f).stem}: no records")
            continue
        rr, br = task_rates(run, keys), task_rates(base, keys)
        diff = {t: rr[t] - br[t] for t in rr}
        lo, hi, p_le0 = bootstrap(diff)
        rt = [tok(r) for k in keys for r in run[k]]
        bt = [tok(r) for k in keys for r in base[k]]
        n_att = len(rt)
        allr = [r for k in keys for r in run[k]]
        ct = sorted(tok(r) for r in allr if r.get("correct"))
        bct = sorted(tok(r) for k in keys for r in base[k] if r.get("correct"))
        pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))] / 1e3 if xs else 0
        if args.table:
            src = Counter(r.get("answer_source") or "none" for r in allr)
            print(f"| {Path(f).stem} | {len(keys)} | {n_att} | {st.mean(rr.values()):.1%} | {st.mean(br.values()):.1%} | "
                  f"{st.mean(diff.values()) * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}] | "
                  f"{st.mean(rt) / 1e3:.1f}k / {st.median(rt) / 1e3:.1f}k | {st.mean(rt) / st.mean(bt) - 1:+.1%} | "
                  f"{st.mean(ct) / 1e3 if ct else 0:.1f}k (p50 {pct(ct, .5):.1f}k, p90 {pct(ct, .9):.1f}k; "
                  f"E004-GL p50 {pct(bct, .5):.1f}k, p90 {pct(bct, .9):.1f}k) | " + ", ".join(f"{s} {n}" for s, n in src.most_common())
                  + f" | ${sum(r.get('est_cost_usd', 0) for r in allr):.2f} |")
            continue
        print(f"\n{Path(f).stem}: {len(keys)} pairs / {len(rr)} tasks, {n_att} attempts")
        atts, per = per_attempt_task_weighted(recs, keys)
        print(f"  task-weighted pass@1: run {st.mean(rr.values()):.1%} (per attempt "
              + ", ".join(f"{x:.1%}" for x in per) + f") vs baseline {st.mean(br.values()):.1%}; "
              f"diff {st.mean(diff.values()) * 100:+.1f} pp, 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}], P(<=0) {p_le0:.2f}")
        if len({a for (_, _, a) in recs}) >= 2 and not args.table:
            base_recs = latest(args.baseline)
            r2, combos = task_pass2(recs, keys)
            b2, _ = task_pass2(base_recs, keys)
            d2 = {t: r2[t] - b2[t] for t in r2}
            lo2, hi2, _ = bootstrap(d2)
            r01, _ = task_pass2(recs, keys, [(0, 1)])
            print(f"  pass@2 (mean over {len(combos)} attempt pairs): run {st.mean(r2.values()):.1%} (attempts 0+1: "
                  f"{st.mean(r01.values()):.1%}) vs baseline {st.mean(b2.values()):.1%} (mean over its attempt pairs); "
                  f"diff {st.mean(d2.values()) * 100:+.1f} pp, 95% CI [{lo2 * 100:+.1f}, {hi2 * 100:+.1f}]")
        pa = st.mean(st.mean(bool(r.get("correct")) for r in run[k]) for k in keys)
        pb = st.mean(st.mean(bool(r.get("correct")) for r in base[k]) for k in keys)
        print(f"  pair accuracy: run {pa:.1%} vs baseline {pb:.1%}")
        print(f"  tokens/attempt: run mean {st.mean(rt) / 1e3:.1f}k median {st.median(rt) / 1e3:.1f}k vs baseline "
              f"mean {st.mean(bt) / 1e3:.1f}k median {st.median(bt) / 1e3:.1f}k ({st.mean(rt) / st.mean(bt) - 1:+.1%})")
        src = Counter(r.get("answer_source", "parse" if r.get("pred") is not None else None) or "none" for r in allr)
        acc = {s: sum(bool(r.get("correct")) for r in allr
                      if (r.get("answer_source", "parse" if r.get("pred") is not None else None) or "none") == s)
               for s in src}
        print("  answer sources: " + ", ".join(f"{s} {n} ({acc[s]} correct)" for s, n in src.most_common()))
        strict = task_rates(run, keys, lambda r: bool(r.get("correct")) and r.get("answer_source") != "lenient")
        print(f"  pass@1 without lenient-parse answers: {st.mean(strict.values()):.1%}")
        stops = Counter(r.get("stop_reason") for r in allr)
        print("  stop reasons: " + ", ".join(f"{s} {n}" for s, n in stops.most_common()))
        # Long tail: pairs whose baseline correct rollouts needed > 40k tokens on average.
        lt = [k for k in keys if any(r.get("correct") for r in base[k])
              and st.mean(tok(r) for r in base[k] if r.get("correct")) > 40000]
        if lt:
            print(f"  long-tail pairs (baseline correct rollouts >40k): {len(lt)}, pair accuracy run "
                  f"{st.mean(st.mean(bool(r.get('correct')) for r in run[k]) for k in lt):.1%} vs baseline "
                  f"{st.mean(st.mean(bool(r.get('correct')) for r in base[k]) for k in lt):.1%}")
        past = [r for r in allr if tok(r) > 40000]
        print(f"  attempts over 40k tokens: {len(past)} of {len(allr)}, {sum(bool(r.get('correct')) for r in past)} correct")
        if ct:
            print(f"  correct-rollout tokens p50 / p90: run {pct(ct, .5):.1f}k / {pct(ct, .9):.1f}k vs baseline "
                  f"{pct(bct, .5):.1f}k / {pct(bct, .9):.1f}k on the same pairs")
            q = st.quantiles(ct, n=4) if len(ct) > 1 else [ct[0]] * 3
            print(f"  correct rollouts: {len(ct)}, tokens mean {st.mean(ct) / 1e3:.1f}k, quartiles "
                  + "/".join(f"{x / 1e3:.1f}k" for x in q) + f", share <=25k {sum(x <= 25000 for x in ct) / len(ct):.0%}")
        wrong = [tok(r) for r in allr if not r.get("correct")]
        if wrong:
            print(f"  failed rollouts: {len(wrong)}, tokens mean {st.mean(wrong) / 1e3:.1f}k; total tokens "
                  f"{sum(rt) / 1e6:.2f}M, est cost ${sum(r.get('est_cost_usd', 0) for r in allr):.2f}")


if __name__ == "__main__":
    main()
