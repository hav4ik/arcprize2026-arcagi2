"""E020: Kaggle-budget numbers for an arm with two real samples (attempt 0 and attempt 1 in one results file).

- 2-sample mean pass@1 (pair- and task-weighted);
- free pass@2: v3 `selection.select_two` over attempt 0 alone (with its alternatives);
- always pass@2: select_two over attempts 0 and 1 (2 attempts on every pair);
- gated pass@2 at a mean token budget B per pair: attempt 1 runs only on pairs `selection.retry_priority` selects, in
  its order, as many as fit (left = B * pairs - attempt-0 tokens; each retry is budgeted at the mean attempt-1
  tokens of the eligible pairs, since a retry's cost is unknown in advance), and the realised tokens are reported.
Paired CIs (95% cluster bootstrap over tasks) are against the free pass@2.

    .venv/bin/python experiments/E020_qwen_harness_80/two_sample.py evals/results/qwen_tools_e020_K2_all.jsonl [--budgets 44000,64000]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_harness.selection import candidates, retry_priority, select_two  # noqa: E402
from select_sim import paired_ci  # noqa: E402

TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"


def tw(vals):
    t = {}
    for (task, _), v in vals.items():
        t.setdefault(task, []).append(v)
    return statistics.mean(statistics.mean(v) for v in t.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--budgets", default="44000,64000")
    args = ap.parse_args()
    recs = {}
    for line in Path(args.results).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if not ("error" in r and r.get("pred") is None):
                recs[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    keys = sorted({(t, p) for t, p, a in recs if (t, p, 0) in recs and (t, p, 1) in recs})
    truth = {k: json.loads((TASKS / f"{k[0]}.json").read_text())["test"][k[1]]["output"] for k in keys}
    a0 = {k: recs[k + (0,)] for k in keys}
    a1 = {k: recs[k + (1,)] for k in keys}
    tok = lambda r: r["usage"]["completion"]  # noqa: E731

    def score(attempts_by_key):
        return {k: float(truth[k] in [c["grid"] for c in select_two(candidates(v))]) for k, v in attempts_by_key.items()}

    p1 = {k: (float(a0[k]["correct"]) + float(a1[k]["correct"])) / 2 for k in keys}
    free = score({k: {0: a0[k]} for k in keys})
    always = score({k: {0: a0[k], 1: a1[k]} for k in keys})
    rows = [("2-sample mean pass@1", p1, [(tok(a0[k]) + tok(a1[k])) / 2 for k in keys], "—"),
            ("free pass@2 (1 attempt + alternatives)", free, [tok(a0[k]) for k in keys], "—"),
            ("always pass@2 (2 attempts)", always, [tok(a0[k]) + tok(a1[k]) for k in keys], f"{len(keys)}")]
    pri = {k: retry_priority({**a0[k], "tokens": tok(a0[k])}) for k in keys}
    elig = sorted([k for k in keys if pri[k] is not None], key=lambda k: pri[k])
    mean_retry = statistics.mean(tok(a1[k]) for k in elig) if elig else 0
    for b in [int(x) for x in args.budgets.split(",")]:
        left = b * len(keys) - sum(tok(a0[k]) for k in keys)
        n = max(0, min(len(elig), int(left // mean_retry))) if mean_retry else 0
        chosen = set(elig[:n])
        g = score({k: ({0: a0[k], 1: a1[k]} if k in chosen else {0: a0[k]}) for k in keys})
        rows.append((f"gated pass@2 at {b // 1000}k mean", g,
                     [tok(a0[k]) + (tok(a1[k]) if k in chosen else 0) for k in keys], f"{n} of {len(elig)} eligible"))
    print(f"{Path(args.results).name}: {len(keys)} pairs with both samples; mean attempt-1 tokens of eligible pairs "
          f"{mean_retry / 1000:.1f}k\n")
    print("| metric | pair-w | task-w | tokens/pair mean / p90 | retries | Δ vs free pass@2 [95% CI] |")
    print("|---|---|---|---|---|---|")
    for name, vals, toks, retries in rows:
        d, lo, hi = paired_ci(free, vals)
        q = sorted(toks)[int(0.9 * (len(toks) - 1))]
        print(f"| {name} | {statistics.mean(vals.values()):.1%} | {tw(vals):.1%} | {statistics.mean(toks) / 1000:.1f}k / "
              f"{q / 1000:.1f}k | {retries} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")


if __name__ == "__main__":
    main()
