"""E031 scratch analysis: per-batch stats + paired comparison vs E025 on the same (task_id, pair).
Usage: .venv/bin/python experiments/E031_flash1017_hdr_rollouts/scripts/analyze.py"""
import collections
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[3] / "evals/results"
E031 = "tools_persistent_grid_lean2sub_deepseek-flash_low_e031_{}.jsonl"
E025 = "tools_persistent_grid_lean2roles_deepseek-flash_low_e025_{}.jsonl"
BATCHES = ["arc2_train", "nvarc_b0"]


def load(path):
    """Latest record per (task_id, pair, attempt=0); resumes append retries of errored records."""
    out = {}
    if not path.exists():
        return out
    for line in path.open():
        r = json.loads(line)
        if r.get("attempt", 0) != 0:
            continue
        k = (r["task_id"], r["pair"])
        if k not in out or "error" in out[k] or "error" not in r:
            out[k] = r
    return out


def pct(x, n):
    return f"{100 * x / n:.1f}%" if n else "-"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def stats(recs, label):
    ok = [r for r in recs if "error" not in r]
    n, ne = len(recs), len(recs) - len(ok)
    # pair-weighted pass@1: each pair is one attempt here, so mean over pairs == mean over records
    cor = sum(bool(r.get("correct")) for r in ok)
    tp = [r for r in ok if r.get("train_pass")]
    ctp = sum(bool(r.get("correct")) for r in tp)
    toks = [r["usage"]["completion"] for r in ok]
    ctoks = [r["usage"]["completion"] for r in ok if r.get("correct")]
    cost = sum(r.get("est_cost_usd", 0) for r in recs)
    stops = collections.Counter(r.get("stop_reason") for r in recs)
    src = collections.Counter(r.get("answer_source") for r in ok)
    print(f"\n## {label}: n={n} errors={ne} ({pct(ne, n)})")
    print(f"pass@1 {pct(cor, len(ok))} ({cor}/{len(ok)}) | train_pass {pct(len(tp), len(ok))} | "
          f"P(correct|train_pass) {pct(ctp, len(tp))} | P(correct|~train_pass) {pct(cor - ctp, len(ok) - len(tp))}")
    if toks:
        print(f"out tokens mean {statistics.mean(toks):.0f} median {statistics.median(toks):.0f} p90 {q(toks, .9)} "
              f"| correct with <=40k: {pct(sum(t <= 40000 for t in ctoks), len(ctoks))}")
    print(f"cost ${cost:.2f} (${cost / max(n, 1):.4f}/attempt)")
    print("stops", dict(stops.most_common()))
    print("answer_source", dict(src.most_common()))
    return dict(n=n, err=ne, cor=cor, ok=len(ok), cost=cost, toks=sum(toks))


def paired(a31, a25, label):
    keys = [k for k in a31 if k in a25 and "error" not in a31[k] and "error" not in a25[k]]
    c = collections.Counter((bool(a25[k].get("correct")), bool(a31[k].get("correct"))) for k in keys)
    t31 = sum(a31[k]["usage"]["completion"] for k in keys)
    t25 = sum(a25[k]["usage"]["completion"] for k in keys)
    c31 = sum(a31[k].get("est_cost_usd", 0) for k in keys)
    c25 = sum(a25[k].get("est_cost_usd", 0) for k in keys)
    tp31 = sum(bool(a31[k].get("train_pass")) for k in keys)
    tp25 = sum(bool(a25[k].get("train_pass")) for k in keys)
    n = len(keys)
    print(f"\n## paired vs E025 {label}: n={n}")
    print(f"both {c[True, True]} | only E025 {c[True, False]} | only E031 {c[False, True]} | neither {c[False, False]}")
    print(f"pass@1 E025 {pct(c[True, True] + c[True, False], n)} vs E031 {pct(c[True, True] + c[False, True], n)} | "
          f"train_pass E025 {pct(tp25, n)} vs E031 {pct(tp31, n)}")
    if n:
        print(f"out tokens/attempt E025 {t25 / n:.0f} vs E031 {t31 / n:.0f} (ratio {t31 / t25:.2f}) | "
              f"cost E025 ${c25:.2f} vs E031 ${c31:.2f} (ratio {c31 / c25:.2f})")
    return c, n, t31, t25, c31, c25


if __name__ == "__main__":
    all31, all25 = {}, {}
    for b in BATCHES:
        a31, a25 = load(R / E031.format(b)), load(R / E025.format(b))
        all31.update({(b,) + k: v for k, v in a31.items()})
        all25.update({(b,) + k: v for k, v in a25.items()})
        stats(list(a31.values()), f"E031 {b}")
        paired(a31, a25, b)
    stats(list(all31.values()), "E031 total")
    paired(all31, all25, "total")
