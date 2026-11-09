"""E035 analysis: no-tool shortening arms on Flash-1017 vs E034 (max, no extras), attempts 0 and 1.

ARC Prize scoring as in E034: task score = fraction of its test pairs solved (attempt 0 for pass@1 a0, either of
0/1 for pass@2), averaged over the 120 tasks. Per (task, pair, attempt) the last non-error record is used;
missing/errored attempts count as wrong.

    .venv/bin/python experiments/E035_notool_shorter/scripts/analyze.py [--status]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"
R = ROOT / "evals/results"
ARMS = {
    "E034 max": R / "arcprize_deepseek-flash_max_e034_flash1017.jsonl",
    "A max+sum": R / "arcprize_deepseek-flash_max_e035_sum.jsonl",
    "B low": R / "arcprize_deepseek-flash_low_e035_plain.jsonl",
    "C low+sum": R / "arcprize_deepseek-flash_low_e035_sum.jsonl",
    "D low+hdr+sum": R / "arcprize_deepseek-flash_low_e035_hdrsum.jsonl",
    "E low+hdr": R / "arcprize_deepseek-flash_low_e035_hdr.jsonl",
}
OUT_PRICE, IN_PRICE = 0.60, 0.14  # $/M, Flash off-peak list (cache-miss input)
COMPARE = [("A max+sum", "E034 max"), ("B low", "E034 max"), ("D low+hdr+sum", "B low"), ("C low+sum", "E034 max"),
           ("D low+hdr+sum", "E034 max"), ("E low+hdr", "E034 max"), ("C low+sum", "B low"),
           ("D low+hdr+sum", "C low+sum"), ("E low+hdr", "B low"), ("D low+hdr+sum", "E low+hdr")]


def load(path):
    recs = [json.loads(l) for l in open(path)] if path.exists() else []
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
    s = {"pass@1 a0": task_scores(best, npairs, [0]), "pass@1 a1": task_scores(best, npairs, [1]),
         "pass@2": task_scores(best, npairs, [0, 1])}
    s["pass@1 mean"] = {t: (s["pass@1 a0"][t] + s["pass@1 a1"][t]) / 2 for t in npairs}
    return s


def boot(delta, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(list(delta.values()))
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cost(recs):
    ok = [d for d in recs if "error" not in d and d.get("usage") and d["attempt"] < 2]
    o = sum(d["usage"]["completion_tokens"] for d in ok) / 1e6 * OUT_PRICE
    i = sum(d["usage"].get("prompt_tokens", 0) for d in ok) / 1e6 * IN_PRICE
    return o, i


def status(npairs, stopped=()):
    total = 2 * sum(npairs.values())
    tot_cost = proj = 0.0
    for name, path in ARMS.items():
        if name.startswith("E034"):
            continue
        recs, best = load(path)
        ok = [d for d in best.values() if "error" not in d]
        nerr = sum("error" in d for d in recs)
        o, i = cost(recs)
        c = o + i
        p = c if name[0] in stopped else (c / max(len(ok), 1) * total if ok else 0)
        tot_cost += c
        proj += p
        toks = [d["usage"]["completion_tokens"] for d in ok if d.get("usage")]
        acc = np.mean([d.get("correct", False) for d in ok]) if ok else 0
        print(f"{name:14s} ok {len(ok):3d}/{total} err {nerr:3d} ({nerr / max(len(recs), 1):.0%}) "
              f"acc/call {acc:.1%} tok mean {np.mean(toks) if toks else 0:,.0f} ${c:.2f} proj ${p:.2f}")
    print(f"TOTAL so far ${tot_cost:.2f}, projected ${proj:.2f} (completed-call mean; biased low early)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="short progress/cost line per arm")
    ap.add_argument("--stopped", default="", help="arm letters stopped early (projected = spent), e.g. A")
    args = ap.parse_args()
    npairs = {p.stem: len(json.load(open(p))["test"]) for p in sorted(TASKS.glob("*.json"))}
    if args.status:
        return status(npairs, tuple(args.stopped))
    out, tok = {}, {}
    rows = []
    for name, path in ARMS.items():
        recs, best = load(path)
        ok = [d for d in best.values() if "error" not in d]
        m = metrics(best, npairs)
        out[name] = m
        u = [d["usage"] for d in ok if d.get("usage")]
        toks = np.array([x["completion_tokens"] for x in u])
        ptoks = np.array([x.get("prompt_tokens", 0) for x in u])
        tok[name] = toks.mean()
        o, i = cost(recs)
        print(f"\n== {name}: {path.name}")
        print(f"  records {len(recs)}, keys ok {len(ok)}/{2 * sum(npairs.values())}, "
              f"errored records {sum('error' in d for d in recs)}")
        for k, v in m.items():
            print(f"  {k:12s} task-weighted {np.mean(list(v.values())):6.1%}")
        pw = pair_rate(best, npairs, [0, 1])
        print(f"  pair-weighted pass@2 {pw:.1%}, pass@1 a0 {pair_rate(best, npairs, [0]):.1%}")
        print(f"  out tok/call: mean {toks.mean():,.0f} median {np.median(toks):,.0f} "
              f"p90 {np.percentile(toks, 90):,.0f} max {toks.max():,}  (n={len(toks)})")
        print(f"  in tok/call: mean {ptoks.mean():,.0f} median {np.median(ptoks):,.0f}")
        print(f"  finish_reason: {dict(Counter(d.get('finish_reason') for d in ok))}, "
              f"unparsed {sum(d.get('pred') is None for d in ok)}")
        print(f"  errors: {dict(Counter(d['error'].split(':')[0] for d in recs if 'error' in d))}")
        print(f"  cost (off-peak list): out ${o:.2f} + in ${i:.2f} = ${o + i:.2f}")
        rows.append((name, np.mean(list(m["pass@2"].values())), np.mean(list(m["pass@1 a0"].values())),
                     np.mean(list(m["pass@1 mean"].values())), pw, toks.mean(), np.median(toks),
                     np.percentile(toks, 90), ptoks.mean(), o + i))
    print("\n| Arm | pass@2 | pass@1 a0 | pass@1 mean | pair pass@2 | out tok mean | median | p90 | in tok | cost |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.1%} | {r[2]:.1%} | {r[3]:.1%} | {r[4]:.1%} | {r[5] / 1e3:.1f}k | "
              f"{r[6] / 1e3:.1f}k | {r[7] / 1e3:.1f}k | {r[8] / 1e3:.1f}k | ${r[9]:.2f} |")
    bests = {n: load(p)[1] for n, p in ARMS.items()}
    full = {n: [t for t in npairs if all("error" not in bests[n].get((t, p, a), {"error": 1})
                                          for p in range(npairs[t]) for a in (0, 1))] for n in ARMS}
    print("\n== Paired deltas, bootstrap 95% CI over tasks complete in both arms (partial arms: stopped early)")
    for a_, b_ in COMPARE:
        ts = [t for t in npairs if t in set(full[a_]) & set(full[b_])]
        ka = [d["usage"]["completion_tokens"] for (t, p, a), d in bests[a_].items() if t in ts]
        kb = [d["usage"]["completion_tokens"] for (t, p, a), d in bests[b_].items() if t in ts]
        print(f"  {a_} - {b_}: {len(ts)} tasks; on them {a_} pass@2 {np.mean([out[a_]['pass@2'][t] for t in ts]):.1%}"
              f" vs {np.mean([out[b_]['pass@2'][t] for t in ts]):.1%}, pass@1 mean "
              f"{np.mean([out[a_]['pass@1 mean'][t] for t in ts]):.1%} vs {np.mean([out[b_]['pass@1 mean'][t] for t in ts]):.1%},"
              f" out tok {np.mean(ka) / 1e3:.1f}k vs {np.mean(kb) / 1e3:.1f}k (x{np.mean(ka) / np.mean(kb):.2f})")
        for k in ["pass@2", "pass@1 mean", "pass@1 a0"]:
            a, b = out[a_][k], out[b_][k]
            mu, lo, hi = boot({t: a[t] - b[t] for t in ts})
            w = sum(a[t] > b[t] for t in ts)
            l_ = sum(a[t] < b[t] for t in ts)
            print(f"    d{k:12s} {mu * 100:+5.1f} pp CI [{lo * 100:+.1f}, {hi * 100:+.1f}] better {w} / worse {l_}")
    return
    for a_, b_ in COMPARE:
        print(f"  {a_} - {b_}: tokens x{tok[a_] / tok[b_]:.2f}")
        for k in ["pass@2", "pass@1 mean", "pass@1 a0"]:
            a, b = out[a_][k], out[b_][k]
            mu, lo, hi = boot({t: a[t] - b[t] for t in npairs})
            w = sum(a[t] > b[t] for t in npairs)
            l_ = sum(a[t] < b[t] for t in npairs)
            print(f"    d{k:12s} {mu * 100:+5.1f} pp CI [{lo * 100:+.1f}, {hi * 100:+.1f}] better {w} / worse {l_}")


if __name__ == "__main__":
    main()
