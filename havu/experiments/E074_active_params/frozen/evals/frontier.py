"""Accuracy-vs-tokens frontier over full-set runs (E009-E011), plus mixed-config attempt pairs for pass@2.

For each results file (latest record per task/pair/attempt; attempts with an API error count as missing):
samples, ARC Prize task-weighted pass@1 (mean over samples), pass@2 (mean over attempt pairs), output tokens per
attempt (mean / median), tokens of correct rollouts (p50 / p90), and recorded cost per sample (est_cost_usd of the
kept records / samples). Only samples covering all 167 pairs count as complete.

    .venv/bin/python evals/frontier.py name=evals/results/<file>.jsonl [...] [--mix nameA,nameB ...]

--mix A,B: pass@2 of a pair made of one attempt from A and one from B (mean over all such attempt pairs) and its
tokens per pair, e.g. a low-effort and a high-effort attempt.
"""
import argparse
import json
import statistics as st
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}
KEYS = [(t, p) for t, task in TASKS.items() for p in range(len(task["test"]))]


def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    by_att = {}
    for (t, p, a), r in out.items():
        if "error" not in r:
            by_att.setdefault(a, {})[(t, p)] = r
    return by_att


def score(ok):
    """ok(t, p) -> float in [0, 1]; ARC Prize task-weighted mean."""
    return st.mean(st.mean(ok(t, p) for p in range(len(task["test"]))) for t, task in TASKS.items())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def summary(name, by_att):
    atts = sorted(by_att)
    complete = [a for a in atts if len(by_att[a]) >= len(KEYS) - 10]  # allow a few unretried API errors (counted wrong)
    p1 = [score(lambda t, p, a=a: bool(by_att[a].get((t, p), {}).get("correct"))) for a in complete]
    p2 = [score(lambda t, p, a=a, b=b: bool(by_att[a].get((t, p), {}).get("correct"))
                or bool(by_att[b].get((t, p), {}).get("correct"))) for a, b in combinations(complete, 2)]
    recs = [r for a in complete for r in by_att[a].values()]
    toks = [r["usage"]["completion"] for r in recs]
    ctoks = [r["usage"]["completion"] for r in recs if r.get("correct")]
    cost = sum(r.get("est_cost_usd", 0) for r in recs) / max(1, len(complete))
    return {"name": name, "samples": len(complete), "pass1": st.mean(p1) if p1 else None,
            "pass1_each": p1, "pass2": st.mean(p2) if p2 else None, "tok_mean": st.mean(toks), "tok_median": st.median(toks),
            "ctok_p50": pct(ctoks, .5), "ctok_p90": pct(ctoks, .9), "cost_per_sample": cost}


def mix(a_att, b_att):
    vals, toks = [], []
    for a in a_att.values():
        for b in b_att.values():
            vals.append(score(lambda t, p: bool(a.get((t, p), {}).get("correct")) or bool(b.get((t, p), {}).get("correct"))))
            toks.append(st.mean(a[k]["usage"]["completion"] + b[k]["usage"]["completion"] for k in KEYS if k in a and k in b))
    return st.mean(vals), st.mean(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path")
    ap.add_argument("--mix", nargs="*", default=[])
    args = ap.parse_args()
    data = {}
    print("| config | samples | pass@1 (per sample) | pass@2 | tok/attempt mean / median | correct p50 / p90 | $/sample |")
    print("|---|---|---|---|---|---|---|")
    for item in args.runs:
        name, path = item.split("=", 1)
        data[name] = load(path)
        s = summary(name, data[name])
        each = " / ".join(f"{x * 100:.1f}" for x in s["pass1_each"])
        print(f"| {name} | {s['samples']} | {s['pass1'] * 100:.1f}% ({each}) | "
              + (f"{s['pass2'] * 100:.1f}%" if s["pass2"] is not None else "—")
              + f" | {s['tok_mean'] / 1e3:.1f}k / {s['tok_median'] / 1e3:.1f}k | {s['ctok_p50'] / 1e3:.1f}k / "
              f"{s['ctok_p90'] / 1e3:.1f}k | ${s['cost_per_sample']:.2f} |")
    for item in args.mix:
        a, b = item.split(",")
        full = lambda d: {k: v for k, v in d.items() if len(v) >= len(KEYS) - 10}
        p2, tok = mix(full(data[a]), full(data[b]))
        print(f"mixed pair {a} + {b}: pass@2 {p2 * 100:.1f}%, {tok / 1e3:.1f}k output tokens per pair")


if __name__ == "__main__":
    main()
