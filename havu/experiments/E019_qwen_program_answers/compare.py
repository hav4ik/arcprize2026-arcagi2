"""E019: paired comparison of Qwen tool-harness arms (1 attempt per pair), e.g. A = submit_answer baseline,
B = program-only. Markdown tables on stdout.

    .venv/bin/python experiments/E019_qwen_program_answers/compare.py A=evals/results/qwen_tools_baseline_local_x.jsonl \
        B=evals/results/qwen_tools_program_x.jsonl [--pairs-file experiments/E009_tool_token_efficiency/dev_pairs.txt]

Scores use attempt 0 of the pairs both arms finished (without error). Task-weighted = ARC Prize (mean over tasks
of the mean over its pairs); pair-weighted = plain mean. The paired CI bootstraps tasks (4000 resamples).
"""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r and r["attempt"] == 0:
                out[(r["task_id"], int(r["pair"]))] = r
    return out


def task_weighted(vals):
    tasks = defaultdict(list)
    for (t, _), v in vals.items():
        tasks[t].append(v)
    return statistics.mean(statistics.mean(v) for v in tasks.values())


def paired(a, b, n=4000, seed=0):
    tasks = defaultdict(list)
    for k in a:
        tasks[k[0]].append(b[k] - a[k])
    d = {t: statistics.mean(v) for t, v in tasks.items()}
    ids = sorted(d)
    rng = random.Random(seed)
    boots = sorted(statistics.mean(d[rng.choice(ids)] for _ in ids) for _ in range(n))
    return statistics.mean(d.values()), boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def pct(x):
    return f"{100 * x:.1f}%" if x == x else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=path; the first is the reference for paired differences")
    ap.add_argument("--pairs-file")
    args = ap.parse_args()
    arms = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        arms[name] = load(path)
    keys = set.intersection(*(set(v) for v in arms.values()))
    if args.pairs_file:
        text = Path(args.pairs_file).read_text().replace("\n", ",")
        sel = {(t, int(p)) for t, p in (x.strip().rsplit(":", 1) for x in text.split(",") if x.strip())}
        keys &= sel
    keys = sorted(keys)
    print(f"{len(keys)} common pairs, {len({k[0] for k in keys})} tasks\n")
    corr = {n: {k: float(r[k]["correct"]) for k in keys} for n, r in arms.items()}

    print("| arm | pass@1 task-weighted | pass@1 pair-weighted | correct | tokens mean / median / correct-median | "
          "cap/no-close stops | submissions / attempt | cache hit (reported / ideal) |")
    print("|---|---|---|---|---|---|---|---|")
    for n, recs in arms.items():
        rs = [recs[k] for k in keys]
        toks = [r["usage"]["completion"] for r in rs]
        ctoks = [r["usage"]["completion"] for r in rs if r["correct"]]
        stops = Counter(r["stop_reason"] for r in rs)
        caps = stops["output_budget"] + stops["context"] + stops["unclosed_think"]
        subs = [r.get("n_submissions") for r in rs if "n_submissions" in r]
        pt = sum(r["usage"].get("prompt_total", 0) for r in rs)
        cached = sum(r["usage"].get("cached", 0) for r in rs if "cached" in r["usage"])
        has_c = any("cached" in r["usage"] for r in rs)
        ideal = sum(r["usage"].get("ideal_cached", 0) for r in rs)
        hit = f"{pct(cached / pt) if has_c and pt else 'n/a'} / {pct(ideal / pt) if pt else 'n/a'}"
        print(f"| {n} | {pct(task_weighted(corr[n]))} | {pct(statistics.mean(corr[n].values()))} | "
              f"{int(sum(corr[n].values()))}/{len(keys)} | {statistics.mean(toks) / 1e3:.1f}k / "
              f"{statistics.median(toks) / 1e3:.1f}k / {statistics.median(ctoks) / 1e3 if ctoks else float('nan'):.1f}k | "
              f"{caps} ({pct(caps / len(rs))}) | "
              f"{statistics.mean(subs) if subs else float('nan'):.2f} | {hit} |")

    names = list(arms)
    ref = names[0]
    print("\n| comparison | Δ pass@1 task-weighted (95% CI) | Δ pair-weighted | discordant (only ref / only other) |")
    print("|---|---|---|---|")
    for n in names[1:]:
        d, lo, hi = paired(corr[ref], corr[n])
        only_ref = sum(corr[ref][k] > corr[n][k] for k in keys)
        only_n = sum(corr[n][k] > corr[ref][k] for k in keys)
        dp = statistics.mean(corr[n].values()) - statistics.mean(corr[ref].values())
        print(f"| {n} − {ref} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {100 * dp:+.1f} | {only_ref} / {only_n} |")

    for n, recs in arms.items():
        rs = [recs[k] for k in keys]
        print(f"\n**{n}: answer sources**\n\n| source | attempts | share | correct | accuracy | tokens median |")
        print("|---|---|---|---|---|---|")
        by = defaultdict(list)
        for r in rs:
            by[r.get("answer_source")].append(r)
        for s, v in sorted(by.items(), key=lambda x: -len(x[1])):
            print(f"| {s} | {len(v)} | {pct(len(v) / len(rs))} | {sum(r['correct'] for r in v)} | "
                  f"{pct(sum(r['correct'] for r in v) / len(v))} | "
                  f"{statistics.median(r['usage']['completion'] for r in v) / 1e3:.1f}k |")
        tp = [r for r in rs if r.get("train_pass") is True]
        tf = [r for r in rs if r.get("train_pass") is False]
        tn = [r for r in rs if r.get("pred") is not None and r.get("train_pass") is None]
        if tp or tf:
            print(f"\nP(correct | train pass) = {sum(r['correct'] for r in tp)}/{len(tp)} = "
                  f"{pct(sum(r['correct'] for r in tp) / len(tp)) if tp else 'n/a'}; P(correct | train fail) = "
                  f"{sum(r['correct'] for r in tf)}/{len(tf)} = {pct(sum(r['correct'] for r in tf) / len(tf)) if tf else 'n/a'}"
                  f"; answers without a train check: {len(tn)} ({sum(r['correct'] for r in tn)} correct)")
        stops = Counter(r["stop_reason"] for r in rs)
        print(f"\nstop reasons: {dict(stops.most_common())}")
        if any("n_submissions" in r for r in rs):
            subs = Counter(r["n_submissions"] for r in rs)
            fails = Counter(k for r in rs for s in r.get("submissions", []) if s["n_pass"] < s["n_train"]
                            for k in s["fail_kinds"])
            test_fail = sum(1 for r in rs for s in r.get("submissions", []) if s["n_pass"] == s["n_train"] and s.get("test_error"))
            print(f"submissions per attempt: {dict(sorted(subs.items()))}; submission-cap hits: "
                  f"{stops['submission_cap']}; rejected-submission failure kinds: {dict(fails.most_common(8))}; "
                  f"train-pass but test error: {test_fail}; nudges: {sum(r.get('nudges', 0) for r in rs)}")
        print(f"tool calls / attempt {statistics.mean(r['tool_calls'] for r in rs):.1f}, tool errors / attempt "
              f"{statistics.mean(r['tool_errors'] for r in rs):.2f}, parse failures {sum(r['parse_failures'] for r in rs)}, "
              f"turns mean {statistics.mean(r['turns'] for r in rs):.1f}")


if __name__ == "__main__":
    main()
