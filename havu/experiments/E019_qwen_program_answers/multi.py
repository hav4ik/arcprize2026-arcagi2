"""E019 multi-sample stats: each arm is a list of files (one sample each, any attempt index) on the same pairs.

    .venv/bin/python experiments/E019_qwen_program_answers/multi.py \
        A=evals/results/qwen_tools_baseline_gw_dev56.jsonl,evals/results/qwen_tools_baseline_gw_dev56_s1.jsonl \
        B=evals/results/qwen_tools_program_gw_dev56.jsonl,evals/results/qwen_tools_program_gw_dev56_s1.jsonl

A file may hold several attempt indices (e.g. E012's 2-attempt runs): every (task, pair, attempt) is a sample.
Only pairs with the same number of finished samples in every arm (the minimum k over arms, first k samples) count.
Reports per arm: mean pass@1 over samples (task- and pair-weighted), pass@2 over the first two samples, tokens;
and the paired difference of mean pass@1 against the first arm with a 95% bootstrap CI over tasks.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(paths, pairs=None):
    out = defaultdict(list)  # (task, pair) -> [records], file order then attempt order
    for path in paths:
        recs = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    recs[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
        for (t, p, a) in sorted(recs, key=lambda k: k[2]):
            if pairs is None or (t, p) in pairs:
                out[(t, p)].append(recs[(t, p, a)])
    return out


def tw(vals):
    tasks = defaultdict(list)
    for (t, _), v in vals.items():
        tasks[t].append(v)
    return statistics.mean(statistics.mean(v) for v in tasks.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", help="NAME=file1[,file2...]")
    ap.add_argument("--pairs-file", default="experiments/E009_tool_token_efficiency/dev_pairs.txt")
    ap.add_argument("--samples", type=int, help="use exactly this many samples per pair (default: the minimum)")
    args = ap.parse_args()
    text = Path(args.pairs_file).read_text().replace("\n", ",")
    pairs = {(t, int(p)) for t, p in (x.strip().rsplit(":", 1) for x in text.split(",") if x.strip())}
    arms = {}
    for spec in args.arms:
        name, files = spec.split("=", 1)
        arms[name] = load(files.split(","), pairs)
    k = args.samples or min(len(v) for a in arms.values() for v in a.values()) if all(arms.values()) else 0
    keys = sorted(key for key in pairs if all(len(a.get(key, [])) >= k for a in arms.values()))
    print(f"{len(keys)} pairs / {len({x[0] for x in keys})} tasks with >= {k} samples in every arm\n")
    if not keys or k == 0:
        return
    mean1 = {n: {key: statistics.mean(r["correct"] for r in a[key][:k]) for key in keys} for n, a in arms.items()}
    print(f"| arm | mean pass@1 over {k} samples (task-weighted) | pair-weighted | pass@2 (first 2 samples, task-w.) | "
          "tokens mean / median |")
    print("|---|---|---|---|---|")
    for n, a in arms.items():
        p2 = {key: float(any(r["correct"] for r in a[key][:2])) for key in keys}
        toks = [r["usage"]["completion"] for key in keys for r in a[key][:k]]
        print(f"| {n} | {100 * tw(mean1[n]):.1f}% | {100 * statistics.mean(mean1[n].values()):.1f}% | "
              f"{100 * tw(p2) if k >= 2 else float('nan'):.1f}% | {statistics.mean(toks) / 1e3:.1f}k / "
              f"{statistics.median(toks) / 1e3:.1f}k |")
    names = list(arms)
    rng = random.Random(0)
    print(f"\n| comparison | Δ mean pass@1 task-weighted (95% CI) |\n|---|---|")
    for n in names[1:]:
        d = defaultdict(list)
        for key in keys:
            d[key[0]].append(mean1[n][key] - mean1[names[0]][key])
        dt = {t: statistics.mean(v) for t, v in d.items()}
        ids = sorted(dt)
        boots = sorted(statistics.mean(dt[rng.choice(ids)] for _ in ids) for _ in range(4000))
        print(f"| {n} − {names[0]} | {100 * statistics.mean(dt.values()):+.1f} [{100 * boots[100]:+.1f}, "
              f"{100 * boots[3899]:+.1f}] |")


if __name__ == "__main__":
    main()
