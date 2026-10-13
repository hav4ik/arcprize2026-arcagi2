"""E020: test-time scaling with answer selection, simulated offline from logged attempts (no GPU, no tokens).

Every finished attempt in the given result files is a candidate for its (task, pair): its test grid, whether its
program passed every training pair (program-mode runs: `train_pass`), its train fraction and its tokens.
No-tool runs (run_qwen_generate.py) give candidates with train_pass None ("free text").

For k = 1..K candidates per pair (random subsets of the pool, `--draws` times, or the first k in file/attempt order
with --ordered) the two submitted answers are the top-2 distinct grids under a policy:

- first2: the first two candidates' grids in order (= plain pass@2 of attempts 0 and 1 when k=2);
- vote: most frequent grid, ties by fewer tokens;
- tp: train-passing grids first, by votes among train-passing candidates, then all votes, then fewer tokens;
      then free-text grids (no-tool) by votes; then failing programs by train fraction and votes (E020 H1/H8).
- tp_frac: like tp, but failing programs rank above free text when their train fraction >= 0.5.
- vote_order: votes, ties by the earliest candidate in spec order (so the spec order sets mode priority);
- vote_tp: pooled cross-mode vote (all candidates count equally), ties broken by the number of train-passing
      candidates, then fewer tokens.

Scores (pass@1 = the first answer, pass@2 = either): Kaggle pair-weighted (primary) and ARC Prize task-weighted;
oracle = any of the k candidates correct; tokens per pair = sum over the k candidates.

    .venv/bin/python experiments/E020_qwen_harness_80/select_sim.py prog=evals/results/qwen_tools_program_x.jsonl \
        [notool=evals/results/qwen_nvfp4_gen_xhigh_240k_t1.jsonl] [--k 1,2,4,8] [--pairs-file F] [--json out.json]
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"


def grid_key(pred):
    if pred is None:
        return None
    if isinstance(pred, str):
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            return None
    try:
        return tuple(tuple(int(v) for v in row) for row in pred)
    except (TypeError, ValueError):
        return None


def tokens(r):
    u = r.get("usage") or {}
    return u.get("completion") or u.get("completion_tokens") or r.get("completion_tokens") or 0


def load(specs, pairs=None, skip_errors=True):
    """{(task, pair): [candidate dicts in spec/attempt order]}."""
    pool = defaultdict(list)
    for spec in specs:
        name, path = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        latest = {}
        for r in recs:
            if skip_errors and "error" in r and r.get("pred") is None:
                continue
            latest[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
        for (t, p, a), r in sorted(latest.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
            if pairs and (t, p) not in pairs:
                continue
            free = "answer_mode" not in json.dumps(r.get("config", "")) and "train_pass" not in r
            pool[(t, p)].append({"src": name, "attempt": a, "grid": grid_key(r.get("pred")),
                                 "correct": bool(r.get("correct")), "train_pass": r.get("train_pass"),
                                 "train_frac": r.get("train_frac"), "tokens": tokens(r), "free": free})
    return pool


def rank(cands, policy):
    """Top-2 distinct grids of the candidates under the policy."""
    cands = [c for c in cands if c["grid"] is not None]
    if policy == "first2":
        out = []
        for c in cands:
            if c["grid"] not in out:
                out.append(c["grid"])
        return out[:2]
    groups = defaultdict(list)
    for c in cands:
        groups[c["grid"]].append(c)

    def key(g):
        cs = groups[g]
        votes, min_tok = len(cs), min(c["tokens"] for c in cs)
        if policy == "vote":
            return (votes, -min_tok)
        if policy == "vote_order":  # votes, ties by the earliest candidate (spec order = mode priority)
            return (votes, -min(cands.index(c) for c in cs))
        if policy == "vote_tp":  # cross-mode pooled vote; train-pass count breaks ties (program answers)
            return (votes, sum(bool(c["train_pass"]) for c in cs), -min_tok)
        tp = sum(bool(c["train_pass"]) for c in cs)
        free = sum(c["free"] for c in cs)
        frac = max((c["train_frac"] or 0) for c in cs)
        if tp:
            cls = 3
        elif policy == "tp_frac" and frac >= 0.5:
            cls = 2
        elif free:
            cls = 1.5
        else:
            cls = 1
        return (cls, tp, votes, frac, -min_tok)
    return sorted(groups, key=key, reverse=True)[:2]


def score_sets(pool, truth, k, policy, draws, ordered, rng):
    """Mean over draws of (pair-weighted pass@1, pass@2, oracle, task-weighted pass@1, pass@2, oracle, tokens/pair)."""
    keys = sorted(pool)
    tot = defaultdict(float)
    n_draws = 1 if ordered else draws
    pair_p2 = defaultdict(float)  # per-pair pass@2 averaged over draws (for the paired CI)
    for _ in range(n_draws):
        per = {}
        toks = []
        for key in keys:
            cands = pool[key]
            sub = cands[:k] if ordered or len(cands) <= k else rng.sample(cands, k)
            if not ordered:
                sub = sorted(sub, key=lambda c: cands.index(c))
            top = rank(sub, policy)
            ans = truth[key]
            per[key] = (float(bool(top) and top[0] == ans), float(ans in top), float(any(c["correct"] for c in sub)))
            pair_p2[key] += per[key][1] / n_draws
            toks.append(sum(c["tokens"] for c in sub))
        by_task = defaultdict(list)
        for (t, _), v in per.items():
            by_task[t].append(v)
        for i, name in enumerate(("p1", "p2", "oracle")):
            tot[name] += statistics.mean(v[i] for v in per.values())
            tot["t_" + name] += statistics.mean(statistics.mean(x[i] for x in vs) for vs in by_task.values())
        tot["tokens"] += statistics.mean(toks)
    res = {k2: v / n_draws for k2, v in tot.items()}
    res["pair_p2"] = dict(pair_p2)
    return res


def paired_ci(a, b, n=4000, seed=0, task_weighted=False):
    """Mean of b - a and its 95% cluster-bootstrap CI over tasks (pair-weighted unless task_weighted)."""
    tasks = defaultdict(list)
    for k in a:
        tasks[k[0]].append(b[k] - a[k])
    ids = sorted(tasks)
    rng = random.Random(seed)

    def stat(sample):
        if task_weighted:
            return statistics.mean(statistics.mean(tasks[t]) for t in sample)
        vals = [v for t in sample for v in tasks[t]]
        return statistics.mean(vals)
    boots = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(n))
    return stat(ids), boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def load_truth(keys):
    truth = {}
    for t in {k[0] for k in keys}:
        task = json.loads((TASKS / f"{t}.json").read_text())
        for p, pair in enumerate(task["test"]):
            truth[(t, p)] = grid_key(pair["output"])
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", help="NAME=results.jsonl (candidates are pooled in this order)")
    ap.add_argument("--k", default="1,2,3,4,6,8")
    ap.add_argument("--policies", default="first2,vote,vote_tp,tp,tp_frac")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--ordered", action="store_true", help="use the first k candidates instead of random subsets")
    ap.add_argument("--pairs-file")
    ap.add_argument("--min-cands", type=int, default=0, help="only pairs with at least this many candidates")
    ap.add_argument("--json")
    ap.add_argument("--ref", default="first2:2", help="policy:k for the paired pass@2 CI column (pair-weighted)")
    args = ap.parse_args()
    pairs = None
    if args.pairs_file:
        text = Path(args.pairs_file).read_text().replace("\n", ",")
        pairs = {(t, int(p)) for t, p in (x.strip().rsplit(":", 1) for x in text.split(",") if x.strip())}
    pool = load(args.specs, pairs)
    if args.min_cands:
        pool = {k: v for k, v in pool.items() if len(v) >= args.min_cands}
    truth = load_truth(pool)
    counts = [len(v) for v in pool.values()]
    print(f"{len(pool)} pairs, {len({k[0] for k in pool})} tasks, candidates per pair min {min(counts)} "
          f"median {statistics.median(counts)} max {max(counts)}")
    cands = [c for v in pool.values() for c in v]
    tp = [c for c in cands if c["train_pass"]]
    if tp:
        print(f"P(correct | train_pass) = {sum(c['correct'] for c in tp) / len(tp):.1%} ({len(tp)} candidates); "
              f"P(correct | not train_pass) = "
              f"{sum(c['correct'] for c in cands if not c['train_pass']) / max(1, len(cands) - len(tp)):.1%}")
    rng = random.Random(0)
    rows = []
    rpol, rk = args.ref.split(":")
    ref = score_sets(pool, truth, int(rk), rpol, args.draws, args.ordered, random.Random(1))["pair_p2"]
    print(f"\n| k | policy | pass@1 pair-w | pass@2 pair-w | oracle pair-w | pass@1 task-w | pass@2 task-w | "
          f"oracle task-w | tokens/pair | Δ pass@2 vs {args.ref} [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in [int(x) for x in args.k.split(",")]:
        if k > max(counts):
            break
        for policy in args.policies.split(","):
            s = score_sets(pool, truth, k, policy, args.draws, args.ordered, rng)
            d, lo, hi = paired_ci(ref, s["pair_p2"])
            rows.append({"k": k, "policy": policy, **{a: b for a, b in s.items() if a != "pair_p2"},
                         "d_p2": d, "d_lo": lo, "d_hi": hi})
            print(f"| {k} | {policy} | {s['p1']:.1%} | {s['p2']:.1%} | {s['oracle']:.1%} | {s['t_p1']:.1%} | "
                  f"{s['t_p2']:.1%} | {s['t_oracle']:.1%} | {s['tokens'] / 1000:.1f}k | "
                  f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
