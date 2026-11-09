"""E012 stats: Qwen tool-harness runs (evals/run_qwen_tools.py) vs no-tool Qwen on the same pairs.

Task-weighted ARC Prize scoring. pass@1 = mean over attempt indices; pass@2 = either of attempts 0/1.
No-tool rows at a cap X are simulated from the 240k no-tool run: an attempt counts as correct only if it was correct
and used <= X tokens, and its cost is min(tokens, X) (a real capped run would add a short forced answer, which
rarely helps: E008).

    .venv/bin/python evals/qwen_tools_stats.py evals/results/qwen_tools_*pilot*.jsonl
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTOOL = ROOT / "evals/results/qwen_nvfp4_gen_xhigh_240k_t1.jsonl"


def load(path):
    latest = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return latest


def scores(correct, pairs):
    """correct: {(t, p, a): bool}; pairs: {(t, p)}. Returns task-weighted (pass@1, pass@2)."""
    tasks = {}
    for t, p in pairs:
        tasks.setdefault(t, []).append(p)

    def tw(f):
        return sum(sum(f(t, p) for p in ps) / len(ps) for t, ps in tasks.items()) / len(tasks) if tasks else float("nan")
    p1 = statistics.mean(tw(lambda t, p, a=a: correct.get((t, p, a), False)) for a in (0, 1))
    p2 = tw(lambda t, p: correct.get((t, p, 0), False) or correct.get((t, p, 1), False))
    return p1, p2


def paired_ci(ca, cb, pairs, n=4000, seed=0):
    """Bootstrap over tasks of (pass@1 diff, pass@2 diff) a - b, each with a 95% CI."""
    import random
    tasks = {}
    for t, p in pairs:
        tasks.setdefault(t, []).append(p)
    ids = sorted(tasks)

    def per_task(c, t):
        ps = tasks[t]
        p1 = sum(sum(c.get((t, p, a), False) for a in (0, 1)) / 2 for p in ps) / len(ps)
        p2 = sum(c.get((t, p, 0), False) or c.get((t, p, 1), False) for p in ps) / len(ps)
        return p1, p2
    d = {t: tuple(x - y for x, y in zip(per_task(ca, t), per_task(cb, t))) for t in ids}
    rng = random.Random(seed)
    out = []
    for k in (0, 1):
        point = statistics.mean(d[t][k] for t in ids)
        boots = sorted(statistics.mean(d[rng.choice(ids)][k] for _ in ids) for _ in range(n))
        out.append((point, boots[int(0.025 * n)], boots[int(0.975 * n)]))
    return out


NO_SALVAGE = False  # --no-salvage: score qwen-tools-3 salvage answers as wrong (v2-equivalent)


def tool_row(path):
    recs = {k: r for k, r in load(path).items() if "error" not in r}
    if NO_SALVAGE:
        for r in recs.values():
            if r.get("answer_source") == "salvage":
                r["correct"], r["answer_source"] = False, None
    errors = sum("error" in r for r in load(path).values())
    toks = [r["usage"]["completion"] for r in recs.values()]
    pairs = {(t, p) for t, p, _ in recs if (t, p, 0) in recs and (t, p, 1) in recs}  # both attempts done
    correct = {k: r["correct"] for k, r in recs.items()}
    p1, p2 = scores(correct, pairs)
    ok_toks = [r["usage"]["completion"] for r in recs.values() if r["correct"]]
    return {
        "run": Path(path).stem.replace("qwen_tools_", ""), "attempts": len(recs), "errors": errors,
        "pass@1": p1, "pass@2": p2, "tok_mean": statistics.mean(toks), "tok_median": statistics.median(toks),
        "tok_correct_median": statistics.median(ok_toks) if ok_toks else 0,
        "turns": statistics.mean(r["turns"] for r in recs.values()),
        "calls": statistics.mean(r["tool_calls"] for r in recs.values()),
        "parse_fail": sum(r["parse_failures"] for r in recs.values()),
        "tool_err": sum(r["tool_errors"] for r in recs.values()),
        "eos_cont": sum(r["eos_continues"] for r in recs.values()),
        "sources": dict(Counter(r["answer_source"] for r in recs.values())),
        "stops": dict(Counter(r["stop_reason"] for r in recs.values())),
        "seconds_mean": statistics.mean(r["seconds"] for r in recs.values()),
        "correct_by_source": dict(Counter(r["answer_source"] for r in recs.values() if r["correct"])),
    }, pairs, correct


def notool_row(pairs, cap):
    recs = {k: r for k, r in load(NOTOOL).items() if (k[0], k[1]) in pairs and k[2] < 2}
    correct = {k: r["correct"] and r["completion_tokens"] <= cap for k, r in recs.items()}
    p1, p2 = scores(correct, pairs)
    toks = [min(r["completion_tokens"], cap) for r in recs.values()]
    return {"run": f"no-tool xhigh, cap {cap // 1000}k (sim.)", "attempts": len(recs), "pass@1": p1, "pass@2": p2,
            "tok_mean": statistics.mean(toks), "tok_median": statistics.median(toks)}, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--caps", default="24000,32000,48000,64000,80000,240000")
    ap.add_argument("--ci-caps", default="48000,240000", help="no-tool caps to compare against with CIs")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-salvage", action="store_true", help="count salvage answers as wrong (v2-equivalent)")
    args = ap.parse_args()
    global NO_SALVAGE
    NO_SALVAGE = args.no_salvage
    rows, all_pairs = [], None
    for path in args.runs:
        row, pairs, correct = tool_row(path)
        rows.append((row, correct))
        all_pairs = pairs if all_pairs is None else all_pairs & pairs
    print("| run | attempts | pass@1 | pass@2 | tok mean / median | correct median | turns | calls | parse fail | "
          "tool err | sources | stops | s/attempt |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row, _ in rows:
        print(f"| {row['run']} | {row['attempts']} ({row['errors']} err) | {row['pass@1']:.1%} | {row['pass@2']:.1%} | "
              f"{row['tok_mean'] / 1000:.1f}k / {row['tok_median'] / 1000:.1f}k | {row['tok_correct_median'] / 1000:.1f}k | "
              f"{row['turns']:.1f} | {row['calls']:.1f} | {row['parse_fail']} | {row['tool_err']} | {row['sources']} | "
              f"{row['stops']} | {row['seconds_mean']:.0f} |")
    print()
    print(f"No-tool Qwen xhigh (240k run, simulated caps) on the {len(all_pairs)} pairs common to all runs:")
    print("| cap | pass@1 | pass@2 | tok mean / median |")
    print("|---|---|---|---|")
    for cap in map(int, args.caps.split(",")):
        row, _ = notool_row(all_pairs, cap)
        print(f"| {cap // 1000}k | {row['pass@1']:.1%} | {row['pass@2']:.1%} | {row['tok_mean'] / 1000:.1f}k / "
              f"{row['tok_median'] / 1000:.1f}k |")
    _, nt = notool_row(all_pairs, 240000)
    print()
    print("Paired per attempt vs no-tool 240k (same pair, same attempt index): both / tools only / no-tool only / neither")
    for row, correct in rows:
        keys = [k for k in correct if (k[0], k[1]) in all_pairs]
        c = Counter((correct[k], nt.get(k, False)) for k in keys)
        print(f"- {row['run']}: {c[(True, True)]} / {c[(True, False)]} / {c[(False, True)]} / {c[(False, False)]}")
    print()
    print("Paired diffs, bootstrap over tasks (pass@1 diff [95% CI]; pass@2 diff [95% CI]), on the common pairs:")
    refs = [(f"no-tool {cap // 1000}k", notool_row(all_pairs, cap)[1]) for cap in map(int, args.ci_caps.split(","))]
    refs += [(rows[0][0]["run"], rows[0][1])] if len(rows) > 1 else []
    for row, correct in rows:
        for name, ref in refs:
            if name == row["run"]:
                continue
            (a, lo, hi), (b, lo2, hi2) = paired_ci(correct, ref, all_pairs)
            print(f"- {row['run']} vs {name}: {a:+.1%} [{lo:+.1%}, {hi:+.1%}]; {b:+.1%} [{lo2:+.1%}, {hi2:+.1%}]")
    print()
    print("Token waste and simulated lower caps (an attempt counts as correct only if it finished correct within the cap;"
          " its cost becomes min(tokens, cap)). Same behaviour before the cap is assumed, so this is approximate:")
    for row, correct in rows:
        recs = {k: r for k, r in load([p for p in args.runs if Path(p).stem.endswith(row['run'])][0]).items()
                if "error" not in r and (k[0], k[1]) in all_pairs}
        if NO_SALVAGE:
            for r in recs.values():
                if r.get("answer_source") == "salvage":
                    r["correct"], r["stop_reason"] = False, r["stop_reason"]
        tot = sum(r["usage"]["completion"] for r in recs.values())
        cap_hit = [r for r in recs.values() if r["stop_reason"] in ("output_budget", "unclosed_think")]
        wrong = [r for r in recs.values() if not r["correct"]]
        print(f"- {row['run']}: {len(cap_hit)}/{len(recs)} attempts end at the cap or unclosed "
              f"({sum(r['correct'] for r in cap_hit)} correct), {sum(r['usage']['completion'] for r in cap_hit) / tot:.0%} of tokens;"
              f" wrong attempts {sum(r['usage']['completion'] for r in wrong) / tot:.0%} of tokens")
        for cap in (16000, 24000, 32000, 40000):
            c = {k: r["correct"] and r["usage"]["completion"] <= cap for k, r in recs.items()}
            p1, p2 = scores(c, all_pairs)
            mean = statistics.mean(min(r["usage"]["completion"], cap) for r in recs.values())
            print(f"    cap {cap // 1000}k: pass@1 {p1:.1%}, pass@2 {p2:.1%}, {mean / 1000:.1f}k tokens/attempt")
    if args.json:
        print(json.dumps([r for r, _ in rows], indent=1))


if __name__ == "__main__":
    main()
