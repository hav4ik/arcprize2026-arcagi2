"""E012 final numbers: medium 48k/24k full set (v2 dev file + v3 remainder), vs no-tool Qwen; strong-wording dev.

    .venv/bin/python experiments/E012_qwen_tools/final_stats.py
"""
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "evals/results"
FULL = [R / "qwen_tools_medium_48k_dev_s24.jsonl", R / "qwen_tools_medium_48k_full_v3.jsonl"]
STRONG = R / "qwen_tools_medium_48k_dev_s24strong.jsonl"
NOTOOL = R / "qwen_nvfp4_gen_xhigh_240k_t1.jsonl"
DEV = {(t, int(p)) for t, p in (x.split(":") for x in
       (ROOT / "experiments/E009_tool_token_efficiency/dev_pairs.txt").read_text().strip().split(","))}
PILOT = {p.stem for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))[:15]}


def load(paths):
    out = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    out[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return out


def scores(c, pairs):
    """(task-weighted pass@1, pass@2, pair-weighted pass@1, pass@2)."""
    tasks = {}
    for t, p in pairs:
        tasks.setdefault(t, []).append(p)
    p1 = lambda t, p: (c.get((t, p, 0), False) + c.get((t, p, 1), False)) / 2  # noqa: E731
    p2 = lambda t, p: c.get((t, p, 0), False) or c.get((t, p, 1), False)  # noqa: E731
    tw = [statistics.mean(statistics.mean(f(t, p) for p in ps) for t, ps in tasks.items()) for f in (p1, p2)]
    pw = [statistics.mean(f(t, p) for t, p in pairs) for f in (p1, p2)]
    return tw + pw


def boot(ca, cb, pairs, n=4000):
    tasks = {}
    for t, p in pairs:
        tasks.setdefault(t, []).append(p)
    ids = sorted(tasks)
    d = {t: [statistics.mean((ca.get((t, p, a), False) - cb.get((t, p, a), False)) for a in (0, 1))
             for p in tasks[t]] for t in ids}
    d = {t: statistics.mean(v) for t, v in d.items()}
    rng = random.Random(0)
    bs = sorted(statistics.mean(d[rng.choice(ids)] for _ in ids) for _ in range(n))
    return statistics.mean(d.values()), bs[int(.025 * n)], bs[int(.975 * n)]


def main():
    full = load(FULL)
    nt = load([NOTOOL])
    pairs = sorted({(t, p) for t, p, _ in full if (t, p, 0) in full and (t, p, 1) in full})
    print(f"full: {len(full)} attempts, {len(pairs)} pairs with both attempts, {len({t for t, _ in pairs})} tasks")
    recs = [full[(t, p, a)] for t, p in pairs for a in (0, 1)]
    toks = sorted(r["usage"]["completion"] for r in recs)
    v3 = {k: r["correct"] for k, r in full.items()}
    v2 = {k: r["correct"] and r.get("answer_source") != "salvage" for k, r in full.items()}
    capped = [r for r in recs if r["stop_reason"] in ("output_budget", "unclosed_think", "context")]
    rows = [("tools medium 48k/24k (v3, with salvage)", v3, statistics.mean(toks), statistics.median(toks),
             toks[int(.9 * len(toks))]),
            ("tools medium 48k/24k (v2-equivalent)", v2, statistics.mean(toks), statistics.median(toks),
             toks[int(.9 * len(toks))])]
    for cap in (24000, 36000, 48000, 240000):
        c = {k: r["correct"] and r["completion_tokens"] <= cap for k, r in nt.items()}
        t = sorted(min(nt[(tt, p, a)]["completion_tokens"], cap) for tt, p in pairs for a in (0, 1))
        rows.append((f"no-tool xhigh, cap {cap // 1000}k (simulated)", c, statistics.mean(t), statistics.median(t),
                     t[int(.9 * len(t))]))
    print("| config | pass@1 | pass@2 | pair-weighted pass@1 / pass@2 | tokens mean / median / p90 |")
    print("|---|---|---|---|---|")
    for name, c, mean, med, p90 in rows:
        a, b, pa, pb = scores(c, pairs)
        print(f"| {name} | {a:.1%} | {b:.1%} | {pa:.1%} / {pb:.1%} | {mean / 1000:.1f}k / {med / 1000:.1f}k / "
              f"{p90 / 1000:.1f}k |")
    print()
    for name, c, *_ in rows[2:]:
        d, lo, hi = boot(v3, c, pairs)
        print(f"- tools vs {name}: pass@1 {d:+.1%} [{lo:+.1%}, {hi:+.1%}]")
    print()
    print(f"cap-hit/unclosed: {len(capped)}/{len(recs)} attempts ({len(capped) / len(recs):.0%}), "
          f"{sum(r['usage']['completion'] for r in capped) / sum(toks):.0%} of tokens, "
          f"{sum(r['correct'] for r in capped)} correct")
    print("stops", dict(Counter(r["stop_reason"] for r in recs)))
    print("answer sources", dict(Counter(r.get("answer_source") for r in recs)),
          "correct by source", dict(Counter(r.get("answer_source") for r in recs if r["correct"])))
    print("versions", dict(Counter(r.get("harness_version", "qwen-tools-2") for r in recs)),
          "salvaged", sum(r.get("answer_source") == "salvage" for r in recs),
          "salvage correct", sum(r.get("answer_source") == "salvage" and r["correct"] for r in recs))
    print("turns mean", round(statistics.mean(r["turns"] for r in recs), 1), "parse failures",
          sum(r["parse_failures"] for r in recs), "tool errors/attempt",
          round(statistics.mean(r["tool_errors"] for r in recs), 2), "eos continues",
          sum(r["eos_continues"] for r in recs), "executor resets", sum(r.get("executor_resets", 0) for r in recs))
    print("seconds/attempt mean", round(statistics.mean(r["seconds"] for r in recs)),
          "max_prompt p50/p90/max", statistics.median(r["usage"]["max_prompt"] for r in recs),
          sorted(r["usage"]["max_prompt"] for r in recs)[int(.9 * len(recs))], max(r["usage"]["max_prompt"] for r in recs))
    ok = sorted(r["usage"]["completion"] for r in recs if r["correct"])
    print("correct-rollout tokens p50/p90", statistics.median(ok), ok[int(.9 * len(ok))])
    for name, sub in (("dev", DEV), ("non-dev", None), ("pilot", PILOT)):
        sel = [x for x in pairs if (x in DEV if name == "dev" else x not in DEV if name == "non-dev" else x[0] in PILOT)]
        a, b, _, _ = scores(v3, sel)
        n48 = scores({k: r["correct"] and r["completion_tokens"] <= 48000 for k, r in nt.items()}, sel)
        n240 = scores({k: r["correct"] for k, r in nt.items()}, sel)
        print(f"{name} ({len(sel)} pairs): tools {a:.1%} / {b:.1%}; no-tool 48k {n48[0]:.1%} / {n48[1]:.1%}; "
              f"no-tool 240k {n240[0]:.1%} / {n240[1]:.1%}")
    # strong wording vs gentle, paired per attempt on dev
    strong = load([STRONG])
    ks = [k for k in strong if k in full]
    s_ok, g_ok = sum(strong[k]["correct"] for k in ks), sum(full[k]["correct"] for k in ks)
    tasks = sorted({k[0] for k in ks})
    d = {t: statistics.mean(strong[k]["correct"] - full[k]["correct"] for k in ks if k[0] == t) for t in tasks}
    rng = random.Random(0)
    bs = sorted(statistics.mean(d[rng.choice(tasks)] for _ in tasks) for _ in range(4000))
    print(f"\nstrong vs gentle (dev, {len(ks)} paired attempts): {s_ok} vs {g_ok} correct; tokens "
          f"{statistics.mean(strong[k]['usage']['completion'] for k in ks) / 1000:.1f}k vs "
          f"{statistics.mean(full[k]['usage']['completion'] for k in ks) / 1000:.1f}k; cap hits "
          f"{sum(strong[k]['stop_reason'] != 'submitted' for k in ks)} vs {sum(full[k]['stop_reason'] != 'submitted' for k in ks)}; "
          f"discordant {sum(strong[k]['correct'] > full[k]['correct'] for k in ks)} vs "
          f"{sum(strong[k]['correct'] < full[k]['correct'] for k in ks)}; task-weighted per-attempt diff "
          f"{statistics.mean(d.values()):+.1%} [{bs[100]:+.1%}, {bs[3900]:+.1%}]")


if __name__ == "__main__":
    sys.exit(main())
