"""E017 phase 2: compare variant runs with sub80s40 (3 samples) on the full public eval. No API calls.

Kaggle metric = pass@1 over the 167 test pairs (pair-weighted), with a cluster bootstrap over tasks for the paired
difference against the mean of sub80s40's 3 samples; the task-weighted (ARC Prize) version is also printed.
Plus pass@2 (variants with 2 samples; the baseline averaged over its 3 attempt pairs), pass@2 with the train-check
ranking and the conditional 3rd attempt, tokens, cost, train-check precision, and mechanism checks from the traces.

    .venv/bin/python -I experiments/E017_reasoning_efficiency/scripts/phase2_compare.py NAME=results.jsonl ... [--json out]
"""
import argparse
import gzip
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
from train_check import ranked_answers  # noqa: E402

RES = ROOT / "evals/results"
BASE = RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl"
TASKS = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}
PAIRS = [(t, p) for t, task in TASKS.items() for p in range(len(task["test"]))]
DIGIT_LINE = re.compile(r"^[\s\[\],0-9.|-]*\d[\s\[\],0-9.|-]*$")  # as scripts/index_attempts.py
ROLES = re.compile(r"\bROLES?\b")
AUDIT = re.compile(r"(test[_ ]?input|the test|test grid|test case)[^\n]{0,120}?(differ|new |unseen|not seen|outside|"
                   r"novel|beyond|unlike|clipped)|(differ|difference)[^\n]{0,60}?(test[_ ]?input|the test)", re.I)
NOISE = re.compile(r"\b(noise|irrelevant|distractor)", re.I)


def latest(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return {k: r for k, r in out.items() if "error" not in r}


def ok(recs, t, p, a):
    return bool(recs.get((t, p, a), {}).get("correct"))


def boot(task_vals, n=10000, seed=0):
    """task_vals: {task: (sum of per-pair values, number of pairs)} -> pair-weighted mean and 95% CI."""
    rng = random.Random(seed)
    items = list(task_vals.values())
    mean = sum(s for s, _ in items) / sum(k for _, k in items)
    means = []
    for _ in range(n):
        smp = rng.choices(items, k=len(items))
        means.append(sum(s for s, _ in smp) / sum(k for _, k in smp))
    means.sort()
    return mean, means[int(0.025 * n)], means[int(0.975 * n)]


def boot_task(task_vals, n=10000, seed=0):
    rng = random.Random(seed)
    vals = [s / k for s, k in task_vals.values()]
    means = sorted(st.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return st.mean(vals), means[int(0.025 * n)], means[int(0.975 * n)]


def pair_rates(recs, samples):
    return {(t, p): st.mean(ok(recs, t, p, a) for a in samples) for t, p in PAIRS}


def pass2_rates(recs, combos):
    return {(t, p): st.mean(ok(recs, t, p, a) or ok(recs, t, p, b) for a, b in combos) for t, p in PAIRS}


def by_task(rates):
    out = defaultdict(lambda: [0.0, 0])
    for (t, _), v in rates.items():
        out[t][0] += v
        out[t][1] += 1
    return {t: tuple(v) for t, v in out.items()}


def diff_rates(a, b):
    return {k: a[k] - b[k] for k in a}


def trace_stats(r):
    """Mechanism checks for one attempt from its saved trace."""
    try:
        with gzip.open(r["trace_path"], "rt") as f:
            msgs = json.load(f)["messages"]
    except Exception:
        return None
    turns = [m for m in msgs if m["role"] == "assistant"]
    grid_lines, text, code = 0, [], []
    for m in turns:
        rc = m.get("reasoning_content") or ""
        grid_lines += sum(1 for l in rc.splitlines()
                          if len(l.strip()) >= 7 and DIGIT_LINE.match(l) and sum(ch.isdigit() for ch in l) >= 5)
        text.append(rc + "\n" + (m.get("content") or ""))
        for c in m.get("tool_calls") or []:
            try:
                code.append(json.loads(c["function"]["arguments"]).get("code", ""))
            except Exception:
                pass
    alltext, allcode = "\n".join(text), "\n".join(code)
    return {"grid_lines": grid_lines, "roles": bool(ROLES.search(alltext)),
            "audit": bool(AUDIT.search(alltext + "\n" + allcode)), "noise": bool(NOISE.search(alltext)),
            "check_calls": len(re.findall(r"\bcheck\(", allcode)),
            "uses_helpers": bool(re.search(r"\b(objects|match_d8|panels|stamp|ray|d8|crop)\(", allcode)),
            "summary_ref": bool(re.search(r"\bsummary\b", alltext, re.I))}


def run_stats(name, recs, samples, policy=False, mech=True):
    attempts = [r for k, r in recs.items() if k[2] in samples]
    toks = [r["usage"]["completion"] for r in attempts]
    corr = sorted(r["usage"]["completion"] for r in attempts if r.get("correct"))
    out = {"name": name, "samples": len(samples), "attempts": len(attempts),
           "missing": len(PAIRS) * len(samples) - len(attempts),
           "tok_mean": st.mean(toks), "tok_median": st.median(toks),
           "correct_p50": corr[len(corr) // 2] if corr else None, "correct_p90": corr[int(0.9 * len(corr))] if corr else None,
           "cost": sum(r.get("est_cost_usd", 0) for r in recs.values()),
           "cost_peak": sum(r.get("est_cost_peak_usd", 0) or 0 for r in recs.values()),
           "no_answer": sum(r.get("pred") is None for r in attempts)}
    out["per_sample_pair_w"] = [st.mean(ok(recs, t, p, a) for t, p in PAIRS) for a in samples]
    labelled = [r for r in attempts if "train_pass" in r]
    if labelled:
        tp = [r for r in labelled if r["train_pass"] is True]
        tf = [r for r in labelled if r["train_pass"] is False]
        out["train_check"] = {"n": len(labelled), "train_pass": len(tp),
                              "p_correct_given_pass": st.mean(bool(r.get("correct")) for r in tp) if tp else None,
                              "p_correct_given_fail": st.mean(bool(r.get("correct")) for r in tf) if tf else None,
                              "untraceable": sum(r.get("train_trace") in ("untraceable", "check_error") for r in labelled),
                              "traces": {k: sum(r.get("train_trace") == k for r in labelled)
                                         for k in ("expr", "cell", "fn", "transcript", "untraceable", "check_error", "no_answer")}}
    if mech:
        ms = [m for m in (trace_stats(r) for r in attempts) if m]
        if ms:
            out["mech"] = {"n": len(ms), "grid_lines_mean": st.mean(m["grid_lines"] for m in ms),
                           "grid_ge10": st.mean(m["grid_lines"] >= 10 for m in ms),
                           "grid_ge40": st.mean(m["grid_lines"] >= 40 for m in ms),
                           "roles": st.mean(m["roles"] for m in ms), "audit": st.mean(m["audit"] for m in ms),
                           "noise_word": st.mean(m["noise"] for m in ms),
                           "check_calls_mean": st.mean(m["check_calls"] for m in ms),
                           "uses_helpers": st.mean(m["uses_helpers"] for m in ms),
                           "summary_ref": st.mean(m["summary_ref"] for m in ms)}
    return out


def longest_turns(recs, samples):
    """Longest single-turn output per attempt (from the trace events)."""
    vals = []
    for k, r in recs.items():
        if k[2] not in samples:
            continue
        try:
            with gzip.open(r["trace_path"], "rt") as f:
                ev = json.load(f)["events"]
        except Exception:
            continue
        vals.append(max([e["usage"].get("completion_tokens", 0) for e in ev if "usage" in e] or [0]))
    vals.sort()
    return {"median": st.median(vals), "p90": vals[int(0.9 * len(vals))], "ge30k": st.mean(v >= 30000 for v in vals),
            "lt10k": st.mean(v < 10000 for v in vals)} if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path.jsonl")
    ap.add_argument("--json")
    ap.add_argument("--no-mech", action="store_true")
    args = ap.parse_args()
    base = latest(BASE)
    base_samples = [0, 1, 2]
    b1 = pair_rates(base, base_samples)
    b2 = pass2_rates(base, [(0, 1), (0, 2), (1, 2)])
    report = {"baseline": run_stats("sub80s40", base, base_samples, mech=not args.no_mech)}
    report["baseline"]["pass1_pair_w"] = st.mean(b1.values())
    report["baseline"]["pass1_task_w"] = st.mean(s / k for s, k in by_task(b1).values())
    report["baseline"]["pass2_pair_w"] = st.mean(b2.values())
    report["baseline"]["pass2_task_w"] = st.mean(s / k for s, k in by_task(b2).values())
    report["baseline"]["longest_turn"] = longest_turns(base, base_samples)
    rows = []
    for spec in args.runs:
        name, path = spec.split("=", 1)
        recs = latest(path)
        samples = sorted({a for (_, _, a), r in recs.items() if not r.get("policy_extra")})
        s = run_stats(name, recs, samples, mech=not args.no_mech)
        v1 = pair_rates(recs, samples)
        d = diff_rates(v1, b1)
        s["pass1_pair_w"] = st.mean(v1.values())
        s["pass1_task_w"] = st.mean(a / k for a, k in by_task(v1).values())
        s["diff1_pair_w"] = boot(by_task(d))
        s["diff1_task_w"] = boot_task(by_task(d))
        if len(samples) >= 2:
            v2 = pass2_rates(recs, [(0, 1)])
            s["pass2_pair_w"] = st.mean(v2.values())
            s["pass2_task_w"] = st.mean(a / k for a, k in by_task(v2).values())
            s["diff2_pair_w"] = boot(by_task(diff_rates(v2, b2)))
            if any("train_pass" in r for r in recs.values()):
                pol = {}
                for t, p in PAIRS:
                    rs = [recs[t, p, a] for a in (0, 1, 2) if (t, p, a) in recs]
                    pol[t, p] = float(any(r.get("correct") for r in ranked_answers(rs)))
                s["pass2_policy_pair_w"] = st.mean(pol.values())
                s["pass2_policy_task_w"] = st.mean(a / k for a, k in by_task(pol).values())
                s["diff2_policy_pair_w"] = boot(by_task(diff_rates(pol, b2)))
                s["diff2_policy_vs_own_pass2"] = boot(by_task(diff_rates(pol, v2)))
                extra = [r for r in recs.values() if r.get("policy_extra")]
                s["policy_extra"] = {"n": len(extra), "correct": sum(bool(r.get("correct")) for r in extra),
                                     "cost": sum(r.get("est_cost_usd", 0) for r in extra),
                                     "tok_mean": st.mean(r["usage"]["completion"] for r in extra) if extra else None}
        s["longest_turn"] = longest_turns(recs, samples)
        report[name] = s
        rows.append(s)
    b = report["baseline"]
    print(f"baseline sub80s40: pass@1 pair-w {b['pass1_pair_w']:.1%} task-w {b['pass1_task_w']:.1%}; "
          f"pass@2 {b['pass2_pair_w']:.1%} / {b['pass2_task_w']:.1%}; tok {b['tok_mean'] / 1e3:.1f}k")
    print("| run | samples | pass@1 pair-w | Δ vs sub80s40 [95% CI] | pass@1 task-w | Δ task-w [CI] | pass@2 | "
          "Δ pass@2 [CI] | pass@2 policy | tok/attempt mean/median | $ (peak part) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        d1, dt = s["diff1_pair_w"], s["diff1_task_w"]
        p2 = f"{s['pass2_pair_w']:.1%}" if "pass2_pair_w" in s else "–"
        d2 = (f"{100 * s['diff2_pair_w'][0]:+.1f} [{100 * s['diff2_pair_w'][1]:+.1f}, {100 * s['diff2_pair_w'][2]:+.1f}]"
              if "diff2_pair_w" in s else "–")
        pp = f"{s['pass2_policy_pair_w']:.1%}" if "pass2_policy_pair_w" in s else "–"
        print(f"| {s['name']} | {s['samples']} | {s['pass1_pair_w']:.1%} ({' / '.join(f'{x:.1%}' for x in s['per_sample_pair_w'])}) | "
              f"{100 * d1[0]:+.1f} [{100 * d1[1]:+.1f}, {100 * d1[2]:+.1f}] | {s['pass1_task_w']:.1%} | "
              f"{100 * dt[0]:+.1f} [{100 * dt[1]:+.1f}, {100 * dt[2]:+.1f}] | {p2} | {d2} | {pp} | "
              f"{s['tok_mean'] / 1e3:.1f}k / {s['tok_median'] / 1e3:.1f}k | ${s['cost']:.2f} (${s['cost_peak']:.2f}) |")
    print("\nmechanism | grid lines mean | ≥10 | ≥40 | longest turn median / p90 | max turn ≥30k | ROLES | test audit | "
          "noise word | check() calls | helpers used | summary ref | no answer | correct p50/p90")
    for s in [b] + rows:
        m, lt = s.get("mech", {}), s.get("longest_turn") or {}
        if m:
            print(f"{s['name']} | {m['grid_lines_mean']:.1f} | {m['grid_ge10']:.0%} | {m['grid_ge40']:.0%} | "
                  f"{lt.get('median', 0) / 1e3:.1f}k / {lt.get('p90', 0) / 1e3:.1f}k | {lt.get('ge30k', 0):.0%} | "
                  f"{m['roles']:.0%} | {m['audit']:.0%} | {m['noise_word']:.0%} | {m['check_calls_mean']:.1f} | "
                  f"{m['uses_helpers']:.0%} | {m['summary_ref']:.0%} | {s['no_answer']} | "
                  f"{(s['correct_p50'] or 0) / 1e3:.1f}k / {(s['correct_p90'] or 0) / 1e3:.1f}k")
    for s in rows:
        if "train_check" in s:
            tc = s["train_check"]
            print(f"{s['name']} train-check: {tc['train_pass']}/{tc['n']} pass; P(correct|pass) "
                  f"{tc['p_correct_given_pass']:.0%}, P(correct|fail) {tc['p_correct_given_fail'] or 0:.0%}; {tc['traces']}")
        if "policy_extra" in s:
            pe = s["policy_extra"]
            print(f"{s['name']} policy: {pe['n']} extra attempts, {pe['correct']} correct, ${pe['cost']:.2f}; "
                  f"pass@2 {s['pass2_pair_w']:.1%} -> with policy {s['pass2_policy_pair_w']:.1%} "
                  f"(Δ {100 * s['diff2_policy_vs_own_pass2'][0]:+.1f} [{100 * s['diff2_policy_vs_own_pass2'][1]:+.1f}, "
                  f"{100 * s['diff2_policy_vs_own_pass2'][2]:+.1f}]); task-w {s['pass2_policy_task_w']:.1%}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
