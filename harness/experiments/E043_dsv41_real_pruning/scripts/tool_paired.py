#!/usr/bin/env python3
"""E043: tool-harness runs compared on exactly the attempts the first run has finished (for interim decisions).

  tool_paired.py e043_e112t_w10_u2_gptq_eng64_sub80 e043_e128t_w10_u2_gptq_eng64_sub80 [more tags ...]

- pass@1 on the same attempts: attempt accuracy, plain and task-weighted (attempts averaged within a task first);
- pass@2 on the pairs where the first run has both attempts: pair-level and task-weighted over those pairs;
- mean output tokens and turns per attempt; Python-error share of tool calls (from the traces); 40-turn and budget stops;
- one-sided wins of the first run vs each other run on the same attempts; --token-exact counters of the first run.
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../evals/results")
PYERR = re.compile(r"Traceback|\w+Error:|\w+Exception:")


def load(tag):
    fs = glob.glob(os.path.join(RES, f"tools_*_{tag}.jsonl"))
    assert len(fs) == 1, (tag, fs)
    rows = {}
    for line in open(fs[0]):
        r = json.loads(line)
        if "task_id" in r:
            rows[(r["task_id"], r["pair"], r["attempt"])] = r
    return {k: r for k, r in rows.items() if "error" not in r}


def py_errors(r):
    p = r.get("trace_path")
    if not p or not os.path.exists(p):
        return 0, 0
    t = json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))
    msgs = t.get("messages", t) if isinstance(t, dict) else t
    tools = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
    return len(tools), sum(bool(PYERR.search(m.get("content") or "")) for m in tools)


def task_weighted(keys, ok):
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append(ok(k))
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)


def main():
    tags = sys.argv[1:]
    runs = {t: load(t) for t in tags}
    first = runs[tags[0]]
    keys = sorted(k for k in first if all(k in runs[t] for t in tags))
    pairs = sorted({k[:2] for k in keys if (k[0], k[1], 0) in first and (k[0], k[1], 1) in first
                    and all((k[0], k[1], a) in runs[t] for t in tags for a in (0, 1))})
    print(f"same attempts: {len(keys)} (first run has {len(first)} finished); pairs with both attempts: {len(pairs)}")
    print(f"{'run':40s} {'pass@1':>14s} {'pass@1 tw':>9s} {'pass@2':>15s} {'pass@2 tw':>9s} {'out tok':>8s} "
          f"{'turns':>6s} {'py-err':>7s} {'cap40':>5s} {'budget':>6s}")
    for t in tags:
        d = runs[t]
        ok = lambda k: bool(d[k].get("correct"))
        c = sum(ok(k) for k in keys)
        p2 = lambda pr: ok((*pr, 0)) or ok((*pr, 1))
        p2n = sum(p2(pr) for pr in pairs)
        p2tw = task_weighted([(*pr, 0) for pr in pairs], lambda k: p2(k[:2])) if pairs else float("nan")
        tok = sum((d[k].get("usage") or {}).get("completion", 0) for k in keys) / len(keys)
        turns = sum(d[k].get("turns", 0) for k in keys) / len(keys)
        calls = errs = 0
        for k in keys:
            a, b = py_errors(d[k])
            calls, errs = calls + a, errs + b
        cap = sum(d[k].get("stop_reason") == "max_turns" for k in keys)
        bud = sum(d[k].get("stop_reason") == "output_budget" for k in keys)
        print(f"{t:40s} {c:4d}/{len(keys)} {100 * c / len(keys):5.1f}% {task_weighted(keys, ok):8.1f}% "
              f"{p2n:4d}/{len(pairs)} {100 * p2n / max(1, len(pairs)):5.1f}% {p2tw:8.1f}% {tok / 1e3:7.1f}K "
              f"{turns:6.1f} {100 * errs / max(1, calls):6.1f}% {cap:5d} {bud:6d}")
    for t in tags[1:]:
        a = sum(bool(first[k].get("correct")) and not runs[t][k].get("correct") for k in keys)
        b = sum(bool(runs[t][k].get("correct")) and not first[k].get("correct") for k in keys)
        print(f"one-sided wins on the same attempts: {tags[0]} {a} vs {t} {b}")
    te = [first[k]["token_exact"] for k in first if "token_exact" in first[k]]
    if te:
        s = {f: sum(x[f] for x in te) for f in ("turns", "breaks", "rerenders", "parse_fallbacks")}
        print(f"--token-exact ({len(te)} attempts): {s['turns']} turns, re-tokenization breaks avoided {s['breaks']} "
              f"({100 * s['breaks'] / max(1, s['turns']):.1f}% of turns), re-renders {s['rerenders']}, "
              f"parse fallbacks {s['parse_fallbacks']}")


if __name__ == "__main__":
    main()
