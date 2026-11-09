"""Absolute pass@2 / pass@1 and output-token stats for every Flash-1017 run on the public eval (tools and no tools).
Task-weighted (ARC Prize). pass@1 = mean of attempts 0 and 1. Partial runs: also scored on their fully finished tasks,
next to E034 on the same tasks."""
import json, collections, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[3] / "evals/results"
TOOL = "tools_persistent_grid_{}_deepseek-flash_low_{}.jsonl"
RUNS = [  # (label, file, kind)
    ("E029 sub80s40", TOOL.format("lean2sub", "e029_flash1017_sub80s40"), "tools"),
    ("E030 sub80s40 + header", TOOL.format("lean2sub", "e030_hdrmin"), "tools"),
    ("E029 sub80s40 lean2code", TOOL.format("lean2code", "e029_flash1017_sub80s40_code"), "tools"),
    ("E030 cap16", TOOL.format("lean2sub", "e030_cap16"), "tools"),
    ("E030 s20x80", TOOL.format("lean2sub", "e030_s20x80"), "tools"),
    ("E030 cap12", TOOL.format("lean2sub", "e030_cap12"), "tools"),
    ("E030 s15x80", TOOL.format("lean2sub", "e030_s15x80"), "tools"),
    ("E030 s20x60 + header", TOOL.format("lean2sub", "e030_s20x60_hdrmin"), "tools"),
    ("E029 s20x60", TOOL.format("lean2sub", "e029_flash1017_s20x60"), "tools"),
    ("E034 max", "arcprize_deepseek-flash_max_e034_flash1017.jsonl", "none"),
    ("E035 A max + stats (partial)", "arcprize_deepseek-flash_max_e035_sum.jsonl", "none"),
    ("E035 C low + stats", "arcprize_deepseek-flash_low_e035_sum.jsonl", "none"),
    ("E035 D low + header + stats", "arcprize_deepseek-flash_low_e035_hdrsum.jsonl", "none"),
    ("E035 E low + header", "arcprize_deepseek-flash_low_e035_hdr.jsonl", "none"),
    ("E035 B low (partial)", "arcprize_deepseek-flash_low_e035_plain.jsonl", "none"),
]
TASKS = {p.stem: len(json.load(open(p))["test"]) for p in sorted((R.parents[1] / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}

def load(f):
    recs = {}
    for r in map(json.loads, open(R / f)):
        if "error" in r or r.get("attempt", 0) > 1:
            continue
        recs[(r["task_id"], r["pair"], r["attempt"])] = r  # last record wins
    return recs

def tok(r):
    u = r.get("usage") or {}
    return u.get("completion", u.get("completion_tokens", 0))

def score(recs, tasks):
    p2 = p1 = 0
    for t in tasks:
        n = TASKS[t]
        p2 += sum(any(recs.get((t, p, a), {}).get("correct") for a in (0, 1)) for p in range(n)) / n
        p1 += sum(bool(recs.get((t, p, a), {}).get("correct")) for p in range(n) for a in (0, 1)) / (2 * n)
    return 100 * p2 / len(tasks), 100 * p1 / len(tasks)

def complete(recs):
    return [t for t, n in TASKS.items() if all((t, p, a) in recs for p in range(n) for a in (0, 1))]

base = load("arcprize_deepseek-flash_max_e034_flash1017.jsonl")
print("| Run | Tools | Calls | pass@2 | pass@1 | Out tokens mean / median / p90 / max | Correct-call median |")
print("|---|---|---|---|---|---|---|")
for label, f, kind in RUNS:
    recs = load(f)
    ts = [tok(r) for r in recs.values()]
    tc = [tok(r) for r in recs.values() if r.get("correct")]
    q = st.quantiles(ts, n=10)
    tokstr = f"{st.mean(ts)/1e3:.1f}k / {st.median(ts)/1e3:.1f}k / {q[8]/1e3:.1f}k / {max(ts)/1e3:.1f}k"
    full = complete(recs)
    if len(full) == len(TASKS):
        p2, p1 = score(recs, TASKS)
        s = f"{p2:.1f}% | {p1:.1f}%"
    else:
        p2, p1 = score(recs, full); b2, b1 = score(base, full)
        s = f"{p2:.1f}% (E034 {b2:.1f}%) | {p1:.1f}% (E034 {b1:.1f}%)"
        label += f", {len(full)} tasks"
    print(f"| {label} | {kind} | {len(recs)} | {s} | {tokstr} | {st.median(tc)/1e3:.1f}k |")
