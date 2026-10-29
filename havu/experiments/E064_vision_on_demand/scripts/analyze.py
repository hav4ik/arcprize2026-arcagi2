#!/usr/bin/env python3
"""E064: vision-on-demand arms vs their baseline on the same tasks (task-weighted pass@1 / pass@2, paired bootstrap
over tasks via E058 compare.py), plus image use: show_grid calls per attempt, how often the image check fired, and
whether the model changed its answer after seeing it (and whether that helped).
Usage (repo root): analyze.py"""
import json, statistics as st, subprocess, sys
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
CMP = "experiments/E058_deepseek_vision/compare.py"
PAIRS = [  # (arm, baseline): DeepSeek API arms vs E058's image arm; DeepInfra arms vs the DeepInfra E058 replicate
    ("e064_show", "e058_pilot30"), ("e064_check", "e058_pilot30"), ("e058_pilot30", "e046p_note_all"),
    ("e064or_strict", "e064or_base"), ("e064or_lazy", "e064or_base"), ("e064or_lazy", "e046p_note_all")]


def load(tag):
    d = {}
    for r in map(json.loads, open(R + tag + ".jsonl")):
        if "task_id" in r and not r.get("error"):
            d[r["task_id"], r["pair"], r["attempt"]] = r
    return d


for arm, base in PAIRS:
    out = subprocess.run([sys.executable, CMP, R + arm + ".jsonl", R + base + ".jsonl"], capture_output=True, text=True)
    print(f"== {arm} vs {base}\n{out.stdout.strip() or out.stderr.strip()[-500:]}")
print()
for tag in ("e064_show", "e064_check", "e064or_base", "e064or_strict", "e064or_lazy"):
    try:
        d = load(tag)
    except FileNotFoundError:
        continue
    rs = [r for r in d.values() if not r.get("multi_share")]
    line = f"{tag:14} attempts {len(rs):3d}  out tok {st.mean(r['usage']['completion'] for r in rs)/1e3:5.1f}k"
    line += f"  prompt tok {st.mean(r['usage']['prompt'] for r in rs)/1e3:6.0f}k  ${sum(r.get('est_cost_usd') or 0 for r in rs):.2f}"
    if any("shows" in r for r in rs):
        sh = [r.get("shows", 0) for r in rs]
        line += f"  show_grid: {st.mean(sh):.2f}/attempt, used in {100*st.mean(x > 0 for x in sh):.0f}%"
    if any("image_check" in r for r in rs):
        ck = [r for r in rs if r.get("image_check")]
        ch = [r for r in ck if r["image_check"]["changed"]]
        # changed answers: right after the change? (pair-level correctness of the final answer, attempt records)
        fixed = sum(bool(d[r["task_id"], r["pair"], r["attempt"]]["correct"]) for r in ch)
        line += (f"  check fired {100*len(ck)/len(rs):.0f}%, answer changed after it {len(ch)} "
                 f"(final correct {fixed}), never resubmitted {sum(r['image_check']['source'] == 'image_check' for r in ck)}")
    print(line)
