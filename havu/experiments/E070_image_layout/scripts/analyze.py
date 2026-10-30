#!/usr/bin/env python3
"""E070: image-layout arms vs E058's per-grid images on the DeepSeek API (task-weighted pass@1 / pass@2, paired
bootstrap over tasks via E058 compare.py; same tasks only), and against each other, plus tokens and cost.
Usage (repo root): analyze.py"""
import json, statistics as st, subprocess, sys
R = "evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_"
CMP = "experiments/E058_deepseek_vision/compare.py"
PAIRS = [("e070_pair", "e058_pilot30"), ("e070_sheet", "e058_pilot30"), ("e070_sheet", "e070_pair"),
         ("e058_pilot30", "e046p_note_all")]  # last: E058 images vs the text harness, for scale

for arm, base in PAIRS:
    out = subprocess.run([sys.executable, CMP, R + arm + ".jsonl", R + base + ".jsonl"], capture_output=True, text=True)
    print(f"== {arm} vs {base}\n{out.stdout.strip() or out.stderr.strip()[-500:]}")
print()
for tag in ("e058_pilot30", "e070_pair", "e070_sheet"):
    d = {}
    for r in map(json.loads, open(R + tag + ".jsonl")):
        if "task_id" in r and not r.get("error"):
            d[r["task_id"], r["pair"], r["attempt"]] = r
    rs = [r for r in d.values() if not r.get("multi_share")]
    print(f"{tag:14} attempts {len(rs):3d}  out tok {st.mean(r['usage']['completion'] for r in rs)/1e3:5.1f}k  "
          f"prompt tok {st.mean(r['usage']['prompt'] for r in rs)/1e3:6.0f}k  turns {st.mean(r.get('turns', 0) for r in rs):4.1f}  "
          f"${sum(r.get('est_cost_usd') or 0 for r in rs):.2f}")
