#!/usr/bin/env python3
"""Accuracy trend of a pod 4 run with the team's acc_series.py (scratchpad dashboard/tools): the ARC task score so far
over all 240 tasks (best of the finished attempts, all pairs, unanswered = 0), tasks answered, first-attempt accuracy;
the score so far at the end of each Kaggle hour; and for each fault (Kaggle h), the score just before it and 1 h after.
Usage: acc_report.py <data dir> [fault_h ...]"""
import json, subprocess, sys
D = sys.argv[1]; faults = [float(x) for x in sys.argv[2:]]
TOOL = "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/dashboard/tools/acc_series.py"
SOL = "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json"
pts = json.loads(subprocess.run([sys.executable, TOOL, "pod4", f"{D}/attempts.jsonl", SOL], capture_output=True,
                                text=True, check=True).stdout)
if not pts:
    print("no finished attempts yet"); sys.exit(0)
last = pts[-1]
print(f"score so far {last['score_so_far']:.2f} / 100 over 240 tasks; tasks answered {last['tasks_answered']}/240; "
      f"first-attempt acc {last['first_attempt_acc']}% (at Kaggle {last['t_h']:.2f} h)")
import os, tempfile
LINES = [l for l in open(f"{D}/attempts.jsonl") if l.strip()]
def at(h):
    """exact state at Kaggle h: the tool on the attempts finished by h (its last point)"""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.writelines(l for l in LINES if float(json.loads(l).get("t_end_h") or 0) <= h)
    try:
        q = json.loads(subprocess.run([sys.executable, TOOL, "pod4", f.name, SOL], capture_output=True, text=True,
                                      check=True).stdout)
    finally:
        os.unlink(f.name)
    return q[-1] if q else {"score_so_far": 0.0, "tasks_answered": 0, "first_attempt_acc": None, "t_h": 0}
print("| Kaggle h (end) | score so far | tasks answered | first-attempt acc % |")
print("|---|---|---|---|")
for h in range(1, int(last["t_h"]) + 2):
    p = at(h)
    if h > last["t_h"] + 1:
        break
    print(f"| {min(h, last['t_h']):.2f} | {p['score_so_far']:.2f} | {p['tasks_answered']} | {p['first_attempt_acc']} |")
for f in faults:
    b, a = at(f), at(f + 1)
    tail = "" if last["t_h"] >= f + 1 else f" (only to {last['t_h']:.2f} h so far)"
    print(f"fault at {f:.2f} h: score {b['score_so_far']:.2f} just before, {a['score_so_far']:.2f} 1 h after "
          f"(+{a['score_so_far'] - b['score_so_far']:.2f}); tasks {b['tasks_answered']} -> {a['tasks_answered']}{tail}")
