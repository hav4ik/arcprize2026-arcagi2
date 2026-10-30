"""Accuracy trend for the reports, by the coordinator's method (dashboard/tools/acc_series.py): ARC task score so far
over all 240 tasks (per task the fraction of test pairs right by any finished attempt; unanswered = 0), tasks with an
answer, first-attempt accuracy so far, and the score at the end of each Kaggle hour. Uses the synced data/.
Usage: acc_report.py"""
import json, subprocess
O = "/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/pod3_runs/b32m_v3/data"
A = "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/dashboard/tools/acc_series.py"
SOL = "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json"
pts = json.loads(subprocess.run(["python3", A, "b32m_v3", f"{O}/attempts.jsonl", SOL], capture_output=True, text=True).stdout)
last = pts[-1]
print(f"now (K{last['t_h']:.2f} h): score {last['score_so_far']:.1f}, tasks answered {last['tasks_answered']}/240, "
      f"first-attempt acc {last['first_attempt_acc']}%")
byh = {}
for p in pts:
    byh[int(p["t_h"])] = p
print("score by Kaggle hour (end of hour): " + ", ".join(f"h{h + 1}: {p['score_so_far']:.1f} ({p['tasks_answered']})" for h, p in sorted(byh.items())))
