"""E062 L4: projection of the unfinished Manh replica to Kaggle, from its first hours and B32's full replica.

Assumptions (all stated in the README):
  - first-attempt accuracy as B32's (the paired comparison ties so far), so attempt_1-only score = B32's 77.01;
  - Manh's progress keeps the time ratio r = t_B32(k) / t_Manh(k) of the last matched task count; r has been growing
    since B32's KV filled (hour 2), so holding it is conservative for Manh;
  - after all first attempts, second attempts finish at Manh's first-attempt rate of the last hour (tasks/h; second
    attempts start again from the small tasks, so this is conservative);
  - each second attempt kept adds B32's mean gain per second attempt ((79.10 - 77.01) / 106 task-weighted points);
  - Kaggle's force deadline in pod time from the replica report's central clock (prefill x1.45, decode x0.94, measured
    with B32 on a real Kaggle run).
Usage: project.py MANH_RUN B32_RUN FORCE_POD_H
"""
import json
import sys

manh, b32, force = sys.argv[1], sys.argv[2], float(sys.argv[3])


def firsts(run):
    t = [json.loads(l) for l in open(run + "/out/dsv41_logs/attempts.jsonl")]
    return sorted(r["t_end_h"] for r in t if r["attempt"] == 0 and r["pair"] == 0)


m, b = firsts(manh), firsts(b32)
k = len(m)
r = b[k - 1] / m[k - 1]
now = m[-1]
last_hour = sum(1 for x in m if x > now - 1.0)
t240 = b[239] / r
n2 = max(0.0, (force - t240) * last_hour)
gain = (79.10 - 77.01) / 106
kc = 77.01 + gain * n2
print(json.dumps({"tasks_done": k, "pod_h": round(now, 2), "ratio_r": round(r, 3), "rate_last_hour_tasks": last_hour,
                  "projected_first_attempts_done_pod_h": round(t240, 2), "force_pod_h": force,
                  "projected_second_attempts_kept": round(n2), "gain_per_second_attempt": round(gain, 4),
                  "projected_kaggle_clock_score": round(kc, 1), "projected_lb_by_rule": round(kc - 1.2, 1),
                  "b32_reference": {"kaggle_clock": 77.43, "lb_estimate": 76.2, "first_attempts_done_pod_h": 8.63,
                                    "second_attempts_kept": 32}}, indent=1))
