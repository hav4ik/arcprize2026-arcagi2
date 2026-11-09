"""Accuracy over time per run: after each finished attempt (by t_end_h), the ARC task score so far over all 240 tasks
(per task the fraction of test pairs right by any finished attempt; unanswered = 0), the tasks with an answer, and the
first-attempt accuracy so far (fully correct / first attempts with an answer). One point per 10-min bin (the last state).
usage: acc_series.py RUN ATTEMPTS.jsonl SOLUTIONS.json -> JSON list on stdout"""
import json, sys
run, path, sol_path = sys.argv[1:4]
sol = json.load(open(sol_path))
def depth(x):
    d = 0
    while isinstance(x, list) and x:
        x = x[0]; d += 1
    return d
def pairs_ok(a):
    t = a["task_id"]; n = len(sol[t]); p = a.get("pred")
    if p is None:
        return None
    preds = p if (n > 1 and depth(p) == 3) else [p]
    return [i < len(preds) and preds[i] == sol[t][i] for i in range(n)]
att = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda a: float(a.get("t_end_h") or 0))
best, first_n, first_ok, out, last_bin, seen = {}, 0, 0, [], None, {}
for a in att:
    ok = pairs_ok(a)
    if ok is None or a["task_id"] not in sol:
        continue
    t = a["task_id"]
    seen.setdefault(t, set()).add(int(a.get("attempt", 0)))
    best[t] = [x or y for x, y in zip(best.get(t, [False] * len(ok)), ok)]
    if int(a.get("attempt", 0)) == 0:
        first_n += 1; first_ok += all(ok)
    th = float(a["t_end_h"])
    point = {"run": run, "t_h": round(th, 2), "score_so_far": round(100 * sum(sum(v) / len(v) for v in best.values()) / len(sol), 2),
             "tasks_answered": len(best), "tasks_two_attempts": sum(1 for v in seen.values() if 0 in v and 1 in v), "first_attempt_acc": round(100 * first_ok / first_n, 1) if first_n else None}
    b = int(th * 6)
    if out and b == last_bin:
        out[-1] = point
    else:
        out.append(point)
    last_bin = b
json.dump(out, sys.stdout)
