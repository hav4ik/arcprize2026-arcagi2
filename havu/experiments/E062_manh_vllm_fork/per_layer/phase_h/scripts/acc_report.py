"""Accuracy trend for a replica (same method as the dashboard's acc_series.py): per task the fraction of test pairs
right by any finished attempt (unanswered = 0), averaged over all tasks of the workload; tasks with an answer;
first- and second-attempt accuracy (fully correct / attempts with an answer); the score so far at each full hour.
Usage: acc_report.py <attempts.jsonl> <solutions.json>"""
import json, sys
path, sol_path = sys.argv[1:3]; sol = json.load(open(sol_path))
def depth(x):
    d = 0
    while isinstance(x, list) and x:
        x = x[0]; d += 1
    return d
def pairs_ok(a):
    t = a["task_id"]; n = len(sol[t]); p = a.get("pred")
    if p is None: return None
    preds = p if (n > 1 and depth(p) == 3) else [p]
    return [i < len(preds) and preds[i] == sol[t][i] for i in range(n)]
att = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda a: float(a.get("t_end_h") or 0))
best, acc, hourly = {}, {0: [0, 0], 1: [0, 0]}, []
score = lambda: 100 * sum(sum(v) / len(v) for v in best.values()) / len(sol)
h_next = 1
for a in att:
    th = float(a.get("t_end_h") or 0)
    while th > h_next:
        hourly.append((h_next, score(), len(best))); h_next += 1
    ok = pairs_ok(a)
    if ok is None or a["task_id"] not in sol: continue
    t = a["task_id"]; best[t] = [x or y for x, y in zip(best.get(t, [False] * len(ok)), ok)]
    k = int(a.get("attempt", 0))
    if k in acc: acc[k][0] += 1; acc[k][1] += all(ok)
last_t = float(att[-1].get("t_end_h") or 0) if att else 0
print(f"at Kaggle {last_t:.2f} h: ARC score so far {score():.1f} (all {len(sol)} tasks; best of finished attempts; unanswered = 0); "
      f"tasks with an answer {len(best)}")
for k, nm in ((0, "first"), (1, "second")):
    n, c = acc[k]; print(f"{nm}-attempt accuracy: {c}/{n} = {100 * c / n:.1f}%" if n else f"{nm}-attempt accuracy: n/a")
print("| by Kaggle h | score so far | tasks answered |\n|---|---|---|")
for h, s_, n in hourly: print(f"| {h} h | {s_:.1f} | {n} |")
print(f"| {last_t:.2f} h (now) | {score():.1f} | {len(best)} |")
