"""Kaggle-style ARC score of a submission.json over the x2t workload: per task the fraction of test pairs whose
attempt_1 or attempt_2 equals the solution (tasks missing from the file score 0), averaged over all tasks x 100;
also tasks with any pair right and the count of tasks whose attempts are the fallback (identical non-answers).
Usage: score_submission.py <submission.json> <solutions.json>"""
import json, sys
sub, sol = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
tot, any_right = 0.0, 0
for t, s in sol.items():
    rows = sub.get(t, [])
    ok = [i < len(rows) and (rows[i].get("attempt_1") == s[i] or rows[i].get("attempt_2") == s[i]) for i in range(len(s))]
    tot += sum(ok) / len(s); any_right += any(ok)
print(f"submission score {100 * tot / len(sol):.2f} over {len(sol)} tasks; tasks with >= 1 pair right {any_right}; tasks in file {len(sub)}")
