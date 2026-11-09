"""Paired per-task comparison of two replicas at the same Kaggle hour: per task the fraction of test pairs right by any
attempt finished by then; gained / lost tasks on the tasks both answered; paired bootstrap of the difference.
usage: paired_at_hour.py NEW_ATTEMPTS BASE_ATTEMPTS SOLUTIONS HOUR"""
import json, sys, random
sol = json.load(open(sys.argv[3])); H = float(sys.argv[4])
def depth(x):
    d = 0
    while isinstance(x, list) and x: x = x[0]; d += 1
    return d
def best(path, h):
    b = {}
    for l in open(path):
        a = json.loads(l)
        if float(a.get("t_end_h") or 99) > h or a["task_id"] not in sol or a.get("pred") is None: continue
        t = a["task_id"]; n = len(sol[t]); p = a["pred"]; preds = p if (n > 1 and depth(p) == 3) else [p]
        ok = [i < len(preds) and preds[i] == sol[t][i] for i in range(n)]
        b[t] = [x or y for x, y in zip(b.get(t, [False] * n), ok)]
    return {t: sum(v) / len(v) for t, v in b.items()}
a, v = best(sys.argv[1], H), best(sys.argv[2], H)
both = set(a) & set(v)
gain = sorted(t for t in both if a[t] > v[t]); lost = sorted(t for t in both if a[t] < v[t])
print(f"at Kaggle {H} h: new score {100 * sum(a.values()) / len(sol):.1f}, base {100 * sum(v.values()) / len(sol):.1f}; answered new {len(a)}, base {len(v)}, both {len(both)}")
print(f"on the {len(both)} tasks both answered: new {100 * sum(a[t] for t in both) / len(both):.1f}% vs base {100 * sum(v[t] for t in both) / len(both):.1f}%")
print(f"gained (new right, base wrong) {len(gain)}: {gain}")
print(f"lost (base right, new wrong) {len(lost)}: {lost}")
d = [a[t] - v[t] for t in both]; random.seed(0)
bs = sorted(sum(random.choice(d) for _ in d) / len(d) * 100 for _ in range(2000))
print(f"paired mean diff {100 * sum(d) / len(d):+.1f} pp, bootstrap 95% CI [{bs[50]:+.1f}, {bs[1950]:+.1f}]")
print(f"answered only by new: {len(set(a) - set(v))} (right {sum(a[t] for t in set(a) - set(v)):.1f}); only by base: {len(set(v) - set(a))} (right {sum(v[t] for t in set(v) - set(a)):.1f})")
