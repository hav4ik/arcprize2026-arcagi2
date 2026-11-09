"""E062 L4: first-attempt accuracy of two replica runs on the tasks both have finished (paired by task), task-weighted
over the task's pairs, with a bootstrap over original tasks (a task and its transformed copy together, via idmap).
Usage: paired_acc.py RUN_A RUN_B [--solutions F --idmap F]"""
import argparse
import json
import random

W = "/home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica/toolkit/workloads/evalx2t_s2026"
ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--solutions", default=W + "_solutions.json"); ap.add_argument("--idmap", default=W + "_idmap.json")
x = ap.parse_args()
sol = json.load(open(x.solutions)); idmap = json.load(open(x.idmap))


def first(run):
    d = {}
    for l in open(run + "/out/dsv41_logs/attempts.jsonl"):
        r = json.loads(l)
        if r["attempt"] == 0:
            d.setdefault(r["task_id"], {})[r["pair"]] = r.get("pred") == sol[r["task_id"]][r["pair"]]
    return {t: sum(v.values()) / len(sol[t]) for t, v in d.items() if len(v) == len(sol[t])}


A, B = first(x.a), first(x.b)
both = sorted(set(A) & set(B))
orig = lambda t: idmap.get(t, {}).get("orig", t) if isinstance(idmap.get(t), dict) else idmap.get(t, t)
clusters = {}
for t in both:
    clusters.setdefault(orig(t), []).append(A[t] - B[t])
ks = list(clusters)
random.seed(0)
bs = []
for _ in range(4000):
    s = [d for k in (random.choice(ks) for _ in ks) for d in clusters[k]]
    bs.append(sum(s) / len(s))
bs.sort()
ma, mb = sum(A[t] for t in both) / len(both), sum(B[t] for t in both) / len(both)
print(f"{len(both)} tasks finished in both ({len(ks)} clusters): first-attempt task-weighted A {100 * ma:.1f} vs B {100 * mb:.1f}, "
      f"A-B {100 * (ma - mb):+.1f} [95% {100 * bs[100]:+.1f}, {100 * bs[3899]:+.1f}]; better {sum(A[t] > B[t] for t in both)}, "
      f"worse {sum(A[t] < B[t] for t in both)}")
