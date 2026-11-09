#!/usr/bin/env python3
"""rep2_compare.py: Manh V12 replica 2 (pod 5) vs the first replica (pod 2). Prints (1) score-so-far by Kaggle hour, (2) final Kaggle-style
score + paired per-task differences (bootstrap over tasks, originals and all 240), (3) decode speed at matched load: mean gen tok/s of
the 1-minute samples binned by the server's running count, same Kaggle hour windows. Files: replica 1 = phase_h/manh_v12/data, data_in/v12_m12_attempts.jsonl;
replica 2 = phase_h/manh_v12_rep2/data. usage: rep2_compare.py [now_h]"""
import json, random, sys, collections
from pathlib import Path
H = Path(__file__).resolve().parent; PH = H.parent / "E062_manh_vllm_fork/per_layer/phase_h"
sys.path.insert(0, str(H / "bin")); import arcscore as A
sol = json.load(open(H.parent / "E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_solutions.json"))
idmap = json.load(open(H.parent / "E055_kaggle_replica/toolkit/workloads/evalx2t_s2026_idmap.json"))
r1 = A.load_jsonl(H / "data_in/v12_m12_attempts.jsonl"); r2 = A.load_jsonl(PH / "manh_v12_rep2/data/dsv41_logs/attempts.jsonl")
now = float(sys.argv[1]) if len(sys.argv) > 1 else 99
def so_far(recs, h):
    res = A.task_results([r for r in recs if float(r.get("t_end_h") or 99) <= h], sol); return A.summarize(res)
print("hour  rep1 score (tasks both)   rep2 score (tasks both)   [score = pass@2 so far, unanswered 0]")
for h in range(1, 13):
    if h > now: break
    a, b = so_far(r1, h), so_far(r2, h); print(f"{h:4d}  {a['score_pass2']:6.1f} ({a['tasks_with_both_attempts']:3d})            {b['score_pass2']:6.1f} ({b['tasks_with_both_attempts']:3d})")
x1, x2 = A.task_results(r1, sol), A.task_results(r2, sol)
print("final rep1", A.summarize(x1)); print("final rep2", A.summarize(x2))
def paired(ts, m="p2"):
    d = [sum(p[m] for p in x2[t]) / len(x2[t]) - sum(p[m] for p in x1[t]) / len(x1[t]) for t in ts]; rng = random.Random(1); n = len(d)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000)); return f"{100*sum(d)/n:+.1f} [{100*bs[250]:+.1f}, {100*bs[9749]:+.1f}] (differ on {sum(1 for v in d if v)} of {n})"
orig = [t for t in sol if idmap[t]["kind"] == "orig"]
for m in ("p2", "p1"): print(f"paired {m} rep2-rep1: all240 {paired(list(sol), m)}; originals {paired(orig, m)}")
def tps(p):
    rows = [l.rstrip("\n").split("\t") for l in open(p)][1:]; return [(float(r[1]), float(r[2]), float(r[3] or 0)) for r in rows if len(r) > 3 and r[3] != ""]
t1, t2 = tps(PH / "manh_v12/data/tps.tsv"), tps(PH / "manh_v12_rep2/data/tps.tsv")
print("decode speed at matched load (mean gen tok/s of 1-min samples with gen>50, by running count; hours 0.25-9.5):")
for lo, hi in ((20, 30), (30, 40), (40, 50), (50, 58), (58, 70)):
    f = lambda t: [g for h, g, r in t if 0.25 <= h <= 9.5 and lo <= r < hi and g > 50]
    a, b = f(t1), f(t2)
    if a and b: print(f"  running {lo}-{hi}: rep1 {sum(a)/len(a):6.1f} (n={len(a)})  rep2 {sum(b)/len(b):6.1f} (n={len(b)})  ratio {sum(b)/len(b)/(sum(a)/len(a)):.3f}")
