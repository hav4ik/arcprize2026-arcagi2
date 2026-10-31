#!/usr/bin/env python3
"""Builds the eval120_kaggle workload (the 120 original Kaggle-version eval tasks of evalx2t_s2026, kind 'orig') and seeded 60/60
dev/holdout halves stratified by V12 replica difficulty (data_in/v12_m12_attempts.jsonl = Manh V12's pod replica: the share of
correct attempts of the task's original, 4 strata). Usage: make_eval120.py"""
import json, random, sys
from pathlib import Path
H = Path(__file__).resolve().parent; WL = Path(__file__).resolve().parents[1] / "E055_kaggle_replica/toolkit/workloads"
sys.path.insert(0, str(H / "bin")); import arcscore as A
idmap = json.load(open(WL / "evalx2t_s2026_idmap.json"))
orig = sorted(t for t, m in idmap.items() if m["kind"] == "orig")
ch = json.load(open(WL / "evalx2t_s2026_challenges.json")); sol = json.load(open(WL / "evalx2t_s2026_solutions.json"))
out = H / "workloads"; out.mkdir(exist_ok=True)
json.dump({t: ch[t] for t in orig}, open(out / "eval120_kaggle_challenges.json", "w"))
json.dump({t: sol[t] for t in orig}, open(out / "eval120_kaggle_solutions.json", "w"))
recs = A.load_jsonl(H / "data_in/v12_m12_attempts.jsonl"); res = A.task_results(recs, {t: sol[t] for t in orig})
def diff(t):  # share of correct attempts over all pairs and both attempts (missing = wrong)
    ps = res[t]; n = 2 * len(ps)
    return sum(v for p in ps for v in p["att"].values()) / n
strata = {"s0_unsolved": [], "s1_hard": [], "s2_mid": [], "s3_easy": []}
for t in orig:
    d = diff(t); strata["s0_unsolved" if d == 0 else "s1_hard" if d <= 0.5 else "s2_mid" if d < 1 else "s3_easy"].append(t)
rng = random.Random(2026); dev, hold = [], []
flip = 0
for k, ts in strata.items():
    ts = sorted(ts); rng.shuffle(ts)
    for i, t in enumerate(ts):
        (dev if (i + flip) % 2 == 0 else hold).append(t)
    flip = (flip + len(ts)) % 2
# exact 60/60: move from the larger half the task whose stratum is largest in it
while len(dev) != len(hold):
    big, small = (dev, hold) if len(dev) > len(hold) else (hold, dev)
    small.append(big.pop())
open(out / "eval120_kaggle_dev_ids.txt", "w").write("\n".join(sorted(dev)) + "\n")
open(out / "eval120_kaggle_holdout_ids.txt", "w").write("\n".join(sorted(hold)) + "\n")
s = A.summarize(res)
print("strata", {k: len(v) for k, v in strata.items()}, "dev", len(dev), "holdout", len(hold))
print("V12 replica on the originals (task-level, pass@2 / pass@1):", s)
print("dev", A.summarize(res, set(dev))["score_pass2"], "holdout", A.summarize(res, set(hold))["score_pass2"])
for k, ts in strata.items(): print(k, "dev", sum(t in dev for t in ts), "holdout", sum(t in hold for t in ts))
