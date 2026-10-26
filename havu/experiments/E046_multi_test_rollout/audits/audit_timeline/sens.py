"""Sensitivity of multi_sim (E046) to hidden-set token demand and late server start. Read-only on the repo."""
import sys, random, collections, statistics
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E042_dsv41_l4_synthetic/scripts")
import deadline_sim as ds
import multi_sim as ms

MULTI = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_mt3_sub80_tx_b05.jsonl"
scale = float(sys.argv[1]); tgen_h = float(sys.argv[2]); reps = int(sys.argv[3])
ds.T_GEN = tgen_h * 3600
ms.GATE = (0.6, 16)
fp = ds.first_prompts()
pp, mp = ms.per_pair_pool(), ms.multi_pool(MULTI)
for pool in (pp, mp):
    for k in pool:
        for r in pool[k]:
            r["tokens"] = max(1, int(r["tokens"] * scale))
tasks = sorted({t for t, _ in pp})
npairs = collections.Counter(t for t, _ in pp)
multi_tasks = {t for t in tasks if npairs[t] > 1 and t in mp}
jobs = dict(pp)
for t in multi_tasks:
    jobs[(t, "m")] = mp[t]
fpj = dict(fp)
for t in multi_tasks:
    fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * 0.6)
sc, miss = [], []
for rep in range(reps):
    rng = random.Random(7000 + rep)
    draw = [rng.choice(tasks) for _ in range(240)]
    keys = []
    for t in draw:
        keys += [(t, "m")] if t in multi_tasks else [(t, q) for q in range(npairs[t])]
    h, n, m = ms.simulate(jobs, keys, fpj, 36, 1.04, rng, 1.556e6 / 0.83)
    sc.append(100 * h / n); miss.append(m)
print(f"scale {scale} T_GEN {tgen_h}h: score {statistics.mean(sc):.1f} +- {statistics.pstdev(sc):.1f}; jobs without a finished attempt 0: {statistics.mean(miss):.1f} of ~{len(keys)}")
