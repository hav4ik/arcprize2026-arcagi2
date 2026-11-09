"""What it takes to reach a projected 75: E046 multi_sim over throughput (s), token scale, workers, gate, thrash."""
import collections, copy, random, statistics, sys
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E042_dsv41_l4_synthetic/scripts")
import deadline_sim as ds
import multi_sim as ms

MULTI = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_mt3_sub80_tx_b05.jsonl"
fp = ds.first_prompts()
PP, MP = ms.per_pair_pool(), ms.multi_pool(MULTI)
tasks = sorted({t for t, _ in PP})
npairs = collections.Counter(t for t, _ in PP)
multi_tasks = {t for t in tasks if npairs[t] > 1 and t in MP}
fpj = dict(fp)
for t in multi_tasks:
    fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * 0.6)

def pools(scale):
    jobs = {k: [dict(r, tokens=max(1, int(r["tokens"] * scale))) for r in v] for k, v in PP.items()}
    for t in multi_tasks:
        jobs[(t, "m")] = [dict(r, tokens=max(1, int(r["tokens"] * scale))) for r in MP[t]]
    return jobs

def run(s=1.04, scale=1.0, w=36, gate=0.6, thrash=True, reps=60):
    ds.THRASH = (0.88, 0.6) if thrash else None
    ms.GATE = (gate, 16) if gate else None
    jobs = pools(scale)
    sc, miss, a1 = [], [], []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(tasks) for _ in range(240)]
        keys = []
        for t in draw:
            keys += [(t, "m")] if t in multi_tasks else [(t, q) for q in range(npairs[t])]
        h, n, m = ms.simulate(jobs, keys, fpj, w, s, rng, 1.556e6 / 0.83)
        sc.append(100 * h / n); miss.append(m); a1.append(ms.LAST["a1"] / ms.LAST["jobs"])
    return statistics.mean(sc), statistics.mean(miss), 100 * statistics.mean(a1)

def ceiling(k):
    """Score if every job finishes exactly k attempts (k=1: pass@1, k=2: pass@2), same draws."""
    jobs = pools(1.0); sc = []
    for rep in range(60):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(tasks) for _ in range(240)]
        hit = n = 0
        for t in draw:
            key = (t, "m") if t in multi_tasks else None
            ks = [key] if key else [(t, q) for q in range(npairs[t])]
            for kk in ks:
                att = rng.sample(jobs[kk], 2) if len(jobs[kk]) >= 2 else jobs[kk] * 2
                for q in range(len(att[0]["correct"])):
                    n += 1; hit += any(att[a]["correct"][q] for a in range(k))
        sc.append(100 * hit / n)
    return statistics.mean(sc)

if __name__ == "__main__":
  print(f"ceilings: every job 1 attempt {ceiling(1):.1f} | every job 2 attempts {ceiling(2):.1f}")
  print("config                                   score  no-attempt0  attempt1%")
  for label, kw in [("V6 now: s1.04 x36 gate0.6 thrash", {}),
                    ("V6, no thrash (churn fixed)", dict(thrash=False)),
                    ("no thrash, no gate, x36", dict(thrash=False, gate=0)),
                    ("no thrash, no gate, x44", dict(thrash=False, gate=0, w=44)),
                    ("no thrash, no gate, x56", dict(thrash=False, gate=0, w=56)),
                    ("tokens x0.85 (effort?), V6 gate", dict(scale=0.85)),
                    ("tokens x0.85, no thrash x44", dict(scale=0.85, thrash=False, gate=0, w=44)),
                    ("throughput +15%, no thrash x44", dict(s=1.04 * 1.15, thrash=False, gate=0, w=44)),
                    ("throughput +30%, no thrash x56", dict(s=1.04 * 1.3, thrash=False, gate=0, w=56)),
                    ("tok x0.85 + thr +15%, no thrash x44", dict(scale=0.85, s=1.04 * 1.15, thrash=False, gate=0, w=44)),
                    ("tok x0.85 + thr +30%, no thrash x56", dict(scale=0.85, s=1.04 * 1.3, thrash=False, gate=0, w=56))]:
      a, b, c = run(**kw)
      print(f"{label:40s} {a:5.1f}  {b:10.1f}  {c:8.1f}")
