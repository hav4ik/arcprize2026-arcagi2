"""Calibration: V5 (per-pair, x36, no gate) and v3 (per-pair, x48, KV 1.44M) with the measured thrash severity."""
import random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds

def run_pp(w, kv, ratio, thrash, s=1.04, reps=60, gate=None):
    ds.THRASH = thrash
    ms.GATE = gate
    jobs = G.pools(1.0); sc = []; miss = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(G.tasks) for _ in range(240)]
        keys = [(t, q) for t in draw for q in range(G.npairs[t])]
        h, n, m = ms.simulate(jobs, keys, G.fpj, w, s, rng, kv / ratio)
        sc.append(100 * h / n); miss.append(m)
    return statistics.mean(sc), statistics.mean(miss)

for label, w, kv, s in (("v3 x48 KV 1.44M", 48, 1.44e6, 1.0), ("V5 x36 KV 1.554M", 36, 1.554e6, 1.04)):
    for ratio in (0.72, 0.83, 0.95):
        for th in ((0.88, 0.6), (0.88, 0.3), (0.82, 0.3)):
            a, m = run_pp(w, kv, ratio, th, s=s)
            print(f"{label:18s} ratio {ratio:.2f} thrash {th}: score {a:5.1f}  jobs without attempt 0: {m:5.1f}")
