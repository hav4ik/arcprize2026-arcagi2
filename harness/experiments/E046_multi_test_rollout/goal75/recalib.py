"""Re-projection under the LB calibration (KV per context token 0.95, thrash 0.88:0.6; fits v3 48.4 vs 50.97, V5 58.5 vs 58.19)."""
import random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds

def run(kvx, w, gate, s=1.04, ratio=0.95, thrash=(0.88, 0.6), reps=60):
    ds.THRASH = thrash; ms.GATE = (gate, 16) if gate else None
    jobs = G.pools(1.0); sc = []; a1 = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(G.tasks) for _ in range(240)]
        keys = []
        for t in draw:
            keys += [(t, "m")] if t in G.multi_tasks else [(t, q) for q in range(G.npairs[t])]
        h, n, m = ms.simulate(jobs, keys, G.fpj, w, s, rng, kvx * 1.556e6 / ratio)
        sc.append(100 * h / n); a1.append(ms.LAST["a1"] / ms.LAST["jobs"])
    return statistics.mean(sc), 100 * statistics.mean(a1)

S = 1.04
for label, kv, w, g, s in [("V6 (multi x36 gate 0.6)", 1.0, 36, 0.6, S),
                           ("V6 config x32 gate 0.6", 1.0, 32, 0.6, S),
                           ("candidate x44 gate 0.7 (interval 512 ~ KV x1.07)", 1.07, 44, 0.7, S),
                           ("candidate x36 gate 0.6 + 512", 1.07, 36, 0.6, S),
                           ("+ #58678 (+9.9% KV) + #156 (+7% decode), x36 g0.6", 1.07 * 1.099, 36, 0.6, S * 1.07),
                           ("+ #58678 + #156, x44 g0.7", 1.07 * 1.099, 44, 0.7, S * 1.07),
                           ("+ E051 offload (KV x2.5), x48 g0.7", 2.5 * 1.18, 48, 0.7, S * 1.07),
                           ("+ E051 offload (KV x2.5), x56 ungated", 2.5 * 1.18, 56, 0, S * 1.07),
                           ("+ E051 offload (KV x2.5), x64 ungated", 2.5 * 1.18, 64, 0, S * 1.07)]:
    a, b = run(kv, w, g, s=s)
    print(f"{label:58s} {a:5.1f}  (2nd attempts {b:4.1f}%)")
