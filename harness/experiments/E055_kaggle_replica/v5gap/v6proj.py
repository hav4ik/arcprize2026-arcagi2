"""V6 projection (grid.run) with the single-test per-pair pool at the Kaggle-flag accuracy (x0.955) instead of w10_u2's."""
import random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts")
import v5proj_lib as P
G, ms, ds = P.G, P.ms, P.ds
orig = G.pools
def run(f_single, ratio, seed=11, reps=40):
    def pools(scale):
        jobs = orig(scale)
        rng = random.Random(seed)
        for k, recs in jobs.items():
            if k[1] != "m" and G.npairs[k[0]] == 1:
                for r in recs:
                    if r["correct"][0] and rng.random() < f_single:
                        r["correct"] = [False]
        return jobs
    G.pools = pools
    try:
        ds.THRASH = (0.88, 0.6); ms.GATE = (0.6, 16)
        jobs = pools(1.0); sc = []
        for rep in range(reps):
            rng = random.Random(7000 + rep)
            draw = [rng.choice(G.tasks) for _ in range(240)]
            keys = []
            for t in draw:
                keys += [(t, "m")] if t in G.multi_tasks else [(t, q) for q in range(G.npairs[t])]
            h, n, m = ms.simulate(jobs, keys, G.fpj, 36, 1.04, rng, 1.556e6 / ratio)
            sc.append(100 * h / n)
        return statistics.mean(sc)
    finally:
        G.pools = orig
for ratio in (0.83, 0.95):
    a = run(0.0, ratio); b = statistics.mean(run(0.045, ratio, seed=s) for s in (11, 12))
    print(f"V6 x36 gate 0.6, ratio {ratio}: single-test pool w10_u2 {a:.1f} -> Kaggle-flag accuracy {b:.1f}", flush=True)
