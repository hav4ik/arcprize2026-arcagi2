import random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/v5gap/scripts")
import v5proj_lib as P
G, ms, ds = P.G, P.ms, P.ds
def scores(jobs, w, kv, ratio, thrash, s=1.04, reps=60):
    ds.THRASH = thrash; ms.GATE = None; sc = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(G.tasks) for _ in range(240)]
        keys = [(t, q) for t in draw for q in range(G.npairs[t])]
        h, n, m = ms.simulate(jobs, keys, G.fpj, w, s, rng, kv / ratio)
        sc.append(100 * h / n)
    return sc
for label, f in [("Kaggle-flag accuracy", (0.045, 0.054)), ("V5-measured accuracy", (0.097, 0.097))]:
    for ratio, th in ((0.72, None), (0.83, (0.88, 0.6))):
        sc = scores(P.flipped(*f, 11), 36, 1.554e6, ratio, th)
        print(f"V5 {label:22s} ratio {ratio} thrash {th}: mean {statistics.mean(sc):.1f} sd over 240-task draws {statistics.stdev(sc):.2f}", flush=True)
