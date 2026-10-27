"""V5 projection (goal75 grid/v5calib simulator) with the per-pair pool's accuracy lowered to V5's measured accuracy."""
import random, statistics, sys
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E046_multi_test_rollout/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds

def flipped(f_single, f_multi, seed):
    jobs = G.pools(1.0)
    rng = random.Random(seed)
    for k, recs in jobs.items():
        if k[1] == "m":
            continue
        f = f_single if G.npairs[k[0]] == 1 else f_multi
        for r in recs:
            if r["correct"][0] and rng.random() < f:
                r["correct"] = [False]
    return jobs

def run_pp(jobs, w, kv, ratio, thrash, s=1.04, reps=60):
    ds.THRASH = thrash
    ms.GATE = None
    sc = []; miss = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(G.tasks) for _ in range(240)]
        keys = [(t, q) for t in draw for q in range(G.npairs[t])]
        h, n, m = ms.simulate(jobs, keys, G.fpj, w, s, rng, kv / ratio)
        sc.append(100 * h / n); miss.append(m)
    return statistics.mean(sc), statistics.mean(miss)

def pool_acc(jobs):
    xs = [r["correct"][0] for k, v in jobs.items() if k[1] != "m" for r in v]
    s = [r["correct"][0] for k, v in jobs.items() if k[1] != "m" and G.npairs[k[0]] == 1 for r in v]
    m = [r["correct"][0] for k, v in jobs.items() if k[1] != "m" and G.npairs[k[0]] > 1 for r in v]
    return 100 * sum(xs) / len(xs), 100 * sum(s) / len(s), 100 * sum(m) / len(m)

variants = {"reference pool (E043 w10_u2, as in v5calib)": (0.0, 0.0),
            "Kaggle-flag per-pair accuracy (single x0.955, multi x0.946)": (1 - 0.955, 1 - 0.946),
            "V5 measured (both V5 runs: x0.903)": (0.097, 0.097),
            "V5 measured, CI ends (x0.85 / x0.96)": None}
which = sys.argv[1] if len(sys.argv) > 1 else "all"
for label, f in variants.items():
    fs = [(0.15, 0.15), (0.04, 0.04)] if f is None else [f]
    for fsg, fmu in fs:
        seeds = [0] if (fsg, fmu) == (0.0, 0.0) else [11, 12, 13]
        for ratio, th in [(0.72, None), (0.83, None), (0.72, (0.88, 0.6)), (0.83, (0.88, 0.6)), (0.95, (0.88, 0.6))]:
            res = []
            for sd in seeds:
                jobs = flipped(fsg, fmu, sd)
                res.append(run_pp(jobs, 36, 1.554e6, ratio, th, reps=40))
            acc = pool_acc(flipped(fsg, fmu, seeds[0]))
            print(f"{label:62s} f=({fsg:.3f},{fmu:.3f}) pool acc {acc[0]:.1f} (s {acc[1]:.1f} m {acc[2]:.1f}) | KV/ctx {ratio:.2f} "
                  f"thrash {th}: score {statistics.mean(r[0] for r in res):5.1f}  no-attempt0 {statistics.mean(r[1] for r in res):5.1f}", flush=True)
