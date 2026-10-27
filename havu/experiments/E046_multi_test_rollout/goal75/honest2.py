"""Fully honest pools: single-test pairs from E046 run 7 (V6 harness, Kaggle flags, 8x box, 2 attempts), multi-test
pairs per pair from e128t_pp_sub80_tx; no E043 w10_u2 records left."""
import collections, json, random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds
R = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_"
def load(path, single_only=False):
    pool = collections.defaultdict(list)
    for r in map(json.loads, open(path)):
        if "task_id" not in r or "error" in r:
            continue
        if single_only and G.npairs.get(r["task_id"], 1) > 1:
            continue
        u = r.get("usage") or {}
        pool[(r["task_id"], r["pair"])].append({"tokens": max(1, int(u.get("completion") or 0)), "turns": int(r.get("turns") or 1),
                                                "correct": [bool(r.get("correct"))], "train_pass": r.get("train_pass") is True})
    return pool
single = load(R + "final_all.jsonl", single_only=True)
pp = load(R + "pp_sub80_tx.jsonl")
pool, src = {}, collections.Counter()
for k in G.PP:
    if G.npairs[k[0]] == 1 and k in single:
        pool[k] = single[k]; src["run7 single"] += 1
    elif k in pp:
        pool[k] = pp[k]; src["pp_sub80_tx"] += 1
    else:
        pool[k] = G.PP[k]; src["w10_u2 (no other record)"] += 1
acc = lambda P: 100 * statistics.mean(sum(x["correct"][0] for x in v) / len(v) for v in P.values())
print("pool sources", dict(src), f"| accuracy {acc(pool):.1f}% vs w10_u2 {acc(G.PP):.1f}%")
G.PP = pool

def run(kvx, w, gate, mode, s=1.04, ratio=0.83, thrash=(0.88, 0.6), kv=1.556e6, reps=80):
    ds.THRASH = thrash; ms.GATE = (gate, 16) if gate else None
    jobs = G.pools(1.0); sc = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(G.tasks) for _ in range(240)]
        keys = []
        for t in draw:
            keys += [(t, "m")] if (mode == "multi" and t in G.multi_tasks) else [(t, q) for q in range(G.npairs[t])]
        h, n, m = ms.simulate(jobs, keys, G.fpj, w, s, rng, kvx * kv / ratio)
        sc.append(100 * h / n)
    return statistics.mean(sc), statistics.pstdev(sc)

for label, kvx, w, g, mode, s, kv in [("v3 [LB 50.97]", 1.0, 48, 0, "pp", 1.0, 1.44e6),
                                      ("V5 [LB 58.19]", 1.0, 36, 0, "pp", 1.04, 1.556e6),
                                      ("V6 (multi x36 g0.6) [pending]", 1.0, 36, 0.6, "multi", 1.04, 1.556e6),
                                      ("candidate x44 g0.7 + 512", 1.07, 44, 0.7, "multi", 1.04, 1.556e6),
                                      ("+ #156 + #58678", 1.07 * 1.099, 44, 0.7, "multi", 1.04 * 1.07, 1.556e6),
                                      ("+ offload x2.5, x56 g0.85", 2.5 * 1.18, 56, 0.85, "multi", 1.04 * 1.07, 1.556e6)]:
    a, sd = run(kvx, w, g, mode, s=s, kv=kv)
    print(f"  {label:34s} {a:5.1f} +- {sd:3.1f}")
