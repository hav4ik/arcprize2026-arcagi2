"""Projections with the honest per-pair pool (Kaggle-flag per-pair records e128t_pp_sub80_tx) instead of E043's
optimistic w10_u2 file (V5_GAP.md). Calibration check: v3 50.97, V5 58.19."""
import collections, json, random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds
pool = collections.defaultdict(list)
for r in map(json.loads, open("/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_pp_sub80_tx.jsonl")):
    if "task_id" not in r or "error" in r:
        continue
    u = r.get("usage") or {}
    pool[(r["task_id"], r["pair"])].append({"tokens": max(1, int(u.get("completion") or 0)), "turns": int(r.get("turns") or 1),
                                            "correct": [bool(r.get("correct"))], "train_pass": r.get("train_pass") is True})
missing = [k for k in G.PP if k not in pool]
for k in missing:  # keep the old records only where the honest file has none
    pool[k] = G.PP[k]
print(f"honest per-pair pool: {len(pool)} pairs ({len(missing)} filled from w10_u2); mean acc "
      f"{100 * statistics.mean(sum(x['correct'][0] for x in v) / len(v) for v in pool.values()):.1f}% "
      f"(w10_u2 {100 * statistics.mean(sum(x['correct'][0] for x in v) / len(v) for v in G.PP.values()):.1f}%)")
G.PP = dict(pool)

def run(kvx, w, gate, mode, s=1.04, ratio=0.83, thrash=(0.88, 0.6), kv=1.556e6, reps=60):
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

for th, ratio in (((0.88, 0.6), 0.83), ((0.88, 0.3), 0.83)):
    print(f"--- thrash {th}, KV/ctx {ratio}")
    for label, kvx, w, g, mode, s, kv in [("v3 (per pair x48, KV 1.44M) [LB 50.97]", 1.0, 48, 0, "pp", 1.0, 1.44e6),
                                          ("V5 (per pair x36, no gate) [LB 58.19]", 1.0, 36, 0, "pp", 1.04, 1.556e6),
                                          ("V6 (multi x36 g0.6) [LB pending]", 1.0, 36, 0.6, "multi", 1.04, 1.556e6),
                                          ("candidate (x44 g0.7, 512 ~KV x1.07)", 1.07, 44, 0.7, "multi", 1.04, 1.556e6),
                                          ("+ #156 + #58678", 1.07 * 1.099, 44, 0.7, "multi", 1.04 * 1.07, 1.556e6),
                                          ("+ offload x2.5, x56 g0.85", 2.5 * 1.18, 56, 0.85, "multi", 1.04 * 1.07, 1.556e6)]:
        a, sd = run(kvx, w, g, mode, s=s, ratio=ratio, thrash=th, kv=kv)
        print(f"  {label:44s} {a:5.1f} +- {sd:3.1f}")
