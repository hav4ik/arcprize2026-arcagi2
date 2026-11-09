"""After L2: thrash is KV-capacity-driven. Projected score vs effective KV pool (E128 accuracy), gated and ungated."""
import random, statistics, sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
import grid as G
import multi_sim as ms, deadline_sim as ds

def run(kvx, w, gate, s=1.04, scale=1.0, reps=60, thrash=(0.88, 0.3)):
    ds.THRASH = thrash
    ms.GATE = (gate, 16) if gate else None
    jobs = G.pools(scale); sc = []; a1 = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(G.tasks) for _ in range(240)]
        keys = []
        for t in draw:
            keys += [(t, "m")] if t in G.multi_tasks else [(t, q) for q in range(G.npairs[t])]
        h, n, m = ms.simulate(jobs, keys, G.fpj, w, s, rng, kvx * 1.556e6 / 0.83)
        sc.append(100 * h / n); a1.append(ms.LAST["a1"] / ms.LAST["jobs"])
    return statistics.mean(sc), 100 * statistics.mean(a1)

if __name__ == "__main__":
  print("thrash at 0.88 of the pool, x0.3 throughput (L2 measured ~0.15-0.3x)")
  for kvx in (1.0, 1.37, 2.0, 3.0):
      row = []
      for w, gate in ((36, 0.6), (44, 0.7), (44, 0), (56, 0)):
          a, b = run(kvx, w, gate)
          row.append(f"x{w} g{gate or '-'}: {a:5.1f} (a1 {b:4.1f}%)")
      print(f"KV x{kvx:<4} " + " | ".join(row))
