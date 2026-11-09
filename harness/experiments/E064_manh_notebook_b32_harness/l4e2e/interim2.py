"""E064b interim for the coordinator: B32-M replica (b32m_x2t) vs pod 1's manh_x2t_w80, B32 (candBf_x2t) and B32+R1
(b32r1_x2t) at the same pod hour. Usage: interim2.py <b32m run dir> (the other runs are read from their local mirrors).
Multi-test rollouts write one attempt record on pair 0 whose pred is a list of grids (one per test input)."""
import json
import random
import re
import sys

R = "/home/vu/Workspace/ARC-AGI-2/experiments/"
W = R + "E055_kaggle_replica/toolkit/workloads/evalx2t_s2026"
RUNS = {"b32m": sys.argv[1], "manh80": R + "E062_manh_vllm_fork/l4/runs/manh_x2t_w80",
        "B32": R + "E055_kaggle_replica/toolkit/runs/candBf_x2t", "R1": R + "E062_manh_vllm_fork/l4pod2/runs/b32r1_x2t"}
sol = json.load(open(W + "_solutions.json"))
idmap = json.load(open(W + "_idmap.json"))
orig = lambda t: (idmap.get(t, {}).get("orig", t) if isinstance(idmap.get(t), dict) else idmap.get(t, t))


def metrics(run):
    return [json.loads(l) for l in open(run + "/out/dsv41_logs/metrics.jsonl")]


def records(run):
    return [json.loads(l) for l in open(run + "/out/dsv41_logs/attempts.jsonl")]


def depth(x):
    d = 0
    while isinstance(x, list) and x:
        x, d = x[0], d + 1
    return d


def first_scores(recs, upto=None):
    """task -> task-weighted first-attempt score, for tasks whose first attempt covers all pairs (by t_end_h <= upto)."""
    per = {}
    for r in recs:
        if r["attempt"] != 0 or (upto is not None and r.get("t_end_h", 1e9) > upto):
            continue
        t, p, pred = r["task_id"], r["pair"], r.get("pred")
        n = len(sol[t])
        if n > 1 and depth(pred) == 3:  # multi-test: all test inputs on pair 0
            for i in range(n):
                per.setdefault(t, {})[i] = i < len(pred) and pred[i] == sol[t][i]
        else:
            per.setdefault(t, {})[p] = pred == sol[t][p]
    return {t: sum(v.values()) / len(sol[t]) for t, v in per.items() if len(v) == len(sol[t])}


M = {k: metrics(v) for k, v in RUNS.items()}
REC = {k: records(v) for k, v in RUNS.items()}
H = M["b32m"][-1]["elapsed_h"]
print(f"## B32-M replica b32m_x2t at pod {H:.2f} h")
# gen tok/s by half hour (window-weighted)
print("\ngen tok/s by half hour (b32m | manh80 pod 1 | B32 old pod):")
for k in range(int(H * 2) + 1):
    lo, hi = k / 2, (k + 1) / 2
    row = []
    for name in ("b32m", "manh80", "B32"):
        ws = [(m["window_s"], m["gen_tok_s"]) for m in M[name] if lo <= m["elapsed_h"] < min(hi, H)]
        row.append(f"{sum(w * g for w, g in ws) / sum(w for w, _ in ws):.0f}" if ws else "-")
    print(f"  {lo:.1f}-{hi:.1f} h: " + " | ".join(row))
# KV / preemptions now
last = M["b32m"][-1]
print(f"\nnow: KV {last['kv']}, running {last['running']}, srv_waiting {last['srv_waiting']}, preemptions (window) "
      f"{last['preemptions']}, prefix hit {last['prefix_hit']}, host restore {last['host_tok_s']} tok/s")
# first attempts done by H
print(f"\nfirst attempts done (tasks, all pairs) by pod {H:.2f} h:")
for name in ("b32m", "manh80", "B32", "R1"):
    print(f"  {name}: {len(first_scores(REC[name], H))}")
# paired first-attempt scores on tasks finished in all four (all finished records)
F = {k: first_scores(v) for k, v in REC.items()}
common = sorted(set.intersection(*(set(f) for f in F.values())))
print(f"\nfirst-attempt task-weighted score on the {len(common)} tasks finished in all four runs:")
for name in F:
    print(f"  {name}: {100 * sum(F[name][t] for t in common) / max(1, len(common)):.1f}")
for other in ("manh80", "B32", "R1"):
    both = sorted(set(F["b32m"]) & set(F[other]))
    cl = {}
    for t in both:
        cl.setdefault(orig(t), []).append(F["b32m"][t] - F[other][t])
    ks = list(cl)
    random.seed(0)
    bs = sorted(sum(s) / len(s) for s in ([d for k in (random.choice(ks) for _ in ks) for d in cl[k]] for _ in range(4000)))
    d = sum(F["b32m"][t] - F[other][t] for t in both) / len(both)
    print(f"  b32m - {other} on {len(both)} common tasks: {100 * d:+.1f} [95% {100 * bs[100]:+.1f}, {100 * bs[3899]:+.1f}]")
# token_exact counters
for name in ("b32m", "manh80"):
    tot = {}
    for r in REC[name]:
        if r.get("t_end_h", 1e9) <= H:
            for k, v in (r.get("token_exact") or {}).items():
                tot[k] = tot.get(k, 0) + (v if isinstance(v, (int, float)) else 0)
    turns = tot.get("turns", 0) or 1
    print(f"\ntoken_exact {name} (records ending by {H:.2f} h): " + ", ".join(f"{k} {v}" for k, v in tot.items())
          + f"; breaks/turn {tot.get('breaks', 0) / turns:.3f}, rerenders/turn {tot.get('rerenders', 0) / turns:.4f}")
# stops
for name in ("b32m", "manh80"):
    st = {}
    for r in REC[name]:
        if r.get("t_end_h", 1e9) <= H:
            st[r.get("stop_reason")] = st.get(r.get("stop_reason"), 0) + 1
    print(f"stops {name}: {st}")
