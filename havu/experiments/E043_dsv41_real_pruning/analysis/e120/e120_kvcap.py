"""E120 vs E128 on Kaggle after E042's L2 result (thrash is KV-capacity-driven, x0.3 throughput above KV 0.88).

Same simulator and settings as experiments/E046_multi_test_rollout/goal75/kvcap.py (E046 multi_sim: 240 tasks,
multi-test jobs, KV-gated admission, s 1.04, seeds 7000+rep), but the accuracy pool varies with the model:
  E128 real   per-pair E128 records + E128 run-3 multi-test records (= kvcap.py, reproduces its numbers)
  E128 synth  per-pair E128 records; multi tasks get synthetic multi jobs (one per-pair record per pair, tokens and
              turns x M, M fitted so E128 synth matches E128 real on multi-task tokens): the proxy check
  E112        same synthetic jobs from the clean single-process E112 runs (main a0 + r2 a0)
  E120 mix    E128 + E112 records pooled (E120 accuracy taken as the midpoint)
KV pool per model (Kaggle v4 settings): E128 1.554M, E120 +0.578M (x1.37), E112 +1.156M (x1.74). Speed s: E128 1.04,
E120 1.04 x 1.125/1.10, E112 1.04 x 1.15/1.10 (fewer experts read per decode step; a guess, checked at s 1.04 too).
Usage: e120_kvcap.py [reps] > e120_kvcap_out.txt
"""
import collections, itertools, json, multiprocessing as mp_, random, statistics, sys
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E042_dsv41_l4_synthetic/scripts")
import deadline_sim as ds
import multi_sim as ms

MULTI = ds.RES / f"{ds.TAG}e128t_mt3_sub80_tx_b05.jsonl"
fp = ds.first_prompts()


def per_pair(files):
    pool = collections.defaultdict(list)
    for f, keep in files:
        for r in map(json.loads, open(ds.RES / f"{ds.TAG}{f}.jsonl")):
            if "task_id" not in r or r.get("attempt") not in keep:
                continue
            u = r.get("usage") or {}
            pool[(r["task_id"], r["pair"])].append({"tokens": max(1, int(u.get("completion") or 0)),
                                                   "turns": int(r.get("turns") or 1), "correct": [bool(r.get("correct"))],
                                                   "train_pass": r.get("train_pass") is True})
    return pool


PP = {"E128": per_pair([("e128t_w10_u2_gptq_eng64_sub80", {0, 1})]),
      "E112": per_pair([("e112t_w10_u2_gptq_eng64_sub80", {0}), ("e112t_w10_u2_gptq_eng64_sub80_r2", {0})])}
PP["E120"] = {k: PP["E128"][k] + PP["E112"].get(k, []) for k in PP["E128"]}
MP = ms.multi_pool(MULTI)
tasks = sorted({t for t, _ in PP["E128"]})
npairs = collections.Counter(t for t, _ in PP["E128"])
multi_tasks = {t for t in tasks if npairs[t] > 1 and t in MP}
fpj = dict(fp)
for t in multi_tasks:
    fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * 0.6)

mean = lambda xs: sum(xs) / len(xs)
M = (sum(mean([r["tokens"] for r in MP[t]]) for t in multi_tasks)
     / sum(sum(mean([r["tokens"] for r in PP["E128"][(t, q)]]) for q in range(npairs[t])) for t in multi_tasks))


def synth(pp, cap=256, seed=1):
    """Every combination of one record per pair (a seeded sample of `cap` when there are more)."""
    rng = random.Random(seed)
    jobs = dict(pp)
    for t in sorted(multi_tasks):
        combos = list(itertools.product(*[pp[(t, q)] for q in range(npairs[t])]))
        out = []
        for rs in (rng.sample(combos, cap) if len(combos) > cap else combos):
            out.append({"tokens": max(1, int(M * sum(r["tokens"] for r in rs))),
                        "turns": max(1, round(M * sum(r["turns"] for r in rs))),
                        "correct": [r["correct"][0] for r in rs], "train_pass": all(r["train_pass"] for r in rs)})
        jobs[(t, "m")] = out
    return jobs


JOBS = {"E128 real": {**PP["E128"], **{(t, "m"): MP[t] for t in multi_tasks}},
        "E128 synth": synth(PP["E128"]), "E120 mix": synth(PP["E120"]), "E112": synth(PP["E112"])}
KVX = {"E128": 1.0, "E120": (1.554 + 0.578) / 1.554, "E112": (1.554 + 1.156) / 1.554}
S = {"E128": 1.04, "E120": 1.04 * 1.125 / 1.10, "E112": 1.04 * 1.15 / 1.10}


def run(args):
    pool, kvx, w, gate, s, reps = args
    ds.THRASH = (0.88, 0.3)
    ms.GATE = (gate, 16) if gate else None
    jobs = JOBS[pool]
    sc = []
    for rep in range(reps):
        rng = random.Random(7000 + rep)
        draw = [rng.choice(tasks) for _ in range(240)]
        keys = []
        for t in draw:
            keys += [(t, "m")] if t in multi_tasks else [(t, q) for q in range(npairs[t])]
        h, n, _ = ms.simulate(jobs, keys, fpj, w, s, rng, kvx * 1.556e6 / 0.83)
        sc.append(100 * h / n)
    return args, statistics.mean(sc), statistics.pstdev(sc)


def ceiling(pool, k):
    jobs = JOBS[pool]; sc = []
    for rep in range(60):
        rng = random.Random(7000 + rep)
        hit = n = 0
        for t in [rng.choice(tasks) for _ in range(240)]:
            for kk in ([(t, "m")] if t in multi_tasks else [(t, q) for q in range(npairs[t])]):
                att = rng.sample(jobs[kk], 2) if len(jobs[kk]) >= 2 else jobs[kk] * 2
                for q in range(len(att[0]["correct"])):
                    n += 1; hit += any(att[a]["correct"][q] for a in range(k))
        sc.append(100 * hit / n)
    return statistics.mean(sc)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"multi tasks {len(multi_tasks)}; synthetic multi-job token factor M = {M:.3f}")
    for p in JOBS:
        print(f"  {p:10s} ceilings: 1 attempt {ceiling(p, 1):5.1f} | 2 attempts {ceiling(p, 2):5.1f}")
    cfgs = [(w, g) for w in (32, 36, 40, 44, 48, 52, 56, 64) for g in (0.6, 0.7)]
    combos = [("E128 real", "E128", 1.0), ("E128 real", "E120", 1.0), ("E128 synth", "E128", 1.0),
              ("E128 synth", "E120", 1.0), ("E120 mix", "E120", 1.0), ("E120 mix", "E120", 0.0),
              ("E112", "E112", 1.0), ("E112", "E120", 1.0)]
    # (pool, KV of model, s): s flag 1.0 = model's own s, 0.0 = s 1.04 (speed-up ignored)
    tasks_ = [(p, KVX[kvm], w, g, S[kvm] if sf else 1.04, reps) for p, kvm, sf in combos for w, g in cfgs]
    with mp_.Pool(22) as pool:
        res = pool.map(run, tasks_, chunksize=1)
    out = collections.defaultdict(dict)
    for (p, kvx, w, g, s, _), m, sd in res:
        out[(p, round(kvx, 2), round(s, 3))][(w, g)] = (m, sd)
    print(f"\nthrash 0.88:0.3, 240 tasks, {reps} reps; score (sd over reps); best config marked *")
    for key, d in out.items():
        best = max(d, key=lambda c: d[c][0])
        cells = " ".join(f"x{w}g{g}:{m:5.1f}{'*' if (w, g) == best else ' '}" for (w, g), (m, sd) in sorted(d.items()))
        print(f"{key[0]:10s} KV x{key[1]:<4} s {key[2]:<5} best {d[best][0]:5.1f} (x{best[0]} g{best[1]}, sd {d[best][1]:.1f})")
        print("    " + cells)
