"""E120 stacked with the other goal-75 levers (util 0.97 KV, tokens x0.9 from E050), plus an E120 proxy without the
mix's diversity bonus ("E120 flip": E128 records, each right answer turned wrong with probability q so pair pass@1
drops to the E128/E112 midpoint, turns x E120/E128). Same sim and settings as e120_kvcap.py.
Usage: e120_stack.py [reps] > e120_stack_out.txt
"""
import collections, multiprocessing as mp_, random, statistics, sys
import e120_kvcap as E

pp1 = lambda pp: statistics.mean(statistics.mean(r["correct"][0] for r in v) for v in pp.values())
turns = lambda pp: statistics.mean(r["turns"] for v in pp.values() for r in v)
q = 1 - pp1(E.PP["E120"]) / pp1(E.PP["E128"])
tf = turns(E.PP["E120"]) / turns(E.PP["E128"])
rng = random.Random(3)
flip = {k: [dict(r, correct=[r["correct"][0] and rng.random() >= q], turns=max(1, round(r["turns"] * tf)))
            for r in v] for k, v in E.PP["E128"].items()}
E.JOBS["E120 flip"] = E.synth(flip)
for p in list(E.JOBS):
    E.JOBS[p + " t0.9"] = {k: [dict(r, tokens=max(1, int(r["tokens"] * 0.9))) for r in v] for k, v in E.JOBS[p].items()}

# util 0.938 -> 0.97 adds ~0.37M KV tokens whatever the model (E042 kvcap2: E128 x1.24)
U97 = 0.24 * 1.554
KV = {"E128": 1.554, "E120": 1.554 + 0.578, "E112": 1.554 + 1.156}

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"E120 flip: q {q:.3f}, turns x{tf:.2f}; pair pass@1 E128 {100 * pp1(E.PP['E128']):.1f}, "
          f"flip {100 * pp1(flip):.1f}, mix {100 * pp1(E.PP['E120']):.1f}, E112 {100 * pp1(E.PP['E112']):.1f}")
    for p in ("E128 synth", "E120 flip", "E120 mix", "E112"):
        print(f"  {p:10s} ceilings: 1 attempt {E.ceiling(p, 1):5.1f} | 2 attempts {E.ceiling(p, 2):5.1f}")
    cfgs = [(w, g) for w in (32, 36, 40, 44, 48, 52, 56, 64) for g in (0.6, 0.7)]
    rows = [(p + t, m, u) for t in ("", " t0.9") for u in (0.0, U97)
            for p, m in (("E128 synth", "E128"), ("E128 real", "E128"), ("E120 flip", "E120"),
                         ("E120 mix", "E120"), ("E112", "E112"))]
    jobs = [(p, (KV[m] + u) / 1.554, w, g, E.S[m], reps) for p, m, u in rows for w, g in cfgs]
    with mp_.Pool(22) as pool:
        res = pool.map(E.run, jobs, chunksize=1)
    out = collections.defaultdict(dict)
    for (p, kvx, w, g, s, _), m, sd in res:
        out[(p, round(kvx, 2))][(w, g)] = (m, sd)
    print(f"\nthrash 0.88:0.3, 240 tasks, {reps} reps; best config per row")
    for (p, kvx), d in out.items():
        best = max(d, key=lambda c: d[c][0])
        print(f"  {p:16s} KV x{kvx:<4}  best {d[best][0]:5.1f}  (x{best[0]} g{best[1]})")
