"""Calibrate the sim's KV / progress against pod runs: the eval set itself (120 tasks once), V5 vs V6 configs."""
import sys, collections, random, statistics, json
sys.dont_write_bytecode = True
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_decision")
from sim2 import *
import sim2
M35 = multi_pool_files([("e128t_mt3_sub80_tx_b05", None)])
fp = ds.first_prompts(); pp = ms.per_pair_pool()
tasks = sorted({t for t, _ in pp}); npairs = collections.Counter(t for t, _ in pp)
mt = {t for t in tasks if npairs[t] > 1 and t in M35}
jobs = dict(pp); fpj = dict(fp)
for t in mt:
    jobs[(t, "m")] = M35[t]; fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * 0.6)
ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.83
cap = 1.556e6 / ratio
# instrument: monkeypatch simulate loop by recording trajectory via a wrapper of ds.G? simpler: re-run with trace hook
def traj(mode, w, gate, reps=30):
    pts = collections.defaultdict(list); recs = collections.defaultdict(list)
    for rep in range(reps):
        keys = []
        for t in tasks:
            if mode == "v6" and t in mt: keys.append((t, "m"))
            else: keys += [(t, q) for q in range(npairs[t])]
        rng = random.Random(99 + rep)
        # copy of sim2.simulate with snapshots
        att = {i: (rng.sample(jobs[k], 2) if len(jobs[k]) >= 2 else jobs[k] * 2) for i, k in enumerate(keys)}
        order = sorted(range(len(keys)), key=lambda i: (fpj.get(keys[i], 7000), i))
        queue = [(i, 0) for i in order] + [(i, 1) for i in order]
        done, running, t = {}, [], ds.T_GEN
        snaps = [1.0, 1.8, 2.5, 3.0, 4.0, 4.8]
        while t < 5.0 * 3600:
            def gate_ok():
                if not gate or len(running) < 16: return True
                return sum(j["ctx0"] + j["gen"] * 1.15 for j in running) < gate * cap
            while queue and len(running) < w and gate_ok():
                i, a = queue.pop(0); r = att[i][a]
                running.append({"i": i, "a": a, "left": r["tokens"], "gen": 0, "ctx0": fpj.get(keys[i], 7000), "over": ds.OVERHEAD_S * r["turns"]})
            active, used = [], 0.0
            for job in running:
                c = job["ctx0"] + job["gen"] * 1.15
                if used + c <= cap or not active: active.append(job); used += c
            n = len(active); per = ds.G(n, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0) / n if n else 0
            ctx = sum(j["ctx0"] + j["gen"] * 1.15 for j in running)
            for job in list(active):
                if job["over"] > 0:
                    share = ds.DT * job["over"] / (job["over"] + job["left"] / max(per, 1e-9))
                    job["over"] -= share; job["left"] -= per * (ds.DT - share); job["gen"] += per * (ds.DT - share)
                else:
                    job["left"] -= per * ds.DT; job["gen"] += per * ds.DT
                if job["left"] <= 0: running.remove(job); done[(job["i"], job["a"])] = 1
            # pods' age is from solver start; T_GEN (0.35h) ~ server start. Use age = t/3600 - 0.0 (pods' age column is server-line age)
            for s_ in snaps:
                if abs(t / 3600 - s_) < ds.DT / 7200:
                    pts[s_].append(ctx * ratio / 1.556e6); recs[s_].append(len(done))
            t += ds.DT
    return {s_: (statistics.mean(pts[s_]), statistics.mean(recs[s_])) for s_ in pts}
for mode, w, g in [(m_, int(w_), float(g_)) for m_, w_, g_ in (x.split(":") for x in sys.argv[2].split(","))]:
    tr = traj(mode, w, g)
    print(mode, w, g, " ".join(f"{h}h KV {kv:.2f} done {n:.0f}" for h, (kv, n) in sorted(tr.items())))
