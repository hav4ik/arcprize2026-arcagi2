"""Audit variant of multi_sim.py: same model, plus per-rep paired V6-V5 diffs, task-weighted score, perturbations,
and a KV trace. V5 = per pair, no gate; V6 = multi + gate."""
import argparse, collections, json, random, statistics, sys
sys.path.insert(0, '/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/audit_evidence/sim')
import deadline_sim as ds
import multi_sim as ms
GROWTH = 1.15
def simulate(jobs_pool, keys, fp, workers, s, rng, kv_cap, gate, trace=None, t0=None, force=None, stop_new=None):
    FORCE = force or ds.FORCE; STOP_NEW = stop_new or ds.STOP_NEW
    att = {i: (rng.sample(jobs_pool[k], 2) if len(jobs_pool[k]) >= 2 else jobs_pool[k] * 2) for i, k in enumerate(keys)}
    order = sorted(range(len(keys)), key=lambda i: (fp.get(keys[i], 7000), i))
    queue = [(i, 0) for i in order] + [(i, 1) for i in order]
    done, running, rates, t, closed = {}, [], [], (t0 if t0 is not None else ds.T_GEN), None
    first_gate = None; thrash_s = 0; peak = 0
    while t < FORCE:
        if closed is None and len(rates) >= 5 and (FORCE - t) * sorted(rates)[len(rates) // 2] < ds.MIN_TOKENS:
            closed = t
        def gate_ok():
            nonlocal first_gate
            if not gate or len(running) < 16: return True
            ok = sum(j["ctx0"] + j["gen"] * GROWTH for j in running) < gate * kv_cap
            if not ok and first_gate is None: first_gate = t
            return ok
        while queue and len(running) < workers and t < STOP_NEW and closed is None and gate_ok():
            idx = 0
            if queue[0][1] == 1:
                best = None
                for j, (i, a) in enumerate(queue):
                    kk = ds.priority(done.get((i, 0)), att[i][0])
                    if best is None or kk < best[0]:
                        best = (kk, j)
                        if kk == 0: break
                idx = best[1]
            i, a = queue.pop(idx)
            r = att[i][a]
            running.append({"i": i, "a": a, "left": r["tokens"], "gen": 0, "start": t, "r": r,
                            "ctx0": fp.get(keys[i], 7000), "over": ds.OVERHEAD_S * r["turns"]})
        if not running and (not queue or closed is not None or t >= STOP_NEW): break
        active, used = [], 0.0
        for job in running:
            c = job["ctx0"] + job["gen"] * GROWTH
            if used + c <= kv_cap or not active:
                active.append(job); used += c
        n = len(active)
        per = ds.G(n, s) / n if n else 0
        ctx = sum(job["ctx0"] + job["gen"] * GROWTH for job in running)
        peak = max(peak, ctx / kv_cap)
        if ds.THRASH and ctx > ds.THRASH[0] * kv_cap:
            per *= ds.THRASH[1]; thrash_s += ds.DT
        if trace is not None and int(t) % 900 == 0: trace.append((t / 3600, ctx / kv_cap, len(running), n))
        for job in list(active):
            if job["over"] > 0:
                share = ds.DT * job["over"] / (job["over"] + job["left"] / max(per, 1e-9))
                job["over"] -= share; job["left"] -= per * (ds.DT - share); job["gen"] += per * (ds.DT - share)
            else:
                job["left"] -= per * ds.DT; job["gen"] += per * ds.DT
            if job["left"] <= 0:
                running.remove(job); done[(job["i"], job["a"])] = "ok"
                rates.append(job["r"]["tokens"] / (t + ds.DT - job["start"]))
        t += ds.DT
    # pair-weighted and task-weighted
    hit = pairs = 0; task_sc = collections.defaultdict(list); a0miss = 0; a1 = 0
    for i, k in enumerate(keys):
        npq = len(att[i][0]["correct"])
        a0miss += done.get((i, 0)) != "ok"; a1 += done.get((i, 1)) == "ok"
        for q in range(npq):
            h = any(done.get((i, a)) == "ok" and att[i][a]["correct"][q] for a in (0, 1))
            hit += h; pairs += 1
            task_sc[k[2] if len(k) > 2 else k[0]].append(h)
    return dict(pair=100 * hit / pairs, task=100 * statistics.mean(statistics.mean(v) for v in task_sc.values()),
                a0miss=a0miss, a1=a1, first_gate=first_gate, thrash_h=thrash_s / 3600, peak=peak)

def build(multi_path):
    fp = ds.first_prompts(); pp, mp = ms.per_pair_pool(), ms.multi_pool(multi_path)
    tasks = sorted({t for t, _ in pp}); npairs = collections.Counter(t for t, _ in pp)
    multi_tasks = {t for t in tasks if npairs[t] > 1 and t in mp}
    jobs = dict(pp)
    for t in multi_tasks: jobs[(t, "m")] = mp[t]
    fpj = dict(fp)
    for t in multi_tasks: fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * 0.6)
    return jobs, fpj, tasks, npairs, multi_tasks

def run(jobs, fpj, tasks, npairs, multi_tasks, w5, w6, s, ratio, kv, gate, reps, ntasks, acc=None):
    out = []
    for rep in range(reps):
        res = {}
        for mode in ("v5", "v6"):
            rng = random.Random(7000 + rep)
            draw = [rng.choice(tasks) for _ in range(ntasks)]
            keys = []
            for j, t in enumerate(draw):
                if mode == "v6" and t in multi_tasks: keys.append((t, "m", j))
                else: keys += [(t, q, j) for q in range(npairs[t])]
            jp = {k: jobs[k[:2]] for k in keys}
            fpk = {k: fpj.get(k[:2], 7000) for k in keys}
            res[mode] = simulate(jp, keys, fpk, w5 if mode == "v5" else w6, s, rng, kv / ratio, gate if mode == "v6" else 0)
        out.append(res)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", required=True); ap.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    jobs, fpj, tasks, npairs, multi_tasks = build(a.multi)
    print(f"{len(multi_tasks)} multi tasks")
    scen = [
        ("baseline (s1.04, ratio .83, thrash .88:.6)", dict(s=1.04, ratio=0.83, thr=(0.88, 0.6))),
        ("no thrash", dict(s=1.04, ratio=0.83, thr=None)),
        ("s -10% (0.936)", dict(s=0.936, ratio=0.83, thr=(0.88, 0.6))),
        ("s -20% (0.83)", dict(s=0.832, ratio=0.83, thr=(0.88, 0.6))),
        ("ratio 0.72 (less KV/ctx)", dict(s=1.04, ratio=0.72, thr=(0.88, 0.6))),
        ("ratio 0.95 (more KV/ctx)", dict(s=1.04, ratio=0.95, thr=(0.88, 0.6))),
        ("thrash .82:.6", dict(s=1.04, ratio=0.83, thr=(0.82, 0.6))),
        ("thrash .92:.6", dict(s=1.04, ratio=0.83, thr=(0.92, 0.6))),
        ("thrash .88:.45", dict(s=1.04, ratio=0.83, thr=(0.88, 0.45))),
        ("s -10% + ratio .95 + thrash .82", dict(s=0.936, ratio=0.95, thr=(0.82, 0.6))),
        ("s -10%, no thrash", dict(s=0.936, ratio=0.83, thr=None)),
    ]
    for name, c in scen:
        ds.THRASH = c["thr"]
        o = run(jobs, fpj, tasks, npairs, multi_tasks, 36, 36, c["s"], c["ratio"], 1.556e6, 0.6, a.reps, 240)
        f = lambda m, k: statistics.mean(r[m][k] for r in o)
        dp = [r["v6"]["pair"] - r["v5"]["pair"] for r in o]; dt = [r["v6"]["task"] - r["v5"]["task"] for r in o]
        print(f"{name:34s} V5 pair {f('v5','pair'):5.1f} task {f('v5','task'):5.1f} a0miss {f('v5','a0miss'):5.1f} thrash_h {f('v5','thrash_h'):4.1f} peak {f('v5','peak'):.2f} | "
              f"V6 pair {f('v6','pair'):5.1f} task {f('v6','task'):5.1f} a0miss {f('v6','a0miss'):5.1f} thrash_h {f('v6','thrash_h'):4.1f} peak {f('v6','peak'):.2f} | "
              f"diff pair {statistics.mean(dp):+5.1f}±{statistics.pstdev(dp):.1f} task {statistics.mean(dt):+5.1f}±{statistics.pstdev(dt):.1f} V6>V5 {100*sum(x>0 for x in dp)/len(dp):.0f}%", flush=True)
