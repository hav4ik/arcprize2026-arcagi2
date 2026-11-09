"""Paired V5-vs-V6 deadline simulation: a copy of E042 multi_sim.simulate (same schedule / throughput / KV / thrash /
gate model) with instrumentation and perturbation knobs. V5 = per pair, W5 workers, no gate; V6 = multi-test, W6, gate.
Both modes see the SAME 240-task draw per rep (paired difference)."""
import argparse, collections, json, random, statistics, sys
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E042_dsv41_l4_synthetic/scripts")
sys.dont_write_bytecode = True
import deadline_sim as ds
import multi_sim as ms

R = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_"

def multi_pool_files(specs):
    pool = collections.defaultdict(list)
    for tag, atts in specs:
        by = collections.defaultdict(dict)
        for l in open(R + tag + ".jsonl"):
            r = json.loads(l)
            if "task_id" in r and (atts is None or r["attempt"] in atts):
                by[(r["task_id"], r["attempt"])][r["pair"]] = r
        for (t, a), prs in by.items():
            if len(prs) < 2: continue  # single-test tasks of final_all
            r0 = prs.get(0)
            if r0 is None or "error" in r0: continue
            u = r0.get("usage") or {}
            pool[t].append({"tokens": max(1, int(u.get("completion") or 0)), "turns": int(r0.get("turns") or 1),
                            "correct": [bool(prs[q].get("correct")) for q in sorted(prs)],
                            "train_pass": r0.get("train_pass") is True})
    return pool

def simulate(jobs_pool, keys, fp, workers, s, rng, kv_cap, gate, thrash, tok_scale=1.0):
    att = {i: (rng.sample(jobs_pool[k], 2) if len(jobs_pool[k]) >= 2 else jobs_pool[k] * 2) for i, k in enumerate(keys)}
    order = sorted(range(len(keys)), key=lambda i: (fp.get(keys[i], 7000), i))
    queue = [(i, 0) for i in order] + [(i, 1) for i in order]
    done, running, rates, t, closed = {}, [], [], ds.T_GEN, None
    kv_peak = thr_s = 0.0; run_sum = n_steps = 0; first_thr = None
    while t < ds.FORCE:
        if closed is None and len(rates) >= 5 and (ds.FORCE - t) * sorted(rates)[len(rates) // 2] < ds.MIN_TOKENS:
            closed = t
        def gate_ok():
            if not gate or kv_cap is None or len(running) < 16:
                return True
            return sum(j["ctx0"] + j["gen"] * 1.15 for j in running) < gate * kv_cap
        while queue and len(running) < workers and t < ds.STOP_NEW and closed is None and gate_ok():
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
            tok = max(1, int(r["tokens"] * (tok_scale if keys[i][1] == "m" else 1.0)))
            running.append({"i": i, "a": a, "left": tok, "tok": tok, "gen": 0, "start": t, "r": r,
                            "ctx0": fp.get(keys[i], 7000), "over": ds.OVERHEAD_S * r["turns"]})
        if not running and (not queue or closed is not None or t >= ds.STOP_NEW):
            break
        active, used = [], 0.0
        for job in running:
            c = job["ctx0"] + job["gen"] * 1.15
            if kv_cap is None or used + c <= kv_cap or not active:
                active.append(job); used += c
        n = len(active)
        per = ds.G(n, s) / n if n else 0
        ctx = sum(job["ctx0"] + job["gen"] * 1.15 for job in running)
        kv_peak = max(kv_peak, ctx / kv_cap); run_sum += len(running); n_steps += 1
        if thrash and ctx > thrash[0] * kv_cap:
            per *= thrash[1]; thr_s += ds.DT
            if first_thr is None: first_thr = t
        for job in list(active):
            if job["over"] > 0:
                share = ds.DT * job["over"] / (job["over"] + job["left"] / max(per, 1e-9))
                job["over"] -= share; job["left"] -= per * (ds.DT - share); job["gen"] += per * (ds.DT - share)
            else:
                job["left"] -= per * ds.DT; job["gen"] += per * ds.DT
            if job["left"] <= 0:
                running.remove(job); done[(job["i"], job["a"])] = "ok"
                rates.append(job["tok"] / (t + ds.DT - job["start"]))
        t += ds.DT
    hit = pairs = 0
    for i in range(len(keys)):
        npairs = len(att[i][0]["correct"]); pairs += npairs
        for q in range(npairs):
            hit += any(done.get((i, a)) == "ok" and att[i][a]["correct"][q] for a in (0, 1))
    return dict(score=100 * hit / pairs, pairs=pairs, a0miss=sum(done.get((i, 0)) != "ok" for i in range(len(keys))),
                a1=sum(done.get((i, 1)) == "ok" for i in range(len(keys))), kv_peak=kv_peak, thr_h=thr_s / 3600,
                first_thr=(first_thr or 0) / 3600, mean_run=run_sum / max(1, n_steps))

def run(cfg, reps=100, seed0=7000):
    """cfg keys: multi (pool), w5, w6, gate6, gate5, s, ratio, kv, thrash, fpj, tok_scale, tasks"""
    fp = ds.first_prompts()
    pp = ms.per_pair_pool(); mp = cfg["multi"]
    tasks = sorted({t for t, _ in pp}); npairs = collections.Counter(t for t, _ in pp)
    multi_tasks = {t for t in tasks if npairs[t] > 1 and t in mp}
    jobs = dict(pp)
    for t in multi_tasks: jobs[(t, "m")] = mp[t]
    fpj = dict(fp)
    for t in multi_tasks:
        fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * cfg.get("fpj", 0.6))
    cap5 = cfg.get("kv", 1.556e6) / cfg.get("ratio5", cfg.get("ratio", 0.83))
    cap6 = cfg.get("kv", 1.556e6) / cfg.get("ratio6", cfg.get("ratio", 0.83))
    flip = cfg.get("flip", 0.0)
    out = []
    for rep in range(reps):
        rng0 = random.Random(seed0 + rep)
        draw = [rng0.choice(tasks) for _ in range(cfg.get("tasks", 240))]
        res = {}
        for mode, w, g, cap in (("v5", cfg.get("w5", 36), cfg.get("gate5", 0), cap5), ("v6", cfg.get("w6", 36), cfg.get("gate6", 0.6), cap6)):
            keys = []
            for t in draw:
                if mode == "v6" and t in multi_tasks: keys.append((t, "m"))
                else: keys += [(t, q) for q in range(npairs[t])]
            rng = random.Random(seed0 * 7 + rep)
            jp = jobs
            if mode == "v6" and flip:  # whole multi attempts turned wrong with prob flip (accuracy penalty, keeps correlation)
                fr = random.Random(seed0 * 13 + rep)
                jp = dict(jobs)
                for t in multi_tasks:
                    jp[(t, "m")] = [dict(r, correct=[False] * len(r["correct"])) if fr.random() < flip else r for r in jobs[(t, "m")]]
            res[mode] = simulate(jp, keys, fpj, w, cfg.get(f"s{mode[1]}", cfg.get("s", 1.04)), rng, cap, g, cfg.get("thrash"), cfg.get("tok_scale", 1.0))
            res[mode]["score"] *= (1 - cfg.get(f"g{mode[1]}", 0.0))
        out.append(res)
    return out

def summ(out, label):
    v5 = [o["v5"]["score"] for o in out]; v6 = [o["v6"]["score"] for o in out]
    d = sorted(b - a for a, b in zip(v5, v6))
    q = lambda p: d[min(len(d) - 1, int(p * len(d)))]
    m = lambda k, mode: statistics.mean(o[mode][k] for o in out)
    print(f"{label:58s} V5 {statistics.mean(v5):5.1f}  V6 {statistics.mean(v6):5.1f}  diff {statistics.mean(d):+5.1f} "
          f"[p5 {q(0.05):+5.1f}, p50 {q(0.5):+5.1f}, p95 {q(0.95):+5.1f}] P(V6<V5) {sum(x < 0 for x in d) / len(d):.2f} | "
          f"a0miss {m('a0miss','v5'):4.1f}/{m('a0miss','v6'):4.1f} kvpk {m('kv_peak','v5'):.2f}/{m('kv_peak','v6'):.2f} "
          f"thr_h {m('thr_h','v5'):4.1f}/{m('thr_h','v6'):4.1f} a1 {m('a1','v5'):4.0f}/{m('a1','v6'):4.0f} run {m('mean_run','v5'):4.1f}/{m('mean_run','v6'):4.1f}", flush=True)
    return d
