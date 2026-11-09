#!/usr/bin/env python3
"""E046: 12-h Kaggle deadline simulation with whole tasks, per-pair vs one rollout per task (--multi-test).

Same server and schedule model as deadline_sim.py (G(n) throughput, KV cap in context tokens, optional thrash,
stop-new / force / admission close, attempt 0 of every job first, then attempt 1 by train-fail priority), but the
test set is drawn as TASKS (240 from the 120 eval tasks, ~335 pairs) and scored per pair. Per-pair mode: one job
per (task, pair) from E128's per-pair records. Multi mode: single-test tasks as before; a multi-pair task is one
job whose attempts come from an E128 --multi-test run (tokens / turns of the attempt, correctness per pair).
Usage: multi_sim.py --multi <e043 multi results.jsonl> [--configs 36,44,52] [--thrash 0.88:0.6] [--reps 100]
"""
import argparse
import collections
import json
import random
import statistics

import deadline_sim as ds


def per_pair_pool():
    pool = collections.defaultdict(list)
    for r in map(json.loads, open(ds.RES / f"{ds.TAG}e128t_w10_u2_gptq_eng64_sub80.jsonl")):
        if "task_id" not in r:
            continue
        u = r.get("usage") or {}
        pool[(r["task_id"], r["pair"])].append({"tokens": max(1, int(u.get("completion") or 0)),
                                               "turns": int(r.get("turns") or 1), "correct": [bool(r.get("correct"))],
                                               "train_pass": r.get("train_pass") is True})
    return pool


def multi_pool(path):
    by = collections.defaultdict(dict)  # (task, attempt) -> {pair: rec}
    for r in map(json.loads, open(path)):
        if "task_id" in r:
            by[(r["task_id"], r["attempt"])][r["pair"]] = r
    pool = collections.defaultdict(list)
    for (t, a), prs in by.items():
        r0 = prs.get(0)
        if r0 is None or "error" in r0:
            continue
        u = r0.get("usage") or {}
        pool[t].append({"tokens": max(1, int(u.get("completion") or 0)), "turns": int(r0.get("turns") or 1),
                        "correct": [bool(prs[q].get("correct")) for q in sorted(prs)],
                        "train_pass": r0.get("train_pass") is True})
    return pool


LAST = {}
GATE = None  # (max usage, min workers): E042 KV-gated admission, usage = running contexts / KV pool


def simulate(jobs_pool, keys, fp, workers, s, rng, kv_cap):
    """keys: job keys; jobs_pool[key]: attempt records with correct = per-pair list. Returns (pairs hit, pairs)."""
    att = {i: (rng.sample(jobs_pool[k], 2) if len(jobs_pool[k]) >= 2 else jobs_pool[k] * 2) for i, k in enumerate(keys)}
    order = sorted(range(len(keys)), key=lambda i: (fp.get(keys[i], 7000), i))
    queue = [(i, 0) for i in order] + [(i, 1) for i in order]
    done, running, rates, t, closed = {}, [], [], ds.T_GEN, None
    while t < ds.FORCE:
        if closed is None and len(rates) >= 5 and (ds.FORCE - t) * sorted(rates)[len(rates) // 2] < ds.MIN_TOKENS:
            closed = t
        def gate_ok():
            if not GATE or kv_cap is None or len(running) < GATE[1]:
                return True
            return sum(j["ctx0"] + j["gen"] * 1.15 for j in running) < GATE[0] * kv_cap

        while queue and len(running) < workers and t < ds.STOP_NEW and closed is None and gate_ok():
            idx = 0
            if queue[0][1] == 1:
                best = None
                for j, (i, a) in enumerate(queue):
                    kk = ds.priority(done.get((i, 0)), att[i][0])
                    if best is None or kk < best[0]:
                        best = (kk, j)
                        if kk == 0:
                            break
                idx = best[1]
            i, a = queue.pop(idx)
            r = att[i][a]
            running.append({"i": i, "a": a, "left": r["tokens"], "gen": 0, "start": t, "r": r,
                            "ctx0": fp.get(keys[i], 7000), "over": ds.OVERHEAD_S * r["turns"]})
        if not running and (not queue or closed is not None or t >= ds.STOP_NEW):
            break
        active, used = [], 0.0
        for job in running:
            c = job["ctx0"] + job["gen"] * 1.15
            if kv_cap is None or used + c <= kv_cap or not active:
                active.append(job)
                used += c
        n = len(active)
        per = ds.G(n, s) / n if n else 0
        ctx = sum(job["ctx0"] + job["gen"] * 1.15 for job in running)
        if ds.THRASH and kv_cap and ctx > ds.THRASH[0] * kv_cap:
            per *= ds.THRASH[1]
        for job in list(active):
            if job["over"] > 0:
                share = ds.DT * job["over"] / (job["over"] + job["left"] / max(per, 1e-9))
                job["over"] -= share
                job["left"] -= per * (ds.DT - share)
                job["gen"] += per * (ds.DT - share)
            else:
                job["left"] -= per * ds.DT
                job["gen"] += per * ds.DT
            if job["left"] <= 0:
                running.remove(job)
                done[(job["i"], job["a"])] = "ok"
                rates.append(job["r"]["tokens"] / (t + ds.DT - job["start"]))
        t += ds.DT
    hit = pairs = 0
    for i in range(len(keys)):
        npairs = len(att[i][0]["correct"])
        pairs += npairs
        for q in range(npairs):
            hit += any(done.get((i, a)) == "ok" and att[i][a]["correct"][q] for a in (0, 1))
    LAST.update(a1=sum(done.get((i, 1)) == "ok" for i in range(len(keys))), jobs=len(keys),
                tok=sum(att[i][a]["tokens"] for (i, a), v in done.items() if v == "ok"))
    return hit, pairs, sum(done.get((i, 0)) != "ok" for i in range(len(keys)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", required=True)
    ap.add_argument("--configs", default="36,44,52")
    ap.add_argument("--kv", type=float, default=1.556e6)
    ap.add_argument("--ratio", type=float, default=0.83)
    ap.add_argument("--s", type=float, default=1.04, help="throughput factor vs Kaggle v3 (Humming on Kaggle: ~1.04)")
    ap.add_argument("--thrash", default="")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--tasks", type=int, default=240)
    ap.add_argument("--gate", type=float, default=0.0, help="KV-gated admission: max usage (min 16 running)")
    a = ap.parse_args()
    ds.THRASH = tuple(map(float, a.thrash.split(":"))) if a.thrash else None
    global GATE
    GATE = (a.gate, 16) if a.gate else None
    fp = ds.first_prompts()
    pp, mp = per_pair_pool(), multi_pool(a.multi)
    tasks = sorted({t for t, _ in pp})
    npairs = collections.Counter(t for t, _ in pp)
    multi_tasks = {t for t in tasks if npairs[t] > 1 and t in mp}
    print(f"{len(tasks)} eval tasks, {len(multi_tasks)} multi-pair with multi records; draws of {a.tasks} tasks; "
          f"KV {a.kv:.3g} / ratio {a.ratio}; s {a.s}; thrash {a.thrash or 'off'}")
    jobs = dict(pp)
    for t in multi_tasks:
        jobs[(t, "m")] = mp[t]
    fpj = dict(fp)
    for t in multi_tasks:
        fpj[(t, "m")] = int(sum(fp.get((t, q), 7000) for q in range(npairs[t])) * 0.6)
    for w in map(int, a.configs.split(",")):
        for mode in ("per-pair", "multi"):
            sc, miss, a1, nj = [], [], [], []
            for rep in range(a.reps):
                rng = random.Random(7000 + rep)
                draw = [rng.choice(tasks) for _ in range(a.tasks)]
                keys = []
                for t in draw:
                    if mode == "multi" and t in multi_tasks:
                        keys.append((t, "m"))
                    else:
                        keys += [(t, q) for q in range(npairs[t])]
                h, n, m = simulate(jobs, keys, fpj, w, a.s, rng, a.kv / a.ratio)
                sc.append(100 * h / n)
                miss.append(m)
                a1.append(LAST["a1"] / LAST["jobs"])
                nj.append(LAST["jobs"])
            print(f"  W {w:3d} {mode:9s}: score {statistics.mean(sc):5.1f} ± {statistics.pstdev(sc):3.1f}   "
                  f"jobs with no finished attempt 0: {statistics.mean(miss):5.1f}   jobs {statistics.mean(nj):5.0f}, "
                  f"with a finished attempt 1: {100 * statistics.mean(a1):4.1f}%")


if __name__ == "__main__":
    main()
