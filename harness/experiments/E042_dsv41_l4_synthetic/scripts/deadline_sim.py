#!/usr/bin/env python3
"""E042: simulate a 12-h Kaggle run of the dsv41 notebook, to see how many attempts finish before the deadlines.

Replays E043's real per-attempt records (completion tokens, turns, correct, train_pass) through the solver's
schedule (kaggle/dsv41_tools/solver_dsv41.py): attempt-0 jobs smallest task first, then attempt-1 jobs with
non-train-passed pairs first; W workers; stop-new 10h50, force 11h10 (running attempts are cut there); admission
closes when (force - now) x median per-stream rate of finished attempts < 15,000 tokens.

Throughput model: total generated tok/s with n running streams is G(n) = s * D(n) / (1 + k D(n) / P), where
D(n) = n / (a + b n) is pure decode (a = 58 ms, b = 0.9 ms per stream: Kaggle decode data at 32K), k = 0.4 uncached
prompt tokens per generated token (real runs), P = 507 tok/s the effective prefill rate that fits Kaggle's E128 x48
plateau (345 tok/s, s = 1). Each turn adds OVERHEAD_S of wall time (tool run, time to first token).

The test set (259 pairs) is drawn with replacement from the eval pairs; each pair's attempts come from its own
recorded attempts, in random order. A cut attempt counts as wrong (1 of 105 cut attempts was right on pod B).
KV: the sum of running contexts (first prompt + tokens generated so far + tool output), scaled so that the E128 x48
run reads 0.75 at 1.74 h as on Kaggle; reported as the peak, not enforced.

Usage: deadline_sim.py [--reps 200]
"""
import argparse
import collections
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"
TAG = "tools_persistent_grid_lean2sub_dsv41_low_e043_"
A, B, K, P = 0.058, 0.0009, 0.4, 507.0
OVERHEAD_S = 4.0
T_GEN = 0.35 * 3600  # install ~5 min + server up ~11 min + first prefills
STOP_NEW, FORCE = 10 * 3600 + 50 * 60, 11 * 3600 + 10 * 60
MIN_TOKENS = 15000
DT = 10.0
FOLLOW = False  # attempt 1 right after a non-train-passed attempt 0 finishes
THRASH = None  # (U, M): while running contexts exceed U of the KV pool, throughput x M (pod B: > 0.88 -> 168-281 of ~380)
FIT = 0.0  # >0: skip attempt-1 jobs whose attempt 0 used more than (time left x rate) / FIT tokens


def G(n, s):
    if n <= 0:
        return 0.0
    d = n / (A + B * n)
    return s * d / (1 + K * d / P)


def load(files):
    """Per-pair attempt pools from result files; an entry is a tag (every attempt) or (tag, attempts to keep)."""
    pairs = collections.defaultdict(list)
    for f in files:
        f, keep = (f, None) if isinstance(f, str) else f
        for r in map(json.loads, open(RES / f"{TAG}{f}.jsonl")):
            if "task_id" not in r or (keep is not None and r.get("attempt") not in keep):
                continue
            u = r.get("usage") or {}
            pairs[(r["task_id"], r["pair"])].append({
                "tokens": max(1, int(u.get("completion") or 0)), "turns": int(r.get("turns") or 1),
                "prompt0": 0, "correct": bool(r.get("correct")), "train_pass": r.get("train_pass") is True})
    return pairs


def first_prompts():
    d = json.load(open(ROOT / "experiments/E043_dsv41_real_pruning/results/tool_shapes_e128t_w10_u2_gptq_eng64_sub80.json"))
    return {(a["task_id"], a["pair"]): a["turns"][0]["prompt"] for a in d["attempts"] if a["turns"]}


POLICY = "current"
# T6: E112's clean pool is main attempt 0 + r2 attempt 0. Main's attempt 1 and r3 ran in a second harness process at
# the same time and are anomalous (95 and 101 of 167 correct vs 116 and 114; dsv4-pruning session, 2026-10-26); with
# them the E112 rows were ~1.5 pp pessimistic (the E128 pick is unaffected).
POOLS = {"E128": ["e128t_w10_u2_gptq_eng64_sub80"],
         "E112": [("e112t_w10_u2_gptq_eng64_sub80", {0}), ("e112t_w10_u2_gptq_eng64_sub80_r2", {0})]}


def priority(st, rec0):
    """Second-attempt priority (lower first) from attempt 0's state: st None = still running, "ok" = finished."""
    tp = rec0.get("tp_raw", rec0["train_pass"])
    if POLICY == "current":  # solver_dsv41.pick(): failed or missing check 0, running 1, passed 2
        return 1 if st is None else (2 if st == "ok" and rec0["train_pass"] else 0)
    if st is None:
        return {"three": 2, "three_run_early": 1}[POLICY]
    if tp is None:  # no answer: never correct in E043's runs
        return 0
    return {"three": 1, "three_run_early": 2}[POLICY] if tp is False else 3


def simulate(pool, keys, fp, workers, s, rng, trace=None, kv_cap=None):
    jobs = []  # (size, key index, attempt)
    att = {}
    for i, k in enumerate(keys):
        recs = rng.sample(pool[k], 2) if len(pool[k]) >= 2 else pool[k] * 2
        att[i] = recs
    order = sorted(range(len(keys)), key=lambda i: (fp.get(keys[i], 7000), i))
    queue = [(i, 0) for i in order] + [(i, 1) for i in order]
    done = {}  # (i, a) -> "ok" | "cut"
    running = []  # dicts
    rates, t, closed_at, kv_peak, kv_at, stall_s = [], T_GEN, None, 0.0, None, 0
    thrash_s, thrash_at = 0, None
    while t < FORCE:
        if closed_at is None and len(rates) >= 5:
            if (FORCE - t) * sorted(rates)[len(rates) // 2] < MIN_TOKENS:
                closed_at = t
        while queue and len(running) < workers and t < STOP_NEW and closed_at is None:
            # pick(): attempt 0 first; then attempt 1 of pairs without a train-passing attempt 0, then running, then passed
            idx = 0
            if FOLLOW and queue[0][1] == 1 and done.get((queue[0][0], 0)) == "ok" and not att[queue[0][0]][0]["train_pass"]:
                pass  # a follow-up queued at the front
            elif queue[0][1] == 1:
                best = None
                rate = sorted(rates)[len(rates) // 2] if rates else None
                for j, (i, a) in enumerate(queue):
                    st = done.get((i, 0))
                    kk = priority(st, att[i][0])
                    if FIT and rate and st == "ok" and att[i][0]["tokens"] * FIT > (FORCE - t) * rate:
                        continue  # attempt 0 of this pair took longer than the time left allows
                    if best is None or kk < best[0]:
                        best = (kk, j)
                        if kk == 0:
                            break
                if best is None:
                    break
                idx = best[1]
            i, a = queue.pop(idx)
            r = att[i][a]
            running.append({"i": i, "a": a, "left": r["tokens"], "gen": 0, "start": t, "r": r,
                            "ctx0": fp.get(keys[i], 7000), "over": OVERHEAD_S * r["turns"]})
        if not running and (not queue or closed_at is not None or t >= STOP_NEW):
            break
        # KV: the oldest streams that fit the pool make progress; the rest wait (preempted / queued turns)
        active, used = [], 0.0
        for job in running:
            c = job["ctx0"] + job["gen"] * 1.15
            if kv_cap is None or used + c <= kv_cap or not active:
                active.append(job)
                used += c
        n = len(active)
        per = G(n, s) / n if n else 0
        ctx = sum(job["ctx0"] + job["gen"] * 1.15 for job in running)
        if THRASH and kv_cap and ctx > THRASH[0] * kv_cap:
            per *= THRASH[1]
            thrash_s += DT
        stalled = len(running) - n
        for job in list(active):
            if job["over"] > 0:  # spread the per-turn overhead in proportion to progress
                share = DT * job["over"] / (job["over"] + job["left"] / max(per, 1e-9))
                job["over"] -= share
                job["left"] -= per * (DT - share)
                job["gen"] += per * (DT - share)
            else:
                job["left"] -= per * DT
                job["gen"] += per * DT
            if job["left"] <= 0:
                running.remove(job)
                done[(job["i"], job["a"])] = "ok"
                if FOLLOW and job["a"] == 0 and not job["r"]["train_pass"] and (job["i"], 1) in queue:
                    queue.remove((job["i"], 1))  # its attempt 1 next, while the first prompt is still cached
                    queue.insert(0, (job["i"], 1))
                rates.append(job["r"]["tokens"] / (t + DT - job["start"]))
        stall_s += stalled * DT
        if trace is not None:
            trace.append((t / 3600, n, ctx))
        if thrash_at is None and thrash_s:
            thrash_at = t
        if kv_at is None and t >= 1.74 * 3600:
            kv_at = ctx
        kv_peak = max(kv_peak, ctx)
        t += DT
    for job in running:
        done[(job["i"], job["a"])] = "cut"
    hit = 0
    for i in range(len(keys)):
        hit += any(done.get((i, a)) == "ok" and att[i][a]["correct"] for a in (0, 1))
    st = collections.Counter(done.values())
    return {"score": hit / len(keys), "ok": st["ok"], "cut": st["cut"], "unstarted": 2 * len(keys) - len(done),
            "first_cut_or_unstarted": sum(done.get((i, 0)) != "ok" for i in range(len(keys))),
            "none_done": sum(all(done.get((i, a)) != "ok" for a in (0, 1)) for i in range(len(keys))),
            "thrash_h": thrash_s / 3600, "thrash_at_h": (thrash_at or FORCE) / 3600,
            "both_done": sum(done.get((i, 0)) == "ok" and done.get((i, 1)) == "ok" for i in range(len(keys))),
            "closed_h": (closed_at or FORCE) / 3600, "kv_peak": kv_peak, "kv_174": kv_at or kv_peak,
            "stalled_frac": stall_s / max(1, workers * (FORCE - T_GEN))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=259)
    ap.add_argument("--ratios", default="none,0.6,0.83", help="KV tokens per context token; none = no KV limit")
    ap.add_argument("--configs", default="", help="model:W:s[,...] (s 1.101 = Humming dense + 15%% KV)")
    ap.add_argument("--fit", type=float, default=0.0)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--thrash", default="", help="U:M, e.g. 0.88:0.6")
    ap.add_argument("--multi-scale", type=float, default=0.0,
                    help="E046: scale tokens and turns of pairs from multi-pair tasks (one rollout per task: 0.59)")
    a = ap.parse_args()
    global FIT, FOLLOW, THRASH
    FIT, FOLLOW = a.fit, a.follow
    THRASH = tuple(map(float, a.thrash.split(":"))) if a.thrash else None
    fp = first_prompts()
    models = {m: load(files) for m, files in POOLS.items()}
    # KV pool in context tokens = reported KV tokens / (KV tokens per context token). The ratio is 0.83 if the
    # simulated E128 x48 contexts at 1.74 h match Kaggle's 0.735 usage, 0.6 from pod B's E112 x128 at 1.41 h.
    kv_tokens = {("E128", 1.00): 1.437e6, ("E128", 1.10): 1.356e6, ("E128", 1.101): 1.356e6 * 1.15, ("E128", 1.155): 1.356e6 * 1.15,
                 ("E112", 1.05): 2.512e6}
    configs = [("E128", 48, 1.00, "v3 as submitted (Kaggle 345 tok/s)"), ("E128", 48, 1.10, "+ Humming dense"),
               ("E128", 56, 1.10, ""), ("E128", 64, 1.10, ""), ("E128", 40, 1.10, ""),
               ("E112", 64, 1.05, "+ Humming dense"), ("E112", 96, 1.05, ""), ("E112", 128, 1.05, "")]
    if a.multi_scale:  # E046: a multi-pair task's one attempt costs multi_scale x the per-pair attempts' tokens
        for pool in models.values():
            n_pairs = collections.Counter(t for t, _ in pool)
            for (t, _), recs in pool.items():
                if n_pairs[t] > 1:
                    for r in recs:
                        r["tokens"] = max(1, int(r["tokens"] * a.multi_scale))
                        r["turns"] = max(1, round(r["turns"] * a.multi_scale))
    print(f"{a.pairs} pairs, {a.reps} reps; G(n) at s=1: " + ", ".join(f"{n}:{G(n, 1):.0f}" for n in (8, 32, 48, 64, 96, 128)))
    print("KV ratio  model  W   s     G(W)  per-stream  score(pass@2)    ok    cut  unstarted  both-done  stalled  closed_h"
          "  a0-missed  none-done  thrash_at  thrash_h")
    if a.configs:  # model:W:s[:KV tokens]
        configs = []
        for c in a.configs.split(","):
            f = c.split(":")
            if len(f) > 3:
                kv_tokens[(f[0], float(f[2]))] = float(f[3])
            configs.append((f[0], int(f[1]), float(f[2]), ""))
    for ratio in [None if r == "none" else float(r) for r in a.ratios.split(",")]:
        for m, w, s, note in configs:
            pool = models[m]
            keys_all = list(pool)
            cap = kv_tokens[(m, s)] / ratio if ratio else None
            rs = []
            for rep in range(a.reps):
                rng = random.Random(1000 + rep)
                keys = [rng.choice(keys_all) for _ in range(a.pairs)]
                rs.append(simulate(pool, keys, fp, w, s, rng, kv_cap=cap))
            f = lambda k: statistics.mean(r[k] for r in rs)
            sd = statistics.pstdev(r["score"] for r in rs)
            print(f"{str(ratio):8s}  {m}  {w:3d}  {s:.2f}  {G(w, s):5.0f}  {G(w, s) / w:6.1f}      "
                  f"{100 * f('score'):5.1f} ± {100 * sd:3.1f}  {f('ok'):5.0f}  {f('cut'):5.1f}  {f('unstarted'):7.1f}  "
                  f"{f('both_done'):9.0f}  {100 * f('stalled_frac'):5.1f}%  {f('closed_h'):7.2f}  {f('first_cut_or_unstarted'):9.1f}  "
                  f"{f('none_done'):9.1f}  {f('thrash_at_h'):9.2f}  {f('thrash_h'):8.2f}  {note}")
    # Reference: no deadline (every attempt finishes)
    for m in models:
        pool = models[m]
        sc = []
        for (k, recs) in pool.items():
            pass
        allp = list(pool)
        p2 = []
        for rep in range(200):
            rng = random.Random(5000 + rep)
            keys = [rng.choice(allp) for _ in range(a.pairs)]
            p2.append(statistics.mean(any(r["correct"] for r in rng.sample(pool[k], 2)) for k in keys))
        print(f"{m} no-deadline pass@2: {100 * statistics.mean(p2):.1f}")


if __name__ == "__main__":
    main()
