#!/usr/bin/env python3
"""E066: ARC eval comparison of the final candidates with earlier runs on the same harnesses, scored like ARC Prize
(task-weighted with partial credit: each test input of a task counts 1/n; pass@1 = mean over attempts 0-1, pass@2 =
either attempt right for that input). Reproduces E058 81.1/90.7, E046 77.6/88.6, E046 run 7 73.7/84.6.

For each run: finished attempts, pass@1 / pass@2 over all 120 tasks (unfinished or errored attempts count as wrong,
final once a run is complete), then paired rows on the tasks a candidate and a reference have both finished.
  compare_evals.py [--paired]
  compare_evals.py --h2h e72 c1x [--harness img|txt]   E72-152 vs candidate 1: pass@1 / pass@2 per model, paired
      difference on the tasks both finished with a 95% bootstrap CI over tasks, tokens / turns per attempt, tool
      errors, error records and stop reasons.
"""
import argparse
import json
import os

R = "/home/vu/Workspace/ARC-AGI-2/evals/results/tools_persistent_grid_lean2sub_"
RUNS = [  # label, file suffix, harness
    ("full model, API (E058)", "deepseek-flash_low_e058_pilot30", "image"),
    ("full model, self-hosted (box 2, partial)", "dsv41_low_e066_full_ref", "image"),
    ("E72-152 avg120 2-bit", "dsv41_low_e066_e72_img", "image"),
    ("candidate 1 E128 2-bit", "dsv41_low_e066_c1x_img", "image"),
    ("full model, API (E046)", "deepseek-flash_low_e046p_note_all", "text"),
    ("full model, self-hosted (box 2, partial)", "dsv41_low_e066_full_ref_txt", "text"),
    ("shipped tool E128 2-bit (E046 run 7)", "dsv41_low_e043_e128t_final_all", "text"),
    ("E72-152 avg120 2-bit", "dsv41_low_e066_e72_txt", "text"),
    ("candidate 1 E128 2-bit", "dsv41_low_e066_c1x_txt", "text"),
]


def load(suffix):
    p = R + suffix + ".jsonl"
    if not os.path.exists(p):
        return None
    latest = {}
    for l in open(p):
        r = json.loads(l)
        latest[(r["task_id"], r["pair"], r["attempt"])] = r
    att = {}  # (task, attempt) -> {pair: correct or None (unfinished / error)}
    for (t, p_, a), r in latest.items():
        if a > 1:
            continue
        att.setdefault((t, a), {})[p_] = None if "error" in r else bool(r.get("correct"))
    return att


def score(att, tasks):
    p1 = p2 = 0
    for t in tasks:
        pairs = sorted({p for a in (0, 1) for p in att.get((t, a), {})}) or [0]
        ok = lambda a, p: bool(att.get((t, a), {}).get(p))
        p1 += sum(ok(a, p) for a in (0, 1) for p in pairs) / (2 * len(pairs))
        p2 += sum(ok(0, p) or ok(1, p) for p in pairs) / len(pairs)
    return 100 * p1 / len(tasks), 100 * p2 / len(tasks)


def done(d, t, a):
    v = d.get((t, a))
    return v is not None and all(x is not None for x in v.values())


def h2h(x, y, harness):
    import random
    import statistics as st
    runs = {n: R + f"dsv41_low_e066_{n}_{harness}.jsonl" for n in (x, y)}
    att, recs = {}, {}
    for n, p in runs.items():
        att[n] = load(f"dsv41_low_e066_{n}_{harness}")
        recs[n] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    if not all(att.values()):
        print("not started:", [n for n in runs if not att[n]]); return
    tasks = sorted({t for d in att.values() for t, _ in d} | {t for t, _ in load("deepseek-flash_low_e058_pilot30")})
    print(f"== {x} vs {y}, {harness} harness")
    for n in (x, y):
        d = att[n]; fin = sum(done(d, t, a) for t in tasks for a in (0, 1))
        p1, p2 = score(d, tasks)
        last = {}
        for r in recs[n]:
            last[(r["task_id"], r["pair"], r["attempt"])] = r
        first = [r for r in last.values() if r["pair"] == min(q["pair"] for q in last.values() if q["task_id"] == r["task_id"] and q["attempt"] == r["attempt"])]
        ok = [r for r in first if "error" not in r]
        toks = [((r.get("usage") or {}).get("completion") or (r.get("usage") or {}).get("completion_tokens") or 0) for r in ok]
        stops = {}
        for r in ok:
            stops[r.get("stop_reason")] = stops.get(r.get("stop_reason"), 0) + 1
        print(f"  {n:4s} finished {fin:3d}/240 | all 120 tasks (unfinished = wrong): pass@1 {p1:5.1f} pass@2 {p2:5.1f} | "
              f"tokens/attempt mean {st.mean(toks) if toks else 0:,.0f} median {st.median(toks) if toks else 0:,.0f} | "
              f"turns mean {st.mean([r.get('turns') or 0 for r in ok]) if ok else 0:.1f} | tool errors/attempt "
              f"{st.mean([r.get('tool_errors') or 0 for r in ok]) if ok else 0:.2f} | error records {sum('error' in r for r in last.values())} | "
              f"stop {stops}")
    both = [t for t in tasks if all(done(att[n], t, a) for n in (x, y) for a in (0, 1))]
    if len(both) < 5:
        print(f"  paired: only {len(both)} tasks finished by both"); return
    def per_task(n, t):
        return score(att[n], [t])
    dx = [(per_task(x, t)[0] - per_task(y, t)[0], per_task(x, t)[1] - per_task(y, t)[1]) for t in both]
    rng = random.Random(0)
    bs = []
    for _ in range(10000):
        smp = [dx[rng.randrange(len(dx))] for _ in dx]
        bs.append((sum(v[0] for v in smp) / len(smp), sum(v[1] for v in smp) / len(smp)))
    for k, name in ((0, "pass@1"), (1, "pass@2")):
        m = sum(v[k] for v in dx) / len(dx); xs = sorted(b[k] for b in bs)
        sx, sy = score(att[x], both)[k], score(att[y], both)[k]
        print(f"  paired on {len(both)} tasks both finished: {name} {x} {sx:5.1f} vs {y} {sy:5.1f} | diff {m:+5.1f} "
              f"[95% CI {xs[250]:+.1f}, {xs[9750]:+.1f}] | tasks better/worse {sum(v[k] > 0 for v in dx)}/{sum(v[k] < 0 for v in dx)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--h2h", nargs=2, metavar=("X", "Y"))
    ap.add_argument("--harness", default="img", choices=["img", "txt"])
    a = ap.parse_args()
    if a.h2h:
        h2h(*a.h2h, a.harness); return
    data = {(lab, h): load(s) for lab, s, h in RUNS}
    tasks = sorted({t for d in data.values() if d for t, _ in d})
    assert len(tasks) == 120, len(tasks)
    for h in ("image", "text"):
        print(f"\n== {h} harness (120 tasks x 2 attempts) | finished | pass@1 | pass@2  (unfinished = wrong)")
        for lab, s, hh in RUNS:
            if hh != h:
                continue
            d = data[(lab, h)]
            if d is None:
                print(f"  {lab:42s} not started"); continue
            fin = sum(done(d, t, x) for t in tasks for x in (0, 1))
            p1, p2 = score(d, tasks)
            print(f"  {lab:42s} {fin:3d}/240 | {p1:5.1f} | {p2:5.1f}")
        if a.paired:
            cands = [lab for lab, s, hh in RUNS if hh == h and "2-bit" in lab and data[(lab, h)]]
            refs = [lab for lab, s, hh in RUNS if hh == h and data[(lab, h)]]
            for c in cands:
                for r in refs:
                    if r == c:
                        continue
                    both = [t for t in tasks if all(done(data[(c, h)], t, x) and done(data[(r, h)], t, x) for x in (0, 1))]
                    if len(both) < 10:
                        continue
                    pc, pr = score(data[(c, h)], both), score(data[(r, h)], both)
                    print(f"  paired, {len(both):3d} tasks both finished: {c} {pc[0]:5.1f}/{pc[1]:5.1f} vs {r} {pr[0]:5.1f}/{pr[1]:5.1f}")


if __name__ == "__main__":
    main()
