#!/usr/bin/env python3
"""E066: one intermediate-results report for the running candidate evals: progress and pace, pass@1 / pass@2 of each
candidate on the tasks it has finished next to every reference on the same tasks, and the head-to-head with a 95%
bootstrap CI (compare_evals.py scoring, ARC Prize partial credit).
  progress_report.py [--harness img|txt]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_evals import R, done, load, score  # noqa: E402

REFS = {"img": [("full model, API", "deepseek-flash_low_e058_pilot30", "image"),
                ("full model, self-hosted", "dsv41_low_e066_full_ref", "image"),
                ("old E128 2-bit (B32)", "dsv41_low_e043_e128t_final_all", "text"),
                ("full model, API", "deepseek-flash_low_e046p_note_all", "text")],
        "txt": [("full model, API", "deepseek-flash_low_e046p_note_all", "text"),
                ("old E128 2-bit (B32)", "dsv41_low_e043_e128t_final_all", "text"),
                ("full model, API", "deepseek-flash_low_e058_pilot30", "image")]}
START = {"img": 18 * 3600 + 9 * 60}  # UTC seconds of day the image runs started (2026-10-29 18:09)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="img", choices=["img", "txt"])
    a = ap.parse_args()
    h = a.harness
    cands = [("E72-152", f"dsv41_low_e066_e72_{h}"), ("candidate 1", f"dsv41_low_e066_c1x_{h}")]
    D = {s: load(s) for _, s in cands + [(x[0], x[1]) for x in REFS[h]]}
    tasks = sorted({t for t, _ in D["deepseek-flash_low_e058_pilot30"]})
    now = time.gmtime()
    print(f"== {h} harness, {time.strftime('%H:%M', now)} UTC")
    for name, s in cands:
        d = D[s]
        if not d:
            print(f"{name}: not started"); continue
        fin = sum(done(d, t, x) for t in tasks for x in (0, 1))
        both = [t for t in tasks if all(done(d, t, x) for x in (0, 1))]
        recs = [json.loads(l) for l in open(R + s + ".jsonl")]
        toks = [((r.get("usage") or {}).get("completion") or 0) for r in recs if "error" not in r and r["pair"] == 0]
        line = f"{name}: {fin}/240 attempts, {len(both)} tasks complete"
        if h in START:
            el = (now.tm_hour * 3600 + now.tm_min * 60 - START[h]) / 3600
            line += f", {el:.1f} h in"
        print(line + f", mean output {sum(toks) / max(1, len(toks)):,.0f} tok/attempt")
        if not both:
            continue
        p1, p2 = score(d, both)
        print(f"   {name + ' (' + h + ')':30s} pass@1 {p1:5.1f}  pass@2 {p2:5.1f}  on its {len(both)} complete tasks")
        for rn, rs, rh in REFS[h]:
            sub = [t for t in both if all(done(D[rs], t, x) for x in (0, 1))]
            if not sub:
                continue
            q1, q2 = score(D[rs], sub)
            c1, c2 = score(d, sub)
            print(f"   {rn + ' (' + rh + ')':30s} pass@1 {q1:5.1f}  pass@2 {q2:5.1f}  on {len(sub)} of them  (candidate {c1 - q1:+.1f} / {c2 - q2:+.1f})")
    x, y = cands[0][1], cands[1][1]
    if D[x] and D[y]:
        both = [t for t in tasks if all(done(D[s], t, a_) for s in (x, y) for a_ in (0, 1))]
        if len(both) >= 5:
            dx = [(score(D[x], [t])[0] - score(D[y], [t])[0], score(D[x], [t])[1] - score(D[y], [t])[1]) for t in both]
            rng = random.Random(0)
            bs = [[dx[rng.randrange(len(dx))] for _ in dx] for _ in range(5000)]
            for k, nm in ((0, "pass@1"), (1, "pass@2")):
                m = sum(v[k] for v in dx) / len(dx)
                xs = sorted(sum(v[k] for v in b) / len(b) for b in bs)
                print(f"head to head, {len(both)} tasks both complete: {nm} E72-152 {score(D[x], both)[k]:5.1f} vs candidate 1 "
                      f"{score(D[y], both)[k]:5.1f}, diff {m:+.1f} [95% CI {xs[125]:+.1f}, {xs[4875]:+.1f}], "
                      f"tasks better/worse {sum(v[k] > 0 for v in dx)}/{sum(v[k] < 0 for v in dx)}")


if __name__ == "__main__":
    main()
