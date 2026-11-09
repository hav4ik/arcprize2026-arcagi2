#!/usr/bin/env python3
"""E042: rank harness configs by their simulated 12-h Kaggle score (deadline_sim.py), from existing result files.

Each config's per-attempt records (completion tokens, turns, correct) come from a results jsonl (DeepSeek API runs
on the full Flash-1017, or E043's E128 runs). API tokens are scaled by --scale (E128 sub80s40 + header averages
35.2K against the API's 27.4K: x1.28) and clipped at the run's max_total_output; accuracy is taken as is, so the
API rows rank configs, they do not predict E128's score. Runs without --train-check use correct as train_pass.
Server: E128 + Humming dense x W (s = 1.10, KV 1.66M tokens / 0.72 per context token), as in v4.
Usage: config_sim.py [--reps 60] [--workers 56] [--scale 1.28]
"""
import argparse
import collections
import json
import random
import statistics
from pathlib import Path

import deadline_sim as ds

RES = ds.ROOT / "evals/results"
API = "tools_persistent_grid_{p}_deepseek-flash_low_{t}.jsonl"
CONFIGS = [  # label, files, is_api
    ("sub80s40 (E029)", [API.format(p="lean2sub", t="e029_flash1017_sub80s40")], True),
    ("sub80s40 + header (Kaggle's)", [API.format(p="lean2sub", t="e030_hdrmin")], True),
    ("s20x80", [API.format(p="lean2sub", t="e030_s20x80")], True),
    ("s15x80", [API.format(p="lean2sub", t="e030_s15x80")], True),
    ("s20x60", [API.format(p="lean2sub", t="e029_flash1017_s20x60")], True),
    ("s20x60 + header", [API.format(p="lean2sub", t="e030_s20x60_hdrmin")], True),
    ("cap16", [API.format(p="lean2sub", t="e030_cap16")], True),
    ("E128 sub80s40 + header (E043)", [f"{ds.TAG}e128t_w10_u2_gptq_eng64_sub80.jsonl"], False),
]


def pool(files, scale):
    out = collections.defaultdict(list)
    for f in files:
        for r in map(json.loads, open(RES / f)):
            if "task_id" not in r:
                continue
            cap = (r.get("config") or {}).get("max_total_output") or 80000
            u = r.get("usage") or {}
            tok = int(u.get("completion") or 0)
            tok = min(int(tok * scale), cap) if scale != 1 else tok
            tp = r.get("train_pass")
            out[(r["task_id"], r["pair"])].append({
                "tokens": max(1, tok), "turns": int(r.get("turns") or 1), "correct": bool(r.get("correct")),
                "train_pass": (tp is True) if "train_pass" in r else bool(r.get("correct")),
                "tp_raw": tp if "train_pass" in r else bool(r.get("correct"))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--scale", type=float, default=1.28)
    ap.add_argument("--pairs", type=int, default=259)
    ap.add_argument("--ratio", type=float, default=0.72, help="KV tokens per context token (0.83 fits the thrash onset)")
    ap.add_argument("--thrash", default="", help="U:M as in deadline_sim.py")
    a = ap.parse_args()
    ds.THRASH = tuple(map(float, a.thrash.split(":"))) if a.thrash else None
    fp = ds.first_prompts()
    cap = 1.356e6 * 1.23 / a.ratio
    print(f"{a.pairs} pairs, {a.reps} reps, E128 + Humming dense x {a.workers} (G {ds.G(a.workers, 1.10):.0f} tok/s), "
          f"API tokens x{a.scale}, KV ratio {a.ratio}, thrash {a.thrash or 'off'}")
    print(f"{'config':32s} {'tok/att':>8s} {'pass@1':>7s} {'pass@2':>7s} | {'Kaggle sim':>11s} {'finished':>8s} "
          f"{'cut':>5s} {'unstarted':>9s} {'none-done':>9s}")
    for label, files, is_api in CONFIGS:
        p = pool(files, a.scale if is_api else 1)
        recs = [r for v in p.values() for r in v]
        tok = statistics.mean(r["tokens"] for r in recs)
        p1 = statistics.mean(r["correct"] for r in recs)
        p2 = statistics.mean(any(r["correct"] for r in v[:2]) for v in p.values())
        keys_all = list(p)
        rs = []
        for rep in range(a.reps):
            rng = random.Random(1000 + rep)
            keys = [rng.choice(keys_all) for _ in range(a.pairs)]
            rs.append(ds.simulate(p, keys, fp, a.workers, 1.10, rng, kv_cap=cap))
        f = lambda k: statistics.mean(r[k] for r in rs)
        print(f"{label:32s} {tok / 1000:7.1f}k {100 * p1:6.1f}% {100 * p2:6.1f}% | {100 * f('score'):5.1f} ± "
              f"{100 * statistics.pstdev(r['score'] for r in rs):3.1f} {f('ok'):8.0f} {f('cut'):5.1f} {f('unstarted'):9.1f} {f('none_done'):9.1f}")


if __name__ == "__main__":
    main()
