#!/usr/bin/env python3
"""E043: pack Flash-1017 eval-task transcripts (evals/results) into the rollout parquet schema for tokenize_rollouts.py.

  make_eval_parquet.py --runs e030_s20x60_hdrmin [e029_flash1017_s20x60 ...] --out eval.parquet
Each run is `evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_<run>{.jsonl,/}` (lean2sub, grid mode,
answer tool). E043 used e030_s20x60_hdrmin alone (data/tok_eval) and the six runs e029_flash1017_s20x60,
e029_flash1017_sub80s40, e030_s15x80, e030_s20x80, e030_hdrmin, e030_s20x60_hdrmin (data/tok_eval6); the cap-nudge
runs (cap12/cap16) are excluded because they contain harness-injected tool calls.
"""
import argparse
import gzip
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--results", default="/home/vu/Workspace/ARC-AGI-2/evals/results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for run in a.runs:
        base = f"{a.results}/tools_persistent_grid_lean2sub_deepseek-flash_low_{run}"
        files = {}
        for f in os.listdir(base):
            k = f.split("_")
            files[(k[0], int(k[1][1:]), int(k[2][1:]))] = os.path.join(base, f)
        for line in open(base + ".jsonl"):
            r = json.loads(line)
            p = files.get((r["task_id"], r["pair"], r["attempt"]))
            if p is None:
                continue
            cfg = r.get("config", {})
            assert cfg.get("answer_tool") and cfg.get("prompt") == "lean2sub" and cfg.get("mode") == "grid", (run, cfg)
            m = json.load(gzip.open(p))
            msgs = m["messages"] if isinstance(m, dict) else m
            out.append({"task_id": r["task_id"], "pair": r["pair"], "attempt": r["attempt"], "batch": run,
                        "parent": r["task_id"], "correct": bool(r.get("correct")), "train_pass": None,
                        "messages": json.dumps(msgs), "events": json.dumps([])})
    df = pd.DataFrame(out)
    df.to_parquet(a.out)
    print(len(df), df.groupby("batch").correct.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
