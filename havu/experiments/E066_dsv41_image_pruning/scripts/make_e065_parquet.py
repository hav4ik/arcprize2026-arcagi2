#!/usr/bin/env python3
"""E066: pack E065 training-pool rollouts (evals/results) into the rollout parquet schema of E043's tokenizer.

One row per rollout (trace file). Multi-test tasks are one rollout with one record per test pair (usage on pair 0):
they become one row, `correct` = every pair correct, `n_pairs` = number of pairs. For each (task, pair, attempt) the
last record without an `error` field wins (the E065 retries append). `parent` = the NVARC parent (task id before
'_'), or the task id for ARC training tasks. `config` (JSON) is kept, and `tools` holds the exact tool schemas the
harness sent (its own tools_for() on that config), so the box needs no harness code.

  make_e065_parquet.py --runs e065_arc2_train e065_nvarc_b0 e065_nvarc_b1 --out e065_image.parquet
"""
import argparse
import ast
import gzip
import json
from collections import defaultdict

import pandas as pd

from tokenize_image_rollouts import harness_tools

PFX = "tools_persistent_grid_lean2sub_deepseek-flash_low_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--results", default="/home/vu/Workspace/ARC-AGI-2/evals/results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        latest = {}
        for line in open(f"{a.results}/{PFX}{run}.jsonl"):
            r = json.loads(line)
            if "error" not in r:
                latest[(r["task_id"], r["pair"], r["attempt"])] = r
        by_trace = defaultdict(list)
        for r in latest.values():
            p = r["trace_path"]
            by_trace[p if p.endswith(".messages.json.gz") else p + ".messages.json.gz"].append(r)
        for path, recs in sorted(by_trace.items()):
            recs.sort(key=lambda r: r["pair"])
            r0 = recs[0]
            cfg = r0["config"] if isinstance(r0["config"], dict) else ast.literal_eval(r0["config"])  # dict repr
            assert cfg["answer_tool"] and cfg["prompt"] == "lean2sub" and cfg["mode"] == "grid", (run, cfg)
            m = json.load(gzip.open(path))
            rows.append({"task_id": r0["task_id"], "pair": r0["pair"], "attempt": r0["attempt"], "batch": run,
                         "parent": r0["task_id"].split("_")[0], "n_pairs": len(recs),
                         "correct": all(bool(r["correct"]) for r in recs),
                         "train_pass": r0.get("train_pass"), "stop_reason": r0.get("stop_reason"),
                         "messages": json.dumps(m["messages"]), "events": json.dumps(m["events"]),
                         "config": json.dumps(cfg), "tools": json.dumps(harness_tools(cfg))})
    df = pd.DataFrame(rows)
    df.to_parquet(a.out)
    print(json.dumps({"rollouts": len(df), "by_batch": df.batch.value_counts().to_dict(),
                      "correct": df.groupby("batch").correct.mean().round(3).to_dict()}))


if __name__ == "__main__":
    main()
