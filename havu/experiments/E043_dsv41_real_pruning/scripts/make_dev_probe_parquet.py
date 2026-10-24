#!/usr/bin/env python3
"""E043: first-turn probe prompts from dev-family training rollouts (gen_probe.py --parquet), so free-generation
probes never touch the 120 eval tasks. Dev = split_dev.py's family rule; one rollout per task, teacher-correct only,
shuffled with a fixed seed.

  make_dev_probe_parquet.py --rollouts e031.parquet [...] --n 24 --seed 0 --out dev_probe.parquet
"""
import argparse
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from split_dev import is_dev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen, rows = set(), []
    for f in a.rollouts:
        for r in pq.read_table(f, columns=["task_id", "parent", "correct", "messages"]).to_pylist():
            if r["correct"] in (True, "True") and is_dev(r) and r["task_id"] not in seen:
                seen.add(r["task_id"])
                rows.append({"task_id": r["task_id"], "messages": r["messages"]})
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    pq.write_table(pa.Table.from_pylist(rows), a.out)
    print(f"{len(seen)} dev tasks with a correct rollout; wrote {len(rows)} to {a.out}")


if __name__ == "__main__":
    main()
