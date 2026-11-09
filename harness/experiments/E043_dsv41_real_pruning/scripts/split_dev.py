#!/usr/bin/env python3
"""E043: deterministic dev split of tokenized training rollouts, held out by task family.

A rollout is dev when sha1(family) % 10 == 0, family = NVARC parent (the ARC training task it was generated from) or,
for ARC training tasks, the task id. Writes calibration and dev token dirs (same format as tokenize_rollouts.py) so
statistics never see dev and screening never sees the eval set.

  split_dev.py --tok DIR [...] --calib-out DIR --dev-out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def family(r):
    p = r.get("parent")
    return p if p not in (None, "None", "") else r["task_id"].split("_")[0]


def is_dev(r):
    return int(hashlib.sha1(family(r).encode()).hexdigest(), 16) % 10 == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--calib-out", required=True)
    ap.add_argument("--dev-out", required=True)
    a = ap.parse_args()
    outs = {}
    for name in ("calib", "dev"):
        d = Path(getattr(a, f"{name}_out"))
        d.mkdir(parents=True, exist_ok=True)
        outs[name] = [open(d / "tokens.u32", "wb"), open(d / "roles.u8", "wb"), open(d / "index.jsonl", "w"), 0, 0]
    for src in a.tok:
        src = Path(src)
        toks = np.fromfile(src / "tokens.u32", dtype=np.uint32)
        roles = np.fromfile(src / "roles.u8", dtype=np.uint8)
        for line in open(src / "index.jsonl"):
            r = json.loads(line)
            o = outs["dev" if is_dev(r) else "calib"]
            o[0].write(toks[r["offset"]:r["offset"] + r["length"]].tobytes())
            o[1].write(roles[r["offset"]:r["offset"] + r["length"]].tobytes())
            o[2].write(json.dumps({**r, "offset": o[3], "family": family(r)}) + "\n")
            o[3] += r["length"]
            o[4] += 1
    for name, o in outs.items():
        for f in o[:3]:
            f.close()
        print(name, o[4], "rollouts", o[3], "tokens")


if __name__ == "__main__":
    main()
