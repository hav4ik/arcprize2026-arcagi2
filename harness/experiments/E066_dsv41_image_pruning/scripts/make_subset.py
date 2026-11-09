#!/usr/bin/env python3
"""E066: a compact token dir with a seeded random subset of rollouts (until --tokens), from tokenize_*.py dirs.

Same format (tokens.u32, roles.u8, index.jsonl with offsets rebased); index records keep their fields plus `src`
(source dir) and `src_offset`. Used to ship only the replayed share of a large calibration split to the box.

  make_subset.py --tok data/e066/tok_txt_calib_s/shard* --tokens 14000000 --seed 2 --out data/e066/tok_txt_calib_sub
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=120_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = []
    for d in sorted(a.tok):
        for line in open(Path(d) / "index.jsonl"):
            r = json.loads(line)
            if r["length"] - 1 <= a.max_len:
                recs.append((d, r))
    random.Random(a.seed).shuffle(recs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    n = off = 0
    with open(out / "tokens.u32", "wb") as ft, open(out / "roles.u8", "wb") as fr, open(out / "index.jsonl", "w") as fi:
        for d, r in recs:
            if n >= a.tokens:
                break
            if d not in data:
                data[d] = (np.memmap(Path(d) / "tokens.u32", dtype=np.uint32, mode="r"),
                           np.memmap(Path(d) / "roles.u8", dtype=np.uint8, mode="r"))
            t, ro = data[d]
            o, L = r["offset"], r["length"]
            ft.write(np.asarray(t[o:o + L]).tobytes())
            fr.write(np.asarray(ro[o:o + L]).tobytes())
            fi.write(json.dumps({**r, "src": str(d), "src_offset": o, "offset": off}) + "\n")
            off += L
            n += L
    print(json.dumps({"out": str(out), "tokens": off, "rollouts": sum(1 for _ in open(out / "index.jsonl"))}))


if __name__ == "__main__":
    main()
