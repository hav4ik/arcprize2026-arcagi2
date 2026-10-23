#!/usr/bin/env python3
"""E043: Engram rows activated on held-out eval traces but never seen in the calibration census.

Counts distinct (layer, row) ids touched by the eval token dirs, how many of them have zero calibration count, the
share of eval lookups that land on such rows, and how often those unseen rows recur within the eval set.

  engram_unseen.py --counts counts_calib.npz --tok DIR [...] [--out unseen.json]
"""
import argparse
import json

import numpy as np

from engram_census import hash_ids, iter_consumed, load_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--model-dir", default="/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--geo", default="/workspace/dsv4/data/engram_geometry.npz")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    geo = load_geometry(a.model_dir, a.geo)
    z = np.load(a.counts)
    seen = [(z[f"l{l}_r0"].astype(np.int64) + z[f"l{l}_r1"]) > 0 for l in range(2)]
    ev = [np.zeros(int(geo["num_embeddings"][l]), np.uint32) for l in range(2)]
    tokens = 0
    for d in a.tok:
        for ids, pos, role in iter_consumed(d):
            h = hash_ids(geo, ids, pos)
            tokens += len(ids)
            for l in range(2):
                ev[l] += np.bincount(h[:, l].ravel(), minlength=len(ev[l])).astype(np.uint32)
    out = {"eval_tokens": tokens}
    for l in range(2):
        touched = ev[l] > 0
        unseen = touched & ~seen[l]
        lk = ev[l].astype(np.int64)
        out[f"layer{l}"] = {
            "eval_distinct_rows": int(touched.sum()),
            "unseen_distinct_rows": int(unseen.sum()),
            "unseen_share_of_distinct": float(unseen.sum() / touched.sum()),
            "eval_lookups": int(lk.sum()),
            "lookups_on_unseen": int(lk[unseen].sum()),
            "lookup_share_on_unseen": float(lk[unseen].sum() / lk.sum()),
            "unseen_rows_hit_once": int((unseen & (ev[l] == 1)).sum()),
            "unseen_rows_hit_ge5": int((unseen & (ev[l] >= 5)).sum()),
        }
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
