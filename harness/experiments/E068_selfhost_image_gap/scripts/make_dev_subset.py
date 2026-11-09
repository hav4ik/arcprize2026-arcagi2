#!/usr/bin/env python3
"""E068: the 32 dev image rollouts nll_eval_mm.py picks (--per-task --n-seq 32 --seed 0 --max-len 60000), as a small
token shard + parquet, so a box needs ~1 MB (zstd) instead of E066's 70 MB tokens + 246 MB parquet.

  make_dev_subset.py OUT_DIR    # writes OUT_DIR/tok_img_dev_s/shard0/{tokens.u32,roles.u8,index.jsonl},
                                #        OUT_DIR/e065_image_dev32.parquet
Every picked rollout is its own task family, so the same selection on the subset returns the same 32 rollouts.
"""
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/E066_dsv41_image_pruning/scripts"))
from mm_common import KEY, load_index  # noqa: E402


def main(out):
    out = Path(out)
    (out / "tok_img_dev_s/shard0").mkdir(parents=True, exist_ok=True)
    recs = load_index(sorted(glob.glob(str(ROOT / "data/e066/tok_img_dev_s/shard*"))), 60000)
    recs.sort(key=lambda r: (r["task_id"], r["pair"], r["attempt"], r.get("batch") or "", r["dir"]))
    rng = random.Random(0)  # nll_eval_mm.py's per-task selection
    by_fam = {}
    for r in recs:
        by_fam.setdefault(r["parent"], []).append(r)
    fams = sorted(by_fam)
    rng.shuffle(fams)
    picked = [rng.choice(by_fam[f]) for f in fams[:32]]
    toks, roles, idx, off = [], [], [], 0
    for r in picked:
        sl = slice(r["offset"], r["offset"] + r["length"])
        toks.append(np.fromfile(f"{r['dir']}/tokens.u32", dtype=np.uint32)[sl])
        roles.append(np.fromfile(f"{r['dir']}/roles.u8", dtype=np.uint8)[sl])
        rr = {k: v for k, v in r.items() if k != "dir"}
        rr["offset"] = off
        off += r["length"]
        idx.append(rr)
    np.concatenate(toks).tofile(out / "tok_img_dev_s/shard0/tokens.u32")
    np.concatenate(roles).tofile(out / "tok_img_dev_s/shard0/roles.u8")
    with open(out / "tok_img_dev_s/shard0/index.jsonl", "w") as f:
        for rr in idx:
            f.write(json.dumps(rr) + "\n")
    keys = {tuple(r[k] for k in KEY) for r in picked}
    t = pq.read_table(ROOT / "data/e066/e065_image.parquet", columns=list(KEY) + ["messages", "tools"])
    rows = [x for x in t.to_pylist() if tuple(x[k] for k in KEY) in keys]
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), out / "e065_image_dev32.parquet", compression="zstd")
    print("picked", len(picked), "families", len({r["parent"] for r in picked}), "rows", len(rows), "tokens", off)


if __name__ == "__main__":
    main(sys.argv[1])
