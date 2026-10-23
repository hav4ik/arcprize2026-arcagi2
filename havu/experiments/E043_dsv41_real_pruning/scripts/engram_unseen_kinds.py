#!/usr/bin/env python3
"""E043: what kind of tokens make Engram lookups on rows never seen in the calibration census?

Splits eval lookups by token source: task prompt (everything before the first generated token: system + task grids),
later prompt tokens (tool results, harness notes, template), and model-generated tokens; reports the share of lookups
on unseen rows per source and n-gram order, and decodes example n-grams that miss.

  engram_unseen_kinds.py --counts counts_calib.npz --tok DIR [...] [--examples 12]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from engram_census import hash_ids, load_geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--tok", nargs="+", required=True)
    ap.add_argument("--model-dir", default="/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--geo", default="/workspace/dsv4/data/engram_geometry.npz")
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    geo = load_geometry(a.model_dir, a.geo)
    tk = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    z = np.load(a.counts)
    seen = [(z[f"l{l}_r0"].astype(np.int64) + z[f"l{l}_r1"]) > 0 for l in range(2)]
    kinds = ["task_prompt", "tool_output_etc", "generated"]
    tot = np.zeros((3, 3), np.int64)   # [kind, order] lookups
    miss = np.zeros((3, 3), np.int64)
    ntok = np.zeros(3, np.int64)
    ex = {k: [] for k in kinds}
    rng = random.Random(0)
    for d in a.tok:
        d = Path(d)
        toks = np.fromfile(d / "tokens.u32", dtype=np.uint32)
        roles = np.fromfile(d / "roles.u8", dtype=np.uint8)
        for line in open(d / "index.jsonl"):
            r = json.loads(line)
            o, n = r["offset"], r["length"] - 1
            ids = toks[o:o + n].astype(np.int64)
            ro = roles[o:o + n]
            first_gen = int(np.argmax(ro == 1)) if (ro == 1).any() else n
            kind = np.where(ro == 1, 2, np.where(np.arange(n) < first_gen, 0, 1))
            h = hash_ids(geo, ids, np.arange(n))
            m = np.stack([~seen[l][h[:, l]] for l in range(2)], axis=1)  # [T, 2, 24] unseen
            for k in range(3):
                sel = kind == k
                ntok[k] += int(sel.sum())
                for oi in range(3):
                    blk = m[sel][:, :, oi * 8:(oi + 1) * 8]
                    tot[k, oi] += blk.size
                    miss[k, oi] += int(blk.sum())
            # Examples: positions whose 4-gram heads mostly miss.
            bad = np.flatnonzero(m[:, :, 8:16].mean(axis=(1, 2)) > 0.5)  # 3-gram heads mostly miss
            for p in rng.sample(list(bad), min(3, len(bad))):
                k = kinds[int(kind[p])]
                if len(ex[k]) < 200:
                    ex[k].append(tk.decode([int(x) for x in ids[max(0, p - 2):p + 1]], skip_special_tokens=False))
    out = {"tokens": dict(zip(kinds, ntok.tolist()))}
    for k in range(3):
        out[kinds[k]] = {"unseen_share": float(miss[k].sum() / max(1, tot[k].sum())),
                         "by_order_2_3_4": [round(float(miss[k, o] / max(1, tot[k, o])), 4) for o in range(3)],
                         "share_of_all_unseen_lookups": float(miss[k].sum() / max(1, miss.sum()))}
    print(json.dumps(out, indent=1))
    for k in kinds:
        print(f"--- example 3-grams missing (ending token at the right), {k}:")
        for s in rng.sample(ex[k], min(a.examples, len(ex[k]))):
            print("   ", repr(s))


if __name__ == "__main__":
    main()
