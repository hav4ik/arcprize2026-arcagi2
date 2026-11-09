#!/usr/bin/env python3
"""E066 (measurement only): per eval task, the share of Engram lookups that fall in rows the 64 GiB table keeps but the
56 GiB table drops (rows chosen from the training census only). Traces: B32's run 7 eval traces, tokenized.
  engram_task_miss.py --tok DIR --geo GEO --mask64 M64 --mask56 M56 --out tasks.json"""
import argparse, json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/workspace/dsv4/repo/experiments/E043_dsv41_real_pruning/scripts")
from engram_census import hash_ids  # noqa: E402
ap = argparse.ArgumentParser()
for k in ("tok", "geo", "mask64", "mask56", "out"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
geo = dict(np.load(a.geo))
m64 = [np.unpackbits(np.load(a.mask64)[f"layer{l}"], bitorder="little").astype(bool) for l in range(2)]
m56 = [np.unpackbits(np.load(a.mask56)[f"layer{l}"], bitorder="little").astype(bool) for l in range(2)]
toks = np.fromfile(f"{a.tok}/tokens.u32", dtype=np.uint32); roles = np.fromfile(f"{a.tok}/roles.u8", dtype=np.uint8)
per = defaultdict(lambda: np.zeros(4))  # lookups, dropped (in 64 not 56), gen lookups, gen dropped
for line in open(f"{a.tok}/index.jsonl"):
    r = json.loads(line); o, L = r["offset"], r["length"]
    ids = toks[o:o + L - 1].astype(np.int64); pos = np.arange(L - 1, dtype=np.int64); g = roles[o:o + L - 1] == 1
    h = hash_ids(geo, ids, pos)
    k64 = np.stack([m64[l][h[:, l]] for l in range(2)], axis=1); k56 = np.stack([m56[l][h[:, l]] for l in range(2)], axis=1)
    drop = k64 & ~k56
    per[r["task_id"]] += [k64.size, drop.sum(), k64[g].size, drop[g].sum()]
res = {t: {"lookups": int(v[0]), "dropped_share": v[1] / max(1, v[0]), "gen_dropped_share": v[3] / max(1, v[2])} for t, v in per.items()}
json.dump(res, open(a.out, "w"), indent=1)
s = sorted(res.items(), key=lambda kv: -kv[1]["dropped_share"])
allv = np.array([v["dropped_share"] for v in res.values()])
print(json.dumps({"tasks": len(res), "mean": float(allv.mean()), "median": float(np.median(allv)),
                  "top10": [(t, round(v["dropped_share"], 4)) for t, v in s[:10]], "p75": float(np.percentile(allv, 75)),
                  "p90": float(np.percentile(allv, 90)), "top30_mean": float(np.mean([v["dropped_share"] for _, v in s[:30]]))}))
