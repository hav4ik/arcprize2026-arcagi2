#!/usr/bin/env python3
"""E043: summary statistics of the Engram row census for the HTML report (small JSON).

  engram_report_stats.py --counts counts_calib.npz --geo engram_geometry.npz --masks-dir masks/ \
      --eval-tok data/tok_eval --out engram_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from engram_census import hash_ids, iter_consumed

ROW_BYTES = 264


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--geo", required=True)
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--eval-tok", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.counts)
    geo = dict(np.load(a.geo))
    out = {"tokens_prompt": int(z["ntok"][0]), "tokens_gen": int(z["ntok"][1])}

    # 1. Histogram of per-row counts (log2 bins), per layer: rows and lookup mass.
    edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 4097, 16385, 65537, 2**40]
    labels = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512",
              "513-1K", "1K-4K", "4K-16K", "16K-64K", ">64K"]
    hist = []
    for l in range(2):
        c = z[f"l{l}_r0"].astype(np.int64) + z[f"l{l}_r1"]
        b = np.digitize(c, edges) - 1  # -1 for zero
        rows = np.bincount(b[c > 0], minlength=len(labels))[: len(labels)]
        mass = np.bincount(b[c > 0], weights=c[c > 0], minlength=len(labels))[: len(labels)]
        hist.append({"layer": [1, 14][l], "unobserved": int((c == 0).sum()), "total": int(len(c)),
                     "rows": rows.tolist(), "lookups": mass.tolist(), "max": int(c.max())})
        # Per bank (order, head): distinct fraction and top-10% concentration.
        banks = []
        for o in range(3):
            for h in range(8):
                j = o * 8 + h
                s, n = int(geo["offsets"][l, j]), int(geo["primes"][l, o, h])
                cb = c[s:s + n]
                nz = np.sort(cb[cb > 0])[::-1]
                top = nz[: max(1, len(nz) // 10)].sum() / max(1, nz.sum())
                banks.append({"order": o + 2, "head": h, "size": n, "distinct": int(len(nz)),
                              "lookups": int(nz.sum()), "top10pct_share": float(top)})
        hist[-1]["banks"] = banks
        # Role split.
        p, g = z[f"l{l}_r0"] > 0, z[f"l{l}_r1"] > 0
        hist[-1]["role_rows"] = {"prompt_only": int((p & ~g).sum()), "gen_only": int((g & ~p).sum()),
                                 "both": int((p & g).sum())}
    out["hist"] = {"labels": labels, "layers": hist}

    # 2. Dense in-sample coverage curve vs retained GiB (global top-frequency order).
    allc = np.concatenate([(z[f"l{l}_r0"].astype(np.int64) + z[f"l{l}_r1"]) for l in range(2)])
    nzc = np.sort(allc[allc > 0])[::-1]
    cum = np.cumsum(nzc) / nzc.sum()
    gib = np.arange(1, len(nzc) + 1) * ROW_BYTES / 2**30
    pts = np.unique(np.concatenate([np.geomspace(1, len(nzc), 120).astype(np.int64) - 1, [len(nzc) - 1]]))
    out["calib_curve"] = [[float(gib[i]), float(cum[i])] for i in pts]
    out["full_table_gib"] = float(len(allc) * ROW_BYTES / 2**30)

    # 3. Held-out (eval-task traces) hit rates per mask, overall and by n-gram order.
    masks = sorted(Path(a.masks_dir).glob("engram_top*gib.npz"), key=lambda p: float(p.stem[10:-3]))
    mbits = {p.stem: [np.unpackbits(np.load(p)[f"layer{l}"], bitorder="little").astype(bool) for l in range(2)]
             for p in masks}
    acc = {k: np.zeros((2, 3, 2)) for k in mbits}  # [role, order, (hits, lookups)]
    all48 = {k: [0, 0] for k in mbits}
    for ids, pos, role in iter_consumed(a.eval_tok):
        h = hash_ids(geo, ids, pos)
        for k, m in mbits.items():
            hit = np.stack([m[l][h[:, l]] for l in range(2)], axis=1)  # [T, 2, 24]
            for r in range(2):
                sel = role == r
                for o in range(3):
                    hh = hit[sel][:, :, o * 8:(o + 1) * 8]
                    acc[k][r, o] += (hh.sum(), hh.size)
            all48[k][0] += int(hit.all(axis=(1, 2)).sum())
            all48[k][1] += len(ids)
    ev = []
    for p in masks:
        k = p.stem
        man = json.load(open(p.with_suffix(".json")))
        A = acc[k]
        ev.append({"mask": k, "budget_gib": man["budget_gib"], "kept_gib": man["kept_gib"],
                   "calib_coverage": man["calib_lookup_coverage"],
                   "eval_hit": float(A[:, :, 0].sum() / A[:, :, 1].sum()),
                   "eval_hit_gen": float(A[1, :, 0].sum() / A[1, :, 1].sum()),
                   "eval_hit_prompt": float(A[0, :, 0].sum() / A[0, :, 1].sum()),
                   "eval_hit_by_order": [float(A[:, o, 0].sum() / A[:, o, 1].sum()) for o in range(3)],
                   "eval_all48": all48[k][0] / all48[k][1]})
    out["eval"] = ev
    out["eval_tokens"] = all48[masks[0].stem][1] if masks else 0
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("tokens_prompt", "tokens_gen", "full_table_gib")}))


if __name__ == "__main__":
    main()
