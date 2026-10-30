#!/usr/bin/env python3
"""E066 check (metric only, nothing is selected from it): how well Engram tables chosen from the TRAINING census
cover the Engram lookups of eval-task traces. Budgets rank the rows of the source table (--mask) by training counts
(engram_census.py counts, r0 + r1), as engram_subset.py does; coverage is then measured on the eval counts.
  engram_evalcheck.py --train-counts counts_calib_img.npz --eval-counts counts_evalcheck.npz --mask tool64.npz
"""
import argparse
import numpy as np

ROW = 264
ap = argparse.ArgumentParser()
ap.add_argument("--train-counts", required=True)
ap.add_argument("--eval-counts", required=True)
ap.add_argument("--mask", required=True)
ap.add_argument("--budget-gib", type=float, nargs="*", default=[64, 56, 48, 40, 32, 24, 16])
a = ap.parse_args()
tz, ez, m = np.load(a.train_counts), np.load(a.eval_counts), np.load(a.mask)
rows = [len(tz[f"l{h}_r0"]) for h in range(2)]
keep = [np.unpackbits(m[f"layer{h}"], bitorder="little")[:rows[h]].astype(bool) for h in range(2)]
tc = np.concatenate([(tz[f"l{h}_r0"].astype(np.int64) + tz[f"l{h}_r1"])[keep[h]] for h in range(2)])
ep = np.concatenate([ez[f"l{h}_r0"].astype(np.int64)[keep[h]] for h in range(2)])
eg = np.concatenate([ez[f"l{h}_r1"].astype(np.int64)[keep[h]] for h in range(2)])
e_all = sum(int(ez[f"l{h}_r0"].sum()) + int(ez[f"l{h}_r1"].sum()) for h in range(2))
t_all = sum(int(tz[f"l{h}_r0"].sum()) + int(tz[f"l{h}_r1"].sum()) for h in range(2))
print(f"eval tokens (prompt, generated): {ez['ntok'].tolist()} | eval lookups inside the table: {(ep.sum() + eg.sum()) / e_all:.2%}"
      f" (training census: {tc.sum() / t_all:.2%})")
order = np.argsort(-tc, kind="stable")
cs_t, cs_e, cs_g, cs_p = (np.cumsum(x[order]) for x in (tc, ep + eg, eg, ep))
print("budget | train lookups kept | EVAL lookups kept | eval generated kept | eval prompt kept  (of the table's own)")
for gib in a.budget_gib:
    k = min(len(order), int(gib * 2**30 // ROW))
    print(f"{gib:4.0f} GiB | {cs_t[k-1]/cs_t[-1]:7.2%} | {cs_e[k-1]/cs_e[-1]:7.2%} | {cs_g[k-1]/cs_g[-1]:7.2%} | {cs_p[k-1]/cs_p[-1]:7.2%}")
never_t = tc == 0
print(f"rows never seen in training: {never_t.sum()/1e6:.1f}M; their share of eval lookups: {(ep + eg)[never_t].sum() / (ep + eg).sum():.2%}")
