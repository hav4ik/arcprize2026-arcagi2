#!/usr/bin/env python3
"""E043: expert-routing statistics for the report (small JSON), from the REAP sufficient statistics.

  expert_report_stats.py --stats stats_all.npz --toolblock toolblock_all.npz --keep-dirs reap/keep_all reap/protect \
      --out expert_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

L, E = 40, 384
ES = [8, 16, 32, 48, 64, 72, 96, 128, 192, 256, 384]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--toolblock", required=True)
    ap.add_argument("--keep-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.stats)
    N = z["N"].sum(axis=(1, 2)).astype(np.float64)       # selections per layer x expert
    A = z["A"].sum(axis=(1, 2))                           # native amplitude sums
    Adec = np.where(np.arange(L)[:, None] >= 20, z["A"][:, :, 1].sum(1), A)
    tb = np.load(a.toolblock)
    Ntb, Atb = tb["N"].astype(np.float64), tb["A"]
    out = {"layers": L, "experts": E, "selections_per_layer": N.sum(1).tolist(),
           "tool_block_selections_per_layer": Ntb.sum(1).tolist()}
    p = N / N.sum(1, keepdims=True)
    ent = -(np.where(p > 0, p * np.log(p), 0)).sum(1)
    out["effective_experts"] = np.exp(ent).round(2).tolist()          # exp(entropy) of routing frequency
    out["mean_amp"] = (A / np.maximum(N, 1)).mean(1).round(4).tolist()
    out["max_share_expert"] = p.max(1).round(4).tolist()
    ptb = Ntb / Ntb.sum(1, keepdims=True)
    out["effective_experts_toolblock"] = np.exp(-(np.where(ptb > 0, ptb * np.log(ptb), 0)).sum(1)).round(2).tolist()

    def coverage(score):
        """Mean / min over layers of the routing share kept when keeping the top-E experts by `score`."""
        res = {}
        for e in ES:
            keep = np.zeros((L, E), bool)
            for li in range(L):
                keep[li, np.lexsort((np.arange(E), -score[li]))[:e]] = True
            sh = (N * keep).sum(1) / N.sum(1)
            shtb = (Ntb * keep).sum(1) / Ntb.sum(1)
            res[e] = {"mean": float(sh.mean()), "min": float(sh.min()), "tb_mean": float(shtb.mean())}
        return res
    out["coverage"] = {"frequency (oracle for routing share)": coverage(N), "contribution": coverage(A),
                       "plain REAP": coverage(np.where(N > 0, A / np.maximum(N, 1), 0)),
                       "tool-block contribution": coverage(Atb), "w30": coverage(Adec + 29 * Atb)}

    keeps = {}
    for d in a.keep_dirs:
        for f in sorted(Path(d).glob("keep_*.npy")):
            keeps[f.stem.replace("keep_", "")] = np.load(f).astype(bool)
    final = {k: v for k, v in keeps.items() if k in ("contrib_dec_e64", "contrib_dec_e72", "contrib_dec_e48",
                                                     "contrib_dec_sp0_w30_e72", "contrib_dec_sp0_w30_e64",
                                                     "contrib_dec_sp0_w30_e48", "reap_e64", "freq_e64",
                                                     "contrib_e64", "wsum_e64", "contrib_dec_sp72_e72")}
    out["per_layer_kept_share"] = {k: ((N * v).sum(1) / N.sum(1)).round(4).tolist() for k, v in final.items()}
    out["per_layer_kept_share_toolblock"] = {k: ((Ntb * v).sum(1) / Ntb.sum(1)).round(4).tolist()
                                             for k, v in final.items()}
    names = sorted(final)
    out["jaccard"] = {"names": names, "matrix": [[float(((final[x] & final[y]).sum(1) / (final[x] | final[y]).sum(1)).mean())
                                                  for y in names] for x in names]}
    top_tb = np.zeros((L, E), bool)
    top_gen = np.zeros((L, E), bool)
    for li in range(L):
        top_tb[li, np.argsort(-Atb[li])[:72]] = True
        top_gen[li, np.argsort(-Adec[li])[:72]] = True
    out["toolblock_vs_general_top72_overlap"] = ((top_tb & top_gen).sum(1) / 72).round(3).tolist()
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("effective_experts",)})[:300])


if __name__ == "__main__":
    main()
