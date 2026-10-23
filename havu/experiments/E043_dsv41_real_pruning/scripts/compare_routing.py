#!/usr/bin/env python3
"""E043: compare expert routing between two workloads (e.g. tool-use vs no-tool) from REAP sufficient statistics.

  compare_routing.py --a stats_all.npz --a-name tool --b stats_nt.npz --b-name notool --out compare.json
Per layer: effective experts exp(H(routing frequency)); routing share covered by the top-E experts (frequency and
contribution ranking) for several E; overlap of the two workloads' top-E contribution sets; and the routing share of
workload B kept by workload A's keep set (how well a tool-derived set serves no-tool traffic, and vice versa).
"""
import argparse
import json

import numpy as np

L, E = 40, 384
ES = [16, 24, 32, 40, 48, 64, 72, 96, 128, 192]


def load(p):
    z = np.load(p)
    N = z["N"].sum(axis=(1, 2)).astype(np.float64)
    Adec = np.where(np.arange(L)[:, None] >= 20, z["A"][:, :, 1].sum(1), z["A"].sum(axis=(1, 2)))
    return N, Adec


def topk(score, e):
    keep = np.zeros((L, E), bool)
    for li in range(L):
        keep[li, np.lexsort((np.arange(E), -score[li]))[:e]] = True
    return keep


def eff(N):
    p = N / N.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return np.exp(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-name", default="tool")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-name", default="notool")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    NA, AA = load(a.a)
    NB, AB = load(a.b)
    out = {"effective_experts": {a.a_name: eff(NA).round(2).tolist(), a.b_name: eff(NB).round(2).tolist()},
           "coverage": {}, "overlap": {}, "cross": {}}
    for e in ES:
        ka, kb = topk(AA, e), topk(AB, e)
        fa, fb = topk(NA, e), topk(NB, e)
        out["coverage"][e] = {
            f"{a.a_name}_freq": float(((NA * fa).sum(1) / NA.sum(1)).mean()),
            f"{a.b_name}_freq": float(((NB * fb).sum(1) / NB.sum(1)).mean()),
            f"{a.a_name}_contrib": float(((NA * ka).sum(1) / NA.sum(1)).mean()),
            f"{a.b_name}_contrib": float(((NB * kb).sum(1) / NB.sum(1)).mean()),
            f"{a.b_name}_contrib_min_layer": float(((NB * kb).sum(1) / NB.sum(1)).min()),
        }
        out["overlap"][e] = float(((ka & kb).sum(1) / e).mean())
        out["cross"][e] = {f"{a.b_name}_traffic_kept_by_{a.a_name}_set": float(((NB * ka).sum(1) / NB.sum(1)).mean()),
                           f"{a.a_name}_traffic_kept_by_{a.b_name}_set": float(((NA * kb).sum(1) / NA.sum(1)).mean())}
    json.dump(out, open(a.out, "w"), indent=1)
    ea, eb = out["effective_experts"][a.a_name], out["effective_experts"][a.b_name]
    print(f"effective experts mean: {a.a_name} {np.mean(ea):.1f}  {a.b_name} {np.mean(eb):.1f}")
    for e in ES:
        c = out["coverage"][e]
        print(f"E{e:3d}: coverage {a.a_name} {c[f'{a.a_name}_contrib']:.3f}  {a.b_name} {c[f'{a.b_name}_contrib']:.3f} "
              f"(min layer {c[f'{a.b_name}_contrib_min_layer']:.3f})  overlap {out['overlap'][e]:.2f}  "
              f"{a.b_name} traffic kept by {a.a_name} set {out['cross'][e][f'{a.b_name}_traffic_kept_by_{a.a_name}_set']:.3f}")


if __name__ == "__main__":
    main()
