#!/usr/bin/env python3
"""E043: non-uniform expert counts per layer from the tool workload's expert scores (HF expert-importance dataset).

Score = recommended tool keep-set score with block weight W (contribution_dec + (W-1) x block_contribution), the rule
behind the tool E96/E112/E128 keep sets (W = 10). Per layer, kept share(E) = sum of the top-E scores / layer total.
Proxy objective: total lost share over the 40 layers (each layer normalised to its own total, since raw scores are
not comparable across layers). Greedy on marginal gains is optimal because each layer's share curve is concave.

  alloc.py [--w 10] [--floor 64] [--cap 384]
"""
import argparse
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="/home/vu/Workspace/ARC-AGI-2/data/dsv4_prune/importance/experts/tool/scores.json")
ap.add_argument("--w", type=float, default=10)
ap.add_argument("--floor", type=int, default=64)
ap.add_argument("--cap", type=int, default=384)
a = ap.parse_args()
s = json.load(open(a.scores))
S = np.array([np.array(L["contribution_dec"]) + (a.w - 1) * np.array(L["block_contribution"]) for L in s["layers"]])
B = np.array([np.array(L["block_contribution"]) for L in s["layers"]])
srt = -np.sort(-S, axis=1)
share = np.cumsum(srt, axis=1) / srt.sum(axis=1, keepdims=True)  # share[L, E-1] = kept share with E experts
bshare = []
for L in range(40):
    order = np.argsort(-S[L])
    bshare.append(np.cumsum(B[L][order]) / B[L].sum())
bshare = np.array(bshare)


def lost(E):  # E per layer -> total lost share, lost tool-block share
    return sum(1 - share[L, E[L] - 1] for L in range(40)), sum(1 - bshare[L, E[L] - 1] for L in range(40))


def greedy(total):
    E = np.full(40, a.floor)
    left = total - E.sum()
    while left > 0:
        gain = np.array([share[L, E[L]] - share[L, E[L] - 1] if E[L] < a.cap else -1 for L in range(40)])
        L = int(np.argmax(gain)); E[L] += 1; left -= 1
    return E


print("uniform per-layer kept share (score W=%g):" % a.w)
for E in (96, 112, 120, 128):
    ls, lb = lost(np.full(40, E))
    print(f"  E{E}: mean kept {1 - ls / 40:.4f}  min layer {share[:, E - 1].min():.4f} (layer {share[:, E - 1].argmin()})"
          f"  tool-block kept {1 - lb / 40:.4f}")
print("\nnon-uniform at the same total (floor %d): lost share vs uniform" % a.floor)
for E in (96, 112, 128):
    tot = 40 * E
    Eu = np.full(40, E); Eg = greedy(tot)
    lu, lbu = lost(Eu); lg, lbg = lost(Eg)
    print(f"  total {tot} (= E{E}): uniform lost {lu:.3f}, non-uniform lost {lg:.3f} ({100 * (lg / lu - 1):+.1f}%); "
          f"tool-block lost {lbu:.3f} -> {lbg:.3f}; per-layer E min {Eg.min()} max {Eg.max()}")
    print("    E per layer:", " ".join(map(str, Eg)))
print("\nsmallest non-uniform total that loses no more share than uniform E128 / E112:")
for E in (128, 112):
    target = lost(np.full(40, E))[0]
    lo, hi = 40 * 64, 40 * E
    while lo < hi:
        mid = (lo + hi) // 2
        if lost(greedy(mid))[0] <= target:
            hi = mid
        else:
            lo = mid + 1
    Eg = greedy(lo)
    print(f"  uniform E{E} lost {target:.3f}: non-uniform total {lo} = mean {lo / 40:.1f} experts/layer "
          f"({100 * (lo / (40 * E) - 1):+.1f}% expert memory); tool-block lost {lost(Eg)[1]:.3f} vs {lost(np.full(40, E))[1]:.3f}")
    print("    E per layer:", " ".join(map(str, Eg)))
