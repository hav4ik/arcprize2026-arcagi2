#!/usr/bin/env python3
"""E066 follow-up to E068 (the fork attended bidirectionally over image spans): how far candidate 1's keep set moves
when its calibration replays run on the fixed (causal) path.

Inputs are class_stats.py aggregate files (N, A, W, pos: sums, so they add and subtract) and keep sets [40, 384]:
  --both-bidir  E066's combined stats (image replay on the bidirectional path + text replay)
  --img-causal / --txt  this run's image replay (fixed path) and text replay (text has no images: unaffected)
  --c1-keep     candidate 1's keep set; --new-keep  class_stats.py select on img-causal + txt (C1's class weights)
Writes stats_img_bidir.npz (= both-bidir - txt) and stats_both_causal.npz next to --out, prints and saves a JSON
report: per-layer overlap new vs C1, the kept share of each class's routing under the causal stats for both sets,
and the image-position routing kept by C1 on either path.
  keep_shift.py --both-bidir stats_both_bidir.npz --img-causal stats_img_causal.npz --txt stats_txt.npz \
      --c1-keep c1_keep.npy --new-keep sel_causal/keep_..._uniform_avg128.npy --out keep_shift.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["prompt", "gen", "block", "image"]


def load(p):
    z = np.load(p)
    return {k: z[k] for k in ("N", "A", "W", "pos")}


def kept_share(N, keep):
    """Per class: share of all expert selections (summed over layers) that land in the keep set."""
    tot = N.sum(axis=(0, 2)).astype(np.float64)
    kept = np.einsum("lce,le->c", N.astype(np.float64), keep.astype(np.float64))
    return {c: round(float(kept[i] / tot[i]), 4) if tot[i] else None for i, c in enumerate(CLASSES)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("both-bidir", "img-causal", "txt", "c1-keep", "new-keep", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    bb, ic, tx = load(a.both_bidir), load(a.img_causal), load(a.txt)
    ib = {k: bb[k] - tx[k] for k in bb}  # E066's image replay alone (bidirectional path)
    # the text stream must be E066's own (positions per class equal); per-expert counts differ by run-to-run routing
    # noise between servers, so rare experts can go slightly negative: clip them and report how many
    assert (ib["pos"] == ic["pos"]).all(), f"text positions differ from E066's: {ib['pos']} vs image {ic['pos']}"
    neg = {"entries": int((ib["N"] < 0).sum()), "selections": int(-ib["N"][ib["N"] < 0].sum()),
           "share_of_selections": float(-ib["N"][ib["N"] < 0].sum() / max(1, ib["N"][ib["N"] > 0].sum()))}
    for k in ("N", "A", "W"):
        ib[k] = np.maximum(ib[k], 0)
    np.savez(out.parent / "stats_img_bidir.npz", **ib)
    bc = {k: ic[k] + tx[k] for k in ic}
    np.savez(out.parent / "stats_both_causal.npz", **bc)
    c1, new = np.load(a.c1_keep).astype(bool), np.load(a.new_keep).astype(bool)
    assert c1.shape == new.shape == (40, 384) and (c1.sum(1) == 128).all() and (new.sum(1) == 128).all()
    ov = (c1 & new).sum(axis=1)
    rep = {
        "positions": {"img_bidir": dict(zip(CLASSES, ib["pos"].tolist())), "img_causal": dict(zip(CLASSES, ic["pos"].tolist())),
                      "txt": dict(zip(CLASSES, tx["pos"].tolist()))},
        "same_image_stream": bool((ib["pos"] == ic["pos"]).all()),
        "img_bidir_clipped": neg,
        "overlap_new_vs_c1": {"mean": round(float(ov.mean()), 2), "min": int(ov.min()),
                              "swapped_mean": round(float(128 - ov.mean()), 2),
                              "layers_swapped_ge4": [int(i) for i in np.flatnonzero(128 - ov >= 4)],
                              "per_layer_swapped": (128 - ov).tolist()},
        # reference: B32's shipped E128 vs C1 swap 3.2 per layer on average (dev NLL -0.0027 image, -0.0020 text)
        "kept_share_causal_stats": {"c1": kept_share(bc["N"], c1), "new": kept_share(bc["N"], new)},
        "image_routing_kept_by_c1": {"bidir": kept_share(ib["N"], c1)["image"], "causal": kept_share(ic["N"], c1)["image"]},
        "gen_routing_kept_by_c1_image_replay": {"bidir": kept_share(ib["N"], c1)["gen"], "causal": kept_share(ic["N"], c1)["gen"]},
    }
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "positions"}, indent=1))


if __name__ == "__main__":
    main()
