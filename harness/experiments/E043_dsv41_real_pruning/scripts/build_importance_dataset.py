#!/usr/bin/env python3
"""E043: build the standalone "expert + Engram importance" dataset for DeepSeek-V4.1-Flash-1017 (for teammates who
prune at other levels or pair the scores with another quantizer).

  build_importance_dataset.py --data /workspace/dsv4/data --code /workspace/dsv4/code --results <dir> --out <stage>

Writes <stage>/: README.md, index.json, model.json,
  experts/{notool,tool}/scores.json + scores.csv (every layer x expert), raw sufficient statistics (*.npz),
  experts/keep_sets/{notool,tool}/keep_*.npy + experts/keep_sets/index.json,
  engram/{notool,tool}/counts.npz, engram/{notool,tool}/masks/*, engram/geometry.npz, engram/masks_index.json,
  results/ (evaluated keep sets: dev NLL, answer-writing probe, ARC), scripts/.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

L, E = 40, 384
WORKLOADS = {
    "notool": {"stats": "reap/stats_nt2.npz", "block": "reap/answerblock_nt2.npz", "block_name": "answer_block",
               "block_weight": 10, "keep_dirs": ["reap/keep_nt2"], "counts": "engram/counts_nt2_calib.npz",
               "masks": "engram/masks_nt2_calib",
               "source": "notool_e044_* configs of livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036 (E044 "
                         "no-tool harness: official ARC Prize prompt, task summary, effort low + minimal header, one "
                         "turn), 4,931 calibration rollouts (family-disjoint dev split excluded)",
               "replayed_tokens": "20.0M (REAP); 109.2M (Engram census)"},
    "tool": {"stats": "reap/stats_all.npz", "block": "reap/toolblock_all.npz", "block_name": "tool_block",
             "block_weight": 30, "keep_dirs": ["reap/keep_all", "reap/protect"], "counts": "engram/counts_calib.npz",
             "masks": "engram/masks",
             "source": "E031/E036/E038 tool-use rollouts (s20x60 + minimal header, persistent python tool) of "
                       "livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036",
             "replayed_tokens": "10.0M (REAP); 251.8M (Engram census)"},
}


def per_layer_scores(stats, block, wblock):
    z, b = np.load(stats), np.load(block)
    N, A, Q, W, MX = z["N"].astype(np.float64), z["A"], z["Q"], z["W"], z["MX"]
    out = {k: np.zeros((L, E)) for k in (
        "selections", "selections_gen_target", "selection_share", "router_weight_sum", "router_weight_mean",
        "contribution", "contribution_dec", "reap", "amp_sq_sum", "amp_max", "block_selections",
        "block_contribution", "recommended_score", "rank_recommended", "rank_contribution_dec")}
    for li in range(L):
        n, a = N[li].sum((0, 1)), A[li].sum((0, 1))
        out["selections"][li] = n
        out["selections_gen_target"][li] = N[li][:, 1].sum(0)
        out["selection_share"][li] = n / max(1.0, n.sum())
        out["router_weight_sum"][li] = W[li].sum((0, 1))
        out["router_weight_mean"][li] = np.where(n > 0, W[li].sum((0, 1)) / np.maximum(n, 1), 0.0)
        out["contribution"][li] = a
        out["contribution_dec"][li] = A[li][:, 1].sum(0) if li >= 20 else a
        out["reap"][li] = np.where(n > 0, a / np.maximum(n, 1), 0.0)
        out["amp_sq_sum"][li] = Q[li].sum((0, 1))
        out["amp_max"][li] = MX[li].max((0, 1))
        out["block_selections"][li] = b["N"][li]
        out["block_contribution"][li] = b["A"][li]
        out["recommended_score"][li] = out["contribution_dec"][li] + (wblock - 1) * b["A"][li]
        for key, src in (("rank_recommended", "recommended_score"), ("rank_contribution_dec", "contribution_dec")):
            order = np.lexsort((np.arange(E), -out[src][li]))
            rank = np.empty(E)
            rank[order] = np.arange(E)
            out[key][li] = rank
    return out, {"pos_count": z["pos_count"].tolist()}


def write_scores(dst, scores, meta, wl, cfg):
    dst.mkdir(parents=True, exist_ok=True)
    ints = {"selections", "selections_gen_target", "block_selections", "rank_recommended", "rank_contribution_dec"}
    js = {"workload": wl, "source": cfg["source"], "replayed_tokens": cfg["replayed_tokens"],
          "block": cfg["block_name"], "recommended_score": f"contribution_dec + {cfg['block_weight'] - 1} x "
          f"block_contribution ({cfg['block_name']} weight x{cfg['block_weight']})", "positions": meta["pos_count"],
          "layers": [{"layer": li, **{k: ([int(x) for x in v[li]] if k in ints else [float(f"{x:.6g}") for x in v[li]])
                                      for k, v in scores.items()}} for li in range(L)]}
    json.dump(js, open(dst / "scores.json", "w"))
    with open(dst / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = list(scores)
        w.writerow(["layer", "expert", *keys])
        for li in range(L):
            for e in range(E):
                w.writerow([li, e, *[int(scores[k][li, e]) if k in ints else f"{scores[k][li, e]:.6g}" for k in keys]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--results", required=True, help="dir with results files to include (json/jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D, out = Path(a.data), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    files = []
    keep_index = []
    for wl, cfg in WORKLOADS.items():
        sc, meta = per_layer_scores(D / cfg["stats"], D / cfg["block"], cfg["block_weight"])
        d = out / "experts" / wl
        write_scores(d, sc, meta, wl, cfg)
        shutil.copy(D / cfg["stats"], d / "reap_stats.npz")
        shutil.copy(D / cfg["block"], d / f"{cfg['block_name']}_stats.npz")
        kd = out / "experts" / "keep_sets" / wl
        kd.mkdir(parents=True, exist_ok=True)
        for src in cfg["keep_dirs"]:
            for f in sorted((D / src).glob("keep_*.npy")):
                k = np.load(f).astype(bool)
                name = f.name if not (kd / f.name).exists() else f"{Path(src).name}_{f.name}"
                np.save(kd / name, k)
                keep_index.append({"workload": wl, "file": f"experts/keep_sets/{wl}/{name}", "rule": f.stem[5:],
                                   "experts_per_layer": int(k.sum(1)[0]), "uniform": bool((k.sum(1) == k.sum(1)[0]).all()),
                                   "kept_ids": [np.nonzero(k[li])[0].tolist() for li in range(L)]})
        ed = out / "engram" / wl
        ed.mkdir(parents=True, exist_ok=True)
        os.link(D / cfg["counts"], ed / "counts.npz") if os.stat(D / cfg["counts"]).st_dev == os.stat(ed).st_dev \
            else shutil.copy(D / cfg["counts"], ed / "counts.npz")
        shutil.copytree(D / cfg["masks"], ed / "masks")
    json.dump(keep_index, open(out / "experts" / "keep_sets" / "index.json", "w"))
    shutil.copy(D / "engram_geometry.npz", out / "engram" / "geometry.npz")
    masks_index = []
    for wl in WORKLOADS:
        for f in sorted((out / "engram" / wl / "masks").glob("*.json")):
            m = json.load(open(f))
            masks_index.append({"workload": wl, "file": f"engram/{wl}/masks/{f.stem}.npz",
                                **{k: m[k] for k in ("budget_gib", "kept_gib", "kept_fraction", "rows_kept", "rows_total",
                                                     "all_observed_kept", "calib_lookup_coverage") if k in m}})
    json.dump(masks_index, open(out / "engram" / "masks_index.json", "w"), indent=1)
    rd = out / "results"
    rd.mkdir(parents=True)
    for f in sorted(Path(a.results).glob("*")):
        shutil.copy(f, rd / f.name)
    sd = out / "scripts"
    sd.mkdir()
    for f in ("scripts/reap_aggregate.py", "scripts/special_stats.py", "scripts/engram_census.py",
              "scripts/compare_routing.py", "scripts/gptq_experts.py", "patches/e043_obs.py", "patches/e043_humming.py",
              "patches/apply_e043_patches.py", "patches/apply_e043_humming.py"):
        if (Path(a.code) / f).exists():
            shutil.copy(Path(a.code) / f, sd / Path(f).name)
    model = {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
             "backbone_layers": L, "routed_experts_per_layer": E, "top_k": 6, "shared_experts": 1,
             "hidden_size": 5120, "expert_intermediate_size": 2304, "routed_expert_format": "MXFP4 (E2M1 + UE8M0 per 32)",
             "router": "sqrtsoftplus scores; correction bias used for selection only; selected weights renormalized; route scale 1.5",
             "ced_towers": {"encoder_prefill": [0, 19], "decoder": [20, 39]},
             "engram": {"backbone_layers": [1, 14], "rows": [384006168, 384016682], "row_bytes": 264,
                        "row_format": "256 FP8 E4M3 + 8 UE8M0 scales"}}
    json.dump(model, open(out / "model.json", "w"), indent=1)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size})
    json.dump({"files": files}, open(out / "index.json", "w"), indent=1)
    print(json.dumps({"files": len(files), "bytes": sum(f["bytes"] for f in files)}))


if __name__ == "__main__":
    main()
