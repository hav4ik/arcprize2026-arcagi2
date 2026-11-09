#!/usr/bin/env python3
"""E066: shrink a compact Engram table to a smaller row budget, keeping a subset of its rows.

The compact export (E043 export_pruned.py) stores, per Engram layer, the kept rows in original-id order
(`layers.L.engram.embed.weight` / `.scale`, whole or as `.rows_<start>` chunks) and `embed_tokens.row_map` (original
row id -> compact row, -1 dropped). This keeps the most-used rows of the current table under a byte budget: rows of
--mask (the table's own keep set, e.g. the tool 64 GiB mask) ranked by census counts (r0 + r1, engram_census.py),
globally over both layers, ties by (layer, row), the same rule as `engram_census.py select`. Rows outside the table are
never added. Writes the new Engram files (weight in <= --chunk-rows row chunks, scale and row_map whole), their
index entries and the new text_config.engram_compact_kept (kept rows per hash column, from the census geometry).

  engram_subset.py --src export/b32_e128 --mask e043tool_top64gib.npz --counts counts_calib_img.npz \
      --geo engram_geometry.npz --budget-gib 48 --out export/engram_b32_48
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROW_BYTES = 256 + 8  # FP8 values + 8 E8M0 scales per row


def load_layer(src, wm, layer):
    """The compact table of one layer: weight [n, 256], scale [n, 8], row_map [rows]."""
    pre = f"layers.{layer}.engram.embed"
    parts = {"weight": [], "scale": []}
    row_map = None
    by_file = defaultdict(list)
    for k, f in wm.items():
        if k.startswith(pre + ".") or k == f"layers.{layer}.engram.embed_tokens.row_map":
            by_file[f].append(k)
    for f, keys in by_file.items():
        with safe_open(str(src / f), framework="pt") as sf:
            for k in keys:
                if k.endswith("row_map"):
                    row_map = sf.get_tensor(k)
                    continue
                kind = "weight" if ".weight" in k else "scale"
                start = int(k.split(".rows_")[1]) if ".rows_" in k else 0
                parts[kind].append((start, sf.get_tensor(k)))
    out = {kind: torch.cat([t for _, t in sorted(v, key=lambda x: x[0])]) for kind, v in parts.items()}
    return out["weight"], out["scale"], row_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mask", required=True, help="packbits keep mask of the source table (layer0, layer1)")
    ap.add_argument("--counts", required=True, help="engram_census.py counts (l{0,1}_r{0,1})")
    ap.add_argument("--geo", required=True, help="engram_census.py geometry (offsets, primes)")
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--chunk-rows", type=int, default=19_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "config.json"))
    tc = cfg.get("text_config", cfg)
    layers = tc["engram_layer_ids"]
    wm = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    m = np.load(a.mask)
    z = np.load(a.counts)
    rows_total = [len(z[f"l{h}_r0"]) for h in range(2)]
    keep_src = [np.unpackbits(m[f"layer{h}"], bitorder="little")[:rows_total[h]].astype(bool) for h in range(2)]
    cnt = [z[f"l{h}_r0"].astype(np.int64) + z[f"l{h}_r1"].astype(np.int64) for h in range(2)]
    # rank the source table's rows: count desc, then (layer, row) asc
    rid = np.concatenate([np.flatnonzero(keep_src[h]) for h in range(2)])
    lay = np.concatenate([np.full(int(keep_src[h].sum()), h, dtype=np.int8) for h in range(2)])
    val = np.concatenate([cnt[h][keep_src[h]] for h in range(2)])
    order = np.lexsort((rid, lay, -val))
    k = min(int(a.budget_gib * 2**30 // ROW_BYTES), len(order))
    sel = order[:k]
    keep_new = []
    for h in range(2):
        b = np.zeros(rows_total[h], dtype=bool)
        b[rid[sel][lay[sel] == h]] = True
        assert not (b & ~keep_src[h]).any()
        keep_new.append(b)
    geo = np.load(a.geo)
    kept_cols = [[int(keep_new[h][int(geo["offsets"][h, j]):int(geo["offsets"][h, j]) + int(geo["primes"][h].reshape(-1)[j])].sum())
                  for j in range(24)] for h in range(2)]
    stats = {"budget_gib": a.budget_gib, "rows_kept": [int(b.sum()) for b in keep_new],
             "rows_source": [int(b.sum()) for b in keep_src], "kept_gib": int(sum(b.sum() for b in keep_new)) * ROW_BYTES / 2**30,
             "census_lookups_kept": float(val[sel].sum() / max(1, val.sum())), "min_count_kept": int(val[sel[-1]]) if k else None,
             "mask_sha256": [hashlib.sha256(np.packbits(b, bitorder="little").tobytes()).hexdigest() for b in keep_new]}
    print(json.dumps(stats))
    files, weight_map, n = [], {}, 0
    plan = []
    for h, layer in enumerate(layers):
        w, s, rm = load_layer(src, wm, layer)
        ids = np.flatnonzero(keep_new[h])
        old = rm[torch.from_numpy(ids)].long()
        assert (old >= 0).all(), "a kept row is missing from the source table"
        nw, ns = w.view(torch.uint8)[old].view(w.dtype), s.view(torch.uint8)[old].view(s.dtype)
        new_rm = torch.full((rows_total[h],), -1, dtype=torch.int32)
        new_rm[torch.from_numpy(ids)] = torch.arange(len(ids), dtype=torch.int32)
        # spot check: 1,000 kept rows equal the source rows through both row maps
        smp = torch.from_numpy(np.random.default_rng(h).choice(ids, size=min(1000, len(ids)), replace=False))
        assert torch.equal(nw.view(torch.uint8)[new_rm[smp].long()], w.view(torch.uint8)[rm[smp].long()])
        plan.append((f"layers.{layer}.engram.embed.scale", {f"layers.{layer}.engram.embed.scale": ns.contiguous(),
                     f"layers.{layer}.engram.embed_tokens.row_map": new_rm}))
        for r0 in range(0, len(ids), a.chunk_rows):
            plan.append((None, {f"layers.{layer}.engram.embed.weight.rows_{r0}": nw[r0:r0 + a.chunk_rows].contiguous()}))
        del w, s, rm
    total = len(plan)
    for i, (_, tensors) in enumerate(plan, 1):
        fn = f"model-engram-{i:05d}-of-{total:05d}.safetensors"
        save_file(tensors, str(out / fn), metadata={"format": "pt"})
        for kname in tensors:
            weight_map[kname] = fn
        files.append(fn)
    json.dump({"weight_map": weight_map, "engram_compact_kept": kept_cols, "stats": stats,
               "masks": {f"layer{h}": None for h in range(2)}}, open(out / "engram_subset.json", "w"), indent=1)
    np.savez(out / f"engram_subset_{a.budget_gib:g}gib_mask.npz",
             **{f"layer{h}": np.packbits(keep_new[h], bitorder="little") for h in range(2)})
    print(json.dumps({"files": len(files), "out": str(out)}))


if __name__ == "__main__":
    main()
