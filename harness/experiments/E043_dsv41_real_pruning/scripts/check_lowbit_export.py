#!/usr/bin/env python3
"""E043: content check of a GPTQ-only low-bit export against its sources, without a GPU server.

  check_lowbit_export.py --export X --src <checkpoint> --gptq <GPTQ dir, original ids> --keep keep.npy

- gptq/layer_LL.safetensors: e{i}.{w1,w3,w2}.{q,s,z} equal the source GPTQ tensors of original expert kept[i];
- router (ffn.gate.weight / bias / bias_vl) rows equal the checkpoint rows at the kept ids, on every layer;
- config n_routed_experts and the manifest's kept ids match the keep set (per-layer exports, E066: the counts in
  n_routed_experts_per_layer, n_routed_experts = their maximum);
- every indexed tensor exists in its file, and no safetensors file is larger than 5 GB.
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def tensors_of(d, names):
    """Load named tensors from an indexed safetensors dir."""
    wm = json.load(open(Path(d) / "model.safetensors.index.json"))["weight_map"]
    out, by_file = {}, {}
    for n in names:
        if n in wm:
            by_file.setdefault(wm[n], []).append(n)
    for fn, ns in by_file.items():
        with safe_open(str(Path(d) / fn), framework="pt") as f:
            for n in ns:
                out[n] = f.get_tensor(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--gptq", required=True)
    ap.add_argument("--keep", required=True)
    a = ap.parse_args()
    X = Path(a.export)
    keep = np.load(a.keep).astype(bool)
    cfg = json.load(open(X / "config.json"))
    tc = cfg.get("text_config", cfg)
    counts = keep.sum(1).astype(int).tolist()
    E = max(counts)
    assert tc["n_routed_experts"] == E, tc["n_routed_experts"]
    per_layer = len(set(counts)) > 1
    assert tc.get("n_routed_experts_per_layer") == (counts if per_layer else None), tc.get("n_routed_experts_per_layer")
    man = json.load(open(X / "e043_manifest.json"))
    ids = man["keep_original_ids"]
    assert man["experts_per_layer"] == (counts if per_layer else E)
    assert all(list(np.nonzero(keep[L])[0]) == list(ids[str(L)]) for L in range(40)), "manifest ids != keep set"
    # gptq tensors
    pk = re.compile(r"^e(\d+)\.(w[123])\.([qsz])$")
    for L in range(40):
        kept = np.nonzero(keep[L])[0]
        with safe_open(str(X / f"gptq/layer_{L:02d}.safetensors"), framework="pt") as fx, \
                safe_open(str(Path(a.gptq) / f"layer_{L:02d}.safetensors"), framework="pt") as fs:
            keys = list(fx.keys())
            assert len(keys) == 9 * counts[L], (L, len(keys))
            for k in keys:
                i, w, p = pk.match(k).groups()
                if not torch.equal(fx.get_tensor(k), fs.get_tensor(f"e{kept[int(i)]}.{w}.{p}")):
                    raise SystemExit(f"layer {L} {k} differs from source e{kept[int(i)]}")
    print(f"gptq: 40 layers x {'per-layer ' + str(min(counts)) + '-' + str(E) if per_layer else E} experts x 9 tensors "
          "equal the source GPTQ at the kept ids")
    # router rows
    names = [f"layers.{L}.ffn.gate.{p}" for L in range(40) for p in ("weight", "bias", "bias_vl")]
    ours, orig = tensors_of(X, names), tensors_of(a.src, names)
    assert set(ours) == set(orig) and len(orig) >= 80, (sorted(set(orig) - set(ours))[:3], len(orig))
    n = 0
    for name, t in ours.items():
        L = int(name.split(".")[1])
        ref = orig[name][torch.from_numpy(np.nonzero(keep[L])[0])]
        assert torch.equal(t, ref), name
        n += 1
    assert n == len(orig), n
    print(f"router: {n} tensors equal the checkpoint rows at the kept ids")
    # index and file sizes
    wm = json.load(open(X / "model.safetensors.index.json"))["weight_map"]
    by_file = {}
    for k, fn in wm.items():
        by_file.setdefault(fn, set()).add(k)
    for fn, ks in by_file.items():
        with safe_open(str(X / fn), framework="pt") as f:
            assert ks <= set(f.keys()), fn
    sizes = [os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(X) for f in fs if f.endswith(".safetensors")]
    assert max(sizes) <= 5_000_000_000, max(sizes)
    print(f"index: {len(wm)} tensors in {len(by_file)} files; {len(sizes)} safetensors, largest {max(sizes) / 1e9:.3f} GB")
    print("CONTENT-CHECK-OK")


if __name__ == "__main__":
    main()
