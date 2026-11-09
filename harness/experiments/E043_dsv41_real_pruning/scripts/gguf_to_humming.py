#!/usr/bin/env python3
"""E043: GGUF routed-expert weights (e.g. unsloth UD-IQ2_XXS) -> precomputed Humming tensors for e043_humming.

Lets the vLLM fork run another quantizer's expert weights on the same engine and harness as our models. Each expert
matrix (blk.L.ffn_{gate,up,down}_exps = w1 / w3 / w2) is decoded exactly with gguf-py, then re-stored as unsigned
B-bit integers with group-G asymmetric bf16 scales, w = (q - z) * s, packed in Humming's layout (32 codes per
B-word little-endian bitstream along K; zero points packed along N), i.e. the tuple
humming.utils.weight.quantize_weight(..., pack=True) returns. With B = 6, G = 32 the re-storing adds ~0.04% of the
weight norm (~0.5% of IQ2_S's own error vs the released FP4 weights); uint4 would add ~12%.

  gguf_to_humming.py --gguf-dir data/unsloth/UD-IQ2_XXS --layers 0 1 ... [--keep keep.npy] --bits 6 --group 32 \
      --out /dev/shm/e043/gguf_e128_u6 --workers 64
With --keep, only kept experts are converted and keys use compact ids (rank among the kept ids, ascending), matching
export_pruned.py; without it, all 384 experts with original ids. Output per layer: layer_LL.safetensors with
e{E}.{w1,w3,w2}.{q,s,z}, plus manifest.json.
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

NAMES = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}
_T = {}
_CFG = {}


def _init(gguf_dir, bits, group):
    import torch
    torch.set_num_threads(1)
    from gguf import GGUFReader
    for f in sorted(glob.glob(f"{gguf_dir}/*.gguf")):
        for t in GGUFReader(f).tensors:
            if "_exps" in t.name:
                _T[t.name] = t
    _CFG.update(bits=bits, group=group)


def pack_k(codes, bits):
    """int64 [n, k] -> int32 [n, k * bits / 32]: each 32 codes form a little-endian bitstream of `bits` words."""
    import torch
    n, k = codes.shape
    b = (codes.view(n, k // 32, 32, 1) >> torch.arange(bits)) & 1           # [n, k/32, 32, bits]: bit j of code i
    b = b.reshape(n, k // 32, bits, 32)                                       # stream position i*bits+j -> word, bit
    w = (b * (1 << torch.arange(32, dtype=torch.int64))).sum(-1)              # [n, k/32, bits] as uint32 in int64
    w = torch.where(w >= (1 << 31), w - (1 << 32), w)                         # reinterpret the uint32 bits as int32
    return w.to(torch.int32).reshape(n, k * bits // 32)


def quantize(w, bits, group):
    """float32 [n, k] -> (codes int64 [n, k], scale bf16 [n, k/g], zero int64 [n, k/g], dequantized float32)."""
    import torch
    n, k = w.shape
    maxq = 2 ** bits - 1
    x = w.view(n, k // group, group)
    lo, hi = x.amin(-1).clamp(max=0), x.amax(-1).clamp(min=0)
    s = ((hi - lo) / maxq).clamp(min=1e-12).to(torch.bfloat16)
    sf = s.float()
    z = torch.round(-lo / sf).clamp(0, maxq)
    q = torch.clamp(torch.round(x / sf[..., None]) + z[..., None], 0, maxq)
    return q.view(n, k).long(), s, z.long(), ((q - z[..., None]) * sf[..., None]).view(n, k)


def convert(task):
    import torch
    from gguf.quants import dequantize
    L, e, key = task
    out, err = {}, {}
    for shard, name in NAMES.items():
        t = _T[f"blk.{L}.{name}.weight"]
        w = torch.from_numpy(np.ascontiguousarray(dequantize(t.data[e], t.tensor_type), dtype=np.float32))
        codes, s, z, deq = quantize(w, _CFG["bits"], _CFG["group"])
        # numpy across the process boundary (torch tensors make the parent's unpickling the bottleneck);
        # the bf16 scale travels as its int16 bit pattern
        out[f"e{key}.{shard}.q"] = pack_k(codes, _CFG["bits"]).numpy()
        out[f"e{key}.{shard}.s"] = s.contiguous().view(torch.int16).numpy()
        out[f"e{key}.{shard}.z"] = pack_k(z.T.contiguous(), _CFG["bits"]).T.contiguous().numpy()
        err[shard] = {"type": t.tensor_type.name,
                      "restore_rel_mse": float((deq - w).pow(2).sum() / w.pow(2).sum().clamp(min=1e-30))}
    return L, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(40)))
    ap.add_argument("--keep", default=None)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    keep = np.load(a.keep).astype(bool) if a.keep else np.ones((40, 384), bool)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    man = {"gguf_dir": a.gguf_dir, "bits": a.bits, "group": a.group, "keep": a.keep, "compact_ids": bool(a.keep),
           "layers": {}}
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.gguf_dir, a.bits, a.group)) as pool:
        for L in a.layers:
            ids = np.nonzero(keep[L])[0]
            tasks = [(L, int(e), (i if a.keep else int(e))) for i, e in enumerate(ids)]
            tensors, errs = {}, []
            import torch
            for _, o, err in pool.imap_unordered(convert, tasks, chunksize=2):
                for k_, v in o.items():
                    t = torch.from_numpy(v)
                    tensors[k_] = t.view(torch.bfloat16) if k_.endswith(".s") else t
                errs.append(err)
            save_file(tensors, str(out / f"layer_{L:02d}.safetensors"))
            types = sorted({v["type"] for e_ in errs for v in e_.values()})
            man["layers"][L] = {"experts": len(tasks), "gguf_types": types,
                                "restore_rel_mse_mean": float(np.mean([v["restore_rel_mse"] for e_ in errs for v in e_.values()]))}
            print(json.dumps({"layer": L, **man["layers"][L], "elapsed_s": round(time.time() - t0)}), flush=True)
    json.dump(man, open(out / "manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
