#!/usr/bin/env python3
"""E043: MXFP4 Engram tables, emulated exactly in the released MXFP8 container.

The compact Engram files (export_pruned.py: layers.{1,14}.engram.embed.weight FP8 E4M3 [rows, 256] + .scale E8M0
[rows, 8], one scale per 32 values) are re-rounded to MXFP4: per group of 32, an E8M0 scale and E2M1 values
{0, 0.5, 1, 1.5, 2, 3, 4, 6}. Of the two standard scale choices (OCP floor(log2 amax) - 2 with saturation at 6, or
ceil(log2(amax / 6)) without clipping) the one with the lower group MSE is kept. E2M1 values are exact in E4M3, so
the result is stored back as FP8 + E8M0 and the engine (unchanged) runs true MXFP4 Engram values; a deployment kernel
would store them at 4.25 bits (half the host memory).

  make_fp4_engram.py --src export/notool_e128_x10_eng56_gptq --out export/engram30_fp4 [--chunk 4000000]
Writes the rewritten model-engram-*.safetensors (row maps unchanged) and stats.json (relative MSE vs MXFP8).
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def to_e2m1(x):
    """Round |x| <= 6 to the nearest E2M1 magnitude (ties to the even code), keep the sign."""
    grid = E2M1.to(x.device)
    a = x.abs().clamp(max=6.0)
    mid = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(a, mid, right=True)                       # nearest; exact midpoints go to the upper bucket
    at_mid = (idx > 0) & (a == mid[(idx - 1).clamp(min=0)])
    idx = torch.where(at_mid & (idx % 2 == 1), idx - 1, idx)        # ties to even code index
    return torch.sign(x) * grid[idx]


def requant(w8, s8):
    """w8 FP8 [n, 256], s8 E8M0 [n, 8] (GPU) -> new FP8 [n, 256], new E8M0 [n, 8], squared errors."""
    n = w8.shape[0]
    x = w8.float().view(n, 8, 32) * torch.exp2(s8.view(torch.uint8).float() - 127)[..., None]
    amax = x.abs().amax(-1).clamp(min=2.0 ** -126)
    best_q, best_e, best_err = None, None, None
    for e in (torch.floor(torch.log2(amax)) - 2, torch.ceil(torch.log2(amax / 6.0))):
        e = e.clamp(-127, 127)
        q = to_e2m1(x / torch.exp2(e)[..., None])
        err = (q * torch.exp2(e)[..., None] - x).pow(2).sum(-1)
        if best_q is None:
            best_q, best_e, best_err = q, e, err
        else:
            m = err < best_err
            best_q = torch.where(m[..., None], q, best_q)
            best_e = torch.where(m, e, best_e)
            best_err = torch.minimum(err, best_err)
    zero = amax <= 2.0 ** -126
    best_e = torch.where(zero, torch.zeros_like(best_e), best_e)
    new_w = best_q.view(n, 256).to(torch.float8_e4m3fn)
    new_s = (best_e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return new_w, new_s, best_err.sum(), x.pow(2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=4_000_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda", 0)
    stats = {}
    for f in sorted(Path(a.src).glob("model-engram-*.safetensors")):
        tensors = {}
        with safe_open(str(f), framework="pt") as sf:
            keys = list(sf.keys())
            for k in keys:
                if not k.endswith(".embed.weight"):
                    if not k.endswith(".embed.scale"):
                        tensors[k] = sf.get_tensor(k)
                    continue
                kw, ks = k, k[: -len("weight")] + "scale"
                ws, ss = sf.get_slice(kw), sf.get_slice(ks)
                n = ws.get_shape()[0]
                new_w = torch.empty((n, 256), dtype=torch.float8_e4m3fn)
                new_s = torch.empty((n, 8), dtype=torch.float8_e8m0fnu)
                err = tot = 0.0
                for r0 in range(0, n, a.chunk):
                    r1 = min(n, r0 + a.chunk)
                    w, s, e_, t_ = requant(ws[r0:r1].to(dev), ss[r0:r1].to(dev))
                    new_w[r0:r1], new_s[r0:r1] = w.cpu(), s.cpu()
                    err, tot = err + float(e_), tot + float(t_)
                tensors[kw], tensors[ks] = new_w, new_s
                stats[kw] = {"rows": n, "rel_mse_vs_mxfp8": err / tot}
                print(json.dumps({kw: stats[kw]}), flush=True)
        save_file(tensors, str(out / f.name), metadata={"format": "pt"})
    json.dump(stats, open(out / "fp4_engram_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
