#!/usr/bin/env python3
"""E066: reconstruction error of candidate low-bit formats for the Engram table, on a sample of real rows.

Source: the shipped MXFP8 rows (FP8 E4M3 values, one UE8M0 scale per 32). Formats, bits per value incl. scales:
  mxfp4 (OCP)    E2M1 + UE8M0 / 32, scale floor(log2 amax) - 2                  4.25  (vLLM PR 61530 / NVIDIA layout)
  mxfp4 (best)   same storage, per group the lower-MSE of the two UE8M0 choices 4.25  (E066 fp4_engram_chunked.py)
  nvfp4          E2M1 + FP8 E4M3 scale / 16 + one FP32 per tensor            4.5
  int4 (ue8m0)   symmetric int [-7, 7] + UE8M0 / 32                             4.25
  int4 (bf16)    symmetric int [-7, 7] + BF16 scale / 32                        4.5
  mxfp6          E2M3 + UE8M0 / 32                                              6.25
Prints relative MSE and relative L2 error vs the MXFP8 source.
  engram_format_error.py <engram file> <weight key> <scale file> <scale key> [--rows 2000000]
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "E043_dsv41_real_pruning" / "scripts"))
from make_fp4_engram import requant, to_e2m1  # noqa: E402


def grid_round(x, grid):
    g = grid.to(x.dtype)
    a = x.abs().clamp(max=float(g[-1]))
    idx = torch.bucketize(a, (g[1:] + g[:-1]) / 2)
    return torch.sign(x) * g[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wfile"); ap.add_argument("wkey"); ap.add_argument("sfile"); ap.add_argument("skey")
    ap.add_argument("--rows", type=int, default=2_000_000)
    a = ap.parse_args()
    torch.set_num_threads(32)
    with safe_open(a.wfile, "pt") as f:
        w8 = f.get_slice(a.wkey)[: a.rows]
    start = int(a.wkey.split("rows_")[1]) if "rows_" in a.wkey else 0
    with safe_open(a.sfile, "pt") as f:
        s8 = f.get_slice(a.skey)[start:start + w8.shape[0]]
    n = w8.shape[0]
    x = w8.float().view(n, 8, 32) * torch.exp2(s8.view(torch.uint8).float() - 127)[..., None]
    ref = x.pow(2).sum()
    out = {}
    amax = x.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -126)
    e = torch.floor(torch.log2(amax)) - 2
    out["mxfp4 (OCP)    4.25"] = to_e2m1(x / torch.exp2(e)) * torch.exp2(e)
    nw, ns, _, _ = requant(w8, s8)
    out["mxfp4 (best)   4.25"] = nw.float().view(n, 8, 32) * torch.exp2(ns.view(torch.uint8).float() - 127)[..., None]
    x16 = x.view(n, 16, 16)
    g = x.abs().max() / (6.0 * 448.0)
    bs = (x16.abs().amax(-1, keepdim=True) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn).float().clamp(min=2.0 ** -9)
    out["nvfp4          4.5 "] = (to_e2m1(x16 / (bs * g)) * bs * g).view(n, 8, 32)
    e = torch.ceil(torch.log2(amax / 7.0))
    out["int4 (ue8m0)   4.25"] = torch.round(x / torch.exp2(e)).clamp(-7, 7) * torch.exp2(e)
    sc = (amax / 7.0).to(torch.bfloat16).float()
    out["int4 (bf16)    4.5 "] = torch.round(x / sc).clamp(-7, 7) * sc
    e23 = torch.tensor([0.0] + [i / 8 for i in range(1, 8)] + [1 + i / 8 for i in range(8)] + [2 + i / 4 for i in range(8)]
                       + [4 + i / 2 for i in range(8)])
    e = torch.floor(torch.log2(amax)) - 2
    out["mxfp6          6.25"] = grid_round(x / torch.exp2(e), e23) * torch.exp2(e)
    print(f"rows {n:,} (key {a.wkey})")
    print("format         bits  rel MSE   rel L2")
    for k, v in out.items():
        r = float((v - x).pow(2).sum() / ref)
        print(f"{k}  {100 * r:6.3f}%  {100 * r ** 0.5:5.1f}%")


if __name__ == "__main__":
    main()
