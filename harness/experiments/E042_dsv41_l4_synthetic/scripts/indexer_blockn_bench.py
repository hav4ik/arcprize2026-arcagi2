#!/usr/bin/env python3
"""E042: decode-shape micro-benchmark of the fork's V4.1 MXFP4 indexer logits kernel over BLOCK_N / warps / stages.

The fork launches `_mxfp4_indexer_logits_kernel` with BLOCK_N 16, 4 warps, 2 stages: every program re-reads the
whole predecoded q (32 heads x 128 BF16 = 8 KB) to score 16 keys. This times the dense decode path (one row per
request, Q_PREDECODED as with DSV41_INDEXER_Q_PREDECODE=1) at our shapes and checks each variant's logits against
the default bit for bit. Random MXFP4 bytes; scales near 2^0.
Usage (on a GPU box, fork venv): indexer_blockn_bench.py [--rows 48,56] [--ctx 16000,36000,64000] [--page 64]
"""
import argparse
import itertools

import torch
import triton

from vllm.model_executor.kernels.attention.dsa import dsv41_indexer as m


def launch(qv, qs, qd, k, w, bt, ends, logits, width, block_n, warps, stages):
    grid, swap = (qv.shape[0], triton.cdiv(width, block_n)), False  # column tiles stay below the grid-y limit here
    m._mxfp4_indexer_logits_kernel[grid](
        qv, qs, qd, k, w, bt, None, ends, None, None, None, logits,
        qv.stride(0), qv.stride(1), qs.stride(0), qs.stride(1), w.stride(0), bt.stride(0), k.stride(0),
        logits.stride(0), bt.shape[1], k.shape[0], width,
        PAGE_SIZE=k.shape[1], ROW_REPEAT=1, PREFILL=False, USE_CANDIDATES=False, CANDIDATE_STRIDE_ROW=0,
        SWAP_GRID_AXES=swap, BLOCK_N=block_n, Q_PREDECODED=True, num_warps=warps, num_stages=stages)


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="48,56")
    ap.add_argument("--ctx", default="16000,36000,64000", help="context tokens; indexer width = ctx / 4")
    ap.add_argument("--page", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    variants = [(16, 4, 2)] + [v for v in itertools.product((32, 64, 128), (4, 8), (2, 3))]
    for rows, ctx in itertools.product(map(int, a.rows.split(",")), map(int, a.ctx.split(","))):
        width = ctx // 4
        pages = triton.cdiv(width, a.page)
        nblocks = rows * pages + 8
        k = torch.randint(0, 256, (nblocks, a.page, 68), dtype=torch.uint8, device=dev, generator=g)
        k[:, :, 64:] = torch.randint(124, 131, (nblocks, a.page, 4), dtype=torch.uint8, device=dev, generator=g)
        perm = torch.randperm(nblocks, device=dev, generator=g)[: rows * pages].to(torch.int32)
        bt = perm.view(rows, pages).contiguous()
        qv = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device=dev, generator=g)
        qs = torch.randint(124, 131, (rows, 32, 4), dtype=torch.uint8, device=dev, generator=g)
        qd = m._predecode_q(qv, qs)
        w = torch.rand((rows, 32), device=dev, generator=g, dtype=torch.float32)
        ends = torch.randint(width // 2, width + 1, (rows,), device=dev, generator=g, dtype=torch.int32)
        ref = torch.empty((rows, width), device=dev, dtype=torch.float32)
        out = torch.empty_like(ref)
        launch(qv, qs, qd, k, w, bt, ends, ref, width, 16, 4, 2)
        kbytes = int(ends.sum()) * 68
        res = []
        for bn, wp, st in variants:
            fn = lambda: launch(qv, qs, qd, k, w, bt, ends, out, width, bn, wp, st)
            try:
                ms = bench(fn)
            except Exception as exc:  # e.g. out of shared memory
                res.append((bn, wp, st, None, f"{type(exc).__name__}"))
                continue
            same = torch.equal(out, ref)
            maxdiff = (out - ref).abs()[torch.isfinite(ref)].max().item()
            res.append((bn, wp, st, ms, "bit-identical" if same else f"max |diff| {maxdiff:.3g}"))
        base = res[0][3]
        print(f"rows {rows}, ctx {ctx} (width {width}, page {a.page}): K bytes {kbytes / 1e6:.1f} MB; "
              f"default {base * 1000:.0f} us = {kbytes / base / 1e6:.0f} GB/s")
        for bn, wp, st, ms, note in sorted(res, key=lambda r: r[3] or 1e9)[:6]:
            print(f"  BLOCK_N {bn:3d} warps {wp} stages {st}: " +
                  (f"{ms * 1000:7.0f} us  x{base / ms:4.2f}  {note}" if ms else note))


if __name__ == "__main__":
    main()
