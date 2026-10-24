#!/usr/bin/env python3
"""E042 P6: time the MXFP4 indexer dense-logits kernel with and without PR #154's q predecode.

Decode-shaped inputs: `rows` queries (one per sequence), each scoring `width` compressed keys from its own
pages. Random E2M1 bytes with UE8M0 scales 2^-9..2^-2, so every logit is finite. Checks that both paths give
bit-identical logits and reports the median kernel time over 20 launches (CUDA events).
"""

import argparse
import json
import statistics

import torch

import vllm.model_executor.kernels.attention.dsa.dsv41_indexer as ix


def inputs(rows, width, page=128):
    g = torch.Generator(device="cuda").manual_seed(rows * 100003 + width)
    nb = (width + page - 1) // page

    def rbytes(shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g)

    q = rbytes((rows, 32, 64))
    qs = rbytes((rows, 32, 4), 118, 126)
    k = rbytes((rows * nb, page, 68))
    k[..., 64:] = rbytes((rows * nb, page, 4), 118, 126)
    w = torch.randn(rows, 32, device="cuda", generator=g)
    bt = torch.arange(rows * nb, dtype=torch.int32, device="cuda").view(rows, nb)
    ends = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    return q, qs, k, w, bt, ends


def run(args, predecode, iters=20):
    ix._use_q_predecode = lambda device: predecode
    out = ix.dsv41_mxfp4_dense_logits(*args[:5], args[5], args[6]).clone()
    times = []
    for _ in range(3):
        ix.dsv41_mxfp4_dense_logits(*args[:5], args[5], args[6])
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        ix.dsv41_mxfp4_dense_logits(*args[:5], args[5], args[6])
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return out, statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json")
    a = p.parse_args()
    rows_out = []
    for rows in (16, 64, 128):
        for width in (512, 4096, 16384, 32768, 65536):
            q, qs, k, w, bt, ends = inputs(rows, width)
            args = (q, qs, k, w, bt, ends, width)
            ref, t_off = run(args, False)
            new, t_on = run(args, True)
            fin = torch.isfinite(ref)
            same = torch.equal(fin, torch.isfinite(new)) and torch.equal(ref[fin], new[fin]) and torch.equal(
                ref[~fin].nan_to_num(), new[~fin].nan_to_num())  # NaN-aware: NaN != NaN under torch.equal
            r = {"rows": rows, "width": width, "ms_inline": round(t_off, 4), "ms_predecode": round(t_on, 4),
                 "speedup": round(t_off / t_on, 2), "bit_identical": bool(same),
                 "frac_finite": round(fin.float().mean().item(), 4)}
            rows_out.append(r)
            print(json.dumps(r), flush=True)
            del q, qs, k, w, bt, ends, ref, new
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"device": torch.cuda.get_device_name(), "rows": rows_out}, f, indent=1)


if __name__ == "__main__":
    main()
