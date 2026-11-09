#!/usr/bin/env python3
"""E054 diagnostic for #58678 under Humming: how large is the full-vs-sharded wkv difference, in bf16 ulps?

Uses gpu/engram_wkv_ab.py's real classes and loaders (ReplicatedLinear vs 4 x ColumnParallelLinear, Humming).
Per layer and T:
  - old (full N=25600) vs new (4 shards of N=6400, concatenated): ulp distance histogram (bit-pattern exact);
  - each against an fp64 reference x @ dequant(W)^T (W = e4m3 x 2^e8m0 per 32x32 block; x is the bf16 input),
    as |out - ref| / ulp(ref): mean and max, and the fraction of outputs that are the correctly rounded value;
  - Humming self-noise: REPS extra calls of each, max ulp distance to the first call;
  - the Engram output (post-wkv kernel) old vs new ulp histogram.

  python wkv_ulp.py --model /root/dsv41/model-e128t --tokens 1,36,48,64,512,2048 --reps 4
"""
import argparse
import importlib.util
import json

import torch

AB = "/root/e054/src/E054_exact_cherrypicks/gpu/engram_wkv_ab.py"


def ordered(t):
    b = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b >= 0, b, -(b & 0x7FFF))


def ulp_dist(a, b):
    return (ordered(a) - ordered(b)).abs()


def hist(d):
    n = d.numel()
    return {"eq": round(float((d == 0).sum()) / n, 5), "1ulp": round(float((d == 1).sum()) / n, 5),
            "2ulp": round(float((d == 2).sum()) / n, 5), "gt2ulp": round(float((d > 2).sum()) / n, 6),
            "max_ulp": int(d.max())}


def ulp_of(ref):
    # bf16 ulp at |ref|: 2^(floor(log2|ref|) - 7); floor at the smallest normal's ulp
    a = ref.abs().clamp_min(2.0 ** -126)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def err_vs_ref(out, ref64):
    u = ulp_of(ref64)
    e = (out.double() - ref64).abs() / u
    rn = ref64.to(torch.bfloat16)  # correctly rounded (fp64 -> bf16 rounds once... via fp32 below to be safe)
    rn = ref64.float().to(torch.bfloat16)
    return {"mean_err_ulp": round(float(e.mean()), 4), "max_err_ulp": round(float(e.max()), 3),
            "frac_correctly_rounded": round(float((out == rn).float().mean()), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/dsv41/model-e128t")
    p.add_argument("--tokens", default="1,36,48,64,512,2048")
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--backend", default="humming")
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("ewa", AB)
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    import vllm  # noqa: F401
    import vllm.models.deepseek_v4_1.common.engram as eg
    from pathlib import Path
    model = Path(a.model)
    qc, _ = ab.quant_config(model)
    tensors = ab.find_tensors(model)
    for L, (w, s, _) in tensors.items():
        full, kf, _ = ab.build(qc, L, w, s, a.backend)
        shards = [ab.build(qc, L, w, s, a.backend, r)[0] for r in range(ab.TP)]
        # fp64 dequantized weight: e4m3 x 2^(e8m0 - 127) per [32, 32] block
        sc = s.view(torch.uint8).cuda().double().sub(127).exp2()
        wd = w.cuda().double() * sc.repeat_interleave(32, 0).repeat_interleave(32, 1)
        n = w.shape[0]
        dim = n // 5
        g = torch.Generator(device="cuda").manual_seed(54_058 + L)
        qw = torch.randn(4, dim, device="cuda", generator=g).to(torch.bfloat16)
        kw = torch.randn(4, dim, device="cuda", generator=g).to(torch.bfloat16)
        for T in map(int, a.tokens.split(",")):
            raw = torch.randint(0, 256, (T, w.shape[1]), dtype=torch.uint8, device="cuda", generator=g)
            raw[(raw & 0x7F) == 0x7F] = 0x38
            x = (raw.view(torch.float8_e4m3fn).float()
                 * torch.pow(2.0, torch.randint(-6, -1, (T, 1), device="cuda", generator=g).float())).to(torch.bfloat16)
            old = ab.gemm(full, x)
            new = torch.cat([ab.gemm(sh, x) for sh in shards], dim=-1)
            self_old = max(int(ulp_dist(old, ab.gemm(full, x)).max()) for _ in range(a.reps))
            self_new = max(int(ulp_dist(new, torch.cat([ab.gemm(sh, x) for sh in shards], -1)).max())
                           for _ in range(a.reps))
            self_old_frac = max(float((ulp_dist(old, ab.gemm(full, x)) > 0).float().mean()) for _ in range(a.reps))
            ref = x.double() @ wd.T
            hidden = torch.randn(T, 4, dim, device="cuda", generator=g).to(torch.bfloat16)
            e_old, e_new = ab.post_wkv(eg, hidden, old, qw, kw), ab.post_wkv(eg, hidden, new, qw, kw)
            rec = {"layer": L, "T": T, "kernel": kf, "old_vs_new": hist(ulp_dist(old, new)),
                   "old_vs_ref": err_vs_ref(old, ref), "new_vs_ref": err_vs_ref(new, ref),
                   "self_noise_max_ulp_old": self_old, "self_noise_frac_old": round(self_old_frac, 5),
                   "self_noise_max_ulp_new": self_new,
                   "engram_out_old_vs_new": hist(ulp_dist(e_old, e_new))}
            print(json.dumps(rec), flush=True)
        del full, shards, wd
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
