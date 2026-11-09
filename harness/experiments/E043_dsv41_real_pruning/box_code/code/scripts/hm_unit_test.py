# E043 low-bit: unit checks on one real expert (GPU 4): decoder vs DeepGEMM reference, Humming round trip, RTN error.
import json, struct, torch
from safetensors import safe_open
from vllm import e043_humming as H
from vllm.third_party.deep_gemm.utils.math import _dequantize_from_fp4_e2m1
import humming.utils.weight as hw
from humming import dtypes
M = "/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/"
idx = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
dev = torch.device("cuda", 0)
for L, E in [(5, 7), (30, 200)]:
    for w in ("w1", "w2"):
        kw, ks = f"layers.{L}.ffn.experts.{E}.{w}.weight", f"layers.{L}.ffn.experts.{E}.{w}.scale"
        with safe_open(M + idx[kw], "pt") as f:
            q = f.get_tensor(kw)
        with safe_open(M + idx[ks], "pt") as f:
            s = f.get_tensor(ks)
        mine = H.mxfp4_to_bf16(q, s, dev).float()
        u = q.view(torch.uint8).to(dev)
        un = torch.zeros((u.shape[0], u.shape[1] * 2), dtype=torch.int32, device=dev)
        un[:, ::2] = (u & 0x0F).int(); un[:, 1::2] = (u >> 4).int()
        ref = _dequantize_from_fp4_e2m1(un) * torch.exp2(s.view(torch.uint8).to(dev).float() - 127).repeat_interleave(32, -1)
        print(f"L{L} E{E} {w} {tuple(mine.shape)}: max|mine-ref| {float((mine - ref).abs().max()):.3g}, rms {float(mine.pow(2).mean().sqrt()):.4g}")
        for bits in (2, 3, 4):
            for g in (64, 128):
                dt = getattr(dtypes, f"uint{bits}")
                qw, sc, zp, _ = hw.quantize_weight(mine.to(torch.bfloat16), dtype=dt, scale_dtype=dtypes.bfloat16, group_size=g,
                                                    has_zero_point=True, pack=True)
                deq = hw.dequantize_weight(qw, sc, zp, None, dt, packed=True)
                rel = float((deq - mine).pow(2).mean() / mine.pow(2).mean())
                print(f"     uint{bits} g{g} asym RTN: rel weight MSE {rel:.4f}  (q {tuple(qw.shape)} {qw.dtype}, s {tuple(sc.shape)} {sc.dtype}, z {tuple(zp.shape)} {zp.dtype})")
