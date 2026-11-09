"""Verify official block-FP8 dequant convention against base BF16, and compare our FP8block vs merged BF16."""
import json
import torch
from safetensors import safe_open

B = 128
BASE = "/workspace/models/Qwen3.8-27B"
SFT = "/workspace/models/Qwen3.8-27B-e028-sft"
OURS = "/workspace/models/Qwen3.8-27B-e028-sft-FP8block"
OFF = "/workspace/audit_e028/official/layers-3.safetensors"


def get(root, name):
    wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{root}/{wm[name]}", "pt") as f:
        return f.get_tensor(name)


def expand(s, r, c):
    return s.float().repeat_interleave(B, 0).repeat_interleave(B, 1)[:r, :c]


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def ourquant(w):
    w = w.float()
    r, c = w.shape
    wb = w.reshape(r // B, B, c // B, B)
    amax = wb.abs().amax(dim=(1, 3))
    s = (amax / 448.0).to(torch.bfloat16)
    q = (w / expand(s, r, c)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, s, amax


with safe_open(OFF, "pt") as f:
    names = [n for n in f.keys() if not n.endswith("_scale_inv") and f.get_slice(n).get_dtype() == "F8_E4M3"]
    print("official F8 tensors in shard:", len(names))
    for n in sorted(names):
        q = f.get_tensor(n)
        s = f.get_tensor(n + "_scale_inv")
        r, c = q.shape
        base = get(BASE, n).float()
        sft = get(SFT, n).float()
        deq_mul = q.float() * expand(s, r, c)
        deq_div = q.float() / expand(s, r, c)
        st = s.t().contiguous()
        line = f"{n.split('layers.')[1]:28s} shape={tuple(q.shape)} scale={tuple(s.shape)} {s.dtype} | "
        line += f"official*s vs base rel={rel(deq_mul, base):.4f} | official/s rel={rel(deq_div, base):.3e} | "
        # how close is official to "amax/448" recipe applied to base?
        oq, os_, amax = ourquant(base)
        line += f"scale==amax/448(bf16) frac={(os_ == s).float().mean().item():.4f} "
        ratio = (s.float() / (amax / 448)).flatten()
        line += f"scale/(amax/448) min={ratio.min().item():.4f} med={ratio.median().item():.4f} max={ratio.max().item():.4f} "
        line += f"fp8 code match={(oq.view(torch.uint8) == q.view(torch.uint8)).float().mean().item():.4f} "
        line += f"max|code|={q.float().abs().max().item():.0f}"
        print(line)
        # ours
        qo = get(OURS, n)
        so = get(OURS, n + "_scale_inv")
        assert qo.dtype == torch.float8_e4m3fn and so.dtype == torch.bfloat16 and so.shape == s.shape, (qo.dtype, so.dtype, so.shape)
        deq_o = qo.float() * expand(so, r, c)
        print(f"    ours*s vs merged rel={rel(deq_o, sft):.4f} maxabs={(deq_o - sft).abs().max().item():.4e} "
              f"(official quant err on base rel={rel(deq_mul, base):.4f}) ours/s rel={rel(qo.float() / expand(so, r, c), sft):.3e} "
              f"ours-vs-transposed-scale rel={rel(qo.float() * expand(so.t().contiguous(), c, r).t() if r == c else deq_o, sft):.4f} "
              f"merged-vs-base rel={rel(sft, base):.4f} sign agree={(torch.sign(deq_o) == torch.sign(sft)).float().mean().item():.4f} "
              f"nan={torch.isnan(qo.float()).sum().item()}")
