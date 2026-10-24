# E043: how faithfully can integer group formats hold unsloth's GGUF expert values? (layer 36, a few experts, CPU)
import json, sys, numpy as np, torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
sys.path.insert(0, "/workspace/dsv4/code/patches")
from e043_humming import mxfp4_to_bf16
M = "/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/"
idx = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
r = GGUFReader("/workspace/dsv4/data/unsloth/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00005-of-00006.gguf")
T = {t.name: t for t in r.tensors}
def orig(L, e, w):
    k = f"layers.{L}.ffn.experts.{e}.{w}"
    with safe_open(M + idx[k + ".weight"], "pt") as f: q = f.get_tensor(k + ".weight")
    with safe_open(M + idx[k + ".scale"], "pt") as f: s = f.get_tensor(k + ".scale")
    return mxfp4_to_bf16(q, s, "cpu").float()
def rtn(w, bits, g):
    n, k = w.shape; maxq = 2 ** bits - 1
    x = w.view(n, k // g, g); lo = x.amin(-1).clamp(max=0); hi = x.amax(-1).clamp(min=0)
    s = ((hi - lo) / maxq).clamp(min=1e-12).to(torch.bfloat16).float(); z = torch.round(-lo / s).clamp(0, maxq)
    q = torch.clamp(torch.round(x / s[..., None]) + z[..., None], 0, maxq)
    return ((q - z[..., None]) * s[..., None]).view(n, k)
L = 36
for gname, w in (("ffn_gate_exps", "w1"), ("ffn_down_exps", "w2")):
    t = T[f"blk.{L}.{gname}.weight"]
    for e in (0, 77, 300):
        wq = torch.from_numpy(dequantize(t.data[e], t.tensor_type).astype(np.float32))
        wo = orig(L, e, w)
        den = wo.pow(2).sum()
        cos = float((wq * wo).sum() / (wq.norm() * wo.norm()))
        row = {"tensor": gname, "type": t.tensor_type.name, "expert": e, "shape": list(wq.shape), "cos_vs_orig": round(cos, 4),
               "iq_rel_err_vs_orig": round(float((wq - wo).pow(2).sum() / den), 4)}
        for bits in (4, 5, 6):
            for g in (16, 32):
                row[f"u{bits}g{g}_extra"] = round(float((rtn(wq, bits, g) - wq).pow(2).sum() / den), 5)
        print(json.dumps(row))
