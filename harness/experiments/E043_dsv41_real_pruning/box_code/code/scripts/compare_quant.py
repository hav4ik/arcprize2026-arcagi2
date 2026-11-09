# E043: relative expert output error tr(dW H dW^T)/tr(W H W^T) for the same E128 experts: unsloth UD-IQ2_XXS vs our
# GPTQ (E128 export) vs plain uint2 RTN. H from the full-model calibration dump (calx_full, same rollouts as GPTQ's
# calibration, so GPTQ has a partial in-sample advantage). Gate/up: H = X^T X of routed tokens; down: H from
# u = r * SwiGLU(X W13_orig) (original weights for all methods).
import glob, json, sys, numpy as np, torch
from safetensors import safe_open
from gguf import GGUFReader
from gguf.quants import dequantize
import humming.utils.weight as hw
from humming import dtypes
sys.path.insert(0, "/workspace/dsv4/code/patches"); sys.path.insert(0, "/workspace/dsv4/code/scripts")
from e043_humming import mxfp4_to_bf16
from gptq_experts import rtn
dev = torch.device("cuda", 0)
M = "/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017/"; idx = json.load(open(M + "model.safetensors.index.json"))["weight_map"]
keep = np.load("/workspace/dsv4/data/reap/keep_nt2/keep_contrib_dec_sp0_w10_e128.npy")
T = {}
for f in sorted(glob.glob("/workspace/dsv4/data/unsloth/UD-IQ2_XXS/*.gguf")):
    for t in GGUFReader(f).tensors:
        if "_exps" in t.name: T[t.name] = t
C = "/workspace/dsv4/data/calx_full_by_layer"
def orig(L, e, w):
    k = f"layers.{L}.ffn.experts.{e}.{w}"
    with safe_open(M + idx[k + ".weight"], "pt") as f: q = f.get_tensor(k + ".weight")
    with safe_open(M + idx[k + ".scale"], "pt") as f: s = f.get_tensor(k + ".scale")
    return mxfp4_to_bf16(q, s, dev).float()
def rel(W, Wq, H): d = Wq - W; return float(((d @ H) * d).sum() / ((W @ H) * W).sum())
for L in [int(x) for x in sys.argv[1:]]:
    X = torch.from_numpy(np.array(np.load(f"{C}/x_{L:02d}.npy", mmap_mode="r"))).view(torch.bfloat16)
    ids = torch.from_numpy(np.load(f"{C}/ids_{L:02d}.npy").astype(np.int64)); rw = torch.from_numpy(np.load(f"{C}/w_{L:02d}.npy").astype(np.float32))
    kept = np.nonzero(keep[L])[0]
    g = safe_open(f"/workspace/dsv4/export/notool_e128_x10_eng56_gptq/gptq/layer_{L:02d}.safetensors", "pt")
    res = {"unsloth": [], "gptq": [], "rtn": []}; res2 = {"unsloth": [], "gptq": [], "rtn": []}; types = set()
    for ci, e in enumerate(kept[::4]):  # every 4th kept expert (32 per layer)
        ci = int(np.nonzero(kept == e)[0][0])
        m = ids == int(e); sel = m.any(-1)
        if sel.sum() < 512: continue
        xs = X[sel]; r_all = (rw * m).sum(-1)[sel]
        W13 = torch.cat([orig(L, e, "w1"), orig(L, e, "w3")]); W2 = orig(L, e, "w2")
        H13 = torch.zeros(5120, 5120, device=dev); H2 = torch.zeros(2304, 2304, device=dev)
        for c0 in range(0, xs.shape[0], 4096):
            x = xs[c0:c0 + 4096].float().to(dev); r = r_all[c0:c0 + 4096].to(dev)
            H13 += x.T @ x
            gu = x @ W13.T; u = r[:, None] * (torch.nn.functional.silu(gu[:, :2304].clamp(max=10)) * gu[:, 2304:].clamp(-10, 10)); H2 += u.T @ u
            del x, gu, u
        uq = {}
        for nm in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            t = T[f"blk.{L}.{nm}.weight"]; types.add(t.tensor_type.name)
            uq[nm] = torch.from_numpy(dequantize(t.data[int(e)], t.tensor_type).astype(np.float32)).to(dev)
        U13 = torch.cat([uq["ffn_gate_exps"], uq["ffn_up_exps"]]); U2 = uq["ffn_down_exps"]
        G = {}
        for sh, bits in (("w1", 2), ("w3", 2), ("w2", 2)):
            q, s, z = (g.get_tensor(f"e{ci}.{sh}.{k}").to(dev) for k in "qsz")
            G[sh] = hw.dequantize_weight(q, s, z, None, dtypes.uint2, packed=True).float()
        G13 = torch.cat([G["w1"], G["w3"]])
        res["unsloth"].append(rel(W13, U13, H13)); res["gptq"].append(rel(W13, G13, H13)); res["rtn"].append(rel(W13, rtn(W13[None], 2, 128, torch.bfloat16)[0], H13))
        res2["unsloth"].append(rel(W2, U2, H2)); res2["gptq"].append(rel(W2, G["w2"], H2)); res2["rtn"].append(rel(W2, rtn(W2[None], 2, 64, torch.bfloat16)[0], H2))
    print(json.dumps({"layer": L, "unsloth_types": sorted(types), "experts": len(res["gptq"]),
                      "gate_up_rel_out_err": {k: round(float(np.mean(v)), 4) for k, v in res.items()},
                      "down_rel_out_err": {k: round(float(np.mean(v)), 4) for k, v in res2.items()}}), flush=True)
