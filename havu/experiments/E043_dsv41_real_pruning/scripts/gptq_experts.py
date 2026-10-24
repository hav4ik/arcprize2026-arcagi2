#!/usr/bin/env python3
"""E043 low-bit round: per-expert GPTQ of DeepSeek-V4.1-Flash-1017's routed experts into Humming's integer format.

Maya-S-style recipe (error-feedback rounding of every expert against its own input statistics), written into the
format Humming runs on L4: unsigned integer codes with a per-(row, group) scale and integer zero point,
w = (q - z) * s, groups along K. Router, attention, shared expert and all other weights are not touched.

Per layer and expert e (calibration rows from calx_transpose.py; teacher = the native model's MoE inputs):
  X_e = rows routed to e, r_e = their top-k weights;  H13 = X_e^T X_e  (shared by w1 = gate and w3 = up)
  GPTQ(w1 | w3, H13, bits13)  ->  h = silu(min(X_e W1q^T, 10)) * clamp(X_e W3q^T, -10, 10)  (the native clamps)
  u = r_e * h  (routing weight inside, as in the native forward);  H2 = u^T u;  GPTQ(w2, H2, bits2)
GPTQ: standard column-wise error feedback (no act-order), percdamp 0.01, block 128, groups computed from the
error-updated weights, per-group min/max range (including 0) with an MSE clipping search; scales rounded to the
kernel's scale dtype before use, so the stored (q, s, z) dequantize exactly to the GPTQ result. Experts with fewer
than --min-tokens calibration rows use H = I (plain rounding with the same clipping search). --heldout F keeps a
random share of each expert's rows out of H and reports the out-of-sample error proxy (GPTQ can fit the in-sample
statistics of an expert with fewer rows than input dimensions almost perfectly); --percdamp sets the damping.

Output: <out>/layer_LL.safetensors with e{E}.{w1,w3,w2}.{q,s,z} in the exact layout of
humming.utils.weight.quantize_weight(..., pack=True) (read by e043_humming.precomputed), and layer_LL.json with
per-expert rows, relative weight MSE and the relative output-error proxy tr(dW H dW^T) / tr(W H W^T) for GPTQ and
for plain rounding (RTN), plus an exact-decode check through humming.utils.weight.dequantize_weight.

  gptq_experts.py --calx /dev/shm/e043/calx_by_layer --layers 0 8 16 24 32 --bits13 2 --bits2 3 --group13 128 \
      --group2 64 [--keep keep.npy] --out /workspace/dsv4/export/gptq_b2b3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
SHRINKS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
N_EXPERTS, HID, INTER = 384, 5120, 2304
CHUNK = 4096


def load_mxfp4(model_dir, index, name, device):
    with safe_open(f"{model_dir}/{index[name + '.weight']}", "pt") as f:
        q = f.get_tensor(name + ".weight")
    with safe_open(f"{model_dir}/{index[name + '.scale']}", "pt") as f:
        s = f.get_tensor(name + ".scale")
    w = q.to(device).view(torch.uint8)
    codes = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(w.shape[0], -1).long()
    mag = E2M1.to(device)[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    return val * torch.exp2(s.to(device).view(torch.uint8).float() - 127).repeat_interleave(32, dim=-1)


def group_params(wg, maxq, sdt):
    """wg [..., g] -> scale, zero [...] minimizing the group's MSE over clipping factors (range includes 0)."""
    xmin = wg.amin(-1).clamp(max=0)
    xmax = wg.amax(-1).clamp(min=0)
    best = None
    for f in SHRINKS:
        lo, hi = xmin * f, xmax * f
        s = ((hi - lo) / maxq).clamp(min=1e-8).to(sdt).float()
        z = torch.round(-lo / s).clamp(0, maxq)
        q = torch.clamp(torch.round(wg / s[..., None]) + z[..., None], 0, maxq)
        err = ((q - z[..., None]) * s[..., None] - wg).pow(2).sum(-1)
        if best is None:
            best = [err, s, z]
        else:
            m = err < best[0]
            best = [torch.where(m, err, best[0]), torch.where(m, s, best[1]), torch.where(m, z, best[2])]
    return best[1], best[2]


def rtn(W, bits, group, sdt):
    B, n, k = W.shape
    maxq = 2 ** bits - 1
    wg = W.view(B, n, k // group, group)
    s, z = group_params(wg, maxq, sdt)
    q = torch.clamp(torch.round(wg / s[..., None]) + z[..., None], 0, maxq)
    return ((q - z[..., None]) * s[..., None]).view(B, n, k)


def gptq(W, H, bits, group, sdt, block=128, percdamp=0.01):
    """W [B, n, k] fp32, H [B, k, k] fp32 -> codes int32 [B, n, k], scales [B, n, k/g], zeros [B, n, k/g], Wq."""
    assert block % group == 0
    B, n, k = W.shape
    maxq = 2 ** bits - 1
    W = W.clone()
    H = H.clone()
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    d[d == 0] = 1.0
    d += percdamp * d.mean(-1, keepdim=True)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros((B, n, k), dtype=torch.int32, device=W.device)
    Wq = torch.zeros_like(W)
    S = torch.zeros((B, n, k // group), device=W.device)
    Z = torch.zeros_like(S)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        W1 = W[:, :, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi = Hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            g = c // group
            if c % group == 0:
                S[:, :, g], Z[:, :, g] = group_params(W1[:, :, i:i + group], maxq, sdt)
            s, z = S[:, :, g], Z[:, :, g]
            w = W1[:, :, i]
            qc = torch.clamp(torch.round(w / s) + z, 0, maxq)
            q = (qc - z) * s
            codes[:, :, c] = qc.to(torch.int32)
            Wq[:, :, c] = q
            err = (w - q) / Hi[:, i, i][:, None]
            W1[:, :, i:] -= err[:, :, None] * Hi[:, i, i:][:, None, :]
            E1[:, :, i] = err
        if i2 < k:
            W[:, :, i2:] -= torch.bmm(E1, Hinv[:, i1:i2, i2:])
    return codes, S, Z, Wq


def rel_out_err(W, Wq, H):
    dW = Wq - W
    return ((torch.bmm(dW, H) * dW).sum((1, 2)) / (torch.bmm(W, H) * W).sum((1, 2)).clamp(min=1e-30))


def pack(codes, S, Z, bits, sdt_torch):
    """Exact layout of humming quantize_weight(pack=True): codes packed along K; zero point packed along N."""
    from humming import ops
    q = ops.pack_weight(codes.contiguous(), bits)
    z = ops.pack_weight(Z.to(torch.int32).transpose(-1, -2).contiguous(), bits).transpose(-1, -2).contiguous()
    return q, S.to(sdt_torch).contiguous(), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calx", required=True)
    ap.add_argument("--model-dir", default="/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--bits13", type=int, default=2)
    ap.add_argument("--bits2", type=int, default=3)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--group13", type=int, default=0, help="gate/up group size (default --group)")
    ap.add_argument("--group2", type=int, default=0, help="down group size (default --group)")
    ap.add_argument("--keep", default=None, help="bool [40, 384] keep set: quantize only kept experts (pruned ones are "
                    "never routed; the loader rounds them itself)")
    ap.add_argument("--scale-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-tokens", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damping, fraction of mean diag(H)")
    ap.add_argument("--heldout", type=float, default=0.0, help="share of each expert's rows kept out of H for an "
                    "out-of-sample error estimate (those rows are then not used for quantization)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g13, g2 = a.group13 or a.group, a.group2 or a.group
    keep = np.load(a.keep).astype(bool) if a.keep else np.ones((40, N_EXPERTS), bool)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", 0)
    sdt = getattr(torch, a.scale_dtype)
    index = json.load(open(f"{a.model_dir}/model.safetensors.index.json"))["weight_map"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from humming import dtypes
    import humming.utils.weight as hw
    for L in a.layers:
        t0 = time.time()
        X = torch.from_numpy(np.array(np.load(f"{a.calx}/x_{L:02d}.npy", mmap_mode="r"))).to(dev).view(torch.bfloat16)
        ids = torch.from_numpy(np.load(f"{a.calx}/ids_{L:02d}.npy").astype(np.int64)).to(dev)
        rw = torch.from_numpy(np.load(f"{a.calx}/w_{L:02d}.npy").astype(np.float32)).to(dev)
        tensors, rows = {}, []
        kept = [int(e) for e in np.nonzero(keep[L])[0]]
        for b0 in range(0, len(kept), a.batch):
            es = kept[b0:b0 + a.batch]
            W13 = torch.stack([torch.cat([load_mxfp4(a.model_dir, index, f"layers.{L}.ffn.experts.{e}.w1", dev),
                                          load_mxfp4(a.model_dir, index, f"layers.{L}.ffn.experts.{e}.w3", dev)]) for e in es])
            W2 = torch.stack([load_mxfp4(a.model_dir, index, f"layers.{L}.ffn.experts.{e}.w2", dev) for e in es])
            H13 = torch.zeros((len(es), HID, HID), device=dev)
            H13h = torch.zeros((len(es), HID, HID), device=dev) if a.heldout else None
            rows_e, ntok = [], []
            for j, e in enumerate(es):
                m = ids == e
                idx = torch.nonzero(m.any(-1)).squeeze(-1)
                r = (rw * m).sum(-1)[idx]
                ntok.append(int(idx.numel()))
                if ntok[-1] < a.min_tokens:
                    H13[j] = torch.eye(HID, device=dev)
                    rows_e.append(None)
                    continue
                perm = torch.randperm(idx.numel(), generator=torch.Generator(device="cpu").manual_seed(L * 1000 + e)).to(dev)
                nh = int(idx.numel() * a.heldout)
                cal, hold = perm[nh:], perm[:nh]
                rows_e.append((idx[cal], r[cal], idx[hold], r[hold]))
                for c0 in range(0, cal.numel(), CHUNK):
                    xe = X[idx[cal[c0:c0 + CHUNK]]].float()
                    H13[j] += xe.T @ xe
                if H13h is not None:
                    for c0 in range(0, hold.numel(), CHUNK):
                        xe = X[idx[hold[c0:c0 + CHUNK]]].float()
                        H13h[j] += xe.T @ xe
            c13, s13, z13, W13q = gptq(W13, H13, a.bits13, g13, sdt, percdamp=a.percdamp)

            def h2_of(j, rows, rws):
                H = torch.zeros((INTER, INTER), device=dev)
                for c0 in range(0, rows.numel(), CHUNK):
                    gu = X[rows[c0:c0 + CHUNK]].float() @ W13q[j].T
                    u = rws[c0:c0 + CHUNK, None] * (torch.nn.functional.silu(gu[:, :INTER].clamp(max=10.0))
                                                    * gu[:, INTER:].clamp(-10.0, 10.0))
                    H += u.T @ u
                return H
            H2 = torch.zeros((len(es), INTER, INTER), device=dev)
            H2h = torch.zeros((len(es), INTER, INTER), device=dev) if a.heldout else None
            for j in range(len(es)):
                if rows_e[j] is None:
                    H2[j] = torch.eye(INTER, device=dev)
                    if H2h is not None:
                        H2h[j] = torch.eye(INTER, device=dev)
                        H13h[j] = torch.eye(HID, device=dev)
                    continue
                H2[j] = h2_of(j, rows_e[j][0], rows_e[j][1])
                if H2h is not None:
                    H2h[j] = h2_of(j, rows_e[j][2], rows_e[j][3])
            c2, s2, z2, W2q = gptq(W2, H2, a.bits2, g2, sdt, percdamp=a.percdamp)
            W13r, W2r = rtn(W13, a.bits13, g13, sdt), rtn(W2, a.bits2, g2, sdt)
            e13_g, e2_g = rel_out_err(W13, W13q, H13), rel_out_err(W2, W2q, H2)
            e13_r, e2_r = rel_out_err(W13, W13r, H13), rel_out_err(W2, W2r, H2)
            if H13h is not None:  # held-out rows (never seen by GPTQ): the honest comparison
                h13_g, h13_r = rel_out_err(W13, W13q, H13h), rel_out_err(W13, W13r, H13h)
                h2_g, h2_r = rel_out_err(W2, W2q, H2h), rel_out_err(W2, W2r, H2h)
            for j, e in enumerate(es):
                for shard, sl, (c, s, z), bits in (("w1", slice(0, INTER), (c13, s13, z13), a.bits13),
                                                   ("w3", slice(INTER, 2 * INTER), (c13, s13, z13), a.bits13),
                                                   ("w2", slice(None), (c2, s2, z2), a.bits2)):
                    q_, s_, z_ = pack(c[j, sl], s[j, sl], z[j, sl], bits, sdt)
                    tensors[f"e{e}.{shard}.q"], tensors[f"e{e}.{shard}.s"], tensors[f"e{e}.{shard}.z"] = q_.cpu(), s_.cpu(), z_.cpu()
                rows.append({"expert": e, "tokens": ntok[j],
                             "w13_rel_out_err": {"gptq": float(e13_g[j]), "rtn": float(e13_r[j])},
                             "w2_rel_out_err": {"gptq": float(e2_g[j]), "rtn": float(e2_r[j])},
                             **({"heldout_w13_rel_out_err": {"gptq": float(h13_g[j]), "rtn": float(h13_r[j])},
                                 "heldout_w2_rel_out_err": {"gptq": float(h2_g[j]), "rtn": float(h2_r[j])}}
                                if H13h is not None else {}),
                             "w13_rel_w_mse": float((W13q[j] - W13[j]).pow(2).mean() / W13[j].pow(2).mean()),
                             "w2_rel_w_mse": float((W2q[j] - W2[j]).pow(2).mean() / W2[j].pow(2).mean())})
            if b0 == 0:  # exact-decode check through Humming for the first quantized expert of the layer
                chk = {}
                e0 = es[0]
                for shard, ref, bits in (("w1", W13q[0, :INTER], a.bits13), ("w2", W2q[0], a.bits2)):
                    deq = hw.dequantize_weight(tensors[f"e{e0}.{shard}.q"].to(dev), tensors[f"e{e0}.{shard}.s"].to(dev),
                                               tensors[f"e{e0}.{shard}.z"].to(dev), None, getattr(dtypes, f"uint{bits}"), packed=True)
                    chk[shard] = float((deq.float() - ref).abs().max())
            del W13, W2, H13, H2, W13q, W2q, W13r, W2r, H13h, H2h
        save_file(tensors, str(out / f"layer_{L:02d}.safetensors"))
        def mean(key, kind):
            v = [r[key][kind] for r in rows if key in r]
            return float(np.mean(v)) if v else None
        err = {f"{p}{m}_{k}": mean(f"{p}{m}_rel_out_err", k) for p in ("", "heldout_") for m in ("w13", "w2") for k in ("gptq", "rtn")}
        summary = {"layer": L, "bits13": a.bits13, "bits2": a.bits2, "group13": g13, "group2": g2, "scale_dtype": a.scale_dtype,
                   "keep": a.keep, "experts_quantized": len(kept),
                   "calib_rows": int(X.shape[0]), "experts_below_min_tokens": sum(r["tokens"] < a.min_tokens for r in rows),
                   "percdamp": a.percdamp, "heldout": a.heldout, "mean_rel_out_err": err,
                   "decode_check_max_abs": chk, "seconds": time.time() - t0, "experts": rows}
        json.dump(summary, open(out / f"layer_{L:02d}.json", "w"))
        print(json.dumps({k: v for k, v in summary.items() if k != "experts"}), flush=True)
        del X, ids, rw, tensors


if __name__ == "__main__":
    main()
