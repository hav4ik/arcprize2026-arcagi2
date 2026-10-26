#!/usr/bin/env python3
"""E054 (c1): upstream #58678 layer check on one GPU with the real E128 Engram wkv weights.

On TP4, #58678 makes rank r compute wkv's output columns [r N/4, (r+1) N/4) (N = 25600) with
the same MXFP8 GEMM on an N/4-column weight shard, then all_gather(dim=-1) concatenates the
four results in rank order (tests/test_engram_wkv_tp.py proves the loader slicing and the
gather order on CPU, with the real vLLM classes). Every rank feeds the same input (the
Engram rows are all-gathered before wkv), and the post-wkv kernel is unchanged. So the
Engram output on TP4 is bit-identical to the replicated one iff, for every token count T,
    concat_r GEMM(x, W[r-th quarter])  ==  GEMM(x, W)     (bitwise),
which is what this script measures, on one GPU, with:
  - the real classes: ReplicatedLinear (old) and ColumnParallelLinear(tp_rank=r, tp_size=4,
    gather_output=True) (new), the checkpoint's DeepseekV4FP8Config (MXFP8, [32, 32] ue8m0
    scales), the real weight loaders, and the server's kernel choice for --linear-backend
    humming (HummingMxfp8LinearKernel; --backend auto picks what the server would pick
    without the flag, the Ada MXFP8 kernel on SM89, whose split-K depends on N);
  - the real E128 tensors layers.{1,14}.engram.wkv.{weight,scale} from --model;
  - T in the decode graph sizes (1..64), 128, 512 and 2048 (a prefill chunk);
  - self-determinism first (each GEMM twice), then old vs new;
  - then the Engram output itself: _fused_engram_post_wkv_kernel on fixed hidden states
    [T, 4, 5120] with kv_old vs kv_new (bitwise; max |diff| if not);
  - per-rank parameter bytes after Humming's repack (the real per-GPU saving).

  /root/dsv41/venv/bin/python gpu/engram_wkv_ab.py --model /root/dsv41/model-e128t --json runs/engram_wkv_ab.json
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import torch

TP = 4
TOKENS = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 128, 512, 2048)


def find_tensors(model: Path):
    idx = json.load(open(model / "model.safetensors.index.json"))["weight_map"]
    pat = re.compile(r"^(?:language_model\.)?(?:model\.)?layers\.(\d+)\.engram\.wkv\.(weight|scale|weight_scale|weight_scale_inv)$")
    found = {}
    for k, f in idx.items():
        m = pat.match(k)
        if m:
            found.setdefault(int(m.group(1)), {})["weight" if m.group(2) == "weight" else "scale"] = (k, f)
    if not found:
        sys.exit("no engram.wkv tensors in the checkpoint index")
    from safetensors import safe_open
    out = {}
    for layer, kv in sorted(found.items()):
        t = {}
        for role, (k, f) in kv.items():
            with safe_open(str(model / f), "pt", device="cpu") as fh:
                t[role] = fh.get_tensor(k)
        out[layer] = (t["weight"], t["scale"], {r: k for r, (k, _) in kv.items()})
    return out


def quant_config(model: Path):
    cfg = json.load(open(model / "config.json"))
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if q is None:
        sys.exit("no quantization_config in config.json")
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
    return DeepseekV4FP8Config.from_config(q), q


class Patch:
    def __init__(self, obj, **attrs):
        self.obj, self.attrs = obj, attrs

    def __enter__(self):
        self.saved = {k: getattr(self.obj, k) for k in self.attrs}
        for k, v in self.attrs.items():
            setattr(self.obj, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.obj, k, v)


def build(qc, layer_id, weight, scale, backend, tp_rank=None):
    import vllm.model_executor.kernels.linear as klin
    import vllm.model_executor.layers.linear as lin
    import vllm.model_executor.parameter as vp

    n, k = weight.shape
    prefix = f"language_model.model.layers.{layer_id}.engram.wkv"
    saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # the server builds the model under the model dtype
    try:
        with Patch(klin, _get_linear_backend=lambda: backend), \
             Patch(vp, get_tensor_model_parallel_rank=lambda: 0, get_tensor_model_parallel_world_size=lambda: 1), \
             Patch(lin, get_tensor_model_parallel_rank=lambda: 0, get_tensor_model_parallel_world_size=lambda: 1), \
             torch.device("cuda"):
            if tp_rank is None:
                layer = lin.ReplicatedLinear(k, n, bias=False, quant_config=qc, return_bias=False, prefix=prefix)
            else:
                layer = lin.ColumnParallelLinear(k, n, bias=False, gather_output=True, quant_config=qc,
                                                 return_bias=False, prefix=prefix, tp_rank=tp_rank, tp_size=TP)
            params = dict(layer.named_parameters())
            params["weight"].weight_loader(params["weight"], weight.cuda())
            params["weight_scale"].weight_loader(params["weight_scale"], scale.cuda())
            layer.quant_method.process_weights_after_loading(layer)
    finally:
        torch.set_default_dtype(saved_dtype)
    kernel = type(getattr(layer.quant_method, "kernel", None)).__name__
    nbytes = sum(p.numel() * p.element_size() for p in layer.parameters()) + \
        sum(b.numel() * b.element_size() for b in layer.buffers())
    return layer, kernel, nbytes


def gemm(layer, x):
    return layer.quant_method.apply(layer, x)


def time_us(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return statistics.median(ts)


def post_wkv(eg, hidden, kv, qw, kw):
    T, hc, dim = hidden.shape
    out = torch.empty_like(hidden)
    block = 1 << (dim - 1).bit_length()
    eg._fused_engram_post_wkv_kernel[(T * hc,)](
        hidden, kv, qw, kw, hidden, out, T,
        hidden.stride(0), hidden.stride(1), hidden.stride(2), kv.stride(0), kv.stride(1),
        qw.stride(0), qw.stride(1), kw.stride(0), kw.stride(1), 0,
        out.stride(0), out.stride(1), out.stride(2), 1e-6, 1e-6,
        DIM=dim, HC_MULT=hc, BLOCK_SIZE=block, HAS_MASK=False, num_warps=8 if block >= 2048 else 4)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/root/dsv41/model-e128t")
    p.add_argument("--backend", default="humming", help="humming (production) or auto (Ada kernel on SM89)")
    p.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    p.add_argument("--hc-mult", type=int, default=4)
    p.add_argument("--json")
    a = p.parse_args()

    import vllm  # noqa: F401
    import vllm.models.deepseek_v4_1.common.engram as eg

    model = Path(a.model)
    qc, qraw = quant_config(model)
    tensors = find_tensors(model)
    head = {"device": torch.cuda.get_device_name(), "backend": a.backend, "quant_config": qraw,
            "layers": {L: {"keys": keys, "weight": [list(w.shape), str(w.dtype)], "scale": [list(s.shape), str(s.dtype)]}
                       for L, (w, s, keys) in tensors.items()}}
    print(json.dumps(head), flush=True)
    rows, mems, ok = [], [], True
    for L, (w, s, _) in tensors.items():
        n = w.shape[0]
        full, k_full, b_full = build(qc, L, w, s, a.backend)
        shards = [build(qc, L, w, s, a.backend, r) for r in range(TP)]
        k_shard, b_shard = shards[0][1], shards[0][2]
        mem = {"layer": L, "kernel_full": k_full, "kernel_shard": k_shard, "bytes_full": b_full,
               "bytes_shard": b_shard, "saved_bytes_per_gpu": b_full - b_shard}
        mems.append(mem)
        print(json.dumps(mem), flush=True)
        dim = n // (a.hc_mult + 1)
        g = torch.Generator(device="cuda").manual_seed(58678 + L)
        qw = torch.randn(a.hc_mult, dim, device="cuda", generator=g).to(torch.bfloat16)
        kw = torch.randn(a.hc_mult, dim, device="cuda", generator=g).to(torch.bfloat16)
        for T in map(int, a.tokens.split(",")):
            # Engram rows after dequant: e4m3 values x 2^-6..2^-2, as bf16 (fixed seed)
            raw = torch.randint(0, 256, (T, w.shape[1]), dtype=torch.uint8, device="cuda", generator=g)
            raw[(raw & 0x7F) == 0x7F] = 0x38
            x = (raw.view(torch.float8_e4m3fn).float()
                 * torch.pow(2.0, torch.randint(-6, -1, (T, 1), device="cuda", generator=g).float())).to(torch.bfloat16)
            o1, o2 = gemm(full, x), gemm(full, x)
            s1 = torch.cat([gemm(sh, x) for sh, _, _ in shards], dim=-1)
            s2 = torch.cat([gemm(sh, x) for sh, _, _ in shards], dim=-1)
            det_old = torch.equal(o1.view(torch.int16), o2.view(torch.int16))
            det_new = torch.equal(s1.view(torch.int16), s2.view(torch.int16))
            same = torch.equal(o1.view(torch.int16), s1.view(torch.int16))
            hidden = torch.randn(T, a.hc_mult, dim, device="cuda", generator=g).to(torch.bfloat16)
            e_old, e_new = post_wkv(eg, hidden, o1, qw, kw), post_wkv(eg, hidden, s1, qw, kw)
            e_same = torch.equal(e_old.view(torch.int16), e_new.view(torch.int16))
            d = (o1.float() - s1.float()).abs()
            ed = (e_old.float() - e_new.float()).abs()
            r = {"layer": L, "T": T, "old_deterministic": det_old, "new_deterministic": det_new,
                 "wkv_bit_identical": same, "engram_out_bit_identical": e_same,
                 "wkv_frac_diff": round(float((d > 0).float().mean()), 6), "wkv_max_abs_diff": float(d.max()),
                 "wkv_max_abs": float(o1.float().abs().max()),
                 "engram_frac_diff": round(float((ed > 0).float().mean()), 6), "engram_max_abs_diff": float(ed.max()),
                 "us_full_gemm": round(time_us(lambda: gemm(full, x)), 1),
                 "us_shard_gemm": round(time_us(lambda: gemm(shards[0][0], x)), 1)}
            ok &= det_old and det_new and same and e_same
            rows.append(r)
            print(json.dumps(r), flush=True)
        del full, shards
        torch.cuda.empty_cache()
    summary = {"summary": True, "all_bit_identical": ok,
               "saved_bytes_per_gpu_all_layers": sum(m["saved_bytes_per_gpu"] for m in mems),
               "note": "bit-identical wkv + Engram output for every T => #58678 is exact on this kernel"}
    print(json.dumps(summary), flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({**head, "memory": mems, "rows": rows, **summary}, open(a.json, "w"), indent=1, default=str)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
