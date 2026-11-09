"""E028: re-quantize the merged BF16 SFT model into the official Qwen/Qwen3.8-27B-FP8 layout.

Block-wise FP8 (e4m3, 128x128 weight blocks, dynamic activations). The tensor set, dtypes and shapes come from the
official checkpoint metadata (official_fp8_meta/tensors.json, from huggingface_hub.get_safetensors_metadata):
F8_E4M3 tensors get a `<name>_scale_inv` (BF16, shape ceil(r/128) x ceil(c/128)); dequant = fp8.float() * scale_inv
broadcast per block (DeepSeek-V3 / Qwen FP8 convention). Everything else is copied in the official dtype.
Scales: amax/448 per block computed in fp32 (tinker_cookbook quantize_blockwise), rounded to BF16 (official dtype),
then the block is re-quantized against the rounded scale so fp8*scale_inv is consistent.

  /opt/sglang/bin/python quantize_blockfp8.py
"""
import datetime
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

SRC = "/workspace/models/Qwen3.8-27B-e028-sft"
DST = "/workspace/models/Qwen3.8-27B-e028-sft-FP8block"
META = "/workspace/e028/official_fp8_meta"
BLOCK = 128
FP8_MAX = 448.0
DT = {"BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F32": "float32", "F16": "float16"}


def quant(w):
    import torch
    from tinker_cookbook.weights._export._quantized import quantize_blockwise
    w = w.float()
    _, s = quantize_blockwise(w, (BLOCK, BLOCK))            # fp32 scale_inv = amax/448
    s = s.to(torch.bfloat16)                                # official scale dtype
    r, c = w.shape
    sf = s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:r, :c]
    q = (w / sf).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


def do_shard(args):
    shard, names, gpu = args
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    off = json.load(open(f"{META}/tensors.json"))
    dev = f"cuda:{gpu}"
    out, maxerr = {}, 0.0
    with safe_open(f"{SRC}/{shard}", "pt", device=dev) as f:
        for n in names:
            t = f.get_tensor(n)
            odt, oshape, _ = off[n]
            assert list(t.shape) == oshape, (n, t.shape, oshape)
            if odt == "F8_E4M3":
                q, s = quant(t)
                sdt, sshape, _ = off[n + "_scale_inv"]
                assert sdt == "BF16" and list(s.shape) == sshape, (n, s.shape, sshape, sdt)
                # round-trip check
                deq = q.float() * s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:t.shape[0], :t.shape[1]]
                rel = ((deq - t.float()).norm() / t.float().norm()).item()
                maxerr = max(maxerr, rel)
                assert rel < 0.05 and torch.isfinite(deq).all(), (n, rel)
                out[n] = q.cpu()
                out[n + "_scale_inv"] = s.cpu()
            else:
                out[n] = t.to(getattr(torch, DT[odt])).cpu()
    save_file(out, f"{DST}/{shard}", metadata={"format": "pt"})
    size = sum(v.numel() * v.element_size() for v in out.values())
    return shard, list(out.keys()), size, maxerr


def main():
    t0 = time.time()
    off = json.load(open(f"{META}/tensors.json"))
    wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    base_keys = {k for k in off if not k.endswith("_scale_inv")}
    assert set(wm) == base_keys, (sorted(set(wm) ^ base_keys)[:10])
    for k in off:
        if k.endswith("_scale_inv"):
            assert off[k[: -len("_scale_inv")]][0] == "F8_E4M3", k
    os.makedirs(DST, exist_ok=True)
    shards = {}
    for n, s in wm.items():
        shards.setdefault(s, []).append(n)
    jobs = [(s, ns, i % 2) for i, (s, ns) in enumerate(sorted(shards.items()))]
    new_map, total = {}, 0
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 6))) as ex:
        for shard, keys, size, err in ex.map(do_shard, jobs):
            for k in keys:
                new_map[k] = shard
            total += size
            print(f"{shard}: {len(keys)} tensors {size/1e9:.2f} GB max rel err {err:.4f} ({time.time()-t0:.0f}s)", flush=True)
    assert set(new_map) == set(off), sorted(set(new_map) ^ set(off))[:10]
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(f"{DST}/model.safetensors.index.json", "w"), indent=2)
    cfg = json.load(open(f"{SRC}/config.json"))
    cfg["quantization_config"] = json.load(open(f"{META}/config.json"))["quantization_config"]
    json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
    for fn in ["chat_template.jinja", "generation_config.json", "merges.txt", "preprocessor_config.json",
               "tokenizer_config.json", "tokenizer.json", "video_preprocessor_config.json", "vocab.json"]:
        if os.path.exists(f"{SRC}/{fn}"):
            shutil.copy2(f"{SRC}/{fn}", f"{DST}/{fn}")
    json.dump({"format": "fp8-block128 (official Qwen3.8-27B-FP8 layout)", "source": SRC,
               "script": "/workspace/e028/quantize_blockfp8.py", "layout_ref": "Qwen/Qwen3.8-27B-FP8",
               "weight_block_size": [BLOCK, BLOCK], "scale_dtype": "bfloat16", "activation_scheme": "dynamic",
               "date": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"},
              open(f"{DST}/e028_quant_info.json", "w"), indent=1)
    print(f"done total {total/1e9:.2f} GB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    main()
