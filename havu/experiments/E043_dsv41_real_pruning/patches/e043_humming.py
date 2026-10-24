"""E043 low-bit round: DeepSeek-V4.1 routed experts through Humming with the *real* 1017 weights.

Installed into the vLLM package by apply_e043_humming.py; active only when DSV41_EXPERT_HUMMING is set.

- dequant_expert_weights(weights): wraps the checkpoint iterator. Each routed-expert matrix arrives as MXFP4
  (`...experts.E.w{1,2,3}.weight` int8 = two E2M1 codes per byte, low nibble first, and `.weight_scale` E8M0 per 32
  along K). The pair is decoded on the GPU into the exact BF16 values (an E2M1 value times a power of two is exact in
  BF16) and yielded as one `.weight`, which Humming's loader then quantizes or replaces with precomputed tensors.
  Router, attention, shared expert, Engram and every non-expert tensor pass through untouched. GPTQ-only checkpoints
  (routed MXFP4 experts dropped, see make_lowbit_export.py) work too: experts covered by E043_GPTQ_DIR but absent from
  the checkpoint get a zero-allocation placeholder, and Humming loads the precomputed tensors.
- precomputed(layer_name, expert_id, shard_id): with E043_GPTQ_DIR set, returns this expert matrix's GPTQ result in
  the exact tuple humming.utils.weight.quantize_weight(..., pack=True) returns
  (packed int32 codes, scale, packed int32 zero point, None), read from `layer_LL.safetensors` (keys
  `e{E}.{w1|w2|w3}.{q,s,z}`). Humming then uses it instead of rounding itself, so there is no second rounding.
"""
import os
import re

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_EXPERT = re.compile(r"^(.*\.experts\.\d+\.w[123])\.(weight|weight_scale)$")
HIDDEN, INTER = 5120, 2304


def mxfp4_to_bf16(weight, scale, device):
    """weight: uint8/int8 [N, K/2] (low nibble = even k), scale: E8M0 [N, K/32] -> exact BF16 [N, K]."""
    w = weight.to(device).view(torch.uint8)
    codes = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(*w.shape[:-1], w.shape[-1] * 2).long()
    mag = E2M1.to(device)[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    s = scale.to(device).view(torch.uint8).to(torch.int32)
    return (val * torch.exp2((s - 127).float()).repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def dequant_expert_weights(weights):
    pending, seen, prefix = {}, set(), None
    device = torch.device("cuda", torch.cuda.current_device())
    for name, tensor in weights:
        if prefix is None:
            m0 = re.match(r"^(.*?)layers\.\d+\.", name)
            if m0:
                prefix = m0.group(1)
        m = _EXPERT.match(name)
        if not m:
            yield name, tensor
            continue
        key, kind = m.group(1), m.group(2)
        d = pending.setdefault(key, {})
        d[kind] = tensor
        if len(d) == 2:
            del pending[key]
            seen.add(key)
            yield key + ".weight", mxfp4_to_bf16(d["weight"], d["weight_scale"], device)
    if pending:
        raise RuntimeError(f"E043 humming: unpaired expert tensors, e.g. {sorted(pending)[:4]}")
    # GPTQ-only checkpoints (no routed MXFP4 tensors): every expert matrix covered by E043_GPTQ_DIR gets a
    # zero-allocation BF16 placeholder (stride-0 view) so Humming's loader is called and takes the precomputed tensors.
    root = os.environ.get("E043_GPTQ_DIR")
    if root:
        import glob
        from safetensors import safe_open
        for path in sorted(glob.glob(os.path.join(root, "layer_*.safetensors"))):
            li = int(os.path.basename(path)[6:8])
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = [k for k in f.keys() if k.endswith(".q")]
            for k in keys:
                e, shard = k[1:].split(".")[:2]
                key = f"{prefix or ''}layers.{li}.ffn.experts.{e}.{shard}"
                if key in seen:
                    continue
                n, kdim = (HIDDEN, INTER) if shard == "w2" else (INTER, HIDDEN)
                yield key + ".weight", torch.empty((1, 1), dtype=torch.bfloat16).expand(n, kdim)


_cache = {"layer": None, "file": None}


def precomputed(layer_name, expert_id, shard_id):
    root = os.environ.get("E043_GPTQ_DIR")
    if not root or expert_id is None:
        return None
    li = int(layer_name.split("layers.")[1].split(".")[0])
    if _cache["layer"] != li:
        from safetensors import safe_open
        path = os.path.join(root, f"layer_{li:02d}.safetensors")
        _cache["file"] = safe_open(path, framework="pt", device="cpu") if os.path.exists(path) else None
        _cache["layer"] = li
    f = _cache["file"]
    if f is None:
        return None
    k = f"e{int(expert_id)}.{shard_id}"
    if f"{k}.q" not in f.keys():
        return None
    z = f.get_tensor(f"{k}.z") if f"{k}.z" in f.keys() else None
    return f.get_tensor(f"{k}.q"), f.get_tensor(f"{k}.s"), z, None
