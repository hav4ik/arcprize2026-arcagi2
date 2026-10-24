# check: Humming's own decoder reproduces our re-stored GGUF values, and they match the GGUF values closely
import sys, glob, torch, numpy as np
from safetensors import safe_open
import humming.utils.weight as hw
from humming import dtypes
from gguf import GGUFReader
from gguf.quants import dequantize
d, L, bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
keep = np.load(sys.argv[4]).astype(bool) if len(sys.argv) > 4 else None
e_orig = int(np.nonzero(keep[L])[0][5]) if keep is not None else 5
key = 5 if keep is not None else e_orig
T = {}
for f in sorted(glob.glob("/workspace/dsv4/data/unsloth/UD-IQ2_XXS/*.gguf")):
    for t in GGUFReader(f).tensors:
        if t.name.startswith(f"blk.{L}.ffn_") and "_exps" in t.name:
            T[t.name] = t
with safe_open(f"{d}/layer_{L:02d}.safetensors", "pt") as f:
    for shard, name in (("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps")):
        q, s, z = (f.get_tensor(f"e{key}.{shard}.{x}").cuda() for x in "qsz")
        deq = hw.dequantize_weight(q, s, z, None, getattr(dtypes, f"uint{bits}"), packed=True).float().cpu()
        t = T[f"blk.{L}.{name}.weight"]
        g = torch.from_numpy(dequantize(t.data[e_orig], t.tensor_type).astype(np.float32))
        print(f"L{L} expert {e_orig} -> e{key} {shard} {t.tensor_type.name}: shape {tuple(deq.shape)} rel MSE vs GGUF "
              f"{float((deq - g).pow(2).sum() / g.pow(2).sum()):.6f}, cos {float((deq * g).sum() / deq.norm() / g.norm()):.6f}")
