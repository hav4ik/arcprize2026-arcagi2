"""Full sweep of the FP8block checkpoint vs official metadata, merged BF16 and base BF16."""
import json
import re
import collections
import torch
from safetensors import safe_open

torch.set_num_threads(16)
B = 128
BASE = "/workspace/models/Qwen3.8-27B"
SFT = "/workspace/models/Qwen3.8-27B-e028-sft"
OURS = "/workspace/models/Qwen3.8-27B-e028-sft-FP8block"
off = json.load(open("/workspace/e028/official_fp8_meta/tensors.json"))
DT = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}


class Reader:
    def __init__(self, root):
        self.root = root
        self.wm = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        self.h = {}

    def get(self, n):
        s = self.wm[n]
        if s not in self.h:
            self.h[s] = safe_open(f"{self.root}/{s}", "pt")
        return self.h[s].get_tensor(n)


def expand(s, r, c):
    return s.float().repeat_interleave(B, 0).repeat_interleave(B, 1)[:r, :c]


ours, sft, base = Reader(OURS), Reader(SFT), Reader(BASE)
print("tensor sets: ours==official", set(ours.wm) == set(off), "sft keys==official non-scale", set(sft.wm) == {k for k in off if not k.endswith('_scale_inv')})
print("base keys == sft keys", set(base.wm) == set(sft.wm))
problems = []
worst = []
grp = collections.defaultdict(list)
changed = collections.Counter()
cos_delta = collections.defaultdict(list)
for n in sorted(off):
    if n.endswith("_scale_inv"):
        continue
    odt, oshape, _ = off[n]
    t = ours.get(n)
    if t.dtype != DT[odt] or list(t.shape) != oshape:
        problems.append((n, "dtype/shape", t.dtype, t.shape, odt, oshape))
        continue
    s_t = sft.get(n)
    b_t = base.get(n)
    p = re.sub(r"\.\d+\.", ".N.", n)
    if not torch.equal(s_t, b_t):
        changed[p] += 1
    if odt == "F8_E4M3":
        s = ours.get(n + "_scale_inv")
        if not (torch.isfinite(s.float()).all() and (s.float() > 0).all()):
            problems.append((n, "bad scale", s.float().min().item()))
        qf = t.float()
        if torch.isnan(qf).any():
            problems.append((n, "nan codes"))
        r, c = t.shape
        deq = qf * expand(s, r, c)
        sf = s_t.float()
        rel = ((deq - sf).norm() / sf.norm()).item()
        grp[p].append(rel)
        worst.append((rel, n))
        # does the SFT delta survive quantization? compare deq(ours) - deq(quant(base)) against (sft - base)
        if p.startswith("model.language_model.layers.N.") and int(n.split(".")[3]) in (0, 3, 31, 63) or p.startswith("mtp"):
            bf = b_t.float()
            wb = bf.reshape(r // B, B, c // B, B)
            sb = (wb.abs().amax(dim=(1, 3)) / 448.0).to(torch.bfloat16)
            qb = (bf / expand(sb, r, c)).clamp(-448, 448).to(torch.float8_e4m3fn).float() * expand(sb, r, c)
            d_true = (sf - bf).flatten()
            d_q = (deq - qb).flatten()
            if d_true.norm() > 0:
                cos = torch.nn.functional.cosine_similarity(d_true, d_q, dim=0).item()
                cos_delta[p].append((n.split(".")[3] if "layers" in n else "mtp", round(cos, 3), round((d_q.norm() / d_true.norm()).item(), 2)))
    else:
        if not torch.equal(t, s_t.to(t.dtype)):
            problems.append((n, "non-quantized tensor differs from merged"))
for p, v in sorted(grp.items()):
    print(f"{p:60s} n={len(v):3d} rel err mean={sum(v)/len(v):.4f} max={max(v):.4f}")
print("worst:", sorted(worst)[-3:])
print("tensors changed by SFT (sft != base):")
for p, c in sorted(changed.items()):
    print(f"  {c:3d} {p}")
print("SFT-delta survival (cos(dq_delta,true_delta), |dq_delta|/|true_delta|):")
for p, v in sorted(cos_delta.items()):
    print(" ", p, v)
print("problems:", problems[:20], len(problems))
