#!/usr/bin/env python3
"""E068: compare two checkpoint dirs tensor by tensor via their model.safetensors.index.json (sha256 of tensor bytes),
skipping Engram table tensors and gptq/. Usage: cmp_models.py DIR_A DIR_B"""
import hashlib, json, re, struct, sys
from collections import Counter
from pathlib import Path

SKIP = re.compile(r"\.engram\.(embed\.|embed_tokens\.)")

def hdr(p, cache={}):
    if p not in cache:
        f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
        cache[p] = (n, h)
    return cache[p]

def digest(d, wm, k):
    p = str(Path(d) / wm[k]); n, h = hdr(p); v = h[k]; a, b = v["data_offsets"]
    with open(p, "rb") as f:
        f.seek(8 + n + a); return v["dtype"], tuple(v["shape"]), hashlib.sha256(f.read(b - a)).hexdigest()

A, B = (json.load(open(Path(d) / "model.safetensors.index.json"))["weight_map"] for d in sys.argv[1:3])
keys = sorted(k for k in set(A) & set(B) if not SKIP.search(k))
diff = []
for k in keys:
    if digest(sys.argv[1], A, k) != digest(sys.argv[2], B, k):
        diff.append(k)
groups = Counter(re.sub(r"layers\.\d+\.", "layers.N.", k) for k in diff)
print(f"{len(keys)} tensors compared, {len(diff)} differ; only in one: {len(set(A) ^ set(B))}")
for g, c in groups.most_common(40):
    print(f"  {c:3d} {g}")
