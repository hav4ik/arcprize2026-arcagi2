#!/usr/bin/env python3
"""E068: compare two safetensors files tensor by tensor (sha256 of each tensor's bytes), ignoring the header.
Usage: cmp_tensors.py A B -> prints '<n_same> same, <n_diff> differ, <n_only> only in one' (+ first differing names)"""
import hashlib, json, struct, sys

def tensors(p):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None)
    out = {}
    for k, v in h.items():
        a, b = v["data_offsets"]; f.seek(8 + n + a); out[k] = (v["dtype"], tuple(v["shape"]), hashlib.sha256(f.read(b - a)).digest())
    return out

A, B = tensors(sys.argv[1]), tensors(sys.argv[2])
same = [k for k in A if k in B and A[k] == B[k]]; diff = sorted(k for k in A if k in B and A[k] != B[k])
only = sorted(set(A) ^ set(B))
print(f"{sys.argv[2]}: {len(same)} same, {len(diff)} differ, {len(only)} only in one; differ e.g. {diff[:3]}; only e.g. {only[:3]}")
