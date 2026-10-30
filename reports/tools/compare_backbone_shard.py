"""Compare a backbone safetensors shard (FP8 block-32 backbone) between base and SFT export.

Pure numpy reader; treats every tensor as raw bytes (dtype-agnostic) and additionally decodes
FP8 e4m3 shared-expert weights to count changed codes.
"""
import json
import struct
import sys
from collections import defaultdict

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def read_raw(path, base, info):
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        return np.frombuffer(f.read(e - s), dtype=np.uint8)


def main(base_path, exp_path):
    hb, ob = read_header(base_path)
    he, oe = read_header(exp_path)
    kb = {k for k in hb if k != "__metadata__"}
    ke = {k for k in he if k != "__metadata__"}
    print(f"base keys={len(kb)} export keys={len(ke)} only_base={sorted(kb-ke)[:10]} only_export={sorted(ke-kb)[:10]}")
    common = sorted(kb & ke)
    stats = defaultdict(lambda: [0, 0, 0])  # group -> [n_tensors, n_changed_tensors, changed_bytes]
    details = []
    for k in common:
        ib, ie = hb[k], he[k]
        if ib["dtype"] != ie["dtype"] or ib["shape"] != ie["shape"]:
            print(f"  DTYPE/SHAPE MISMATCH {k}: base {ib['dtype']}{ib['shape']} export {ie['dtype']}{ie['shape']}")
            continue
        rb = read_raw(base_path, ob, ib)
        re_ = read_raw(exp_path, oe, ie)
        if rb.size != re_.size:
            print(f"  SIZE MISMATCH {k}")
            continue
        diff = int((rb != re_).sum())
        grp = "shared_experts" if ".shared_experts." in k else ("engram" if "engram" in k else ("attn/other" ))
        if ".shared_experts." in k and k.endswith(".scale"):
            grp = "shared_experts.scale"
        st = stats[grp]
        st[0] += 1
        st[1] += int(diff > 0)
        st[2] += diff
        if ".shared_experts." in k:
            frac = diff / rb.size
            if ib["dtype"] == "F8_E4M3":
                details.append((k, ib["dtype"], ib["shape"], rb.size, diff, frac))
            else:
                details.append((k, ib["dtype"], ib["shape"], rb.size, diff, frac))
    print("\n=== per-group summary (bytes changed) ===")
    for g, (n, nc, cb) in sorted(stats.items()):
        print(f"  {g:22s}: tensors={n:4d} changed_tensors={nc:4d} changed_bytes={cb}")
    print("\n=== shared-expert tensors (fraction of bytes/codes changed) ===")
    fr_w = [d[5] for d in details if d[1] == "F8_E4M3"]
    fr_s = [d[5] for d in details if d[1] != "F8_E4M3"]
    if fr_w:
        print(f"  FP8 weight codes changed: mean={100*np.mean(fr_w):.4f}% min={100*min(fr_w):.4f}% max={100*max(fr_w):.4f}% over {len(fr_w)} tensors")
    if fr_s:
        print(f"  scale bytes changed:      mean={100*np.mean(fr_s):.4f}% max={100*max(fr_s):.4f}% over {len(fr_s)} tensors")
    for d in details[:12]:
        print(f"    {d[0]} {d[1]} {d[2]} changed {d[4]}/{d[3]} ({100*d[5]:.4f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
