#!/usr/bin/env python3
"""E043: check that a resharded export holds exactly the tensors of the original, byte for byte.

  verify_reshard_equiv.py <original export dir> <new dir> [--max-bytes 5e9]

For every tensor of the original index: sha256 of its bytes vs the bytes in the new dir, either the same name or its
`<name>.rows_<start>` chunks concatenated in row order (dtype and trailing shape equal, rows summing to the original).
The new index must hold no other tensors, every safetensors file under the new dir (gptq/ included, symlinks
followed) must be at most --max-bytes, and files outside the index (gptq/, configs, tokenizer) must be byte-identical
to the original's where both exist. Pure Python; reads both copies once.
"""
import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

BLOCK = 64 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def digest(path, start, end, h=None):
    h = h or hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start
        while left:
            b = f.read(min(BLOCK, left))
            h.update(b)
            left -= len(b)
    return h


def tensors(d):
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for fn in sorted(set(wm.values())):
        base, h = header(d / fn)
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (d / fn, base + v["data_offsets"][0], base + v["data_offsets"][1], v["dtype"], v["shape"])
    missing = set(wm) - set(out)
    assert not missing, f"{d}: index names missing from files: {sorted(missing)[:3]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig")
    ap.add_argument("new")
    ap.add_argument("--max-bytes", type=float, default=5e9)
    ap.add_argument("--allow-diff", action="append", default=[], help="non-index files expected to differ (README.md)")
    ap.add_argument("--files-only", action="store_true", help="skip the tensor stage (sizes and non-index files only)")
    a = ap.parse_args()
    o, n = Path(a.orig), Path(a.new)
    to, tn = tensors(o), tensors(n)
    if a.files_only:
        to = {}
    chunks = {}
    for k in tn:
        m = re.match(r"^(.*)\.rows_(\d+)$", k)
        if m and k not in to:
            chunks.setdefault(m.group(1), []).append((int(m.group(2)), k))
    used = set()
    for name, (p, s, e, dtype, shape) in to.items():
        want = digest(p, s, e).hexdigest()
        if name in tn:
            q = tn[name]
            assert (q[3], q[4]) == (dtype, shape), name
            got = digest(q[0], q[1], q[2]).hexdigest()
            used.add(name)
        else:
            parts = sorted(chunks.get(name, []))
            assert parts, f"{name} missing in the new dir"
            h, rows = hashlib.sha256(), 0
            for r, k in parts:
                q = tn[k]
                assert q[3] == dtype and q[4][1:] == shape[1:] and r == rows, k
                digest(q[0], q[1], q[2], h)
                rows += q[4][0]
                used.add(k)
            assert rows == shape[0], (name, rows, shape[0])
            got = h.hexdigest()
        assert got == want, f"sha256 mismatch: {name}"
    extra = set(tn) - used
    assert a.files_only or not extra, f"tensors only in the new dir: {sorted(extra)[:3]}"
    big, same, nfiles = [], 0, 0
    for root, _, files in os.walk(n, followlinks=True):
        for f in files:
            p = Path(root) / f
            rel = p.relative_to(n)
            if f.endswith(".safetensors"):
                nfiles += 1
                if p.stat().st_size > a.max_bytes:
                    big.append(str(rel))
            if not f.startswith("model") and str(rel) not in a.allow_diff and (o / rel).is_file():
                assert digest(p, 0, p.stat().st_size).digest() == digest(o / rel, 0, (o / rel).stat().st_size).digest(), rel
                same += 1
    assert not big, f"files above {a.max_bytes:g} bytes: {big}"
    print(f"OK{' (files only)' if a.files_only else ''}: {len(to)} original tensors byte-identical in the new dir ({len(tn)} tensors there, "
          f"{sum(len(v) for v in chunks.values())} row chunks); {nfiles} safetensors files <= {a.max_bytes / 1e9:g} GB; "
          f"{same} non-index files identical to the original")


if __name__ == "__main__":
    main()
