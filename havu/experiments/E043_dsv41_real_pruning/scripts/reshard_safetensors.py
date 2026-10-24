#!/usr/bin/env python3
"""E043: re-shard an export's safetensors so that no file is larger than --max-bytes (default 5e9; user rule
2026-10-24: HF uploads are capped at 5 GB per file).

  reshard_safetensors.py <export dir> [--max-bytes 5e9] [--dry-run]

Byte-level and dtype-agnostic (pure Python, no torch): tensors are copied as raw byte ranges. Tensors larger than the
cap are split along dim 0 into `<name>.rows_<start>` chunks. Only names matching --split may be split (default: the
Engram tables `layers.N.engram.embed.{weight,scale}`); serving them needs patches/apply_compact_engram.py, whose
E043C row-chunk loader copies each chunk into the rank's rows. File groups keep their prefix: tensors from
model-engram-* files go to model-engram-XXXXX-of-YYYYY.safetensors, the rest to model-XXXXX-of-YYYYY.safetensors.
Within a file, tensors are ordered by dtype size (largest first) so every tensor stays aligned.

The new files are written to <dir>/.reshard/. Every source tensor's sha256 is compared with its bytes re-read from
the new files (all chunks in row order) before the indexed old files are unlinked (a hardlink shared with another
export stays there) and model.safetensors.index.json is replaced. Files outside the index (gptq/, configs) are
untouched.
"""
import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

ITEM = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
        "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1}
RESERVE = 1 << 20  # header bytes kept free under the cap
BLOCK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def group_of(fname):
    return re.sub(r"-\d{5}(-of-\d{5})?\.safetensors$", "", fname)


def copy_range(path, start, end, dst=None, h=None):
    with open(path, "rb") as src:
        src.seek(start)
        left = end - start
        while left:
            buf = src.read(min(BLOCK, left))
            if not buf:
                raise IOError(f"short read in {path}")
            if dst is not None:
                dst.write(buf)
            if h is not None:
                h.update(buf)
            left -= len(buf)


def plan(d, index, max_bytes, split):
    """Source tensors (group, file, byte start, byte end, name, dtype, shape) and, per group, the new files' pieces
    (new name, dtype, shape, source file, byte start, byte end, source name, start row)."""
    files = sorted(set(index["weight_map"].values()))
    tensors = []
    for fn in files:
        base, h = read_header(d / fn)
        for name, v in h.items():
            if name != "__metadata__":
                s, e = v["data_offsets"]
                tensors.append((group_of(fn), fn, base + s, base + e, name, v["dtype"], v["shape"]))
    missing = set(index["weight_map"]) - {t[4] for t in tensors}
    if missing:
        raise SystemExit(f"index names missing from the files: {sorted(missing)[:5]}")
    tensors.sort(key=lambda t: (t[0] != "model", t[0], files.index(t[1]), t[2]))
    cap = max_bytes - RESERVE
    groups = {}
    for grp, fn, s, e, name, dtype, shape in tensors:
        if e - s <= cap:
            pieces = [(name, dtype, shape, fn, s, e, name, 0)]
        else:
            if not split.match(name):
                raise SystemExit(f"{name} is {(e - s) / 1e9:.2f} GB, above the cap, and may not be split (--split)")
            row = (e - s) // shape[0]
            assert row * shape[0] == e - s
            rows = cap // row
            pieces = [(f"{name}.rows_{r}", dtype, [min(rows, shape[0] - r)] + shape[1:], fn, s + r * row,
                       s + min(r + rows, shape[0]) * row, name, r) for r in range(0, shape[0], rows)]
        out = groups.setdefault(grp, [[]])
        for p in pieces:
            if out[-1] and sum(q[5] - q[4] for q in out[-1]) + p[5] - p[4] > cap:
                out.append([])
            out[-1].append(p)
    return tensors, groups


def write_file(path, pieces, d):
    pieces = sorted(pieces, key=lambda p: (-ITEM[p[1]], p[0]))
    header, off = {"__metadata__": {"format": "pt"}}, 0
    for name, dtype, shape, _, s, e, _, _ in pieces:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + e - s]}
        off += e - s
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-(8 + len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for _, _, _, fn, s, e, _, _ in pieces:
            copy_range(d / fn, s, e, dst=f)
    return {p[0]: path.name for p in pieces}


def check(d, new_dir, weight_map, tensors):
    """Every source tensor re-read from the new files (chunks in row order) has the source's sha256."""
    headers = {fn: read_header(new_dir / fn) for fn in set(weight_map.values())}
    source = {t[4] for t in tensors}
    chunks = {}
    for name, fn in weight_map.items():
        m = re.match(r"^(.*)\.rows_(\d+)$", name)
        src, r = (m.group(1), int(m.group(2))) if m and name not in source else (name, 0)
        chunks.setdefault(src, []).append((r, name, fn))
    for grp, fn, s, e, name, dtype, shape in tensors:
        want, got, rows, n = hashlib.sha256(), hashlib.sha256(), 0, 0
        copy_range(d / fn, s, e, h=want)
        for r, cname, cfn in sorted(chunks[name]):
            base, hd = headers[cfn]
            v = hd[cname]
            assert v["dtype"] == dtype and v["shape"][1:] == shape[1:] and r == rows, cname
            a, b = v["data_offsets"]
            copy_range(new_dir / cfn, base + a, base + b, h=got)
            rows += v["shape"][0]
            n += b - a
        assert rows == shape[0] and n == e - s, name
        if got.digest() != want.digest():
            raise SystemExit(f"sha256 mismatch for {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--max-bytes", type=float, default=5e9)
    ap.add_argument("--split", default=r"^layers\.\d+\.engram\.embed\.(weight|scale)$")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d, max_bytes = Path(a.dir), int(a.max_bytes)
    index = json.loads((d / "model.safetensors.index.json").read_text())
    tensors, groups = plan(d, index, max_bytes, re.compile(a.split))
    names = {(grp, i): f"{grp}-{i + 1:05d}-of-{len(shards):05d}.safetensors"
             for grp, shards in groups.items() for i in range(len(shards))}
    for (grp, i), fn in names.items():
        pieces = groups[grp][i]
        chunks = sum(p[0] != p[6] for p in pieces)
        print(f"{fn}: {len(pieces)} tensors ({chunks} row chunks), {sum(p[5] - p[4] for p in pieces) / 1e9:.3f} GB")
    old = sorted(set(index["weight_map"].values()))
    print(f"{len(old)} files -> {len(names)} files, cap {max_bytes / 1e9:g} GB", flush=True)
    if a.dry_run:
        return
    new_dir = d / ".reshard"
    new_dir.mkdir(exist_ok=True)
    weight_map = {}
    for (grp, i), fn in names.items():
        weight_map.update(write_file(new_dir / fn, groups[grp][i], d))
        size = (new_dir / fn).stat().st_size
        assert size <= max_bytes, f"{fn} is {size} bytes"
    check(d, new_dir, weight_map, tensors)
    for fn in old:
        os.unlink(d / fn)
    for fn in names.values():
        os.rename(new_dir / fn, d / fn)
    new_dir.rmdir()
    meta = dict(index.get("metadata", {}))
    meta["total_size"] = sum(t[3] - t[2] for t in tensors)
    meta["e043_reshard"] = {"max_bytes": max_bytes, "row_chunk_suffix": ".rows_<start>"}
    tmp = d / "model.safetensors.index.json.tmp"
    tmp.write_text(json.dumps({"metadata": meta, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    os.replace(tmp, d / "model.safetensors.index.json")
    largest = max((d / fn).stat().st_size for fn in names.values())
    print(f"done: {len(tensors)} tensors verified (sha256), largest file {largest / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
