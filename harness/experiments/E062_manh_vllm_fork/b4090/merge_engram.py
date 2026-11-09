#!/usr/bin/env python3
"""E062: an HF-layout view of the Kaggle E128 copy for loaders that do not know row-chunked Engram tables.

The Kaggle copy stores each Engram table as `<base>.rows_<start>` chunks (<= 5 GB files; our B32 overlay's loader
reassembles them). Manh's fork loads only whole tensors, as in the HF export. This writes, into <dst>:
  - symlinks to every file and folder of <src> except the safetensors index and the row-chunk files,
  - one `model-engram-merged-<k>.safetensors` per chunked tensor (chunks concatenated in start order; contiguity and
    the total row count checked),
  - a new model.safetensors.index.json (chunk names replaced by the merged names).
Exact bytes: the chunks are row slices of the original tensors (E043 verify_reshard_equiv.py sha256-matched them
against the HF export), so concatenation restores those tensors.

    python merge_engram.py /root/dsv41/model-e128k /root/dsv41/model-e128m
"""
import collections
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROWS = re.compile(r"^(.*)\.rows_(\d+)$")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    src = os.path.realpath(src)
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = index["weight_map"]
    groups = collections.defaultdict(list)
    for name, f in wm.items():
        m = ROWS.match(name)
        if m:
            groups[m.group(1)].append((int(m.group(2)), name, f))
    chunk_files = {f for g in groups.values() for _, _, f in g}
    # files holding anything besides chunks stay as they are
    mixed = {f for name, f in wm.items() if f in chunk_files and not ROWS.match(name)}
    for e in os.listdir(src):
        if e == "model.safetensors.index.json" or (e in chunk_files and e not in mixed):
            continue
        p = os.path.join(dst, e)
        if not os.path.lexists(p):
            os.symlink(os.path.join(src, e), p)
    new_wm = {n: f for n, f in wm.items() if not ROWS.match(n)}
    for k, (base, parts) in enumerate(sorted(groups.items())):
        parts.sort()
        t0 = time.time()
        shapes, dtype = [], None
        for start, name, f in parts:
            with safe_open(os.path.join(src, f), framework="pt") as sf:
                sl = sf.get_slice(name)
                shapes.append(sl.get_shape())
                dtype = sl.get_dtype()
        rows = 0
        for (start, name, f), shp in zip(parts, shapes):
            assert start == rows, f"{base}: chunk {name} starts at {start}, expected {rows}"
            rows += shp[0]
            assert list(shp[1:]) == list(shapes[0][1:]), f"{base}: chunk shapes differ"
        out_name = f"model-engram-merged-{k + 1:02d}.safetensors"
        out_path = os.path.join(dst, out_name)
        if os.path.exists(out_path):
            print(f"{base}: {out_name} exists, kept", flush=True)
        else:
            full = None
            for (start, name, f), shp in zip(parts, shapes):
                with safe_open(os.path.join(src, f), framework="pt") as sf:
                    t = sf.get_tensor(name)
                if full is None:
                    full = torch.empty((rows, *t.shape[1:]), dtype=t.dtype)
                full[start:start + t.shape[0]].copy_(t)
                del t
            save_file({base: full}, out_path + ".tmp", metadata={"format": "pt"})
            os.replace(out_path + ".tmp", out_path)
            del full
        new_wm[base] = out_name
        print(f"{base}: {len(parts)} chunks, {rows} rows x {shapes[0][1:]} {dtype} -> {out_name} "
              f"({os.path.getsize(out_path) / 1e9:.2f} GB, {time.time() - t0:.0f}s)", flush=True)
    meta = dict(index.get("metadata", {}))
    json.dump({"metadata": meta, "weight_map": dict(sorted(new_wm.items()))},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    print(f"MERGE_OK: {len(groups)} tensors merged, {len(new_wm)} weight names", flush=True)


if __name__ == "__main__":
    main()
