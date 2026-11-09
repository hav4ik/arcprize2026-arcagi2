#!/usr/bin/env python3
"""E066: path / size / sha256 manifest of an export folder (the entries format of E043
results/kaggle_e128_manifest.json), for dsv4-infer's mounted-file check. Hashes in parallel processes.

  sha_manifest.py <export dir> <out.json> [--procs 16]
"""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts
                   and "__pycache__" not in p.relative_to(root).parts)
    with Pool(a.procs) as pool:
        digests = pool.map(sha, [str(p) for p in files], chunksize=1)
    entries = [{"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": d} for p, d in zip(files, digests)]
    m = {"root": str(root), "files": len(entries), "total_bytes": sum(e["size"] for e in entries), "entries": entries}
    json.dump(m, open(a.out, "w"), indent=1)
    print(json.dumps({"out": a.out, "files": m["files"], "total_bytes": m["total_bytes"]}))


if __name__ == "__main__":
    main()
