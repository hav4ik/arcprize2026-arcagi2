#!/usr/bin/env python3
"""E043: manifest of a model folder: relative path, size and sha256 of every file (symlinks followed), as JSON.

  file_manifest.py <dir> <out.json> [--workers 16]

Skips the kaggle metadata JSON and dot-dirs. Files are hashed in parallel processes.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SKIP = {"model-instance-metadata.json", "model-metadata.json"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = Path(a.dir)
    files = []
    for d, dirs, names in os.walk(root, followlinks=True):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x != "__pycache__"]  # the kaggle CLI skips both
        files += [Path(d) / n for n in names if str((Path(d) / n).relative_to(root)) not in SKIP]
    files.sort(key=lambda p: -p.stat().st_size)
    with ProcessPoolExecutor(a.workers) as pool:
        digests = list(pool.map(sha, files))
    rows = sorted(({"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": h}
                   for p, h in zip(files, digests)), key=lambda r: r["path"])
    json.dump({"root": str(root), "files": len(rows), "total_bytes": sum(r["size"] for r in rows), "entries": rows},
              open(a.out, "w"), indent=1)
    print(f"{len(rows)} files, {sum(r['size'] for r in rows) / 1e9:.2f} GB -> {a.out}")


if __name__ == "__main__":
    main()
