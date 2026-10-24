#!/usr/bin/env python3
"""Stage the model checkpoint for the vLLM load, in the background, while the notebook installs vLLM.

Usage: stage_model.py <src dir> <dst dir> <marker file> <mode: copy | warm | none> [threads]

- copy: copy every file (symlinks resolved) to <dst> with N threads, largest first, then check sizes. Kaggle's
  /kaggle/input is a lazily fetched mount, and 125-131 GB of weights cannot stay in the page cache next to the server's
  ~100 GiB of pinned Engram and processes (173 GiB cgroup), so a page-cache warm-up alone would be re-read slowly at
  load. The local disk (/tmp, ~1 TB free) keeps the copy. Falls back to `warm` if the disk lacks space.
- warm: read every file once (64 MiB chunks) with N threads, to fetch the mount's blocks before the load.
- none: nothing.
The marker gets one line: "ok <dir to load from>" or "fail <reason>" (the solver then loads from <src>).
"""
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CHUNK = 64 << 20


def files(src):
    out = []
    for root, _, names in os.walk(src, followlinks=True):
        if "/.cache" in root or "/." in root[len(src):]:
            continue
        for n in names:
            p = os.path.join(root, n)
            out.append((os.path.relpath(p, src), os.path.getsize(p)))
    return sorted(out, key=lambda x: -x[1])


def copy_one(src, dst, rel):
    os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
    tmp = os.path.join(dst, rel) + ".partial"
    with open(os.path.join(src, rel), "rb") as fi, open(tmp, "wb") as fo:
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
    os.replace(tmp, os.path.join(dst, rel))


def warm_one(src, _dst, rel):
    with open(os.path.join(src, rel), "rb") as fi:
        while fi.read(CHUNK):
            pass


def main():
    src, dst, marker, mode = sys.argv[1:5]
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    t0 = time.time()
    try:
        if mode == "none":
            open(marker, "w").write(f"ok {src}\n")
            return
        fl = files(src)
        total = sum(s for _, s in fl)
        if mode == "copy":
            os.makedirs(dst, exist_ok=True)
            free = shutil.disk_usage(dst).free
            if free < total + (10 << 30):
                print(f"copy: {total / 2**30:.1f} GiB needed, {free / 2**30:.1f} GiB free; warming instead", flush=True)
                mode = "warm"
        fn = copy_one if mode == "copy" else warm_one
        with ThreadPoolExecutor(threads) as ex:
            for i, _ in enumerate(ex.map(lambda x: fn(src, dst, x[0]), fl), 1):
                if i % 10 == 0 or i == len(fl):
                    el = time.time() - t0
                    print(f"{mode}: {i}/{len(fl)} files, {el:.0f}s", flush=True)
        if mode == "copy":
            bad = [r for r, s in fl if os.path.getsize(os.path.join(dst, r)) != s]
            if bad:
                raise RuntimeError(f"size mismatch: {bad[:3]}")
        el = time.time() - t0
        print(f"{mode} done: {total / 2**30:.1f} GiB in {el:.0f}s ({total / 2**20 / max(el, 1):.0f} MiB/s)", flush=True)
        open(marker, "w").write(f"ok {dst if mode == 'copy' else src}\n")
    except Exception as exc:
        open(marker, "w").write(f"fail {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
