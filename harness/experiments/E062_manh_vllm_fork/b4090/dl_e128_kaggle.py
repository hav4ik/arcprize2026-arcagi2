#!/usr/bin/env python3
"""E062: download the Kaggle copy of E128 (chankhavu/dsv41-flash-1017-arc-tool-e128-2bit, Version 1, 136 files, the
model the B32 notebook mounts) file by file with kagglehub, verify every file's size and sha256 against the E043
manifest, and expose it at <out> (symlink to the kagglehub cache dir).

The HF repo (chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit) is not readable with the workspace's
fine-grained HF token (404), so the box uses the Kaggle copy. KAGGLE_API_TOKEN comes from the caller's environment
(sent over SSH stdin, never written to disk). Log: stdout.

    KAGGLEHUB_CACHE=/root/dsv41/kh_cache python dl_e128_kaggle.py manifest.json /root/dsv41/model-e128k [threads]
"""
import concurrent.futures
import hashlib
import json
import os
import sys
import time

HANDLE = "chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pyTorch/default/1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    import kagglehub
    manifest, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    entries = json.load(open(manifest))["entries"]
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {len(entries)} files, {sum(e['size'] for e in entries) / 1e9:.1f} GB, "
          f"{threads} threads", flush=True)
    done = {}

    def one(e):
        for attempt in range(5):
            try:
                p = kagglehub.model_download(HANDLE, path=e["path"])
                if os.path.getsize(p) == e["size"]:
                    return e["path"], p
                print(f"size mismatch {e['path']}: {os.path.getsize(p)} vs {e['size']}; retry", flush=True)
                os.remove(p)
            except Exception as exc:
                print(f"{e['path']}: {type(exc).__name__}: {str(exc)[:200]}; retry {attempt + 1}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"failed: {e['path']}")

    big_first = sorted(entries, key=lambda e: -e["size"])
    with concurrent.futures.ThreadPoolExecutor(threads) as ex:
        for i, (rel, p) in enumerate(ex.map(one, big_first)):
            done[rel] = p
            if i % 10 == 0 or i == len(entries) - 1:
                got = sum(os.path.getsize(q) for q in done.values())
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(entries)} files, {got / 1e9:.1f} GB, "
                      f"{got / (time.time() - t0) / 1e6:.0f} MB/s", flush=True)
    # the version root: <cache>/.../<version>/ + rel
    rel0, p0 = next(iter(done.items()))
    root = p0[: -len(rel0)].rstrip("/")
    bad = []

    def check(e):
        p = os.path.join(root, e["path"])
        return e["path"], (os.path.getsize(p) == e["size"] and sha256(p) == e["sha256"])

    with concurrent.futures.ThreadPoolExecutor(16) as ex:
        for rel, ok in ex.map(check, entries):
            if not ok:
                bad.append(rel)
    if os.path.islink(out):
        os.remove(out)
    if not os.path.exists(out):
        os.symlink(root, out)
    print(f"[{time.strftime('%H:%M:%S')}] root {root} -> {out}; sha256 checked {len(entries)}, bad {len(bad)} {bad[:5]}",
          flush=True)
    print("DL_OK" if not bad else "DL_BAD", flush=True)


if __name__ == "__main__":
    main()
