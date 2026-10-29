"""E064b (pod): download listed files of a Kaggle dataset or model version one by one with kagglehub, gently.

    KAGGLE_API_TOKEN=... python dl_kaggle_files.py dataset|model <handle incl. version> <files.tsv: path<TAB>size> <dest>

Sequential (Kaggle's API answered 429 to 12 parallel downloads, 2026-10-28 ~20:05 UTC): one file at a time, 2 s apart,
backoff 60 s x attempt on errors (6 attempts). Pin the version in the handle (owner/slug/versions/N, or the model's
.../<version>) so every file comes from the same version. Each file lands at <dest>/<path>; a file already there with
the listed size is skipped. Every downloaded path is logged with the version kagglehub resolved ("GOT <path> v<N>").
Ends with "DL_DONE ok=<n> failed=<n>".
"""
import os
import re
import shutil
import sys
import time

import kagglehub

kind, handle, listing, dest = sys.argv[1:5]
files = [l.rstrip("\n").split("\t") for l in open(listing) if l.strip()]
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(dest, ".khcache"))
fn = kagglehub.dataset_download if kind == "dataset" else kagglehub.model_download
t0, ok, failed = time.time(), 0, 0
for path, size in files:
    out = os.path.join(dest, path)
    if os.path.exists(out) and os.path.getsize(out) == int(size):
        ok += 1
        continue
    err = None
    for attempt in range(6):
        try:
            got = fn(handle, path=path)
            if os.path.getsize(got) != int(size):
                raise IOError(f"size {os.path.getsize(got)} != {size}")
            ver = (re.search(r"/versions/(\d+)/", got) or re.search(r"/(\d+)/", got))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.move(got, out)
            print("GOT", path, "v" + (ver.group(1) if ver else "?"), flush=True)
            err = None
            break
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print("RETRY", path, attempt, err[:160], flush=True)
            time.sleep(60 * (attempt + 1))
    if err:
        failed += 1
        print(path, "FAILED", err[:200], flush=True)
    else:
        ok += 1
    time.sleep(2)
print(f"DL_DONE ok={ok} failed={failed} in {time.time() - t0:.0f}s", flush=True)
