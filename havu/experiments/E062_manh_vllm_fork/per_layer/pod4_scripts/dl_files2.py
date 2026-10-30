"""Like E064b's dl_kaggle_files.py (sequential, version-pinned, skip files already present at the listed size), but a
404 also tries the URL-decoded name (e.g. '%3a' -> ':'), with short backoff. The file lands under the listed name.
Usage: dl_files2.py <handle incl. version> <files.tsv: path<TAB>size> <dest>"""
import os, shutil, sys, time, urllib.parse
import kagglehub
handle, listing, dest = sys.argv[1:4]
os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join(dest, ".khcache"))
files = [l.rstrip("\n").split("\t") for l in open(listing) if l.strip()]
ok = failed = 0
for path, size in files:
    out = os.path.join(dest, path)
    if os.path.exists(out) and os.path.getsize(out) == int(size):
        ok += 1; continue
    got, err = None, None
    for cand in dict.fromkeys([path, urllib.parse.unquote(path)]):
        for attempt in range(3):
            try:
                got = kagglehub.dataset_download(handle, path=cand); break
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[:120]}"
                if "404" in err: break
                time.sleep(15 * (attempt + 1))
        if got: break
    if got and os.path.getsize(got) == int(size):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True); shutil.move(got, out); ok += 1
        print("GOT", path, flush=True)
    else:
        failed += 1; print("FAILED", path, err, flush=True)
    time.sleep(1)
print(f"DL_DONE ok={ok} failed={failed}", flush=True)
