"""Download an HF model repo at a pinned revision to a local dir (resumable snapshot_download), then check every LFS
file's size and sha256 against the Hub's metadata. The token is read from stdin (first line), never an argument.
usage: hf_snap.py REPO REV LOCAL_DIR [WORKERS] [--allow PATTERN ...]"""
import hashlib, os, sys, time
from concurrent.futures import ThreadPoolExecutor
os.environ["HF_TOKEN"] = sys.stdin.readline().strip()
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi, snapshot_download
repo, rev, out = sys.argv[1:4]
workers = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 8
allow = sys.argv[sys.argv.index("--allow") + 1:] if "--allow" in sys.argv else None
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
print("REPO_OK", repo, "rev", info.sha, "files", len(info.siblings), "bytes", sum(f.size or 0 for f in info.siblings), flush=True)
t0 = time.time()
snapshot_download(repo, revision=info.sha, local_dir=out, max_workers=workers, allow_patterns=allow)
print("DL_DONE in %.0fs" % (time.time() - t0), flush=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()
lfs = [s for s in info.siblings if s.lfs and os.path.exists(os.path.join(out, s.rfilename))]
bad = []
with ThreadPoolExecutor(12) as ex:
    for s, h in zip(lfs, ex.map(lambda s: sha(os.path.join(out, s.rfilename)), lfs)):
        if h != s.lfs.sha256 or os.path.getsize(os.path.join(out, s.rfilename)) != s.size:
            bad.append(s.rfilename)
print(f"LFS sha256 checked {len(lfs)} files; bad {bad}", flush=True)
print("VERIFY_OK" if not bad else "VERIFY_FAIL", flush=True)
