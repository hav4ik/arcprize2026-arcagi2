"""Download one HF model repo to a local dir with huggingface_hub's resumable snapshot_download. The token comes only
from the HF_TOKEN environment variable (never an argument). Usage: hf_download.py REPO LOCAL_DIR [WORKERS]"""
import os, sys, time
from huggingface_hub import HfApi, snapshot_download
repo, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
info = HfApi().model_info(repo, files_metadata=True)
print("REPO_OK", repo, "rev", info.sha, "files", len(info.siblings), "bytes", sum(f.size or 0 for f in info.siblings), flush=True)
t0 = time.time()
snapshot_download(repo, revision=info.sha, local_dir=out, max_workers=workers)
print("DL_DONE in %.0fs" % (time.time() - t0), flush=True)
