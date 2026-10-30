#!/bin/bash
# E066 box 4 (Vast 58284661): vLLM fork env (setup_env.sh), candidate 1's export from HF (needs $HF_TOKEN), sha256 check
# against the box 1 manifest, then candidate 1's eval server (serve_eval.sh c1x).
set -u
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
(bash $S/setup_env.sh > logs/setup_env.log 2>&1; tail -1 logs/setup_env.log) &
pip install -q huggingface_hub hf_transfer > /dev/null 2>&1 || python3 -m pip install -q huggingface_hub > /dev/null 2>&1
HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit", local_dir="/workspace/dsv4/export/c1_gptq",
                  token=os.environ["HF_TOKEN"], max_workers=16)
print("DOWNLOAD_DONE")
PY
wait
grep -q SETUP_DONE logs/setup_env.log || { echo SETUP_FAILED; tail -20 logs/setup_env.log; exit 1; }
venv/bin/python - <<'PY'
import hashlib, json, os
from concurrent.futures import ProcessPoolExecutor
m = json.load(open("results/c1_gptq_sha256_manifest.json"))
root = "/workspace/dsv4/export/c1_gptq"
def h(e):
    p = os.path.join(root, e["path"]); x = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): x.update(b)
    return e["path"], os.path.getsize(p) == e["size"] and x.hexdigest() == e["sha256"]
with ProcessPoolExecutor(16) as ex:
    bad = [p for p, ok in ex.map(h, m["entries"]) if not ok]
print("SHA_OK" if not bad else f"SHA_BAD {bad[:5]}")
PY
bash $S/serve_eval.sh c1x
