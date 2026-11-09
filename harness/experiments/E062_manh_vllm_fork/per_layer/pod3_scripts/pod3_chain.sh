#!/bin/bash
# pod 3 setup chain (C1 on Manh's stack). Needs KAGGLE_API_TOKEN and HF_TOKEN (read-only) in the environment.
# Logs: /root/chain.log (steps), per-step logs beside it. Ends with CHAIN_DONE or CHAIN_FAIL <step>.
L=/root/chain.log; exec >> $L 2>&1
log() { echo "$(date -u +%T) $*"; }
fail() { log "CHAIN_FAIL $*"; exit 1; }
log "start"
python3 -m venv /root/kgl || fail venv
/root/kgl/bin/pip install -q kagglehub uv "huggingface_hub[hf_xet]" || fail pip
[ -x /usr/bin/uv ] || ln -s /root/kgl/bin/uv /usr/bin/uv
/root/kgl/bin/uv python install 3.13 > /root/uvpy.log 2>&1 || fail uv-python
log "bootstrap ok: $(/root/kgl/bin/uv python find 3.13)"
# C1 from HF (read-only key), in the background
( /root/kgl/bin/hf download chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-img-E128-2bit --local-dir /root/models/c1 > /root/dl_c1.log 2>&1 \
  && echo DL_DONE >> /root/dl_c1.log || echo DL_FAILED >> /root/dl_c1.log ) &
DLC1=$!
unset HF_TOKEN
# Manh's wheels (vllm-wheels v8, sequential) and his source dataset v22
cd /root/b32m && /root/kgl/bin/python -I dl_kaggle_files.py dataset nguyennguyen599/vllm-wheels/versions/10 manh_wheels_files_v10.tsv /root/b32m/dl > /root/dl_wheels.log 2>&1
grep -q "DL_DONE ok=294 failed=0" /root/dl_wheels.log || fail wheels
log "wheels ok"
KAGGLEHUB_CACHE=/root/khsrc /root/kgl/bin/python -I -c 'import kagglehub,shutil; p=kagglehub.dataset_download("nguyennguyen599/arc2-2026-src/versions/22"); print("SRC", p); shutil.copytree(p, "/root/e066/work", dirs_exist_ok=True)' > /root/dl_src.log 2>&1 || fail src
log "src ok: $(grep SRC /root/dl_src.log)"
unset KAGGLE_API_TOKEN
sha256sum /root/b32m/dl/wheels/wheels/vllm-0.32.1rc1.dev0+gf506afa210-*.whl | cut -c1-16 >> $L
log "installing vllm-wheels v10 (coordinator decision)"
bash /root/b32m/install_b32m.sh; grep -q INSTALL_DONE /root/b32m/install.log || fail install
grep -E "STEP_FAIL|IMPORT_OK" /root/b32m/install.log
bash /root/e062pl/apply_patch_f506.sh > /root/patch.log 2>&1 || fail patch
cat /root/patch.log
wait $DLC1
tail -1 /root/dl_c1.log | grep -q DL_DONE || fail c1-download
/root/b32m/venv/bin/python -I - <<'PY' || fail c1-verify
import json, hashlib, os
m = json.load(open("/root/e066/c1_gptq_sha256_manifest.json")); root = "/root/models/c1"
ents = m.get("entries") or m.get("files")
bad = 0
for e in ents:
    p = os.path.join(root, e["path"])
    if not os.path.exists(p) or os.path.getsize(p) != e["size"]:
        bad += 1; print("SIZE_BAD", e["path"]); continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""): h.update(b)
    if h.hexdigest() != e["sha256"]:
        bad += 1; print("SHA_BAD", e["path"])
print("C1_VERIFY", "OK" if bad == 0 else f"BAD {bad}", len(ents), "files")
raise SystemExit(1 if bad else 0)
PY
log "CHAIN_DONE"
