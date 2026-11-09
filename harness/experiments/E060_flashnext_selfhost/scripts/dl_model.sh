#!/bin/bash
# E060: download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE pinned to c6af9fee (129.0 GB, 88 files) into
# /root/e060/model over plain HTTP (no xet: E057b saw xet stall 3x), resumable, 5 tries. Needs ~130 GB free disk.
# Token: $HF_TOKEN from the caller's environment, sent over SSH stdin and never written to disk, e.g. from the
# workstation:
#   export HF_TOKEN="$(sed -n 's/^huggingface_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
#   printf '%s\n' "$HF_TOKEN" | ssh -p PORT root@HOST 'read -r HF_TOKEN; export HF_TOKEN; \
#     setsid nohup bash /root/e060/repo/scripts/dl_model.sh > /dev/null 2>&1 < /dev/null &'
# Log: /root/e060/runs/dl_model.log (DL_DONE / DL_FAILED). Uses /root/hfenv/bin/hf if present, else its own
# /root/e060/hfenv (huggingface_hub CLI).
REPO=albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE
REV=c6af9fee7ad2cb8ef44b534a99b8c1b81fb0f6da
M=/root/e060/model; L=/root/e060/runs/dl_model.log
export HF_HOME=/root/e060/hf_home HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
mkdir -p $M /root/e060/runs
HF=/root/hfenv/bin/hf
if [ ! -x $HF ]; then
  [ -x /root/e060/hfenv/bin/hf ] || { uv venv -q --python 3.12 /root/e060/hfenv && uv pip install -q --python /root/e060/hfenv/bin/python "huggingface_hub[cli]"; }
  HF=/root/e060/hfenv/bin/hf
fi
echo "[$(date -u +%T)] start (http, no xet) $REPO@$REV; disk free $(df -h / | awk 'NR==2{print $4}')" >> $L
for try in 1 2 3 4 5; do
  if $HF download $REPO --revision $REV --local-dir $M --max-workers 8 >> $L 2>&1; then
    echo "[$(date -u +%T)] DL_DONE: $(find $M -path $M/.cache -prune -o -type f -printf '%s\n' | awk '{s+=$1} END {print s}') bytes, $(find $M -path $M/.cache -prune -o -type f -print | wc -l) files; disk free $(df -h / | awk 'NR==2{print $4}')" >> $L
    exit 0
  fi
  echo "[$(date -u +%T)] try $try failed; retrying in 30 s" >> $L
  sleep 30
done
echo "[$(date -u +%T)] DL_FAILED" >> $L
