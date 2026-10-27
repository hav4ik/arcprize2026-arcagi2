#!/bin/bash
# E059 B16 box 57983124 (plain HTTP, resumable, from the start; E057b: xet stalled 3x): download the tool E128 2-bit checkpoint in full to disk (no /dev/shm split; 204 GB disk is enough).
# Token: $HF_TOKEN from the caller's environment (sent over SSH stdin, never written to disk). Log: /root/dsv41/runs/dl_e128t.log
REPO=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit
REV=9bce98c5ecda0a9226edc6eab72a689089db1e37
M=/root/dsv41/model-e128t; L=/root/dsv41/runs/dl_e128t.log
export HF_HOME=/root/dsv41/cache/hf_dl HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
mkdir -p $M
echo "[$(date -u +%T)] start (http, no xet) $REPO@$REV" >> $L
if /root/hfenv/bin/hf download $REPO --revision $REV --local-dir $M >> $L 2>&1; then
  echo "[$(date -u +%T)] DL_DONE: $(find $M -path $M/.cache -prune -o -type f -printf '%s\n' | awk '{s+=$1} END {print s}') bytes (expect 131090911045); disk free $(df -h / | awk 'NR==2{print $4}')" >> $L
else
  echo "[$(date -u +%T)] DL_FAILED" >> $L
fi
