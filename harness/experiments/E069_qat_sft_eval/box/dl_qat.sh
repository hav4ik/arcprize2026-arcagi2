#!/bin/bash
# E068 box: download the QAT + SFT E128 (chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch, weights only, no Engram table)
# to local disk. HF_TOKEN from the environment only.
REPO=chankhavu/ds-qat-e0128-arazdSFT-v01-025epoch
M=/root/e068/model-qat025; L=/root/e068/runs/dl_qat025.log
mkdir -p $M /root/e068/runs
log() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
log start
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_PROGRESS_BARS=1
/venv/main/bin/hf download $REPO --local-dir $M >> $L 2>&1 || { log "DL FAILED"; exit 1; }
log "DL_DONE: $(find $M -path $M/.cache -prune -o -type f -print | wc -l) files; $(du -sb --exclude=.cache $M | cut -f1) bytes"
