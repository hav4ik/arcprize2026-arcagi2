#!/bin/bash
# E068 box: download tool E128 2-bit (E043 HF layout) to local disk. HF_TOKEN comes from the environment only.
REPO=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit
M=/root/e068/model-e128t; L=/root/e068/runs/dl_e128t.log
mkdir -p $M /root/e068/runs
log() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
log start
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_PROGRESS_BARS=1
/venv/main/bin/hf download $REPO --local-dir $M >> $L 2>&1 || { log "DL FAILED"; exit 1; }
log "DL_DONE: $(ls $M/gptq/*.safetensors 2>/dev/null | wc -l) gptq files, $(ls $M/model-engram-*.safetensors 2>/dev/null | wc -l) engram files; $(du -sbL $M | cut -f1) bytes (expect 131090961684)"
