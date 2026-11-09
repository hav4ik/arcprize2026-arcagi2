#!/bin/bash
# E042 RW (Runpod, 15:20): download E043's tool E128 2-bit checkpoint (chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-
# E128-2bit, 122.1 GiB) for the user's real-weight check ("Yes, drop the 48 replay"). It fits neither /dev/shm (117
# GiB) nor the free disk (34 GiB) alone. Disk (/root/dsv41/model-e128t): backbone, configs, gptq layers 00-17 (29.9
# GiB). /dev/shm/e128t: the Engram files and gptq layers 18-39 (92.2 GiB), symlinked into the model dir. Token: $HF_TOKEN
# from the caller's environment (never written to disk). One flag per pattern: `hf download` reads extra patterns after
# one --include as file names. Log: /root/dsv41/runs/dl_e128t.log
REPO=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit
M=/root/dsv41/model-e128t; X=/dev/shm/e128t; HF=/root/dsv41/venv/bin/hf; L=/root/dsv41/runs/dl_e128t.log
log() { echo "[$(date -u +%H:%M:%S)] $*" >> $L; }
mkdir -p $M $X
log "start"
$HF download $REPO --local-dir $X --include "model-engram-*" --include "gptq/layer_1[89].*" --include "gptq/layer_[23]?.*" >> $L 2>&1 \
  || { log "DL FAILED (shm part)"; exit 1; }
log "shm part done: $(du -sh $X | cut -f1)"
$HF download $REPO --local-dir $M --exclude "model-engram-*" --exclude "gptq/layer_1[89].*" --exclude "gptq/layer_[23]?.*" >> $L 2>&1 \
  || { log "DL FAILED (disk part)"; exit 1; }
log "disk part done: $(du -sh $M | cut -f1)"
mkdir -p $M/gptq
for f in $X/model-engram-*; do ln -sf $f $M/; done
for f in $X/gptq/*; do ln -sf $f $M/gptq/; done
log "DL_DONE: $(ls $M/gptq/*.safetensors | wc -l) gptq layers, $(ls $M/model-engram-*.safetensors | wc -l) engram files; $(du -sbL $M | cut -f1) bytes (expect 131090961684); disk free $(df -h / | awk 'NR==2{print $4}'), shm free $(df -h /dev/shm | awk 'NR==2{print $4}')"
