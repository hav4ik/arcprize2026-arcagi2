#!/bin/bash
# E042 RW112 (Runpod 4xL4, 2026-10-24 ~19:45): the user's E112-vs-E128 Kaggle decision ("continue with E112").
# Stops the E128 server (rw_e128t3) and deletes the E128 checkpoint (on HF, ~9 min to restore), downloads E043's tool
# E112 2-bit export (chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit, 124,890,565,440 B; Engram as <= 5 GB row
# chunks, loaded by apply_compact_engram_v2.py's model.py edit, applied 19:40) with dl_e128t.sh's disk / shm split,
# and serves it at the E128 flags plus --enable-prompt-tokens-details. Token: $HF_TOKEN from the caller's environment
# (never written). Log: /root/dsv41/runs/rw.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/rw.log; HF=/root/dsv41/venv/bin/hf
REPO=chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit
M=/root/dsv41/model-e112t; X=/dev/shm/e112t
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1000 ]; }
[ -n "${HF_TOKEN:-}" ] || { log "ABORT: no HF_TOKEN"; exit 1; }
$S/stop.sh rw_e128t3 | tee -a $L
for i in $(seq 60); do busy || break; sleep 5; done
busy && { log "ABORT: box busy"; exit 1; }
rm -rf /dev/shm/e128t /root/dsv41/model-e128t
log "E128 deleted; disk free $(df -h / | awk 'NR==2{print $4}'), shm free $(df -h /dev/shm | awk 'NR==2{print $4}')"
mkdir -p $M $X
$HF download $REPO --local-dir $X --include "model-engram-*" --include "gptq/layer_1[89].*" --include "gptq/layer_[23]?.*" >> $L 2>&1 \
  || { log "DL FAILED (shm part)"; exit 1; }
log "E112 shm part done: $(du -sh $X | cut -f1)"
$HF download $REPO --local-dir $M --exclude "model-engram-*" --exclude "gptq/layer_1[89].*" --exclude "gptq/layer_[23]?.*" >> $L 2>&1 \
  || { log "DL FAILED (disk part)"; exit 1; }
mkdir -p $M/gptq
for f in $X/model-engram-*; do ln -sf $f $M/; done
for f in $X/gptq/*; do ln -sf $f $M/gptq/; done
log "E112 DL_DONE: $(ls $M/gptq/*.safetensors | wc -l) gptq layers, $(ls $M/model-engram-*.safetensors | wc -l) engram files; $(du -sbL $M | cut -f1) bytes (expect 124890565440); disk free $(df -h / | awk 'NR==2{print $4}'), shm free $(df -h /dev/shm | awk 'NR==2{print $4}')"
N=rw_e112t; R=/root/dsv41/runs/$N
env DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}' \
  DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$M/gptq DSV41_INDEXER_Q_PREDECODE=1 \
  MODEL=$M SEQS=128 UTIL=0.93 MAXLEN=163840 $S/serve_real.sh $N \
  --enable-prefix-caching --prefix-cache-retention-interval 128 --enable-prompt-tokens-details >> $L
(setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
t=0
until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
  sleep 10; t=$((t + 10))
  if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 3600 ]; then
    log "$N SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-400)"; exit 1
  fi
done
log "$N up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
  "$(grep -o -E "Model loading took [0-9.]+ GiB" $R/server.log | head -1)"
(setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
log "$N host: $(free -g | awk '/Mem:/{print $3 " GiB used / " $7 " available"}'); shm $(df -h /dev/shm | awk 'NR==2{print $3}') used"
a=$(curl -s -m 600 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "dsv41",
  "messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}], "max_tokens": 2048,
  "temperature": 0, "chat_template_kwargs": {"thinking": true, "reasoning_effort": 50}}')
log "$N smoke: $(echo "$a" | /root/dsv41/venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:200]), "| reasoning", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:120]))' 2>&1 | cut -c1-500)"
log "RW112_UP"
