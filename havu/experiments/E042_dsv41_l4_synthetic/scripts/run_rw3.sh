#!/bin/bash
# E042 RW3 (Runpod, 2026-10-24 ~19:45): restart the real tool E128 server as rw_e128t3 with
# --enable-prompt-tokens-details, so each response reports its cached prompt tokens (prefix_hitlog.py measures
# per-turn hits under real 40-stream ARC load: re-tokenization breaks vs sliding-window eviction). Same flags as rw_e128t2.
S=/root/dsv41/scripts; L=/root/dsv41/runs/rw.log; M=/root/dsv41/model-e128t
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1000 ]; }
$S/stop.sh rw_e128t2 | tee -a $L
for i in $(seq 60); do busy || break; sleep 5; done
busy && { log "ABORT: box busy"; exit 1; }
N=rw_e128t3; R=/root/dsv41/runs/$N
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
log "$N up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
(setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
log "RW3_UP"
