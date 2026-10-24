#!/bin/bash
# E042 KV4 (Runpod A, 2026-10-24 ~20:45): the vLLM blog (2026-10-23) puts V4.1's KV at ~890 B/token with FP4
# compression; our fp8 KV is ~1.5 KB/token/GPU. Same E112 server as rw_e112t with --kv-cache-dtype nvfp4_ds_mla:
# does it start on sm89, how many KV tokens, and (kv_nll_probe.py) does the NLL match fp8?
S=/root/dsv41/scripts; L=/root/dsv41/runs/rw.log; M=/root/dsv41/model-e112t
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1000 ]; }
$S/stop.sh rw_e112t | tee -a $L
for i in $(seq 60); do busy || break; sleep 5; done
busy && { log "ABORT: box busy"; exit 1; }
N=rw_e112t_fp4; R=/root/dsv41/runs/$N
# serve_real.sh hard-codes --kv-cache-dtype fp8; a later flag wins in vLLM's argparse
env DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}' \
  DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$M/gptq DSV41_INDEXER_Q_PREDECODE=1 \
  MODEL=$M SEQS=128 UTIL=0.93 MAXLEN=163840 $S/serve_real.sh $N \
  --enable-prefix-caching --prefix-cache-retention-interval 128 --enable-prompt-tokens-details --kv-cache-dtype nvfp4_ds_mla >> $L
t=0
until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
  sleep 10; t=$((t + 10))
  if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 1800 ]; then
    log "$N SERVE FAILED after ${t}s: $(grep -E 'Error|error' $R/server.log | grep -v Warning | tail -3 | cut -c1-500)"; exit 1
  fi
done
log "$N up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1); $(grep -o -E 'Available KV cache memory: [0-9.]+ GiB' $R/server.log | head -1)"
(setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
log "KVFP4_UP"
