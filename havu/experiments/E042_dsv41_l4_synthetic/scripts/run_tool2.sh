#!/bin/bash
# E042 T2 (Runpod 4xL4, 2026-10-24 10:15): tool-mode throughput for E043's tool-harness targets (the user's
# direction, relayed by E043: "a pruned + quantized model that does tool-integrated reasoning well", E128 and E96).
# Dummy weights in E043's schemas (w13 uint2 g128 + w2 uint2 g64 for E128, w2 uint3 g64 for E96), the 64 GiB tool
# Engram table (32.6M synthetic rows per rank and layer), util 0.93 (KV 1.47M / 2.72M), 128 seqs, 160K, prefix caching,
# retention 128, PR #154 flag. Traffic: T72's multi-turn tool replay (tools_e030_hdrmin.json, the full model's
# tool traces) until E043 sends its tool E128 / E96 per-turn shapes. Window 30 min after 5 min warm. Log: runs/t2.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/t2.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1000 ]; }
wait_up() {
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-400)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
    "$(grep -o -E "Model loading took [0-9.]+ GiB" $R/server.log | head -1)"
}
H() { echo "{\"quant_method\":\"humming\",\"weight_dtype\":\"$1\",\"weight_scale_dtype\":\"bfloat16\",\"group_size\":$2,\"has_zero_point\":true}"; }
# tl <name> <model> <C> <w2 override JSON>
tl() {
  local N=$1 m=$2 c=$3 w2=$4; local R=/root/dsv41/runs/$N
  for i in $(seq 60); do busy || break; sleep 5; done
  busy && { log "ABORT before $N: busy"; exit 1; }
  env DSV41_EXPERT_HUMMING="$(H uint2 128)" DSV41_EXPERT_HUMMING_W2="$w2" ENGRAM_ROWS=32600000 DSV41_INDEXER_Q_PREDECODE=1 \
    MODEL=/root/dsv41/model-$m SEQS=128 UTIL=0.93 MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching --prefix-cache-retention-interval 128 >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    log "$N tools C $c: $(timeout 4200 $PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc $c --warm-s 300 \
      --window-s 1800 --tool-delay 1 --seed 1 --json $R/r2_tools_c$c.json 2>&1 | tail -1 | cut -c1-900)"
    log "$N host: $(free -g | awk '/Mem:/{print $3 " GiB used / " $7 " available"}')"
    log "$N health: $($PY $S/run_health.py $R 2>&1 | cut -c1-500)"
  fi
  $S/stop.sh $N | tee -a $L
}
for i in $(seq 90); do busy || break; sleep 2; done
log "t2 start"
tl t2_x128_c48 e128 48 '{"group_size": 64}'
tl t2_x96_c80 e96 80 '{"weight_dtype": "uint3", "group_size": 64}'
log "T2_DONE (server stopped; box idle)"
