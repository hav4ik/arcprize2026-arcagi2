#!/bin/bash
# E042 R6, after R5:
#   1. Engram table size: 8x the compact table (ENGRAM_ROWS 32M per rank and layer, 63 GiB pinned host memory vs
#      7.9 GiB), default kernels, P12's flags and points 1K x 64, 1K x 128, 8K x 16 (vs P12's base mean). Row ids
#      are folded onto the table, so lookups spread over 63 GiB: a check of how lookup cost (TLB, host memory)
#      scales toward the full table (130-190 GiB on Kaggle). The fork has no THP option.
#   2. E48 x 128 tool replay (the Kaggle run-7 config, tool_replay v4, flag on, seed 1), default kernels, then the
#      Humming dense kernel (--linear-backend humming): the A/B for the strongest Kaggle candidate.
# Log: /root/dsv41/runs/r6.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/r6.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
wait_up() {
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
    "$(grep -o -E "Using '[A-Z_]+' Mxfp4 MoE backend|Using [A-Za-z0-9]+ for MXFP8 GEMM" $R/server.log | sort -u | tr '\n' ' ')"
}
until grep -qE "R5_DONE|ABORT" /root/dsv41/runs/r5.log 2>/dev/null; do sleep 60; done
for i in $(seq 90); do busy || break; sleep 2; done
log "r6 start"

# 1. Engram 8x
N=r6_engram8x; R=/root/dsv41/runs/$N/dw
busy && { log "ABORT before $N: busy"; exit 1; }
log "host memory before: $(free -g | awk '/Mem:/{print $3 " used / " $7 " available GiB"}')"
ENGRAM_ROWS=32000000 DSV41_INDEXER_Q_PREDECODE=1 SEQS=128 UTIL=0.88 $S/serve.sh $N >> $L
(setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
if wait_up $N; then
  log "$N host memory: $(free -g | awk '/Mem:/{print $3 " used / " $7 " available GiB"}'); $(grep -o -E 'Engram table offloaded[^;]*' $R/../server.log | head -1 | cut -c1-200)"
  mkdir -p $R
  for p in 1024:64 1024:128 8192:16; do
    ctx=${p%%:*}; c=${p##*:}
    log "$N ctx $ctx C $c: $($PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c --sock-read 10800 \
      --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
  done
fi
$S/stop.sh $N | tee -a $L

# 2. E48 x 128 tools, default vs Humming dense
for v in base hlin; do
  N=r6_e48$v; X=()
  [ $v = hlin ] && X=(--linear-backend humming)
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-e48 SEQS=128 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching --prefix-cache-retention-interval 128 "${X[@]}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    log "$N R2 tools C 128: $($PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc 128 --warm-s 300 \
      --window-s 2400 --tool-delay 1 --seed 1 --json /root/dsv41/runs/$N/r2_tools_c128.json 2>&1 | tail -1 | cut -c1-900)"
  fi
  $S/stop.sh $N | tee -a $L
done
log "R6_DONE (server stopped; box idle)"
