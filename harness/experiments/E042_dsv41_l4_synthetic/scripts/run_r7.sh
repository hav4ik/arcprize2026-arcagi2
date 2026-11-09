#!/bin/bash
# E042 R7: clock-controlled A/B of the Humming dense MXFP8 kernel (--linear-backend humming, patches/humming_v41.py).
# R5's tool replay (289.4 tok/s vs R3's 209.0) is confounded: GPU2 throttles at 87 C, its clock gates every TP4 step,
# and R5's slowest GPU ran 44% faster than R3's (window means 1038 vs 721 MHz). nvidia-smi -lgc is not permitted in
# this container, so the A/B interleaves short decode benches B H H B E B H (B base, H Humming dense, E base with a 4x
# Engram table: 16M rows per rank and layer, 31.5 GiB pinned; R6's 8x table was OOM-killed by the 108 GiB cgroup).
# decode_window records t0_unix, so each point's prefill and window line up with gpu.csv clocks. Then R6's step 2:
# E48 x 128 tool replays, base then Humming dense. Log: /root/dsv41/runs/r7.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/r7.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
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
# dw <name> <rows> [vllm args]: P12's decode-bench server (E64, 128 seqs, util 0.88, MBT 2048, 64K, no prefix caching)
dw() {
  local N=$1 rows=$2; shift 2; local R=/root/dsv41/runs/$N/dw
  busy && { log "ABORT before $N: busy"; exit 1; }
  ENGRAM_ROWS=$rows DSV41_INDEXER_Q_PREDECODE=1 SEQS=128 UTIL=0.88 $S/serve.sh $N "$@" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    [ $rows != 4000000 ] && log "$N host memory: $(free -g | awk '/Mem:/{print $3 " used / " $7 " available GiB"}'); $(grep -o -E 'Engram table offloaded[^;]*' $R/../server.log | head -1 | cut -c1-200)"
    mkdir -p $R
    for p in 1024:128 8192:32; do
      ctx=${p%%:*}; c=${p##*:}
      log "$N ctx $ctx C $c: $($PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c --sock-read 10800 \
        --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
    done
  fi
  $S/stop.sh $N | tee -a $L
}
for i in $(seq 90); do busy || break; sleep 2; done
log "r7 start"

# 1. Interleaved decode benches
dw r7_b1 4000000
dw r7_h1 4000000 --linear-backend humming
dw r7_h2 4000000 --linear-backend humming
dw r7_b2 4000000
dw r7_e4x 16000000
dw r7_b3 4000000
dw r7_h3 4000000 --linear-backend humming
log "$($PY $S/dw_table.py --base r7_b1,r7_b2,r7_b3 'r7_[bhe][0-9]*' 2>&1)"

# 2. E48 x 128 tools (the Kaggle run-7 config, tool_replay v4, flag on, seed 1), base then Humming dense
for v in base hlin; do
  N=r7_e48$v; X=()
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
log "R7_DONE (server stopped; box idle)"
