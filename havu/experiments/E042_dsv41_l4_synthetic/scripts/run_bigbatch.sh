#!/bin/bash
# E042 S4: E64 batch ceiling (max-num-seqs 128, util 0.88): clean decode at C=96/128 (1K) and C=64 (8K). Log: runs/bigbatch.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/bigbatch.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
$S/stop.sh c2_e72 | tee -a $L
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ] && { log "ABORT: busy"; exit 1; }
SEQS=128 UTIL=0.88 $S/serve.sh s4_c128 >> $L
(setsid nohup $S/gpumon.sh s4_c128 > /dev/null 2>&1 < /dev/null &)
R=/root/dsv41/runs/s4_c128; t=0
until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
  sleep 10; t=$((t + 10))
  if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then log "SERVE FAILED: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; exit 1; fi
done
log "s4_c128 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1); $(grep -oE 'Graph capturing finished in [0-9.]+ secs, took [0-9.-]+ GiB' $R/server.log | tail -1)"
$S/dw_grid.sh s4_c128 "1024:96 1024:128 8192:64"
$S/stop.sh s4_c128 | tee -a $L
log "BIGBATCH_DONE (server stopped; box idle)"
