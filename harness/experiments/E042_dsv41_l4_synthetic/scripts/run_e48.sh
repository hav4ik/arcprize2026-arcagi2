#!/bin/bash
# E042 C3: E48 (48 routed experts/layer, top-6), S4 flags (max-num-seqs 128, util 0.88). Starts after P5; then hands
# over to P6 (run_bottleneck.sh), whose first server is E64 with the same flags and grid, for a back-to-back
# comparison at similar GPU2 temperatures. Log: /root/dsv41/runs/e48.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/e48.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
until grep -qE "AGENTIC_DONE|ABORT|SERVE FAILED" /root/dsv41/runs/agentic.log 2>/dev/null; do sleep 30; done
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
if busy; then log "ABORT: port 8100 or GPUs busy"; else
  R=/root/dsv41/runs/c3_e48
  MODEL=/root/dsv41/model-e48 SEQS=128 UTIL=0.88 $S/serve.sh c3_e48 >> $L
  (setsid nohup $S/gpumon.sh c3_e48 > /dev/null 2>&1 < /dev/null &)
  t=0; ok=1
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "c3_e48 SERVE FAILED: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; ok=0; break; fi; done
  if [ $ok = 1 ]; then
    log "c3_e48 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1); $(grep -o 'Model loading took [0-9.]* GiB' $R/server.log | head -1)"
    log "c3_e48 smoke: $($S/smoke_request.sh 2>&1 | grep -E 'finite' | cut -c1-200)"
    $S/dw_grid.sh c3_e48 "1024:1 1024:8 1024:32 1024:64 1024:128 8192:32"
    log "c3_e48 grid: $(sed -E 's/"first_token_s": \[[^]]*\], //' $R/dw/grid.log | tr '\n' ' ' | cut -c1-2500)"
  fi
  $S/stop.sh c3_e48 | tee -a $L
fi
log "E48_DONE; starting P6"
exec $S/run_bottleneck.sh
