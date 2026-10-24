#!/bin/bash
# E042 follow-ups 2: (P3) torch-profile steady C=64 decode (1K ctx, default NCCL); (C2) E72 decode/prefill grid.
# Log: /root/dsv41/runs/followups2.log
set -u
S=/root/dsv41/scripts; L=/root/dsv41/runs/followups2.log
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
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1); $(grep -o 'Model loading took [0-9.]* GiB' $R/server.log | head -1)"
}
log "followups2 start"; $S/stop.sh s3_p2p | tee -a $L
busy && { log "ABORT: busy"; exit 1; }

P=/root/dsv41/runs/p3_prof_decode; mkdir -p $P/trace
SEQS=64 $S/serve.sh p3_prof_decode --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$P/trace\",\"torch_profiler_with_stack\":false,\"warmup_iterations\":3,\"active_iterations\":10}" >> $L
(setsid nohup $S/gpumon.sh p3_prof_decode > /dev/null 2>&1 < /dev/null &)
if wait_up p3_prof_decode; then
  /root/dsv41/venv/bin/python $S/decode_window.py --ctx 1024 --conc 64 --window 150 --json $P/dw_c64.json > $P/dw_c64.txt 2>&1 &
  BP=$!
  sleep 110   # 64 x 1K prefill takes ~87 s; then steady decode
  curl -s -m 30 -X POST localhost:8100/start_profile; sleep 20; curl -s -m 600 -X POST localhost:8100/stop_profile
  log "P3 decode capture stopped; files: $(find $P/trace -type f | wc -l)"
  wait $BP; log "P3 window (profiler inside): $(cat $P/dw_c64.txt | cut -c1-400)"
fi
$S/stop.sh p3_prof_decode | tee -a $L

busy && { log "ABORT before C2: busy"; exit 1; }
MODEL=/root/dsv41/model-e72 SEQS=64 $S/serve.sh c2_e72 >> $L
(setsid nohup $S/gpumon.sh c2_e72 > /dev/null 2>&1 < /dev/null &)
if wait_up c2_e72; then
  log "C2 smoke: $($S/smoke_request.sh 2>&1 | tr '\n' ' ' | cut -c1-250)"
  $S/dw_grid.sh c2_e72 "1024:1 1024:8 1024:32 1024:64 8192:8 8192:32"
fi
log "FOLLOWUPS2_DONE (c2_e72 server left running)"
