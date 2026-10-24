#!/bin/bash
# E042 follow-ups after s1_full: (P1) torch-profile steady C=32 decode with FULL graphs; (EP1) expert parallel sweep.
# Usage: run_followups.sh   Log: /root/dsv41/runs/followups.log
set -u
S=/root/dsv41/scripts; L=/root/dsv41/runs/followups.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
wait_up() {  # name
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
}
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }

log "followups start"; $S/stop.sh s1_full | tee -a $L
busy && { log "ABORT: port 8100 or GPUs still busy"; exit 1; }

# P1: profile. Profiler records 10 active iterations after 3 warmup ones, starting at /start_profile.
P=/root/dsv41/runs/p1_prof; mkdir -p $P/trace
SEQS=64 $S/serve.sh p1_prof --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$P/trace\",\"torch_profiler_with_stack\":false,\"warmup_iterations\":3,\"active_iterations\":10}" >> $L
(setsid nohup $S/gpumon.sh p1_prof > /dev/null 2>&1 < /dev/null &)
if wait_up p1_prof; then
  # Capture 1 during prefill (32 x 1024-token prompts in 2048-token chunks), capture 2 in steady C=32 decode.
  /root/dsv41/venv/bin/python $S/decode_window.py --ctx 1024 --conc 32 --window 120 --json $P/dw_c32.json > $P/dw_c32.txt 2>&1 &
  BP=$!
  sleep 8;  curl -s -m 30 -X POST localhost:8100/start_profile; sleep 40; curl -s -m 600 -X POST localhost:8100/stop_profile
  log "P1 prefill capture stopped; files: $(find $P/trace -type f | wc -l)"
  sleep 20; curl -s -m 30 -X POST localhost:8100/start_profile; sleep 15; curl -s -m 600 -X POST localhost:8100/stop_profile
  log "P1 decode capture stopped; files: $(find $P/trace -type f | wc -l)"
  wait $BP; log "P1 window (profiler overhead inside): $(cat $P/dw_c32.txt)"
fi
$S/stop.sh p1_prof | tee -a $L

# EP1: expert parallel (16 full-width experts per rank), same flags otherwise; clean decode windows.
busy && { log "ABORT before EP1: port or GPUs busy"; exit 1; }
SEQS=64 $S/serve.sh ep1 --enable-expert-parallel >> $L
(setsid nohup $S/gpumon.sh ep1 > /dev/null 2>&1 < /dev/null &)
if wait_up ep1; then
  log "EP1 smoke: $($S/smoke_request.sh 2>&1 | tr '\n' ' ' | cut -c1-300)"
  $S/dw_grid.sh ep1 "1024:1 1024:8 1024:32 1024:64 8192:8 8192:32"
  log "EP1 grid: $(tr '\n' ' ' < /root/dsv41/runs/ep1/dw/grid.log | cut -c1-2500)"
fi
$S/stop.sh ep1 | tee -a $L

# P2: larger prefill chunk (max-num-batched-tokens 8192): prefill rate and decode at 8K.
busy && { log "ABORT before P2: port or GPUs busy"; exit 1; }
SEQS=64 MBT=8192 $S/serve.sh p2_mbt8k >> $L
(setsid nohup $S/gpumon.sh p2_mbt8k > /dev/null 2>&1 < /dev/null &)
if wait_up p2_mbt8k; then
  $S/dw_grid.sh p2_mbt8k "8192:1 8192:8 1024:32"
  log "P2 grid: $(tr '\n' ' ' < /root/dsv41/runs/p2_mbt8k/dw/grid.log | cut -c1-1500)"
fi
log "FOLLOWUPS_DONE (p2_mbt8k server left running)"
