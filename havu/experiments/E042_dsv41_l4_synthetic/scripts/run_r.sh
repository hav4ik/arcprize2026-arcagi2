#!/bin/bash
# E042 R: trace-shaped load from the real Flash-1017 ARC runs (scripts/trace_shapes.py), PR #154 on, prefix caching on,
# max-model-len 160K (no-tools traces reach 146K). Per model, at the concurrency its KV holds in steady state
# (E64: 64, E48: 128; trace_shapes.py fit):
#   R1 no tools: steady-state snapshot (length-biased traces, uniform position; mean slot context ~42K), decode window,
#      with the server's one profiler capture.
#   R2 tools: E030 attempt shapes, slots start mid-attempt, back to back, 1 s tool delay; 40-min window.
# Log: /root/dsv41/runs/r.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/r.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
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
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
}
log "r start"
for v in e64 e48; do
  case $v in
    e64) M=/root/dsv41/model-e64; C=64;;
    e48) M=/root/dsv41/model-e48; C=128;;
  esac
  N=r_$v; R=/root/dsv41/runs/$N; T=$R/trace; mkdir -p $T
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=$M SEQS=128 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N --enable-prefix-caching \
    --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$T\",\"torch_profiler_with_stack\":false,\"warmup_iterations\":3,\"active_iterations\":10}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    log "$N R1 no-tools snapshot C $C: $($PY $S/decode_window.py --ctx-file $TS/snap_notools_c${C}_s1.json --window 120 \
      --seed $C --sock-read 14400 --max-model-len 163840 --profile-s 20 --json $R/r1_notools_c$C.json 2>&1 | tail -1 \
      | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-600)"
    log "$N R1 profile: $($PY $S/prof_summary.py $T --indexer-per-step 8 --json $T/summary.json 2>&1 | grep -o '"per_step_ms": {[^}]*}' | head -4 | tr '\n' ' ')"
    log "$N R2 tools C $C: $($PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc $C --warm-s 300 \
      --window-s 2400 --tool-delay 1 --seed 1 --json $R/r2_tools_c$C.json 2>&1 | tail -1 | cut -c1-900)"
  fi
  $S/stop.sh $N | tee -a $L
done
log "R_DONE (server stopped; box idle)"
