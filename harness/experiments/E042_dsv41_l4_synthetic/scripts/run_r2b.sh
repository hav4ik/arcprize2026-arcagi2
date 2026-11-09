#!/bin/bash
# E042 R2b: generated-token prefix caching. The fork keeps sliding-window KV only at the prompt end by default
# (prefix_cache_retention_interval 0), so a re-sent transcript hits only the previous prompt's blocks (P7: 2944 of
# 3456 / 3968). --prefix-cache-retention-interval 128 keeps every SWA block. Takes over from run_r.sh once its E64 R2
# (no flag) has reported, then:
#   1. P7 gencache with the flag (expect 3456 / 3968 hits),
#   2. R2 tools E64 x 64 with the flag (A/B against run_r.sh's R2, same seed),
#   3. E48 x 128: R1 no-tools snapshot (profiled), then R2 tools with the flag,
#   4. R_DONE in r.log, which releases P10.
# Log: /root/dsv41/runs/r2b.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/r2b.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
RET=(--prefix-cache-retention-interval 128)
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
serve() {  # serve <name> <model> [extra...]
  local N=$1 M=$2; shift 2
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=$M SEQS=128 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N --enable-prefix-caching "$@" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  wait_up $N && { (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &); return 0; }
}
r2() {  # r2 <name> <conc>
  log "$1 R2 tools C $2 (retention 128): $($PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc $2 --warm-s 300 \
    --window-s 2400 --tool-delay 1 --seed 1 --json /root/dsv41/runs/$1/r2_tools_c$2.json 2>&1 | tail -1 | cut -c1-900)"
}

# Take over from run_r.sh as soon as its E64 R2 reports (before it starts the E48 server).
until grep -qE "r_e64 R2 tools|ABORT|FAILED" /root/dsv41/runs/r.log 2>/dev/null; do sleep 2; done
P=$(pgrep -f "^/bin/bash /root/dsv41/scripts/run_r.sh$")
log "r2b start; stopping run_r.sh pid ${P:-none} before its E48 phase"
[ -n "$P" ] && kill $P
for i in $(seq 90); do busy || break; sleep 2; done
if [ -s /root/dsv41/runs/r_e48/server.pid ]; then
  $S/stop.sh r_e48 | tee -a $L; mv /root/dsv41/runs/r_e48 /root/dsv41/runs/r_e48_preempted
fi
busy && { $S/stop.sh r_e64 | tee -a $L; }
busy && { log "ABORT: GPUs still busy"; exit 1; }

# 1. gencache with the flag
N=p11_ret128
if serve $N /root/dsv41/model-e64 "${RET[@]}"; then
  log "$N gencache: $($PY $S/prefix_probe.py gencache --seed 11 --json /root/dsv41/runs/$N/gencache.json 2>&1 | tr '\n' ' ' | cut -c1-600)"
fi
$S/stop.sh $N | tee -a $L

# 2. R2 E64 x 64 with the flag
N=r_e64ret
busy && { log "ABORT before $N: busy"; exit 1; }
serve $N /root/dsv41/model-e64 "${RET[@]}" && r2 $N 64
$S/stop.sh $N | tee -a $L

# 3. E48 x 128: R1 (no tools; the flag does not matter there) and R2 with the flag, one server
N=r_e48ret; T=/root/dsv41/runs/$N/trace; mkdir -p $T
busy && { log "ABORT before $N: busy"; exit 1; }
if serve $N /root/dsv41/model-e48 "${RET[@]}" --profiler-config \
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$T\",\"torch_profiler_with_stack\":false,\"warmup_iterations\":3,\"active_iterations\":10}"; then
  log "$N R1 no-tools snapshot C 128: $($PY $S/decode_window.py --ctx-file $TS/snap_notools_c128_s1.json --window 120 \
    --seed 128 --sock-read 14400 --max-model-len 163840 --profile-s 20 --json /root/dsv41/runs/$N/r1_notools_c128.json 2>&1 \
    | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-600)"
  log "$N R1 profile: $($PY $S/prof_summary.py $T --indexer-per-step 8 --json $T/summary.json 2>&1 | grep -o '"per_step_ms": {[^}]*}' | head -4 | tr '\n' ' ')"
  r2 $N 128
fi
$S/stop.sh $N | tee -a $L
log "R2B_DONE (server stopped)"
echo "[$(date -u +%H:%M:%S)] R_DONE (via run_r2b.sh)" >> /root/dsv41/runs/r.log
