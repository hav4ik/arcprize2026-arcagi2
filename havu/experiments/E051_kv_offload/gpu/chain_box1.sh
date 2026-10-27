#!/usr/bin/env bash
# Box 1 follow-up after the cap2 offload-on arms (chained on stop_after_phase's line in cap2.progress):
#  1. byte-level KV parity with shared_region=false (private pinned host memory, the planned Kaggle setting);
#  2. heavy-context pair (replay --scale 1.25) at 56 streams / 20 s idle, offload on (28 GiB, private) then off.
set -uo pipefail
G=/root/e051/gpu; R=/root/e051/runs
PROG=$R/chain_box1.progress
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$PROG"; }
until grep -q "stop_after_phase" "$R/cap2.progress" 2>/dev/null; do sleep 15; done
sleep 20
log "cap2 on-arms finished; parity with shared_region=false"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 100 ] || { log "GPUs busy (${used} MiB), abort"; exit 1; }
RUN=par_private OFFLOAD_GIB=28 POLICY=checkpoint SHARED_REGION=false PYTHONPATH=$G \
  EXTRA_ARGS="--worker-extension-cls e051_ext.E051Ext" bash $G/start_run.sh >> "$PROG" 2>&1 || { log "start failed"; exit 1; }
D=$R/par_private; L=$D/server_offload_28g_checkpoint.log; P=$(cat $D/server_offload_28g_checkpoint.pid); t0=$(date +%s)
until grep -q "Application startup complete" "$L"; do
  kill -0 "$P" 2>/dev/null || { log "parity server exited"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { log "parity server timeout"; exit 1; }
  sleep 10
done
log "parity server ready after $(( $(date +%s) - t0 ))s; $(grep -c 'shared_offload_region' "$L") shm-region lines (expect 0)"
(cd "$D" && /root/dsv41/venv_offload/bin/python -I $G/kv_parity.py --lengths 3000 20000 60000 --max-tokens 300 \
  --json "$D/kv_parity_private.json" > "$D/kv_parity_private.out" 2>&1)
log "parity: $(tail -1 "$D/kv_parity_private.out") ; tracebacks $(grep -c -E 'Traceback|AssertionError' "$L")"
PG=$(cat "$D/server.pgid"); kill -- -"$PG"; kill "$(cat "$D/metrics_logger.pid")" 2>/dev/null
for i in $(seq 1 60); do sleep 2; pgrep -g "$PG" > /dev/null || break; done
sleep 10
log "parity server stopped; heavy pair next"
TAG=cap3 PHASES="on:28:checkpoint:false off:0:-" RUNS="56:20:1.25" bash $G/run_capacity2.sh >> "$PROG" 2>&1
log "chain done"
