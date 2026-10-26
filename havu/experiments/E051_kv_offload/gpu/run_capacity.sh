#!/usr/bin/env bash
# E051 gate 4: capacity A/B on a dev box. For each phase (server config), start the server, wait until ready, run the
# tool replay for each "streams:tool_idle_s" pair (after resetting the GPU and host prefix caches), then stop the server
# by process group. Progress lines go to /root/e051/runs/$TAG.progress.
#
#   TAG=cap1 PHASES="on:28:checkpoint off:0:-" RUNS="56:20 72:20 36:1" setsid bash run_capacity.sh &
set -uo pipefail
TAG=${TAG:?set TAG}
PHASES=${PHASES:-"on:28:checkpoint off:0:-"}
RUNS=${RUNS:-"56:20 72:20 36:1"}
WARM=${WARM:-600}
WINDOW=${WINDOW:-2400}
MAXWALL=${MAXWALL:-5400}
PY=/root/dsv41/venv_offload/bin/python
G=/root/e051/gpu
PROG=/root/e051/runs/$TAG.progress
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$PROG"; }

for ph in $PHASES; do
  IFS=: read -r name gib pol <<< "$ph"
  [ "$pol" = "-" ] && pol=checkpoint
  RUN=${TAG}_${name}
  D=/root/e051/runs/$RUN
  if ! RUN=$RUN OFFLOAD_GIB=$gib POLICY=$pol bash $G/start_run.sh >> "$PROG" 2>&1; then log "$RUN: start failed"; exit 1; fi
  L=$D/server_offload_${gib}g_${pol}.log
  P=$(cat "$D"/server_offload_*.pid)
  t0=$(date +%s)
  until grep -q "Application startup complete" "$L"; do
    kill -0 "$P" 2>/dev/null || { log "$RUN: server exited during startup"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { log "$RUN: startup timeout"; exit 1; }
    sleep 10
  done
  log "$RUN: ready after $(( $(date +%s) - t0 ))s"
  # Smoke: 4 streams for ~2 min, so a broken workload or script fails before the 50-min runs.
  $PY -I $G/tool_replay_kv.py --shapes $G/tools_e043_e128t.json --url http://127.0.0.1:8100 --model dsv41 \
    --conc 4 --tool-delay 1 --warm-s 30 --window-s 60 --max-wall-s 900 --seed 1 --json "$D/smoke.json" > "$D/smoke.out" 2>&1
  rc=$?
  log "$RUN: smoke rc=$rc $(tail -1 "$D/smoke.out" | cut -c1-300)"
  [ "$rc" -eq 0 ] || { log "$RUN: smoke failed, stopping (server left running)"; exit 1; }
  for r in $RUNS; do
    C=${r%%:*}; DL=${r##*:}
    curl -s -X POST "http://127.0.0.1:8100/reset_prefix_cache?reset_external=true" > /dev/null
    sleep 5
    out=$D/replay_C${C}_D${DL}
    log "$RUN: replay C=$C D=$DL start"
    $PY -I $G/tool_replay_kv.py --shapes $G/tools_e043_e128t.json \
      --url http://127.0.0.1:8100 --model dsv41 --conc "$C" --tool-delay "$DL" --warm-s "$WARM" --window-s "$WINDOW" \
      --max-wall-s "$MAXWALL" --seed 0 --json "$out.json" > "$out.out" 2>&1
    rc=$?
    log "$RUN: replay C=$C D=$DL rc=$rc $(tail -1 "$out.out" | cut -c1-400)"
    [ "$rc" -eq 0 ] || { log "$RUN: replay failed, stopping the matrix (server left running)"; exit 1; }
    kill -0 "$P" 2>/dev/null || { log "$RUN: server died"; exit 1; }
  done
  PG=$(cat "$D/server.pgid")
  kill -- -"$PG"; kill "$(cat "$D/metrics_logger.pid")" 2>/dev/null
  for i in $(seq 1 60); do sleep 2; pgrep -g "$PG" > /dev/null || break; done
  sleep 10
  log "$RUN: stopped; tracebacks $(grep -c -E 'Traceback|AssertionError' "$L")"
done
log "ALL DONE"
