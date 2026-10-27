#!/usr/bin/env bash
# Stop a run_capacity.sh driver once a given phase has finished, before it starts the next phase's server (the next
# phase runs on another box). Also stops anything the driver already started for the next phase, by process group.
#   setsid nohup bash stop_after_phase.sh cap2 on off <driver pid> &
TAG=$1; DONE=$2; NEXT=$3; DRV=$4
F=/root/e051/runs/$TAG.progress
until grep -q "$TAG_$DONE: stopped\|${TAG}_${DONE}: stopped" "$F" 2>/dev/null; do
  kill -0 "$DRV" 2>/dev/null || { echo "driver gone"; exit 0; }
  sleep 3
done
kill -- -"$DRV" 2>/dev/null
sleep 5
D=/root/e051/runs/${TAG}_${NEXT}
if [ -f "$D/server.pgid" ]; then kill -- -"$(cat "$D/server.pgid")"; fi
P=$(cat "$D"/server_offload_*.pid 2>/dev/null)
if [ -n "$P" ]; then PG=$(ps -o pgid= -p "$P" | tr -d ' '); [ -n "$PG" ] && kill -- -"$PG"; fi
[ -f "$D/metrics_logger.pid" ] && kill "$(cat "$D/metrics_logger.pid")" 2>/dev/null
echo "$(date -u +%FT%TZ) stop_after_phase: driver $DRV stopped after ${TAG}_${DONE}; ${TAG}_${NEXT} cleaned" >> "$F"
