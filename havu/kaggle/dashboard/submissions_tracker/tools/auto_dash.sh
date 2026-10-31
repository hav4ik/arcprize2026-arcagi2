#!/bin/bash
# Keeps the replica dashboard data current with no agent: every INTERVAL s rebuild into DASH_DIR, copy the JSONs
# here, refresh publish_queue.json. Start: setsid nohup tools/auto_dash.sh >/dev/null 2>&1 &   Stop: touch auto_dash.stop
T="$(cd "$(dirname "$0")/.." && pwd)"
DASH_DIR="${DASH_DIR:-/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/dashboard}"
INTERVAL="${INTERVAL:-300}"; PIDF="$T/auto_dash.pid"; STOPF="$T/auto_dash.stop"; LOG="$T/auto_dash.log"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "already running: $(cat "$PIDF")"; exit 1; fi
rm -f "$STOPF"; echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT
export DASH_DIR
cycle() {
  date -u +"%F %T cycle" >> "$LOG"
  ( cd "$DASH_DIR" && [ -f tools/events_auto.py ] && python3 tools/events_auto.py ) >> "$LOG" 2>&1
  ( cd "$DASH_DIR" && python3 tools/update_dash.py ) >> "$LOG" 2>&1 || { echo "update_dash failed" >> "$LOG"; return; }
  for f in runs mem tps acc outages events findings submissions ref_b32; do [ -f "$DASH_DIR/$f.json" ] && cp "$DASH_DIR/$f.json" "$T/$f.json"; done
  cp "$DASH_DIR"/frozen_*.json "$T/" 2>/dev/null
  python3 "$T/tools/publish_list.py" --update >> "$LOG" 2>&1
  tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
}
while [ ! -f "$STOPF" ]; do
  cycle
  i=0; while [ "$i" -lt "$INTERVAL" ] && [ ! -f "$STOPF" ]; do sleep 1; i=$((i+1)); done
done
rm -f "$STOPF"
