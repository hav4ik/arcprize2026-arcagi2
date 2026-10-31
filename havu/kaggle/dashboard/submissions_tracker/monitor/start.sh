#!/bin/bash
set -euo pipefail
MONITOR_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$MONITOR_DIR/../../../.." && pwd)
mkdir -p "$MONITOR_DIR/state"
if [ -f "$MONITOR_DIR/state/server.pid" ]; then
  MONITOR_PID=$(cat "$MONITOR_DIR/state/server.pid")
  if [ -r "/proc/$MONITOR_PID/cmdline" ] && tr '\0' ' ' < "/proc/$MONITOR_PID/cmdline" | rg -q "$MONITOR_DIR/server.py"; then
    echo "ARC monitor already running (PID $MONITOR_PID)"
    exit 0
  fi
fi
nohup setsid "$REPO_DIR/.venv/bin/python" -u "$MONITOR_DIR/server.py" --port 8765 \
  > "$MONITOR_DIR/state/server.log" 2>&1 < /dev/null &
echo "Started ARC monitor (PID $!): http://localhost:8765/local.html"
