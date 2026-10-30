#!/usr/bin/env bash
# Stop a server started by run_server.sh (only our own process group).
SCR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
LOGDIR="${LOGDIR:-$SCR/memory/logs}"
NAME="${1:?name}"
PG="$(cat "$LOGDIR/$NAME.pid")"
kill -TERM -- "-$PG" 2>/dev/null || true
for i in $(seq 1 60); do
  kill -0 -- "-$PG" 2>/dev/null || { echo "stopped $NAME"; exit 0; }
  sleep 1
done
kill -KILL -- "-$PG" 2>/dev/null || true
echo "killed $NAME"
