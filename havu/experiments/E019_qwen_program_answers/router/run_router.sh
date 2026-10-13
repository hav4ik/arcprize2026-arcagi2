#!/usr/bin/env bash
# Supervisor loop for the local SGLang Model Gateway (smg from sglang-router 0.3.2, E019). Restarts the gateway
# if it exits and then re-registers every worker in workers.json (register_worker.py --restore). Workers are
# local SSH-tunnel ports (tunnels.sh); no token is needed on them (see README "Auth").
#   start:   experiments/E019_qwen_program_answers/router/run_router.sh start
#   stop:    experiments/E019_qwen_program_answers/router/run_router.sh stop
#   status:  experiments/E019_qwen_program_answers/router/run_router.sh status
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DIR="$ROOT/experiments/E019_qwen_program_answers/router"
PORT="${ROUTER_PORT:-30100}"
PROM_PORT="${ROUTER_PROM_PORT:-29100}"
PY="$ROOT/.venv/bin/python"

router_cmd() {
  exec "$ROOT/.venv/bin/smg" launch \
    --host 127.0.0.1 --port "$PORT" \
    --policy manual --assignment-mode min_load --max-idle-secs 14400 \
    --request-timeout-secs 3600 \
    --retry-max-retries 3 \
    --cb-failure-threshold 10 --cb-timeout-duration-secs 60 \
    --health-check-interval-secs 30 --health-failure-threshold 3 --health-check-timeout-secs 10 \
    --prometheus-host 127.0.0.1 --prometheus-port "$PROM_PORT" \
    --log-level info
}

loop() {
  echo $$ > "$DIR/supervisor.pid"
  while true; do
    echo "$(date -u +%FT%TZ) starting router on 127.0.0.1:$PORT" >> "$DIR/supervisor.log"
    ( router_cmd ) >> "$DIR/router.log" 2>&1 &
    rpid=$!
    echo "$rpid" > "$DIR/router.pid"
    sleep 3
    "$PY" "$DIR/register_worker.py" --gateway "http://127.0.0.1:$PORT" --restore >> "$DIR/supervisor.log" 2>&1
    wait "$rpid"
    echo "$(date -u +%FT%TZ) router exited with $?; restarting in 5 s" >> "$DIR/supervisor.log"
    [ -f "$DIR/stop" ] && { rm -f "$DIR/stop"; echo "stop flag; exiting" >> "$DIR/supervisor.log"; exit 0; }
    sleep 5
  done
}

case "${1:-start}" in
  start)
    rm -f "$DIR/stop"
    nohup "$0" loop > /dev/null 2>&1 &
    echo "supervisor started (pid $!); logs in $DIR/{supervisor,router}.log" ;;
  loop) loop ;;
  stop)
    touch "$DIR/stop"
    [ -f "$DIR/supervisor.pid" ] && kill "$(cat "$DIR/supervisor.pid")" 2>/dev/null
    [ -f "$DIR/router.pid" ] && kill "$(cat "$DIR/router.pid")" 2>/dev/null
    rm -f "$DIR/stop"; echo "stopped" ;;
  status)
    for f in supervisor router; do
      p=$(cat "$DIR/$f.pid" 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "$f: running ($p)"; else echo "$f: not running"; fi
    done
    curl -s -m 5 "http://127.0.0.1:$PORT/workers" | "$PY" -c 'import json,sys; [print(w["url"], "healthy" if w["is_healthy"] else "UNHEALTHY", "load", w["load"]) for w in json.load(sys.stdin)["workers"]]' ;;
esac
