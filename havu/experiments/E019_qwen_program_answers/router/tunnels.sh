#!/usr/bin/env bash
# SSH tunnels (restart loops) from this machine to each Vast box's internal SGLang port 127.0.0.1:18000.
# The router talks to http://127.0.0.1:<local port> with no token: this bypasses the Vast Caddy auth edge,
# whose /health requires a bearer token that sglang-router's health/detection probes do not send.
#   start | stop | status
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
# Tunnels come from tunnels.conf ("name local_port ssh_host ssh_port"), written by register_worker.py.
TUNNELS=""
[ -f "$DIR/tunnels.conf" ] && TUNNELS="$(grep -v '^#' "$DIR/tunnels.conf")"

loop() {  # name lport host port
  local name=$1 lport=$2 host=$3 port=$4
  echo $$ > "$DIR/tunnel_$name.pid"
  while true; do
    [ -f "$DIR/stop_tunnels" ] && exit 0
    echo "$(date -u +%FT%TZ) [$name] connecting $host:$port -> 127.0.0.1:$lport" >> "$DIR/tunnel_$name.log"
    ssh -N -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes -o TCPKeepAlive=yes \
      -L "127.0.0.1:$lport:127.0.0.1:18000" -p "$port" "root@$host" >> "$DIR/tunnel_$name.log" 2>&1
    echo "$(date -u +%FT%TZ) [$name] ssh exited $?; retry in 5 s" >> "$DIR/tunnel_$name.log"
    sleep 5
  done
}

case "${1:-start}" in
  start)
    rm -f "$DIR/stop_tunnels"
    echo "$TUNNELS" | while read -r name lport host port; do
      [ -z "$name" ] && continue
      p=$(cat "$DIR/tunnel_$name.pid" 2>/dev/null)
      if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "$name already running ($p)"; continue; fi
      nohup "$0" loop "$name" "$lport" "$host" "$port" > /dev/null 2>&1 &
      echo "$name tunnel loop started (127.0.0.1:$lport -> $host:$port)"
    done ;;
  loop) shift; loop "$@" ;;
  stop)
    touch "$DIR/stop_tunnels"
    for f in "$DIR"/tunnel_*.pid; do
      [ -f "$f" ] || continue; p=$(cat "$f"); pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; rm -f "$f"
    done
    echo "tunnels stopped" ;;
  status)
    echo "$TUNNELS" | while read -r name lport host port; do
      [ -z "$name" ] && continue
      p=$(cat "$DIR/tunnel_$name.pid" 2>/dev/null)
      st=$( [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo up || echo down )
      code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$lport/health")
      echo "$name loop=$st 127.0.0.1:$lport /health=$code"
    done ;;
esac
