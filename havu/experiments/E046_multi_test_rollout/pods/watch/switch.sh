#!/bin/bash
# usage: switch.sh <port> <host> <oldrun> <newrun> [ENV=VALUE ...]: stop old run cleanly, start new, wait for server up
D=$(dirname $0); PORT=$1; HOST=$2; OLD=$3; NEW=$4; shift 4
echo "$(date -u +%T) switch $HOST:$PORT $OLD -> $NEW $*"
timeout 600 ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -p $PORT root@$HOST "bash -s $OLD $NEW $*" < $D/remote_switch.sh 2>&1
for i in $(seq 1 60); do
  up=$(timeout 30 ssh -n -o ConnectTimeout=15 -p $PORT root@$HOST "grep -E 'server up|server exited|server failed|server not up' /root/runs/$NEW/run.log | tail -1" 2>/dev/null)
  [ -n "$up" ] && { echo "$(date -u +%T) SERVERLINE $NEW $up"; break; }
  sleep 20
done
