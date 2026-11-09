#!/bin/bash
# E042: stop a server started by serve.sh: its whole session (sid = server.pid), never by name.
R=/root/dsv41/runs/$1; SID=$(cat $R/server.pid 2>/dev/null) || { echo "no pid"; exit 0; }
pgrep -s $SID > /dev/null || { echo "not running"; exit 0; }
pkill -TERM -s $SID
for i in $(seq 30); do pgrep -s $SID > /dev/null || break; sleep 2; done
pkill -KILL -s $SID 2>/dev/null
for i in $(seq 15); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 2; done
echo "stopped $1 (sid $SID)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
