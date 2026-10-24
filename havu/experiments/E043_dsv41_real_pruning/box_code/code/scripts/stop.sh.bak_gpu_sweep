#!/bin/bash
# E043: stop a server started by serve.sh: its session, then any process still holding its GPUs (orphaned workers).
# Usage: stop.sh <run_name> <gpu_list e.g. 0,1,2,3>
R=/workspace/dsv4/runs/$1
[ -s $R/server.pid ] && kill -- -$(cat $R/server.pid) 2>/dev/null
sleep 8
for i in $(seq 1 10); do
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $2 | sort -u)
  [ -z "$pids" ] && break
  kill $pids 2>/dev/null; sleep 4; kill -9 $pids 2>/dev/null; sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $2 | tr "\n" " "; echo
