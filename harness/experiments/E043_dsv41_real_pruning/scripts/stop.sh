#!/bin/bash
# E043: stop a server started by serve.sh: kill its session (serve.sh starts it with setsid, server.pid = session id),
# then any process still holding its GPUs *in that session* (orphaned workers). Other jobs on the same GPUs (GPTQ
# workers, calibration replays) are left alone; they used to be killed here.
# Usage: stop.sh <run_name> <gpu_list e.g. 0,1,2,3>
R=/workspace/dsv4/runs/$1
SID=$(cat $R/server.pid 2>/dev/null)
[ -z "$SID" ] && { echo "stop.sh: no $R/server.pid"; exit 1; }
kill -- -$SID 2>/dev/null
sleep 8
for i in $(seq 1 10); do
  pids=""
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $2 | sort -u); do
    [ "$(ps -o sid= -p $p 2>/dev/null | tr -d ' ')" = "$SID" ] && pids="$pids $p"
  done
  [ -z "$pids" ] && break
  kill $pids 2>/dev/null; sleep 4; kill -9 $pids 2>/dev/null; sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $2 | tr "\n" " "; echo
