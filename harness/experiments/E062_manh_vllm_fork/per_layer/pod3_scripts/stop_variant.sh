#!/bin/bash
# stop one variant by PID: the solver (local_run.py, found by its cwd = the run dir) and Manh's server (launch.json pid)
N=$1; D=/root/e066/$N; R=/root/runs/${N}_x80
for p in /proc/[0-9]*; do
  [ "$(readlink $p/cwd 2>/dev/null)" = "$R" ] || continue
  pid=${p#/proc/}; pg=$(ps -o pgid= -p $pid | tr -d ' ')
  echo "solver pid $pid pgid $pg"; kill -TERM -- -$pg 2>/dev/null
done
sleep 10
SP=$(python3 -c "import json;print(json.load(open('$D/cell2m.json'))['MANH_SERVER_PID'])")
echo "server pid $SP"; kill -TERM -- -$SP 2>/dev/null || kill -TERM $SP
for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "still on GPU; SIGKILL"; kill -KILL -- -$SP 2>/dev/null; sleep 10; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader; echo "stopped $N $(date -u +%T)" >> $D/status
