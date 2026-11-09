#!/bin/bash
# E042: per-GPU memory/util every 2 s into runs/<name>/gpu.csv until the server pid exits.
R=/root/dsv41/runs/$1; PID=$(cat $R/server.pid)
while kill -0 $PID 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw --format=csv,noheader >> $R/gpu.csv
  echo "$(date -u +%FT%T) $(free -m | awk '/Mem:/{print $3}') $(grep -E '^(VmHWM|VmRSS)' /proc/$PID/status 2>/dev/null | tr -s ' ' | tr '\n' ' ')" >> $R/host_mem.log
  sleep 2
done
