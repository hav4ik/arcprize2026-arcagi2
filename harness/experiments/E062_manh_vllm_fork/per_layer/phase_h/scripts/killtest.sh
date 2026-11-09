#!/bin/bash
# TP-worker kill test for the replica: SIGKILL one VLLM::Worker_TP (by PID), then sample every 2 s for <secs>:
# host RAM (used / MemAvailable / AnonPages), vLLM process count, GPU memory, the cell-2s supervisor status file,
# the solver's server-status lines. Usage: killtest.sh <rank 0-3> <secs> <log>
RANK=$1; SECS=$2; LOG=$3; W=/kaggle/working; L=$W/dsv41_logs
pid=$(ps -eo pid,args | awk -v r="VLLM::Worker_TP$RANK" '$2 == r {print $1}' | head -1)
{ echo "== pre-kill $(date -u +%T): target Worker_TP$RANK pid $pid"
  ps -eo pid,ppid,pgid,rss,args | grep -E "[V]LLM::|[v]llm.entrypoints" | cut -c1-120
  free -g | head -2; grep -E "AnonPages|MemAvailable|Shmem:" /proc/meminfo | tr -s " " | tr "\n" " "; echo
  echo "supervisor: $(cat $L/e066_supervisor.json 2>/dev/null)"
  echo "attempts: $(wc -l < $L/attempts.jsonl) records; last solver line: $(tail -1 $L/solver.log | cut -c1-200)"
  [ -z "$pid" ] && { echo "no worker pid found"; exit 1; }
  kill -KILL $pid; echo "== KILLED pid $pid (Worker_TP$RANK) at $(date -u +%T.%N | cut -c1-12)"
  t0=$(date +%s); maxu=0
  while [ $(( $(date +%s) - t0 )) -lt $SECS ]; do
    u=$(free -g | awk 'NR==2{print $3}'); [ "$u" -gt "$maxu" ] && maxu=$u
    echo "$(date -u +%T) +$(( $(date +%s) - t0 ))s used ${u}G anon $(awk '/AnonPages/{printf "%.1f", $2/1048576}' /proc/meminfo)G vllm_procs $(ps -eo args | grep -cE '^VLLM::|vllm.entrypoints') gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/) sup $(tr -d '\n ' < $L/e066_supervisor.json 2>/dev/null | cut -c1-160)"
    sleep 2
  done
  echo "== end $(date -u +%T): max host used ${maxu}G"
  echo "supervisor: $(cat $L/e066_supervisor.json 2>/dev/null)"
  grep -anE "server-side|re-?queue|status|restart|supervisor|not up|waiting" $L/solver.log | tail -30 | cut -c1-250
} >> $LOG 2>&1
