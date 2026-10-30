#!/bin/bash
# B32-M v3 replica watcher (local, read-only on the pod): every 180 s reads the newest mon.jsonl record and the logs.
# Exits with FLAG at once on: Kaggle-eq host RAM crossing the next level (155 WARN / 165 ALERT / 173 CRIT) above the
# given one, min GPU free < 1024 MiB (WARN) / < 512 (ALERT) below the given one, a new CUDA OOM / engine death / server
# death or restart line, a burst of >= 5 new error records, the notebook ending, more vLLM allocator retries, any vLLM
# OOM count, or cap - peak < 0.26 GiB on any rank (gpumem_now.py). At <HH:MM> UTC prints HOURLY_DUE.
# Usage: watch_b32.sh <HH:MM> <ram level already reported: 0|155|165|173> <vram level already reported: 0|1024|512>
S=$(dirname "$0"); read IP PORT < $S/pod3_ssh.txt
end=$(date -u -d "$1 UTC" +%s); [ $end -lt $(date +%s) ] && end=$((end + 86400)); RL=$2; VL=$3
q() { timeout 60 ssh -o BatchMode=yes -p $PORT root@$IP 'D=/root/e066/b32mv3; W=/kaggle/working/dsv41_logs
  tail -1 $D/mon.jsonl | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); g=[x for x in r.get(\"gpu\",[]) if isinstance(x,list)]; print(r.get(\"kaggle_eq_gib\",0), int(min([x[2] for x in g] or [99999])))"
  c() { echo "${1:-0}"; }
  c $(cat $W/server*.log $W/*/server*.log 2>/dev/null | grep -acE "out of memory|OutOfMemoryError|EngineDeadError|EngineCore encountered|Traceback|WorkerProc hit an exception|RPC call to execute_model timed out")
  c $(grep -acE "server died|restarting the server|server restart|restart [0-9]" $W/solver.log 2>/dev/null)
  c $(grep -ac "\"error\"" $W/attempts.jsonl 2>/dev/null)
  c $(grep -ac "nb end" $D/run.log); /usr/bin/python3 $D/gpumem_now.py' 2>/dev/null | tr '\n' ' '; }
set -- $(q); e0=${3:-0}; d0=${4:-0}; rt0=${7:-0}; hist=(${5:-0} ${5:-0} ${5:-0})  # error records 3 checks (~9 min) back
while [ $(date +%s) -lt $end ]; do
  sleep 180; o=$(q); set -- $o; [ -z "$1" ] && continue
  k=${1%.*}; v=$2; lvl=0; for x in 155 165 173; do [ "$k" -ge $x ] && lvl=$x; done
  vl=0; [ "$v" -lt 1024 ] && vl=1024; [ "$v" -lt 512 ] && vl=512
  if [ $lvl -gt $RL ] || { [ $vl -ne 0 ] && { [ $VL -eq 0 ] || [ $vl -lt $VL ]; }; } || [ "${3:-0}" != "$e0" ] || [ "${4:-0}" != "$d0" ] || [ $(( ${5:-0} - ${hist[0]} )) -ge 5 ] || [ "${6:-0}" != 0 ] || [ "${7:-0}" -gt "$rt0" ] || [ "${8:-0}" != 0 ] || python3 -c "import sys; sys.exit(0 if float('${9:-9}') < 0.26 else 1)"; then
    date -u +%T; echo "FLAG kaggle_eq=$1 GiB (level $lvl, reported $RL) min_gpu_free=$v MiB (level $vl, reported $VL) server_err_lines=$3 (was $e0) solver_death_lines=$4 (was $d0) error_records=$5 (~9 min ago ${hist[0]}) nb_end=$6 alloc_retries_max=$7 (was $rt0) ooms=$8 min_cap_minus_peak=$9"; exit 0; fi
  hist=(${hist[1]} ${hist[2]} ${5:-0})
done
date -u +%T; echo "HOURLY_DUE $(q)"
