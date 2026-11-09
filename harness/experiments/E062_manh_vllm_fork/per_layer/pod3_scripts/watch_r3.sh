#!/bin/bash
# Watch run 3 until <HH:MM UTC>: flag at once a supervisor state/start change, a new "server died", papermill exit,
# a new error record, or a GPU with < 1024 MiB free; at the deadline print the hourly report. Usage: watch_r3.sh <HH:MM>
S=$(dirname "$0"); read IP PORT < $S/pod3_ssh.txt; end=$(date -u -d "$1 UTC" +%s); [ $end -lt $(date +%s) ] && end=$((end + 86400))
e0=$(timeout 30 ssh -o BatchMode=yes -p $PORT root@$IP 'grep -c "\"error\"" /kaggle/working/dsv41_logs/attempts.jsonl')
while [ $(date +%s) -lt $end ]; do
  o=$(timeout 30 ssh -o BatchMode=yes -p $PORT root@$IP 'python3 -c "import json;d=json.load(open(\"/kaggle/working/dsv41_logs/e066_supervisor.json\"));print(d[\"state\"],d[\"start\"])"; grep -ac "server died" /kaggle/working/dsv41_logs/solver.log; pgrep -fc "papermill --kernel"; grep -c "\"error\"" /kaggle/working/dsv41_logs/attempts.jsonl; nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1' 2>/dev/null | tr "\n" " ")
  set -- $o
  if [ -n "$1" ] && { [ "$1" != "up" ] || [ "${2:-1}" != "1" ] || [ "${3:-1}" != "1" ] || [ "${4:-1}" = "0" ] || [ "${5:-0}" != "${e0:-0}" ] || [ "${6:-9999}" -lt 1024 ]; }; then
    date -u +%T; echo "FLAG state start deaths papermill errors(was $e0) minfree: $o"; exit 0; fi
  sleep 60
done
date -u +%T; echo HOURLY_DUE; $S/pull_r3.sh 600 2>&1 | tail -34; python3 $S/vram_series.py $S/r3copy/sampler.jsonl 600 | tail -7
