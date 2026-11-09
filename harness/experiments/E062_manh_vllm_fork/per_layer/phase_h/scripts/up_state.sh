#!/bin/bash
# One state line for run $1: supervisor state/start, "server up" count in solver.log, NOTEBOOK_DONE count, error lines.
RUN=$1; W=/root/kwork_$RUN/dsv41_logs
s=$(python3 -c "import json; d=json.load(open('$W/supervisor.json')); print(d.get('state'), d.get('start'))" 2>/dev/null)
u=$(grep -c "server up" $W/solver.log 2>/dev/null); n=$(grep -c NOTEBOOK_DONE /root/nbruns/$RUN/nb.timeline 2>/dev/null)
e=$(grep -acE "CUDA out of memory|Traceback|EngineDeadError" $W/server/server.log 2>/dev/null)
echo "${s:-none} up=${u:-0} nbdone=${n:-0} err=${e:-0}"
