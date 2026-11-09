#!/bin/bash
# rep_dash_loop.sh: every 300 s run rep_dash_sync.sh (dashboard-format data of one replica) until runs/rep_dash_<run>.stop or the run is over (> 12.2 Kaggle h).
# env: POD_PORT POD_IP REPLICA_RUN REPLICA_DATA
H=$(cd "$(dirname "$0")" && pwd); STOP=$H/runs/rep_dash_$REPLICA_RUN.stop; S=$REPLICA_DATA/status.json
while [ ! -e $STOP ]; do
  t=$(date +%s); out=$(bash $H/rep_dash_sync.sh 2>&1 | tail -1); echo "$(date -u +%T) ${out:0:200}"
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$S')).get('kaggle_h', 0) < 12.2 else 1)" 2>/dev/null || { [ -s $S ] && echo "run over?"; }
  sleep $(( 300 - ($(date +%s) - t) > 10 ? 300 - ($(date +%s) - t) : 10 ))
done
