#!/bin/bash
# Dashboard data for a pod 4 run (overwritten each time, atomic renames; read-only on the pod):
#   data/gpumem.log, ram.tsv, tps.tsv, status.json (dash.py on the pod), attempts.jsonl, solver.log (copies).
# Usage: export_dash.sh <run> [verdict]     (data dir: pod4_runs/<run>/data)
S=$(cd "$(dirname "$0")" && pwd); read IP PORT < $S/pod4_ssh.txt; RUN=$1
O=$S/../pod4_runs/$RUN/data; mkdir -p $O
SSH=(timeout 120 ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$IP)
for w in gpumem ram tps; do
  f=$w.tsv; [ $w = gpumem ] && f=gpumem.log
  "${SSH[@]}" "/usr/bin/python3 -I /root/p4/scripts/dash.py $RUN $w" > $O/$f.tmp && [ -s $O/$f.tmp ] && mv $O/$f.tmp $O/$f
done
"${SSH[@]}" "/usr/bin/python3 -I /root/p4/scripts/dash.py $RUN status $(printf '%q' "${2:-}")" > $O/status.json.tmp && [ -s $O/status.json.tmp ] && mv $O/status.json.tmp $O/status.json
for f in attempts.jsonl solver.log; do
  timeout 120 scp -q -i $HOME/.ssh/id_ed25519 -P $PORT root@$IP:/root/kwork_$RUN/dsv41_logs/$f $O/$f.tmp 2>/dev/null && mv $O/$f.tmp $O/$f
done
rm -f $O/*.tmp
echo "$(date -u +%T) dash $RUN: $(cat $O/status.json 2>/dev/null | head -c 600)"
