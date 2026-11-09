#!/bin/bash
# sync_queue.sh <pod name> <ssh port> <ip> [interval s=300]: loop (no agent turns). Every interval, rsync each queued run's data of
# that pod into runs/<label>/ (nbruns files, dsv41_logs of kwork_<run>, submission.json, score.json) plus runs/<pod>/queue/.
# Stop: touch runs/sync_<pod>.stop. Start detached: setsid nohup bash sync_queue.sh pod2 18070 1.2.3.4 > runs/sync_pod2.log 2>&1 &
POD=$1; PORT=$2; IP=$3; EVERY=${4:-300}; H=$(cd "$(dirname "$0")" && pwd); O=$H/runs; mkdir -p $O/$POD/queue
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$IP"; RS="rsync -a --timeout=120 -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p $PORT'"
while [ ! -e $O/sync_$POD.stop ]; do
  t=$(date +%s)
  eval $RS root@$IP:/root/queue/index.tsv root@$IP:/root/queue/logs/runner.log $O/$POD/queue/ 2>/dev/null
  $SSH 'cd /root/queue; for d in pending running done failed; do echo "== $d"; ls -1 $d; done; echo "== disk"; df -h / | tail -1; echo "== gpu"; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr "\n" " "' > $O/$POD/queue/state.txt 2>/dev/null
  while IFS=$'\t' read -r label run wl start; do
    [ -z "$run" ] && continue; D=$O/$label; mkdir -p $D/dsv41_logs
    eval $RS --include="'score.json'" --include="'score.txt'" --include="'END_REASON'" --include="'variant.json'" --include="'job.env'" --include="'nb.timeline'" --include="'notebook.sha256'" \
      --include="'samp.jsonl'" --include="'samp.alerts'" --include="'samp.tps.tsv'" --include="'tree.txt'" --include="'nb.ipynb'" --exclude="'*'" root@$IP:/root/nbruns/$run/ $D/ 2>/dev/null
    eval $RS --include="'attempts.jsonl'" --include="'metrics.jsonl'" --include="'solver.log'" --include="'status.json'" --include="'supervisor.json'" --include="'stage.log'" --exclude="'*'" root@$IP:/root/kwork_$run/dsv41_logs/ $D/dsv41_logs/ 2>/dev/null
    eval $RS root@$IP:/root/kwork_$run/submission.json $D/ 2>/dev/null
  done < $O/$POD/queue/index.tsv
  s=$((EVERY - ($(date +%s) - t))); [ $s -lt 10 ] && s=10; sleep $s
done
