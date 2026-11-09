#!/bin/bash
# Copy the notebook run's solver logs (no traces), the metrics scrape, memprobe and mem.log from pod 3, then print the
# per-window report (windows.py) and the latest status. Read-only on the pod. Usage: pull_nb.sh [bucket_s=300]
S=$(dirname "$0"); read IP PORT < $S/pod3_ssh.txt; D=$S/nbcopy; mkdir -p $D/out/dsv41_logs
timeout 120 ssh -o BatchMode=yes -p $PORT root@$IP 'cd /kaggle/working/dsv41_logs && tar czf - --exclude=traces . ; ' > $D/logs.tgz && tar xzf $D/logs.tgz -C $D/out/dsv41_logs
for f in metrics_scrape.txt memprobe.jsonl mem.log run.log; do timeout 60 scp -q -P $PORT root@$IP:/root/e066/nbrun/$f $D/ ; done
timeout 60 scp -q -P $PORT root@$IP:/root/data/evalx2t_s2026_solutions.json $D/ 2>/dev/null
python3 $S/windows.py $D $D/metrics_scrape.txt $D/evalx2t_s2026_solutions.json ${1:-300}
echo; cat $D/out/dsv41_logs/status.json; echo; tail -3 $D/out/dsv41_logs/solver.log | cut -c1-300
