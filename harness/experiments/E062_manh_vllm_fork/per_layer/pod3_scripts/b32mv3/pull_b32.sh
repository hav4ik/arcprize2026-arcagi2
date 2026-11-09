#!/bin/bash
# Copy the B32-M v3 replica's logs from pod 3 (no traces unless "full"), then print b32_report.py and the per-window
# solver/server report (windows.py, 30-min buckets). Read-only on the pod. Usage: pull_b32.sh [full]
S=$(dirname "$0"); read IP PORT < $S/pod3_ssh.txt; D=$S/b32copy; mkdir -p $D/out
EX='--exclude=traces'; [ "${1:-}" = full ] && EX=''
timeout 600 ssh -o BatchMode=yes -p $PORT root@$IP "cd /kaggle/working && tar czf - $EX dsv41_logs; " > $D/logs.tgz && rm -rf $D/dsv41_logs && tar xzf $D/logs.tgz -C $D
for f in mon.jsonl mem.log run.log metrics_scrape.txt gpu_hold.log setup.log verify_e128.log; do timeout 120 scp -q -P $PORT root@$IP:/root/e066/b32mv3/$f $D/ 2>/dev/null; done
[ "${1:-}" = full ] && timeout 120 scp -q -P $PORT root@$IP:/root/e066/b32mv3/nb_rerun_out.ipynb $D/ 2>/dev/null
rm -rf $D/out/dsv41_logs; ln -s ../dsv41_logs $D/out/dsv41_logs
python3 $S/b32_trends.py $D 2
cat $D/dsv41_logs/status.json 2>/dev/null; echo
python3 $S/windows.py $D $D/metrics_scrape.txt $S/run2_final/evalx2t_s2026_solutions.json 1800 2>&1 | tail -26
