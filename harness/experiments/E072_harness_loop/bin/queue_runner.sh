#!/bin/bash
# queue_runner.sh: detached loop. Waits until no replica notebook runs and the GPUs are empty, then takes the oldest
# /root/queue/pending/*.job (or a filler: V12 incumbent replicate on eval120), runs it via run_job.sh, moves it to done/.
# Start: setsid nohup bash /root/queue/bin/queue_runner.sh > /root/queue/logs/runner.out 2>&1 < /dev/null &  (see bin/start_runner.sh)
# Stop: touch /root/queue/runner.stop (finishes after the current job's wait; never kills a run). PID in /root/queue/runner.pid.
Q=/root/queue; B=$Q/bin; cd $Q; mkdir -p pending running done failed logs
echo $$ > $Q/runner.pid
log() { echo "$(date -u +%FT%TZ) [runner] $*" >> $Q/logs/runner.log; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{if ($1>m) m=$1} END {print m+0}'; }
busy() { pgrep -f "e062pl/run_nb.py" > /dev/null || [ "$(gpu_used)" -gt 2000 ]; }
log "runner started pid $$"
fails=0; n=0
while [ ! -e $Q/runner.stop ]; do
  ok=0; for i in 1 2 3; do busy && { ok=1; break; }; sleep 15; done
  [ $ok -eq 1 ] && { sleep 30; continue; }
  j=$(ls -1tr pending/*.job 2>/dev/null | head -1)
  if [ -z "$j" ]; then
    n=$((n+1)); lab=filler_v12_$(date -u +%m%d%H%M); j=pending/${lab}.job
    sed "s/@LABEL@/$lab/" $Q/filler.job.tpl > $j; log "queue empty: filler $lab"
  fi
  base=$(basename $j); mv $j running/$base; t=$(date +%s)
  bash $B/run_job.sh running/$base >> $Q/logs/run_job.out 2>&1; rc=$?
  if [ $rc -eq 0 ]; then mv running/$base done/$base; fails=0; else mv running/$base failed/$base; fails=$((fails+1)); log "job $base FAILED rc=$rc"; fi
  # failure backoff only (a failing queue must not spin or burn the filler cycle): 2 min, then 10 min after 3 in a row
  [ $rc -ne 0 ] && sleep $([ $fails -ge 3 ] && echo 600 || echo 120)
  # back-to-back launches: GPU memory needs a moment to clear
  sleep 20
done
log "runner stop file seen: exiting"; rm -f $Q/runner.pid
