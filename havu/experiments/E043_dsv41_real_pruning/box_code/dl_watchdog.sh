#!/bin/bash
# Download the pinned checkpoint; restart hf download whenever no bytes land for 180 s.
REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
DIR=/workspace/dsv4/models/DeepSeek-V4.1-Flash-1017
LOG=/workspace/dsv4/logs/download.log
for attempt in $(seq 1 30); do
  echo "=== attempt $attempt $(date -u +%T)" >> $LOG
  /opt/sglang/bin/hf download $REPO --revision $REV --local-dir $DIR --max-workers 8 >> $LOG 2>&1 &
  pid=$!
  last=$(du -sb $DIR | cut -f1); idle=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    now=$(du -sb $DIR | cut -f1)
    if [ "$now" -gt "$last" ]; then idle=0; last=$now; else idle=$((idle+30)); fi
    if [ $idle -ge 180 ]; then echo "stall, killing $pid $(date -u +%T)" >> $LOG; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; break; fi
  done
  wait $pid; rc=$?
  n=$(ls $DIR/*.safetensors 2>/dev/null | wc -l)
  echo "rc=$rc shards=$n $(date -u +%T)" >> $LOG
  if [ $rc -eq 0 ] && [ $n -eq 48 ]; then echo DONE >> $LOG; exit 0; fi
done
