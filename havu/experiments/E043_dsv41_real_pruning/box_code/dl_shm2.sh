#!/bin/bash
# Download the pinned checkpoint into /dev/shm; restart hf only if the NIC receives < 50 MB in 300 s.
REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
REV=1bae0e1088993b9b82de08f59de5be5a34e42170
DIR=/dev/shm/dsv4/DeepSeek-V4.1-Flash-1017
LOG=/workspace/dsv4/logs/download.log
rx() { awk '/eth0/{print $2}' /proc/net/dev; }
pid=${1:-}
for attempt in $(seq 1 30); do
  if [ -z "$pid" ]; then
    echo "=== attempt $attempt $(date -u +%T)" >> $LOG
    HF_XET_HIGH_PERFORMANCE=1 /opt/sglang/bin/hf download $REPO --revision $REV --local-dir $DIR --max-workers 16 >> $LOG 2>&1 &
    pid=$!
  fi
  last=$(rx); idle=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    now=$(rx)
    if [ $((now-last)) -gt 50000000 ]; then idle=0; last=$now; else idle=$((idle+30)); fi
    if [ $idle -ge 300 ]; then echo "stall, killing $pid $(date -u +%T)" >> $LOG; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; break; fi
  done
  wait $pid 2>/dev/null; rc=$?
  n=$(ls $DIR/*.safetensors 2>/dev/null | wc -l)
  echo "rc=$rc shards=$n $(date -u +%T)" >> $LOG
  pid=""
  if [ $n -eq 48 ] && [ -z "$(ls $DIR/.cache/huggingface/download/*.incomplete 2>/dev/null)" ]; then echo DONE >> $LOG; exit 0; fi
done
