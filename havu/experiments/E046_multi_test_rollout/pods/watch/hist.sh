#!/bin/bash
# fetch every server/gate line + mem.log of each run in pods.txt into hist_<run>.txt / mem_<run>.log
D=$(dirname $0)
while read -r id port host run; do
  [ -z "$id" ] && continue
  timeout 60 ssh -n -o ConnectTimeout=15 -p $port root@$host "grep -E '\] server: |KV gate:|server up|server died|server start|done at|local score' /root/runs/$run/run.log" > $D/hist_$run.txt 2>/dev/null
  timeout 60 scp -q -o ConnectTimeout=15 -P $port root@$host:/root/runs/$run/mem.log $D/mem_$run.log 2>/dev/null
done < $D/pods.txt
