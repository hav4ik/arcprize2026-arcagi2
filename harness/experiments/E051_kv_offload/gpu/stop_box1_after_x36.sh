#!/usr/bin/env bash
# Box 1 wind-down (user: scale the 4090s down): once late1_on28p's hour-8 x36 replay has finished, stop the chain
# (chain_box1b.sh -> run_capacity3.sh late1), the queued late1b waiter, the on28p server and its metrics logger, by
# process group only.
F=/root/e051/runs/late1.progress
R=/root/e051/runs
until grep -q "late1_on28p: replay C=36 D=20 S=1.25 T=0.3 M=30000 rc=" "$F" 2>/dev/null; do sleep 10; done
sleep 3
# Explicit process groups started by E051 (recorded at launch): chain_box1b.sh (incl. run_capacity3.sh late1 and its
# replay) = 73420; the queued late1b waiter = 88020.
for pg in 73420 88020; do
  kill -- -"$pg" 2>/dev/null
done
D=$R/late1_on28p
[ -f "$D/server.pgid" ] && kill -- -"$(cat "$D/server.pgid")" 2>/dev/null
[ -f "$D/metrics_logger.pid" ] && kill "$(cat "$D/metrics_logger.pid")" 2>/dev/null
for i in $(seq 1 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 100 ] && break
  sleep 2
done
echo "$(date -u +%FT%TZ) box 1 wind-down: chain, waiter, server stopped; GPUs $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')" >> "$F"
