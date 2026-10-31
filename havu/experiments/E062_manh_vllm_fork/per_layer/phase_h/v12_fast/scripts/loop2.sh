#!/bin/bash
# sync every 5 min until data/STOP exists
cd "$(dirname "$0")/.."
while [ ! -e data/STOP ]; do
  out=$(REPLICA_RUN=v12fast2_rerun REPLICA_DATA=$PWD/data bash scripts/sync.sh 2>&1 | tail -1); echo "$(date -u +%T) ${out:0:200}" >> data/sync_loop.log
  sleep 300
done
