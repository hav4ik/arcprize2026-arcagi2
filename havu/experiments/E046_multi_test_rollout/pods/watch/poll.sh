#!/bin/bash
# poll every pod in pods.txt; raw output per run
D=$(dirname $0)
while read -r id port host run; do
  [ -z "$id" ] && continue
  echo "=== $id $run $(date -u +%T)"
  timeout 60 ssh -o ConnectTimeout=15 -o ServerAliveInterval=10 -o StrictHostKeyChecking=accept-new -p $port root@$host "bash -s $run" < $D/remote.sh 2>&1 || echo "SSH_FAIL rc=$?"
done < $D/pods.txt
