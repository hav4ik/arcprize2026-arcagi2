#!/bin/bash
S=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/pod3
read IP PORT < $S/pod3_ssh.txt; N=$1; last=""
for i in $(seq 1 120); do
  st=$(timeout 30 ssh -o BatchMode=yes -p $PORT root@$IP "cat /root/e066/$N/status; tail -1 /root/chain.log" < /dev/null 2>/dev/null)
  [ "$st" != "$last" ] && { echo "== $(date -u +%T)"; echo "$st"; last="$st"; }
  case "$st" in *LAUNCHED*|*FAIL*) exit 0;; esac
  sleep 30
done
