#!/bin/bash
S=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/pod3
read IP PORT < $S/pod3_ssh.txt
last=""
for i in $(seq 1 240); do
  st=$(timeout 30 ssh -o BatchMode=yes -p $PORT root@$IP 'tail -1 /root/chain.log; du -s --block-size=1G /root/models/c1 2>/dev/null | cut -f1; grep -c "^GOT" /root/dl_wheels.log 2>/dev/null' 2>/dev/null | tr '\n' ' ')
  [ "$st" != "$last" ] && echo "$(date -u +%T) $st" && last="$st"
  case "$st" in *CHAIN_DONE*|*CHAIN_FAIL*) exit 0;; esac
  sleep 60
done
