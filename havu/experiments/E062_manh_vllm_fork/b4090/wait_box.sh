#!/bin/bash
# Poll Vast instance 58063361 until SSH answers (or 09:00 UTC). Prints one line per change and READY/GIVEUP at the end.
cd /home/vu/Workspace/ARC-AGI-2
K="$(sed -n 's/^vast_ai_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
prev=""
while true; do
  st=$(.venv/bin/vastai show instances --api-key "$K" --raw 2>/dev/null | .venv/bin/python -c "
import json,sys
for i in json.load(sys.stdin):
    if i['id']==58063361: print(i.get('actual_status'), i.get('cur_state'), i.get('ssh_host'), i.get('ssh_port'))" 2>/dev/null)
  if timeout 25 ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes -p 25442 root@ssh7.vast.ai 'echo SSHOK' 2>/dev/null | grep -q SSHOK; then
    echo "$(date -u +%H:%M) READY vast=[$st]"; exit 0
  fi
  [ "$st" != "$prev" ] && echo "$(date -u +%H:%M) vast=[$st] ssh=down"
  prev="$st"
  if [ "$(date -u +%H%M)" -ge 0900 ] && [ "$(date -u +%H%M)" -lt 1200 ]; then echo "$(date -u +%H:%M) GIVEUP vast=[$st]"; exit 1; fi
  sleep 180
done
