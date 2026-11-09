#!/bin/bash
# usage: launch_box.sh <instance id> <eval label> <mtp> <kv dtype> <attn backend>
# Waits until the Vast instance exposes SSH, uploads bundle + scripts, starts setup, queues eval_6000.sh.
ID=$1; L=$2; MTP=$3; KV=$4; ATTN=$5
cd /home/vu/Workspace/ARC-AGI-2
J=/home/vu/.claude/jobs/821d4fe1/tmp
K="$(sed -n 's/^vast_ai_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
while true; do
  r=$(.venv/bin/vastai show instances --api-key "$K" --raw | python3 -c "
import json, sys
for i in json.load(sys.stdin):
    if i['id'] == $ID:
        p = ((i.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort')
        print(i.get('actual_status'), i.get('public_ipaddr'), p)")
  set -- $r
  [ "$1" = running ] && [ -n "$3" ] && [ "$3" != None ] && break
  sleep 30
done
H=$2; P=$3
echo "$ID ssh root@$H -p $P"
for i in $(seq 1 20); do timeout 30 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $P root@$H 'mkdir -p /root/arc' 2>/dev/null && break; sleep 15; done
scp -q -P $P -o BatchMode=yes $J/arc_bundle.tgz $J/box/setup_6000.sh $J/box/eval_6000.sh root@$H:/root/
timeout 30 ssh -o BatchMode=yes -p $P root@$H "mv /root/setup_6000.sh /root/eval_6000.sh /root/arc/ && cd /root/arc && \
  (setsid nohup ./setup_6000.sh > logs_setup.txt 2>&1 < /dev/null &); sleep 2; \
  (setsid nohup ./eval_6000.sh $L $MTP $KV $ATTN > eval_$L.log 2>&1 < /dev/null &); nvidia-smi --query-gpu=name --format=csv,noheader; echo queued $L"
