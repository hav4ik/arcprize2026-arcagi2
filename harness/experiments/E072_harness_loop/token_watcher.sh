#!/bin/bash
# token_watcher.sh <pod> <port> <ip>: local loop (no agent turns). When the pod's stage_model.sh writes /root/queue/need_kaggle_token,
# pipe the Kaggle token from SECRETS.yaml (read at that moment) to bin/dl_with_token.sh over ssh stdin. Stop: touch runs/token_$pod.stop
POD=$1; PORT=$2; IP=$3; H=$(cd "$(dirname "$0")" && pwd); S=$H/../../SECRETS.yaml
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$IP"
while [ ! -e $H/runs/token_$POD.stop ]; do
  if $SSH 'test -s /root/queue/need_kaggle_token' 2>/dev/null; then
    echo "$(date -u +%T) $POD needs the token: starting download"
    sed -n 's/^kaggle_api_token:[[:space:]]*//p' $S | tr -d "\"' \n" | $SSH 'bash /root/queue/bin/dl_with_token.sh' 2>&1 | tail -2
    echo "$(date -u +%T) $POD download call returned"
  fi
  sleep 20
done
