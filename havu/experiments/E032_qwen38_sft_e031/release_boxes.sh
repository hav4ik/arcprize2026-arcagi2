#!/usr/bin/env bash
# Wait for base's k93b first attempts (the last eval job), confirm no API tokens remain on the boxes, then destroy the
# three E032 RTX boxes (user-authorized 2026-10-19) and close the local tunnels. The 4xL4 box (55889595) is not touched.
#   release_boxes.sh WAIT_PID
set -uo pipefail
cd /home/vu/Workspace/ARC-AGI-2
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== $(date -u +%FT%TZ) last eval job finished"
python3 experiments/E032_qwen38_sft_e031/eval_status.py | grep -v effmed
for b in "20904 root@23.16.187.177" "24522 root@76.100.228.184" "30835 root@82.98.239.123"; do
  set -- $b
  echo "$2: tokens left: $(timeout 30 ssh -o ConnectTimeout=15 -o BatchMode=yes -p $1 $2 'ls /root/.e032_* /root/.e028_* 2>/dev/null | wc -l' 2>/dev/null | tail -1)"
done
K="$(sed -n 's/^vast_ai_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
for id in 56613651 56598413 56630374; do
  echo "destroy $id: $(.venv/bin/vastai destroy instance $id -y --api-key "$K" 2>&1 | tail -1)"
done
kill 1689973 1755349 1824939 2>/dev/null && echo "tunnels closed"
sleep 20
.venv/bin/vastai show instances --api-key "$K" --raw 2>/dev/null | python3 -c "
import sys, json
print('remaining instances:', [(i['id'], i.get('num_gpus'), i.get('gpu_name'), i.get('actual_status')) for i in json.load(sys.stdin)])"
echo "=== $(date -u +%FT%TZ) release done"
