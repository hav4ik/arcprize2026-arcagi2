#!/usr/bin/env bash
# Start one E051 server run on a dev box: box_serve.sh in its own session (so it can be stopped by process group),
# then metrics_logger.py with the server's process group, both under /root/e051/runs/$RUN.
#
#   RUN=s1_off28_ckpt OFFLOAD_GIB=28 POLICY=checkpoint bash /root/e051/gpu/start_run.sh
#   stop:  kill -- -$(cat /root/e051/runs/$RUN/server.pgid); kill $(cat /root/e051/runs/$RUN/metrics_logger.pid)
set -euo pipefail
: "${RUN:?set RUN}"
D=/root/e051/runs/$RUN
[ ! -e "$D" ] || { echo "$D exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 100 ] || { echo "a GPU already has ${used} MiB in use: check for leftovers first"; exit 1; }
setsid bash /root/e051/gpu/box_serve.sh < /dev/null > /root/e051/runs/$RUN.launch.out 2>&1 || true
cat /root/e051/runs/$RUN.launch.out
PIDF=$(ls "$D"/*.pid)
P=$(cat "$PIDF")
PG=$(ps -o pgid= -p "$P" | tr -d ' ')
echo "$PG" > "$D/server.pgid"
cd "$D"
setsid nohup "${VENV:-/root/dsv41/venv_offload}/bin/python" -I /root/e051/gpu/metrics_logger.py \
  --out "$D/metrics.jsonl" --pgid "$PG" --interval 5 < /dev/null > "$D/metrics_logger.out" 2>&1 &
echo $! > "$D/metrics_logger.pid"
echo "server pid $P pgid $PG, logger $(cat "$D/metrics_logger.pid"), dir $D"
