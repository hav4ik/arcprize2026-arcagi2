#!/bin/bash
# Local watchdog for the E128 download on box 57983124 (E059 B16; copy of E057b dl_watchdog.sh): every 60 s read the bytes on disk (incl. .incomplete files);
# no growth for 3 checks AND eth0 rx < 5 MiB/s (xet writes in bursts) and no DL_DONE -> stop the download's own process group (by PGID) and restart it with the
# token over SSH stdin. Exits on DL_DONE / DL_FAILED.
cd /home/vu/Workspace/ARC-AGI-2
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -p 62174 root@159.112.204.189)
prev=0; still=0
while true; do
  s=$(timeout 40 "${SSH[@]}" 'grep -cE "DL_DONE|DL_FAILED" /root/dsv41/runs/dl_e128t.log; du -sb /root/dsv41/model-e128t | cut -f1; pgrep -f "^/bin/bash /root/dsv41/dl_e128_(full|http).sh" | head -1 || echo none; a=$(awk "/eth0/{print \$2}" /proc/net/dev); sleep 5; b=$(awk "/eth0/{print \$2}" /proc/net/dev); echo $(( (b-a)/5/1048576 ))' 2>/dev/null)
  set -- $s
  [ "${1:-0}" -ge 1 ] && { echo "$(date -u +%T) download finished: $(timeout 30 "${SSH[@]}" 'tail -1 /root/dsv41/runs/dl_e128t.log' 2>/dev/null | cut -c1-200)"; exit 0; }
  b=${2:-0}; pid=${3:-none}; rx=${4:-0}; [ "$pid" = none ] && pid=""
  if [ "$b" -gt "$prev" ] || [ "$rx" -ge 5 ]; then still=0; else still=$((still + 1)); fi
  echo "$(date -u +%T) bytes $b (+$(( (b - prev) / 1048576 )) MiB) rx ${rx} MiB/s still=$still pid=$pid"
  prev=$b
  if [ "$still" -ge 3 ] || [ -z "$pid" ]; then
    echo "$(date -u +%T) stalled or gone: restarting"
    timeout 60 "${SSH[@]}" "PG=\$(ps -o pgid= -p ${pid:-1} 2>/dev/null | tr -d ' '); if [ -n \"$pid\" ] && [ -n \"\$PG\" ] && [ \"\$PG\" != 1 ]; then kill -- -\$PG; sleep 5; pgrep -g \$PG >/dev/null && kill -9 -- -\$PG; fi; echo \"[\$(date -u +%T)] watchdog: no progress for 3 min, restarting over plain HTTP (group \$PG)\" >> /root/dsv41/runs/dl_e128t.log; find /root/dsv41/model-e128t/.cache -regextype posix-extended -regex '.*/[^./]+[.][0-9a-f]{64}[.][0-9a-f]+[.]incomplete' -delete" 2>/dev/null
    sed -n 's/^huggingface_full_access_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' " | timeout 60 "${SSH[@]}" 'read -r T; HF_TOKEN="$T" setsid nohup /root/dsv41/dl_e128_http.sh > /dev/null 2>&1 < /dev/null & sleep 2; echo restarted' 2>/dev/null
    still=0
  fi
  sleep 60
done
