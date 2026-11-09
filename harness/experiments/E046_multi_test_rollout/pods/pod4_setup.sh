#!/bin/bash
# Wait for the 4th Runpod pod, install the E042 L4 stack + E128 + pod kit, then start the KV-gate multi-test run.
cd /home/vu/Workspace/ARC-AGI-2
SP=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad
KNOWN="137m3l13x3ei2n 23jiwzwlii3uoa sltm3bor1tz1t8"
log() { echo "$(date -u +%T) $*"; }
while true; do
  R=$(RUNPOD_KEY="$(sed -n 's/^runpod_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")" KNOWN="$KNOWN" .venv/bin/python - <<'PY'
import os, httpx
q = "query { myself { pods { id runtime { ports { ip isIpPublic privatePort publicPort } } } } }"
d = httpx.post("https://api.runpod.io/graphql", json={"query": q}, headers={"Authorization": f"Bearer {os.environ['RUNPOD_KEY']}"}, timeout=30).json()
for p in d["data"]["myself"]["pods"]:
    if p["id"] in os.environ["KNOWN"].split():
        continue
    for x in ((p.get("runtime") or {}).get("ports") or []):
        if x.get("isIpPublic") and x["privatePort"] == 22:
            print(p["id"], x["ip"], x["publicPort"])
PY
)
  [ -n "$R" ] && break
  sleep 30
done
set -- $R; ID=$1; IP=$2; PORT=$3
log "pod4 $ID ssh $IP:$PORT"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $PORT root@$IP)
until timeout 30 "${SSH[@]}" true 2>/dev/null; do sleep 15; done
log "ssh ok"
(cd experiments/E042_dsv41_l4_synthetic && timeout 300 bash scripts/setup_box.sh root@$IP $PORT | tail -1)
sed -n 's/^huggingface_full_access_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' " | timeout 180 "${SSH[@]}" 'read -r T; mkdir -p /root/dsv41/runs /root/dsv41/model-e128t; (python3 -m venv /root/hfenv >/dev/null 2>&1 && /root/hfenv/bin/pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1) || exit 3; printf "%s\n" "#!/bin/bash" "L=/root/dsv41/runs/dl_e128t.log" "/root/hfenv/bin/hf download chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit --local-dir /root/dsv41/model-e128t >> \$L 2>&1 && echo DL_DONE >> \$L || echo DL_FAILED >> \$L" > /root/dsv41/dl_e128.sh; chmod +x /root/dsv41/dl_e128.sh; HF_TOKEN="$T" setsid nohup /root/dsv41/dl_e128.sh > /dev/null 2>&1 < /dev/null & echo dl started'
log "download started"
for i in $(seq 1 120); do
  st=$(timeout 30 "${SSH[@]}" 'tail -1 /root/dsv41/install.log; grep -cE "DL_DONE" /root/dsv41/runs/dl_e128t.log; grep -c "DL_FAILED" /root/dsv41/runs/dl_e128t.log' 2>/dev/null)
  if echo "$st" | grep -q "INSTALL OK" && [ "$(echo "$st" | sed -n 2p)" = "1" ]; then break; fi
  if echo "$st" | grep -qiE "timed out|error:" ; then log "install retry: $(echo "$st" | head -1)"; timeout 30 "${SSH[@]}" 'setsid nohup /root/dsv41/box_install.sh > /dev/null 2>&1 < /dev/null &'; fi
  [ "$(echo "$st" | sed -n 3p)" = "1" ] && { log "DOWNLOAD FAILED"; exit 1; }
  sleep 30
done
log "install + download done"
tar czf - -C $SP podkit | timeout 280 "${SSH[@]}" 'tar xzf - -C /root && bash /root/podkit/podprep.sh 2>&1 | tail -1 && /root/podrun.sh mt52_gate75 ARC_WORKERS=52 ARC_MULTI_TEST=1 ARC_GPU_UTIL=0.938 ARC_KV_ADMIT_MAX=0.75 ARC_KV_ADMIT_MIN_WORKERS=16 > /dev/null 2>&1 < /dev/null; sleep 5; tail -1 /root/runs/mt52_gate75/run.log; setsid nohup /root/memlog.sh /root/runs/mt52_gate75/mem.log > /dev/null 2>&1 < /dev/null &' 2>&1 | tail -3
log "STARTED mt52_gate75 on $IP:$PORT"
