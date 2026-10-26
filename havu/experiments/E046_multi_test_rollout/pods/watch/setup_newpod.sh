#!/bin/bash
# Set up a fresh Runpod 4xL4 pod end to end (E042 stack + E128 + podkit + kit2 = kit + kit_next overlay) and start a run.
# Usage: setup_newpod.sh <id> <ip> <port> <run name> [ENV=VALUE ...]   (ARC_KIT=/root/kit2 expected among the env)
cd /home/vu/Workspace/ARC-AGI-2
SP=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad
ID=$1; IP=$2; PORT=$3; RUN=$4; shift 4; ENVS="$*"
log() { echo "$(date -u +%T) [$ID $RUN] $*"; }
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=15 -p $PORT root@$IP)
log "waiting for ssh $IP:$PORT"
until timeout 30 "${SSH[@]}" true 2>/dev/null; do sleep 15; done
log "ssh ok"
(cd experiments/E042_dsv41_l4_synthetic && timeout 300 bash scripts/setup_box.sh root@$IP $PORT | tail -1)
sed -n 's/^huggingface_full_access_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' " | timeout 240 "${SSH[@]}" 'read -r T; mkdir -p /root/dsv41/runs /root/dsv41/model-e128t; (python3 -m venv /root/hfenv >/dev/null 2>&1 && /root/hfenv/bin/pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1) || exit 3; printf "%s\n" "#!/bin/bash" "L=/root/dsv41/runs/dl_e128t.log" "/root/hfenv/bin/hf download chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit --local-dir /root/dsv41/model-e128t >> \$L 2>&1 && echo DL_DONE >> \$L || echo DL_FAILED >> \$L" > /root/dsv41/dl_e128.sh; chmod +x /root/dsv41/dl_e128.sh; HF_TOKEN="$T" setsid nohup /root/dsv41/dl_e128.sh > /dev/null 2>&1 < /dev/null & echo dl started'
log "download started (rc=$?)"
ok=0
for i in $(seq 1 160); do
  st=$(timeout 30 "${SSH[@]}" 'tail -1 /root/dsv41/install.log; grep -cE "DL_DONE" /root/dsv41/runs/dl_e128t.log; grep -c "DL_FAILED" /root/dsv41/runs/dl_e128t.log' 2>/dev/null)
  if echo "$st" | grep -q "INSTALL OK" && [ "$(echo "$st" | sed -n 2p)" = "1" ]; then ok=1; break; fi
  if echo "$st" | head -1 | grep -qiE "timed out|error|failed" ; then log "install retry: $(echo "$st" | head -1 | cut -c1-200)"; timeout 30 "${SSH[@]}" 'setsid nohup /root/dsv41/box_install.sh > /dev/null 2>&1 < /dev/null &'; fi
  [ "$(echo "$st" | sed -n 3p)" = "1" ] && { log "DOWNLOAD FAILED"; exit 1; }
  [ $((i % 10)) = 0 ] && log "waiting: $(echo "$st" | head -1 | cut -c1-150) dl_done=$(echo "$st" | sed -n 2p)"
  sleep 30
done
[ $ok = 1 ] || { log "INSTALL/DOWNLOAD NOT DONE after 80 min"; exit 1; }
log "install + download done"
# kit_next: the two changed files (solver + harness), from the workspace
tar czf - -C kaggle/dsv41_tools solver_dsv41.py -C ../../ evals/run_deepseek_tools.py | timeout 120 "${SSH[@]}" 'rm -rf /root/kit_next && mkdir -p /root/kit_next && tar xzf - -C /root/kit_next && ls -R /root/kit_next | head'
tar czf - -C $SP podkit | timeout 280 "${SSH[@]}" 'tar xzf - -C /root && bash /root/podkit/podprep.sh 2>&1 | tail -1'
timeout 120 "${SSH[@]}" 'set -e; rm -rf /root/kit2; cp -a /root/kit /root/kit2; cp /root/kit_next/solver_dsv41.py /root/kit2/solver_dsv41.py; cp /root/kit_next/evals/run_deepseek_tools.py /root/kit2/repo/evals/run_deepseek_tools.py; sha256sum /root/kit2/solver_dsv41.py /root/kit2/repo/evals/run_deepseek_tools.py'
timeout 120 "${SSH[@]}" "/root/podrun.sh $RUN $ENVS > /dev/null 2>&1 < /dev/null; sleep 5; tail -2 /root/runs/$RUN/run.log; cat /root/runs/$RUN/env.txt | grep -E 'KIT|WORKERS|MULTI|KV_ADMIT|GPU_UTIL'; setsid nohup /root/memlog.sh /root/runs/$RUN/mem.log > /dev/null 2>&1 < /dev/null &" 2>&1 | tail -12
log "STARTED $RUN"
for i in $(seq 1 40); do
  up=$(timeout 30 "${SSH[@]}" "grep -E 'server up|server exited|server failed|server not up' /root/runs/$RUN/run.log | tail -1" 2>/dev/null)
  [ -n "$up" ] && { log "SERVERLINE $up"; break; }
  sleep 30
done
