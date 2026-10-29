#!/bin/bash
# wait until ssh to pod 3 works again (user fixed /root perms or restarted the pod), then: fix /root perms, re-stage the
# bundle without ownership/dir-mode changes, start the setup chain with the keys over stdin.
cd /home/vu/Workspace/ARC-AGI-2
S=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/pod3
for i in $(seq 1 120); do
  R="$(sed -n 's/^runpod_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
  L=$(curl -s -H "Authorization: Bearer $R" https://rest.runpod.io/v1/pods/o5r3oa6tgfpc8i | .venv/bin/python -c '
import json,sys
p=json.load(sys.stdin); pm=p.get("portMappings") or {}
if p.get("publicIp") and pm.get("22"): print(p["publicIp"], pm["22"])' 2>/dev/null)
  unset R
  if [ -n "$L" ]; then
    set -- $L
    if timeout 20 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -p $2 root@$1 true 2>/dev/null; then
      echo "$L" > $S/pod3_ssh.txt; echo "SSH_BACK $L $(date -u +%T)"; break
    fi
  fi
  sleep 30
done
[ -n "${L:-}" ] || { echo "TIMEOUT"; exit 1; }
set -- $L; SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -p $2 root@$1)
timeout 60 "${SSH[@]}" 'chown root:root /root && chmod 700 /root && ls -ld /root' || { echo "PERM_FIX_FAILED"; exit 2; }
tar czf - -C $S/bundle/root . | timeout 180 "${SSH[@]}" 'tar xzf - -C /root --no-same-owner --no-overwrite-dir && chown -R root:root /root/b32m /root/e062pl /root/e066 /root/kit_img /root/data && chmod 700 /root && ls -ld /root && mkdir -p /dev/shm/uvcache /root/models && ln -sfn /dev/shm/uvcache /root/b32m/uvcache && echo STAGED' || { echo "STAGE_FAILED"; exit 3; }
timeout 20 "${SSH[@]}" true || { echo "SSH_BROKE_AFTER_STAGE"; exit 4; }
K="$(sed -n 's/^kaggle_api_token:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"; H="$(sed -n 's/^huggingface_readonly_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
printf '%s\n%s\n' "$K" "$H" | timeout 60 "${SSH[@]}" '[ -e /root/chain.log ] && grep -q start /root/chain.log && { echo CHAIN_ALREADY_STARTED; exit 0; }; read -r KAGGLE_API_TOKEN; read -r HF_TOKEN; export KAGGLE_API_TOKEN HF_TOKEN; setsid nohup bash /root/pod3_chain.sh > /dev/null 2>&1 < /dev/null & sleep 5; tail -3 /root/chain.log'
unset K H
echo "CHAIN_STARTED $(date -u +%T)"
