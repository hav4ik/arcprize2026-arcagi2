#!/usr/bin/env bash
# After E032's xhigh first attempts finish on the three GPU0 servers (port 30002), restart those servers as base
# (official FP8, same minimal-capable template) and run base's missing first attempts (k93b) and its second
# attempts (pass@2) on dev56, k93a, k93b and rest across the three.  switch_gpu0_to_base.sh WAIT_PID
set -uo pipefail
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "=== $(date -u +%FT%TZ) GPU0 E032 xhigh fill-in done; switching GPU0 servers to base"
SSHO="-o ConnectTimeout=15 -o BatchMode=yes"
for b in "20904 root@23.16.187.177 /workspace/models/Qwen3.8-27B-FP8" \
         "24522 root@76.100.228.184 /workspace/models/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a" \
         "30835 root@82.98.239.123 /workspace/models/Qwen3.8-27B-FP8"; do
  set -- $b
  ssh $SSHO -p $1 $2 "cd /workspace/e032; P=\$(ps -eo pid,args | grep '^ *[0-9]* /opt/sglang/bin/python -m sglang.launch_server' | grep -- '--port 30002' | awk '{print \$1}'); kill \$P; for i in \$(seq 1 30); do [ \$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ] && break; sleep 2; done; COMMON='--host 127.0.0.1 --mem-fraction-static 0.88 --context-length 131072 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chat-template /workspace/e032/chat_template_minimal.jinja'; CUDA_VISIBLE_DEVICES=0 setsid nohup /opt/sglang/bin/python -m sglang.launch_server --model-path $3 --served-model-name base --port 30002 \$COMMON > sg_base_gpu0.log 2>&1 < /dev/null & echo \"$2 GPU0 -> base\"" 2>&1 | grep -v "^Welcome\|^Have fun\|AI agents\|^$\|vast-agents\|authentication fails"
done
for i in $(seq 1 60); do ok=0; for p in 32002 32012 32022; do [ "$(curl -s -m 5 127.0.0.1:$p/v1/models | grep -o '"id":"[a-z0-9]*"' | head -1)" = '"id":"base"' ] && ok=$((ok+1)); done; [ $ok = 3 ] && break; sleep 10; done
echo "base GPU0 servers ready $ok/3 at $(date -u +%H:%M:%S)"; [ $ok = 3 ] || exit 1
cd /home/vu/Workspace/ARC-AGI-2/experiments/E032_qwen38_sft_e031
echo "format base@32002: $(/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/a7e03fe1-bf32-49e2-bd9a-9300af5223c6/scratchpad/tokvenv/bin/python -I format_check.py http://127.0.0.1:32002/v1 base 2>&1 | grep -v WARNING | tail -1)"
U=http://127.0.0.1:32002/v1,http://127.0.0.1:32012/v1,http://127.0.0.1:32022/v1
./eval_chain.sh base $U 60 k93b
ATTEMPTS=2 ./eval_chain.sh base $U 60 "dev56 k93a k93b rest"
echo "=== $(date -u +%FT%TZ) base fill-in and pass@2 done"
