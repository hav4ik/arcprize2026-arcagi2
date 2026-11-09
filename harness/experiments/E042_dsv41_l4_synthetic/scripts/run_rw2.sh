#!/bin/bash
# E042 RW2 (Runpod, 15:55): restart run_rw.sh's server with the reasoning and tool-call parsers (serve_real.sh now
# passes E043 serve.sh's --reasoning-parser / --tool-call-parser deepseek_v41 --enable-auto-tool-choice; rw_e128t's
# smoke answer came back with its thinking inline). vLLM is already switched to E043's loaders. Helpers from run_rw.sh::
# E042 RW (Runpod 4xL4, 2026-10-24 15:30): the user's real-weight check of E043's tool E128 2-bit checkpoint
# (chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit; tool ARC pass@2 82.5% on 8x RTX PRO 6000) after T4's C = 40
# point ("Yes, drop the 48 replay"). Waits for T4_CUT_DONE and the download (dl_e128t.sh), switches the venv's vLLM
# from E042's synthetic Engram fold to E043's compact Engram loader (files backed up to vllm_backup_e042synth first),
# serves the checkpoint at the T2 config (util 0.93, 128 seqs, 160K, prefix caching, retention 128, PR #154 flag), and
# checks one chat answer. The harness runs from the workstation (eval_rw.sh). Log: /root/dsv41/runs/rw.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/rw.log; PY=/root/dsv41/venv/bin/python
V=/root/dsv41/venv/lib/python3.12/site-packages/vllm; P=/root/dsv41/e043p_new; M=/root/dsv41/model-e128t
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -gt 1000 ]; }
$S/stop.sh rw_e128t | tee -a $L
for i in $(seq 60); do busy || break; sleep 5; done
busy && { log "ABORT: box busy"; exit 1; }
N=rw_e128t2; R=/root/dsv41/runs/$N
env DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}' \
  DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$M/gptq DSV41_INDEXER_Q_PREDECODE=1 \
  MODEL=$M SEQS=128 UTIL=0.93 MAXLEN=163840 $S/serve_real.sh $N \
  --enable-prefix-caching --prefix-cache-retention-interval 128 >> $L
(setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
t=0
until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
  sleep 10; t=$((t + 10))
  if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 3600 ]; then
    log "$N SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-400)"; exit 1
  fi
done
log "$N up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
  "$(grep -o -E "Model loading took [0-9.]+ GiB" $R/server.log | head -1);" \
  "$(grep -o -E "Using [A-Za-z0-9_' ]+ (MoE|Mxfp4)[A-Za-z0-9_ ]*" $R/server.log | sort -u | head -2 | tr '\n' ' ')"
(setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
log "$N host: $(free -g | awk '/Mem:/{print $3 " GiB used / " $7 " available"}'); shm $(df -h /dev/shm | awk 'NR==2{print $3}') used"
a=$(curl -s -m 600 localhost:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "dsv41",
  "messages": [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}], "max_tokens": 2048,
  "temperature": 0, "chat_template_kwargs": {"thinking": true, "reasoning_effort": 50}}')
log "$N smoke: $(echo "$a" | $PY -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:200]), "| reasoning", repr((m.get("reasoning_content") or m.get("reasoning") or "")[:120]), "| usage", d.get("usage"))' 2>&1 | cut -c1-600)"
log "RW2_UP"
