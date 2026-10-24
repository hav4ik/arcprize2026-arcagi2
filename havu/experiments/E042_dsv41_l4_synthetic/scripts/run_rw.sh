#!/bin/bash
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
log "rw waiting for T4_CUT_DONE and DL_DONE"
until grep -q T4_CUT_DONE /root/dsv41/runs/t4.log; do sleep 10; done
until grep -q -E "DL_DONE|DL FAILED" /root/dsv41/runs/dl_e128t.log; do sleep 10; done
grep -q "DL FAILED" /root/dsv41/runs/dl_e128t.log && { log "ABORT: download failed"; exit 1; }
b=$(du -sbL $M --exclude=.cache | cut -f1); log "checkpoint $b bytes (HF 131090961684)"
for i in $(seq 60); do busy || break; sleep 5; done
busy && { log "ABORT: box busy"; exit 1; }
B=/root/dsv41/vllm_backup_e042synth
if [ ! -d $B ]; then
  for f in models/deepseek_v4_1/common/engram.py model_executor/layers/quantization/utils/humming_utils.py \
      model_executor/layers/quantization/humming.py models/deepseek_v4_1/nvidia/router.py e043_humming.py \
      models/deepseek_v4_1/quant_config.py models/deepseek_v4_1/nvidia/model.py; do
    mkdir -p $B/$(dirname $f); cp $V/$f $B/$f
  done
  log "backed up the E042 vLLM files to $B"
fi
if grep -q E042 $V/models/deepseek_v4_1/common/engram.py; then
  $PY $P/unpatch_engram.py $P/apply_patches_e042.py $V/models/deepseek_v4_1/common/engram.py >> $L 2>&1 \
    || { log "ABORT: unpatch_engram failed"; exit 1; }
fi
$PY $P/apply_compact_engram.py $V >> $L 2>&1 || { log "ABORT: apply_compact_engram failed"; exit 1; }
$PY $P/apply_e043_humming.py $V >> $L 2>&1 || { log "ABORT: apply_e043_humming failed"; exit 1; }
N=rw_e128t; R=/root/dsv41/runs/$N
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
log "$N smoke: $(echo "$a" | $PY -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:200]), "| reasoning", len(m.get("reasoning_content") or m.get("reasoning") or ""), "chars | usage", d.get("usage"))' 2>&1 | cut -c1-600)"
log "RW_UP"
