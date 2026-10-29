#!/bin/bash
# E062 acc_r1 (on the 48 GB box): after acc_b32 ends, the same arm with R1 (humming_utils.py 4dd6392b +
# DSV41_HUMMING_MOE_PAD=64,64). Same podrun env as acc_b32 (its env.txt, minus the per-run paths and salt).
# R1's file is swapped into /root/dsv41/venv only for this run; B32's file (9ae7acb8) is restored at the end.
set -uo pipefail
L=/root/e062/acc_r1_chain.log
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
F=/root/dsv41/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/humming_utils.py
sha() { sha256sum "$1" | cut -c1-8; }
until grep -q "^done at +" /root/runs/acc_b32/run.log 2>/dev/null; do sleep 60; done
log "acc_b32 done"
for _ in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1000 ] && break; sleep 10; done
[ "$u" -lt 1000 ] || { log "GPU still busy ($u MiB): not starting"; exit 1; }
[ "$(sha $F)" = 9ae7acb8 ] || { log "unexpected humming_utils $(sha $F)"; exit 1; }
[ "$(sha /root/e062/humming_utils_r1.py)" = 4dd6392b ] || { log "R1 file sha mismatch"; exit 1; }
cp $F /root/e062/humming_utils_b32_48g.py && cp /root/e062/humming_utils_r1.py $F && log "R1 humming_utils installed ($(sha $F))"
ARGS=()
while IFS= read -r line; do
  case "${line%%=*}" in ARC_OUT|ARC_SCRATCH|ARC_CACHE_SALT) continue ;; esac
  ARGS+=("$line")
done < <(grep -E '^ARC_' /root/runs/acc_b32/env.txt)
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HF_API_TOKEN$)'); do unset "$v"; done
bash /root/podrun.sh acc_r1 "${ARGS[@]}" PATH=/root/dsv41/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin DSV41_HUMMING_MOE_PAD=64,64 >> $L 2>&1
sleep 5
setsid nohup /root/memlog.sh /root/runs/acc_r1/mem.log > /dev/null 2>&1 < /dev/null &
M=$!
log "acc_r1 launched, memlog $M"
until grep -q "^done at +" /root/runs/acc_r1/run.log 2>/dev/null; do sleep 60; done
log "acc_r1 done: $(grep 'local score' /root/runs/acc_r1/out/dsv41_logs/solver.log | tail -1)"
kill $M 2>/dev/null
for _ in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1000 ] && break; sleep 10; done
cp /root/e062/humming_utils_b32_48g.py $F && log "B32 humming_utils restored ($(sha $F)); GPUs max $u MiB; ACC_R1 CHAIN DONE"
