#!/bin/bash
# E042 P12: kernel and prefill A/B on E64 (PR #154, no prefix caching, max-model-len 64K), from the vLLM V4.1 and
# Humming posts (2026-10-07, 10-21). Waits for R3. Each server: decode_window at 1K x 64, 1K x 128 (decode) and
# 8K x 16, 32K x 4 (prefill tok/s of 131K cold tokens). Base runs first and last, to bound GPU2's thermal drift.
#   0. NCCL all-reduce micro-bench: default vs NCCL_SHM_USE_CUDA_MEMCPY=1 (copy engine; modes 1 and 3)
#   1. base (Marlin MXFP4 experts, AdaMxfp8 dense)       5. Humming experts with FP8 activations (W4A8)
#   2. Humming experts (--moe-backend humming)            6. Marlin experts with FP8 activations (W4A8)
#   3. Humming dense (--linear-backend humming)           7. CED prefill (hf_overrides ced_prefill)
#   4. both Humming                                       8. base again
# Log: /root/dsv41/runs/p12.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/p12.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
wait_up() {
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1);" \
    "$(grep -o -E "Using '[A-Z_]+' Mxfp4 MoE backend|Using [A-Za-z0-9]+ for MXFP8 GEMM" $R/server.log | sort -u | tr '\n' ' ')"
}
until grep -qE "R3_DONE|ABORT" /root/dsv41/runs/r3.log 2>/dev/null; do sleep 30; done
for i in $(seq 90); do busy || break; sleep 2; done
busy && { log "ABORT: GPUs busy after R3"; exit 1; }
log "p12 start"

O=/root/dsv41/runs/nccl; mkdir -p $O; T=/root/dsv41/venv/bin/torchrun; B=$S/nccl_allreduce_bench.py
(cd $O; unset NCCL_PROTO NCCL_ALGO NCCL_P2P_LEVEL
 $T --nproc-per-node 4 $B --label p12_default --out p12_default.json 2>&1 | grep -E "^p12_default" > p12_default.txt
 NCCL_SHM_USE_CUDA_MEMCPY=1 $T --nproc-per-node 4 $B --label shm_ce1 --out shm_ce1.json 2>&1 | grep -E "^shm_ce1" > shm_ce1.txt
 NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3 $T --nproc-per-node 4 $B --label shm_ce3 --out shm_ce3.json 2>&1 \
   | grep -E "^shm_ce3" > shm_ce3.txt)
for f in p12_default shm_ce1 shm_ce3; do log "nccl $f: $(grep -E 'T=( 2048|   64|    1) ' $O/$f.txt | awk '{print $2, $5 "ms"}' | tr '\n' ' ')"; done

HW='{"quant_method":"humming","dtype":"float8e4m3"}'
for v in base hmoe hlin hboth hw4a8 mw4a8 ced base2; do
  N=p12_$v; R=/root/dsv41/runs/$N/dw; X=(); E=()
  case $v in
    hmoe) X=(--moe-backend humming);;
    hlin) X=(--linear-backend humming);;
    hboth) X=(--moe-backend humming --linear-backend humming);;
    hw4a8) X=(--moe-backend humming); E=(VLLM_HUMMING_INPUT_QUANT_CONFIG=$HW);;
    mw4a8) E=(MARLIN_IN=fp8);;
    ced) X=(--hf-overrides '{"ced_prefill": true}');;
  esac
  busy && { log "ABORT before $N: busy"; exit 1; }
  env "${E[@]}" DSV41_INDEXER_Q_PREDECODE=1 SEQS=128 UTIL=0.88 $S/serve.sh $N "${X[@]}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    mkdir -p $R
    for p in 1024:64 1024:128 8192:16 32768:4; do
      ctx=${p%%:*}; c=${p##*:}
      log "$N ctx $ctx C $c: $($PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c --sock-read 10800 \
        --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
    done
  fi
  $S/stop.sh $N | tee -a $L
done
log "P12_DONE (server stopped; box idle)"
