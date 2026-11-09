#!/bin/bash
# E042 P6: bottleneck attribution and the two untried kernel leads. Starts after P5 (AGENTIC_DONE).
# 1) Fork PR #154 (indexer q predecode, env-gated): kernel bench on GPU3, then decode grids with it off and on.
# 2) FlashInfer PCIe IPC all-reduce (VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1; box-only, needs P2P): same grid.
# 3) Decode profiles at C=16 and C=128 (C=64 is P3) to split each kernel into per-step and per-token cost.
# Log: /root/dsv41/runs/bottleneck.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/bottleneck.log; PY=/root/dsv41/venv/bin/python
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
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
}
until grep -qE "AGENTIC_DONE|ABORT|SERVE FAILED" /root/dsv41/runs/agentic.log 2>/dev/null; do sleep 30; done
log "bottleneck start"
busy && { log "ABORT: port 8100 or GPUs busy"; exit 1; }
$PY -I /root/dsv41/patches/pr154_apply.py /root/dsv41/patches/pr154/pr154_kernel.diff 2>&1 | tee -a $L
CUDA_VISIBLE_DEVICES=3 $PY $S/indexer_bench.py --json /root/dsv41/runs/indexer_bench.json 2>&1 | tail -16 | tee -a $L

G="1024:64 1024:128 8192:32"
for v in base pred ipc; do
  case $v in
    base) E=();;
    pred) E=(DSV41_INDEXER_Q_PREDECODE=1);;
    ipc) E=(VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1);;
  esac
  busy && { log "ABORT before p6_$v: busy"; exit 1; }
  env "${E[@]}" SEQS=128 UTIL=0.88 $S/serve.sh p6_$v >> $L
  (setsid nohup $S/gpumon.sh p6_$v > /dev/null 2>&1 < /dev/null &)
  if wait_up p6_$v; then
    [ $v = ipc ] && log "ipc log lines: $(grep -iE 'pcie.?ipc|all.?reduce' /root/dsv41/runs/p6_ipc/server.log | grep -v Warning | cut -c1-200 | head -4 | tr '\n' ' ')"
    log "p6_$v smoke: $($S/smoke_request.sh 2>&1 | grep -E 'finite' | cut -c1-200)"
    $S/dw_grid.sh p6_$v "$G"
    log "p6_$v grid: $(sed -E 's/"first_token_s": \[[^]]*\], //' /root/dsv41/runs/p6_$v/dw/grid.log | tr '\n' ' ' | cut -c1-1500)"
  fi
  $S/stop.sh p6_$v | tee -a $L
done

for c in 16 128; do
  N=p6_prof_c$c; P=/root/dsv41/runs/$N; mkdir -p $P/trace
  busy && { log "ABORT before $N: busy"; exit 1; }
  SEQS=128 UTIL=0.88 $S/serve.sh $N --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$P/trace\",\"torch_profiler_with_stack\":false,\"warmup_iterations\":3,\"active_iterations\":10}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    $PY $S/decode_window.py --ctx 1024 --conc $c --window 150 --seed $((500 + c)) --json $P/dw.json > $P/dw.txt 2>&1 &
    BP=$!
    sleep $((c * 1024 / 700 + 30))   # prefill of all C prompts, then steady decode
    curl -s -m 30 -X POST localhost:8100/start_profile; sleep 20; curl -s -m 600 -X POST localhost:8100/stop_profile
    log "$N capture stopped; files: $(find $P/trace -type f | wc -l)"
    wait $BP; log "$N window (profiler inside): $(sed -E 's/"first_token_s": \[[^]]*\], //' $P/dw.txt | cut -c1-400)"
  fi
  $S/stop.sh $N | tee -a $L
done
log "BOTTLENECK_DONE (server stopped; box idle)"
