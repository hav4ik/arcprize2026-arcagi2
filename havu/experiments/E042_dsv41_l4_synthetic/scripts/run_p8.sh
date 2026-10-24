#!/bin/bash
# E042 P8: is PR #154's decode gain real? P6 measured pred +28% at 1K C=128, but GPU2 (thermally throttled) ran
# 1035 MHz there vs 900 for base. Interleave base/pred twice (ABAB) on E64, then the stacks pred+ipc (E64) and
# pred on E48. Log: /root/dsv41/runs/p8.log; per-variant GPU2 clocks are in runs/p8_*/gpu.csv.
S=/root/dsv41/scripts; L=/root/dsv41/runs/p8.log
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
log "p8 start"
G="1024:64 1024:128"
for v in base1 pred1 base2 pred2 predipc e48pred; do
  E=(); M=/root/dsv41/model-e64
  case $v in
    pred*) E=(DSV41_INDEXER_Q_PREDECODE=1);;
    e48pred) E=(DSV41_INDEXER_Q_PREDECODE=1); M=/root/dsv41/model-e48;;
  esac
  [ $v = predipc ] && E+=(VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1)
  busy && { log "ABORT before p8_$v: busy"; exit 1; }
  env "${E[@]}" MODEL=$M SEQS=128 UTIL=0.88 $S/serve.sh p8_$v >> $L
  (setsid nohup $S/gpumon.sh p8_$v > /dev/null 2>&1 < /dev/null &)
  if wait_up p8_$v; then
    log "p8_$v smoke: $($S/smoke_request.sh 2>&1 | grep -E 'finite' | cut -c1-200)"
    $S/dw_grid.sh p8_$v "$G"
    log "p8_$v grid: $(sed -E 's/"first_token_s": \[[^]]*\], //' /root/dsv41/runs/p8_$v/dw/grid.log | tr '\n' ' ' | cut -c1-1200)"
  fi
  $S/stop.sh p8_$v | tee -a $L
done
log "P8_DONE (server stopped; box idle)"
