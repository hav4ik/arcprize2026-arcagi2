#!/bin/bash
# E042 P10: box twins of Kaggle run 5 (E48, PR #154, max-model-len 64K, no prefix caching, same points: 1K x 128,
# 32K x 64, 32K x 128), for a box vs Kaggle comparison at long context. Starts after R (R_DONE). Only the first
# profiler capture per server records (fork), so only 1K x 128 is profiled.
# Log: /root/dsv41/runs/p10.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/p10.log; PY=/root/dsv41/venv/bin/python
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
until grep -qE "R_DONE|ABORT" /root/dsv41/runs/r.log 2>/dev/null; do sleep 60; done
log "p10 start"
for v in e48; do
  M=/root/dsv41/model-e48; G="1024:128 32768:64 32768:128"
  N=p10_$v; R=/root/dsv41/runs/$N/dw; T=/root/dsv41/runs/$N/trace; mkdir -p $T
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=$M SEQS=128 UTIL=0.88 $S/serve.sh $N --profiler-config \
    "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$T\",\"torch_profiler_with_stack\":false,\"warmup_iterations\":3,\"active_iterations\":10}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    mkdir -p $R
    for p in $G; do
      ctx=${p%%:*}; c=${p##*:}
      log "$N ctx $ctx C $c: $($PY $S/decode_window.py --ctx $ctx --conc $c --window 60 --seed $c --sock-read 10800 \
        --profile-s 20 --json $R/ctx${ctx}_c${c}.json 2>&1 | tail -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-500)"
      mkdir -p $T/ctx${ctx}_c$c; find $T -maxdepth 1 -type f -exec mv {} $T/ctx${ctx}_c$c/ \;
      log "$N ctx $ctx C $c profile: $($PY $S/prof_summary.py $T/ctx${ctx}_c$c --indexer-per-step 8 --json $T/ctx${ctx}_c$c/summary.json 2>&1 | grep -o '"per_step_ms": {[^}]*}' | head -1)"
    done
  fi
  $S/stop.sh $N | tee -a $L
done
log "P10_DONE (server stopped; box idle)"
