#!/bin/bash
# E042 P9: decode at long context and high concurrency (mode-1 ARC traces run to 50-80K tokens). All with PR #154 on.
# Context cost (indexer, attention) does not depend on the expert count, so the series runs on E48 (2.45M KV tokens,
# room for 128 x 32K), plus one E64 point to compare. Each point first prefills C x ctx tokens at ~700 tok/s
# (32K x 128 = 4.2M tokens, ~100 min), hence the long client read timeout.
# Each point also captures a 10-step torch profile before its window (decode_window --profile-s), filed per point.
# Log: /root/dsv41/runs/p9.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/p9.log; PY=/root/dsv41/venv/bin/python
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
log "p9 start"
for v in e48 e64; do
  case $v in
    e48) M=/root/dsv41/model-e48; G="1024:32 32768:32 32768:64 32768:128 60000:64";;
    e64) M=/root/dsv41/model-e64; G="32768:64";;
  esac
  N=p9_$v; R=/root/dsv41/runs/$N/dw; T=/root/dsv41/runs/$N/trace; mkdir -p $T
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
log "P9_DONE (server stopped; box idle)"
