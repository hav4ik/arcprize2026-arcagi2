#!/bin/bash
# E042 R3: clean A/B of --prefix-cache-retention-interval 128 on tool traffic, E64 x 64, PR #154, prefix caching,
# max-model-len 160K. tool_replay.py v4 (first turn starts inside its completion; queued setup turns drained before
# the warm-up). Flag on first, then off. Waits for P10. Log: /root/dsv41/runs/r3.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/r3.log; PY=/root/dsv41/venv/bin/python; TS=/root/dsv41/metadata/trace_shapes
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
until grep -qE "P10_DONE|ABORT" /root/dsv41/runs/p10.log 2>/dev/null; do sleep 30; done
for i in $(seq 90); do busy || break; sleep 2; done
log "r3 start"
for v in ret off; do
  N=r3_e64$v
  case $v in ret) X=(--prefix-cache-retention-interval 128);; off) X=();; esac
  busy && { log "ABORT before $N: busy"; exit 1; }
  DSV41_INDEXER_Q_PREDECODE=1 MODEL=/root/dsv41/model-e64 SEQS=128 UTIL=0.88 MAXLEN=163840 $S/serve.sh $N \
    --enable-prefix-caching "${X[@]}" >> $L
  (setsid nohup $S/gpumon.sh $N > /dev/null 2>&1 < /dev/null &)
  if wait_up $N; then
    (setsid nohup $S/kvmon.sh $N > /dev/null 2>&1 < /dev/null &)
    log "$N R2 tools C 64: $($PY $S/tool_replay.py --shapes $TS/tools_e030_hdrmin.json --conc 64 --warm-s 300 \
      --window-s 2400 --tool-delay 1 --seed 1 --json /root/dsv41/runs/$N/r2_tools_c64.json 2>&1 | tail -1 | cut -c1-900)"
  fi
  $S/stop.sh $N | tee -a $L
done
log "R3_DONE (server stopped; box idle)"
