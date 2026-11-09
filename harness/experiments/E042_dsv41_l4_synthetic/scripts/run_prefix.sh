#!/bin/bash
# E042 P4: prefix caching on vs off (plan Task 8 order: graphs, then prefix caching). Starts after S4 is done.
# Per server: parity probe (cold vs cached P / P+S), tool-shaped multi-turn replay at C=16 and C=32, and
# (cache on only) one clean decode point. Log: /root/dsv41/runs/prefix.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/prefix.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
wait_up() {  # name
  local R=/root/dsv41/runs/$1 t=0
  until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
    sleep 10; t=$((t + 10))
    if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
      log "$1 SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; return 1
    fi
  done
  log "$1 up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
  log "$1 prefix-cache log lines: $(grep -i 'prefix' $R/server.log | grep -v 'Avg prompt' | cut -c1-250 | head -5 | tr '\n' ' ')"
}
busy() { curl -s -m 3 localhost:8100/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; }
probe() {  # name
  local R=/root/dsv41/runs/$1
  log "$1 parity: $($PY $S/prefix_probe.py parity --json $R/parity.json 2>&1 | tail -4 | tr '\n' ' ' | cut -c1-900)"
  for c in 16 32; do
    log "$1 replay C$c: $($PY $S/prefix_probe.py replay --conc $c --json $R/replay_c$c.json 2>&1 | tail -1 | cut -c1-900)"
  done
}

until grep -q BIGBATCH_DONE /root/dsv41/runs/bigbatch.log; do sleep 30; done
log "prefix start"
busy && { log "ABORT: port 8100 or GPUs still busy"; exit 1; }
SEQS=64 $S/serve.sh p4_pc_on --enable-prefix-caching >> $L
(setsid nohup $S/gpumon.sh p4_pc_on > /dev/null 2>&1 < /dev/null &)
if wait_up p4_pc_on; then
  probe p4_pc_on
  $S/dw_grid.sh p4_pc_on "1024:32"
  log "p4_pc_on decode: $(tail -2 /root/dsv41/runs/p4_pc_on/dw/grid.log | head -1 | sed -E 's/"first_token_s": \[[^]]*\], //' | cut -c1-400)"
fi
$S/stop.sh p4_pc_on | tee -a $L
busy && { log "ABORT before cache-off run: port or GPUs busy"; exit 1; }
SEQS=64 $S/serve.sh p4_pc_off >> $L
(setsid nohup $S/gpumon.sh p4_pc_off > /dev/null 2>&1 < /dev/null &)
wait_up p4_pc_off && probe p4_pc_off
$S/stop.sh p4_pc_off | tee -a $L
log "PREFIX_DONE (server stopped; box idle)"
