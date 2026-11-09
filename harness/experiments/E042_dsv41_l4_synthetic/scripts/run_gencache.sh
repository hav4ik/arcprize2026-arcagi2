#!/bin/bash
# E042 P7: are generated-token KV blocks reused by the prefix cache? (P5 hit rates match "prompt + tool tokens only".)
# Starts after P6 (BOTTLENECK_DONE). Log: /root/dsv41/runs/gencache.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/gencache.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
until grep -qE "BOTTLENECK_DONE|ABORT|SERVE FAILED" /root/dsv41/runs/bottleneck.log 2>/dev/null; do sleep 30; done
curl -s -m 3 localhost:8100/v1/models > /dev/null && { log "ABORT: port busy"; exit 1; }
R=/root/dsv41/runs/p7_gencache
SEQS=64 $S/serve.sh p7_gencache --enable-prefix-caching >> $L
t=0; until curl -s -m 3 localhost:8100/v1/models > /dev/null; do sleep 10; t=$((t + 10));
  if grep -q Traceback $R/server.log || [ $t -gt 2700 ]; then log "SERVE FAILED"; $S/stop.sh p7_gencache; exit 1; fi; done
log "gencache: $($PY $S/prefix_probe.py gencache --seed 77 --json $R/gencache.json 2>&1 | tr '\n' ' ' | cut -c1-900)"
log "cache config lines: $(grep -iE 'prefix|hybrid|kv cache group|block_size|KVCacheSpec' $R/server.log | grep -v 'Avg prompt' | cut -c1-220 | head -8 | tr '\n' ' ')"
$S/stop.sh p7_gencache | tee -a $L
log "GENCACHE_DONE (server stopped; box idle)"
