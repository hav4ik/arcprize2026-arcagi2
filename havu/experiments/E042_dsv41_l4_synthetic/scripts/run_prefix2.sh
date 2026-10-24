#!/bin/bash
# E042 P4 rerun: the first cache-on C=32 replay reused seeds 0-15 from the C=16 replay just before it, so half its
# attempts hit that run's cache. Re-run C=32 (cache on) with seed 7 (attempt prompts Random(7000+i), unused so far).
S=/root/dsv41/scripts; L=/root/dsv41/runs/prefix.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
R=/root/dsv41/runs/p4_pc_on2
curl -s -m 3 localhost:8100/v1/models > /dev/null && { log "ABORT rerun: port busy"; exit 1; }
SEQS=64 $S/serve.sh p4_pc_on2 --enable-prefix-caching >> $L
(setsid nohup $S/gpumon.sh p4_pc_on2 > /dev/null 2>&1 < /dev/null &)
t=0; until curl -s -m 3 localhost:8100/v1/models > /dev/null; do sleep 10; t=$((t + 10));
  if grep -q Traceback $R/server.log || [ $t -gt 2700 ]; then log "rerun SERVE FAILED"; $S/stop.sh p4_pc_on2; exit 1; fi; done
log "p4_pc_on2 replay C32 seed7: $($PY $S/prefix_probe.py replay --conc 32 --seed 7 --json $R/replay_c32_s7.json 2>&1 | tail -1 | cut -c1-900)"
$S/stop.sh p4_pc_on2 | tee -a $L
log "PREFIX2_DONE (server stopped; box idle)"
