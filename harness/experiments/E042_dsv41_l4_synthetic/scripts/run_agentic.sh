#!/bin/bash
# E042 P5: ARC-tool-shaped replay with prefix caching on: 3K prompt, 12 turns of 1024 generated + 400 tool tokens
# (final context ~19K), at C=32 and C=64, fresh seeds. Starts after the P4 rerun. Log: /root/dsv41/runs/agentic.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/agentic.log; PY=/root/dsv41/venv/bin/python
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
until grep -qE "PREFIX2_DONE|ABORT rerun|rerun SERVE FAILED" /root/dsv41/runs/prefix.log; do sleep 30; done
curl -s -m 3 localhost:8100/v1/models > /dev/null && { log "ABORT: port busy"; exit 1; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ] && { log "ABORT: GPUs busy"; exit 1; }
R=/root/dsv41/runs/p5_agentic
SEQS=64 $S/serve.sh p5_agentic --enable-prefix-caching >> $L
(setsid nohup $S/gpumon.sh p5_agentic > /dev/null 2>&1 < /dev/null &)
t=0; until curl -s -m 3 localhost:8100/v1/models > /dev/null; do sleep 10; t=$((t + 10));
  if grep -q Traceback $R/server.log || [ $t -gt 2700 ]; then log "SERVE FAILED"; $S/stop.sh p5_agentic; exit 1; fi; done
log "p5_agentic up: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1)"
for c in 32 64; do
  log "replay C$c: $($PY $S/prefix_probe.py replay --conc $c --turns 12 --base 3000 --gen 1024 --tool 400 --seed $((10 + c)) --json $R/replay_c$c.json 2>&1 | tail -1 | cut -c1-1200)"
done
$S/stop.sh p5_agentic | tee -a $L
log "AGENTIC_DONE (server stopped; box idle)"
