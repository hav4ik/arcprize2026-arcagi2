#!/bin/bash
# E042 sweep chain: FULL-CUDA-graph server (B<=64), smoke, then TPS vs concurrency at three prompt lengths.
# Usage: run_sweeps.sh <run_name>   (env passed through to serve.sh: SEQS, MBT, UTIL, MAXLEN, ...)
# Log: /root/dsv41/runs/<run_name>/chain.log
set -u
NAME=$1; shift; R=/root/dsv41/runs/$NAME; S=/root/dsv41/scripts; mkdir -p $R
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $R/chain.log; }
log "chain start: SEQS=${SEQS:-64} MBT=${MBT:-2048} UTIL=${UTIL:-0.85} EAGER=${EAGER:-0}"
if curl -s -m 3 localhost:${PORT:-8100}/v1/models > /dev/null || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then
  log "ABORT: port ${PORT:-8100} answers or GPUs are in use"; exit 1
fi
SEQS=${SEQS:-64} $S/serve.sh $NAME "$@" >> $R/chain.log
(setsid nohup $S/gpumon.sh $NAME > /dev/null 2>&1 < /dev/null &)
t=0
until curl -s -m 3 localhost:${PORT:-8100}/v1/models > /dev/null; do
  sleep 10; t=$((t + 10))
  if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then
    log "SERVE FAILED after ${t}s: $(grep -E 'Error' $R/server.log | grep -v Warning | tail -2 | cut -c1-300)"; exit 1
  fi
done
log "server up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1); $(grep -oE 'Graph capturing finished in [0-9.]+ secs, took [0-9.-]+ GiB' $R/server.log | tail -1)"
log "smoke: $($S/smoke_request.sh 2>&1 | tr '\n' ' ' | cut -c1-400)"
log "sweep A in1024/out512"; $S/bench.sh $NAME 1024 512 1 2 4 8 16 32 64 | while read l; do log "  A $l"; done
log "sweep B in8192/out512"; $S/bench.sh $NAME 8192 512 1 8 16 32 | while read l; do log "  B $l"; done
log "sweep C in32768/out256"; $S/bench.sh $NAME 32768 256 1 4 8 16 | while read l; do log "  C $l"; done
log "CHAIN_DONE"
