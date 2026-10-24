#!/bin/bash
# E042: thermal check, then the decode grid with NCCL_P2P_LEVEL=SYS (box-only comm ceiling). Log: runs/p2p.log
S=/root/dsv41/scripts; L=/root/dsv41/runs/p2p.log
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L; }
log "thermal check: $(/root/dsv41/venv/bin/python $S/gpu_burn_check.py 60 2>&1 | tr '\n' ';')"
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active --format=csv,noheader >> $L
P2P=SYS SEQS=64 $S/serve.sh s3_p2p >> $L
(setsid nohup $S/gpumon.sh s3_p2p > /dev/null 2>&1 < /dev/null &)
R=/root/dsv41/runs/s3_p2p; t=0
until curl -s -m 3 localhost:8100/v1/models > /dev/null; do
  sleep 10; t=$((t + 10))
  if grep -q Traceback $R/server.log || ! kill -0 $(cat $R/server.pid) 2>/dev/null || [ $t -gt 2700 ]; then log "SERVE FAILED"; exit 1; fi
done
log "s3_p2p up after ${t}s: $(grep -o 'GPU KV cache size: [0-9,]* tokens' $R/server.log | tail -1); $(grep -c 'via P2P' $R/server.log) P2P channel lines"
$S/dw_grid.sh s3_p2p "1024:1 1024:8 1024:32 1024:64 8192:1 8192:8 8192:32 32768:1 32768:8"
log "P2P_DONE (s3_p2p server left running)"
