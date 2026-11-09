#!/bin/bash
# E042 (Runpod, 08:40): after run_seq7.sh (run_rest2.sh's E64 controls), run_rest3.sh. Usage: run_seq8.sh <run_seq7.sh pid>
while kill -0 $1 2>/dev/null; do sleep 10; done
/root/dsv41/scripts/run_rest3.sh
echo "[$(date -u +%H:%M:%S)] SEQ8_DONE (box idle)" >> /root/dsv41/runs/t72.log
