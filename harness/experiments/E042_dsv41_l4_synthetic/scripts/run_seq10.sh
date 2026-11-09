#!/bin/bash
# E042 (Runpod, 12:40): after run_seq9.sh (run_rest4.sh), run_tool3.sh. Usage: run_seq10.sh <run_seq9.sh pid>
while kill -0 $1 2>/dev/null; do sleep 10; done
/root/dsv41/scripts/run_tool3.sh
echo "[$(date -u +%H:%M:%S)] SEQ10_DONE (box idle)" >> /root/dsv41/runs/t72.log
