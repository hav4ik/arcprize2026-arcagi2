#!/bin/bash
# E042 (Runpod, 11:20): after run_tool2.sh, run_rest4.sh. Usage: run_seq9.sh <run_tool2.sh pid>
while kill -0 $1 2>/dev/null; do sleep 10; done
/root/dsv41/scripts/run_rest4.sh
echo "[$(date -u +%H:%M:%S)] SEQ9_DONE (box idle)" >> /root/dsv41/runs/t72.log
