#!/bin/bash
# E042 (Runpod, 14:20): after run_seq10.sh (T3), run_tool4.sh. Usage: run_seq11.sh <run_seq10.sh pid>
while kill -0 $1 2>/dev/null; do sleep 10; done
/root/dsv41/scripts/run_tool4.sh
echo "[$(date -u +%H:%M:%S)] SEQ11_DONE (box idle)" >> /root/dsv41/runs/t72.log
