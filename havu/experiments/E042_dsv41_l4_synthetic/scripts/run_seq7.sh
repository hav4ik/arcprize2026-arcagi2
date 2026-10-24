#!/bin/bash
# E042 (Runpod, 06:45): replaces run_seq6.sh. Waits for run_ds16.sh (whose run_shape_rest.sh run_cut2.sh stops), then
# run_rest2.sh. Usage: run_seq7.sh <run_ds16.sh pid>
while kill -0 $1 2>/dev/null; do sleep 10; done
/root/dsv41/scripts/run_rest2.sh
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
