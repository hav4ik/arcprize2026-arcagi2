#!/bin/bash
# E042 (Runpod, 2026-10-24 06:00): replaces run_seq5.sh's tail. run_ds16.sh (running) ends with run_shape_rest.sh (now
# the util-0.93 replan); the E64 x 64 tool anchor is dropped. Usage: run_seq6.sh <run_ds16.sh pid>
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
