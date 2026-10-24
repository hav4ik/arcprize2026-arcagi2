#!/bin/bash
# E042 (Runpod, 2026-10-24 03:50): after DS1, the user's E128 question (run_lowbit.sh), then the E64 x 64 tool anchor.
# Replaces run_seq.sh's tail (stopped while DS1 ran). Usage: run_seq2.sh <run_dspark.sh pid>
S=/root/dsv41/scripts
while kill -0 $1 2>/dev/null; do sleep 20; done
$S/run_lowbit.sh
echo "e64:64" > /root/dsv41/runs/t72.points; $S/run_e72t.sh
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
