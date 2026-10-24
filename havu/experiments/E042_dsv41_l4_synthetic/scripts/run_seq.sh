#!/bin/bash
# E042 (Runpod, 2026-10-24 02:05): the user's focus is the no-tool harness, so E72's no-tool snapshots run first, then
# DS1 (DSpark cost), then the E64 x 64 tool anchor. Replaces run_dspark.sh's order (stopped during its base server).
S=/root/dsv41/scripts
echo "nt:e72:28 nt:e72:32 nt:e72:24" > /root/dsv41/runs/t72.points; $S/run_e72t.sh
$S/run_dspark.sh
echo "e64:64" > /root/dsv41/runs/t72.points; $S/run_e72t.sh
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
