#!/bin/bash
# E042 (Runpod, 2026-10-24 03:50): LB with E128 first (user: "real quick"), then DS1's d16 points, then the E64 x 64
# tool anchor. Replaces run_seq3.sh (its LB start collided with a dying server).
S=/root/dsv41/scripts
$S/run_lowbit.sh
$S/run_ds16.sh
echo "e64:64" > /root/dsv41/runs/t72.points; $S/run_e72t.sh
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
