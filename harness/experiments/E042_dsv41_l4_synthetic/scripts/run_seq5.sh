#!/bin/bash
# E042 (Runpod, 2026-10-24 04:00): after LB's E128 uint2 point, the user's target shape first (run_shape.sh: mixed
# 2/3-bit E104 and E128 uint2, both with the 30% Engram table, then the rest of LB), then DS1's d16 points, then the
# E64 x 64 tool anchor. Usage: run_seq5.sh <run_seq4.sh pid> <run_lowbit.sh pid>
S=/root/dsv41/scripts; L=/root/dsv41/runs/lb.log
until sed -n "/restarted by run_seq4/,\$p" $L | grep -q -E "lb_e128_g2 health:|ABORT"; do sleep 3; done
kill $1 $2 2>/dev/null && echo "[$(date -u +%H:%M:%S)] run_seq5: stopped run_seq4/run_lowbit after lb_e128_g2; next run_shape.sh" >> $L
$S/run_shape.sh
$S/run_ds16.sh
echo "e64:64" > /root/dsv41/runs/t72.points; $S/run_e72t.sh
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
