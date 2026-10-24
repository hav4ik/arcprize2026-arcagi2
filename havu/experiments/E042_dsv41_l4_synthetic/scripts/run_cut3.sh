#!/bin/bash
# E042 (Runpod, 15:05): the user dropped T4's C = 48 point ("Yes, drop the 48 replay"): stop run_tool4.sh right after
# t4_x128t_c40 is stopped, and stop a t4_x128t_c48 server if one got started. Usage: run_cut3.sh <run_tool4.sh pid>
L=/root/dsv41/runs/t4.log
until grep -q -E "stopped t4_x128t_c40|ABORT|t4_x128t_c40 SERVE FAILED" $L; do sleep 0.2; done
kill $1 && echo "[$(date -u +%H:%M:%S)] run_cut3: stopped run_tool4.sh after t4_x128t_c40 (C = 48 dropped by the user)" >> $L
sleep 20
if grep -q "started t4_x128t_c48" $L; then
  /root/dsv41/scripts/stop.sh t4_x128t_c48 >> $L 2>&1
  echo "[$(date -u +%H:%M:%S)] run_cut3: stopped the t4_x128t_c48 server that had started" >> $L
fi
echo "[$(date -u +%H:%M:%S)] T4_CUT_DONE (box idle)" >> $L
