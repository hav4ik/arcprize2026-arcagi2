#!/bin/bash
# E042 (Runpod, 2026-10-24 03:45): the user wants the low-bit throughput "real quick", so after DS1's E72 d32 point
# stop run_dspark.sh (its d16 points move to run_ds16.sh), run LB, then the d16 points, then the E64 x 64 anchor.
# Usage: run_seq3.sh <run_dspark.sh pid>
S=/root/dsv41/scripts; D=/root/dsv41/runs/ds.log
until grep -q -E "ds_e72_d32_n2 (no-tools snapshot|SERVE FAILED)|ds_e72_d32_n2: no snapshot" $D; do sleep 3; done
kill $1 2>/dev/null && echo "[$(date -u +%H:%M:%S)] run_seq3: stopped run_dspark.sh after d32 (d16 points -> run_ds16.sh after LB)" >> $D
for i in $(seq 60); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 5; done
$S/stop.sh ds_e72_d32_n2 >> $D
$S/run_lowbit.sh
$S/run_ds16.sh
echo "e64:64" > /root/dsv41/runs/t72.points; $S/run_e72t.sh
echo "[$(date -u +%H:%M:%S)] SEQ_DONE (box idle)" >> /root/dsv41/runs/t72.log
