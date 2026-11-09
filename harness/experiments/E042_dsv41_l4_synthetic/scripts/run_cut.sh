#!/bin/bash
# E042 (Runpod, 04:20): stop run_shape.sh after its E72 util-0.93 point, so run_seq5.sh moves on to run_ds16.sh (the
# low-bit + DSpark points, then run_shape_rest.sh: E96, E160, E64 controls). Usage: run_cut.sh <run_shape.sh pid>
L=/root/dsv41/runs/lb.log
until grep -q -E "sh_e72_util93 health:|sh_e72_util93 SERVE FAILED|ABORT" $L; do sleep 3; done
kill $1 2>/dev/null && echo "[$(date -u +%H:%M:%S)] run_cut: stopped run_shape.sh after sh_e72_util93; next run_ds16.sh, then run_shape_rest.sh" >> $L
