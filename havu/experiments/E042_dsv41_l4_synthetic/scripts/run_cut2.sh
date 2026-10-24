#!/bin/bash
# E042 (Runpod, 06:45): stops run_shape_rest.sh after ds_x96_n4_u93, before its E96 gptq uint3 point (replaced by
# run_rest2.sh's ds_x96_u93). Usage: run_cut2.sh <run_shape_rest.sh pid>
L=/root/dsv41/runs/lb.log
until sed -n "/sh rest start (util 0.93 replan)/,\$p" $L | grep -q -E "ds_x96_n4_u93 (health:|SERVE FAILED)|lb_e96_g3_u93|ABORT"; do sleep 2; done
kill $1 && echo "[$(date -u +%H:%M:%S)] run_cut2: stopped run_shape_rest.sh after ds_x96_n4_u93; next run_rest2.sh" >> $L
