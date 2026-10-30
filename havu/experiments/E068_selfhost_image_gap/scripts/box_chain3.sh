#!/bin/bash
# E068 chain 3 (node 5, 2026-10-30): let the running ref_t06 probe (pid $1) finish, then the causal, vitdp and noprefix
# arms of box_run.sh.
cd /workspace/dsv4
S=repo/experiments/E068_selfhost_image_gap/scripts
while kill -0 $1 2>/dev/null; do sleep 20; done
tvenv/bin/python $S/perception_probe.py --score results/e068/probe_ref_t06.jsonl
echo "ARM_DONE ref_t06 $(date -u +%H:%M)"
bash $S/box_run.sh causal vitdp noprefix
