#!/bin/bash
# E066 box 1: free-generation probe of the step-8 (cap 160) and step-16 (cap 152) floor-72 layouts on the native
# server (imgC) once it serves; same 50 dev prompts as the box 3 probes. Restores the control file afterwards.
cd /workspace/dsv4
for i in $(seq 1 80); do curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 && break; sleep 15; done
curl -s -m 5 http://127.0.0.1:18001/v1/models | grep -q dsv41 || { echo SERVER_NOT_UP; exit 1; }
MODE=gen bash repo/experiments/E066_dsv41_image_pruning/scripts/nll_masks_mm.sh results/gen_probe_box1.jsonl \
  s8_f72c160:cand/e72c160_keep.npy s16_f72c152:cand/e72c152_keep.npy
echo PROBE_DONE
