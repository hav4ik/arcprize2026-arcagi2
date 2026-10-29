#!/bin/bash
# E066 box 3: after chain_box3f (pid 18077), dev-loss screen of coarse-step variants at average 120 (fewer distinct
# expert counts, which may save Humming per-count memory on Kaggle: dsv4-infer).
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
M=sel_box3/keep_contrib_enc1-1-10-1_dec0-1-10-1
while kill -0 18077 2>/dev/null; do sleep 15; done
bash $S/nll_masks_mm.sh results/c3_screen.jsonl f72c152s16_120:${M}_calblend_f72c152s16_avg120.npy \
  f64c160s32_120:${M}_calblend_f64c160s32_avg120.npy
echo STEP_DONE
