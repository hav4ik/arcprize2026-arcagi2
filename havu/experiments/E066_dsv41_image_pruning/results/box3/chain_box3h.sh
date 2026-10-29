#!/bin/bash
# E066 box 3: re-measure uniform 120 on this box (the 120 point came from box 1 and looks off next to the box 3 uniform 112)
cd /workspace/dsv4
while kill -0 20409 2>/dev/null; do sleep 15; done
bash repo/experiments/E066_dsv41_image_pruning/scripts/nll_masks_mm.sh results/c3_screen.jsonl u120b3:sel_box3/keep_contrib_enc1-1-10-1_dec0-1-10-1_uniform_avg120.npy
echo U120_DONE
