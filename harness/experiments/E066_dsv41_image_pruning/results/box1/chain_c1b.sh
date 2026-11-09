#!/bin/bash
# E066: after the combined aggregation (pid 27768) and screen batch A (pid 27770) exit, select the candidate 1
# variants on both harnesses (image weight 0 / 1 in layers 20-39) and screen them into the same file.
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
while kill -0 27768 2>/dev/null || kill -0 27770 2>/dev/null; do sleep 15; done
grep -q "\"requests\"" results/agg_both.log || { echo "aggregation failed"; tail -5 results/agg_both.log; exit 1; }
for w in 0 1; do
  tvenv/bin/python $S/class_stats.py select --stats obs/stats_both.npz --w-enc 1,1,10,1 --w-dec 0,1,10,$w --avg 128 \
    --out-dir sel/both
done
bash $S/nll_masks_mm.sh results/c1_screen.jsonl \
  w0:sel/both/keep_contrib_enc1-1-10-1_dec0-1-10-0_uniform_avg128.npy \
  w1:sel/both/keep_contrib_enc1-1-10-1_dec0-1-10-1_uniform_avg128.npy
echo C1B_DONE
