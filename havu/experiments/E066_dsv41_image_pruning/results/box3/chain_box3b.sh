#!/bin/bash
# E066 box 3: replaces chain_box3.sh after its screen (pid 11547): free-generation probe only; no GPTQ capture until
# the user picks the dynamic-count candidate.
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
M=sel_box3/keep_contrib_enc1-1-10-1_dec0-1-10-1
while kill -0 11547 2>/dev/null; do sleep 15; done
echo SCREEN_DONE
MODE=gen NSEQ=25 bash $S/nll_masks_mm.sh results/gen_probe.jsonl full:none c1:${M}_uniform_avg128.npy \
  f96_120:${M}_calblend_f96c160s8_avg120.npy f72_120:${M}_calblend_f72c160s8_avg120.npy \
  f72c224_120:${M}_calblend_f72c224s8_avg120.npy
echo GEN_DONE
