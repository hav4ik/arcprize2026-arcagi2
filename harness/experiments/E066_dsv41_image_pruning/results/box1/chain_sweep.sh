#!/bin/bash
# E066: candidate 1 (image weight 1) GPTQ capture, then the candidate 2 layer sweep (ref 128, cut 64, w1 weights).
set -u
cd /workspace/dsv4
S=repo/experiments/E066_dsv41_image_pruning/scripts
C1=sel/both/keep_contrib_enc1-1-10-1_dec0-1-10-1_uniform_avg128.npy
mkdir -p cand && cp $C1 cand/c1_keep.npy
bash $S/gptq_candidate.sh dump c1 cand/c1_keep.npy || { echo DUMP_FAILED; exit 1; }
echo C1_DUMP_DONE
tvenv/bin/python $S/class_stats.py sweep-masks --stats obs/stats_both.npz --w-enc 1,1,10,1 --w-dec 0,1,10,1 \
  --ref 128 --cut 64 --out-dir sel/sweep
SPECS="base:sel/sweep/keep_base.npy base2:sel/sweep/keep_base.npy"
for l in $(seq -w 0 39); do SPECS="$SPECS l$l:sel/sweep/keep_l$l.npy"; done
NSEQ=8 MAXLEN=40000 SEED=1 bash $S/nll_masks_mm.sh results/sweep.jsonl $SPECS
tvenv/bin/python $S/class_stats.py sens --nll results/sweep.jsonl --ref 128 --cut 64 --out results/sens.json
echo SWEEP_DONE
