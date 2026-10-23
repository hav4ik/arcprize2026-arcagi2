#!/bin/bash
# E043: upload the analysis artifacts to the user's private HF dataset (token in $HF_TOKEN, from ssh stdin).
set -u
cd /workspace/dsv4
R=chankhavu/dsv41-flash1017-e043-pruning-artifacts
U="tvenv/bin/python code/scripts/hf_upload.py --repo $R --type dataset"
S=hf_stage; rm -rf $S; mkdir -p $S/engram $S/reap $S/results $S/code
cp data/engram/engram_stats.json data/engram_geometry.npz $S/engram/
cp -r data/engram/masks $S/engram/masks_full_data; cp -r data/engram/masks_half $S/engram/masks_half_data 2>/dev/null
cp data/reap/stats_*.npz $S/reap/; cp -r data/reap/keep_all data/reap/keep_pilot $S/reap/
cp data/nll_*.jsonl $S/results/ 2>/dev/null; cp logs/half_census.log $S/results/ 2>/dev/null
cp -r code/scripts code/patches $S/code/
cat > $S/README.md <<'MD'
# E043 DeepSeek-V4.1-Flash-1017 pruning artifacts (private)

From experiment E043 (repo hav4ik/arcprize2026-arcagi2, havu/experiments/E043_dsv41_real_pruning):
- `engram/`: row-lookup census geometry, report stats, top-frequency row masks (packbits per Engram layer).
- `engram_counts/`: raw per-row lookup counts (uint32, per layer and input role) on 251.8M calibration tokens.
- `reap/`: REAP sufficient statistics (`stats_*.npz`: N, A, Q, W, MX per layer x input role x target role x expert)
  and keep masks (`keep_*/keep_<rule>_e<E>.npy`, bool [40, 384]).
- `tok/`: rebuilt token streams (uint32) + roles of the calibration rollouts and the eval-task traces.
- `obs/`: raw observer step files (per token: input id, position, 40 x 6 expert ids, router weights, native amplitudes).
- `results/`: NLL candidate results; `code/`: scripts and vLLM-fork patches as run.
MD
$U --path $S --message "E043 analysis artifacts"
$U --path data/engram/counts_calib.npz --in-repo engram_counts
$U --path data/engram/counts_half.npz --in-repo engram_counts
$U --path data/tok --in-repo tok/calib --message "calibration token streams"
$U --path data/tok_eval --in-repo tok/eval_e030_s20x60hdr
$U --path data/tok_eval6 --in-repo tok/eval_flash1017_6runs
$U --path /dev/shm/e043/obs/pilot --in-repo obs/pilot --message "observer pilot"
$U --path /dev/shm/e043/obs/main --in-repo obs/main --message "observer main"
echo UPLOAD_DONE
