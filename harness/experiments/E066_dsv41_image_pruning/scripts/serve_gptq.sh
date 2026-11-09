#!/bin/bash
# E066: serve a 2-bit candidate (full checkpoint + runtime keep mask + precomputed GPTQ experts) with the vision tower,
# as the tool E128 eval server (E043 chain_t2): Humming uint2 g128 gate/up, g64 down, no expert parallel, util 0.86.
# Usage: serve_gptq.sh <run_name> <keep.npy> <gptq_dir>   env: GPUS (0,1,2,3), PORT (18001), CTRL (/dev/shm/e043/ctrlB.json),
#        MAXLEN (262144: the image harness's chat path always asks for 64K output tokens, so prompts past ~67K
#        tokens fail at 131072; the same limit for every E066 eval server), SEQS (192)
set -eu
D=/workspace/dsv4
NAME=$1; K=$(realpath "$2"); O=$(realpath "$3")
CTRL=${CTRL:-/dev/shm/e043/ctrlB.json}
mkdir -p $(dirname $CTRL)
echo "{\"engram_keep\": \"$D/data/e066/engram/e043tool_top64gib.npz\", \"expert_keep\": \"$K\", \"observe\": null}" > $CTRL
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR=$O
VISION=1 GPUS=${GPUS:-0,1,2,3} TP=4 EP=0 PORT=${PORT:-18001} CTRL=$CTRL UTIL=${UTIL:-0.86} SEQS=${SEQS:-192} MBT=8192 \
  MAXLEN=${MAXLEN:-262144} \
  MODEL=$D/DeepSeek-V4.1-Flash-1017 bash $D/repo/experiments/E043_dsv41_real_pruning/scripts/serve.sh "$NAME" \
  --limit-mm-per-prompt '{"image": 64}'
