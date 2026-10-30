#!/bin/bash
# Burst 1: NCCL env matrix + host all-reduce prototypes (under both GPU locks).
cd "$(dirname "$0")/.."
export LAB_TMP=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/tmp
export TORCH_EXTENSIONS_DIR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/torch_ext TORCH_CUDA_ARCH_LIST=8.6
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date
timeout 900 /home/vu/miniconda3/envs/vllm/bin/python scripts/host_ar_proto.py --out results/har_proto.json > results/har_proto.log 2>&1; echo "proto rc=$?"
date
scripts/run_ar_matrix.sh
date
