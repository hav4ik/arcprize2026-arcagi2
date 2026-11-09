#!/bin/bash
# One vLLM decode config: e2e_one.sh LABEL TP [ENV=VAL ...]. Call under both GPU locks.
cd "$(dirname "$0")/.."
label=$1; tp=$2; shift 2
export LAB=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab
export TORCH_EXTENSIONS_DIR=$LAB/torch_ext TORCH_CUDA_ARCH_LIST=8.6
M=/home/vu/.cache/kagglehub/models/yeoyunsianggeremie/aimo-sft-7b/Transformers/default/6
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; date +%T
env "$@" timeout 900 /home/vu/miniconda3/envs/vllm/bin/python scripts/e2e_decode.py --model $M --tp $tp \
  --label $label --out results/e2e_$label.json > results/e2e_$label.log 2>&1
echo "$label rc=$? $(date +%T)"; grep "^$label {" results/e2e_$label.log
