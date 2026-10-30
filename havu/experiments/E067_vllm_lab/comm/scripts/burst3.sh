#!/bin/bash
# Burst 3: host-AR kernel parameter sweep (large sizes) + first vLLM TP2 decode configs.
cd "$(dirname "$0")/.."
export LAB_TMP=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/tmp
export TORCH_EXTENSIONS_DIR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/torch_ext TORCH_CUDA_ARCH_LIST=8.6
P=/home/vu/miniconda3/envs/vllm/bin/python
date +%T
for cfg in "t6 --threads 6" "t6w32 --threads 6 --writers 32 --readers 32" "t6w32u64 --threads 6 --writers 32 --readers 32 --unit-kb 64" "t2w4u4 --threads 2 --writers 4 --readers 4 --unit-kb 4"; do
  set -- $cfg; label=$1; shift
  timeout 300 $P scripts/host_ar_proto.py --paths kernel-warm --gap-us 0 "$@" --out results/har_sweep_$label.json > results/har_sweep_$label.log 2>&1; echo "sweep $label rc=$?"
done
date +%T
scripts/e2e_one.sh tp2_default 2
scripts/e2e_one.sh tp2_har_warm 2 HAR_ENABLE=1 PYTHONPATH=$PWD/scripts/har_vllm
