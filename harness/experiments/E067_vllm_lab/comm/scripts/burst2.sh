#!/bin/bash
# Burst 2: host all-reduce prototypes + NCCL repeat checks with PCIe link logging.
cd "$(dirname "$0")/.."
export LAB_TMP=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/tmp
export TORCH_EXTENSIONS_DIR=/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/vllmlab/torch_ext TORCH_CUDA_ARCH_LIST=8.6
P=/home/vu/miniconda3/envs/vllm/bin/python
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; date
nvidia-smi --query-gpu=timestamp,index,pcie.link.gen.current,pcie.link.width.current,clocks.sm,pstate --format=csv,noheader -lms 200 > results/pcie_burst2.csv &
SMI=$!
timeout 900 $P scripts/host_ar_proto.py --out results/har_proto.json > results/har_proto.log 2>&1; echo "proto rc=$?"; date
for rep in 2; do
  for cfg in "default" "p2p_disable NCCL_P2P_DISABLE=1" "proto_simple NCCL_PROTO=Simple"; do
    set -- $cfg; label=${1}_r$rep; shift
    env "$@" timeout 300 $P scripts/ar_bench.py --label $label --out results/ar_$label.json > results/ar_$label.log 2>&1; echo "$label rc=$?"
  done
done
kill $SMI; date
