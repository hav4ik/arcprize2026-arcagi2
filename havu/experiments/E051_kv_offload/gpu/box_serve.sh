#!/usr/bin/env bash
# Box wrapper for serve_dsv41_offload.sh on the Vast 4x4090 dev boxes: the environment of /root/dsv41/smoke_launch.sh
# (CUDA 13 toolkit for JIT, shared JIT caches) around the Kaggle server command + offload flags.
#
#   OFFLOAD_GIB=28 POLICY=checkpoint /root/e051/gpu/box_serve.sh      # offload on
#   OFFLOAD_GIB=0 /root/e051/gpu/box_serve.sh                          # baseline, no connector
#
# Runs from /root/e051/runs/<RUN> so logs and the PID file stay out of other agents' directories.
set -euo pipefail
C=/root/dsv41/cache
export MODEL_DIR=${MODEL_DIR:-/root/dsv41/model-e128t}
export VENV=${VENV:-/root/dsv41/venv_offload}
export VLLM_CACHE_ROOT=$C/vllm FLASHINFER_WORKSPACE_BASE=$C HF_HOME=$C/hf CUDA_CACHE_PATH=$C/nv
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=$VENV/bin:/usr/local/cuda-13.0/bin:$PATH
# Host RAM may be shared with other renters/agents: plan for at most half of it (user, 2026-10-26).
# Planned = Engram pinned CPU offload (~94 GiB) + KV offload budget + ~20 GiB process overhead.
PLAN=$(( 94 + ${OFFLOAD_GIB:-28} + 20 ))
TOTAL=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$PLAN" -le $(( TOTAL / 2 )) ] || { echo "planned host RAM ${PLAN} GiB > half of ${TOTAL} GiB"; exit 1; }
[ "$PLAN" -le "$AVAIL" ] || { echo "planned host RAM ${PLAN} GiB > MemAvailable ${AVAIL} GiB"; exit 1; }
RUN=${RUN:-$(date -u +%m%d_%H%M)_off${OFFLOAD_GIB:-28}_${POLICY:-checkpoint}}
mkdir -p /root/e051/runs/$RUN && cd /root/e051/runs/$RUN
exec bash /root/e051/gpu/serve_dsv41_offload.sh
