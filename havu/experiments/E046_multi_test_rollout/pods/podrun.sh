#!/bin/bash
# E042/E046: the Kaggle notebook pipeline (kaggle/dsv41_tools/out/local_run.py) on a Runpod 4xL4 pod, against the
# box_install.sh venv with the kit's vLLM overlay. Eval set, submit-mode deadlines unless overridden.
# Usage: podrun.sh <name> [ENV=VALUE ...]   (e.g. ARC_WORKERS=44 ARC_MULTI_TEST=1 ARC_GPU_UTIL=0.938)
set -u
N=$1; shift
R=/root/runs/$N; mkdir -p $R/out $R/scratch
export ARC_SKIP_INSTALL=1 ARC_VENV=/root/dsv41/venv ARC_KIT=/root/kit ARC_MODEL_SRC=/root/dsv41/model-e128t \
  ARC_CHALLENGES=/root/data/arc-agi_evaluation_challenges.json ARC_SOLUTIONS=/root/data/arc-agi_evaluation_solutions.json \
  ARC_MODE=submit ARC_OUT=$R/out ARC_SCRATCH=$R/scratch ARC_TOOL_PYTHON=/root/toolenv/bin/python ARC_INPUT_ROOT=/root/none \
  ARC_MAX_NUM_SEQS=64 ARC_GPU_UTIL=0.945 ARC_CACHE_SALT=$N \
  ARC_SERVER_EXTRA_ARGS='["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}"]' \
  CUDA_HOME=/usr/local/cuda-13.0 PATH=/root/dsv41/venv/bin:/usr/local/cuda-13.0/bin:$PATH
for kv in "$@"; do export "$kv"; done
env | grep -E '^ARC_' | sort > $R/env.txt
cd $R && setsid nohup /usr/bin/python3 -u /root/local_run.py > $R/run.log 2>&1 < /dev/null &
echo "started $N: $R/run.log"
