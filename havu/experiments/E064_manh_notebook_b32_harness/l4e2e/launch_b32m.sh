#!/bin/bash
# E064b L4 pod #2: the E055 replica of the submission candidate B32-M (kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m).
# - Code: B32's kit unchanged (/root/kit_b: solver 77fe80a3, harness 882c817f, train_check 8fe36175, auto-repair
#   2292369c / 571f47eb).
# - Environment: Manh's: Python 3.13 venv /root/b32m/venv with his vllm-wheels set (vllm 0.32.1rc1.dev0+g5e1bccd07c,
#   torch 2.14.0+cu130, ...) installed as his notebook cell 4 does, plus B32-M's plan B row-chunk patch (cell 2r).
# - Model: the Kaggle Model copy of E128 (row-chunked Engram), /root/dsv41/model-e128k, sha256 = kaggle_e128_manifest.
# - Settings: B32-M's cell 00 (80 workers, gate 0.85 + queue hold, multi-test 0.5, context 163840, seqs 96, the same
#   ARC_SERVER_EXTRA_ARGS / ARC_SERVER_ENV_JSON), util 0.938 = the pods' equivalent of Kaggle's 0.945.
# - Tool sandbox: the same 3.13 venv (on Kaggle the image's Python 3.13).
# Usage: launch_b32m.sh <name>
NAME=${1:?name}
cd /home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica
REPLICA_LOCAL_DIR=/home/vu/Workspace/ARC-AGI-2/experiments/E064_manh_notebook_b32_harness/l4e2e/runs/$NAME \
exec toolkit/replica_run.sh 11660 213.173.105.27 $NAME evalx2t_s2026 \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/b32m/venv \
  ARC_TOOL_PYTHON=/root/b32m/venv/bin/python \
  ARC_MODEL_SRC=/root/dsv41/model-e128k \
  CUDA_HOME=/usr/local/cuda-13.0 \
  PATH=/root/b32m/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=80 \
  ARC_MAX_NUM_SEQS=96 \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  'ARC_SERVER_ENV_JSON={"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_PLUGINS": ""}' \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--decode-context-parallel-size", "4", "--cp-kv-cache-interleave-size", "64", "--cpu-offload-gb", "1", "--cpu-offload-params", "embed_tokens", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 96}", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"eviction_policy\":\"lru\"}}"]'
