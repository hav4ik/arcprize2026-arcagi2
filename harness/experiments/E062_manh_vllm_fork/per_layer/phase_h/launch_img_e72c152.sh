#!/bin/bash
# Phase H step 4: image-harness load run of the E72-152 avg120 candidate on L4 pod 2 (Manh's fork + 04c + router
# patch), with the E055 replica toolkit, B32-M's settings (as E064b's b32m_x2t) plus ARC_GRID_FORMAT=text+image and
# --limit-mm-per-prompt {"image": 64}. Kit /root/kit_img = kaggle_kit_v5 + kaggle/dsv41_tools/pinned_img (solver 5d75b1c4,
# harness 1d1a281e, grid_image 18ed78fb, train_check 8fe36175, auto-repair 2292369c / 571f47eb).
# Usage: launch_img_e72c152.sh <name>   (stop it by PID after ~75-90 min; the toolkit never kills anything)
NAME=${1:?name}
cd /home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica
REPLICA_LOCAL_DIR=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/per_layer/phase_h/runs/$NAME \
exec toolkit/replica_run.sh 18070 213.173.105.27 $NAME evalx2t_s2026 \
  ARC_KIT=/root/kit_img \
  ARC_VENV=/root/b32m/venv \
  ARC_TOOL_PYTHON=/root/b32m/venv/bin/python \
  ARC_MODEL_SRC=/root/models/e72c152 \
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
  ARC_GRID_FORMAT=text+image \
  'ARC_SERVER_ENV_JSON={"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_PLUGINS": ""}' \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--decode-context-parallel-size", "4", "--cp-kv-cache-interleave-size", "64", "--cpu-offload-gb", "1", "--cpu-offload-params", "embed_tokens", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 96}", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"eviction_policy\":\"lru\"}}", "--limit-mm-per-prompt", "{\"image\": 64}"]'
