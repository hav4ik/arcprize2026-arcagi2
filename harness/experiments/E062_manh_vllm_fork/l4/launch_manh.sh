#!/bin/bash
# E062 L4: the E055 replica of Manh's fork (nguyen599/vllm b9ad670a67 wheel + patches 01/02 = branch arc/e128-kaggle
# e4988d15) with B32's solver kit unchanged (kit_b) and B32's settings, except the server stack:
#   ARC_VENV=/root/manh/venv; extra args: Humming dense, decode context parallel 4, FULL CUDA graphs up to SEQS,
#   upstream KV offload connector 32 GiB with offload_prompt_only=false (READY.md translation; B32's
#   swa_store_policy / shared_region keys dropped), DCP interleave 64 (must be a multiple of E128's candidate block 8
#   and divide block 128), retention interval 512 (the solver hard-codes 128; DCP4 needs a multiple of 512), and
#   Manh's capacity options: embed_tokens UVA (--cpu-offload-gb 1 --cpu-offload-params embed_tokens) and
#   VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64; server env also: driver reserve 512 MiB, GPU memory log every 300 s.
# Usage: launch_manh.sh <name> <workers> <max_num_seqs>
NAME=${1:?name}; W=${2:-80}; SEQS=${3:-96}
cd /home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica
REPLICA_LOCAL_DIR=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/l4/runs/$NAME \
exec toolkit/replica_run.sh 19843 213.173.105.17 $NAME evalx2t_s2026 \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/manh/venv \
  PATH=/root/manh/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=$W \
  ARC_MAX_NUM_SEQS=$SEQS \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  'ARC_SERVER_ENV_JSON={"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64"}' \
  "ARC_SERVER_EXTRA_ARGS=[\"--linear-backend\", \"humming\", \"--decode-context-parallel-size\", \"4\", \"--cp-kv-cache-interleave-size\", \"64\", \"--prefix-cache-retention-interval\", \"512\", \"--cpu-offload-gb\", \"1\", \"--cpu-offload-params\", \"embed_tokens\", \"--compilation-config\", \"{\\\"cudagraph_mode\\\": \\\"FULL\\\", \\\"max_cudagraph_capture_size\\\": $SEQS}\", \"--kv-transfer-config\", \"{\\\"kv_connector\\\":\\\"OffloadingConnector\\\",\\\"kv_role\\\":\\\"kv_both\\\",\\\"kv_connector_extra_config\\\":{\\\"cpu_bytes_to_use\\\":34359738368,\\\"offload_prompt_only\\\":false,\\\"eviction_policy\\\":\\\"lru\\\"}}\"]"
