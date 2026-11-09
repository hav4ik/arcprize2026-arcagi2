#!/bin/bash
# E062 L4: B32 itself on this pod (same-pod check against E055's candBf_x2t): candBf_x2t's exact settings and workload
# (launch_candBf_x2t.sh), B32's stack = /root/dsv41/venv with the 16-file overlay (b4090/b32_after_install.sh), kit_b.
# Run for a ~75-90 min window, then stop with SIGTERM to the solver (final write, server stopped).
cd /home/vu/Workspace/ARC-AGI-2/experiments/E055_kaggle_replica
REPLICA_LOCAL_DIR=/home/vu/Workspace/ARC-AGI-2/experiments/E062_manh_vllm_fork/l4/runs/${1:-b32pod_x2t} \
exec toolkit/replica_run.sh 19843 213.173.105.17 ${1:-b32pod_x2t} evalx2t_s2026 \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/dsv41/venv \
  PATH=/root/dsv41/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=56 \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}"]'
