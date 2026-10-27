#!/bin/bash
# E055 final candidate B replica: next-candidate code + auto-repair + S15/S16/S17 (solver 77fe80a3), multi-test,
# interval 512, E054 indexer skip (#156), E051 KV offload 32 GiB private, x56 + KV gate 0.85 + hold on server queue;
# transformed eval x2 workload (evalx2t_s2026), toolkit deadlines (-600 s), util 0.938 (Kaggle's KV pool).
cd "$(dirname "$0")"
exec toolkit/replica_run.sh 43647 157.157.221.29 candBf_x2t evalx2t_s2026 ARC_CACHE_SALT=candBf_x2t_040739 \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/dsv41/venv_offload \
  PATH=/root/dsv41/venv_offload/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=56 \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}"]'
