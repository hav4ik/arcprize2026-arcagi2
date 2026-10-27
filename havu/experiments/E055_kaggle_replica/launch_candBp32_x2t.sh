#!/bin/bash
# E055 full candidate B+ replica (32 GiB): candidate B (pinned_candb code, E056 kit, 32 GiB offload, gate 0.85 + queue hold) +
# E051 overlay v2 scheduler (94b06f85, incremental offload keys) + 64 workers; evalx2t_s2026, toolkit deadlines.
# wd44 venv_candb = E056 kit installed via the notebook cells, then install_overlay_v2.sh.
cd "$(dirname "$0")"
exec toolkit/replica_run.sh 30092 213.173.105.28 candBp32_x2t evalx2t_s2026 \
  ARC_KIT=/root/kit_bp \
  ARC_VENV=/root/dsv41/venv_candb \
  PATH=/root/dsv41/venv_candb/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=64 \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}"]'
