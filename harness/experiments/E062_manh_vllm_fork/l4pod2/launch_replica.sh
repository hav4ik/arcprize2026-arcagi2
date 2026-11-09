#!/bin/bash
# E062 L4 pod #2 (Runpod 5z3srm3108wcwl): E055-toolkit replica of B32 (+ R1 when PAD is set), exactly candBf_x2t's
# settings (launch_candBf_x2t.sh): kit_b (solver 77fe80a3, harness 882c817f, train_check 8fe36175, auto-repair),
# B32's 16-file overlay (1a3a6679) in /root/dsv41/venv, x56 + KV gate 0.85 + queue hold, multi-test 0.5, context limit,
# retention 512, Humming, FULL graphs cap 64, E051 offload 32 GiB, util 0.938, workload evalx2t_s2026, toolkit deadlines.
# R1: humming_utils.py with DSV41_HUMMING_MOE_PAD (4dd6392b; unset = B32's 256,128) and PAD passed to the server env.
#   NAME=b32r1_x2t PAD=64,64 WORKERS=56 bash launch_replica.sh
#   NAME=b32_x2t_p2 WORKERS=56 bash launch_replica.sh            # baseline (restore B32's humming_utils.py first)
cd "$(dirname "$0")/../../E055_kaggle_replica"
NAME=${NAME:?}; WORKERS=${WORKERS:-56}
EXTRA=()
[ -n "${PAD:-}" ] && EXTRA+=(DSV41_HUMMING_MOE_PAD=$PAD)
REPLICA_LOCAL_DIR=$(cd "$OLDPWD" && pwd)/runs/$NAME exec toolkit/replica_run.sh 11660 213.173.105.27 "$NAME" evalx2t_s2026 \
  ARC_KIT=/root/kit_b \
  ARC_VENV=/root/dsv41/venv \
  PATH=/root/dsv41/venv/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  ARC_WORKERS=$WORKERS \
  ARC_MULTI_TEST=1 \
  ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 \
  ARC_KV_ADMIT_MAX=0.85 \
  ARC_KV_ADMIT_MIN_WORKERS=16 \
  ARC_ADMIT_MAX_WAITING=1 \
  ARC_AUTO_REPAIR=1 \
  "${EXTRA[@]}" \
  'ARC_SERVER_EXTRA_ARGS=["--linear-backend", "humming", "--compilation-config", "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": 64}", "--prefix-cache-retention-interval", "512", "--kv-transfer-config", "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":34359738368,\"offload_prompt_only\":false,\"swa_store_policy\":\"checkpoint\",\"shared_region\":false,\"eviction_policy\":\"lru\"}}"]'
