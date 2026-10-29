#!/bin/bash
# E062b: one fixed-window throughput run of the B32 solver (kit_b, B32 settings) on the E055 240-task workload
# (evalx2t_s2026) against Manh's fork on an L4 pod (E062_PORT / E062_IP; default pod #2 5z3srm3108wcwl, used on pod #1
# ddj7xv53e5rgre after the reassignment), via E055's replica_run.sh. Copied from
# b4090/ab_run.sh (fork arm), with pod #1's replica server config (l4/launch_manh.sh: DCP4, interleave 64,
# retention 512, embed_tokens UVA, Humming MoE pad 64,64, driver reserve 512 MiB, upstream offload 32 GiB with
# offload_prompt_only=false, FULL graphs up to SEQS). Model: the pod's default /root/dsv41/model-e128t (HF copy).
#   ab_run_pod2.sh <base|flags> <name> <window_min> <workers>        (E062_SEQS default 96)
#   flags  = the same + Manh's recommended env (host all-reduce MODE 60 + tuning, TRITON_SPARSE_INDEXER_LOGITS).
#   flags2 = the local merge arc/e128-kaggle-main5e1b (95376a1: Manh's main 5e1bccd07c + our branch) in /root/manh2/venv,
#            his flags with MODE 572 (= 512 warm AVX-512 reducers + 60) and VLLM_DSA_DECODE_INDEXER_ROW_SHARD=1.
#   lossless = flags2 with MODE 556 (= 572 - 16: no MODE_Q8, so the host all-reduce stays lossless bf16/fp32).
# Window: stop-new at window, force +4 min, hard stop +6, kill +8 (pod clock, no Kaggle shift).
set -euo pipefail
ARM=$1; NAME=$2; WIN=$3; WORKERS=$4; shift 4   # any further ENV=VAL go to the solver
ROOT=/home/vu/Workspace/ARC-AGI-2
T=$ROOT/experiments/E055_kaggle_replica/toolkit
PORT=${E062_PORT:-11660}; IP=${E062_IP:-213.173.105.27}
W=$(( WIN * 60 )); SEQS=${E062_SEQS:-96}; OFF=34359738368
case $ARM in
  base)  SENV='{"VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "300", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64"}' ;;
  flags) SENV='{"VLLM_HOST_ALLREDUCE": "1", "VLLM_HOST_ALLREDUCE_MODE": "60", "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16", "VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB": "36", "VLLM_HOST_ALLREDUCE_LARGE_SHAPE": "32x32x256", "VLLM_HOST_ALLREDUCE_SHAPE": "12x12x256", "VLLM_HOST_ALLREDUCE_UNIT_KB": "16", "VLLM_DCP_ASYNC_QUERY_GATHER": "1", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_TRITON_SPARSE_INDEXER_LOGITS": "1", "VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "60"}' ;;
  flags2) SENV='{"VLLM_HOST_ALLREDUCE": "1", "VLLM_HOST_ALLREDUCE_MODE": "572", "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16", "VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB": "36", "VLLM_HOST_ALLREDUCE_LARGE_SHAPE": "32x32x256", "VLLM_HOST_ALLREDUCE_SHAPE": "12x12x256", "VLLM_HOST_ALLREDUCE_UNIT_KB": "16", "VLLM_DCP_ASYNC_QUERY_GATHER": "1", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_TRITON_SPARSE_INDEXER_LOGITS": "1", "VLLM_DSA_DECODE_INDEXER_ROW_SHARD": "1", "VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "60"}'
          VENV=/root/manh2/venv ;;
  lossless) SENV='{"VLLM_HOST_ALLREDUCE": "1", "VLLM_HOST_ALLREDUCE_MODE": "556", "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16", "VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB": "36", "VLLM_HOST_ALLREDUCE_LARGE_SHAPE": "32x32x256", "VLLM_HOST_ALLREDUCE_SHAPE": "12x12x256", "VLLM_HOST_ALLREDUCE_UNIT_KB": "16", "VLLM_DCP_ASYNC_QUERY_GATHER": "1", "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64", "VLLM_TRITON_SPARSE_INDEXER_LOGITS": "1", "VLLM_DSA_DECODE_INDEXER_ROW_SHARD": "1", "VLLM_GPU_DRIVER_RESERVE_MB": "512", "VLLM_GPU_MEMORY_LOG_INTERVAL": "60"}'
          VENV=/root/manh2/venv ;;
  *) echo "arm: base | flags | flags2 | lossless"; exit 2 ;;
esac
VENV=${VENV:-/root/manh/venv}
REPLICA_LOCAL_DIR=$ROOT/experiments/E062_manh_vllm_fork/e062b/runs/$NAME REPLICA_SHIFT_S=0 \
exec "$T/replica_run.sh" "$PORT" "$IP" "$NAME" evalx2t_s2026 \
  ARC_KIT=/root/kit_b ARC_WORKERS=$WORKERS ARC_MAX_NUM_SEQS=$SEQS ARC_MULTI_TEST=1 ARC_MULTI_BUDGET=0.5 \
  ARC_CONTEXT_LIMIT=163840 ARC_KV_ADMIT_MAX=0.85 ARC_KV_ADMIT_MIN_WORKERS=16 ARC_ADMIT_MAX_WAITING=1 ARC_AUTO_REPAIR=1 \
  ARC_GPU_UTIL=0.938 ARC_STOP_NEW_S=$W ARC_FORCE_AT_S=$(( W + 240 )) ARC_HARD_STOP_S=$(( W + 360 )) \
  ARC_KILL_AT_S=$(( W + 480 )) \
  ARC_VENV=$VENV PATH=$VENV/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  "ARC_SERVER_ENV_JSON=$SENV" "$@" \
  "ARC_SERVER_EXTRA_ARGS=[\"--linear-backend\", \"humming\", \"--decode-context-parallel-size\", \"4\", \"--cp-kv-cache-interleave-size\", \"64\", \"--prefix-cache-retention-interval\", \"512\", \"--cpu-offload-gb\", \"1\", \"--cpu-offload-params\", \"embed_tokens\", \"--compilation-config\", \"{\\\"cudagraph_mode\\\": \\\"FULL\\\", \\\"max_cudagraph_capture_size\\\": $SEQS}\", \"--kv-transfer-config\", \"{\\\"kv_connector\\\":\\\"OffloadingConnector\\\",\\\"kv_role\\\":\\\"kv_both\\\",\\\"kv_connector_extra_config\\\":{\\\"cpu_bytes_to_use\\\":$OFF,\\\"offload_prompt_only\\\":false,\\\"eviction_policy\\\":\\\"lru\\\"}}\"]"
