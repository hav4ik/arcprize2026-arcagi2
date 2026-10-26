#!/usr/bin/env bash
# DSv4.1 E128 server = the Kaggle command (E046 kaggle_bench/dsv41_logs/server_cmd.json) + KV CPU offload.
#
#   MODEL_DIR=/data/dsv41-flash-1017-arc-tool-e128-2bit VENV=/opt/dsv41/venv \
#   OFFLOAD_GIB=28 POLICY=checkpoint gpu/serve_dsv41_offload.sh
#
# OFFLOAD_GIB=0 starts the unchanged baseline (no connector at all).
# KV_BLOCKS pins the GPU pool to Kaggle's L4 size (2.28 GiB/GPU / 113,920 B = ~21,490 blocks,
# logged there as "GPU KV cache size: 1,554,183 tokens"); empty = use --gpu-memory-utilization.
# Never start it with a name another agent's pkill pattern matches; stop it by PID only.
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR}" "${VENV:?set VENV}"
PORT=${PORT:-8100}
OFFLOAD_GIB=${OFFLOAD_GIB:-28}
POLICY=${POLICY:-checkpoint}          # checkpoint | all
SHARED_REGION=${SHARED_REGION:-true}  # false: private pinned tensors instead of /dev/shm
KV_BLOCKS=${KV_BLOCKS:-21490}
LOG=${LOG:-server_offload_${OFFLOAD_GIB}g_${POLICY}.log}

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1
export DSV41_INDEXER_Q_PREDECODE=1
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export VLLM_SERVER_DEV_MODE=1          # enables POST /reset_prefix_cache for the parity test
unset PYTORCH_CUDA_ALLOC_CONF          # expandable_segments is rejected with KV connectors

ARGS=(
  serve "$MODEL_DIR" --served-model-name dsv41 --host 127.0.0.1 --port "$PORT"
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840 --max-num-seqs 64
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.945
  --enable-prefix-caching --prefix-cache-retention-interval 128
  --engram-config '{"cpu_offload":true}' --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details --linear-backend humming
  --compilation-config '{"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64}'
)
[ -n "$KV_BLOCKS" ] && ARGS+=(--num-gpu-blocks-override "$KV_BLOCKS")
# EXTRA_ARGS: whitespace-separated extra server flags for test runs (e.g. --worker-extension-cls e051_ext.E051Ext).
if [ -n "${EXTRA_ARGS:-}" ]; then read -ra X <<< "$EXTRA_ARGS"; ARGS+=("${X[@]}"); fi

if [ "$OFFLOAD_GIB" != "0" ]; then
  BYTES=$(( OFFLOAD_GIB * 1024 * 1024 * 1024 ))
  KVT=$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%d,"offload_prompt_only":false,"swa_store_policy":"%s","shared_region":%s,"eviction_policy":"lru"}}' \
        "$BYTES" "$POLICY" "$SHARED_REGION")
  ARGS+=(--kv-transfer-config "$KVT")
  if [ "$SHARED_REGION" = "true" ]; then
    need=$(( BYTES / 1024 / 1024 / 1024 + 1 ))
    have=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
    [ "$have" -ge "$need" ] || { echo "/dev/shm has ${have}G < ${need}G: set SHARED_REGION=false"; exit 1; }
  fi
fi

echo "starting: ${ARGS[*]}" > "$LOG"
nohup "$VENV/bin/vllm" "${ARGS[@]}" >> "$LOG" 2>&1 &
echo $! > "${LOG%.log}.pid"
echo "pid $(cat "${LOG%.log}.pid"), log $LOG"
