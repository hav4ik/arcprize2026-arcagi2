#!/usr/bin/env bash
# E056: start the E128 server for one arm on a 4x4090 dev box: the Kaggle command (E046 kaggle_bench server_cmd.json,
# as E051's serve_dsv41_offload.sh / E054's serve_e054.sh) with candidate B's flags: Humming, FULL CUDA graphs cap 64,
# max-num-seqs 64, prefix caching, retention interval 512, KV pool pinned to Kaggle's (21,490 blocks of 113,920 B),
# and the E051 OffloadingConnector (32 GiB private pinned host memory, checkpoint SWA policy).
#
#   ARM=candb   RUN=<name> gpu/serve_candb.sh   # venv_candb, DSV41_INDEXER_SKIP unset (#156 skip on)
#   ARM=offload RUN=<name> gpu/serve_candb.sh   # venv_candb, DSV41_INDEXER_SKIP=0 (old indexer kernel exactly; E054)
#   EXT=1: --worker-extension-cls e056_ext.E056Ext (KV-byte parity + indexer probe; adds no serving behaviour)
#   stop:  kill -- -$(cat /root/e056/runs/$RUN/server.pgid)      (never by process name: other agents share boxes)
# Env (defaults): VENV=/root/dsv41/venv_candb MODEL_DIR=/root/dsv41/model-e128t PORT=8100 KV_BLOCKS=21490 UTIL=0.92
# (E054's value on this box) OFFLOAD_BYTES=34359738368. Each arm has its own VLLM_CACHE_ROOT (E054 practice).
set -euo pipefail
ARM=${ARM:?set ARM=candb|offload}
case "$ARM" in candb) SKIP="" ;; offload) SKIP=0 ;; *) echo "ARM must be candb|offload"; exit 2 ;; esac
VENV=${VENV:-/root/dsv41/venv_candb}
MODEL_DIR=${MODEL_DIR:-/root/dsv41/model-e128t}
PORT=${PORT:-8100}
KV_BLOCKS=${KV_BLOCKS-21490}
UTIL=${UTIL:-0.92}
OFFLOAD_BYTES=${OFFLOAD_BYTES:-34359738368}
RUN=${RUN:?set RUN}
D=/root/e056/runs/$RUN
[ ! -e "$D" ] || { echo "$D exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || { echo "a GPU already has ${used} MiB in use: check for leftovers (by PID) first"; exit 1; }
# Host RAM is shared: plan for at most half. Engram pins ~94 GiB, the offload pool 4 x PowerOf2Ceil(bytes / 4)
# (PyTorch's pinned allocator rounds per rank; 32 GiB exactly -> 32 GiB), ~20 GiB process overhead.
PLAN=$(( 94 + OFFLOAD_BYTES / 1073741824 + 20 ))
TOTAL=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$PLAN" -le $(( TOTAL / 2 )) ] || { echo "planned host RAM ${PLAN} GiB > half of ${TOTAL} GiB"; exit 1; }
[ "$PLAN" -le "$AVAIL" ] || { echo "planned host RAM ${PLAN} GiB > MemAvailable ${AVAIL} GiB"; exit 1; }
SITE=$("$VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
grep -q DSV41_INDEXER_SKIP "$SITE/vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py" \
  || { echo "E054 indexer not installed in $VENV"; exit 1; }
grep -q SWA_STORE_CHECKPOINT "$SITE/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py" \
  || { echo "E051 offload files not installed in $VENV"; exit 1; }
mkdir -p "$D"

C=/root/dsv41/cache
export VLLM_CACHE_ROOT=$C/vllm_e056_$ARM FLASHINFER_WORKSPACE_BASE=$C HF_HOME=$C/hf CUDA_CACHE_PATH=$C/nv
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=$VENV/bin:/usr/local/cuda-13.0/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1
export DSV41_INDEXER_Q_PREDECODE=1
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export VLLM_SERVER_DEV_MODE=1                  # /reset_prefix_cache and /collective_rpc for the tests
unset PYTORCH_CUDA_ALLOC_CONF                  # expandable_segments is rejected with KV connectors
unset DSV41_INDEXER_SKIP DSV41_INDEXER_SKIP_CANDIDATES DSV41_INDEXER_TARGET_PROGRAMS DSV41_ENGRAM_WKV_TP
[ -n "$SKIP" ] && export DSV41_INDEXER_SKIP=$SKIP
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HF_API_TOKEN  # offline; keep tokens out of the server

KVT=$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%d,"offload_prompt_only":false,"swa_store_policy":"checkpoint","shared_region":false,"eviction_policy":"lru"}}' "$OFFLOAD_BYTES")
ARGS=(
  serve "$MODEL_DIR" --served-model-name dsv41 --host 127.0.0.1 --port "$PORT"
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840 --max-num-seqs 64
  --max-num-batched-tokens 2048 --gpu-memory-utilization "$UTIL"
  --enable-prefix-caching --prefix-cache-retention-interval 512
  --engram-config '{"cpu_offload":true}' --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details --linear-backend humming
  --compilation-config '{"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64}'
  --kv-transfer-config "$KVT"
)
[ -n "$KV_BLOCKS" ] && ARGS+=(--num-gpu-blocks-override "$KV_BLOCKS")
if [ "${EXT:-0}" = 1 ]; then
  export PYTHONPATH=/root/e056/gpu:/root/e056/e051/gpu
  ARGS+=(--worker-extension-cls e056_ext.E056Ext)
fi

{ echo "E056 ARM=$ARM DSV41_INDEXER_SKIP=${SKIP:-unset} UTIL=$UTIL KV_BLOCKS=${KV_BLOCKS:-none} OFFLOAD_BYTES=$OFFLOAD_BYTES EXT=${EXT:-0}"
  echo "starting: $VENV/bin/vllm ${ARGS[*]}"; } > "$D/server.log"
env | grep -E '^(DSV41_|VLLM_|E043_|CUDA_HOME|PYTORCH_|PYTHONPATH)' | sort > "$D/env.txt"
cd "$D"
setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" >> "$D/server.log" 2>&1 < /dev/null &
P=$!
echo "$P" > "$D/server.pid"
sleep 1
ps -o pgid= -p "$P" | tr -d ' ' > "$D/server.pgid"
echo "arm $ARM: pid $P pgid $(cat "$D/server.pgid"), dir $D"
