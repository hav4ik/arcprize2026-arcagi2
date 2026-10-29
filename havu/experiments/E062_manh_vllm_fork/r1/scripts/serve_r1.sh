#!/usr/bin/env bash
# E062 R1: start B32's E128 server (E056 serve_candb.sh's Kaggle command: Humming, FULL graphs cap 64, max-num-seqs 64,
# prefix caching, retention 512, E051 offload 32 GiB) with the R1 humming_utils.py, which reads DSV41_HUMMING_MOE_PAD
# ("N,K"; unset = B32's 256,128). No --num-gpu-blocks-override: the KV pool is whatever util leaves, so the R1 gain shows.
#
#   PAD=64,64 RUN=<name> UTIL=0.938 bash serve_r1.sh      # PAD unset or "" = B32's default padding
#   stop: kill -TERM -- -$(cat /root/e062/runs/<name>/server.pgid)   (by process group, never by name)
set -euo pipefail
VENV=${VENV:-/root/dsv41/venv}
MODEL_DIR=${MODEL_DIR:-/root/dsv41/model-e128t}
PORT=${PORT:-8100}
UTIL=${UTIL:-0.938}
OFFLOAD_BYTES=${OFFLOAD_BYTES:-34359738368}
MAXSEQS=${MAXSEQS:-64}
CAP=${CAP:-64}
RUN=${RUN:?set RUN}
D=/root/e062/runs/$RUN
[ ! -e "$D" ] || { echo "$D exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || { echo "a GPU already has ${used} MiB in use: check for leftovers (by PID) first"; exit 1; }
SITE=$("$VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
grep -q DSV41_HUMMING_MOE_PAD "$SITE/vllm/model_executor/layers/quantization/utils/humming_utils.py" \
  || { echo "R1 humming_utils.py not installed in $VENV"; exit 1; }
grep -q SWA_STORE_CHECKPOINT "$SITE/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py" \
  || { echo "E051 offload files not installed in $VENV"; exit 1; }
mkdir -p "$D"
C=/root/dsv41/cache
export VLLM_CACHE_ROOT=$C/vllm_e062_${RUN} FLASHINFER_WORKSPACE_BASE=$C HF_HOME=$C/hf CUDA_CACHE_PATH=$C/nv
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=$VENV/bin:/usr/local/cuda-13.0/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export DSV41_INDEXER_Q_PREDECODE=1 MAX_JOBS=8
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export VLLM_SERVER_DEV_MODE=1                  # /reset_prefix_cache for the cold probes
unset PYTORCH_CUDA_ALLOC_CONF NCCL_P2P_LEVEL VLLM_MARLIN_INPUT_DTYPE
unset DSV41_INDEXER_SKIP DSV41_INDEXER_SKIP_CANDIDATES DSV41_INDEXER_TARGET_PROGRAMS DSV41_ENGRAM_WKV_TP DSV41_HUMMING_MOE_PAD
[ -n "${PAD:-}" ] && export DSV41_HUMMING_MOE_PAD=$PAD
for v in $(env | cut -d= -f1 | grep -E '^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN|HF_API_TOKEN|VLLM_MODEL|VLLM_ARGS|VLLM_TEST_ENDPOINT)$'); do unset "$v"; done
KVT=$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":%d,"offload_prompt_only":false,"swa_store_policy":"checkpoint","shared_region":false,"eviction_policy":"lru"}}' "$OFFLOAD_BYTES")
ARGS=(
  serve "$MODEL_DIR" --served-model-name dsv41 --host 127.0.0.1 --port "$PORT"
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840 --max-num-seqs "$MAXSEQS"
  --max-num-batched-tokens 2048 --gpu-memory-utilization "$UTIL"
  --enable-prefix-caching --prefix-cache-retention-interval 512
  --engram-config '{"cpu_offload":true}' --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details --linear-backend humming
  --compilation-config "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": $CAP}"
  --kv-transfer-config "$KVT"
)
{ echo "E062 R1 RUN=$RUN PAD=${PAD:-default(256,128)} UTIL=$UTIL OFFLOAD_BYTES=$OFFLOAD_BYTES MAXSEQS=$MAXSEQS CAP=$CAP"
  echo "humming_utils sha: $(sha256sum "$SITE/vllm/model_executor/layers/quantization/utils/humming_utils.py" | cut -c1-8)"
  echo "starting: $VENV/bin/vllm ${ARGS[*]}"; } > "$D/server.log"
env | grep -E '^(DSV41_|VLLM_|E043_|CUDA_HOME|PYTORCH_)' | sort > "$D/env.txt"
cd "$D"
setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" >> "$D/server.log" 2>&1 < /dev/null &
P=$!
echo "$P" > "$D/server.pid"
sleep 1
ps -o pgid= -p "$P" | tr -d ' ' > "$D/server.pgid"
echo "R1 $RUN: pid $P pgid $(cat "$D/server.pgid"), dir $D"
