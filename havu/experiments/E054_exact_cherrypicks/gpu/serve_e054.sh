#!/usr/bin/env bash
# E054: start the E128 server for one A/B arm on a 4x4090 dev box: the Kaggle command (E046
# kaggle_bench/dsv41_logs/server_cmd.json; same flags as E051's serve script) + the E054 kill switches.
#
#   ARM=old  gpu/serve_e054.sh     # DSV41_INDEXER_SKIP=0 DSV41_ENGRAM_WKV_TP=0  (behaves as the kit overlay)
#   ARM=idx  gpu/serve_e054.sh     # #156 only                     (SKIP=1, WKV_TP=0)
#   ARM=both gpu/serve_e054.sh     # #156 + #58678                  (SKIP=1, WKV_TP=1)
#   ARM=eng  gpu/serve_e054.sh     # #58678 only                    (SKIP=0, WKV_TP=1)
#   stop:    kill -- -$(cat $RUNDIR/server.pgid)          (never by process name: other agents share boxes)
#
# Needs gpu/install_e054.sh first (the switches only exist in the E054 files; ARM=old on the kit files is the
# same as no E054 at all). Env (defaults): VENV=/root/dsv41/venv MODEL_DIR=/root/dsv41/model-e128t PORT=8100
# KV_BLOCKS=21490 (Kaggle's pool, 113,920-B blocks; empty = size by UTIL) UTIL=0.92 (E050's box value)
# RUN=<name> (dir /root/e054/runs/$RUN). WAIT=1 blocks until /v1/models answers (timeout 30 min).
# Each arm gets its own VLLM_CACHE_ROOT: the switches are env vars, which vLLM's compile-cache key does not see,
# and DSV41_ENGRAM_WKV_TP changes the traced Engram graph. The JIT caches (FlashInfer, Triton, CUDA) stay shared.
set -euo pipefail
ARM=${ARM:?set ARM=old|idx|both|eng}
case "$ARM" in
  old) SKIP=0; WKV=0 ;; idx) SKIP=1; WKV=0 ;; both) SKIP=1; WKV=1 ;; eng) SKIP=0; WKV=1 ;;
  *) echo "ARM must be old|idx|both|eng"; exit 2 ;;
esac
VENV=${VENV:-/root/dsv41/venv}
MODEL_DIR=${MODEL_DIR:-/root/dsv41/model-e128t}
PORT=${PORT:-8100}
KV_BLOCKS=${KV_BLOCKS-21490}
UTIL=${UTIL:-0.92}
RUN=${RUN:-$(date -u +%m%d_%H%M)_$ARM}
D=/root/e054/runs/$RUN
[ ! -e "$D" ] || { echo "$D exists"; exit 1; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[ "$used" -lt 500 ] || { echo "a GPU already has ${used} MiB in use: check for leftovers (by PID) first"; exit 1; }
grep -q DSV41_INDEXER_SKIP "$("$VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py" \
  || { echo "E054 files not installed in $VENV (gpu/install_e054.sh)"; exit 1; }
mkdir -p "$D"

C=/root/dsv41/cache
export VLLM_CACHE_ROOT=$C/vllm_e054_$ARM FLASHINFER_WORKSPACE_BASE=$C HF_HOME=$C/hf CUDA_CACHE_PATH=$C/nv
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=$VENV/bin:/usr/local/cuda-13.0/bin:$PATH
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1
export DSV41_INDEXER_Q_PREDECODE=1
export DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'
export DSV41_EXPERT_HUMMING_W2='{"group_size": 64}'
export E043_GPTQ_DIR="$MODEL_DIR/gptq"
export VLLM_SERVER_DEV_MODE=1                 # POST /reset_prefix_cache for logprob_probe.py
export DSV41_INDEXER_SKIP=$SKIP DSV41_ENGRAM_WKV_TP=$WKV
unset DSV41_INDEXER_SKIP_CANDIDATES DSV41_INDEXER_TARGET_PROGRAMS
[ -n "${TARGET_PROGRAMS:-}" ] && export DSV41_INDEXER_TARGET_PROGRAMS=$TARGET_PROGRAMS

ARGS=(
  serve "$MODEL_DIR" --served-model-name dsv41 --host 127.0.0.1 --port "$PORT"
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp
  --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840 --max-num-seqs 64
  --max-num-batched-tokens 2048 --gpu-memory-utilization "$UTIL"
  --enable-prefix-caching --prefix-cache-retention-interval 128
  --engram-config '{"cpu_offload":true}' --tokenizer-mode deepseek_v41 --language-model-only
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --enable-prompt-tokens-details --linear-backend humming
  --compilation-config '{"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64}'
)
[ -n "$KV_BLOCKS" ] && ARGS+=(--num-gpu-blocks-override "$KV_BLOCKS")

{ echo "E054 ARM=$ARM DSV41_INDEXER_SKIP=$SKIP DSV41_ENGRAM_WKV_TP=$WKV UTIL=$UTIL KV_BLOCKS=${KV_BLOCKS:-none}"
  echo "starting: $VENV/bin/vllm ${ARGS[*]}"; } > "$D/server.log"
env | grep -E '^(DSV41_|VLLM_|E043_|CUDA_HOME)' | sort > "$D/env.txt"
cd "$D"
setsid nohup "$VENV/bin/vllm" "${ARGS[@]}" >> "$D/server.log" 2>&1 < /dev/null &
P=$!
echo "$P" > "$D/server.pid"
sleep 1
ps -o pgid= -p "$P" | tr -d ' ' > "$D/server.pgid"
echo "arm $ARM: pid $P pgid $(cat "$D/server.pgid"), dir $D"
if [ "${WAIT:-1}" = 1 ]; then
  for _ in $(seq 360); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "ready after $SECONDS s"; exit 0; fi
    kill -0 "$P" 2>/dev/null || { echo "server exited; tail of log:"; tail -30 "$D/server.log"; exit 1; }
    sleep 5
  done
  echo "not ready after 30 min"; exit 1
fi
