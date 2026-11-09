#!/bin/bash
# E062 (on the box): start one standalone server for parity / unit checks, detached, in its own session.
#   serve.sh b32  [port] [extra args...]   B32's stack (/root/dsv41/venv, 16-file overlay), the solver's B32 flags
#   serve.sh manh [port] [extra args...]   Manh's fork (/root/manh/venv), the same flags + TP4/DCP4
# The flags mirror solver_dsv41.server_cmd() + B32's ARC_SERVER_EXTRA_ARGS (minus KV offload, which only matters
# under load). Log: /root/e062/serve_<stack>_<port>.log; pid file beside it. Stop with: kill -- -$(cat <pidfile>)
set -euo pipefail
S=$1; PORT=${2:-8000}; shift 2 || shift $#
LOG=/root/e062/serve_${S}_${PORT}.log
case $S in
  b32)  V=/root/dsv41/venv; M=${E062_MODEL:-/root/dsv41/model-e128k}; STACK_ARGS=(--linear-backend humming) ;;
  manh) V=/root/manh/venv; M=${E062_MODEL:-/root/dsv41/model-e128m}; STACK_ARGS=(--decode-context-parallel-size 4) ;;
  *) echo "stack: b32 | manh"; exit 2 ;;
esac
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HF_API_TOKEN$)'); do unset "$v"; done
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 DSV41_INDEXER_Q_PREDECODE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false MAX_JOBS=8 \
  DSV41_EXPERT_HUMMING='{"quant_method": "humming", "weight_dtype": "uint2", "weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}' \
  DSV41_EXPERT_HUMMING_W2='{"group_size": 64}' E043_GPTQ_DIR=$M/gptq \
  CUDA_HOME=/usr/local/cuda-13.0 PATH=$V/bin:/usr/local/cuda-13.0/bin:$PATH
[ -d /usr/local/cuda-13.0 ] || export CUDA_HOME=/usr/local/cuda PATH=$V/bin:/usr/local/cuda/bin:$PATH
# extra env for a run: E062_ENV="A=1 B=2"
for kv in ${E062_ENV:-}; do export "$kv"; done
cd /root/e062
setsid nohup $V/bin/vllm serve $M --served-model-name dsv41 --host 127.0.0.1 --port $PORT \
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp \
  --kv-cache-dtype fp8 --block-size 128 --max-model-len ${E062_MAX_LEN:-163840} \
  --max-num-seqs ${E062_SEQS:-64} --max-num-batched-tokens 2048 \
  --gpu-memory-utilization ${E062_UTIL:-0.92} --enable-prefix-caching \
  --prefix-cache-retention-interval 512 --engram-config '{"cpu_offload":true}' \
  --tokenizer-mode deepseek_v41 --language-model-only --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 64}' \
  "${STACK_ARGS[@]}" "$@" > $LOG 2>&1 < /dev/null &
echo $! > /root/e062/serve_${S}_${PORT}.pid
echo "started $S on $PORT (pid $!), log $LOG"
