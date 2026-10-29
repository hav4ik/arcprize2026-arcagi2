#!/bin/bash
# E062 L4 (on the pod): one standalone start of Manh's fork with the replica's exact server flags (solver_dsv41.py
# server_cmd() fixed flags + the replica's ARC_SERVER_EXTRA_ARGS / ARC_SERVER_ENV_JSON), to read the KV pool and run
# a few API checks before the replica. Detached, own session; stop with: kill -- -$(cat /root/e062/serve_manh.pid)
set -euo pipefail
PORT=${PORT:-8011}; SEQS=${SEQS:-96}; UTIL=${UTIL:-0.938}; CAP=${CAP:-96}
M=/root/dsv41/model-e128t; V=/root/manh/venv; LOG=/root/e062/serve_manh_$PORT.log
for v in $(env | cut -d= -f1 | grep -E '^(VLLM_|HF_TOKEN$|HUGGING_FACE_HUB_TOKEN$|HF_API_TOKEN$)'); do unset "$v"; done
export VLLM_USE_V2_MODEL_RUNNER=1 FLASHINFER_CUDA_ARCH_LIST=8.9 TORCH_CUDA_ARCH_LIST=8.9 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false MAX_JOBS=8 VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_GPU_DRIVER_RESERVE_MB=${RESERVE:-512} VLLM_GPU_MEMORY_LOG_INTERVAL=300 \
  CUDA_HOME=/usr/local/cuda-13.0 PATH=$V/bin:/usr/local/cuda-13.0/bin:$PATH
for kv in ${E062_ENV:-}; do export "$kv"; done
mkdir -p /root/e062/scratch_serve; cd /root/e062/scratch_serve
setsid nohup $V/bin/vllm serve $M --served-model-name dsv41 --host 127.0.0.1 --port $PORT \
  --trust-remote-code --tensor-parallel-size 4 --distributed-executor-backend mp \
  --kv-cache-dtype fp8 --block-size 128 --max-model-len 163840 \
  --max-num-seqs $SEQS --max-num-batched-tokens 2048 \
  --gpu-memory-utilization $UTIL --enable-prefix-caching \
  --prefix-cache-retention-interval 128 --engram-config '{"cpu_offload":true}' \
  --tokenizer-mode deepseek_v41 --language-model-only --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --linear-backend humming --decode-context-parallel-size 4 \
  --compilation-config "{\"cudagraph_mode\": \"FULL\", \"max_cudagraph_capture_size\": $CAP}" \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":34359738368,"offload_prompt_only":false,"eviction_policy":"lru"}}' \
  "$@" > $LOG 2>&1 < /dev/null &
echo $! > /root/e062/serve_manh.pid
echo "started manh on $PORT (pid $!), log $LOG"
