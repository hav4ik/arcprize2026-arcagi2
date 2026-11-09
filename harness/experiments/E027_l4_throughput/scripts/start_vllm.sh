#!/bin/bash
# usage: [VENV=...] [ATTN=...] start_vllm.sh <mtp tokens, 0=off> <extra flags...>
MTP=$1; shift
source ${VENV:-/root/vllm31-venv}/bin/activate
ATTN=${ATTN:-TRITON_ATTN}
export VLLM_FORCE_ATTN_BACKEND=$ATTN VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
ARGS=(--model ${MODEL:-Qwen/Qwen3.8-27B-FP8} --download-dir /workspace/models --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8001 --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --max-model-len 131072 --max-num-seqs 64 --kv-cache-dtype fp8 --attention-backend $ATTN --enable-prefix-caching --trust-remote-code --no-async-scheduling)
if [ "$MTP" -gt 0 ]; then ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP}"); fi
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" "$@"
