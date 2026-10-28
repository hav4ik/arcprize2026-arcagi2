#!/bin/bash
# E060: serve Qwen3.8-Flash-Next (albucino W4A16 + FP8 PLE) with upstream vLLM 0.32.0 on 4x RTX 4090 (SM89).
# Runs in the foreground (start it with setsid; stop it by its process group). Settings via env:
#   MODEL (default /root/e060/model), PORT (8000), TP (4), UTIL (0.92), MAXLEN (196608), MAX_SEQS (64),
#   MNBT (max-num-batched-tokens, 8192), SPEC (speculative-config JSON or empty), LOAD_FORMAT (auto | dummy),
#   KV_DTYPE (auto), QSA_LOGITS_MB (128), BLOCKS (num-gpu-blocks-override, empty), EXTRA (more flags), LOG (log file)
set -u
MODEL=${MODEL:-/root/e060/model}; PORT=${PORT:-8000}; TP=${TP:-4}; UTIL=${UTIL:-0.92}; MAXLEN=${MAXLEN:-196608}
MAX_SEQS=${MAX_SEQS:-64}; MNBT=${MNBT:-8192}; SPEC=${SPEC:-}; LOAD_FORMAT=${LOAD_FORMAT:-auto}; KV_DTYPE=${KV_DTYPE:-auto}
BLOCKS=${BLOCKS:-}; EXTRA=${EXTRA:-}
export VLLM_CACHE_ROOT=/root/e060/vllm_cache TRITON_CACHE_DIR=/root/e060/triton_cache
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
# QSA prefill allocates a logits workspace of this size outside the profiled memory (default 512 MiB: OOM at util 0.96)
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${QSA_LOGITS_MB:-128}
ARGS=(serve "$MODEL" --served-model-name flashnext --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP" --enable-expert-parallel
  --engram-config '{"cpu_offload": true}'
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$MNBT" --enable-prefix-caching --kv-cache-dtype "$KV_DTYPE"
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --load-format "$LOAD_FORMAT" --enable-prompt-tokens-details)
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ -n "$BLOCKS" ] && ARGS+=(--num-gpu-blocks-override "$BLOCKS")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)
echo "[$(date -u +%FT%TZ)] serve: ${ARGS[*]}"
exec /root/e060/svenv/bin/python -m vllm.entrypoints.cli.main "${ARGS[@]}"
