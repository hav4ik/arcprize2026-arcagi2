#!/bin/bash
# Manh's server config (his launcher dsv41_vllm.py command_for + environment(), notebook cell 7 env and vllm-args), with
# the changes agreed with the user: context 163840, max-num-seqs 120 (capture 120), host all-reduce MODE 556 (lossless),
# limit-mm 64; plus our --tool-call-parser deepseek_v41 --enable-auto-tool-choice. kv_offload_mode = his default
# "eager" (lazy_offload false). VLLM_BATCH_SHARD_TOPK = his profile top_k 64 (environment() sets it).
# Usage: serve_manh.sh <model dir> <tag>
set -u
M=$1; TAG=$2; OUT=/root/e062pl/runs/$TAG; mkdir -p $OUT /root/e062pl/triton_manh /tmp/arc_vision
export CUDA_VISIBLE_DEVICES=0,2,1,3 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_PLUGINS="" \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
  TORCH_CUDA_ARCH_LIST=8.9 FLASHINFER_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 TRITON_CACHE_DIR=/root/e062pl/triton_manh \
  VLLM_BATCH_SHARD_TOPK=64 CUDA_HOME=/usr/local/cuda-13.0 \
  VLLM_HOST_ALLREDUCE=1 VLLM_HOST_ALLREDUCE_MODE=556 VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS=16 \
  VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB=36 VLLM_HOST_ALLREDUCE_LARGE_SHAPE=32x32x256 VLLM_HOST_ALLREDUCE_SHAPE=12x12x256 \
  VLLM_HOST_ALLREDUCE_UNIT_KB=16 VLLM_DCP_ASYNC_QUERY_GATHER=1 VLLM_HUMMING_MOE_PAD_MULTIPLE=64,64 \
  VLLM_HUMMING_MOE_TUNING_CONFIG="$(cat /root/e062pl/manh_moe_tuning.json)" VLLM_TRITON_SPARSE_INDEXER_LOGITS=1 \
  VLLM_GPU_DRIVER_RESERVE_MB=512 VLLM_GPU_MEMORY_LOG_INTERVAL=60
ARGS=(serve "$M" --served-model-name dsv41 --host 127.0.0.1 --port 18084 --trust-remote-code
  --tensor-parallel-size 4 --distributed-executor-backend mp --kv-cache-dtype fp8 --block-size 128
  --max-model-len 163840 --max-num-seqs 120 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.92
  --enable-prefix-caching --enable-batch-sharded-sampling --engram-config '{"cpu_offload": true}'
  --linear-backend humming --compilation-config '{"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 120}'
  --reasoning-parser deepseek_v41 --enable-scale-out --enable-prompt-tokens-details
  --kv-transfer-config '{"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": 12884901888, "lazy_offload": false}}'
  --limit-mm-per-prompt '{"image": 64}'
  --offload-backend=uva --cpu-offload-gb=0.25 --cpu-offload-params=embed_tokens --engram-config.use_thp=true --watermark=0.1
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice)
{ echo "$(date -u +%FT%TZ) serve_manh $TAG"; printf '%s\n' "${ARGS[@]}"; env | grep -E '^(VLLM|CUDA_VISIBLE|TORCH_CUDA|FLASHINFER|TRITON_CACHE)' | grep -v TUNING_CONFIG | sort; echo "VLLM_HUMMING_MOE_TUNING_CONFIG sha $(sha256sum /root/e062pl/manh_moe_tuning.json | cut -c1-16)"; } > $OUT/meta.txt
setsid nohup /root/b32m/venv/bin/python -m vllm.entrypoints.cli.main "${ARGS[@]}" > $OUT/server.log 2>&1 < /dev/null &
echo $! > $OUT/server.pid; date +%s > $OUT/t0
echo "started $TAG pid $(cat $OUT/server.pid)"
