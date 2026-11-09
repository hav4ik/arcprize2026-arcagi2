# Manh's server cell (his notebook as pulled), verbatim except (E066, agreed with the user): --dsv41-vllm-model {E066_MODEL}; --context 163840; --concurrency 120; --qwen38-port 18084; --vision-max-images 64; VLLM_HOST_ALLREDUCE_MODE=556; --tool-call-parser deepseek_v41 --enable-auto-tool-choice
import json
import os

# Tuned humming MoE tiles for the TP4 routed-expert shards, L4 only (num_sms 58); see Requirements.
_moe_cfg = lambda block, warp, stages, ctas, stream_k: {
    "block_shape": block, "warp_shape": warp, "num_stages": stages, "num_ctas_per_sm": ctas,
    "use_stream_k": stream_k, "num_sms": 58, "mma_type": "mma", "output_chunk_rows": 0,
    "use_f16_accum": False, "use_pdl": False, "use_tma_a": False, "use_tma_as": False,
    "use_tma_as2": False, "use_tma_c": False}
DSV41_MOE_TUNING = {  # "NxK": [[lo, hi, config], ...], used when lo < tokens x top_k 6 <= hi
    "1152x5120": [
        [1926, 3680, _moe_cfg([32, 128, 128], [32, 64, 64], 5, 2, False)],
        [3888, 5504, _moe_cfg([48, 128, 128], [48, 64, 64], 3, 1, False)],
        [5504, 8192, _moe_cfg([64, 128, 128], [64, 32, 128], 3, 1, True)],
        [8192, 12288, _moe_cfg([48, 128, 128], [48, 64, 64], 3, 1, True)],
    ],
    "5120x576": [
        [1840, 4248, _moe_cfg([32, 256, 64], [32, 64, 64], 5, 2, False)],
        [4248, 5592, _moe_cfg([48, 256, 64], [48, 64, 32], 5, 1, False)],
        [5592, 8192, _moe_cfg([64, 256, 64], [64, 32, 64], 5, 1, False)],
    ],
}
os.environ.update({
    "VLLM_HOST_ALLREDUCE": "1",
    "VLLM_HOST_ALLREDUCE_MODE": "556",
    "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16",
    "VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB": "36",
    "VLLM_HOST_ALLREDUCE_LARGE_SHAPE": "32x32x256",
    "VLLM_HOST_ALLREDUCE_SHAPE": "12x12x256",
    "VLLM_HOST_ALLREDUCE_UNIT_KB": "16",
    "VLLM_DCP_ASYNC_QUERY_GATHER": "1",           # DCP only: fp8 query all-gather on a side stream (default)
    "VLLM_HUMMING_MOE_PAD_MULTIPLE": "64,64",     # routed experts unpadded: 576-wide TP4 shard, not 640
    "VLLM_HUMMING_MOE_TUNING_CONFIG": json.dumps(DSV41_MOE_TUNING),  # 321+ token steps only
    "VLLM_TRITON_SPARSE_INDEXER_LOGITS": "1",     # indexer consumers score only layer 20's candidates
    "VLLM_GPU_DRIVER_RESERVE_MB": "512",          # keep 512 MiB out of the torch cache (see Requirements)
    "VLLM_GPU_MEMORY_LOG_INTERVAL": "60",         # per-rank "GPU memory: ..." line every 60 s
})

# Server only (no ARC run, e.g. for another harness): run the env cells above, then this. It starts
# the same vLLM server as the command above, prints its URL and returns; the engine keeps running in
# its own session.
#   Prints: PID: <pid> Log: {server_root}/server/server.log
#           Base URL: http://127.0.0.1:18084/v1 Model: dsv41
#   Ready once GET http://127.0.0.1:18084/health returns 200
import time
server_root = f"/kaggle/working/dsv41_server_{time.time_ns()}"
!python -u /kaggle/working/dsv41_vllm.py \
  --qwen38-variant dsv4-vllm \
  --qwen38-recipe vision-repl-audit \
  --dsv41-vllm-model {E066_MODEL} \
  --dsv41-vllm-triton-cache /kaggle/working/dsv41_triton \
  --output-root {server_root} \
  --tensor-parallel-size 4 \
  --gpu-ids 0,2,1,3 \
  --qwen38-port 18084 \
  --temp-root /tmp/arc_vision \
  --context 163840 \
  --concurrency 120 \
  --gpu-memory-utilization 0.92 \
  --kv-offload-gib-per-rank 12 \
  --max-num-batched-tokens 2048 \
  --vision-max-pixels 2097152 \
  --vision-max-images 32 \
  --vllm-arg=--offload-backend=uva \
  --vllm-arg=--cpu-offload-gb=0.25 \
  --vllm-arg=--cpu-offload-params=embed_tokens \
  --vllm-arg=--engram-config.use_thp=true \
  --vllm-arg=--watermark=0.1 \
  --vllm-arg=--tool-call-parser=deepseek_v41 \
  --vllm-arg=--enable-auto-tool-choice
os.environ["ARC_EXTERNAL_SERVER_LOG"] = f"{server_root}/server/server.log"
print("E066 external server log:", os.environ["ARC_EXTERNAL_SERVER_LOG"], flush=True)
