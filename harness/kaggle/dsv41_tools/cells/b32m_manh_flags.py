# Cell 2m (B32-M): MANH'S EXTRA FLAGS: OFF. E062b is still testing them. Host all-reduce MODE 60 includes MODE_Q8
# (16): lossy int8 sums across the GPUs. The others are speed options from his notebook: indexer logits, tuned Humming
# MoE tiles, his GPU order, watermark. Flip MANH_FLAGS (build_b32m.py
# --manh-flags) only after E062b's results. The solver applies them via ARC_SERVER_ENV_JSON / ARC_SERVER_EXTRA_ARGS.
MANH_FLAGS = False
if MANH_FLAGS:
    _moe_cfg = lambda block, warp, stages, ctas, stream_k: {
        "block_shape": block, "warp_shape": warp, "num_stages": stages, "num_ctas_per_sm": ctas,
        "use_stream_k": stream_k, "num_sms": 58, "mma_type": "mma", "output_chunk_rows": 0,
        "use_f16_accum": False, "use_pdl": False, "use_tma_a": False, "use_tma_as": False,
        "use_tma_as2": False, "use_tma_c": False}
    _moe_tuning = {  # his cell 7: "NxK": [[lo, hi, config], ...] for the TP4 routed-expert shards, L4 only
        "1152x5120": [
            [1926, 3680, _moe_cfg([32, 128, 128], [32, 64, 64], 5, 2, False)],
            [3888, 5504, _moe_cfg([48, 128, 128], [48, 64, 64], 3, 1, False)],
            [5504, 8192, _moe_cfg([64, 128, 128], [64, 32, 128], 3, 1, True)],
            [8192, 12288, _moe_cfg([48, 128, 128], [48, 64, 64], 3, 1, True)]],
        "5120x576": [
            [1840, 4248, _moe_cfg([32, 256, 64], [32, 64, 64], 5, 2, False)],
            [4248, 5592, _moe_cfg([48, 256, 64], [48, 64, 32], 5, 1, False)],
            [5592, 8192, _moe_cfg([64, 256, 64], [64, 32, 64], 5, 1, False)]]}
    _se = json.loads(BASE_ENV.get("ARC_SERVER_ENV_JSON") or "{}")
    _se.update({"VLLM_HOST_ALLREDUCE": "1", "VLLM_HOST_ALLREDUCE_MODE": "60", "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16",
                "VLLM_HOST_ALLREDUCE_PREFETCH_MAX_MB": "36", "VLLM_HOST_ALLREDUCE_LARGE_SHAPE": "32x32x256",
                "VLLM_HOST_ALLREDUCE_SHAPE": "12x12x256", "VLLM_HOST_ALLREDUCE_UNIT_KB": "16",
                "VLLM_DCP_ASYNC_QUERY_GATHER": "1", "VLLM_TRITON_SPARSE_INDEXER_LOGITS": "1",
                "VLLM_HUMMING_MOE_TUNING_CONFIG": json.dumps(_moe_tuning), "VLLM_GPU_MEMORY_LOG_INTERVAL": "60",
                "CUDA_VISIBLE_DEVICES": "0,2,1,3"})
    BASE_ENV["ARC_SERVER_ENV_JSON"] = json.dumps(_se)
    _sx = json.loads(BASE_ENV["ARC_SERVER_EXTRA_ARGS"])
    # not taken: his --enable-batch-sharded-sampling (ranks send their top-k logits only; B32 samples the full
    # vocabulary at temperature 1.0, top_p 1.0) and --engram-config.use_thp (B32 passes --engram-config as JSON)
    BASE_ENV["ARC_SERVER_EXTRA_ARGS"] = json.dumps(_sx + ["--watermark", "0.1"])
print("Manh's extra flags:", "ON" if MANH_FLAGS else "OFF", flush=True)
