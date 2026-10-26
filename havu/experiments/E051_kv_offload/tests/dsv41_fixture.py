# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixtures that rebuild the DeepSeek-V4.1 (E128 2-bit) KV layout.

The per-layer specs mirror models/deepseek_v4_1/attention.py:109-123 (SWA),
:880-899 (compressed KV), :936-946 (indexer K) and compressor.py:158-166 (ring).
The groups come from the fork's own packing code
(_get_packed_kv_cache_groups), so the tests see the server's real
9-group layout: 7 SWA groups (32-token blocks), 1 MLA+indexer group
(128-token blocks), 1 compressor ring group (8-row CircularBufferSpec).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

# text_config of chankhavu/dsv41-flash-1017-arc-tool-e128-2bit (config.json).
E128_TEXT_CONFIG = {
    "num_hidden_layers": 40,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "sliding_window": 128,
}
BLOCK_SIZE = 128  # --block-size 128
POOL_BLOCK_BYTES = 113_920


def build_layer_specs(cfg=E128_TEXT_CONFIG, block_size=BLOCK_SIZE, cache_dtype="fp8"):
    cr = cfg["compress_ratios"]
    kvs = cfg["kv_source_layer_ids"]
    ids = cfg["index_source_layer_ids"]
    specs = {}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}.self_attn"
        if i in kvs:
            specs[p] = MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.uint8,
                tokens_per_state=cr[i],
                cache_dtype_str="fp4_mixed",
                alignment=32,
                model_version="deepseek_v4_1",
                kv_quant_mode=get_kv_quant_mode(cache_dtype),
                state_content_bytes=288,
            )
            if i in ids:
                specs[f"{p}.indexer.k_cache"] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=68,
                    dtype=torch.uint8,
                    tokens_per_state=cr[i],
                    alignment=128,
                    state_content_bytes=68,
                    model_version="deepseek_v4_1",
                )
            if cr[i] > 1:
                specs[f"{p}.compressor.state_cache"] = CircularBufferSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=1024,
                    head_size_v=0,
                    dtype=torch.float32,
                )
        specs[f"{p}.swa_cache"] = SlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            sliding_window=cfg["sliding_window"],
            cache_dtype_str=cache_dtype,
            state_content_bytes=528,
            alignment=16,
            model_version="deepseek_v4_1",
            kv_quant_mode=get_kv_quant_mode(cache_dtype),
        )
    return specs


def build_groups():
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                is_block_outermost=True
            )
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        ),
        speculative_config=None,
    )
    groups = U._get_packed_kv_cache_groups(vc, build_layer_specs())
    assert groups is not None
    return groups


def build_kv_cache_config(num_blocks=1024):
    """The fork's own placement (get_kv_cache_config_from_groups, BLHNC)."""
    from vllm.v1.kv_cache_layout import KVCacheLayout

    groups = build_groups()
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=128,
        )
    )
    cfg = U.get_kv_cache_config_from_groups(
        vc, groups, available_memory=POOL_BLOCK_BYTES * num_blocks
    )
    assert cfg.num_blocks == num_blocks
    return cfg


def allocate_cpu_kv_caches(kv_cache_config):
    """Per-layer views over one int8 CPU buffer, as the GPU runner builds them."""
    from vllm.v1.kv_cache_interface import create_kv_cache_views
    from vllm.v1.kv_cache_layout import KVCacheLayout

    size = kv_cache_config.kv_cache_tensors[0].size
    raw = torch.zeros(size, dtype=torch.int8)
    specs = {}
    for group in kv_cache_config.kv_cache_groups:
        inner = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            specs[name] = inner[name] if inner is not None else group.kv_cache_spec
    kv_caches = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        spec = specs[tensor.layers[0]]
        views = create_kv_cache_views(
            raw, spec, kv_cache_config.num_blocks, KVCacheLayout.BLHNC, tensor
        )
        for name, view in zip(tensor.layers, views):
            kv_caches[name] = view
    return raw, kv_caches


def make_vllm_config(extra_config=None, tp=4, prefix_match_unit=None):
    """Duck-typed VllmConfig with only the fields the offloading code reads."""
    extra = {"cpu_bytes_to_use": 8 << 30, "offload_prompt_only": False}
    extra.update(extra_config or {})
    parallel = SimpleNamespace(
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        world_size=tp,
        distributed_executor_backend="mp",
        nnodes_within_dp=1,
        rank=0,
        data_parallel_index=0,
        data_parallel_size=1,
        data_parallel_rank_local=None,
        cp_kv_cache_interleave_size=1,
    )
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            engine_id="test-engine",
            kv_connector_extra_config=extra,
        ),
        parallel_config=parallel,
        model_config=SimpleNamespace(
            use_mla=True,
            model="dsv41",
            dtype=torch.bfloat16,
            get_total_num_kv_heads=lambda: 1,
            hf_config=SimpleNamespace(model_type="deepseek_v41"),
        ),
        cache_config=SimpleNamespace(
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
            prefix_match_unit=prefix_match_unit,
            cache_dtype="fp8",
            kv_cache_layout="BLHNC",
            prefix_cache_retention_interval=128,
        ),
        use_v2_model_runner=True,
        kv_events_config=None,
        speculative_config=None,
    )
