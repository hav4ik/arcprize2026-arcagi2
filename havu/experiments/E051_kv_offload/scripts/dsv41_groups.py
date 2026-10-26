"""Rebuild DSv4.1 (E128 config) per-layer KV specs and run the fork's packed grouping on CPU."""
import json, sys
from types import SimpleNamespace
import torch
from vllm.v1.kv_cache_interface import (MLAAttentionSpec, SlidingWindowMLASpec, CircularBufferSpec,
    get_kv_quant_mode, iter_layer_specs)
from vllm.v1.core import kv_cache_utils as U

cfg = json.load(open(sys.argv[1]))["text_config"]
cr = cfg["compress_ratios"]; kvs = cfg["kv_source_layer_ids"]; ids = cfg["index_source_layer_ids"]
L = cfg["num_hidden_layers"]

def build_specs(block_size=128, cache_dtype="fp8"):
    specs = {}
    for i in range(L):
        p = f"model.layers.{i}.self_attn"
        if i in kvs:
            specs[p] = MLAAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=512,
                dtype=torch.uint8, tokens_per_state=cr[i], cache_dtype_str="fp4_mixed",
                alignment=32, model_version="deepseek_v4_1",
                kv_quant_mode=get_kv_quant_mode(cache_dtype), state_content_bytes=288)
            if i in ids:
                specs[f"{p}.indexer.k_cache"] = MLAAttentionSpec(block_size=block_size, num_kv_heads=1,
                    head_size=68, dtype=torch.uint8, tokens_per_state=cr[i], alignment=128,
                    state_content_bytes=68, model_version="deepseek_v4_1")
            if cr[i] > 1:
                specs[f"{p}.compressor.state_cache"] = CircularBufferSpec(block_size=8, num_kv_heads=1,
                    head_size=1024, head_size_v=0, dtype=torch.float32)
        specs[f"{p}.swa_cache"] = SlidingWindowMLASpec(block_size=32, num_kv_heads=1, head_size=512,
            dtype=torch.uint8, sliding_window=cfg["sliding_window"], cache_dtype_str=cache_dtype,
            state_content_bytes=528, alignment=16, model_version="deepseek_v4_1",
            kv_quant_mode=get_kv_quant_mode(cache_dtype))
    return specs

if __name__ == "__main__":
    specs = build_specs()
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_block_outermost=True)),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41")),
        speculative_config=None)
    groups = U._get_packed_kv_cache_groups(vc, specs)
    tot = 0
    for gi, g in enumerate(groups):
        ls = list(iter_layer_specs(g.kv_cache_spec))
        kinds = sorted({type(s).__name__ for s in ls})
        print(gi, kinds, "block_size", g.kv_cache_spec.block_size, "layers", len(g.layer_names),
              "page_bytes", g.kv_cache_spec.page_size_bytes, "cacheable", g.kv_cache_spec.prefix_cacheable)
    print("per-block bytes (max group page)", max(g.kv_cache_spec.page_size_bytes for g in groups))
    print("bytes_per_block fn", U._get_kv_cache_bytes_per_block(groups))
