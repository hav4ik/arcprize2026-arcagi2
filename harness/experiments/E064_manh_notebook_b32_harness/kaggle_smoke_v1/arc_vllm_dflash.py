"""Opt-in, checkpoint-validated DFlash2 configuration for ARC task LoRAs.

The target still verifies every proposal with the active task adapter. No
draft weights are trained or attached as a target LoRA by this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


OPTION_NAMES = (
    "vllm_draft_model", "vllm_draft_tokens", "vllm_draft_quantization",
    "vllm_draft_kv_cache_dtype", "vllm_draft_attention_backend",
)


def add_dflash_arguments(parser):
    parser.add_argument("--vllm-draft-model", default=os.getenv("ARC_VLLM_DRAFT_MODEL"),
                        help="Opt-in local DFlash2 checkpoint; no draft by default.")
    parser.add_argument("--vllm-draft-tokens", type=int,
                        default=int(os.getenv("ARC_VLLM_DRAFT_TOKENS", "0")),
                        help="Draft count; 0 reads checkpoint (block_size minus anchor).")
    parser.add_argument("--vllm-draft-quantization", choices=("none", "fp8"),
                        default=os.getenv("ARC_VLLM_DRAFT_QUANTIZATION", "none"),
                        help="Draft weights only; does not change target or LoRA dtype.")
    parser.add_argument("--vllm-draft-kv-cache-dtype", choices=("auto", "fp8", "fp8_e4m3"),
                        default=os.getenv("ARC_VLLM_DRAFT_KV_CACHE_DTYPE", "auto"),
                        help="Draft cache only; auto inherits target cache dtype.")
    parser.add_argument("--vllm-draft-attention-backend",
                        choices=("auto", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"),
                        default=os.getenv("ARC_VLLM_DRAFT_ATTENTION_BACKEND", "auto"))


def dflash_options(args):
    return {name: getattr(args, name) for name in OPTION_NAMES}


def build_dflash_speculative_config(model_path, options):
    """Fail early on invalid combinations; leave ordinary serving untouched."""
    draft_path = options.get("vllm_draft_model")
    count = int(options.get("vllm_draft_tokens", 0) or 0)
    if not draft_path:
        if count:
            raise ValueError("--vllm-draft-tokens requires --vllm-draft-model")
        return None
    if options.get("vllm_speculative_ngram_tokens", 0):
        raise ValueError("DFlash2 and n-gram speculation cannot be enabled together")
    if options.get("vllm_cumulative_nll_stop", False):
        raise ValueError("DFlash2 requires --no-vllm-cumulative-nll-stop; the stateful "
                         "processor is not a speculative verifier")
    if options.get("bidirectional_prompt_attention", False):
        raise ValueError("DFlash2 currently requires causal prompt attention")
    draft_root, target_root = Path(draft_path), Path(model_path)
    draft = json.loads((draft_root / "config.json").read_text())
    target = json.loads((target_root / "config.json").read_text())
    if (draft.get("speculators_model_type") != "dflash2"
            or "DFlash2DraftModel" not in draft.get("architectures", [])):
        raise ValueError("--vllm-draft-model must be a speculators DFlash2 checkpoint")
    if (target.get("architectures") != ["Qwen3ForCausalLM"]
            or target.get("unsloth_delta_mhar")
            or target.get("arc_rope_3d")
            or (target_root / "delta_mhar_config.json").exists()
            or (target_root / "arc_rope_3d.json").exists()):
        raise ValueError("DFlash2 currently supports ordinary Qwen3 targets, not custom "
                         "Delta-MHAR/3D-RoPE or hybrid architectures")
    block = draft.get("block_size")
    if isinstance(block, bool) or not isinstance(block, int) or block < 2:
        raise ValueError("DFlash2 checkpoint needs a valid block_size >= 2")
    if draft.get("sample_from_anchor", False):
        raise ValueError("DFlash2 sample_from_anchor=True is not supported")
    if count == 0:
        count = block - 1
    if not 1 <= count < block:
        raise ValueError(f"DFlash2 block_size={block} includes one anchor: "
                         f"choose 1..{block - 1} draft tokens, not {count}")
    layer = draft.get("transformer_layer_config", {})
    hidden = draft.get("target_hidden_size") or layer.get("hidden_size")
    if hidden != target.get("hidden_size") or layer.get("vocab_size") != target.get("vocab_size"):
        raise ValueError("DFlash2 target hidden size/vocabulary do not match")
    if target.get("vocab_size") != 16:
        raise ValueError("ARC DFlash2 requires the 16-token grid vocabulary; "
                         "speculative DFS requests all 16 raw token logprobs")
    ids = draft.get("aux_hidden_state_layer_ids", [])
    if not ids or any(type(i) is not int or not 0 <= i < target.get("num_hidden_layers", 0) for i in ids):
        raise ValueError("DFlash2 auxiliary hidden-state layer IDs are invalid for target")
    # ARC uses a custom 16-token vocabulary. Equal size alone is insufficient.
    for filename in ("vocab.json", "special_tokens_map.json"):
        left, right = draft_root / filename, target_root / filename
        if left.is_file() and right.is_file():
            if json.loads(left.read_text()) != json.loads(right.read_text()):
                raise ValueError(f"DFlash2 target token mapping differs: {filename}")
    quant = options.get("vllm_draft_quantization", "none")
    cache = options.get("vllm_draft_kv_cache_dtype", "auto")
    backend = options.get("vllm_draft_attention_backend", "auto")
    if quant not in ("none", "fp8") or cache not in ("auto", "fp8", "fp8_e4m3"):
        raise ValueError("Unsupported DFlash2 draft quantization/cache dtype")
    if backend not in ("auto", "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"):
        raise ValueError("Unsupported DFlash2 draft attention backend")
    spec = {"method": "dflash", "model": str(draft_root),
            "num_speculative_tokens": count, "draft_sample_method": "greedy"}
    if quant != "none":
        spec["quantization"] = quant
    if cache != "auto":
        spec["kv_cache_dtype"] = cache
    if backend != "auto":
        spec["attention_backend"] = backend
    return spec


def prepare_dflash_runtime(model_path, options):
    """DFlash2's candidate selector requires V2, never silently use DFlash1."""
    spec = build_dflash_speculative_config(model_path, options)
    if spec is None:
        return
    if os.getenv("VLLM_USE_V2_MODEL_RUNNER") == "0":
        raise ValueError("DFlash2 needs the V2 model runner; unset VLLM_USE_V2_MODEL_RUNNER=0")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    print(f"[vLLM] DFlash2: {spec}; target verifies with active BF16 task LoRA", flush=True)
