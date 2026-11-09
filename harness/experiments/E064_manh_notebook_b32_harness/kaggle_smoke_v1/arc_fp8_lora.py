"""Configuration and dispatch checks for experimental Qwen3 FP8 LoRA SFT."""

PROJECTIONS = frozenset((
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
))


def validate_fp8_lora_options(
    enabled, *, fp8_frozen_linear=False, fp8_dynamic_backward=True,
    lora_method="rslora", lora_init="default", optimizer="adamw_torch",
    task_rl=False, deepspeed=False, distributed=False, keep_training_state=False,
    torch_compile_training=False, inference_backend="vllm", train_only=False,
    model_type=None,
):
    if not enabled:
        return
    incompatible = []
    if not fp8_frozen_linear or not fp8_dynamic_backward:
        incompatible.append("requires --fp8-frozen-linear and --fp8-dynamic-backward")
    if lora_method not in {"lora", "rslora"} or lora_init not in {"default", "gaussian"}:
        incompatible.append("requires ordinary LoRA/RSLoRA with default or gaussian initialization")
    if optimizer != "adamw_torch":
        incompatible.append("requires --optimizer adamw_torch")
    for active, label in (
        (task_rl, "task-time GRPO"), (deepspeed, "DeepSpeed"),
        (distributed, "distributed training"), (keep_training_state, "carried training state"),
        (torch_compile_training, "compiled training"),
    ):
        if active:
            incompatible.append(label)
    if not train_only and inference_backend != "vllm":
        incompatible.append("requires vLLM serving or --train-only")
    if model_type is not None and model_type != "qwen3":
        incompatible.append(f"only dense Qwen3 is supported, got {model_type!r}")
    if incompatible:
        raise ValueError("--fp8-lora: " + "; ".join(incompatible))


def precision_manifest(*, base_fp8, lora_fp8, stats=None):
    return {
        "frozen_projection_storage": "fp8_e4m3" if base_fp8 else "bfloat16",
        "lora_gemm": "fp8_current_scaling_e4m3_e5m2" if lora_fp8 else "bfloat16",
        "lora_master_weights": "bfloat16",
        "lora_gradients": "bfloat16",
        "optimizer_precision": "unchanged",
        "adapter_export_dtype": "bfloat16",
        "fp8_lora_gemms": None if stats is None else vars(stats).copy(),
    }


def validate_fp8_lora_model(model):
    """Reject silent fallback to unfused BF16 adapters before Trainer runs."""
    import torch
    from unsloth.kernels import fast_lora

    if not getattr(fast_lora, "UNSLOTH_SUPPORTS_FP8_LORA_COMPUTE", False):
        raise RuntimeError("--fp8-lora requires the updated Unsloth fork (FP8 LoRA compute)")
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("--fp8-lora currently supports dense Qwen3 only")
    layers = [m for m in model.modules() if hasattr(m, "self_attn") and hasattr(m, "mlp")]
    if len(layers) != int(model.config.num_hidden_layers):
        raise RuntimeError("Cannot audit all Qwen3 layers for FP8 LoRA dispatch")
    for layer in layers:
        hooks = (
            (layer.self_attn, "apply_qkv", fast_lora.apply_lora_qkv),
            (layer.self_attn, "apply_o", fast_lora.apply_lora_o),
            (layer.mlp, "forward", fast_lora.apply_lora_mlp_swiglu),
        )
        for owner, name, expected in hooks:
            actual = getattr(owner, name, None)
            if getattr(actual, "__func__", actual) is not expected:
                raise RuntimeError(f"FP8 LoRA requires fused Qwen3 {name}; update the Unsloth fork")
    count = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        if name.rsplit(".", 1)[-1] not in PROJECTIONS:
            raise ValueError(f"FP8 LoRA does not support adapter projection {name}")
        adapters = list(module.active_adapters)
        if len(adapters) != 1 or module.disable_adapters or module.merged:
            raise ValueError("FP8 LoRA requires one active, unmerged task adapter")
        adapter = adapters[0]
        a, b = module.lora_A[adapter], module.lora_B[adapter]
        if any(p.weight.dtype != torch.bfloat16 or not p.weight.requires_grad for p in (a, b)):
            raise ValueError("FP8 LoRA requires trainable BF16 master A/B parameters")
        base = module.base_layer
        if base.weight.requires_grad or not hasattr(base.weight, "_unsloth_te_fp8_matmul"):
            raise ValueError("FP8 LoRA requires frozen TE FP8 base projections")
        if base.bias is not None or getattr(module.lora_dropout[adapter], "p", 0) != 0:
            raise ValueError("FP8 LoRA requires bias-free projections and zero dropout")
        if module.use_dora.get(adapter, False) or getattr(module, "use_nora", {}).get(adapter, False):
            raise ValueError("FP8 LoRA does not support DoRA/NoRA")
        if any(getattr(m, "activation_fake_quantizer", None) is not None for m in (base, a, b)):
            raise ValueError("FP8 LoRA cannot be combined with activation QAT")
        count += 1
    if count != 7 * len(layers):
        raise RuntimeError(f"FP8 LoRA expected {7 * len(layers)} projections, found {count}")
    return count
