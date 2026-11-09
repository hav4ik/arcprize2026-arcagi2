from flash_attention3_runtime import bootstrap_flash_attention3_runtime

bootstrap_flash_attention3_runtime()

from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import (
    ArcDataset,
    QwenFormatter,
    infer_demonstrated_output_shape,
    parse_train_prediction_key,
    token_lengths as batch_token_lengths,
)
from arc_model_reuse import (
    InitialAdapterSnapshot,
    freeze_worker_gc,
    restore_fp8_base_before_unload,
    restore_modules_to_save_before_unload,
    worker_gc_scope,
)
from arc_rl import (
    ArcGridReward,
    assistant_completion_spans,
    build_task_rl_rows,
    color_mapping_signatures,
    dataset_colors,
    select_balanced_warmup_rows,
)
from run_config import (
    configure_task_peft_params_for_model,
    is_pissa_init,
    lora_export_rank,
    lora_effective_scaling,
    normalize_gradient_checkpointing_mode,
    resolve_challenge_path,
    resolve_evaluation_solution_path,
    resolve_model_path,
    resolve_torch_compile_training_options,
    resolve_train_prediction_output_dir,
    resolve_gradient_checkpointing_value,
    resolve_lora_method,
    resolve_task_gradient_checkpointing_profile,
    select_auto_gradient_checkpointing_profile,
    select_task_auto_gradient_checkpointing_percentage,
    select_grpo_auto_gradient_checkpointing_percentage,
    select_rl_micro_batch_size,
    select_rl_micro_batch_token_limit,
    stable_task_seed,
    validate_pissa_options,
)
from output_nll_report import (
    apply_output_nll_scores,
    load_available_ground_truths,
    prepare_output_nll_report,
    write_output_nll_report,
)
from task_optimizers import (
    TaskOptimizerTrainerMixin,
    configure_task_optimizer,
)
from training_args_compat import warmup_dataclass_arguments
from arc_fp8_lora import (
    precision_manifest,
    validate_fp8_lora_model,
    validate_fp8_lora_options,
)
from arc_deepspeed import (
    build_zero3_param_offload_config,
    deepspeed_standard_gradient_checkpointing_environment,
    destroy_deepspeed_engine,
    gather_zero3_peft_state_dict,
    single_process_deepspeed_environment,
    use_long_context_deepspeed,
    validate_long_context_deepspeed_options,
)

import gc
import functools
import os
import io
import sys
import time
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling, TrainerCallback

import logging
from contextlib import ExitStack, redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15
IM_START_ID = 14
MODEL_PATH = resolve_model_path()
DELTA_MHAR_PARAMETER_MARKER = "_unsloth_delta_mhar"


def configure_qwen3_long_context_blas(model, max_train_tokens):
    """Select cuBLASLt only for its measured Qwen3 long-context regime.

    This is deliberately environment-gated because the same backend loses on
    the short ARC projection shapes.  The setting is process-wide, but a
    training lane handles one task at a time and calls this function before
    constructing each task trainer, so a later short task is switched back to
    ordinary cuBLAS.
    """
    requested = os.getenv(
        "ARC_QWEN3_AUTO_CUBLASLT",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not requested:
        return False, None, None

    threshold_value = os.getenv(
        "ARC_QWEN3_CUBLASLT_MIN_TOKENS",
        "6144",
    ).strip()
    try:
        threshold = int(threshold_value)
    except ValueError as error:
        raise ValueError(
            "ARC_QWEN3_CUBLASLT_MIN_TOKENS must be a positive integer"
        ) from error
    if threshold <= 0:
        raise ValueError(
            "ARC_QWEN3_CUBLASLT_MIN_TOKENS must be a positive integer"
        )

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen3":
        return False, None, threshold
    use_cublaslt = bool(int(max_train_tokens) >= threshold)
    backend = "cublaslt" if use_cublaslt else "cublas"
    preferred_blas_library = getattr(
        getattr(torch.backends, "cuda", None),
        "preferred_blas_library",
        None,
    )
    if not callable(preferred_blas_library):
        raise RuntimeError(
            "ARC_QWEN3_AUTO_CUBLASLT requires "
            "torch.backends.cuda.preferred_blas_library"
        )
    preferred_blas_library(backend)
    return use_cublaslt, backend, threshold


def acquire_pending_adapter_slot(
    semaphore,
    *,
    abort_event=None,
    end_time=float("inf"),
    poll_seconds=0.5,
):
    """Wait for publication capacity without hiding aborts or run expiry."""
    poll_seconds = float(poll_seconds)
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    while True:
        if abort_event is not None and abort_event.is_set():
            return False
        remaining = float(end_time) - time.time()
        if remaining <= 0:
            return False
        if semaphore.acquire(timeout=min(poll_seconds, remaining)):
            return True


def qwen35_gdn_replay_cache_options():
    """Parse the default-off exact GDN checkpoint-replay cache controls."""
    requested = os.getenv(
        "ARC_QWEN35_GDN_REPLAY_CACHE",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    projection_value = os.getenv(
        "ARC_QWEN35_GDN_REPLAY_CACHE_PROJECTIONS",
        "qkv+z",
    ).strip().lower().replace(" ", "")
    projection_aliases = {
        "qkv+z+out": ("in_proj_qkv", "in_proj_z", "out_proj"),
        "all": ("in_proj_qkv", "in_proj_z", "out_proj"),
        "qkv+z": ("in_proj_qkv", "in_proj_z"),
        "qkv,z": ("in_proj_qkv", "in_proj_z"),
        "in_proj_qkv,in_proj_z": ("in_proj_qkv", "in_proj_z"),
        "qkv": ("in_proj_qkv",),
        "in_proj_qkv": ("in_proj_qkv",),
        "z": ("in_proj_z",),
        "in_proj_z": ("in_proj_z",),
        "out": ("out_proj",),
        "out_proj": ("out_proj",),
    }
    if projection_value not in projection_aliases:
        raise ValueError(
            "ARC_QWEN35_GDN_REPLAY_CACHE_PROJECTIONS must be "
            "qkv+z+out, qkv+z, qkv, z, or out"
        )

    max_mib_value = os.getenv(
        "ARC_QWEN35_GDN_REPLAY_CACHE_MAX_MIB",
        "",
    ).strip()
    max_cache_bytes = None
    if max_mib_value:
        try:
            max_mib = float(max_mib_value)
        except ValueError as error:
            raise ValueError(
                "ARC_QWEN35_GDN_REPLAY_CACHE_MAX_MIB must be a "
                "non-negative number"
            ) from error
        if not np.isfinite(max_mib) or max_mib < 0:
            raise ValueError(
                "ARC_QWEN35_GDN_REPLAY_CACHE_MAX_MIB must be a "
                "finite non-negative number"
            )
        max_cache_bytes = int(max_mib * 1024**2)
    return (
        requested,
        projection_aliases[projection_value],
        max_cache_bytes,
    )


def load_checkpoint_attention_residuals(model, model_path=MODEL_PATH):
    """Strictly restore an experimental routed base when one is declared."""
    try:
        from unsloth.models.attention_residuals import (
            checkpoint_has_delta_mhar,
            load_qwen3_delta_mhar_sidecar,
        )
    except ImportError as error:
        if os.path.isfile(os.path.join(model_path, "delta_mhar_config.json")):
            raise RuntimeError(
                "This checkpoint requires Delta-MH, but the installed Unsloth "
                "build has no attention-residual support"
            ) from error
        return None

    if not checkpoint_has_delta_mhar(model_path):
        return None
    return load_qwen3_delta_mhar_sidecar(
        model,
        model_path,
        required=True,
    )


def ensure_training_optimization_support(
    selective_activation_recompute=False,
    hybrid_activation_recompute=False,
    fp8_frozen_linear=False,
    partial_gradient_checkpointing=False,
    bidirectional_prompt_attention=False,
    fused_qknorm_rope=False,
    attention_checkpoint_input_offload=False,
    smart_attention_checkpoint_input_offload=False,
):
    """Fail clearly when an optimization needs a newer Unsloth build."""
    from unsloth.kernels import fast_lora
    from unsloth.models import llama

    missing = []
    if selective_activation_recompute and not getattr(
        fast_lora,
        "UNSLOTH_SUPPORTS_SELECTIVE_MLP_RECOMPUTATION",
        False,
    ):
        missing.append("fused-LoRA MLP recomputation")
    if selective_activation_recompute and not getattr(
        llama,
        "UNSLOTH_SUPPORTS_SELECTIVE_ATTENTION_RECOMPUTATION",
        False,
    ):
        missing.append("decoder-layer attention recomputation")
    if hybrid_activation_recompute and not getattr(
        llama,
        "UNSLOTH_SUPPORTS_SEPARATE_MLP_RECOMPUTATION",
        False,
    ):
        missing.append("separate fused-LoRA MLP recomputation")
    if fp8_frozen_linear and not getattr(
        fast_lora,
        "UNSLOTH_SUPPORTS_FP8_FROZEN_LINEAR",
        False,
    ):
        missing.append("FP8 frozen-linear training")
    if partial_gradient_checkpointing and not getattr(
        llama,
        "UNSLOTH_SUPPORTS_PARTIAL_GRADIENT_CHECKPOINTING",
        False,
    ):
        missing.append("percentage-based gradient checkpointing")
    if bidirectional_prompt_attention:
        try:
            from unsloth.utils import arc_prompt_attention
        except ImportError:
            arc_prompt_attention = None
        if arc_prompt_attention is None or not getattr(
            arc_prompt_attention,
            "UNSLOTH_SUPPORTS_ARC_BIDIRECTIONAL_PROMPT_ATTENTION",
            False,
        ):
            missing.append("ARC segmented bidirectional prompt attention")
    if fused_qknorm_rope:
        try:
            from unsloth.models import qwen3
            from unsloth.kernels import qwen3_qknorm_rope
        except ImportError:
            qwen3 = qwen3_qknorm_rope = None
        if (
            qwen3 is None
            or qwen3_qknorm_rope is None
            or not getattr(
                qwen3,
                "UNSLOTH_SUPPORTS_FUSED_QWEN3_QKNORM_ROPE",
                False,
            )
            or not getattr(
                qwen3_qknorm_rope,
                "UNSLOTH_SUPPORTS_FUSED_QWEN3_QKNORM_ROPE",
                False,
            )
        ):
            missing.append("fused Qwen3 Q/K normalization and RoPE")
    if attention_checkpoint_input_offload and not getattr(
        llama,
        "UNSLOTH_SUPPORTS_ATTENTION_CHECKPOINT_INPUT_OFFLOAD",
        False,
    ):
        missing.append("pinned-CPU attention checkpoint inputs")
    if smart_attention_checkpoint_input_offload:
        try:
            from unsloth_zoo import gradient_checkpointing as zoo_gc
        except ImportError:
            zoo_gc = None
        if zoo_gc is None or not getattr(
            zoo_gc,
            "UNSLOTH_SUPPORTS_DEFERRED_DOUBLE_BUFFER_RELEASE",
            False,
        ):
            missing.append("safe in-place smart-checkpoint gradients")
    if missing:
        raise RuntimeError(
            "The requested training optimization requires the patched "
            "Unsloth package from nguyen599/unsloth main. Missing support "
            "for: "
            f"{', '.join(missing)}. Install the matching Unsloth build or "
            "disable the corresponding ARC optimization option."
        )


def ensure_arc_tokenizer_backend(tokenizer, model_path=MODEL_PATH):
    """Restore the checkpoint's WordLevel backend if Transformers rebuilt it."""
    probe = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    expected_ids = [
        IM_START_ID,
        USER_TOKEN_ID,
        ARC_VOCAB["Ċ"],
        EOS_ID,
        ARC_VOCAB["Ċ"],
        IM_START_ID,
        ASSISTANT_TOKEN_ID,
        ARC_VOCAB["Ċ"],
    ]
    if tokenizer.encode(probe, add_special_tokens=False) == expected_ids:
        return tokenizer

    from tokenizers import Tokenizer

    if not hasattr(tokenizer, "_tokenizer"):
        raise RuntimeError(
            "The loaded tokenizer dropped ARC chat control tokens and its "
            "backend cannot be replaced"
        )
    tokenizer._tokenizer = Tokenizer.from_file(
        os.path.join(model_path, "tokenizer.json")
    )
    if tokenizer.encode(probe, add_special_tokens=False) != expected_ids:
        raise RuntimeError("Failed to restore ARC chat control tokens")
    return tokenizer


def configure_gradient_checkpointing_interval(model, interval):
    """Checkpoint only every Nth decoder layer while keeping model-level GC."""
    interval = max(1, int(interval))
    model._unsloth_gradient_checkpointing_interval = interval
    configured_modules = 0
    checkpointed_layers = 0
    total_layers = 0
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is None or not hasattr(module, "gradient_checkpointing"):
            continue
        module._gradient_checkpointing_interval = interval
        configured_modules += 1
        for index, layer in enumerate(layers):
            if not hasattr(layer, "gradient_checkpointing"):
                continue
            layer.gradient_checkpointing = (
                bool(module.gradient_checkpointing)
                and index % interval == 0
            )
            total_layers += 1
            checkpointed_layers += int(layer.gradient_checkpointing)
    return configured_modules, checkpointed_layers, total_layers


def configure_gradient_checkpointing_percentage(model, percentage):
    """Checkpoint an exact, uniformly distributed percentage of decoder layers."""
    percentage = float(percentage)
    model._unsloth_gradient_checkpointing_percentage = percentage
    checkpointed_layers, total_layers = (
        FastLanguageModel.set_gradient_checkpointing_percentage(
            model,
            percentage,
        )
    )
    return 1, checkpointed_layers, total_layers


def gradient_checkpointing_layer_counts(model):
    """Return enabled and total decoder-layer checkpoint flags."""
    checkpointed_layers = 0
    total_layers = 0
    seen_layers = set()
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is None:
            continue
        for layer in layers:
            layer_id = id(layer)
            if (
                layer_id in seen_layers
                or not hasattr(layer, "gradient_checkpointing")
            ):
                continue
            seen_layers.add(layer_id)
            total_layers += 1
            checkpointed_layers += int(layer.gradient_checkpointing)
    return checkpointed_layers, total_layers


def wrapped_callable_name(function):
    """Return a checkpoint callable's name through partial wrappers."""
    seen = set()
    while isinstance(function, functools.partial) and id(function) not in seen:
        seen.add(id(function))
        function = function.func
    return getattr(function, "__name__", type(function).__name__)


def validate_unsloth_gradient_checkpointing_boundary(model):
    """Ensure enabled decoder layers use Unsloth's CPU-offload checkpoint."""
    checkpointed_layers = 0
    invalid_functions = {}
    seen_layers = set()
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is None:
            continue
        for index, layer in enumerate(layers):
            layer_id = id(layer)
            if layer_id in seen_layers:
                continue
            seen_layers.add(layer_id)
            if not bool(getattr(layer, "gradient_checkpointing", False)):
                continue
            checkpointed_layers += 1
            function_name = wrapped_callable_name(
                getattr(layer, "_gradient_checkpointing_func", None)
            )
            if function_name != "unsloth_checkpoint":
                invalid_functions[index] = function_name
    if checkpointed_layers == 0:
        raise RuntimeError(
            "Unsloth gradient-checkpointing mode is active, but no decoder "
            "layers are checkpointed"
        )
    if invalid_functions:
        raise RuntimeError(
            "Unsloth gradient-checkpointing mode was not installed at the "
            f"decoder boundary: {invalid_functions}"
        )
    return checkpointed_layers


def reset_torch_compiler_cache():
    """Release compiled graphs that can retain a completed task's tensors."""
    compiler_reset = getattr(
        getattr(torch, "compiler", None),
        "reset",
        None,
    )
    if callable(compiler_reset):
        compiler_reset()
        return
    dynamo_reset = getattr(
        getattr(torch, "_dynamo", None),
        "reset",
        None,
    )
    if callable(dynamo_reset):
        dynamo_reset()


def configure_activation_recomputation(
    model,
    recompute_mlp,
    recompute_attention,
):
    """Set the two sublayer recomputation modes on a loaded model."""
    mlp_count = 0
    attention_count = 0
    for module in model.modules():
        if all(
            hasattr(module, projection)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ):
            module._unsloth_recompute_mlp_activations = recompute_mlp
            module._unsloth_checkpoint_mlp_module = False
            mlp_count += int(bool(recompute_mlp))
        if (
            hasattr(module, "self_attn")
            and hasattr(module, "input_layernorm")
            and hasattr(module, "mlp")
        ):
            module._unsloth_recompute_attention_activations = (
                recompute_attention
            )
            attention_count += int(bool(recompute_attention))
    return mlp_count, attention_count


def configure_hybrid_activation_recomputation(
    model,
    recompute_mlp,
    recompute_attention,
    attention_recompute_percentage=100.0,
):
    """Recompute sublayers only where the complete decoder is not checkpointed."""
    attention_recompute_percentage = float(attention_recompute_percentage)
    if not 0.0 <= attention_recompute_percentage <= 100.0:
        raise ValueError(
            "Hybrid attention recomputation percentage must be in [0, 100]"
        )
    mlp_count = 0
    attention_count = 0
    seen_layers = set()
    decoder_layer_groups = []
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is None:
            continue
        layer_group = []
        for layer in layers:
            if id(layer) in seen_layers or not hasattr(layer, "mlp"):
                continue
            seen_layers.add(id(layer))
            layer_group.append(layer)
        if layer_group:
            decoder_layer_groups.append(layer_group)

    for layers in decoder_layer_groups:
        eligible_attention_layers = [
            layer
            for layer in layers
            if not bool(getattr(layer, "gradient_checkpointing", False))
        ]
        active_attention_count = int(
            np.ceil(
                len(eligible_attention_layers)
                * attention_recompute_percentage
                / 100.0
            )
        )
        if active_attention_count:
            active_attention_indices = {
                int(
                    index
                    * len(eligible_attention_layers)
                    / active_attention_count
                )
                for index in range(active_attention_count)
            }
        else:
            active_attention_indices = set()
        active_attention_layer_ids = {
            id(eligible_attention_layers[index])
            for index in active_attention_indices
        }

        for layer in layers:
            fully_checkpointed = bool(
                getattr(layer, "gradient_checkpointing", False)
            )
            mlp_active = bool(recompute_mlp) and not fully_checkpointed
            attention_active = (
                bool(recompute_attention)
                and not fully_checkpointed
                and id(layer) in active_attention_layer_ids
            )
            layer.mlp._unsloth_recompute_mlp_activations = mlp_active
            layer.mlp._unsloth_checkpoint_mlp_module = False
            layer._unsloth_recompute_attention_activations = attention_active
            mlp_count += int(mlp_active)
            attention_count += int(attention_active)
    return mlp_count, attention_count


def configure_fused_qknorm_rope(model, enabled):
    """Opt eligible Qwen3 attention modules into exact-BF16 QKNorm/RoPE."""
    count = 0
    seen = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if not (
            hasattr(module, "q_norm")
            and hasattr(module, "k_norm")
            and hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
        ):
            continue
        module._unsloth_fused_qknorm_rope = bool(enabled)
        count += 1
    return count


def configure_attention_checkpoint_input_offload(
    model,
    enabled,
    checkpoint_mode="standard",
):
    """Offload only inputs retained by attention-only checkpoints."""
    saved_cpu_count = 0
    smart_buffer_count = 0
    seen = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if not (hasattr(module, "self_attn") and hasattr(module, "mlp")):
            continue
        active = bool(enabled) and bool(
            getattr(
                module,
                "_unsloth_recompute_attention_activations",
                False,
            )
        )
        use_saved_cpu = active and checkpoint_mode == "standard"
        use_smart_double_buffer = active and checkpoint_mode == "unsloth"
        module._unsloth_offload_attention_checkpoint_input = use_saved_cpu
        saved_cpu_count += int(use_saved_cpu)
        smart_buffer_count += int(use_smart_double_buffer)
    return saved_cpu_count, smart_buffer_count


class UnslothFixedTrainer(TaskOptimizerTrainerMixin, UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def _wrap_model(self, model, training=True, dataloader=None):
        checkpoint_interval = getattr(
            model,
            "_unsloth_gradient_checkpointing_interval",
            1,
        )
        checkpoint_percentage = getattr(
            model,
            "_unsloth_gradient_checkpointing_percentage",
            None,
        )
        if training and self.args.gradient_checkpointing:
            if checkpoint_percentage is None:
                configure_gradient_checkpointing_interval(
                    model,
                    checkpoint_interval,
                )
            else:
                configure_gradient_checkpointing_percentage(
                    model,
                    checkpoint_percentage,
                )
        model = super()._wrap_model(
            model,
            training=training,
            dataloader=dataloader,
        )
        if training and self.args.gradient_checkpointing:
            if checkpoint_percentage is None:
                configure_gradient_checkpointing_interval(
                    model,
                    checkpoint_interval,
                )
            else:
                configure_gradient_checkpointing_percentage(
                    model,
                    checkpoint_percentage,
                )
        ddp_handler = getattr(self.accelerator, "ddp_handler", None)
        if (
            training
            and torch.distributed.is_initialized()
            and self.args.gradient_checkpointing
            and ddp_handler is not None
        ):
            # Unsloth's Qwen path uses reentrant gradient checkpointing.
            # Static DDP supports repeated checkpoint hooks for a fixed graph.
            ddp_handler.static_graph = True
        return model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        graph_diagnostics = (
            os.getenv("ARC_TRAIN_GRAPH_DIAGNOSTICS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if graph_diagnostics and not getattr(
            self,
            "_arc_graph_input_reported",
            False,
        ):
            diagnostic_labels = inputs.get("labels")
            supervised = (
                int(diagnostic_labels.ne(-100).sum().item())
                if isinstance(diagnostic_labels, torch.Tensor)
                else None
            )
            label_values = (
                sorted(
                    int(value)
                    for value in diagnostic_labels[diagnostic_labels.ne(-100)]
                    .unique()
                    .detach()
                    .cpu()
                    .tolist()
                )
                if isinstance(diagnostic_labels, torch.Tensor)
                else None
            )
            dynamo_plugin = getattr(
                getattr(self.accelerator, "state", None),
                "dynamo_plugin",
                None,
            )
            print(
                "[ARC graph diagnostic] input: "
                f"labels_shape={getattr(diagnostic_labels, 'shape', None)}, "
                f"supervised_tokens={supervised}, "
                f"supervised_label_ids={label_values}, "
                f"trainer_torch_compile={self.args.torch_compile}, "
                f"dynamo_backend={getattr(dynamo_plugin, 'backend', None)}",
                file=sys.__stdout__,
                flush=True,
            )
            self._arc_graph_input_reported = True
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if graph_diagnostics and not getattr(
            self,
            "_arc_graph_forward_reported",
            False,
        ):
            logits = (
                outputs.get("logits")
                if isinstance(outputs, dict)
                else getattr(outputs, "logits", None)
            )
            print(
                "[ARC graph diagnostic] forward: "
                f"loss_requires_grad={getattr(loss, 'requires_grad', None)}, "
                f"loss_grad_fn={type(getattr(loss, 'grad_fn', None)).__name__}, "
                f"logits_requires_grad="
                f"{getattr(logits, 'requires_grad', None)}, "
                f"logits_grad_fn="
                f"{type(getattr(logits, 'grad_fn', None)).__name__}",
                file=sys.__stdout__,
                flush=True,
            )
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                def report_loss_gradient(gradient):
                    print(
                        "[ARC graph diagnostic] loss gradient: "
                        f"shape={gradient.shape}, "
                        f"dtype={gradient.dtype}, "
                        f"value={gradient.detach().float().cpu().tolist()}",
                        file=sys.__stdout__,
                        flush=True,
                    )
                    return gradient

                loss.register_hook(report_loss_gradient)
            self._arc_graph_forward_reported = True
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, *step_args, **step_kwargs):
        graph_diagnostics = (
            os.getenv("ARC_TRAIN_GRAPH_DIAGNOSTICS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if graph_diagnostics and not getattr(
            self,
            "_arc_graph_model_reported",
            False,
        ):
            print(
                "[ARC graph diagnostic] training model: "
                f"type={type(model).__module__}.{type(model).__name__}, "
                f"has_orig_mod={hasattr(model, '_orig_mod')}",
                file=sys.__stdout__,
                flush=True,
            )
            self._arc_graph_model_reported = True
        if (
            getattr(self, "_arc_gradient_checkpointing_mode", "standard")
            == "unsloth"
            and not getattr(
                self,
                "_arc_unsloth_checkpoint_boundary_validated",
                False,
            )
        ):
            validate_unsloth_gradient_checkpointing_boundary(model)
            self._arc_unsloth_checkpoint_boundary_validated = True
        replay_cache_setup = getattr(
            self,
            "_arc_qwen35_gdn_replay_cache_setup",
            None,
        )
        if (
            replay_cache_setup is not None
            and getattr(
                self,
                "_arc_qwen35_gdn_replay_cache_controller",
                None,
            )
            is None
        ):
            # Trainer installs the final model-local checkpoint callable only
            # inside the train loop.  Patch immediately after the boundary is
            # validated and before this first forward.
            self._arc_qwen35_gdn_replay_cache_controller = (
                replay_cache_setup(model)
            )
        step_result = super().training_step(
            model,
            *step_args,
            **step_kwargs,
        )
        if graph_diagnostics and not getattr(
            self,
            "_arc_graph_backward_reported",
            False,
        ):
            selected = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                category = None
                if "lm_head" in name and ".lora_B." in name:
                    category = "lm_head_lora_B"
                elif ".lora_B." in name:
                    category = "projection_lora_B"
                elif (
                    "embed_tokens" in name
                    and ".modules_to_save." in name
                ):
                    category = "embed_tokens"
                if category is not None and category not in selected:
                    selected[category] = (name, parameter)
            details = []
            for category, (name, parameter) in selected.items():
                grad = parameter.grad
                details.append(
                    f"{category}={name}:"
                    + (
                        "grad=None"
                        if grad is None
                        else f"grad_norm={grad.float().norm().item():.9g}"
                    )
                )
            print(
                "[ARC graph diagnostic] backward: " + "; ".join(details),
                file=sys.__stdout__,
                flush=True,
            )
            self._arc_graph_backward_reported = True
        return step_result


class TrainingProfilerCallback(TrainerCallback):

    def __init__(self, profiler):
        self.profiler = profiler

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def __init__(
        self,
        *args,
        final_assistant_only=False,
        bidirectional_prompt_attention=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.final_assistant_only = bool(final_assistant_only)
        self.bidirectional_prompt_attention = bool(
            bidirectional_prompt_attention
        )

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            batch["labels"][i, :] = -100
            spans = assistant_completion_spans(
                labels.tolist(),
                user_token_id=USER_TOKEN_ID,
                assistant_token_id=ASSISTANT_TOKEN_ID,
                eos_token_id=EOS_ID,
                final_only=self.final_assistant_only,
            )
            for start, end in spans:
                batch["labels"][i, start:end] = labels[start:end]
        if self.bidirectional_prompt_attention:
            prompt_mask = batch["labels"].eq(-100)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                real_tokens = attention_mask.to(dtype=torch.bool)
                if not bool(real_tokens.all().item()):
                    raise RuntimeError(
                        "ARC bidirectional prompt attention currently "
                        "requires an unpadded physical batch; use "
                        "per_device_train_batch_size=1"
                    )
                prompt_mask &= real_tokens
            batch["arc_bidirectional_prompt_mask"] = prompt_mask
        return batch


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:

    n = logits.size(0)

    nll = arc_token_nll(logits, scores)

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_index, t in enumerate(ARC_TOKENS):
            score = nll[i, token_index].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]

    while time.time() - start_time < 540*20 and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


def arc_token_nll(logits, scores):
    """Return candidate NLLs without copying the full vocabulary to CPU."""
    float_logits = logits.float()
    token_ids = torch.tensor(
        ARC_TOKENS,
        device=logits.device,
        dtype=torch.long,
    )
    selected_logits = float_logits.index_select(-1, token_ids)
    normalizer = torch.logsumexp(float_logits, dim=-1, keepdim=True)
    accumulated_scores = torch.as_tensor(
        scores,
        device=logits.device,
        dtype=torch.float32,
    ).view(-1, 1)
    return (
        accumulated_scores + normalizer - selected_logits
    ).cpu()


def select_cache_batch(cache, indices):
    """Select and duplicate cache rows for a batched search frontier."""
    from arc_rope_3d_lora import ArcRopeCache

    if isinstance(cache, ArcRopeCache):
        return cache.select(indices)
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(indices)
        return cache

    if torch.is_tensor(cache):
        return cache.index_select(0, indices.to(cache.device))
    if isinstance(cache, tuple):
        return tuple(select_cache_batch(item, indices) for item in cache)
    if isinstance(cache, list):
        return [select_cache_batch(item, indices) for item in cache]
    raise TypeError(
        f"Unsupported cache type for frontier decoding: {type(cache).__name__}"
    )


def reset_unsloth_fast_inference_state(model):
    """Force Unsloth to rebuild fixed-batch scratch buffers from the KV cache."""
    modules = getattr(
        model,
        "_unsloth_fast_inference_attention_modules",
        None,
    )
    if modules is None:
        modules = [
            module
            for module in model.modules()
            if all(
                hasattr(module, projection)
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        ]
        model._unsloth_fast_inference_attention_modules = modules
    for module in modules:
        # Qwen's fast path uses the existence of paged_attention as its
        # prefill marker, but every tensor below is also fixed to that
        # prefill batch size. Clear all of them so a frontier that grows or
        # shrinks cannot reuse a stale RH_Q/temp buffer from the prior depth.
        for attribute in (
            "paged_attention_K",
            "paged_attention_V",
            "paged_attention",
            "temp_QA",
            "temp_KV",
            "temp_O",
            "RH_Q",
            "attention",
        ):
            if hasattr(module, attribute):
                delattr(module, attribute)


def _inference_turbo_frontier_group(
    model,
    logits,
    cache,
    origin_count,
    position,
    max_new_tokens,
    max_score,
    search_started_at,
    search_deadline,
    max_live_branches,
):
    """Explore one bounded group of live DFS origins at each depth."""
    origins = list(range(origin_count))
    scores = [0.0] * origin_count
    paths = [[] for _ in origins]
    suffixes = defaultdict(list)
    remaining_tokens = max_new_tokens
    while origins and remaining_tokens > 0:
        if len(origins) > max_live_branches:
            recursive_suffixes = turbo_dfs(
                model,
                logits=logits,
                max_new_tokens=remaining_tokens,
                max_score=max_score,
                scores=scores,
                pos=position,
                cache=cache,
                start_time=search_started_at,
                end_time=search_deadline,
            )
            for frontier_index, beams in recursive_suffixes.items():
                origin = origins[frontier_index]
                path = paths[frontier_index]
                suffixes[origin].extend(
                    (score, path + suffix)
                    for score, suffix in beams
                )
            break

        nll = arc_token_nll(logits, scores)

        parent_indices = []
        next_tokens = []
        next_origins = []
        next_scores = []
        next_paths = []

        for frontier_index, origin in enumerate(origins):
            candidates = []
            for token_index, token in enumerate(ARC_TOKENS):
                score = nll[frontier_index, token_index].item()
                if score < max_score:
                    candidates.append((score, token))

            for score, token in sorted(candidates):
                if token == EOS_ID:
                    suffixes[origin].append(
                        (score, paths[frontier_index] + [token])
                    )
                elif remaining_tokens > 1:
                    parent_indices.append(frontier_index)
                    next_tokens.append(token)
                    next_origins.append(origin)
                    next_scores.append(score)
                    next_paths.append(paths[frontier_index] + [token])

        if not next_tokens:
            break
        if time.time() >= search_deadline:
            break

        cache_indices = torch.tensor(
            parent_indices,
            device=model.device,
            dtype=torch.long,
        )
        cache = select_cache_batch(cache, cache_indices)
        # Unsloth's fast Qwen inference stores a private paged-attention cache
        # and scratch tensors sized to the prefill batch. A frontier can grow
        # or shrink on every token, so force the next call to repopulate those
        # private buffers from our newly selected public KV cache.
        reset_unsloth_fast_inference_state(model)
        outputs = model(
            input_ids=torch.tensor(
                next_tokens,
                device=model.device,
                dtype=torch.long,
            ).view(-1, 1),
            position_ids=torch.full(
                (len(next_tokens), 1),
                position,
                device=model.device,
            ),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        logits = outputs.logits[:, -1]
        cache = outputs.past_key_values
        origins = next_origins
        scores = next_scores
        paths = next_paths
        remaining_tokens -= 1
        position += 1

    return suffixes


@torch.no_grad()
def inference_turbo_frontier(
    model,
    prefix_tokens,
    max_new_tokens,
    max_score,
    end_time,
    origin_batch_size=4,
    max_live_branches=8,
):
    """Batch small frontiers, then fall back to DFS before they grow large."""
    input_ids = torch.tensor(
        prefix_tokens,
        device=model.device,
        dtype=torch.long,
    )
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)

    logits = outputs.logits[:, -1]
    cache = outputs.past_key_values
    origin_count = input_ids.size(0)
    origin_batch_size = max(
        1,
        min(int(origin_batch_size), origin_count),
    )
    position = input_ids.size(1)
    result = []
    search_started_at = time.time()
    search_deadline = min(search_started_at + 540, end_time)
    max_live_branches = max(1, int(max_live_branches))

    for origin_start in range(0, origin_count, origin_batch_size):
        if time.time() >= search_deadline:
            break
        origin_stop = min(
            origin_start + origin_batch_size,
            origin_count,
        )
        origin_indices = torch.arange(
            origin_start,
            origin_stop,
            device=model.device,
            dtype=torch.long,
        )
        group_suffixes = _inference_turbo_frontier_group(
            model,
            logits.index_select(0, origin_indices),
            select_cache_batch(cache, origin_indices),
            origin_stop - origin_start,
            position,
            max_new_tokens,
            max_score,
            search_started_at,
            search_deadline,
            max_live_branches,
        )
        result.extend(
            (
                origin_start + batch_id,
                sorted(beams, key=lambda item: item[0]),
            )
            for batch_id, beams in group_suffixes.items()
        )

    return [
        item
        for item in sorted(result, key=lambda item: item[0])
    ]


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


def infer_same_shape_reply(query):
    """Backward-compatible wrapper for conservative shape inference."""
    return infer_demonstrated_output_shape(query)


@torch.no_grad()
def inference_shape_beam(
    model,
    prefix_tokens,
    output_shape,
    end_time,
    beam_width=4,
):
    """Decode fixed-shape ARC grids with a small non-monochrome beam.

    Cell positions retain the best ``beam_width`` continuations per prompt.
    Row separators and the final end token are forced by ``output_shape``.
    The final cell must make the grid contain at least two colors. This is a
    bounded last-resort path for outputs where threshold DFS produced no
    usable multi-color grid.
    """
    if not prefix_tokens:
        return []
    token_lengths = {len(tokens) for tokens in prefix_tokens}
    if len(token_lengths) != 1:
        raise ValueError(
            "Shape-greedy prompts must have equal token lengths, got "
            f"{sorted(token_lengths)}"
        )
    height, width = output_shape
    if not (0 < height <= 30 and 0 < width <= 30):
        raise ValueError(f"Invalid ARC output shape: {output_shape}")
    if height * width < 2:
        raise ValueError(
            "A non-monochrome fallback requires at least two cells, got "
            f"{output_shape}"
        )
    beam_width = int(beam_width)
    if beam_width < 2:
        raise ValueError(
            f"Shape fallback beam width must be >=2, got {beam_width}"
        )

    # Normal recursive calls keep a fixed batch of four. The fallback may use
    # a smaller batch, so invalidate Unsloth's private fixed-batch buffers
    # before its new prefill.
    reset_unsloth_fast_inference_state(model)
    input_ids = torch.tensor(
        prefix_tokens,
        device=model.device,
        dtype=torch.long,
    )
    outputs = model(
        input_ids=input_ids,
        return_dict=True,
        use_cache=True,
    )
    logits = outputs.logits[:, -1]
    cache = outputs.past_key_values
    position = input_ids.size(1)
    origin_count = input_ids.size(0)
    current_width = 1
    scores = [0.0] * origin_count
    paths = [[] for _ in range(origin_count)]
    color_masks = [0] * origin_count

    for row in range(height):
        for column in range(width):
            if time.time() >= end_time:
                return []
            nll = arc_token_nll(logits, scores)
            is_final_cell = (
                row == height - 1 and column == width - 1
            )
            parent_indices = []
            next_tokens = []
            next_scores = []
            next_paths = []
            next_color_masks = []

            for origin in range(origin_count):
                beam_start = origin * current_width
                beam_stop = beam_start + current_width
                candidate_scores = nll[beam_start:beam_stop, :10].clone()
                if is_final_cell:
                    for local_parent in range(current_width):
                        mask = color_masks[beam_start + local_parent]
                        for digit in range(10):
                            if (mask | (1 << digit)).bit_count() < 2:
                                candidate_scores[local_parent, digit] = (
                                    float("inf")
                                )
                flat_scores = candidate_scores.reshape(-1)
                selected_scores, selected_indices = torch.topk(
                    flat_scores,
                    k=beam_width,
                    largest=False,
                    sorted=True,
                )
                for selected_score, selected_index in zip(
                    selected_scores.tolist(),
                    selected_indices.tolist(),
                ):
                    local_parent, digit = divmod(selected_index, 10)
                    parent = beam_start + local_parent
                    parent_indices.append(parent)
                    next_tokens.append(ARC_TOKENS[digit])
                    next_scores.append(float(selected_score))
                    next_paths.append(
                        paths[parent] + [ARC_TOKENS[digit]]
                    )
                    next_color_masks.append(
                        color_masks[parent] | (1 << digit)
                    )

            cache_indices = torch.tensor(
                parent_indices,
                device=model.device,
                dtype=torch.long,
            )
            identity_indices = torch.arange(
                len(parent_indices),
                device=model.device,
                dtype=torch.long,
            )
            if (
                len(parent_indices) != len(paths)
                or not torch.equal(cache_indices, identity_indices)
            ):
                cache = select_cache_batch(cache, cache_indices)
                reset_unsloth_fast_inference_state(model)
            scores = next_scores
            paths = next_paths
            color_masks = next_color_masks
            current_width = beam_width

            outputs = model(
                input_ids=torch.tensor(
                    next_tokens,
                    device=model.device,
                    dtype=torch.long,
                ).view(-1, 1),
                position_ids=torch.full(
                    (len(next_tokens), 1),
                    position,
                    device=model.device,
                ),
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )
            logits = outputs.logits[:, -1]
            cache = outputs.past_key_values
            position += 1

        terminal_token = EOS_ID if row == height - 1 else ARC_VOCAB["Ċ"]
        nll = arc_token_nll(logits, scores)
        terminal_index = ARC_TOKENS.index(terminal_token)
        scores = [
            float(nll[i, terminal_index].item()) for i in range(len(paths))
        ]
        for path in paths:
            path.append(terminal_token)
        if terminal_token == EOS_ID:
            break
        if time.time() >= end_time:
            return []
        outputs = model(
            input_ids=torch.full(
                (len(paths), 1),
                terminal_token,
                device=model.device,
                dtype=torch.long,
            ),
            position_ids=torch.full(
                (len(paths), 1),
                position,
                device=model.device,
            ),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )
        logits = outputs.logits[:, -1]
        cache = outputs.past_key_values
        position += 1

    result = []
    for origin in range(origin_count):
        beam_start = origin * current_width
        beam_stop = beam_start + current_width
        result.append((
            origin,
            sorted(
                [
                    (scores[index], paths[index])
                    for index in range(beam_start, beam_stop)
                ],
                key=lambda item: item[0],
            ),
        ))
    return result


@torch.no_grad()
def calc_scores_batch(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    logits_to_keep = max(
        max_len - length + len(answer_tokens) + 1
        for length, answer_tokens in zip(
            batch_lengths,
            batch_answer_tokens,
        )
    )
    outputs = model(
        input_ids=input_ids,
        return_dict=True,
        use_cache=True,
        logits_to_keep=logits_to_keep,
    )
    batch_logits = outputs.logits.float()
    window_start = max_len - logits_to_keep
    result_tensors = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_start = query_length - 1 - window_start
        answer_logits = logits[
            answer_start:answer_start + len(answer_tokens)
        ]
        answer_ids = torch.tensor(
            answer_tokens,
            device=answer_logits.device,
            dtype=torch.long,
        )
        selected_logits = answer_logits.gather(
            1,
            answer_ids.view(-1, 1),
        ).squeeze(1)
        result_tensors.append(
            (
                torch.logsumexp(answer_logits, dim=-1)
                - selected_logits
            ).sum()
        )
    return torch.stack(result_tensors).cpu().tolist()


def calc_scores(
    queries,
    answers,
    tokenizer,
    model,
    batch_size=4,
):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Score batch size must be positive")
    result = []
    for offset in range(0, len(queries), batch_size):
        result.extend(
            calc_scores_batch(
                queries[offset:offset + batch_size],
                answers[offset:offset + batch_size],
                tokenizer,
                model,
            )
        )
    return result


def build_inference_batches(test_id_to_subkeys, batch_size):
    """Group compatible augmentations up to any positive batch capacity."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Inference batch size must be positive")
    if batch_size == 4:
        batches = []
        # Preserve the original ordering across multiple test outputs.
        for offset_groups in (
            ((0, 4), (2, 6)),
            ((8, 12), (10, 14)),
        ):
            for subkeys in test_id_to_subkeys.values():
                for offsets in offset_groups:
                    batch = []
                    for offset in offsets:
                        batch.extend(subkeys[offset:offset + 2])
                    batches.append(batch)
        return batches

    if batch_size == 8:
        # Original-orientation families have matching row counts, as do the
        # rotated-orientation families. Mixing those two groups can produce
        # ragged token lists for rectangular ARC grids.
        offset_groups = (
            (0, 4, 10, 14),
            (2, 6, 8, 12),
        )
    else:
        # Each family contains the eight views known to share rectangular
        # tensor shapes in the Transformers/Unsloth backend. Split those
        # families by the requested maximum instead of imposing 4 or 8.
        compatible_offset_families = (
            (0, 4, 10, 14),
            (2, 6, 8, 12),
        )
        batches = []
        for subkeys in test_id_to_subkeys.values():
            for offsets in compatible_offset_families:
                family = []
                for offset in offsets:
                    family.extend(subkeys[offset:offset + 2])
                batches.extend(
                    family[start:start + batch_size]
                    for start in range(0, len(family), batch_size)
                    if family[start:start + batch_size]
                )
        return batches

    batches = []
    for subkeys in test_id_to_subkeys.values():
        for offsets in offset_groups:
            batch = []
            for offset in offsets:
                batch.extend(subkeys[offset:offset + 2])
            batches.append(batch)
    return batches


def unwrap_optimizer(optimizer):
    """Remove Accelerate-style optimizer wrappers without losing state."""
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        wrapped_optimizer = getattr(optimizer, "optimizer", None)
        if wrapped_optimizer is None or wrapped_optimizer is optimizer:
            break
        optimizer = wrapped_optimizer
    return optimizer


def optimizer_state_memory(optimizer):
    """Summarize unique optimizer-state tensors by device and dtype."""
    optimizer = unwrap_optimizer(optimizer)
    result = {
        "total_bytes": 0,
        "cuda_bytes": 0,
        "cpu_bytes": 0,
        "tensor_count": 0,
        "by_device_dtype": defaultdict(int),
    }
    seen_tensors = set()

    def visit(value):
        if torch.is_tensor(value):
            tensor_id = id(value)
            if tensor_id in seen_tensors:
                return
            seen_tensors.add(tensor_id)
            num_bytes = value.numel() * value.element_size()
            device_type = value.device.type
            dtype_name = str(value.dtype).removeprefix("torch.")
            result["total_bytes"] += num_bytes
            result[f"{device_type}_bytes"] = (
                result.get(f"{device_type}_bytes", 0) + num_bytes
            )
            result["tensor_count"] += 1
            result["by_device_dtype"][(device_type, dtype_name)] += num_bytes
            return
        if isinstance(value, dict):
            for nested_value in value.values():
                visit(nested_value)
            return
        if isinstance(value, (list, tuple)):
            for nested_value in value:
                visit(nested_value)

    if optimizer is not None:
        visit(getattr(optimizer, "state", {}))
    result["by_device_dtype"] = dict(result["by_device_dtype"])
    return result


def log_optimizer_state_memory(rank, optimizer, phase):
    """Print optimizer-state residency without allocating CUDA tensors."""
    summary = optimizer_state_memory(optimizer)
    mib = 1024**2
    details = ", ".join(
        f"{device}/{dtype}={num_bytes / mib:.1f}MiB"
        for (device, dtype), num_bytes in sorted(
            summary["by_device_dtype"].items()
        )
    )
    if not details:
        details = "none"
    print(
        f"[Rank {rank}] optimizer state {phase}: "
        f"total={summary['total_bytes'] / mib:.1f}MiB, "
        f"cuda={summary['cuda_bytes'] / mib:.1f}MiB, "
        f"cpu={summary['cpu_bytes'] / mib:.1f}MiB, "
        f"tensors={summary['tensor_count']} ({details})",
        file=sys.__stdout__,
        flush=True,
    )


def restart_optimizer_learning_rate(optimizer, learning_rate):
    """Start a fresh per-task scheduler while retaining optimizer moments."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        param_group["initial_lr"] = learning_rate


def run_task_grpo(
    *,
    model,
    tokenizer,
    puzzle_dataset,
    formatter,
    task_key,
    rank,
    max_seq_length,
    max_new_tokens,
    gradient_checkpointing,
    gradient_checkpointing_mode,
    gradient_checkpointing_interval,
    gradient_checkpointing_percentage,
    gradient_checkpointing_profile,
    max_grad_norm,
    max_steps,
    num_generations,
    micro_batch_size,
    auto_micro_batch,
    augmentations,
    max_prompt_tokens,
    min_cell_accuracy,
    exact_bonus,
    temperature,
    learning_rate,
    optimizer,
    flash_adamw_quantize,
    flash_adamw_master_weight_bits,
    normuon_adjust_lr,
    normuon_momentum,
    normuon_beta2,
    normuon_megabatch_size,
    normuon_use_triton,
    muon_adjust_lr,
    muon_momentum,
    muon_nesterov,
    muon_ns_chunk_size,
    muon_min_8bit_size,
    muon_scalar_optimizer,
    color_signature_colors,
    excluded_color_signatures,
    rollout_backend,
    vllm_job_queue=None,
    vllm_response_queue=None,
    vllm_adapter_root=None,
    vllm_run_id=None,
    vllm_task_adapter_id=None,
    vllm_initial_embedding_state=None,
    vllm_peft_config=None,
    vllm_service_index=None,
    vllm_response_timeout_seconds=1200.0,
):
    """Run one task-local GRPO phase with native or external generation."""
    from trl import GRPOConfig, GRPOTrainer

    rl_rows = build_task_rl_rows(
        puzzle_dataset,
        formatter,
        augmentations=augmentations,
        seed=stable_task_seed(
            task_key,
            seed=7,
            namespace="arc-rl-data",
        ),
        max_prompt_tokens=max_prompt_tokens,
        color_signature_colors=color_signature_colors,
        excluded_color_signatures=excluded_color_signatures,
    )
    if not rl_rows:
        raise ValueError(f"Puzzle {task_key} has no leave-one-out RL rows")

    prompt_token_counts = [
        len(tokenizer.encode(row["prompt"])) for row in rl_rows
    ]
    actual_max_prompt_tokens = max(prompt_token_counts)
    rl_color_signatures = color_mapping_signatures(
        rl_rows,
        color_signature_colors,
    )
    color_overlap = rl_color_signatures.intersection(
        excluded_color_signatures
    )
    if color_overlap:
        raise RuntimeError(
            "RL/SFT color split failed; overlapping permutations: "
            f"{sorted(color_overlap)}"
        )
    # A valid reward completion must have one of the known held-out target
    # shapes.  The generic ARC limit reserves a 30x30 grid (931 tokens), which
    # lets a malformed sample run hundreds of useless tokens and then be
    # masked as truncated.  Bound rollout by the largest valid target in this
    # task, plus one safety token for tokenizer-version differences.
    target_completion_limit = max(
        len(
            tokenizer.encode(
                formatter.fmt_reply([row["target_grid"]]),
                add_special_tokens=False,
            )
        )
        for row in rl_rows
    ) + 1
    completion_limit = min(
        int(max_new_tokens),
        int(max_seq_length) - int(max_prompt_tokens),
        target_completion_limit,
    )
    if completion_limit <= 0:
        raise ValueError(
            "RL max prompt tokens must leave space for a completion inside "
            f"the {max_seq_length}-token model context"
        )

    grpo_sequence_tokens = actual_max_prompt_tokens + completion_limit
    requested_micro_batch_size = micro_batch_size
    micro_batch_token_limit = select_rl_micro_batch_token_limit(optimizer)
    active_micro_batch_size = (
        select_rl_micro_batch_size(
            grpo_sequence_tokens,
            requested_micro_batch_size,
            token_limit=micro_batch_token_limit,
        )
        if auto_micro_batch
        else requested_micro_batch_size
    )
    requested_grpo_checkpointing_percentage = (
        gradient_checkpointing_percentage
    )
    grpo_checkpoint_policy_tokens = (
        grpo_sequence_tokens * active_micro_batch_size
    )
    if requested_grpo_checkpointing_percentage == "auto":
        active_grpo_checkpointing_percentage = (
            select_grpo_auto_gradient_checkpointing_percentage(
                grpo_sequence_tokens,
                active_micro_batch_size,
                profile=gradient_checkpointing_profile,
            )
        )
    else:
        active_grpo_checkpointing_percentage = (
            requested_grpo_checkpointing_percentage
        )
    active_grpo_gradient_checkpointing = bool(gradient_checkpointing)
    if (
        active_grpo_checkpointing_percentage is not None
        and active_grpo_checkpointing_percentage <= 0
    ):
        active_grpo_gradient_checkpointing = False
    active_grpo_gradient_checkpointing_value = (
        resolve_gradient_checkpointing_value(
            active_grpo_gradient_checkpointing,
            gradient_checkpointing_mode,
        )
    )

    reward = ArcGridReward(
        min_cell_accuracy=min_cell_accuracy,
        exact_bonus=exact_bonus,
    )
    runtime_root = os.getenv(
        "ARC_RL_RUNTIME_DIR",
        "/kaggle/working/arc_rl_runtime",
    )
    output_dir = os.path.join(runtime_root, f"rank{rank}", task_key)
    print(
        f"[Rank {rank}] GRPO data for puzzle {task_key}: "
        f"rows={len(rl_rows)}, max_prompt_tokens={actual_max_prompt_tokens}, "
        f"max_completion_tokens={completion_limit}, "
        f"physical_micro_batch={active_micro_batch_size} "
        f"(requested={requested_micro_batch_size}, "
        f"auto={auto_micro_batch}, "
        f"token_limit={micro_batch_token_limit}), "
        f"checkpoint_policy_tokens={grpo_checkpoint_policy_tokens}, "
        f"checkpoint_percentage={active_grpo_checkpointing_percentage} "
        f"(requested={requested_grpo_checkpointing_percentage}), "
        f"checkpoint_mode={gradient_checkpointing_mode}, "
        f"checkpoint_profile={gradient_checkpointing_profile}, "
        f"rollout_backend={rollout_backend}, optimizer={optimizer}, "
        f"flash_adamw_quantize={flash_adamw_quantize}, "
        "flash_adamw_master_weight_bits="
        f"{flash_adamw_master_weight_bits}, "
        f"normuon_adjust_lr={normuon_adjust_lr}, "
        f"normuon_megabatch_size={normuon_megabatch_size}, "
        f"normuon_use_triton={normuon_use_triton}, "
        f"muon_adjust_lr={muon_adjust_lr}, "
        f"muon_ns_chunk_size={muon_ns_chunk_size}, "
        f"muon_scalar_optimizer={muon_scalar_optimizer}",
        flush=True,
    )
    print(
        f"[Rank {rank}] RL/SFT color split for puzzle {task_key}: "
        f"observed_colors={list(color_signature_colors)}, "
        f"sft_effective_mappings={len(excluded_color_signatures)}, "
        f"rl_effective_mappings={len(rl_color_signatures)}, overlap=0",
        flush=True,
    )

    rl_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=active_micro_batch_size,
        gradient_accumulation_steps=(
            num_generations // active_micro_batch_size
        ),
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        **warmup_dataclass_arguments(GRPOConfig, 0.0),
        # Custom optimizers are constructed by
        # TaskOptimizerTrainerMixin. Keep a Transformers-recognized
        # placeholder in TrainingArguments.
        optim=(
            "adamw_torch"
            if optimizer in {
                "flash_adamw",
                "normuon",
                "muon32bit",
                "muon8bit",
            }
            else optimizer
        ),
        weight_decay=0.0,
        max_grad_norm=max_grad_norm,
        bf16=True,
        fp16=False,
        gradient_checkpointing=active_grpo_gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": True}
            if active_grpo_gradient_checkpointing
            and gradient_checkpointing_mode == "unsloth"
            else None
        ),
        max_prompt_length=max_prompt_tokens,
        max_completion_length=completion_limit,
        num_generations=num_generations,
        generation_batch_size=num_generations,
        temperature=temperature,
        beta=0.001,
        loss_type="bnpo",
        mask_truncated_completions=True,
        use_vllm=False,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        seed=stable_task_seed(
            task_key,
            seed=42,
            namespace="arc-rl-trainer",
        ),
        dataloader_num_workers=0,
    )
    def restore_grpo_checkpointing(model_to_restore):
        if not active_grpo_gradient_checkpointing:
            return model_to_restore
        model_to_restore = FastLanguageModel.for_training(
            model_to_restore,
            use_gradient_checkpointing=(
                active_grpo_gradient_checkpointing_value
            ),
        )
        model_to_restore._unsloth_gradient_checkpointing = (
            active_grpo_gradient_checkpointing_value
        )
        if active_grpo_checkpointing_percentage is None:
            configure_gradient_checkpointing_interval(
                model_to_restore,
                gradient_checkpointing_interval,
            )
        else:
            configure_gradient_checkpointing_percentage(
                model_to_restore,
                active_grpo_checkpointing_percentage,
            )
        return model_to_restore

    class ArcGRPOTrainer(TaskOptimizerTrainerMixin, GRPOTrainer):
        """Preserve Unsloth's per-layer GC policy across TRL wrapping."""

        def _wrap_model(self, model_to_wrap, training=True, dataloader=None):
            model_to_wrap = restore_grpo_checkpointing(model_to_wrap)
            wrapped = super()._wrap_model(
                model_to_wrap,
                training=training,
                dataloader=dataloader,
            )
            return restore_grpo_checkpointing(wrapped)

        def training_step(self, model_to_train, *step_args, **step_kwargs):
            if (
                gradient_checkpointing_mode == "unsloth"
                and active_grpo_gradient_checkpointing
                and not getattr(
                    self,
                    "_arc_unsloth_checkpoint_boundary_validated",
                    False,
                )
            ):
                validate_unsloth_gradient_checkpointing_boundary(
                    model_to_train
                )
                self._arc_unsloth_checkpoint_boundary_validated = True
            return super().training_step(
                model_to_train,
                *step_args,
                **step_kwargs,
            )

    trainer = ArcGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=Dataset.from_list(rl_rows),
        args=rl_args,
    )
    configure_task_optimizer(
        trainer,
        optimizer_name=optimizer,
        flash_adamw_quantize=flash_adamw_quantize,
        flash_adamw_master_weight_bits=(
            flash_adamw_master_weight_bits
        ),
        normuon_adjust_lr=normuon_adjust_lr,
        normuon_momentum=normuon_momentum,
        normuon_beta2=normuon_beta2,
        normuon_megabatch_size=normuon_megabatch_size,
        normuon_use_triton=normuon_use_triton,
        muon_adjust_lr=muon_adjust_lr,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        muon_ns_chunk_size=muon_ns_chunk_size,
        muon_min_8bit_size=muon_min_8bit_size,
        muon_scalar_optimizer=muon_scalar_optimizer,
        log_prefix=f"[Rank {rank}] GRPO",
    )
    original_trainer_model = trainer.model
    trainer.model = restore_grpo_checkpointing(trainer.model)
    if getattr(trainer, "model_wrapped", None) is original_trainer_model:
        trainer.model_wrapped = trainer.model
    grpo_checkpointed_layers, grpo_total_layers = (
        gradient_checkpointing_layer_counts(trainer.model)
    )
    grpo_base_model = trainer.model
    get_base_model = getattr(grpo_base_model, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
        if candidate is not None:
            grpo_base_model = candidate
    hidden_state_support_marker = (
        "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__"
    )
    grpo_native_hidden_states = any(
        bool(getattr(forward_candidate, hidden_state_support_marker, False))
        for forward_candidate in (
            getattr(grpo_base_model, "forward", None),
            getattr(
                getattr(grpo_base_model, "forward", None),
                "__func__",
                None,
            ),
        )
    )
    grpo_hidden_state_fallback = any(
        bool(
            getattr(
                candidate,
                "_unsloth_grpo_hidden_states_forward_wrapped",
                False,
            )
        )
        for candidate in (trainer.model, grpo_base_model)
    )
    print(
        f"[Rank {rank}] GRPO checkpoint state before train: "
        f"requested={gradient_checkpointing}, active="
        f"{bool(getattr(trainer.model, 'is_gradient_checkpointing', False))}, "
        f"mode={gradient_checkpointing_mode}, "
        f"percentage={active_grpo_checkpointing_percentage}, "
        f"checkpointed_layers="
        f"{grpo_checkpointed_layers}/{grpo_total_layers}, "
        f"native_hidden_states={grpo_native_hidden_states}, "
        f"generic_hidden_state_fallback={grpo_hidden_state_fallback}",
        flush=True,
    )
    rollout_client = None
    if rollout_backend == "vllm":
        required = {
            "vllm_job_queue": vllm_job_queue,
            "vllm_response_queue": vllm_response_queue,
            "vllm_adapter_root": vllm_adapter_root,
            "vllm_run_id": vllm_run_id,
            "vllm_task_adapter_id": vllm_task_adapter_id,
            "vllm_initial_embedding_state": vllm_initial_embedding_state,
            "vllm_peft_config": vllm_peft_config,
            "vllm_service_index": vllm_service_index,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Incomplete external vLLM rollout configuration: "
                f"{missing}"
            )
        from arc_vllm_rl import (
            ArcVllmRolloutClient,
            attach_external_vllm_rollout,
        )

        rollout_client = ArcVllmRolloutClient(
            tokenizer=tokenizer,
            job_queue=vllm_job_queue,
            response_queue=vllm_response_queue,
            adapter_root=vllm_adapter_root,
            run_id=vllm_run_id,
            task_key=task_key,
            trainer_rank=rank,
            service_index=vllm_service_index,
            task_adapter_id=vllm_task_adapter_id,
            initial_embedding_state=vllm_initial_embedding_state,
            peft_config=vllm_peft_config,
            max_prompt_tokens=max_prompt_tokens,
            response_timeout_seconds=vllm_response_timeout_seconds,
        )
        attach_external_vllm_rollout(trainer, rollout_client)
    elif rollout_backend != "native":
        raise ValueError(f"Unknown GRPO rollout backend: {rollout_backend}")
    stats = trainer.train()
    model = trainer.accelerator.unwrap_model(
        model,
        keep_fp32_wrapper=False,
    )
    reward_summary = reward.summary()
    rollout_history = [] if rollout_client is None else rollout_client.history
    del trainer
    return model, stats, reward_summary, rollout_history


@worker_gc_scope()
def worker(
    rank,
    queue,
    end_time,
    distributed=False,
    keep_training_state=False,
    optimizer="adamw_torch",
    flash_adamw_quantize=False,
    flash_adamw_master_weight_bits=32,
    normuon_adjust_lr="rms_norm",
    normuon_momentum=0.95,
    normuon_beta2=0.95,
    normuon_megabatch_size=8,
    normuon_use_triton=False,
    muon_adjust_lr="rms_norm",
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_chunk_size=1,
    muon_min_8bit_size=4096,
    muon_scalar_optimizer="adamw_torch",
    te_optimizer_state_dtype="fp32",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_mode="standard",
    gradient_checkpointing_interval=1,
    gradient_checkpointing_percentage=None,
    deepspeed_long_context=False,
    deepspeed_token_threshold=8192,
    deepspeed_gradient_checkpointing_percentage=0.0,
    deepspeed_offload_optimizer=False,
    deepspeed_param_persistence_threshold=2_000_000,
    deepspeed_model_persistence_threshold=300_000_000,
    deepspeed_max_live_parameters=100_000_000,
    deepspeed_prefetch_bucket_size=25_000_000,
    num_train_epochs=4.0,
    learning_rate=5e-5,
    sft_loss_scope="all",
    use_liger_kernel=False,
    lora_method="rslora",
    lora_init="default",
    pissa_base_mode="original",
    lora_r=256,
    lora_alpha=32.0,
    train_embed_tokens=True,
    max_seq_length=8192,
    recompute_mlp_activations=False,
    recompute_attention_activations=False,
    hybrid_recompute_mlp_activations=False,
    hybrid_recompute_attention_activations=False,
    hybrid_attention_recompute_percentage=100.0,
    fused_qknorm_rope=False,
    attention_checkpoint_input_offload=False,
    fp8_frozen_linear=False,
    fp8_min_train_tokens=0,
    fp8_lora=False,
    fp8_dynamic_backward=False,
    fp8_dynamic_backward_split_accumulator=True,
    auto_activation_recompute=False,
    mlp_recompute_token_threshold=3000,
    attention_recompute_token_threshold=4096,
    max_grad_norm=1.0,
    torch_compile_training=False,
    bidirectional_prompt_attention=False,
    train_only=False,
    task_rl=False,
    rl_sft_steps=32,
    rl_max_steps=8,
    rl_num_generations=2,
    rl_micro_batch_size=2,
    rl_auto_micro_batch=True,
    rl_rollout_backend="native",
    rl_augmentations=2,
    rl_max_prompt_tokens=2048,
    rl_reward_min_cell_accuracy=0.7,
    rl_exact_bonus=0.0,
    rl_temperature=0.8,
    rl_learning_rate=5e-6,
    predict_train=False,
    report_output_nll=False,
    min_train_pairs=0,
    extra_train_pair_mode="augment",
    profile_training=False,
    profile_wait_steps=1,
    profile_warmup_steps=1,
    profile_active_steps=4,
    inference_timeout_seconds=1200,
    inference_batch_size=4,
    score_batch_size=4,
    rescore_views=8,
    rescore_schedule="candidate",
    vllm_prefix_score=False,
    vllm_prefix_score_tile_views=4,
    dfs_mode="recursive",
    frontier_origin_batch_size=4,
    frontier_max_live_branches=8,
    frontier_max_prompt_tokens=3000,
    max_beam_score=-np.log(0.2),
    zero_candidate_shape_fallback=False,
    shape_fallback_beam_width=4,
    vllm_job_queues=None,
    vllm_rollout_response_queues=None,
    vllm_pending_adapter_slots=None,
    vllm_service_outstanding=None,
    vllm_service_load_lock=None,
    vllm_adapter_root=None,
    vllm_adapter_roots=None,
    vllm_adapter_staging_slots=None,
    vllm_run_id=None,
    vllm_adapter_ids=None,
    vllm_service_active=None,
    vllm_service_ready=None,
    vllm_published_adapter_count=None,
    vllm_ramp_ready_event=None,
    vllm_ramp_after_adapters=0,
    vllm_final_inference=False,
    vllm_preserve_adapter_root=None,
    abort_event=None,
    vllm_ramp_service_limit=None,
    vllm_shared_pending_queue=None,
    vllm_shared_pending_count=None,
):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    reuse_base_model = os.getenv(
        "ARC_REUSE_BASE_MODEL",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    # With a reused base, reset the task adapter in place to the bit-exact
    # state of the first fresh attach instead of PEFT unload + re-attach.
    # ``verify`` keeps the re-attach and checks it against that snapshot.
    inplace_adapter_reset = os.getenv(
        "ARC_INPLACE_ADAPTER_RESET",
        "0",
    ).strip().lower()
    if inplace_adapter_reset in {"true", "yes", "on"}:
        inplace_adapter_reset = "1"
    if inplace_adapter_reset not in {"0", "1", "verify"}:
        raise ValueError("ARC_INPLACE_ADAPTER_RESET must be 0, 1 or verify")

    max_seq_length = int(max_seq_length)
    from arc_train_pair_expansion import validate_pair_expansion, write_expansion_manifest

    validate_pair_expansion(min_train_pairs, extra_train_pair_mode)
    pair_expansion_options = dict(
        min_train_pairs=min_train_pairs,
        extra_train_pair_mode=extra_train_pair_mode,
    )
    if min_train_pairs and task_rl:
        raise ValueError("Training-pair expansion currently supports task SFT only")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    learning_rate = float(learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if sft_loss_scope not in ("all", "final"):
        raise ValueError("sft_loss_scope must be all or final")
    final_assistant_only = bool(task_rl) or sft_loss_scope == "final"
    lora_method_config = resolve_lora_method(lora_method, lora_init)
    validate_fp8_lora_options(
        fp8_lora, fp8_frozen_linear=fp8_frozen_linear,
        fp8_dynamic_backward=fp8_dynamic_backward,
        lora_method=lora_method, lora_init=lora_init, optimizer=optimizer,
        task_rl=task_rl, deepspeed=deepspeed_long_context,
        distributed=distributed, keep_training_state=keep_training_state,
        torch_compile_training=torch_compile_training,
        inference_backend="vllm" if vllm_final_inference else "native",
        train_only=train_only,
    )
    pissa_enabled = is_pissa_init(lora_method_config["init_lora_weights"])
    from arc_pissa_shared import validate_runtime_base

    shared_pissa_manifest = validate_runtime_base(
        MODEL_PATH, mode=pissa_base_mode, initialization=lora_init,
        method=lora_method, rank=lora_r, alpha=lora_alpha,
    )
    validate_pissa_options(
        lora_method, lora_init,
        fp8_frozen_linear=fp8_frozen_linear,
        task_rl=task_rl, deepspeed=deepspeed_long_context,
        distributed=distributed, keep_training_state=keep_training_state,
    )
    effective_lora_scaling = lora_effective_scaling(
        lora_method_config["method"],
        lora_r,
        lora_alpha,
    )
    gradient_checkpointing_mode = normalize_gradient_checkpointing_mode(
        gradient_checkpointing_mode
    )
    (
        qwen35_gdn_replay_cache_requested,
        qwen35_gdn_replay_cache_projections,
        qwen35_gdn_replay_cache_max_bytes,
    ) = qwen35_gdn_replay_cache_options()
    qwen35_gdn_replay_cache_partial = os.getenv(
        "ARC_QWEN35_GDN_REPLAY_CACHE_PARTIAL", "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if (
        distributed
        and gradient_checkpointing
        and gradient_checkpointing_mode == "unsloth"
    ):
        raise ValueError(
            "Unsloth pinned-CPU gradient checkpointing currently supports "
            "independent one-GPU task workers only; use "
            "--single-task-gpus 1"
        )
    if gradient_checkpointing_mode == "unsloth" and max_seq_length < 512:
        gradient_checkpointing_mode = "standard"
        if not distributed or rank == 0:
            print(
                f"[Rank {rank}] max_seq_length={max_seq_length} is below "
                "Unsloth's smart-offload threshold; using standard "
                "gradient checkpointing",
                flush=True,
            )

    if task_rl and distributed:
        raise ValueError(
            "Task-time GRPO currently supports independent one-GPU workers "
            "only; use --single-task-gpus 1"
        )
    if task_rl and bidirectional_prompt_attention:
        raise ValueError(
            "ARC bidirectional prompt attention currently supports the SFT "
            "task-training path, not task-time GRPO"
        )
    if task_rl and keep_training_state:
        raise ValueError(
            "Task-time GRPO requires a fresh per-task adapter and cannot be "
            "combined with --keep-training-state"
        )
    if task_rl and fp8_frozen_linear:
        raise ValueError(
            "FP8 frozen-linear task training is not yet supported by the "
            "task-time GRPO reload path"
        )
    if keep_training_state and fp8_frozen_linear:
        raise ValueError(
            "FP8 frozen-linear training currently requires fresh per-task "
            "adapters and optimizers; disable --keep-training-state"
        )
    if task_rl and gradient_accumulation_steps != 1:
        raise ValueError(
            "Task-time GRPO currently requires "
            "--gradient-accumulation-steps 1 so --rl-sft-steps is exact"
        )
    if rl_sft_steps <= 0 or rl_max_steps <= 0:
        raise ValueError("RL SFT and GRPO step counts must be positive")
    if rl_num_generations < 2:
        raise ValueError("rl_num_generations must be at least 2")
    if rl_micro_batch_size <= 0:
        raise ValueError("rl_micro_batch_size must be positive")
    if rl_num_generations % rl_micro_batch_size:
        raise ValueError(
            "rl_num_generations must be divisible by rl_micro_batch_size"
        )
    if rl_rollout_backend not in {"native", "vllm"}:
        raise ValueError("rl_rollout_backend must be native or vllm")
    if rl_rollout_backend == "vllm" and not task_rl:
        raise ValueError("vLLM RL rollout requires task_rl")
    if rl_rollout_backend == "native" and rl_num_generations > 4:
        raise ValueError(
            "Native RL rollout supports at most 4 generations; use "
            "--rl-rollout-backend vllm for 8 or 16 generations"
        )
    if rl_augmentations <= 0 or rl_max_prompt_tokens <= 0:
        raise ValueError("RL augmentations and prompt limit must be positive")
    if not 0.0 <= rl_reward_min_cell_accuracy <= 1.0:
        raise ValueError("RL reward minimum cell accuracy must be in [0, 1]")
    if rl_exact_bonus < 0.0:
        raise ValueError("RL exact bonus must be non-negative")
    if rl_temperature <= 0.0 or rl_learning_rate <= 0.0:
        raise ValueError("RL temperature and learning rate must be positive")

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "lm_head",
    ]
    if train_embed_tokens:
        target_modules.append("embed_tokens")

    if (
        use_liger_kernel
        and (
            auto_activation_recompute
            or recompute_mlp_activations
            or hybrid_recompute_mlp_activations
        )
    ):
        raise ValueError(
            "MLP activation recomputation is incompatible with Liger because "
            "Liger replaces the fused MLP path."
        )
    selective_activation_recompute = (
        auto_activation_recompute
        or recompute_mlp_activations
        or recompute_attention_activations
    )
    hybrid_activation_recompute = (
        hybrid_recompute_mlp_activations
        or hybrid_recompute_attention_activations
    )
    hybrid_attention_recompute_percentage = float(
        hybrid_attention_recompute_percentage
    )
    if not 0.0 <= hybrid_attention_recompute_percentage <= 100.0:
        raise ValueError(
            "Hybrid attention recomputation percentage must be in [0, 100]"
        )
    if selective_activation_recompute and hybrid_activation_recompute:
        raise ValueError(
            "Hybrid recomputation cannot be combined with selective or "
            "automatic activation recomputation"
        )
    if hybrid_activation_recompute and not gradient_checkpointing:
        raise ValueError(
            "Hybrid recomputation requires gradient checkpointing"
        )
    if hybrid_activation_recompute and distributed:
        raise ValueError(
            "Hybrid recomputation currently supports one GPU per task"
        )
    attention_checkpoint_input_offload = bool(
        attention_checkpoint_input_offload
    )
    if (
        attention_checkpoint_input_offload
        and not (
            recompute_attention_activations
            or hybrid_recompute_attention_activations
        )
    ):
        raise ValueError(
            "Attention checkpoint-input offload requires attention-only "
            "recomputation"
        )
    if attention_checkpoint_input_offload and (
        gradient_checkpointing_mode == "unsloth"
        and (
            not hybrid_recompute_attention_activations
            or gradient_checkpointing_percentage in {None, "auto"}
            or float(gradient_checkpointing_percentage) > 0.0
        )
    ):
        raise ValueError(
            "Unsloth attention checkpoint-input offload requires hybrid "
            "attention recomputation with exactly zero full-layer gradient "
            "checkpoints"
        )
    if hybrid_activation_recompute and task_rl:
        raise ValueError(
            "Hybrid recomputation is not yet supported by task-time GRPO"
        )
    deepspeed_gradient_checkpointing_percentage = float(
        0.0
        if deepspeed_gradient_checkpointing_percentage is None
        else deepspeed_gradient_checkpointing_percentage
    )
    if not 0.0 <= deepspeed_gradient_checkpointing_percentage <= 100.0:
        raise ValueError(
            "DeepSpeed gradient-checkpointing percentage must be in [0, 100]"
        )
    deepspeed_can_activate = (
        bool(deepspeed_long_context)
        and max_seq_length > int(deepspeed_token_threshold)
    )
    if reuse_base_model:
        incompatible_reuse_options = []
        if keep_training_state:
            incompatible_reuse_options.append("--keep-training-state")
        if distributed:
            incompatible_reuse_options.append("distributed training")
        if task_rl:
            incompatible_reuse_options.append("task-time GRPO")
        if deepspeed_can_activate:
            incompatible_reuse_options.append("DeepSpeed long-context routing")
        if torch_compile_training:
            incompatible_reuse_options.append("compiled training")
        if incompatible_reuse_options:
            raise ValueError(
                "ARC_REUSE_BASE_MODEL currently supports independent "
                "LoRA SFT workers only; incompatible with "
                + ", ".join(incompatible_reuse_options)
            )
        if not distributed or rank == 0:
            print(
                f"[Rank {rank}] base-model reuse enabled: retain frozen "
                "BF16 weights, unload the completed PEFT wrapper, and "
                "attach a newly initialized LoRA for every later task"
                + (
                    "; FP8 training preserves exact BF16 base weights in CPU "
                    "RAM and restores them before every reattachment"
                    if fp8_frozen_linear else ""
                ),
                flush=True,
            )
        if fp8_frozen_linear and inplace_adapter_reset != "0":
            print(
                f"[Rank {rank}] FP8 base reuse uses exact BF16 CPU restore + "
                "fresh PEFT reattachment; ARC_INPLACE_ADAPTER_RESET is not used",
                flush=True,
            )
    if deepspeed_can_activate and hybrid_activation_recompute:
        raise ValueError(
            "Hybrid recomputation cannot be combined with DeepSpeed routing"
        )
    validate_long_context_deepspeed_options(
        enabled=deepspeed_can_activate,
        optimizer=optimizer,
        distributed=distributed,
        keep_training_state=keep_training_state,
        task_rl=task_rl,
        train_only=train_only,
        routed_inference=bool(vllm_final_inference),
        torch_compile_training=torch_compile_training,
        selective_activation_recompute=selective_activation_recompute,
        fp8_frozen_linear=fp8_frozen_linear,
    )
    if (
        gradient_checkpointing_percentage is not None
        and selective_activation_recompute
    ):
        raise ValueError(
            "Percentage gradient checkpointing cannot be combined with "
            "selective or automatic activation recomputation."
        )
    if (
        gradient_checkpointing
        and gradient_checkpointing_mode == "unsloth"
        and selective_activation_recompute
    ):
        raise ValueError(
            "Unsloth pinned-CPU gradient checkpointing cannot currently be "
            "combined with selective or automatic activation recomputation"
        )
    if (
        selective_activation_recompute
        or fp8_frozen_linear
        or gradient_checkpointing_percentage is not None
        or hybrid_activation_recompute
        or bidirectional_prompt_attention
        or fused_qknorm_rope
        or attention_checkpoint_input_offload
    ):
        ensure_training_optimization_support(
            selective_activation_recompute=(
                selective_activation_recompute
                or hybrid_activation_recompute
            ),
            hybrid_activation_recompute=hybrid_activation_recompute,
            fp8_frozen_linear=fp8_frozen_linear,
            partial_gradient_checkpointing=(
                gradient_checkpointing_percentage is not None
                or (
                    deepspeed_can_activate
                    and deepspeed_gradient_checkpointing_percentage > 0
                )
            ),
            bidirectional_prompt_attention=(
                bidirectional_prompt_attention
            ),
            fused_qknorm_rope=fused_qknorm_rope,
            attention_checkpoint_input_offload=(
                attention_checkpoint_input_offload
            ),
            smart_attention_checkpoint_input_offload=(
                attention_checkpoint_input_offload
                and gradient_checkpointing_mode == "unsloth"
            ),
        )
    adaptive_percentage_checkpointing = (
        gradient_checkpointing_percentage == "auto"
    )
    explicit_zero_checkpointing = (
        gradient_checkpointing_percentage is not None
        and gradient_checkpointing_percentage != "auto"
        and float(gradient_checkpointing_percentage) <= 0
    )
    # Adaptive mode chooses only after task tokenization. Construct the base
    # model and LoRA without checkpointing so its zero-percent tier is the same
    # true fast path as --no-gradient-checkpointing; longer tasks enable their
    # selected layer set immediately before Trainer construction.
    base_gradient_checkpointing = (
        gradient_checkpointing
        and not selective_activation_recompute
        and not adaptive_percentage_checkpointing
        and not explicit_zero_checkpointing
    )
    # Adaptive percentage mode may start with a zero-checkpoint task and turn
    # checkpointing on for a later, longer task. Install Unsloth's global smart
    # checkpoint function while constructing PEFT whenever that later path is
    # possible; the per-task for_training call still disables all layer flags
    # for a zero-percent task.
    model_setup_gradient_checkpointing = resolve_gradient_checkpointing_value(
        (
            gradient_checkpointing
            and not selective_activation_recompute
            and (
                not explicit_zero_checkpointing
                or (
                    attention_checkpoint_input_offload
                    and hybrid_recompute_attention_activations
                    and gradient_checkpointing_mode == "unsloth"
                )
            )
        ),
        gradient_checkpointing_mode,
    )

    peft_params = dict(
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=model_setup_gradient_checkpointing,
        random_state=42,
        use_rslora=lora_method_config["use_rslora"],
        use_nora=lora_method_config["use_nora"],
        init_lora_weights=lora_method_config["init_lora_weights"],
        loftq_config=None,
        ensure_weight_tying=False,
    )
    if not distributed or rank == 0:
        print(
            f"[Rank {rank}] task-adapter method: "
            f"{lora_method_config['method']} (r={int(lora_r)}, "
            f"alpha={float(lora_alpha):g}, "
            f"effective_scale={effective_lora_scaling:.8g}, "
            f"persistent_normalization="
            f"{lora_method_config['use_nora'] is True}, "
            f"init={lora_method_config['init_lora_weights']!r})",
            flush=True,
        )
    if model_setup_gradient_checkpointing == "unsloth":
        unsloth_temporary_root = os.path.join(
            os.path.expanduser(
                os.getenv(
                    "ARC_UNSLOTH_TEMP_DIR",
                    "/dev/shm/arc_unsloth_temporary_saved_buffers",
                )
            ),
            f"rank{rank}",
        )
        os.makedirs(unsloth_temporary_root, exist_ok=True)
        if not os.access(unsloth_temporary_root, os.W_OK):
            raise RuntimeError(
                "Unsloth temporary offload root is not writable: "
                f"{unsloth_temporary_root}"
            )
        peft_params["temporary_location"] = unsloth_temporary_root
        if not distributed or rank == 0:
            print(
                f"[Rank {rank}] Unsloth checkpoint offload root: "
                f"{unsloth_temporary_root}",
                flush=True,
            )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        **warmup_dataclass_arguments(UnslothTrainingArguments, 0.1),
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        # Custom optimizers are constructed lazily by the shared mixin.
        optim=(
            "adamw_torch"
            if optimizer in {
                "flash_adamw",
                "normuon",
                "muon32bit",
                "muon8bit",
            }
            else optimizer
        ),
        te_fused_adamw_state_dtype=te_optimizer_state_dtype,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        # Independent task workers normally keep Trainer output quiet because
        # multiple bars can interleave.  Opt in when measuring long-context
        # task-time training with ARC_TRAIN_TQDM=1.  This is deliberately
        # separate from UNSLOTH_DISABLE_STATISTICS, which controls Unsloth's
        # environment telemetry rather than Hugging Face Trainer progress.
        disable_tqdm=os.getenv("ARC_TRAIN_TQDM", "0").strip().lower()
        not in {"1", "true", "yes", "on"},
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        # The tied lm_head LoRA parameters can be skipped by a forward pass.
        ddp_find_unused_parameters=distributed,
        dataloader_num_workers=0,
        gradient_checkpointing=base_gradient_checkpointing,
        use_liger_kernel=use_liger_kernel,
        # Transformers treats a non-None backend or mode as an implicit request
        # to enable torch.compile.  Supplying these unconditionally therefore
        # made the default-off CLI flag report False in worker options while
        # Trainer silently wrapped the whole PEFT model in OptimizedModule.
        **resolve_torch_compile_training_options(torch_compile_training),
        # tf32=True
    )

    # Unsloth only makes its default device map rank-aware for quantized
    # models. This model is BF16, so bind every DDP replica explicitly or all
    # ranks will load onto cuda:0 and exhaust that GPU.
    model_device_map = (
        {"": f"cuda:{torch.cuda.current_device()}"}
        if distributed
        else "sequential"
    )

    pissa_initialization = None
    initial_adapter_snapshot = None

    def attach_fresh_task_adapter(
        task_model,
        *,
        stage_embeddings_to_disk=True,
        reuse_pissa_residual=False,
    ):
        nonlocal pissa_initialization, initial_adapter_snapshot
        attach_started = time.perf_counter()
        task_peft_params = configure_task_peft_params_for_model(
            peft_params,
            task_model.config,
        )
        if pissa_enabled:
            from arc_pissa import (
                PiSSAInitialization, prepare_pissa_targets, validate_pissa_base,
            )

            task_peft_params = prepare_pissa_targets(task_peft_params)
            validate_pissa_base(task_model, task_peft_params)
            if reuse_pissa_residual:
                if pissa_initialization is None:
                    raise RuntimeError("PiSSA residual reuse is missing its initial factors")
                # Retain W_res and reset A0/B0 below. A second PiSSA SVD
                # would subtract principal components from W_res again.
                task_peft_params["init_lora_weights"] = True
            elif shared_pissa_manifest is not None:
                task_peft_params["init_lora_weights"] = True
        if not stage_embeddings_to_disk:
            # The first attach uses Unsloth's disk staging to minimize model
            # construction peak.  Once the old wrapper has been unloaded,
            # only the frozen base remains on an otherwise empty 22-GiB L4;
            # bypassing that repeated staging avoids several seconds of disk
            # I/O.  The task-specific checkpoint policy is applied later by
            # ``for_training`` and the per-layer configuration helpers.
            task_peft_params["use_gradient_checkpointing"] = False
            from unsloth.models import llama as unsloth_llama

            if getattr(
                unsloth_llama,
                "UNSLOTH_SUPPORTS_FAST_PEFT_REATTACH",
                False,
            ):
                task_peft_params["skip_redundant_cleanup"] = True
        task_model = FastLanguageModel.get_peft_model(
            task_model,
            **task_peft_params,
        )
        # PEFT delegates unknown attributes to its wrapped model.  After
        # ``unload()`` + re-wrap, ``hasattr(wrapper,
        # "_flag_for_generation")`` can therefore be true even though the
        # flag is owned by a child.  Unsloth's ``for_training`` checks with
        # hasattr and then deletes from the current wrapper, which raises in
        # that delegated-attribute case.  Give every delegated level a local
        # marker so Unsloth can consume it normally on the next mode switch.
        flag_owner = task_model
        seen_flag_owners = set()
        while id(flag_owner) not in seen_flag_owners:
            seen_flag_owners.add(id(flag_owner))
            if (
                hasattr(flag_owner, "_flag_for_generation")
                and "_flag_for_generation" not in flag_owner.__dict__
            ):
                flag_owner.__dict__["_flag_for_generation"] = True
            next_flag_owner = getattr(flag_owner, "model", None)
            if next_flag_owner is None or next_flag_owner is flag_owner:
                break
            flag_owner = next_flag_owner
        configure_activation_recomputation(
            task_model,
            recompute_mlp_activations,
            recompute_attention_activations,
        )
        for _name, parameter in task_model.named_parameters():
            if (
                parameter.dtype == torch.float32
                and DELTA_MHAR_PARAMETER_MARKER not in _name
            ):
                parameter.data = parameter.data.to(torch.bfloat16)
        if pissa_enabled:
            if reuse_pissa_residual:
                count = pissa_initialization.restore(task_model)
                action = "restored initial factors on retained frozen residual"
            elif shared_pissa_manifest is not None:
                pissa_initialization = PiSSAInitialization.from_shared_checkpoint(task_model, MODEL_PATH)
                count = len(pissa_initialization.scalings)
                action = "loaded shared residual and immutable initial factors (no SVD)"
            else:
                pissa_initialization = PiSSAInitialization.capture(task_model)
                count = len(pissa_initialization.scalings)
                action = "initialized frozen residual and saved initial factors"
            print(
                f"[Rank {rank}] PiSSA {action}: projections={count}, "
                f"init={lora_init}, training_rank={lora_r}, "
                f"export_rank={lora_export_rank(lora_r, lora_init, pissa_base_mode)}, "
                f"base_mode={pissa_base_mode}, "
                f"immutable_cpu_factors={pissa_initialization.tensor_bytes / 1024**2:.1f}MiB, "
                f"attach_seconds={time.perf_counter() - attach_started:.3f}",
                flush=True,
            )
        if (
            inplace_adapter_reset != "0"
            and reuse_base_model
            and not pissa_enabled
            and not fp8_frozen_linear
        ):
            if initial_adapter_snapshot is None:
                initial_adapter_snapshot = InitialAdapterSnapshot.capture(
                    task_model
                )
                print(
                    f"[Rank {rank}] captured initial task adapter for in-place "
                    f"reset: tensors={len(initial_adapter_snapshot.entries)}, "
                    f"pinned_cpu={initial_adapter_snapshot.tensor_bytes / 1024**2:.1f}MiB",
                    flush=True,
                )
            elif inplace_adapter_reset == "verify":
                matching, total, rng_matches = initial_adapter_snapshot.matches(
                    task_model
                )
                print(
                    f"[Rank {rank}] in-place adapter reset verify: "
                    f"exact_tensors={matching}/{total}, rng_state_exact={rng_matches}",
                    flush=True,
                )
        return task_model

    def load_fresh_task_model():
        task_model, task_tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=model_setup_gradient_checkpointing,
            max_seq_length=max_seq_length,
            device_map=model_device_map,
        )
        attention_residual_config = load_checkpoint_attention_residuals(
            task_model,
            MODEL_PATH,
        )
        from arc_rope_3d_lora import load_checkpoint_rope_3d

        rope_config = load_checkpoint_rope_3d(task_model, MODEL_PATH)
        if rope_config is not None:
            print(
                f"[Rank {rank}] loaded ARC 3D RoPE base={rope_config['base']:g}; "
                "task LoRA, scoring and KV decoding use grid coordinates",
                flush=True,
            )
        task_tokenizer = ensure_arc_tokenizer_backend(task_tokenizer)
        task_model = attach_fresh_task_adapter(task_model)
        if (
            attention_residual_config is not None
            and os.getenv("UNSLOTH_DELTA_MHAR_FROZEN_BACKWARD", "0") == "1"
        ):
            from unsloth.kernels import attention_residuals as residual_kernels

            if not getattr(
                residual_kernels, "UNSLOTH_SUPPORTS_DELTA_MHAR_FROZEN_BACKWARD", False,
            ):
                raise RuntimeError(
                    "Frozen Delta-MH backward requires the matching Unsloth patch; "
                    "update Unsloth or disable UNSLOTH_DELTA_MHAR_FROZEN_BACKWARD"
                )
            if rank == 0:
                print(
                    "[Rank 0] Delta-MH frozen-router backward enabled: skips only "
                    "unrequested router parameter gradients; input gradients are retained",
                    flush=True,
                )
        if attention_residual_config is not None and rank == 0:
            print(
                "[Rank 0] loaded frozen Delta-MH base: "
                f"heads={attention_residual_config.num_heads}, "
                f"block_size={attention_residual_config.block_size}, "
                f"fused={attention_residual_config.fused}",
                flush=True,
            )
        return task_model, task_tokenizer

    model, tokenizer = load_fresh_task_model()

    def quantize_fp8_base_weights(preserve_bf16):
        from unsloth.kernels.fast_lora import (
            quantize_model_lora_base_weights_fp8,
        )

        converted = quantize_model_lora_base_weights_fp8(
            model,
            dynamic_backward=fp8_dynamic_backward,
            dynamic_backward_split_accumulator=(
                fp8_dynamic_backward_split_accumulator
            ),
            preserve_bf16=preserve_bf16,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[Rank {rank}] converted {converted} frozen LoRA base "
            "projection weights to rowwise TE FP8 "
            f"(dynamic_backward={fp8_dynamic_backward}, "
            "dynamic_backward_split_accumulator="
            f"{fp8_dynamic_backward_split_accumulator}, "
            f"preserve_bf16={preserve_bf16})",
            flush=True,
        )
        if converted == 0:
            raise RuntimeError(
                "FP8 frozen-linear training was requested, but no PEFT base "
                "projection weights were converted"
            )
        return converted

    gradient_checkpointing_profile = (
        select_auto_gradient_checkpointing_profile(model.config)
    )
    qwen35_deepspeed_experiment = os.getenv(
        "ARC_DEEPSPEED_QWEN35_EXPERIMENT",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if (
        deepspeed_can_activate
        and gradient_checkpointing_profile != "qwen3_4b"
        and not (
            gradient_checkpointing_profile == "qwen3_5_9b"
            and qwen35_deepspeed_experiment
        )
    ):
        raise ValueError(
            "Long-context DeepSpeed parameter offload is currently calibrated "
            "only for the Qwen3-4B ARC model, not profile "
            f"{gradient_checkpointing_profile!r}"
        )
    if (
        deepspeed_can_activate
        and gradient_checkpointing_profile == "qwen3_5_9b"
        and qwen35_deepspeed_experiment
        and (not distributed or rank == 0)
    ):
        print(
            f"[Rank {rank}] enabling experimental Qwen3.5-9B DeepSpeed "
            "long-context routing; this path is benchmark-only and remains "
            "default-off",
            flush=True,
        )
    if gradient_checkpointing_percentage == "auto" and (
        not distributed or rank == 0
    ):
        print(
            f"[Rank {rank}] auto gradient-checkpointing profile: "
            f"{gradient_checkpointing_profile} "
            f"(model_type={getattr(model.config, 'model_type', None)}, "
            f"hidden_size={getattr(model.config, 'hidden_size', None)}, "
            "layers="
            f"{getattr(model.config, 'num_hidden_layers', None)})",
            flush=True,
        )

    initial_vllm_embedding_state = None
    vllm_peft_config = None
    shared_pending_pool = vllm_shared_pending_queue is not None
    if shared_pending_pool != (vllm_shared_pending_count is not None):
        raise ValueError("The shared adapter pool requires both queue and counter")
    if shared_pending_pool and vllm_job_queues is None:
        raise ValueError("The shared adapter pool requires the vLLM pipeline")
    if vllm_job_queues is not None:
        if distributed:
            raise ValueError("The vLLM handoff does not support DDP training")
        if len(vllm_job_queues) < 1:
            raise ValueError("At least one vLLM job queue is required")
        if vllm_rollout_response_queues is None:
            raise ValueError("vLLM rollout response queues are required")
        if not 0 <= rank < len(vllm_rollout_response_queues):
            raise ValueError(f"No vLLM rollout response queue for rank {rank}")
        if any(
            value is None
            for value in (
                vllm_pending_adapter_slots,
                vllm_service_outstanding,
                vllm_service_load_lock,
                vllm_adapter_root,
                vllm_run_id,
                vllm_adapter_ids,
            )
        ):
            raise ValueError("Incomplete vLLM adapter handoff configuration")
        if vllm_adapter_staging_slots is not None and (
            not vllm_adapter_roots
            or len(vllm_adapter_roots) != len(vllm_adapter_staging_slots)
        ):
            raise ValueError(
                "vLLM adapter staging roots and slot semaphores must align"
            )
        if int(vllm_ramp_after_adapters) > 0 and any(
            value is None
            for value in (
                vllm_service_active,
                vllm_published_adapter_count,
                vllm_ramp_ready_event,
            )
        ):
            raise ValueError("Incomplete adaptive inference-ramp configuration")
        from arc_vllm_adapter import (
            clone_embedding_state_to_cpu,
            peft_config_dict,
        )

        initial_peft_state = get_peft_model_state_dict(
            model,
            adapter_name="default",
        )
        initial_vllm_embedding_state = clone_embedding_state_to_cpu(
            initial_peft_state
        )
        if pissa_enabled:
            initial_vllm_embedding_state.update(pissa_initialization.weights)
        # Ordinary LoRA needs only the tiny vocabulary bases. PiSSA also
        # retains immutable CPU A0/B0 factors for original-base conversion.
        # Drop the full PEFT mapping so it cannot retain the
        # first model's LoRA tensors when this lane reloads a fresh model for
        # its next task.
        del initial_peft_state
        vllm_peft_config = peft_config_dict(model.peft_config["default"])
    persistent_optimizer = None
    completed_task_count = 0
    adaptive_service_index = None

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        final_assistant_only=final_assistant_only,
        bidirectional_prompt_attention=bidirectional_prompt_attention,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    if max_seq_length <= max_new_tokens:
        raise ValueError(
            f"max_seq_length={max_seq_length} leaves no room for an ARC "
            f"prompt after reserving {max_new_tokens} generation tokens"
        )

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    if frontier_origin_batch_size <= 0:
        raise ValueError("frontier_origin_batch_size must be positive")
    if frontier_max_live_branches <= 0:
        raise ValueError("frontier_max_live_branches must be positive")
    if frontier_max_prompt_tokens <= 0:
        raise ValueError("frontier_max_prompt_tokens must be positive")
    if shape_fallback_beam_width < 2:
        raise ValueError("shape_fallback_beam_width must be at least 2")
    max_score = float(max_beam_score)
    if rescore_views <= 0:
        raise ValueError("rescore_views must be positive")
    if rescore_schedule not in {"candidate", "view", "prefix"}:
        raise ValueError(
            "rescore_schedule must be 'candidate', 'view', or 'prefix'"
        )
    if vllm_prefix_score_tile_views <= 0:
        raise ValueError("vllm_prefix_score_tile_views must be positive")

    test_path = resolve_challenge_path(rerun_mode)

    arc_test_set = ArcDataset.from_file(test_path)
    output_nll_ground_truths = load_available_ground_truths(
        bool(report_output_nll and not rerun_mode),
        resolve_evaluation_solution_path(),
    )

    dir_outputs = os.getenv(
        "ARC_INFERENCE_OUTPUT_DIR",
        "/kaggle/inference_outputs",
    )
    os.makedirs(dir_outputs, exist_ok=True)
    train_prediction_output_dir = None
    if predict_train:
        train_prediction_output_dir = resolve_train_prediction_output_dir(
            dir_outputs
        )
        os.makedirs(train_prediction_output_dir, exist_ok=True)

    if (
        reuse_base_model
        and inplace_adapter_reset == "1"
        and initial_adapter_snapshot is not None
        and os.getenv("ARC_GC_FREEZE", "1") == "1"
    ):
        # With the in-place reset the model, adapter wrappers, snapshot,
        # tokenizer and task set live for the whole lane. Moving them to the
        # permanent generation lets each of the ~9 full collections per task
        # (solver, Unsloth, Accelerate) skip them instead of costing ~0.6s.
        # Refcount freeing is unchanged; only cyclic GC ignores these.
        # The worker scope thaws this pool on every exit, before starter's
        # trainer->inference handoff collects dead model reference cycles.
        freeze_started = time.perf_counter()
        if freeze_worker_gc(rank):
            print(
                f"[Rank {rank}] gc.freeze: objects={gc.get_freeze_count()}, "
                f"seconds={time.perf_counter() - freeze_started:.3f}",
                flush=True,
            )

    while True:
        if abort_event is not None and abort_event.is_set():
            print(
                f"[Rank {rank}] fatal sibling failure received; "
                "stop claiming tasks",
                flush=True,
            )
            break
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break
        if abort_event is not None and abort_event.is_set():
            print(
                f"[Rank {rank}] fatal sibling failure received after "
                f"claiming {key}; aborting the run",
                flush=True,
            )
            break

        start_time = time.time()
        print(f"[Rank {rank}] claimed task {key}")

        torch.cuda.reset_peak_memory_stats()

        if keep_training_state:
            if completed_task_count:
                print(
                    f"[Rank {rank}] carrying LoRA weights and optimizer "
                    f"state into puzzle {key}"
                )
        elif completed_task_count:
            reset_started = time.perf_counter()
            allocated_before_reload = torch.cuda.memory_allocated()
            if (
                reuse_base_model
                and inplace_adapter_reset == "1"
                and initial_adapter_snapshot is not None
            ):
                model.zero_grad(set_to_none=True)
                compiler_reset_started = time.perf_counter()
                reset_torch_compiler_cache()
                gc.collect()
                torch.cuda.empty_cache()
                compiler_reset_seconds = (
                    time.perf_counter() - compiler_reset_started
                )
                allocated_after_model_release = torch.cuda.memory_allocated()
                allocated_after_cleanup = allocated_after_model_release
                adapter_restore_started = time.perf_counter()
                (
                    restored_adapter_tensors,
                    restored_adapter_bytes,
                ) = initial_adapter_snapshot.restore(model)
                # Mirror the attach-time setup that a re-attach would redo.
                configure_activation_recomputation(
                    model,
                    recompute_mlp_activations,
                    recompute_attention_activations,
                )
                # A re-attach runs ``prepare_model_for_training`` with
                # ``use_gradient_checkpointing=False``, which drops both
                # Unsloth checkpoint patches (and their grown GPU/CPU
                # buffers) before the solver reinstalls the smart one.
                from unsloth_zoo.gradient_checkpointing import (
                    patch_unsloth_smart_gradient_checkpointing,
                    unpatch_unsloth_gradient_checkpointing,
                    unpatch_unsloth_smart_gradient_checkpointing,
                )

                unpatch_unsloth_gradient_checkpointing()
                unpatch_unsloth_smart_gradient_checkpointing()
                patch_unsloth_smart_gradient_checkpointing(
                    dtype=torch.bfloat16,
                )
                reset_label = (
                    "reset the task LoRA in place to its initial state "
                    f"(compiler_reset={compiler_reset_seconds:.3f}s, "
                    "adapter_restore="
                    f"{time.perf_counter() - adapter_restore_started:.3f}s/"
                    f"{restored_adapter_tensors} tensors/"
                    f"{restored_adapter_bytes / (1024 ** 2):.1f}MiB, "
                    "wrapper_changes="
                    f"{initial_adapter_snapshot.restored_wrapper_changes})"
                )
            elif reuse_base_model:
                model.zero_grad(set_to_none=True)
                compiler_reset_started = time.perf_counter()
                reset_torch_compiler_cache()
                gc.collect()
                torch.cuda.empty_cache()
                compiler_reset_seconds = (
                    time.perf_counter() - compiler_reset_started
                )
                unload_started = time.perf_counter()
                fp8_restore_started = time.perf_counter()
                restored_fp8_base_count = restored_fp8_base_bytes = 0
                if fp8_frozen_linear:
                    (
                        restored_fp8_base_count,
                        restored_fp8_base_bytes,
                    ) = restore_fp8_base_before_unload(model)
                fp8_restore_seconds = time.perf_counter() - fp8_restore_started
                (
                    restored_modules_to_save,
                    restored_modules_to_save_bytes,
                ) = restore_modules_to_save_before_unload(model)
                unload = getattr(
                    getattr(model, "base_model", None),
                    "unload",
                    None,
                )
                if not callable(unload):
                    raise RuntimeError(
                        "ARC_REUSE_BASE_MODEL requires PEFT "
                        "base_model.unload()"
                    )
                base_model = unload()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                allocated_after_model_release = (
                    torch.cuda.memory_allocated()
                )
                gc.collect()
                torch.cuda.empty_cache()
                allocated_after_cleanup = (
                    torch.cuda.memory_allocated()
                )
                unload_seconds = time.perf_counter() - unload_started
                adapter_attach_started = time.perf_counter()
                model = attach_fresh_task_adapter(
                    base_model,
                    stage_embeddings_to_disk=False,
                    reuse_pissa_residual=pissa_enabled,
                )
                # ``get_peft_model(..., use_gradient_checkpointing=False)``
                # intentionally skips both embedding staging and Unsloth's
                # global smart-checkpoint installation.  Reinstall only the
                # checkpoint callable here; Trainer will bind it to whatever
                # decoder-layer subset the next task's policy selects.
                from unsloth_zoo.gradient_checkpointing import (
                    patch_unsloth_smart_gradient_checkpointing,
                )

                patch_unsloth_smart_gradient_checkpointing(
                    dtype=torch.bfloat16,
                )
                adapter_attach_seconds = (
                    time.perf_counter() - adapter_attach_started
                )
                reset_label = (
                    "reused frozen base and attached a fresh LoRA "
                    f"(compiler_reset={compiler_reset_seconds:.3f}s, "
                    f"unload_cleanup={unload_seconds:.3f}s, "
                    f"adapter_attach={adapter_attach_seconds:.3f}s, "
                    f"fp8_restore={fp8_restore_seconds:.3f}s/"
                    f"{restored_fp8_base_count} projections/"
                    f"{restored_fp8_base_bytes / (1024 ** 2):.1f}MiB CPU, "
                    "restored_modules_to_save="
                    f"{restored_modules_to_save}/"
                    f"{restored_modules_to_save_bytes / (1024 ** 2):.3f}MiB)"
                )
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                allocated_after_model_release = torch.cuda.memory_allocated()
                reset_torch_compiler_cache()
                gc.collect()
                torch.cuda.empty_cache()
                allocated_after_cleanup = torch.cuda.memory_allocated()
                model, tokenizer = load_fresh_task_model()
                reset_label = "loaded a fresh model and LoRA"
            if pissa_enabled and vllm_job_queues is not None:
                # A fresh (non-reused) base may have a newly computed SVD.
                # Export must subtract this model's A0/B0, never task one's.
                initial_vllm_embedding_state = clone_embedding_state_to_cpu(
                    get_peft_model_state_dict(model, adapter_name="default")
                )
                initial_vllm_embedding_state.update(pissa_initialization.weights)
                vllm_peft_config = peft_config_dict(model.peft_config["default"])
            collator = QwenDataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                mlm=False,
                final_assistant_only=final_assistant_only,
                bidirectional_prompt_attention=(
                    bidirectional_prompt_attention
                ),
            )
            formatter = QwenFormatter(tokenizer=tokenizer)
            max_new_tokens = formatter.max_new_tokens()
            print(
                f"[Rank {rank}] {reset_label} before "
                f"puzzle {key} in "
                f"{time.perf_counter() - reset_started:.3f}s; "
                f"prior-task CUDA cleanup="
                f"{allocated_before_reload // (1024 ** 2)}MiB -> "
                f"{allocated_after_model_release // (1024 ** 2)}MiB "
                f"after model release -> "
                f"{allocated_after_cleanup // (1024 ** 2)}MiB",
                flush=True,
            )
        puzzle_ds = arc_test_set.change_keys([key])

        train_augmentation_count = puzzle_ds.train_augmentation_count(
            default_n=int(os.getenv("ARC_TRAIN_AUGMENTATIONS", "16")),
            small_n=int(os.getenv("ARC_SMALL_TASK_TRAIN_AUGMENTATIONS", "32")),
            max_train_pairs=int(
                os.getenv("ARC_SMALL_TASK_MAX_TRAIN_PAIRS", "2")
            ),
            max_average_grid_area=float(
                os.getenv("ARC_SMALL_TASK_MAX_AVERAGE_GRID_AREA", "100")
            ),
        )
        print(
            f"[Rank {rank}] using {train_augmentation_count} color "
            f"augmentations ({8 * train_augmentation_count} training rows) "
            f"for puzzle {key}",
            flush=True,
        )
        train_ds = puzzle_ds.augment(
            n=train_augmentation_count,
            shfl_keys=True,
            seed=1,
            unique_colors=True,
            balanced_examples=True,
            **pair_expansion_options,
        )
        if any(getattr(train_ds, "pair_expansion_records", {}).values()):
            oversized = [
                (row_key, train_ds.get_length(row_key, formatter, "text"))
                for row_key in train_ds.keys
                if train_ds.get_length(row_key, formatter, "text") > max_seq_length
            ]
            if oversized:
                raise ValueError(
                    "Expanded training conversation exceeds --max-seq-length; "
                    "reduce --min-train-pairs or increase the context limit. "
                    f"Original demonstrations were not truncated: {oversized[:3]}"
                )
            expansion_path = write_expansion_manifest(
                train_ds, formatter, root=dir_outputs, task_key=key,
                phase="train", min_train_pairs=min_train_pairs, mode=extra_train_pair_mode,
            )
            print(
                f"[Rank {rank}] training-pair expansion: original={len(puzzle_ds.queries[key]['train'])}, "
                f"minimum={min_train_pairs}, mode={extra_train_pair_mode}, "
                f"rows={len(train_ds.keys)} (unchanged); manifest={expansion_path}",
                flush=True,
            )
        else:
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)
        formatted_train_rows = train_ds.as_list(formatter)
        print(
            f"[Rank {rank}] SFT loss scope for puzzle {key}: "
            f"requested={sft_loss_scope}, "
            f"active={'final' if final_assistant_only else 'all'}; "
            "preceding demonstrations retained as causal context",
            file=sys.__stdout__,
            flush=True,
        )
        task_observed_colors = dataset_colors(puzzle_ds)
        warmup_sft_color_signatures = set()
        if task_rl:
            formatted_train_rows = select_balanced_warmup_rows(
                formatted_train_rows,
                limit=rl_sft_steps,
                seed=stable_task_seed(
                    key,
                    seed=42,
                    namespace="arc-rl-sft",
                ),
            )
            warmup_sft_color_signatures = color_mapping_signatures(
                formatted_train_rows,
                task_observed_colors,
            )
            print(
                f"[Rank {rank}] task RL warm start: "
                f"{len(formatted_train_rows)} final-answer-only SFT rows, "
                f"{len(warmup_sft_color_signatures)} effective color "
                f"mappings, "
                f"then {rl_max_steps} GRPO steps x "
                f"{rl_num_generations} generations "
                f"(micro_batch={rl_micro_batch_size}, "
                f"accumulation={rl_num_generations // rl_micro_batch_size}); "
                f"reward=0 below {rl_reward_min_cell_accuracy:.3f} cell "
                f"accuracy, exact_bonus={rl_exact_bonus:g}",
                flush=True,
            )
        train_token_counts = batch_token_lengths(
            tokenizer,
            (row["text"] for row in formatted_train_rows),
        )
        max_train_tokens = max(train_token_counts, default=0)
        (
            active_qwen3_cublaslt,
            active_qwen3_blas_backend,
            qwen3_cublaslt_min_tokens,
        ) = configure_qwen3_long_context_blas(model, max_train_tokens)
        active_deepspeed = use_long_context_deepspeed(
            deepspeed_long_context,
            max_train_tokens,
            deepspeed_token_threshold,
        )
        active_deepspeed_config = None
        task_peft_state = None
        if active_deepspeed:
            active_deepspeed_config = build_zero3_param_offload_config(
                offload_optimizer=deepspeed_offload_optimizer,
                param_persistence_threshold=(
                    deepspeed_param_persistence_threshold
                ),
                model_persistence_threshold=(
                    deepspeed_model_persistence_threshold
                ),
                max_live_parameters=deepspeed_max_live_parameters,
                prefetch_bucket_size=deepspeed_prefetch_bucket_size,
            )
        active_fp8_frozen_linear = (
            fp8_frozen_linear
            and max_train_tokens >= max(0, fp8_min_train_tokens)
        )
        active_mlp_recompute = recompute_mlp_activations
        active_attention_recompute = recompute_attention_activations
        active_hybrid_mlp_recompute = hybrid_recompute_mlp_activations
        active_hybrid_attention_recompute = (
            hybrid_recompute_attention_activations
        )
        active_gradient_checkpointing = (
            gradient_checkpointing and not selective_activation_recompute
        )
        active_checkpointing_interval = gradient_checkpointing_interval
        requested_checkpointing_percentage = gradient_checkpointing_percentage
        active_checkpointing_profile = resolve_task_gradient_checkpointing_profile(
            gradient_checkpointing_profile,
            lora_r=lora_r,
            fp8_frozen_linear=active_fp8_frozen_linear,
        )
        active_gradient_checkpointing_mode = gradient_checkpointing_mode
        if active_deepspeed:
            # The benchmark intentionally replaces high-percentage Unsloth
            # recomputation with frozen-parameter streaming. A non-zero value
            # is an explicit emergency fallback and uses ordinary GPU-held
            # checkpoint inputs to avoid combining two CPU-offload policies.
            requested_checkpointing_percentage = (
                deepspeed_gradient_checkpointing_percentage
            )
            active_gradient_checkpointing = (
                deepspeed_gradient_checkpointing_percentage > 0
            )
            active_gradient_checkpointing_mode = "standard"
        # SFT always has physical batch one. GRPO resolves a separate policy
        # later from its actual prompt, completion, and micro-batch workload.
        checkpointing_micro_batch_size = 1
        checkpointing_policy_tokens = max_train_tokens
        if requested_checkpointing_percentage == "auto":
            active_checkpointing_percentage = (
                select_task_auto_gradient_checkpointing_percentage(
                    max_train_tokens,
                    profile=gradient_checkpointing_profile,
                    lora_r=lora_r,
                    fp8_frozen_linear=active_fp8_frozen_linear,
                )
            )
        else:
            active_checkpointing_percentage = (
                requested_checkpointing_percentage
            )
        if (
            active_checkpointing_percentage is not None
            and active_checkpointing_percentage <= 0
        ):
            # A zero-percent policy should take the true no-GC path. Merely
            # selecting zero layers while leaving model-level GC enabled keeps
            # unnecessary checkpointing state instead of fully disabling it.
            active_gradient_checkpointing = False
        if auto_activation_recompute:
            active_gradient_checkpointing = (
                gradient_checkpointing
                and max_train_tokens <= mlp_recompute_token_threshold
            )
            active_mlp_recompute = (
                max_train_tokens > mlp_recompute_token_threshold
            )
            active_attention_recompute = (
                max_train_tokens > attention_recompute_token_threshold
            )

        mlp_count, attention_count = configure_activation_recomputation(
            model,
            active_mlp_recompute,
            active_attention_recompute,
        )
        if active_fp8_frozen_linear:
            quantize_fp8_base_weights(
                preserve_bf16=(
                    reuse_base_model
                    or (not train_only and not vllm_final_inference)
                ),
            )
        model._unsloth_gradient_checkpointing_interval = (
            active_checkpointing_interval
        )
        active_gradient_checkpointing_value = (
            resolve_gradient_checkpointing_value(
                active_gradient_checkpointing,
                active_gradient_checkpointing_mode,
            )
        )
        model = FastLanguageModel.for_training(
            model,
            use_gradient_checkpointing=active_gradient_checkpointing_value,
        )
        # Unsloth's train/inference transitions consult this marker. Keep the
        # literal smart mode rather than collapsing it back to bool.
        model._unsloth_gradient_checkpointing = (
            active_gradient_checkpointing_value
        )
        checkpoint_modules = checkpointed_layers = total_layers = 0
        if active_gradient_checkpointing:
            if active_checkpointing_percentage is None:
                (
                    checkpoint_modules,
                    checkpointed_layers,
                    total_layers,
                ) = configure_gradient_checkpointing_interval(
                    model,
                    active_checkpointing_interval,
                )
            else:
                (
                    checkpoint_modules,
                    checkpointed_layers,
                    total_layers,
                ) = configure_gradient_checkpointing_percentage(
                    model,
                    active_checkpointing_percentage,
                )
        hybrid_mlp_count = hybrid_attention_count = 0
        if hybrid_activation_recompute:
            (
                hybrid_mlp_count,
                hybrid_attention_count,
            ) = configure_hybrid_activation_recomputation(
                model,
                active_hybrid_mlp_recompute,
                active_hybrid_attention_recompute,
                hybrid_attention_recompute_percentage,
            )
        fused_qknorm_rope_count = configure_fused_qknorm_rope(
            model,
            fused_qknorm_rope and not active_deepspeed,
        )
        (
            attention_checkpoint_saved_cpu_count,
            attention_checkpoint_smart_buffer_count,
        ) = (
            configure_attention_checkpoint_input_offload(
                model,
                attention_checkpoint_input_offload,
                active_gradient_checkpointing_mode,
            )
        )
        qwen35_gdn_replay_cache_inactive_reasons = []
        if getattr(model.config, "model_type", None) != "qwen3_5_text":
            qwen35_gdn_replay_cache_inactive_reasons.append(
                "model is not qwen3_5_text"
            )
        if not active_fp8_frozen_linear:
            qwen35_gdn_replay_cache_inactive_reasons.append(
                "frozen FP8 is inactive"
            )
        if active_gradient_checkpointing_mode != "unsloth":
            qwen35_gdn_replay_cache_inactive_reasons.append(
                "checkpoint mode is not unsloth"
            )
        if not active_gradient_checkpointing:
            qwen35_gdn_replay_cache_inactive_reasons.append(
                "gradient checkpointing is inactive"
            )
        if checkpointed_layers <= 0 or (
            checkpointed_layers != total_layers and not qwen35_gdn_replay_cache_partial
        ):
            qwen35_gdn_replay_cache_inactive_reasons.append(
                f"{'any-layer' if qwen35_gdn_replay_cache_partial else 'full-layer'} "
                f"checkpointing is inactive "
                f"({checkpointed_layers}/{total_layers})"
            )
        qwen35_gdn_replay_cache_active = bool(
            qwen35_gdn_replay_cache_requested
            and not qwen35_gdn_replay_cache_inactive_reasons
        )
        qwen35_gdn_replay_cache_max_mib_display = (
            "unlimited"
            if qwen35_gdn_replay_cache_max_bytes is None
            else f"{qwen35_gdn_replay_cache_max_bytes / 1024**2:g}"
        )
        if not distributed or rank == 0:
            print(
                f"[Rank {rank}] activation policy: max_tokens="
                f"{max_train_tokens}, physical_micro_batch="
                f"{checkpointing_micro_batch_size}, "
                f"checkpoint_policy_tokens={checkpointing_policy_tokens}, "
                f"auto_activation_recompute="
                f"{auto_activation_recompute}, "
                f"mlp_recompute={active_mlp_recompute} ({mlp_count} layers), "
                f"attention_recompute={active_attention_recompute} "
                f"({attention_count} layers), hybrid_mlp_recompute="
                f"{active_hybrid_mlp_recompute} "
                f"({hybrid_mlp_count} non-full layers), "
                f"hybrid_attention_recompute="
                f"{active_hybrid_attention_recompute} "
                f"({hybrid_attention_count} non-full layers, "
                f"requested_percentage="
                f"{hybrid_attention_recompute_percentage:g})",
                flush=True,
            )
            print(
                f"[Rank {rank}] gradient checkpointing requested="
                f"{gradient_checkpointing if not active_deepspeed else deepspeed_gradient_checkpointing_percentage > 0}, active="
                f"{bool(getattr(model, 'is_gradient_checkpointing', False))}, "
                f"mode={active_gradient_checkpointing_mode}, "
                f"interval={active_checkpointing_interval}, "
                f"percentage={active_checkpointing_percentage} "
                f"(requested={requested_checkpointing_percentage}), "
                f"profile={active_checkpointing_profile}, "
                f"configured_modules={checkpoint_modules}, "
                f"checkpointed_layers={checkpointed_layers}/{total_layers}",
                flush=True,
            )
            print(
                f"[Rank {rank}] fused Qwen3 QKNorm/RoPE: requested="
                f"{fused_qknorm_rope}, active_modules="
                f"{fused_qknorm_rope_count if not active_deepspeed else 0}",
                flush=True,
            )
            print(
                f"[Rank {rank}] attention checkpoint-input offload: "
                f"requested={attention_checkpoint_input_offload}, "
                f"mode={active_gradient_checkpointing_mode}, "
                f"saved_cpu_layers={attention_checkpoint_saved_cpu_count}, "
                f"smart_buffer_layers="
                f"{attention_checkpoint_smart_buffer_count}, "
                f"safe_inplace_qkv_release="
                f"{attention_checkpoint_smart_buffer_count > 0}",
                flush=True,
            )
            print(
                f"[Rank {rank}] training backend: "
                f"{'deepspeed-zero3-param-offload' if active_deepspeed else 'unsloth'}, "
                f"long_context_requested={deepspeed_long_context}, "
                f"strict_token_threshold={deepspeed_token_threshold}, "
                f"optimizer_offload="
                f"{deepspeed_offload_optimizer if active_deepspeed else False}",
                flush=True,
            )
            print(
                f"[Rank {rank}] frozen-linear policy: requested_fp8="
                f"{fp8_frozen_linear}, active_fp8="
                f"{active_fp8_frozen_linear}, "
                f"min_train_tokens={fp8_min_train_tokens}",
                flush=True,
            )
            if qwen3_cublaslt_min_tokens is not None:
                print(
                    f"[Rank {rank}] Qwen3 BLAS policy: auto_cublaslt=True, "
                    f"backend={active_qwen3_blas_backend or 'unchanged'}, "
                    f"active_cublaslt={active_qwen3_cublaslt}, "
                    f"min_train_tokens={qwen3_cublaslt_min_tokens}",
                    flush=True,
                )
            if qwen35_gdn_replay_cache_requested:
                print(
                    f"[Rank {rank}] Qwen3.5 GDN replay cache: active="
                    f"{qwen35_gdn_replay_cache_active}, projections="
                    f"{list(qwen35_gdn_replay_cache_projections)}, "
                    f"checkpointed_layers_only={qwen35_gdn_replay_cache_partial}, "
                    f"max_cache_mib="
                    f"{qwen35_gdn_replay_cache_max_mib_display}"
                    + (
                        ""
                        if qwen35_gdn_replay_cache_active
                        else "; reasons="
                        + ", ".join(
                            qwen35_gdn_replay_cache_inactive_reasons
                        )
                    ),
                    flush=True,
                )

        if formatted_train_rows and (not distributed or rank == 0):
            sample = formatted_train_rows[0]
            sample_token_count = train_token_counts[0]
            if os.getenv("ARC_TRAIN_SAMPLE_PRINT", "1") == "1":
                print(
                    f"[Rank {rank}] training sample {sample['key']} "
                    f"({sample_token_count} tokens):\n"
                    "----- BEGIN TRAINING SAMPLE -----\n"
                    f"{sample['text']}\n"
                    "----- END TRAINING SAMPLE -----",
                    flush=True,
                )

        with (
            io.StringIO() as buf,
            redirect_stdout(buf),
            redirect_stderr(buf),
            ExitStack() as training_runtime,
        ):
            fp8_lora_stats = None
            active_fp8_lora = bool(fp8_lora and active_fp8_frozen_linear)
            if active_fp8_lora:
                projection_count = validate_fp8_lora_model(model)
                from unsloth.kernels.fp8_lora import fp8_lora_training

                fp8_lora_stats = training_runtime.enter_context(fp8_lora_training())
                print(
                    f"[Rank {rank}] FP8 LoRA compute enabled: projections={projection_count}, "
                    "weights/activations=E4M3, gradient_operands=E5M2, "
                    "master_weights/gradients/export=BF16, optimizer=adamw_torch; "
                    "attention/norms/embeddings/head retain their existing precision",
                    file=sys.__stdout__, flush=True,
                )
            if active_deepspeed:
                deepspeed_master_port = training_runtime.enter_context(
                    single_process_deepspeed_environment()
                )
                training_runtime.enter_context(
                    deepspeed_standard_gradient_checkpointing_environment(
                        model,
                        dtype=torch.bfloat16,
                    )
                )
                print(
                    f"[Rank {rank}] DeepSpeed world-size-one runtime ready "
                    f"on local CUDA device 0 (master_port="
                    f"{deepspeed_master_port}); ZeRO-3 CPU parameter "
                    "offload active",
                    file=sys.__stdout__,
                    flush=True,
                )

            trainer_kwargs = {}
            if persistent_optimizer is not None:
                restart_optimizer_learning_rate(
                    persistent_optimizer,
                    train_args["learning_rate"],
                )
                trainer_kwargs["optimizers"] = (
                    persistent_optimizer,
                    None,
                )

            task_train_args = dict(
                train_args,
                gradient_checkpointing=active_gradient_checkpointing,
            )
            if active_deepspeed:
                task_train_args.update(
                    deepspeed=active_deepspeed_config,
                    local_rank=0,
                )
            if active_gradient_checkpointing and (
                active_gradient_checkpointing_mode == "unsloth"
                or active_deepspeed
            ):
                # Smart checkpointing uses Unsloth's reentrant autograd
                # function. The DeepSpeed route also requires pristine
                # reentrant PyTorch checkpointing: Transformers 5 defaults to
                # non-reentrant checkpointing, whose AOT backward currently
                # fails inside Qwen3.5's FLA Triton operators.
                task_train_args["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": True,
                }
            if task_rl:
                # The selected dataset is already balanced to this size.  An
                # explicit step cap keeps the warm start at exactly the
                # requested number of optimizer updates.
                task_train_args.update(
                    num_train_epochs=1.0,
                    max_steps=rl_sft_steps,
                )
            trainer = UnslothFixedTrainer(
                model=model,
                processing_class=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(formatted_train_rows),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**task_train_args),
                **trainer_kwargs,
            )
            trainer._arc_gradient_checkpointing_mode = (
                active_gradient_checkpointing_mode
                if active_gradient_checkpointing
                else "standard"
            )
            if os.environ.get("ARC_PIPELINE_REPORT_DIR"):
                from pipeline_runtime import attach_training_runtime_report

                attach_training_runtime_report(trainer, key)
            configure_task_optimizer(
                trainer,
                optimizer_name=optimizer,
                flash_adamw_quantize=flash_adamw_quantize,
                flash_adamw_master_weight_bits=(
                    flash_adamw_master_weight_bits
                ),
                normuon_adjust_lr=normuon_adjust_lr,
                normuon_momentum=normuon_momentum,
                normuon_beta2=normuon_beta2,
                normuon_megabatch_size=normuon_megabatch_size,
                normuon_use_triton=normuon_use_triton,
                muon_adjust_lr=muon_adjust_lr,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_ns_chunk_size=muon_ns_chunk_size,
                muon_min_8bit_size=muon_min_8bit_size,
                muon_scalar_optimizer=muon_scalar_optimizer,
                log_prefix=f"[Rank {rank}] SFT",
            )

            qwen35_gdn_replay_cache_controller = None
            if qwen35_gdn_replay_cache_active:
                from unsloth.kernels.qwen35_gdn_replay_cache import (
                    estimate_qwen35_gdn_replay_cache_bytes,
                    patch_qwen35_gdn_projection_replay_cache,
                )

                estimated_cache_bytes = (
                    estimate_qwen35_gdn_replay_cache_bytes(
                        model,
                        batch_size=1,
                        sequence_length=max_train_tokens,
                        dtype=torch.bfloat16,
                        projection_names=(
                            qwen35_gdn_replay_cache_projections
                        ),
                    )
                )
                print(
                    f"[Rank {rank}] prepared exact GDN checkpoint replay; "
                    f"estimated_full_cache="
                    f"{estimated_cache_bytes / 1024**2:.1f}MiB, "
                    f"budget="
                    f"{qwen35_gdn_replay_cache_max_mib_display}"
                    + (
                        ""
                        if qwen35_gdn_replay_cache_max_bytes is None
                        else "MiB"
                    ),
                    file=sys.__stdout__,
                    flush=True,
                )

                def setup_qwen35_gdn_replay_cache(model_to_patch):
                    cache_kwargs = {}
                    if qwen35_gdn_replay_cache_partial:
                        from unsloth.kernels import qwen35_gdn_replay_cache as cache_module

                        if not getattr(cache_module, "UNSLOTH_SUPPORTS_PARTIAL_GDN_REPLAY_CACHE", False):
                            raise RuntimeError(
                                "Partial GDN replay cache requires the matching Unsloth patch; "
                                "update Unsloth or disable ARC_QWEN35_GDN_REPLAY_CACHE_PARTIAL"
                            )
                        cache_kwargs["checkpointed_layers_only"] = True
                    controller = patch_qwen35_gdn_projection_replay_cache(
                        model_to_patch,
                        enabled=True,
                        projection_names=(
                            qwen35_gdn_replay_cache_projections
                        ),
                        max_cache_bytes=(
                            qwen35_gdn_replay_cache_max_bytes
                        ),
                        assume_unsloth_checkpointing=True,
                        **cache_kwargs,
                    )
                    if controller.stats.patched_modules == 0:
                        raise RuntimeError(
                            "Qwen3.5 GDN replay caching was requested, but "
                            "no eligible frozen FP8 GDN projections were "
                            "found. Ensure UNSLOTH_QWEN35_FP8_GDN=1 is "
                            "active before training."
                        )
                    print(
                        f"[Rank {rank}] patched "
                        f"{controller.stats.patched_modules} frozen GDN "
                        "projections for exact checkpoint replay",
                        file=sys.__stdout__,
                        flush=True,
                    )
                    return controller

                trainer._arc_qwen35_gdn_replay_cache_setup = (
                    setup_qwen35_gdn_replay_cache
                )

            parameter_diagnostics = os.getenv(
                "ARC_TRAIN_PARAMETER_DIAGNOSTICS",
                "0",
            ).strip().lower() in {"1", "true", "yes", "on"}
            diagnostic_parameters = []
            if parameter_diagnostics:
                wanted = {
                    "lora_A": False,
                    "lora_B": False,
                    "embed_tokens": False,
                    "lm_head": False,
                }
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    category = None
                    if "lora_A" in parameter_name and not wanted["lora_A"]:
                        category = "lora_A"
                    elif "lora_B" in parameter_name and not wanted["lora_B"]:
                        category = "lora_B"
                    elif (
                        "embed_tokens" in parameter_name
                        and not wanted["embed_tokens"]
                    ):
                        category = "embed_tokens"
                    elif "lm_head" in parameter_name and not wanted["lm_head"]:
                        category = "lm_head"
                    if category is None:
                        continue
                    wanted[category] = True
                    diagnostic_parameters.append(
                        (
                            category,
                            parameter_name,
                            parameter,
                            parameter.detach().float().cpu().clone(),
                        )
                    )
                print(
                    f"[Rank {rank}] parameter diagnostics selected: "
                    + ", ".join(
                        f"{category}={name}"
                        for category, name, _parameter, _before in (
                            diagnostic_parameters
                        )
                    ),
                    file=sys.__stdout__,
                    flush=True,
                )

            try:
                if profile_training:
                    profile_path = (
                        f"/kaggle/working/training_profile_{key}_rank{rank}.txt"
                    )
                    trace_path = (
                        f"/kaggle/working/training_profile_{key}_rank{rank}.json"
                    )

                    def save_profile(profiler):
                        table = profiler.key_averages(
                            group_by_input_shape=True,
                        ).table(
                            sort_by="self_cuda_time_total",
                            row_limit=120,
                        )
                        with open(profile_path, "w") as profile_file:
                            profile_file.write(table)
                        profiler.export_chrome_trace(trace_path)

                    profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=profile_wait_steps,
                            warmup=profile_warmup_steps,
                            active=profile_active_steps,
                            repeat=1,
                        ),
                        on_trace_ready=save_profile,
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=False,
                        with_flops=True,
                    )
                    trainer.add_callback(TrainingProfilerCallback(profiler))
                    with profiler:
                        stats = trainer.train()
                else:
                    stats = trainer.train()

                if fp8_lora_stats is not None:
                    if getattr(stats, "global_step", 0) > 0 and (
                        fp8_lora_stats.forward_gemms == 0 or fp8_lora_stats.backward_gemms == 0
                    ):
                        raise RuntimeError("FP8 LoRA requested but no forward/backward GEMMs ran")
                    print(
                        f"[Rank {rank}] FP8 LoRA GEMM counts: {vars(fp8_lora_stats)}",
                        file=sys.__stdout__, flush=True,
                    )

                if parameter_diagnostics:
                    optimizer = unwrap_optimizer(trainer.optimizer)
                    for category, name, parameter, before in (
                        diagnostic_parameters
                    ):
                        after = parameter.detach().float().cpu()
                        delta = after - before
                        state = optimizer.state.get(parameter, {})
                        exp_avg = state.get("exp_avg")
                        exp_avg_sq = state.get("exp_avg_sq")
                        print(
                            f"[Rank {rank}] parameter diagnostic "
                            f"{category}: name={name}, "
                            f"changed={(delta != 0).sum().item()}/"
                            f"{delta.numel()}, max_abs_delta="
                            f"{delta.abs().max().item():.9g}, "
                            f"before_norm={before.norm().item():.9g}, "
                            f"after_norm={after.norm().item():.9g}, "
                            f"exp_avg_norm="
                            f"{None if exp_avg is None else exp_avg.float().norm().item()}, "
                            f"exp_avg_sq_norm="
                            f"{None if exp_avg_sq is None else exp_avg_sq.float().norm().item()}",
                            file=sys.__stdout__,
                            flush=True,
                        )
            except BaseException as error:
                print(
                    f"[Rank {rank}] training failed after global_step="
                    f"{getattr(trainer.state, 'global_step', None)} with "
                    f"{type(error).__name__}",
                    flush=True,
                )
                try:
                    log_optimizer_state_memory(
                        rank,
                        getattr(trainer, "optimizer", None),
                        "at training failure",
                    )
                except BaseException as diagnostic_error:
                    print(
                        f"[Rank {rank}] optimizer-state diagnostic failed: "
                        f"{type(diagnostic_error).__name__}: "
                        f"{diagnostic_error}",
                        flush=True,
                    )
                if active_deepspeed:
                    destroy_deepspeed_engine(trainer)
                raise
            finally:
                qwen35_gdn_replay_cache_controller = getattr(
                    trainer,
                    "_arc_qwen35_gdn_replay_cache_controller",
                    None,
                )
                if qwen35_gdn_replay_cache_controller is not None:
                    replay_stats = (
                        qwen35_gdn_replay_cache_controller.stats
                    )
                    print(
                        f"[Rank {rank}] Qwen3.5 GDN replay cache stats: "
                        f"patched={replay_stats.patched_modules}, "
                        f"captures={replay_stats.captures}, "
                        f"hits={replay_stats.hits}, "
                        f"misses={replay_stats.misses}, "
                        f"budget_skips={replay_stats.budget_skips}, "
                        f"stale={replay_stats.stale_entries}, "
                        f"peak={replay_stats.peak_retained_bytes / 1024**2:.1f}MiB, "
                        f"retained_after_train="
                        f"{replay_stats.retained_bytes / 1024**2:.1f}MiB",
                        file=sys.__stdout__,
                        flush=True,
                    )
                    # Restore the ordinary FP8 forwards before any BF16 base
                    # restoration, inference transition, or model teardown.
                    qwen35_gdn_replay_cache_controller.unpatch()

            log_optimizer_state_memory(
                rank,
                trainer.optimizer,
                "after training",
            )

            if keep_training_state:
                persistent_optimizer = unwrap_optimizer(trainer.optimizer)

            if active_deepspeed:
                model = trainer.accelerator.unwrap_model(
                    trainer.model_wrapped,
                    keep_fp32_wrapper=False,
                )
                if not train_only:
                    task_peft_state = dict(
                        gather_zero3_peft_state_dict(
                            model,
                            adapter_name="default",
                        )
                    )
                    task_peft_bytes = sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in task_peft_state.values()
                    )
                    print(
                        f"[Rank {rank}] gathered "
                        f"{len(task_peft_state)} ZeRO-3 PEFT tensors/"
                        f"{task_peft_bytes / 1024**2:.1f} MiB to CPU",
                        file=sys.__stdout__,
                        flush=True,
                    )
                destroy_deepspeed_engine(trainer)
                # A ZeRO-offloaded model is not safe for native post-training
                # inference after its world-one process group is destroyed.
                # Routed services consume the immutable CPU adapter snapshot.
                model = None
            else:
                model = trainer.accelerator.unwrap_model(
                    model,
                    keep_fp32_wrapper=False,
                )

            del trainer
            # The task-RL phase constructs a second Trainer and optimizer for
            # the same model.  Collect cycles from the SFT Trainer before that
            # allocation so optimizer state and temporary buffers cannot
            # overlap the first GRPO forward on a 22 GiB L4.
            gc.collect()
            torch.cuda.empty_cache()

        if task_rl:
            reload_started = time.perf_counter()
            sft_adapter_state = {
                name: tensor.detach().to(device="cpu", copy=True)
                for name, tensor in get_peft_model_state_dict(
                    model,
                    adapter_name="default",
                ).items()
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()
            reset_torch_compiler_cache()
            gc.collect()
            torch.cuda.empty_cache()

            model, tokenizer = load_fresh_task_model()
            set_peft_model_state_dict(
                model,
                sft_adapter_state,
                adapter_name="default",
            )
            del sft_adapter_state
            gc.collect()
            torch.cuda.empty_cache()
            collator = QwenDataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                mlm=False,
                final_assistant_only=True,
                bidirectional_prompt_attention=False,
            )
            formatter = QwenFormatter(tokenizer=tokenizer)
            max_new_tokens = formatter.max_new_tokens()
            print(
                f"[Rank {rank}] reloaded the SFT adapter into a fresh "
                f"GRPO model in "
                f"{time.perf_counter() - reload_started:.3f}s; "
                f"CUDA allocated="
                f"{torch.cuda.memory_allocated() // (1024 ** 2)}MiB, "
                f"reserved="
                f"{torch.cuda.memory_reserved() // (1024 ** 2)}MiB",
                flush=True,
            )
            rl_started = time.perf_counter()
            vllm_service_index = None
            if rl_rollout_backend == "vllm":
                vllm_service_index = rank % len(vllm_job_queues)
            (
                model,
                rl_stats,
                rl_reward_summary,
                rl_rollout_history,
            ) = run_task_grpo(
                model=model,
                tokenizer=tokenizer,
                puzzle_dataset=puzzle_ds,
                formatter=formatter,
                task_key=key,
                rank=rank,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                gradient_checkpointing=(
                    gradient_checkpointing
                    and not selective_activation_recompute
                ),
                gradient_checkpointing_mode=(
                    gradient_checkpointing_mode
                ),
                gradient_checkpointing_interval=(
                    active_checkpointing_interval
                ),
                gradient_checkpointing_percentage=(
                    requested_checkpointing_percentage
                ),
                gradient_checkpointing_profile=(
                    gradient_checkpointing_profile
                ),
                max_grad_norm=0.1,
                max_steps=rl_max_steps,
                num_generations=rl_num_generations,
                micro_batch_size=rl_micro_batch_size,
                auto_micro_batch=rl_auto_micro_batch,
                augmentations=rl_augmentations,
                max_prompt_tokens=rl_max_prompt_tokens,
                min_cell_accuracy=rl_reward_min_cell_accuracy,
                exact_bonus=rl_exact_bonus,
                temperature=rl_temperature,
                learning_rate=rl_learning_rate,
                optimizer=optimizer,
                flash_adamw_quantize=flash_adamw_quantize,
                flash_adamw_master_weight_bits=(
                    flash_adamw_master_weight_bits
                ),
                normuon_adjust_lr=normuon_adjust_lr,
                normuon_momentum=normuon_momentum,
                normuon_beta2=normuon_beta2,
                normuon_megabatch_size=normuon_megabatch_size,
                normuon_use_triton=normuon_use_triton,
                muon_adjust_lr=muon_adjust_lr,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_ns_chunk_size=muon_ns_chunk_size,
                muon_min_8bit_size=muon_min_8bit_size,
                muon_scalar_optimizer=muon_scalar_optimizer,
                color_signature_colors=task_observed_colors,
                excluded_color_signatures=(
                    warmup_sft_color_signatures
                ),
                rollout_backend=rl_rollout_backend,
                vllm_job_queue=(
                    None
                    if vllm_service_index is None
                    else vllm_job_queues[vllm_service_index]
                ),
                vllm_response_queue=(
                    None
                    if vllm_service_index is None
                    else vllm_rollout_response_queues[rank]
                ),
                vllm_adapter_root=vllm_adapter_root,
                vllm_run_id=vllm_run_id,
                vllm_task_adapter_id=(
                    None
                    if vllm_service_index is None
                    else int(vllm_adapter_ids[key])
                ),
                vllm_initial_embedding_state=(
                    None
                    if vllm_service_index is None
                    else initial_vllm_embedding_state
                ),
                vllm_peft_config=(
                    None
                    if vllm_service_index is None
                    else vllm_peft_config
                ),
                vllm_service_index=vllm_service_index,
                vllm_response_timeout_seconds=(
                    # Final DFS may intentionally use a very short diagnostic
                    # timeout. Do not reuse it for synchronous GRPO rollout:
                    # vLLM cold start alone can take roughly one minute.
                    1200.0
                ),
            )
            print(
                f"[Rank {rank}] GRPO stats for puzzle {key}: {rl_stats}; "
                f"reward={rl_reward_summary}; "
                f"rollouts={rl_rollout_history}; "
                f"runtime={time.perf_counter() - rl_started:.3f}s",
                flush=True,
            )

        completed_task_count += 1

        if distributed:
            current_device = torch.cuda.current_device()
            torch.distributed.barrier(device_ids=[current_device])
            torch.distributed.destroy_process_group()
            for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
                os.environ.pop(name, None)
            if rank != 0:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"[Rank {rank}] synchronized training complete for "
                    f"puzzle {key}; inference runs on rank 0"
                )
                break

        gc.collect()
        torch.cuda.empty_cache()

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

        torch.cuda.reset_peak_memory_stats()

        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        if train_only:
            print(f"[Rank {rank}] train-only mode finished puzzle {key}")
            continue

        if vllm_final_inference:
            from arc_vllm_adapter import (
                acquire_adapter_staging_root,
                make_adapter_name,
                preserve_adapter_artifact,
                publish_vllm_adapter,
                remove_adapter_artifact,
                remove_job_adapter_artifact,
            )
            from arc_vllm_router import (
                NoReadyVllmServiceError,
                record_adapter_publication_and_maybe_activate,
                release_service_task,
                reserve_least_loaded_service,
                service_load_snapshot,
            )

            adapter_id = int(vllm_adapter_ids[key])
            adapter_name = make_adapter_name(
                vllm_run_id,
                adapter_id,
                key,
            )
            print(
                f"[Rank {rank}] waiting for vLLM pending-adapter capacity "
                f"for {key}",
                flush=True,
            )
            pending_wait_started = time.perf_counter()
            if not acquire_pending_adapter_slot(
                vllm_pending_adapter_slots,
                abort_event=abort_event,
                end_time=end_time,
            ):
                stop_reason = (
                    "fatal sibling/service failure"
                    if abort_event is not None and abort_event.is_set()
                    else "run deadline"
                )
                print(
                    f"[Rank {rank}] stopped waiting for vLLM pending-adapter "
                    f"capacity for {key}: {stop_reason}",
                    flush=True,
                )
                break
            pending_wait_seconds = time.perf_counter() - pending_wait_started
            service_index = None
            service_loads = service_load_snapshot(
                vllm_service_outstanding, vllm_service_load_lock,
            )
            while not shared_pending_pool and service_index is None:
                try:
                    service_index, service_loads = reserve_least_loaded_service(
                        vllm_service_outstanding,
                        vllm_service_load_lock,
                        preferred_service_index=(
                            rank % len(vllm_job_queues)
                            if task_rl and rl_rollout_backend == "vllm"
                            else None
                        ),
                        service_active=vllm_service_active,
                        service_ready=vllm_service_ready,
                    )
                except NoReadyVllmServiceError:
                    if (
                        abort_event is not None
                        and abort_event.is_set()
                    ) or time.time() >= end_time:
                        break
                    time.sleep(0.5)
            if not shared_pending_pool and service_index is None:
                vllm_pending_adapter_slots.release()
                stop_reason = (
                    "fatal sibling/service failure"
                    if abort_event is not None and abort_event.is_set()
                    else "run deadline"
                )
                print(
                    f"[Rank {rank}] stopped waiting for a ready vLLM "
                    f"service for {key}: {stop_reason}",
                    flush=True,
                )
                break
            if not shared_pending_pool:
                print(
                    f"[Rank {rank}] assigned {key} to vLLM service "
                    f"{service_index}; outstanding={list(service_loads)}",
                    flush=True,
                )
            adapter_staging_index = None
            adapter_publish_root = vllm_adapter_root
            try:
                if vllm_adapter_staging_slots:
                    adapter_staging_index, adapter_publish_root = (
                        acquire_adapter_staging_root(
                            vllm_adapter_roots,
                            vllm_adapter_staging_slots,
                            abort_event=abort_event,
                            end_time=end_time,
                        )
                    )
            except BaseException:
                if service_index is not None:
                    release_service_task(
                        service_index,
                        vllm_service_outstanding,
                        vllm_service_load_lock,
                    )
                vllm_pending_adapter_slots.release()
                raise
            adapter_path = None
            try:
                current_peft_state = (
                    task_peft_state
                    if task_peft_state is not None
                    else get_peft_model_state_dict(
                        model,
                        adapter_name="default",
                    )
                )
                try:
                    adapter_path, adapter_manifest = publish_vllm_adapter(
                        root=adapter_publish_root,
                        adapter_name=adapter_name,
                        adapter_id=adapter_id,
                        task_key=key,
                        current_state=current_peft_state,
                        initial_state=initial_vllm_embedding_state,
                        config=vllm_peft_config,
                        pissa_base_mode=pissa_base_mode,
                        pissa_base_id=(shared_pissa_manifest["base_id"] if shared_pissa_manifest else None),
                        extra_manifest={
                            "trainer_rank": rank,
                            "max_train_tokens": max_train_tokens,
                            "training_seconds": time.time() - start_time,
                            "training_global_step": getattr(
                                stats,
                                "global_step",
                                None,
                            ),
                            "training_loss": getattr(
                                stats,
                                "training_loss",
                                None,
                            ),
                            "training_backend": (
                                "deepspeed-zero3-param-offload"
                                if active_deepspeed
                                else "unsloth"
                            ),
                            "lora_method": lora_method_config["method"],
                            "training_precision": precision_manifest(
                                base_fp8=active_fp8_frozen_linear,
                                lora_fp8=active_fp8_lora,
                                stats=fp8_lora_stats,
                            ),
                            "lora_effective_scaling": (
                                effective_lora_scaling
                            ),
                            "deepspeed_token_threshold": (
                                int(deepspeed_token_threshold)
                                if active_deepspeed
                                else None
                            ),
                            "deepspeed_gc_percentage": (
                                float(
                                    deepspeed_gradient_checkpointing_percentage
                                )
                                if active_deepspeed
                                else None
                            ),
                            "pending_capacity_wait_seconds": (
                                pending_wait_seconds
                            ),
                            "vllm_service_index": service_index,
                            "vllm_shared_pending_pool": shared_pending_pool,
                            "vllm_service_loads_after_reservation": list(
                                service_loads
                            ),
                            "adapter_staging_index": adapter_staging_index,
                            "adapter_staging_root": adapter_publish_root,
                        },
                    )
                    if vllm_preserve_adapter_root:
                        preserve_started = time.perf_counter()
                        preserved_path = preserve_adapter_artifact(
                            adapter_path,
                            vllm_preserve_adapter_root,
                        )
                        print(
                            f"[Rank {rank}] preserved {adapter_name} at "
                            f"{preserved_path} in "
                            f"{time.perf_counter() - preserve_started:.3f}s",
                            flush=True,
                        )
                finally:
                    # PEFT state_dict values alias the live GPU parameters.
                    # Keeping this local across loop iterations pins the
                    # completed task's ~1 GiB adapter beside the next model.
                    del current_peft_state
                    task_peft_state = None
            except BaseException:
                if adapter_path is not None:
                    try:
                        remove_adapter_artifact(
                            adapter_path,
                            adapter_publish_root,
                        )
                    except FileNotFoundError:
                        pass
                if adapter_staging_index is not None:
                    vllm_adapter_staging_slots[
                        adapter_staging_index
                    ].release()
                if service_index is not None:
                    release_service_task(
                        service_index,
                        vllm_service_outstanding,
                        vllm_service_load_lock,
                    )
                vllm_pending_adapter_slots.release()
                raise
            published_at = time.time()
            job = {
                "kind": "adapter",
                "task_key": key,
                "adapter_id": adapter_id,
                "adapter_name": adapter_name,
                "adapter_path": str(adapter_path),
                "adapter_root": str(adapter_publish_root),
                "adapter_staging_index": adapter_staging_index,
                "max_train_tokens": max_train_tokens,
                "published_at": published_at,
                "trainer_rank": rank,
                "export_seconds": adapter_manifest["export_seconds"],
                "tensor_bytes": adapter_manifest["tensor_bytes"],
                "pending_capacity_wait_seconds": pending_wait_seconds,
                "service_index": service_index,
                "service_loads_after_reservation": list(service_loads),
                "ground_truth_outputs": (
                    output_nll_ground_truths.get(key)
                    if report_output_nll
                    else None
                ),
            }
            pool_pending = None
            try:
                if shared_pending_pool:
                    from arc_vllm_pending_pool import publish_pending_adapter

                    pool_pending = publish_pending_adapter(
                        vllm_shared_pending_queue, vllm_shared_pending_count,
                        vllm_service_load_lock, job,
                    )
                else:
                    vllm_job_queues[service_index].put(job)
            except BaseException:
                remove_job_adapter_artifact(
                    job,
                    vllm_adapter_root,
                    vllm_adapter_staging_slots,
                )
                vllm_pending_adapter_slots.release()
                if service_index is not None:
                    release_service_task(
                        service_index,
                        vllm_service_outstanding,
                        vllm_service_load_lock,
                    )
                raise
            published_count = None
            active_services = None
            if int(vllm_ramp_after_adapters) > 0:
                (
                    published_count,
                    adaptive_service_index,
                    service_loads,
                    active_services,
                ) = record_adapter_publication_and_maybe_activate(
                    vllm_published_adapter_count,
                    int(vllm_ramp_after_adapters),
                    vllm_service_outstanding,
                    vllm_service_active,
                    vllm_service_load_lock,
                    vllm_ramp_ready_event,
                    service_limit=vllm_ramp_service_limit,
                )
            print(
                f"[Rank {rank}] published {adapter_name} in "
                f"{adapter_manifest['export_seconds']:.3f}s "
                f"({adapter_manifest['tensor_bytes'] / 1024**2:.1f} MiB); "
                f"pending-capacity wait={pending_wait_seconds:.3f}s; "
                + (
                    f"shared pending pool={pool_pending}; "
                    if shared_pending_pool else f"vLLM service={service_index}; "
                )
                + (
                    "converting this trainer into an inference service"
                    if adaptive_service_index is not None
                    else (
                        "continuing to the next training task without "
                        "waiting for inference"
                    )
                )
                + (
                    f"; adaptive publications={published_count}, "
                    f"active_services={list(active_services)}"
                    if published_count is not None
                    else ""
                ),
                flush=True,
            )
            if adaptive_service_index is not None:
                print(
                    f"[Rank {rank}] adaptive inference threshold reached; "
                    f"stop claiming training tasks and convert this GPU to "
                    f"service {adaptive_service_index}",
                    flush=True,
                )
                break
            continue

        if active_fp8_frozen_linear:
            from unsloth.kernels.fast_lora import (
                restore_model_lora_base_weights_bf16,
            )

            restored_fp8_linears = restore_model_lora_base_weights_bf16(
                model
            )
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"[Rank {rank}] restored {restored_fp8_linears} exact "
                "BF16 base weights for native inference",
                flush=True,
            )

        model = FastLanguageModel.for_inference(model)

        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        inference_setup_start = time.perf_counter()
        puzzle_ds_multi = puzzle_ds.split_multi_replies()
        eval_ds = puzzle_ds_multi.augment(n=2, seed=2, **pair_expansion_options)
        if predict_train:
            train_prediction_ds_multi = puzzle_ds.split_train_replies()
            puzzle_ds_multi = puzzle_ds_multi.append(
                train_prediction_ds_multi
            )
            # Keep the historical test-view RNG stream unchanged. Augmenting
            # the combined dataset would consume extra random permutations
            # between test keys and silently alter normal predictions.
            eval_ds = eval_ds.append(
                train_prediction_ds_multi.augment(n=2, seed=2, **pair_expansion_options)
            )
        if any(
            len(q["train"]) < min_train_pairs for q in puzzle_ds_multi.queries.values()
        ) and any(
            eval_ds.get_length(k, formatter, "input") > max_seq_length - max_new_tokens
            for k in eval_ds.keys
        ):
            raise ValueError(
                "Expanded native inference prompt would be truncated; use vLLM's "
                "per-view context gating, lower --min-train-pairs, or increase context"
            )
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)
        rescore_seeds = {
            subkeys[0].split(".")[0]: stable_task_seed(
                subkeys[0].split(".")[0],
                seed=int(os.getenv("ARC_RESCORE_SEED", "0")),
                namespace="rescore",
            )
            for subkeys in test_id_to_subkeys.values()
        }

        batches = build_inference_batches(
            test_id_to_subkeys,
            inference_batch_size,
        )
        active_dfs_mode = dfs_mode
        if dfs_mode == "auto":
            active_dfs_mode = (
                "frontier"
                if (
                    len(test_id_to_subkeys) == 1
                    and max_train_tokens <= frontier_max_prompt_tokens
                )
                else "recursive"
            )
        print(
            f"[Rank {rank}] inference policy for puzzle {key}: "
            f"requested={dfs_mode}, active={active_dfs_mode}, "
            f"decode_groups={len(batches)}, "
            f"max_train_tokens={max_train_tokens}, "
            f"frontier_token_limit={frontier_max_prompt_tokens}, "
            f"max_beam_score={max_score:.6f}, "
            f"zero_candidate_shape_fallback="
            f"{zero_candidate_shape_fallback}, "
            f"shape_fallback_beam_width={shape_fallback_beam_width}, "
            f"rescore_views={rescore_views}, "
            f"rescore_schedule={rescore_schedule}, "
            f"predict_train={predict_train}, "
            f"rescore_seeds={rescore_seeds}",
            flush=True,
        )
        inference_setup_seconds = (
            time.perf_counter() - inference_setup_start
        )
        inference_start_time = time.time()
        decode_seconds = 0.0
        scoring_seconds = 0.0
        serialization_seconds = 0.0
        decode_calls = 0
        score_calls = 0
        saved_candidate_sets = 0
        raw_beam_paths = 0
        invalid_beam_paths = 0
        monochrome_beam_paths = 0
        valid_beam_paths = 0
        fallback_decode_seconds = 0.0
        fallback_decode_calls = 0
        output_nll_seconds = 0.0
        output_nll_count = 0
        valid_base_keys = set()
        with torch.inference_mode():

            known_scores = {}

            def process_decode_results(subkeys, dfs_result, phase):
                nonlocal scoring_seconds
                nonlocal serialization_seconds
                nonlocal score_calls
                nonlocal saved_candidate_sets
                nonlocal raw_beam_paths
                nonlocal invalid_beam_paths
                nonlocal monochrome_beam_paths
                nonlocal valid_beam_paths

                for subkey_id, scored_beams in dfs_result:

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []
                    raw_beam_paths += len(scored_beams)

                    for beam_score, suffix_tokens in scored_beams:

                        array = formatter.convert_tokens_to_array(
                            suffix_tokens
                        )
                        if array is None:
                            invalid_beam_paths += 1
                            continue

                        solution = puzzle_ds_multi.invert_mod(
                            array,
                            subkey,
                            inv_perm=True,
                        )
                        if (
                            zero_candidate_shape_fallback
                            and np.unique(solution).size < 2
                        ):
                            monochrome_beam_paths += 1
                            print(
                                f"[Rank {rank}] rejecting monochrome "
                                f"candidate from {subkey} ({phase})"
                            )
                            continue
                        valid_beam_paths += 1

                        grid_id = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(
                                f"[Rank {rank}] scoring {subkey} "
                                f"#{len(decoded_result)} ({phase})"
                            )
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={
                                    bk: puzzle_ds_multi.queries.get(bk)
                                },
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment_exact_views(
                                rescore_views,
                                seed=rescore_seeds[bk],
                                **pair_expansion_options,
                            )
                            if len(puzzle_ds_multi.queries[bk]["train"]) < min_train_pairs and any(
                                aug_dataset.get_length(k, formatter, "input") > max_seq_length - max_new_tokens
                                for k in aug_dataset.keys
                            ):
                                raise ValueError("Expanded native rescore prompt would be truncated")
                            aug_dataset = aug_dataset.cut_to_len(
                                formatter=formatter,
                                name="input",
                                max_len=(
                                    max_seq_length - max_new_tokens
                                ),
                            )
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(
                                formatter
                            ):
                                aug_queries.append(
                                    augmented_sample["input"]
                                )
                                aug_answers.append(
                                    augmented_sample["reply"]
                                )
                            score_start = time.perf_counter()
                            augmented_scores = calc_scores(
                                aug_queries,
                                aug_answers,
                                tokenizer,
                                model,
                                batch_size=score_batch_size,
                            )
                            scoring_seconds += (
                                time.perf_counter() - score_start
                            )
                            score_calls += 1
                            known_scores[grid_id] = augmented_scores

                        decoded_result.append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "score_aug_count": len(augmented_scores),
                            "score_aug_seed": rescore_seeds[bk],
                            "solution": solution,
                        })

                    if len(decoded_result):
                        serialization_start = time.perf_counter()
                        output_dir = dir_outputs
                        if parse_train_prediction_key(subkey) is not None:
                            if train_prediction_output_dir is None:
                                raise RuntimeError(
                                    "Training prediction output was produced "
                                    "while --predict-train is disabled"
                                )
                            output_dir = train_prediction_output_dir
                        with bz2.BZ2File(
                            os.path.join(output_dir, subkey),
                            "w",
                        ) as output_file:
                            pickle.dump(decoded_result, output_file)
                        serialization_seconds += (
                            time.perf_counter() - serialization_start
                        )
                        saved_candidate_sets += 1
                        valid_base_keys.add(bk)

            for subkeys in batches:

                inference_spend_time = time.time() - inference_start_time
                if (
                    inference_spend_time > inference_timeout_seconds
                    or time.time() > end_time
                ):
                    print(
                        f"[Rank {rank}] inference timeout after "
                        f"{inference_spend_time:.1f}s for puzzle {key}"
                    )
                    break

                print(f"[Rank {rank}] decoding {subkeys}")

                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                token_lengths = {len(token_ids) for token_ids in tokens}
                if len(token_lengths) != 1:
                    raise RuntimeError(
                        f"Inference batch contains different token lengths: "
                        f"{sorted(token_lengths)} for {subkeys}"
                    )

                decode_start = time.perf_counter()
                if active_dfs_mode == "frontier":
                    dfs_result = inference_turbo_frontier(
                        model,
                        tokens,
                        max_new_tokens,
                        max_score,
                        end_time,
                        origin_batch_size=frontier_origin_batch_size,
                        max_live_branches=(
                            frontier_max_live_branches
                        ),
                    )
                else:
                    dfs_result = inference_turbo_dfs(
                        model,
                        tokens,
                        max_new_tokens,
                        max_score,
                        end_time,
                    )
                decode_seconds += time.perf_counter() - decode_start
                decode_calls += 1
                process_decode_results(subkeys, dfs_result, "threshold DFS")

            if zero_candidate_shape_fallback:
                fallback_groups = defaultdict(list)
                for subkeys in test_id_to_subkeys.values():
                    bk = subkeys[0].split(".")[0]
                    if bk in valid_base_keys:
                        continue
                    fallback_subkey = subkeys[0]
                    output_shape = infer_same_shape_reply(
                        eval_ds.queries[fallback_subkey]
                    )
                    if output_shape is None:
                        print(
                            f"[Rank {rank}] no safe shape fallback for "
                            f"{bk}: demonstrations do not consistently "
                            "share an output-shape offset or constant output "
                            "shape"
                        )
                        continue
                    if int(np.prod(output_shape)) < 2:
                        print(
                            f"[Rank {rank}] no non-monochrome shape "
                            f"fallback for {bk}: inferred output shape "
                            f"{output_shape} has fewer than two cells"
                        )
                        continue
                    data = eval_ds.get(fallback_subkey, formatter)
                    prefix = tokenizer.encode(data["input"])
                    fallback_groups[
                        (output_shape, len(prefix))
                    ].append((fallback_subkey, prefix))

                fallback_deadline = min(
                    end_time,
                    inference_start_time + inference_timeout_seconds,
                )
                for (
                    output_shape,
                    _prompt_length,
                ), requests in fallback_groups.items():
                    if time.time() >= fallback_deadline:
                        print(
                            f"[Rank {rank}] shape fallback skipped at "
                            f"inference deadline for "
                            f"{[item[0] for item in requests]}"
                        )
                        break
                    fallback_subkeys = [item[0] for item in requests]
                    fallback_prefixes = [item[1] for item in requests]
                    print(
                        f"[Rank {rank}] shape-beam fallback "
                        f"{fallback_subkeys} with output shape "
                        f"{output_shape} and width "
                        f"{shape_fallback_beam_width}",
                        flush=True,
                    )
                    fallback_start = time.perf_counter()
                    fallback_result = inference_shape_beam(
                        model,
                        fallback_prefixes,
                        output_shape,
                        fallback_deadline,
                        beam_width=shape_fallback_beam_width,
                    )
                    elapsed = time.perf_counter() - fallback_start
                    fallback_decode_seconds += elapsed
                    fallback_decode_calls += 1
                    decode_seconds += elapsed
                    decode_calls += 1
                    process_decode_results(
                        fallback_subkeys,
                        fallback_result,
                        "shape-beam fallback",
                    )

        if report_output_nll:
            output_nll_started = time.perf_counter()
            test_output_count = len(puzzle_ds.queries[key]["test"])
            test_base_keys = {
                f"{key}_{test_index}"
                for test_index in range(test_output_count)
            }
            candidate_solutions = defaultdict(list)
            for grid_id in known_scores:
                if not isinstance(grid_id, tuple) or len(grid_id) != 2:
                    continue
                base_key, grid_rows = grid_id
                if base_key not in test_base_keys:
                    continue
                candidate_solutions[base_key].append(
                    np.asarray(grid_rows, dtype=np.uint8)
                )
            output_nll_report, output_nll_targets = (
                prepare_output_nll_report(
                    **pair_expansion_options,
                    task_key=key,
                    puzzle_ds_multi=puzzle_ds_multi,
                    test_output_count=test_output_count,
                    candidate_solutions=candidate_solutions,
                    ground_truth_outputs=output_nll_ground_truths.get(key),
                    formatter=formatter,
                    tokenizer=tokenizer,
                    max_model_len=max_seq_length,
                    metadata={
                        "backend": "unsloth",
                        "score_batch_size": int(score_batch_size),
                    },
                )
            )
            if output_nll_targets and time.time() < end_time:
                output_nll_scores = calc_scores(
                    [target.query for target in output_nll_targets],
                    [target.answer for target in output_nll_targets],
                    tokenizer,
                    model,
                    batch_size=score_batch_size,
                )
                apply_output_nll_scores(
                    output_nll_report,
                    output_nll_targets,
                    output_nll_scores,
                )
                output_nll_count = len(output_nll_scores)
            output_nll_path = write_output_nll_report(
                dir_outputs,
                output_nll_report,
            )
            output_nll_seconds = time.perf_counter() - output_nll_started
            print(
                f"[Rank {rank}] output NLL report for {key}: "
                f"scored={output_nll_count}/{len(output_nll_targets)}, "
                f"ground_truth_available="
                f"{key in output_nll_ground_truths}, "
                f"time={output_nll_seconds:.3f}s, path={output_nll_path}",
                flush=True,
            )

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        inference_seconds = time.time() - inference_start_time
        print(
            f"[Rank {rank}] inference stats for puzzle {key}: "
            f"total={inference_seconds:.3f}s, "
            f"setup={inference_setup_seconds:.3f}s, "
            f"decode={decode_seconds:.3f}s/{decode_calls} calls, "
            f"score={scoring_seconds:.3f}s/{score_calls} calls, "
            f"serialize={serialization_seconds:.3f}s/"
            f"{saved_candidate_sets} files, mode={active_dfs_mode}, "
            f"raw_beams={raw_beam_paths}, "
            f"invalid_beams={invalid_beam_paths}, "
            f"monochrome_beams={monochrome_beam_paths}, "
            f"valid_beams={valid_beam_paths}, "
            f"fallback_decode={fallback_decode_seconds:.3f}s/"
            f"{fallback_decode_calls} calls, "
            f"requested_mode={dfs_mode}, "
            f"decode_batch={inference_batch_size}, "
            f"frontier_origins={frontier_origin_batch_size}, "
            f"frontier_max_live={frontier_max_live_branches}, "
            f"frontier_token_limit={frontier_max_prompt_tokens}, "
            f"score_batch={score_batch_size}, "
            f"rescore_views={rescore_views}, "
            f"rescore_schedule={rescore_schedule}, "
            f"output_nll={output_nll_seconds:.3f}s/"
            f"{output_nll_count} grids",
            flush=True,
        )

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")

    return adaptive_service_index
