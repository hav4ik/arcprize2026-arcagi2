"""Opt-in DeepSpeed runtime for long-context task-time LoRA training.

The normal ARC worker deliberately does not depend on DeepSpeed.  This module
keeps the experimental ZeRO-3 parameter-offload policy isolated so tasks at or
below the configured token threshold retain the existing Unsloth path.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch


DEFAULT_LONG_CONTEXT_TOKEN_THRESHOLD = 8192
DEFAULT_PARAM_PERSISTENCE_THRESHOLD = 2_000_000
DEFAULT_MODEL_PERSISTENCE_THRESHOLD = 300_000_000
DEFAULT_MAX_LIVE_PARAMETERS = 100_000_000
DEFAULT_PREFETCH_BUCKET_SIZE = 25_000_000

_DEEPSPEED_RUNTIME_ENV = (
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_MIXED_PRECISION",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
)

_UNSLOTH_CHECKPOINT_NAMES = frozenset(
    {
        "unsloth_checkpoint",
        "unsloth_gradient_checkpoint",
        "unsloth_offloaded_gradient_checkpoint",
    }
)


def _checkpoint_callable_name(function: Any) -> str:
    """Return a checkpoint callable's underlying name through partials."""
    seen = set()
    while isinstance(function, partial) and id(function) not in seen:
        seen.add(id(function))
        function = function.func
    return getattr(function, "__name__", type(function).__name__)


@contextmanager
def deepspeed_standard_gradient_checkpointing_environment(
    model: torch.nn.Module,
    *,
    dtype: torch.dtype = torch.bfloat16,
):
    """Temporarily force genuine PyTorch reentrant checkpointing.

    A worker may load PEFT with ``use_gradient_checkpointing="unsloth"``
    before it knows whether the materialized task crosses the DeepSpeed token
    gate.  Unsloth's smart mode patches both PyTorch and Transformers globals,
    and merely changing the later per-task mode marker to ``"standard"`` does
    not undo that patch.  Without this scope a ZeRO-3 parameter-offload task
    also CPU-offloads checkpoint inputs, invalidating both its memory policy
    and timing comparison.

    This scope releases the smart-checkpoint buffers, installs the pristine
    ``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=True`` on every
    checkpoint-capable module, and restores the exact prior globals and
    module-local callables on exit.  Reentrant PyTorch checkpointing is
    intentional: the Qwen3.5 FLA backward path failed under the non-reentrant
    variant in the live partial-checkpoint screen.
    """
    if model is None or not hasattr(model, "modules"):
        raise TypeError("model must be a torch.nn.Module-like object")

    import torch.utils.checkpoint as torch_checkpoint
    import transformers.modeling_utils as transformers_modeling_utils
    from unsloth_zoo import gradient_checkpointing as unsloth_checkpointing

    smart_checkpoint = getattr(
        unsloth_checkpointing,
        "unsloth_checkpoint",
        None,
    )
    smart_checkpoint_function = getattr(
        unsloth_checkpointing,
        "UnslothCheckpointFunction",
        None,
    )
    previous_torch_checkpoint = torch_checkpoint.checkpoint
    previous_transformers_checkpoint = transformers_modeling_utils.checkpoint
    previous_checkpoint_function = torch_checkpoint.CheckpointFunction
    missing = object()
    previous_old_checkpoint = getattr(
        torch_checkpoint,
        "_old_checkpoint",
        missing,
    )
    previous_old_checkpoint_function = getattr(
        torch_checkpoint,
        "_old_CheckpointFunction",
        missing,
    )

    pristine_checkpoint = getattr(
        torch_checkpoint,
        "_unsloth_pristine_checkpoint",
        None,
    )
    if pristine_checkpoint is None:
        candidate = getattr(torch_checkpoint, "_old_checkpoint", None)
        if (
            callable(candidate)
            and _checkpoint_callable_name(candidate)
            not in _UNSLOTH_CHECKPOINT_NAMES
        ):
            pristine_checkpoint = candidate
    if pristine_checkpoint is None and (
        _checkpoint_callable_name(previous_torch_checkpoint)
        not in _UNSLOTH_CHECKPOINT_NAMES
    ):
        pristine_checkpoint = previous_torch_checkpoint
    if not callable(pristine_checkpoint):
        raise RuntimeError(
            "Cannot locate the pristine torch checkpoint callable while "
            "entering the DeepSpeed standard-checkpoint scope"
        )

    unexpected_shims = {
        _checkpoint_callable_name(previous_torch_checkpoint),
        _checkpoint_callable_name(previous_transformers_checkpoint),
    } & (_UNSLOTH_CHECKPOINT_NAMES - {"unsloth_checkpoint"})
    if unexpected_shims:
        raise RuntimeError(
            "DeepSpeed standard-checkpoint scope found an unsupported "
            "nested Unsloth checkpoint shim: "
            + ", ".join(sorted(unexpected_shims))
        )

    smart_was_active = any(
        (
            previous_torch_checkpoint is smart_checkpoint,
            previous_transformers_checkpoint is smart_checkpoint,
            previous_checkpoint_function is smart_checkpoint_function,
            _checkpoint_callable_name(previous_torch_checkpoint)
            == "unsloth_checkpoint",
            _checkpoint_callable_name(previous_transformers_checkpoint)
            == "unsloth_checkpoint",
            getattr(previous_checkpoint_function, "__name__", "")
            == "UnslothCheckpointFunction",
        )
    )

    module_checkpoint_functions = []
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing") or hasattr(
            module,
            "_gradient_checkpointing_func",
        ):
            module_checkpoint_functions.append(
                (
                    module,
                    getattr(module, "_gradient_checkpointing_func", missing),
                )
            )

    standard_checkpoint = partial(
        pristine_checkpoint,
        use_reentrant=True,
    )
    try:
        if smart_was_active:
            unsloth_checkpointing.unpatch_unsloth_smart_gradient_checkpointing()

        # Enforce the pristine function even if an earlier nested patch left
        # `_old_checkpoint` pointing at another wrapper.
        original_checkpoint_function = getattr(
            torch_checkpoint,
            "_old_CheckpointFunction",
            torch_checkpoint.CheckpointFunction,
        )
        if getattr(original_checkpoint_function, "__name__", "") != (
            "UnslothCheckpointFunction"
        ):
            torch_checkpoint.CheckpointFunction = original_checkpoint_function
        torch_checkpoint.checkpoint = pristine_checkpoint
        transformers_modeling_utils.checkpoint = pristine_checkpoint
        for module, _ in module_checkpoint_functions:
            module._gradient_checkpointing_func = standard_checkpoint

        if (
            _checkpoint_callable_name(torch_checkpoint.checkpoint)
            in _UNSLOTH_CHECKPOINT_NAMES
            or _checkpoint_callable_name(
                transformers_modeling_utils.checkpoint
            )
            in _UNSLOTH_CHECKPOINT_NAMES
            or getattr(torch_checkpoint.CheckpointFunction, "__name__", "")
            == "UnslothCheckpointFunction"
        ):
            raise RuntimeError(
                "Failed to remove Unsloth smart checkpointing from the "
                "DeepSpeed task"
            )
        yield standard_checkpoint
    finally:
        try:
            if smart_was_active:
                # Recreate Unsloth's released smart buffers before restoring
                # the exact pre-scope function objects below.
                unsloth_checkpointing.patch_unsloth_smart_gradient_checkpointing(
                    dtype=dtype,
                )
        finally:
            # Restore function globals and their saved-pristine slots even if
            # Unsloth buffer reinitialization itself fails.  This leaves a
            # mixed-task worker in the same callable state it entered with,
            # while still surfacing the reinitialization exception.
            torch_checkpoint.CheckpointFunction = previous_checkpoint_function
            torch_checkpoint.checkpoint = previous_torch_checkpoint
            transformers_modeling_utils.checkpoint = (
                previous_transformers_checkpoint
            )
            if previous_old_checkpoint is missing:
                try:
                    delattr(torch_checkpoint, "_old_checkpoint")
                except AttributeError:
                    pass
            else:
                torch_checkpoint._old_checkpoint = previous_old_checkpoint
            if previous_old_checkpoint_function is missing:
                try:
                    delattr(torch_checkpoint, "_old_CheckpointFunction")
                except AttributeError:
                    pass
            else:
                torch_checkpoint._old_CheckpointFunction = (
                    previous_old_checkpoint_function
                )
            for module, previous_function in module_checkpoint_functions:
                if previous_function is missing:
                    try:
                        delattr(module, "_gradient_checkpointing_func")
                    except AttributeError:
                        pass
                else:
                    module._gradient_checkpointing_func = previous_function


def use_long_context_deepspeed(
    enabled: bool,
    max_train_tokens: int,
    token_threshold: int = DEFAULT_LONG_CONTEXT_TOKEN_THRESHOLD,
) -> bool:
    """Return whether this task crosses the strict DeepSpeed token gate."""
    max_train_tokens = int(max_train_tokens)
    token_threshold = int(token_threshold)
    if max_train_tokens < 0:
        raise ValueError("max_train_tokens must be non-negative")
    if token_threshold < 1:
        raise ValueError("DeepSpeed token threshold must be positive")
    return bool(enabled) and max_train_tokens > token_threshold


def build_zero3_param_offload_config(
    *,
    offload_optimizer: bool = False,
    param_persistence_threshold: int = DEFAULT_PARAM_PERSISTENCE_THRESHOLD,
    model_persistence_threshold: int = DEFAULT_MODEL_PERSISTENCE_THRESHOLD,
    max_live_parameters: int = DEFAULT_MAX_LIVE_PARAMETERS,
    prefetch_bucket_size: int = DEFAULT_PREFETCH_BUCKET_SIZE,
) -> dict[str, Any]:
    """Build the single-GPU ZeRO-3 configuration used by the experiment.

    Persistence thresholds are expressed in parameter elements, following
    DeepSpeed.  The defaults keep rank-128 LoRA tensors resident while the
    much larger frozen Qwen projection weights remain eligible for CPU
    offload.  This is a size-based policy, not a claim that ZeRO supports an
    explicit list of selected layers.
    """
    values = {
        "param_persistence_threshold": int(param_persistence_threshold),
        "model_persistence_threshold": int(model_persistence_threshold),
        "max_live_parameters": int(max_live_parameters),
        "prefetch_bucket_size": int(prefetch_bucket_size),
    }
    if any(value < 1 for value in values.values()):
        raise ValueError(
            "DeepSpeed persistence, live-parameter, and prefetch values "
            "must be positive"
        )

    zero_config: dict[str, Any] = {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": False,
        "contiguous_gradients": True,
        "stage3_param_persistence_threshold": values[
            "param_persistence_threshold"
        ],
        "stage3_model_persistence_threshold": values[
            "model_persistence_threshold"
        ],
        "stage3_max_live_parameters": values["max_live_parameters"],
        "stage3_prefetch_bucket_size": values["prefetch_bucket_size"],
        # Adapter extraction gathers only PEFT tensors.  Avoid an accidental
        # full 4B-model consolidation through Trainer's ordinary save path.
        "stage3_gather_16bit_weights_on_model_save": False,
    }
    if offload_optimizer:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return {
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_allow_untested_optimizer": True,
        "zero_optimization": zero_config,
    }


def validate_long_context_deepspeed_options(
    *,
    enabled: bool,
    optimizer: str,
    distributed: bool,
    keep_training_state: bool,
    task_rl: bool,
    train_only: bool,
    routed_inference: bool,
    torch_compile_training: bool,
    selective_activation_recompute: bool,
    fp8_frozen_linear: bool,
) -> None:
    """Reject combinations not covered by the first live benchmark."""
    if not enabled:
        return
    conflicts = []
    if optimizer != "adamw_torch":
        conflicts.append("--optimizer adamw_torch")
    if distributed:
        conflicts.append("independent one-GPU task workers")
    if keep_training_state:
        conflicts.append("fresh per-task adapters")
    if task_rl:
        conflicts.append("task SFT without --task-rl")
    if not train_only and not routed_inference:
        conflicts.append("--train-only or routed vLLM/SGLang inference")
    if torch_compile_training:
        conflicts.append("--no-torch-compile-training")
    if selective_activation_recompute:
        conflicts.append("no selective activation-recompute flags")
    if fp8_frozen_linear:
        conflicts.append("--no-fp8-frozen-linear")
    if conflicts:
        raise ValueError(
            "Long-context DeepSpeed is an experimental ZeRO-3 parameter-"
            "offload path and currently requires: " + ", ".join(conflicts)
        )


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def reset_accelerate_deepspeed_state() -> None:
    """Reset process-global Trainer state between normal and ZeRO engines."""
    try:
        from transformers.integrations.deepspeed import (
            unset_hf_deepspeed_config,
        )

        unset_hf_deepspeed_config()
    except (ImportError, AttributeError):
        pass

    try:
        from accelerate.state import AcceleratorState, GradientState

        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
    except (ImportError, AttributeError):
        pass


@contextmanager
def single_process_deepspeed_environment():
    """Expose one CUDA-visible worker as an isolated DeepSpeed world of one."""
    previous = {name: os.environ.get(name) for name in _DEEPSPEED_RUNTIME_ENV}
    previous_cuda_autocast_dtype = torch.get_autocast_dtype("cuda")
    reset_accelerate_deepspeed_state()
    os.environ.update(
        {
            "ACCELERATE_USE_DEEPSPEED": "true",
            # Trainer creates a fresh Accelerator inside this context.  Make
            # its autocast contract explicit: relying on Accelerate's process
            # default can select FP16 even though TrainingArguments and the
            # DeepSpeed JSON both request BF16, which leaves Unsloth's fused
            # LoRA addmm with a Half base result and BFloat16 adapter tensor.
            "ACCELERATE_MIXED_PRECISION": "bf16",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_available_local_port()),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            # Accelerate requires these for Ada workstation-class devices;
            # collectives are world-size one, so neither disables useful ARC
            # inter-GPU communication.
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        }
    )
    # DeepSpeed's native BF16 mode casts parameters correctly, but its
    # world-one Trainer integration can still enter CUDA autocast without an
    # explicit dtype. PyTorch then defaults that context to FP16, producing a
    # Half base projection beside Unsloth's BF16 LoRA branch. Keep this change
    # scoped to the routed task and restore the process default afterwards.
    torch.set_autocast_dtype("cuda", torch.bfloat16)
    try:
        yield int(os.environ["MASTER_PORT"])
    finally:
        # The engine should already be destroyed by the caller.  Tear down its
        # one-rank process group before restoring the ordinary Trainer state.
        if torch.distributed.is_initialized():
            try:
                import deepspeed.comm as ds_comm

                ds_comm.destroy_process_group()
            except (ImportError, AttributeError, RuntimeError):
                torch.distributed.destroy_process_group()
        reset_accelerate_deepspeed_state()
        torch.set_autocast_dtype("cuda", previous_cuda_autocast_dtype)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def is_zero3_parameter(parameter: torch.nn.Parameter) -> bool:
    """Return whether DeepSpeed attached ZeRO metadata to a parameter."""
    return hasattr(parameter, "ds_id") or hasattr(parameter, "ds_status")


def is_adapter_export_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    """Select adapter and tiny ARC embedding/head tensors for PEFT export."""
    return bool(parameter.requires_grad) or any(
        marker in str(name) for marker in ("embed_tokens", "lm_head")
    )


def gather_zero3_peft_state_dict(
    model: torch.nn.Module,
    *,
    adapter_name: str = "default",
) -> Mapping[str, torch.Tensor]:
    """Materialize only PEFT-relevant ZeRO-3 tensors into immutable CPU RAM."""
    from peft import get_peft_model_state_dict

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if is_adapter_export_parameter(name, parameter)
    ]
    unique_parameters = list(
        {id(parameter): parameter for _, parameter in named_parameters}.values()
    )
    zero_parameters = [
        parameter
        for parameter in unique_parameters
        if is_zero3_parameter(parameter)
    ]
    if not zero_parameters:
        return {
            name: tensor.detach().to(device="cpu", copy=True).contiguous()
            for name, tensor in get_peft_model_state_dict(
                model,
                adapter_name=adapter_name,
            ).items()
        }

    import deepspeed

    with deepspeed.zero.GatheredParameters(
        zero_parameters,
        modifier_rank=None,
    ):
        selected_state = {
            name: parameter.detach().to(
                device="cpu",
                copy=True,
            ).contiguous()
            for name, parameter in named_parameters
        }
    return get_peft_model_state_dict(
        model,
        state_dict=selected_state,
        adapter_name=adapter_name,
    )


def destroy_deepspeed_engine(trainer: Any) -> None:
    """Release ZeRO hooks and offload buffers without hiding real failures."""
    engine = getattr(trainer, "model_wrapped", None)
    destroy = getattr(engine, "destroy", None)
    if callable(destroy):
        destroy()
