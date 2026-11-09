"""Task-time DiffusionGemma LoRA lifecycle and vLLM PEFT export.

The REAM training wrappers store each routed expert as one packed gate/up
adapter.  vLLM's universal MoE LoRA loader consumes ordinary PEFT tensors per
expert and projection instead.  This module is the explicit compatibility
boundary between those layouts.

Self-conditioning LoRA is deliberately frozen to an exact zero update.  The
current native vLLM DiffusionGemma model keeps self-conditioning as ordinary
``nn.Linear`` modules, so training those adapters would create a silent
train/serve mismatch.  The frozen BF16 self-conditioning base remains active
during both training and inference.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ORDINARY_LORA_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "router.proj",
)
SELF_CONDITIONING_PATHS = ("gate_proj", "up_proj", "down_proj")
PEFT_FORMAT_VERSION = 1


def _resolve(root: torch.nn.Module, dotted_path: str) -> Any:
    value: Any = root
    for part in dotted_path.split("."):
        value = getattr(value, part)
    return value


def _adapter_pair(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    adapter = getattr(module, "adapter", None)
    adapter_a = getattr(adapter, "A", None)
    adapter_b = getattr(adapter, "B", None)
    if not isinstance(adapter_a, torch.Tensor) or not isinstance(
        adapter_b, torch.Tensor
    ):
        raise TypeError(
            "expected a REAM FP8LoRALinear with adapter.A/adapter.B, got "
            f"{type(module).__name__}"
        )
    return adapter_a, adapter_b


@torch.no_grad()
def freeze_unserved_self_conditioning_lora(model: torch.nn.Module) -> dict[str, int]:
    """Make self-conditioning's LoRA update identically zero and frozen."""
    self_conditioning = model.model.decoder.self_conditioning
    tensors = 0
    values = 0
    for path in SELF_CONDITIONING_PATHS:
        module = _resolve(self_conditioning, path)
        adapter_a, adapter_b = _adapter_pair(module)
        adapter_b.zero_()
        adapter_a.requires_grad_(False)
        adapter_b.requires_grad_(False)
        tensors += 2
        values += adapter_a.numel() + adapter_b.numel()
    return {"tensors": tensors, "values": values}


@torch.no_grad()
def reset_task_adapter(
    model: torch.nn.Module,
    seed: int,
    *,
    expert_parallel_task_seeds: Iterable[int] | None = None,
) -> dict[str, int]:
    """Reinitialize every *trainable* BF16 adapter for a new ARC task.

    Frozen FP8 bases and the zero self-conditioning adapters are untouched.
    The operation is deterministic for a task seed and can therefore be used
    with persistent trainer processes without reloading the base model.
    """
    devices = sorted(
        {
            parameter.device.index
            for parameter in model.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    tensors = 0
    values = 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        expert_parallel_seeds = (
            None
            if expert_parallel_task_seeds is None
            else tuple(int(value) for value in expert_parallel_task_seeds)
        )
        for module in model.modules():
            adapter = getattr(module, "adapter", None)
            adapter_a = getattr(adapter, "A", None)
            adapter_b = getattr(adapter, "B", None)
            if isinstance(adapter_a, torch.nn.Parameter) and adapter_a.requires_grad:
                adapter_a.grad = None
                adapter_b.grad = None
                torch.nn.init.kaiming_uniform_(adapter_a, a=math.sqrt(5))
                adapter_b.zero_()
                tensors += 2
                values += adapter_a.numel() + adapter_b.numel()

            gate_up_a = getattr(module, "gate_up_A", None)
            gate_up_b = getattr(module, "gate_up_B", None)
            down_a = getattr(module, "down_A", None)
            down_b = getattr(module, "down_B", None)
            if bool(getattr(module, "is_expert_parallel", False)):
                if expert_parallel_seeds is None:
                    raise ValueError(
                        "expert-parallel DiffusionGemma reset requires every "
                        "task lane's seed"
                    )
                reset = module.reset_adapter_slots(expert_parallel_seeds)
                tensors += int(reset["tensors"])
                values += int(reset["values"])
            elif isinstance(gate_up_a, torch.nn.Parameter) and gate_up_a.requires_grad:
                gate_up_a.grad = None
                gate_up_b.grad = None
                down_a.grad = None
                down_b.grad = None
                torch.nn.init.kaiming_uniform_(gate_up_a, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(down_a, a=math.sqrt(5))
                gate_up_b.zero_()
                down_b.zero_()
                tensors += 4
                values += sum(
                    tensor.numel()
                    for tensor in (gate_up_a, gate_up_b, down_a, down_b)
                )
    if tensors == 0:
        raise RuntimeError("no trainable DiffusionGemma task adapters were reset")
    return {"tensors": tensors, "values": values}


def _cpu_bf16_clone(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"DiffusionGemma LoRA tensor must be BF16, got {tensor.dtype}")
    return tensor.detach().to(device="cpu").contiguous().clone()


def _ordinary_peft_prefix(layer_index: int, path: str) -> str:
    # The native DiffusionGemma mapper turns model.decoder.* into model.*.
    return f"base_model.model.model.decoder.layers.{layer_index}.{path}"


def _expert_peft_prefix(
    layer_index: int, expert_index: int, projection: str
) -> str:
    # Its mapper additionally turns .experts. into .moe.experts.; vLLM then
    # builds the universal 2-D FusedMoE LoRA buffers from these expert rows.
    return (
        "base_model.model.model.decoder.layers."
        f"{layer_index}.experts.{expert_index}.{projection}"
    )


def build_vllm_peft_state(
    model: torch.nn.Module,
    *,
    expert_state: dict[tuple[int, str], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return a universal 2-D PEFT state accepted by native vLLM MoE LoRA."""
    state: dict[str, torch.Tensor] = {}
    decoder = model.model.decoder
    for layer_index, layer in enumerate(decoder.layers):
        for path in ORDINARY_LORA_PATHS:
            try:
                module = _resolve(layer, path)
            except AttributeError:
                continue
            if module is None or not hasattr(module, "adapter"):
                continue
            adapter_a, adapter_b = _adapter_pair(module)
            prefix = _ordinary_peft_prefix(layer_index, path)
            state[f"{prefix}.lora_A.weight"] = _cpu_bf16_clone(adapter_a)
            state[f"{prefix}.lora_B.weight"] = _cpu_bf16_clone(adapter_b)

        experts = layer.experts
        required = ("gate_up_A", "gate_up_B", "down_A", "down_B")
        if not all(hasattr(experts, name) for name in required):
            raise TypeError(
                f"decoder layer {layer_index} does not contain packed REAM experts"
            )
        intermediate_size = int(experts.intermediate_dim)
        expert_parallel = bool(getattr(experts, "is_expert_parallel", False))
        if expert_parallel:
            if expert_state is None:
                raise ValueError(
                    "expert-parallel PEFT export requires gathered expert state"
                )
            gate_up_a = expert_state[(layer_index, "gate_up_A")]
            gate_up_b = expert_state[(layer_index, "gate_up_B")]
            down_a = expert_state[(layer_index, "down_A")]
            down_b = expert_state[(layer_index, "down_B")]
            num_experts = int(experts.global_num_experts)
        else:
            gate_up_a = experts.gate_up_A
            gate_up_b = experts.gate_up_B
            down_a = experts.down_A
            down_b = experts.down_B
            num_experts = int(experts.num_experts)
        if gate_up_b.shape[1] != 2 * intermediate_size:
            raise RuntimeError(
                f"invalid packed gate/up shape at decoder layer {layer_index}: "
                f"{tuple(gate_up_b.shape)}"
            )
        for expert_index in range(num_experts):
            # REAM shares A between gate and up. PEFT's universal 2-D format
            # needs one independently owned tensor per logical projection.
            shared_a = gate_up_a[expert_index]
            for projection, adapter_b in (
                ("gate_proj", gate_up_b[expert_index, :intermediate_size]),
                ("up_proj", gate_up_b[expert_index, intermediate_size:]),
            ):
                prefix = _expert_peft_prefix(
                    layer_index, expert_index, projection
                )
                state[f"{prefix}.lora_A.weight"] = _cpu_bf16_clone(shared_a)
                state[f"{prefix}.lora_B.weight"] = _cpu_bf16_clone(adapter_b)
            prefix = _expert_peft_prefix(layer_index, expert_index, "down_proj")
            state[f"{prefix}.lora_A.weight"] = _cpu_bf16_clone(
                down_a[expert_index]
            )
            state[f"{prefix}.lora_B.weight"] = _cpu_bf16_clone(
                down_b[expert_index]
            )

    if not state:
        raise RuntimeError("DiffusionGemma PEFT export produced no tensors")
    return state


def _trainable_scale(model: torch.nn.Module) -> tuple[int, float]:
    for layer in model.model.decoder.layers:
        for path in ORDINARY_LORA_PATHS:
            try:
                module = _resolve(layer, path)
            except AttributeError:
                continue
            adapter = getattr(module, "adapter", None)
            adapter_a = getattr(adapter, "A", None)
            if isinstance(adapter_a, torch.Tensor) and adapter_a.requires_grad:
                rank = int(adapter_a.shape[0])
                scale = float(adapter.scale)
                return rank, scale
    raise RuntimeError("cannot infer DiffusionGemma task adapter rank/scale")


@torch.no_grad()
def gather_expert_parallel_slot_state(
    model: torch.nn.Module,
    *,
    slot: int,
    process_group: Any,
) -> dict[tuple[int, str], torch.Tensor]:
    """Gather one task slot's expert shards for ordinary vLLM PEFT export.

    All EP ranks must call this function in the same layer/field order. Each
    owner contributes every task slot, allowing rank ``slot`` to reconstruct
    its complete expert adapter without moving frozen expert weights.
    """

    if not torch.distributed.is_initialized():
        raise RuntimeError("expert-parallel export requires torch.distributed")
    world_size = int(torch.distributed.get_world_size(group=process_group))
    slot = int(slot)
    if not 0 <= slot < world_size:
        raise ValueError(f"invalid expert-parallel slot {slot}/{world_size}")
    state: dict[tuple[int, str], torch.Tensor] = {}
    for layer_index, layer in enumerate(model.model.decoder.layers):
        experts = layer.experts
        if not bool(getattr(experts, "is_expert_parallel", False)):
            raise TypeError(
                f"decoder layer {layer_index} is not expert-parallel"
            )
        partition = experts.partition
        for field in ("gate_up_A", "gate_up_B", "down_A", "down_B"):
            parameter = getattr(experts, field).detach()
            padded_shape = (
                world_size,
                int(partition.max_local_experts),
                *parameter.shape[2:],
            )
            padded = parameter.new_zeros(padded_shape)
            padded[:, : int(partition.local_experts)].copy_(parameter)
            gathered_flat = parameter.new_empty(
                (world_size * padded_shape[0], *padded_shape[1:])
            )
            torch.distributed.all_gather_into_tensor(
                gathered_flat,
                padded,
                group=process_group,
            )
            gathered = gathered_flat.view(world_size, *padded_shape)
            shards = [
                gathered[owner, slot, : partition.stops[owner] - partition.starts[owner]]
                for owner in range(world_size)
            ]
            state[(layer_index, field)] = (
                torch.cat(shards, dim=0).to(device="cpu").contiguous().clone()
            )
            del gathered, gathered_flat, padded
    return state


def install_expert_parallel_gradient_clipping(
    trainer: Any,
    model: torch.nn.Module,
    *,
    local_slot: int,
    process_group: Any,
) -> None:
    """Clip every task LoRA independently across its distributed shards."""

    local_slot = int(local_slot)
    experts = [
        module
        for module in model.modules()
        if bool(getattr(module, "is_expert_parallel", False))
    ]
    if not experts:
        raise RuntimeError("cannot install EP clipping without EP experts")
    world_size = int(torch.distributed.get_world_size(group=process_group))
    expert_parameter_ids = {
        id(parameter)
        for module in experts
        for parameter in (
            module.gate_up_A,
            module.gate_up_B,
            module.down_A,
            module.down_B,
        )
    }
    ordinary = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in expert_parameter_ids
    ]
    original = trainer.accelerator.clip_grad_norm_

    def clip_grad_norm_ep(parameters, max_norm, norm_type=2):
        del parameters
        if float(norm_type) != 2.0:
            return original(model.parameters(), max_norm, norm_type)
        device = experts[0].gate_up_A.device
        squares = torch.zeros(world_size, dtype=torch.float32, device=device)
        for module in experts:
            for parameter in (
                module.gate_up_A,
                module.gate_up_B,
                module.down_A,
                module.down_B,
            ):
                if parameter.grad is None:
                    continue
                gradients = parameter.grad.detach().reshape(world_size, -1)
                norms = torch.linalg.vector_norm(
                    gradients, ord=2, dim=1, dtype=torch.float32
                )
                squares.add_(norms.square())
        for parameter in ordinary:
            if parameter.grad is not None:
                norm = torch.linalg.vector_norm(
                    parameter.grad.detach().reshape(-1),
                    ord=2,
                    dtype=torch.float32,
                )
                squares[local_slot].add_(norm.square())
        torch.distributed.all_reduce(
            squares,
            op=torch.distributed.ReduceOp.SUM,
            group=process_group,
        )
        norms = squares.sqrt()
        scales = (float(max_norm) / (norms + 1e-6)).clamp(max=1.0)
        for module in experts:
            for parameter in (
                module.gate_up_A,
                module.gate_up_B,
                module.down_A,
                module.down_B,
            ):
                if parameter.grad is not None:
                    view = (world_size,) + (1,) * (parameter.grad.ndim - 1)
                    parameter.grad.mul_(scales.to(parameter.grad.dtype).view(view))
        local_scale = scales[local_slot]
        for parameter in ordinary:
            if parameter.grad is not None:
                parameter.grad.mul_(local_scale.to(parameter.grad.dtype))
        return norms[local_slot].to(dtype=torch.float32)

    trainer.accelerator.clip_grad_norm_ = clip_grad_norm_ep
    trainer._arc_expert_parallel_clip_grad_norm = True


def export_vllm_peft_adapter(
    model: torch.nn.Module,
    output_dir: str | os.PathLike[str],
    *,
    metadata: dict[str, Any] | None = None,
    expert_state: dict[tuple[int, str], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Atomically export an adapter and validate its on-disk tensor contract."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rank, scale = _trainable_scale(model)
    alpha = scale * rank
    if not float(alpha).is_integer():
        raise ValueError(
            f"PEFT/vLLM requires an integral lora_alpha; inferred {alpha}"
        )
    state = build_vllm_peft_state(model, expert_state=expert_state)
    weights_path = destination / "adapter_model.safetensors"
    temporary_weights = destination / f".adapter_model.{os.getpid()}.tmp"
    tensor_metadata = {
        "format": "arc-diffusion-gemma-vllm-peft-v1",
        "self_conditioning_lora": "false",
    }
    if metadata:
        tensor_metadata.update({str(key): str(value) for key, value in metadata.items()})
    save_file(state, str(temporary_weights), metadata=tensor_metadata)
    os.replace(temporary_weights, weights_path)

    config = {
        "base_model_name_or_path": "",
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": int(alpha),
        "peft_type": "LORA",
        "r": rank,
        "target_modules": sorted(
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "proj",
            }
        ),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
        "arc_diffusion_gemma_peft_format": PEFT_FORMAT_VERSION,
        "arc_self_conditioning_lora": False,
    }
    config_path = destination / "adapter_config.json"
    temporary_config = destination / f".adapter_config.{os.getpid()}.tmp"
    temporary_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_config, config_path)

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        disk_keys = set(handle.keys())
        if disk_keys != set(state):
            raise RuntimeError("DiffusionGemma PEFT adapter changed during save")
        tensor_count = len(disk_keys)
    return {
        "path": str(destination),
        "rank": rank,
        "alpha": int(alpha),
        "scale": scale,
        "tensors": tensor_count,
        "values": sum(tensor.numel() for tensor in state.values()),
        "bytes": weights_path.stat().st_size,
        "self_conditioning_lora": False,
    }


def iter_trainable_adapter_parameters(
    model: torch.nn.Module,
) -> Iterable[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("DiffusionGemma has no trainable task adapter parameters")
    for parameter in parameters:
        if parameter.dtype != torch.bfloat16:
            raise TypeError(
                f"task adapter parameters must be BF16, got {parameter.dtype}"
            )
    return parameters


__all__ = [
    "ORDINARY_LORA_PATHS",
    "build_vllm_peft_state",
    "export_vllm_peft_adapter",
    "freeze_unserved_self_conditioning_lora",
    "gather_expert_parallel_slot_state",
    "install_expert_parallel_gradient_clipping",
    "iter_trainable_adapter_parameters",
    "reset_task_adapter",
]
