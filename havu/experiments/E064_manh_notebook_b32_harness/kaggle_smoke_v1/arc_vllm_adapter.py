"""Export trained ARC PEFT adapters in native vLLM LoRA format.

The ARC model trains ``embed_tokens`` and, with current Unsloth, ``lm_head``
as full PEFT ``modules_to_save`` modules. vLLM intentionally rejects that PEFT
feature, but the checkpoint has only 16 vocabulary rows. Their full deltas can
therefore be represented exactly as embedding/linear LoRAs inside the existing
ARC rank (normally 128 or 256).

This module deliberately has no vLLM or Unsloth imports.  Trainer processes can
use it without initializing an inference runtime, and unit tests can exercise
the on-disk protocol on CPU.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from run_config import is_pissa_init, normalize_lora_init, normalize_pissa_base_mode


ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_MANIFEST_FILE = "arc_adapter_manifest.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

_LORA_WEIGHT_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A",
    ".lora_embedding_B",
)
_EMBEDDING_WEIGHT_SUFFIXES = (
    ".embed_tokens.weight",
    ".embed_tokens.modules_to_save.default.weight",
)
_LM_HEAD_WEIGHT_SUFFIXES = (
    ".lm_head.weight",
    ".lm_head.modules_to_save.default.weight",
)
_IGNORABLE_BASE_EMBEDDING_SUFFIXES = (
    ".embed_tokens.base_layer.weight",
    ".lm_head.base_layer.weight",
)
_NORA_EPS = 1e-6


def make_adapter_name(run_id: str, adapter_id: int, task_key: str) -> str:
    """Return a stable, vLLM-cache-safe name for one adapter generation."""
    if int(adapter_id) < 1:
        raise ValueError(f"adapter_id must be positive, got {adapter_id}")
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(run_id)).strip("-")
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(task_key)).strip("-")
    if not safe_run or not safe_task:
        raise ValueError("run_id and task_key must contain a safe character")
    return f"arc-{safe_run}-{int(adapter_id)}-{safe_task}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, tuple, list)):
        items = [_jsonable(item) for item in value]
        return sorted(items) if isinstance(value, set) else items
    enum_value = getattr(value, "value", None)
    if enum_value is not None and not isinstance(value, (str, bytes)):
        return _jsonable(enum_value)
    if isinstance(value, Path):
        return str(value)
    return value


def peft_config_dict(config: Any) -> dict[str, Any]:
    """Convert a PEFT config object or mapping to a JSON-safe dictionary."""
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, Mapping):
        raise TypeError(
            "PEFT config must be a mapping or expose to_dict(), got "
            f"{type(config).__name__}"
        )
    return _jsonable(dict(config))


def clone_peft_state_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Take an immutable CPU snapshot suitable for later delta calculation."""
    return {
        key: value.detach().to(device="cpu", copy=True).contiguous()
        for key, value in state_dict.items()
    }


def clone_embedding_state_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Snapshot full embedding/head weights needed as export delta bases.

    The public name is retained for callers from the older embedding-only
    protocol. Current Unsloth redirects both tiny ARC vocabulary matrices to
    ``modules_to_save``, so both immutable bases must survive task training.
    """
    keys = (
        _find_embedding_weight_key(state_dict),
        _find_lm_head_weight_key(state_dict),
    )
    return {
        key: state_dict[key].detach().to(device="cpu", copy=True).contiguous()
        for key in keys
        if key is not None
    }


def _is_lora_weight(key: str) -> bool:
    return any(marker in key for marker in _LORA_WEIGHT_MARKERS)


def _find_full_weight_key(
    state_dict: Mapping[str, torch.Tensor],
    suffixes: tuple[str, ...],
    *,
    module_name: str,
) -> str | None:
    keys = [
        key
        for key in state_dict
        if key.endswith(suffixes)
        and ".base_layer." not in key
        and ".original_module." not in key
    ]
    if not keys:
        return None
    if len(keys) != 1:
        raise ValueError(
            f"Expected exactly one trainable {module_name} full weight, got "
            f"{keys}"
        )
    return keys[0]


def _find_embedding_weight_key(
    state_dict: Mapping[str, torch.Tensor],
) -> str | None:
    return _find_full_weight_key(
        state_dict,
        _EMBEDDING_WEIGHT_SUFFIXES,
        module_name="embed_tokens",
    )


def _find_lm_head_weight_key(
    state_dict: Mapping[str, torch.Tensor],
) -> str | None:
    return _find_full_weight_key(
        state_dict,
        _LM_HEAD_WEIGHT_SUFFIXES,
        module_name="lm_head",
    )


def _full_module_prefix(
    weight_key: str,
    suffixes: tuple[str, ...],
    module_name: str,
) -> str:
    for suffix in suffixes:
        if weight_key.endswith(suffix):
            return weight_key[: -len(suffix)] + f".{module_name}"
    raise ValueError(f"Not a {module_name} weight key: {weight_key}")


def _embedding_module_prefix(weight_key: str) -> str:
    return _full_module_prefix(
        weight_key,
        _EMBEDDING_WEIGHT_SUFFIXES,
        "embed_tokens",
    )


def _lm_head_module_prefix(weight_key: str) -> str:
    return _full_module_prefix(
        weight_key,
        _LM_HEAD_WEIGHT_SUFFIXES,
        "lm_head",
    )


def _module_leaves(names: Any) -> set[str]:
    if names is None:
        return set()
    if isinstance(names, str):
        names = [names]
    return {str(name).rsplit(".", 1)[-1] for name in names}


def _rslora_scaling(config: Mapping[str, Any]) -> float:
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"Invalid LoRA rank/alpha: r={rank}, alpha={alpha}")
    return alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)


def _source_lora_method(config: Mapping[str, Any]) -> str:
    """Describe PEFT's training-time parameterization before export."""
    use_nora = config.get("use_nora", False)
    if use_nora is True:
        return "nora"
    if use_nora == "init":
        return "nora-init"
    if use_nora not in {False, None}:
        raise ValueError(
            "Unsupported PEFT use_nora value during vLLM export: "
            f"{use_nora!r}"
        )
    if config.get("init_lora_weights") == "bimi":
        return "bimi"
    return "rslora" if config.get("use_rslora") else "lora"


def _normalize_nora_lora_a(weight: torch.Tensor) -> torch.Tensor:
    """Materialize NoRA's exact training-time A matrix for plain LoRA runtimes."""
    compute_weight = (
        weight.float()
        if weight.dtype in (torch.float16, torch.bfloat16)
        else weight
    )
    denominator = torch.linalg.vector_norm(
        compute_weight,
        dim=0,
        keepdim=True,
    ).clamp_min(_NORA_EPS)
    return (compute_weight / denominator).to(weight.dtype).contiguous()


def _nonfinite_count(tensor: torch.Tensor) -> int:
    if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
        return 0
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return 0
    return int((~finite).sum().item())


def _raise_if_nonfinite(name: str, nonfinite_count: int, numel: int) -> None:
    if nonfinite_count:
        raise ValueError(
            f"Adapter tensor {name!r} contains {nonfinite_count}/"
            f"{numel} non-finite value(s)"
        )


def _require_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    """Reject a corrupted adapter tensor before publishing it to vLLM."""
    _raise_if_nonfinite(name, _nonfinite_count(tensor), tensor.numel())


def factor_embedding_delta(
    delta: torch.Tensor,
    *,
    rank: int,
    scaling: float,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor ``[vocab, hidden]`` delta into vLLM embedding A/B tensors.

    vLLM consumes ``A[rank, vocab]`` and ``B[hidden, rank]`` and merges the
    configured LoRA scaling into B.  With an identity prefix in A this is an
    exact factorization whenever ``rank >= vocab``.
    """
    if delta.ndim != 2:
        raise ValueError(
            f"Embedding delta must have shape [vocab, hidden], got {delta.shape}"
        )
    vocab_size, hidden_size = map(int, delta.shape)
    if rank < vocab_size:
        raise ValueError(
            f"LoRA rank {rank} cannot exactly encode {vocab_size} vocabulary rows"
        )
    if not math.isfinite(scaling) or scaling == 0:
        raise ValueError(f"Invalid LoRA scaling: {scaling}")
    output_dtype = dtype or delta.dtype
    lora_a = torch.zeros((rank, vocab_size), dtype=output_dtype)
    lora_b = torch.zeros((hidden_size, rank), dtype=output_dtype)
    diagonal = torch.arange(vocab_size)
    lora_a[diagonal, diagonal] = 1
    lora_b[:, :vocab_size] = (delta.T / scaling).to(output_dtype)
    return lora_a.contiguous(), lora_b.contiguous()


def reconstruct_embedding_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Reconstruct a ``[vocab, hidden]`` embedding delta in FP32."""
    return (
        lora_a.float().T.matmul(lora_b.float().T) * float(scaling)
    )


def factor_linear_delta(
    delta: torch.Tensor,
    *,
    rank: int,
    scaling: float,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor ``[output, input]`` into standard linear LoRA A/B tensors.

    ARC's output dimension is the 16-token vocabulary, so an identity prefix
    in B represents every full ``lm_head`` row exactly whenever rank >= vocab.
    """
    if delta.ndim != 2:
        raise ValueError(
            f"Linear delta must have shape [output, input], got {delta.shape}"
        )
    output_size, input_size = map(int, delta.shape)
    if rank < output_size:
        raise ValueError(
            f"LoRA rank {rank} cannot exactly encode {output_size} output rows"
        )
    if not math.isfinite(scaling) or scaling == 0:
        raise ValueError(f"Invalid LoRA scaling: {scaling}")
    output_dtype = dtype or delta.dtype
    lora_a = torch.zeros((rank, input_size), dtype=output_dtype)
    lora_b = torch.zeros((output_size, rank), dtype=output_dtype)
    lora_a[:output_size] = (delta / scaling).to(output_dtype)
    diagonal = torch.arange(output_size)
    lora_b[diagonal, diagonal] = 1
    return lora_a.contiguous(), lora_b.contiguous()


def reconstruct_linear_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Reconstruct a standard linear ``[output, input]`` delta in FP32."""
    return lora_b.float().matmul(lora_a.float()) * float(scaling)


def convert_peft_state_for_vllm(
    current_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    config: Any,
    *,
    pissa_base_mode: str = "original",
    pissa_base_id: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    """Convert a trained ARC PEFT state into tensors/config accepted by vLLM."""
    exported_config = peft_config_dict(config)
    rank = int(exported_config["r"])
    scaling = _rslora_scaling(exported_config)
    source_method = _source_lora_method(exported_config)
    current_embedding_key = _find_embedding_weight_key(current_state)
    initial_embedding_key = _find_embedding_weight_key(initial_state)
    current_lm_head_key = _find_lm_head_weight_key(current_state)
    initial_lm_head_key = _find_lm_head_weight_key(initial_state)
    saved_modules = _module_leaves(exported_config.get("modules_to_save"))
    convert_lm_head = "lm_head" in saved_modules

    if current_lm_head_key is not None and not convert_lm_head:
        if initial_lm_head_key is None:
            raise ValueError(
                "PEFT state contains an unexpected full lm_head without an "
                "immutable initial snapshot"
            )
        current_lm_head = current_state[current_lm_head_key].detach().cpu()
        initial_lm_head = initial_state[initial_lm_head_key].detach().cpu()
        if not torch.equal(current_lm_head, initial_lm_head):
            raise ValueError(
                "PEFT state contains a changed full lm_head that is not "
                "declared in modules_to_save"
            )

    unsupported = [
        key
        for key in current_state
        if not _is_lora_weight(key)
        and key != current_embedding_key
        and key != current_lm_head_key
        and not key.endswith(_IGNORABLE_BASE_EMBEDDING_SUFFIXES)
    ]
    if unsupported:
        raise ValueError(
            "vLLM export found unsupported non-LoRA PEFT tensors: "
            f"{unsupported}"
        )

    exported_state = {}
    # Non-finite counts of the training-device source tensors, keyed by the
    # exact CPU copy they describe. Scanning ~0.5 GiB of BF16 on one CPU
    # thread takes over a second per task; on CUDA it is a few milliseconds.
    source_nonfinite = {}
    for key, value in current_state.items():
        if not _is_lora_weight(key):
            continue
        value = value.detach()
        exported = value.to(device="cpu", copy=True).contiguous()
        exported_state[key] = exported
        source_nonfinite[key] = (exported, _nonfinite_count(value))
    source_initialization = exported_config.get("init_lora_weights", True)
    pissa_base_mode = normalize_pissa_base_mode(pissa_base_mode)
    pissa_enabled = is_pissa_init(source_initialization)
    pissa_shared = pissa_base_mode == "shared-residual"
    if pissa_shared and (
        not pissa_enabled or not re.fullmatch(r"[0-9a-f]{64}", pissa_base_id or "")
    ):
        raise ValueError("Direct PiSSA export requires a shared residual base identity and PiSSA factors")
    if not pissa_shared and pissa_base_id is not None:
        raise ValueError("An original-base adapter cannot declare a residual base identity")
    pissa_converted = pissa_enabled and not pissa_shared
    if isinstance(source_initialization, str) and source_initialization.startswith("pissa"):
        normalize_lora_init(source_initialization)  # Reject malformed PiSSA modes.
    pissa_projection_count = 0
    training_rank = rank
    if pissa_enabled:
        if (
            exported_config.get("rank_pattern")
            or exported_config.get("alpha_pattern")
            or exported_config.get("use_dora")
            or exported_config.get("use_nora")
        ):
            raise ValueError("PiSSA export requires uniform rank/alpha without DoRA/NoRA")
        if any(".lora_embedding_" in key for key in exported_state):
            raise ValueError("PiSSA export requires full vocabulary modules, not embedding LoRA")
        initial_factor_keys = {key for key in initial_state if _is_lora_weight(key)}
        if initial_factor_keys != exported_state.keys():
            raise ValueError(
                "PiSSA export requires immutable initial A0/B0 for every projection; "
                f"missing={sorted(set(exported_state) - initial_factor_keys)}, "
                f"unexpected={sorted(initial_factor_keys - set(exported_state))}"
            )
        for key in tuple(exported_state):
            if ".lora_A." not in key:
                continue
            b_key = key.replace(".lora_A.", ".lora_B.")
            if b_key not in exported_state:
                raise ValueError(f"PiSSA export is missing {b_key!r}")
            a, b = exported_state[key], exported_state[b_key]
            a0 = initial_state[key].detach().cpu()
            b0 = initial_state[b_key].detach().cpu()
            if (
                a.ndim != 2 or b.ndim != 2
                or a.shape != a0.shape or b.shape != b0.shape
                or a.shape[0] != rank or b.shape[1] != rank
                or a.dtype != a0.dtype or b.dtype != b0.dtype
            ):
                raise ValueError(f"PiSSA initial/trained factor shape or dtype mismatch at {key!r}")
            _require_finite_tensor(a0, name=key + " (PiSSA initial)")
            _require_finite_tensor(b0, name=b_key + " (PiSSA initial)")
            # [B, -B0] @ [A; A0] == B@A - B0@A0. No truncation/SVD
            # approximation of the learned update is used during export.
            if not pissa_shared:
                exported_state[key] = torch.cat((a, a0), dim=0).contiguous()
                exported_state[b_key] = torch.cat((b, -b0), dim=1).contiguous()
            pissa_projection_count += 1
        if not pissa_projection_count or 2 * pissa_projection_count != len(exported_state):
            raise ValueError("PiSSA export has unpaired/non-projection LoRA factors")
        if not pissa_shared:
            rank *= 2
            exported_config["r"] = rank
            exported_config["lora_alpha"] = float(exported_config["lora_alpha"]) * (
                math.sqrt(2.0) if exported_config.get("use_rslora") else 2.0
            )
        exported_config["init_lora_weights"] = True
    materialized_nora_a_keys: list[str] = []
    if source_method == "nora":
        for key, value in tuple(exported_state.items()):
            if ".lora_A." not in key:
                continue
            exported_state[key] = _normalize_nora_lora_a(value)
            materialized_nora_a_keys.append(key)
    for key, value in exported_state.items():
        exported, nonfinite_count = source_nonfinite.get(key, (None, 0))
        if exported is value:
            _raise_if_nonfinite(key, nonfinite_count, value.numel())
        else:
            _require_finite_tensor(value, name=key)
    conversion: dict[str, Any] = {
        "embedding_converted": False,
        "embedding_max_abs_error": 0.0,
        "lm_head_converted": False,
        "lm_head_max_abs_error": 0.0,
        "scaling": scaling,
        "source_lora_method": source_method,
        "persistent_nora_materialized": source_method == "nora",
        "materialized_nora_a_count": len(materialized_nora_a_keys),
        "materialized_nora_a_keys": materialized_nora_a_keys,
        "source_initialization": source_initialization,
        "pissa_converted": pissa_converted,
        "pissa_base_mode": pissa_base_mode,
        "pissa_base_id": pissa_base_id,
        "pissa_projection_count": pissa_projection_count,
        "training_rank": training_rank,
        "export_rank": rank,
    }

    if current_embedding_key is not None:
        if initial_embedding_key is None:
            raise ValueError(
                "Current PEFT state contains a full embedding but the immutable "
                "initial snapshot does not"
            )
        current_embedding = current_state[current_embedding_key].detach().cpu()
        initial_embedding = initial_state[initial_embedding_key].detach().cpu()
        _require_finite_tensor(
            current_embedding,
            name=current_embedding_key,
        )
        _require_finite_tensor(
            initial_embedding,
            name=initial_embedding_key,
        )
        if current_embedding.shape != initial_embedding.shape:
            raise ValueError(
                "Embedding shape changed during training: "
                f"{initial_embedding.shape} -> {current_embedding.shape}"
            )
        delta = current_embedding.float() - initial_embedding.float()
        lora_a, lora_b = factor_embedding_delta(
            delta,
            rank=rank,
            scaling=scaling,
            dtype=current_embedding.dtype,
        )
        reconstructed = reconstruct_embedding_delta(lora_a, lora_b, scaling)
        max_abs_error = float((reconstructed - delta).abs().max().item())
        # BF16 conversion can round the trained full weight delta.  Validate
        # against a dtype-aware tolerance while retaining an explicit metric.
        tolerance = 5e-3 if current_embedding.dtype == torch.bfloat16 else 5e-5
        if max_abs_error > tolerance:
            raise ValueError(
                "Embedding LoRA reconstruction failed: max_abs_error="
                f"{max_abs_error:.6g}, tolerance={tolerance:.6g}"
            )
        prefix = _embedding_module_prefix(current_embedding_key)
        exported_state[f"{prefix}.lora_embedding_A"] = lora_a
        exported_state[f"{prefix}.lora_embedding_B"] = lora_b
        conversion.update(
            embedding_converted=True,
            embedding_key=current_embedding_key,
            embedding_shape=list(current_embedding.shape),
            embedding_max_abs_error=max_abs_error,
        )
    elif "embed_tokens" in saved_modules:
        raise ValueError(
            "PEFT config declares modules_to_save but no full embed_tokens "
            "weight was found"
        )

    if convert_lm_head:
        if current_lm_head_key is None:
            raise ValueError(
                "PEFT config declares lm_head in modules_to_save but no full "
                "lm_head weight was found"
            )
        if initial_lm_head_key is None:
            raise ValueError(
                "Current PEFT state contains a full lm_head but the immutable "
                "initial snapshot does not"
            )
        current_lm_head = current_state[current_lm_head_key].detach().cpu()
        initial_lm_head = initial_state[initial_lm_head_key].detach().cpu()
        _require_finite_tensor(current_lm_head, name=current_lm_head_key)
        _require_finite_tensor(initial_lm_head, name=initial_lm_head_key)
        if current_lm_head.shape != initial_lm_head.shape:
            raise ValueError(
                "lm_head shape changed during training: "
                f"{initial_lm_head.shape} -> {current_lm_head.shape}"
            )
        delta = current_lm_head.float() - initial_lm_head.float()
        lora_a, lora_b = factor_linear_delta(
            delta,
            rank=rank,
            scaling=scaling,
            dtype=current_lm_head.dtype,
        )
        reconstructed = reconstruct_linear_delta(lora_a, lora_b, scaling)
        max_abs_error = float((reconstructed - delta).abs().max().item())
        tolerance = 5e-3 if current_lm_head.dtype == torch.bfloat16 else 5e-5
        if max_abs_error > tolerance:
            raise ValueError(
                "lm_head LoRA reconstruction failed: max_abs_error="
                f"{max_abs_error:.6g}, tolerance={tolerance:.6g}"
            )
        prefix = _lm_head_module_prefix(current_lm_head_key)
        lora_a_key = f"{prefix}.lora_A.weight"
        lora_b_key = f"{prefix}.lora_B.weight"
        collisions = [
            key for key in (lora_a_key, lora_b_key) if key in exported_state
        ]
        if collisions:
            raise ValueError(
                "Cannot combine full lm_head training with existing lm_head "
                f"LoRA tensors: {collisions}"
            )
        exported_state[lora_a_key] = lora_a
        exported_state[lora_b_key] = lora_b
        conversion.update(
            lm_head_converted=True,
            lm_head_key=current_lm_head_key,
            lm_head_shape=list(current_lm_head.shape),
            lm_head_max_abs_error=max_abs_error,
        )

    exported_config["modules_to_save"] = None
    target_modules = exported_config.get("target_modules") or []
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    target_modules = list(target_modules)
    if current_embedding_key is not None and "embed_tokens" not in target_modules:
        target_modules.append("embed_tokens")
    if convert_lm_head and "lm_head" not in target_modules:
        target_modules.append("lm_head")
    exported_config["target_modules"] = sorted(set(target_modules))
    exported_config["inference_mode"] = True
    # vLLM consumes an ordinary static LoRA adapter. Persistent NoRA has
    # already been materialized into every trained projection A above;
    # one-time NoRA and BIMI require no inference-time behavior. Remove the
    # fork-only control so stock serving runtimes can parse the artifact.
    exported_config.pop("use_nora", None)
    if exported_config.get("init_lora_weights") == "bimi":
        exported_config["init_lora_weights"] = True

    if not exported_state:
        raise ValueError("Refusing to publish an empty LoRA adapter")
    return exported_state, exported_config, conversion


def publish_vllm_adapter(
    *,
    root: str | os.PathLike[str],
    adapter_name: str,
    adapter_id: int,
    task_key: str,
    current_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    config: Any,
    extra_manifest: Mapping[str, Any] | None = None,
    pissa_base_mode: str = "original",
    pissa_base_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate and atomically publish one native vLLM adapter directory."""
    started_at = time.perf_counter()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    final_path = root_path / adapter_name
    if final_path.exists():
        raise FileExistsError(
            f"Adapter generation already exists and names cannot be reused: {final_path}"
        )
    temporary_path = root_path / (
        f".{adapter_name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary_path.mkdir(parents=False, exist_ok=False)
    try:
        state, exported_config, conversion = convert_peft_state_for_vllm(
            current_state,
            initial_state,
            config,
            pissa_base_mode=pissa_base_mode,
            pissa_base_id=pissa_base_id,
        )
        save_file(state, temporary_path / ADAPTER_WEIGHTS_FILE)
        with (temporary_path / ADAPTER_CONFIG_FILE).open("w") as config_file:
            json.dump(exported_config, config_file, indent=2, sort_keys=True)
            config_file.write("\n")

        manifest = {
            "format_version": 1,
            "adapter_name": adapter_name,
            "adapter_id": int(adapter_id),
            "task_key": task_key,
            "tensor_count": len(state),
            "tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in state.values()
            ),
            "conversion": conversion,
        }
        if extra_manifest:
            manifest.update(_jsonable(dict(extra_manifest)))
        with (temporary_path / ADAPTER_MANIFEST_FILE).open("w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")

        # All files live under one parent, so rename is an atomic publication
        # boundary.  Consumers never observe a half-written adapter.
        os.replace(temporary_path, final_path)
        manifest["export_seconds"] = time.perf_counter() - started_at
        return final_path, manifest
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def preserve_adapter_artifact(
    path: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
) -> Path:
    """Atomically copy one published adapter into a persistent replay root.

    The live inference service owns and eventually deletes ``path``.  A full
    copy (rather than hard links) keeps the replay artifact independent of that
    lifecycle and also works when temporary adapters live in ``/dev/shm``.
    """
    source_path = Path(path).resolve()
    if not source_path.is_dir() or not source_path.name.startswith("arc-"):
        raise ValueError(f"Refusing to preserve invalid ARC adapter: {source_path}")
    for required_file in (
        ADAPTER_WEIGHTS_FILE,
        ADAPTER_CONFIG_FILE,
        ADAPTER_MANIFEST_FILE,
    ):
        if not (source_path / required_file).is_file():
            raise ValueError(
                f"Refusing to preserve incomplete ARC adapter {source_path}: "
                f"missing {required_file}"
            )

    root_path = Path(destination_root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    final_path = root_path / source_path.name
    if final_path.exists():
        raise FileExistsError(
            f"Preserved adapter generation already exists: {final_path}"
        )
    temporary_path = root_path / (
        f".{source_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(source_path, temporary_path)
        os.replace(temporary_path, final_path)
        return final_path
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise


def remove_adapter_artifact(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> None:
    """Remove one completed adapter without permitting broad recursive deletes."""
    root_path = Path(root).resolve()
    adapter_path = Path(path).resolve()
    if adapter_path.parent != root_path or not adapter_path.name.startswith("arc-"):
        raise ValueError(
            f"Refusing to remove adapter outside its run root: {adapter_path}"
        )
    shutil.rmtree(adapter_path)


def acquire_adapter_staging_root(
    roots,
    staging_slots,
    *,
    abort_event=None,
    end_time=float("inf"),
    poll_seconds=0.1,
):
    """Reserve the first available tier in the planner's priority order."""
    roots = tuple(map(str, roots or ()))
    slots = tuple(staging_slots or ())
    if not roots or len(roots) != len(slots):
        raise ValueError("Adapter staging roots and slot semaphores must align")
    while True:
        for index, (root, semaphore) in enumerate(zip(roots, slots)):
            if semaphore.acquire(block=False):
                return index, root
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError("Adapter staging reservation aborted")
        if time.time() >= float(end_time):
            raise TimeoutError("Adapter staging reservation reached the run deadline")
        time.sleep(float(poll_seconds))


def release_adapter_staging_slot(job, staging_slots) -> bool:
    """Return one tier reservation exactly once within this job owner."""
    if not staging_slots or job.get("_adapter_staging_slot_released"):
        return False
    index = job.get("adapter_staging_index")
    if index is None:
        return False
    staging_slots[int(index)].release()
    job["_adapter_staging_slot_released"] = True
    return True


def remove_job_adapter_artifact(
    job,
    default_root,
    staging_slots=None,
) -> None:
    """Remove a queued job's artifact and release its filesystem tier slot."""
    root = job.get("adapter_root") or default_root
    try:
        remove_adapter_artifact(job["adapter_path"], root)
    except FileNotFoundError:
        release_adapter_staging_slot(job, staging_slots)
        raise
    else:
        release_adapter_staging_slot(job, staging_slots)
