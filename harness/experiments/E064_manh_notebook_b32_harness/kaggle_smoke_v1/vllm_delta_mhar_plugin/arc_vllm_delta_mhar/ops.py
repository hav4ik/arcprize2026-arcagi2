"""Flattened-token Delta-MH routing used by the vLLM model plugin."""

from __future__ import annotations

import os
import importlib
import inspect
from typing import Any, Callable

import torch

try:
    # Import Triton directly so importing the CPU reference operator does not
    # initialize vLLM in trainer/test processes.  The targeted vLLM wheels
    # depend on the same top-level Triton package.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU and minimal unit-test environments
    try:
        from vllm.triton_utils import tl, triton

        HAS_TRITON = True
    except Exception:
        triton = None
        tl = None
        HAS_TRITON = False


def reference_delta_mhar_mix(
    *,
    partial: torch.Tensor,
    delta_bank: torch.Tensor,
    delta_count: int,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    null_source: torch.Tensor,
    num_heads: int,
    variance_epsilon: float,
) -> torch.Tensor:
    """Reference B4S1 depth mix over flattened vLLM token rows."""
    if partial.ndim != 2:
        raise ValueError(
            f"Expected [tokens, hidden] activations, got {tuple(partial.shape)}"
        )
    token_count, hidden_size = map(int, partial.shape)
    delta_count = int(delta_count)
    if not 0 <= delta_count <= int(delta_bank.shape[1]):
        raise ValueError(
            f"Invalid completed-block count {delta_count} for bank "
            f"{tuple(delta_bank.shape)}"
        )
    if delta_bank.ndim != 3 or tuple(delta_bank.shape[::2]) != (
        token_count,
        hidden_size,
    ):
        raise ValueError(
            "Delta bank must have shape [tokens, blocks, hidden], got "
            f"{tuple(delta_bank.shape)}"
        )
    if hidden_size % int(num_heads):
        raise ValueError(
            f"hidden size {hidden_size} is not divisible by {num_heads} heads"
        )
    if tuple(query.shape) != (int(num_heads), hidden_size // int(num_heads)):
        raise ValueError(f"Invalid routing query shape {tuple(query.shape)}")
    if norm_weight.numel() != hidden_size or null_source.numel() != hidden_size:
        raise ValueError("Delta-MH norm/null parameter shape mismatch")

    null = null_source.to(device=partial.device, dtype=partial.dtype)
    null = null.view(1, 1, hidden_size).expand(token_count, 1, hidden_size)
    sources = torch.cat((null, delta_bank[:, :delta_count]), dim=1)
    source_float = sources.float()
    inverse_rms = torch.rsqrt(
        source_float.square().mean(dim=-1, keepdim=True) + float(variance_epsilon)
    )
    # Match the Transformers/Unsloth contract: normalized values are rounded
    # to the activation dtype before the FP32 RMS scale and router dot product.
    keys = (source_float * inverse_rms).to(partial.dtype).float()
    keys = keys * norm_weight.float().view(1, 1, hidden_size)
    head_size = hidden_size // int(num_heads)
    keys = keys.view(token_count, 1 + delta_count, int(num_heads), head_size)
    values = source_float.view_as(keys)
    scores = (keys * query.float().view(1, 1, int(num_heads), head_size)).sum(dim=-1)
    probabilities = scores.softmax(dim=1)
    routed = (probabilities[..., None] * values).sum(dim=1)
    return routed.reshape_as(partial).to(partial.dtype)


if HAS_TRITON:

    @triton.jit
    def _delta_mhar_bank_forward_kernel(
        delta_bank,
        output,
        query,
        norm_weight,
        null_source,
        token_count,
        hidden_size,
        num_heads,
        head_size,
        stride_bank_token,
        stride_bank_source,
        variance_epsilon,
        inverse_hidden_size,
        DELTA_COUNT: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
        WIDTH_BLOCK: tl.constexpr,
    ):
        token_id = tl.program_id(0).to(tl.int64)
        head_offsets = tl.arange(0, HEAD_BLOCK)
        width_offsets = tl.arange(0, WIDTH_BLOCK)
        head_mask = head_offsets < num_heads
        value_mask = head_mask[:, None] & (width_offsets[None, :] < head_size)
        hidden_offsets = head_offsets[:, None] * head_size + width_offsets[None, :]

        q = tl.load(query + hidden_offsets, mask=value_mask, other=0.0).to(tl.float32)
        norm = tl.load(norm_weight + hidden_offsets, mask=value_mask, other=0.0).to(
            tl.float32
        )
        running_max = tl.full((HEAD_BLOCK,), -1.0e30, tl.float32)
        running_sum = tl.zeros((HEAD_BLOCK,), tl.float32)
        accumulator = tl.zeros((HEAD_BLOCK, WIDTH_BLOCK), tl.float32)

        for source_id in tl.static_range(0, DELTA_COUNT + 1):
            if source_id == 0:
                # The learned null is FP32 but the reference first rounds it to
                # the activation dtype used by the residual stream.
                value = (
                    tl.load(
                        null_source + hidden_offsets,
                        mask=value_mask,
                        other=0.0,
                    )
                    .to(output.dtype.element_ty)
                    .to(tl.float32)
                )
            else:
                value = tl.load(
                    delta_bank
                    + token_id * stride_bank_token
                    + (source_id - 1) * stride_bank_source
                    + hidden_offsets,
                    mask=value_mask,
                    other=0.0,
                ).to(tl.float32)
            square_sum = tl.sum(tl.sum(value * value, axis=1), axis=0)
            inverse_rms = tl.rsqrt(square_sum * inverse_hidden_size + variance_epsilon)
            key = (value * inverse_rms).to(output.dtype.element_ty).to(
                tl.float32
            ) * norm
            score = tl.sum(q * key, axis=1)
            score = tl.where(head_mask, score, -1.0e30)
            next_max = tl.maximum(running_max, score)
            old_scale = tl.exp(running_max - next_max)
            source_scale = tl.exp(score - next_max)
            running_sum = running_sum * old_scale + source_scale
            accumulator = (
                accumulator * old_scale[:, None] + source_scale[:, None] * value
            )
            running_max = next_max

        routed = accumulator / running_sum[:, None]
        tl.store(
            output + token_id * hidden_size + hidden_offsets,
            routed.to(output.dtype.element_ty),
            mask=value_mask,
        )


def _launch_fused_delta_mhar_mix(
    *,
    partial: torch.Tensor,
    delta_bank: torch.Tensor,
    delta_count: int,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    null_source: torch.Tensor,
    num_heads: int,
    variance_epsilon: float,
) -> torch.Tensor:
    """Launch the inference-only contiguous-bank Triton router.

    This implementation deliberately stays behind a stable Torch dispatcher
    operator.  Calling a raw Triton JIT function from vLLM's compiled model
    graph records a process-local ``kernel_idx`` in the on-disk AOT artifact;
    a later engine process cannot resolve that index and has to recompile the
    entire model.  The dispatcher boundary keeps the same fused kernel while
    making the cached graph refer to ``arc_delta_mhar::fused_mix`` instead.
    """
    if not HAS_TRITON or triton is None:
        raise RuntimeError("Triton is unavailable for fused Delta-MH inference")
    if not partial.is_cuda:
        raise ValueError("Fused Delta-MH inference requires CUDA activations")
    if partial.ndim != 2:
        raise ValueError("Fused Delta-MH expects flattened 2D activations")
    token_count, hidden_size = map(int, partial.shape)
    delta_count = int(delta_count)
    num_heads = int(num_heads)
    # AOTAutograd represents the model's in-place-updated delta bank as a
    # synthetic storage base.  Inductor can pass that base (1D) through an
    # opaque custom-op boundary even though its fake/compile-time value is the
    # original 3D view. Reconstruct the contiguous logical bank without a copy.
    if delta_bank.ndim == 1:
        bank_row_size = token_count * hidden_size
        if bank_row_size == 0 or delta_bank.numel() % bank_row_size:
            raise ValueError(
                "Invalid flattened Delta-MH bank storage: "
                f"shape={tuple(delta_bank.shape)}, tokens={token_count}, "
                f"hidden={hidden_size}"
            )
        delta_bank = delta_bank.view(
            token_count,
            delta_bank.numel() // bank_row_size,
            hidden_size,
        )
    if delta_bank.ndim != 3:
        raise ValueError(
            f"Fused Delta-MH expects a 3D bank, got {tuple(delta_bank.shape)}"
        )
    if hidden_size % num_heads:
        raise ValueError(
            f"hidden size {hidden_size} is not divisible by {num_heads} heads"
        )
    if tuple(delta_bank.shape) != (
        token_count,
        int(delta_bank.shape[1]),
        hidden_size,
    ):
        raise ValueError(f"Invalid delta bank shape {tuple(delta_bank.shape)}")
    if not 0 <= delta_count <= int(delta_bank.shape[1]):
        raise ValueError(f"Invalid completed-block count {delta_count}")
    delta_bank = delta_bank.contiguous()
    output = torch.empty_like(partial, memory_format=torch.contiguous_format)
    head_size = hidden_size // num_heads
    head_block = triton.next_power_of_2(num_heads)
    width_block = triton.next_power_of_2(head_size)
    tile_size = head_block * width_block
    num_warps = 8 if tile_size >= 2048 else 4
    _delta_mhar_bank_forward_kernel[(token_count,)](
        delta_bank,
        output,
        query.contiguous(),
        norm_weight.contiguous(),
        null_source.contiguous(),
        token_count,
        hidden_size,
        num_heads,
        head_size,
        delta_bank.stride(0),
        delta_bank.stride(1),
        float(variance_epsilon),
        1.0 / hidden_size,
        DELTA_COUNT=delta_count,
        HEAD_BLOCK=head_block,
        WIDTH_BLOCK=width_block,
        num_warps=num_warps,
    )
    return output


def _fused_delta_mhar_mix_impl(
    partial: torch.Tensor,
    delta_bank: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    null_source: torch.Tensor,
    delta_count: int,
    num_heads: int,
    variance_epsilon: float,
) -> torch.Tensor:
    return _launch_fused_delta_mhar_mix(
        partial=partial,
        delta_bank=delta_bank,
        delta_count=delta_count,
        query=query,
        norm_weight=norm_weight,
        null_source=null_source,
        num_heads=num_heads,
        variance_epsilon=variance_epsilon,
    )


def _fused_delta_mhar_mix_fake(
    partial: torch.Tensor,
    delta_bank: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    null_source: torch.Tensor,
    delta_count: int,
    num_heads: int,
    variance_epsilon: float,
) -> torch.Tensor:
    del delta_bank, query, norm_weight, null_source
    del delta_count, num_heads, variance_epsilon
    return torch.empty_like(partial, memory_format=torch.contiguous_format)


_DELTA_MHAR_LIBRARY = torch.library.Library("arc_delta_mhar", "FRAGMENT")
_DELTA_MHAR_LIBRARY.define(
    "fused_mix(Tensor partial, Tensor delta_bank, Tensor query, "
    "Tensor norm_weight, Tensor null_source, int delta_count, int num_heads, "
    "float variance_epsilon) -> Tensor"
)

# Qwen3.5's native vLLM executor also calls a raw Triton JIT kernel from its
# compiled forward for fused Q/K RMSNorm + RoPE + gate extraction.  Like the
# Delta-MH kernel above, that stores a process-local ``kernel_idx`` in vLLM's
# on-disk AOT artifact.  A second EngineCore process cannot resolve the index
# and recompiles the model.  Keep the native implementation and its exact
# numerics, but put the launch behind the same stable dispatcher boundary.
_ORIGINAL_QWEN35_FUSED_QK: Callable[..., Any] | None = None
_ORIGINAL_QWEN35_FUSED_QK_PARAMETERS: frozenset[str] = frozenset()


def _qwen35_fused_qk_impl(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None,
    norm_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _ORIGINAL_QWEN35_FUSED_QK is None:
        raise RuntimeError(
            "The compile-stable Qwen3.5 fused-QK wrapper was not installed"
        )
    optional_kwargs: dict[str, Any] = {}
    if "mrope_section" in _ORIGINAL_QWEN35_FUSED_QK_PARAMETERS:
        optional_kwargs["mrope_section"] = mrope_section
    elif mrope_section is not None:
        raise RuntimeError(
            "The installed vLLM fused-QK kernel does not support MRoPE"
        )
    if "norm_beta" in _ORIGINAL_QWEN35_FUSED_QK_PARAMETERS:
        optional_kwargs["norm_beta"] = float(norm_beta)
    elif float(norm_beta) != 0.0:
        raise RuntimeError(
            "The installed vLLM fused-QK kernel does not support norm_beta"
        )
    return _ORIGINAL_QWEN35_FUSED_QK(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        float(eps),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(rotary_dim),
        **optional_kwargs,
    )


def _qwen35_fused_qk_fake(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | None,
    norm_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, cos_sin_cache, positions
    del eps, num_kv_heads, rotary_dim, mrope_section, norm_beta
    q_shape = (q_gate.shape[0], int(num_q_heads) * int(head_dim))
    q = q_gate.new_empty(q_shape)
    return q, torch.empty_like(k), torch.empty_like(q)


_DELTA_MHAR_LIBRARY.define(
    "qwen35_fused_qk(Tensor q_gate, Tensor k, Tensor q_weight, "
    "Tensor k_weight, Tensor cos_sin_cache, Tensor positions, float eps, "
    "int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim, "
    "int[]? mrope_section, float norm_beta) "
    "-> (Tensor, Tensor, Tensor)"
)
if HAS_TRITON:
    _DELTA_MHAR_LIBRARY.impl(
        "qwen35_fused_qk",
        _qwen35_fused_qk_impl,
        "CUDA",
    )
_DELTA_MHAR_LIBRARY._register_fake(
    "qwen35_fused_qk",
    _qwen35_fused_qk_fake,
)
if HAS_TRITON:
    _DELTA_MHAR_LIBRARY.impl(
        "fused_mix",
        _fused_delta_mhar_mix_impl,
        "CUDA",
    )
_DELTA_MHAR_LIBRARY._register_fake(
    "fused_mix",
    _fused_delta_mhar_mix_fake,
)


def fused_delta_mhar_mix(
    *,
    partial: torch.Tensor,
    delta_bank: torch.Tensor,
    delta_count: int,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    null_source: torch.Tensor,
    num_heads: int,
    variance_epsilon: float,
) -> torch.Tensor:
    """Run the fused router through its compile-stable dispatcher operator."""
    if not HAS_TRITON or triton is None:
        raise RuntimeError("Triton is unavailable for fused Delta-MH inference")
    if not partial.is_cuda:
        raise ValueError("Fused Delta-MH inference requires CUDA activations")
    return torch.ops.arc_delta_mhar.fused_mix(
        partial,
        delta_bank,
        query,
        norm_weight,
        null_source,
        int(delta_count),
        int(num_heads),
        float(variance_epsilon),
    )


def compile_stable_qwen35_fused_qk(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | tuple[int, int, int] | None = None,
    norm_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call vLLM's native fused Qwen3.5 Q/K kernel via a stable custom op."""
    return torch.ops.arc_delta_mhar.qwen35_fused_qk(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        float(eps),
        int(num_q_heads),
        int(num_kv_heads),
        int(head_dim),
        int(rotary_dim),
        None if mrope_section is None else list(map(int, mrope_section)),
        float(norm_beta),
    )


def install_compile_stable_qwen35_fused_qk() -> None:
    """Patch only the Python call boundary used by vLLM's Qwen3.5 executor.

    The underlying Triton implementation is left untouched.  Installation is
    idempotent because a model module can be imported more than once while
    vLLM probes supported architectures.
    """
    global _ORIGINAL_QWEN35_FUSED_QK
    global _ORIGINAL_QWEN35_FUSED_QK_PARAMETERS
    qwen3_next = importlib.import_module(
        "vllm.model_executor.models.qwen3_next"
    )
    current = qwen3_next.fused_qk_rmsnorm_rope_gate
    if current is compile_stable_qwen35_fused_qk:
        return
    _ORIGINAL_QWEN35_FUSED_QK = current
    _ORIGINAL_QWEN35_FUSED_QK_PARAMETERS = frozenset(
        inspect.signature(current).parameters
    )
    qwen3_next.fused_qk_rmsnorm_rope_gate = compile_stable_qwen35_fused_qk


def delta_mhar_mix(
    *,
    partial: torch.Tensor,
    delta_bank: torch.Tensor,
    delta_count: int,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    null_source: torch.Tensor,
    num_heads: int,
    variance_epsilon: float,
    fused: bool,
) -> torch.Tensor:
    """Choose the fused production path with an explicit reference bisect."""
    use_reference = os.getenv("ARC_VLLM_DELTA_MHAR_REFERENCE", "0") == "1"
    if fused and partial.is_cuda and HAS_TRITON and not use_reference:
        return fused_delta_mhar_mix(
            partial=partial,
            delta_bank=delta_bank,
            delta_count=delta_count,
            query=query,
            norm_weight=norm_weight,
            null_source=null_source,
            num_heads=num_heads,
            variance_epsilon=variance_epsilon,
        )
    return reference_delta_mhar_mix(
        partial=partial,
        delta_bank=delta_bank,
        delta_count=delta_count,
        query=query,
        norm_weight=norm_weight,
        null_source=null_source,
        num_heads=num_heads,
        variance_epsilon=variance_epsilon,
    )


__all__ = [
    "HAS_TRITON",
    "delta_mhar_mix",
    "fused_delta_mhar_mix",
    "reference_delta_mhar_mix",
]
