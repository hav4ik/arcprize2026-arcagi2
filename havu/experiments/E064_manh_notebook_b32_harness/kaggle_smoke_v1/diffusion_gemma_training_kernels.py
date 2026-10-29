"""Opt-in exact-BF16 training kernels for the compact DiffusionGemma model.

The patches in this module deliberately preserve module objects, Parameters,
and state-dict names.  They only replace repeated elementwise forward paths;
the dense and routed-expert matrix multiplications remain owned by PyTorch and
Transformers (including the native ``grouped_mm`` expert implementation).
"""

from __future__ import annotations

import os
from collections import Counter
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F


KERNEL_BACKENDS = (
    "torch",
    "liger-gelu",
    "liger-rms",
    "liger-rms-cutedsl",
    "liger",
)
PACKED_PROJECTION_MODES = ("off", "dense", "attention", "all")
ENCODER_ATTENTION_BACKENDS = ("sdpa", "flash-attention-2", "hybrid-flex")
DECODER_ATTENTION_BACKENDS = ("sdpa", "flash-attention-2")

_COMPILED_GLOBAL_FLEX_ATTENTION: dict[tuple[int, int], Any] = {}


def _project_attention_qkv(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Use the opt-in shared-input frozen-FP8 LoRA projection path."""

    enabled = os.environ.get(
        "REAM_DIFFUSION_GEMMA_FUSED_ATTENTION_QKV",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return (
            attention.q_proj(hidden_states),
            attention.k_proj(hidden_states),
            (
                attention.v_proj(hidden_states)
                if attention.v_proj is not None
                else None
            ),
        )
    try:
        from ream.diffusion_gemma_lora import fused_attention_projections
    except ImportError as error:
        raise RuntimeError(
            "fused DiffusionGemma attention projections require the REAM "
            "frozen-FP8 LoRA runtime"
        ) from error
    return fused_attention_projections(attention, hidden_states)


def diffusion_gemma_flex_forward_block_size() -> int:
    """Return the opt-in global-attention forward tile.

    Backward remains on 16-token tiles because a 32-token backward tile needs
    more shared memory than an L4 exposes.  Keep 16 as the default until the
    isolated parity and production trainer gates are both recorded.
    """

    raw_value = os.environ.get(
        "ARC_DIFFUSION_GEMMA_FLEX_FORWARD_BLOCK_SIZE",
        "16",
    )
    try:
        block_size = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "ARC_DIFFUSION_GEMMA_FLEX_FORWARD_BLOCK_SIZE must be 16 or 32, "
            f"got {raw_value!r}"
        ) from error
    if block_size not in (16, 32):
        raise ValueError(
            "ARC_DIFFUSION_GEMMA_FLEX_FORWARD_BLOCK_SIZE must be 16 or 32, "
            f"got {block_size}"
        )
    return block_size


def diffusion_gemma_flex_num_stages() -> int:
    """Return the opt-in SM89 FlexAttention pipeline depth."""

    raw_value = os.environ.get(
        "ARC_DIFFUSION_GEMMA_FLEX_NUM_STAGES",
        "1",
    )
    try:
        num_stages = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "ARC_DIFFUSION_GEMMA_FLEX_NUM_STAGES must be 1 or 2, "
            f"got {raw_value!r}"
        ) from error
    if num_stages not in (1, 2):
        raise ValueError(
            "ARC_DIFFUSION_GEMMA_FLEX_NUM_STAGES must be 1 or 2, "
            f"got {num_stages}"
        )
    return num_stages


def build_diffusion_gemma_flex_causal_block_mask(
    attention_mask: torch.Tensor,
    *,
    block_size: int | None = None,
):
    """Build one head-independent causal FlexAttention mask per microbatch.

    The NVARC packer puts independent right-padded documents on the batch
    axis. Every valid or padded query may see the valid causal prefix; loss
    and decoder-cache masks subsequently discard padded query positions. The
    production width-512 kernel uses 16-token blocks; the explicit option lets
    the isolated tuner test larger compatible query/key tiles without changing
    that default. The returned object is shared by all five global encoder
    layers and their checkpoint replays.
    """

    if block_size is None:
        block_size = diffusion_gemma_flex_forward_block_size()
    if attention_mask.ndim != 2:
        raise ValueError("FlexAttention requires a 2-D right-padding mask")
    if attention_mask.device.type != "cuda":
        raise ValueError("DiffusionGemma FlexAttention requires CUDA")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("FlexAttention block_size must be a positive power of two")
    from torch.nn.attention.flex_attention import create_block_mask

    batch_size, sequence_length = attention_mask.shape
    valid_lengths = attention_mask.to(dtype=torch.int32).sum(dim=-1)

    def causal_padding_mask(batch, _head, query_index, key_index):
        return (query_index >= key_index) & (
            key_index < valid_lengths[batch]
        )

    return create_block_mask(
        causal_padding_mask,
        int(batch_size),
        None,
        int(sequence_length),
        int(sequence_length),
        device=attention_mask.device,
        BLOCK_SIZE=int(block_size),
        _compile=True,
    )


def _compiled_global_flex_attention():
    """Return the dynamic exact-BF16 width-512 GQA FlexAttention graph."""

    forward_block_size = diffusion_gemma_flex_forward_block_size()
    num_stages = diffusion_gemma_flex_num_stages()
    cache_key = (forward_block_size, num_stages)
    if cache_key not in _COMPILED_GLOBAL_FLEX_ATTENTION:
        from torch.nn.attention.flex_attention import flex_attention

        # Width 512 exceeds FA2 and cuDNN's supported training head widths.
        # Backward stays at 16-token tiles to remain below L4's shared-memory
        # limit.  The parity-gated forward may opt into 32-token tiles. GQA
        # avoids materializing eight copies of each of the two KV heads.
        kernel_options = {
            "BLOCK_M": forward_block_size,
            "BLOCK_N": forward_block_size,
            "BLOCK_M1": 16,
            "BLOCK_N1": 16,
            "BLOCK_M2": 16,
            "BLOCK_N2": 16,
            "num_stages": num_stages,
            "num_warps": 4,
            "ROWS_GUARANTEED_SAFE": True,
            "BLOCKS_ARE_CONTIGUOUS": True,
        }

        def run(query, key, value, block_mask):
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                scale=1.0,
                enable_gqa=True,
                kernel_options=kernel_options,
            )

        _COMPILED_GLOBAL_FLEX_ATTENTION[cache_key] = torch.compile(
            run,
            fullgraph=True,
            dynamic=True,
            mode="default",
        )
    return _COMPILED_GLOBAL_FLEX_ATTENTION[cache_key]


def _hidden_activation(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    return str(getattr(text_config, "hidden_activation", ""))


def _install_liger_kernels(
    model: torch.nn.Module,
    *,
    patch_rms_norm: bool,
    patch_gelu_mul: bool,
    rms_norm_implementation: str = "triton",
) -> dict[str, Any]:
    if rms_norm_implementation not in ("triton", "cutedsl"):
        raise ValueError(
            f"unsupported Liger RMSNorm implementation {rms_norm_implementation!r}"
        )
    try:
        if patch_gelu_mul:
            from liger_kernel.ops.geglu import LigerGELUMulFunction
        if patch_rms_norm:
            if rms_norm_implementation == "cutedsl":
                # Import the implementation explicitly.  Liger's global
                # LIGER_KERNEL_IMPL switch only rewrites names exported from
                # liger_kernel.ops; importing ops.rms_norm directly (our
                # historical path) always selects Triton.
                from liger_kernel.ops.cutedsl.ops.rms_norm import (
                    LigerRMSNormFunction,
                )
            else:
                from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    except Exception as error:  # pragma: no cover - depends on Kaggle wheels
        raise RuntimeError(
            "the selected --diffusion-kernel-backend requires a working "
            "liger-kernel installation and its optional backend dependencies"
        ) from error

    activation = _hidden_activation(model)
    if patch_gelu_mul and activation != "gelu_pytorch_tanh":
        raise RuntimeError(
            "the fused DiffusionGemma GELU-multiply kernel is exact only for "
            f"gelu_pytorch_tanh; model reports {activation!r}"
        )

    counts: Counter[str] = Counter()
    patched_classes: Counter[str] = Counter()
    parameter_ids_before = {id(parameter) for parameter in model.parameters()}

    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name == "DiffusionGemmaRMSNorm" and patch_rms_norm:
            if getattr(module, "_arc_liger_rms_norm", False):
                continue

            def liger_rms_forward(self, hidden_states):
                weight = self.weight if self.with_scale else None
                # DiffusionGemma computes both statistics and affine scaling in
                # FP32, then casts back.  offset=0 and casting_mode="gemma"
                # reproduce that contract.  In-place backward is unsafe around
                # the model's residual branches, so keep it disabled.
                return LigerRMSNormFunction.apply(
                    hidden_states,
                    weight,
                    self.eps,
                    0.0,
                    "gemma",
                    False,
                )

            module.forward = MethodType(liger_rms_forward, module)
            module._arc_liger_rms_norm = True
            counts["rms_norm"] += 1
            patched_classes[class_name] += 1
        elif class_name == "DiffusionGemmaText4MLP" and patch_gelu_mul:
            if getattr(module, "_arc_liger_gelu_mul", False):
                continue

            def liger_dense_mlp_forward(self, hidden_states):
                gate = self.gate_proj(hidden_states)
                up = self.up_proj(hidden_states)
                return self.down_proj(LigerGELUMulFunction.apply(gate, up))

            module.forward = MethodType(liger_dense_mlp_forward, module)
            module._arc_liger_gelu_mul = True
            counts["dense_gelu_mul"] += 1
            patched_classes[class_name] += 1
        elif class_name == "DiffusionGemmaTextExperts" and patch_gelu_mul:
            if getattr(module, "_arc_liger_gelu_mul", False):
                continue

            def liger_expert_apply_gate(self, gate_up_out):
                gate, up = gate_up_out.chunk(2, dim=-1)
                return LigerGELUMulFunction.apply(gate, up)

            # Transformers' grouped-mm expert dispatcher calls this hook after
            # the packed gate/up projection.  The GEMMs and routing are left
            # untouched.
            module._apply_gate = MethodType(liger_expert_apply_gate, module)
            module._arc_liger_gelu_mul = True
            counts["expert_gelu_mul"] += 1
            patched_classes[class_name] += 1

    parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if parameter_ids_after != parameter_ids_before:
        raise RuntimeError("training-kernel installation changed model Parameters")
    if patch_rms_norm and not counts["rms_norm"]:
        raise RuntimeError("Liger DiffusionGemma patch found no RMSNorm modules")
    if patch_gelu_mul and not counts["dense_gelu_mul"]:
        raise RuntimeError("Liger DiffusionGemma patch found no dense MLP modules")
    if patch_rms_norm and patch_gelu_mul:
        backend = "liger"
    elif patch_rms_norm:
        backend = (
            "liger-rms-cutedsl"
            if rms_norm_implementation == "cutedsl"
            else "liger-rms"
        )
    else:
        backend = "liger-gelu"
    return {
        "backend": backend,
        "rms_norm_implementation": (
            rms_norm_implementation if patch_rms_norm else None
        ),
        "activation": activation,
        "patched": dict(sorted(counts.items())),
        "patched_classes": dict(sorted(patched_classes.items())),
        "parameter_identity_preserved": True,
    }


def install_diffusion_gemma_training_kernels(
    model: torch.nn.Module,
    backend: str,
) -> dict[str, Any]:
    """Install an opt-in kernel backend and return an auditable summary."""
    backend = str(backend).lower()
    if backend not in KERNEL_BACKENDS:
        raise ValueError(
            f"unsupported DiffusionGemma kernel backend {backend!r}; "
            f"expected one of {KERNEL_BACKENDS}"
        )
    if backend == "torch":
        return {
            "backend": "torch",
            "activation": _hidden_activation(model),
            "patched": {},
            "patched_classes": {},
            "parameter_identity_preserved": True,
        }
    return _install_liger_kernels(
        model,
        patch_rms_norm=backend in (
            "liger-rms",
            "liger-rms-cutedsl",
            "liger",
        ),
        patch_gelu_mul=backend in ("liger-gelu", "liger"),
        rms_norm_implementation=(
            "cutedsl" if backend == "liger-rms-cutedsl" else "triton"
        ),
    )


def install_diffusion_gemma_encoder_attention(
    model: torch.nn.Module,
    backend: str,
) -> dict[str, Any]:
    """Install optimized causal attention paths for the long text encoder.

    DiffusionGemma shares one text configuration between its encoder and
    decoder, so changing ``config._attn_implementation`` would also send the
    decoder's query-dependent 4-D cache mask to FlashAttention.  Patch only
    encoder attention instances instead.  The caller supplies the original
    right-padding mask as a 2-D mapping; FlashAttention's varlen path removes
    padding without changing the independent batch-axis document contract.
    """

    backend = str(backend).lower()
    if backend not in ENCODER_ATTENTION_BACKENDS:
        raise ValueError(
            f"unsupported DiffusionGemma encoder attention backend {backend!r}; "
            f"expected one of {ENCODER_ATTENTION_BACKENDS}"
        )
    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    if backend == "sdpa":
        model._arc_diffusion_gemma_encoder_attention_backend = backend
        return {
            "backend": backend,
            "patched_modules": 0,
            "mask_contract": "transformers-generated-additive",
            "parameter_identity_preserved": True,
        }

    try:
        from transformers.modeling_flash_attention_utils import (
            _flash_attention_forward,
        )
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            apply_rotary_pos_emb,
        )
    except Exception as error:  # pragma: no cover - runtime wheel dependent
        raise RuntimeError(
            "--encoder-attention-backend flash-attention-2 requires a "
            "Transformers build with DiffusionGemma and FlashAttention support"
        ) from error
    try:
        import flash_attn  # noqa: F401
    except Exception as error:  # pragma: no cover - runtime wheel dependent
        raise RuntimeError(
            "--encoder-attention-backend flash-attention-2 requires the "
            "flash-attn package"
        ) from error

    use_global_flex = backend == "hybrid-flex"
    patched = 0
    retained = 0
    layer_types: Counter[str] = Counter()
    for module in model.modules():
        if module.__class__.__name__ != "DiffusionGemmaEncoderTextAttention":
            continue
        # The five global layers use width-512 heads, beyond FA2's limit.
        # hybrid-flex patches them separately below; the ordinary FA2 backend
        # deliberately retains the reference SDPA implementation.
        if not module.is_sliding:
            if not use_global_flex:
                retained += 1
                layer_types["full-sdpa"] += 1
                continue
            if int(module.head_dim) != 512:
                raise RuntimeError(
                    "ARC global FlexAttention expects width-512 heads, got "
                    f"{module.head_dim} at encoder layer {module.layer_idx}"
                )
            if getattr(module, "_arc_flex_attention_encoder", False):
                continue

            def flex_encoder_forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                **kwargs,
            ):
                del kwargs
                # Avoid importing FlexAttention in the reference and ordinary
                # FA2 paths. BlockMask is intentionally checked by contract
                # instead of importing its private concrete type here.
                if attention_mask is None or not hasattr(
                    attention_mask, "kv_num_blocks"
                ):
                    raise ValueError(
                        "ARC global FlexAttention expects the prebuilt causal "
                        "BlockMask"
                    )

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                cos, sin = position_embeddings

                (
                    query_states,
                    raw_key_states,
                    raw_value_states,
                ) = _project_attention_qkv(self, hidden_states)
                query_states = query_states.view(hidden_shape)
                query_states = self.q_norm(query_states)
                query_states = apply_rotary_pos_emb(
                    query_states, cos, sin, unsqueeze_dim=2
                ).transpose(1, 2)

                raw_key_states = raw_key_states.view(hidden_shape)
                raw_value_states = (
                    raw_value_states.view(hidden_shape)
                    if raw_value_states is not None
                    else raw_key_states
                )
                value_states = self.v_norm(raw_value_states).transpose(1, 2)
                key_states = self.k_norm(raw_key_states)
                key_states = apply_rotary_pos_emb(
                    key_states, cos, sin, unsqueeze_dim=2
                ).transpose(1, 2)
                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        self.layer_idx,
                    )

                try:
                    attn_output = _compiled_global_flex_attention()(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask,
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        "DiffusionGemma global FlexAttention failed for "
                        f"Q={tuple(query_states.shape)}, "
                        f"K={tuple(key_states.shape)}, "
                        f"V={tuple(value_states.shape)}, "
                        f"dtype={query_states.dtype}"
                    ) from error
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                return self.o_proj(attn_output), None

            module.forward = MethodType(flex_encoder_forward, module)
            module._arc_flex_attention_encoder = True
            patched += 1
            layer_types["full-flex-attention"] += 1
            continue
        if int(module.head_dim) > 256:
            raise RuntimeError(
                "FlashAttention-2 cannot patch DiffusionGemma sliding "
                f"attention layer {module.layer_idx} with head_dim="
                f"{module.head_dim}; maximum supported width is 256"
            )
        if getattr(module, "_arc_flash_attention_2_encoder", False):
            continue

        def flash_encoder_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            **kwargs,
        ):
            if attention_mask is not None:
                if attention_mask.ndim != 2:
                    raise ValueError(
                        "ARC FlashAttention encoder expects the original 2-D "
                        "right-padding mask"
                    )
                if attention_mask.shape != hidden_states.shape[:2]:
                    raise ValueError(
                        "ARC FlashAttention encoder mask shape does not match "
                        "the encoder batch and sequence dimensions"
                    )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            cos, sin = position_embeddings

            (
                query_states,
                key_states,
                value_states,
            ) = _project_attention_qkv(self, hidden_states)
            query_states = query_states.view(hidden_shape)
            query_states = self.q_norm(query_states)
            query_states = apply_rotary_pos_emb(
                query_states, cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)

            key_states = key_states.view(hidden_shape)
            value_states = (
                value_states.view(hidden_shape)
                if value_states is not None
                else key_states
            )
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(
                key_states, cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            value_states = self.v_norm(value_states).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )

            # The utility consumes [batch, sequence, heads, head_dim].  Pin the
            # implementation explicitly because the shared config must remain
            # on SDPA for the decoder's 4-D prefix/canvas masks.
            flash_query = query_states.transpose(1, 2)
            flash_key = key_states.transpose(1, 2)
            flash_value = value_states.transpose(1, 2)
            try:
                attn_output = _flash_attention_forward(
                    flash_query,
                    flash_key,
                    flash_value,
                    attention_mask,
                    query_length=input_shape[1],
                    is_causal=self.is_causal,
                    dropout=(self.attention_dropout if self.training else 0.0),
                    softmax_scale=self.scaling,
                    sliding_window=self.sliding_window,
                    attn_implementation="flash_attention_2",
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "DiffusionGemma encoder FlashAttention failed for "
                    f"Q={tuple(flash_query.shape)}, K={tuple(flash_key.shape)}, "
                    f"V={tuple(flash_value.shape)}, dtype={flash_query.dtype}, "
                    f"sliding_window={self.sliding_window}"
                ) from error
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        module.forward = MethodType(flash_encoder_forward, module)
        module._arc_flash_attention_2_encoder = True
        patched += 1
        layer_types["sliding-flash-attention-2"] += 1

    if not patched:
        raise RuntimeError(
            "FlashAttention encoder patch found no DiffusionGemma encoder "
            "attention modules"
        )
    parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if parameter_ids_after != parameter_ids_before:
        raise RuntimeError(
            "FlashAttention encoder installation changed model Parameters"
        )
    model._arc_diffusion_gemma_encoder_attention_backend = backend
    return {
        "backend": backend,
        "patched_modules": patched,
        "retained_sdpa_modules": retained,
        "layer_types": dict(sorted(layer_types.items())),
        "mask_contract": (
            "full-block-causal-and-sliding-2d-varlen"
            if use_global_flex
            else "hybrid-full-additive-and-sliding-2d-varlen"
        ),
        "decoder_backend": "unchanged",
        "parameter_identity_preserved": True,
    }


def install_diffusion_gemma_decoder_attention(
    model: torch.nn.Module,
    backend: str,
) -> dict[str, Any]:
    """Install exact varlen FA2 for the 256-wide diffusion decoder layers.

    ARC's decoder mask is query invariant: every noisy-canvas query sees the
    same visible encoder prefix (or sliding suffix) plus the complete noisy
    canvas.  Representing that contract as a 2-D key-visibility mask lets
    FlashAttention unpad K/V exactly, without approximating the clean-canvas
    isolation rule.  The five 512-wide global layers remain on SDPA because
    FlashAttention-2 supports head dimensions only up to 256.
    """

    backend = str(backend).lower()
    if backend not in DECODER_ATTENTION_BACKENDS:
        raise ValueError(
            f"unsupported DiffusionGemma decoder attention backend {backend!r}; "
            f"expected one of {DECODER_ATTENTION_BACKENDS}"
        )
    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    if backend == "sdpa":
        model._arc_diffusion_gemma_decoder_attention_backend = backend
        return {
            "backend": backend,
            "patched_modules": 0,
            "mask_contract": "query-invariant-additive-4d",
            "parameter_identity_preserved": True,
        }

    try:
        from transformers.modeling_flash_attention_utils import (
            _flash_attention_forward,
        )
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            apply_rotary_pos_emb,
        )
    except Exception as error:  # pragma: no cover - runtime wheel dependent
        raise RuntimeError(
            "--decoder-attention-backend flash-attention-2 requires a "
            "Transformers build with DiffusionGemma and FlashAttention support"
        ) from error
    try:
        import flash_attn  # noqa: F401
    except Exception as error:  # pragma: no cover - runtime wheel dependent
        raise RuntimeError(
            "--decoder-attention-backend flash-attention-2 requires the "
            "flash-attn package"
        ) from error

    patched = 0
    retained = 0
    layer_types: Counter[str] = Counter()
    for module in model.modules():
        if module.__class__.__name__ != "DiffusionGemmaDecoderTextAttention":
            continue
        if not module.is_sliding:
            retained += 1
            layer_types["full-sdpa"] += 1
            continue
        if int(module.head_dim) > 256:
            raise RuntimeError(
                "FlashAttention-2 cannot patch DiffusionGemma sliding "
                f"decoder layer {module.layer_idx} with head_dim="
                f"{module.head_dim}; maximum supported width is 256"
            )
        if getattr(module, "_arc_flash_attention_2_decoder", False):
            continue

        def flash_decoder_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            **kwargs,
        ):
            if past_key_values is None:
                raise ValueError(
                    "the DiffusionGemma decoder requires the encoder KV cache"
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            cos, sin = position_embeddings

            (
                query_states,
                key_states,
                value_states,
            ) = _project_attention_qkv(self, hidden_states)
            query_states = query_states.view(hidden_shape)
            query_states = self.q_norm(query_states)
            query_states = apply_rotary_pos_emb(
                query_states, cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)

            key_states = key_states.view(hidden_shape)
            value_states = (
                value_states.view(hidden_shape)
                if value_states is not None
                else key_states
            )
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(
                key_states, cos, sin, unsqueeze_dim=2
            ).transpose(1, 2)
            value_states = self.v_norm(value_states).transpose(1, 2)
            key_states, value_states = self.append_to_cache(
                past_key_values, key_states, value_states
            )

            key_visibility = attention_mask
            if key_visibility is not None:
                if key_visibility.ndim == 4:
                    if (
                        key_visibility.shape[1] != 1
                        or key_visibility.shape[2] != input_shape[1]
                        or key_visibility.shape[3] != key_states.shape[2]
                    ):
                        raise ValueError(
                            "ARC decoder additive mask does not match Q/K lengths"
                        )
                    # ARC constructs a query-invariant visibility mask.  The
                    # first query therefore contains the complete key mask.
                    key_visibility = key_visibility[:, 0, 0, :] == 0
                elif key_visibility.ndim == 2:
                    if key_visibility.shape != (
                        input_shape[0],
                        key_states.shape[2],
                    ):
                        raise ValueError(
                            "ARC decoder key-visibility mask does not match K"
                        )
                    key_visibility = key_visibility.bool()
                else:
                    raise ValueError(
                        "ARC FlashAttention decoder expects a 2-D key mask "
                        "or query-invariant 4-D additive mask"
                    )

            flash_query = query_states.transpose(1, 2)
            flash_key = key_states.transpose(1, 2)
            flash_value = value_states.transpose(1, 2)
            try:
                attn_output = _flash_attention_forward(
                    flash_query,
                    flash_key,
                    flash_value,
                    key_visibility,
                    query_length=input_shape[1],
                    is_causal=False,
                    dropout=(self.attention_dropout if self.training else 0.0),
                    softmax_scale=self.scaling,
                    # Visibility was already reduced to the exact prefix/suffix
                    # plus canvas.  Applying FA's positional window again would
                    # change that set, so deliberately leave it disabled.
                    sliding_window=None,
                    attn_implementation="flash_attention_2",
                )
            except RuntimeError as error:
                raise RuntimeError(
                    "DiffusionGemma decoder FlashAttention failed for "
                    f"Q={tuple(flash_query.shape)}, K={tuple(flash_key.shape)}, "
                    f"V={tuple(flash_value.shape)}, dtype={flash_query.dtype}"
                ) from error
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None

        module.forward = MethodType(flash_decoder_forward, module)
        module._arc_flash_attention_2_decoder = True
        patched += 1
        layer_types["sliding-flash-attention-2"] += 1

    if not patched:
        raise RuntimeError(
            "FlashAttention decoder patch found no DiffusionGemma decoder "
            "attention modules"
        )
    parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if parameter_ids_after != parameter_ids_before:
        raise RuntimeError(
            "FlashAttention decoder installation changed model Parameters"
        )
    model._arc_diffusion_gemma_decoder_attention_backend = backend
    return {
        "backend": backend,
        "patched_modules": patched,
        "retained_sdpa_modules": retained,
        "layer_types": dict(sorted(layer_types.items())),
        "mask_contract": "hybrid-full-additive-and-sliding-2d-varlen",
        "encoder_backend": "unchanged",
        "parameter_identity_preserved": True,
    }


def _install_shared_input_linear_group(
    owner: torch.nn.Module,
    linears: tuple[torch.nn.Linear, ...],
    *,
    cache_attribute: str,
) -> None:
    """Execute adjacent same-input linears as one packed BF16 GEMM.

    DiffusionGemma evaluates these projections in a fixed left-to-right order.
    The first Linear computes every output, while the remaining Linear calls
    consume their slices from a short-lived owner-local cache.  Parameters and
    module objects stay untouched, so checkpoint/state-dict names and tied
    parameter identity remain stable.
    """

    if len(linears) < 2:
        raise ValueError("a packed projection group needs at least two linears")
    if any(not isinstance(linear, torch.nn.Linear) for linear in linears):
        raise TypeError("packed projection groups require torch.nn.Linear modules")
    in_features = linears[0].in_features
    if any(linear.in_features != in_features for linear in linears):
        raise ValueError("packed projections must share their input dimension")
    bias_modes = {linear.bias is None for linear in linears}
    if len(bias_modes) != 1:
        raise ValueError("packed projections must either all use bias or all omit it")

    output_sizes = tuple(int(linear.out_features) for linear in linears)
    first = linears[0]

    def packed_first_forward(_self, hidden_states):
        if getattr(owner, cache_attribute, None) is not None:
            raise RuntimeError(
                "DiffusionGemma packed projection cache was not consumed by "
                "the preceding forward"
            )
        packed_weight = torch.cat(tuple(linear.weight for linear in linears), dim=0)
        packed_bias = (
            None
            if first.bias is None
            else torch.cat(tuple(linear.bias for linear in linears), dim=0)
        )
        outputs = F.linear(hidden_states, packed_weight, packed_bias).split(
            output_sizes, dim=-1
        )
        setattr(
            owner,
            cache_attribute,
            [id(hidden_states), list(outputs[1:])],
        )
        return outputs[0]

    first.forward = MethodType(packed_first_forward, first)

    for projection_index, linear in enumerate(linears[1:], start=1):

        def packed_following_forward(
            _self,
            hidden_states,
            *,
            __projection_index=projection_index,
        ):
            cached = getattr(owner, cache_attribute, None)
            if cached is None or cached[0] != id(hidden_states):
                raise RuntimeError(
                    "DiffusionGemma packed projections were called out of order"
                )
            outputs = cached[1]
            output = outputs.pop(0)
            if __projection_index == len(linears) - 1:
                delattr(owner, cache_attribute)
            return output

        linear.forward = MethodType(packed_following_forward, linear)


def install_packed_shared_input_projections(
    model: torch.nn.Module,
    mode: str,
) -> dict[str, Any]:
    """Pack Q/K/V and/or dense gate/up projections without replacing modules."""

    mode = str(mode).lower()
    if mode not in PACKED_PROJECTION_MODES:
        raise ValueError(
            f"unsupported packed projection mode {mode!r}; "
            f"expected one of {PACKED_PROJECTION_MODES}"
        )
    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    counts: Counter[str] = Counter()
    if mode == "off":
        return {
            "mode": mode,
            "patched": {},
            "parameter_identity_preserved": True,
        }

    for module in model.modules():
        class_name = module.__class__.__name__
        if mode in ("dense", "all") and class_name == "DiffusionGemmaText4MLP":
            if getattr(module, "_arc_packed_dense_projections", False):
                continue
            _install_shared_input_linear_group(
                module,
                (module.gate_proj, module.up_proj),
                cache_attribute="_arc_packed_dense_outputs",
            )
            module._arc_packed_dense_projections = True
            counts["dense_gate_up"] += 1
        elif mode in ("attention", "all") and class_name in (
            "DiffusionGemmaEncoderTextAttention",
            "DiffusionGemmaDecoderTextAttention",
        ):
            if getattr(module, "_arc_packed_attention_projections", False):
                continue
            linears = [module.q_proj, module.k_proj]
            if module.v_proj is not None:
                linears.append(module.v_proj)
            _install_shared_input_linear_group(
                module,
                tuple(linears),
                cache_attribute="_arc_packed_attention_outputs",
            )
            module._arc_packed_attention_projections = True
            counts["attention_qkv" if len(linears) == 3 else "attention_qk"] += 1

    parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if parameter_ids_after != parameter_ids_before:
        raise RuntimeError("packed projection installation changed model Parameters")
    if mode in ("dense", "all") and not counts["dense_gate_up"]:
        raise RuntimeError("packed projection patch found no dense MLP modules")
    if mode in ("attention", "all") and not (
        counts["attention_qkv"] or counts["attention_qk"]
    ):
        raise RuntimeError("packed projection patch found no attention modules")
    return {
        "mode": mode,
        "patched": dict(sorted(counts.items())),
        "parameter_identity_preserved": True,
    }


def install_topk_router_softmax(model: torch.nn.Module) -> dict[str, Any]:
    """Normalize only selected router logits while retaining diagnostic output.

    ``softmax(scores)[topk] / sum(softmax(scores)[topk])`` is algebraically
    equal to ``softmax(topk(scores))`` because the full-vocabulary denominator
    cancels.  The first return value remains the reference full FP32 softmax,
    computed without an autograd graph because ARC training has no router aux
    loss.  Main-loss gradients still flow through the selected logits.
    """

    parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "DiffusionGemmaTextRouter":
            continue
        if getattr(module, "_arc_topk_router_softmax", False):
            continue

        def topk_router_forward(self, hidden_states):
            normalized = self.norm(hidden_states)
            normalized = normalized * self.scale * self.scalar_root_size
            expert_scores = self.proj(normalized)
            top_k_scores, top_k_index = torch.topk(
                expert_scores,
                k=self.config.top_k_experts,
                dim=-1,
            )
            top_k_weights = torch.softmax(top_k_scores, dim=-1, dtype=torch.float32)
            top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
            with torch.no_grad():
                router_probabilities = torch.softmax(
                    expert_scores, dim=-1, dtype=torch.float32
                )
            return router_probabilities, top_k_weights, top_k_index

        module.forward = MethodType(topk_router_forward, module)
        module._arc_topk_router_softmax = True
        count += 1

    if not count:
        raise RuntimeError("top-k router patch found no DiffusionGemma routers")
    parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if parameter_ids_after != parameter_ids_before:
        raise RuntimeError("top-k router installation changed model Parameters")
    return {
        "patched_routers": count,
        "diagnostic_probabilities": "full-softmax-detached",
        "parameter_identity_preserved": True,
    }


def nonzero_dropout_settings(model: torch.nn.Module) -> list[str]:
    """Return every active dropout source that makes RNG-free replay unsafe."""
    active: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            active.add(f"module:{name or '<root>'}.p={float(module.p)}")

    config = getattr(model, "config", None)
    payload = config.to_dict() if hasattr(config, "to_dict") else {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if "dropout" in str(key).lower() and isinstance(child, (int, float)):
                    if float(child) != 0.0:
                        active.add(f"config:{child_path}={float(child)}")
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")
    return sorted(active)


def assert_rng_free_checkpoint_replay_safe(model: torch.nn.Module) -> None:
    """Reject ``preserve_rng_state=False`` when model dropout is active."""
    active = nonzero_dropout_settings(model)
    if active:
        raise RuntimeError(
            "checkpoint RNG preservation can only be disabled for a dropout-free "
            "DiffusionGemma model; active settings: " + ", ".join(active)
        )
