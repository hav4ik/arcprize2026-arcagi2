"""Native vLLM executors for B4S1 routed Qwen3 and Qwen3.5."""

from __future__ import annotations

from collections.abc import Iterable
import inspect
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .config import read_delta_mhar_spec, spec_from_hf_config
from .ops import (
    delta_mhar_mix,
    install_compile_stable_qwen35_fused_qk,
)


# vLLM 0.23's native Qwen3.5 fused-QK helper launches a raw Triton function
# inside the compiled graph.  Install the dispatcher wrapper before any
# Qwen3.5 layer is constructed so AOT artifacts can be reused by the second
# independent vLLM service process.
install_compile_stable_qwen35_fused_qk()


def _require_single_gpu(vllm_config: VllmConfig) -> None:
    parallel = vllm_config.parallel_config
    tensor_parallel_size = int(parallel.tensor_parallel_size)
    pipeline_parallel_size = int(parallel.pipeline_parallel_size)
    if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
        raise RuntimeError(
            "Delta-MH vLLM inference currently requires TP=1 and PP=1; "
            f"got TP={tensor_parallel_size}, PP={pipeline_parallel_size}"
        )


def _checkpoint_path(vllm_config: VllmConfig) -> Path:
    model_path = Path(vllm_config.model_config.model).expanduser().resolve()
    if not model_path.is_dir():
        raise RuntimeError(
            "Delta-MH vLLM inference requires a local checkpoint directory, "
            f"got {model_path}"
        )
    return model_path


def _weights_with_routing_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    routing_weight_paths: tuple[Path, ...],
    *,
    embedded: bool,
) -> Iterable[tuple[str, torch.Tensor]]:
    # Ordinary vLLM loading already yields tensors embedded in HF model shards.
    # Only legacy sidecar checkpoints need a second iterator appended.
    if embedded:
        return weights

    from safetensors import safe_open

    def routing_weights() -> Iterable[tuple[str, torch.Tensor]]:
        for path in routing_weight_paths:
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if "._unsloth_delta_mhar." in name:
                        yield name, checkpoint.get_tensor(name)

    return chain(weights, routing_weights())


def _without_weight_prefixes(
    weights: Iterable[tuple[str, torch.Tensor]],
    prefixes: Iterable[str],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Filter checkpoint-only branches across vLLM loader API versions.

    Older vLLM builds accepted ``skip_prefixes`` in
    :class:`AutoWeightsLoader`, while recent development builds replaced that
    constructor contract with unexpected-weight controls. Filtering the lazy
    checkpoint iterator is equivalent for the two branches used here and
    keeps this plugin compatible with both loader contracts.
    """
    normalized = tuple(prefixes)
    if not normalized:
        return weights
    return (
        (name, tensor)
        for name, tensor in weights
        if not name.startswith(normalized)
    )


class DeltaMHARRouter(nn.Module):
    """Frozen FP32 depth router loaded from the pre-SFT sidecar."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        variance_epsilon: float,
        fused: bool,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden size {hidden_size} is not divisible by {num_heads} heads"
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.variance_epsilon = float(variance_epsilon)
        self.fused = bool(fused)
        self.query = nn.Parameter(
            torch.empty(
                num_heads,
                hidden_size // num_heads,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.norm_weight = nn.Parameter(
            torch.empty(hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.null_source = nn.Parameter(
            torch.empty(hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.alpha = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        partial: torch.Tensor,
        delta_bank: torch.Tensor,
        delta_count: int,
    ) -> torch.Tensor:
        routed = delta_mhar_mix(
            partial=partial,
            delta_bank=delta_bank,
            delta_count=delta_count,
            query=self.query,
            norm_weight=self.norm_weight,
            null_source=self.null_source,
            num_heads=self.num_heads,
            variance_epsilon=self.variance_epsilon,
            fused=self.fused,
        )
        return partial + self.alpha.to(partial.dtype) * routed


class DeltaMHARCarry(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.delta = nn.Parameter(
            torch.empty(hidden_size, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, mixer_output: torch.Tensor) -> torch.Tensor:
        return mixer_output + mixer_output * self.delta.to(mixer_output.dtype)


class DeltaMHARSingleSiteLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        variance_epsilon: float,
        fused: bool,
    ) -> None:
        super().__init__()
        self.attention = DeltaMHARRouter(
            hidden_size=hidden_size,
            num_heads=num_heads,
            variance_epsilon=variance_epsilon,
            fused=fused,
        )
        self.carry = DeltaMHARCarry(hidden_size)


def _routing_layer(config: Any) -> DeltaMHARSingleSiteLayer:
    declaration = spec_from_hf_config(config)
    return DeltaMHARSingleSiteLayer(
        hidden_size=int(config.hidden_size),
        num_heads=int(declaration["num_heads"]),
        variance_epsilon=float(config.rms_norm_eps),
        fused=bool(declaration["fused"]),
    )


class ArcQwen3DeltaMHARDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        self._unsloth_delta_mhar = _routing_layer(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        delta_bank: torch.Tensor,
        delta_count: int,
    ) -> torch.Tensor:
        residual = hidden_states
        routed_input = self._unsloth_delta_mhar.attention(
            residual, delta_bank, delta_count
        )
        normalized = self.input_layernorm(routed_input)
        attention_output = self.self_attn(
            positions=positions,
            hidden_states=normalized,
        )
        residual_after_attention = residual + attention_output
        mlp_input = routed_input + self._unsloth_delta_mhar.carry(attention_output)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_input))
        return residual_after_attention + mlp_output


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class ArcQwen3DeltaMHARModel(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=ArcQwen3DeltaMHARDecoderLayer,
        )
        declaration = spec_from_hf_config(self.config)
        self._arc_delta_mhar_block_size = int(declaration["block_size"])

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None or not get_pp_group().is_first_rank:
            raise RuntimeError("Delta-MH Qwen3 vLLM executor requires PP=1")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        # Keep the depth bank functional. Torch 2.13's AOT synthetic-base
        # wrapper flattens a preallocated bank mutated through ``copy_`` and
        # then violates the compiled custom-op stride guard at runtime.
        # The initial view is ignored while ``delta_count == 0``.
        delta_bank = hidden_states.unsqueeze(1)
        delta_count = 0
        block_start = hidden_states
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                positions,
                hidden_states,
                delta_bank,
                delta_count,
            )
            if (layer_index + 1) % self._arc_delta_mhar_block_size == 0:
                completed_delta = (hidden_states - block_start).unsqueeze(1)
                delta_bank = (
                    completed_delta
                    if delta_count == 0
                    else torch.cat((delta_bank, completed_delta), dim=1)
                )
                delta_count += 1
                block_start = hidden_states
        return self.norm(hidden_states)


class ArcQwen3DeltaMHARForCausalLM(Qwen3ForCausalLM):
    """Qwen3 causal LM with only the residual-stream model loop replaced."""

    allow_patterns_overrides = ["model*.safetensors"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_single_gpu(vllm_config)
        model_path = _checkpoint_path(vllm_config)
        spec = read_delta_mhar_spec(model_path)
        if spec.family != "qwen3":
            raise RuntimeError(f"Qwen3 executor received {spec.model_type}")
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self._arc_delta_mhar_weight_paths = spec.routing_weight_paths
        self._arc_delta_mhar_weights_embedded = spec.routing_weights_embedded
        self.model = ArcQwen3DeltaMHARModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _weights_with_routing_weights(
            weights,
            self._arc_delta_mhar_weight_paths,
            embedded=self._arc_delta_mhar_weights_embedded,
        )
        if self.config.tie_word_embeddings:
            weights = _without_weight_prefixes(weights, ["lm_head."])
        return AutoWeightsLoader(self).load_weights(weights)


class ArcQwen3_5DeltaMHARDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config, layer_type, prefix)
        if self.layer_scale:
            raise RuntimeError(
                "Delta-MH Qwen3.5 does not support layer_scale checkpoints"
            )
        mixer = (
            self.linear_attn
            if self.layer_type == "linear_attention"
            else self.self_attn
        )
        # vLLM 0.23 stable writes into an ``output`` tensor, while later
        # 0.23 wheel snapshots return the mixer tensor.  Resolve that API
        # difference once during construction so the compiled forward has no
        # exception-driven branch.
        self._arc_mixer_returns_tensor = (
            "output" not in inspect.signature(mixer.forward).parameters
        )
        self._unsloth_delta_mhar = _routing_layer(
            vllm_config.model_config.hf_text_config
        )

    def forward(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        delta_bank: torch.Tensor,
        delta_count: int,
    ) -> torch.Tensor:
        residual = hidden_states
        routed_input = self._unsloth_delta_mhar.attention(
            residual, delta_bank, delta_count
        )
        normalized = self.input_layernorm(routed_input)
        if self.layer_type == "linear_attention":
            if self._arc_mixer_returns_tensor:
                mixer_output = self.linear_attn(hidden_states=normalized)
            else:
                mixer_output = torch.empty_like(normalized)
                self.linear_attn(hidden_states=normalized, output=mixer_output)
        elif self.layer_type == "full_attention":
            if self._arc_mixer_returns_tensor:
                mixer_output = self.self_attn(
                    positions=positions,
                    hidden_states=normalized,
                )
            else:
                mixer_output = torch.empty_like(normalized)
                self.self_attn(
                    positions=positions,
                    hidden_states=normalized,
                    output=mixer_output,
                )
        else:  # pragma: no cover - constructor already validates this
            raise RuntimeError(f"Unsupported Qwen3.5 layer type {self.layer_type}")
        residual_after_mixer = residual + mixer_output
        mlp_input = routed_input + self._unsloth_delta_mhar.carry(mixer_output)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_input))
        return residual_after_mixer + mlp_output


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class ArcQwen3_5DeltaMHARModel(Qwen3_5Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Keep Qwen3_5Model's packed QKV/GDN/MLP weight loader while replacing
        # only construction and the residual-stream forward loop.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        self.num_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        def get_layer(prefix: str):
            index = extract_layer_index(prefix)
            return ArcQwen3_5DeltaMHARDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[index],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        declaration = spec_from_hf_config(config)
        self._arc_delta_mhar_block_size = int(declaration["block_size"])
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None or not get_pp_group().is_first_rank:
            raise RuntimeError("Delta-MH Qwen3.5 vLLM executor requires PP=1")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        # See ArcQwen3DeltaMHARModel: this must remain functional for Torch
        # 2.13 AOT compilation rather than mutating a lifted synthetic base.
        delta_bank = hidden_states.unsqueeze(1)
        delta_count = 0
        block_start = hidden_states
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                delta_bank=delta_bank,
                delta_count=delta_count,
            )
            if (layer_index + 1) % self._arc_delta_mhar_block_size == 0:
                completed_delta = (hidden_states - block_start).unsqueeze(1)
                delta_bank = (
                    completed_delta
                    if delta_count == 0
                    else torch.cat((delta_bank, completed_delta), dim=1)
                )
                delta_count += 1
                block_start = hidden_states
        return self.norm(hidden_states)


class ArcQwen3_5DeltaMHARForCausalLM(
    Qwen3_5ForCausalLM,
    IsHybrid,
    SupportsMRoPE,
):
    """Text-only hybrid Qwen3.5 with native GDN state and routed residuals."""

    allow_patterns_overrides = ["model*.safetensors"]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _require_single_gpu(vllm_config)
        model_path = _checkpoint_path(vllm_config)
        spec = read_delta_mhar_spec(model_path)
        if spec.family != "qwen3_5":
            raise RuntimeError(f"Qwen3.5 executor received {spec.model_type}")
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Delta-MH Qwen3.5 requires mamba_cache_mode='align'"
            )
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self._arc_delta_mhar_weight_paths = spec.routing_weight_paths
        self._arc_delta_mhar_weights_embedded = spec.routing_weights_embedded
        self.model = ArcQwen3_5DeltaMHARModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_text_config
        num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        """Return equal temporal/spatial axes for a text-only checkpoint."""
        if mm_features:
            raise ValueError("Text-only routed Qwen3.5 received multimodal features")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.view(1, -1).expand(3, -1).clone(), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = _weights_with_routing_weights(
            weights,
            self._arc_delta_mhar_weight_paths,
            embedded=self._arc_delta_mhar_weights_embedded,
        )
        weights = _without_weight_prefixes(weights, ["mtp."])
        return AutoWeightsLoader(self).load_weights(weights)


class ArcQwen3_5TextForCausalLM(
    Qwen3_5ForCausalLM,
    IsHybrid,
    SupportsMRoPE,
):
    """Complete vLLM 0.23's text-only Qwen3.5 runtime interfaces.

    The bundled executor builds the right dense text model, but it is absent
    from ModelRegistry and does not advertise the hybrid GDN or MRoPE
    protocols.  The V1 worker requires both protocols before allocating model
    state.  This wrapper leaves model construction and weight loading entirely
    in vLLM and supplies only the missing runtime contract.
    """

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        config = vllm_config.model_config.hf_text_config
        num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_speculative_tokens,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        """Use identical MRoPE axes for a text-only token sequence."""
        if mm_features:
            raise ValueError("Text-only Qwen3.5 received multimodal features")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.view(1, -1).expand(3, -1).clone(), 0


__all__ = [
    "ArcQwen3DeltaMHARForCausalLM",
    "ArcQwen3_5DeltaMHARForCausalLM",
    "ArcQwen3_5TextForCausalLM",
    "DeltaMHARCarry",
    "DeltaMHARRouter",
    "DeltaMHARSingleSiteLayer",
]
