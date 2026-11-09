"""LoRA-capable wrapper around vLLM's native DiffusionGemma implementation."""

from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration,
    DiffusionGemmaProcessingInfo,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4MultiModalProcessor,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsLoRA
from vllm.multimodal import MULTIMODAL_REGISTRY


class ArcDiffusionGemmaProcessingInfo(DiffusionGemmaProcessingInfo):
    """Treat a vision-pruned ARC checkpoint as an explicitly text-only model."""

    def _is_text_only(self) -> bool:
        return getattr(self.get_hf_config(), "vision_config", None) is None

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        if self._is_text_only():
            return {}
        return super().get_supported_mm_limits()

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        if self._is_text_only():
            return {}
        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=ArcDiffusionGemmaProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class ArcDiffusionGemmaForConditionalGeneration(
    DiffusionGemmaForConditionalGeneration,
    SupportsLoRA,
    MixtureOfExperts,
):
    """Expose the existing Gemma4 FusedMoE stack to vLLM's LoRA manager.

    The native model already has the correct checkpoint mapper and packed
    qkv/gate-up mappings. It previously omitted only the marker/protocol state
    used by vLLM when replacing linear and FusedMoE modules with LoRA-aware
    wrappers.
    """

    # Explicit class attributes make model-registry inspection succeed before
    # an engine instance exists. The inherited packed mapping remains the
    # authoritative projection layout.
    supports_lora = True
    embedding_modules: dict[str, str] = {}
    lora_skip_prefixes = ["vision_tower.", "embed_vision.", "self_conditioning."]

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.moe_layers: list[nn.Module] = []
        example_moe = None
        for layer in self.model.layers:
            moe = getattr(layer, "moe", None)
            experts = getattr(moe, "experts", None)
            if experts is not None:
                example_moe = moe
                self.moe_layers.append(experts)

        self.num_moe_layers = len(self.moe_layers)
        expert_count = int(getattr(example_moe, "num_experts", 0))
        self.num_logical_experts = expert_count
        self.num_physical_experts = expert_count
        self.num_local_physical_experts = expert_count
        self.num_routed_experts = expert_count
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_redundant_experts = 0

        if not self.moe_layers:
            raise RuntimeError(
                "ARC DiffusionGemma vLLM plugin found no Gemma4 FusedMoE layers"
            )


__all__ = [
    "ArcDiffusionGemmaForConditionalGeneration",
    "ArcDiffusionGemmaProcessingInfo",
]
