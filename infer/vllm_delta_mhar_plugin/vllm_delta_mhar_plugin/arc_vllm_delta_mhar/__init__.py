"""vLLM plugin registration for routed ARC Qwen checkpoints."""

from __future__ import annotations


PLUGIN_NAME = "arc_delta_mhar"
QWEN3_ARCHITECTURE = "ArcQwen3DeltaMHARForCausalLM"
QWEN35_ARCHITECTURE = "ArcQwen3_5DeltaMHARForCausalLM"
QWEN35_TEXT_ARCHITECTURE = "Qwen3_5ForCausalLM"


def register() -> None:
    """Register routed models and vLLM 0.23's missing Qwen3.5 text alias."""
    from vllm import ModelRegistry

    # The targeted vLLM 0.23.1 snapshot ships the native text-only executor
    # but omits its public HF architecture from ModelRegistry.  Generic
    # ``*ForCausalLM`` normalization then incorrectly selects multimodal
    # Qwen3_5ForConditionalGeneration for a Qwen3_5TextConfig checkpoint.
    # The thin plugin wrapper also supplies the hybrid GDN and text-only MRoPE
    # protocols omitted from that native executor in this vLLM snapshot.
    ModelRegistry.register_model(
        QWEN35_TEXT_ARCHITECTURE,
        "arc_vllm_delta_mhar.model:ArcQwen3_5TextForCausalLM",
    )
    ModelRegistry.register_model(
        QWEN3_ARCHITECTURE,
        "arc_vllm_delta_mhar.model:ArcQwen3DeltaMHARForCausalLM",
    )
    ModelRegistry.register_model(
        QWEN35_ARCHITECTURE,
        "arc_vllm_delta_mhar.model:ArcQwen3_5DeltaMHARForCausalLM",
    )


__all__ = [
    "PLUGIN_NAME",
    "QWEN3_ARCHITECTURE",
    "QWEN35_ARCHITECTURE",
    "QWEN35_TEXT_ARCHITECTURE",
    "register",
]
