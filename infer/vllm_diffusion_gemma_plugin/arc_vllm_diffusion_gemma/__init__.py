"""vLLM plugin registration kept lazy for multiprocessing-safe startup."""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DiffusionGemmaForBlockDiffusion",
        "arc_vllm_diffusion_gemma.model:ArcDiffusionGemmaForConditionalGeneration",
    )


__all__ = ["register"]
