#!/usr/bin/env python3
"""Runtime compatibility for DiffusionGemma artifacts without vision weights.

Transformers' DiffusionGemma configuration permits ``vision_config=None``,
but some generated modeling releases still instantiate the vision tower and
multimodal projector unconditionally.  This narrowly patches the native
encoder only for text-only configurations.  Ordinary multimodal checkpoints
continue through the original constructor and methods unchanged.
"""

from __future__ import annotations

from typing import Any


_PATCH_MARKER = "_arc_text_only_runtime_installed"


def install_diffusion_gemma_text_only_runtime(modeling_module: Any | None = None):
    """Make native DiffusionGemma honor ``vision_config=None``.

    Returns the patched encoder class.  Calling this function repeatedly is
    safe.  ``modeling_module`` is injectable for focused unit tests.
    """
    if modeling_module is None:
        from transformers.models.diffusion_gemma import (
            modeling_diffusion_gemma as modeling_module,
        )

    encoder_class = modeling_module.DiffusionGemmaEncoderModel
    if getattr(encoder_class, _PATCH_MARKER, False):
        return encoder_class

    original_init = encoder_class.__init__
    original_get_image_features = encoder_class.get_image_features
    original_get_placeholder_mask = encoder_class.get_placeholder_mask

    def text_only_aware_init(self, config):
        if config.vision_config is not None:
            return original_init(self, config)
        modeling_module.DiffusionGemmaPreTrainedModel.__init__(self, config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = modeling_module.DiffusionGemmaEncoderTextModel(
            config=config.text_config
        )
        self.vision_tower = None
        self.embed_vision = None
        self.post_init()

    def text_only_aware_get_image_features(self, pixel_values, *args, **kwargs):
        if self.vision_tower is None:
            raise RuntimeError(
                "this DiffusionGemma artifact is text-only and has no vision tower"
            )
        return original_get_image_features(self, pixel_values, *args, **kwargs)

    def text_only_aware_get_placeholder_mask(
        self,
        input_ids=None,
        inputs_embeds=None,
    ):
        if self.config.image_token_id is not None:
            return original_get_placeholder_mask(self, input_ids, inputs_embeds)
        import torch

        if input_ids is not None:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        if inputs_embeds is not None:
            return torch.zeros(
                inputs_embeds.shape[:-1],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        raise ValueError("input_ids or inputs_embeds is required")

    encoder_class.__init__ = text_only_aware_init
    encoder_class.get_image_features = text_only_aware_get_image_features
    encoder_class.get_placeholder_mask = text_only_aware_get_placeholder_mask
    setattr(encoder_class, _PATCH_MARKER, True)
    return encoder_class


__all__ = ["install_diffusion_gemma_text_only_runtime"]
