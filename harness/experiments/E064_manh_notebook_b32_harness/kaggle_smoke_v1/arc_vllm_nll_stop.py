"""vLLM adapter for exact ARC cumulative-NLL leaf stopping.

The CPU DFS owns branching and candidate collection. This processor only
prevents a constrained greedy rollout from continuing after its best non-EOS
ARC child has crossed the same cumulative-NLL threshold. vLLM computes the
requested raw logprobs before applying custom processors, so forcing EOS here
does not alter the probabilities returned to the CPU search.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from arc_vllm_nll_stop_core import (
    ARC_CUMULATIVE_NLL_EXTRA_KEY,
    ArcCumulativeNllRequest,
    EOS_TOKEN_ID,
    cumulative_nll_stop_spec,
)
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


def _parse_spec(params: SamplingParams) -> tuple[float, float] | None:
    extra_args = params.extra_args or {}
    payload = extra_args.get(ARC_CUMULATIVE_NLL_EXTRA_KEY)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"{ARC_CUMULATIVE_NLL_EXTRA_KEY} must be a mapping")
    try:
        normalized = cumulative_nll_stop_spec(
            payload["initial_nll"],
            payload["max_nll"],
        )
    except KeyError as error:
        raise ValueError(
            f"{ARC_CUMULATIVE_NLL_EXTRA_KEY} requires initial_nll and max_nll"
        ) from error
    return normalized["initial_nll"], normalized["max_nll"]


class ArcCumulativeNllLogitsProcessor(AdapterLogitsProcessor):
    """vLLM V1 adapter processor enabled by one SamplingParams payload."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        spec = _parse_spec(sampling_params)
        if spec is None:
            return
        if sampling_params.temperature != 0.0:
            raise ValueError("ARC cumulative-NLL stopping requires greedy sampling")
        if sampling_params.allowed_token_ids is not None:
            raise ValueError(
                "ARC cumulative-NLL stopping owns the ARC-token constraint; "
                "allowed_token_ids must be unset"
            )
        if EOS_TOKEN_ID not in (sampling_params.stop_token_ids or ()):
            raise ValueError(
                "ARC cumulative-NLL stopping requires token 15 as a stop token"
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Callable[[list[int], torch.Tensor], torch.Tensor] | None:
        spec = _parse_spec(params)
        if spec is None:
            return None
        return ArcCumulativeNllRequest(*spec)


__all__ = [
    "ARC_CUMULATIVE_NLL_EXTRA_KEY",
    "ArcCumulativeNllLogitsProcessor",
    "cumulative_nll_stop_spec",
]
