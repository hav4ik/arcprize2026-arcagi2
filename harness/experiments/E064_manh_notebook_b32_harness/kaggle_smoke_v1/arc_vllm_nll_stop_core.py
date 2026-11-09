"""Backend-light cumulative-NLL state used by ARC's vLLM processor."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


ARC_TOKENS = tuple(range(11)) + (15,)
ARC_NON_EOS_TOKENS = tuple(range(11))
EOS_TOKEN_ID = 15
ARC_CUMULATIVE_NLL_EXTRA_KEY = "arc_cumulative_nll_stop"


def cumulative_nll_stop_spec(
    initial_nll: float,
    max_nll: float,
) -> dict[str, float]:
    """Return the serializable per-request processor payload."""
    initial_nll = float(initial_nll)
    max_nll = float(max_nll)
    if not math.isfinite(initial_nll) or initial_nll < 0.0:
        raise ValueError("ARC cumulative-NLL initial score must be finite and >= 0")
    if not math.isfinite(max_nll) or max_nll <= 0.0:
        raise ValueError("ARC cumulative-NLL cutoff must be finite and > 0")
    return {"initial_nll": initial_nll, "max_nll": max_nll}


class ArcCumulativeNllRequest:
    """Request-local exact NLL state consumed by the vLLM adapter."""

    def __init__(self, initial_nll: float, max_nll: float) -> None:
        self.initial_nll = float(initial_nll)
        self.max_nll = float(max_nll)
        self._cumulative_nll: torch.Tensor | None = None
        self._previous_logprobs: torch.Tensor | None = None
        self._processed_output_tokens = 0
        self._resumed_without_prefix_nll = False
        self._non_arc_mask: torch.Tensor | None = None
        self._non_eos_token_ids: torch.Tensor | None = None

    @property
    def cumulative_nll(self) -> float:
        """Expose state for focused CPU tests; this synchronizes CUDA."""
        if self._cumulative_nll is None:
            return self.initial_nll
        return float(self._cumulative_nll.item())

    @property
    def resumed_without_prefix_nll(self) -> bool:
        """Whether vLLM recreated this processor after request preemption."""
        return self._resumed_without_prefix_nll

    def _initialize(self, logits: torch.Tensor) -> None:
        self._cumulative_nll = torch.tensor(
            self.initial_nll,
            dtype=torch.float64,
            device=logits.device,
        )
        non_arc_mask = torch.ones(
            logits.shape[-1],
            dtype=torch.bool,
            device=logits.device,
        )
        non_arc_mask[list(ARC_TOKENS)] = False
        self._non_arc_mask = non_arc_mask
        self._non_eos_token_ids = torch.tensor(
            ARC_NON_EOS_TOKENS,
            dtype=torch.long,
            device=logits.device,
        )

    def __call__(
        self,
        output_token_ids: Sequence[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 1:
            raise ValueError(
                f"ARC cumulative-NLL processor expects 1D logits, got {logits.shape}"
            )
        if logits.shape[0] <= EOS_TOKEN_ID:
            raise ValueError(
                "ARC cumulative-NLL processor requires token IDs through 15"
            )
        if self._cumulative_nll is None:
            self._initialize(logits)
        assert self._cumulative_nll is not None
        assert self._non_arc_mask is not None
        assert self._non_eos_token_ids is not None

        output_count = len(output_token_ids)
        if self._previous_logprobs is not None:
            new_tokens = output_count - self._processed_output_tokens
            if new_tokens not in (0, 1):
                raise RuntimeError(
                    "ARC cumulative-NLL stopping supports at most one sampled "
                    f"token between processor calls, got {new_tokens}"
                )
            if new_tokens == 1:
                sampled_token = int(output_token_ids[-1])
                if sampled_token not in ARC_TOKENS:
                    raise RuntimeError(
                        "ARC constrained rollout sampled non-ARC token "
                        f"{sampled_token}"
                    )
                # Returned vLLM raw logprobs are FP32. Promote each term
                # before accumulating so this follows the CPU DFS's
                # Python-float sum. vLLM may sample the same scheduler step
                # more than once before appending its token; a zero-token
                # delta refreshes the pending logits below without adding NLL.
                self._cumulative_nll = (
                    self._cumulative_nll
                    - self._previous_logprobs[sampled_token].to(torch.float64)
                )
        elif output_count != 0:
            # vLLM removes a request-level processor when KV pressure preempts
            # its request and constructs a fresh one when that request resumes.
            # The resumed processor sees the already-sampled token IDs, but the
            # API does not provide their historical logits. Start from the
            # request's original NLL and account only for newly observed suffix
            # tokens. Because token NLLs are non-negative, this is a lower bound
            # on the true score: it can delay an in-engine stop, but can never
            # prune a valid DFS child. The CPU receives every raw position
            # logprob and remains the exact authority when the request returns.
            self._processed_output_tokens = output_count
            self._resumed_without_prefix_nll = True

        raw_logprobs = torch.log_softmax(
            logits,
            dim=-1,
            dtype=torch.float32,
        )
        best_non_eos_logprob = raw_logprobs.index_select(
            0,
            self._non_eos_token_ids,
        ).max()
        should_stop = (
            self._cumulative_nll
            - best_non_eos_logprob.to(torch.float64)
            >= self.max_nll
        )

        # Match allowed_token_ids=ARC_TOKENS on the ordinary path. When no
        # non-EOS child survives, mask those 11 tokens as well so the explicit
        # stop-token rule terminates the request. Raw logprobs were captured
        # by vLLM before this returned tensor is sampled.
        constrained = logits.masked_fill(self._non_arc_mask, -float("inf"))
        non_eos_logits = constrained.index_select(0, self._non_eos_token_ids)
        constrained.index_copy_(
            0,
            self._non_eos_token_ids,
            torch.where(
                should_stop,
                torch.full_like(non_eos_logits, -float("inf")),
                non_eos_logits,
            ),
        )

        self._previous_logprobs = raw_logprobs.detach()
        self._processed_output_tokens = output_count
        return constrained


__all__ = [
    "ARC_CUMULATIVE_NLL_EXTRA_KEY",
    "ARC_TOKENS",
    "ArcCumulativeNllRequest",
    "EOS_TOKEN_ID",
    "cumulative_nll_stop_spec",
]
