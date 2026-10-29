"""Backend-independent ARC search state machines for vLLM inference.

vLLM is used only to provide raw, full-vocabulary token log-probabilities.
These classes retain the repository's cumulative-NLL threshold search, DFS
ordering, bounded frontier fallback, and fixed-shape recovery semantics.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


ARC_TOKENS = tuple(range(11)) + (15,)
DIGIT_TOKENS = tuple(range(10))
NEWLINE_TOKEN_ID = 10
EOS_TOKEN_ID = 15


@dataclass(frozen=True)
class SearchRequest:
    """One raw-logprob query owned by a search state."""

    request_id: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int = 1
    constrained_rollout: bool = False
    cumulative_nll: float = 0.0


def _logprob_value(value: Any) -> float:
    if hasattr(value, "logprob"):
        value = value.logprob
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite token logprob: {result}")
    return result


def _logprob_values_at_position(logprobs: Any, position: int) -> Mapping:
    """Return vLLM's unnormalized mapping at one output position."""
    if logprobs is None:
        raise ValueError("vLLM returned no logprobs")
    values = logprobs[position]
    if values is None:
        raise ValueError(f"vLLM returned no logprobs at position {position}")
    if not isinstance(values, Mapping):
        raise TypeError(
            "vLLM logprobs at position "
            f"{position} must be a mapping, got {type(values).__name__}"
        )
    return values


def logprobs_at_position(logprobs: Any, position: int = -1) -> dict[int, float]:
    """Normalize vLLM FlatLogprobs/list output at one position."""
    values = _logprob_values_at_position(logprobs, position)
    return {
        int(token_id): _logprob_value(value)
        for token_id, value in values.items()
    }


def requested_arc_logprobs(logprobs: Any, position: int = -1) -> dict[int, float]:
    """Extract all 12 ARC-token raw logprobs, ignoring a sampled duplicate."""
    values = _logprob_values_at_position(logprobs, position)
    normalized_keys = {int(token_id): token_id for token_id in values}
    missing = [
        token_id for token_id in ARC_TOKENS if token_id not in normalized_keys
    ]
    if missing:
        raise ValueError(
            f"vLLM response is missing requested ARC token IDs: {missing}"
        )
    result = {}
    for token_id in ARC_TOKENS:
        try:
            result[token_id] = _logprob_value(
                values[normalized_keys[token_id]]
            )
        except ValueError as exc:
            raise ValueError(
                f"Non-finite logprob for requested ARC token {token_id} "
                f"at position {position}"
            ) from exc
    return result


def answer_nll_from_prompt_logprobs(
    prompt_logprobs: Any,
    answer_token_ids: Sequence[int],
    query_length: int,
    cached_tokens: int = 0,
) -> float:
    """Sum answer NLL from a full or safely prefix-cached vLLM response.

    vLLM normally disables prefix-cache reads for prompt-logprob requests. If
    that guard is explicitly relaxed, the returned container contains only
    the recomputed suffix (plus its leading ``None`` sentinel). Taking the
    final answer-length positions remains exact only while the cache hit ends
    strictly before the answer. Callers must retry without cache otherwise.
    """
    if query_length < 1:
        raise ValueError("query_length must be positive")
    if cached_tokens < 0:
        raise ValueError("cached_tokens must be non-negative")
    if not answer_token_ids:
        return 0.0
    expected_length = query_length + len(answer_token_ids)
    if cached_tokens == 0 and len(prompt_logprobs) < expected_length:
        raise ValueError(
            "Prompt-logprob response is shorter than query+answer: "
            f"{len(prompt_logprobs)} < {expected_length}"
        )
    if cached_tokens >= query_length:
        raise ValueError(
            "Prefix-cache hit reaches the answer boundary; cached answer "
            "logprobs are unavailable"
        )
    if len(prompt_logprobs) < len(answer_token_ids) + 1:
        raise ValueError(
            "Prefix-cached prompt-logprob response does not contain every "
            "answer position"
        )
    first_answer_position = len(prompt_logprobs) - len(answer_token_ids)
    total = 0.0
    for offset, token_id in enumerate(answer_token_ids):
        position = first_answer_position + offset
        values = logprobs_at_position(prompt_logprobs, position)
        if int(token_id) not in values:
            raise ValueError(
                f"Prompt logprobs at position {position} omit actual token "
                f"{token_id}"
            )
        total -= values[int(token_id)]
    return total


def candidate_scores(
    token_logprobs: Mapping[int, float],
    accumulated_score: float,
    token_ids: Sequence[int] = ARC_TOKENS,
) -> list[tuple[float, int]]:
    """Convert raw logprobs into sorted cumulative NLL candidates."""
    missing = [token_id for token_id in token_ids if token_id not in token_logprobs]
    if missing:
        raise ValueError(f"Missing token logprobs: {missing}")
    return sorted(
        (
            float(accumulated_score) - float(token_logprobs[token_id]),
            int(token_id),
        )
        for token_id in token_ids
    )


@dataclass
class _DfsNode:
    path: tuple[int, ...]
    score: float
    remaining_tokens: int
    candidates: list[tuple[float, int]] | None = None
    next_candidate: int = 0


class RecursiveOriginSearch:
    """Explicit stack equivalent of one origin in ``turbo_dfs``."""

    def __init__(
        self,
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        max_score: float,
        deadline: float,
        *,
        origin: int = 0,
        initial_path: Sequence[int] = (),
        initial_score: float = 0.0,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.prefix_token_ids = tuple(map(int, prefix_token_ids))
        self.max_score = float(max_score)
        self.deadline = float(deadline)
        self.origin = int(origin)
        self.beams: list[tuple[float, list[int]]] = []
        self._stack = [
            _DfsNode(
                path=tuple(map(int, initial_path)),
                score=float(initial_score),
                remaining_tokens=int(max_new_tokens),
            )
        ]
        self._outstanding = False

    @property
    def done(self) -> bool:
        return not self._stack

    def expire(self, now: float | None = None) -> None:
        if (time.time() if now is None else now) >= self.deadline:
            self._stack.clear()
            self._outstanding = False

    def prompt(self, now: float | None = None) -> tuple[int, ...] | None:
        self.expire(now)
        if self.done or self._outstanding:
            return None
        node = self._stack[-1]
        if node.candidates is not None:
            self._advance()
            if self.done:
                return None
            node = self._stack[-1]
        self._outstanding = True
        return self.prefix_token_ids + node.path

    def accept(
        self,
        token_logprobs: Mapping[int, float],
        now: float | None = None,
    ) -> None:
        if not self._outstanding or self.done:
            raise RuntimeError("Recursive search has no outstanding query")
        self._outstanding = False
        self.expire(now)
        if self.done:
            return
        node = self._stack[-1]
        children = []
        for score, token_id in candidate_scores(token_logprobs, node.score):
            if score >= self.max_score:
                continue
            if token_id == EOS_TOKEN_ID:
                self.beams.append((score, list(node.path + (token_id,))))
            elif node.remaining_tokens > 1:
                children.append((score, token_id))
        node.candidates = children
        self._advance()

    def _advance(self) -> None:
        while self._stack:
            node = self._stack[-1]
            if node.candidates is None:
                return
            if node.next_candidate < len(node.candidates):
                score, token_id = node.candidates[node.next_candidate]
                node.next_candidate += 1
                self._stack.append(
                    _DfsNode(
                        path=node.path + (token_id,),
                        score=score,
                        remaining_tokens=node.remaining_tokens - 1,
                    )
                )
                return
            self._stack.pop()


class RecursiveSearchBatch:
    """Round-robin collection of independent DFS origins."""

    def __init__(
        self,
        prefix_token_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        max_score: float,
        deadline: float,
    ) -> None:
        self.origins = [
            RecursiveOriginSearch(
                prefix,
                max_new_tokens,
                max_score,
                deadline,
                origin=origin,
            )
            for origin, prefix in enumerate(prefix_token_ids)
        ]
        self._request_counter = 0
        self._pending: dict[int, RecursiveOriginSearch] = {}

    @classmethod
    def from_frontier_nodes(
        cls,
        nodes: Sequence["_FrontierNode"],
        prefixes: Sequence[Sequence[int]],
        max_score: float,
        deadline: float,
    ) -> "RecursiveSearchBatch":
        obj = cls.__new__(cls)
        obj.origins = [
            RecursiveOriginSearch(
                prefixes[node.origin],
                node.remaining_tokens,
                max_score,
                deadline,
                origin=node.origin,
                initial_path=node.path,
                initial_score=node.score,
            )
            for node in nodes
        ]
        obj._request_counter = 0
        obj._pending = {}
        return obj

    @property
    def done(self) -> bool:
        return not self._pending and all(state.done for state in self.origins)

    def take_requests(
        self,
        limit: int,
        now: float | None = None,
    ) -> list[SearchRequest]:
        requests = []
        for state in self.origins:
            if len(requests) >= limit:
                break
            prompt = state.prompt(now)
            if prompt is None:
                continue
            request_id = self._request_counter
            self._request_counter += 1
            self._pending[request_id] = state
            requests.append(SearchRequest(request_id, prompt))
        return requests

    def accept(
        self,
        request_id: int,
        token_logprobs: Mapping[int, float],
        now: float | None = None,
    ) -> None:
        try:
            state = self._pending.pop(int(request_id))
        except KeyError as exc:
            raise KeyError(f"Unknown recursive request ID {request_id}") from exc
        state.accept(token_logprobs, now)

    def results(self) -> list[tuple[int, list[tuple[float, list[int]]]]]:
        if not self.done:
            raise RuntimeError("Search results requested before completion")
        by_origin: dict[int, list[tuple[float, list[int]]]] = defaultdict(list)
        for state in self.origins:
            by_origin[state.origin].extend(state.beams)
        return [
            (origin, sorted(beams, key=lambda item: item[0]))
            for origin, beams in sorted(by_origin.items())
            if beams
        ]


@dataclass(frozen=True)
class _RolloutNode:
    path: tuple[int, ...]
    score: float
    remaining_tokens: int


class RecursiveRolloutOriginSearch:
    """DFS origin that consumes one constrained greedy leaf per request.

    vLLM returns raw log probabilities before applying ``allowed_token_ids``.
    The sampled tokens can therefore follow the best ARC-token child while
    every alternative child is retained on the explicit DFS stack with its
    original full-vocabulary cumulative NLL.
    """

    def __init__(
        self,
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        max_score: float,
        deadline: float,
        *,
        origin: int = 0,
        rollout_chunk_tokens: int = 32,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if rollout_chunk_tokens < 1:
            raise ValueError("rollout_chunk_tokens must be positive")
        self.prefix_token_ids = tuple(map(int, prefix_token_ids))
        self.max_score = float(max_score)
        self.deadline = float(deadline)
        self.origin = int(origin)
        self.rollout_chunk_tokens = int(rollout_chunk_tokens)
        self.beams: list[tuple[float, list[int]]] = []
        self._stack = [_RolloutNode((), 0.0, int(max_new_tokens))]
        self._outstanding: dict[int, _RolloutNode] = {}
        self._request_counter = 0

    @property
    def done(self) -> bool:
        return not self._stack and not self._outstanding

    @property
    def outstanding_count(self) -> int:
        return len(self._outstanding)

    def expire(self, now: float | None = None) -> None:
        if (time.time() if now is None else now) >= self.deadline:
            self._stack.clear()

    def request(
        self,
        now: float | None = None,
    ) -> tuple[int, tuple[int, ...], int, float] | None:
        self.expire(now)
        if not self._stack:
            return None
        node = self._stack.pop()
        request_id = self._request_counter
        self._request_counter += 1
        self._outstanding[request_id] = node
        return (
            request_id,
            self.prefix_token_ids + node.path,
            min(node.remaining_tokens, self.rollout_chunk_tokens),
            node.score,
        )

    def accept(
        self,
        request_id: int,
        sampled_token_ids: Sequence[int],
        position_logprobs: Sequence[Mapping[int, float]],
        now: float | None = None,
    ) -> None:
        try:
            node = self._outstanding.pop(int(request_id))
        except KeyError as exc:
            raise KeyError(
                f"Rollout search has no outstanding query {request_id}"
            ) from exc
        self.expire(now)
        if (time.time() if now is None else now) >= self.deadline:
            return
        if len(sampled_token_ids) != len(position_logprobs):
            raise ValueError(
                "Rollout sampled-token/logprob length mismatch: "
                f"{len(sampled_token_ids)} != {len(position_logprobs)}"
            )
        if not sampled_token_ids:
            raise ValueError("vLLM returned an empty constrained rollout")

        path = node.path
        score = node.score
        remaining_tokens = node.remaining_tokens
        for sampled_token_id, token_logprobs in zip(
            sampled_token_ids,
            position_logprobs,
        ):
            if remaining_tokens < 1:
                break
            candidates = [
                (candidate_score, token_id)
                for candidate_score, token_id in candidate_scores(
                    token_logprobs,
                    score,
                )
                if candidate_score < self.max_score
            ]
            for candidate_score, token_id in candidates:
                if token_id == EOS_TOKEN_ID:
                    self.beams.append(
                        (candidate_score, list(path + (token_id,)))
                    )

            children = [
                (candidate_score, token_id)
                for candidate_score, token_id in candidates
                if token_id != EOS_TOKEN_ID and remaining_tokens > 1
            ]
            if not children:
                break

            first_score, first_token = children[0]
            for alternative_score, alternative_token in reversed(children[1:]):
                self._stack.append(
                    _RolloutNode(
                        path + (alternative_token,),
                        alternative_score,
                        remaining_tokens - 1,
                    )
                )

            if int(sampled_token_id) != first_token:
                # The constrained greedy sampler should choose the first DFS
                # child. Ties or a best EOS can differ; resume that child in
                # the next request rather than consuming mismatched logits.
                self._stack.append(
                    _RolloutNode(
                        path + (first_token,),
                        first_score,
                        remaining_tokens - 1,
                    )
                )
                break

            path = path + (first_token,)
            score = first_score
            remaining_tokens -= 1
        else:
            # The chunk ended before this DFS leaf did. Resume the same child
            # first, ahead of all alternatives already placed on the stack.
            if remaining_tokens > 0:
                self._stack.append(
                    _RolloutNode(path, score, remaining_tokens)
                )


class RecursiveRolloutSearchBatch:
    """Round-robin DFS origins using constrained whole-leaf generation."""

    def __init__(
        self,
        prefix_token_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        max_score: float,
        deadline: float,
        *,
        rollout_chunk_tokens: int = 32,
        branch_window: int = 1,
    ) -> None:
        if branch_window < 1:
            raise ValueError("branch_window must be positive")
        self.origins = [
            RecursiveRolloutOriginSearch(
                prefix,
                max_new_tokens,
                max_score,
                deadline,
                origin=origin,
                rollout_chunk_tokens=rollout_chunk_tokens,
            )
            for origin, prefix in enumerate(prefix_token_ids)
        ]
        self.branch_window = int(branch_window)
        self._request_counter = 0
        self._pending: dict[int, tuple[RecursiveRolloutOriginSearch, int]] = {}

    @property
    def done(self) -> bool:
        return not self._pending and all(state.done for state in self.origins)

    def cap_rollout_chunk_tokens(self, max_tokens: int) -> None:
        """Cap future requests without disturbing an outstanding rollout."""
        max_tokens = int(max_tokens)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        for state in self.origins:
            state.rollout_chunk_tokens = min(
                state.rollout_chunk_tokens,
                max_tokens,
            )

    def take_requests(
        self,
        limit: int,
        now: float | None = None,
    ) -> list[SearchRequest]:
        requests = []
        for state in self.origins:
            available = self.branch_window - state.outstanding_count
            for _ in range(max(0, available)):
                if len(requests) >= limit:
                    break
                requested = state.request(now)
                if requested is None:
                    break
                origin_request_id, prompt, max_tokens, cumulative_nll = requested
                request_id = self._request_counter
                self._request_counter += 1
                self._pending[request_id] = (state, origin_request_id)
                requests.append(
                    SearchRequest(
                        request_id,
                        prompt,
                        max_tokens=max_tokens,
                        constrained_rollout=True,
                        cumulative_nll=float(cumulative_nll),
                    )
                )
            if len(requests) >= limit:
                break
        return requests

    def accept_rollout(
        self,
        request_id: int,
        sampled_token_ids: Sequence[int],
        position_logprobs: Sequence[Mapping[int, float]],
        now: float | None = None,
    ) -> None:
        try:
            state, origin_request_id = self._pending.pop(int(request_id))
        except KeyError as exc:
            raise KeyError(f"Unknown rollout request ID {request_id}") from exc
        state.accept(
            origin_request_id,
            sampled_token_ids,
            position_logprobs,
            now,
        )

    def results(self) -> list[tuple[int, list[tuple[float, list[int]]]]]:
        if not self.done:
            raise RuntimeError("Search results requested before completion")
        return [
            (state.origin, sorted(state.beams, key=lambda item: item[0]))
            for state in self.origins
            if state.beams
        ]


@dataclass
class _FrontierNode:
    origin: int
    path: tuple[int, ...]
    score: float
    remaining_tokens: int


class _FrontierGroup:
    def __init__(
        self,
        prefixes: Sequence[Sequence[int]],
        origin_offset: int,
        max_new_tokens: int,
        max_score: float,
        deadline: float,
        max_live_branches: int,
    ) -> None:
        self.prefixes = [tuple(map(int, prefix)) for prefix in prefixes]
        self.origin_offset = int(origin_offset)
        self.max_score = float(max_score)
        self.deadline = float(deadline)
        self.max_live_branches = max(1, int(max_live_branches))
        self.nodes = [
            _FrontierNode(origin, (), 0.0, int(max_new_tokens))
            for origin in range(len(prefixes))
        ]
        self.beams: dict[int, list[tuple[float, list[int]]]] = defaultdict(list)
        self.recursive: RecursiveSearchBatch | None = None
        self._next_unissued = 0
        self._request_counter = 0
        self._pending: dict[int, int] = {}
        self._responses: dict[int, Mapping[int, float]] = {}
        self._finished = False

    @property
    def done(self) -> bool:
        if self.recursive is not None:
            return self.recursive.done
        return self._finished and not self._pending

    def take_requests(self, limit: int, now: float) -> list[SearchRequest]:
        if now >= self.deadline:
            self._finished = True
            self.nodes = []
            return []
        if self.recursive is not None:
            return self.recursive.take_requests(limit, now)
        requests = []
        while self._next_unissued < len(self.nodes) and len(requests) < limit:
            node_index = self._next_unissued
            self._next_unissued += 1
            node = self.nodes[node_index]
            request_id = self._request_counter
            self._request_counter += 1
            self._pending[request_id] = node_index
            requests.append(
                SearchRequest(
                    request_id,
                    self.prefixes[node.origin] + node.path,
                )
            )
        return requests

    def accept(
        self,
        request_id: int,
        token_logprobs: Mapping[int, float],
        now: float,
    ) -> None:
        if self.recursive is not None:
            self.recursive.accept(request_id, token_logprobs, now)
            return
        try:
            node_index = self._pending.pop(int(request_id))
        except KeyError as exc:
            raise KeyError(f"Unknown frontier request ID {request_id}") from exc
        self._responses[node_index] = token_logprobs
        if self._next_unissued == len(self.nodes) and not self._pending:
            self._finish_level(now)

    def _finish_level(self, now: float) -> None:
        next_nodes = []
        for node_index, node in enumerate(self.nodes):
            for score, token_id in candidate_scores(
                self._responses[node_index],
                node.score,
            ):
                if score >= self.max_score:
                    continue
                if token_id == EOS_TOKEN_ID:
                    self.beams[node.origin].append(
                        (score, list(node.path + (token_id,)))
                    )
                elif node.remaining_tokens > 1:
                    next_nodes.append(
                        _FrontierNode(
                            node.origin,
                            node.path + (token_id,),
                            score,
                            node.remaining_tokens - 1,
                        )
                    )
        self._responses.clear()
        self._next_unissued = 0
        if now >= self.deadline or not next_nodes:
            self.nodes = []
            self._finished = True
            return
        self.nodes = next_nodes
        if len(next_nodes) > self.max_live_branches:
            self.recursive = RecursiveSearchBatch.from_frontier_nodes(
                next_nodes,
                self.prefixes,
                self.max_score,
                self.deadline,
            )

    def results(self) -> dict[int, list[tuple[float, list[int]]]]:
        if not self.done:
            raise RuntimeError("Frontier group results requested before completion")
        result = defaultdict(list)
        for origin, beams in self.beams.items():
            result[origin].extend(beams)
        if self.recursive is not None:
            for origin, beams in self.recursive.results():
                result[origin].extend(beams)
        return result


class FrontierSearchBatch:
    """Frontier search with the same sequential origin grouping as Unsloth."""

    def __init__(
        self,
        prefix_token_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        max_score: float,
        deadline: float,
        *,
        origin_batch_size: int = 4,
        max_live_branches: int = 8,
    ) -> None:
        prefixes = [tuple(map(int, prefix)) for prefix in prefix_token_ids]
        group_size = max(1, min(int(origin_batch_size), len(prefixes)))
        self.groups = [
            _FrontierGroup(
                prefixes[offset : offset + group_size],
                offset,
                max_new_tokens,
                max_score,
                deadline,
                max_live_branches,
            )
            for offset in range(0, len(prefixes), group_size)
        ]
        self._active_group = 0

    def _advance_groups(self) -> None:
        while (
            self._active_group < len(self.groups)
            and self.groups[self._active_group].done
        ):
            self._active_group += 1

    @property
    def done(self) -> bool:
        self._advance_groups()
        return self._active_group >= len(self.groups)

    def take_requests(
        self,
        limit: int,
        now: float | None = None,
    ) -> list[SearchRequest]:
        self._advance_groups()
        if self.done:
            return []
        return self.groups[self._active_group].take_requests(
            limit,
            time.time() if now is None else now,
        )

    def accept(
        self,
        request_id: int,
        token_logprobs: Mapping[int, float],
        now: float | None = None,
    ) -> None:
        if self.done:
            raise RuntimeError("Frontier search is already complete")
        self.groups[self._active_group].accept(
            request_id,
            token_logprobs,
            time.time() if now is None else now,
        )
        self._advance_groups()

    def results(self) -> list[tuple[int, list[tuple[float, list[int]]]]]:
        if not self.done:
            raise RuntimeError("Search results requested before completion")
        result = defaultdict(list)
        for group in self.groups:
            for local_origin, beams in group.results().items():
                result[group.origin_offset + local_origin].extend(beams)
        return [
            (origin, sorted(beams, key=lambda item: item[0]))
            for origin, beams in sorted(result.items())
            if beams
        ]


@dataclass
class _ShapeBeam:
    origin: int
    path: tuple[int, ...] = ()
    score: float = 0.0
    color_mask: int = 0


class ShapeBeamSearch:
    """Full-prompt state machine matching ``inference_shape_beam``."""

    def __init__(
        self,
        prefix_token_ids: Sequence[Sequence[int]],
        output_shape: tuple[int, int],
        deadline: float,
        *,
        beam_width: int = 4,
    ) -> None:
        if not prefix_token_ids:
            raise ValueError("Shape beam requires at least one prompt")
        height, width = map(int, output_shape)
        if not (0 < height <= 30 and 0 < width <= 30):
            raise ValueError(f"Invalid ARC output shape: {output_shape}")
        if height * width < 2:
            raise ValueError("A non-monochrome fallback needs at least two cells")
        if int(beam_width) < 2:
            raise ValueError("beam_width must be at least 2")
        self.prefixes = [tuple(map(int, prefix)) for prefix in prefix_token_ids]
        self.height = height
        self.width = width
        self.deadline = float(deadline)
        self.beam_width = int(beam_width)
        self.beams = [_ShapeBeam(origin) for origin in range(len(self.prefixes))]
        self.row = 0
        self.column = 0
        self._terminal_stage = False
        self._done = False
        self._request_counter = 0
        self._next_unissued = 0
        self._pending: dict[int, int] = {}
        self._responses: dict[int, Mapping[int, float]] = {}

    @property
    def done(self) -> bool:
        return self._done and not self._pending

    def take_requests(
        self,
        limit: int,
        now: float | None = None,
    ) -> list[SearchRequest]:
        now = time.time() if now is None else now
        if now >= self.deadline:
            self._done = True
            self.beams = []
            return []
        requests = []
        while self._next_unissued < len(self.beams) and len(requests) < limit:
            index = self._next_unissued
            self._next_unissued += 1
            beam = self.beams[index]
            request_id = self._request_counter
            self._request_counter += 1
            self._pending[request_id] = index
            requests.append(
                SearchRequest(
                    request_id,
                    self.prefixes[beam.origin] + beam.path,
                )
            )
        return requests

    def accept(
        self,
        request_id: int,
        token_logprobs: Mapping[int, float],
        now: float | None = None,
    ) -> None:
        try:
            index = self._pending.pop(int(request_id))
        except KeyError as exc:
            raise KeyError(f"Unknown shape-beam request ID {request_id}") from exc
        self._responses[index] = token_logprobs
        if self._next_unissued == len(self.beams) and not self._pending:
            self._finish_stage(time.time() if now is None else now)

    def _finish_stage(self, now: float) -> None:
        if now >= self.deadline:
            self.beams = []
            self._done = True
            return
        if self._terminal_stage:
            terminal = EOS_TOKEN_ID if self.row == self.height - 1 else NEWLINE_TOKEN_ID
            for index, beam in enumerate(self.beams):
                beam.score -= float(self._responses[index][terminal])
                beam.path = beam.path + (terminal,)
            if terminal == EOS_TOKEN_ID:
                self._done = True
            else:
                self.row += 1
                self.column = 0
                self._terminal_stage = False
        else:
            is_final_cell = self.row == self.height - 1 and self.column == self.width - 1
            by_origin: dict[int, list[_ShapeBeam]] = defaultdict(list)
            for index, beam in enumerate(self.beams):
                for score, token_id in candidate_scores(
                    self._responses[index],
                    beam.score,
                    DIGIT_TOKENS,
                ):
                    color_mask = beam.color_mask | (1 << token_id)
                    if is_final_cell and color_mask.bit_count() < 2:
                        continue
                    by_origin[beam.origin].append(
                        _ShapeBeam(
                            origin=beam.origin,
                            path=beam.path + (token_id,),
                            score=score,
                            color_mask=color_mask,
                        )
                    )
            self.beams = [
                beam
                for origin in range(len(self.prefixes))
                for beam in sorted(
                    by_origin[origin],
                    key=lambda item: (item.score, item.path[-1]),
                )[: self.beam_width]
            ]
            self.column += 1
            if self.column == self.width:
                self._terminal_stage = True
        self._responses.clear()
        self._next_unissued = 0

    def results(self) -> list[tuple[int, list[tuple[float, list[int]]]]]:
        if not self.done:
            raise RuntimeError("Shape-beam results requested before completion")
        by_origin = defaultdict(list)
        for beam in self.beams:
            by_origin[beam.origin].append((beam.score, list(beam.path)))
        return [
            (origin, sorted(beams, key=lambda item: item[0]))
            for origin, beams in sorted(by_origin.items())
            if beams
        ]
