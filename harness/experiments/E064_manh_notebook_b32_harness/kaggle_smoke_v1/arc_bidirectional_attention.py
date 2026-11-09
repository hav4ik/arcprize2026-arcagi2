"""Shared ARC intermittent bidirectional-attention metadata.

ARC conversations alternate user/query spans and assistant/answer spans.  The
query span is allowed to attend to its complete span, while answer tokens stay
strictly causal.  This module owns the token-to-segment contract used by both
Unsloth training and vLLM inference; neither backend is allowed to infer a
different mask from model activations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Sequence


IM_START_TOKEN_ID = 14
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
EOS_TOKEN_ID = 15


@dataclass(frozen=True)
class ArcAttentionSegment:
    """One half-open query or answer run in an ARC token stream."""

    start: int
    end: int
    is_prompt: bool

    def as_tuple(self) -> tuple[int, int, bool]:
        return self.start, self.end, self.is_prompt


def arc_prompt_mask_from_tokens(
    token_ids: Sequence[int],
    *,
    user_token_id: int = USER_TOKEN_ID,
    assistant_token_id: int = ASSISTANT_TOKEN_ID,
    eos_token_id: int = EOS_TOKEN_ID,
) -> tuple[bool, ...]:
    """Return the exact assistant-only-SFT prompt mask for one ARC chat.

    ``True`` means a query/prompt token receives intermittent bidirectional
    attention.  ``False`` means an assistant answer token remains causal.  The
    assistant role token and following newline are prompt context, matching
    ``QwenDataCollatorForCompletionOnlyLM``; answer supervision begins two
    tokens after the assistant role marker.

    The final assistant answer may be incomplete during constrained decoding,
    so only completed user turns are required to have an EOS marker.
    """
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids:
        raise ValueError("ARC attention metadata requires at least one token")

    role_positions = [
        index
        for index, token_id in enumerate(ids)
        if token_id in {user_token_id, assistant_token_id}
    ]
    if not role_positions:
        raise ValueError("ARC token stream contains no user/assistant role markers")
    role_ids = [ids[index] for index in role_positions]
    expected = [
        user_token_id if index % 2 == 0 else assistant_token_id
        for index in range(len(role_ids))
    ]
    if role_ids != expected:
        raise ValueError(
            "ARC roles must alternate user/assistant starting with user: "
            f"got {role_ids}"
        )

    mask = [True] * len(ids)
    for role_index, role_position in enumerate(role_positions):
        if ids[role_position] != assistant_token_id:
            continue
        answer_start = role_position + 2
        if answer_start > len(ids):
            raise ValueError("assistant role marker is missing its newline token")
        next_role = (
            role_positions[role_index + 1]
            if role_index + 1 < len(role_positions)
            else len(ids)
        )
        eos_position = next(
            (
                index
                for index in range(answer_start, next_role)
                if ids[index] == eos_token_id
            ),
            None,
        )
        answer_end = next_role if eos_position is None else eos_position + 1
        for index in range(answer_start, answer_end):
            mask[index] = False

    # Every completed user turn must terminate before the following assistant
    # marker.  This catches malformed streams that would otherwise silently
    # turn part of a user grid causal.
    for role_index in range(0, len(role_positions), 2):
        user_position = role_positions[role_index]
        if role_index + 1 >= len(role_positions):
            break
        assistant_position = role_positions[role_index + 1]
        if eos_token_id not in ids[user_position + 1 : assistant_position]:
            raise ValueError("user turn is missing EOS before assistant marker")

    return tuple(mask)


def attention_segments_from_prompt_mask(
    prompt_mask: Sequence[bool],
) -> tuple[ArcAttentionSegment, ...]:
    """Compress a token mask into contiguous prompt/answer runs."""
    mask = tuple(bool(value) for value in prompt_mask)
    if not mask:
        raise ValueError("ARC prompt mask cannot be empty")
    segments: list[ArcAttentionSegment] = []
    start = 0
    current = mask[0]
    for index in range(1, len(mask)):
        if mask[index] == current:
            continue
        segments.append(ArcAttentionSegment(start, index, current))
        start = index
        current = mask[index]
    segments.append(ArcAttentionSegment(start, len(mask), current))
    return tuple(segments)


def arc_attention_segments_from_tokens(
    token_ids: Sequence[int],
) -> tuple[ArcAttentionSegment, ...]:
    """Parse and compress one ARC chat in a single call."""
    return attention_segments_from_prompt_mask(
        arc_prompt_mask_from_tokens(token_ids)
    )


def serialized_arc_attention_segments(
    token_ids: Sequence[int],
) -> tuple[tuple[int, int, bool], ...]:
    """Return an IPC-safe representation for vLLM ``SamplingParams``."""
    return tuple(
        segment.as_tuple()
        for segment in arc_attention_segments_from_tokens(token_ids)
    )


def arc_attention_cache_salt(
    token_ids: Sequence[int],
    *,
    block_size: int = 16,
) -> str:
    """Return a safe vLLM cache namespace for intermittent attention.

    A prompt token's K/V depends on later tokens through the end of its prompt
    run, while vLLM normally hashes only preceding cache blocks.  The digest
    therefore covers every token through the last prompt run.  Until the
    answer fills the cache block containing that run's end, every length gets
    a temporary namespace so vLLM cannot resume *inside* a bidirectional run.
    Once that block is complete, recursive answer decoding reuses one stable
    namespace and ordinary causal prefix caching resumes.
    """
    if block_size <= 0:
        raise ValueError("vLLM cache block_size must be positive")
    ids = tuple(int(token_id) for token_id in token_ids)
    segments = arc_attention_segments_from_tokens(ids)
    prompt_end = max(
        segment.end for segment in segments if segment.is_prompt
    )
    digest = hashlib.blake2b(digest_size=16)
    digest.update(b"arc-bidirectional-v1\0")
    digest.update(struct.pack("<Q", prompt_end))
    for token_id in ids[:prompt_end]:
        if token_id < 0:
            raise ValueError("token IDs must be non-negative")
        digest.update(struct.pack("<Q", token_id))
    identity = f"arc-bidirectional-v1:{digest.hexdigest()}"
    stable_length = (
        (prompt_end + block_size - 1) // block_size
    ) * block_size
    if len(ids) < stable_length:
        return f"{identity}:warmup-{len(ids)}"
    return identity


def intersect_attention_segments(
    serialized_segments: Sequence[Sequence[int | bool]],
    *,
    query_start: int,
    query_end: int,
) -> tuple[ArcAttentionSegment, ...]:
    """Intersect full-request segments with one scheduled vLLM query slice.

    Positions after the serialized request are generated assistant tokens and
    therefore causal.  Prompt runs must be scheduled through their full end;
    accepting a chunk that stops inside a prompt run would produce an invalid
    KV cache and is rejected by the backend.
    """
    query_start = int(query_start)
    query_end = int(query_end)
    if query_start < 0 or query_end <= query_start:
        raise ValueError(
            f"invalid query interval [{query_start}, {query_end})"
        )

    full_segments = tuple(
        ArcAttentionSegment(int(start), int(end), bool(is_prompt))
        for start, end, is_prompt in serialized_segments
    )
    if not full_segments:
        raise ValueError("serialized ARC attention segments cannot be empty")
    expected = 0
    for segment in full_segments:
        if segment.start != expected or segment.end <= segment.start:
            raise ValueError(
                "serialized ARC attention segments must be contiguous from "
                f"zero; expected {expected}, got {segment}"
            )
        expected = segment.end

    intersections: list[ArcAttentionSegment] = []
    for segment in full_segments:
        start = max(query_start, segment.start)
        end = min(query_end, segment.end)
        if start < end:
            if segment.is_prompt and query_end < segment.end:
                raise RuntimeError(
                    "vLLM split an ARC prompt run before its bidirectional "
                    f"end: scheduled=[{query_start}, {query_end}), "
                    f"prompt={segment}"
                )
            intersections.append(
                ArcAttentionSegment(start, end, segment.is_prompt)
            )

    generated_start = max(query_start, expected)
    if generated_start < query_end:
        intersections.append(
            ArcAttentionSegment(generated_start, query_end, False)
        )
    if not intersections:
        raise RuntimeError(
            "ARC attention metadata does not overlap the scheduled query"
        )
    if intersections[0].start != query_start:
        raise RuntimeError("ARC attention metadata leaves a query prefix uncovered")
    for previous, current in zip(intersections, intersections[1:]):
        if previous.end != current.start:
            raise RuntimeError("ARC attention metadata leaves a query gap")
    if intersections[-1].end != query_end:
        raise RuntimeError("ARC attention metadata leaves a query suffix uncovered")
    return tuple(intersections)


__all__ = [
    "ASSISTANT_TOKEN_ID",
    "ArcAttentionSegment",
    "EOS_TOKEN_ID",
    "IM_START_TOKEN_ID",
    "USER_TOKEN_ID",
    "arc_attention_segments_from_tokens",
    "arc_attention_cache_salt",
    "arc_prompt_mask_from_tokens",
    "attention_segments_from_prompt_mask",
    "intersect_attention_segments",
    "serialized_arc_attention_segments",
]
