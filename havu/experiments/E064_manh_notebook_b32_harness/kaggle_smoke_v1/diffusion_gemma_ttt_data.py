"""ARC task-time conversations and DiffusionGemma inference views."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from arc_loader import ArcDataset, convert_grid_to_string, is_valid_solution
from diffusion_gemma_arc_data import (
    BOS_ID,
    CHANNEL_END_ID,
    CHANNEL_START_ID,
    END_TURN_ID,
    EOS_ID,
    MODEL_ID,
    NEWLINE_ID,
    START_ID,
    THOUGHT_ID,
    encode_complete_message,
    expand_saved_messages,
)


DEFAULT_TTT_COLOR_AUGMENTATIONS = 16
DEFAULT_INFERENCE_VIEWS = 8


def stable_task_seed(task_key: str, seed: int = 42) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{task_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _training_messages(query: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for example in query.get("train", ()):
        messages.extend(
            (
                {
                    "role": "user",
                    "content": convert_grid_to_string(example["input"]),
                },
                {
                    "role": "assistant",
                    "content": convert_grid_to_string(example["output"]),
                },
            )
        )
    if not messages:
        raise ValueError("DiffusionGemma TTT requires at least one training pair")
    return messages


def build_ttt_conversation_rows(
    task_key: str,
    query: dict[str, Any],
    *,
    instruction_token_ids: Iterable[int] = (),
    color_augmentations: int = DEFAULT_TTT_COLOR_AUGMENTATIONS,
    max_seq_length: int = 12288,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build exactly ``8 * color_augmentations`` conversation-level rows.

    Every augmented view contains all transformed demonstrations.  The final
    demonstration is the only diffusion target; earlier pairs remain clean
    causal context. ``balanced_examples=True`` rotates every demonstration
    through that final supervised position.
    """
    if color_augmentations <= 0:
        raise ValueError("color_augmentations must be positive")
    source = ArcDataset(
        queries={task_key: query},
        keys=[task_key],
    )
    augmented = source.augment(
        n=int(color_augmentations),
        seed=stable_task_seed(task_key, seed),
        unique_colors=True,
        balanced_examples=True,
    )
    expected = 8 * int(color_augmentations)
    if len(augmented.keys) != expected:
        raise RuntimeError(
            f"expected {expected} augmented TTT views, got {len(augmented.keys)}"
        )

    rows: list[dict[str, Any]] = []
    instruction_token_ids = tuple(int(value) for value in instruction_token_ids)
    for view_key in augmented.keys:
        active_train = list(augmented.queries[view_key].get("train", ()))
        dropped_train_pairs = 0
        while True:
            examples = expand_saved_messages(
                _training_messages({"train": active_train}),
                max_seq_length=int(max_seq_length),
                instruction_token_ids=instruction_token_ids,
                sample_mode="conversation",
            )
            if examples or len(active_train) <= 1:
                break
            # Keep the final supervised pair and remove only complete oldest
            # demonstrations. This mirrors inference prompt truncation and
            # never byte-truncates a grid or changes the conversation target.
            active_train.pop(0)
            dropped_train_pairs += 1
        if len(examples) != 1:
            physical = (
                "over the configured sequence limit"
                if not examples
                else f"expanded into {len(examples)} rows"
            )
            raise ValueError(
                f"conversation TTT view {view_key!r} is {physical}; "
                "all training pairs must remain in one atomic conversation"
            )
        rows.append(
            {
                "puzzle_name": task_key,
                "view_key": view_key,
                "dropped_train_pairs": dropped_train_pairs,
                **examples[0],
            }
        )
    return rows


def _assistant_generation_prefix() -> bytearray:
    return bytearray(
        (
            START_ID,
            MODEL_ID,
            NEWLINE_ID,
            CHANNEL_START_ID,
            THOUGHT_ID,
            NEWLINE_ID,
            CHANNEL_END_ID,
        )
    )


def build_inference_prompt_ids(
    query: dict[str, Any],
    *,
    instruction_token_ids: Iterable[int] = (),
    max_prompt_tokens: int = 11264,
) -> list[int]:
    """Encode demonstrations plus one test input using the ARC31 chat contract.

    If required, oldest complete demonstrations are removed as atomic pairs;
    the test turn and most recent demonstrations are never byte-truncated.
    """
    tests = query.get("test", ())
    if len(tests) != 1:
        raise ValueError(
            f"inference view must contain exactly one test pair, got {len(tests)}"
        )
    demonstrations = list(query.get("train", ()))
    instruction_token_ids = tuple(int(value) for value in instruction_token_ids)

    def encode(active: Sequence[dict[str, Any]]) -> bytearray:
        prompt = bytearray((BOS_ID,))
        first_user = True
        for example in active:
            user = encode_complete_message(
                "user", convert_grid_to_string(example["input"])
            )
            if first_user and instruction_token_ids:
                user[3:3] = bytearray(instruction_token_ids)
            prompt.extend(user)
            prompt.extend(
                encode_complete_message(
                    "assistant", convert_grid_to_string(example["output"])
                )
            )
            first_user = False
        test_user = encode_complete_message(
            "user", convert_grid_to_string(tests[0]["input"])
        )
        if first_user and instruction_token_ids:
            test_user[3:3] = bytearray(instruction_token_ids)
        prompt.extend(test_user)
        prompt.extend(_assistant_generation_prefix())
        return prompt

    while True:
        prompt = encode(demonstrations)
        if len(prompt) <= int(max_prompt_tokens):
            return list(prompt)
        if not demonstrations:
            raise ValueError(
                f"test input alone requires {len(prompt)} prompt tokens, above "
                f"the limit {max_prompt_tokens}"
            )
        demonstrations.pop(0)


@dataclass(frozen=True)
class DiffusionInferenceView:
    task_key: str
    split_key: str
    view_key: str
    prompt_token_ids: tuple[int, ...]


def build_inference_views(
    task_key: str,
    query: dict[str, Any],
    *,
    instruction_token_ids: Iterable[int] = (),
    views: int = DEFAULT_INFERENCE_VIEWS,
    max_prompt_tokens: int = 11264,
    seed: int = 42,
) -> list[DiffusionInferenceView]:
    """Build deterministic augmented prompts for every test pair."""
    if views <= 0:
        raise ValueError("views must be positive")
    source = ArcDataset(queries={task_key: query}, keys=[task_key])
    split = source.split_multi_replies()
    result: list[DiffusionInferenceView] = []
    for split_key in split.keys:
        one = split.change_keys([split_key])
        augmented = one.augment_exact_views(
            int(views), seed=stable_task_seed(split_key, seed)
        )
        for view_key in augmented.keys:
            result.append(
                DiffusionInferenceView(
                    task_key=task_key,
                    split_key=split_key,
                    view_key=view_key,
                    prompt_token_ids=tuple(
                        build_inference_prompt_ids(
                            augmented.queries[view_key],
                            instruction_token_ids=instruction_token_ids,
                            max_prompt_tokens=max_prompt_tokens,
                        )
                    ),
                )
            )
    return result


def parse_generated_grid(token_ids: Iterable[int]) -> np.ndarray | None:
    """Strictly parse one compact DiffusionGemma ARC response."""
    rows: list[list[int]] = [[]]
    saw_content = False
    for raw_token in token_ids:
        token = int(raw_token)
        if token in (END_TURN_ID, EOS_ID):
            break
        if 0 <= token <= 9:
            rows[-1].append(token)
            saw_content = True
        elif token == NEWLINE_ID:
            if rows[-1]:
                rows.append([])
        else:
            return None
    if rows and not rows[-1]:
        rows.pop()
    if not saw_content or not rows:
        return None
    widths = {len(row) for row in rows}
    if len(widths) != 1 or 0 in widths:
        return None
    grid = np.asarray(rows, dtype=np.int64)
    return grid if is_valid_solution(grid) else None


@dataclass(frozen=True)
class ViewPrediction:
    split_key: str
    view_key: str
    token_ids: tuple[int, ...]
    score: float | None = None


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid.tolist())


def select_augmented_candidates(
    predictions: Iterable[ViewPrediction],
    *,
    attempts: int = 2,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """Invert views and rank candidates by consensus, then mean score."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    grouped: dict[str, dict[tuple[tuple[int, ...], ...], dict[str, Any]]] = defaultdict(dict)
    invalid: dict[str, int] = defaultdict(int)
    for order, prediction in enumerate(predictions):
        grid = parse_generated_grid(prediction.token_ids)
        if grid is None:
            invalid[prediction.split_key] += 1
            continue
        restored = ArcDataset.invert_mod(grid, prediction.view_key)
        if not is_valid_solution(restored):
            invalid[prediction.split_key] += 1
            continue
        key = _grid_key(restored)
        row = grouped[prediction.split_key].setdefault(
            key,
            {
                "grid": np.asarray(restored, dtype=np.int64),
                "count": 0,
                "scores": [],
                "first_order": order,
            },
        )
        row["count"] += 1
        if prediction.score is not None and np.isfinite(prediction.score):
            row["scores"].append(float(prediction.score))

    selected: dict[str, list[np.ndarray]] = {}
    diagnostics: dict[str, Any] = {}
    all_splits = set(grouped) | set(invalid)
    for split_key in sorted(all_splits):
        candidates = list(grouped.get(split_key, {}).values())
        for candidate in candidates:
            candidate["mean_score"] = (
                sum(candidate["scores"]) / len(candidate["scores"])
                if candidate["scores"]
                else float("-inf")
            )
        candidates.sort(
            key=lambda item: (
                -item["count"],
                -item["mean_score"],
                item["first_order"],
            )
        )
        selected[split_key] = [
            item["grid"] for item in candidates[: int(attempts)]
        ]
        diagnostics[split_key] = {
            "valid_views": sum(item["count"] for item in candidates),
            "invalid_views": invalid.get(split_key, 0),
            "unique_candidates": len(candidates),
            "top_counts": [item["count"] for item in candidates[:attempts]],
            "top_mean_scores": [
                None if not np.isfinite(item["mean_score"]) else item["mean_score"]
                for item in candidates[:attempts]
            ],
        }
    return selected, diagnostics


__all__ = [
    "DEFAULT_INFERENCE_VIEWS",
    "DEFAULT_TTT_COLOR_AUGMENTATIONS",
    "DiffusionInferenceView",
    "ViewPrediction",
    "build_inference_prompt_ids",
    "build_inference_views",
    "build_ttt_conversation_rows",
    "parse_generated_grid",
    "select_augmented_candidates",
    "stable_task_seed",
]
