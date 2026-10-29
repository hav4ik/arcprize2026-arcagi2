"""Task-local ARC reinforcement-learning data and reward helpers.

This module deliberately has no Unsloth or TRL imports.  The reward and data
selection logic can therefore be tested on CPU, while ``arc_solver`` imports
the heavyweight GRPO trainer only when the opt-in experiment is enabled.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np


ARC_MIN_GRID_SIZE = 1
ARC_MAX_GRID_SIZE = 30


def _completion_text(completion: Any) -> str:
    """Normalize a plain or conversational TRL completion to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def parse_grid_completion(completion: Any) -> np.ndarray | None:
    """Parse the first rectangular ARC grid block from a completion.

    The ARC tokenizer normally emits only digits, newlines, and EOS.  Being a
    little defensive here makes reward failures explicit when a generic
    tokenizer or a conversational completion is used in a smoke test.
    """
    text = _completion_text(completion)
    text = text.split("<|im_end|>", 1)[0]
    blocks: list[list[list[int]]] = []
    current: list[list[int]] = []

    def finish_block() -> None:
        nonlocal current
        if current:
            blocks.append(current)
            current = []

    for raw_line in text.splitlines():
        line = "".join(raw_line.strip().split())
        if line and all(character in "0123456789" for character in line):
            current.append([int(character) for character in line])
        else:
            finish_block()
    finish_block()

    for rows in blocks:
        width = len(rows[0])
        if not ARC_MIN_GRID_SIZE <= len(rows) <= ARC_MAX_GRID_SIZE:
            continue
        if not ARC_MIN_GRID_SIZE <= width <= ARC_MAX_GRID_SIZE:
            continue
        if any(len(row) != width for row in rows):
            continue
        return np.asarray(rows, dtype=np.int64)
    return None


@dataclass(frozen=True)
class GridRewardResult:
    reward: float
    valid_grid: bool
    correct_shape: bool
    cell_accuracy: float
    macro_color_recall: float
    exact_match: bool
    above_threshold: bool


def score_grid_completion(
    completion: Any,
    target_grid: Sequence[Sequence[int]],
    *,
    min_cell_accuracy: float = 0.7,
    exact_bonus: float = 0.0,
) -> GridRewardResult:
    """Score one completion with a thresholded dense ARC-grid reward.

    Invalid or wrong-shaped grids receive zero.  A correctly shaped grid also
    receives zero when fewer than ``min_cell_accuracy`` cells are correct.
    Above that threshold the dense reward is its cell accuracy, with an
    optional exact-match bonus to keep a clear gap between almost-correct and
    completely correct outputs.
    """
    if not 0.0 <= min_cell_accuracy <= 1.0:
        raise ValueError("min_cell_accuracy must be between 0 and 1")
    if exact_bonus < 0.0:
        raise ValueError("exact_bonus must be non-negative")

    target = np.asarray(target_grid, dtype=np.int64)
    if (
        target.ndim != 2
        or not ARC_MIN_GRID_SIZE <= target.shape[0] <= ARC_MAX_GRID_SIZE
        or not ARC_MIN_GRID_SIZE <= target.shape[1] <= ARC_MAX_GRID_SIZE
    ):
        raise ValueError("target_grid must be a non-empty ARC grid up to 30x30")

    prediction = parse_grid_completion(completion)
    if prediction is None:
        return GridRewardResult(0.0, False, False, 0.0, 0.0, False, False)
    if prediction.shape != target.shape:
        return GridRewardResult(0.0, True, False, 0.0, 0.0, False, False)

    cell_accuracy = float(np.mean(prediction == target))
    recalls = []
    for color in np.unique(target):
        target_mask = target == color
        recalls.append(float(np.mean(prediction[target_mask] == color)))
    macro_color_recall = float(np.mean(recalls)) if recalls else 0.0
    exact_match = bool(np.array_equal(prediction, target))
    above_threshold = cell_accuracy >= min_cell_accuracy
    reward = cell_accuracy if above_threshold else 0.0
    if exact_match:
        reward += exact_bonus
    return GridRewardResult(
        reward=float(reward),
        valid_grid=True,
        correct_shape=True,
        cell_accuracy=cell_accuracy,
        macro_color_recall=macro_color_recall,
        exact_match=exact_match,
        above_threshold=above_threshold,
    )


class ArcGridReward:
    """TRL-compatible reward callable with lightweight task diagnostics."""

    __name__ = "arc_grid_reward"

    def __init__(self, min_cell_accuracy=0.7, exact_bonus=0.0):
        if not 0.0 <= float(min_cell_accuracy) <= 1.0:
            raise ValueError("min_cell_accuracy must be between 0 and 1")
        if float(exact_bonus) < 0.0:
            raise ValueError("exact_bonus must be non-negative")
        self.min_cell_accuracy = float(min_cell_accuracy)
        self.exact_bonus = float(exact_bonus)
        self.calls = 0
        self.valid_grids = 0
        self.correct_shapes = 0
        self.above_threshold = 0
        self.exact_matches = 0
        self.reward_sum = 0.0
        self.cell_accuracy_sum = 0.0
        self.macro_color_recall_sum = 0.0

    def __call__(self, completions, target_grid, **_kwargs):
        if len(completions) != len(target_grid):
            raise ValueError(
                "completions and target_grid must have the same batch length"
            )
        rewards = []
        for completion, target in zip(completions, target_grid):
            result = score_grid_completion(
                completion,
                target,
                min_cell_accuracy=self.min_cell_accuracy,
                exact_bonus=self.exact_bonus,
            )
            rewards.append(result.reward)
            self.calls += 1
            self.valid_grids += int(result.valid_grid)
            self.correct_shapes += int(result.correct_shape)
            self.above_threshold += int(result.above_threshold)
            self.exact_matches += int(result.exact_match)
            self.reward_sum += result.reward
            self.cell_accuracy_sum += result.cell_accuracy
            self.macro_color_recall_sum += result.macro_color_recall
        return rewards

    def summary(self) -> dict[str, float | int]:
        denominator = max(1, self.calls)
        return {
            "samples": self.calls,
            "valid_fraction": self.valid_grids / denominator,
            "shape_fraction": self.correct_shapes / denominator,
            "above_threshold_fraction": self.above_threshold / denominator,
            "exact_fraction": self.exact_matches / denominator,
            "mean_cell_accuracy": self.cell_accuracy_sum / denominator,
            "mean_macro_color_recall": (
                self.macro_color_recall_sum / denominator
            ),
            "mean_reward": self.reward_sum / denominator,
        }


def assistant_completion_spans(
    token_ids: Sequence[int],
    *,
    user_token_id: int,
    assistant_token_id: int,
    eos_token_id: int,
    final_only: bool = False,
) -> list[tuple[int, int]]:
    """Return half-open label spans for assistant replies in an ARC chat."""
    starts = sorted(
        index
        for index, token_id in enumerate(token_ids)
        if token_id in {user_token_id, assistant_token_id}
    )
    ends = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == eos_token_id
    ]
    if len(starts) != len(ends):
        raise ValueError(
            f"Unbalanced ARC chat markers: {len(starts)} starts, "
            f"{len(ends)} ends"
        )
    spans = []
    for turn_index, (start, end) in enumerate(zip(starts, ends)):
        if start >= end:
            raise ValueError("ARC chat turn starts after its EOS token")
        if turn_index % 2 == 1:
            spans.append((start + 2, end + 1))
    return spans[-1:] if final_only else spans


def _warmup_stratum(key: str) -> tuple[tuple[str, ...], int | None]:
    parts = key.split(".")[1:]
    geometry = tuple(part for part in parts if part in {"transpose", "rot90"})
    order_text = next(
        (part[2:] for part in reversed(parts) if part.startswith("ex")),
        "",
    )
    if not order_text:
        return geometry, None
    try:
        order = (
            [int(value) for value in order_text.split("-")]
            if "-" in order_text
            else [int(value) for value in order_text]
        )
    except ValueError:
        return geometry, None
    return geometry, order[-1] if order else None


def color_permutation_from_key(key: str) -> str | None:
    """Return the full color-permutation descriptor encoded in a row key."""
    return next(
        (part for part in key.split(".") if part.startswith("permute")),
        None,
    )


def color_permutations(rows: Iterable[dict[str, Any]]) -> set[str]:
    """Collect color mappings used by formatted or RL rows."""
    result = set()
    for row in rows:
        key = str(row.get("key", row.get("rl_key", "")))
        descriptor = color_permutation_from_key(key)
        if descriptor is not None:
            result.add(descriptor)
    return result


def dataset_colors(dataset) -> tuple[int, ...]:
    """Return every color observed in the selected task's known grids."""
    colors = set()
    for key in dataset.keys:
        for examples in dataset.queries[key].values():
            for example in examples:
                for grid in example.values():
                    colors.update(
                        int(value) for value in np.asarray(grid).reshape(-1)
                    )
        for grid in dataset.replies.get(key, []):
            colors.update(int(value) for value in np.asarray(grid).reshape(-1))
    return tuple(sorted(colors))


def color_mapping_signature(key: str, colors: Iterable[int]) -> tuple[int, ...]:
    """Return a mapping's effect on task-observed colors, ignoring unused ones."""
    descriptor = color_permutation_from_key(key)
    if descriptor is None:
        raise ValueError(f"Row key has no color permutation: {key}")
    mapping = [int(value) for value in descriptor[len("permute"):]]
    if len(mapping) != 10 or sorted(mapping) != list(range(10)):
        raise ValueError(f"Invalid ARC color permutation: {descriptor}")
    return tuple(mapping[int(color)] for color in colors)


def color_mapping_signatures(
    rows: Iterable[dict[str, Any]],
    colors: Iterable[int],
) -> set[tuple[int, ...]]:
    """Collect effective mappings on the colors actually used by a task."""
    colors = tuple(colors)
    return {
        color_mapping_signature(
            str(row.get("key", row.get("rl_key", ""))),
            colors,
        )
        for row in rows
    }


def select_balanced_warmup_rows(
    rows: Iterable[dict[str, Any]],
    limit: int = 32,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select deterministic rows balanced over geometry and final target."""
    rows = list(rows)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(rows) <= limit:
        return rows

    buckets: dict[
        tuple[tuple[str, ...], int | None],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        buckets[_warmup_stratum(str(row.get("key", "")))].append(row)

    def stable_order(row: dict[str, Any]) -> bytes:
        payload = f"{seed}:{row.get('key', '')}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    for bucket in buckets.values():
        bucket.sort(key=stable_order)

    bucket_keys = sorted(buckets, key=repr)
    selected = []
    depth = 0
    while len(selected) < limit:
        made_progress = False
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
        depth += 1
    return selected


def build_task_rl_rows(
    puzzle_dataset,
    formatter,
    *,
    augmentations: int = 2,
    seed: int = 7,
    max_prompt_tokens: int = 2048,
    color_signature_colors: Iterable[int] = tuple(range(10)),
    excluded_color_signatures: Iterable[tuple[int, ...]] = (),
) -> list[dict[str, Any]]:
    """Construct augmented leave-one-out prompts and known reward targets."""
    if augmentations <= 0:
        raise ValueError("augmentations must be positive")
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")

    color_signature_colors = tuple(color_signature_colors)
    excluded_color_signatures = {
        tuple(signature) for signature in excluded_color_signatures
    }
    base_dataset = puzzle_dataset.split_train_replies()
    if not base_dataset.keys:
        return []
    selected_keys: list[str] = []
    selected_queries = {}
    selected_replies = {}
    selected_per_stratum: dict[
        tuple[str, tuple[str, ...]],
        int,
    ] = defaultdict(int)
    selected_signatures: dict[
        tuple[str, tuple[str, ...]],
        set[tuple[int, ...]],
    ] = defaultdict(set)
    expected_strata = len(base_dataset.keys) * 8
    # Usually the first attempt is sufficient. Retries avoid materializing
    # `len(excluded)` throwaway variants for every held-out view.
    for attempt in range(16):
        candidate_dataset = base_dataset.augment(
            n=max(augmentations, 4),
            shfl_keys=True,
            seed=seed + attempt * 104729,
            unique_colors=True,
            balanced_examples=True,
        )
        for key in candidate_dataset.keys:
            signature = color_mapping_signature(key, color_signature_colors)
            if signature in excluded_color_signatures:
                continue
            base_key = key.split(".", 1)[0]
            geometry, _target = _warmup_stratum(key)
            stratum = (base_key, geometry)
            if selected_per_stratum[stratum] >= augmentations:
                continue
            if signature in selected_signatures[stratum]:
                continue
            selected_keys.append(key)
            selected_queries[key] = candidate_dataset.queries[key]
            selected_replies[key] = candidate_dataset.replies[key]
            selected_signatures[stratum].add(signature)
            selected_per_stratum[stratum] += 1
        if (
            len(selected_per_stratum) == expected_strata
            and all(
                count == augmentations
                for count in selected_per_stratum.values()
            )
        ):
            break

    if (
        len(selected_per_stratum) != expected_strata
        or any(count != augmentations for count in selected_per_stratum.values())
    ):
        raise ValueError(
            "Could not construct the requested RL color split; the task may "
            "not have enough distinct effective color permutations"
        )
    dataset = base_dataset.__class__(
        queries=selected_queries,
        replies=selected_replies,
        keys=selected_keys,
    )
    dataset = dataset.cut_to_len(
        formatter=formatter,
        name="input",
        max_len=max_prompt_tokens,
    )
    rows = []
    for key in dataset.keys:
        formatted = dataset.get(key, formatter)
        rows.append(
            {
                "prompt": formatted["input"],
                "target_grid": dataset.replies[key][0],
                "rl_key": key,
                "color_permutation": color_permutation_from_key(key),
                "color_signature": list(
                    color_mapping_signature(key, color_signature_colors)
                ),
            }
        )
    return rows
