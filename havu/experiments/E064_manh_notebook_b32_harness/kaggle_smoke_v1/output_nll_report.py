"""Canonical teacher-forced NLL reports for ARC test outputs.

The trained model is available during inference but not when ``submit.py``
later selects its two attempts.  This module defines a small sidecar artifact
that scores every unique candidate (plus the submission fallback and optional
ground truth) against the original, unaugmented task prompt.  ``submit.py`` can
then look up the NLL of whichever attempts its selected ranking policy emits.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from arc_loader import ArcDataset


OUTPUT_NLL_REPORT_SCHEMA_VERSION = 1
OUTPUT_NLL_REPORT_SUBDIR = "output_nll_reports"


@dataclass(frozen=True)
class OutputNllTarget:
    """One canonical prompt/answer pair that still needs model scoring."""

    base_key: str
    entry_index: int
    grid: tuple[tuple[int, ...], ...]
    query: str
    answer: str
    prompt_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]


def output_nll_report_dir(inference_output_dir: str) -> str:
    """Return the isolated sidecar directory below one candidate store."""
    override = os.getenv("ARC_OUTPUT_NLL_REPORT_DIR")
    if override:
        return os.path.expanduser(override)
    return os.path.join(inference_output_dir, OUTPUT_NLL_REPORT_SUBDIR)


def load_available_ground_truths(enabled: bool, solution_path: str) -> dict:
    """Load labels when present without treating hidden-test absence as fatal."""
    if not enabled:
        return {}
    try:
        with open(solution_path, encoding="utf-8") as solution_file:
            solutions = json.load(solution_file)
    except FileNotFoundError:
        print(
            "*** Output-NLL ground truth unavailable: solution file is "
            f"missing at {solution_path!r}",
            flush=True,
        )
        return {}
    if not isinstance(solutions, dict):
        raise TypeError(
            "ARC solution file must contain a task-key mapping, got "
            f"{type(solutions).__name__}"
        )
    return solutions


def normalize_output_grid(grid: Any) -> list[list[int]]:
    """Validate and JSON-normalize one non-empty ARC grid."""
    array = np.asarray(grid)
    if array.ndim != 2 or not all(int(size) > 0 for size in array.shape):
        raise ValueError(f"Expected a non-empty 2D ARC grid, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(array, np.floor(array))):
            raise ValueError("ARC output grid contains non-integer values")
    if np.any(array < 0) or np.any(array > 9):
        raise ValueError("ARC output grid values must be in [0, 9]")
    return [[int(value) for value in row] for row in array.tolist()]


def output_grid_key(grid: Any) -> tuple[tuple[int, ...], ...]:
    """Return the exact hashable identity used for report lookups."""
    return tuple(tuple(row) for row in normalize_output_grid(grid))


def _canonical_score_row(
    puzzle_ds_multi: ArcDataset,
    base_key: str,
    grid: list[list[int]],
    formatter: Any,
) -> dict[str, Any]:
    dataset = ArcDataset(
        keys=[base_key],
        queries={base_key: puzzle_ds_multi.queries[base_key]},
        replies={base_key: [grid]},
    )
    return dataset.get(base_key, formatter)


def prepare_output_nll_report(
    *,
    task_key: str,
    puzzle_ds_multi: ArcDataset,
    test_output_count: int,
    candidate_solutions: Mapping[str, Sequence[Any]],
    ground_truth_outputs: Sequence[Any] | None,
    formatter: Any,
    tokenizer: Any,
    max_model_len: int,
    metadata: Mapping[str, Any] | None = None,
    min_train_pairs: int = 0,
    extra_train_pair_mode: str = "augment",
) -> tuple[dict[str, Any], list[OutputNllTarget]]:
    """Build an unscored report and its context-safe scoring targets.

    The prompt contains the demonstrations and test input only.  Ground truth
    is added exclusively as an answer, exactly like an ordinary candidate, so
    enabling this diagnostic cannot leak labels into training, decoding, or
    candidate selection.
    """
    test_output_count = int(test_output_count)
    max_model_len = int(max_model_len)
    if test_output_count < 0:
        raise ValueError("test_output_count must be non-negative")
    if max_model_len < 1:
        raise ValueError("max_model_len must be positive")
    if min_train_pairs:
        # One deterministic expansion of the canonical view, shared by all
        # answers. Never derive this prompt from any candidate/ground truth.
        puzzle_ds_multi = puzzle_ds_multi.with_min_train_pairs(
            min_train_pairs, extra_train_pair_mode, seed=0,
        )

    report: dict[str, Any] = {
        "schema_version": OUTPUT_NLL_REPORT_SCHEMA_VERSION,
        "task_key": str(task_key),
        "score_definition": (
            "sum(-log P(answer_token | canonical_prompt, previous_answer_tokens))"
        ),
        "prompt_policy": (
            "unaugmented training examples plus the selected test input"
        ),
        "answer_includes_terminal_token": True,
        "max_model_len": max_model_len,
        "metadata": dict(metadata or {}),
        "outputs": {},
    }
    targets: list[OutputNllTarget] = []
    if min_train_pairs:
        report["prompt_policy"] += (
            f"; train-only minimum-pair expansion={min_train_pairs}, "
            f"mode={extra_train_pair_mode}, seed=0"
        )

    truths = list(ground_truth_outputs or [])
    for test_index in range(test_output_count):
        base_key = f"{task_key}_{test_index}"
        if base_key not in puzzle_ds_multi.queries:
            raise KeyError(f"Canonical ARC query is missing for {base_key}")

        truth = (
            normalize_output_grid(truths[test_index])
            if test_index < len(truths)
            else None
        )
        ordered_grids: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}

        def add_grid(grid: Any, source: str) -> None:
            normalized = normalize_output_grid(grid)
            key = tuple(tuple(row) for row in normalized)
            existing = ordered_grids.get(key)
            if existing is None:
                ordered_grids[key] = {
                    "grid": normalized,
                    "sources": [source],
                }
            elif source not in existing["sources"]:
                existing["sources"].append(source)

        if truth is not None:
            add_grid(truth, "ground_truth")
        for solution in candidate_solutions.get(base_key, ()):
            add_grid(solution, "candidate")
        # ``ArcDataset.get_submission`` uses this when no candidate exists.
        # Scoring it guarantees that both displayed attempts have an NLL even
        # for partially completed runs.
        add_grid([[0]], "submission_fallback")

        output_record = {
            "test_index": test_index,
            "ground_truth": truth,
            "scores": [],
        }
        report["outputs"][base_key] = output_record

        for grid_key, grid_record in ordered_grids.items():
            row = _canonical_score_row(
                puzzle_ds_multi,
                base_key,
                grid_record["grid"],
                formatter,
            )
            prompt_token_ids = tuple(tokenizer.encode(row["input"]))
            answer_token_ids = tuple(tokenizer.encode(row["reply"]))
            # vLLM prompt-logprob scoring requests one unused generation token.
            # Reserve it for identical availability across native and routed
            # inference backends.
            required_context_tokens = (
                len(prompt_token_ids) + len(answer_token_ids) + 1
            )
            entry = {
                **grid_record,
                "prompt_tokens": len(prompt_token_ids),
                "answer_tokens": len(answer_token_ids),
                "required_context_tokens": required_context_tokens,
                "total_nll": None,
                "mean_nll": None,
                "status": "pending",
            }
            entry_index = len(output_record["scores"])
            output_record["scores"].append(entry)
            if required_context_tokens > max_model_len:
                entry["status"] = "context_exceeded"
                entry["reason"] = (
                    f"requires {required_context_tokens} tokens but "
                    f"max_model_len is {max_model_len}"
                )
                continue
            targets.append(
                OutputNllTarget(
                    base_key=base_key,
                    entry_index=entry_index,
                    grid=grid_key,
                    query=row["input"],
                    answer=row["reply"],
                    prompt_token_ids=prompt_token_ids,
                    answer_token_ids=answer_token_ids,
                )
            )

    return report, targets


def apply_output_nll_scores(
    report: dict[str, Any],
    targets: Sequence[OutputNllTarget],
    scores: Sequence[float],
) -> None:
    """Attach finite total and per-token NLLs to prepared report entries."""
    if len(targets) != len(scores):
        raise ValueError(
            f"Expected {len(targets)} output NLLs, got {len(scores)}"
        )
    for target, raw_score in zip(targets, scores):
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(
                f"Non-finite output NLL for {target.base_key}: {score}"
            )
        entry = report["outputs"][target.base_key]["scores"][
            target.entry_index
        ]
        if output_grid_key(entry["grid"]) != target.grid:
            raise RuntimeError(
                f"Output-NLL report entry moved for {target.base_key}"
            )
        answer_tokens = int(entry["answer_tokens"])
        if answer_tokens < 1:
            raise RuntimeError("ARC answer must contain at least its end token")
        entry["total_nll"] = score
        entry["mean_nll"] = score / answer_tokens
        entry["status"] = "ok"
        entry.pop("reason", None)


def finalize_output_nll_report(report: dict[str, Any]) -> None:
    """Convert any unfinished diagnostic entries into explicit unavailable rows."""
    for output in report.get("outputs", {}).values():
        for entry in output.get("scores", []):
            if entry.get("status") == "pending":
                entry["status"] = "not_scored"
                entry["reason"] = "model scoring did not complete"


def output_nll_report_path(inference_output_dir: str, task_key: str) -> str:
    if os.path.basename(str(task_key)) != str(task_key):
        raise ValueError(f"Unsafe ARC task key for report path: {task_key!r}")
    return os.path.join(
        output_nll_report_dir(inference_output_dir),
        f"{task_key}.json",
    )


def write_output_nll_report(
    inference_output_dir: str,
    report: dict[str, Any],
) -> str:
    """Atomically publish one task report beside, not inside, candidate files."""
    finalize_output_nll_report(report)
    task_key = str(report["task_key"])
    path = output_nll_report_path(inference_output_dir, task_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, sort_keys=True)
        os.replace(temporary_path, path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    return path


def load_output_nll_reports(
    inference_output_dir: str,
    selected_task_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load only selected task reports; a missing sidecar is not an error."""
    reports = {}
    for task_key in selected_task_keys:
        path = output_nll_report_path(inference_output_dir, task_key)
        try:
            with open(path, encoding="utf-8") as report_file:
                report = json.load(report_file)
        except FileNotFoundError:
            continue
        if report.get("schema_version") != OUTPUT_NLL_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported output-NLL report schema in {path!r}: "
                f"{report.get('schema_version')!r}"
            )
        if report.get("task_key") != task_key:
            raise ValueError(
                f"Output-NLL report {path!r} belongs to "
                f"{report.get('task_key')!r}, expected {task_key!r}"
            )
        reports[task_key] = report
    return reports


def find_output_nll_entry(
    report: Mapping[str, Any] | None,
    base_key: str,
    grid: Any,
) -> Mapping[str, Any] | None:
    """Find the canonical NLL row for an exact grid, if it was scored."""
    if report is None:
        return None
    output = report.get("outputs", {}).get(base_key)
    if not isinstance(output, Mapping):
        return None
    wanted = output_grid_key(grid)
    for entry in output.get("scores", ()):
        if output_grid_key(entry.get("grid")) == wanted:
            return entry
    return None


__all__ = [
    "OutputNllTarget",
    "apply_output_nll_scores",
    "find_output_nll_entry",
    "finalize_output_nll_report",
    "load_available_ground_truths",
    "load_output_nll_reports",
    "output_grid_key",
    "output_nll_report_dir",
    "output_nll_report_path",
    "prepare_output_nll_report",
    "write_output_nll_report",
]
