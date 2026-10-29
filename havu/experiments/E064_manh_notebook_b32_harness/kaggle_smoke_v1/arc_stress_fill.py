"""Opt-in ARC prompt filling for long-context train/inference stress tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


Grid = list[list[int]]
Pair = dict[str, Grid]


def _grid_signature(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def _pair_signature(pair: Pair) -> tuple[Any, Any]:
    return _grid_signature(pair["input"]), _grid_signature(pair["output"])


def _dihedral_transform(grid: Grid, transform_id: int) -> Grid:
    array = np.asarray(grid, dtype=np.uint8)
    if transform_id == 0:
        transformed = array
    elif transform_id == 1:
        transformed = np.rot90(array, k=1)
    elif transform_id == 2:
        transformed = np.rot90(array, k=2)
    elif transform_id == 3:
        transformed = np.rot90(array, k=3)
    elif transform_id == 4:
        transformed = np.fliplr(array)
    elif transform_id == 5:
        transformed = np.flipud(array)
    elif transform_id == 6:
        transformed = array.T
    elif transform_id == 7:
        transformed = np.fliplr(np.rot90(array, k=1))
    else:
        raise ValueError(f"Invalid dihedral transform {transform_id}")
    return transformed.tolist()


def _augment_pair(pair: Pair, transform_id: int, permutation: list[int]) -> Pair:
    lookup = np.asarray(permutation, dtype=np.uint8)
    result: Pair = {}
    for name in ("input", "output"):
        transformed = np.asarray(
            _dihedral_transform(pair[name], transform_id), dtype=np.uint8
        )
        result[name] = lookup[transformed].tolist()
    return result


def _exact_train_tokens(formatter, tokenizer, train_pairs: list[Pair]) -> int:
    return len(tokenizer.encode(formatter.fmt_train(train_pairs)))


def _generate_fill_candidates(
    train_pairs: list[Pair],
    test_inputs: list[Grid],
    *,
    seed_text: str,
    count: int = 128,
) -> list[tuple[Pair, dict[str, Any]]]:
    seed = int.from_bytes(
        hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], byteorder="big"
    )
    rng = random.Random(seed)
    seen_pairs = {_pair_signature(pair) for pair in train_pairs}
    heldout_inputs = {_grid_signature(grid) for grid in test_inputs}
    candidates: list[tuple[Pair, dict[str, Any]]] = []

    for _ in range(max(4096, count * 64)):
        base_pair_index = rng.randrange(len(train_pairs))
        transform_id = rng.randrange(8)
        permutation = list(range(10))
        rng.shuffle(permutation)
        pair = _augment_pair(
            train_pairs[base_pair_index], transform_id, permutation
        )
        signature = _pair_signature(pair)
        if signature in seen_pairs or _grid_signature(pair["input"]) in heldout_inputs:
            continue
        seen_pairs.add(signature)
        candidates.append(
            (
                pair,
                {
                    "kind": "reaugmented_train_pair",
                    "base_pair_index": base_pair_index,
                    "transform_id": transform_id,
                    "color_mapping": permutation,
                },
            )
        )
        if len(candidates) == count:
            break
    return candidates


def _choose_fill_candidates(
    candidates: list[tuple[Pair, dict[str, Any]]],
    formatter,
    tokenizer,
    capacity: int,
) -> list[tuple[Pair, dict[str, Any]]]:
    if capacity <= 0:
        return []
    weighted = [
        (pair, provenance, _exact_train_tokens(formatter, tokenizer, [pair]))
        for pair, provenance in candidates
    ]
    paths: dict[int, tuple[int, ...]] = {0: ()}
    for candidate_index, (_, _, contribution) in enumerate(weighted):
        additions: dict[int, tuple[int, ...]] = {}
        for current_tokens, path in paths.items():
            new_tokens = current_tokens + contribution
            if new_tokens <= capacity and new_tokens not in paths:
                additions.setdefault(new_tokens, path + (candidate_index,))
        paths.update(additions)
    return [
        (weighted[index][0], weighted[index][1]) for index in paths[max(paths)]
    ]


def fill_challenge_tasks(
    challenges: dict[str, Any],
    task_keys: list[str],
    formatter,
    tokenizer,
    target_tokens: int,
    *,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a copy with selected task training prefixes filled to a ceiling.

    Added examples are deterministic independent dihedral/color augmentations
    of existing supervised pairs. They live in the ordinary ``train`` list,
    so every downstream training and inference backend sees the same prefix.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    result = dict(challenges)
    records: list[dict[str, Any]] = []

    for task_key in task_keys:
        task = copy.deepcopy(challenges[task_key])
        base_train = task.get("train", [])
        if not base_train:
            raise ValueError(f"{task_key}: cannot fill a task without training pairs")
        base_tokens = _exact_train_tokens(formatter, tokenizer, base_train)
        candidates = _generate_fill_candidates(
            base_train,
            [example["input"] for example in task.get("test", [])],
            seed_text=f"{seed}:{task_key}:fill",
        )
        selected = _choose_fill_candidates(
            candidates,
            formatter,
            tokenizer,
            max(0, target_tokens - base_tokens),
        )
        task["train"] = base_train + [pair for pair, _ in selected]
        final_tokens = _exact_train_tokens(formatter, tokenizer, task["train"])
        if final_tokens > max(base_tokens, target_tokens):
            raise RuntimeError(f"{task_key}: filled prompt crossed its token ceiling")
        if len({_pair_signature(pair) for pair in task["train"]}) != len(
            task["train"]
        ):
            raise RuntimeError(f"{task_key}: prompt filling produced a duplicate pair")
        result[task_key] = task
        records.append(
            {
                "task_id": task_key,
                "base_train_pairs": len(base_train),
                "added_train_pairs": len(selected),
                "train_pairs": len(task["train"]),
                "base_train_tokens": base_tokens,
                "train_tokens": final_tokens,
                "fill_gap_tokens": max(0, target_tokens - final_tokens),
                "fill_examples": [provenance for _, provenance in selected],
            }
        )
    return result, records


def measure_augmented_context(
    challenges: dict[str, Any],
    task_keys: list[str],
    formatter,
    tokenizer,
) -> dict[str, int]:
    """Measure all eight geometric views used by ARC train/vLLM augmentation."""

    max_train_tokens = 0
    max_inference_prompt_tokens = 0
    for task_key in task_keys:
        task = challenges[task_key]
        for transform_id in range(8):
            transformed_train = [
                {
                    "input": _dihedral_transform(pair["input"], transform_id),
                    "output": _dihedral_transform(pair["output"], transform_id),
                }
                for pair in task["train"]
            ]
            train_text = formatter.fmt_train(transformed_train)
            train_tokens = len(tokenizer.encode(train_text))
            max_train_tokens = max(max_train_tokens, train_tokens)
            for test_example in task.get("test", []):
                transformed_test = {
                    "input": _dihedral_transform(
                        test_example["input"], transform_id
                    )
                }
                prompt = train_text + formatter.fmt_query([transformed_test])
                max_inference_prompt_tokens = max(
                    max_inference_prompt_tokens,
                    len(tokenizer.encode(prompt)),
                )
    generation_reserve = int(formatter.max_new_tokens())
    return {
        "max_train_tokens": max_train_tokens,
        "max_inference_prompt_tokens": max_inference_prompt_tokens,
        "generation_reserve": generation_reserve,
        "required_vllm_context": max_inference_prompt_tokens
        + generation_reserve,
    }


def write_runtime_challenge(
    challenges: dict[str, Any],
    records: list[dict[str, Any]],
    context: dict[str, int],
    *,
    source_path: str,
    target_tokens: int,
    output_root: str | None = None,
) -> tuple[str, str]:
    root = Path(
        output_root
        or os.getenv("ARC_STRESS_FILL_OUTPUT_DIR")
        or (
            "/kaggle/working/arc_stress_fill"
            if Path("/kaggle/working").is_dir()
            else ".arc_stress_fill"
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    identity = json.dumps(
        {
            "source": os.path.abspath(source_path),
            "target_tokens": target_tokens,
            "task_ids": [record["task_id"] for record in records],
        },
        sort_keys=True,
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    challenge_path = root / f"arc_filled_{target_tokens}_{suffix}.json"
    manifest_path = root / f"arc_filled_{target_tokens}_{suffix}.manifest.json"

    def atomic_dump(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)

    atomic_dump(challenge_path, challenges)
    atomic_dump(
        manifest_path,
        {
            "source_challenge_path": os.path.abspath(source_path),
            "fill_train_to_tokens": target_tokens,
            "context": context,
            "records": records,
        },
    )
    return str(challenge_path), str(manifest_path)
