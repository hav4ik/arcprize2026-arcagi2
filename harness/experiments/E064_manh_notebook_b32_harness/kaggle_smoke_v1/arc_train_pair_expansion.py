"""Opt-in, train-only demonstration expansion within an ARC conversation.

Independent pair augmentation is an experiment, not a guarantee that the
unknown task rule is preserved. Repetition is the rule-preserving control.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path

from arc_stress_fill import _augment_pair, _grid_signature, _pair_signature


def validate_pair_expansion(min_train_pairs: int, mode: str) -> None:
    if not isinstance(min_train_pairs, int) or min_train_pairs < 0:
        raise ValueError("min_train_pairs must be a non-negative integer")
    if mode not in {"augment", "repeat"}:
        raise ValueError("extra_train_pair_mode must be 'augment' or 'repeat'")


def expand_training_query(query, *, key, min_train_pairs=0, mode="augment", seed=0):
    """Keep originals and add pairs from this view's training data only.

    Neither candidate replies nor test outputs participate in generation or
    RNG seeding. Test *inputs* are used only to reject accidental collisions.
    Call after whole-task augmentation (or after holding out a training pair).
    """
    validate_pair_expansion(min_train_pairs, mode)
    original = query.get("train", [])
    if len(original) >= min_train_pairs or min_train_pairs == 0:
        return query, []
    if not original:
        raise ValueError(f"{key}: cannot expand a prompt without training pairs")
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(f"arc-pair-expansion-v1:{seed}:{key}".encode()).digest()[:8],
            "big",
        )
    )
    heldout = {_grid_signature(pair["input"]) for pair in query.get("test", [])}
    seen = {_pair_signature(pair) for pair in original}
    input_outputs = {}
    for pair in original:
        input_outputs.setdefault(_grid_signature(pair["input"]), set()).add(
            _grid_signature(pair["output"])
        )
    repeat_sources = [
        index
        for index, pair in enumerate(original)
        if _grid_signature(pair["input"]) not in heldout
    ]
    result = copy.deepcopy(query)
    records = []
    for extra_index in range(min_train_pairs - len(original)):
        pair = None
        record = None
        if mode == "augment":
            for _ in range(256):
                source_index = rng.randrange(len(original))
                transform_id = rng.randrange(8)
                permutation = list(range(10))
                rng.shuffle(permutation)
                candidate = _augment_pair(
                    original[source_index], transform_id, permutation
                )
                input_sig, output_sig = _pair_signature(candidate)
                if input_sig in heldout or (input_sig, output_sig) in seen:
                    continue
                # This checks observable contradictions only, not the unknown
                # task rule (e.g. move-right vs move-left after rotation).
                if input_sig in input_outputs and input_outputs[input_sig] != {
                    output_sig
                }:
                    continue
                pair = candidate
                record = {
                    "kind": "independent_augmented_train_pair",
                    "source_index": source_index,
                    "transform_id": transform_id,
                    "color_mapping": permutation,
                }
                break
        if pair is None:
            if not repeat_sources:
                raise ValueError(f"{key}: no extra pair avoids the held-out input")
            source_index = rng.choice(repeat_sources)
            pair = copy.deepcopy(original[source_index])
            record = {
                "kind": "repeat" if mode == "repeat" else "repeat_fallback",
                "source_index": source_index,
            }
        seen.add(_pair_signature(pair))
        input_outputs.setdefault(_grid_signature(pair["input"]), set()).add(
            _grid_signature(pair["output"])
        )
        # Do not always put a copy/augmented answer in the final position.
        # The relative order of all original demonstrations stays unchanged.
        position = rng.randrange(len(result["train"]) + 1)
        result["train"].insert(position, pair)
        records.append(
            {**record, "extra_index": extra_index, "insert_position": position}
        )
    return result, records


def write_expansion_manifest(
    dataset, formatter, *, root, task_key, phase, min_train_pairs, mode
):
    """Small reproducible sidecar; candidate loaders ignore this subdirectory."""
    rows = []
    for key in dataset.keys:
        query = dataset.queries[key]
        train_text = formatter.fmt_train(query["train"])
        prompt = train_text + formatter.fmt_query(query["test"])
        rows.append(
            {
                "key": key,
                "train_pairs": len(query["train"]),
                "train_tokens": len(formatter.tokenizer.encode(train_text)),
                "prompt_tokens": len(formatter.tokenizer.encode(prompt)),
                "reply_tokens": sum(
                    len(formatter.tokenizer.encode(formatter.fmt_reply([p["output"]])))
                    for p in query["train"]
                ),
                "train_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "added_pairs": getattr(dataset, "pair_expansion_records", {}).get(
                    key, []
                ),
            }
        )
    path = Path(root) / "train_pair_expansion" / f"{task_key}.{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_key": task_key,
                "phase": phase,
                "min_train_pairs": min_train_pairs,
                "mode": mode,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(temporary, path)
    return str(path)
