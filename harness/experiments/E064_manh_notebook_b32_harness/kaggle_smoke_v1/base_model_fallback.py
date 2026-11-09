"""Label-free policy and artifact handling for one base-model recovery pass.

Only deserialize trusted candidate artifacts produced by this application.
Base and LoRA likelihoods are deliberately never pooled for selection.
"""

from __future__ import annotations

import bz2
import copy
import json
import os
from pathlib import Path
import pickle

import numpy as np

from arc_decoder import DEFAULT_SELECTION_ALGORITHM, get_selection_algorithm
from task_accuracy_report import read_completed_candidates


def is_black_placeholder(grid) -> bool:
    array = np.asarray(grid)
    return array.shape == (1, 1) and bool(array[0, 0] == 0)


def black_placeholder_outputs(task_key, output_count, output_dir, candidate_paths):
    """Find outputs whose production top two attempts are both [[0]]."""
    records = read_completed_candidates(
        output_dir, task_key, output_count, candidate_paths,
    )
    select = get_selection_algorithm(
        os.getenv("ARC_SELECTION_ALGORITHM", DEFAULT_SELECTION_ALGORITHM)
    )
    targets = []
    for index in range(output_count):
        base_key = f"{task_key}_{index}"
        attempts = list(select(dict(records[base_key]))[:2])
        if not attempts or all(is_black_placeholder(grid) for grid in attempts):
            targets.append(base_key)
    return tuple(targets)


def write_fallback_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def publish_base_candidates(
    *, primary_dir, retry_dir, targets, original_paths, owned_paths, adapter_name,
):
    """Publish only non-black retry candidates; archive replaced LoRA files.

    ``owned_paths`` tracks every published/moved file immediately, including
    backups, so interrupted-attempt cleanup never touches another task's files.
    Everything from unsuccessful retry outputs stays in its diagnostic folder.
    """
    primary = Path(primary_dir).resolve()
    retry = Path(retry_dir).resolve()
    groups = []
    for name in sorted(owned_paths):
        path = Path(name)
        base_key = path.name.split(".")[0]
        if path.parent.resolve() != retry or base_key not in targets:
            continue
        with bz2.BZ2File(path) as f:
            candidates = pickle.load(f)
        candidates = [
            dict(candidate, inference_model="base", fallback_from_adapter=adapter_name)
            for candidate in candidates
            if not is_black_placeholder(candidate["solution"])
        ]
        if candidates:
            groups.append((path, base_key, candidates))
    recovered = {base_key for _, base_key, _ in groups}
    for path, _, candidates in groups:
        destination = primary / (path.name + ".base_model")
        temporary = retry / (path.name + ".publish.tmp")
        try:
            with bz2.BZ2File(temporary, "w") as f:
                pickle.dump(candidates, f)
            temporary.replace(destination)
            owned_paths.add(str(destination))
        finally:
            temporary.unlink(missing_ok=True)
    for name in sorted(original_paths):
        path = Path(name)
        if path.parent.resolve() != primary or path.name.split(".")[0] not in recovered:
            continue
        backup = retry / "lora_before" / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        path.replace(backup)
        owned_paths.discard(str(path))
        owned_paths.add(str(backup))
    return tuple(sorted(recovered))


def merge_base_output_nll(primary_report, base_report, recovered):
    """Use the supplying model's canonical NLL for each final output."""
    merged = copy.deepcopy(primary_report)
    model_by_output = {key: "lora" for key in merged["outputs"]}
    for key in recovered:
        merged["outputs"][key] = copy.deepcopy(base_report["outputs"][key])
        merged["outputs"][key]["inference_model"] = "base"
        model_by_output[key] = "base"
    merged["metadata"]["inference_model_by_output"] = model_by_output
    merged["metadata"]["base_model_fallback_outputs"] = list(recovered)
    merged["metadata"]["inference_model"] = (
        "base" if all(model == "base" for model in model_by_output.values()) else "mixed"
    )
    merged["metadata"]["lora_adapter_name"] = merged["metadata"].get("adapter_name")
    merged["metadata"]["adapter_name"] = None  # No one adapter supplied all NLLs.
    return merged
