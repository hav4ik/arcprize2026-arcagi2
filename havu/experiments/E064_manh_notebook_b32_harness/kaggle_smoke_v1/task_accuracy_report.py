"""Post-inference diagnostics only: labels never enter search or selection.

Candidate files are trusted, application-produced bzip2 pickles. Do not point
this helper at untrusted downloads. Each run has its own atomic JSON sidecars,
so independent inference services can report cumulative completed-task scores.
"""

from __future__ import annotations

import bz2
import json
import os
from pathlib import Path
import pickle
import re
import time
import uuid

import numpy as np

from arc_decoder import DEFAULT_SELECTION_ALGORITHM, get_selection_algorithm
from output_nll_report import normalize_output_grid
from run_config import resolve_evaluation_solution_path


def load_task_truths(task_key, query, solution_path=None):
    """Load inline or validation-file labels, but never during private reruns."""
    count = len(query["test"])
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        return [None] * count, "private_rerun"
    truths = [p.get("output") for p in query["test"]]
    source = "inline_test_outputs"
    if any(t is None for t in truths):
        try:
            with open(solution_path or resolve_evaluation_solution_path()) as f:
                values = json.load(f).get(task_key)
        except FileNotFoundError:
            values = None
        if values is not None:
            if len(values) != count:
                raise ValueError(f"Test/solution count mismatch for {task_key}")
            truths = [old if old is not None else values[i] for i, old in enumerate(truths)]
            source = "validation_solution_file"
        elif all(t is None for t in truths):
            source = "unavailable"
    return [None if t is None else normalize_output_grid(t) for t in truths], source


def read_completed_candidates(output_dir, task_key, output_count, candidate_paths=None):
    allowed = {f"{task_key}_{i}" for i in range(output_count)}
    root = Path(output_dir).resolve()
    paths = root.iterdir() if candidate_paths is None else map(Path, candidate_paths)
    records = {k: {} for k in sorted(allowed)}
    for path in sorted(paths, key=lambda p: p.name):
        base_key = path.name.split(".")[0]
        if base_key not in allowed or path.parent.resolve() != root or not path.is_file():
            continue  # Excludes train predictions, other tasks and sidecar JSON.
        with bz2.BZ2File(path) as f:
            outputs = pickle.load(f)
        for i, candidate in enumerate(outputs):
            records[base_key][f"{path.name}.out{i}"] = candidate
    return records


def evaluate_task_candidates(task_key, truths, records, selection_algorithm):
    """Match submit.py pass@2, including duplication and [[0]] fallback."""
    select = get_selection_algorithm(selection_algorithm)
    rows = []
    for index, truth in enumerate(truths):
        key = f"{task_key}_{index}"
        candidates = records.get(key, {})
        ranked = select(dict(candidates))  # No truth is passed to the selector.
        attempts = list(ranked[:2]) or [np.asarray([[0]], dtype=np.int64)]
        if len(attempts) == 1:
            attempts.append(attempts[-1])
        rows.append({
            "base_key": key,
            "labeled": truth is not None,
            "correct": None if truth is None else any(np.array_equal(a, truth) for a in attempts),
            "oracle_correct": None if truth is None else any(np.array_equal(c["solution"], truth) for c in candidates.values()),
            "candidate_count": len(candidates),
            "empty": not candidates,
        })
    labeled = sum(r["labeled"] for r in rows)
    correct = sum(r["correct"] is True for r in rows)
    oracle = sum(r["oracle_correct"] is True for r in rows)
    complete = labeled == len(rows) and bool(rows)
    return {
        "schema_version": 1, "task_key": task_key,
        "selection_algorithm": selection_algorithm,
        "status": "complete" if complete else ("partial_labels" if labeled else "unavailable"),
        "test_outputs": len(rows), "labeled_outputs": labeled,
        "correct_outputs": correct, "oracle_correct_outputs": oracle,
        "score": correct / len(rows) if complete else None,
        "oracle_score": oracle / len(rows) if complete else None,
        "empty_outputs": sum(r["empty"] for r in rows),
        "outputs": rows,
    }


def report_completed_task_accuracy(
    *, task_key, query, output_dir, run_id, candidate_paths=None,
    selection_algorithm=None, solution_path=None, log_prefix="[vLLM]",
):
    """Called only after all task candidates and rescoring files are complete."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_id)) or not re.fullmatch(r"[A-Za-z0-9_-]+", str(task_key)):
        raise ValueError("Accuracy report run/task IDs must be safe path components")
    selection = selection_algorithm or os.getenv("ARC_SELECTION_ALGORITHM", DEFAULT_SELECTION_ALGORITHM)
    truths, source = load_task_truths(task_key, query, solution_path)
    # A hidden/missing-label report does not even need to deserialize candidates.
    records = read_completed_candidates(output_dir, task_key, len(truths), candidate_paths) if any(t is not None for t in truths) else {}
    report = evaluate_task_candidates(task_key, truths, records, selection)
    if not any(t is not None for t in truths):
        report["empty_outputs"] = None  # Candidate files were not read.
        for row in report["outputs"]:
            row.update(candidate_count=None, empty=None)
    report.update(run_id=str(run_id), label_source=source, created_unix=time.time())
    root = Path(output_dir) / "task_accuracy_reports" / str(run_id)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{task_key}.json"
    temporary = root / f".{task_key}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if report["status"] == "unavailable":
        print(f"{log_prefix} accuracy for {task_key}: unavailable ({source}); not counted as zero", flush=True)
        return report
    completed = []
    for p in root.glob("*.json"):
        value = json.loads(p.read_text())
        if value.get("run_id") == str(run_id) and value.get("selection_algorithm") == selection and value.get("status") == "complete":
            completed.append(value)
    completed.sort(key=lambda r: r["task_key"])
    cumulative = {
        "completed_labeled_tasks": len(completed),
        "score": sum(r["score"] for r in completed),
        "oracle_score": sum(r["oracle_score"] for r in completed),
        "correct_outputs": sum(r["correct_outputs"] for r in completed),
        "test_outputs": sum(r["test_outputs"] for r in completed),
    }
    report["cumulative_at_report_time"] = cumulative
    task_score = f"{report['score']:.3f}/1" if report["score"] is not None else "partial labels"
    print(
        f"{log_prefix} accuracy for {task_key}: selector={selection}, "
        f"score={task_score}, correct={report['correct_outputs']}/{report['labeled_outputs']} outputs, "
        f"oracle={report['oracle_correct_outputs']}/{report['labeled_outputs']} outputs, "
        f"empty={report['empty_outputs']}; completed-so-far="
        f"{cumulative['score']:.3f}/{cumulative['completed_labeled_tasks']} tasks, "
        f"oracle={cumulative['oracle_score']:.3f}/{cumulative['completed_labeled_tasks']}, "
        f"correct={cumulative['correct_outputs']}/{cumulative['test_outputs']} outputs",
        flush=True,
    )
    return report
