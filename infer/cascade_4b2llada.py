"""Run a confidence-routed NVARC Qwen3-4B -> ARChitects LLaDA cascade.

The two solvers intentionally remain separate Python processes. NVARC owns
confidence routing from its raw decoded candidates; ARChitects receives only
the routed whole-task keys and writes an ordinary ARC submission JSON. The
driver then merges the independently ranked attempts without exposing either
runtime's imported model classes or CUDA state to the other.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

from arc_decoder import ArcDecoder, DEFAULT_SELECTION_ALGORITHM
from arc_loader import ArcDataset
from cascade_4b28b import (
    apply_environment_profile,
    atomic_write_json,
    json_safe_number,
    parse_environment_overrides,
    prepare_output_directory,
    run_subprocess,
    validate_starter_args,
)
from confidence_routing import route_tasks_by_confidence
from run_config import (
    resolve_data_challenge_path,
    resolve_evaluation_solution_path,
    select_task_keys,
)
from run_status import (
    STATUS_SUCCESS,
    extract_run_status_path,
    run_entrypoint_with_status,
)


PASSTHROUGH_MARKER = "--stage1-starter-args"
LLADA_MERGE_POLICIES = ("one-each", "prefer-llada", "prefer-4b")
ATTEMPT_NAMES = ("attempt_1", "attempt_2")
ARCHITECTS_INTERNAL_RESERVE_SECONDS = 10 * 60
_FINAL_SUBMISSION_PATH: Path | None = None


@dataclass(frozen=True)
class LLaDAOutputMerge:
    """Audit record for one routed ARC test-output merge."""

    base_key: str
    primary_candidates: int
    llada_candidates: int
    selected_sources: tuple[str, ...]


def split_passthrough_args(argv):
    """Separate cascade options from NVARC ``starter.py`` options."""

    if argv.count(PASSTHROUGH_MARKER) > 1:
        raise ValueError(f"Repeated passthrough marker: {PASSTHROUGH_MARKER}")
    if PASSTHROUGH_MARKER not in argv:
        return list(argv), []
    marker_index = argv.index(PASSTHROUGH_MARKER)
    return list(argv[:marker_index]), list(argv[marker_index + 1 :])


def _starter_option_value(arguments, names, *, label):
    """Read one value option from passthrough arguments in either CLI form."""

    matches = []
    names = tuple(names)
    for index, token in enumerate(arguments):
        for name in names:
            if token == name:
                if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                    raise ValueError(f"{label} requires a value")
                matches.append(arguments[index + 1])
                break
            prefix = name + "="
            if token.startswith(prefix):
                matches.append(token[len(prefix) :])
                break
    if len(matches) > 1:
        raise ValueError(f"{label} may be specified only once")
    return matches[0] if matches else None


def _starter_boolean_option(arguments, positive, negative, *, label):
    """Read one BooleanOptionalAction-style passthrough flag."""

    matches = []
    for token in arguments:
        if token == positive:
            matches.append(True)
        elif token == negative:
            matches.append(False)
    if len(matches) > 1:
        raise ValueError(f"{label} may be specified only once")
    return matches[0] if matches else None


def _environment_boolean(value, *, label):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must be a Boolean value; got {value!r}")


def resolve_shared_adapter_staging(args, starter_args, environment):
    """Return normalized equivalent CLI options for both isolated stages."""

    starter_args = list(starter_args)
    starter_storage = _starter_option_value(
        starter_args,
        ("--inference-adapter-storage",),
        label="--inference-adapter-storage",
    )
    requested_storage = args.inference_adapter_storage
    if (
        requested_storage is not None
        and starter_storage is not None
        and requested_storage != starter_storage
    ):
        raise ValueError(
            "Conflicting shared adapter storage values: top-level "
            f"{requested_storage!r} versus stage-1 {starter_storage!r}"
        )
    storage = (
        requested_storage
        or starter_storage
        or environment.get("ARC_INFERENCE_ADAPTER_STORAGE")
        or "auto"
    )
    if storage not in {"auto", "ram", "disk"}:
        raise ValueError(
            "Shared adapter storage must be auto, ram, or disk; "
            f"got {storage!r}"
        )
    if starter_storage is None:
        starter_args.extend(("--inference-adapter-storage", storage))
    architects_args = ["--inference-adapter-storage", storage]

    starter_root = _starter_option_value(
        starter_args,
        ("--inference-adapter-root", "--vllm-adapter-root"),
        label="--inference-adapter-root/--vllm-adapter-root",
    )
    requested_root = args.inference_adapter_root
    if requested_root and starter_root and requested_root != starter_root:
        raise ValueError(
            "Conflicting shared adapter roots: top-level "
            f"{requested_root!r} versus stage-1 {starter_root!r}"
        )
    root = (
        requested_root
        or starter_root
        or environment.get("ARC_INFERENCE_ADAPTER_ROOT")
        or environment.get("ARC_VLLM_ADAPTER_ROOT")
    )
    if root:
        root = os.fspath(root)
        if starter_root is None:
            starter_args.extend(("--inference-adapter-root", root))
        architects_args.extend(("--inference-adapter-root", root))

    starter_ram_root = _starter_option_value(
        starter_args,
        ("--inference-ram-adapter-root",),
        label="--inference-ram-adapter-root",
    )
    if (
        args.inference_ram_adapter_root
        and starter_ram_root
        and args.inference_ram_adapter_root != starter_ram_root
    ):
        raise ValueError("Conflicting shared RAM adapter roots")
    ram_root = (
        args.inference_ram_adapter_root
        or starter_ram_root
        or environment.get("ARC_INFERENCE_RAM_ADAPTER_ROOT")
    )
    starter_disk_root = _starter_option_value(
        starter_args,
        ("--inference-disk-adapter-root",),
        label="--inference-disk-adapter-root",
    )
    if (
        args.inference_disk_adapter_root
        and starter_disk_root
        and args.inference_disk_adapter_root != starter_disk_root
    ):
        raise ValueError("Conflicting shared disk adapter roots")
    disk_root = (
        args.inference_disk_adapter_root
        or starter_disk_root
        or environment.get("ARC_INFERENCE_DISK_ADAPTER_ROOT")
    )
    for value, option, starter_value in (
        (ram_root, "--inference-ram-adapter-root", starter_ram_root),
        (disk_root, "--inference-disk-adapter-root", starter_disk_root),
    ):
        if value:
            value = os.fspath(value)
            if starter_value is None:
                starter_args.extend((option, value))
            architects_args.extend((option, value))

    starter_pending = _starter_option_value(
        starter_args,
        ("--vllm-max-pending-adapters",),
        label="--vllm-max-pending-adapters",
    )
    requested_pending = args.vllm_max_pending_adapters
    if (
        requested_pending is not None
        and starter_pending is not None
        and int(requested_pending) != int(starter_pending)
    ):
        raise ValueError(
            "Conflicting shared pending-adapter limits: top-level "
            f"{requested_pending!r} versus stage-1 {starter_pending!r}"
        )
    pending = (
        requested_pending
        if requested_pending is not None
        else starter_pending
    )
    if pending is None:
        pending = environment.get("ARC_VLLM_MAX_PENDING_ADAPTERS")
    if pending is not None:
        pending = int(pending)
        if pending < 1:
            raise ValueError("Shared pending-adapter limit must be positive")
        if starter_pending is None:
            starter_args.extend(("--vllm-max-pending-adapters", str(pending)))
        architects_args.extend(("--vllm-max-pending-adapters", str(pending)))

    starter_scale = _starter_boolean_option(
        starter_args,
        "--vllm-scale-after-training",
        "--no-vllm-scale-after-training",
        label="--[no-]vllm-scale-after-training",
    )
    requested_scale = args.vllm_scale_after_training
    if (
        requested_scale is not None
        and starter_scale is not None
        and requested_scale != starter_scale
    ):
        raise ValueError(
            "Conflicting shared post-training scale flags between the "
            "top-level cascade and stage-1 passthrough"
        )
    scale = requested_scale if requested_scale is not None else starter_scale
    if scale is None and "ARC_VLLM_SCALE_AFTER_TRAINING" in environment:
        scale = _environment_boolean(
            environment["ARC_VLLM_SCALE_AFTER_TRAINING"],
            label="ARC_VLLM_SCALE_AFTER_TRAINING",
        )
    if scale is not None:
        scale_option = (
            "--vllm-scale-after-training"
            if scale
            else "--no-vllm-scale-after-training"
        )
        if starter_scale is None:
            starter_args.append(scale_option)
        architects_args.append(scale_option)
    return starter_args, architects_args


def _grid_key(grid):
    """Return a stable hashable representation of an ARC grid."""

    return tuple(tuple(int(value) for value in row) for row in grid)


def _is_zero_placeholder(grid):
    return _grid_key(grid) == ((0,),)


def submission_output_candidates(output, *, generated_count=None):
    """Extract real ranked attempts from one submission output.

    ``generated_count`` comes from the ARChitects result manifest and removes
    ambiguity between its schema fallback ``[[0]]`` and a genuinely generated
    all-black 1x1 grid. Without a manifest, an all-zero pair is treated as
    missing, while a mixed pair retains both attempts conservatively.
    """

    attempts = [deepcopy(output[name]) for name in ATTEMPT_NAMES]
    if generated_count is not None:
        count = max(0, min(len(attempts), int(generated_count)))
        return attempts[:count]
    if all(_is_zero_placeholder(attempt) for attempt in attempts):
        return []
    return attempts


def merge_nvarc_llada_submissions(
    primary_submission,
    llada_submission,
    decisions,
    queries,
    *,
    policy="one-each",
    primary_output_counts=None,
    llada_output_counts=None,
):
    """Merge LLaDA attempts only for tasks selected by the NVARC router."""

    if policy not in LLADA_MERGE_POLICIES:
        supported = ", ".join(LLADA_MERGE_POLICIES)
        raise ValueError(
            f"Unknown LLaDA merge policy {policy!r}; expected {supported}"
        )
    primary_output_counts = primary_output_counts or {}
    llada_output_counts = llada_output_counts or {}
    merged = deepcopy(primary_submission)
    audit = []

    for decision in decisions:
        if not decision.route:
            continue
        task_key = decision.task_key
        llada_task = llada_submission.get(task_key, [])
        for test_index in range(len(queries[task_key]["test"])):
            base_key = f"{task_key}_{test_index}"
            primary_output = primary_submission[task_key][test_index]
            primary = submission_output_candidates(
                primary_output,
                generated_count=primary_output_counts.get(base_key, 2),
            )
            llada_output = (
                llada_task[test_index]
                if test_index < len(llada_task)
                else {name: [[0]] for name in ATTEMPT_NAMES}
            )
            llada = submission_output_candidates(
                llada_output,
                generated_count=llada_output_counts.get(base_key),
            )

            if policy == "one-each":
                ranked_groups = (
                    ("llada", llada[:1]),
                    ("4b", primary[:1]),
                    ("llada", llada[1:]),
                    ("4b", primary[1:]),
                )
            elif policy == "prefer-llada":
                ranked_groups = (("llada", llada), ("4b", primary))
            else:
                ranked_groups = (("4b", primary), ("llada", llada))

            selected = []
            selected_sources = []
            seen = set()
            for source, candidates in ranked_groups:
                for candidate in candidates:
                    key = _grid_key(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    selected.append(candidate)
                    selected_sources.append(source)
                    if len(selected) == len(ATTEMPT_NAMES):
                        break
                if len(selected) == len(ATTEMPT_NAMES):
                    break

            while len(selected) < len(ATTEMPT_NAMES):
                selected.append(deepcopy(selected[-1]) if selected else [[0]])
                selected_sources.append(
                    selected_sources[-1] if selected_sources else "missing"
                )
            merged[task_key][test_index] = {
                name: selected[index]
                for index, name in enumerate(ATTEMPT_NAMES)
            }
            audit.append(
                LLaDAOutputMerge(
                    base_key=base_key,
                    primary_candidates=len(primary),
                    llada_candidates=len(llada),
                    selected_sources=tuple(selected_sources),
                )
            )

    return merged, audit


def load_llada_output_counts(path):
    """Load exact per-output candidate counts from an ARChitects manifest."""

    if not path or not Path(path).is_file():
        return {}
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported ARChitects result manifest: {path}")
    return {
        str(base_key): int(count)
        for base_key, count in payload.get("output_candidate_counts", {}).items()
    }


def validate_submission_shape(submission, queries, *, label):
    """Fail closed on malformed cross-pipeline submission JSON."""

    if not isinstance(submission, dict):
        raise ValueError(f"{label} submission must be a JSON object")
    missing = [task_key for task_key in queries if task_key not in submission]
    if missing:
        raise ValueError(f"{label} submission is missing tasks: {missing[:8]}")
    for task_key, task in queries.items():
        outputs = submission[task_key]
        if not isinstance(outputs, list) or len(outputs) != len(task["test"]):
            raise ValueError(
                f"{label} submission has the wrong output count for {task_key}"
            )
        for output in outputs:
            if not isinstance(output, dict) or any(
                name not in output for name in ATTEMPT_NAMES
            ):
                raise ValueError(
                    f"{label} submission has malformed attempts for {task_key}"
                )


def score_submission(data, submission, task_keys):
    """Return the task-weighted ARC score, or ``None`` without labels."""

    replies = getattr(data, "replies", None)
    if not replies or any(task_key not in replies for task_key in task_keys):
        return None

    score = 0.0
    for task_key in task_keys:
        targets = replies[task_key]
        for test_index, target in enumerate(targets):
            output = submission[task_key][test_index]
            target_key = _grid_key(target)
            if any(
                _grid_key(output[attempt_name]) == target_key
                for attempt_name in ATTEMPT_NAMES
            ):
                score += 1.0 / len(targets)
    return score


def cascade_evaluation_scores(
    data,
    selected_keys,
    primary_submission,
    llada_submission,
    final_submission,
):
    """Build comparable stage and merged scores when solutions are loaded."""

    stage1_score = score_submission(data, primary_submission, selected_keys)
    if stage1_score is None:
        return None
    stage2_score = (
        score_submission(data, llada_submission, selected_keys)
        if llada_submission
        else None
    )
    final_score = score_submission(data, final_submission, selected_keys)
    return {
        "max_score": len(selected_keys),
        "stage1_4b": stage1_score,
        "stage2_llada": stage2_score,
        "final_merged": final_score,
        "delta_vs_stage1": final_score - stage1_score,
    }


def print_cascade_evaluation_scores(scores):
    """Print one grep-friendly score summary at the end of the cascade."""

    if scores is None:
        return
    stage2 = scores["stage2_llada"]
    stage2_text = "n/a" if stage2 is None else f"{stage2:.5f}"
    print(
        "*** Cascade score summary: "
        f"stage1_4b={scores['stage1_4b']:.5f}, "
        f"stage2_llada={stage2_text}, "
        f"final_merged={scores['final_merged']:.5f}, "
        f"delta_vs_stage1={scores['delta_vs_stage1']:+.5f}, "
        f"max_score={scores['max_score']}"
    )


def build_primary_submission_and_decisions(
    data,
    selected_keys,
    output_dir,
    *,
    selection_algorithm,
    kgmon_threshold,
    beam_threshold,
    candidate_read_errors=None,
):
    """Rank fresh NVARC candidates once and derive whole-task routing."""

    from early_routing import early_routed_task_keys

    early_skipped = early_routed_task_keys(output_dir, data.queries, selected_keys)

    split_data = data.split_multi_replies()
    decoder = ArcDecoder(split_data, n_guesses=2)
    allowed_base_keys = {
        f"{task_key}_{test_index}"
        for task_key in selected_keys
        for test_index in range(len(data.queries[task_key]["test"]))
    }
    decoder.load_decoded_results(
        os.fspath(output_dir),
        allowed_base_keys={key for key in allowed_base_keys if key.rsplit("_", 1)[0] not in early_skipped},
        skip_unreadable=candidate_read_errors is not None,
    )
    if candidate_read_errors is not None:
        candidate_read_errors.extend(decoder.candidate_read_errors)
    primary_predictions = decoder.run_selection_algo(selection_algorithm)
    primary_submission = data.get_submission(primary_predictions)
    primary_output_counts = {
        base_key: min(len(ATTEMPT_NAMES), len(primary_predictions.get(base_key, ())))
        for base_key in allowed_base_keys
    }
    decisions = route_tasks_by_confidence(
        data.queries,
        decoder.decoded_results,
        selected_keys,
        kgmon_threshold=kgmon_threshold,
        beam_threshold=beam_threshold,
    )
    return primary_submission, decisions, primary_output_counts


def infer_architects_split(challenge_path):
    """Map an ARC challenge filename to ARChitects contest split naming."""

    name = Path(challenge_path).name.lower()
    for split in ("evaluation", "test", "training"):
        if f"_{split}_" in name:
            return split
    return "test" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") else "evaluation"


def routing_manifest_payload(
    decisions,
    *,
    selected_keys,
    output_4b,
    llada_submission,
    model_4b,
    llada_base_model,
    llada_size_model,
    kgmon_threshold,
    beam_threshold,
    merge_policy,
):
    records = []
    for decision in decisions:
        record = asdict(decision)
        record["reasons"] = list(record["reasons"])
        record["missing_outputs"] = list(record["missing_outputs"])
        record["min_kgmon"] = json_safe_number(record["min_kgmon"])
        record["max_beam_score"] = json_safe_number(
            record["max_beam_score"]
        )
        records.append(record)
    return {
        "schema_version": 1,
        "created_at": time.time(),
        "routing_rule": {
            "missing_output": True,
            "kgmon_threshold": float(kgmon_threshold),
            "beam_threshold": float(beam_threshold),
        },
        "merge_policy": merge_policy,
        "selected_task_keys": list(selected_keys),
        "routed_task_keys": [
            decision.task_key for decision in decisions if decision.route
        ],
        "models": {
            "nvarc_4b": str(model_4b),
            "llada_base": str(llada_base_model),
            "llada_size": str(llada_size_model),
        },
        "artifacts": {
            "nvarc_candidates": str(output_4b),
            "llada_submission": str(llada_submission),
        },
        "decisions": records,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run NVARC Qwen3-4B, route uncertain whole tasks to an isolated "
            "ARChitects LLaDA stage, and merge the final two attempts. "
            f"NVARC starter options follow {PASSTHROUGH_MARKER}."
        )
    )
    parser.add_argument("--model-4b", required=True)
    parser.add_argument("--architects-base-model", required=True)
    parser.add_argument("--architects-size-model", required=True)
    parser.add_argument(
        "--architects-dir",
        required=True,
        help="Directory containing the ARChitects submit.py entrypoint.",
    )
    parser.add_argument("--architects-data-path")
    parser.add_argument("--architects-split")
    parser.add_argument(
        "--output-root",
        default="/kaggle/working/arc_4b2llada_cascade",
    )
    parser.add_argument("--output-4b")
    parser.add_argument("--primary-submission")
    parser.add_argument("--llada-submission")
    parser.add_argument("--llada-result-manifest")
    parser.add_argument("--routing-manifest")
    parser.add_argument(
        "--final-submission",
        default="/kaggle/working/submission.json",
    )
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--finalization-reserve-seconds", type=float, default=120.0)
    parser.add_argument(
        "--routing-kgmon-threshold",
        type=float,
        default=float(os.getenv("ARC_ROUTING_KGMON_THRESHOLD", "-5")),
    )
    parser.add_argument(
        "--routing-beam-threshold",
        type=float,
        default=float(os.getenv("ARC_ROUTING_BEAM_THRESHOLD", "0.7")),
    )
    parser.add_argument(
        "--merge-policy",
        choices=LLADA_MERGE_POLICIES,
        default="one-each",
    )
    parser.add_argument(
        "--selection-algorithm",
        default=os.getenv("ARC_SELECTION_ALGORITHM", DEFAULT_SELECTION_ALGORITHM),
    )
    parser.add_argument("--stage1-env", action="append", default=[])
    parser.add_argument("--architects-env", action="append", default=[])
    parser.add_argument(
        "--inference-adapter-storage",
        choices=("auto", "ram", "disk"),
        default=None,
        help=(
            "Shared temporary adapter policy for both NVARC and LLaDA. "
            "If omitted, a stage-1 passthrough value is promoted to LLaDA; "
            "otherwise the cascade defaults to auto."
        ),
    )
    parser.add_argument(
        "--inference-adapter-root",
        help=(
            "Optional explicit shared adapter parent for both stages. This "
            "disables RAM-to-disk spill, matching each solver's root override."
        ),
    )
    parser.add_argument(
        "--inference-ram-adapter-root",
        help="Optional shared RAM-tier parent used by auto or ram mode.",
    )
    parser.add_argument(
        "--inference-disk-adapter-root",
        help="Optional shared disk-tier parent used by auto or disk mode.",
    )
    parser.add_argument(
        "--vllm-max-pending-adapters",
        type=int,
        default=None,
        help=(
            "Optional shared bounded adapter-queue capacity. A value after "
            "--stage1-starter-args is also promoted to LLaDA."
        ),
    )
    parser.add_argument(
        "--vllm-scale-after-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable trainer-GPU conversion to TP=1 vLLM services "
            "in both cascade stages."
        ),
    )
    parser.add_argument("--reuse-4b-output", action="store_true")
    parser.add_argument("--reuse-llada-submission", action="store_true")
    parser.add_argument(
        "--require-llada-success",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--run-status-file")
    return parser


def _load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def validate_llada_artifact_paths(
    submission_path,
    result_manifest_path,
    *,
    reuse,
):
    """Prevent a failed secondary run from accepting stale JSON artifacts."""

    submission_path = Path(submission_path)
    result_manifest_path = Path(result_manifest_path)
    if reuse:
        if not submission_path.is_file():
            raise FileNotFoundError(
                f"Cannot reuse missing LLaDA submission: {submission_path}"
            )
        return
    existing = [
        path
        for path in (submission_path, result_manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing stale LLaDA cascade artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    result_manifest_path.parent.mkdir(parents=True, exist_ok=True)


def _final_submission_exists():
    return (
        _FINAL_SUBMISSION_PATH is not None
        and _FINAL_SUBMISSION_PATH.is_file()
        and _FINAL_SUBMISSION_PATH.stat().st_size > 0
    )


def main():
    global _FINAL_SUBMISSION_PATH

    cascade_argv, starter_args = split_passthrough_args(sys.argv[1:])
    args = build_parser().parse_args(cascade_argv)
    validate_starter_args(starter_args, "stage 1 starter arguments")
    if args.finalization_reserve_seconds < 0:
        raise ValueError("--finalization-reserve-seconds cannot be negative")

    script_dir = Path(__file__).resolve().parent
    architects_dir = Path(args.architects_dir).expanduser().resolve()
    architects_entrypoint = architects_dir / "submit.py"
    if not architects_entrypoint.is_file():
        raise FileNotFoundError(
            f"ARChitects entrypoint does not exist: {architects_entrypoint}"
        )

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_4b = Path(args.output_4b or output_root / "4b").resolve()
    primary_path = Path(
        args.primary_submission or output_root / "nvarc_submission.json"
    ).resolve()
    llada_path = Path(
        args.llada_submission or output_root / "llada_submission.json"
    ).resolve()
    llada_result_path = Path(
        args.llada_result_manifest or output_root / "llada_results.json"
    ).resolve()
    routing_path = Path(
        args.routing_manifest or output_root / "routing_manifest.json"
    ).resolve()
    final_path = Path(args.final_submission).expanduser().resolve()
    _FINAL_SUBMISSION_PATH = final_path
    output_4b = prepare_output_directory(
        output_4b,
        reuse=args.reuse_4b_output,
        label="4b",
    )

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    challenge_path = resolve_data_challenge_path(rerun_mode)
    data = ArcDataset.from_file(challenge_path)
    if not rerun_mode:
        solution_path = Path(resolve_evaluation_solution_path())
        if solution_path.is_file():
            try:
                data.load_replies(solution_path)
            except (OSError, KeyError, json.JSONDecodeError) as error:
                print(
                    "*** Cascade score summary disabled: failed to load "
                    f"solutions from {solution_path}: {error}"
                )
    selected_keys = select_task_keys(data.queries.keys())
    print(f"*** 4B -> LLaDA cascade selected {len(selected_keys)} task(s)")

    base_environment = os.environ.copy()
    base_environment.pop("ARC_RUN_STATUS_PATH", None)
    stage1_environment = apply_environment_profile(
        base_environment,
        "qwen3-4b-bf16",
        args.stage1_env,
    )
    starter_args, shared_architects_args = resolve_shared_adapter_staging(
        args,
        starter_args,
        stage1_environment,
    )
    stage1_environment.update(
        {
            "ARC_MODEL_PATH": os.path.abspath(os.path.expanduser(args.model_4b)),
            "ARC_INFERENCE_OUTPUT_DIR": os.fspath(output_4b),
            "ARC_TRAIN_INFERENCE_OUTPUT_DIR": os.fspath(
                output_4b / "train_predictions"
            ),
            "ARC_TASK_KEY": ",".join(selected_keys),
        }
    )
    stage1_environment.pop("ARC_TASK_INDEX", None)
    stage1_command = [sys.executable, "-u", "starter.py", *starter_args]
    if args.end_time > 0:
        stage1_command.extend(("--end-time", str(args.end_time)))
    if not args.reuse_4b_output:
        returncode = run_subprocess(
            stage1_command,
            environment=stage1_environment,
            cwd=script_dir,
            label="Stage 1/NVARC 4B training and inference",
        )
        if returncode:
            raise RuntimeError(f"NVARC stage failed with return code {returncode}")
    else:
        print(f"*** Reusing NVARC 4B candidates from {output_4b}")

    (
        primary_submission,
        decisions,
        primary_output_counts,
    ) = build_primary_submission_and_decisions(
        data,
        selected_keys,
        output_4b,
        selection_algorithm=args.selection_algorithm,
        kgmon_threshold=args.routing_kgmon_threshold,
        beam_threshold=args.routing_beam_threshold,
    )
    validate_submission_shape(primary_submission, data.queries, label="NVARC")
    atomic_write_json(primary_path, primary_submission)
    routed_keys = [decision.task_key for decision in decisions if decision.route]
    print(
        f"*** Routed {len(routed_keys)}/{len(selected_keys)} NVARC task(s) "
        "to ARChitects LLaDA"
    )
    print(f"*** Routed task keys: {routed_keys}")
    if routed_keys:
        validate_llada_artifact_paths(
            llada_path,
            llada_result_path,
            reuse=args.reuse_llada_submission,
        )

    manifest = routing_manifest_payload(
        decisions,
        selected_keys=selected_keys,
        output_4b=output_4b,
        llada_submission=llada_path,
        model_4b=args.model_4b,
        llada_base_model=args.architects_base_model,
        llada_size_model=args.architects_size_model,
        kgmon_threshold=args.routing_kgmon_threshold,
        beam_threshold=args.routing_beam_threshold,
        merge_policy=args.merge_policy,
    )
    manifest["stage1_status"] = "reused" if args.reuse_4b_output else "ok"
    manifest["nvarc_output_candidate_counts"] = primary_output_counts
    atomic_write_json(routing_path, manifest)

    llada_status = "not-routed"
    llada_submission = {}
    if routed_keys and not args.reuse_llada_submission:
        remaining = math.inf if args.end_time <= 0 else args.end_time - time.time()
        minimum_stage_budget = (
            ARCHITECTS_INTERNAL_RESERVE_SECONDS
            + args.finalization_reserve_seconds
        )
        if remaining <= minimum_stage_budget:
            llada_status = "skipped-deadline"
            print(
                "*** Skipping LLaDA because only "
                f"{remaining:.1f}s remain (need more than "
                f"{minimum_stage_budget:.1f}s including reserves); "
                "retaining NVARC attempts"
            )
        else:
            architects_environment = dict(base_environment)
            architects_environment.update(
                parse_environment_overrides(args.architects_env)
            )
            architects_data_path = (
                Path(args.architects_data_path).resolve()
                if args.architects_data_path
                else Path(challenge_path).resolve().parent
            )
            architects_environment.update(
                {
                    "ARCHITECTS_TASK_KEYS": ",".join(routed_keys),
                    "ARCHITECTS_BASE_MODEL_PATH": os.path.abspath(
                        os.path.expanduser(args.architects_base_model)
                    ),
                    "ARCHITECTS_SIZE_MODEL_PATH": os.path.abspath(
                        os.path.expanduser(args.architects_size_model)
                    ),
                    "ARCHITECTS_DATA_PATH": str(architects_data_path),
                    "ARCHITECTS_SPLIT": (
                        args.architects_split or infer_architects_split(challenge_path)
                    ),
                    "ARCHITECTS_MAX_INFER_TASKS": str(len(routed_keys)),
                    "ARCHITECTS_SUBMISSION_PATH": os.fspath(llada_path),
                    "ARCHITECTS_RESULT_MANIFEST_PATH": os.fspath(
                        llada_result_path
                    ),
                }
            )
            if args.end_time > 0:
                architects_environment["ARCHITECTS_END_TIME"] = str(args.end_time)
            architects_environment.pop("ARC_TASK_KEY", None)
            architects_environment.pop("ARC_TASK_INDEX", None)
            architects_environment.pop("ARCHITECTS_TRAIN_ONLY", None)
            llada_returncode = run_subprocess(
                [
                    sys.executable,
                    "-u",
                    os.fspath(architects_entrypoint),
                    *shared_architects_args,
                ],
                environment=architects_environment,
                cwd=architects_dir,
                label="Stage 2/ARChitects LLaDA routed TTT and inference",
            )
            if llada_returncode:
                llada_status = "failed-fallback-4b"
                if args.require_llada_success:
                    raise RuntimeError(
                        f"LLaDA stage failed with return code {llada_returncode}"
                    )
            elif not llada_path.is_file():
                llada_status = "missing-output-fallback-4b"
                if args.require_llada_success:
                    raise FileNotFoundError(
                        f"LLaDA did not write its submission: {llada_path}"
                    )
            else:
                try:
                    llada_submission = _load_json(llada_path)
                    validate_submission_shape(
                        llada_submission,
                        data.queries,
                        label="LLaDA",
                    )
                except Exception as error:
                    llada_status = "invalid-output-fallback-4b"
                    print(f"*** Invalid LLaDA output; retaining 4B: {error}")
                    if args.require_llada_success:
                        raise
                    llada_submission = {}
                else:
                    llada_status = "ok"
    elif routed_keys:
        llada_submission = _load_json(llada_path)
        validate_submission_shape(llada_submission, data.queries, label="LLaDA")
        llada_status = "reused"

    if llada_submission:
        output_counts = load_llada_output_counts(llada_result_path)
        final_submission, merge_audit = merge_nvarc_llada_submissions(
            primary_submission,
            llada_submission,
            decisions,
            data.queries,
            policy=args.merge_policy,
            primary_output_counts=primary_output_counts,
            llada_output_counts=output_counts,
        )
    else:
        final_submission = primary_submission
        merge_audit = []
    validate_submission_shape(final_submission, data.queries, label="final")
    atomic_write_json(final_path, final_submission)

    patterns = {}
    for record in merge_audit:
        pattern = "+".join(record.selected_sources)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    print(
        f"*** Final merge policy={args.merge_policy}; "
        f"merged_outputs={len(merge_audit)}; source_patterns={patterns}"
    )
    print(f"*** Wrote final submission to {final_path}")
    evaluation_scores = cascade_evaluation_scores(
        data,
        selected_keys,
        primary_submission,
        llada_submission,
        final_submission,
    )
    print_cascade_evaluation_scores(evaluation_scores)
    manifest["llada_status"] = llada_status
    manifest["merge_audit"] = [asdict(record) for record in merge_audit]
    manifest["final_submission"] = str(final_path)
    if evaluation_scores is not None:
        manifest["evaluation_scores"] = evaluation_scores
    manifest["updated_at"] = time.time()
    atomic_write_json(routing_path, manifest)


if __name__ == "__main__":
    run_entrypoint_with_status(
        main,
        path=extract_run_status_path(),
        phase="cascade_4b2llada",
        success_status=STATUS_SUCCESS,
        success_check=_final_submission_exists,
    )
