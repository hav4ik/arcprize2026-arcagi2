"""Run a confidence-routed 4B -> 8B ARC task-LoRA cascade.

The stages intentionally run as separate Python processes.  This prevents the
4B BF16 model and Qwen3.5 FP8 runtime patches from sharing imported modules,
CUDA state, or model-family environment flags.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

from arc_decoder import ArcDecoder, DEFAULT_SELECTION_ALGORITHM
from arc_loader import ArcDataset
from confidence_routing import (
    CASCADE_MERGE_POLICIES,
    route_tasks_by_confidence,
)
from run_config import (
    resolve_data_challenge_path,
    select_task_keys,
)
from run_status import (
    STATUS_SUCCESS,
    extract_run_status_path,
    run_entrypoint_with_status,
)


PASSTHROUGH_MARKERS = (
    "--common-starter-args",
    "--stage1-starter-args",
    "--stage2-starter-args",
)
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


STAGE_ENVIRONMENT_PROFILES = {
    "inherit": {},
    "qwen3-4b-bf16": {
        "ARC_FP8_FROZEN_LINEAR": "0",
        "ARC_FP8_DYNAMIC_BACKWARD": "0",
        "ARC_FP8_DYNAMIC_BACKWARD_SPLIT_ACCUMULATOR": "0",
        "ARC_CACHE_UP_PROJECTION_FP8": "0",
        "ARC_VLLM_QUANTIZATION": None,
        "UNSLOTH_QWEN35_FAST_LORA_MLP": None,
        "UNSLOTH_QWEN35_FP8_GDN": None,
        "UNSLOTH_QWEN35_MLP_REPLAY_BYPASS": None,
        "ARC_QWEN35_GDN_REPLAY_CACHE": None,
        "ARC_QWEN35_GDN_REPLAY_CACHE_PROJECTIONS": None,
        "ARC_QWEN35_GDN_REPLAY_CACHE_MAX_MIB": None,
    },
    "qwen35-9b-fp8": {
        # Frozen base projections are FP8; task-LoRA parameters remain BF16.
        "ARC_FP8_FROZEN_LINEAR": "1",
        "ARC_FP8_DYNAMIC_BACKWARD": "1",
        "ARC_FP8_DYNAMIC_BACKWARD_SPLIT_ACCUMULATOR": "0",
        "ARC_CACHE_UP_PROJECTION_FP8": "0",
        "ARC_VLLM_QUANTIZATION": "fp8",
        "UNSLOTH_QWEN35_FAST_LORA_MLP": "1",
        "UNSLOTH_QWEN35_FP8_GDN": "1",
        "UNSLOTH_QWEN35_MLP_REPLAY_BYPASS": "1",
        "UNSLOTH_LORA_MLP_CACHE_XA": "0",
        "ARC_QWEN35_GDN_REPLAY_CACHE": "1",
        "ARC_QWEN35_GDN_REPLAY_CACHE_PROJECTIONS": "qkv+z",
        "ARC_QWEN35_GDN_REPLAY_CACHE_MAX_MIB": "6000",
    },
}


def split_passthrough_args(argv):
    """Split cascade options from common and per-stage starter options."""
    segments = {
        "cascade": [],
        "common": [],
        "stage1": [],
        "stage2": [],
    }
    marker_to_segment = {
        marker: name
        for marker, name in zip(
            PASSTHROUGH_MARKERS,
            ("common", "stage1", "stage2"),
            strict=True,
        )
    }
    current = "cascade"
    seen = []
    for token in argv:
        if token in marker_to_segment:
            if token in seen:
                raise ValueError(f"Repeated passthrough marker: {token}")
            seen.append(token)
            current = marker_to_segment[token]
            continue
        segments[current].append(token)

    marker_positions = [
        PASSTHROUGH_MARKERS.index(marker) for marker in seen
    ]
    if marker_positions != sorted(marker_positions):
        raise ValueError(
            "Starter passthrough markers must appear in common, stage1, "
            "stage2 order"
        )
    return segments


def parse_environment_overrides(values):
    """Parse repeated KEY=VALUE stage overrides without shell evaluation."""
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Expected KEY=VALUE environment override, got {value!r}"
            )
        name, contents = value.split("=", 1)
        if not ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment name: {name!r}")
        overrides[name] = contents
    return overrides


def apply_environment_profile(environment, profile, overrides=()):
    """Return an isolated stage environment with explicit profile resets."""
    try:
        profile_values = STAGE_ENVIRONMENT_PROFILES[profile]
    except KeyError as error:
        supported = ", ".join(STAGE_ENVIRONMENT_PROFILES)
        raise ValueError(
            f"Unknown stage environment profile {profile!r}; "
            f"expected {supported}"
        ) from error
    result = dict(environment)
    for name, value in profile_values.items():
        if value is None:
            result.pop(name, None)
        else:
            result[name] = str(value)
    result.update(parse_environment_overrides(overrides))
    return result


def validate_starter_args(arguments, label):
    """Reject options whose ownership must stay with the cascade driver."""
    managed = ("--end-time", "--run-status-file")
    forbidden = ("--train-only",)
    for token in arguments:
        if token in forbidden:
            raise ValueError(
                f"{label} cannot use {token}; routing requires inference "
                "artifacts from both stages"
            )
        if any(token == option or token.startswith(option + "=") for option in managed):
            raise ValueError(
                f"{label} must not set {token}; the cascade driver owns "
                "the shared deadline and status file"
            )


def prepare_output_directory(path, *, reuse, label):
    """Create a fresh stage store or validate an explicitly reused store."""
    path = Path(path)
    if reuse:
        if not path.is_dir():
            raise FileNotFoundError(
                f"Cannot reuse missing {label} output directory: {path}"
            )
        return path
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to mix stale candidates into non-empty {label} "
            f"output directory {path}; choose a new --output-root or use "
            f"--reuse-{label}-output intentionally"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def json_safe_number(value):
    if value is None or math.isfinite(float(value)):
        return value
    return str(value)


def routing_manifest_payload(
    decisions,
    *,
    selected_keys,
    kgmon_threshold,
    beam_threshold,
    output_4b,
    output_8b,
    model_4b,
    model_8b,
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
        "models": {"4b": str(model_4b), "8b": str(model_8b)},
        "candidate_stores": {"4b": str(output_4b), "8b": str(output_8b)},
        "decisions": records,
    }


def compute_routing_decisions(
    data,
    selected_keys,
    output_dir,
    *,
    kgmon_threshold,
    beam_threshold,
):
    split_data = data.split_multi_replies()
    decoder = ArcDecoder(split_data, n_guesses=2)
    selected_base_keys = {
        f"{task_key}_{test_index}"
        for task_key in selected_keys
        for test_index in range(len(data.queries[task_key]["test"]))
    }
    decoder.load_decoded_results(
        os.fspath(output_dir),
        allowed_base_keys=selected_base_keys,
    )
    return route_tasks_by_confidence(
        data.queries,
        decoder.decoded_results,
        selected_keys,
        kgmon_threshold=kgmon_threshold,
        beam_threshold=beam_threshold,
    )


def run_subprocess(command, *, environment, cwd, label):
    print(f"*** {label}", flush=True)
    print(" ", shlex.join(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, env=environment, cwd=cwd, check=False)
    elapsed = time.perf_counter() - started
    print(
        f"*** {label} finished: returncode={result.returncode}, "
        f"wall={elapsed:.1f}s",
        flush=True,
    )
    return result.returncode


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run 4B task LoRAs, confidence-route uncertain tasks to an "
            "independent 8B stage, then merge a final submission. Cascade "
            "options must precede the three starter-argument markers."
        ),
    )
    parser.add_argument("--model-4b", required=True)
    parser.add_argument("--model-8b", required=True)
    parser.add_argument(
        "--output-root",
        default="/kaggle/working/arc_4b28b_cascade",
    )
    parser.add_argument("--output-4b")
    parser.add_argument("--output-8b")
    parser.add_argument("--routing-manifest")
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument(
        "--finalization-reserve-seconds",
        type=float,
        default=120.0,
    )
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
        choices=CASCADE_MERGE_POLICIES,
        default=os.getenv("ARC_CASCADE_MERGE_POLICY", "one-each"),
    )
    parser.add_argument(
        "--selection-algorithm",
        default=os.getenv(
            "ARC_SELECTION_ALGORITHM",
            DEFAULT_SELECTION_ALGORITHM,
        ),
    )
    parser.add_argument(
        "--stage1-profile",
        choices=tuple(STAGE_ENVIRONMENT_PROFILES),
        default="qwen3-4b-bf16",
    )
    parser.add_argument(
        "--stage2-profile",
        choices=tuple(STAGE_ENVIRONMENT_PROFILES),
        default="qwen35-9b-fp8",
    )
    parser.add_argument(
        "--stage1-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--stage2-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--reuse-4b-output", action="store_true")
    parser.add_argument("--reuse-8b-output", action="store_true")
    parser.add_argument(
        "--require-8b-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail the whole cascade when stage 2 exits nonzero. By default "
            "the final merge falls back to available 4B candidates."
        ),
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--run-status-file")
    return parser


def main():
    segments = split_passthrough_args(sys.argv[1:])
    args = build_parser().parse_args(segments["cascade"])
    common_starter_args = segments["common"]
    stage1_starter_args = segments["stage1"]
    stage2_starter_args = segments["stage2"]
    validate_starter_args(
        common_starter_args + stage1_starter_args,
        "stage 1 starter arguments",
    )
    validate_starter_args(
        common_starter_args + stage2_starter_args,
        "stage 2 starter arguments",
    )
    if args.finalization_reserve_seconds < 0:
        raise ValueError("--finalization-reserve-seconds cannot be negative")

    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_4b = Path(args.output_4b or output_root / "4b").resolve()
    output_8b = Path(args.output_8b or output_root / "8b").resolve()
    manifest_path = Path(
        args.routing_manifest or output_root / "routing_manifest.json"
    ).resolve()
    output_4b = prepare_output_directory(
        output_4b,
        reuse=args.reuse_4b_output,
        label="4b",
    )
    output_8b = prepare_output_directory(
        output_8b,
        reuse=args.reuse_8b_output,
        label="8b",
    )

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    data = ArcDataset.from_file(resolve_data_challenge_path(rerun_mode))
    selected_keys = select_task_keys(data.queries.keys())
    print(f"*** Cascade selected {len(selected_keys)} task(s)")

    base_environment = os.environ.copy()
    base_environment.pop("ARC_RUN_STATUS_PATH", None)
    stage1_environment = apply_environment_profile(
        base_environment,
        args.stage1_profile,
        args.stage1_env,
    )
    stage1_environment.update(
        {
            "ARC_MODEL_PATH": os.path.abspath(
                os.path.expanduser(args.model_4b)
            ),
            "ARC_INFERENCE_OUTPUT_DIR": os.fspath(output_4b),
            "ARC_TRAIN_INFERENCE_OUTPUT_DIR": os.fspath(
                output_4b / "train_predictions"
            ),
            "ARC_TASK_KEY": ",".join(selected_keys),
        }
    )
    stage1_environment.pop("ARC_TASK_INDEX", None)

    stage1_command = [
        sys.executable,
        "-u",
        "starter.py",
        *common_starter_args,
        *stage1_starter_args,
    ]
    if args.end_time > 0:
        stage1_command.extend(("--end-time", str(args.end_time)))
    if not args.reuse_4b_output:
        returncode = run_subprocess(
            stage1_command,
            environment=stage1_environment,
            cwd=script_dir,
            label="Stage 1/4B training and inference",
        )
        if returncode:
            raise subprocess.CalledProcessError(returncode, stage1_command)
    else:
        print(f"*** Reusing 4B candidates from {output_4b}")

    decisions = compute_routing_decisions(
        data,
        selected_keys,
        output_4b,
        kgmon_threshold=args.routing_kgmon_threshold,
        beam_threshold=args.routing_beam_threshold,
    )
    routed_keys = [
        decision.task_key for decision in decisions if decision.route
    ]
    reason_counts = {
        reason: sum(reason in decision.reasons for decision in decisions)
        for reason in ("missing-output", "low-kgmon", "high-beam")
    }
    print(
        f"*** Routed {len(routed_keys)}/{len(selected_keys)} task(s) to 8B; "
        f"missing={reason_counts['missing-output']}, "
        f"low_kgmon={reason_counts['low-kgmon']}, "
        f"high_beam={reason_counts['high-beam']} "
        "(reason counts overlap)"
    )
    print(f"*** Routed task keys: {routed_keys}")
    manifest = routing_manifest_payload(
        decisions,
        selected_keys=selected_keys,
        kgmon_threshold=args.routing_kgmon_threshold,
        beam_threshold=args.routing_beam_threshold,
        output_4b=output_4b,
        output_8b=output_8b,
        model_4b=args.model_4b,
        model_8b=args.model_8b,
        merge_policy=args.merge_policy,
    )
    manifest["stage1_status"] = "reused" if args.reuse_4b_output else "ok"
    atomic_write_json(manifest_path, manifest)
    print(f"*** Wrote routing manifest to {manifest_path}")

    stage2_status = "not-routed"
    if routed_keys and not args.reuse_8b_output:
        remaining = (
            math.inf
            if args.end_time <= 0
            else args.end_time - time.time()
        )
        if remaining <= args.finalization_reserve_seconds:
            stage2_status = "skipped-deadline"
            print(
                "*** Skipping 8B stage because only "
                f"{remaining:.1f}s remain before the shared deadline; "
                "the final merge will retain 4B candidates"
            )
        else:
            stage2_environment = apply_environment_profile(
                base_environment,
                args.stage2_profile,
                args.stage2_env,
            )
            stage2_environment.update(
                {
                    "ARC_MODEL_PATH": os.path.abspath(
                        os.path.expanduser(args.model_8b)
                    ),
                    "ARC_INFERENCE_OUTPUT_DIR": os.fspath(output_8b),
                    "ARC_TRAIN_INFERENCE_OUTPUT_DIR": os.fspath(
                        output_8b / "train_predictions"
                    ),
                    "ARC_TASK_KEY": ",".join(routed_keys),
                }
            )
            stage2_environment.pop("ARC_TASK_INDEX", None)
            stage2_command = [
                sys.executable,
                "-u",
                "starter.py",
                *common_starter_args,
                *stage2_starter_args,
            ]
            if args.end_time > 0:
                stage2_command.extend(("--end-time", str(args.end_time)))
            stage2_returncode = run_subprocess(
                stage2_command,
                environment=stage2_environment,
                cwd=script_dir,
                label="Stage 2/8B routed training and inference",
            )
            stage2_status = (
                "ok" if stage2_returncode == 0 else "failed-fallback-4b"
            )
            if stage2_returncode and args.require_8b_success:
                raise subprocess.CalledProcessError(
                    stage2_returncode,
                    stage2_command,
                )
            if stage2_returncode:
                print(
                    "*** 8B stage failed; merging any completed 8B "
                    "artifacts and falling back to 4B elsewhere"
                )
    elif routed_keys:
        stage2_status = "reused"
        print(f"*** Reusing 8B candidates from {output_8b}")

    manifest["stage2_status"] = stage2_status
    manifest["updated_at"] = time.time()
    atomic_write_json(manifest_path, manifest)

    submit_environment = dict(base_environment)
    submit_environment.update(
        {
            "ARC_INFERENCE_OUTPUT_DIR": os.fspath(output_4b),
            "ARC_8B_INFERENCE_OUTPUT_DIR": os.fspath(output_8b),
            "ARC_TASK_KEY": ",".join(selected_keys),
            "ARC_SELECTION_ALGORITHM": args.selection_algorithm,
        }
    )
    submit_environment.pop("ARC_TASK_INDEX", None)
    submit_environment.pop("ARC_RUN_STATUS_PATH", None)
    submit_command = [
        sys.executable,
        "-u",
        "submit.py",
        "--cascade-8b-inference-output-dir",
        os.fspath(output_8b),
        "--cascade-merge-policy",
        args.merge_policy,
        "--routing-kgmon-threshold",
        str(args.routing_kgmon_threshold),
        "--routing-beam-threshold",
        str(args.routing_beam_threshold),
        "--selection-algorithm",
        args.selection_algorithm,
    ]
    if args.visualize:
        submit_command.append("--visualize")
    submit_returncode = run_subprocess(
        submit_command,
        environment=submit_environment,
        cwd=script_dir,
        label="Final 4B/8B merged submission",
    )
    if submit_returncode:
        raise subprocess.CalledProcessError(
            submit_returncode,
            submit_command,
        )


if __name__ == "__main__":
    run_entrypoint_with_status(
        main,
        path=extract_run_status_path(),
        phase="cascade_4b28b",
        success_status=STATUS_SUCCESS,
        success_check=lambda: (Path(__file__).resolve().parent / "submission.json").is_file(),
    )
