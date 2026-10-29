"""NVARC Qwen3-4B TTT -> confidence-routed Qwen3.8 (27B, Flash-Next on vLLM or llama.cpp), DeepSeek-V4 or dots3-note
cascade."""

from __future__ import annotations

import argparse
import asyncio
import codecs
from copy import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import qwen38_pipeline as qwen38
from early_routing import add_early_routing_arguments, early_routing_starter_args, read_early_routing_manifest
from run_config import resolve_challenge_path, select_task_keys
from run_status import STATUS_SUCCESS, extract_run_status_path, run_entrypoint_with_status


FINAL_PATH = None
MARKER = "--stage1-starter-args"
# Same 4B inference/search settings as the user's current 4B->LLaDA command.
# Training environment options remain caller-controlled, as with that cascade.
DEFAULT_STARTER_ARGS = """
--inference-backend vllm --multi-task-gpus 2 --vllm-gpus 2 --vllm-max-loras 3
--vllm-auto-recover --vllm-max-adapter-recoveries 3 --no-vllm-enforce-eager
--vllm-leaf-rollout --no-vllm-batch-invariant --vllm-kv-resident-fraction 0.6
--vllm-async-scheduling --kv-offloading-size 0 --vllm-rollout-max-tokens 32
--vllm-prefix-score --inference-batch-size 24 --vllm-max-num-seqs 24
--no-vllm-cumulative-nll-stop --vllm-rollout-branch-window 1 --lora-r 128
--gradient-checkpointing-mode unsloth --max-seq-length 16384
--inference-timeout-seconds 3000 --vllm-search-limit-seconds 1000
--rescore-views 8 --rescore-schedule prefix --vllm-prefix-score-tile-views 4
--score-batch-size 16 --dfs-mode auto
""".split()


def split_args(argv):
    if argv.count(MARKER) > 1:
        raise ValueError(f"Repeated {MARKER}")
    if MARKER not in argv:
        return list(argv), list(DEFAULT_STARTER_ARGS)
    index = argv.index(MARKER)
    return list(argv[:index]), list(argv[index+1:])


def build_parser():
    p = qwen38.ProfileParser(description=__doc__)
    qwen38.add_qwen38_arguments(p)
    add_early_routing_arguments(p)
    p.add_argument("--model-4b", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-4b", type=Path)
    p.add_argument("--reuse-4b-output", action="store_true")
    p.add_argument("--challenges", type=Path)
    p.add_argument("--solutions", type=Path)
    p.add_argument("--final-submission", type=Path, default=Path("/kaggle/working/submission.json"))
    p.add_argument("--end-time", type=float, default=0)
    p.add_argument("--finalization-reserve-seconds", type=float, default=120)
    p.add_argument("--stage1-max-seconds", type=float, default=21600)
    p.add_argument("--routing-kgmon-threshold", type=float, default=-10)
    p.add_argument("--routing-beam-threshold", type=float, default=1.0)
    p.add_argument("--merge-policy", choices=("one-each", "prefer-qwen38", "prefer-4b"), default="one-each")
    p.add_argument("--selection-algorithm", default=os.getenv("ARC_SELECTION_ALGORITHM", "kgmon"))
    p.add_argument("--stage1-env", action="append", default=[])
    p.add_argument("--inference-adapter-storage", choices=("auto", "ram", "disk"), default=None)
    p.add_argument("--inference-adapter-root")
    p.add_argument("--inference-ram-adapter-root")
    p.add_argument("--inference-disk-adapter-root")
    p.add_argument("--vllm-max-pending-adapters", type=int)
    p.add_argument("--vllm-scale-after-training", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--require-stage1-success", action="store_true")
    p.add_argument("--run-status-file")
    return p


def run_stage1(command, environment, cwd, log_path, deadline):
    """Mirror the durable child log live without blocking the parent deadline."""
    # The child writes directly to disk so its complete log survives even if the
    # notebook disconnects. A separate binary reader also forwards partial lines
    # (e.g. tqdm's carriage-return updates), unlike a blocking readline() pipe.
    with log_path.open("xb") as log, log_path.open("rb") as live_log:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def emit(chunk, *, final=False):
            text = decoder.decode(chunk, final=final)
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()

        proc = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status = "ok"
        try:
            qwen38.atomic_json(log_path.with_suffix(".launch.json"), dict(
                command=command, pid=proc.pid, deadline_epoch=deadline,
                environment={k:v for k,v in environment.items()
                             if k.startswith(("ARC_", "UNSLOTH_", "VLLM_"))
                             and not any(secret in k.upper() for secret in
                                         ("API_KEY", "PASSWORD", "SECRET", "ACCESS_TOKEN", "AUTH_TOKEN"))}))
            print(f"*** Stage 1 PID={proc.pid}; log={log_path}; deadline={deadline}", flush=True)
            while proc.poll() is None:
                # Bound each read so a noisy child cannot starve deadline checks.
                emit(live_log.read(65536))
                if time.time() >= deadline:
                    status = "deadline"
                    break
                time.sleep(min(0.1, max(0, deadline-time.time())))
        finally:
            try:
                qwen38.stop_owned_process(proc)
            finally:
                # Include the final traceback / shutdown output, even when the
                # child exits between polls or the parent terminates it.
                while chunk := live_log.read(65536):
                    emit(chunk)
                emit(b"", final=True)
        if status == "ok" and proc.returncode != 0:
            status = "failed"
        print(f"\n*** Stage 1 finished: status={status}; returncode={proc.returncode}; log={log_path}",
              flush=True)
        return dict(status=status, returncode=proc.returncode, pid=proc.pid)


def build_routing_manifest(decisions, counts, selected, primary_path, output_4b, args):
    def finite_or_string(value):
        return value if value is None or math.isfinite(value) else str(value)
    records = []
    for decision in decisions:
        record = asdict(decision)
        record["min_kgmon"] = finite_or_string(record["min_kgmon"])
        record["max_beam_score"] = finite_or_string(record["max_beam_score"])
        records.append(record)
    return dict(schema_version=1, selected_task_keys=selected,
                routed_task_keys=[d.task_key for d in decisions if d.route],
                decisions=records, nvarc_output_candidate_counts=counts,
                primary_submission=str(primary_path), output_4b=str(output_4b),
                models={"4b": args.model_4b, **({"dsv4": str(args.dsv4_model)} if args.qwen38_variant == "dsv4" else
                                                {"flash-llama": str(args.qf_model)} if args.qwen38_variant == "flash-llama"
                                                else {"dots3": str(args.d3_model)} if args.qwen38_variant == "dots3"
                                                else {"qwen38": args.model_qwen38})},
                merge_policy=args.merge_policy,
                flash_next_profile=getattr(args, "flash_next_profile", "manual"),
                routing_rule=dict(missing_output=True, kgmon_threshold=args.routing_kgmon_threshold,
                                  beam_threshold=args.routing_beam_threshold))


def run(args, starter_args):
    global FINAL_PATH
    # Reuse the exact existing ranking/routing and adapter-option resolution.
    # These imports stay out of the standalone language-model entrypoint.
    from arc_loader import ArcDataset
    from cascade_4b28b import apply_environment_profile, prepare_output_directory, validate_starter_args
    from cascade_4b2llada import build_primary_submission_and_decisions, resolve_shared_adapter_staging

    qwen38.validate_options(args)
    starter_args = early_routing_starter_args(args, starter_args)
    if args.reuse_4b_output and any(arg.startswith("--early-route-") for arg in starter_args):
        raise ValueError("Cannot run new early-routing probes with --reuse-4b-output; reuse honors its saved manifest")
    validate_starter_args(starter_args, "stage 1 starter arguments")
    if not math.isfinite(args.stage1_max_seconds) or args.stage1_max_seconds <= 0:
        raise ValueError("--stage1-max-seconds must be positive and finite")
    if not math.isfinite(args.routing_kgmon_threshold) or not math.isfinite(args.routing_beam_threshold) or args.routing_beam_threshold < 0:
        raise ValueError("Invalid routing thresholds")
    args.challenges = args.challenges or Path(resolve_challenge_path(os.getenv("KAGGLE_IS_COMPETITION_RERUN")))
    # Do not attach labels to the decoder. Stage-2 evaluation is post-hoc only.
    data = ArcDataset.from_file(args.challenges)
    selected = select_task_keys(data.queries)
    args.output_root.mkdir(parents=True, exist_ok=False)
    FINAL_PATH = args.final_submission
    qwen38.atomic_json(args.final_submission, qwen38.empty_submission(data.queries))
    output_4b = prepare_output_directory(args.output_4b or args.output_root / "4b",
                                        reuse=args.reuse_4b_output, label="4b")
    environment = apply_environment_profile(os.environ, "qwen3-4b-bf16", args.stage1_env)
    environment.pop("ARC_RUN_STATUS_PATH", None)
    environment.pop("ARC_TASK_INDEX", None)
    environment.setdefault("ARC_REUSE_BASE_MODEL", "1")
    environment.update(ARC_MODEL_PATH=str(Path(args.model_4b).expanduser().resolve()),
                       ARC_INFERENCE_OUTPUT_DIR=str(output_4b),
                       ARC_TRAIN_INFERENCE_OUTPUT_DIR=str(output_4b / "train_predictions"),
                       ARC_CHALLENGE_PATH=str(args.challenges.resolve()), ARC_TASK_KEY=",".join(selected))
    starter_args, _ = resolve_shared_adapter_staging(args, starter_args, environment)
    stage1_deadline = qwen38.deadline_from_limits(args.end_time, args.stage1_max_seconds,
                                                args.finalization_reserve_seconds)
    command = [sys.executable, "-u", "starter.py", *starter_args, "--end-time", str(stage1_deadline)]
    if args.reuse_4b_output:
        stage1 = dict(status="reused", returncode=0)
    elif time.time() >= stage1_deadline or not selected:
        stage1 = dict(status="skipped-deadline" if selected else "not-selected", returncode=0)
    else:
        stage1 = run_stage1(command, environment, Path(__file__).resolve().parent,
                           args.output_root / "stage1.log", stage1_deadline)
    qwen38.atomic_json(args.output_root / "stage1_status.json", stage1)
    candidate_read_errors = []
    primary, decisions, counts = build_primary_submission_and_decisions(
        data, selected, output_4b, selection_algorithm=args.selection_algorithm,
        kgmon_threshold=args.routing_kgmon_threshold, beam_threshold=args.routing_beam_threshold,
        candidate_read_errors=candidate_read_errors)
    qwen38.validate_submission(primary, data.queries)
    primary_path = args.output_root / "nvarc_submission.json"
    qwen38.atomic_json(primary_path, primary)
    qwen38.atomic_json(args.final_submission, primary)
    manifest = build_routing_manifest(decisions, counts, selected, primary_path, output_4b, args)
    early_manifest = read_early_routing_manifest(output_4b, data.queries, selected)
    if early_manifest is not None:
        manifest["early_routing"] = early_manifest
        early_keys = set(early_manifest["skipped_task_keys"])
        for record in manifest["decisions"]:
            record["early_routed"] = record["task_key"] in early_keys
            if record["early_routed"]:
                record["reasons"] = ["early-base-score", *record["reasons"]]
    manifest["stage1"] = stage1
    manifest["candidate_read_errors"] = candidate_read_errors
    routing_path = args.output_root / "routing_manifest.json"
    qwen38.atomic_json(routing_path, manifest)
    print(f"*** Routed {len(manifest['routed_task_keys'])}/{len(selected)} tasks to Qwen3.8", flush=True)
    if stage1["status"] == "failed" and args.require_stage1_success:
        raise RuntimeError(f"Stage 1 failed: {stage1}; primary submission was preserved")

    secondary = copy(args)
    secondary.output_root = args.output_root / "qwen38"
    secondary.primary_submission = primary_path
    secondary.routing_manifest = routing_path
    secondary.task_keys = ",".join(selected)
    summary = asyncio.run(qwen38.run_inference(secondary))
    manifest["qwen38_status"] = summary["status"]
    manifest["final_submission"] = str(args.final_submission)
    manifest["scores"] = {k:summary[k] for k in ("primary_score", "qwen38_raw_score", "merged_score")}
    qwen38.atomic_json(routing_path, manifest)
    return summary


def main():
    argv, starter_args = split_args(sys.argv[1:])
    args = build_parser().parse_args(argv)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    run(args, starter_args)


if __name__ == "__main__":
    run_entrypoint_with_status(main, path=extract_run_status_path(), phase="cascade_4b2qwen38",
                             success_status=STATUS_SUCCESS,
                             success_check=lambda: FINAL_PATH is not None and FINAL_PATH.is_file())
