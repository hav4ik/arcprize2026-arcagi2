"""Replay preserved ARC LoRA adapters through one vLLM inference service.

This is an experiment utility, not the competition entry point.  It stages
hard links to already-trained adapter artifacts so multiple vLLM cache and
scheduler configurations can be compared with identical model weights.
"""

from __future__ import annotations

from pipeline_runtime_launcher import add_replay_runtime_arguments, maybe_reexec_replay

if __name__ == "__main__":
    maybe_reexec_replay(__file__)

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import time
import uuid
from pathlib import Path

from arc_vllm_adapter import ADAPTER_MANIFEST_FILE
from arc_vllm_dflash import (
    add_dflash_arguments,
    build_dflash_speculative_config,
    dflash_options,
)
from arc_vllm_infer import run_vllm_inference_worker
from run_config import (
    VLLM_KV_CACHE_DTYPES,
    normalize_lora_init,
    normalize_pissa_base_mode,
    normalize_vllm_kv_cache_dtype,
    normalize_vllm_kv_resident_fraction,
    normalize_vllm_search_limit_seconds,
)


def positive_int(value: object) -> int:
    """Parse an unrestricted positive count for replay tuning options."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def stage_adapter_jobs(
    source_root: Path,
    staging_root: Path,
    task_keys: set[str] | None = None,
) -> list[dict[str, object]]:
    """Hard-link complete adapter directories into a disposable run root."""
    source_root = source_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, object]] = []
    for source in sorted(source_root.iterdir()):
        manifest_path = source / ADAPTER_MANIFEST_FILE
        if not source.is_dir() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if task_keys is not None and str(manifest["task_key"]) not in task_keys:
            continue
        destination = staging_root / source.name
        shutil.copytree(source, destination, copy_function=os.link)
        jobs.append(
            {
                "kind": "adapter",
                "task_key": str(manifest["task_key"]),
                "adapter_id": int(manifest["adapter_id"]),
                "adapter_name": str(manifest["adapter_name"]),
                "adapter_path": str(destination),
                "max_train_tokens": int(manifest["max_train_tokens"]),
                "published_at": time.time(),
                "trainer_rank": int(manifest.get("trainer_rank", 0)),
                "export_seconds": 0.0,
                "tensor_bytes": int(manifest["tensor_bytes"]),
                "pending_capacity_wait_seconds": 0.0,
                "service_index": 0,
                "service_loads_after_reservation": [len(jobs) + 1],
            }
        )
    if not jobs:
        staging_root.rmdir()
        raise ValueError(f"No complete ARC adapters found under {source_root}")
    return jobs


def build_options(args: argparse.Namespace, adapter_count: int) -> dict[str, object]:
    """Build the same inference policy used by the optimized starter path."""
    return {
        **dflash_options(args),
        "inference_timeout_seconds": args.inference_timeout_seconds,
        "vllm_search_limit_seconds": args.vllm_search_limit_seconds,
        "inference_batch_size": args.inference_batch_size,
        "score_batch_size": args.score_batch_size,
        "rescore_views": args.rescore_views,
        "rescore_schedule": args.rescore_schedule,
        "vllm_prefix_score": args.vllm_prefix_score,
        "vllm_prefix_score_tile_views": args.prefix_score_tile_views,
        "dfs_mode": args.dfs_mode,
        "frontier_origin_batch_size": args.frontier_origin_batch_size,
        "frontier_max_live_branches": args.frontier_max_live_branches,
        "frontier_max_prompt_tokens": args.frontier_max_prompt_tokens,
        "max_beam_score": args.max_beam_score,
        "zero_candidate_shape_fallback": False,
        "shape_fallback_beam_width": 2,
        "predict_train": False,
        "base_model_fallback": args.base_model_fallback,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "pissa_base_mode": getattr(args, "pissa_base_mode", "original"),
        "lora_init": getattr(args, "lora_init", "default"),
        "lora_method": getattr(args, "lora_method", "rslora"),
        "lora_alpha": getattr(args, "lora_alpha", 32.0),
        "vllm_max_loras": args.max_loras or adapter_count,
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_quantization": args.vllm_quantization,
        "vllm_kv_cache_dtype": args.vllm_kv_cache_dtype,
        "vllm_kv_offloading_size": args.kv_offloading_size,
        "vllm_kv_offloading_backend": "native",
        "vllm_enforce_eager": args.vllm_enforce_eager,
        "vllm_async_scheduling": args.vllm_async_scheduling,
        "vllm_batch_invariant": args.vllm_batch_invariant,
        "vllm_targeted_logprobs_only": args.vllm_targeted_logprobs_only,
        "vllm_speculative_ngram_tokens": args.vllm_speculative_ngram_tokens,
        "vllm_speculative_ngram_method": args.vllm_speculative_ngram_method,
        "vllm_speculative_ngram_min": args.vllm_speculative_ngram_min,
        "vllm_speculative_ngram_max": args.vllm_speculative_ngram_max,
        "vllm_max_num_seqs": args.max_num_seqs,
        "vllm_kv_resident_batching": args.vllm_kv_resident_batching,
        "vllm_kv_resident_fraction": args.vllm_kv_resident_fraction,
        "vllm_max_num_batched_tokens": args.max_num_batched_tokens,
        "vllm_leaf_rollout": args.vllm_leaf_rollout,
        "vllm_rollout_max_tokens": args.vllm_rollout_max_tokens,
        "vllm_rollout_branch_window": args.vllm_rollout_branch_window,
        "vllm_cumulative_nll_stop": args.vllm_cumulative_nll_stop,
        "vllm_telemetry_dir": (
            None if args.telemetry_dir is None else str(args.telemetry_dir)
        ),
        "rl_rollout_backend": "native",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_runtime_arguments(parser)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument(
        "--task-key",
        action="append",
        help="Replay only this task key; repeat to select multiple tasks",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staging-parent", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path)
    parser.add_argument(
        "--base-model-fallback", action=argparse.BooleanOptionalAction, default=False,
        help="Retry all-black submitted outputs using the base model without LoRA",
    )
    parser.add_argument("--kv-offloading-size", type=float, default=0.0)
    parser.add_argument(
        "--max-loras",
        type=positive_int,
        help="Preallocated vLLM LoRA slots (defaults to selected adapters)",
    )
    parser.add_argument("--max-num-seqs", type=positive_int, default=16)
    parser.add_argument(
        "--vllm-kv-resident-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vllm-kv-resident-fraction",
        type=normalize_vllm_kv_resident_fraction,
        default=0.95,
        metavar="FRACTION|auto",
        help=(
            "Per-task decode-root fraction (default: 0.95). Auto uses 0.95 "
            "for one --max-loras slot, otherwise 0.90 / slots. Without "
            "--max-loras, use the selected adapter count, not active occupancy."
        ),
    )
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=16384)
    parser.add_argument("--max-seq-length", type=positive_int, default=16384)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--pissa-base-mode", type=normalize_pissa_base_mode, default="original")
    parser.add_argument("--lora-init", type=normalize_lora_init, default="default")
    parser.add_argument("--lora-method", choices=("lora", "rslora"), default="rslora")
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.89)
    parser.add_argument(
        "--vllm-quantization",
        choices=("none", "fp8"),
        default="fp8",
    )
    parser.add_argument(
        "--vllm-kv-cache-dtype",
        type=normalize_vllm_kv_cache_dtype,
        choices=VLLM_KV_CACHE_DTYPES,
        default=os.getenv("ARC_VLLM_KV_CACHE_DTYPE", "auto"),
        help=(
            "Main-model KV precision, independent of base/LoRA weights "
            "(default: auto; environment: ARC_VLLM_KV_CACHE_DTYPE)."
        ),
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vllm-batch-invariant",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vllm-targeted-logprobs-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Request only the explicit ARC token logprobs, avoiding vLLM's "
            "otherwise redundant full-vocabulary top-k (default: disabled)."
        ),
    )
    add_dflash_arguments(parser)
    parser.add_argument("--vllm-speculative-ngram-tokens", type=int, default=0)
    parser.add_argument(
        "--vllm-speculative-ngram-method",
        choices=("ngram", "ngram_gpu"),
        default="ngram_gpu",
    )
    parser.add_argument("--vllm-speculative-ngram-min", type=int, default=2)
    parser.add_argument("--vllm-speculative-ngram-max", type=int, default=5)
    parser.add_argument(
        "--vllm-leaf-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vllm-rollout-max-tokens", type=positive_int, default=32)
    parser.add_argument("--vllm-rollout-branch-window", type=positive_int, default=1)
    parser.add_argument(
        "--vllm-cumulative-nll-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--inference-batch-size", type=positive_int, default=4)
    parser.add_argument("--score-batch-size", type=positive_int, default=16)
    parser.add_argument("--rescore-views", type=positive_int, default=8)
    parser.add_argument(
        "--rescore-schedule",
        choices=("candidate", "view", "prefix"),
        default="prefix",
    )
    parser.add_argument(
        "--vllm-prefix-score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prefix-score-tile-views",
        type=positive_int,
        default=4,
    )
    parser.add_argument(
        "--dfs-mode",
        choices=("auto", "recursive", "frontier"),
        default="auto",
    )
    parser.add_argument("--frontier-origin-batch-size", type=positive_int, default=4)
    parser.add_argument("--frontier-max-live-branches", type=positive_int, default=8)
    parser.add_argument("--frontier-max-prompt-tokens", type=positive_int, default=3000)
    parser.add_argument("--max-beam-score", type=float, default=math.log(5.0))
    parser.add_argument("--inference-timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--vllm-search-limit-seconds",
        type=normalize_vllm_search_limit_seconds,
        default=os.getenv("ARC_VLLM_SEARCH_LIMIT_SECONDS", "540"),
        metavar="SECONDS",
        help=(
            "Positive wall-time limit per threshold-DFS augmentation group, "
            "clipped by the task and global deadlines (default: 540; "
            "environment: ARC_VLLM_SEARCH_LIMIT_SECONDS)."
        ),
    )
    parser.add_argument("--end-time-seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dflash_speculative_config(args.model_path, vars(args))
    from arc_pissa_shared import validate_runtime_base

    validate_runtime_base(
        args.model_path, mode=args.pissa_base_mode, initialization=args.lora_init,
        method=args.lora_method, rank=args.lora_r, alpha=args.lora_alpha,
        verify_files=True,
    )
    if args.kv_offloading_size < 0:
        raise ValueError("--kv-offloading-size must be non-negative")
    if args.max_num_seqs < 1 or args.max_num_batched_tokens < 1:
        raise ValueError("vLLM scheduler limits must be positive")
    if args.max_seq_length < 1 or args.lora_r < 1:
        raise ValueError("Model context and LoRA rank must be positive")
    if args.vllm_rollout_max_tokens < 1:
        raise ValueError("--vllm-rollout-max-tokens must be positive")
    if args.vllm_rollout_branch_window < 1:
        raise ValueError("--vllm-rollout-branch-window must be positive")
    if args.vllm_cumulative_nll_stop and not args.vllm_leaf_rollout:
        raise ValueError("--vllm-cumulative-nll-stop requires leaf rollout")
    if (
        args.vllm_cumulative_nll_stop
        and args.vllm_speculative_ngram_tokens > 0
    ):
        raise ValueError(
            "Cumulative-NLL stop is incompatible with speculative decoding"
        )
    if args.vllm_speculative_ngram_tokens < 0:
        raise ValueError("--vllm-speculative-ngram-tokens must be non-negative")
    if (
        args.vllm_speculative_ngram_min < 1
        or args.vllm_speculative_ngram_max < args.vllm_speculative_ngram_min
    ):
        raise ValueError("Invalid speculative n-gram lookup range")
    if args.max_loras is not None and args.max_loras < 1:
        raise ValueError("--max-loras must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.telemetry_dir is not None:
        args.telemetry_dir.mkdir(parents=True, exist_ok=False)
    args.staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = args.staging_parent / f"replay-{uuid.uuid4().hex[:12]}"
    jobs = stage_adapter_jobs(
        args.adapter_root,
        staging_root,
        set(args.task_key) if args.task_key else None,
    )
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    pending_slots = context.BoundedSemaphore(len(jobs))
    outstanding = context.Array("i", [len(jobs)], lock=False)
    service_lock = context.Lock()
    startup_lock = context.Lock()
    abort_event = context.Event()
    for job in jobs:
        pending_slots.acquire()
        job_queue.put(job)

    os.environ["ARC_INFERENCE_OUTPUT_DIR"] = str(args.output_dir)
    try:
        run_vllm_inference_worker(
            job_queue=job_queue,
            rollout_response_queues=[],
            pending_adapter_slots=pending_slots,
            service_index=0,
            service_outstanding=outstanding,
            service_load_lock=service_lock,
            startup_lock=startup_lock,
            trainer_count=0,
            end_time=time.time() + args.end_time_seconds,
            model_path=args.model_path,
            adapter_root=str(staging_root),
            options=build_options(args, len(jobs)),
            abort_event=abort_event,
        )
    finally:
        job_queue.close()
        job_queue.join_thread()
        if staging_root.exists():
            # The worker removes admitted adapters.  Only clean an incomplete
            # staging root, never the preserved source artifacts.
            shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()
