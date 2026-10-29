# !UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}

import os
import sys

from pipeline_runtime import (
    add_runtime_arguments,
    runtime_specs_from_args,
    split_runtime_requested,
    validate_split_mode,
)

from runtime_bootstrap import (
    bootstrap_transformer_engine_cpp_runtime,
    patch_vllm_compact_vocab_sampler_warmup,
    patch_vllm_diffusion_gemma_cudagraph_causal_buffer,
)


# Mixed-mode coordinators must not patch their global serving installation
# while launching a different one. Role children activate before this import.
_pipeline_role = os.environ.get("ARC_PIPELINE_ROLE")
if _pipeline_role or not split_runtime_requested():
    bootstrap_transformer_engine_cpp_runtime()
    if _pipeline_role != "trainer":
        patch_vllm_compact_vocab_sampler_warmup()
        patch_vllm_diffusion_gemma_cudagraph_causal_buffer()

import atexit
import gc
import json
import inspect
import importlib.metadata
import math
import shutil
import socket
import subprocess
import time
import torch
import argparse
import uuid
from dataclasses import dataclass
from contextlib import ExitStack
import torch.distributed as dist
import torch.multiprocessing as mp
from queue import Queue

from run_config import (
    GRADIENT_CHECKPOINTING_MODES,
    LORA_METHODS,
    VLLM_KV_CACHE_DTYPES,
    is_pissa_init,
    lora_engine_max_rank,
    lora_export_rank,
    normalize_lora_init,
    normalize_pissa_base_mode,
    validate_pissa_options,
    TASK_ORDER_MODES,
    normalize_gradient_checkpointing_mode,
    normalize_vllm_kv_cache_dtype,
    normalize_vllm_kv_resident_fraction,
    normalize_vllm_search_limit_seconds,
    order_task_keys,
    resolve_challenge_path,
    resolve_model_path,
    resolve_vllm_kv_resident_fraction,
    select_task_keys,
)
from arc_deepspeed import (
    DEFAULT_LONG_CONTEXT_TOKEN_THRESHOLD,
    DEFAULT_MAX_LIVE_PARAMETERS,
    DEFAULT_MODEL_PERSISTENCE_THRESHOLD,
    DEFAULT_PARAM_PERSISTENCE_THRESHOLD,
    DEFAULT_PREFETCH_BUCKET_SIZE,
    validate_long_context_deepspeed_options,
)
from arc_fp8_lora import validate_fp8_lora_options
from early_routing import (
    add_early_routing_arguments,
    run_early_routing,
    validate_early_routing_arguments,
)
from run_status import (
    STATUS_STARTER_SUCCESS,
    extract_run_status_path,
    run_entrypoint_with_status,
)


MAX_GPU_COUNT = 4
DEFAULT_RAM_ADAPTER_ROOT = "/dev/shm/arc_inference_adapters"
DEFAULT_DISK_ADAPTER_ROOT = "/kaggle/working/arc_vllm_adapters"
DEFAULT_TMP_ADAPTER_ROOT = "/kaggle/tmp/arc_vllm_adapters"
# The current rank-256 ARC adapter, including converted embedding and output
# head deltas, is about 1010.5 MiB. Keep a small filesystem/headroom margin per
# bounded pending slot so a zero-service ramp fails before training work is
# lost.
ADAPTER_STAGING_ESTIMATE_MIB = 1100
DISTRIBUTED_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
SPAWN_FAILURE_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class AdapterStagingTier:
    label: str
    parent: str
    free_bytes: int
    slot_capacity: int


def _adapter_staging_filesystem_id(path):
    """Identify shared free space, including symlinks and bind mounts."""
    return os.stat(path).st_dev


def choose_adapter_staging_plan(
    storage,
    requested_root,
    max_artifacts,
    *,
    ram_root=None,
    disk_root=None,
    tmp_root=None,
    bytes_per_slot_mib=ADAPTER_STAGING_ESTIMATE_MIB,
):
    """Plan one or more prioritized staging tiers for a bounded live set."""
    storage = str(storage).strip().lower()
    if storage not in {"auto", "ram", "disk"}:
        raise ValueError(
            f"adapter storage must be one of auto, ram, or disk; got {storage!r}"
        )
    max_artifacts = int(max_artifacts)
    if max_artifacts < 1:
        raise ValueError("max_artifacts must be positive")

    ram_root = ram_root or os.getenv(
        "ARC_INFERENCE_RAM_ADAPTER_ROOT",
        DEFAULT_RAM_ADAPTER_ROOT,
    )
    disk_root = disk_root or os.getenv(
        "ARC_INFERENCE_DISK_ADAPTER_ROOT",
        DEFAULT_DISK_ADAPTER_ROOT,
    )
    tmp_root = tmp_root or os.getenv(
        "ARC_INFERENCE_TMP_ADAPTER_ROOT",
        DEFAULT_TMP_ADAPTER_ROOT,
    )
    if requested_root:
        candidates = [("custom", requested_root)]
        allow_spill = False
    elif storage == "ram":
        candidates = [("ram", ram_root)]
        allow_spill = False
    elif storage == "disk":
        candidates = [("disk", disk_root)]
        allow_spill = False
    else:
        candidates = [("ram", ram_root), ("disk", disk_root), ("tmp", tmp_root)]
        allow_spill = True

    bytes_per_slot_mib = int(bytes_per_slot_mib)
    if bytes_per_slot_mib < 1:
        raise ValueError("bytes_per_slot_mib must be positive")
    bytes_per_slot = bytes_per_slot_mib * 1024**2
    required_bytes = max_artifacts * bytes_per_slot
    remaining = max_artifacts
    tiers: list[AdapterStagingTier] = []
    failures: list[str] = []
    seen_paths: set[str] = set()
    seen_filesystems: dict[int, str] = {}
    total_free_bytes = 0
    for label, candidate in candidates:
        path = os.path.abspath(os.path.expanduser(os.fspath(candidate)))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            os.makedirs(path, exist_ok=True)
            if not os.path.isdir(path) or not os.access(path, os.W_OK):
                raise OSError("directory is not writable")
            filesystem_id = _adapter_staging_filesystem_id(path)
            if filesystem_id in seen_filesystems:
                # Different directories can expose the same free-space pool.
                # Counting both would reserve slots beyond the disk capacity.
                failures.append(
                    f"{label} {path}: shares a filesystem with "
                    f"{seen_filesystems[filesystem_id]}; free space already counted"
                )
                continue
            free_bytes = int(shutil.disk_usage(path).free)
            raw_capacity = free_bytes // bytes_per_slot
            if raw_capacity < 1:
                raise OSError(
                    f"only {free_bytes / 1024**3:.2f} GiB free; less than "
                    f"one {bytes_per_slot_mib} MiB staging slot"
                )
        except OSError as exc:
            failures.append(f"{label} {path}: {exc}")
            continue

        seen_filesystems[filesystem_id] = f"{label} {path}"
        total_free_bytes += free_bytes
        assigned_capacity = min(remaining, int(raw_capacity))
        if assigned_capacity:
            tiers.append(
                AdapterStagingTier(
                    label=label,
                    parent=path,
                    free_bytes=free_bytes,
                    slot_capacity=assigned_capacity,
                )
            )
            remaining -= assigned_capacity
        if remaining == 0:
            break
        if not allow_spill:
            failures.append(
                f"{label} {path}: only {free_bytes / 1024**3:.2f} GiB free; "
                f"need about {required_bytes / 1024**3:.2f} GiB"
            )
            break

    if remaining:
        available_slots = max_artifacts - remaining
        detail = "; ".join(failures) if failures else "no writable tier"
        if allow_spill and tiers:
            tier_details = ", ".join(
                f"{tier.label}={tier.slot_capacity} slot(s)"
                for tier in tiers
            )
            detail = (
                f"combined capacity is {available_slots}/{max_artifacts} "
                f"slot(s) ({tier_details}); {detail}"
            )
        raise RuntimeError(
            "No adapter staging filesystem pool has enough free space for "
            f"{max_artifacts} staged artifact(s): {detail}"
        )
    return tuple(tiers), total_free_bytes, required_bytes


def choose_adapter_staging_parent(
    storage,
    requested_root,
    max_artifacts,
    *,
    ram_root=None,
    disk_root=None,
    tmp_root=None,
):
    """Compatibility wrapper around the tiered staging planner.

    A hybrid auto plan returns a tuple of parent paths and joined tier labels;
    single-tier callers retain the historical string path.
    """
    tiers, total_free_bytes, required_bytes = choose_adapter_staging_plan(
        storage,
        requested_root,
        max_artifacts,
        ram_root=ram_root,
        disk_root=disk_root,
        tmp_root=tmp_root,
    )
    parents = tuple(tier.parent for tier in tiers)
    labels = "+".join(tier.label for tier in tiers)
    parent_or_parents = parents[0] if len(parents) == 1 else parents
    return parent_or_parents, labels, total_free_bytes, required_bytes


def model_declares_delta_mhar(model_path):
    if os.path.isfile(os.path.join(model_path, "delta_mhar_config.json")):
        return True
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as file_handle:
        return json.load(file_handle).get("unsloth_delta_mhar") is not None


def preflight_vllm_delta_mhar_plugin(model_path):
    """Validate the routed checkpoint and its installed vLLM entry point."""
    project_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "vllm_delta_mhar_plugin",
    )
    try:
        entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    except TypeError:  # Python 3.10 compatibility
        entry_points = importlib.metadata.entry_points().get("vllm.general_plugins", ())
    matching = [entry for entry in entry_points if entry.name == "arc_delta_mhar"]
    if len(matching) != 1:
        raise RuntimeError(
            "Routed vLLM inference requires the local model plugin. Run "
            f"`python -m pip install --no-deps -e {project_root}` before "
            "starter.py."
        )
    from arc_vllm_delta_mhar.config import read_delta_mhar_spec

    spec = read_delta_mhar_spec(model_path)
    allowed_plugins = [
        name for name in os.getenv("VLLM_PLUGINS", "").split(",") if name
    ]
    if allowed_plugins and "arc_delta_mhar" not in allowed_plugins:
        raise RuntimeError(
            f"VLLM_PLUGINS is set but excludes 'arc_delta_mhar': {allowed_plugins}"
        )
    # Set the filter before multiprocessing spawn and before any vLLM import.
    if not allowed_plugins:
        os.environ["VLLM_PLUGINS"] = "arc_delta_mhar"
    return spec


def env_flag(name, default):
    """Read a boolean environment variable with explicit accepted values."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def vllm_batch_invariant_supports_cuda_graphs(device_capability):
    """Return whether vLLM batch invariance supports graphs on this GPU."""
    major, minor = map(int, device_capability)
    return (major, minor) >= (9, 0)


def validate_zero_service_adapter_capacity(
    *,
    selected_count,
    max_pending_adapters,
    post_training_scale,
    initial_service_count,
    ramp_after_adapters=0,
):
    """Require enough staging capacity to reach the first inference phase."""
    ramp_after_adapters = int(ramp_after_adapters)
    required = (
        min(ramp_after_adapters, int(selected_count))
        if ramp_after_adapters > 0 else int(selected_count)
    )
    if (
        (post_training_scale or ramp_after_adapters > 0)
        and int(initial_service_count) == 0
        and int(max_pending_adapters) < required
    ):
        if ramp_after_adapters > 0:
            raise ValueError(
                "A zero-initial-service ramp must buffer every pre-ramp "
                f"adapter ({required}); set --vllm-max-pending-adapters "
                f"to at least {required} (the smaller of the ramp threshold "
                "and selected task count)"
            )
        raise ValueError(
            "A zero-initial-service post-training run must buffer every "
            f"selected adapter ({selected_count}); set "
            "--vllm-max-pending-adapters to at least that value and choose "
            "adapter storage with enough free space"
        )


def validate_routed_vllm_runtime_options(*, model_family, batch_invariant):
    """Reject vLLM modes that the routed model family cannot execute."""
    if str(model_family) == "qwen3_5" and bool(batch_invariant):
        raise ValueError(
            "vLLM batch-invariant mode is not supported by Qwen3.5's hybrid "
            "GDN attention backend; use --no-vllm-batch-invariant"
        )


def training_worker_options_for_model(worker_options, *, model_family):
    """Return only options accepted by the selected training worker.

    DiffusionGemma's trainer deliberately accepts a model-specific ``**options``
    payload.  The Qwen trainer has an explicit signature so leaking routing or
    DiffusionGemma-only fields into that payload fails before training starts.
    Keep ``model_family`` in the inference-service options instead.
    """
    options = dict(worker_options)
    if str(model_family) == "diffusion_gemma":
        if options.pop("sft_loss_scope", "all") != "all":
            raise ValueError("--sft-loss-scope final supports Qwen task SFT only")
    else:
        options.pop("model_family", None)
        options = {
            key: value
            for key, value in options.items()
            if not key.startswith("diffusion_gemma_")
        }
    return options


def validate_diffusion_gemma_pair_fsdp_topology(
    *,
    enabled,
    selected_count,
    worker_count,
    service_gpu_count,
    ramp_gpu_count,
    post_training_scale,
    experts_backend,
):
    """Validate the first audited independent-adapter pair-FSDP topology."""
    if not enabled:
        return
    if int(selected_count) < 2 or int(selected_count) % 2:
        raise ValueError(
            "--diffusion-gemma-pair-fsdp requires an even selected task "
            "count of at least two"
        )
    if int(worker_count) != 2 or int(service_gpu_count) != 2:
        raise ValueError(
            "--diffusion-gemma-pair-fsdp currently requires exactly "
            "--multi-task-gpus 2 and --vllm-gpus 2"
        )
    if int(ramp_gpu_count) or bool(post_training_scale):
        raise ValueError(
            "--diffusion-gemma-pair-fsdp cannot yet convert a sharded "
            "trainer rank into an inference service; use two persistent "
            "vLLM GPUs without an inference ramp"
        )
    if str(experts_backend) != "cutlass-ada-triton":
        raise ValueError(
            "--diffusion-gemma-pair-fsdp requires "
            "--diffusion-gemma-lora-experts-backend cutlass-ada-triton"
        )


def validate_diffusion_gemma_expert_parallel_topology(
    *,
    size,
    pair_fsdp,
    selected_count,
    worker_count,
    service_gpu_count,
    ramp_gpu_count,
    post_training_scale,
    experts_backend,
):
    """Validate activation-dispatch EP with one independent task per rank."""
    size = int(size)
    if size == 1:
        return
    if size not in {2, 3}:
        raise ValueError("--diffusion-gemma-expert-parallel-size must be 1, 2, or 3")
    if bool(pair_fsdp):
        raise ValueError(
            "--diffusion-gemma-expert-parallel-size cannot be combined with "
            "--diffusion-gemma-pair-fsdp"
        )
    if int(selected_count) < size or int(selected_count) % size:
        raise ValueError(
            "DiffusionGemma expert parallelism requires the selected task "
            f"count to be a positive multiple of {size}"
        )
    if int(worker_count) != size or int(service_gpu_count) != 4 - size:
        raise ValueError(
            f"EP size {size} requires --multi-task-gpus {size} and "
            f"--vllm-gpus {4 - size} on the four-L4 topology"
        )
    if int(ramp_gpu_count) or bool(post_training_scale):
        raise ValueError(
            "DiffusionGemma expert-parallel ranks must remain in their EP "
            "group; inference ramp/scale-after-training is not yet supported"
        )
    if str(experts_backend) != "cutlass-ada-triton":
        raise ValueError(
            "DiffusionGemma expert parallelism requires "
            "--diffusion-gemma-lora-experts-backend cutlass-ada-triton"
        )


def prebuild_diffusion_gemma_pair_fsdp_extension(*, enabled, ream_root):
    """Build the shared CUTLASS extension once before pair workers spawn.

    Two fresh trainer processes otherwise enter ``cpp_extension.load`` at the
    same first expert projection.  PyTorch serializes them with an empty lock
    file; if a prior notebook run was interrupted during JIT, that lock is
    left behind and both ranks wait forever.  A short-lived subprocess avoids
    retaining a CUDA context in the starter and makes that boundary explicit.
    Pair-FSDP already forbids concurrent starter topologies, so an existing
    lock at this pre-spawn point is necessarily stale for this run.
    """
    if not enabled:
        return
    ream_root = os.path.abspath(os.fspath(ream_root))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item
        for item in (ream_root, env.get("PYTHONPATH", ""))
        if item
    )
    code = """
from pathlib import Path
from torch.utils.cpp_extension import _get_build_directory

build_dir = Path(_get_build_directory("ream_ada_grouped_fp8", verbose=False))
lock = build_dir / "lock"
if lock.exists():
    lock.unlink()
from ream.diffusion_gemma_grouped_fp8_cutlass import load_ada_grouped_fp8_extension
extension = load_ada_grouped_fp8_extension()
if not hasattr(extension, "batched_transpose_fp8"):
    raise RuntimeError("Ada grouped-FP8 extension lacks transpose support")
print("DiffusionGemma pair-FSDP CUTLASS extension ready", flush=True)
"""
    subprocess.run(
        [sys.executable, "-u", "-c", code],
        check=True,
        env=env,
    )


def parse_gradient_checkpointing_percentage(value):
    """Parse an explicit checkpoint percentage or the adaptive policy."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        percentage = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a number from 0 through 100"
        ) from exc
    if not 0.0 <= percentage <= 100.0:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a number from 0 through 100"
        )
    return percentage


def max_beam_score_from_probability(value):
    """Convert an intuitive retained-path probability to cumulative NLL."""
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a probability greater than 0 and at most 1"
        ) from exc
    if not 0.0 < probability <= 1.0:
        raise argparse.ArgumentTypeError(
            "must be a probability greater than 0 and at most 1"
        )
    return -math.log(probability)


def positive_int(value):
    """Parse a strictly positive integer for size and capacity options."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def clear_distributed_env():
    for name in DISTRIBUTED_ENV_VARS:
        os.environ.pop(name, None)


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def select_process_gpu(gpu_index):
    """Restrict a spawned process to one physical GPU before CUDA use."""
    selected = os.environ.get("ARC_PIPELINE_SELECTED_GPU")
    if selected is not None and os.environ.get("ARC_PIPELINE_ROLE"):
        if int(selected) != gpu_index:
            raise RuntimeError(f"Split worker GPU mismatch: preselected {selected}, requested {gpu_index}")
        # The subprocess environment was restricted before importing Torch.
        return
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        device_ids = [
            device_id.strip()
            for device_id in visible_devices.split(",")
            if device_id.strip()
        ]
        if gpu_index >= len(device_ids):
            raise RuntimeError(
                f"GPU index {gpu_index} is outside CUDA_VISIBLE_DEVICES="
                f"{visible_devices!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids[gpu_index]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


def record_ramp_trainer_completion(
    *, ramp_trainers_finished_count, trainer_count, ramp_ready_event,
):
    """Release an unreached ramp once no lane can publish another adapter.

    Count immediately after the training worker returns, BEFORE waiting for
    ramp slots or recording final scale-out completion. Those later barriers
    depend on the ramp being released and cannot be used to detect exhaustion.
    Early-converted lanes also count: they will not publish more adapters.
    """
    with ramp_trainers_finished_count.get_lock():
        completed = int(ramp_trainers_finished_count.value) + 1
        if completed > int(trainer_count):
            raise RuntimeError("A ramp trainer reported completion more than once")
        ramp_trainers_finished_count.value = completed
        released = completed == int(trainer_count) and not ramp_ready_event.is_set()
        if released:
            ramp_ready_event.set()
    return completed, released


def record_post_training_trainer_completion(
    *,
    trainers_finished_count,
    trainer_count,
    service_active,
    initial_service_count,
    service_load_lock,
    training_complete_event,
    rebalance_complete_event,
    trainer_service_indices=None,
    shared_pending_pool=False,
):
    """Record one exhausted trainer and activate future services at the end.

    Every trainer has published its last adapter and emitted ``trainer_done``
    before calling this helper, including lanes converted by an early ramp.
    ``initial_service_count`` includes those early-ramped services. Only the
    last training lane exposes final services to pending work. With a shared
    pool there is no per-engine backlog to redistribute.
    """
    trainer_count = int(trainer_count)
    initial_service_count = int(initial_service_count)
    with trainers_finished_count.get_lock():
        trainers_finished_count.value = int(trainers_finished_count.value) + 1
        completed = int(trainers_finished_count.value)
        if completed > trainer_count:
            raise RuntimeError("A trainer reported completion more than once")
        last_trainer = completed == trainer_count

    if last_trainer:
        with service_load_lock:
            if trainer_service_indices is not None:
                # Threshold conversions can be claimed by any physical rank.
                # Once every lane has stopped training, map the remaining
                # ranks to unique final slots without reusing an early slot.
                claimed = [int(i) for i in trainer_service_indices if i >= 0]
                unassigned = [
                    rank
                    for rank, index in enumerate(trainer_service_indices)
                    if index < 0
                ]
                future = list(range(initial_service_count, len(service_active)))
                if (
                    len(trainer_service_indices) != trainer_count
                    or len(set(claimed)) != len(claimed)
                    or any(index >= initial_service_count for index in claimed)
                    or len(unassigned) != len(future)
                ):
                    raise RuntimeError(
                        "Incomplete or duplicate trainer/service mapping"
                    )
                for rank, service_index in zip(unassigned, future):
                    trainer_service_indices[rank] = service_index
            for service_index in range(
                initial_service_count,
                len(service_active),
            ):
                service_active[service_index] = 1
            active_services = tuple(bool(value) for value in service_active)
        # The shared pool needs no redistribution barrier. The no-initial-
        # engine legacy path also has no local scheduler backlog to migrate.
        if initial_service_count == 0 or shared_pending_pool:
            rebalance_complete_event.set()
        training_complete_event.set()
    else:
        with service_load_lock:
            active_services = tuple(bool(value) for value in service_active)
    return completed, last_trainer, active_services


def join_spawn_context(
    process_context,
    grace_period=SPAWN_FAILURE_GRACE_SECONDS,
):
    """Join spawned workers and bound sibling cleanup after one fails."""
    while not poll_spawn_context(
        process_context,
        timeout=None,
        grace_period=grace_period,
    )[0]:
        pass


def poll_spawn_context(
    process_context,
    *,
    timeout=0.0,
    grace_period=SPAWN_FAILURE_GRACE_SECONDS,
):
    """Poll one torch spawn context and report newly reaped process ranks."""
    try:
        join_parameters = inspect.signature(process_context.join).parameters
        supports_timeout = "timeout" in join_parameters
        supports_grace_period = "grace_period" in join_parameters
    except (TypeError, ValueError):
        supports_timeout = False
        supports_grace_period = False
    sentinels = getattr(process_context, "sentinels", {})
    before = set(sentinels.values())
    join_kwargs = {}
    if supports_timeout:
        join_kwargs["timeout"] = timeout
    elif timeout is not None:
        raise RuntimeError(
            "This torch multiprocessing build cannot poll spawned workers"
        )
    if supports_grace_period:
        join_kwargs["grace_period"] = float(grace_period)
    complete = bool(process_context.join(**join_kwargs))
    after = set(sentinels.values())
    return complete, tuple(sorted(before - after))


def _run_worker_on_assigned_rank(
    _local_spawn_rank,
    function,
    assigned_rank,
    function_args,
):
    """Let a one-process spawn context retain an existing physical rank."""
    function(int(assigned_rank), *tuple(function_args))


def _terminate_spawn_contexts(
    process_contexts,
    *,
    grace_period=SPAWN_FAILURE_GRACE_SECONDS,
):
    """Best-effort cleanup for independent trainer and service contexts."""
    processes = []
    seen = set()
    for process_context in process_contexts:
        for process in getattr(process_context, "processes", ()):
            identity = id(process)
            if identity not in seen:
                seen.add(identity)
                processes.append(process)

    deadline = time.monotonic() + float(grace_period)
    while any(process.is_alive() for process in processes):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + float(grace_period)
    while any(process.is_alive() for process in processes):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join(timeout=float(grace_period))


def spawn_trainers_with_per_lane_services(
    trainer_function,
    service_function,
    *,
    trainer_args,
    service_args,
    nprocs,
    abort_event,
    poll_seconds=0.25,
    spawn_function=mp.spawn,
):
    """Replace each exhausted trainer with a fresh same-GPU service.

    A DiffusionGemma trainer owns CUDA and Transformer Engine state that cannot
    be safely reused by vLLM.  The parent therefore waits until the exact
    trainer child has exited, reaps it, and only then starts a one-process
    service context carrying the same logical/physical rank.  Other trainer
    children keep running while that engine loads and performs inference.
    """
    nprocs = int(nprocs)
    if nprocs < 1:
        raise ValueError("Per-lane handoff requires at least one trainer")
    poll_seconds = max(0.01, float(poll_seconds))
    trainer_context = spawn_function(
        trainer_function,
        args=tuple(trainer_args),
        nprocs=nprocs,
        join=False,
    )
    remaining_trainers = set(range(nprocs))
    service_contexts = {}
    all_contexts = [trainer_context]
    try:
        while remaining_trainers or service_contexts:
            made_progress = False
            if remaining_trainers:
                _trainers_complete, completed_ranks = poll_spawn_context(
                    trainer_context,
                    timeout=poll_seconds,
                )
                for rank in completed_ranks:
                    if rank not in remaining_trainers:
                        raise RuntimeError(
                            f"Trainer rank {rank} completed more than once"
                        )
                    remaining_trainers.remove(rank)
                    print(
                        f"[GPU {rank}] trainer process exited cleanly; "
                        "starting its vLLM service immediately while "
                        f"{len(remaining_trainers)} trainer(s) continue",
                        flush=True,
                    )
                    service_context = spawn_function(
                        _run_worker_on_assigned_rank,
                        args=(
                            service_function,
                            rank,
                            tuple(service_args),
                        ),
                        nprocs=1,
                        join=False,
                    )
                    service_contexts[rank] = service_context
                    all_contexts.append(service_context)
                    made_progress = True

            for rank, service_context in tuple(service_contexts.items()):
                service_complete, _completed = poll_spawn_context(
                    service_context,
                    timeout=0.0,
                )
                if service_complete:
                    del service_contexts[rank]
                    print(
                        f"[GPU {rank}] per-lane vLLM service exited cleanly",
                        flush=True,
                    )
                    made_progress = True

            if not remaining_trainers and service_contexts and not made_progress:
                time.sleep(min(poll_seconds, 0.05))
        return
    except BaseException:
        abort_event.set()
        _terminate_spawn_contexts(all_contexts)
        raise


def spawn_and_join(function, *, args, nprocs):
    """Spawn without the legacy 30-second failure wait, then join safely."""
    process_context = mp.spawn(
        function,
        args=args,
        nprocs=nprocs,
        join=False,
    )
    join_spawn_context(process_context)


def close_process_queue(process_queue, *, successful):
    """Close a multiprocessing queue without masking a worker exception.

    A feeder thread cannot flush into a reader that PyTorch just terminated.
    Joining that feeder on a failed run can therefore hang forever.  Failed
    runs explicitly cancel feeder joining; successful runs retain the normal
    flush-and-join behavior.
    """
    if not successful:
        try:
            process_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
    try:
        process_queue.close()
    except (OSError, ValueError):
        return
    if successful:
        try:
            process_queue.join_thread()
        except (AttributeError, OSError, ValueError):
            pass


def independent_worker(
    rank,
    task_queue,
    import_lock,
    abort_event,
    end_time,
    keep_training_state,
    worker_options,
):
    clear_distributed_env()
    select_process_gpu(rank)
    torch.set_default_device("cpu")

    # Unsloth may patch or generate shared files during import.
    with import_lock:
        from arc_solver import worker

    print(f"[GPU {rank}] ready for dynamic task scheduling")
    try:
        worker(
            rank,
            task_queue,
            end_time,
            distributed=False,
            keep_training_state=keep_training_state,
            abort_event=abort_event,
            **worker_options,
        )
    except BaseException:
        abort_event.set()
        raise
    print(f"[GPU {rank}] no more tasks")


def vllm_pipeline_worker(
    process_rank,
    trainer_count,
    vllm_gpu_count,
    task_queue,
    job_queues,
    rollout_response_queues,
    pending_adapter_slots,
    service_outstanding,
    service_load_lock,
    vllm_startup_lock,
    import_lock,
    abort_event,
    end_time,
    keep_training_state,
    worker_options,
    inference_options,
    model_path,
    adapter_root,
    run_id,
    adapter_ids,
    service_active=None,
    published_adapter_count=None,
    ramp_ready_event=None,
    ramp_after_adapters=0,
    total_service_count=None,
    post_training_scale=False,
    trainers_finished_count=None,
    training_complete_event=None,
    post_training_rebalance_count=None,
    post_training_rebalance_event=None,
    initial_service_count=None,
    service_ready=None,
    adapter_roots=None,
    adapter_staging_slots=None,
    defer_post_training_service=False,
    service_only=False,
    trainer_service_indices=None,
    shared_pending_queue=None,
    shared_pending_count=None,
    ramp_trainers_finished_count=None,
):
    """Run one trainer or one independent TP=1 inference service.

    The historical function name is retained because spawned-worker tests and
    old notebooks import it directly. ``inference_options['service_backend']``
    selects vLLM or SGLang; the trainer-side queue protocol is shared.
    """
    clear_distributed_env()
    service_backend = str(inference_options.get("service_backend", "vllm"))
    model_family = str(inference_options.get("model_family", "qwen"))
    shared_pending_pool = shared_pending_queue is not None
    if shared_pending_pool != (shared_pending_count is not None):
        raise RuntimeError("The shared adapter pool requires both queue and counter")
    if shared_pending_pool and (service_backend != "vllm" or model_family != "qwen"):
        raise RuntimeError("The shared adapter pool is currently for Qwen vLLM only")
    if service_backend not in {"vllm", "sglang"}:
        raise RuntimeError(f"Unsupported inference service backend: {service_backend}")
    if model_family == "diffusion_gemma" and service_backend != "vllm":
        raise RuntimeError("DiffusionGemma requires the native vLLM service")
    if model_family not in {"qwen", "diffusion_gemma"}:
        raise RuntimeError(f"Unsupported model family: {model_family}")
    initial_service_gpu_indices = list(
        range(MAX_GPU_COUNT - vllm_gpu_count, MAX_GPU_COUNT)
    )
    total_service_count = int(
        len(job_queues) if total_service_count is None else total_service_count
    )
    if total_service_count != len(job_queues):
        raise RuntimeError(
            f"Expected {total_service_count} service queues, got {len(job_queues)}"
        )
    adaptive_ramp = int(ramp_after_adapters) > 0
    post_training_scale = bool(post_training_scale)
    combined_ramp = adaptive_ramp and post_training_scale
    initial_service_count = int(
        vllm_gpu_count if initial_service_count is None else initial_service_count
    )
    if post_training_scale:
        # "Initial" here includes early-ramped engines. This separates warm
        # service slots from the final trainer-to-service conversions.
        if combined_ramp:
            if (
                not int(vllm_gpu_count)
                < initial_service_count
                < total_service_count
                or trainer_service_indices is None
                or len(trainer_service_indices) != trainer_count
            ):
                raise RuntimeError(
                    "Incomplete combined inference-ramp configuration"
                )
        elif initial_service_count != int(vllm_gpu_count):
            raise RuntimeError(
                "The post-training initial-service count must match the "
                "number of startup inference GPUs"
            )
        if any(
            value is None
            for value in (
                service_active,
                trainers_finished_count,
                training_complete_event,
                post_training_rebalance_count,
                post_training_rebalance_event,
            )
        ):
            raise RuntimeError(
                "Incomplete post-training inference scale-out configuration"
            )

    def wait_for_post_training_service():
        for event in (training_complete_event, post_training_rebalance_event):
            while not event.wait(timeout=0.5):
                if abort_event.is_set() or time.time() >= end_time:
                    return None
        service_index = (
            int(trainer_service_indices[process_rank])
            if combined_ramp
            else initial_service_count + process_rank
        )
        if not initial_service_count <= service_index < total_service_count:
            raise RuntimeError(
                f"Invalid post-training service assignment {service_index}"
            )
        from arc_vllm_router import service_load_snapshot

        service_loads = service_load_snapshot(
            service_outstanding, service_load_lock,
        )
        if shared_pending_pool:
            from arc_vllm_router import SERVICE_FINISHED

            # Final services need no assigned backlog or migration barrier.
            # If every job is already claimed, avoid a needless cold start.
            # Serialize closing with any recovery requeue into the pool.
            with service_load_lock:
                pool_pending = int(shared_pending_count.value)
                if pool_pending == 0:
                    service_ready[service_index] = SERVICE_FINISHED
            if pool_pending == 0:
                print(
                    f"[GPU {process_rank}] shared adapter pool empty; skip cold "
                    f"start of post-training service {service_index}", flush=True,
                )
                return None
            print(
                f"[GPU {process_rank}] all training is complete; starting "
                f"post-training service {service_index}; shared pending={pool_pending}",
                flush=True,
            )
            return service_index
        if service_loads[service_index] < 1:
            print(
                f"[GPU {process_rank}] no queued adapter was moved to "
                f"post-training service {service_index}; skip its cold start; "
                f"outstanding={list(service_loads)}",
                flush=True,
            )
            return None
        print(
            f"[GPU {process_rank}] all training is complete; starting "
            f"post-training service {service_index} with "
            f"{service_loads[service_index]} queued adapter(s); "
            f"outstanding={list(service_loads)}",
            flush=True,
        )
        return service_index

    def run_service(service_index, gpu_index, *, gpu_already_selected):
        if not gpu_already_selected:
            select_process_gpu(gpu_index)
        torch.set_default_device("cpu")
        if service_backend == "sglang":
            from arc_sglang_infer import (
                run_sglang_inference_worker as run_inference_worker,
            )
        elif model_family == "diffusion_gemma":
            from diffusion_gemma_pipeline import (
                run_diffusion_gemma_vllm_worker as run_inference_worker,
            )
        else:
            from arc_vllm_infer import (
                run_vllm_inference_worker as run_inference_worker,
            )

        phase = "adaptive" if gpu_already_selected else "initial"
        print(
            f"[GPU {gpu_index}] starting {phase} persistent "
            f"{service_backend} service {service_index}/"
            f"{total_service_count - 1} (TP=1)"
        )
        run_inference_worker(
            job_queue=job_queues[service_index],
            rollout_response_queues=rollout_response_queues,
            pending_adapter_slots=pending_adapter_slots,
            service_index=service_index,
            service_outstanding=service_outstanding,
            service_load_lock=service_load_lock,
            startup_lock=vllm_startup_lock,
            trainer_count=trainer_count,
            end_time=end_time,
            model_path=model_path,
            adapter_root=adapter_root,
            adapter_staging_slots=adapter_staging_slots,
            options=inference_options,
            abort_event=abort_event,
            **(
                {
                    "all_job_queues": job_queues,
                    "service_active": service_active,
                    "service_ready": service_ready,
                    "initial_service_count": initial_service_count,
                    "post_training_scale": post_training_scale,
                    "post_training_rebalance_count": (post_training_rebalance_count),
                    "post_training_rebalance_event": (post_training_rebalance_event),
                    **(
                        {
                            "shared_pending_queue": shared_pending_queue,
                            "shared_pending_count": shared_pending_count,
                        }
                        if model_family == "qwen" else {}
                    ),
                }
                if service_backend == "vllm"
                else {}
            ),
        )

    if service_only:
        if not post_training_scale:
            raise RuntimeError(
                "A fresh service-only phase requires post-training scale-out"
            )
        service_index = int(process_rank)
        if int(vllm_gpu_count) > 0:
            service_index = wait_for_post_training_service()
            if service_index is None:
                return
        elif combined_ramp:
            service_index = int(trainer_service_indices[process_rank])
            if service_index < 0 or service_index >= initial_service_count:
                service_index = wait_for_post_training_service()
                if service_index is None:
                    return
        if not 0 <= service_index < total_service_count:
            raise RuntimeError(
                f"Invalid fresh post-training service index {service_index} "
                f"for {total_service_count} service(s)"
            )
        print(
            f"[GPU {process_rank}] starting post-training service "
            "in a fresh process after trainer CUDA-context exit",
            flush=True,
        )
        run_service(
            service_index,
            process_rank,
            gpu_already_selected=False,
        )
        return

    if process_rank < trainer_count:
        select_process_gpu(process_rank)
        torch.set_default_device("cpu")
        try:
            with import_lock:
                if model_family == "diffusion_gemma":
                    from diffusion_gemma_pipeline import (
                        diffusion_gemma_trainer_worker as worker,
                    )
                else:
                    from arc_solver import worker

            delegated = initial_service_gpu_indices or ["adaptive after threshold"]
            print(
                f"[GPU {process_rank}] ready for dynamic training; "
                f"{service_backend} inference is delegated to {delegated}"
            )
            adaptive_service_index = worker(
                process_rank,
                task_queue,
                end_time,
                distributed=False,
                keep_training_state=keep_training_state,
                vllm_job_queues=job_queues,
                vllm_rollout_response_queues=rollout_response_queues,
                vllm_pending_adapter_slots=pending_adapter_slots,
                vllm_service_outstanding=service_outstanding,
                vllm_service_load_lock=service_load_lock,
                vllm_adapter_root=adapter_root,
                vllm_adapter_roots=adapter_roots,
                vllm_adapter_staging_slots=adapter_staging_slots,
                vllm_run_id=run_id,
                vllm_adapter_ids=adapter_ids,
                vllm_service_active=service_active,
                vllm_service_ready=(
                    service_ready
                    if service_backend == "vllm" and int(vllm_gpu_count) > 0
                    else None
                ),
                vllm_published_adapter_count=published_adapter_count,
                vllm_ramp_ready_event=ramp_ready_event,
                vllm_ramp_after_adapters=ramp_after_adapters,
                abort_event=abort_event,
                **(
                    {
                        "vllm_shared_pending_queue": shared_pending_queue,
                        "vllm_shared_pending_count": shared_pending_count,
                    }
                    if shared_pending_pool else {}
                ),
                **(
                    {"vllm_ramp_service_limit": initial_service_count}
                    if combined_ramp else {}
                ),
                **worker_options,
            )
            if adaptive_ramp and ramp_trainers_finished_count is not None:
                completed, released = record_ramp_trainer_completion(
                    ramp_trainers_finished_count=ramp_trainers_finished_count,
                    trainer_count=trainer_count,
                    ramp_ready_event=ramp_ready_event,
                )
                if released:
                    print(
                        f"[GPU {process_rank}] all {completed} training lanes "
                        "finished before the inference ramp threshold "
                        f"({int(published_adapter_count.value)}/{ramp_after_adapters} "
                        "adapters); releasing deferred inference ramp",
                        flush=True,
                    )
            if combined_ramp:
                if adaptive_service_index is None:
                    # A lane may exhaust the task queue before the threshold
                    # publisher finishes, or before the last lane releases an
                    # unreached ramp. Claim any unfilled EARLY slot before
                    # counting this lane as done; late slots remain reserved.
                    while not ramp_ready_event.wait(timeout=0.5):
                        if abort_event.is_set() or time.time() >= end_time:
                            return
                    from arc_vllm_router import activate_inactive_service

                    adaptive_service_index, _, _ = activate_inactive_service(
                        service_outstanding, service_active, service_load_lock,
                        service_limit=initial_service_count,
                    )
                if adaptive_service_index is not None:
                    if not (
                        int(vllm_gpu_count)
                        <= adaptive_service_index
                        < initial_service_count
                    ):
                        raise RuntimeError(
                            "Trainer claimed a non-ramp inference service"
                        )
                    with service_load_lock:
                        trainer_service_indices[process_rank] = adaptive_service_index
            if adaptive_service_index is None:
                print(f"[GPU {process_rank}] no more training tasks")
            for job_queue in job_queues:
                job_queue.put(
                    {
                        "kind": "trainer_done",
                        "rank": process_rank,
                    }
                )

            if post_training_scale:
                (
                    finished_trainers,
                    last_trainer,
                    active_services,
                ) = record_post_training_trainer_completion(
                    trainers_finished_count=trainers_finished_count,
                    trainer_count=trainer_count,
                    service_active=service_active,
                    initial_service_count=initial_service_count,
                    service_load_lock=service_load_lock,
                    training_complete_event=training_complete_event,
                    rebalance_complete_event=(post_training_rebalance_event),
                    trainer_service_indices=(
                        trainer_service_indices if combined_ramp else None
                    ),
                    shared_pending_pool=shared_pending_pool,
                )
                print(
                    f"[GPU {process_rank}] training lane finished "
                    f"({finished_trainers}/{trainer_count}); "
                    f"last={last_trainer}, active_services="
                    f"{list(active_services)}",
                    flush=True,
                )
                if defer_post_training_service:
                    print(
                        f"[GPU {process_rank}] deferring service startup to "
                        "a fresh post-training process",
                        flush=True,
                    )
                    return
                # Early conversions must start serving immediately. Waiting
                # for the final barrier here deadlocks a bounded adapter queue.
                if adaptive_service_index is None:
                    adaptive_service_index = wait_for_post_training_service()
                    if adaptive_service_index is None:
                        return

            if adaptive_ramp and adaptive_service_index is None:
                # A fast trainer can exhaust the queue before another worker
                # publishes the threshold adapter (or all training ends below
                # the threshold). Keep that process alive so every promised
                # future service can still be claimed.
                while not ramp_ready_event.wait(timeout=0.5):
                    if abort_event.is_set() or time.time() >= end_time:
                        return
                from arc_vllm_router import activate_inactive_service

                (
                    adaptive_service_index,
                    service_loads,
                    active_services,
                ) = activate_inactive_service(
                    service_outstanding,
                    service_active,
                    service_load_lock,
                )
                if adaptive_service_index is not None:
                    print(
                        f"[GPU {process_rank}] claimed adaptive service "
                        f"{adaptive_service_index} after training queue "
                        f"exhaustion; outstanding={list(service_loads)}, "
                        f"active={list(active_services)}",
                        flush=True,
                    )

            if adaptive_service_index is None:
                return

            # ``worker`` returned, so its model and Trainer locals are no
            # longer reachable. Release cached allocator blocks before the
            # inference engine child reserves this same physical GPU.
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            run_service(
                int(adaptive_service_index),
                process_rank,
                gpu_already_selected=True,
            )
        except BaseException:
            # Any trainer or converted-service failure must wake siblings so
            # the original exception propagates instead of leaving the
            # pipeline waiting indefinitely.
            abort_event.set()
            raise
        return

    service_index = process_rank - trainer_count
    if not 0 <= service_index < vllm_gpu_count:
        raise RuntimeError(
            f"Invalid initial {service_backend} service index "
            f"{service_index} for {vllm_gpu_count} service(s)"
        )
    gpu_index = initial_service_gpu_indices[service_index]
    try:
        run_service(
            service_index,
            gpu_index,
            gpu_already_selected=False,
        )
    except BaseException:
        abort_event.set()
        raise


def enqueue_tasks(task_queue, selected_keys, worker_count):
    """Fill a shared FIFO queue and add one stop marker per GPU worker."""
    for key in selected_keys:
        task_queue.put(key)
    for _ in range(worker_count):
        task_queue.put(None)


def distributed_worker(
    rank,
    selected_key,
    import_lock,
    end_time,
    master_port,
    keep_training_state,
    worker_options,
):
    clear_distributed_env()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_default_device("cpu")

    # Import before process-group initialization. This preserves the original
    # serialized Unsloth patching without allowing import-time collectives to
    # wait on ranks that are blocked by the import lock.
    with import_lock:
        from arc_solver import worker

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(MAX_GPU_COUNT)
    os.environ["LOCAL_WORLD_SIZE"] = str(MAX_GPU_COUNT)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=MAX_GPU_COUNT,
        device_id=device,
    )

    try:
        dist.barrier(device_ids=[rank])

        task_queue = Queue()
        task_queue.put(selected_key)
        task_queue.put(None)

        print(f"[DDP rank {rank}/GPU {rank}] start task {selected_key}")
        worker(
            rank,
            task_queue,
            end_time,
            distributed=True,
            keep_training_state=keep_training_state,
            **worker_options,
        )
        print(f"[DDP rank {rank}/GPU {rank}] done task {selected_key}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    with ExitStack() as runtime_cleanup:
        return _main(runtime_cleanup)


def _main(runtime_cleanup):
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    add_early_routing_arguments(parser)
    parser.add_argument(
        "--run-status-file",
        default=os.getenv("ARC_RUN_STATUS_PATH"),
        help=(
            "Atomically record running, starter_success, out_of_memory, or "
            "exception as JSON for a private-rerun notebook bridge "
            "(environment: ARC_RUN_STATUS_PATH)."
        ),
    )
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument(
        "--max-seq-length",
        type=positive_int,
        default=os.getenv("ARC_MAX_SEQ_LENGTH", "8192"),
        help=(
            "Shared task-training and inference context limit. vLLM preserves "
            "complete inference prompts, skips views with no decode room, and "
            "bounds DFS by the remaining context; reserved engines use this "
            "same model context (default: 8192; environment: "
            "ARC_MAX_SEQ_LENGTH)."
        ),
    )
    parser.add_argument(
        "--fill-train-to-tokens",
        type=positive_int,
        default=os.getenv("ARC_FILL_TRAIN_TO_TOKENS"),
        metavar="TOKENS",
        help=(
            "stress-test mode: append deterministic augmented training pairs "
            "to each selected task until its formatted training prefix is as "
            "close as possible to TOKENS without crossing it. The generated "
            "challenge is shared by training and inference (environment: "
            "ARC_FILL_TRAIN_TO_TOKENS)."
        ),
    )
    parser.add_argument(
        "--single-task-gpus",
        type=int,
        choices=(1, MAX_GPU_COUNT),
        default=1,
        help="Number of GPUs for a single selected task (default: 1)",
    )
    parser.add_argument(
        "--min-train-pairs", type=int,
        default=os.getenv("ARC_MIN_TRAIN_PAIRS", "0"), metavar="PAIRS",
        help=("Experimental minimum demonstrations per Qwen conversation, in "
              "both training and inference. Adds train-derived pairs only; "
              "does not increase rows or optimizer steps. 0 disables "
              "(environment: ARC_MIN_TRAIN_PAIRS)."),
    )
    parser.add_argument(
        "--extra-train-pair-mode", choices=("augment", "repeat"),
        default=os.getenv("ARC_EXTRA_TRAIN_PAIR_MODE", "augment"),
        help=("How --min-train-pairs adds demonstrations: independent dihedral/"
              "color augmentation (experimental; may contradict task rules), "
              "or repetition in the current view as a control."),
    )
    parser.add_argument(
        "--multi-task-gpus",
        type=int,
        choices=tuple(range(1, MAX_GPU_COUNT + 1)),
        default=None,
        help=(
            "Independent GPU workers sharing a dynamic task queue. By "
            "default, use up to 4 GPUs."
        ),
    )
    parser.add_argument(
        "--task-order",
        choices=TASK_ORDER_MODES,
        default=os.getenv("ARC_TASK_ORDER", "input"),
        help=(
            "Task queue order: preserve challenge/selection order with "
            "'input' (default), or stably process fewer training pairs first "
            "with 'train-pairs', more pairs first with 'train-pairs-desc'. "
            "This does not reorder pairs within tasks or sort by token length "
            "(environment: ARC_TASK_ORDER)."
        ),
    )
    parser.add_argument(
        "--keep-training-state",
        action="store_true",
        help=(
            "Carry LoRA weights and optimizer moments to the next task in "
            "each GPU worker. Dynamic scheduling makes the exact task order "
            "per GPU timing-dependent. The default resets both for every task."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=(
            "adamw_torch",
            "adamw_8bit",
            "paged_adamw_8bit",
            "paged_adamw_32bit",
            "te_fused_adamw",
            "flash_adamw",
            "normuon",
            "muon32bit",
            "muon8bit",
        ),
        default=os.getenv("ARC_OPTIM", "adamw_torch"),
        help=(
            "Training optimizer shared by task SFT and GRPO. "
            "te_fused_adamw requires a compatible Transformer Engine build; "
            "flash_adamw requires FlashOptim; normuon requires DION; "
            "muon32bit/muon8bit require the pinned bitsandbytes Muon build. "
            "All Muon variants keep embedding/head tensors on AdamW "
            "(default: adamw_torch; "
            "environment: ARC_OPTIM)."
        ),
    )
    flash_adamw_master_weight_bits_default = int(
        os.getenv("ARC_FLASH_ADAMW_MASTER_WEIGHT_BITS", "32")
    )
    if flash_adamw_master_weight_bits_default not in {24, 32}:
        parser.error("ARC_FLASH_ADAMW_MASTER_WEIGHT_BITS must be 24 or 32")
    parser.add_argument(
        "--flash-adamw-master-weight-bits",
        type=int,
        choices=(24, 32),
        default=flash_adamw_master_weight_bits_default,
        help=(
            "Effective FlashAdamW master-weight precision. 32 is the "
            "accuracy-conservative default; 24 uses less VRAM (environment: "
            "ARC_FLASH_ADAMW_MASTER_WEIGHT_BITS)."
        ),
    )
    parser.add_argument(
        "--flash-adamw-quantize",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_FLASH_ADAMW_QUANTIZE", False),
        help=(
            "Quantize FlashAdamW first and second moments to 8-bit. This "
            "shared SFT/GRPO option is off by default (environment: "
            "ARC_FLASH_ADAMW_QUANTIZE=1/0)."
        ),
    )
    normuon_adjust_lr_default = os.getenv(
        "ARC_NORMUON_ADJUST_LR",
        "rms_norm",
    )
    if normuon_adjust_lr_default not in {
        "rms_norm",
        "spectral_norm",
        "none",
    }:
        parser.error("ARC_NORMUON_ADJUST_LR must be rms_norm, spectral_norm, or none")
    parser.add_argument(
        "--normuon-adjust-lr",
        choices=("rms_norm", "spectral_norm", "none"),
        default=normuon_adjust_lr_default,
        help=(
            "NorMuon matrix learning-rate scaling shared by task SFT and "
            "GRPO (default: rms_norm; environment: ARC_NORMUON_ADJUST_LR)."
        ),
    )
    parser.add_argument(
        "--normuon-momentum",
        type=float,
        default=float(os.getenv("ARC_NORMUON_MOMENTUM", "0.95")),
        help="NorMuon momentum (environment: ARC_NORMUON_MOMENTUM).",
    )
    parser.add_argument(
        "--normuon-beta2",
        type=float,
        default=float(os.getenv("ARC_NORMUON_BETA2", "0.95")),
        help="NorMuon neuron-wise beta2 (environment: ARC_NORMUON_BETA2).",
    )
    parser.add_argument(
        "--normuon-megabatch-size",
        type=int,
        default=int(os.getenv("ARC_NORMUON_MEGABATCH_SIZE", "8")),
        help=(
            "Maximum same-shape matrices per DION optimizer mega-batch; "
            "the L4-safe default is 8 and zero is unlimited (environment: "
            "ARC_NORMUON_MEGABATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--normuon-use-triton",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_NORMUON_USE_TRITON", False),
        help=(
            "Use DION's Triton Polar Express kernel. Off by default because "
            "one-epoch task runs cannot amortize shape autotuning "
            "(environment: ARC_NORMUON_USE_TRITON=1/0)."
        ),
    )
    muon_adjust_lr_default = os.getenv("ARC_MUON_ADJUST_LR", "rms_norm")
    if muon_adjust_lr_default not in {
        "rms_norm",
        "spectral_norm",
        "none",
    }:
        parser.error("ARC_MUON_ADJUST_LR must be rms_norm, spectral_norm, or none")
    parser.add_argument(
        "--muon-adjust-lr",
        choices=("rms_norm", "spectral_norm", "none"),
        default=muon_adjust_lr_default,
        help=(
            "bitsandbytes Muon matrix learning-rate scaling (environment: "
            "ARC_MUON_ADJUST_LR)."
        ),
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=float(os.getenv("ARC_MUON_MOMENTUM", "0.95")),
    )
    parser.add_argument(
        "--muon-nesterov",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_MUON_NESTEROV", True),
    )
    parser.add_argument(
        "--muon-ns-chunk-size",
        type=int,
        default=int(os.getenv("ARC_MUON_NS_CHUNK_SIZE", "1")),
        help=(
            "Maximum same-shape matrices per Newton-Schulz call. The global "
            "compatibility default is one; DiffusionGemma rank-32 task LoRA "
            "has audited short/near-12k L4 gates at eight "
            "(environment: ARC_MUON_NS_CHUNK_SIZE)."
        ),
    )
    parser.add_argument(
        "--muon-min-8bit-size",
        type=int,
        default=int(os.getenv("ARC_MUON_MIN_8BIT_SIZE", "4096")),
        help=(
            "Minimum matrix values for quantized Muon momentum (environment: "
            "ARC_MUON_MIN_8BIT_SIZE)."
        ),
    )
    parser.add_argument(
        "--muon-scalar-optimizer",
        choices=("adamw_torch", "adamw_8bit"),
        default=os.getenv("ARC_MUON_SCALAR_OPTIMIZER", "adamw_torch"),
        help=(
            "Optimizer for embeddings, lm_head, norms, and other non-Muon "
            "parameters. adamw_torch is the fair comparison default."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=positive_int,
        default=1,
        help=(
            "Number of microbatches accumulated per optimizer update "
            "(default: 1). On one GPU, use 4 to match the four-GPU DDP "
            "global batch and update count; any positive value is accepted."
        ),
    )
    te_optimizer_state_dtype_default = os.getenv(
        "ARC_TE_OPTIMIZER_STATE_DTYPE",
        "fp32",
    )
    if te_optimizer_state_dtype_default not in {"fp32", "fp16"}:
        parser.error("ARC_TE_OPTIMIZER_STATE_DTYPE must be fp32 or fp16")
    parser.add_argument(
        "--te-optimizer-state-dtype",
        choices=("fp32", "fp16"),
        default=te_optimizer_state_dtype_default,
        help=(
            "Moment-state dtype for te_fused_adamw. fp16 uses TE's "
            "scaled reduced-precision states to save VRAM (default: fp32; "
            "environment: ARC_TE_OPTIMIZER_STATE_DTYPE)."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_GRADIENT_CHECKPOINTING", True),
        help=(
            "Enable activation/gradient checkpointing (default: enabled; "
            "environment: ARC_GRADIENT_CHECKPOINTING=1/0). Use "
            "--no-gradient-checkpointing to disable it."
        ),
    )
    gradient_checkpointing_mode_default = os.getenv(
        "ARC_GRADIENT_CHECKPOINTING_MODE",
        "standard",
    )
    try:
        gradient_checkpointing_mode_default = (
            normalize_gradient_checkpointing_mode(
                gradient_checkpointing_mode_default
            )
        )
    except ValueError as error:
        parser.error(str(error))
    parser.add_argument(
        "--gradient-checkpointing-mode",
        choices=GRADIENT_CHECKPOINTING_MODES,
        default=gradient_checkpointing_mode_default,
        help=(
            "Checkpoint-input storage policy. standard keeps selected layer "
            "inputs on GPU; unsloth asynchronously offloads eligible inputs "
            "to pinned CPU RAM and reloads them for backward. Smart offload "
            "supports independent one-GPU task workers, not shared DDP "
            "(environment: ARC_GRADIENT_CHECKPOINTING_MODE)."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing-interval",
        type=positive_int,
        default=int(os.getenv("ARC_GRADIENT_CHECKPOINTING_INTERVAL", "1")),
        help=(
            "Checkpoint every Nth transformer layer when checkpointing is "
            "enabled. 1 checkpoints all layers; larger values trade memory "
            "for speed (environment: ARC_GRADIENT_CHECKPOINTING_INTERVAL)."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing-percent",
        type=parse_gradient_checkpointing_percentage,
        default=os.getenv("ARC_GRADIENT_CHECKPOINTING_PERCENT"),
        help=(
            "Checkpoint this percentage of transformer layers, distributed "
            "uniformly through the stack. Overrides --gradient-checkpointing-"
            "interval when set (environment: "
            "ARC_GRADIENT_CHECKPOINTING_PERCENT). Use 'auto' for the L4 "
            "lower-bound policy: for rank-128 Qwen3-4B SFT, 0%% below "
            "2,048 tokens, 30%% from 2,048 through 6,143, 40%% from 6,144, "
            "50%% from 7,168, and 55%% from 7,680 through the measured "
            "8,244-token formatting band. Rank-256 and Qwen3.5 use their "
            "own conservative profiles. GRPO raises the upper "
            "3,584--4,096 band to "
            "50%%, the 4,096--5,120 band to 55%%, uses 75%% through 10,240 "
            "effective tokens, and fully checkpoints above that range."
        ),
    )
    parser.add_argument(
        "--deepspeed-long-context",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_DEEPSPEED_LONG_CONTEXT", False),
        help=(
            "Experimentally use one-rank ZeRO-3 CPU parameter offload only "
            "for task views longer than --deepspeed-token-threshold. Tasks "
            "at or below the threshold retain the ordinary Unsloth trainer "
            "(environment: ARC_DEEPSPEED_LONG_CONTEXT=1/0)."
        ),
    )
    parser.add_argument(
        "--deepspeed-token-threshold",
        type=positive_int,
        default=os.getenv(
            "ARC_DEEPSPEED_TOKEN_THRESHOLD",
            str(DEFAULT_LONG_CONTEXT_TOKEN_THRESHOLD),
        ),
        help=(
            "Strict token boundary for experimental DeepSpeed routing; the "
            "default means only max_train_tokens > 8192 are offloaded "
            "(environment: ARC_DEEPSPEED_TOKEN_THRESHOLD)."
        ),
    )
    parser.add_argument(
        "--deepspeed-gradient-checkpointing-percent",
        type=float,
        default=os.getenv("ARC_DEEPSPEED_GC_PERCENT"),
        metavar="PERCENT",
        help=(
            "Checkpoint percentage inside a routed DeepSpeed task. This must "
            "be supplied explicitly because the safe value depends on the "
            "actual token length; zero is only an OOM diagnostic "
            "(environment: ARC_DEEPSPEED_GC_PERCENT)."
        ),
    )
    parser.add_argument(
        "--deepspeed-offload-optimizer",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_DEEPSPEED_OFFLOAD_OPTIMIZER", False),
        help=(
            "Also move AdamW state to CPU for routed long tasks. Disabled by "
            "default so the first comparison isolates parameter offload "
            "(environment: ARC_DEEPSPEED_OFFLOAD_OPTIMIZER=1/0)."
        ),
    )
    parser.add_argument(
        "--deepspeed-param-persistence-threshold",
        type=positive_int,
        default=os.getenv(
            "ARC_DEEPSPEED_PARAM_PERSISTENCE_THRESHOLD",
            str(DEFAULT_PARAM_PERSISTENCE_THRESHOLD),
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--deepspeed-model-persistence-threshold",
        type=positive_int,
        default=os.getenv(
            "ARC_DEEPSPEED_MODEL_PERSISTENCE_THRESHOLD",
            str(DEFAULT_MODEL_PERSISTENCE_THRESHOLD),
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--deepspeed-max-live-parameters",
        type=positive_int,
        default=os.getenv(
            "ARC_DEEPSPEED_MAX_LIVE_PARAMETERS",
            str(DEFAULT_MAX_LIVE_PARAMETERS),
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--deepspeed-prefetch-bucket-size",
        type=positive_int,
        default=os.getenv(
            "ARC_DEEPSPEED_PREFETCH_BUCKET_SIZE",
            str(DEFAULT_PREFETCH_BUCKET_SIZE),
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=float(os.getenv("ARC_TRAIN_EPOCHS", "4")),
        help="Training epochs per ARC task (environment: ARC_TRAIN_EPOCHS).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("ARC_LEARNING_RATE", "5e-5")),
        help=(
            "Task-SFT learning rate (default: 5e-5; environment: "
            "ARC_LEARNING_RATE). This does not change the separate GRPO "
            "learning rate."
        ),
    )
    parser.add_argument(
        "--sft-loss-scope",
        choices=("all", "final"),
        default=os.getenv("ARC_SFT_LOSS_SCOPE", "all").strip().lower(),
        help=(
            "Qwen task-SFT labels: all demonstration answers (default) or only "
            "the final answer, with preceding demonstrations retained as "
            "context. Task-RL always uses final-only warm-start labels. "
            "Environment: ARC_SFT_LOSS_SCOPE."
        ),
    )
    parser.add_argument(
        "--liger-kernel",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_USE_LIGER_KERNEL", False),
        help=(
            "Enable Transformers' Liger Qwen3 kernels (default: disabled; "
            "environment: ARC_USE_LIGER_KERNEL=1/0)."
        ),
    )
    parser.add_argument(
        "--lora-method",
        choices=LORA_METHODS,
        default=os.getenv("ARC_LORA_METHOD", "rslora").strip().lower(),
        help=(
            "Task-adapter parameterization: production RSLoRA, ordinary "
            "LoRA, persistent NoRA normalization, one-time NoRA "
            "initialization, or BIMI initialization (default: rslora; "
            "environment: ARC_LORA_METHOD)."
        ),
    )
    parser.add_argument(
        "--lora-init",
        type=normalize_lora_init,
        default=os.getenv("ARC_LORA_INIT", "default"),
        help=(
            "LoRA/rsLoRA initialization: default (unchanged), pissa (full SVD), "
            "or pissa_niter_N (fast SVD, e.g. pissa_niter_4). PiSSA keeps a "
            "frozen residual base during training. Export is rank 2*r on the "
            "original base, or rank r with --pissa-base-mode shared-residual. "
            "Unquantized Qwen SFT only. "
            "Environment: ARC_LORA_INIT."
        ),
    )
    parser.add_argument(
        "--pissa-base-mode",
        type=normalize_pissa_base_mode,
        default=os.getenv("ARC_PISSA_BASE_MODE", "original"),
        help=(
            "original: convert PiSSA to rank 2*r against the original base (default); "
            "shared-residual: ARC_MODEL_PATH is a checkpoint made by prepare_pissa_base.py; "
            "reset from its saved factors and serve rank r without another SVD."
        ),
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        choices=(16, 32, 64, 96, 128, 192, 256),
        default=int(os.getenv("ARC_LORA_R", "256")),
        help=(
            "Task LoRA rank. Qwen defaults to RSLoRA rank 256; compact "
            "DiffusionGemma on L4 supports 16/32 and experimental rank 64 "
            "(environment: ARC_LORA_R). vLLM rounds its engine rank capacity "
            "up to a supported size without changing the training/export rank."
        ),
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=float(os.getenv("ARC_LORA_ALPHA", "32")),
        help=(
            "Task-adapter alpha (default: 32; environment: "
            "ARC_LORA_ALPHA). Standard LoRA/NoRA experiments should set "
            "alpha equal to rank for unit scaling."
        ),
    )
    diffusion_gemma_group = parser.add_argument_group(
        "DiffusionGemma task-time training and vLLM inference"
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-delta-checkpoint",
        default=os.getenv("ARC_DIFFUSION_GEMMA_DELTA_CHECKPOINT"),
        help=(
            "Exact SFT delta archive (including .tar.gz.xx) reconstructed "
            "against ARC_MODEL_PATH before workers start "
            "(environment: ARC_DIFFUSION_GEMMA_DELTA_CHECKPOINT)."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-runtime-model-dir",
        default=os.getenv("ARC_DIFFUSION_GEMMA_RUNTIME_MODEL_DIR"),
        help=(
            "Optional persistent destination for the reconstructed complete "
            "HF checkpoint. By default starter.py uses and removes a unique "
            "/dev/shm directory."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-delta-work-dir",
        default=os.getenv("ARC_DIFFUSION_GEMMA_DELTA_WORK_DIR", "/dev/shm"),
        help="Temporary exact-delta decode directory (default: /dev/shm).",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-delta-workers",
        type=positive_int,
        default=int(os.getenv("ARC_DIFFUSION_GEMMA_DELTA_WORKERS", "1")),
        help=(
            "Bounded exact-delta shard decode workers (default: 1; "
            "environment: ARC_DIFFUSION_GEMMA_DELTA_WORKERS)."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-ream-root",
        default=os.getenv("ARC_DIFFUSION_GEMMA_REAM_ROOT", "/kaggle/working/ream"),
        help="REAM checkout used for frozen-FP8 LoRA conversion.",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-ttt-steps",
        type=positive_int,
        default=int(os.getenv("ARC_DIFFUSION_GEMMA_TTT_STEPS", "128")),
        help="Per-task optimizer steps (default: 128).",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-color-augmentations",
        type=positive_int,
        default=int(
            os.getenv("ARC_DIFFUSION_GEMMA_COLOR_AUGMENTATIONS", "16")
        ),
        help=(
            "Color permutations for each of eight geometric TTT views; 16 "
            "produces exactly 128 conversation rows."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-inference-views",
        type=positive_int,
        default=int(os.getenv("ARC_DIFFUSION_GEMMA_INFERENCE_VIEWS", "8")),
        help="Augmented denoising views per test pair (default: 8).",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-max-output-tokens",
        type=positive_int,
        default=int(
            os.getenv("ARC_DIFFUSION_GEMMA_MAX_OUTPUT_TOKENS", "1024")
        ),
        help=(
            "Diffusion canvas/output budget per vLLM request (default: 1024; "
            "supports a complete 30x30 ARC grid plus framing)."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-learning-rate",
        type=float,
        default=float(os.getenv("ARC_DIFFUSION_GEMMA_LEARNING_RATE", "5e-5")),
        help="BF16 task-LoRA learning rate (default: 5e-5).",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-optimizer",
        choices=("muon8bit", "adamw_torch"),
        default=os.getenv("ARC_DIFFUSION_GEMMA_OPTIMIZER", "muon8bit"),
        help=(
            "Task-LoRA optimizer. Both paths can offload moments; Muon8bit "
            "is the L4 memory default."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-optimizer-state-offload",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_DIFFUSION_GEMMA_OPTIMIZER_STATE_OFFLOAD", True),
        help="Keep DiffusionGemma optimizer state in host RAM (default: on).",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-optimizer-state-offload-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=env_flag(
            "ARC_DIFFUSION_GEMMA_OPTIMIZER_STATE_OFFLOAD_PIN_MEMORY", True
        ),
        help="Pin offloaded optimizer state for asynchronous copies.",
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-lora-token-chunk-size",
        type=int,
        default=int(
            os.getenv("ARC_DIFFUSION_GEMMA_LORA_TOKEN_CHUNK_SIZE", "1024")
        ),
        help=(
            "Maximum token rows in each BF16 LoRA update temporary; zero "
            "disables chunking (default: 1024)."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-lora-experts-backend",
        choices=(
            "loop",
            "te-grouped",
            "te-grouped-liger",
            "te-grouped-triton",
            "cutlass-ada-triton",
        ),
        default=os.getenv(
            "ARC_DIFFUSION_GEMMA_LORA_EXPERTS_BACKEND", "loop"
        ),
        help=(
            "Packed MoE implementation for frozen-FP8 task LoRA. 'loop' is "
            "the compatibility fallback; 'te-grouped' batches frozen bases "
            "with Transformer Engine and BF16 adapters with grouped_mm. "
            "'te-grouped-liger' additionally fuses the exact tanh-GELU and "
            "expert activation multiply. 'te-grouped-triton' also replaces "
            "the per-expert BF16 adapter GEMMs with an SM-sized persistent "
            "GPU-offset-native rank-32 kernel. 'cutlass-ada-triton' retains "
            "those adapter/activation kernels and replaces Transformer "
            "Engine's frozen expert bases with native SM89 grouped-FP8 "
            "forward/dgrad kernels. The "
            "grouped path has short/12k training gates but remains opt-in "
            "until the representative score gate."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-pair-fsdp",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_DIFFUSION_GEMMA_PAIR_FSDP", False),
        help=(
            "Experimental two-trainer topology: each GPU trains an "
            "independent task LoRA while FSDP shards only the shared frozen "
            "CUTLASS FP8 expert base across the pair. Requires two trainer "
            "and two persistent vLLM GPUs, an even task count, and the "
            "cutlass-ada-triton expert backend."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-expert-parallel-size",
        type=int,
        choices=(1, 2, 3),
        default=int(os.getenv("ARC_DIFFUSION_GEMMA_EXPERT_PARALLEL_SIZE", "1")),
        help=(
            "Experimental activation-dispatch MoE expert parallelism. Each "
            "rank trains one independent task and permanently owns a balanced "
            "expert shard plus every lane's matching expert-LoRA shard. Use 3 "
            "with three trainer GPUs and one persistent vLLM GPU (default: 1)."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-decoder-checkpointing-percentage",
        type=parse_gradient_checkpointing_percentage,
        default=os.getenv(
            "ARC_DIFFUSION_GEMMA_DECODER_CHECKPOINTING_PERCENTAGE", "auto"
        ),
        help=(
            "Diffusion decoder layers checkpointed during L4 TTT, or 'auto' "
            "for rank-aware L4 token-length tiers (default: auto). Rank 32 "
            "is required for the measured near-12k training envelope."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-encoder-checkpointing-percentage",
        type=parse_gradient_checkpointing_percentage,
        default=os.getenv(
            "ARC_DIFFUSION_GEMMA_ENCODER_CHECKPOINTING_PERCENTAGE", "auto"
        ),
        help=(
            "Causal encoder layers checkpointed during L4 TTT, or 'auto' "
            "for rank-aware L4 token-length tiers (default: auto). Rank 32 "
            "uses pinned-CPU checkpoint inputs for the longest tier."
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-kernel-backend",
        choices=("torch", "liger-gelu", "liger-rms", "liger-rms-cutedsl", "liger"),
        default=os.getenv("ARC_DIFFUSION_GEMMA_KERNEL_BACKEND", "liger-rms"),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-encoder-attention-backend",
        choices=("sdpa", "flash-attention-2", "hybrid-flex"),
        default=os.getenv(
            "ARC_DIFFUSION_GEMMA_ENCODER_ATTENTION_BACKEND", "hybrid-flex"
        ),
    )
    diffusion_gemma_group.add_argument(
        "--diffusion-gemma-decoder-attention-backend",
        choices=("sdpa", "flash-attention-2"),
        default=os.getenv(
            "ARC_DIFFUSION_GEMMA_DECODER_ATTENTION_BACKEND", "flash-attention-2"
        ),
    )
    parser.add_argument(
        "--train-embed-tokens",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_TRAIN_EMBED_TOKENS", True),
        help=(
            "Train the full input embedding table (default: enabled; "
            "environment: ARC_TRAIN_EMBED_TOKENS=1/0). Disabling this is an "
            "experimental memory and throughput optimization."
        ),
    )
    parser.add_argument(
        "--recompute-mlp-activations",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_RECOMPUTE_MLP_ACTIVATIONS", False),
        help=(
            "Recompute fused LoRA MLP gate/up activations during backward "
            "instead of storing them (default: disabled; environment: "
            "ARC_RECOMPUTE_MLP_ACTIVATIONS=1/0)."
        ),
    )
    parser.add_argument(
        "--recompute-attention-activations",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_RECOMPUTE_ATTENTION_ACTIVATIONS", False),
        help=(
            "Checkpoint only the attention sublayer during training "
            "(default: disabled; environment: "
            "ARC_RECOMPUTE_ATTENTION_ACTIVATIONS=1/0)."
        ),
    )
    parser.add_argument(
        "--hybrid-recompute-mlp-activations",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_HYBRID_RECOMPUTE_MLP_ACTIVATIONS", False),
        help=(
            "On layers not selected for full gradient checkpointing, use "
            "the fused LoRA MLP's exact internal gate/up recomputation. "
            "This experimental mode can be combined with percentage GC "
            "(environment: ARC_HYBRID_RECOMPUTE_MLP_ACTIVATIONS=1/0)."
        ),
    )
    parser.add_argument(
        "--hybrid-recompute-attention-activations",
        action=argparse.BooleanOptionalAction,
        default=env_flag(
            "ARC_HYBRID_RECOMPUTE_ATTENTION_ACTIVATIONS",
            False,
        ),
        help=(
            "Checkpoint attention only on layers not selected for full "
            "gradient checkpointing (environment: "
            "ARC_HYBRID_RECOMPUTE_ATTENTION_ACTIVATIONS=1/0)."
        ),
    )
    parser.add_argument(
        "--hybrid-attention-recompute-percentage",
        type=float,
        default=float(
            os.getenv("ARC_HYBRID_ATTENTION_RECOMPUTE_PERCENTAGE", "100")
        ),
        help=(
            "Checkpoint attention on this percentage of decoder layers that "
            "are not already fully checkpointed when hybrid attention "
            "recomputation is enabled (default: 100; environment: "
            "ARC_HYBRID_ATTENTION_RECOMPUTE_PERCENTAGE). This lets the MLP "
            "use exact internal recomputation on every non-full layer while "
            "retaining selected attention activations on the remaining "
            "layers."
        ),
    )
    parser.add_argument(
        "--fused-qknorm-rope",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_FUSED_QKNORM_ROPE", False),
        help=(
            "Fuse frozen Qwen3 Q/K RMSNorm with rotary embedding while "
            "preserving the existing BF16 rounding boundaries. This "
            "experimental training path is disabled by default "
            "(environment: ARC_FUSED_QKNORM_ROPE=1/0)."
        ),
    )
    parser.add_argument(
        "--attention-checkpoint-input-offload",
        action=argparse.BooleanOptionalAction,
        default=env_flag(
            "ARC_ATTENTION_CHECKPOINT_INPUT_OFFLOAD",
            False,
        ),
        help=(
            "Keep only attention-only checkpoint inputs in pinned CPU RAM. "
            "Standard mode uses independent saved tensors; Unsloth mode uses "
            "its corrected asynchronous double buffer and requires zero "
            "full-layer checkpoints. Both exact paths require the reentrant "
            "engine (environment: "
            "ARC_ATTENTION_CHECKPOINT_INPUT_OFFLOAD=1/0)."
        ),
    )
    parser.add_argument(
        "--fp8-lora",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_FP8_LORA", False),
        help=(
            "Experimental Qwen3 LoRA/RSLoRA FP8 forward/backward GEMMs; "
            "AdamW master weights, gradients and exported adapters remain BF16. "
            "Requires --fp8-frozen-linear and --fp8-dynamic-backward. "
            "vLLM base quantization is separately --vllm-quantization fp8 "
            "(environment: ARC_FP8_LORA=1/0)."
        ),
    )
    parser.add_argument(
        "--fp8-frozen-linear",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_FP8_FROZEN_LINEAR", False),
        help=(
            "Quantize frozen PEFT base projections to rowwise TE W8A8 FP8 "
            "while retaining BF16 LoRA parameters and gradients "
            "(environment: ARC_FP8_FROZEN_LINEAR=1/0)."
        ),
    )
    parser.add_argument(
        "--fp8-dynamic-backward",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_FP8_DYNAMIC_BACKWARD", True),
        help=(
            "Materialize each FP8 transpose only for its dgrad GEMM instead "
            "of retaining a second FP8 weight copy (environment: "
            "ARC_FP8_DYNAMIC_BACKWARD=1/0)."
        ),
    )
    parser.add_argument(
        "--fp8-dynamic-backward-split-accumulator",
        action=argparse.BooleanOptionalAction,
        default=env_flag(
            "ARC_FP8_DYNAMIC_BACKWARD_SPLIT_ACCUMULATOR",
            False,
        ),
        help=(
            "Use TE's more accurate but slower split accumulator for FP8 "
            "dgrad (environment: "
            "ARC_FP8_DYNAMIC_BACKWARD_SPLIT_ACCUMULATOR=1/0)."
        ),
    )
    parser.add_argument(
        "--fp8-min-train-tokens",
        type=int,
        default=int(os.getenv("ARC_FP8_MIN_TRAIN_TOKENS", "0")),
        help=(
            "Apply frozen-linear FP8 only when the longest task row reaches "
            "this token count; zero enables every task (environment: "
            "ARC_FP8_MIN_TRAIN_TOKENS)."
        ),
    )
    parser.add_argument(
        "--auto-activation-recompute",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_AUTO_ACTIVATION_RECOMPUTE", False),
        help=(
            "Select no recomputation, MLP recomputation, or MLP+attention "
            "recomputation from each task's token length (default: disabled; "
            "environment: ARC_AUTO_ACTIVATION_RECOMPUTE=1/0)."
        ),
    )
    parser.add_argument(
        "--mlp-recompute-token-threshold",
        type=int,
        default=int(os.getenv("ARC_MLP_RECOMPUTE_TOKEN_THRESHOLD", "3000")),
        help=(
            "Auto mode starts MLP recomputation above this token count (default: 3000)."
        ),
    )
    parser.add_argument(
        "--attention-recompute-token-threshold",
        type=int,
        default=int(os.getenv("ARC_ATTENTION_RECOMPUTE_TOKEN_THRESHOLD", "4096")),
        help=(
            "Auto mode also recomputes attention above this token count "
            "(default: 4096)."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=float(os.getenv("ARC_MAX_GRAD_NORM", "1")),
        help=(
            "Gradient clipping norm; use 0 to disable clipping "
            "(default: 1; environment: ARC_MAX_GRAD_NORM)."
        ),
    )
    parser.add_argument(
        "--torch-compile-training",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_TORCH_COMPILE_TRAINING", False),
        help=(
            "Compile the training model with TorchInductor reduce-overhead "
            "mode (default: disabled; environment: "
            "ARC_TORCH_COMPILE_TRAINING=1/0)."
        ),
    )
    parser.add_argument(
        "--bidirectional-prompt-attention",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_BIDIRECTIONAL_PROMPT_ATTENTION", False),
        help=(
            "Use ARC intermittent attention: every user/query run is "
            "bidirectional through its own end and assistant/answer runs "
            "remain causal. Requires the patched Unsloth package and vLLM "
            "for inference (default: disabled; environment: "
            "ARC_BIDIRECTIONAL_PROMPT_ATTENTION=1/0)."
        ),
    )
    parser.add_argument(
        "--bidirectional-vllm-compile",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_BIDIRECTIONAL_VLLM_COMPILE", False),
        help=(
            "Compile the bidirectional vLLM model and use full decode plus "
            "piecewise CUDA graphs. This is opt-in because it helps long "
            "rollouts but was slower for rank-256 exact one-token DFS on "
            "SM120 (default: disabled; environment: "
            "ARC_BIDIRECTIONAL_VLLM_COMPILE=1/0)."
        ),
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Skip inference after each task's LoRA training; intended for "
            "throughput benchmarks."
        ),
    )
    parser.add_argument(
        "--task-rl",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_TASK_RL", False),
        help=(
            "Replace the normal task adaptation schedule with a balanced "
            "final-answer-only SFT warm start followed by native Unsloth "
            "GRPO (experimental and disabled by default; environment: "
            "ARC_TASK_RL=1/0)."
        ),
    )
    parser.add_argument(
        "--rl-sft-steps",
        type=positive_int,
        default=int(os.getenv("ARC_RL_SFT_STEPS", "32")),
        help=(
            "Balanced final-answer-only SFT optimizer steps before GRPO "
            "(default: 32; environment: ARC_RL_SFT_STEPS)."
        ),
    )
    parser.add_argument(
        "--rl-max-steps",
        type=int,
        default=int(os.getenv("ARC_RL_MAX_STEPS", "8")),
        help="GRPO optimizer steps per task (default: 8).",
    )
    parser.add_argument(
        "--rl-num-generations",
        type=positive_int,
        default=int(os.getenv("ARC_RL_NUM_GENERATIONS", "2")),
        help=(
            "Completions sampled per GRPO prompt. Two is the native L4 "
            "correctness-smoke default; eight is intended for the external "
            "vLLM backend; sixteen is the higher-diversity throughput "
            "experiment. Any value of at least two is accepted when it also "
            "satisfies the backend and micro-batch constraints "
            "(environment: ARC_RL_NUM_GENERATIONS)."
        ),
    )
    parser.add_argument(
        "--rl-micro-batch-size",
        type=positive_int,
        default=int(os.getenv("ARC_RL_MICRO_BATCH_SIZE", "2")),
        help=(
            "GRPO sequences processed together on the training GPU. The full "
            "generation group is accumulated into one optimizer update "
            "(default: 2; environment: ARC_RL_MICRO_BATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--rl-auto-micro-batch",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_RL_AUTO_MICRO_BATCH", True),
        help=(
            "Reduce the GRPO micro-batch to its largest safe divisor when "
            "prompt plus completion would exceed the optimizer-aware L4 "
            "physical workload: 4,096 tokens for resident state or 8,192 "
            "for paged state. The generation group and optimizer batch are "
            "unchanged (default: enabled; environment: "
            "ARC_RL_AUTO_MICRO_BATCH=1/0)."
        ),
    )
    parser.add_argument(
        "--rl-rollout-backend",
        choices=("native", "vllm"),
        default=os.getenv("ARC_RL_ROLLOUT_BACKEND", "native"),
        help=(
            "GRPO generation runtime. vllm publishes the exact current LoRA "
            "to a persistent service on a reserved GPU; native remains the "
            "default (environment: ARC_RL_ROLLOUT_BACKEND)."
        ),
    )
    parser.add_argument(
        "--rl-augmentations",
        type=positive_int,
        default=int(os.getenv("ARC_RL_AUGMENTATIONS", "2")),
        help="Color augmentations for leave-one-out GRPO prompts (default: 2).",
    )
    parser.add_argument(
        "--rl-max-prompt-tokens",
        type=int,
        default=int(os.getenv("ARC_RL_MAX_PROMPT_TOKENS", "2048")),
        help=(
            "Maximum GRPO prompt length after dropping old "
            "demonstrations (default: 2048; environment: "
            "ARC_RL_MAX_PROMPT_TOKENS)."
        ),
    )
    parser.add_argument(
        "--rl-reward-min-cell-accuracy",
        type=float,
        default=float(os.getenv("ARC_RL_REWARD_MIN_CELL_ACCURACY", "0.7")),
        help=(
            "Correctly shaped predictions below this cell accuracy receive "
            "zero reward (default: 0.7; environment: "
            "ARC_RL_REWARD_MIN_CELL_ACCURACY)."
        ),
    )
    parser.add_argument(
        "--rl-exact-bonus",
        type=float,
        default=float(os.getenv("ARC_RL_EXACT_BONUS", "0")),
        help=(
            "Optional extra reward for an exactly correct grid. Zero keeps "
            "the requested reward range at [0, 1] (default: 0)."
        ),
    )
    parser.add_argument(
        "--rl-temperature",
        type=float,
        default=float(os.getenv("ARC_RL_TEMPERATURE", "0.8")),
        help="Native GRPO sampling temperature (default: 0.8).",
    )
    parser.add_argument(
        "--rl-learning-rate",
        type=float,
        default=float(os.getenv("ARC_RL_LEARNING_RATE", "5e-6")),
        help="GRPO learning rate after the SFT warm start (default: 5e-6).",
    )
    parser.add_argument(
        "--predict-train",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_PREDICT_TRAIN", False),
        help=(
            "After task-time training, also infer every demonstration as a "
            "leave-one-out pseudo-test. Diagnostic candidates are stored "
            "separately and never enter submission.json (default: disabled; "
            "environment: ARC_PREDICT_TRAIN=1/0)."
        ),
    )
    parser.add_argument(
        "--report-task-accuracy",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_REPORT_TASK_ACCURACY", False),
        help=(
            "After each Qwen vLLM task completes, print pass@2 accuracy and "
            "oracle coverage using available validation labels. Diagnostics "
            "only; no test-label feedback to training/search/selection. Missing "
            "labels/private reruns are skipped. Environment: ARC_REPORT_TASK_ACCURACY."
        ),
    )
    parser.add_argument(
        "--report-output-nll",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_REPORT_OUTPUT_NLL", False),
        help=(
            "After decoding, teacher-force every unique canonical test "
            "candidate, the [[0]] submission fallback, and available ground "
            "truth. Store total/per-token NLL sidecars for submit.py "
            "--visualize (default: disabled; environment: "
            "ARC_REPORT_OUTPUT_NLL=1/0)."
        ),
    )
    parser.add_argument(
        "--inference-backend",
        choices=("unsloth", "vllm", "sglang"),
        default=os.getenv("ARC_INFERENCE_BACKEND", "unsloth"),
        help=(
            "Inference runtime. vllm and sglang reserve the configured "
            "highest-numbered GPU(s) and are experimental; the default "
            "remains unsloth "
            "(environment: "
            "ARC_INFERENCE_BACKEND)."
        ),
    )
    parser.add_argument(
        "--vllm-gpus",
        type=int,
        choices=tuple(range(0, MAX_GPU_COUNT)),
        default=int(os.getenv("ARC_VLLM_GPUS", "1")),
        help=(
            "Number of independent TP=1 vLLM services on the highest-numbered "
            "GPUs (default: 1; environment: ARC_VLLM_GPUS)."
        ),
    )
    parser.add_argument(
        "--vllm-max-loras",
        type=positive_int,
        default=int(os.getenv("ARC_VLLM_MAX_LORAS", "4")),
        help=(
            "Maximum adapters resident in each vLLM service. Any positive "
            "value is accepted, but larger capacities consume more GPU "
            "memory (default: 4; environment: ARC_VLLM_MAX_LORAS)."
        ),
    )
    parser.add_argument(
        "--vllm-auto-recover",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_AUTO_RECOVER", False),
        help=(
            "Restart only a dead/poisoned vLLM EngineCore and replay its "
            "unfinished task adapters from retained staging artifacts. This "
            "is opt-in and bounded (default: disabled; environment: "
            "ARC_VLLM_AUTO_RECOVER=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-max-adapter-recoveries",
        type=int,
        default=int(os.getenv("ARC_VLLM_MAX_ADAPTER_RECOVERIES", "3")),
        help=(
            "Maximum fatal EngineCore recoveries attempted per adapter. A "
            "later incident beyond this limit drops only that adapter while "
            "the replacement service continues other work (default: 3; "
            "environment: ARC_VLLM_MAX_ADAPTER_RECOVERIES)."
        ),
    )
    parser.add_argument(
        "--vllm-adapter-timeout-seconds",
        type=float,
        default=float(os.getenv("ARC_VLLM_ADAPTER_TIMEOUT_SECONDS", "3600")),
        help=(
            "Hard wall-clock watchdog for each admitted final-inference "
            "adapter. The first timeout restarts its vLLM service and "
            "replays all unfinished resident adapters; a second timeout "
            "drops only the repeatedly hung adapter. This retry budget is "
            "separate from fatal EngineCore recovery (default: 3600; "
            "environment: ARC_VLLM_ADAPTER_TIMEOUT_SECONDS)."
        ),
    )
    parser.add_argument(
        "--vllm-startup-timeout-seconds",
        type=float,
        default=float(os.getenv("ARC_VLLM_STARTUP_TIMEOUT_SECONDS", "600")),
        help=(
            "Maximum wall time for one vLLM EngineCore construction after "
            "acquiring the serialized startup lock. A timed-out initial "
            "service retries without permanently holding that lock "
            "(default: 600; environment: "
            "ARC_VLLM_STARTUP_TIMEOUT_SECONDS)."
        ),
    )
    parser.add_argument(
        "--vllm-startup-attempts",
        type=int,
        default=int(os.getenv("ARC_VLLM_STARTUP_ATTEMPTS", "2")),
        help=(
            "Maximum EngineCore construction attempts for an initial or "
            "replacement vLLM service (default: 2; environment: "
            "ARC_VLLM_STARTUP_ATTEMPTS)."
        ),
    )
    parser.add_argument(
        "--vllm-restart-backoff-seconds",
        type=float,
        default=float(os.getenv("ARC_VLLM_RESTART_BACKOFF_SECONDS", "3")),
        help=(
            "Delay after shutting down a failed EngineCore and before loading "
            "its replacement (default: 3 seconds; environment: "
            "ARC_VLLM_RESTART_BACKOFF_SECONDS)."
        ),
    )
    parser.add_argument(
        "--sglang-gpus",
        type=int,
        choices=tuple(range(0, MAX_GPU_COUNT)),
        default=int(os.getenv("ARC_SGLANG_GPUS", "1")),
        help=(
            "Number of independent TP=1 SGLang services on the "
            "highest-numbered GPUs (default: 1; environment: "
            "ARC_SGLANG_GPUS)."
        ),
    )
    parser.add_argument(
        "--inference-ramp-after-adapters",
        type=int,
        default=int(os.getenv("ARC_INFERENCE_RAMP_AFTER_ADAPTERS", "0")),
        help=(
            "After this many adapters are published, let trainer GPUs "
            "convert into additional inference services. If training finishes "
            "before the threshold (including a threshold above the selected "
            "task count), start the ramp at training completion. Existing "
            "inference services continue normally. Zero disables the "
            "adaptive ramp (environment: "
            "ARC_INFERENCE_RAMP_AFTER_ADAPTERS)."
        ),
    )
    parser.add_argument(
        "--inference-ramp-gpus",
        type=int,
        choices=tuple(range(0, MAX_GPU_COUNT)),
        default=int(os.getenv("ARC_INFERENCE_RAMP_GPUS", "0")),
        help=(
            "Number of trainer GPUs that stop after their current adapter "
            "and become TP=1 inference services once the ramp threshold is "
            "reached or all training finishes. Zero disables the ramp (environment: "
            "ARC_INFERENCE_RAMP_GPUS)."
        ),
    )
    parser.add_argument(
        "--vllm-scale-after-training",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_SCALE_AFTER_TRAINING", False),
        help=(
            "After every training task finishes, redistribute adapters still "
            "queued behind the initial/threshold-ramped vLLM services and "
            "convert the remaining trainer GPUs into additional TP=1 "
            "services. Compatible with --inference-ramp-after-adapters and "
            "--inference-ramp-gpus. A converted GPU "
            "skips engine startup when no backlog was assigned to it. This "
            "never stops training early and is disabled by default "
            "(environment: ARC_VLLM_SCALE_AFTER_TRAINING=1/0)."
        ),
    )
    parser.add_argument(
        "--sglang-max-loaded-loras",
        type=int,
        default=int(os.getenv("ARC_SGLANG_MAX_LOADED_LORAS", "4")),
        help=(
            "Maximum adapters registered in CPU memory per SGLang service "
            "(default: 4; environment: ARC_SGLANG_MAX_LOADED_LORAS)."
        ),
    )
    parser.add_argument(
        "--sglang-max-loras-per-batch",
        type=int,
        default=int(os.getenv("ARC_SGLANG_MAX_LORAS_PER_BATCH", "2")),
        help=(
            "Maximum distinct adapters in one SGLang GPU scheduler batch. "
            "This is independent of the CPU loaded-adapter cap (default: 2; "
            "environment: ARC_SGLANG_MAX_LORAS_PER_BATCH)."
        ),
    )
    parser.add_argument(
        "--sglang-lora-backend",
        choices=("csgmv", "triton"),
        default=os.getenv("ARC_SGLANG_LORA_BACKEND", "csgmv"),
        help=(
            "SGLang multi-LoRA kernel backend (default: csgmv; environment: "
            "ARC_SGLANG_LORA_BACKEND)."
        ),
    )
    parser.add_argument(
        "--sglang-overlap-loading",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_SGLANG_OVERLAP_LOADING", False),
        help=(
            "Overlap SGLang adapter H2D loading with inference. Disabled by "
            "default until rank-256 Kaggle validation (environment: "
            "ARC_SGLANG_OVERLAP_LOADING=1/0)."
        ),
    )
    parser.add_argument(
        "--sglang-deterministic-inference",
        "--enable-deterministic-inference",
        dest="sglang_deterministic_inference",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_SGLANG_DETERMINISTIC_INFERENCE", False),
        help=(
            "Enable SGLang deterministic inference with batch-invariant "
            "kernels. This is experimental and disabled by default "
            "(environment: ARC_SGLANG_DETERMINISTIC_INFERENCE=1/0)."
        ),
    )
    parser.add_argument(
        "--sglang-leaf-rollout",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_SGLANG_LEAF_ROLLOUT", False),
        help=(
            "Generate several unmasked greedy DFS tokens per SGLang request "
            "while retaining raw ARC-token logprobs. With --dfs-mode auto, "
            "this selects recursive DFS so the rollout can apply. "
            "Experimental and disabled by default (environment: "
            "ARC_SGLANG_LEAF_ROLLOUT=1/0)."
        ),
    )
    parser.add_argument(
        "--sglang-rollout-max-tokens",
        type=int,
        default=int(os.getenv("ARC_SGLANG_ROLLOUT_MAX_TOKENS", "32")),
        help=(
            "Maximum tokens in one SGLang leaf-rollout request (default: "
            "32; environment: ARC_SGLANG_ROLLOUT_MAX_TOKENS)."
        ),
    )
    parser.add_argument(
        "--sglang-mem-fraction-static",
        type=float,
        default=float(os.getenv("ARC_SGLANG_MEM_FRACTION_STATIC", "0")),
        help=(
            "SGLang static model/KV memory fraction. Zero uses SGLang's "
            "automatic sizing (default: 0; environment: "
            "ARC_SGLANG_MEM_FRACTION_STATIC)."
        ),
    )
    parser.add_argument(
        "--sglang-max-running-requests",
        type=int,
        default=int(os.getenv("ARC_SGLANG_MAX_RUNNING_REQUESTS", "16")),
        help=(
            "Maximum concurrent requests in each SGLang service (default: "
            "16; environment: ARC_SGLANG_MAX_RUNNING_REQUESTS)."
        ),
    )
    parser.add_argument(
        "--sglang-chunked-prefill-size",
        type=int,
        default=int(os.getenv("ARC_SGLANG_CHUNKED_PREFILL_SIZE", "0")),
        help=(
            "SGLang chunked-prefill token size. Zero uses automatic sizing "
            "(default: 0; environment: ARC_SGLANG_CHUNKED_PREFILL_SIZE)."
        ),
    )
    parser.add_argument(
        "--vllm-max-pending-adapters",
        "--inference-max-pending-adapters",
        type=int,
        default=int(os.getenv("ARC_VLLM_MAX_PENDING_ADAPTERS", "10")),
        help=(
            "Maximum adapter handoffs being exported or waiting to be loaded "
            "by vLLM. Loaded adapters do not consume this disk-buffer "
            "capacity (default: 10; environment: "
            "ARC_VLLM_MAX_PENDING_ADAPTERS)."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=float(os.getenv("ARC_VLLM_GPU_MEMORY_UTILIZATION", "0.88")),
        help=(
            "Per-service vLLM GPU memory reservation fraction (default: 0.88; "
            "environment: ARC_VLLM_GPU_MEMORY_UTILIZATION)."
        ),
    )
    parser.add_argument(
        "--vllm-quantization",
        choices=("none", "fp8"),
        default=os.getenv("ARC_VLLM_QUANTIZATION", "none"),
        help=(
            "Online vLLM base-model quantization. 'fp8' keeps LoRA adapters "
            "in their normal dtype while quantizing the frozen base "
            "(default: none; environment: ARC_VLLM_QUANTIZATION)."
        ),
    )
    parser.add_argument(
        "--vllm-kv-cache-dtype",
        type=normalize_vllm_kv_cache_dtype,
        choices=VLLM_KV_CACHE_DTYPES,
        default=os.getenv("ARC_VLLM_KV_CACHE_DTYPE", "auto"),
        help=(
            "Main-model vLLM KV-cache precision, independent of base-weight "
            "quantization and LoRA dtype. auto preserves the model dtype "
            "(BF16 for Qwen); fp8 enables FP8 KV. Does not change the early-"
            "routing probe (environment: ARC_VLLM_KV_CACHE_DTYPE)."
        ),
    )
    parser.add_argument(
        "--vllm-kv-offloading-size",
        "--kv-offloading-size",
        dest="vllm_kv_offloading_size",
        type=float,
        default=float(os.getenv("ARC_VLLM_KV_OFFLOADING_SIZE", "0")),
        help=(
            "GiB of host RAM per TP=1 vLLM service used for KV-cache "
            "offloading. Zero disables offloading (default: 0; environment: "
            "ARC_VLLM_KV_OFFLOADING_SIZE)."
        ),
    )
    parser.add_argument(
        "--vllm-kv-offloading-backend",
        "--kv-offloading-backend",
        dest="vllm_kv_offloading_backend",
        choices=("native",),
        default=os.getenv("ARC_VLLM_KV_OFFLOADING_BACKEND", "native"),
        help=(
            "KV-cache offloading backend. This pipeline currently exposes "
            "vLLM's dependency-free native CPU backend (default: native; "
            "environment: ARC_VLLM_KV_OFFLOADING_BACKEND)."
        ),
    )
    parser.add_argument(
        "--inference-adapter-storage",
        choices=("auto", "ram", "disk"),
        default=os.getenv("ARC_INFERENCE_ADAPTER_STORAGE", "auto"),
        help=(
            "Temporary trained-adapter staging location. auto fills bounded "
            "/dev/shm slots first and spills additional live artifacts into "
            "/kaggle/working, then /kaggle/tmp (shared filesystems counted once); "
            "ram and disk remain strict single-tier modes "
            "(default: auto; environment: "
            "ARC_INFERENCE_ADAPTER_STORAGE)."
        ),
    )
    parser.add_argument(
        "--inference-adapter-root",
        "--vllm-adapter-root",
        dest="vllm_adapter_root",
        default=(
            os.getenv("ARC_INFERENCE_ADAPTER_ROOT")
            or os.getenv("ARC_VLLM_ADAPTER_ROOT")
        ),
        help=(
            "Explicit parent for atomic temporary adapter generations. "
            "This overrides --inference-adapter-storage; the historical "
            "--vllm-adapter-root spelling remains an alias."
        ),
    )
    parser.add_argument(
        "--inference-ram-adapter-root",
        default=os.getenv("ARC_INFERENCE_RAM_ADAPTER_ROOT"),
        help=(
            "Optional RAM-tier parent for auto/ram staging (environment: "
            "ARC_INFERENCE_RAM_ADAPTER_ROOT)."
        ),
    )
    parser.add_argument(
        "--inference-disk-adapter-root",
        default=os.getenv("ARC_INFERENCE_DISK_ADAPTER_ROOT"),
        help=(
            "Optional disk-tier parent for auto/disk staging (environment: "
            "ARC_INFERENCE_DISK_ADAPTER_ROOT)."
        ),
    )
    parser.add_argument(
        "--inference-tmp-adapter-root",
        default=os.getenv("ARC_INFERENCE_TMP_ADAPTER_ROOT"),
        help=(
            "Optional final spill-tier parent for auto staging "
            "(default: /kaggle/tmp/arc_vllm_adapters; environment: "
            "ARC_INFERENCE_TMP_ADAPTER_ROOT)."
        ),
    )
    parser.add_argument(
        "--vllm-preserve-adapter-root",
        default=os.getenv("ARC_VLLM_PRESERVE_ADAPTER_ROOT"),
        help=(
            "Optional persistent directory that receives an independent "
            "atomic copy of every trained adapter before vLLM consumes and "
            "deletes its temporary handoff (environment: "
            "ARC_VLLM_PRESERVE_ADAPTER_ROOT)."
        ),
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_ENFORCE_EAGER", False),
        help=(
            "Disable vLLM compilation and CUDA graphs. The performance "
            "default is disabled; enable only for parity diagnosis "
            "(environment: ARC_VLLM_ENFORCE_EAGER=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_ASYNC_SCHEDULING", False),
        help=(
            "Allow vLLM to overlap two internal engine batches. ARC uses "
            "the ordinary full-vocabulary output-logprob path in this mode "
            "to avoid vLLM 0.23's staged target-ID race; the option remains "
            "disabled by default while long-run prompt scoring is validated "
            "(environment: ARC_VLLM_ASYNC_SCHEDULING=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-batch-invariant",
        action=argparse.BooleanOptionalAction,
        default=env_flag("VLLM_BATCH_INVARIANT", False),
        help=(
            "Use vLLM's beta batch-invariant CUDA kernels so chunked leaf "
            "log-probabilities match one-token replay exactly (default: "
            "disabled; requires --vllm-enforce-eager below SM90; "
            "environment: VLLM_BATCH_INVARIANT=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-targeted-logprobs-only",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_TARGETED_LOGPROBS_ONLY", False),
        help=(
            "Return exact logprobs only for the 12 ARC tokens and skip "
            "vLLM's redundant vocabulary top-k. Async scheduling safely "
            "overrides this with all 16 checkpoint tokens (default: disabled; "
            "environment: ARC_VLLM_TARGETED_LOGPROBS_ONLY=1/0)."
        ),
    )
    from arc_vllm_dflash import (
        add_dflash_arguments,
        build_dflash_speculative_config,
        dflash_options,
    )
    add_dflash_arguments(parser)
    parser.add_argument(
        "--vllm-speculative-ngram-tokens",
        type=int,
        default=int(os.getenv("ARC_VLLM_SPECULATIVE_NGRAM_TOKENS", "0")),
        help=(
            "Number of prompt n-gram draft tokens for experimental vLLM "
            "speculative decoding; zero disables it (default: 0)."
        ),
    )
    parser.add_argument(
        "--vllm-speculative-ngram-method",
        choices=("ngram", "ngram_gpu"),
        default=os.getenv("ARC_VLLM_SPECULATIVE_NGRAM_METHOD", "ngram_gpu"),
    )
    parser.add_argument(
        "--vllm-speculative-ngram-min",
        type=int,
        default=int(os.getenv("ARC_VLLM_SPECULATIVE_NGRAM_MIN", "2")),
    )
    parser.add_argument(
        "--vllm-speculative-ngram-max",
        type=int,
        default=int(os.getenv("ARC_VLLM_SPECULATIVE_NGRAM_MAX", "5")),
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=int(os.getenv("ARC_VLLM_MAX_NUM_SEQS", "16")),
        help="Maximum requests per vLLM scheduler batch (default: 16).",
    )
    parser.add_argument(
        "--vllm-kv-resident-batching",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_KV_RESIDENT_BATCHING", True),
        help=(
            "Cap each augmented decode group to fit vLLM's profiled GPU KV "
            "token capacity, preventing long prompts from evicting one "
            "another on every rollout chunk (default: enabled; environment: "
            "ARC_VLLM_KV_RESIDENT_BATCHING=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-kv-resident-fraction",
        type=normalize_vllm_kv_resident_fraction,
        default=os.getenv("ARC_VLLM_KV_RESIDENT_FRACTION", "0.95"),
        metavar="FRACTION|auto",
        help=(
            "Fraction of the profiled GPU KV-token capacity available to one "
            "decode group's prompt roots. A number in (0, 1] stays fixed; "
            "auto uses 0.95 for --vllm-max-loras 1, otherwise 0.90 divided "
            "by --vllm-max-loras (2: 0.45, 3: 0.30). Auto is based on "
            "configured slots, not current active tasks (default: 0.95; environment: "
            "ARC_VLLM_KV_RESIDENT_FRACTION)."
        ),
    )
    parser.add_argument(
        "--vllm-telemetry-dir",
        default=os.getenv("ARC_VLLM_TELEMETRY_DIR"),
        help=(
            "Optional fresh directory for one JSONL scheduler/engine trace "
            "per vLLM service (environment: ARC_VLLM_TELEMETRY_DIR)."
        ),
    )
    parser.add_argument(
        "--vllm-forensics-dir",
        default=os.getenv("ARC_VLLM_FORENSICS_DIR"),
        help=(
            "Optional directory for bounded JSONL request/response evidence "
            "when a rare vLLM failure occurs. The failure is still re-raised "
            "unchanged (environment: ARC_VLLM_FORENSICS_DIR)."
        ),
    )
    parser.add_argument(
        "--vllm-rollout-branch-window",
        type=int,
        default=int(os.getenv("ARC_VLLM_ROLLOUT_BRANCH_WINDOW", "1")),
        help=(
            "Maximum concurrent DFS leaf branches per augmented origin. "
            "One preserves the historical scheduler; larger values expose "
            "more independent work to vLLM (default: 1; environment: "
            "ARC_VLLM_ROLLOUT_BRANCH_WINDOW)."
        ),
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=int(os.getenv("ARC_VLLM_MAX_BATCHED_TOKENS", "16384")),
        help="vLLM scheduler token budget (default: 16384).",
    )
    parser.add_argument(
        "--vllm-leaf-rollout",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_LEAF_ROLLOUT", False),
        help=(
            "Generate each recursive DFS leaf in one constrained vLLM "
            "request while retaining every alternative branch from raw "
            "log-probabilities. This experimental optimization defaults to "
            "disabled (environment: ARC_VLLM_LEAF_ROLLOUT=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-rollout-max-tokens",
        type=int,
        default=int(os.getenv("ARC_VLLM_ROLLOUT_MAX_TOKENS", "32")),
        help=(
            "Maximum generated tokens returned to the CPU between DFS "
            "cutoff checks when leaf rollout is enabled (default: 32; "
            "environment: ARC_VLLM_ROLLOUT_MAX_TOKENS)."
        ),
    )
    parser.add_argument(
        "--vllm-cumulative-nll-stop",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_CUMULATIVE_NLL_STOP", False),
        help=(
            "Stop a leaf rollout inside vLLM as soon as no non-EOS ARC "
            "child survives the CPU DFS cumulative-NLL cutoff. Raw "
            "logprobs and alternative branches remain unchanged. This "
            "experimental optimization defaults to disabled (environment: "
            "ARC_VLLM_CUMULATIVE_NLL_STOP=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-prefix-score",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_VLLM_PREFIX_SCORE", False),
        help=(
            "Allow teacher-forced candidate scoring to reuse cached query "
            "prefixes. Block-aligned answer-prefix scores are reconstructed "
            "exactly; an unsafe hit is retried without cache (default: "
            "disabled; environment: ARC_VLLM_PREFIX_SCORE=1/0)."
        ),
    )
    parser.add_argument(
        "--vllm-prefix-score-tile-views",
        type=positive_int,
        default=int(os.getenv("ARC_VLLM_PREFIX_SCORE_TILE_VIEWS", "4")),
        help=(
            "Number of augmented query prefixes warmed together before "
            "rescoring the other candidates when --rescore-schedule=prefix. "
            "Four keeps long ARC prefixes inside an L4 vLLM KV cache "
            "(default: 4; environment: ARC_VLLM_PREFIX_SCORE_TILE_VIEWS)."
        ),
    )
    parser.add_argument(
        "--profile-training",
        action="store_true",
        help=(
            "Capture a short post-warmup CPU/CUDA training profile under "
            "/kaggle/working; intended for optimization work."
        ),
    )
    parser.add_argument(
        "--profile-wait-steps",
        type=positive_int,
        default=int(os.getenv("ARC_PROFILE_WAIT_STEPS", "1")),
        help="Profiler schedule wait steps (default: 1).",
    )
    parser.add_argument(
        "--profile-warmup-steps",
        type=positive_int,
        default=int(os.getenv("ARC_PROFILE_WARMUP_STEPS", "1")),
        help="Profiler schedule warmup steps (default: 1).",
    )
    parser.add_argument(
        "--profile-active-steps",
        type=positive_int,
        default=int(os.getenv("ARC_PROFILE_ACTIVE_STEPS", "4")),
        help="Profiler schedule active steps (default: 4).",
    )
    parser.add_argument(
        "--inference-timeout-seconds",
        type=float,
        default=float(os.getenv("ARC_INFERENCE_TIMEOUT_SECONDS", "1200")),
        help=(
            "Per-puzzle inference soft wall-time limit after training. With "
            "vLLM it starts when the loaded adapter becomes an active task, "
            "excludes the pending-adapter queue, and includes time sharing "
            "the GPU with other tasks. In-flight calls and queued rescoring "
            "can finish after this limit "
            "(default: 1200; environment: ARC_INFERENCE_TIMEOUT_SECONDS)."
        ),
    )
    parser.add_argument(
        "--vllm-search-limit-seconds",
        type=normalize_vllm_search_limit_seconds,
        default=os.getenv("ARC_VLLM_SEARCH_LIMIT_SECONDS", "540"),
        metavar="SECONDS",
        help=(
            "Positive wall-time limit per vLLM threshold-DFS augmentation "
            "group, shared by its branches and rollout chunks. Clipped by "
            "the remaining task timeout and global deadline; clocks keep "
            "running while other LoRAs use the GPU. Does not change native "
            "Unsloth search or shape-fallback limits "
            "(default: 540; environment: ARC_VLLM_SEARCH_LIMIT_SECONDS)."
        ),
    )
    parser.add_argument(
        "--inference-batch-size",
        type=positive_int,
        default=int(os.getenv("ARC_INFERENCE_BATCH_SIZE", "4")),
        help=(
            "Shape-compatible constrained-decode batch size. Any positive "
            "value is accepted; the legacy default is four "
            "(environment: ARC_INFERENCE_BATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--score-batch-size",
        type=positive_int,
        default=int(os.getenv("ARC_SCORE_BATCH_SIZE", "4")),
        help=(
            "Candidate rescoring batch size. Any positive value is accepted; "
            "BF16 scores can vary with padding shape. The default is four "
            "(environment: ARC_SCORE_BATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--rescore-views",
        type=positive_int,
        default=int(os.getenv("ARC_RESCORE_VIEWS", "8")),
        help=(
            "Number of deterministic geometry/color views used to rescore each "
            "unique candidate. Any positive count is accepted (default: 8; "
            "environment: ARC_RESCORE_VIEWS)."
        ),
    )
    parser.add_argument(
        "--rescore-schedule",
        choices=("candidate", "view", "prefix"),
        default=os.getenv("ARC_RESCORE_SCHEDULE", "candidate"),
        help=(
            "vLLM score-job order. 'view' interleaves candidates by view. "
            "'prefix' uses cache-sized warm/reuse barriers and is the intended "
            "schedule for --vllm-prefix-score, especially with 32 views. "
            "Direct Unsloth scoring ignores the order (default: candidate; "
            "environment: ARC_RESCORE_SCHEDULE)."
        ),
    )
    parser.add_argument(
        "--dfs-mode",
        choices=("recursive", "frontier", "auto"),
        default=os.getenv("ARC_DFS_MODE", "recursive"),
        help=(
            "Constrained search implementation. Frontier batches all live "
            "branches; auto uses recursive search for multi-output puzzles "
            "and frontier for single-output puzzles. The default is recursive "
            "(environment: ARC_DFS_MODE)."
        ),
    )
    parser.add_argument(
        "--frontier-origin-batch-size",
        type=positive_int,
        default=int(os.getenv("ARC_FRONTIER_ORIGIN_BATCH_SIZE", "4")),
        help=(
            "Maximum independent input origins combined in one frontier "
            "search. Four preserves the original prefill batch; reduce this "
            "only for extra inference headroom (default: 4; environment: "
            "ARC_FRONTIER_ORIGIN_BATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--frontier-max-live-branches",
        type=int,
        default=int(os.getenv("ARC_FRONTIER_MAX_LIVE_BRANCHES", "8")),
        help=(
            "Switch a frontier group to recursive DFS when its live batch "
            "exceeds this size. This bounds memory while retaining batched "
            "decoding for small frontiers (default: 8; environment: "
            "ARC_FRONTIER_MAX_LIVE_BRANCHES)."
        ),
    )
    parser.add_argument(
        "--frontier-max-prompt-tokens",
        type=int,
        default=int(os.getenv("ARC_FRONTIER_MAX_PROMPT_TOKENS", "3000")),
        help=(
            "In auto mode, use recursive inference above this training-prompt "
            "length to avoid rebuilding oversized paged caches (default: "
            "3000; environment: ARC_FRONTIER_MAX_PROMPT_TOKENS)."
        ),
    )
    parser.add_argument(
        "--max-beam-score",
        type=float,
        default=float(os.getenv("ARC_MAX_BEAM_SCORE", "1.6094379124341003")),
        help=(
            "Maximum cumulative constrained-decoding NLL retained by the "
            "search (default: -log(0.2); environment: ARC_MAX_BEAM_SCORE)."
        ),
    )
    parser.add_argument(
        "--dfs-cutoff-probability",
        type=max_beam_score_from_probability,
        default=os.getenv("ARC_DFS_CUTOFF_PROBABILITY"),
        metavar="P",
        help=(
            "Retain DFS paths with cumulative probability at least P. This is "
            "converted to --max-beam-score=-log(P) and takes precedence over "
            "that argument; useful values are 0.20, 0.10, and 0.07 "
            "(environment: ARC_DFS_CUTOFF_PROBABILITY)."
        ),
    )
    parser.add_argument(
        "--base-model-fallback",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_BASE_MODEL_FALLBACK", False),
        help=(
            "Qwen vLLM only: retry test outputs whose selected attempts are "
            "both [[0]], using the loaded base model without LoRA. One extra "
            "pass with the same inference budget, bounded by --end-time. "
            "Other outputs stay unchanged. Default off; environment: "
            "ARC_BASE_MODEL_FALLBACK."
        ),
    )
    parser.add_argument(
        "--zero-candidate-shape-fallback",
        action=argparse.BooleanOptionalAction,
        default=env_flag("ARC_ZERO_CANDIDATE_SHAPE_FALLBACK", False),
        help=(
            "When demonstrations share an input/output shape offset or a "
            "constant output shape and threshold decoding yields no multi-color "
            "candidate, generate a fixed-shape non-monochrome beam. Monochrome "
            "normal candidates are rejected when enabled. The default is off "
            "while this recovery path is evaluated (environment: "
            "ARC_ZERO_CANDIDATE_SHAPE_FALLBACK=1/0)."
        ),
    )
    parser.add_argument(
        "--shape-fallback-beam-width",
        type=positive_int,
        default=int(os.getenv("ARC_SHAPE_FALLBACK_BEAM_WIDTH", "4")),
        help=(
            "Beam width for the fixed-shape fallback (default: 4; "
            "environment: ARC_SHAPE_FALLBACK_BEAM_WIDTH)."
        ),
    )
    args = parser.parse_args()
    try:
        validate_early_routing_arguments(args)
        if args.early_route_metric != "off" and (
            args.inference_backend != "vllm" or args.train_only or args.task_rl
        ):
            raise ValueError("Early routing requires Qwen3-4B SFT with --inference-backend vllm")
        args.lora_init = normalize_lora_init(args.lora_init)
        validate_fp8_lora_options(
            args.fp8_lora, fp8_frozen_linear=args.fp8_frozen_linear,
            fp8_dynamic_backward=args.fp8_dynamic_backward,
            lora_method=args.lora_method, lora_init=args.lora_init,
            optimizer=args.optimizer, task_rl=args.task_rl,
            deepspeed=args.deepspeed_long_context,
            distributed=args.single_task_gpus > 1,
            keep_training_state=args.keep_training_state,
            torch_compile_training=args.torch_compile_training,
            inference_backend=args.inference_backend, train_only=args.train_only,
        )
        args.pissa_base_mode = normalize_pissa_base_mode(args.pissa_base_mode)
        lora_export_rank(args.lora_r, args.lora_init, args.pissa_base_mode)
        validate_pissa_options(
            args.lora_method, args.lora_init,
            fp8_frozen_linear=args.fp8_frozen_linear,
            quantization=args.vllm_quantization,
            task_rl=args.task_rl, deepspeed=args.deepspeed_long_context,
            distributed=args.single_task_gpus > 1,
            keep_training_state=args.keep_training_state,
            inference_backend=args.inference_backend,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.min_train_pairs < 0:
        parser.error("--min-train-pairs cannot be negative")
    if args.min_train_pairs and (args.fill_train_to_tokens or args.task_rl):
        parser.error("--min-train-pairs cannot be combined with stress filling or --task-rl")
    try:
        split_specs = runtime_specs_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.runtime_preflight_only:
        from pipeline_runtime_launcher import create_report_directory, prepare_runtimes

        report_dir = create_report_directory(args.runtime_report_dir)
        prepare_runtimes(split_specs, report_dir, check_only=True)
        print(f"Split runtime preflight complete (no training): {report_dir}", flush=True)
        return
    if args.dfs_cutoff_probability is not None:
        args.max_beam_score = args.dfs_cutoff_probability
    if args.flash_adamw_quantize and args.optimizer != "flash_adamw":
        parser.error("--flash-adamw-quantize requires --optimizer flash_adamw")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.sft_loss_scope not in ("all", "final"):
        parser.error("--sft-loss-scope must be all or final")
    if not 0.0 <= args.normuon_momentum < 1.0:
        parser.error("--normuon-momentum must be in [0, 1)")
    if not 0.0 <= args.normuon_beta2 < 1.0:
        parser.error("--normuon-beta2 must be in [0, 1)")
    if args.normuon_megabatch_size < 0:
        parser.error("--normuon-megabatch-size cannot be negative")
    if not 0.0 <= args.muon_momentum < 1.0:
        parser.error("--muon-momentum must be in [0, 1)")
    if args.muon_ns_chunk_size <= 0:
        parser.error("--muon-ns-chunk-size must be positive")
    if args.muon_min_8bit_size <= 0:
        parser.error("--muon-min-8bit-size must be positive")
    if args.rl_max_steps < 1:
        parser.error("--rl-max-steps must be positive")
    if args.rl_num_generations < 2:
        parser.error("--rl-num-generations must be at least 2")
    if args.rl_max_prompt_tokens < 1:
        parser.error("--rl-max-prompt-tokens must be positive")
    if not 0.0 <= args.rl_reward_min_cell_accuracy <= 1.0:
        parser.error("--rl-reward-min-cell-accuracy must be in [0, 1]")
    if args.rl_exact_bonus < 0.0:
        parser.error("--rl-exact-bonus must be non-negative")
    if args.rl_temperature <= 0.0:
        parser.error("--rl-temperature must be positive")
    if args.rl_learning_rate <= 0.0:
        parser.error("--rl-learning-rate must be positive")
    if args.task_rl and args.keep_training_state:
        parser.error("--task-rl cannot be combined with --keep-training-state")
    if args.task_rl and args.gradient_accumulation_steps != 1:
        parser.error("--task-rl currently requires --gradient-accumulation-steps 1")
    if args.rl_rollout_backend == "vllm" and not args.task_rl:
        parser.error("--rl-rollout-backend vllm requires --task-rl")
    if (
        args.task_rl
        and args.rl_rollout_backend == "native"
        and args.rl_num_generations > 4
    ):
        parser.error(
            "Native L4 rollout supports at most 4 generations; use "
            "--rl-rollout-backend vllm for 8 or 16"
        )
    if args.rl_num_generations % args.rl_micro_batch_size:
        parser.error("--rl-num-generations must be divisible by --rl-micro-batch-size")
    if args.shape_fallback_beam_width < 2:
        parser.error("--shape-fallback-beam-width must be at least 2")
    if args.report_output_nll and args.train_only:
        parser.error("--report-output-nll cannot be combined with --train-only")
    if args.report_task_accuracy and (args.train_only or args.inference_backend != "vllm"):
        parser.error("--report-task-accuracy currently requires Qwen vLLM inference")
    if args.base_model_fallback and (
        args.train_only or args.inference_backend != "vllm"
    ):
        parser.error("--base-model-fallback requires the Qwen vLLM inference pipeline")

    base_model_path = resolve_model_path()
    if args.vllm_draft_model and (args.inference_backend != "vllm" or args.train_only):
        parser.error("--vllm-draft-model requires vLLM inference, not --train-only")
    try:
        build_dflash_speculative_config(base_model_path, vars(args))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    from arc_pissa_shared import validate_runtime_base

    try:
        shared_pissa_manifest = validate_runtime_base(
            base_model_path, mode=args.pissa_base_mode, initialization=args.lora_init,
            method=args.lora_method, rank=args.lora_r, alpha=args.lora_alpha,
            verify_files=True,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    from pre_sft_rope_3d import read_rope_3d_config

    checkpoint_rope_3d = read_rope_3d_config(base_model_path)
    if checkpoint_rope_3d is not None:
        if split_specs is not None or args.task_rl or args.torch_compile_training:
            parser.error(
                "ARC 3D RoPE currently supports ordinary task SFT/DFS, not "
                "split execution, GRPO or --torch-compile-training"
            )
        if not args.train_only and args.inference_backend == "sglang":
            parser.error("ARC 3D RoPE requires Unsloth or the ARC vLLM executor")
        if args.inference_backend == "vllm" and not args.train_only:
            if (
                args.bidirectional_prompt_attention
                or args.vllm_async_scheduling
                or args.vllm_speculative_ngram_tokens
                or args.vllm_draft_model
            ):
                parser.error(
                    "ARC 3D RoPE vLLM currently requires causal, synchronous, "
                    "non-speculative inference"
                )
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        print(
            f"Checkpoint ARC 3D RoPE: base={checkpoint_rope_3d['base']:g} "
            "(automatically restored for task LoRA/inference)",
            flush=True,
        )
    if split_specs is not None:
        with open(os.path.join(base_model_path, "config.json")) as model_file:
            split_model_config = json.load(model_file)
        try:
            validate_split_mode(args, model_config=split_model_config)
            if model_declares_delta_mhar(base_model_path):
                raise ValueError("Split runtime does not yet support Delta-MH routed checkpoints")
        except ValueError as error:
            parser.error(str(error))
    from diffusion_gemma_pipeline import model_declares_diffusion_gemma

    diffusion_gemma_mode = model_declares_diffusion_gemma(base_model_path)
    model_family = "diffusion_gemma" if diffusion_gemma_mode else "qwen"
    if args.fp8_lora:
        with open(os.path.join(base_model_path, "config.json")) as model_file:
            fp8_model_config = json.load(model_file)
        if fp8_model_config.get("model_type") != "qwen3":
            parser.error("--fp8-lora currently supports dense Qwen3 only")
    pipeline_model_path = base_model_path
    diffusion_gemma_runtime_cleanup_path = None
    if args.diffusion_gemma_delta_checkpoint and not diffusion_gemma_mode:
        parser.error(
            "--diffusion-gemma-delta-checkpoint requires ARC_MODEL_PATH to "
            "be the exact compact DiffusionGemma base used to create that delta"
        )
    if diffusion_gemma_mode:
        if is_pissa_init(args.lora_init):
            parser.error("--lora-init PiSSA currently supports Qwen only")
        if args.base_model_fallback:
            parser.error("--base-model-fallback currently supports Qwen vLLM tasks only")
        if args.report_task_accuracy:
            parser.error("--report-task-accuracy currently supports Qwen vLLM tasks only")
        if args.sft_loss_scope != "all":
            parser.error("--sft-loss-scope final supports Qwen task SFT only")
        if args.inference_backend != "vllm" or args.train_only:
            parser.error(
                "DiffusionGemma task-time training currently requires "
                "--inference-backend vllm and cannot use --train-only"
            )
        if args.task_rl:
            parser.error("DiffusionGemma does not yet support --task-rl")
        if args.keep_training_state:
            parser.error(
                "DiffusionGemma resets an independent LoRA for every ARC task; "
                "remove --keep-training-state"
            )
        if args.lora_r not in {16, 32, 64, 128}:
            parser.error(
                "DiffusionGemma frozen-FP8 TTT on a 22GiB L4 requires "
                "--lora-r 16, 32, 64, or opt-in rank 128"
            )
        if args.lora_r > 64 and os.getenv(
            "ARC_DIFFUSION_GEMMA_CHECKPOINTING_PROFILE", "legacy"
        ).strip().lower() != "l4-exp48":
            parser.error(
                "DiffusionGemma rank 128 requires "
                "ARC_DIFFUSION_GEMMA_CHECKPOINTING_PROFILE=l4-exp48 "
                "and a compatible 48-expert model"
            )
        from diffusion_gemma_pipeline import (
            _read_config,
            validate_diffusion_gemma_checkpointing_profile,
        )

        try:
            validate_diffusion_gemma_checkpointing_profile(
                os.getenv("ARC_DIFFUSION_GEMMA_CHECKPOINTING_PROFILE", "legacy").strip().lower(),
                _read_config(base_model_path),
                args.lora_r,
                vars(args),
            )
        except ValueError as exc:
            parser.error(str(exc))
        max_l4_lora_slots = max(1, 64 // args.lora_r)
        if args.vllm_max_loras > max_l4_lora_slots:
            parser.error(
                f"DiffusionGemma rank {args.lora_r} reserves a large MoE LoRA "
                "buffer on each 22GiB L4; set --vllm-max-loras at most "
                f"{max_l4_lora_slots}"
            )
        expected_ttt_steps = 8 * args.diffusion_gemma_color_augmentations
        if args.diffusion_gemma_ttt_steps != expected_ttt_steps:
            parser.error(
                "DiffusionGemma conversation TTT uses all eight geometric "
                "views: --diffusion-gemma-ttt-steps must equal 8 * "
                "--diffusion-gemma-color-augmentations "
                f"({expected_ttt_steps})"
            )
        if (
            args.diffusion_gemma_decoder_checkpointing_percentage != "auto"
            and not 0.0
            <= args.diffusion_gemma_decoder_checkpointing_percentage
            <= 100.0
        ):
            parser.error(
                "--diffusion-gemma-decoder-checkpointing-percentage must be "
                "between 0 and 100"
            )
        if (
            args.diffusion_gemma_encoder_checkpointing_percentage != "auto"
            and not 0.0
            <= args.diffusion_gemma_encoder_checkpointing_percentage
            <= 100.0
        ):
            parser.error(
                "--diffusion-gemma-encoder-checkpointing-percentage must be "
                "between 0 and 100"
            )
        if args.diffusion_gemma_lora_token_chunk_size < 0:
            parser.error(
                "--diffusion-gemma-lora-token-chunk-size cannot be negative"
            )
        if args.diffusion_gemma_learning_rate <= 0.0:
            parser.error("--diffusion-gemma-learning-rate must be positive")
        if args.diffusion_gemma_max_output_tokens >= args.max_seq_length:
            parser.error(
                "--diffusion-gemma-max-output-tokens must leave room for the "
                "ARC prompt inside --max-seq-length"
            )
        if args.max_seq_length > 12288:
            parser.error(
                "The audited compact DiffusionGemma TTT/vLLM path supports at "
                "most --max-seq-length 12288"
            )
        if args.fill_train_to_tokens is not None:
            parser.error(
                "--fill-train-to-tokens is a Qwen formatter stress mode and "
                "is not supported by DiffusionGemma"
            )
        if args.min_train_pairs:
            parser.error("--min-train-pairs currently supports Qwen task SFT only")
        if args.report_output_nll:
            parser.error(
                "--report-output-nll is autoregressive and is not defined for "
                "DiffusionGemma's iterative denoising output"
            )
        if args.predict_train:
            parser.error(
                "--predict-train is not implemented for DiffusionGemma yet"
            )
        # These are family invariants rather than optional tuning choices.
        args.fp8_frozen_linear = True
        args.vllm_quantization = "fp8"
        print(
            "Detected compact DiffusionGemma: starter.py will use frozen-FP8 "
            "BF16-LoRA conversation TTT and native online-FP8 vLLM serving",
            flush=True,
        )

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    test_path = resolve_challenge_path(rerun_mode)

    with open(test_path, "r") as f:
        data = json.load(f)

    selected_keys = select_task_keys(data.keys())
    original_selected_keys = list(selected_keys)

    # Custom task filtering: add or change `if ...: continue` rules here.
    filtered_keys = []
    for key in selected_keys:
        task = data[key]

        # Example rule: only solve tasks with exactly 2 training examples.
        if len(task["train"]) >3:
            pass
            # print("skip", key)
            # continue

        filtered_keys.append(key)
    selected_keys = filtered_keys

    if not selected_keys:
        parser.error("No tasks remain after applying task filters")

    # Before any trainer/service allocates a model: disposable base-only probes.
    # Keep the original cohort for submission scoring; only its work queue shrinks.
    try:
        selected_keys = run_early_routing(
            args, data, selected_keys, base_model_path,
            os.getenv("ARC_INFERENCE_OUTPUT_DIR", "/kaggle/inference_outputs"),
            runtime_specs=split_specs, identity_task_keys=original_selected_keys,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not selected_keys:
        print("[early routing] All selected tasks bypass 4B; no trainers/services started. "
              "submit.py will keep zero placeholders, or the cascade will route them to stage 2.", flush=True)
        return
    selected_keys = order_task_keys(selected_keys, data, args.task_order)
    if args.task_order in {"train-pairs", "train-pairs-desc"}:
        pair_count_groups = {}
        for key in selected_keys:
            count = len(data[key]["train"])
            pair_count_groups[count] = pair_count_groups.get(count, 0) + 1
        print(
            "Task queue order: "
            f"{'descending' if args.task_order == 'train-pairs-desc' else 'ascending'} "
            "training-pair count, stable within "
            f"ties; groups={pair_count_groups}",
            flush=True,
        )

    if args.fill_train_to_tokens is not None:
        if args.fill_train_to_tokens > args.max_seq_length:
            parser.error(
                "--fill-train-to-tokens cannot exceed --max-seq-length"
            )
        from transformers import AutoTokenizer

        from arc_loader import QwenFormatter
        from arc_stress_fill import (
            fill_challenge_tasks,
            measure_augmented_context,
            write_runtime_challenge,
        )

        stress_model_path = resolve_model_path()
        stress_tokenizer = AutoTokenizer.from_pretrained(
            stress_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        stress_formatter = QwenFormatter(tokenizer=stress_tokenizer)
        data, fill_records = fill_challenge_tasks(
            data,
            selected_keys,
            stress_formatter,
            stress_tokenizer,
            args.fill_train_to_tokens,
        )
        stress_context = measure_augmented_context(
            data,
            selected_keys,
            stress_formatter,
            stress_tokenizer,
        )
        if stress_context["max_train_tokens"] > args.max_seq_length:
            parser.error(
                "Filled ARC training views require context "
                f"{stress_context['max_train_tokens']}, exceeding "
                f"--max-seq-length {args.max_seq_length}"
            )
        if (
            not args.train_only
            and stress_context["max_inference_prompt_tokens"]
            >= args.max_seq_length
        ):
            print(
                "ARC prompt-fill warning: at least one augmented inference "
                "prompt has no decode room and will be skipped without "
                "trimming demonstrations "
                f"(max_prompt={stress_context['max_inference_prompt_tokens']}, "
                f"max_seq_length={args.max_seq_length})",
                flush=True,
            )
        filled_path, fill_manifest_path = write_runtime_challenge(
            data,
            fill_records,
            stress_context,
            source_path=test_path,
            target_tokens=args.fill_train_to_tokens,
        )
        os.environ["ARC_CHALLENGE_PATH"] = filled_path
        test_path = filled_path
        added_counts = [record["added_train_pairs"] for record in fill_records]
        filled_lengths = [record["train_tokens"] for record in fill_records]
        print(
            "ARC prompt-fill stress mode: "
            f"tasks={len(fill_records)}, target={args.fill_train_to_tokens}, "
            f"added_pairs={min(added_counts)}..{max(added_counts)}, "
            f"train_tokens={min(filled_lengths)}..{max(filled_lengths)}, "
            f"max_augmented_train={stress_context['max_train_tokens']}, "
            f"max_augmented_inference_prompt="
            f"{stress_context['max_inference_prompt_tokens']}, "
            f"required_context={stress_context['required_vllm_context']}",
            flush=True,
        )
        print(
            f"ARC prompt-fill challenge: {filled_path}; "
            f"manifest: {fill_manifest_path}",
            flush=True,
        )
        del stress_formatter, stress_tokenizer

    selected_count = len(selected_keys)

    if args.task_rl and selected_count == 1 and args.single_task_gpus == MAX_GPU_COUNT:
        parser.error(
            "--task-rl currently supports one GPU per task; use --single-task-gpus 1"
        )
    if (
        args.gradient_checkpointing
        and args.gradient_checkpointing_mode == "unsloth"
        and selected_count == 1
        and args.single_task_gpus == MAX_GPU_COUNT
    ):
        parser.error(
            "--gradient-checkpointing-mode unsloth supports independent "
            "one-GPU task workers only; use --single-task-gpus 1 for the "
            "selected task"
        )

    external_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if external_world_size != 1:
        raise RuntimeError(
            "Run starter.py with Python, not torchrun. It manages its own "
            "multi-GPU processes."
        )

    use_vllm_final_inference = args.inference_backend == "vllm" and not args.train_only
    use_sglang_final_inference = (
        args.inference_backend == "sglang" and not args.train_only
    )
    use_vllm_rl_rollout = args.task_rl and args.rl_rollout_backend == "vllm"
    if (
        args.deepspeed_gradient_checkpointing_percent is not None
        and not 0.0
        <= args.deepspeed_gradient_checkpointing_percent
        <= 100.0
    ):
        parser.error(
            "--deepspeed-gradient-checkpointing-percent must be between "
            "0 and 100"
        )
    deepspeed_can_activate = (
        args.deepspeed_long_context
        and args.max_seq_length > args.deepspeed_token_threshold
    )
    if (
        deepspeed_can_activate
        and args.deepspeed_gradient_checkpointing_percent is None
    ):
        parser.error(
            "--deepspeed-long-context requires an explicit "
            "--deepspeed-gradient-checkpointing-percent; live L4 tests "
            "found 56% was the minimum at 9,743 tokens and 70% at 13,639 "
            "tokens"
        )
    try:
        validate_long_context_deepspeed_options(
            enabled=deepspeed_can_activate,
            optimizer=args.optimizer,
            distributed=(
                selected_count == 1
                and args.single_task_gpus == MAX_GPU_COUNT
            ),
            keep_training_state=args.keep_training_state,
            task_rl=args.task_rl,
            train_only=args.train_only,
            routed_inference=(
                use_vllm_final_inference or use_sglang_final_inference
            ),
            torch_compile_training=args.torch_compile_training,
            selective_activation_recompute=(
                args.auto_activation_recompute
                or args.recompute_mlp_activations
                or args.recompute_attention_activations
            ),
            fp8_frozen_linear=args.fp8_frozen_linear,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.deepspeed_long_context and not deepspeed_can_activate:
        print(
            "Long-context DeepSpeed cannot activate because "
            f"max_seq_length={args.max_seq_length} is not greater than "
            f"the token threshold {args.deepspeed_token_threshold}; all "
            "tasks will retain ordinary Unsloth training.",
            flush=True,
        )
    if args.bidirectional_prompt_attention and args.task_rl:
        parser.error(
            "--bidirectional-prompt-attention currently supports task SFT, "
            "not --task-rl"
        )
    if args.bidirectional_vllm_compile and not args.bidirectional_prompt_attention:
        parser.error(
            "--bidirectional-vllm-compile requires --bidirectional-prompt-attention"
        )
    if args.bidirectional_vllm_compile and args.train_only:
        parser.error(
            "--bidirectional-vllm-compile starts no engine under "
            "--train-only; remove the compile flag"
        )
    if args.bidirectional_vllm_compile and args.vllm_enforce_eager:
        parser.error("--bidirectional-vllm-compile conflicts with --vllm-enforce-eager")
    if (
        args.bidirectional_prompt_attention
        and not args.train_only
        and not use_vllm_final_inference
    ):
        parser.error(
            "--bidirectional-prompt-attention requires "
            "--inference-backend vllm unless --train-only is set, so "
            "training and inference cannot silently use different masks"
        )
    if args.bidirectional_prompt_attention and args.torch_compile_training:
        parser.error(
            "--bidirectional-prompt-attention is incompatible with "
            "--torch-compile-training; segmented FlashAttention uses a "
            "data-dependent run table"
        )
    if (
        args.bidirectional_prompt_attention
        and not args.vllm_batch_invariant
        and not args.train_only
    ):
        # Exact DFS mixes scheduler batch shapes. vLLM's batch-invariant
        # kernels keep raw logprobs stable across those shapes. This does not
        # promise bitwise equality between a cold full prefill and a later
        # prefix-cache resume, which can use different attention kernels.
        args.vllm_batch_invariant = True
        print(
            "ARC bidirectional prompt attention enabled vLLM batch "
            "invariance for batch-shape-stable raw logprobs",
            flush=True,
        )
    if use_sglang_final_inference and use_vllm_rl_rollout:
        parser.error(
            "SGLang final inference and vLLM GRPO rollout cannot share one "
            "reserved-service topology yet; use --rl-rollout-backend native"
        )
    service_backend = None
    if use_sglang_final_inference:
        service_backend = "sglang"
    elif use_vllm_final_inference or use_vllm_rl_rollout:
        service_backend = "vllm"
    use_inference_pipeline = service_backend is not None
    service_gpu_count = (
        args.sglang_gpus if service_backend == "sglang" else args.vllm_gpus
    )
    ramp_after_adapters = int(args.inference_ramp_after_adapters)
    ramp_gpu_count = int(args.inference_ramp_gpus)
    adaptive_inference_ramp = bool(ramp_after_adapters or ramp_gpu_count)
    post_training_scale = bool(args.vllm_scale_after_training)
    if post_training_scale and not use_vllm_final_inference:
        parser.error(
            "--vllm-scale-after-training requires final inference through "
            "--inference-backend vllm"
        )
    if post_training_scale and use_vllm_rl_rollout:
        parser.error(
            "--vllm-scale-after-training currently supports asynchronous "
            "final inference, not synchronous vLLM GRPO rollout"
        )
    try:
        validate_zero_service_adapter_capacity(
            selected_count=selected_count,
            max_pending_adapters=args.vllm_max_pending_adapters,
            post_training_scale=post_training_scale,
            initial_service_count=service_gpu_count,
            ramp_after_adapters=ramp_after_adapters,
        )
    except ValueError as error:
        parser.error(str(error))
    if adaptive_inference_ramp and not use_inference_pipeline:
        parser.error(
            "The adaptive inference ramp requires --inference-backend vllm or sglang"
        )
    if (ramp_after_adapters > 0) != (ramp_gpu_count > 0):
        parser.error(
            "--inference-ramp-after-adapters and --inference-ramp-gpus "
            "must both be positive, or both be zero"
        )
    final_service_count = (
        MAX_GPU_COUNT if post_training_scale else service_gpu_count + ramp_gpu_count
    )
    if use_inference_pipeline:
        # vLLM reads this switch during its lazy import inside the reserved
        # inference process, so set it before multiprocessing spawn.
        if service_backend == "vllm":
            os.environ["VLLM_BATCH_INVARIANT"] = (
                "1" if args.vllm_batch_invariant else "0"
            )
        model_path = base_model_path
        if diffusion_gemma_mode:
            try:
                from diffusion_gemma_pipeline import (
                    preflight_diffusion_gemma_plugin,
                    prepare_diffusion_gemma_runtime_model,
                    remove_created_runtime_model,
                )

                plugin_target = preflight_diffusion_gemma_plugin()
                (
                    pipeline_model_path,
                    delta_metadata,
                    remove_runtime_model,
                ) = prepare_diffusion_gemma_runtime_model(
                    model_path,
                    args.diffusion_gemma_delta_checkpoint,
                    args.diffusion_gemma_runtime_model_dir,
                    work_dir=args.diffusion_gemma_delta_work_dir,
                    reconstruct_workers=args.diffusion_gemma_delta_workers,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                parser.error(str(error))
            if remove_runtime_model:
                diffusion_gemma_runtime_cleanup_path = pipeline_model_path
                atexit.register(
                    remove_created_runtime_model,
                    diffusion_gemma_runtime_cleanup_path,
                )
            delta_summary = (
                "base checkpoint (no SFT delta)"
                if delta_metadata is None
                else f"exact SFT delta reconstructed in "
                f"{delta_metadata.get('elapsed_seconds', 'unknown')}s"
            )
            print(
                "Validated DiffusionGemma runtime: "
                f"plugin={plugin_target}, model={pipeline_model_path}, "
                f"artifact={delta_summary}, lora_rank={args.lora_r}",
                flush=True,
            )
        elif model_declares_delta_mhar(model_path):
            if service_backend != "vllm":
                parser.error(
                    "Delta-MH checkpoints require the Unsloth or vLLM routed "
                    f"executor; {service_backend} has no equivalent forward"
                )
            try:
                routed_spec = preflight_vllm_delta_mhar_plugin(model_path)
                validate_routed_vllm_runtime_options(
                    model_family=routed_spec.family,
                    batch_invariant=args.vllm_batch_invariant,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                parser.error(str(error))
            print(
                "Validated routed vLLM checkpoint: "
                f"family={routed_spec.family}, mode={routed_spec.mode}, "
                f"heads={routed_spec.num_heads}, "
                f"block_size={routed_spec.block_size}, "
                f"routing_tensors={routed_spec.tensor_count}",
                "routing_source="
                + (
                    "embedded_hf"
                    if routed_spec.routing_weights_embedded
                    else "sidecar"
                ),
                flush=True,
            )
        if final_service_count < 1:
            parser.error(
                f"--{service_backend}-gpus 0 requires a positive adaptive "
                "inference ramp"
            )
        if final_service_count > MAX_GPU_COUNT:
            parser.error(
                f"Initial services ({service_gpu_count}) plus adaptive "
                f"services ({ramp_gpu_count}) exceed {MAX_GPU_COUNT} GPUs"
            )
        if ramp_after_adapters > selected_count:
            print(
                f"Inference ramp threshold {ramp_after_adapters} exceeds the "
                f"selected task count ({selected_count}); ramp GPUs will start "
                "after all training lanes finish. Existing inference services "
                "continue normally.",
                flush=True,
            )
        if use_vllm_rl_rollout and service_gpu_count < 1:
            parser.error(
                "Synchronous vLLM GRPO rollout requires at least one "
                "initial --vllm-gpus service"
            )
        if (
            diffusion_gemma_mode
            and service_gpu_count < 1
            and not post_training_scale
        ):
            parser.error(
                "DiffusionGemma with --vllm-gpus 0 requires "
                "--vllm-scale-after-training so every adapter can be staged "
                "before trainer GPUs become inference services"
            )
        max_trainer_count = MAX_GPU_COUNT - service_gpu_count
        if selected_count == 1 and args.single_task_gpus == MAX_GPU_COUNT:
            parser.error(
                f"The {service_backend} service topology cannot be combined with "
                "--single-task-gpus 4 because at least one GPU is reserved "
                "for inference"
            )
        if (
            args.multi_task_gpus is not None
            and args.multi_task_gpus > max_trainer_count
        ):
            gpu_flag = "--sglang-gpus" if service_backend == "sglang" else "--vllm-gpus"
            parser.error(
                f"{gpu_flag} {service_gpu_count} leaves at most "
                f"{max_trainer_count} training GPU(s), but "
                f"--multi-task-gpus {args.multi_task_gpus} was requested"
            )
        worker_count = min(
            selected_count,
            args.multi_task_gpus or max_trainer_count,
            max_trainer_count,
        )
        try:
            validate_diffusion_gemma_pair_fsdp_topology(
                enabled=(
                    diffusion_gemma_mode and args.diffusion_gemma_pair_fsdp
                ),
                selected_count=selected_count,
                worker_count=worker_count,
                service_gpu_count=service_gpu_count,
                ramp_gpu_count=ramp_gpu_count,
                post_training_scale=post_training_scale,
                experts_backend=args.diffusion_gemma_lora_experts_backend,
            )
            validate_diffusion_gemma_expert_parallel_topology(
                size=(
                    args.diffusion_gemma_expert_parallel_size
                    if diffusion_gemma_mode
                    else 1
                ),
                pair_fsdp=args.diffusion_gemma_pair_fsdp,
                selected_count=selected_count,
                worker_count=worker_count,
                service_gpu_count=service_gpu_count,
                ramp_gpu_count=ramp_gpu_count,
                post_training_scale=post_training_scale,
                experts_backend=args.diffusion_gemma_lora_experts_backend,
            )
        except ValueError as error:
            parser.error(str(error))
        if ramp_gpu_count >= worker_count:
            parser.error(
                "--inference-ramp-gpus must be smaller than the initial "
                f"trainer count ({worker_count}) so training can continue"
            )
        post_training_gpu_count = worker_count if post_training_scale else 0
        if post_training_scale:
            final_service_count = service_gpu_count + post_training_gpu_count
    elif selected_count == 1:
        worker_count = args.single_task_gpus
    else:
        worker_count = min(
            selected_count,
            args.multi_task_gpus or MAX_GPU_COUNT,
        )
    visible_gpu_count = torch.cuda.device_count()
    # A zero-initial-service post-training topology intentionally reuses the
    # trainer devices after their models are released. It is the supported
    # one-RTX-PRO-6000 train-then-infer mode and therefore requires only its
    # trainer count, not the historical fixed four-L4 topology.
    sequential_post_training_inference = bool(
        use_inference_pipeline and post_training_scale and service_gpu_count == 0
    )
    required_gpu_count = (
        worker_count
        if sequential_post_training_inference
        else MAX_GPU_COUNT
        if use_inference_pipeline
        else worker_count
    )
    if visible_gpu_count < required_gpu_count:
        raise RuntimeError(
            f"This mode requires {required_gpu_count} GPUs, but only "
            f"{visible_gpu_count} CUDA devices are visible"
        )
    if (
        service_backend == "vllm"
        and args.vllm_batch_invariant
        and not args.vllm_enforce_eager
    ):
        service_gpu_indices = range(
            MAX_GPU_COUNT - args.vllm_gpus,
            MAX_GPU_COUNT,
        )
        unsupported_services = []
        for gpu_index in service_gpu_indices:
            capability = torch.cuda.get_device_capability(gpu_index)
            if not vllm_batch_invariant_supports_cuda_graphs(capability):
                unsupported_services.append(
                    f"GPU {gpu_index} (SM{capability[0]}{capability[1]})"
                )
        if unsupported_services and 0:  # seem supported
            parser.error(
                "--vllm-batch-invariant cannot be combined with "
                "--no-vllm-enforce-eager on "
                f"{', '.join(unsupported_services)}. vLLM 0.23 requires "
                "eager execution for batch invariance below SM90. Use "
                "--no-vllm-batch-invariant to retain compiled/CUDA-graph "
                "performance, or use --vllm-enforce-eager to retain "
                "batch-invariant log-probabilities."
            )
    if service_backend == "vllm" and args.bidirectional_vllm_compile:
        service_gpu_indices = range(
            MAX_GPU_COUNT - args.vllm_gpus,
            MAX_GPU_COUNT,
        )
        unsupported_services = []
        for gpu_index in service_gpu_indices:
            capability = torch.cuda.get_device_capability(gpu_index)
            if not vllm_batch_invariant_supports_cuda_graphs(capability):
                unsupported_services.append(
                    f"GPU {gpu_index} (SM{capability[0]}{capability[1]})"
                )
        if unsupported_services:
            parser.error(
                "--bidirectional-vllm-compile is currently validated only "
                "on SM90 or newer because exact batch-invariant graph "
                "execution is required; unsupported: "
                f"{', '.join(unsupported_services)}"
            )
    if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1]")
    if args.vllm_kv_offloading_size < 0.0:
        parser.error("--vllm-kv-offloading-size must be non-negative")
    if args.vllm_max_num_seqs < 1:
        parser.error("--vllm-max-num-seqs must be positive")
    try:
        resolved_kv_resident_fraction = resolve_vllm_kv_resident_fraction(
            args.vllm_kv_resident_fraction, max_loras=args.vllm_max_loras
        )
    except ValueError as exc:
        parser.error(f"--vllm-kv-resident-fraction: {exc}")
    if args.vllm_max_pending_adapters < 1:
        parser.error("--vllm-max-pending-adapters must be positive")
    if args.vllm_max_adapter_recoveries < 1:
        parser.error("--vllm-max-adapter-recoveries must be positive")
    if args.vllm_adapter_timeout_seconds <= 0.0:
        parser.error("--vllm-adapter-timeout-seconds must be positive")
    if args.vllm_startup_timeout_seconds <= 0.0:
        parser.error("--vllm-startup-timeout-seconds must be positive")
    if args.vllm_startup_attempts < 1:
        parser.error("--vllm-startup-attempts must be positive")
    if args.vllm_restart_backoff_seconds < 0.0:
        parser.error("--vllm-restart-backoff-seconds must be non-negative")
    if args.vllm_auto_recover and args.inference_backend != "vllm":
        parser.error("--vllm-auto-recover requires --inference-backend vllm")
    if args.vllm_auto_recover and args.task_rl and args.rl_rollout_backend == "vllm":
        parser.error(
            "--vllm-auto-recover currently supports asynchronous final "
            "inference only, not synchronous vLLM GRPO rollouts"
        )
    if args.vllm_max_num_batched_tokens < 1:
        parser.error("--vllm-max-num-batched-tokens must be positive")
    if args.vllm_rollout_max_tokens < 1:
        parser.error("--vllm-rollout-max-tokens must be positive")
    if args.vllm_rollout_branch_window < 1:
        parser.error("--vllm-rollout-branch-window must be positive")
    if args.vllm_cumulative_nll_stop and not args.vllm_leaf_rollout:
        parser.error("--vllm-cumulative-nll-stop requires --vllm-leaf-rollout")
    if (
        args.vllm_cumulative_nll_stop
        and args.vllm_speculative_ngram_tokens > 0
    ):
        parser.error(
            "--vllm-cumulative-nll-stop is incompatible with speculative "
            "decoding in vLLM 0.23"
        )
    if args.vllm_speculative_ngram_tokens < 0:
        parser.error("--vllm-speculative-ngram-tokens must be non-negative")
    if (
        args.vllm_speculative_ngram_min < 1
        or args.vllm_speculative_ngram_max < args.vllm_speculative_ngram_min
    ):
        parser.error("Invalid speculative n-gram lookup range")
    if args.sglang_max_loaded_loras < 1:
        parser.error("--sglang-max-loaded-loras must be positive")
    if args.sglang_max_loras_per_batch < 1:
        parser.error("--sglang-max-loras-per-batch must be positive")
    if args.sglang_max_loras_per_batch > args.sglang_max_loaded_loras:
        parser.error(
            "--sglang-max-loras-per-batch cannot exceed --sglang-max-loaded-loras"
        )
    if (
        args.sglang_overlap_loading
        and args.sglang_max_loaded_loras > 2 * args.sglang_max_loras_per_batch
    ):
        parser.error(
            "SGLang overlap loading requires --sglang-max-loaded-loras "
            "to be at most twice --sglang-max-loras-per-batch"
        )
    if not (
        args.sglang_mem_fraction_static == 0.0
        or 0.0 < args.sglang_mem_fraction_static <= 1.0
    ):
        parser.error("--sglang-mem-fraction-static must be zero or in (0, 1]")
    if args.sglang_max_running_requests < 1:
        parser.error("--sglang-max-running-requests must be positive")
    if args.sglang_rollout_max_tokens < 1:
        parser.error("--sglang-rollout-max-tokens must be positive")
    if args.sglang_chunked_prefill_size < 0:
        parser.error("--sglang-chunked-prefill-size must be non-negative")
    if (
        args.train_only
        and args.inference_backend in {"vllm", "sglang"}
        and not use_vllm_rl_rollout
    ):
        print(
            "Train-only mode does not start a reserved inference service; "
            "using the existing trainer topology."
        )

    print(f"Selected {selected_count} task(s): {selected_keys}")
    prebuild_diffusion_gemma_pair_fsdp_extension(
        enabled=(
            diffusion_gemma_mode
            and (
                args.diffusion_gemma_pair_fsdp
                or args.diffusion_gemma_expert_parallel_size > 1
            )
        ),
        ream_root=args.diffusion_gemma_ream_root,
    )
    pair_fsdp_port = (
        find_available_port()
        if diffusion_gemma_mode and args.diffusion_gemma_pair_fsdp
        else None
    )
    expert_parallel_port = (
        find_available_port()
        if diffusion_gemma_mode
        and args.diffusion_gemma_expert_parallel_size > 1
        else None
    )
    worker_options = {
        "model_family": model_family,
        "diffusion_gemma_model_path": pipeline_model_path,
        "diffusion_gemma_ream_root": args.diffusion_gemma_ream_root,
        "diffusion_gemma_ttt_steps": args.diffusion_gemma_ttt_steps,
        "diffusion_gemma_color_augmentations": (
            args.diffusion_gemma_color_augmentations
        ),
        "diffusion_gemma_inference_views": args.diffusion_gemma_inference_views,
        "diffusion_gemma_max_output_tokens": (
            args.diffusion_gemma_max_output_tokens
        ),
        "diffusion_gemma_learning_rate": args.diffusion_gemma_learning_rate,
        "diffusion_gemma_optimizer": args.diffusion_gemma_optimizer,
        "diffusion_gemma_optimizer_state_offload": (
            args.diffusion_gemma_optimizer_state_offload
        ),
        "diffusion_gemma_optimizer_state_offload_pin_memory": (
            args.diffusion_gemma_optimizer_state_offload_pin_memory
        ),
        "diffusion_gemma_lora_token_chunk_size": (
            args.diffusion_gemma_lora_token_chunk_size
        ),
        "diffusion_gemma_lora_experts_backend": (
            args.diffusion_gemma_lora_experts_backend
        ),
        "diffusion_gemma_pair_fsdp": args.diffusion_gemma_pair_fsdp,
        "diffusion_gemma_pair_fsdp_port": pair_fsdp_port,
        "diffusion_gemma_expert_parallel_size": (
            args.diffusion_gemma_expert_parallel_size
        ),
        "diffusion_gemma_expert_parallel_port": expert_parallel_port,
        "diffusion_gemma_decoder_checkpointing_percentage": (
            args.diffusion_gemma_decoder_checkpointing_percentage
        ),
        "diffusion_gemma_encoder_checkpointing_percentage": (
            args.diffusion_gemma_encoder_checkpointing_percentage
        ),
        "diffusion_gemma_kernel_backend": args.diffusion_gemma_kernel_backend,
        "diffusion_gemma_encoder_attention_backend": (
            args.diffusion_gemma_encoder_attention_backend
        ),
        "diffusion_gemma_decoder_attention_backend": (
            args.diffusion_gemma_decoder_attention_backend
        ),
        "optimizer": args.optimizer,
        "flash_adamw_quantize": args.flash_adamw_quantize,
        "flash_adamw_master_weight_bits": (args.flash_adamw_master_weight_bits),
        "normuon_adjust_lr": args.normuon_adjust_lr,
        "normuon_momentum": args.normuon_momentum,
        "normuon_beta2": args.normuon_beta2,
        "normuon_megabatch_size": args.normuon_megabatch_size,
        "normuon_use_triton": args.normuon_use_triton,
        "muon_adjust_lr": args.muon_adjust_lr,
        "muon_momentum": args.muon_momentum,
        "muon_nesterov": args.muon_nesterov,
        "muon_ns_chunk_size": args.muon_ns_chunk_size,
        "muon_min_8bit_size": args.muon_min_8bit_size,
        "muon_scalar_optimizer": args.muon_scalar_optimizer,
        "te_optimizer_state_dtype": args.te_optimizer_state_dtype,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_mode": args.gradient_checkpointing_mode,
        "gradient_checkpointing_interval": (args.gradient_checkpointing_interval),
        "gradient_checkpointing_percentage": (args.gradient_checkpointing_percent),
        "deepspeed_long_context": args.deepspeed_long_context,
        "deepspeed_token_threshold": args.deepspeed_token_threshold,
        "deepspeed_gradient_checkpointing_percentage": (
            args.deepspeed_gradient_checkpointing_percent
        ),
        "deepspeed_offload_optimizer": args.deepspeed_offload_optimizer,
        "deepspeed_param_persistence_threshold": (
            args.deepspeed_param_persistence_threshold
        ),
        "deepspeed_model_persistence_threshold": (
            args.deepspeed_model_persistence_threshold
        ),
        "deepspeed_max_live_parameters": (
            args.deepspeed_max_live_parameters
        ),
        "deepspeed_prefetch_bucket_size": (
            args.deepspeed_prefetch_bucket_size
        ),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "sft_loss_scope": args.sft_loss_scope,
        "use_liger_kernel": args.liger_kernel,
        "lora_method": args.lora_method,
        "lora_init": args.lora_init,
        "pissa_base_mode": args.pissa_base_mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_embed_tokens": args.train_embed_tokens,
        "max_seq_length": args.max_seq_length,
        "recompute_mlp_activations": args.recompute_mlp_activations,
        "recompute_attention_activations": (args.recompute_attention_activations),
        "hybrid_recompute_mlp_activations": (
            args.hybrid_recompute_mlp_activations
        ),
        "hybrid_recompute_attention_activations": (
            args.hybrid_recompute_attention_activations
        ),
        "hybrid_attention_recompute_percentage": (
            args.hybrid_attention_recompute_percentage
        ),
        "fused_qknorm_rope": args.fused_qknorm_rope,
        "attention_checkpoint_input_offload": (
            args.attention_checkpoint_input_offload
        ),
        "fp8_frozen_linear": args.fp8_frozen_linear,
        "fp8_lora": args.fp8_lora,
        "fp8_min_train_tokens": args.fp8_min_train_tokens,
        "fp8_dynamic_backward": args.fp8_dynamic_backward,
        "fp8_dynamic_backward_split_accumulator": (
            args.fp8_dynamic_backward_split_accumulator
        ),
        "auto_activation_recompute": args.auto_activation_recompute,
        "mlp_recompute_token_threshold": (args.mlp_recompute_token_threshold),
        "attention_recompute_token_threshold": (
            args.attention_recompute_token_threshold
        ),
        "max_grad_norm": args.max_grad_norm,
        "torch_compile_training": args.torch_compile_training,
        "bidirectional_prompt_attention": (args.bidirectional_prompt_attention),
        "train_only": args.train_only,
        "task_rl": args.task_rl,
        "rl_sft_steps": args.rl_sft_steps,
        "rl_max_steps": args.rl_max_steps,
        "rl_num_generations": args.rl_num_generations,
        "rl_micro_batch_size": args.rl_micro_batch_size,
        "rl_auto_micro_batch": args.rl_auto_micro_batch,
        "rl_rollout_backend": args.rl_rollout_backend,
        "rl_augmentations": args.rl_augmentations,
        "rl_max_prompt_tokens": args.rl_max_prompt_tokens,
        "rl_reward_min_cell_accuracy": (args.rl_reward_min_cell_accuracy),
        "rl_exact_bonus": args.rl_exact_bonus,
        "rl_temperature": args.rl_temperature,
        "rl_learning_rate": args.rl_learning_rate,
        "predict_train": args.predict_train,
        "report_output_nll": args.report_output_nll,
        "min_train_pairs": args.min_train_pairs,
        "extra_train_pair_mode": args.extra_train_pair_mode,
        "profile_training": args.profile_training,
        "profile_wait_steps": args.profile_wait_steps,
        "profile_warmup_steps": args.profile_warmup_steps,
        "profile_active_steps": args.profile_active_steps,
        "inference_timeout_seconds": args.inference_timeout_seconds,
        "inference_batch_size": args.inference_batch_size,
        "score_batch_size": args.score_batch_size,
        "rescore_views": args.rescore_views,
        "rescore_schedule": args.rescore_schedule,
        "vllm_prefix_score": args.vllm_prefix_score,
        "vllm_prefix_score_tile_views": (args.vllm_prefix_score_tile_views),
        "dfs_mode": args.dfs_mode,
        "frontier_origin_batch_size": (args.frontier_origin_batch_size),
        "frontier_max_live_branches": (args.frontier_max_live_branches),
        "frontier_max_prompt_tokens": (args.frontier_max_prompt_tokens),
        "max_beam_score": args.max_beam_score,
        "zero_candidate_shape_fallback": (args.zero_candidate_shape_fallback),
        "shape_fallback_beam_width": args.shape_fallback_beam_width,
        "vllm_preserve_adapter_root": args.vllm_preserve_adapter_root,
        # The trainer-side publisher predates the SGLang backend; retain its
        # internal argument name while using the same atomic adapter handoff
        # for either reserved inference service.
        "vllm_final_inference": (
            use_vllm_final_inference or use_sglang_final_inference
        ),
    }
    worker_options = training_worker_options_for_model(
        worker_options,
        model_family=model_family,
    )
    print(f"Worker options: {worker_options}")
    if use_vllm_final_inference and not is_pissa_init(args.lora_init):
        print(
            f"vLLM LoRA ranks: training={args.lora_r}; "
            f"export={lora_export_rank(args.lora_r, args.lora_init, args.pissa_base_mode)}; "
            f"engine capacity={lora_engine_max_rank(args.lora_r, args.lora_init, args.pissa_base_mode)}",
            flush=True,
        )
    if is_pissa_init(args.lora_init):
        print(
            f"PiSSA opt-in: training rank={args.lora_r}; "
            f"export rank={lora_export_rank(args.lora_r, args.lora_init, args.pissa_base_mode)}; "
            f"vLLM rank capacity={lora_engine_max_rank(args.lora_r, args.lora_init, args.pissa_base_mode)}; "
            f"base_mode={args.pissa_base_mode}. "
            + (
                f"Shared residual id={shared_pissa_manifest['base_id']}; no SVD or rank doubling in workers."
                if shared_pissa_manifest else
                "Original-base conversion doubles projection storage; BF16 parity needs a live gate."
            ),
            flush=True,
        )

    split_broker = None
    split_prepared = None
    split_report_dir = None
    if split_specs is not None:
        from pipeline_runtime_ipc import PipelineBroker
        from pipeline_runtime_launcher import create_report_directory, prepare_runtimes

        split_report_dir = create_report_directory(args.runtime_report_dir)
        split_prepared = prepare_runtimes(split_specs, split_report_dir)
        split_broker = runtime_cleanup.enter_context(PipelineBroker())
        spawn_context = split_broker
        print(f"Split runtime reports: {split_report_dir}", flush=True)
    else:
        spawn_context = mp.get_context("spawn")
    import_lock = spawn_context.Lock()
    if use_inference_pipeline:
        abort_event = spawn_context.Event()
        task_queue = spawn_context.Queue()
        job_queues = [spawn_context.Queue() for _ in range(final_service_count)]
        # Per-service queues now carry control messages / directed GRPO only.
        # Qwen vLLM adapters stay unassigned until a ready engine has a slot.
        shared_pending_pool = service_backend == "vllm" and not diffusion_gemma_mode
        shared_pending_queue = spawn_context.Queue() if shared_pending_pool else None
        shared_pending_count = (
            spawn_context.Value("i", 0, lock=False) if shared_pending_pool else None
        )
        rollout_response_queues = [spawn_context.Queue() for _ in range(worker_count)]
        pending_adapter_slots = spawn_context.BoundedSemaphore(
            args.vllm_max_pending_adapters
        )
        service_outstanding = spawn_context.Array(
            "i",
            [0] * final_service_count,
            lock=False,
        )
        service_active = spawn_context.Array(
            "b",
            [1] * service_gpu_count + [0] * (final_service_count - service_gpu_count),
            lock=False,
        )
        # ``active`` describes topology; ``ready`` describes engine health.
        # Keeping them separate prevents trainers from assigning adapters to
        # a service that is still starting or is being recovered.
        service_ready = spawn_context.Array(
            "b",
            [0] * final_service_count,
            lock=False,
        )
        published_adapter_count = spawn_context.Value(
            "i",
            0,
            lock=False,
        )
        ramp_ready_event = spawn_context.Event()
        ramp_trainers_finished_count = (
            spawn_context.Value("i", 0) if adaptive_inference_ramp else None
        )
        trainers_finished_count = spawn_context.Value(
            "i",
            0,
        )
        training_complete_event = spawn_context.Event()
        post_training_rebalance_count = spawn_context.Value(
            "i",
            0,
        )
        post_training_rebalance_event = spawn_context.Event()
        trainer_service_indices = (
            spawn_context.Array("i", [-1] * worker_count, lock=False)
            if post_training_scale and adaptive_inference_ramp
            else None
        )
        service_load_lock = spawn_context.Lock()
        inference_startup_lock = spawn_context.Lock()
        enqueue_tasks(task_queue, selected_keys, worker_count)
        resident_adapter_capacity = final_service_count * (
            args.sglang_max_loaded_loras
            if service_backend == "sglang"
            else args.vllm_max_loras
        )
        # Completed tasks retain their staging artifact until the engine
        # unloads the adapter. Bound the combined staging pool by queued plus
        # resident slots, never by the full hidden-test task count.
        max_staged_adapters = min(
            selected_count,
            args.vllm_max_pending_adapters + resident_adapter_capacity,
        )
        try:
            (
                adapter_staging_plan,
                adapter_free_bytes,
                adapter_required_bytes,
            ) = choose_adapter_staging_plan(
                args.inference_adapter_storage,
                args.vllm_adapter_root,
                max_staged_adapters,
                ram_root=args.inference_ram_adapter_root,
                disk_root=args.inference_disk_adapter_root,
                tmp_root=args.inference_tmp_adapter_root,
                bytes_per_slot_mib=(
                    48 * args.lora_r
                    if diffusion_gemma_mode
                    else ADAPTER_STAGING_ESTIMATE_MIB * max(
                        1.0, lora_export_rank(args.lora_r, args.lora_init, args.pissa_base_mode) / 256
                    )
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        run_id = uuid.uuid4().hex[:12]
        adapter_roots = tuple(
            os.path.join(tier.parent, run_id) for tier in adapter_staging_plan
        )
        for adapter_run_root in adapter_roots:
            os.makedirs(adapter_run_root, exist_ok=False)
        adapter_root = adapter_roots[0]
        adapter_staging_slots = tuple(
            spawn_context.BoundedSemaphore(tier.slot_capacity)
            for tier in adapter_staging_plan
        )
        adapter_storage = "+".join(
            tier.label for tier in adapter_staging_plan
        )
        adapter_capacity_details = ", ".join(
            f"{tier.label}:{tier.slot_capacity} slot(s) at "
            f"{adapter_roots[index]}"
            for index, tier in enumerate(adapter_staging_plan)
        )
        adapter_ids = {key: index + 1 for index, key in enumerate(selected_keys)}
        inference_options = dict(
            worker_options,
            pissa_base_id=(shared_pissa_manifest["base_id"] if shared_pissa_manifest else None),
            base_model_fallback=args.base_model_fallback,
            report_task_accuracy=args.report_task_accuracy,
            accuracy_run_id=run_id,
            model_family=model_family,
            service_backend=service_backend,
            bidirectional_vllm_compile=(args.bidirectional_vllm_compile),
            vllm_max_loras=args.vllm_max_loras,
            vllm_auto_recover=args.vllm_auto_recover,
            vllm_max_adapter_recoveries=(args.vllm_max_adapter_recoveries),
            vllm_adapter_timeout_seconds=(args.vllm_adapter_timeout_seconds),
            vllm_startup_timeout_seconds=(args.vllm_startup_timeout_seconds),
            vllm_startup_attempts=args.vllm_startup_attempts,
            vllm_restart_backoff_seconds=(args.vllm_restart_backoff_seconds),
            vllm_gpu_memory_utilization=(args.vllm_gpu_memory_utilization),
            vllm_quantization=args.vllm_quantization,
            vllm_kv_cache_dtype=args.vllm_kv_cache_dtype,
            vllm_kv_offloading_size=args.vllm_kv_offloading_size,
            vllm_kv_offloading_backend=(args.vllm_kv_offloading_backend),
            vllm_enforce_eager=args.vllm_enforce_eager,
            vllm_async_scheduling=args.vllm_async_scheduling,
            vllm_batch_invariant=args.vllm_batch_invariant,
            vllm_targeted_logprobs_only=(
                args.vllm_targeted_logprobs_only
            ),
            vllm_speculative_ngram_tokens=(
                args.vllm_speculative_ngram_tokens
            ),
            vllm_speculative_ngram_method=(
                args.vllm_speculative_ngram_method
            ),
            vllm_speculative_ngram_min=args.vllm_speculative_ngram_min,
            vllm_speculative_ngram_max=args.vllm_speculative_ngram_max,
            **dflash_options(args),
            vllm_max_num_seqs=args.vllm_max_num_seqs,
            vllm_kv_resident_batching=args.vllm_kv_resident_batching,
            vllm_kv_resident_fraction=args.vllm_kv_resident_fraction,
            vllm_max_num_batched_tokens=(args.vllm_max_num_batched_tokens),
            vllm_telemetry_dir=args.vllm_telemetry_dir,
            vllm_forensics_dir=args.vllm_forensics_dir,
            vllm_leaf_rollout=args.vllm_leaf_rollout,
            vllm_search_limit_seconds=args.vllm_search_limit_seconds,
            vllm_rollout_max_tokens=args.vllm_rollout_max_tokens,
            vllm_rollout_branch_window=(args.vllm_rollout_branch_window),
            vllm_cumulative_nll_stop=(args.vllm_cumulative_nll_stop),
            rl_rollout_backend=args.rl_rollout_backend,
            sglang_max_loaded_loras=args.sglang_max_loaded_loras,
            sglang_max_loras_per_batch=(args.sglang_max_loras_per_batch),
            sglang_lora_backend=args.sglang_lora_backend,
            sglang_overlap_loading=args.sglang_overlap_loading,
            sglang_deterministic_inference=(args.sglang_deterministic_inference),
            sglang_leaf_rollout=args.sglang_leaf_rollout,
            sglang_rollout_max_tokens=(args.sglang_rollout_max_tokens),
            sglang_mem_fraction_static=(args.sglang_mem_fraction_static),
            sglang_max_running_requests=(args.sglang_max_running_requests),
            sglang_chunked_prefill_size=(args.sglang_chunked_prefill_size),
        )
        service_gpu_indices = list(
            range(MAX_GPU_COUNT - service_gpu_count, MAX_GPU_COUNT)
        )
        if service_backend == "sglang":
            service_details = (
                f"loaded LoRAs per service="
                f"{args.sglang_max_loaded_loras}; "
                f"LoRAs per GPU batch="
                f"{args.sglang_max_loras_per_batch}; "
                f"LoRA backend={args.sglang_lora_backend}; "
                f"overlap loading={args.sglang_overlap_loading}; "
                f"deterministic inference="
                f"{args.sglang_deterministic_inference}; "
                f"leaf rollout={args.sglang_leaf_rollout}; "
                f"rollout chunk={args.sglang_rollout_max_tokens}; "
                f"decode={'chunked DFS' if args.sglang_leaf_rollout else 'exact one-token DFS'}"
            )
        else:
            service_details = (
                f"resident LoRAs per service={args.vllm_max_loras}; "
                f"auto recovery={args.vllm_auto_recover}/"
                f"{args.vllm_max_adapter_recoveries} recovery attempt(s) "
                "per adapter; "
                f"adapter watchdog={args.vllm_adapter_timeout_seconds:g}s/"
                "1 independent restart; "
                f"startup watchdog={args.vllm_startup_timeout_seconds:g}s/"
                f"{args.vllm_startup_attempts} attempt(s); "
                f"leaf rollout={args.vllm_leaf_rollout}; "
                f"search limit={args.vllm_search_limit_seconds:g}s/group; "
                f"task timeout={args.inference_timeout_seconds:g}s (wall); "
                f"async scheduling={args.vllm_async_scheduling}; "
                f"batch invariant={args.vllm_batch_invariant}; "
                f"KV cache dtype={args.vllm_kv_cache_dtype}; "
                f"KV-resident batching={args.vllm_kv_resident_batching}/"
                f"{resolved_kv_resident_fraction:.3f}"
                f"{' (auto)' if args.vllm_kv_resident_fraction == 'auto' else ''}; "
                f"KV offload={args.vllm_kv_offloading_size:g} GiB/"
                f"{args.vllm_kv_offloading_backend}; "
                f"RL rollout backend={args.rl_rollout_backend}; "
                f"rollout chunk={args.vllm_rollout_max_tokens}; "
                f"branch window={args.vllm_rollout_branch_window}; "
                f"cumulative-NLL stop={args.vllm_cumulative_nll_stop}"
            )
        print(
            f"Using {worker_count} dynamic trainer(s) on GPU(s) "
            f"{list(range(worker_count))} and {service_gpu_count} independent "
            f"initial TP=1 {service_backend} service(s) on GPU(s) "
            f"{service_gpu_indices}; {service_details}; "
            f"adaptive ramp={ramp_gpu_count} GPU(s) after "
            f"{ramp_after_adapters} adapter(s), final services="
            f"{final_service_count}; "
            f"post-training scale-out={post_training_scale}/"
            f"{worker_count - ramp_gpu_count if post_training_scale else 0} GPU(s); "
            f"global pending adapter capacity="
            f"{args.vllm_max_pending_adapters}; pending pool="
            f"{'shared' if shared_pending_pool else 'per-service'}; adapter storage="
            f"{adapter_storage}; adapter tiers={adapter_capacity_details}; "
            f"capacity estimate={adapter_required_bytes / 1024**3:.2f} "
            f"GiB/{adapter_free_bytes / 1024**3:.2f} GiB free"
        )
        fresh_diffusion_gemma_service_phase = bool(
            diffusion_gemma_mode
            and post_training_scale
            and service_gpu_count == 0
        )
        pipeline_worker_args = (
            worker_count,
            service_gpu_count,
            task_queue,
            job_queues,
            rollout_response_queues,
            pending_adapter_slots,
            service_outstanding,
            service_load_lock,
            inference_startup_lock,
            import_lock,
            abort_event,
            args.end_time,
            args.keep_training_state,
            worker_options,
            inference_options,
            pipeline_model_path,
            adapter_root,
            run_id,
            adapter_ids,
            service_active,
            published_adapter_count,
            ramp_ready_event,
            ramp_after_adapters,
            final_service_count,
            post_training_scale,
            trainers_finished_count,
            training_complete_event,
            post_training_rebalance_count,
            post_training_rebalance_event,
            service_gpu_count + (ramp_gpu_count if post_training_scale else 0),
            service_ready,
            adapter_roots,
            adapter_staging_slots,
            fresh_diffusion_gemma_service_phase,
            False,
            trainer_service_indices,
            shared_pending_queue,
            shared_pending_count,
            ramp_trainers_finished_count,
        )
        pipeline_succeeded = False
        try:
            if split_broker is not None:
                from pipeline_runtime_launcher import SplitRuntimeLauncher

                binding = inspect.signature(vllm_pipeline_worker).bind(0, *pipeline_worker_args)
                pipeline_kwargs = dict(binding.arguments)
                del pipeline_kwargs["process_rank"]
                SplitRuntimeLauncher(
                    split_prepared, split_broker, split_report_dir,
                    environment=dict(
                        os.environ,
                        ARC_MODEL_PATH=pipeline_model_path,
                        ARC_CHALLENGE_PATH=test_path,
                    ),
                ).run(
                    pipeline_kwargs, max_gpu_count=MAX_GPU_COUNT,
                )
            elif fresh_diffusion_gemma_service_phase:
                print(
                    "DiffusionGemma per-lane handoff enabled: each exhausted "
                    "trainer will exit before a clean same-GPU vLLM process "
                    "starts; remaining trainer lanes continue concurrently",
                    flush=True,
                )
                fresh_service_args = (
                    *pipeline_worker_args[:-6],
                    False,
                    True,
                    *pipeline_worker_args[-4:],
                )
                spawn_trainers_with_per_lane_services(
                    vllm_pipeline_worker,
                    vllm_pipeline_worker,
                    trainer_args=pipeline_worker_args,
                    service_args=fresh_service_args,
                    nprocs=worker_count,
                    abort_event=abort_event,
                )
            else:
                spawn_and_join(
                    vllm_pipeline_worker,
                    args=pipeline_worker_args,
                    nprocs=worker_count + service_gpu_count,
                )
            pipeline_succeeded = True
        finally:
            if shared_pending_queue is not None:
                close_process_queue(
                    shared_pending_queue, successful=pipeline_succeeded,
                )
            close_process_queue(
                task_queue,
                successful=pipeline_succeeded,
            )
            for job_queue in job_queues:
                close_process_queue(
                    job_queue,
                    successful=pipeline_succeeded,
                )
            for response_queue in rollout_response_queues:
                close_process_queue(
                    response_queue,
                    successful=pipeline_succeeded,
                )
            # Services normally unlink each artifact after task inference and
            # adapter unload. A failed/aborted pipeline may still leave queued
            # staged adapters, so reclaim every exact per-run UUID directory
            # after all children have exited.
            for adapter_run_root in adapter_roots:
                shutil.rmtree(adapter_run_root, ignore_errors=True)
            if diffusion_gemma_runtime_cleanup_path is not None:
                from diffusion_gemma_pipeline import remove_created_runtime_model

                remove_created_runtime_model(diffusion_gemma_runtime_cleanup_path)
                diffusion_gemma_runtime_cleanup_path = None
    elif selected_count == 1 and args.single_task_gpus == MAX_GPU_COUNT:
        print(f"Using all {MAX_GPU_COUNT} GPUs with DDP for one task")
        spawn_and_join(
            distributed_worker,
            args=(
                selected_keys[0],
                import_lock,
                args.end_time,
                find_available_port(),
                args.keep_training_state,
                worker_options,
            ),
            nprocs=MAX_GPU_COUNT,
        )
    else:
        abort_event = spawn_context.Event()
        task_queue = spawn_context.Queue()
        enqueue_tasks(task_queue, selected_keys, worker_count)
        print(
            f"Using dynamic scheduling on GPU(s): "
            f"{list(range(worker_count))}; "
            f"keep training state: {args.keep_training_state}"
        )
        if args.keep_training_state:
            print(
                "Warning: --keep-training-state carries state through each "
                "GPU's timing-dependent task sequence."
            )
        workers_succeeded = False
        try:
            spawn_and_join(
                independent_worker,
                args=(
                    task_queue,
                    import_lock,
                    abort_event,
                    args.end_time,
                    args.keep_training_state,
                    worker_options,
                ),
                nprocs=worker_count,
            )
            workers_succeeded = True
        finally:
            close_process_queue(
                task_queue,
                successful=workers_succeeded,
            )


if __name__ == "__main__":
    if "--runtime-preflight-only" in sys.argv[1:]:
        # Checking an environment is not a completed solver run.
        main()
    else:
        run_entrypoint_with_status(
            main,
            path=extract_run_status_path(),
            phase="starter",
            success_status=STATUS_STARTER_SUCCESS,
        )
