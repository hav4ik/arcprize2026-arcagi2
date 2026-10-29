"""Persistent independent TP=1 vLLM multi-LoRA services for ARC.

The module is imported only by the opt-in vLLM worker.  It deliberately keeps
all ARC search and scoring policy on the CPU and treats vLLM as a batched raw
log-probability provider.
"""

from __future__ import annotations

import bz2
import gc
import hashlib
import inspect
import json
import math
import os
import pickle
import signal
import subprocess
import tempfile
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing.util import Finalize
from queue import Empty
from typing import Any

import numpy as np

from arc_loader import (
    ArcDataset,
    QwenFormatter,
    infer_demonstrated_output_shape,
    parse_train_prediction_key,
)
from arc_vllm_adapter import remove_job_adapter_artifact
from arc_vllm_dflash import build_dflash_speculative_config, prepare_dflash_runtime
from arc_vllm_pending_pool import (
    claim_pending_adapter,
    finish_service_if_pool_drained,
    pending_pool_snapshot,
    return_pending_adapter,
)
from arc_vllm_search import (
    ARC_TOKENS,
    FrontierSearchBatch,
    RecursiveRolloutSearchBatch,
    RecursiveSearchBatch,
    ShapeBeamSearch,
    answer_nll_from_prompt_logprobs,
    logprobs_at_position,
    requested_arc_logprobs,
)
from arc_vllm_router import (
    finish_service_if_idle,
    mark_service_dead,
    reassign_task_to_live_service,
    rebalance_pending_service_task,
    release_service_task,
    service_load_snapshot,
    transfer_service_task_assignment,
)
from output_nll_report import (
    OutputNllTarget,
    apply_output_nll_scores,
    finalize_output_nll_report,
    prepare_output_nll_report,
    write_output_nll_report,
)
from run_config import (
    lora_engine_max_rank,
    normalize_vllm_kv_cache_dtype,
    normalize_vllm_search_limit_seconds,
    resolve_challenge_path,
    resolve_train_prediction_output_dir,
    resolve_vllm_kv_resident_fraction,
    stable_task_seed,
    validate_pissa_options,
)


USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
EOS_TOKEN_ID = 15
IM_START_TOKEN_ID = 14
_PENDING_SLOT_HELD = "_arc_pending_slot_held"
_ADMISSION_IN_PROGRESS = "_arc_vllm_admission_in_progress"
_ENGINE_FAILURE_COUNT = "_arc_vllm_engine_failures"
_ADAPTER_TIMEOUT_COUNT = "_arc_vllm_adapter_timeouts"
_MAX_ADAPTER_TIMEOUT_RESTARTS = 1
_ENGINE_LIVENESS_POLL_SECONDS = 2.0
_ARC_CHECKPOINT_VOCAB_SIZE = max(ARC_TOKENS) + 1


class RecoverableVllmResponseError(RuntimeError):
    """A malformed vLLM response that warrants a bounded engine replay.

    This is intentionally narrower than a generic output-validation failure.
    ARC requests explicitly ask vLLM for every grammar-token logprob; a
    response that silently omits those requested IDs cannot be consumed or
    safely retried against the same potentially-corrupted EngineCore state.
    """


class EngineCoreDeadError(RuntimeError):
    """The EngineCore child exited or stopped answering a blocking client call.

    vLLM's synchronous MP client flags a dead core but never wakes a caller
    already blocked on its output queue, so liveness is polled externally.
    """


class AdapterInferenceTimeoutError(TimeoutError):
    """A hard per-adapter inference watchdog expired.

    ``timed_out_adapter_ids`` identifies only the adapter(s) whose independent
    wall-clock budget expired.  Other adapters resident on the same poisoned
    or wedged EngineCore are replayed after restart without consuming either
    their timeout retry or fatal-engine recovery budget.
    """

    def __init__(
        self,
        timed_out_adapters: list[tuple[int, str, float]],
        timeout_seconds: float,
    ) -> None:
        if not timed_out_adapters:
            raise ValueError("timed_out_adapters must not be empty")
        self.timed_out_adapters = tuple(
            (int(adapter_id), str(task_key), float(elapsed_seconds))
            for adapter_id, task_key, elapsed_seconds in timed_out_adapters
        )
        self.timed_out_adapter_ids = frozenset(
            adapter_id for adapter_id, _task_key, _elapsed in self.timed_out_adapters
        )
        self.timeout_seconds = float(timeout_seconds)
        details = ", ".join(
            f"{task_key}#{adapter_id} elapsed={elapsed_seconds:.1f}s"
            for adapter_id, task_key, elapsed_seconds in self.timed_out_adapters
        )
        super().__init__(
            "vLLM adapter inference watchdog expired after "
            f"{self.timeout_seconds:.1f}s: {details}"
        )


class VllmEngineStartupTimeoutError(TimeoutError):
    """A serialized EngineCore construction exceeded its wall-clock budget."""


def summarize_invalid_arc_logprobs(
    logprobs: Any,
    sampled_token_ids: Any,
    *,
    max_positions: int = 4,
) -> dict[str, Any]:
    """Build a bounded, JSON-safe description of malformed vLLM logprobs."""
    sampled = tuple(map(int, sampled_token_ids or ()))
    try:
        position_count = 1 if isinstance(logprobs, Mapping) else len(logprobs)
    except (TypeError, AttributeError):
        position_count = 1
    invalid_positions: list[dict[str, Any]] = []
    scanned_positions = 0
    for position in range(max(1, int(position_count))):
        scanned_positions = position + 1
        try:
            returned = logprobs_at_position(logprobs, position=position)
        except (TypeError, ValueError, IndexError, KeyError) as error:
            invalid_positions.append(
                {
                    "position": position,
                    "parse_error": f"{type(error).__name__}: {error}",
                    "raw_type": type(logprobs).__name__,
                    "raw_repr": repr(logprobs)[:1000],
                }
            )
        else:
            missing = [
                token_id for token_id in ARC_TOKENS if token_id not in returned
            ]
            nonfinite = [
                token_id
                for token_id, value in returned.items()
                if not math.isfinite(float(value))
            ]
            if missing or nonfinite:
                invalid_positions.append(
                    {
                        "position": position,
                        "missing_arc_token_ids": missing,
                        "nonfinite_token_ids": sorted(nonfinite),
                        "returned": {
                            str(token_id): (
                                round(float(value), 8)
                                if math.isfinite(float(value))
                                else repr(float(value))
                            )
                            for token_id, value in sorted(returned.items())
                        },
                    }
                )
        if len(invalid_positions) >= int(max_positions):
            break
    return {
        "sampled_token_ids": list(sampled),
        "position_count": int(position_count),
        "invalid_positions": invalid_positions,
        "diagnostic_truncated": int(position_count) > scanned_positions,
    }


def iter_exception_chain(error: BaseException):
    """Yield an exception and its explicit/implicit causes once each."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def find_adapter_timeout_error(
    error: BaseException,
) -> AdapterInferenceTimeoutError | None:
    """Return the watchdog error embedded in an exception chain, if any."""
    for chained_error in iter_exception_chain(error):
        if isinstance(chained_error, AdapterInferenceTimeoutError):
            return chained_error
    return None


def is_recoverable_vllm_engine_error(error: BaseException) -> bool:
    """Recognize a dead/poisoned vLLM EngineCore, not model-result errors.

    An illegal CUDA access permanently poisons the EngineCore child process.
    vLLM may surface either the original ``AcceleratorError`` or only a later
    ``EngineDeadError`` through its MP client, depending on which request first
    observes the failure. Both require a fresh engine. Generic numerical or
    validation failures intentionally do not match this classifier. A
    specifically typed malformed requested-logprob response is replayed
    because vLLM violated the request contract and the active search state
    cannot safely advance.
    """
    fatal_fragments = (
        "illegal memory access",
        "device-side assert",
        "cudaerrorassert",
        "cudaerrorillegaladdress",
        "enginecore encountered an issue",
        "enginecore encountered a fatal error",
        "engine core proc failed",
        "engine core is dead",
        "enginecore is dead",
    )
    for chained_error in iter_exception_chain(error):
        if isinstance(chained_error, RecoverableVllmResponseError):
            return True
        type_name = type(chained_error).__name__.lower()
        message = str(chained_error).lower()
        if type_name in {"enginedeaderror", "acceleratorerror"}:
            return True
        if isinstance(chained_error, AssertionError) and any(
            "/vllm/" in frame.filename.replace("\\", "/")
            for frame in traceback.extract_tb(chained_error.__traceback__)
        ):
            # An internal vLLM invariant (for example the scheduler's
            # prompt-logprob assertion) means EngineCore state is corrupt.
            return True
        if any(fragment in message for fragment in fatal_fragments):
            return True
    return False


def vllm_engine_core_dead(engine: Any) -> bool:
    """Best-effort check that the EngineCore child is gone or flagged dead."""
    core = getattr(getattr(engine, "llm_engine", None), "engine_core", None)
    resources = getattr(core, "resources", None)
    if bool(getattr(resources, "engine_dead", False)):
        return True
    processes = getattr(getattr(resources, "engine_manager", None), "processes", None)
    try:
        return any(not process.is_alive() for process in (processes or ()))
    except (AttributeError, OSError, ValueError):
        return False


def run_engine_call_with_liveness(
    engine: Any,
    call: Any,
    *args: Any,
    poll_seconds: float = 2.0,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run one blocking vLLM client call without letting a dead core hang it.

    ``SIGALRM`` polls EngineCore liveness (and an optional hard timeout) and
    raises ``EngineCoreDeadError`` in the main thread, so the normal recovery
    path can restart the engine.  Must be called from the main thread.
    """
    if vllm_engine_core_dead(engine):
        raise EngineCoreDeadError("vLLM engine core is dead")
    started = time.monotonic()

    def handle_poll(_signum, _frame) -> None:
        if vllm_engine_core_dead(engine):
            raise EngineCoreDeadError("vLLM engine core is dead")
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise EngineCoreDeadError(
                f"vLLM engine core is dead or unresponsive after "
                f"{timeout_seconds:.1f}s in a blocking client call"
            )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, handle_poll)
    signal.setitimer(signal.ITIMER_REAL, poll_seconds, poll_seconds)
    try:
        return call(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_delay, previous_interval = previous_timer
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_delay - elapsed),
                previous_interval,
            )


def mark_pending_slot_held(job: dict[str, Any]) -> None:
    """Record ownership of the bounded publication slot on queue receipt."""
    job[_PENDING_SLOT_HELD] = True


def release_job_pending_slot(job: dict[str, Any], semaphore: Any) -> bool:
    """Release a job's pending slot at most once.

    Recovered active jobs are put back through normal admission, but their
    publication slot was already returned during the first engine load.
    """
    if not bool(job.get(_PENDING_SLOT_HELD, False)):
        return False
    semaphore.release()
    job[_PENDING_SLOT_HELD] = False
    return True


def increment_adapter_engine_failures(job: dict[str, Any]) -> int:
    """Increment and return one adapter's fatal EngineCore incident count."""
    failures = int(job.get(_ENGINE_FAILURE_COUNT, 0)) + 1
    job[_ENGINE_FAILURE_COUNT] = failures
    return failures


def record_adapter_engine_failure(
    job: dict[str, Any],
    max_recoveries: int,
) -> tuple[int, bool]:
    """Record one failure and say whether another replay is allowed."""
    max_recoveries = int(max_recoveries)
    if max_recoveries < 1:
        raise ValueError("max_recoveries must be positive")
    failures = increment_adapter_engine_failures(job)
    return failures, failures <= max_recoveries


def increment_adapter_timeout_count(job: dict[str, Any]) -> int:
    """Increment one adapter's watchdog count, separate from engine failures."""
    timeouts = int(job.get(_ADAPTER_TIMEOUT_COUNT, 0)) + 1
    job[_ADAPTER_TIMEOUT_COUNT] = timeouts
    return timeouts


def record_adapter_timeout(
    job: dict[str, Any],
    max_restarts: int = _MAX_ADAPTER_TIMEOUT_RESTARTS,
) -> tuple[int, bool]:
    """Record a watchdog expiry and report whether this adapter is replayed."""
    max_restarts = int(max_restarts)
    if max_restarts < 0:
        raise ValueError("max_restarts must be non-negative")
    timeouts = increment_adapter_timeout_count(job)
    return timeouts, timeouts <= max_restarts


def resolve_active_dfs_mode(
    requested_mode: str,
    *,
    num_test_outputs: int,
    max_train_tokens: int,
    frontier_max_prompt_tokens: int,
    leaf_rollout: bool = False,
    leaf_rollout_prefer_recursive: bool = False,
) -> str:
    """Resolve ``auto`` without changing either explicit search policy.

    Whole-leaf rollout accelerates recursive DFS, not frontier search.  A
    backend may therefore opt into recursive search for ``auto`` only when its
    leaf implementation is active.  vLLM and exact SGLang retain the historical
    short-task frontier heuristic unless they explicitly request this policy.
    """
    requested_mode = str(requested_mode)
    if requested_mode != "auto":
        return requested_mode
    if leaf_rollout and leaf_rollout_prefer_recursive:
        return "recursive"
    return (
        "frontier"
        if int(num_test_outputs) == 1
        and int(max_train_tokens) <= int(frontier_max_prompt_tokens)
        else "recursive"
    )


def resolve_task_search_deadline(
    now: float,
    *,
    inference_deadline: float,
    global_end_time: float,
    search_limit_seconds: float = 540.0,
) -> float:
    """Bound one threshold group by task/global wall-time deadlines.

    The group budget resets for the next augmentation group, never for a
    rollout chunk or branch. Concurrent adapters do not pause these clocks.
    """
    return min(
        float(now) + normalize_vllm_search_limit_seconds(search_limit_seconds),
        float(inference_deadline),
        float(global_end_time),
    )


def shutdown_vllm_engine(engine: Any, timeout_seconds: float = 3.0) -> bool:
    """Stop vLLM's EngineCore process so it cannot outlive its worker.

    vLLM V1 places CUDA ownership and its output pipes in an EngineCore child.
    Terminating only this scheduler process can leave that child holding the
    notebook's stdout pipe open.  Return False only for an engine layout that
    exposes no shutdown API.
    """
    llm_engine = getattr(engine, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if not callable(shutdown):
        return False
    try:
        shutdown(timeout=float(timeout_seconds))
    except TypeError:
        # Keep compatibility with older in-process clients whose shutdown
        # method predates the timeout keyword.
        shutdown()
    return True


def query_visible_gpu_memory_mb() -> int:
    """Sample physical GPU memory even when vLLM owns CUDA in a child.

    vLLM V1 keeps its CUDA allocations in EngineCore, so
    ``torch.cuda.max_memory_allocated`` in this scheduler process always
    reports zero.  At process launch ``CUDA_VISIBLE_DEVICES`` contains the
    single physical index or UUID selected by ``starter.py``; nvidia-smi can
    therefore sample the actual engine device without initializing CUDA here.
    """
    selector = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
    if not selector:
        selector = "0"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # A GPU that just faulted can wedge nvidia-smi indefinitely.
            timeout=10.0,
        )
        samples = [
            int(float(line.strip()))
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        return max(samples, default=-1)
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1


def release_pending_adapter_artifact(
    job: dict[str, Any],
    adapter_root: str,
    pending_adapter_slots: Any,
    adapter_staging_slots: Any = None,
) -> None:
    """Delete one staged handoff before returning its bounded queue slot."""
    remove_job_adapter_artifact(
        job,
        adapter_root,
        adapter_staging_slots,
    )
    pending_adapter_slots.release()


def rebalance_pending_jobs_after_training(
    *,
    pending_jobs: deque[dict[str, Any]],
    service_index: int,
    initial_service_count: int,
    all_job_queues: Any,
    service_outstanding: Any,
    service_load_lock: Any,
    rebalance_count: Any,
    rebalance_complete_event: Any,
) -> dict[str, Any]:
    """Redistribute one initial service's queued adapters after training.

    Active adapters stay on their warm engine.  Only jobs that have not yet
    been admitted are eligible, and each source balances between itself and
    the newly activated post-training services.  The latter do not start
    their engines until every initial service has completed this pass.
    """
    service_index = int(service_index)
    initial_service_count = int(initial_service_count)
    if not 0 <= service_index < initial_service_count:
        raise ValueError("Only an initial inference service can rebalance its backlog")
    if len(all_job_queues) != len(service_outstanding):
        raise ValueError(
            "Service queues and outstanding counters must have equal length"
        )
    post_service_indices = tuple(range(initial_service_count, len(all_job_queues)))
    if not post_service_indices:
        raise ValueError("Post-training scale-out requires future services")

    eligible = (service_index, *post_service_indices)
    retained: deque[dict[str, Any]] = deque()
    migrated_by_service: dict[int, int] = defaultdict(int)
    original_pending = len(pending_jobs)
    while pending_jobs:
        job = pending_jobs.popleft()
        target_service_index, _loads = rebalance_pending_service_task(
            service_index,
            eligible,
            service_outstanding,
            service_load_lock,
        )
        if target_service_index == service_index:
            retained.append(job)
            continue

        moved_job = dict(job)
        moved_job["service_index"] = target_service_index
        moved_job["post_training_source_service_index"] = service_index
        moved_job["post_training_rebalanced"] = True
        try:
            all_job_queues[target_service_index].put(moved_job)
        except BaseException:
            transfer_service_task_assignment(
                target_service_index,
                service_index,
                service_outstanding,
                service_load_lock,
            )
            retained.append(job)
            while pending_jobs:
                retained.append(pending_jobs.popleft())
            pending_jobs.extend(retained)
            raise
        migrated_by_service[target_service_index] += 1

    pending_jobs.extend(retained)
    # Future queues already contain trainer_done messages.  Send one sentinel
    # from every initial service *after* all of that source's migrated jobs.
    # multiprocessing.Queue preserves per-producer ordering, so a future
    # scheduler cannot mistake trainer completion for an empty final queue.
    for target_service_index in post_service_indices:
        all_job_queues[target_service_index].put(
            {
                "kind": "post_training_rebalance_done",
                "source_service_index": service_index,
                "service_index": target_service_index,
            }
        )
    with rebalance_count.get_lock():
        rebalance_count.value = int(rebalance_count.value) + 1
        completed_services = int(rebalance_count.value)
        if completed_services > initial_service_count:
            raise RuntimeError(
                "An initial inference service rebalanced its backlog twice"
            )
        if completed_services == initial_service_count:
            rebalance_complete_event.set()

    return {
        "original_pending": original_pending,
        "retained_pending": len(retained),
        "migrated_by_service": dict(migrated_by_service),
        "completed_services": completed_services,
        "service_loads": service_load_snapshot(
            service_outstanding,
            service_load_lock,
        ),
    }


def ensure_arc_tokenizer_backend(tokenizer, model_path):
    """Restore the checkpoint WordLevel tokenizer on Transformers 4.x/5.x."""
    probe = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    expected_ids = [14, 11, 10, 15, 10, 14, 12, 10]
    if tokenizer.encode(probe, add_special_tokens=False) == expected_ids:
        return tokenizer
    from tokenizers import Tokenizer

    if not hasattr(tokenizer, "_tokenizer"):
        raise RuntimeError(
            "The tokenizer dropped ARC control tokens and its backend cannot "
            "be replaced"
        )
    tokenizer._tokenizer = Tokenizer.from_file(
        os.path.join(model_path, "tokenizer.json")
    )
    if tokenizer.encode(probe, add_special_tokens=False) != expected_ids:
        raise RuntimeError("Failed to restore ARC chat control tokens")
    return tokenizer


def build_inference_batches(test_id_to_subkeys, batch_size):
    """Build decode batches up to any positive vLLM request capacity."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Inference batch size must be positive")
    if batch_size == 4:
        batches = []
        for offset_groups in (
            ((0, 4), (2, 6)),
            ((8, 12), (10, 14)),
        ):
            for subkeys in test_id_to_subkeys.values():
                for offsets in offset_groups:
                    batch = []
                    for offset in offsets:
                        batch.extend(subkeys[offset : offset + 2])
                    if batch:
                        batches.append(batch)
        return batches
    if batch_size == 8:
        offset_groups = ((0, 4, 10, 14), (2, 6, 8, 12))
    else:
        # vLLM accepts independently sized prompts and sampling parameters.
        # Preserve the established 4/8 ordering above, while treating every
        # other value as an ordinary maximum request count.
        return [
            list(subkeys[offset : offset + batch_size])
            for subkeys in test_id_to_subkeys.values()
            for offset in range(0, len(subkeys), batch_size)
            if subkeys[offset : offset + batch_size]
        ]
    batches = []
    for subkeys in test_id_to_subkeys.values():
        for offsets in offset_groups:
            batch = []
            for offset in offsets:
                batch.extend(subkeys[offset : offset + 2])
            if batch:
                batches.append(batch)
    return batches


def resolve_kv_resident_inference_batch_size(
    requested_batch_size,
    prefix_lengths,
    *,
    cache_capacity_tokens=None,
    enabled=True,
    resident_fraction=0.95,
):
    """Keep one augmented-root batch inside the profiled GPU KV cache.

    A request group whose roots exceed the cache has a pathological access
    pattern under chunked leaf rollout: every continuation evicts the roots
    needed by the next continuation, so every 32-token chunk re-prefills the
    complete prompts.  Resolve a smaller established 16/8/4/2/1 grouping from
    vLLM's measured cache capacity while leaving batches that already fit
    byte-for-byte unchanged.
    """
    requested_batch_size = int(requested_batch_size)
    if requested_batch_size <= 0:
        raise ValueError("Inference batch size must be positive")
    resident_fraction = float(resident_fraction)
    if not 0.0 < resident_fraction <= 1.0:
        raise ValueError("KV-resident fraction must be in (0, 1]")
    lengths = [int(length) for length in prefix_lengths]
    if any(length <= 0 for length in lengths):
        raise ValueError("Inference prefix lengths must be positive")
    if not enabled or not lengths or cache_capacity_tokens is None:
        return requested_batch_size
    cache_capacity_tokens = int(cache_capacity_tokens)
    if cache_capacity_tokens <= 0:
        return requested_batch_size

    resident_budget = max(1, int(cache_capacity_tokens * resident_fraction))
    max_resident_requests = max(1, resident_budget // max(lengths))
    if requested_batch_size <= max_resident_requests:
        return requested_batch_size

    maximum = min(requested_batch_size, max_resident_requests)
    if requested_batch_size in {2, 4, 8, 16}:
        # These sizes have deliberate augmentation orderings in
        # build_inference_batches.  Preserve one of those tested orderings.
        for candidate in (16, 8, 4, 2, 1):
            if candidate <= maximum:
                return candidate
    return maximum


def split_inference_batches_by_context(
    batches,
    inference_prefixes,
    decode_token_budgets,
    *,
    allow_mixed_prompt_lengths=False,
):
    """Filter batches and preserve each search's shared decode budget."""
    compatible_batches = []
    for batch in batches:
        eligible = [subkey for subkey in batch if subkey in inference_prefixes]
        if allow_mixed_prompt_lengths:
            grouped = {}
            for subkey in eligible:
                grouped.setdefault(int(decode_token_budgets[subkey]), []).append(
                    subkey
                )
            compatible_batches.extend(
                group for group in grouped.values() if group
            )
            continue
        grouped = {}
        for subkey in eligible:
            signature = (
                len(inference_prefixes[subkey]),
                int(decode_token_budgets[subkey]),
            )
            grouped.setdefault(signature, []).append(subkey)
        compatible_batches.extend(group for group in grouped.values() if group)
    return compatible_batches


def infer_same_shape_reply(query):
    """Backward-compatible wrapper for conservative shape inference."""
    return infer_demonstrated_output_shape(query)


@dataclass
class _ScoreAggregate:
    grid_id: tuple[Any, ...]
    values: list[float | None]

    @property
    def complete(self) -> bool:
        return all(value is not None for value in self.values)

    def result(self) -> list[float]:
        if not self.complete:
            raise RuntimeError("Score aggregate is incomplete")
        return [float(value) for value in self.values if value is not None]


@dataclass
class _ScoreJob:
    request_id: int
    prompt_token_ids: tuple[int, ...]
    query_length: int
    answer_token_ids: tuple[int, ...]
    aggregate: _ScoreAggregate
    aggregate_index: int
    allow_prefix_cache: bool = True
    barrier_after: bool = False


@dataclass
class _CandidateEntry:
    beam_score: float
    grid_id: tuple[Any, ...]
    solution: np.ndarray


@dataclass
class _EngineWork:
    task: "ArcVllmTask"
    kind: str
    request_id: int
    prompt_token_ids: tuple[int, ...]
    max_tokens: int = 1
    constrained_rollout: bool = False
    cumulative_nll: float = 0.0
    score_job: _ScoreJob | None = None
    lora_request: Any = field(init=False, repr=False)
    model_adapter_id: int = field(init=False)
    inference_model: str = field(init=False)

    def __post_init__(self) -> None:
        # A response can advance its task from LoRA to base inference. Keep
        # request identity stable for dispatch, exact replay and telemetry.
        self.lora_request = getattr(self.task, "lora_request", None)
        self.model_adapter_id = (
            0 if hasattr(self.task, "lora_request") and self.lora_request is None
            else int(getattr(self.lora_request, "lora_int_id", getattr(self.task, "job", {}).get("adapter_id", -1)))
        )
        self.inference_model = (
            "base" if self.model_adapter_id == 0 or getattr(self.task, "base_fallback_active", False) else "lora"
        )


def _bounded_repr(value: Any, limit: int = 512) -> str:
    """Return a diagnostic repr without letting one object flood the log."""
    try:
        result = repr(value)
    except BaseException as error:
        result = f"<repr failed: {type(error).__name__}: {error}>"
    if len(result) <= limit:
        return result
    return result[: max(0, limit - 18)] + "...<truncated>"


def _prompt_fingerprint(token_ids: Any) -> dict[str, Any]:
    """Fingerprint a prompt while retaining only a bounded token sample."""
    try:
        normalized = tuple(map(int, token_ids))
    except BaseException as error:
        return {
            "type": type(token_ids).__name__,
            "error": f"{type(error).__name__}: {error}",
        }
    digest = hashlib.sha256(
        b"".join(
            int(token_id).to_bytes(8, byteorder="little", signed=True)
            for token_id in normalized
        )
    ).hexdigest()
    return {
        "length": len(normalized),
        "sha256": digest,
        "head": list(normalized[:16]),
        "tail": list(normalized[-32:]),
    }


def _logprob_value_summary(value: Any) -> dict[str, Any]:
    """Serialize the useful, CPU-side fields of one vLLM Logprob object."""
    result: dict[str, Any] = {"type": type(value).__name__}
    raw_logprob = getattr(value, "logprob", value)
    try:
        logprob = float(raw_logprob)
        result["logprob"] = logprob
        result["finite"] = math.isfinite(logprob)
    except BaseException as error:
        result["logprob_error"] = f"{type(error).__name__}: {error}"
        result["repr"] = _bounded_repr(value)
    for name in ("rank", "decoded_token"):
        try:
            attr = getattr(value, name)
        except BaseException:
            continue
        if attr is not None:
            result[name] = attr if isinstance(attr, (int, float, str, bool)) else _bounded_repr(attr)
    return result


def _logprob_container_summary(
    logprobs: Any,
    *,
    include_values: bool,
    max_positions: int = 64,
) -> dict[str, Any]:
    """Summarize FlatLogprobs/list output without assuming its concrete type."""
    result: dict[str, Any] = {"type": type(logprobs).__name__}
    if logprobs is None:
        result["is_none"] = True
        return result
    try:
        length = len(logprobs)
    except BaseException as error:
        result["length_error"] = f"{type(error).__name__}: {error}"
        result["repr"] = _bounded_repr(logprobs)
        return result
    result["length"] = int(length)
    if length <= max_positions:
        positions = range(length)
    else:
        positions = (*range(4), *range(length - 4, length))
        result["positions_truncated"] = True
    position_summaries = []
    for position in positions:
        item: dict[str, Any] = {"position": int(position)}
        try:
            values = logprobs[position]
        except BaseException as error:
            item["error"] = f"{type(error).__name__}: {error}"
            position_summaries.append(item)
            continue
        item["mapping_type"] = type(values).__name__
        if values is None:
            item["is_none"] = True
            position_summaries.append(item)
            continue
        if not isinstance(values, Mapping):
            item["repr"] = _bounded_repr(values)
            position_summaries.append(item)
            continue
        normalized: dict[int, Any] = {}
        bad_keys = []
        for key, value in values.items():
            try:
                normalized[int(key)] = value
            except BaseException:
                bad_keys.append(_bounded_repr(key, limit=128))
        keys = sorted(normalized)
        item["size"] = len(values)
        item["token_ids"] = keys
        item["missing_arc_token_ids"] = [
            int(token_id) for token_id in ARC_TOKENS if token_id not in normalized
        ]
        if bad_keys:
            item["non_integer_keys"] = bad_keys
        if include_values:
            item["values"] = {
                str(token_id): _logprob_value_summary(normalized[token_id])
                for token_id in keys
            }
        position_summaries.append(item)
    result["positions"] = position_summaries
    return result


def _sampling_params_summary(params: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(params).__name__}
    for name in (
        "max_tokens",
        "min_tokens",
        "temperature",
        "ignore_eos",
        "logprobs",
        "prompt_logprobs",
        "flat_logprobs",
        "skip_reading_prefix_cache",
        "include_stop_str_in_output",
    ):
        try:
            value = getattr(params, name)
        except BaseException:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[name] = value
        else:
            result[name] = _bounded_repr(value)
    for name in ("allowed_token_ids", "logprob_token_ids", "stop_token_ids"):
        try:
            value = getattr(params, name)
        except BaseException:
            continue
        if value is None:
            result[name] = None
            continue
        try:
            result[name] = list(map(int, value))
        except BaseException:
            result[name] = _bounded_repr(value)
    try:
        extra_args = getattr(params, "extra_args")
    except BaseException:
        extra_args = None
    if extra_args:
        result["extra_args"] = _bounded_repr(extra_args, limit=1024)
    return result


def _engine_work_summary(work: _EngineWork) -> dict[str, Any]:
    job = getattr(work.task, "job", {})
    result = {
        "task_key": str(getattr(work.task, "key", "<unknown>")),
        "kind": str(work.kind),
        "request_id": int(work.request_id),
        "adapter_id": work.model_adapter_id,
        "job_adapter_id": int(job.get("adapter_id", -1)),
        "adapter_name": str(job.get("adapter_name", "<unknown>")),
        "inference_model": work.inference_model,
        "prompt": _prompt_fingerprint(work.prompt_token_ids),
        "max_tokens": int(work.max_tokens),
        "constrained_rollout": bool(work.constrained_rollout),
        "cumulative_nll": float(work.cumulative_nll),
    }
    if work.score_job is not None:
        result["score"] = {
            "query_length": int(work.score_job.query_length),
            "answer_tokens": len(work.score_job.answer_token_ids),
            "allow_prefix_cache": bool(work.score_job.allow_prefix_cache),
            "barrier_after": bool(work.score_job.barrier_after),
        }
    return result


def _engine_output_summary(output: Any, *, include_values: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(output).__name__}
    for name in ("request_id", "finished", "num_cached_tokens"):
        try:
            value = getattr(output, name)
        except BaseException:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[name] = value
        else:
            result[name] = _bounded_repr(value)
    prompt_ids = getattr(output, "prompt_token_ids", None)
    if prompt_ids is not None:
        result["prompt"] = _prompt_fingerprint(prompt_ids)
    metrics = getattr(output, "metrics", None)
    if metrics is not None:
        metric_values = {}
        for name in (
            "arrival_time",
            "first_scheduled_time",
            "first_token_time",
            "time_in_queue",
            "finished_time",
            "scheduler_time",
            "model_forward_time",
            "model_execute_time",
            "num_preemptions",
        ):
            try:
                value = getattr(metrics, name)
            except BaseException:
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                metric_values[name] = value
        result["metrics"] = metric_values
    samples = []
    try:
        raw_samples = list(getattr(output, "outputs", None) or [])
    except BaseException as error:
        result["outputs_error"] = f"{type(error).__name__}: {error}"
        raw_samples = []
    for sample_index, sample in enumerate(raw_samples):
        sample_result: dict[str, Any] = {
            "index": sample_index,
            "type": type(sample).__name__,
        }
        try:
            token_ids = tuple(map(int, sample.token_ids))
            sample_result["token_ids"] = list(token_ids[:256])
            sample_result["token_count"] = len(token_ids)
            if len(token_ids) > 256:
                sample_result["token_ids_truncated"] = True
        except BaseException as error:
            sample_result["token_ids_error"] = f"{type(error).__name__}: {error}"
        for name in ("finish_reason", "stop_reason", "cumulative_logprob"):
            try:
                value = getattr(sample, name)
            except BaseException:
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                sample_result[name] = value
            else:
                sample_result[name] = _bounded_repr(value)
        sample_result["logprobs"] = _logprob_container_summary(
            getattr(sample, "logprobs", None),
            include_values=include_values,
        )
        samples.append(sample_result)
    result["outputs"] = samples
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        # Teacher-forced score responses can contain thousands of positions.
        # Their length/type is enough to distinguish a decode/score mismatch;
        # do not copy every score into the forensic artifact.
        result["prompt_logprobs"] = _logprob_container_summary(
            prompt_logprobs,
            include_values=False,
            max_positions=8,
        )
    return result


def _engine_output_brief_summary(output: Any) -> dict[str, Any]:
    """Summarize a successful response without scanning token dictionaries."""
    result: dict[str, Any] = {"type": type(output).__name__}
    for name in ("request_id", "finished", "num_cached_tokens"):
        try:
            value = getattr(output, name)
        except BaseException:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[name] = value
    metrics = getattr(output, "metrics", None)
    if metrics is not None:
        try:
            result["num_preemptions"] = getattr(metrics, "num_preemptions")
        except BaseException:
            pass
    samples = []
    try:
        raw_samples = list(getattr(output, "outputs", None) or [])
    except BaseException as error:
        result["outputs_error"] = f"{type(error).__name__}: {error}"
        raw_samples = []
    for sample in raw_samples:
        sample_result: dict[str, Any] = {"type": type(sample).__name__}
        for source_name, result_name in (
            ("token_ids", "token_count"),
            ("logprobs", "logprob_positions"),
        ):
            try:
                sample_result[result_name] = len(getattr(sample, source_name))
            except BaseException:
                sample_result[result_name] = None
        for name in ("finish_reason", "stop_reason"):
            try:
                value = getattr(sample, name)
            except BaseException:
                continue
            if isinstance(value, (int, float, str, bool)) or value is None:
                sample_result[name] = value
        samples.append(sample_result)
    result["outputs"] = samples
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        try:
            result["prompt_logprob_positions"] = len(prompt_logprobs)
        except BaseException:
            result["prompt_logprob_positions"] = None
    return result


class _VllmForensics:
    """Opt-in, line-buffered evidence for rare long-run vLLM failures."""

    def __init__(self, service_index: int, directory: str | None):
        self.service_index = int(service_index)
        self.call_index = 0
        self.recent_calls: deque[dict[str, Any]] = deque(maxlen=32)
        self.path: Path | None = None
        self._stream = None
        if directory:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            self.path = root / f"service-{self.service_index}-pid-{os.getpid()}.jsonl"
            self._stream = self.path.open("a", buffering=1)
            self._write(
                {
                    "type": "forensic_start",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "service_index": self.service_index,
                }
            )

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def _write(self, event: dict[str, Any]) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(event, sort_keys=True, default=_bounded_repr) + "\n")

    def begin_call(
        self,
        works: list[_EngineWork],
        sampling_params: list[Any],
        *,
        active: list[Any],
        pending_jobs: int,
        loaded_loras: Any,
        options: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        self.call_index += 1
        event = {
            "type": "call_start",
            "timestamp": time.time(),
            "service_index": self.service_index,
            "call_index": self.call_index,
            "works": [_engine_work_summary(work) for work in works],
            "sampling_params": [
                _sampling_params_summary(params) for params in sampling_params
            ],
            "active": [
                {
                    "task_key": str(getattr(task, "key", "<unknown>")),
                    "adapter_id": int(getattr(task, "job", {}).get("adapter_id", -1)),
                }
                for task in active
            ],
            "pending_jobs": int(pending_jobs),
            "loaded_loras": _bounded_repr(loaded_loras, limit=1024),
            "runtime": {
                name: options.get(name)
                for name in (
                    "max_seq_length",
                    "vllm_max_num_seqs",
                    "vllm_max_num_batched_tokens",
                    "vllm_max_loras",
                    "vllm_enforce_eager",
                    "vllm_async_scheduling",
                    "vllm_batch_invariant",
                    "vllm_leaf_rollout",
                    "vllm_rollout_max_tokens",
                    "vllm_cumulative_nll_stop",
                    "vllm_prefix_score",
                )
            },
        }
        self.recent_calls.append(event)
        self._write(event)

    def record_result(self, outputs: list[Any], elapsed: float) -> None:
        if not self.enabled:
            return
        self._write(
            {
                "type": "call_result",
                "timestamp": time.time(),
                "service_index": self.service_index,
                "call_index": self.call_index,
                "engine_seconds": float(elapsed),
                "outputs": [
                    _engine_output_brief_summary(output)
                    for output in outputs
                ],
            }
        )

    def record_failure(
        self,
        phase: str,
        error: BaseException,
        *,
        works: list[_EngineWork] | None = None,
        outputs: list[Any] | None = None,
        sampling_params: list[Any] | None = None,
        active: list[Any] | None = None,
        pending_jobs: int | None = None,
        loaded_loras: Any = None,
        gpu_memory_mb: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "type": "failure",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "service_index": self.service_index,
            "call_index": self.call_index,
            "phase": str(phase),
            "exception_chain": [
                {
                    "type": type(chained).__name__,
                    "message": str(chained),
                }
                for chained in iter_exception_chain(error)
            ],
            "works": [
                _engine_work_summary(work) for work in (works or [])
            ],
            "sampling_params": [
                _sampling_params_summary(params)
                for params in (sampling_params or [])
            ],
            "outputs": [
                _engine_output_summary(output, include_values=True)
                for output in (outputs or [])
            ],
            "active": [
                {
                    "task_key": str(getattr(task, "key", "<unknown>")),
                    "adapter_id": int(getattr(task, "job", {}).get("adapter_id", -1)),
                    "adapter_name": str(
                        getattr(task, "job", {}).get("adapter_name", "<unknown>")
                    ),
                }
                for task in (active or [])
            ],
            "pending_jobs": pending_jobs,
            "loaded_loras": _bounded_repr(loaded_loras, limit=1024),
            "gpu_memory_mb": gpu_memory_mb,
            "recent_calls": list(self.recent_calls),
        }
        self._write(event)
        # Emit the same self-contained event to the combined notebook log. The
        # JSONL path is preferable, but this preserves evidence if a later
        # sibling shutdown prevents the user from downloading the file.
        print(
            f"[vLLM {self.service_index}] ARC_VLLM_FORENSIC_FAILURE "
            + json.dumps(event, sort_keys=True, default=_bounded_repr),
            flush=True,
        )

    def close(self) -> None:
        if self._stream is None:
            return
        self._write(
            {
                "type": "forensic_end",
                "timestamp": time.time(),
                "service_index": self.service_index,
                "call_index": self.call_index,
            }
        )
        self._stream.close()
        self._stream = None


class _ServiceTelemetry:
    """Low-overhead service-call telemetry for fixed-adapter benchmarks."""

    def __init__(self, service_index: int, telemetry_dir: str | None = None):
        self.service_index = int(service_index)
        self.calls = 0
        self.requests = 0
        self.decode_requests = 0
        self.score_requests = 0
        self.mixed_calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.generated_tokens = 0
        self.engine_seconds = 0.0
        self.prepare_seconds = 0.0
        self.response_seconds = 0.0
        self.max_batch_size = 0
        self.max_task_count = 0
        self.max_adapter_count = 0
        self._stream = None
        if telemetry_dir:
            directory = Path(telemetry_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"service-{self.service_index}.jsonl"
            self._stream = path.open("x", buffering=1)

    def record(
        self,
        *,
        phase: str,
        kinds: list[str],
        prompt_lengths: list[int],
        cached_tokens: list[int],
        generated_tokens: int,
        adapter_ids: list[int],
        task_keys: list[str],
        max_tokens: list[int],
        engine_seconds: float,
        prepare_seconds: float = 0.0,
        response_seconds: float = 0.0,
    ) -> None:
        batch_size = len(prompt_lengths)
        if not (
            len(kinds)
            == len(cached_tokens)
            == len(adapter_ids)
            == len(task_keys)
            == len(max_tokens)
            == batch_size
        ):
            raise ValueError("Telemetry request fields have different lengths")
        decode_requests = sum(kind == "decode" for kind in kinds)
        score_requests = sum(kind == "score" for kind in kinds)
        unique_adapters = len(set(adapter_ids))
        unique_tasks = len(set(task_keys))
        prompt_total = sum(map(int, prompt_lengths))
        cached_total = sum(map(int, cached_tokens))
        sorted_prompt_lengths = sorted(map(int, prompt_lengths))
        if sorted_prompt_lengths:
            middle = len(sorted_prompt_lengths) // 2
            if len(sorted_prompt_lengths) % 2:
                prompt_median = float(sorted_prompt_lengths[middle])
            else:
                prompt_median = (
                    sorted_prompt_lengths[middle - 1]
                    + sorted_prompt_lengths[middle]
                ) / 2.0
        else:
            prompt_median = 0.0
        max_token_histogram = {
            str(value): max_tokens.count(value) for value in sorted(set(max_tokens))
        }
        event = {
            "type": "engine_call",
            "service_index": self.service_index,
            "call_index": self.calls + 1,
            "phase": str(phase),
            "batch_size": batch_size,
            "decode_requests": decode_requests,
            "score_requests": score_requests,
            "mixed_decode_score": bool(decode_requests and score_requests),
            "task_count": unique_tasks,
            "adapter_count": unique_adapters,
            "prompt_tokens": prompt_total,
            "prompt_tokens_min": min(sorted_prompt_lengths, default=0),
            "prompt_tokens_median": prompt_median,
            "prompt_tokens_max": max(sorted_prompt_lengths, default=0),
            "cached_tokens": cached_total,
            "uncached_tokens": max(0, prompt_total - cached_total),
            "generated_tokens": int(generated_tokens),
            "max_tokens": max(max_tokens, default=0),
            "max_tokens_histogram": max_token_histogram,
            "engine_seconds": float(engine_seconds),
            "prepare_seconds": float(prepare_seconds),
            "response_seconds": float(response_seconds),
        }
        self.calls += 1
        self.requests += batch_size
        self.decode_requests += decode_requests
        self.score_requests += score_requests
        self.mixed_calls += int(event["mixed_decode_score"])
        self.prompt_tokens += prompt_total
        self.cached_tokens += cached_total
        self.generated_tokens += int(generated_tokens)
        self.engine_seconds += float(engine_seconds)
        self.prepare_seconds += float(prepare_seconds)
        self.response_seconds += float(response_seconds)
        self.max_batch_size = max(self.max_batch_size, batch_size)
        self.max_task_count = max(self.max_task_count, unique_tasks)
        self.max_adapter_count = max(self.max_adapter_count, unique_adapters)
        if self._stream is not None:
            self._stream.write(json.dumps(event, sort_keys=True) + "\n")

    def record_works(
        self,
        works: list[_EngineWork],
        outputs: list[Any],
        *,
        phase: str,
        engine_seconds: float,
        prepare_seconds: float = 0.0,
        response_seconds: float = 0.0,
    ) -> None:
        generated_tokens = sum(
            len(sample.token_ids)
            for output in outputs
            for sample in (getattr(output, "outputs", None) or [])
        )
        self.record(
            phase=phase,
            kinds=[work.kind for work in works],
            prompt_lengths=[len(work.prompt_token_ids) for work in works],
            cached_tokens=[
                int(getattr(output, "num_cached_tokens", 0) or 0)
                for output in outputs
            ],
            generated_tokens=generated_tokens,
            adapter_ids=[work.model_adapter_id for work in works],
            task_keys=[str(work.task.key) for work in works],
            max_tokens=[int(work.max_tokens) for work in works],
            engine_seconds=engine_seconds,
            prepare_seconds=prepare_seconds,
            response_seconds=response_seconds,
        )

    def summary(self) -> dict[str, int | float]:
        mean_batch = self.requests / self.calls if self.calls else 0.0
        cache_fraction = (
            self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0
        )
        return {
            "calls": self.calls,
            "requests": self.requests,
            "decode_requests": self.decode_requests,
            "score_requests": self.score_requests,
            "mixed_calls": self.mixed_calls,
            "mean_batch_size": mean_batch,
            "max_batch_size": self.max_batch_size,
            "max_task_count": self.max_task_count,
            "max_adapter_count": self.max_adapter_count,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_fraction": cache_fraction,
            "generated_tokens": self.generated_tokens,
            "engine_seconds": self.engine_seconds,
            "prepare_seconds": self.prepare_seconds,
            "response_seconds": self.response_seconds,
        }

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.write(
            json.dumps(
                {"type": "summary", "service_index": self.service_index, **self.summary()},
                sort_keys=True,
            )
            + "\n"
        )
        self._stream.close()
        self._stream = None


def validate_vllm_context_request(
    prompt_token_ids: Any,
    *,
    max_tokens: int,
    max_model_len: int,
    request_label: str,
) -> int:
    """Reject an oversized request before it can poison a vLLM batch.

    vLLM rejects a generation prompt that already fills its model context and
    length-caps a completion when a smaller prompt later reaches that limit.
    ARC keeps that behavior explicit by preserving the prompt and bounding
    every search request by its remaining context. Keep this final boundary
    check because recursive paths grow after initial preprocessing and future
    request types may use a different generation budget.
    """
    prompt_tokens = len(prompt_token_ids)
    max_tokens = int(max_tokens)
    max_model_len = int(max_model_len)
    if max_tokens < 1:
        raise ValueError(
            f"{request_label} has invalid max_tokens={max_tokens}; expected >= 1"
        )
    if max_model_len < 1:
        raise ValueError(
            f"{request_label} has invalid max_model_len={max_model_len}; expected >= 1"
        )
    total_tokens = prompt_tokens + max_tokens
    if total_tokens > max_model_len:
        raise ValueError(
            f"{request_label} exceeds the vLLM context window: "
            f"prompt_tokens={prompt_tokens}, max_tokens={max_tokens}, "
            f"total_tokens={total_tokens}, max_model_len={max_model_len}"
        )
    return total_tokens


def validate_engine_work_context(
    work: _EngineWork,
    max_model_len: int,
) -> int:
    """Validate one ARC decode or teacher-forced score request."""
    return validate_vllm_context_request(
        work.prompt_token_ids,
        max_tokens=work.max_tokens,
        max_model_len=max_model_len,
        request_label=(
            f"ARC {work.kind} request for task {work.task.key} "
            f"(request_id={work.request_id})"
        ),
    )


def _engine_prompt(work: _EngineWork) -> dict[str, Any]:
    """Build one token prompt, isolating a task after numerical recovery."""
    prompt: dict[str, Any] = {
        "prompt_token_ids": list(work.prompt_token_ids),
    }
    cache_salt = getattr(work.task, "decode_cache_salt", None)
    if bool(
        getattr(work.task, "options", {}).get(
            "bidirectional_prompt_attention",
            False,
        )
    ):
        from arc_bidirectional_attention import arc_attention_cache_salt

        # Prefix hashes do not include SamplingParams.extra_args.  Isolate
        # intermittent-attention K/V from causal cache entries and prevent a
        # block hit from resuming inside a bidirectional prompt run.
        attention_salt = arc_attention_cache_salt(work.prompt_token_ids)
        prompt["cache_salt"] = (
            attention_salt
            if work.kind != "decode" or cache_salt is None
            else f"{attention_salt}:recovery-{cache_salt}"
        )
    elif work.kind == "decode" and cache_salt is not None:
        prompt["cache_salt"] = cache_salt
    return prompt


def _sampling_params_with_arc_attention(
    sampling_params: Any,
    prompt_token_ids: tuple[int, ...],
    *,
    enabled: bool,
) -> Any:
    """Attach compact role runs to one vLLM request without mutating a base."""
    if not enabled:
        return sampling_params
    from arc_bidirectional_attention import (
        serialized_arc_attention_segments,
    )
    from arc_vllm_bidirectional import ARC_VLLM_SEGMENTS_EXTRA_KEY

    clone = sampling_params.clone()
    extra_args = dict(getattr(clone, "extra_args", None) or {})
    extra_args[ARC_VLLM_SEGMENTS_EXTRA_KEY] = serialized_arc_attention_segments(
        prompt_token_ids
    )
    clone.extra_args = extra_args
    return clone


class ArcVllmTask:
    """All CPU inference state for one trained task/LoRA adapter."""

    def __init__(
        self,
        *,
        job: dict[str, Any],
        lora_request: Any,
        arc_test_set: ArcDataset,
        tokenizer: Any,
        formatter: QwenFormatter,
        output_dir: str,
        global_end_time: float,
        options: dict[str, Any],
    ) -> None:
        self.job = job
        self.key = str(job["task_key"])
        self.service_index = int(job.get("service_index", 0))
        engine_label = str(options.get("engine_label", "vLLM"))
        self.log_prefix = f"[{engine_label} {self.service_index}]"
        self.lora_request = lora_request
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.output_dir = output_dir
        self.train_prediction_output_dir: str | None = None
        self.global_end_time = float(global_end_time)
        self.options = options
        self.max_new_tokens = formatter.max_new_tokens()
        self.max_score = float(options["max_beam_score"])
        self.requested_inference_batch_size = int(
            options["inference_batch_size"]
        )
        self.inference_batch_size = self.requested_inference_batch_size
        self.score_batch_size = int(options["score_batch_size"])
        if self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        if self.score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")
        if int(options["frontier_origin_batch_size"]) <= 0:
            raise ValueError("frontier_origin_batch_size must be positive")
        if int(options["frontier_max_live_branches"]) <= 0:
            raise ValueError("frontier_max_live_branches must be positive")
        if int(options["frontier_max_prompt_tokens"]) <= 0:
            raise ValueError("frontier_max_prompt_tokens must be positive")
        self.rescore_views = int(options.get("rescore_views", 8))
        self.rescore_schedule = str(options.get("rescore_schedule", "candidate"))
        if self.rescore_views <= 0:
            raise ValueError("rescore_views must be positive")
        if self.rescore_schedule not in {"candidate", "view", "prefix"}:
            raise ValueError(
                "rescore_schedule must be 'candidate', 'view', or 'prefix'"
            )
        self.prefix_score_tile_views = int(
            options.get("vllm_prefix_score_tile_views", 4)
        )
        if self.prefix_score_tile_views <= 0:
            raise ValueError("vllm_prefix_score_tile_views must be positive")
        self.search_limit_seconds = normalize_vllm_search_limit_seconds(
            options.get("vllm_search_limit_seconds", 540.0)
        )
        self.inference_timeout_seconds = float(
            options["inference_timeout_seconds"]
        )
        # Admission happens after loading the LoRA. Queue wait is excluded;
        # preprocessing and time sharing this service with other tasks are not.
        self.started_at = time.time()
        self.inference_deadline = self.started_at + self.inference_timeout_seconds
        self.adapter_watchdog_timeout_seconds = float(
            options.get("vllm_adapter_timeout_seconds", 3600.0)
        )
        if self.adapter_watchdog_timeout_seconds <= 0:
            raise ValueError("vllm_adapter_timeout_seconds must be positive")
        self.adapter_watchdog_deadline = (
            self.started_at + self.adapter_watchdog_timeout_seconds
        )

        puzzle_ds = arc_test_set.change_keys([self.key])
        self.accuracy_query = puzzle_ds.queries[self.key]
        self.accuracy_reported = False
        self.test_output_count = len(puzzle_ds.queries[self.key]["test"])
        self.ground_truth_outputs = job.get("ground_truth_outputs")
        self.pair_expansion_options = dict(
            min_train_pairs=int(options.get("min_train_pairs", 0)),
            extra_train_pair_mode=str(options.get("extra_train_pair_mode", "augment")),
        )
        self.puzzle_ds_multi = puzzle_ds.split_multi_replies()
        self.eval_ds = self.puzzle_ds_multi.augment(n=2, seed=2, **self.pair_expansion_options)
        if self.pair_expansion_options["min_train_pairs"]:
            from arc_train_pair_expansion import write_expansion_manifest

            write_expansion_manifest(
                self.eval_ds, formatter, root=output_dir, task_key=self.key,
                phase="decode", min_train_pairs=self.pair_expansion_options["min_train_pairs"],
                mode=self.pair_expansion_options["extra_train_pair_mode"],
            )
        self.predict_train = bool(options.get("predict_train", False))
        if self.predict_train:
            train_prediction_ds_multi = puzzle_ds.split_train_replies()
            self.puzzle_ds_multi = self.puzzle_ds_multi.append(
                train_prediction_ds_multi
            )
            # Preserve the exact historical test augmentation RNG stream.
            # Diagnostic targets are augmented independently and appended
            # only after the ordinary test views have been materialized.
            self.eval_ds = self.eval_ds.append(
                train_prediction_ds_multi.augment(n=2, seed=2, **self.pair_expansion_options)
            )
            self.train_prediction_output_dir = resolve_train_prediction_output_dir(
                output_dir
            )
            os.makedirs(self.train_prediction_output_dir, exist_ok=True)
        self.max_model_len = int(options["max_seq_length"])
        self.inference_prefixes: dict[str, tuple[int, ...]] = {}
        self.decode_token_budgets: dict[str, int] = {}
        skipped_context_views: list[tuple[str, int]] = []
        for subkey in self.eval_ds.keys:
            sample = self.eval_ds.get(subkey, formatter)
            prefix = tuple(tokenizer.encode(sample["input"]))
            if len(prefix) >= self.max_model_len:
                skipped_context_views.append((subkey, len(prefix)))
                continue
            self.inference_prefixes[subkey] = prefix
            self.decode_token_budgets[subkey] = min(
                self.max_new_tokens,
                self.max_model_len - len(prefix),
            )
        self.context_skipped_views = len(skipped_context_views)
        decode_budgets = tuple(self.decode_token_budgets.values())
        self.decode_budget_range = (
            (min(decode_budgets), max(decode_budgets))
            if decode_budgets
            else None
        )
        if skipped_context_views:
            examples = ", ".join(
                f"{subkey}:{length}"
                for subkey, length in skipped_context_views[:4]
            )
            if len(skipped_context_views) > 4:
                examples += ", ..."
            print(
                f"{self.log_prefix} skipping {len(skipped_context_views)} "
                "inference view(s) whose complete prompt leaves no vLLM "
                f"decode room at max_model_len={self.max_model_len}: "
                f"{examples}",
                flush=True,
            )
        self.kv_cache_capacity_tokens = options.get(
            "vllm_kv_cache_capacity_tokens"
        )
        self.kv_resident_fraction = resolve_vllm_kv_resident_fraction(
            options.get("vllm_kv_resident_fraction", 0.95),
            max_loras=options.get("vllm_max_loras", 1),
        )
        self.inference_batch_size = resolve_kv_resident_inference_batch_size(
            self.requested_inference_batch_size,
            (len(prefix) for prefix in self.inference_prefixes.values()),
            cache_capacity_tokens=self.kv_cache_capacity_tokens,
            enabled=bool(options.get("vllm_kv_resident_batching", True)),
            resident_fraction=self.kv_resident_fraction,
        )
        self.test_id_to_subkeys: dict[str, list[str]] = defaultdict(list)
        for subkey in sorted(self.eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            self.test_id_to_subkeys[test_id].append(subkey)
        self.rescore_seeds = {
            subkeys[0].split(".")[0]: stable_task_seed(
                subkeys[0].split(".")[0],
                seed=int(os.getenv("ARC_RESCORE_SEED", "0")),
                namespace="rescore",
            )
            for subkeys in self.test_id_to_subkeys.values()
        }
        unfiltered_batches = build_inference_batches(
            self.test_id_to_subkeys,
            self.inference_batch_size,
        )
        self.batches = split_inference_batches_by_context(
            unfiltered_batches,
            self.inference_prefixes,
            self.decode_token_budgets,
            allow_mixed_prompt_lengths=True,
        )
        requested_mode = str(options["dfs_mode"])
        self.active_dfs_mode = resolve_active_dfs_mode(
            requested_mode,
            num_test_outputs=len(self.test_id_to_subkeys),
            max_train_tokens=int(job["max_train_tokens"]),
            frontier_max_prompt_tokens=int(options["frontier_max_prompt_tokens"]),
            leaf_rollout=bool(options.get("vllm_leaf_rollout", False)),
            leaf_rollout_prefer_recursive=bool(
                options.get("leaf_rollout_prefer_recursive", False)
            ),
        )

        self.batch_index = 0
        self.fallback_groups: (
            list[tuple[tuple[int, int], list[tuple[str, tuple[int, ...]]]]] | None
        ) = None
        self.fallback_index = 0
        self.current_subkeys: list[str] = []
        self.current_phase = ""
        self.search: Any | None = None
        self.pending_output_groups: list[tuple[str, list[_CandidateEntry]]] = []
        self.known_scores: dict[tuple[Any, ...], list[float]] = {}
        self.score_seed_by_grid: dict[tuple[Any, ...], int | None] = {}
        self.pending_aggregates: dict[tuple[Any, ...], _ScoreAggregate] = {}
        self.report_output_nll = bool(
            options.get("report_output_nll", False)
        )
        self.output_nll_report: dict[str, Any] | None = None
        self.output_nll_targets: list[OutputNllTarget] = []
        self.output_nll_aggregates: list[
            tuple[OutputNllTarget, _ScoreAggregate]
        ] = []
        self.output_nll_queued = False
        self.output_nll_written = False
        self.output_nll_seconds = 0.0
        self.output_nll_scored = 0
        self.score_jobs: deque[_ScoreJob] = deque()
        self.outstanding_scores: dict[int, _ScoreJob] = {}
        self._score_request_counter = 0
        self.valid_base_keys: set[str] = set()
        self.written_candidate_paths: set[str] = set()
        self._done = False
        self.base_fallback_checked = False
        self.base_fallback_active = False
        self.base_fallback_targets: tuple[str, ...] = ()
        self.base_fallback_recovered: tuple[str, ...] = ()
        self.base_fallback_seconds = 0.0
        self.base_fallback_original: dict[str, Any] | None = None

        self.decode_requests = 0
        self.decode_exact_fallbacks = 0
        self.decode_cache_salt: str | None = None
        self.score_requests = 0
        self.decode_engine_seconds = 0.0
        self.score_engine_seconds = 0.0
        self.serialization_seconds = 0.0
        self.saved_candidate_sets = 0
        self.raw_beam_paths = 0
        self.invalid_beam_paths = 0
        self.monochrome_beam_paths = 0
        self.valid_beam_paths = 0
        self.context_fallback_scored_candidates = 0
        self.generated_prompt_tokens = 0
        self.decode_prompt_tokens = 0
        self.decode_cached_tokens = 0
        self.score_prompt_tokens = 0
        self.score_cached_tokens = 0
        self.score_prefix_retries = 0
        self.score_prefix_reconstructions = 0
        self.score_prefix_cache_disabled = False
        self.score_prefix_boundaries: dict[
            tuple[int, bytes],
            tuple[float, dict[int, float]],
        ] = {}
        self.queue_wait_seconds = max(
            0.0,
            self.started_at - float(job.get("published_at", self.started_at)),
        )

        print(
            f"{self.log_prefix} inference policy for puzzle {self.key}: "
            f"requested={requested_mode}, active={self.active_dfs_mode}, "
            f"decode_groups={len(self.batches)}, "
            f"decode_batch={self.requested_inference_batch_size}->"
            f"{self.inference_batch_size}, "
            f"kv_capacity={self.kv_cache_capacity_tokens}, "
            f"kv_resident_fraction={self.kv_resident_fraction:.3f}, "
            f"context_skipped_views={self.context_skipped_views}, "
            f"decode_budget_range={self.decode_budget_range}, "
            f"max_train_tokens={job['max_train_tokens']}, "
            f"min_train_pairs={self.pair_expansion_options['min_train_pairs']}, "
            f"extra_train_pair_mode={self.pair_expansion_options['extra_train_pair_mode']}, "
            f"max_beam_score={self.max_score:.6f}, "
            f"leaf_rollout={bool(options.get('vllm_leaf_rollout', False))}, "
            f"rollout_chunk={options.get('vllm_rollout_max_tokens', 1)}, "
            f"rollout_branch_window="
            f"{options.get('vllm_rollout_branch_window', 1)}, "
            f"batch_invariant="
            f"{bool(options.get('vllm_batch_invariant', False))}, "
            f"rescore_views={self.rescore_views}, "
            f"rescore_schedule={self.rescore_schedule}, "
            f"prefix_score="
            f"{bool(options.get('vllm_prefix_score', False))}, "
            f"prefix_tile_views={self.prefix_score_tile_views}, "
            f"predict_train={self.predict_train}, "
            f"report_output_nll={self.report_output_nll}, "
            f"search_limit={self.search_limit_seconds:g}s/group, "
            f"inference_timeout={self.inference_timeout_seconds:g}s/task (wall), "
            f"adapter_watchdog={self.adapter_watchdog_timeout_seconds:g}s, "
            f"watchdog_attempt="
            f"{int(job.get(_ADAPTER_TIMEOUT_COUNT, 0)) + 1}, "
            f"adapter={job['adapter_name']}#{job['adapter_id']}",
            flush=True,
        )
        self.settle()

    @property
    def done(self) -> bool:
        return self._done

    def _start_base_model_fallback(self, now: float) -> bool:
        if (
            not self.options.get("base_model_fallback", False)
            or self.base_fallback_checked
            or self.lora_request is None
        ):
            return False
        self.base_fallback_checked = True
        from base_model_fallback import black_placeholder_outputs, write_fallback_json

        self.base_fallback_targets = black_placeholder_outputs(
            self.key, self.test_output_count, self.output_dir,
            self.written_candidate_paths,
        )
        if not self.base_fallback_targets:
            return False
        if now >= self.global_end_time:
            print(f"{self.log_prefix} base-model fallback skipped: global deadline", flush=True)
            return False
        original = self.base_fallback_original = {
            "output_dir": self.output_dir,
            "paths": set(self.written_candidate_paths),
            "valid_base_keys": set(self.valid_base_keys),
            "nll_report": self.output_nll_report,
            "started_at": now,
        }
        retry_root = Path(self.output_dir) / "base_model_fallback"
        retry_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = tempfile.mkdtemp(prefix=f"{self.key}-", dir=retry_root)
        original["retry_dir"] = self.output_dir
        if self.output_nll_report is not None:
            path = write_fallback_json(
                Path(self.output_dir) / "lora_output_nll.json", self.output_nll_report,
            )
            self.written_candidate_paths.add(path)
        self.base_fallback_active = True
        # Keep the task adapter's ownership for final unload. An ordinary
        # engine uses no adapter; a residual engine needs the immutable A0/B0
        # adapter for BOTH decode and rescore to reconstruct its base model.
        self.lora_request = self.options.get("_pissa_fallback_lora_request")
        if self.options.get("pissa_base_mode") == "shared-residual" and self.lora_request is None:
            raise RuntimeError("Residual-base fallback is missing its initial PiSSA adapter")
        self.known_scores.clear()
        self.score_seed_by_grid.clear()
        self.score_prefix_boundaries.clear()
        self.score_prefix_cache_disabled = False
        self.decode_cache_salt = f"base-fallback-{Path(self.output_dir).name}"
        self.valid_base_keys.clear()
        self.batches = [
            filtered for batch in self.batches
            if (filtered := [key for key in batch if key.split(".")[0] in self.base_fallback_targets])
        ]
        self.test_id_to_subkeys = {
            key: subkeys for key, subkeys in self.test_id_to_subkeys.items()
            if subkeys[0].split(".")[0] in self.base_fallback_targets
        }
        self.batch_index = 0
        self.fallback_groups = None
        self.fallback_index = 0
        self.current_subkeys = []
        self.current_phase = ""
        self.output_nll_report = None
        self.output_nll_targets = []
        self.output_nll_aggregates = []
        self.output_nll_queued = False
        self.output_nll_written = False
        self.inference_deadline = now + self.inference_timeout_seconds
        self.adapter_watchdog_deadline = now + self.adapter_watchdog_timeout_seconds
        print(
            f"{self.log_prefix} base-model fallback for {self.key}: "
            f"outputs={list(self.base_fallback_targets)}, "
            f"fallback_adapter={getattr(self.lora_request, 'lora_name', None)}, "
            f"decode_groups={len(self.batches)}, "
            f"budget={max(0.0, min(self.inference_deadline, self.global_end_time) - now):.1f}s; "
            "other LoRA outputs retained",
            flush=True,
        )
        return True

    def _finish_base_model_fallback(self, *, abort_reason: str | None = None) -> None:
        if not self.base_fallback_active:
            return
        from base_model_fallback import (
            merge_base_output_nll, publish_base_candidates, write_fallback_json,
        )

        original = self.base_fallback_original
        assert original is not None
        if abort_reason is None:
            self.base_fallback_recovered = publish_base_candidates(
                primary_dir=original["output_dir"], retry_dir=self.output_dir,
                targets=self.base_fallback_targets, original_paths=original["paths"],
                owned_paths=self.written_candidate_paths,
                adapter_name=self.job["adapter_name"],
            )
        self.base_fallback_seconds = time.time() - original["started_at"]
        self.output_dir = original["output_dir"]
        self.valid_base_keys = original["valid_base_keys"] | set(self.base_fallback_recovered)
        if self.report_output_nll:
            self.output_nll_report = (
                merge_base_output_nll(original["nll_report"], self.output_nll_report, self.base_fallback_recovered)
                if self.base_fallback_recovered else original["nll_report"]
            )
            if self.base_fallback_recovered:
                path = write_output_nll_report(self.output_dir, self.output_nll_report)
                self.written_candidate_paths.add(path)
        path = write_fallback_json(Path(original["retry_dir"]) / "summary.json", {
            "schema_version": 1, "task_key": self.key,
            "source_adapter": self.job["adapter_name"],
            "trigger": "both_selected_attempts_are_1x1_black",
            "targets": list(self.base_fallback_targets),
            "recovered_outputs": list(self.base_fallback_recovered),
            "unrecovered_outputs": sorted(set(self.base_fallback_targets) - set(self.base_fallback_recovered)),
            "seconds": self.base_fallback_seconds,
            "abort_reason": abort_reason,
            "recovered_means": "non-placeholder prediction, not ground-truth accuracy",
            "decode_and_rescore_lora_request": getattr(self.lora_request, "lora_name", None),
            "pissa_base_id": self.options.get("pissa_base_id"),
            "base_reconstruction": (
                "residual_plus_initial_adapter_bf16" if self.lora_request is not None else "original_checkpoint"
            ),
        })
        self.written_candidate_paths.add(path)
        self.base_fallback_active = False
        print(
            f"{self.log_prefix} base-model fallback finished for {self.key}: "
            f"recovered={list(self.base_fallback_recovered)} (not an accuracy claim), "
            f"seconds={self.base_fallback_seconds:.3f}, abort_reason={abort_reason}; "
            f"report={path}", flush=True,
        )

    def abort_base_model_fallback(self, reason: str) -> None:
        """Optional retry failure must not erase the completed LoRA result."""
        self._finish_base_model_fallback(abort_reason=reason)
        self._done = True

    def _new_threshold_search(self, subkeys: list[str]) -> Any:
        prefixes = [self.inference_prefixes[subkey] for subkey in subkeys]
        # vLLM accepts a ragged list of independently sized prompts. Decode
        # budgets still remain homogeneous because the search state shares
        # one max_new_tokens value across its origins.
        token_budgets = {self.decode_token_budgets[subkey] for subkey in subkeys}
        if len(token_budgets) != 1:
            raise RuntimeError(
                "Inference batch contains different decode budgets: "
                f"{sorted(token_budgets)} for {subkeys}"
            )
        max_new_tokens = next(iter(token_budgets))
        deadline = resolve_task_search_deadline(
            time.time(),
            inference_deadline=self.inference_deadline,
            global_end_time=self.global_end_time,
            search_limit_seconds=self.search_limit_seconds,
        )
        if self.active_dfs_mode == "frontier":
            return FrontierSearchBatch(
                prefixes,
                max_new_tokens,
                self.max_score,
                deadline,
                origin_batch_size=int(self.options["frontier_origin_batch_size"]),
                max_live_branches=int(self.options["frontier_max_live_branches"]),
            )
        if self.options.get("vllm_leaf_rollout", False):
            return RecursiveRolloutSearchBatch(
                prefixes,
                max_new_tokens,
                self.max_score,
                deadline,
                rollout_chunk_tokens=int(self.options["vllm_rollout_max_tokens"]),
                branch_window=int(
                    self.options.get("vllm_rollout_branch_window", 1)
                ),
            )
        return RecursiveSearchBatch(
            prefixes,
            max_new_tokens,
            self.max_score,
            deadline,
        )

    def _build_fallback_groups(self) -> None:
        grouped: dict[
            tuple[tuple[int, int], int],
            list[tuple[str, tuple[int, ...]]],
        ] = defaultdict(list)
        if not self.options["zero_candidate_shape_fallback"]:
            self.fallback_groups = []
            return
        for subkeys in self.test_id_to_subkeys.values():
            base_key = subkeys[0].split(".")[0]
            if base_key in self.valid_base_keys:
                continue
            eligible_subkeys = [
                subkey
                for subkey in subkeys
                if subkey in self.inference_prefixes
            ]
            if not eligible_subkeys:
                continue
            shaped_candidates = []
            monochrome_only = True
            for subkey in eligible_subkeys:
                output_shape = infer_same_shape_reply(self.eval_ds.queries[subkey])
                if output_shape is None:
                    continue
                if int(np.prod(output_shape)) < 2:
                    continue
                monochrome_only = False
                required_reply_tokens = len(
                    self.tokenizer.encode(
                        self.formatter.fmt_reply(
                            [np.zeros(output_shape, dtype=np.uint8)]
                        )
                    )
                )
                prefix = self.inference_prefixes[subkey]
                shaped_candidates.append(
                    (subkey, prefix, output_shape, required_reply_tokens)
                )
            if not shaped_candidates and monochrome_only:
                print(
                    f"{self.log_prefix} no safe shape fallback for {base_key}: "
                    "no augmentation has a supported non-monochrome output shape"
                )
                continue
            fitting_candidates = [
                candidate
                for candidate in shaped_candidates
                if len(candidate[1]) + candidate[3] <= self.max_model_len
            ]
            if not fitting_candidates:
                print(
                    f"{self.log_prefix} skipping shape fallback for {base_key}: "
                    "no complete prompt has enough room for its inferred output",
                    flush=True,
                )
                continue
            fallback_subkey, prefix, output_shape, _ = fitting_candidates[0]
            grouped[(output_shape, len(prefix))].append((fallback_subkey, prefix))
        self.fallback_groups = [
            (shape_and_length[0], requests)
            for shape_and_length, requests in grouped.items()
        ]

    def _start_next_search(self, now: float) -> bool:
        if self.batch_index < len(self.batches):
            if now > self.inference_deadline or now > self.global_end_time:
                print(
                    f"{self.log_prefix} inference timeout after "
                    f"{now - self.started_at:.1f}s for puzzle {self.key}"
                )
                self.batch_index = len(self.batches)
            else:
                self.current_subkeys = self.batches[self.batch_index]
                self.batch_index += 1
                self.current_phase = "threshold DFS"
                print(f"{self.log_prefix} decoding {self.current_subkeys}")
                self.search = self._new_threshold_search(self.current_subkeys)
                return True

        if self.fallback_groups is None:
            self._build_fallback_groups()
        assert self.fallback_groups is not None
        if self.fallback_index < len(self.fallback_groups):
            if now >= min(self.inference_deadline, self.global_end_time):
                self.fallback_index = len(self.fallback_groups)
            else:
                output_shape, requests = self.fallback_groups[self.fallback_index]
                self.fallback_index += 1
                self.current_subkeys = [request[0] for request in requests]
                prefixes = [request[1] for request in requests]
                self.current_phase = "shape-beam fallback"
                print(
                    f"{self.log_prefix} shape-beam fallback "
                    f"{self.current_subkeys} "
                    f"with output shape {output_shape} and width "
                    f"{self.options['shape_fallback_beam_width']}",
                    flush=True,
                )
                self.search = ShapeBeamSearch(
                    prefixes,
                    output_shape,
                    min(self.inference_deadline, self.global_end_time),
                    beam_width=int(self.options["shape_fallback_beam_width"]),
                )
                return True
        return False

    def _prepare_score_aggregate(
        self,
        base_key: str,
        solution: np.ndarray,
        grid_id: tuple[Any, ...],
        fallback_beam_score: float,
    ) -> tuple[_ScoreAggregate, list[dict[str, Any]]] | None:
        augmented = ArcDataset(
            keys=[base_key],
            queries={base_key: self.puzzle_ds_multi.queries.get(base_key)},
            replies={base_key: [solution.tolist()]},
        )
        augmented = augmented.augment_exact_views(
            self.rescore_views,
            seed=self.rescore_seeds[base_key],
            **self.pair_expansion_options,
        )
        rows = augmented.as_list(self.formatter)
        request_lengths = []
        for row in rows:
            query_length = len(self.tokenizer.encode(row["input"]))
            answer_length = len(self.tokenizer.encode(row["reply"]))
            request_lengths.append(query_length + answer_length + 1)
        oversized = [
            length for length in request_lengths if length > self.max_model_len
        ]
        if oversized:
            self.context_fallback_scored_candidates += 1
            self.known_scores[grid_id] = [float(fallback_beam_score)]
            # The fallback value belongs to the candidate's originating DFS
            # view, not the shared rescore seed. Marking it unknown makes
            # shared-view selectors safely fall back to kgmon.
            self.score_seed_by_grid[grid_id] = None
            print(
                f"{self.log_prefix} keeping candidate for {base_key} with "
                "its threshold-DFS NLL because "
                f"{len(oversized)}/{len(rows)} untrimmed rescore view(s) "
                "exceed the vLLM context window "
                f"(max_required={max(oversized)}, "
                f"max_model_len={self.max_model_len})",
                flush=True,
            )
            return None
        aggregate = _ScoreAggregate(grid_id, [None] * len(rows))
        self.pending_aggregates[grid_id] = aggregate
        self.score_seed_by_grid[grid_id] = self.rescore_seeds[base_key]
        return aggregate, rows

    def _append_score_job(
        self,
        aggregate: _ScoreAggregate,
        aggregate_index: int,
        row: dict[str, Any],
        *,
        barrier_after: bool = False,
    ) -> _ScoreJob:
        query_tokens = tuple(self.tokenizer.encode(row["input"]))
        answer_tokens = tuple(self.tokenizer.encode(row["reply"]))
        request_id = self._score_request_counter
        self._score_request_counter += 1
        job = _ScoreJob(
            request_id=request_id,
            prompt_token_ids=query_tokens + answer_tokens,
            query_length=len(query_tokens),
            answer_token_ids=answer_tokens,
            aggregate=aggregate,
            aggregate_index=aggregate_index,
            barrier_after=barrier_after,
        )
        self.score_jobs.append(job)
        return job

    def _queue_score_aggregates(
        self,
        candidates: list[
            tuple[str, np.ndarray, tuple[Any, ...]]
            | tuple[str, np.ndarray, tuple[Any, ...], float]
        ],
    ) -> None:
        prepared = []
        for candidate in candidates:
            base_key, solution, grid_id = candidate[:3]
            fallback_beam_score = (
                float(candidate[3]) if len(candidate) == 4 else self.max_score
            )
            score_data = self._prepare_score_aggregate(
                base_key,
                solution,
                grid_id,
                fallback_beam_score,
            )
            if score_data is not None:
                prepared.append((*score_data, base_key))
        if not prepared:
            return

        if self.rescore_schedule == "candidate":
            for aggregate, rows, _base_key in prepared:
                for index, row in enumerate(rows):
                    self._append_score_job(aggregate, index, row)
            return

        # Candidate answers under the same augmentation share the complete
        # examples/challenge query prefix.
        prepared_by_base_key: dict[
            str,
            list[tuple[_ScoreAggregate, list[dict[str, Any]]]],
        ] = defaultdict(list)
        for aggregate, rows, base_key in prepared:
            prepared_by_base_key[base_key].append((aggregate, rows))
        for same_task_candidates in prepared_by_base_key.values():
            row_counts = {len(rows) for _, rows in same_task_candidates}
            if len(row_counts) != 1:
                raise RuntimeError(
                    "Candidates produced different rescore view counts: "
                    f"{sorted(row_counts)}"
                )
            row_count = next(iter(row_counts), 0)
            if row_count == 0:
                continue
            if self.rescore_schedule == "view":
                for index in range(row_count):
                    for aggregate, rows in same_task_candidates:
                        self._append_score_job(aggregate, index, rows[index])
                continue

            # vLLM cannot reuse a prefix from another request in the same
            # engine call. Warm a small tile of views with candidate zero,
            # place a barrier, consume all remaining candidates while those
            # prefixes are resident, then advance to the next tile. Four long
            # ARC prefixes fit comfortably in the 63k-token L4 KV cache where
            # all 32 often do not.
            if len(same_task_candidates) == 1:
                aggregate, rows = same_task_candidates[0]
                for index, row in enumerate(rows):
                    self._append_score_job(aggregate, index, row)
                continue
            tile_size = self.prefix_score_tile_views
            for tile_start in range(0, row_count, tile_size):
                tile_stop = min(row_count, tile_start + tile_size)
                warm_aggregate, warm_rows = same_task_candidates[0]
                warm_jobs = [
                    self._append_score_job(
                        warm_aggregate,
                        index,
                        warm_rows[index],
                    )
                    for index in range(tile_start, tile_stop)
                ]
                warm_jobs[-1].barrier_after = True
                reuse_jobs = [
                    self._append_score_job(
                        aggregate,
                        index,
                        rows[index],
                    )
                    for aggregate, rows in same_task_candidates[1:]
                    for index in range(tile_start, tile_stop)
                ]
                reuse_jobs[-1].barrier_after = True

    def _process_decode_results(self, results) -> None:
        new_candidates: dict[
            tuple[Any, ...],
            tuple[str, np.ndarray, tuple[Any, ...], float],
        ] = {}
        for subkey_id, scored_beams in results:
            subkey = self.current_subkeys[subkey_id]
            base_key = subkey.split(".")[0]
            entries = []
            self.raw_beam_paths += len(scored_beams)
            for beam_score, suffix_tokens in scored_beams:
                array = self.formatter.convert_tokens_to_array(suffix_tokens)
                if array is None:
                    self.invalid_beam_paths += 1
                    continue
                solution = self.puzzle_ds_multi.invert_mod(
                    array,
                    subkey,
                    inv_perm=True,
                )
                if (
                    self.options["zero_candidate_shape_fallback"]
                    and np.unique(solution).size < 2
                ):
                    self.monochrome_beam_paths += 1
                    print(
                        f"{self.log_prefix} rejecting monochrome candidate from "
                        f"{subkey} ({self.current_phase})"
                    )
                    continue
                self.valid_beam_paths += 1
                grid_id = (base_key, tuple(map(tuple, solution)))
                if (
                    grid_id not in self.known_scores
                    and grid_id not in self.pending_aggregates
                    and grid_id not in new_candidates
                ):
                    print(
                        f"{self.log_prefix} scoring {subkey} #{len(entries)} "
                        f"({self.current_phase})"
                    )
                    new_candidates[grid_id] = (
                        base_key,
                        solution,
                        grid_id,
                        float(beam_score),
                    )
                elif (
                    grid_id in new_candidates
                    and float(beam_score) < new_candidates[grid_id][3]
                ):
                    new_candidates[grid_id] = (
                        base_key,
                        solution,
                        grid_id,
                        float(beam_score),
                    )
                entries.append(_CandidateEntry(float(beam_score), grid_id, solution))
            if entries:
                self.pending_output_groups.append((subkey, entries))
        self._queue_score_aggregates(list(new_candidates.values()))
        scorable_grid_ids = set(self.known_scores) | set(self.pending_aggregates)
        self.pending_output_groups = [
            (subkey, scorable_entries)
            for subkey, entries in self.pending_output_groups
            if (
                scorable_entries := [
                    entry
                    for entry in entries
                    if entry.grid_id in scorable_grid_ids
                ]
            )
        ]

    def _queue_output_nll_scores(self) -> None:
        """Queue canonical candidate/label diagnostics after all decoding."""
        if self.output_nll_queued:
            return
        self.output_nll_queued = True
        started = time.perf_counter()
        test_base_keys = {
            f"{self.key}_{test_index}"
            for test_index in range(self.test_output_count)
        }
        candidate_solutions: dict[str, list[np.ndarray]] = defaultdict(list)
        for grid_id in self.known_scores:
            if not isinstance(grid_id, tuple) or len(grid_id) != 2:
                continue
            base_key, grid_rows = grid_id
            if base_key not in test_base_keys:
                continue
            candidate_solutions[base_key].append(
                np.asarray(grid_rows, dtype=np.uint8)
            )

        self.output_nll_report, self.output_nll_targets = (
            prepare_output_nll_report(
                **self.pair_expansion_options,
                task_key=self.key,
                puzzle_ds_multi=self.puzzle_ds_multi,
                test_output_count=self.test_output_count,
                candidate_solutions=candidate_solutions,
                ground_truth_outputs=self.ground_truth_outputs,
                formatter=self.formatter,
                tokenizer=self.tokenizer,
                max_model_len=self.max_model_len,
                metadata={
                    "backend": str(
                        self.options.get("engine_label", "vLLM")
                    ).lower(),
                    "score_batch_size": int(self.score_batch_size),
                    "adapter_name": None if self.base_fallback_active else str(self.job["adapter_name"]),
                    "inference_model": "base" if self.base_fallback_active else "lora",
                },
            )
        )
        if self.base_fallback_active:
            self.output_nll_report["outputs"] = {
                key: value for key, value in self.output_nll_report["outputs"].items()
                if key in self.base_fallback_targets
            }
            self.output_nll_targets = [
                target for target in self.output_nll_targets
                if target.base_key in self.base_fallback_targets
            ]
        score_deadline = min(self.inference_deadline, self.global_end_time)
        if time.time() < score_deadline:
            for target in self.output_nll_targets:
                aggregate = _ScoreAggregate(
                    ("output_nll", target.base_key, target.grid),
                    [None],
                )
                self.output_nll_aggregates.append((target, aggregate))
                request_id = self._score_request_counter
                self._score_request_counter += 1
                self.score_jobs.append(
                    _ScoreJob(
                        request_id=request_id,
                        prompt_token_ids=(
                            target.prompt_token_ids
                            + target.answer_token_ids
                        ),
                        query_length=len(target.prompt_token_ids),
                        answer_token_ids=target.answer_token_ids,
                        aggregate=aggregate,
                        aggregate_index=0,
                    )
                )
        self.output_nll_seconds += time.perf_counter() - started

    def _finalize_output_nll_report(self) -> None:
        if self.output_nll_written:
            return
        if self.output_nll_report is None:
            raise RuntimeError("Output-NLL report was not prepared")
        if any(
            not aggregate.complete
            for _target, aggregate in self.output_nll_aggregates
        ):
            raise RuntimeError("Cannot publish incomplete output-NLL scores")
        started = time.perf_counter()
        active_targets = [
            target for target, _aggregate in self.output_nll_aggregates
        ]
        scores = [
            aggregate.result()[0]
            for _target, aggregate in self.output_nll_aggregates
        ]
        apply_output_nll_scores(
            self.output_nll_report,
            active_targets,
            scores,
        )
        if self.base_fallback_active:
            from base_model_fallback import write_fallback_json

            # Do not let ARC_OUTPUT_NLL_REPORT_DIR overwrite the original
            # LoRA report before the retry is complete and published.
            finalize_output_nll_report(self.output_nll_report)
            output_path = write_fallback_json(
                Path(self.output_dir) / "base_output_nll.json", self.output_nll_report,
            )
        else:
            output_path = write_output_nll_report(self.output_dir, self.output_nll_report)
        self.written_candidate_paths.add(output_path)
        elapsed = time.perf_counter() - started
        self.output_nll_seconds += elapsed
        self.serialization_seconds += elapsed
        self.output_nll_scored += len(scores)
        self.output_nll_written = True
        print(
            f"{self.log_prefix} output NLL report for {self.key}: "
            f"scored={len(scores)}/"
            f"{len(self.output_nll_targets)}, "
            f"ground_truth_available="
            f"{self.ground_truth_outputs is not None}, "
            f"time={self.output_nll_seconds:.3f}s, path={output_path}",
            flush=True,
        )

    def _scores_complete(self) -> bool:
        return (
            not self.score_jobs
            and not self.outstanding_scores
            and all(
                aggregate.complete for aggregate in self.pending_aggregates.values()
            )
            and all(
                aggregate.complete
                for _target, aggregate in self.output_nll_aggregates
            )
        )

    def _finalize_scores_and_serialize(self) -> None:
        if not self._scores_complete():
            raise RuntimeError("Cannot serialize before all augmented scores finish")
        for grid_id, aggregate in self.pending_aggregates.items():
            self.known_scores[grid_id] = aggregate.result()
        self.pending_aggregates.clear()
        for subkey, entries in self.pending_output_groups:
            decoded_result = [
                {
                    "beam_score": entry.beam_score,
                    "score_aug": self.known_scores[entry.grid_id],
                    "score_aug_count": len(self.known_scores[entry.grid_id]),
                    "score_aug_seed": self.score_seed_by_grid.get(
                        entry.grid_id,
                        self.rescore_seeds[subkey.split(".")[0]],
                    ),
                    "solution": entry.solution,
                }
                for entry in entries
            ]
            started = time.perf_counter()
            output_dir = self.output_dir
            if parse_train_prediction_key(subkey) is not None:
                if self.train_prediction_output_dir is None:
                    raise RuntimeError(
                        "Training prediction output was produced while "
                        "--predict-train is disabled"
                    )
                output_dir = self.train_prediction_output_dir
            output_path = os.path.join(output_dir, subkey)
            with bz2.BZ2File(output_path, "w") as output_file:
                pickle.dump(decoded_result, output_file)
            self.written_candidate_paths.add(output_path)
            self.serialization_seconds += time.perf_counter() - started
            self.saved_candidate_sets += 1
            self.valid_base_keys.add(subkey.split(".")[0])
        self.pending_output_groups.clear()

    def remove_written_candidate_files(self) -> int:
        """Remove only files produced by this interrupted task attempt."""
        removed = 0
        for output_path in tuple(self.written_candidate_paths):
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass
            else:
                removed += 1
        self.written_candidate_paths.clear()
        return removed

    def settle(self, now: float | None = None) -> None:
        """Advance local phases until another model request is required."""
        now = time.time() if now is None else now
        while not self._done:
            if self.search is not None:
                if not self.search.done:
                    return
                results = self.search.results()
                self.search = None
                self._process_decode_results(results)
                if not self._scores_complete():
                    return
                self._finalize_scores_and_serialize()
                now = time.time()
                continue
            if not self._scores_complete():
                return
            if self.pending_output_groups:
                self._finalize_scores_and_serialize()
                now = time.time()
                continue
            if self._start_next_search(now):
                return
            if self.report_output_nll:
                if not self.output_nll_queued:
                    self._queue_output_nll_scores()
                    if not self._scores_complete():
                        return
                if not self.output_nll_written:
                    self._finalize_output_nll_report()
            if self._start_base_model_fallback(time.time()):
                now = time.time()
                continue
            self._finish_base_model_fallback()
            self._done = True

    def take_work(self, limit: int) -> list[_EngineWork]:
        self.settle()
        if self.done or limit <= 0:
            return []
        if self.search is not None:
            return [
                _EngineWork(
                    task=self,
                    kind="decode",
                    request_id=request.request_id,
                    prompt_token_ids=request.prompt_token_ids,
                    max_tokens=request.max_tokens,
                    constrained_rollout=request.constrained_rollout,
                    cumulative_nll=request.cumulative_nll,
                )
                for request in self.search.take_requests(limit)
            ]
        score_limit = min(limit, self.score_batch_size)
        works = []
        while self.score_jobs and len(works) < score_limit:
            job = self.score_jobs.popleft()
            if self.score_prefix_cache_disabled:
                job.allow_prefix_cache = False
            self.outstanding_scores[job.request_id] = job
            works.append(
                _EngineWork(
                    task=self,
                    kind="score",
                    request_id=job.request_id,
                    prompt_token_ids=job.prompt_token_ids,
                    score_job=job,
                )
            )
            if job.barrier_after:
                break
        return works

    def accept_decode(
        self,
        request_id: int,
        token_logprobs,
        *,
        sampled_token_ids=None,
    ) -> None:
        if self.search is None:
            raise RuntimeError("Decode result arrived without an active search")
        if sampled_token_ids is None:
            self.search.accept(request_id, token_logprobs)
        else:
            self.search.accept_rollout(
                request_id,
                sampled_token_ids,
                token_logprobs,
            )
        self.decode_requests += 1
        self.settle()

    def accept_score(
        self,
        request_id: int,
        prompt_logprobs,
        *,
        cached_tokens: int = 0,
    ) -> None:
        try:
            job = self.outstanding_scores.pop(request_id)
        except KeyError as exc:
            raise KeyError(f"Unknown score request {request_id}") from exc
        prefix_score = (
            self.options.get("vllm_prefix_score", False)
            and not self.options.get("score_returns_complete_prompt", False)
        )
        try:
            if prefix_score:
                score = self._answer_nll_with_prefix_cache(
                    job,
                    prompt_logprobs,
                    cached_tokens,
                )
            else:
                score = answer_nll_from_prompt_logprobs(
                    prompt_logprobs,
                    job.answer_token_ids,
                    job.query_length,
                )
        except ValueError:
            if (
                not self.options.get("vllm_prefix_score", False)
                or not job.allow_prefix_cache
                or cached_tokens <= 0
            ):
                raise
            job.allow_prefix_cache = False
            self.score_jobs.appendleft(job)
            self.score_prefix_retries += 1
            disable_threshold = max(1, int(self.score_batch_size))
            if (
                not self.score_prefix_cache_disabled
                and self.score_prefix_retries >= disable_threshold
            ):
                self.score_prefix_cache_disabled = True
                for queued_job in self.score_jobs:
                    queued_job.allow_prefix_cache = False
                print(
                    f"{self.log_prefix} disabling score prefix-cache reads "
                    "for the remainder of this task after repeated unsafe "
                    f"answer-side hits ({self.score_prefix_retries} retries, "
                    f"{self.score_requests} completed scores); decode prefix "
                    "caching remains enabled",
                    flush=True,
                )
            print(
                f"{self.log_prefix} retrying score request {request_id} "
                f"without prefix cache after an unsafe {cached_tokens}-token "
                "hit",
                flush=True,
            )
            self.settle()
            return
        job.aggregate.values[job.aggregate_index] = score
        self.score_requests += 1
        self.settle()

    def _answer_nll_with_prefix_cache(
        self,
        job: _ScoreJob,
        prompt_logprobs,
        cached_tokens: int,
    ) -> float:
        """Reconstruct cached answer-prefix NLLs from an earlier score.

        Prefix-cache hits are block aligned. For each scored answer, retain the
        cumulative NLL and raw 16-token distribution at answer-side block
        boundaries. A later candidate that shares those cached tokens can use
        the retained values for the omitted positions and consume vLLM's
        returned suffix normally. If no matching boundary exists, the caller
        retries that request with cache reads disabled.
        """
        prompt_tokens = job.prompt_token_ids
        total_length = len(prompt_tokens)
        query_length = job.query_length
        if cached_tokens < 0 or cached_tokens >= total_length:
            raise ValueError(f"Invalid cached token count: {cached_tokens}")
        if not prompt_logprobs:
            raise ValueError("vLLM returned no prompt logprobs")

        returned_positions = len(prompt_logprobs) - 1
        first_absolute_position = total_length - returned_positions
        if first_absolute_position < 1:
            raise ValueError("Invalid prefix-cached prompt-logprob length")
        if cached_tokens and first_absolute_position != cached_tokens + 1:
            raise ValueError(
                "Prompt-logprob suffix does not match reported cache hit: "
                f"first={first_absolute_position}, cached={cached_tokens}"
            )

        digest = hashlib.sha256()
        boundary_digests: dict[int, bytes] = {}
        for length, token_id in enumerate(prompt_tokens, start=1):
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
            if length % 16 == 0 or length == cached_tokens:
                boundary_digests[length] = digest.copy().digest()

        cumulative_nll = 0.0
        first_scored_position = query_length
        reconstructed_prefix = False
        if cached_tokens >= query_length:
            boundary_key = (
                cached_tokens,
                boundary_digests[cached_tokens],
            )
            try:
                cumulative_nll, next_token_logprobs = self.score_prefix_boundaries[
                    boundary_key
                ]
                next_token_nll = -next_token_logprobs[int(prompt_tokens[cached_tokens])]
            except KeyError as exc:
                raise ValueError(
                    "No scored boundary matches the cached answer prefix"
                ) from exc
            cumulative_nll += next_token_nll
            first_scored_position = cached_tokens + 1
            reconstructed_prefix = True

        if first_scored_position < first_absolute_position:
            raise ValueError(
                "Cached response omits answer positions without a retained prefix score"
            )

        for absolute_position in range(
            first_scored_position,
            total_length,
        ):
            response_position = absolute_position - first_absolute_position + 1
            values = logprobs_at_position(
                prompt_logprobs,
                response_position,
            )
            if absolute_position % 16 == 0:
                boundary_key = (
                    absolute_position,
                    boundary_digests[absolute_position],
                )
                self.score_prefix_boundaries[boundary_key] = (
                    cumulative_nll,
                    values,
                )
            token_id = int(prompt_tokens[absolute_position])
            try:
                cumulative_nll -= values[token_id]
            except KeyError as exc:
                raise ValueError(
                    f"Prompt logprobs at position {absolute_position} omit "
                    f"actual token {token_id}"
                ) from exc
        if reconstructed_prefix:
            self.score_prefix_reconstructions += 1
        return cumulative_nll

    def record_engine_call(
        self,
        kind: str,
        seconds: float,
        prompt_tokens: int,
        cached_tokens: int,
    ) -> None:
        if kind == "decode":
            self.decode_engine_seconds += seconds
            self.decode_prompt_tokens += int(prompt_tokens)
            self.decode_cached_tokens += int(cached_tokens)
        else:
            self.score_engine_seconds += seconds
            self.score_prompt_tokens += int(prompt_tokens)
            self.score_cached_tokens += int(cached_tokens)
        self.generated_prompt_tokens += int(prompt_tokens)

    def report_accuracy(self) -> None:
        """Read validation labels only after decoding/rescoring has finished."""
        if not self.options.get("report_task_accuracy", False) or self.accuracy_reported:
            return
        if not self.done:
            raise RuntimeError("Task accuracy cannot be reported before inference completes")
        self.accuracy_reported = True
        try:
            from task_accuracy_report import report_completed_task_accuracy

            report_completed_task_accuracy(
                task_key=self.key,
                query=self.accuracy_query,
                output_dir=self.output_dir,
                run_id=self.options["accuracy_run_id"],
                candidate_paths=tuple(self.written_candidate_paths),
                log_prefix=self.log_prefix,
            )
        except Exception as error:
            # An optional diagnostic must not discard already completed grids.
            print(
                f"{self.log_prefix} accuracy report failed for {self.key}: "
                f"{type(error).__name__}: {error}; candidates retained",
                flush=True,
            )

    def summary(self) -> str:
        return (
            f"{self.log_prefix} inference stats for puzzle {self.key}: "
            f"total={time.time() - self.started_at:.3f}s, "
            f"queue_wait={self.queue_wait_seconds:.3f}s, "
            f"decode_service={self.decode_engine_seconds:.3f}s/"
            f"{self.decode_requests} requests, "
            f"score_service={self.score_engine_seconds:.3f}s/"
            f"{self.score_requests} requests, "
            f"prompt_tokens={self.generated_prompt_tokens}, "
            f"decode_cache={self.decode_cached_tokens}/"
            f"{self.decode_prompt_tokens}, "
            f"score_cache={self.score_cached_tokens}/"
            f"{self.score_prompt_tokens}, "
            f"score_prefix_mode="
            f"{bool(self.options.get('vllm_prefix_score', False))}, "
            f"score_prefix_retries={self.score_prefix_retries}, "
            f"score_prefix_reconstructions="
            f"{self.score_prefix_reconstructions}, "
            f"score_prefix_cache_disabled="
            f"{self.score_prefix_cache_disabled}, "
            f"output_nll={self.output_nll_seconds:.3f}s/"
            f"{self.output_nll_scored} grids, "
            f"serialize={self.serialization_seconds:.3f}s/"
            f"{self.saved_candidate_sets} files, mode={self.active_dfs_mode}, "
            f"decode_batch={self.requested_inference_batch_size}->"
            f"{self.inference_batch_size}, "
            f"leaf_rollout={bool(self.options.get('vllm_leaf_rollout', False))}, "
            f"rollout_chunk={self.options.get('vllm_rollout_max_tokens', 1)}, "
            f"search_limit={self.search_limit_seconds:g}s/group, "
            f"inference_timeout={self.inference_timeout_seconds:g}s/task (wall), "
            f"cumulative_nll_stop="
            f"{bool(self.options.get('vllm_cumulative_nll_stop', False))}, "
            f"rescore_views={self.rescore_views}, "
            f"rescore_schedule={self.rescore_schedule}, "
            f"prefix_tile_views={self.prefix_score_tile_views}, "
            f"context_skipped_views={self.context_skipped_views}, "
            "context_fallback_scored_candidates="
            f"{self.context_fallback_scored_candidates}, "
            f"raw_beams={self.raw_beam_paths}, "
            f"decode_exact_fallbacks={self.decode_exact_fallbacks}, "
            f"base_model_fallback_outputs={len(self.base_fallback_targets)}, "
            f"base_model_fallback_recovered={len(self.base_fallback_recovered)}, "
            f"base_model_fallback_seconds={self.base_fallback_seconds:.3f}, "
            f"invalid_beams={self.invalid_beam_paths}, "
            f"monochrome_beams={self.monochrome_beam_paths}, "
            f"valid_beams={self.valid_beam_paths}"
        )


def _vllm_engine_kwargs(
    model_path: str,
    options: dict[str, Any],
    *,
    hf_overrides: dict[str, Any] | None = None,
    disable_cutedsl_warmup: bool = False,
) -> dict[str, Any]:
    """Build vLLM arguments, omitting KV offload on the default path."""
    validate_pissa_options(
        options.get("lora_method", "rslora"), options.get("lora_init", "default"),
        model_family=options.get("model_family", "qwen"),
        fp8_frozen_linear=options.get("fp8_frozen_linear", False),
        quantization=options.get("vllm_quantization", "none"),
    )
    from arc_pissa_shared import validate_runtime_base

    shared = validate_runtime_base(
        model_path, mode=options.get("pissa_base_mode", "original"),
        initialization=options.get("lora_init", "default"),
        method=options.get("lora_method", "rslora"), rank=options["lora_r"],
        alpha=options.get("lora_alpha", 32.0),
    )
    if shared is not None:
        if options.get("pissa_base_id") not in (None, shared["base_id"]):
            raise ValueError("vLLM shared PiSSA base identity differs from the launcher")
        options["pissa_base_id"] = shared["base_id"]
    # Fallback has one additional fixed adapter, never an extra task. Reserving
    # it avoids evicting an active task or exceeding a mixed batch's slot count.
    fallback_slot = int(shared is not None and options.get("base_model_fallback", False))
    kwargs: dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "skip_tokenizer_init": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(options["vllm_gpu_memory_utilization"]),
        "enforce_eager": bool(options["vllm_enforce_eager"]),
        # vLLM 0.23 enables a two-batch asynchronous scheduler by default.
        # Its shared Punica metadata and prompt-logprob completion state are
        # not safe for this persistent multi-LoRA workload: the next batch can
        # be prepared while the previous batch still owns those buffers.  Keep
        # request-level batching, but serialize engine batches by default.
        # The V2 model runner keeps that state on one CUDA stream, so
        # --vllm-async-scheduling is safe there (stress- and accuracy-tested
        # with multi-LoRA, prompt logprobs and DFlash on the fork wheel).
        "async_scheduling": bool(options.get("vllm_async_scheduling", False)),
        "enable_prefix_caching": True,
        "max_model_len": int(options["max_seq_length"]),
        "max_num_seqs": int(options["vllm_max_num_seqs"]),
        "max_num_batched_tokens": int(options["vllm_max_num_batched_tokens"]),
        "enable_lora": True,
        "max_loras": int(options["vllm_max_loras"]) + fallback_slot,
        "max_cpu_loras": int(options["vllm_max_loras"]) + fallback_slot,
        "max_lora_rank": lora_engine_max_rank(
            options["lora_r"], options.get("lora_init", "default"),
            options.get("pissa_base_mode", "original"),
        ),
        "logprobs_mode": "raw_logprobs",
    }
    if options.get("vllm_cumulative_nll_stop", False):
        kwargs["logits_processors"] = [
            "arc_vllm_nll_stop:ArcCumulativeNllLogitsProcessor"
        ]
    quantization = str(options.get("vllm_quantization", "none"))
    if quantization != "none":
        kwargs["quantization"] = quantization
    kv_cache_dtype = normalize_vllm_kv_cache_dtype(
        options.get("vllm_kv_cache_dtype", "auto")
    )
    if kv_cache_dtype != "auto":
        # Leave the default engine configuration untouched. Quantizing the
        # cache does not change model/LoRA dtype or force an attention backend.
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if options.get("fp8_lora", False):
        # FP8 is a training GEMM recipe, not a vLLM adapter storage dtype.
        kwargs["lora_dtype"] = "bfloat16"
    speculative_tokens = int(
        options.get("vllm_speculative_ngram_tokens", 0) or 0
    )
    if speculative_tokens > 0:
        kwargs["speculative_config"] = {
            "method": str(
                options.get("vllm_speculative_ngram_method", "ngram_gpu")
            ),
            "num_speculative_tokens": speculative_tokens,
            "prompt_lookup_min": int(
                options.get("vllm_speculative_ngram_min", 2)
            ),
            "prompt_lookup_max": int(
                options.get("vllm_speculative_ngram_max", 5)
            ),
        }
    draft_spec = build_dflash_speculative_config(model_path, options)
    if draft_spec is not None:
        kwargs["speculative_config"] = draft_spec
        if draft_spec.get("kv_cache_dtype") not in (None, "auto"):
            kwargs["worker_cls"] = "arc_vllm_dflash_worker.ArcDFlashWorker"
    if disable_cutedsl_warmup:
        # Some Kaggle vLLM 0.23 snapshots expose a CUTLASS-DSL warmup switch.
        # Their wheel set can pair a newer CUTLASS wrapper with an older call
        # and fail on Blackwell before the dense Qwen model runs. Exact v0.23.0
        # has no such config field, so the caller capability-gates this entry.
        kwargs["kernel_config"] = {"enable_cutedsl_warmup": False}
    if hf_overrides is not None:
        kwargs["hf_overrides"] = dict(hf_overrides)
    kv_offloading_size = float(options.get("vllm_kv_offloading_size", 0.0) or 0.0)
    if kv_offloading_size > 0.0:
        kwargs.update(
            kv_offloading_size=kv_offloading_size,
            kv_offloading_backend=str(
                options.get("vllm_kv_offloading_backend", "native")
            ),
        )
    if options.get("bidirectional_prompt_attention", False):
        compiled_attention = bool(options.get("bidirectional_vllm_compile", False))
        if compiled_attention and options.get("vllm_enforce_eager", False):
            raise ValueError(
                "bidirectional_vllm_compile conflicts with vllm_enforce_eager"
            )
        # Chunked prompt runs would truncate the role-run contract.  Eager is
        # the accuracy/speed default for one-token DFS.  The optional compiled
        # mode uses the same Inductor partitions for prefill and decode plus
        # full decode graphs; using full-decode-only graphs caused measurable
        # cache/full raw-logprob drift on the validated stack.
        kwargs.update(
            enforce_eager=not compiled_attention,
            enable_chunked_prefill=False,
            block_size=16,
            attention_backend="CUSTOM",
            worker_cls=("arc_vllm_bidirectional.ArcBidirectionalWorker"),
        )
        if compiled_attention:
            max_num_seqs = int(options["vllm_max_num_seqs"])
            kwargs["compilation_config"] = {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                # Avoid graph padding for variable multi-task scheduler
                # batches.  The configured maximum is normally only 16.
                "cudagraph_capture_sizes": list(range(1, max_num_seqs + 1)),
            }
    return kwargs


def vllm_kv_cache_capacity_tokens(
    engine: Any,
    model_path: str,
) -> tuple[int | None, str]:
    """Read the usable per-engine KV-token capacity after vLLM profiling.

    New vLLM exposes a group-aware token count, which is also correct for
    hybrid Qwen3.5 caches.  Older releases expose only block counts; that
    fallback is safe for the homogeneous full-attention Qwen3 checkpoint used
    by NVARC, but intentionally stays disabled for hybrid model families.
    """
    llm_engine = getattr(engine, "llm_engine", None)
    vllm_config = getattr(llm_engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return None, "unavailable"

    exact_capacity = getattr(cache_config, "kv_cache_size_tokens", None)
    if exact_capacity is not None and int(exact_capacity) > 0:
        return int(exact_capacity), "group-aware"

    model_type = None
    config_path = Path(model_path) / "config.json"
    try:
        with config_path.open() as file_handle:
            model_type = json.load(file_handle).get("model_type")
    except (OSError, TypeError, ValueError):
        pass
    if model_type != "qwen3":
        return None, f"unavailable-{model_type or 'unknown-model'}"

    num_blocks = getattr(cache_config, "num_gpu_blocks", None)
    block_size = getattr(cache_config, "block_size", None)
    if num_blocks is None or block_size is None:
        return None, "unavailable-qwen3-blocks"
    # vLLM permanently reserves one null block.  Excluding it avoids claiming
    # a few tokens that the prefix cache can never use.
    usable_blocks = max(0, int(num_blocks) - 1)
    capacity = usable_blocks * int(block_size)
    if capacity <= 0:
        return None, "unavailable-qwen3-empty"
    return capacity, "qwen3-block-fallback"


def configure_runtime_kv_capacity(
    engine: Any,
    runtime_options: dict[str, Any],
    model_path: str,
    *,
    log_prefix: str,
) -> None:
    """Publish one engine's profiled KV capacity to task batch planning."""
    capacity, source = vllm_kv_cache_capacity_tokens(engine, model_path)
    runtime_options["vllm_kv_cache_capacity_tokens"] = capacity
    runtime_options["vllm_kv_cache_capacity_source"] = source
    if capacity is None:
        print(
            f"{log_prefix} KV-resident decode batching unavailable "
            f"({source}); retaining requested inference batches",
            flush=True,
        )
        return
    requested_fraction = runtime_options.get("vllm_kv_resident_fraction", 0.95)
    max_loras = runtime_options.get("vllm_max_loras", 1)
    fraction = resolve_vllm_kv_resident_fraction(
        requested_fraction, max_loras=max_loras
    )
    auto_detail = (
        f" (auto, max_loras={max_loras})"
        if isinstance(requested_fraction, str)
        and requested_fraction.strip().lower() == "auto"
        else ""
    )
    print(
        f"{log_prefix} profiled GPU KV capacity={capacity} tokens "
        f"({source}); resident decode budget="
        f"{int(capacity * fraction)} tokens at fraction={fraction:.3f}{auto_detail}",
        flush=True,
    )


def _load_vllm_engine(model_path: str, options: dict[str, Any]):
    """Import and initialize vLLM only inside the reserved GPU process."""
    from pre_sft_rope_3d import read_rope_3d_config

    prepare_dflash_runtime(model_path, options)

    # Select V1 before vLLM/plugin imports in the frontend and spawned worker.
    if read_rope_3d_config(model_path) is not None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    model_root = Path(model_path)
    config_path = model_root / "config.json"
    routed = (model_root / "delta_mhar_config.json").is_file()
    if not routed and config_path.is_file():
        with config_path.open() as file_handle:
            routed = json.load(file_handle).get("unsloth_delta_mhar") is not None
    routed_spec = None
    hf_overrides = None
    if routed:
        if options.get("bidirectional_prompt_attention", False):
            raise RuntimeError(
                "Delta-MH and bidirectional prompt attention cannot share one "
                "vLLM executor yet"
            )
        try:
            from arc_vllm_delta_mhar import (
                PLUGIN_NAME,
                QWEN3_ARCHITECTURE,
                QWEN35_ARCHITECTURE,
                register as register_delta_mhar,
            )
            from arc_vllm_delta_mhar.config import read_delta_mhar_spec
        except ImportError as error:
            raise RuntimeError(
                "Routed vLLM inference requires the local plugin. Install it "
                "offline with `python -m pip install --no-deps -e "
                "vllm_delta_mhar_plugin`."
            ) from error
        allowed_plugins = [
            name for name in os.getenv("VLLM_PLUGINS", "").split(",") if name
        ]
        if allowed_plugins and PLUGIN_NAME not in allowed_plugins:
            raise RuntimeError(
                f"VLLM_PLUGINS excludes required plugin {PLUGIN_NAME!r}: "
                f"{allowed_plugins}"
            )
        routed_spec = read_delta_mhar_spec(model_path)
        hf_overrides = {
            "architectures": [
                QWEN35_ARCHITECTURE
                if routed_spec.family == "qwen3_5"
                else QWEN3_ARCHITECTURE
            ]
        }
        # Entry points reproduce registration in EngineCore/worker processes;
        # explicit registration makes the current frontend deterministic even
        # when another module imported vLLM before general plugins were loaded.
        register_delta_mhar()
    if options.get("bidirectional_prompt_attention", False):
        # Force the runner selected by ArcBidirectionalWorker in both the
        # frontend and EngineCore child before vLLM reads its environment.
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        from arc_vllm_bidirectional import (
            register_vllm_bidirectional_support,
        )

        register_vllm_bidirectional_support()
    from vllm import LLM, SamplingParams
    from vllm.config.kernel import KernelConfig
    from runtime_bootstrap import install_vllm_token_only_renderer_warmup

    # ARC sends prompt_token_ids with skip_tokenizer_init=True. Chat-template
    # warmup cannot run without a tokenizer; GPU and MM warmup stay intact.
    install_vllm_token_only_renderer_warmup()

    # Exact v0.23.0 predates this optional switch; newer 0.23 Kaggle wheels
    # add it. Detect the installed schema after the backend's lazy import.
    disable_cutedsl_warmup = (
        "enable_cutedsl_warmup" in inspect.signature(KernelConfig).parameters
    )

    # Construct both parameter sets before loading the model.  This is a
    # capability check for the v0.23 targeted/flat-logprob API.
    targeted_logprobs_only = bool(
        options.get("vllm_targeted_logprobs_only", False)
    )
    async_scheduling = bool(options.get("vllm_async_scheduling", False))
    decode_params = _make_decode_sampling_params(
        targeted_logprobs_only=targeted_logprobs_only,
        async_scheduling=async_scheduling,
        speculative_decoding=bool(options.get("vllm_draft_model")),
    )
    score_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
        prompt_logprobs=1,
        flat_logprobs=True,
    )
    cached_score_params = None
    uncached_prefix_score_params = None
    if options.get("vllm_prefix_score", False):
        prefix_score_kwargs = dict(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
            # The checkpoint vocabulary contains only 16 tokens. Returning all
            # of them lets us reconstruct the one boundary token whose logits
            # are omitted when a block-aligned cache hit enters the answer.
            prompt_logprobs=-1,
            flat_logprobs=True,
        )
        cached_score_params = SamplingParams(
            **prefix_score_kwargs,
            # vLLM defaults this to True for prompt-logprob requests because a
            # cache hit omits cached-position logprobs. ArcVllmTask reconstructs
            # matching answer-side block boundaries and retries unsafe hits.
            skip_reading_prefix_cache=False,
        )
        # An unsafe hit is retried without reading cache. Keep the full
        # 16-token distribution so that successful retry also seeds exact
        # answer-boundary metadata for later candidates instead of making each
        # candidate incur the same failed cached request again.
        uncached_prefix_score_params = SamplingParams(
            **prefix_score_kwargs,
            skip_reading_prefix_cache=True,
        )
    if routed_spec is not None:
        print(
            "[vLLM] routed checkpoint preflight: "
            f"family={routed_spec.family}, mode={routed_spec.mode}, "
            f"heads={routed_spec.num_heads}, block_size={routed_spec.block_size}, "
            f"layers={routed_spec.num_hidden_layers}, "
            f"routing_tensors={routed_spec.tensor_count}, "
            f"routing_source={'embedded_hf' if routed_spec.routing_weights_embedded else 'sidecar'}, "
            f"operator={'triton' if routed_spec.fused else 'reference'}",
            flush=True,
        )
    engine_kwargs = _vllm_engine_kwargs(
        model_path,
        options,
        hf_overrides=hf_overrides,
        disable_cutedsl_warmup=disable_cutedsl_warmup,
    )
    if routed_spec is not None:
        engine_kwargs["worker_cls"] = (
            "arc_vllm_delta_mhar.worker.ArcDeltaMHARWorker"
        )
    from arc_vllm_rope_3d import configure_vllm_rope_3d

    if configure_vllm_rope_3d(model_path, options, engine_kwargs):
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        print(
            "[vLLM] ARC 3D RoPE: checkpoint base restored; "
            "V1 synchronous executor with graph-safe grid-coordinate input buffer",
            flush=True,
        )
    if os.environ.get("ARC_PIPELINE_EXPECTED_MANIFEST"):
        from pipeline_runtime import configure_engine_runtime

        configure_engine_runtime(engine_kwargs)
    engine = LLM(**engine_kwargs)
    if os.environ.get("ARC_PIPELINE_EXPECTED_MANIFEST"):
        from pipeline_runtime import report_engine_runtime

        report_engine_runtime(engine)
    for method in ("add_lora", "remove_lora", "list_loras"):
        if not hasattr(engine.llm_engine, method):
            raise RuntimeError(
                f"Installed vLLM lacks required runtime LoRA method {method}"
            )
    try:
        install_pissa_fallback_adapter(engine, model_path, options)
    except BaseException:
        shutdown_vllm_engine(engine)
        raise
    return (
        engine,
        decode_params,
        score_params,
        cached_score_params,
        uncached_prefix_score_params,
    )


def install_pissa_fallback_adapter(engine, model_path, options):
    """Also called after engine recovery, before admitting any task adapter."""
    if options.get("pissa_base_mode") != "shared-residual" or not options.get("base_model_fallback"):
        return
    from arc_pissa_shared import (
        INITIAL_ADAPTER_DIR, FALLBACK_ADAPTER_ID, file_sha256, read_shared_manifest,
    )
    from vllm.lora.request import LoRARequest

    manifest = read_shared_manifest(model_path)
    relative = f"{INITIAL_ADAPTER_DIR}/adapter_model.safetensors"
    if (
        manifest["base_id"] != options["pissa_base_id"]
        or file_sha256(Path(model_path) / relative) != manifest["files"][relative]["sha256"]
    ):
        raise ValueError("PiSSA fallback adapter identity/checksum mismatch")
    request = LoRARequest(
        lora_name=f"arc-pissa-init-{options['pissa_base_id'][:16]}",
        lora_int_id=FALLBACK_ADAPTER_ID,
        lora_path=str(Path(model_path) / INITIAL_ADAPTER_DIR),
    )
    loaded = engine.llm_engine.add_lora(request)
    if not loaded and FALLBACK_ADAPTER_ID not in engine.llm_engine.list_loras():
        raise RuntimeError("vLLM could not load the initial PiSSA fallback adapter")
    options["_pissa_fallback_lora_request"] = request
    print("[vLLM] shared PiSSA fallback uses initial A0/B0; one extra LoRA slot reserved", flush=True)


def _load_vllm_engine_serialized(
    model_path: str,
    options: dict[str, Any],
    startup_lock: Any,
    *,
    log_prefix: str,
):
    """Load an EngineCore with bounded lock ownership and bounded retries."""
    timeout_seconds = float(options.get("vllm_startup_timeout_seconds", 600.0))
    attempts = int(options.get("vllm_startup_attempts", 2))
    if timeout_seconds <= 0:
        raise ValueError("vLLM startup timeout must be positive")
    if attempts < 1:
        raise ValueError("vLLM startup attempts must be positive")

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        lock_requested = time.perf_counter()
        acquired = False
        entered_context = False
        try:
            acquire = getattr(startup_lock, "acquire", None)
            if callable(acquire):
                acquired = bool(acquire(timeout=timeout_seconds + 30.0))
                if not acquired:
                    raise VllmEngineStartupTimeoutError(
                        f"{log_prefix} waited more than "
                        f"{timeout_seconds + 30.0:.1f}s for the serialized "
                        "EngineCore startup lock"
                    )
            else:
                startup_lock.__enter__()
                entered_context = True
                acquired = True
            lock_wait_seconds = time.perf_counter() - lock_requested

            previous_alarm_handler = signal.getsignal(signal.SIGALRM)
            previous_alarm = signal.getitimer(signal.ITIMER_REAL)

            def handle_startup_timeout(_signum, _frame) -> None:
                raise VllmEngineStartupTimeoutError(
                    f"{log_prefix} EngineCore construction exceeded "
                    f"{timeout_seconds:.1f}s"
                )

            signal.signal(signal.SIGALRM, handle_startup_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            load_started = time.perf_counter()
            try:
                result = _load_vllm_engine(model_path, options)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_alarm_handler)
                if previous_alarm[0] > 0.0:
                    signal.setitimer(
                        signal.ITIMER_REAL,
                        previous_alarm[0],
                        previous_alarm[1],
                    )
            return (
                result,
                lock_wait_seconds,
                time.perf_counter() - load_started,
                attempt,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            last_error = error
            if attempt >= attempts:
                raise
            # Do not serialize the recovery backoff. A sibling service must be
            # able to start while this process is preparing its next attempt.
            if acquired:
                if entered_context:
                    startup_lock.__exit__(None, None, None)
                else:
                    startup_lock.release()
                acquired = False
            print(
                f"{log_prefix} warning: EngineCore startup attempt "
                f"{attempt}/{attempts} failed: {type(error).__name__}: "
                f"{error}; retrying",
                flush=True,
            )
            gc.collect()
            backoff = float(options.get("vllm_restart_backoff_seconds", 3.0))
            if backoff > 0:
                time.sleep(backoff)
        finally:
            if acquired:
                if entered_context:
                    startup_lock.__exit__(None, None, None)
                else:
                    startup_lock.release()
    raise RuntimeError("Unreachable vLLM startup retry state") from last_error


def _arc_output_logprob_kwargs(
    *,
    targeted_logprobs_only: bool,
    async_scheduling: bool,
    speculative_decoding: bool = False,
) -> dict[str, Any]:
    """Select the stable vLLM output-logprob path for this scheduler mode.

    vLLM 0.23's asynchronous scheduler can overlap the request-index staging
    used by ``logprob_token_ids``.  A stale zero-length entry then returns an
    ordinary top-k response that is missing the explicitly requested ARC IDs.
    The ARC checkpoint has only 16 output tokens, so requesting the complete
    ordinary top-k avoids that shared mutable state with negligible work and
    preserves the exact raw full-vocabulary log probabilities.
    """
    # V2 speculative accepted-token logprobs do not consistently honor the
    # explicit ID list either. Top-16 covers the complete ARC vocabulary,
    # including the EOS probability needed at every DFS branch position.
    if async_scheduling or speculative_decoding:
        return {"logprobs": _ARC_CHECKPOINT_VOCAB_SIZE}
    return {
        "logprobs": None if targeted_logprobs_only else len(ARC_TOKENS),
        "logprob_token_ids": list(ARC_TOKENS),
    }


def _make_decode_sampling_params(
    *,
    targeted_logprobs_only: bool = False,
    async_scheduling: bool = False,
    speculative_decoding: bool = False,
):
    """Build exact-DFS params without sampling an unused non-ARC token."""
    from vllm import SamplingParams

    # raw_logprobs are captured by vLLM before applying allowed_token_ids, so
    # this does not renormalize or otherwise change the 12 probabilities used
    # by threshold DFS. It only keeps vLLM's automatically included sampled
    # token inside that requested set.
    logprob_kwargs = _arc_output_logprob_kwargs(
        targeted_logprobs_only=targeted_logprobs_only,
        async_scheduling=async_scheduling,
        speculative_decoding=speculative_decoding,
    )
    return SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        allowed_token_ids=list(ARC_TOKENS),
        detokenize=False,
        # When explicit token IDs are present, vLLM can compute their exact
        # full-vocabulary log-softmax values without also sorting the vocab.
        # Keep the historical top-k request as the default until the live
        # fixed-adapter A/B gate has passed.
        flat_logprobs=True,
        **logprob_kwargs,
    )


def _make_rollout_sampling_params(
    max_tokens: int,
    *,
    targeted_logprobs_only: bool = False,
    async_scheduling: bool = False,
    speculative_decoding: bool = False,
    cumulative_nll_stop: bool = False,
    initial_nll: float = 0.0,
    max_nll: float | None = None,
):
    """Constrain sampling while returning pre-mask full-vocabulary NLLs."""
    from vllm import SamplingParams

    # ARC token 15 terminates a completed grid.  It remains an explicit stop
    # token even though it is also the tokenizer EOS: with detokenization
    # disabled, vLLM 0.23 returns the stopping token ID and its matching
    # logprob position.  This lets RecursiveRolloutOriginSearch record the
    # completed beam without making vLLM generate the unused remainder of the
    # bounded rollout.  ``ignore_eos=True`` disables only vLLM's implicit EOS
    # handling; the explicit stop_token_ids rule still stops at token 15.
    extra_args = None
    if cumulative_nll_stop:
        if max_nll is None:
            raise ValueError("Cumulative-NLL rollout stopping requires max_nll")
        from arc_vllm_nll_stop_core import (
            ARC_CUMULATIVE_NLL_EXTRA_KEY,
            cumulative_nll_stop_spec,
        )

        extra_args = {
            ARC_CUMULATIVE_NLL_EXTRA_KEY: cumulative_nll_stop_spec(
                initial_nll,
                max_nll,
            )
        }
    logprob_kwargs = _arc_output_logprob_kwargs(
        targeted_logprobs_only=targeted_logprobs_only,
        async_scheduling=async_scheduling,
        speculative_decoding=speculative_decoding,
    )
    return SamplingParams(
        temperature=0.0,
        max_tokens=int(max_tokens),
        ignore_eos=True,
        stop_token_ids=[EOS_TOKEN_ID],
        include_stop_str_in_output=True,
        allowed_token_ids=None if cumulative_nll_stop else list(ARC_TOKENS),
        extra_args=extra_args,
        detokenize=False,
        flat_logprobs=True,
        **logprob_kwargs,
    )


def run_vllm_inference_worker(
    *,
    job_queue: Any,
    rollout_response_queues: Any,
    pending_adapter_slots: Any,
    service_index: int,
    service_outstanding: Any,
    service_load_lock: Any,
    startup_lock: Any,
    trainer_count: int,
    end_time: float,
    model_path: str,
    adapter_root: str,
    options: dict[str, Any],
    adapter_staging_slots: Any = None,
    abort_event: Any = None,
    all_job_queues: Any = None,
    service_active: Any = None,
    service_ready: Any = None,
    initial_service_count: int | None = None,
    post_training_scale: bool = False,
    post_training_rebalance_count: Any = None,
    post_training_rebalance_event: Any = None,
    shared_pending_queue: Any = None,
    shared_pending_count: Any = None,
) -> None:
    """Run the persistent multi-LoRA scheduler until all trainers finish."""
    # CUDA_LAUNCH_BLOCKING is intentionally scoped to EngineCore children.
    # Setting the ARC-prefixed control in the top-level command therefore
    # localizes an asynchronous illegal access without serializing LoRA
    # training on the other GPUs.
    if os.environ.get(
        "ARC_VLLM_CUDA_LAUNCH_BLOCKING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # EngineCore is a child of this service process, so promote the CLI option
    # to an environment variable before importing/spawning vLLM.  The patched
    # engine-side diagnostics intentionally have no dependency on ARC modules.
    forensics_dir = options.get("vllm_forensics_dir")
    if forensics_dir:
        os.environ["ARC_VLLM_FORENSICS_DIR"] = str(forensics_dir)
    from transformers import AutoTokenizer
    from vllm.lora.request import LoRARequest

    service_index = int(service_index)
    log_prefix = f"[vLLM {service_index}]"
    if service_ready is not None and len(service_ready) != len(
        service_outstanding
    ):
        raise ValueError(
            "service_ready and service_outstanding must have equal length"
        )

    service_retired = False

    def set_service_ready(ready: bool) -> None:
        # Finalization must not reopen a permanently closed routing slot.
        if service_ready is None or service_retired:
            return
        with service_load_lock:
            if int(service_ready[service_index]) >= 0:
                service_ready[service_index] = int(bool(ready))

    set_service_ready(False)
    initial_service_count = int(
        0 if initial_service_count is None else initial_service_count
    )
    post_training_scale = bool(post_training_scale)
    shared_pending_pool = shared_pending_queue is not None
    if shared_pending_pool != (shared_pending_count is not None):
        raise ValueError("The shared adapter pool requires both queue and counter")
    if post_training_scale:
        if any(
            value is None
            for value in (
                all_job_queues,
                service_active,
                post_training_rebalance_count,
                post_training_rebalance_event,
            )
        ):
            raise ValueError("Incomplete post-training vLLM scale-out configuration")
        if not 0 <= initial_service_count < len(all_job_queues):
            raise ValueError(
                "Post-training vLLM scale-out requires at least one future service"
            )
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def handle_startup_sigterm(signum, _frame) -> None:
        # Raising through LLM/MPClient construction lets vLLM's constructor
        # finalizers tear down any EngineCore process that was already spawned
        # but has not yet returned an engine object to this function.
        print(
            f"{log_prefix} received signal {signum} during engine startup; "
            "unwinding vLLM initialization",
            flush=True,
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, handle_startup_sigterm)
    startup_started = time.perf_counter()
    runtime_options = dict(options)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = ensure_arc_tokenizer_backend(tokenizer, model_path)
    formatter = QwenFormatter(tokenizer=tokenizer)
    (
        (
            engine,
            decode_params,
            score_params,
            cached_score_params,
            uncached_prefix_score_params,
        ),
        startup_lock_wait_seconds,
        engine_load_seconds,
        startup_attempt,
    ) = _load_vllm_engine_serialized(
        model_path,
        runtime_options,
        startup_lock,
        log_prefix=log_prefix,
    )
    configure_runtime_kv_capacity(
        engine,
        runtime_options,
        model_path,
        log_prefix=log_prefix,
    )

    # multiprocessing.util.Finalize is used instead of atexit because spawned
    # multiprocessing children may finish through os._exit().  The explicit
    # SIGTERM handler also covers PyTorch terminating sibling workers after a
    # trainer OOM.
    engine_finalizer = Finalize(
        None,
        shutdown_vllm_engine,
        args=(engine,),
        kwargs={"timeout_seconds": 3.0},
        exitpriority=20,
    )

    def finalize_engine(reason: str) -> None:
        set_service_ready(False)
        if not engine_finalizer.still_active():
            return
        print(f"{log_prefix} shutting down EngineCore ({reason})", flush=True)
        try:
            shutdown_found = engine_finalizer()
            if shutdown_found is False:
                print(
                    f"{log_prefix} warning: vLLM engine exposes no "
                    "EngineCore shutdown method",
                    flush=True,
                )
        except BaseException as error:
            # Cleanup errors must not replace the trainer's original OOM.
            print(
                f"{log_prefix} warning: EngineCore shutdown failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    def handle_sigterm(signum, _frame) -> None:
        finalize_engine(f"signal {signum}")
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, handle_sigterm)
    set_service_ready(True)
    startup_seconds = time.perf_counter() - startup_started
    print(
        f"{log_prefix} engine ready in {startup_seconds:.3f}s "
        f"(startup-lock wait={startup_lock_wait_seconds:.3f}s, "
        f"engine load={engine_load_seconds:.3f}s, "
        f"attempt={startup_attempt}) with "
        f"{runtime_options['vllm_max_loras']} LoRA slots",
        flush=True,
    )

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    arc_test_set = ArcDataset.from_file(resolve_challenge_path(rerun_mode))
    output_dir = os.getenv(
        "ARC_INFERENCE_OUTPUT_DIR",
        "/kaggle/inference_outputs",
    )
    os.makedirs(output_dir, exist_ok=True)
    telemetry = _ServiceTelemetry(
        service_index,
        options.get("vllm_telemetry_dir"),
    )
    forensics = _VllmForensics(
        service_index,
        options.get("vllm_forensics_dir"),
    )

    active: list[ArcVllmTask] = []
    pending_jobs: deque[dict[str, Any]] = deque()
    pending_rollouts: deque[dict[str, Any]] = deque()
    trainers_done = 0
    post_training_rebalance_sources: set[int] = set()
    total_generate_calls = 0
    total_engine_seconds = 0.0
    engine_restart_count = 0
    engine_restart_seconds = 0.0
    peak_sampled_gpu_memory_mb = query_visible_gpu_memory_mb()
    rollout_params: dict[int, Any] = {}
    service_started = time.time()
    round_robin_offset = 0
    deadline_mode = False
    aborted = False
    invalid_logprob_diagnostics = 0
    post_training_rebalanced = (
        shared_pending_pool or not post_training_scale or service_index >= initial_service_count
    )

    def log_invalid_logprob_response(
        work: _EngineWork,
        sample: Any,
        error: BaseException,
        *,
        source: str,
    ) -> None:
        """Log what vLLM returned without allowing a corrupt-engine flood."""
        nonlocal invalid_logprob_diagnostics
        limit = 8
        if invalid_logprob_diagnostics >= limit:
            return
        invalid_logprob_diagnostics += 1
        payload = summarize_invalid_arc_logprobs(
            getattr(sample, "logprobs", None),
            getattr(sample, "token_ids", ()),
        )
        payload.update(
            {
                "service": service_index,
                "task": work.task.key,
                "adapter": work.task.job.get("adapter_name"),
                "model_adapter_id": work.model_adapter_id,
                "inference_model": work.inference_model,
                "request_id": work.request_id,
                "source": source,
                "prompt_tokens": len(work.prompt_token_ids),
                "error": f"{type(error).__name__}: {error}",
                "diagnostic_index": invalid_logprob_diagnostics,
                "diagnostic_limit": limit,
            }
        )
        print(
            f"{log_prefix} invalid ARC logprob response: "
            f"{json.dumps(payload, sort_keys=True)}",
            flush=True,
        )

    def expired_adapter_tasks(now: float | None = None) -> list[ArcVllmTask]:
        """Return unfinished resident tasks whose hard watchdog expired."""
        current_time = time.time() if now is None else float(now)
        return [
            task
            for task in active
            if not task.done
            and current_time
            >= float(getattr(task, "adapter_watchdog_deadline", math.inf))
        ]

    def adapter_timeout_error(
        tasks: list[ArcVllmTask],
        now: float | None = None,
    ) -> AdapterInferenceTimeoutError:
        current_time = time.time() if now is None else float(now)
        return AdapterInferenceTimeoutError(
            [
                (
                    int(task.job["adapter_id"]),
                    task.key,
                    max(
                        0.0,
                        current_time - float(getattr(task, "started_at", current_time)),
                    ),
                )
                for task in tasks
            ],
            float(options.get("vllm_adapter_timeout_seconds", 3600.0)),
        )

    def engine_utility(method: Any, *args: Any) -> Any:
        """Bound a non-generate vLLM client call (LoRA load/unload/list)."""
        return run_engine_call_with_liveness(
            engine,
            method,
            *args,
            timeout_seconds=float(
                options.get("vllm_utility_timeout_seconds", 600.0)
            ),
        )

    def generate_with_adapter_watchdog(*args, **kwargs):
        """Interrupt a synchronous vLLM call at the earliest task deadline.

        The vLLM V1 client waits in this service process while EngineCore runs
        in a child.  ``SIGALRM`` therefore breaks a wedged MP-client wait so
        the normal replacement-engine path can reload all resident adapters.
        """
        unfinished = [task for task in active if not task.done]
        if not unfinished:
            return run_engine_call_with_liveness(
                engine, engine.generate, *args, **kwargs
            )

        now = time.time()
        expired = expired_adapter_tasks(now)
        if expired:
            raise adapter_timeout_error(expired, now)
        earliest_deadline = min(
            float(getattr(task, "adapter_watchdog_deadline", math.inf))
            for task in unfinished
        )
        if not math.isfinite(earliest_deadline):
            return run_engine_call_with_liveness(
                engine, engine.generate, *args, **kwargs
            )
        alarm_seconds = max(0.001, earliest_deadline - now)
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        alarm_started = time.monotonic()

        def handle_adapter_watchdog(_signum, _frame) -> None:
            # The periodic tick also catches an EngineCore that died while
            # this call was blocked; vLLM never wakes the caller itself.
            if vllm_engine_core_dead(engine):
                raise EngineCoreDeadError("vLLM engine core is dead")
            alarm_now = time.time()
            timed_out = expired_adapter_tasks(alarm_now)
            if not timed_out and alarm_now < earliest_deadline - 0.01:
                return
            if not timed_out:
                # Timer delivery can precede the wall-clock comparison by a
                # sub-millisecond. Attribute it to the earliest task(s).
                timed_out = [
                    task
                    for task in unfinished
                    if float(getattr(task, "adapter_watchdog_deadline", math.inf))
                    <= earliest_deadline
                ]
            raise adapter_timeout_error(timed_out, alarm_now)

        signal.signal(signal.SIGALRM, handle_adapter_watchdog)
        signal.setitimer(
            signal.ITIMER_REAL,
            min(alarm_seconds, _ENGINE_LIVENESS_POLL_SECONDS),
            _ENGINE_LIVENESS_POLL_SECONDS,
        )
        try:
            return engine.generate(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            previous_delay, previous_interval = previous_timer
            if previous_delay > 0:
                elapsed = time.monotonic() - alarm_started
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.001, previous_delay - elapsed),
                    previous_interval,
                )

    def receive(message: dict[str, Any]) -> None:
        nonlocal trainers_done
        kind = message.get("kind")
        if kind == "adapter":
            target_service = int(message.get("service_index", service_index))
            if target_service != service_index:
                raise RuntimeError(
                    f"{log_prefix} received adapter for service {target_service}"
                )
            if _PENDING_SLOT_HELD not in message:
                # Jobs forwarded from a peer already carry slot ownership.
                mark_pending_slot_held(message)
            pending_jobs.append(message)
        elif kind == "rl_rollout":
            target_service = int(message.get("service_index", service_index))
            if target_service != service_index:
                raise RuntimeError(
                    f"{log_prefix} received RL rollout for service {target_service}"
                )
            trainer_rank = int(message["trainer_rank"])
            if not 0 <= trainer_rank < len(rollout_response_queues):
                raise RuntimeError(
                    f"{log_prefix} received invalid trainer rank {trainer_rank}"
                )
            pending_rollouts.append(message)
        elif kind == "trainer_done":
            trainers_done += 1
            print(
                f"{log_prefix} trainer {message.get('rank')} finished "
                f"({trainers_done}/{trainer_count})",
                flush=True,
            )
        elif kind == "post_training_rebalance_done":
            target_service = int(message.get("service_index", service_index))
            if target_service != service_index:
                raise RuntimeError(
                    f"{log_prefix} received a rebalance sentinel for service "
                    f"{target_service}"
                )
            source_service = int(message["source_service_index"])
            if not 0 <= source_service < initial_service_count:
                raise RuntimeError(
                    f"{log_prefix} received invalid rebalance source {source_service}"
                )
            if source_service in post_training_rebalance_sources:
                raise RuntimeError(
                    f"{log_prefix} received duplicate rebalance sentinel "
                    f"from service {source_service}"
                )
            post_training_rebalance_sources.add(source_service)
            print(
                f"{log_prefix} received post-training rebalance sentinel "
                f"from initial service {source_service} "
                f"({len(post_training_rebalance_sources)}/"
                f"{initial_service_count})",
                flush=True,
            )
        else:
            raise RuntimeError(f"Unknown vLLM pipeline message: {message}")

    def upstream_complete() -> bool:
        if trainers_done < trainer_count:
            return False
        if shared_pending_pool or not post_training_scale or service_index < initial_service_count:
            return True
        return len(post_training_rebalance_sources) >= initial_service_count

    def close_idle_service() -> bool:
        if not upstream_complete() or active or pending_jobs or pending_rollouts:
            return False
        if shared_pending_pool and not service_retired:
            return finish_service_if_pool_drained(
                service_index, shared_pending_count, service_outstanding,
                service_ready, service_load_lock,
            )
        return finish_service_if_idle(
            service_index, service_outstanding, service_ready, service_load_lock,
        )

    def drain_nonblocking() -> None:
        while True:
            try:
                receive(job_queue.get_nowait())
            except Empty:
                break

    def claim_shared_job() -> dict[str, Any] | None:
        if not shared_pending_pool:
            return None
        job = claim_pending_adapter(
            shared_pending_queue, shared_pending_count, service_index,
            service_outstanding, service_load_lock,
        )
        if job is not None:
            if _PENDING_SLOT_HELD not in job:
                mark_pending_slot_held(job)
            pending_count, loads = pending_pool_snapshot(
                shared_pending_count, service_outstanding, service_load_lock,
            )
            print(
                f"{log_prefix} claimed {job['adapter_name']} from shared pool; "
                f"pending={pending_count}, claimed={list(loads)}", flush=True,
            )
        return job

    def discard_job(
        job: dict[str, Any],
        reason: str = "run deadline",
    ) -> None:
        remove_job_adapter_artifact(
            job,
            adapter_root,
            adapter_staging_slots,
        )
        release_job_pending_slot(job, pending_adapter_slots)
        loads = release_service_task(
            service_index,
            service_outstanding,
            service_load_lock,
        )
        print(
            f"{log_prefix} discarded adapter {job['adapter_name']}: "
            f"{reason}; outstanding={list(loads)}"
        )

    def admit_jobs() -> None:
        nonlocal peak_sampled_gpu_memory_mb
        # Keep one adapter slot available while trainers may issue synchronous
        # GRPO rollouts.  Once all trainers finish, final inference can consume
        # every configured resident slot.
        active_limit = int(runtime_options["vllm_max_loras"])
        if (
            options.get("rl_rollout_backend") == "vllm"
            and trainers_done < trainer_count
        ):
            active_limit = max(0, active_limit - 1)
        while len(active) < active_limit and not deadline_mode:
            # Never prefetch a private backlog. Recovery replays are the only
            # locally pending jobs in shared-pool mode and keep their owner.
            job = pending_jobs.popleft() if pending_jobs else claim_shared_job()
            if job is None:
                break
            try:
                job[_ADMISSION_IN_PROGRESS] = True
                from arc_pissa_shared import FALLBACK_ADAPTER_ID, validate_adapter_base

                if int(job["adapter_id"]) == FALLBACK_ADAPTER_ID:
                    raise ValueError("Task adapter ID collides with the reserved PiSSA fallback ID")
                validate_adapter_base(
                    job["adapter_path"], mode=runtime_options.get("pissa_base_mode", "original"),
                    base_id=runtime_options.get("pissa_base_id"), rank=runtime_options["lora_r"],
                    alpha=runtime_options.get("lora_alpha", 32.0),
                    method=runtime_options.get("lora_method", "rslora"),
                )
                request = LoRARequest(
                    lora_name=job["adapter_name"],
                    lora_int_id=int(job["adapter_id"]),
                    lora_path=job["adapter_path"],
                )
                load_started = time.perf_counter()
                loaded = engine_utility(engine.llm_engine.add_lora, request)
                load_seconds = time.perf_counter() - load_started
                loaded_ids = set(engine_utility(engine.llm_engine.list_loras))
                if not loaded and int(job["adapter_id"]) not in loaded_ids:
                    raise RuntimeError(
                        f"vLLM failed to load adapter {job['adapter_name']} "
                        f"from {job['adapter_path']}"
                    )
                print(
                    f"{log_prefix} loaded {job['adapter_name']} in "
                    f"{load_seconds:.3f}s; active adapters="
                    f"{sorted(loaded_ids)}",
                    flush=True,
                )
                task = ArcVllmTask(
                    job=job,
                    lora_request=request,
                    arc_test_set=arc_test_set,
                    tokenizer=tokenizer,
                    formatter=formatter,
                    output_dir=output_dir,
                    global_end_time=end_time,
                    options=runtime_options,
                )
                task.adapter_load_seconds = load_seconds
                job.pop(_ADMISSION_IN_PROGRESS, None)
            except BaseException:
                # add_lora can kill EngineCore before the task becomes active.
                # Keep both the job and its bounded publication-slot ownership
                # intact so a replacement engine can retry it.
                pending_jobs.appendleft(job)
                raise
            active.append(task)
            # The engine synchronously materialized this adapter, so it no
            # longer occupies a *pending* queue slot.  Retain its RAM/disk
            # artifact until inference and engine unload both complete; this
            # makes ownership explicit and still lets trainers publish more.
            release_job_pending_slot(job, pending_adapter_slots)
            queue_seconds = time.time() - float(job["published_at"])
            print(
                f"{log_prefix} accepted {job['adapter_name']} after "
                f"{queue_seconds:.3f}s queued; staging artifact retained "
                "until task inference completes",
                flush=True,
            )
            peak_sample = query_visible_gpu_memory_mb()
            if peak_sample >= 0:
                peak_sampled_gpu_memory_mb = max(
                    peak_sampled_gpu_memory_mb,
                    peak_sample,
                )

    def forward_job_to_live_service(job: dict[str, Any]) -> bool:
        """Hand one non-resident job to a live peer; False when none exists."""
        if shared_pending_pool:
            if not return_pending_adapter(
                shared_pending_queue, shared_pending_count, service_index,
                service_outstanding, service_ready, service_load_lock, job,
            ):
                # Keep the artifact for diagnosis. No engine can drain the pool.
                raise RuntimeError("All vLLM services retired with unfinished shared-pool work")
            print(
                f"{log_prefix} failover: returned {job['adapter_name']} to shared pool",
                flush=True,
            )
            return True
        if (
            all_job_queues is None
            or service_active is None
            or service_ready is None
        ):
            return False
        rebalance_done = (
            post_training_rebalance_event is not None
            and post_training_rebalance_event.is_set()
        )
        route = reassign_task_to_live_service(
            service_index,
            service_outstanding,
            service_active,
            service_ready,
            service_load_lock,
            include_unstarted=not rebalance_done,
        )
        if route is None:
            return False
        target_service_index, loads = route
        moved_job = dict(job)
        moved_job.pop(_ADMISSION_IN_PROGRESS, None)
        moved_job["service_index"] = target_service_index
        moved_job["failover_source_service_index"] = service_index
        try:
            all_job_queues[target_service_index].put(moved_job)
        except BaseException:
            transfer_service_task_assignment(
                target_service_index,
                service_index,
                service_outstanding,
                service_load_lock,
            )
            return False
        print(
            f"{log_prefix} failover: moved adapter {job['adapter_name']} to "
            f"service {target_service_index}; outstanding={list(loads)}",
            flush=True,
        )
        return True

    def forward_or_discard_job(job: dict[str, Any]) -> None:
        if not forward_job_to_live_service(job):
            discard_job(job, reason="service retired and no live peer available")

    def retire_service(error: BaseException, phase: str) -> bool:
        """Mark this service dead and hand all of its work to live peers."""
        nonlocal service_retired
        if (
            all_job_queues is None
            or service_active is None
            or service_ready is None
        ):
            return False
        mark_service_dead(service_ready, service_load_lock, service_index)
        service_retired = True
        print(
            f"{log_prefix} warning: retiring service after unrecoverable "
            f"{type(error).__name__} during {phase}: {error}; "
            f"active={len(active)}, pending={len(pending_jobs)}",
            flush=True,
        )
        if shared_pending_pool:
            with service_load_lock:
                has_peer = any(
                    index != service_index and int(ready) >= 0
                    and (bool(service_active[index]) or trainers_done >= trainer_count)
                    for index, ready in enumerate(service_ready)
                )
                unfinished = (
                    int(shared_pending_count.value) > 0
                    or any(int(n) > 0 for n in service_outstanding)
                    or trainers_done < trainer_count
                )
            if not has_peer and unfinished:
                # Do not fill a bounded pool waiting for an inactive ramp that
                # cannot be reached without first draining that same pool.
                raise RuntimeError("No live vLLM service remains to drain the shared adapter pool")
        for task in list(active):
            if getattr(task, "base_fallback_active", False):
                task.abort_base_model_fallback(f"{type(error).__name__}: {error}")
            if task.done:
                try:
                    remove_job_adapter_artifact(
                        task.job,
                        adapter_root,
                        adapter_staging_slots,
                    )
                except FileNotFoundError:
                    pass
                release_service_task(
                    service_index,
                    service_outstanding,
                    service_load_lock,
                )
                continue
            task.remove_written_candidate_files()
            pending_jobs.append(task.job)
        active.clear()
        while pending_rollouts:
            discard_rollout(
                pending_rollouts.popleft(),
                "vLLM service retired before GRPO rollout",
            )
        while pending_jobs:
            forward_or_discard_job(pending_jobs.popleft())
        return True

    def recover_engine(error: BaseException, phase: str) -> bool:
        """Restart a failed/wedged EngineCore and replay unfinished jobs."""
        nonlocal engine
        nonlocal decode_params
        nonlocal score_params
        nonlocal cached_score_params
        nonlocal uncached_prefix_score_params
        nonlocal engine_finalizer
        nonlocal engine_restart_count
        nonlocal engine_restart_seconds
        nonlocal round_robin_offset
        nonlocal runtime_options

        timeout_error = find_adapter_timeout_error(error)
        if timeout_error is None:
            if not bool(options.get("vllm_auto_recover", False)):
                return False
            if not is_recoverable_vllm_engine_error(error):
                return False
        set_service_ready(False)
        formatted_traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ).rstrip()
        incident_label = (
            "adapter watchdog timeout"
            if timeout_error is not None
            else "recoverable vLLM failure"
        )
        print(
            f"{log_prefix} warning: {incident_label} traceback during "
            f"{phase}:\n{formatted_traceback}",
            flush=True,
        )
        max_adapter_recoveries = int(
            options.get("vllm_max_adapter_recoveries", 3)
        )
        restart_number = engine_restart_count + 1
        restart_started = time.perf_counter()
        requeued_jobs: list[dict[str, Any]] = []
        dropped_jobs: list[dict[str, Any]] = []
        failure_counts: dict[int, int] = {}
        timeout_counts: dict[int, int] = {}
        completed = 0
        removed_partial_files = 0
        timed_out_adapter_ids = (
            timeout_error.timed_out_adapter_ids
            if timeout_error is not None
            else frozenset()
        )
        for task in list(active):
            if getattr(task, "base_fallback_active", False):
                task.abort_base_model_fallback(f"{type(error).__name__}: {error}")
                print(task.summary(), flush=True)
                task.report_accuracy()
            if task.done:
                # Its candidate files are complete. The dead EngineCore no
                # longer needs an explicit adapter unload, so finish ownership
                # locally and retain the valid result.
                try:
                    remove_job_adapter_artifact(
                        task.job,
                        adapter_root,
                        adapter_staging_slots,
                    )
                except FileNotFoundError:
                    pass
                release_service_task(
                    service_index,
                    service_outstanding,
                    service_load_lock,
                )
                completed += 1
                continue
            removed_partial_files += task.remove_written_candidate_files()
            adapter_id = int(task.job["adapter_id"])
            if timeout_error is not None:
                if adapter_id in timed_out_adapter_ids:
                    timeout_count, retry = record_adapter_timeout(task.job)
                    timeout_counts[adapter_id] = timeout_count
                else:
                    # The service restart invalidates every resident LoRA, but
                    # peers that did not expire consume no retry budget.
                    retry = True
            else:
                failure_count, retry = record_adapter_engine_failure(
                    task.job,
                    max_adapter_recoveries,
                )
                failure_counts[adapter_id] = failure_count
            if retry:
                requeued_jobs.append(task.job)
            else:
                dropped_jobs.append(task.job)
        active.clear()

        # add_lora/list_loras can kill EngineCore before ArcVllmTask exists.
        # Only the marked admission job participated in that failure; ordinary
        # queued adapters must not inherit its counter.
        retained_pending: deque[dict[str, Any]] = deque()
        while pending_jobs:
            job = pending_jobs.popleft()
            if job.pop(_ADMISSION_IN_PROGRESS, False):
                if timeout_error is None:
                    failure_count, retry = record_adapter_engine_failure(
                        job,
                        max_adapter_recoveries,
                    )
                    failure_counts[int(job["adapter_id"])] = failure_count
                    if not retry:
                        dropped_jobs.append(job)
                        continue
            retained_pending.append(job)
        pending_jobs.extend(retained_pending)
        for job in reversed(requeued_jobs):
            pending_jobs.appendleft(job)
        for job in dropped_jobs:
            discard_job(
                job,
                reason=(
                    "adapter inference watchdog expired twice"
                    if timeout_error is not None
                    else "fatal EngineCore recovery budget exhausted"
                ),
            )
        round_robin_offset = 0

        runtime_options = dict(options)
        if timeout_error is not None:
            retry_details = (
                f"adapter_timeouts={timeout_counts}, timeout_restarts="
                f"{_MAX_ADAPTER_TIMEOUT_RESTARTS}, "
                "fatal-recovery counters unchanged"
            )
        else:
            retry_details = (
                f"adapter_failures={failure_counts}, "
                f"limit={max_adapter_recoveries} recoveries"
            )
        print(
            f"{log_prefix} warning: {incident_label} during {phase}: "
            f"{type(error).__name__}: {error}; restarting engine "
            f"#{restart_number}, {retry_details}, "
            f"requeued={len(requeued_jobs)}, dropped={len(dropped_jobs)}, "
            f"completed={completed}, removed_partial="
            f"{removed_partial_files}, pending={len(pending_jobs)}, "
            f"max_loras={runtime_options['vllm_max_loras']}",
            flush=True,
        )

        finalize_engine(f"auto recovery engine #{restart_number}")
        engine = None
        gc.collect()
        backoff = float(options.get("vllm_restart_backoff_seconds", 3.0))
        if backoff > 0:
            time.sleep(backoff)

        try:
            signal.signal(signal.SIGTERM, handle_startup_sigterm)
            (
                (
                    engine,
                    decode_params,
                    score_params,
                    cached_score_params,
                    uncached_prefix_score_params,
                ),
                lock_wait,
                load_seconds,
                startup_attempt,
            ) = _load_vllm_engine_serialized(
                model_path,
                runtime_options,
                startup_lock,
                log_prefix=log_prefix,
            )
            configure_runtime_kv_capacity(
                engine,
                runtime_options,
                model_path,
                log_prefix=log_prefix,
            )
            engine_finalizer = Finalize(
                None,
                shutdown_vllm_engine,
                args=(engine,),
                kwargs={"timeout_seconds": 3.0},
                exitpriority=20,
            )
        except BaseException as restart_error:
            if isinstance(restart_error, Exception) and retire_service(
                restart_error, f"engine restart after {phase}"
            ):
                return True
            raise RuntimeError(
                f"{log_prefix} failed to restart EngineCore after "
                f"{type(error).__name__} during {phase}"
            ) from restart_error
        finally:
            signal.signal(signal.SIGTERM, handle_sigterm)

        set_service_ready(True)
        engine_restart_count = restart_number
        restart_seconds = time.perf_counter() - restart_started
        engine_restart_seconds += restart_seconds
        print(
            f"{log_prefix} replacement EngineCore ready in "
            f"{restart_seconds:.3f}s (startup-lock wait={lock_wait:.3f}s, "
            f"engine load={load_seconds:.3f}s, attempt={startup_attempt}); "
            "unfinished adapters will "
            "reload from retained staging artifacts",
            flush=True,
        )
        return True

    def finish_tasks() -> None:
        for task in list(active):
            if not task.done:
                continue
            unload_started = time.perf_counter()
            removed = engine_utility(
                engine.llm_engine.remove_lora, int(task.job["adapter_id"])
            )
            unload_seconds = time.perf_counter() - unload_started
            if not removed:
                raise RuntimeError(
                    f"vLLM failed to unload adapter {task.job['adapter_name']}"
                )
            remove_job_adapter_artifact(
                task.job,
                adapter_root,
                adapter_staging_slots,
            )
            active.remove(task)
            loads = release_service_task(
                service_index,
                service_outstanding,
                service_load_lock,
            )
            print(task.summary(), flush=True)
            task.report_accuracy()
            print(
                f"{log_prefix} unloaded {task.job['adapter_name']} in "
                f"{unload_seconds:.3f}s; active adapters="
                f"{sorted(engine_utility(engine.llm_engine.list_loras))}; "
                f"RAM/disk artifact removed; outstanding={list(loads)}",
                flush=True,
            )

    def respond_rollout_error(job: dict[str, Any], error: BaseException) -> None:
        trainer_rank = int(job["trainer_rank"])
        rollout_response_queues[trainer_rank].put(
            {
                "request_id": job.get("request_id"),
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def discard_rollout(job: dict[str, Any], reason: str) -> None:
        try:
            remove_job_adapter_artifact(
                job,
                adapter_root,
                adapter_staging_slots,
            )
        except FileNotFoundError:
            pass
        respond_rollout_error(job, RuntimeError(reason))

    def run_rollout(job: dict[str, Any]) -> None:
        """Load one exact adapter version and sample its GRPO completions."""
        nonlocal total_generate_calls
        nonlocal total_engine_seconds
        nonlocal peak_sampled_gpu_memory_mb

        from vllm import SamplingParams

        trainer_rank = int(job["trainer_rank"])
        response_queue = rollout_response_queues[trainer_rank]
        request = LoRARequest(
            lora_name=job["adapter_name"],
            lora_int_id=int(job["adapter_id"]),
            lora_path=job["adapter_path"],
        )
        loaded = False
        artifact_released = False
        load_seconds = 0.0
        generate_seconds = 0.0
        queue_seconds = time.time() - float(job["published_at"])
        try:
            load_started = time.perf_counter()
            loaded = bool(engine.llm_engine.add_lora(request))
            load_seconds = time.perf_counter() - load_started
            if (
                not loaded
                and int(job["adapter_id"]) not in engine.llm_engine.list_loras()
            ):
                raise RuntimeError(
                    f"vLLM failed to load RL adapter {job['adapter_name']}"
                )
            loaded = True
            # add_lora synchronously owns the tensors.  Free the 1 GiB handoff
            # before generation so the trainer can publish its next version.
            remove_job_adapter_artifact(
                job,
                adapter_root,
                adapter_staging_slots,
            )
            artifact_released = True

            params = SamplingParams(
                n=int(job["num_generations"]),
                temperature=float(job["temperature"]),
                top_p=float(job["top_p"]),
                top_k=int(job["top_k"]),
                min_p=float(job["min_p"]),
                repetition_penalty=float(job["repetition_penalty"]),
                max_tokens=int(job["max_tokens"]),
                ignore_eos=True,
                stop_token_ids=[EOS_TOKEN_ID],
                include_stop_str_in_output=True,
                detokenize=False,
                seed=int(job["seed"]),
            )
            for prompt_index, token_ids in enumerate(job["prompt_token_ids"]):
                validate_vllm_context_request(
                    token_ids,
                    max_tokens=int(job["max_tokens"]),
                    max_model_len=int(options["max_seq_length"]),
                    request_label=(
                        f"ARC RL rollout {job['request_id']} prompt {prompt_index}"
                    ),
                )
            prompts = [
                {"prompt_token_ids": list(map(int, token_ids))}
                for token_ids in job["prompt_token_ids"]
            ]
            generate_started = time.perf_counter()
            outputs = engine.generate(
                prompts,
                sampling_params=params,
                lora_request=request,
                use_tqdm=False,
            )
            generate_seconds = time.perf_counter() - generate_started
            total_generate_calls += 1
            total_engine_seconds += generate_seconds
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} prompt groups; "
                    f"expected {len(prompts)}"
                )
            completion_ids = [
                list(map(int, sample.token_ids))
                for output in outputs
                for sample in output.outputs
            ]
            expected = len(prompts) * int(job["num_generations"])
            if len(completion_ids) != expected:
                raise RuntimeError(
                    f"vLLM returned {len(completion_ids)} RL completions; "
                    f"expected {expected}"
                )
            telemetry.record(
                phase="rl_rollout",
                kinds=["rl"] * len(prompts),
                prompt_lengths=[len(tokens) for tokens in job["prompt_token_ids"]],
                cached_tokens=[
                    int(getattr(output, "num_cached_tokens", 0) or 0)
                    for output in outputs
                ],
                generated_tokens=sum(len(tokens) for tokens in completion_ids),
                adapter_ids=[int(job["adapter_id"])] * len(prompts),
                task_keys=[str(job.get("task_key", "rl"))] * len(prompts),
                max_tokens=[int(job["max_tokens"])] * len(prompts),
                engine_seconds=generate_seconds,
            )
            response_queue.put(
                {
                    "request_id": job["request_id"],
                    "completion_ids": completion_ids,
                    "queue_seconds": queue_seconds,
                    "load_seconds": load_seconds,
                    "generate_seconds": generate_seconds,
                    "completion_lengths": [
                        len(token_ids) for token_ids in completion_ids
                    ],
                }
            )
            print(
                f"{log_prefix} RL rollout {job['request_id']}: "
                f"prompts={len(prompts)}, generations={expected}, "
                f"load={load_seconds:.3f}s, "
                f"generate={generate_seconds:.3f}s",
                flush=True,
            )
            peak_sample = query_visible_gpu_memory_mb()
            if peak_sample >= 0:
                peak_sampled_gpu_memory_mb = max(
                    peak_sampled_gpu_memory_mb,
                    peak_sample,
                )
        except BaseException as error:
            respond_rollout_error(job, error)
        finally:
            if loaded:
                removed = engine.llm_engine.remove_lora(int(job["adapter_id"]))
                if not removed:
                    print(
                        f"{log_prefix} warning: failed to unload RL adapter "
                        f"{job['adapter_name']}",
                        flush=True,
                    )
            if not artifact_released:
                try:
                    remove_job_adapter_artifact(
                        job,
                        adapter_root,
                        adapter_staging_slots,
                    )
                except FileNotFoundError:
                    pass

    def run_exact_decode_fallback(
        work: _EngineWork,
        chunk_error: BaseException,
    ) -> tuple[tuple[int, ...], tuple[dict[int, float], ...]]:
        """Replay one failed leaf chunk as an exact one-token DFS step."""
        nonlocal total_generate_calls
        nonlocal total_engine_seconds

        if work.task.decode_cache_salt is None:
            work.task.decode_cache_salt = (
                f"arc-exact-{service_index}-"
                f"{work.task.job['adapter_id']}-{time.time_ns()}"
            )
        fallback_started = time.perf_counter()
        fallback_outputs = generate_with_adapter_watchdog(
            [_engine_prompt(work)],
            sampling_params=[decode_params],
            lora_request=[work.lora_request],
            use_tqdm=False,
        )
        fallback_seconds = time.perf_counter() - fallback_started
        total_generate_calls += 1
        total_engine_seconds += fallback_seconds
        if len(fallback_outputs) != 1 or len(fallback_outputs[0].outputs) != 1:
            raise RuntimeError(
                "vLLM exact fallback returned an unexpected output count: "
                f"groups={len(fallback_outputs)}, samples="
                f"{[len(item.outputs) for item in fallback_outputs]}"
            ) from chunk_error
        fallback_output = fallback_outputs[0]
        sample = fallback_output.outputs[0]
        sampled_token_ids = tuple(map(int, sample.token_ids))
        if len(sampled_token_ids) != 1 or sample.logprobs is None:
            raise RuntimeError(
                "vLLM exact fallback did not return one token with logprobs: "
                f"tokens={sampled_token_ids}, "
                f"has_logprobs={sample.logprobs is not None}"
            ) from chunk_error
        try:
            token_logprobs = requested_arc_logprobs(
                sample.logprobs,
                position=-1,
            )
        except (TypeError, ValueError) as fallback_error:
            log_invalid_logprob_response(
                work,
                sample,
                fallback_error,
                source="exact-one-token-fallback",
            )
            raise RecoverableVllmResponseError(
                "vLLM leaf chunk and exact one-token replay both returned "
                "invalid requested ARC logprobs: "
                f"task={work.task.key}, "
                f"adapter={work.task.job['adapter_name']}, "
                f"request_id={work.request_id}, "
                f"prompt_tokens={len(work.prompt_token_ids)}"
            ) from fallback_error

        search = work.task.search
        cap_rollout = getattr(search, "cap_rollout_chunk_tokens", None)
        if not callable(cap_rollout):
            raise RuntimeError(
                "Cannot downgrade a failed constrained rollout because its "
                f"search type is {type(search).__name__}"
            ) from chunk_error
        cap_rollout(1)
        work.task.decode_exact_fallbacks += 1
        work.task.record_engine_call(
            "decode",
            fallback_seconds,
            len(work.prompt_token_ids),
            int(getattr(fallback_output, "num_cached_tokens", 0) or 0),
        )
        telemetry.record_works(
            [work],
            fallback_outputs,
            phase="exact_fallback",
            engine_seconds=fallback_seconds,
        )
        print(
            f"{work.task.log_prefix} warning: downgraded puzzle "
            f"{work.task.key} to exact one-token DFS after a non-finite "
            f"leaf-rollout logprob at request {work.request_id}; "
            f"prompt_tokens={len(work.prompt_token_ids)}, "
            f"fallback={fallback_seconds:.3f}s, "
            f"reason={chunk_error}",
            flush=True,
        )
        return sampled_token_ids, (token_logprobs,)

    def accept_engine_outputs(
        works: list[_EngineWork],
        outputs: list[Any],
        elapsed: float,
    ) -> None:
        """Apply one successful service batch to its CPU search states."""
        if len(outputs) != len(works):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(works)} requests"
            )
        call_participants: dict[
            tuple[int, str],
            tuple[ArcVllmTask, int, int],
        ] = {}
        for work, output in zip(works, outputs):
            participant_key = (id(work.task), work.kind)
            task, prompt_tokens, cached_tokens = call_participants.get(
                participant_key,
                (work.task, 0, 0),
            )
            call_participants[participant_key] = (
                task,
                prompt_tokens + len(work.prompt_token_ids),
                cached_tokens + int(getattr(output, "num_cached_tokens", 0) or 0),
            )
        for (_task_id, kind), (
            task,
            prompt_tokens,
            cached_tokens,
        ) in call_participants.items():
            # A service call is shared by every request in its batch. Charge
            # its wall time once per participating task/phase, not once per
            # sequence, while still counting every prompt token.
            task.record_engine_call(
                kind,
                elapsed,
                prompt_tokens,
                cached_tokens,
            )
        for work, output in zip(works, outputs):
            if work.kind == "decode":
                if not output.outputs or output.outputs[0].logprobs is None:
                    raise RuntimeError("vLLM decode response has no sample logprobs")
                try:
                    if work.constrained_rollout:
                        sampled_token_ids = tuple(output.outputs[0].token_ids)
                        if len(output.outputs[0].logprobs) != len(sampled_token_ids):
                            raise RuntimeError(
                                "vLLM constrained rollout returned different "
                                "token/logprob lengths: "
                                f"{len(sampled_token_ids)} != "
                                f"{len(output.outputs[0].logprobs)}"
                            )
                        try:
                            position_logprobs = tuple(
                                requested_arc_logprobs(
                                    output.outputs[0].logprobs,
                                    position=position,
                                )
                                for position in range(len(output.outputs[0].logprobs))
                            )
                        except (TypeError, ValueError) as chunk_error:
                            log_invalid_logprob_response(
                                work,
                                output.outputs[0],
                                chunk_error,
                                source="constrained-rollout",
                            )
                            (
                                sampled_token_ids,
                                position_logprobs,
                            ) = run_exact_decode_fallback(
                                work,
                                chunk_error,
                            )
                        work.task.accept_decode(
                            work.request_id,
                            position_logprobs,
                            sampled_token_ids=sampled_token_ids,
                        )
                    else:
                        token_logprobs = requested_arc_logprobs(
                            output.outputs[0].logprobs,
                            position=-1,
                        )
                        work.task.accept_decode(
                            work.request_id,
                            token_logprobs,
                        )
                except (TypeError, ValueError) as exc:
                    log_invalid_logprob_response(
                        work,
                        output.outputs[0],
                        exc,
                        source=(
                            "constrained-rollout"
                            if work.constrained_rollout
                            else "exact-one-token-decode"
                        ),
                    )
                    raise RecoverableVllmResponseError(
                        "vLLM returned invalid requested ARC logprobs: "
                        f"task={work.task.key}, "
                        f"adapter={work.task.job['adapter_name']}, "
                        f"request_id={work.request_id}, "
                        f"constrained_rollout={work.constrained_rollout}, "
                        f"prompt_tokens={len(work.prompt_token_ids)}"
                    ) from exc
            else:
                if output.prompt_logprobs is None:
                    raise RuntimeError("vLLM score response has no prompt logprobs")
                work.task.accept_score(
                    work.request_id,
                    output.prompt_logprobs,
                    cached_tokens=int(getattr(output, "num_cached_tokens", 0) or 0),
                )

    while True:
        if service_retired:
            break
        if abort_event is not None and abort_event.is_set():
            aborted = True
            print(
                f"{log_prefix} fatal trainer/service failure received; abort inference",
                flush=True,
            )
            break
        timed_out_tasks = expired_adapter_tasks()
        if timed_out_tasks:
            if recover_engine(
                adapter_timeout_error(timed_out_tasks),
                "adapter watchdog check",
            ):
                continue
        drain_nonblocking()
        if not post_training_rebalanced and trainers_done >= trainer_count:
            rebalance_summary = rebalance_pending_jobs_after_training(
                pending_jobs=pending_jobs,
                service_index=service_index,
                initial_service_count=initial_service_count,
                all_job_queues=all_job_queues,
                service_outstanding=service_outstanding,
                service_load_lock=service_load_lock,
                rebalance_count=post_training_rebalance_count,
                rebalance_complete_event=post_training_rebalance_event,
            )
            post_training_rebalanced = True
            print(
                f"{log_prefix} post-training backlog rebalance: "
                f"original_pending={rebalance_summary['original_pending']}, "
                f"retained={rebalance_summary['retained_pending']}, "
                f"migrated={rebalance_summary['migrated_by_service']}, "
                f"completed_initial_services="
                f"{rebalance_summary['completed_services']}/"
                f"{initial_service_count}, outstanding="
                f"{list(rebalance_summary['service_loads'])}",
                flush=True,
            )
        if time.time() >= end_time:
            deadline_mode = True
        if deadline_mode:
            while pending_rollouts:
                discard_rollout(
                    pending_rollouts.popleft(),
                    "run deadline reached before vLLM GRPO rollout",
                )
            while pending_jobs:
                discard_job(pending_jobs.popleft())
            while (job := claim_shared_job()) is not None:
                discard_job(job)
            # Mark active searches expired and serialize only work that already
            # completed naturally.  Published adapter artifacts are cleaned on
            # a normal deadline, but retained on exceptions for diagnosis.
            for task in active:
                if getattr(task, "base_fallback_active", False):
                    task.abort_base_model_fallback(
                        "run deadline reached during base-model fallback"
                    )
                    continue
                task.global_end_time = 0.0
                task.inference_deadline = 0.0
                if task.search is not None:
                    task.search = None
                task.score_jobs.clear()
                task.outstanding_scores.clear()
                task.pending_aggregates.clear()
                task.pending_output_groups.clear()
                task.batch_index = len(task.batches)
                task.fallback_groups = []
                task.settle()
            try:
                finish_tasks()
            except BaseException as error:
                if recover_engine(error, "deadline adapter unload"):
                    continue
                raise
            if close_idle_service():
                break
            try:
                receive(job_queue.get(timeout=0.1 if shared_pending_pool else 0.5))
            except Empty:
                continue
            continue

        if pending_rollouts:
            # GRPO trainers synchronously wait for these responses.  Prioritize
            # them over asynchronous final-answer DFS work.
            run_rollout(pending_rollouts.popleft())
            continue

        try:
            admit_jobs()
            for task in active:
                task.settle()
            finish_tasks()
            admit_jobs()
        except BaseException as error:
            if recover_engine(error, "adapter admission/unload"):
                continue
            raise

        if not active:
            if close_idle_service():
                break
            try:
                receive(job_queue.get(timeout=0.1 if shared_pending_pool else 0.5))
            except Empty:
                continue
            continue

        max_requests = int(options["vllm_max_num_seqs"])
        works: list[_EngineWork] = []
        ordered_tasks = active[round_robin_offset:] + active[:round_robin_offset]
        fair_quota = max(1, math.ceil(max_requests / len(ordered_tasks)))
        for task in ordered_tasks:
            if len(works) >= max_requests:
                break
            works.extend(task.take_work(min(fair_quota, max_requests - len(works))))
        round_robin_offset = (round_robin_offset + 1) % max(1, len(active))
        if not works:
            # Local state may have completed after the last response.
            for task in active:
                task.settle()
            try:
                finish_tasks()
            except BaseException as error:
                if recover_engine(error, "idle adapter unload"):
                    continue
                raise
            if active:
                time.sleep(0.01)
            continue

        prepare_started = time.perf_counter()
        for work in works:
            validate_engine_work_context(
                work,
                int(options["max_seq_length"]),
            )
        prompts = [_engine_prompt(work) for work in works]
        sampling_params = []
        bidirectional_prompt_attention = bool(
            options.get("bidirectional_prompt_attention", False)
        )
        for work in works:
            if work.kind == "score":
                if (
                    options.get("vllm_prefix_score", False)
                    and work.score_job is not None
                ):
                    params = (
                        cached_score_params
                        if work.score_job.allow_prefix_cache
                        else uncached_prefix_score_params
                    )
                else:
                    params = score_params
            elif work.constrained_rollout:
                cumulative_nll_stop = bool(
                    options.get("vllm_cumulative_nll_stop", False)
                )
                params = (
                    None
                    if cumulative_nll_stop
                    else rollout_params.get(work.max_tokens)
                )
                if params is None:
                    params = _make_rollout_sampling_params(
                        work.max_tokens,
                        targeted_logprobs_only=bool(
                            options.get("vllm_targeted_logprobs_only", False)
                        ),
                        async_scheduling=bool(
                            options.get("vllm_async_scheduling", False)
                        ),
                        speculative_decoding=bool(options.get("vllm_draft_model")),
                        cumulative_nll_stop=cumulative_nll_stop,
                        initial_nll=work.cumulative_nll,
                        max_nll=float(options["max_beam_score"]),
                    )
                    if not cumulative_nll_stop:
                        rollout_params[work.max_tokens] = params
            else:
                params = decode_params
            sampling_params.append(
                _sampling_params_with_arc_attention(
                    params,
                    work.prompt_token_ids,
                    enabled=bidirectional_prompt_attention,
                )
            )
        lora_requests = [work.lora_request for work in works]
        prepare_seconds = time.perf_counter() - prepare_started
        try:
            loaded_loras_snapshot: Any = sorted(
                engine_utility(engine.llm_engine.list_loras)
            )
        except BaseException as snapshot_error:
            loaded_loras_snapshot = (
                f"{type(snapshot_error).__name__}: {snapshot_error}"
            )
        forensics.begin_call(
            works,
            sampling_params,
            active=active,
            pending_jobs=len(pending_jobs),
            loaded_loras=loaded_loras_snapshot,
            options=runtime_options,
        )
        generate_started = time.perf_counter()
        try:
            outputs = generate_with_adapter_watchdog(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )
        except BaseException as error:
            forensics.record_failure(
                "generate",
                error,
                works=works,
                sampling_params=sampling_params,
                active=active,
                pending_jobs=len(pending_jobs),
                loaded_loras=loaded_loras_snapshot,
                gpu_memory_mb=query_visible_gpu_memory_mb(),
            )
            if recover_engine(error, "generate"):
                continue
            raise
        elapsed = time.perf_counter() - generate_started
        forensics.record_result(outputs, elapsed)
        total_generate_calls += 1
        total_engine_seconds += elapsed
        response_started = time.perf_counter()
        try:
            accept_engine_outputs(works, outputs, elapsed)
        except BaseException as error:
            try:
                loaded_loras_snapshot = sorted(
                    engine_utility(engine.llm_engine.list_loras)
                )
            except BaseException as snapshot_error:
                loaded_loras_snapshot = (
                    f"{type(snapshot_error).__name__}: {snapshot_error}"
                )
            forensics.record_failure(
                "response_processing",
                error,
                works=works,
                outputs=outputs,
                sampling_params=sampling_params,
                active=active,
                pending_jobs=len(pending_jobs),
                loaded_loras=loaded_loras_snapshot,
                gpu_memory_mb=query_visible_gpu_memory_mb(),
            )
            if recover_engine(error, "response processing/exact replay"):
                continue
            raise
        response_seconds = time.perf_counter() - response_started
        telemetry.record_works(
            works,
            outputs,
            phase="search",
            engine_seconds=elapsed,
            prepare_seconds=prepare_seconds,
            response_seconds=response_seconds,
        )

    while service_retired and not aborted:
        # Keep draining this queue so a trainer that reserved the service
        # before it died cannot strand an adapter, and so a retired initial
        # service still completes its share of the post-training rebalance.
        if abort_event is not None and abort_event.is_set():
            aborted = True
            break
        if trainers_done >= trainer_count and not post_training_rebalanced:
            rebalance_pending_jobs_after_training(
                pending_jobs=pending_jobs,
                service_index=service_index,
                initial_service_count=initial_service_count,
                all_job_queues=all_job_queues,
                service_outstanding=service_outstanding,
                service_load_lock=service_load_lock,
                rebalance_count=post_training_rebalance_count,
                rebalance_complete_event=post_training_rebalance_event,
            )
            post_training_rebalanced = True
        if post_training_rebalanced and close_idle_service():
            break
        if time.time() >= end_time:
            break
        try:
            receive(job_queue.get(timeout=0.5))
        except Empty:
            continue
        while pending_jobs:
            forward_or_discard_job(pending_jobs.popleft())
        while pending_rollouts:
            discard_rollout(
                pending_rollouts.popleft(),
                "vLLM service retired before GRPO rollout",
            )

    if not aborted:
        try:
            Path(adapter_root).rmdir()
        except OSError:
            pass
    gc.collect()
    final_memory_sample = query_visible_gpu_memory_mb()
    if final_memory_sample >= 0:
        peak_sampled_gpu_memory_mb = max(
            peak_sampled_gpu_memory_mb,
            final_memory_sample,
        )
    final_loads = service_load_snapshot(
        service_outstanding,
        service_load_lock,
    )
    if not aborted and not service_retired and final_loads[service_index] != 0:
        raise RuntimeError(
            f"{log_prefix} exited with {final_loads[service_index]} outstanding task(s)"
        )
    telemetry_summary = telemetry.summary()
    telemetry.close()
    forensics.close()
    print(
        f"{log_prefix} service {'aborted' if aborted else 'finished'}: "
        f"wall={time.time() - service_started:.3f}s, "
        f"startup={startup_seconds:.3f}s, restarts={engine_restart_count}/"
        f"{engine_restart_seconds:.3f}s, "
        f"engine={total_engine_seconds:.3f}s/"
        f"{total_generate_calls} calls, sampled_peak_gpu_memory="
        f"{peak_sampled_gpu_memory_mb}MB, outstanding={list(final_loads)}; "
        f"scheduler=requests={telemetry_summary['requests']}, "
        f"mean_batch={telemetry_summary['mean_batch_size']:.2f}, "
        f"max_batch={telemetry_summary['max_batch_size']}, "
        f"max_tasks={telemetry_summary['max_task_count']}, "
        f"max_adapters={telemetry_summary['max_adapter_count']}, "
        f"mixed_calls={telemetry_summary['mixed_calls']}, "
        f"cache={telemetry_summary['cache_fraction']:.3f}, "
        f"prepare={telemetry_summary['prepare_seconds']:.3f}s, "
        f"response={telemetry_summary['response_seconds']:.3f}s",
        flush=True,
    )
    finalize_engine("fatal pipeline abort" if aborted else "normal exit")
    signal.signal(signal.SIGTERM, previous_sigterm_handler)
