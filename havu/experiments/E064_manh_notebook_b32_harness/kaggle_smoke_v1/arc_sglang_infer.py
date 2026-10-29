"""Persistent independent TP=1 SGLang multi-LoRA services for ARC.

This backend deliberately reuses the existing CPU search/scoring state machine
from :mod:`arc_vllm_infer`.  SGLang is only a batched raw-logprob provider; it
does not own candidate selection or submission serialization.
"""

from __future__ import annotations

import gc
import json
import math
import os
import signal
import time
from collections import deque
from collections.abc import Mapping, Sequence
from multiprocessing.util import Finalize
from pathlib import Path
from queue import Empty
from typing import Any

from arc_loader import ArcDataset, QwenFormatter
from arc_vllm_adapter import remove_job_adapter_artifact
from arc_vllm_infer import (
    ARC_TOKENS,
    ArcVllmTask,
    _EngineWork,
    ensure_arc_tokenizer_backend,
    query_visible_gpu_memory_mb,
    release_pending_adapter_artifact,
)
from arc_vllm_router import release_service_task, service_load_snapshot
from run_config import resolve_challenge_path


def _finite_float(value: Any, *, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite SGLang logprob for {context}: {result}")
    return result


def _logprob_pair(entry: Any, *, context: str) -> tuple[float, int]:
    """Normalize SGLang's ``[logprob, token_id, optional_text]`` entry."""
    if isinstance(entry, Mapping):
        if "logprob" not in entry or "token_id" not in entry:
            raise ValueError(
                f"SGLang {context} mapping omits logprob/token_id: {entry!r}"
            )
        value = entry["logprob"]
        token_id = entry["token_id"]
    elif isinstance(entry, Sequence) and not isinstance(
        entry, (str, bytes, bytearray)
    ):
        if len(entry) < 2:
            raise ValueError(
                f"SGLang {context} entry is too short: {entry!r}"
            )
        value, token_id = entry[0], entry[1]
    else:
        raise TypeError(
            f"SGLang {context} entry has unsupported type "
            f"{type(entry).__name__}"
        )
    return _finite_float(value, context=context), int(token_id)


def normalize_sglang_batch(result: Any, expected: int) -> list[dict[str, Any]]:
    """Return one response dictionary per submitted offline-engine request."""
    if isinstance(result, Mapping):
        results = [dict(result)]
    elif isinstance(result, Sequence) and not isinstance(
        result, (str, bytes, bytearray)
    ):
        results = list(result)
    else:
        raise TypeError(
            "SGLang generate returned unsupported type "
            f"{type(result).__name__}"
        )
    if len(results) != int(expected):
        raise RuntimeError(
            f"SGLang returned {len(results)} responses for {expected} requests"
        )
    for index, item in enumerate(results):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"SGLang response {index} is {type(item).__name__}, not a mapping"
            )
    return [dict(item) for item in results]


def requested_arc_logprobs_from_sglang(
    response: Mapping[str, Any],
    *,
    position: int = -1,
) -> dict[int, float]:
    """Extract all 12 requested ARC token logprobs at one decode position."""
    try:
        positions = response["meta_info"]["output_token_ids_logprobs"]
        entries = positions[position]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"SGLang decode response omits position {position} ARC logprobs"
        ) from exc
    result: dict[int, float] = {}
    for entry in entries:
        value, token_id = _logprob_pair(entry, context="decode")
        result[token_id] = value
    missing = [token_id for token_id in ARC_TOKENS if token_id not in result]
    if missing:
        raise ValueError(
            f"SGLang decode response omits requested ARC token IDs: {missing}"
        )
    return {token_id: result[token_id] for token_id in ARC_TOKENS}


def rollout_from_sglang_response(
    response: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[dict[int, float], ...]]:
    """Extract a multi-token greedy rollout and every raw ARC distribution."""
    try:
        output_ids = tuple(map(int, response["output_ids"]))
        positions = response["meta_info"]["output_token_ids_logprobs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("SGLang rollout response omits IDs or logprobs") from exc
    if not output_ids:
        raise ValueError("SGLang returned an empty leaf rollout")
    if len(output_ids) != len(positions):
        raise ValueError(
            "SGLang rollout token/logprob length mismatch: "
            f"{len(output_ids)} != {len(positions)}"
        )
    return output_ids, tuple(
        requested_arc_logprobs_from_sglang(response, position=position)
        for position in range(len(output_ids))
    )


def cached_tokens_from_sglang(response: Mapping[str, Any]) -> int:
    """Return SGLang's radix-cache hit count for timing diagnostics."""
    try:
        cached_tokens = int(response.get("meta_info", {}).get("cached_tokens", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("SGLang returned an invalid cached_tokens value") from exc
    if cached_tokens < 0:
        raise ValueError("SGLang cached_tokens must be non-negative")
    return cached_tokens


def prompt_logprobs_from_sglang(
    response: Mapping[str, Any],
    *,
    prompt_length: int,
    query_length: int,
    answer_token_ids: Sequence[int],
) -> list[dict[int, float] | None]:
    """Build the logical full-prompt container expected by the ARC scorer.

    SGLang is asked to begin at ``query_length - 1`` because the first returned
    input position is an undefined sentinel and the following position is the
    first answer token. The shared scorer needs a full-length positional
    container, so the unrequested query prefix is represented by ``None``
    entries.
    """
    answer_token_ids = tuple(map(int, answer_token_ids))
    if prompt_length != query_length + len(answer_token_ids):
        raise ValueError(
            "Prompt length does not equal query plus answer: "
            f"{prompt_length} != {query_length} + {len(answer_token_ids)}"
        )
    try:
        entries = list(response["meta_info"]["input_token_logprobs"])
    except (KeyError, TypeError) as exc:
        raise ValueError("SGLang score response has no input logprobs") from exc

    # Some SGLang revisions retain a leading undefined sentinel even when
    # logprob_start_len is nonzero.  Accept only that documented sentinel; do
    # not silently trim a finite score and shift every target position.
    if len(entries) == len(answer_token_ids) + 1:
        sentinel = entries[0]
        if sentinel is None or (
            isinstance(sentinel, Sequence)
            and not isinstance(sentinel, (str, bytes, bytearray))
            and sentinel
            and sentinel[0] is None
        ):
            entries = entries[1:]
    if len(entries) != len(answer_token_ids):
        raise ValueError(
            "SGLang returned the wrong answer-logprob count: "
            f"{len(entries)} != {len(answer_token_ids)}"
        )

    logical: list[dict[int, float] | None] = [None] * query_length
    for offset, (entry, expected_token_id) in enumerate(
        zip(entries, answer_token_ids)
    ):
        value, token_id = _logprob_pair(entry, context=f"score position {offset}")
        # SGLang 0.5.17's prompt-logprob serializer maps every input token ID
        # equal to ``vocab_size - 1`` to zero to sanitize multimodal hash
        # placeholders.  ARC's custom vocabulary uses that final row (15) as
        # EOS, so only this exact reported-ID mismatch is benign; the value is
        # still the positionally aligned EOS logprob.
        clipped_arc_eos = expected_token_id == ARC_TOKENS[-1] and token_id == 0
        if token_id != expected_token_id and not clipped_arc_eos:
            raise ValueError(
                "SGLang score token alignment mismatch at answer offset "
                f"{offset}: returned {token_id}, expected {expected_token_id}"
            )
        logical.append({expected_token_id: value})
    return logical


def sglang_engine_kwargs(
    model_path: str,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build conservative TP=1 SGLang engine arguments."""
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        # Every request already contains the exact custom 16-token IDs. Avoid
        # a second Transformers tokenizer load inside SGLang, which can change
        # control-token handling across Transformers 4.x/5.x releases.
        "skip_tokenizer_init": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tp_size": 1,
        "context_length": int(options["max_seq_length"]),
        "enable_lora": True,
        "max_lora_rank": int(options["lora_r"]),
        # Runtime loading without an initial adapter requires an explicit
        # target union. "all" is resolved against modules present in Qwen3 and
        # includes the separately trained embeddings and LM head.
        "lora_target_modules": ["all"],
        "max_loaded_loras": int(options["sglang_max_loaded_loras"]),
        "max_loras_per_batch": int(options["sglang_max_loras_per_batch"]),
        "lora_backend": str(options["sglang_lora_backend"]),
        "enable_lora_overlap_loading": bool(
            options.get("sglang_overlap_loading", False)
        ),
        "max_running_requests": int(options["sglang_max_running_requests"]),
        "log_level": "error",
    }
    mem_fraction = float(options.get("sglang_mem_fraction_static", 0.0) or 0.0)
    if mem_fraction > 0.0:
        kwargs["mem_fraction_static"] = mem_fraction
    chunked_prefill = int(
        options.get("sglang_chunked_prefill_size", 0) or 0
    )
    if chunked_prefill > 0:
        kwargs["chunked_prefill_size"] = chunked_prefill
    # Keep the kwarg absent by default so older SGLang builds retain their
    # existing behavior. SGLang 0.5.17 resolves this to its batch-invariant
    # attention and sampling paths.
    if options.get("sglang_deterministic_inference", False):
        kwargs["enable_deterministic_inference"] = True
    return kwargs


def normalize_rslora_config_for_sglang(
    adapter_path: str | os.PathLike[str],
) -> tuple[bool, float, float]:
    """Rewrite rsLoRA metadata to an equivalent ordinary LoRA scale.

    SGLang 0.5.17 computes every adapter scale as ``lora_alpha / r`` and does
    not inspect PEFT's ``use_rslora`` flag. ARC adapters use rsLoRA, whose
    scale is ``lora_alpha / sqrt(r)``. Multiplying alpha by ``sqrt(r)`` and
    clearing the flag preserves the exact effective scale without touching
    any tensor. The handoff directory is private to this SGLang service and
    is deleted after a successful load, so vLLM artifacts remain unchanged.
    """
    config_path = Path(adapter_path) / "adapter_config.json"
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read SGLang adapter config {config_path}"
        ) from exc

    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(
            "SGLang ARC inference does not support PEFT rank_pattern or "
            "alpha_pattern adapters"
        )

    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"Invalid adapter scale metadata: r={rank}, lora_alpha={alpha}"
        )
    original_scale = alpha / (
        math.sqrt(rank) if config.get("use_rslora") else rank
    )
    if not config.get("use_rslora"):
        return False, original_scale, original_scale

    normalized_alpha = alpha * math.sqrt(rank)
    normalized_scale = normalized_alpha / rank
    if not math.isclose(
        original_scale,
        normalized_scale,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            "Internal rsLoRA normalization changed the effective scale: "
            f"{original_scale} != {normalized_scale}"
        )
    config["lora_alpha"] = normalized_alpha
    config["use_rslora"] = False

    temporary_path = config_path.with_name(
        f".{config_path.name}.sglang-{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("x") as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)
            config_file.write("\n")
        os.replace(temporary_path, config_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return True, original_scale, normalized_scale


def _engine_result_success(result: Any, *, action: str) -> None:
    """Raise when a SGLang adapter lifecycle RPC reports failure."""
    if isinstance(result, bool):
        success = result
        error = None
    elif isinstance(result, Mapping):
        success = result.get("success")
        error = result.get("error_message") or result.get("message")
    else:
        success = getattr(result, "success", None)
        error = getattr(result, "error_message", None)
    if success is False:
        raise RuntimeError(f"SGLang failed to {action}: {error or result!r}")


def _load_sglang_engine(model_path: str, options: Mapping[str, Any]) -> Any:
    """Import SGLang only inside a reserved inference process."""
    import sglang as sgl

    engine = sgl.Engine(**sglang_engine_kwargs(model_path, options))
    for method_name in (
        "generate",
        "load_lora_adapter",
        "unload_lora_adapter",
        "shutdown",
    ):
        if not callable(getattr(engine, method_name, None)):
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise RuntimeError(
                "Installed SGLang lacks required offline-engine method "
                f"{method_name}; install a build matching the local sglang "
                "checkout"
            )
    return engine


def shutdown_sglang_engine(engine: Any) -> bool:
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        return False
    shutdown()
    return True


def _decode_sampling_params() -> dict[str, Any]:
    # The sampled token is not consumed by exact DFS. Avoid a sampling mask so
    # returned ARC logprobs retain full-vocabulary normalization.
    return {
        "temperature": 0.0,
        "max_new_tokens": 1,
        "ignore_eos": True,
        "skip_special_tokens": False,
    }


def _score_sampling_params() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_new_tokens": 0,
        "skip_special_tokens": False,
    }


def _rollout_sampling_params(max_new_tokens: int) -> dict[str, Any]:
    """Build an unmasked greedy rollout with full-vocabulary logprobs.

    No grammar or logit bias is applied. If the model samples a control token
    outside ``ARC_TOKENS``, the shared rollout state machine stops consuming
    that chunk and resumes the best valid ARC child in a new request.
    """
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 1:
        raise ValueError("SGLang rollout max_new_tokens must be positive")
    return {
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": False,
        "stop_token_ids": [ARC_TOKENS[-1]],
        "no_stop_trim": True,
        "skip_special_tokens": False,
    }


def run_sglang_inference_worker(
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
) -> None:
    """Run one persistent SGLang service until every trainer has finished."""
    del rollout_response_queues  # SGLang GRPO rollout is intentionally absent.

    from transformers import AutoTokenizer

    service_index = int(service_index)
    log_prefix = f"[SGLang {service_index}]"
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def startup_sigterm(signum, _frame) -> None:
        print(
            f"{log_prefix} received signal {signum} during engine startup",
            flush=True,
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, startup_sigterm)
    startup_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = ensure_arc_tokenizer_backend(tokenizer, model_path)
    formatter = QwenFormatter(tokenizer=tokenizer)
    lock_started = time.perf_counter()
    with startup_lock:
        lock_wait_seconds = time.perf_counter() - lock_started
        engine_started = time.perf_counter()
        engine = _load_sglang_engine(model_path, options)
        engine_load_seconds = time.perf_counter() - engine_started

    engine_finalizer = Finalize(
        None,
        shutdown_sglang_engine,
        args=(engine,),
        exitpriority=20,
    )

    def finalize_engine(reason: str) -> None:
        if not engine_finalizer.still_active():
            return
        print(f"{log_prefix} shutting down engine ({reason})", flush=True)
        try:
            found = engine_finalizer()
            if found is False:
                print(
                    f"{log_prefix} warning: engine exposes no shutdown method",
                    flush=True,
                )
        except BaseException as error:
            print(
                f"{log_prefix} warning: shutdown failed: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    def handle_sigterm(signum, _frame) -> None:
        finalize_engine(f"signal {signum}")
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, handle_sigterm)
    startup_seconds = time.perf_counter() - startup_started
    print(
        f"{log_prefix} engine ready in {startup_seconds:.3f}s "
        f"(startup-lock wait={lock_wait_seconds:.3f}s, "
        f"engine load={engine_load_seconds:.3f}s); "
        f"loaded={options['sglang_max_loaded_loras']}, "
        f"per_batch={options['sglang_max_loras_per_batch']}",
        flush=True,
    )

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    arc_test_set = ArcDataset.from_file(resolve_challenge_path(rerun_mode))
    output_dir = os.getenv("ARC_INFERENCE_OUTPUT_DIR", "/kaggle/inference_outputs")
    os.makedirs(output_dir, exist_ok=True)

    task_options = dict(options)
    task_options.update(
        engine_label="SGLang",
        # This is the SGLang implementation of the same behavioral contract:
        # request batching must not change token probabilities. It is not the
        # vLLM environment-variable implementation.
        vllm_batch_invariant=bool(
            options.get("sglang_deterministic_inference", False)
        ),
        vllm_leaf_rollout=bool(options.get("sglang_leaf_rollout", False)),
        vllm_rollout_max_tokens=int(
            options.get("sglang_rollout_max_tokens", 32)
        ),
        # ``auto`` otherwise sends short one-output tasks to frontier search,
        # where whole-leaf rollout cannot reduce engine calls.  This override
        # is scoped to the explicit SGLang leaf flag; exact mode and both
        # explicit DFS choices remain unchanged.
        leaf_rollout_prefer_recursive=bool(
            options.get("sglang_leaf_rollout", False)
        ),
        # SGLang returns the requested answer positions even when its radix
        # cache reuses the query prefix; vLLM's omitted-prefix reconstruction
        # is therefore neither needed nor valid.
        score_returns_complete_prompt=True,
    )

    active: list[ArcVllmTask] = []
    pending_jobs: deque[dict[str, Any]] = deque()
    trainers_done = 0
    service_started = time.time()
    total_generate_calls = 0
    total_engine_seconds = 0.0
    peak_sampled_gpu_memory_mb = query_visible_gpu_memory_mb()
    round_robin_offset = 0
    deadline_mode = False
    aborted = False

    def receive(message: dict[str, Any]) -> None:
        nonlocal trainers_done
        kind = message.get("kind")
        if kind == "adapter":
            target_service = int(message.get("service_index", service_index))
            if target_service != service_index:
                raise RuntimeError(
                    f"{log_prefix} received adapter for service {target_service}"
                )
            pending_jobs.append(message)
        elif kind == "trainer_done":
            trainers_done += 1
            print(
                f"{log_prefix} trainer {message.get('rank')} finished "
                f"({trainers_done}/{trainer_count})",
                flush=True,
            )
        elif kind == "rl_rollout":
            raise RuntimeError(
                "SGLang final inference does not implement GRPO rollout; use "
                "--rl-rollout-backend native"
            )
        else:
            raise RuntimeError(f"Unknown SGLang pipeline message: {message}")

    def drain_nonblocking() -> None:
        while True:
            try:
                receive(job_queue.get_nowait())
            except Empty:
                break

    def discard_job(job: dict[str, Any]) -> None:
        release_pending_adapter_artifact(
            job,
            adapter_root,
            pending_adapter_slots,
            adapter_staging_slots,
        )
        loads = release_service_task(
            service_index,
            service_outstanding,
            service_load_lock,
        )
        print(
            f"{log_prefix} discarded adapter {job['adapter_name']} at "
            f"deadline; outstanding={list(loads)}",
            flush=True,
        )

    def admit_jobs() -> None:
        nonlocal peak_sampled_gpu_memory_mb
        active_limit = int(options["sglang_max_loaded_loras"])
        while pending_jobs and len(active) < active_limit and not deadline_mode:
            job = pending_jobs.popleft()
            normalized_rslora, original_scale, normalized_scale = (
                normalize_rslora_config_for_sglang(job["adapter_path"])
            )
            if normalized_rslora:
                print(
                    f"{log_prefix} normalized rsLoRA metadata for "
                    f"{job['adapter_name']}: effective_scale="
                    f"{original_scale:.8g} (SGLang alpha/r="
                    f"{normalized_scale:.8g})",
                    flush=True,
                )
            load_started = time.perf_counter()
            result = engine.load_lora_adapter(
                str(job["adapter_name"]),
                str(job["adapter_path"]),
            )
            _engine_result_success(
                result,
                action=f"load adapter {job['adapter_name']}",
            )
            load_seconds = time.perf_counter() - load_started
            task = ArcVllmTask(
                job=job,
                lora_request=str(job["adapter_name"]),
                arc_test_set=arc_test_set,
                tokenizer=tokenizer,
                formatter=formatter,
                output_dir=output_dir,
                global_end_time=end_time,
                options=task_options,
            )
            task.adapter_load_seconds = load_seconds
            # Loading is synchronous, so the bounded queue slot can be reused.
            # Keep the artifact itself until this task finishes and SGLang
            # unloads the resident adapter.
            pending_adapter_slots.release()
            active.append(task)
            queue_seconds = time.time() - float(job["published_at"])
            print(
                f"{log_prefix} loaded and accepted {job['adapter_name']} in "
                f"{load_seconds:.3f}s after {queue_seconds:.3f}s queued; "
                "staging artifact retained until task inference completes; "
                f"active={[item.job['adapter_name'] for item in active]}",
                flush=True,
            )
            sample = query_visible_gpu_memory_mb()
            if sample >= 0:
                peak_sampled_gpu_memory_mb = max(
                    peak_sampled_gpu_memory_mb,
                    sample,
                )

    def finish_tasks() -> None:
        for task in list(active):
            if not task.done:
                continue
            unload_started = time.perf_counter()
            result = engine.unload_lora_adapter(str(task.job["adapter_name"]))
            _engine_result_success(
                result,
                action=f"unload adapter {task.job['adapter_name']}",
            )
            unload_seconds = time.perf_counter() - unload_started
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
            print(
                f"{log_prefix} unloaded {task.job['adapter_name']} in "
                f"{unload_seconds:.3f}s; RAM/disk artifact removed; "
                f"outstanding={list(loads)}",
                flush=True,
            )

    def run_exact_decode_fallback(
        work: _EngineWork,
        chunk_error: BaseException,
    ) -> tuple[tuple[int, ...], tuple[dict[int, float], ...]]:
        """Replay one malformed leaf chunk as an exact one-token request."""
        nonlocal total_generate_calls
        nonlocal total_engine_seconds

        fallback_started = time.perf_counter()
        raw_output = engine.generate(
            input_ids=[list(work.prompt_token_ids)],
            sampling_params=[_decode_sampling_params()],
            return_logprob=[True],
            logprob_start_len=[-1],
            top_logprobs_num=[0],
            token_ids_logprob=[list(ARC_TOKENS)],
            lora_path=[str(work.task.lora_request)],
        )
        elapsed = time.perf_counter() - fallback_started
        output = normalize_sglang_batch(raw_output, 1)[0]
        try:
            sampled_token_ids, position_logprobs = (
                rollout_from_sglang_response(output)
            )
            if len(sampled_token_ids) != 1:
                raise ValueError(
                    "SGLang exact fallback returned "
                    f"{len(sampled_token_ids)} tokens"
                )
        except (KeyError, TypeError, ValueError) as fallback_error:
            raise RuntimeError(
                "SGLang leaf rollout and exact one-token replay both "
                "returned invalid ARC logprobs: "
                f"task={work.task.key}, request_id={work.request_id}, "
                f"prompt_tokens={len(work.prompt_token_ids)}"
            ) from fallback_error

        cap_rollout = getattr(work.task.search, "cap_rollout_chunk_tokens", None)
        if not callable(cap_rollout):
            raise RuntimeError(
                "Cannot downgrade malformed SGLang rollout for search type "
                f"{type(work.task.search).__name__}"
            ) from chunk_error
        cap_rollout(1)
        work.task.decode_exact_fallbacks += 1
        cached_tokens = cached_tokens_from_sglang(output)
        work.task.record_engine_call(
            "decode",
            elapsed,
            len(work.prompt_token_ids),
            cached_tokens,
        )
        total_generate_calls += 1
        total_engine_seconds += elapsed
        print(
            f"{work.task.log_prefix} warning: downgraded puzzle "
            f"{work.task.key} to exact one-token DFS after an invalid "
            f"SGLang leaf response at request {work.request_id}; "
            f"fallback={elapsed:.3f}s, reason={chunk_error}",
            flush=True,
        )
        return sampled_token_ids, position_logprobs

    try:
        while True:
            if abort_event is not None and abort_event.is_set():
                aborted = True
                print(
                    f"{log_prefix} fatal trainer/service failure received; "
                    "abort inference",
                    flush=True,
                )
                break
            drain_nonblocking()
            if time.time() >= end_time:
                deadline_mode = True
            if deadline_mode:
                while pending_jobs:
                    discard_job(pending_jobs.popleft())
                for task in active:
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
                finish_tasks()
                if trainers_done >= trainer_count:
                    break
                try:
                    receive(job_queue.get(timeout=0.5))
                except Empty:
                    continue
                continue

            admit_jobs()
            for task in active:
                task.settle()
            finish_tasks()
            admit_jobs()

            if not active:
                if trainers_done >= trainer_count and not pending_jobs:
                    break
                try:
                    receive(job_queue.get(timeout=0.5))
                except Empty:
                    continue
                continue

            max_requests = int(options["sglang_max_running_requests"])
            works: list[_EngineWork] = []
            ordered_tasks = (
                active[round_robin_offset:] + active[:round_robin_offset]
            )
            fair_quota = max(1, math.ceil(max_requests / len(ordered_tasks)))
            for task in ordered_tasks:
                if len(works) >= max_requests:
                    break
                works.extend(
                    task.take_work(min(fair_quota, max_requests - len(works)))
                )
            round_robin_offset = (round_robin_offset + 1) % max(1, len(active))
            if not works:
                for task in active:
                    task.settle()
                finish_tasks()
                if active:
                    time.sleep(0.01)
                continue

            input_ids = [list(work.prompt_token_ids) for work in works]
            sampling_params = []
            for work in works:
                if work.kind == "score":
                    sampling_params.append(_score_sampling_params())
                elif work.constrained_rollout:
                    sampling_params.append(
                        _rollout_sampling_params(work.max_tokens)
                    )
                else:
                    sampling_params.append(_decode_sampling_params())
            logprob_start_len = [
                max(0, int(work.score_job.query_length) - 1)
                if work.kind == "score" and work.score_job is not None
                else -1
                for work in works
            ]
            generate_started = time.perf_counter()
            raw_outputs = engine.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=[True] * len(works),
                logprob_start_len=logprob_start_len,
                top_logprobs_num=[0] * len(works),
                token_ids_logprob=[list(ARC_TOKENS) for _ in works],
                lora_path=[str(work.task.lora_request) for work in works],
            )
            elapsed = time.perf_counter() - generate_started
            outputs = normalize_sglang_batch(raw_outputs, len(works))
            total_generate_calls += 1
            total_engine_seconds += elapsed

            participants: dict[
                tuple[int, str],
                tuple[ArcVllmTask, int, int],
            ] = {}
            for work, output in zip(works, outputs):
                key = (id(work.task), work.kind)
                task, prompt_tokens, cached_tokens = participants.get(
                    key,
                    (work.task, 0, 0),
                )
                participants[key] = (
                    task,
                    prompt_tokens + len(work.prompt_token_ids),
                    cached_tokens + cached_tokens_from_sglang(output),
                )
            for (
                _task_id,
                kind,
            ), (task, prompt_tokens, cached_tokens) in participants.items():
                task.record_engine_call(
                    kind,
                    elapsed,
                    prompt_tokens,
                    cached_tokens,
                )

            for work, output in zip(works, outputs):
                try:
                    if work.kind == "decode":
                        if work.constrained_rollout:
                            try:
                                (
                                    sampled_token_ids,
                                    position_logprobs,
                                ) = rollout_from_sglang_response(output)
                            except (KeyError, TypeError, ValueError) as error:
                                (
                                    sampled_token_ids,
                                    position_logprobs,
                                ) = run_exact_decode_fallback(work, error)
                            work.task.accept_decode(
                                work.request_id,
                                position_logprobs,
                                sampled_token_ids=sampled_token_ids,
                            )
                        else:
                            token_logprobs = requested_arc_logprobs_from_sglang(
                                output,
                                position=-1,
                            )
                            work.task.accept_decode(
                                work.request_id,
                                token_logprobs,
                            )
                    else:
                        if work.score_job is None:
                            raise RuntimeError("Score work has no score job")
                        prompt_logprobs = prompt_logprobs_from_sglang(
                            output,
                            prompt_length=len(work.prompt_token_ids),
                            query_length=work.score_job.query_length,
                            answer_token_ids=work.score_job.answer_token_ids,
                        )
                        work.task.accept_score(
                            work.request_id,
                            prompt_logprobs,
                            cached_tokens=cached_tokens_from_sglang(output),
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "SGLang returned invalid ARC logprobs: "
                        f"task={work.task.key}, "
                        f"adapter={work.task.job['adapter_name']}, "
                        f"request_id={work.request_id}, kind={work.kind}, "
                        f"prompt_tokens={len(work.prompt_token_ids)}"
                    ) from exc
    except BaseException:
        aborted = True
        if abort_event is not None:
            abort_event.set()
        raise
    finally:
        if not aborted:
            try:
                Path(adapter_root).rmdir()
            except OSError:
                pass
        gc.collect()
        final_sample = query_visible_gpu_memory_mb()
        if final_sample >= 0:
            peak_sampled_gpu_memory_mb = max(
                peak_sampled_gpu_memory_mb,
                final_sample,
            )
        final_loads = service_load_snapshot(
            service_outstanding,
            service_load_lock,
        )
        print(
            f"{log_prefix} service {'aborted' if aborted else 'finished'}: "
            f"wall={time.time() - service_started:.3f}s, "
            f"startup={startup_seconds:.3f}s, "
            f"engine={total_engine_seconds:.3f}s/{total_generate_calls} calls, "
            f"sampled_peak_gpu_memory={peak_sampled_gpu_memory_mb}MB, "
            f"outstanding={list(final_loads)}",
            flush=True,
        )
        finalize_engine("pipeline abort" if aborted else "normal exit")
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    if not aborted and final_loads[service_index] != 0:
        raise RuntimeError(
            f"{log_prefix} exited with {final_loads[service_index]} "
            "outstanding task(s)"
        )
