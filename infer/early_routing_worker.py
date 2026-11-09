"""Disposable frozen-base vLLM probe worker; launched by early_routing.py."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import time

from early_routing import RECIPE, digest, encode_job, probe_jobs, score_cells, validate_probe_model, visible_task
from pipeline_runtime import atomic_json, model_manifest


def _probe_results(engine, pending, params, *, batch_size, metrics):
    """Keep vLLM supplied across task boundaries; yield each completed request."""
    core = getattr(engine, "llm_engine", None)
    if (callable(getattr(engine, "enqueue", None)) and callable(getattr(core, "step", None))
            and callable(getattr(core, "has_unfinished_requests", None))):
        metrics["dispatch"] = "continuous-cross-task"
        requests = [{"prompt_token_ids": q + a} for _, _, q, a in pending]
        request_ids = engine.enqueue(requests, params, use_tqdm=False)
        if len(request_ids) != len(pending) or len(set(request_ids)) != len(pending):
            raise RuntimeError("Invalid early-routing enqueue request IDs")
        # Newer vLLM enqueue() returns randomized internal IDs, but step()
        # emits external_req_id. Snapshot the mapping before step() removes
        # completed RequestStates. Never guess IDs by stripping UUID suffixes
        # or assuming the request counter starts at zero.
        states = getattr(getattr(core, "output_processor", None), "request_states", {})
        output_ids = [getattr(states.get(request_id), "external_req_id", request_id)
                      for request_id in request_ids]
        if any(not isinstance(request_id, str) or not request_id for request_id in output_ids):
            raise RuntimeError("Invalid early-routing external request IDs")
        if len(set(output_ids)) != len(output_ids):
            raise RuntimeError("Duplicate early-routing external request IDs")
        metrics["remapped_request_ids"] = sum(a != b for a, b in zip(request_ids, output_ids))
        waiting = dict(zip(output_ids, pending))
        metrics["submitted_requests"] += len(pending)
        # max_num_seqs caps active GPU requests, not this CPU-side backlog.
        # Using step() exposes finished requests without draining the queue
        # after every task/batch, and permits durable progress before timeout.
        while core.has_unfinished_requests():
            results = core.step()
            metrics["engine_steps"] += 1
            for result in results:
                if not result.finished:
                    continue
                request = waiting.pop(result.request_id, None)
                if request is None:
                    raise RuntimeError(f"Unknown/duplicate early-routing request ID: {result.request_id!r}; "
                                       f"expected output IDs (sample): {list(waiting)[:3]!r}")
                metrics["completed_requests"] += 1
                yield request, result
        if waiting:
            raise RuntimeError(f"Early-routing engine stopped with {len(waiting)} unfinished requests")
    else:
        # Compatibility with older LLM wrappers lacking enqueue(). They still
        # get full cross-task batches, not one generate() call per task.
        metrics["dispatch"] = "cross-task-batches"
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            metrics["generate_calls"] += 1
            metrics["submitted_requests"] += len(batch)
            results = engine.generate([{"prompt_token_ids": q + a} for _, _, q, a in batch],
                                      params, use_tqdm=False)
            if len(results) != len(batch):
                raise RuntimeError("Incomplete early-routing probe batch")
            for request, result in zip(batch, results):
                metrics["completed_requests"] += 1
                yield request, result


def probe_tasks(tasks, tokenizer, get_engine, params, *, context, batch_size, on_report=None, metrics=None):
    """Batch all valid tasks together; only publish scores after all pair/views."""
    from arc_vllm_search import logprobs_at_position

    if batch_size < 1:
        raise ValueError("Probe batch size must be positive")
    started = time.monotonic()
    reports, pending, expected = {}, [], {}
    if metrics is None:
        metrics = {}
    metrics.update(dispatch="no-requests", submitted_requests=0, completed_requests=0,
                   engine_steps=0, generate_calls=0, prompt_tokens=0)

    def publish(key):
        reports[key]["seconds"] = time.monotonic() - started
        if on_report is not None:
            on_report(key, reports[key])

    for key, task in tasks.items():
        report = reports[key] = dict(visible_task_sha256=digest(visible_task(task)), rows=[], status="pending")
        try:
            jobs = probe_jobs(task)
            encoded = [(key, job, *encode_job(job, tokenizer)) for job in jobs]
            if not jobs:
                report["status"] = "too-few-train-pairs"
            elif any(len(q) + len(a) + 1 > context for _, _, q, a in encoded):
                report["status"] = "context-limit"  # No truncation or partial mean.
            else:
                expected[key] = len(encoded)
                pending.extend(encoded)
        except ValueError as error:
            report.update(status="invalid-probe", error=str(error))
        if report["status"] != "pending":
            publish(key)

    metrics["prepared_requests"] = len(pending)
    metrics["prompt_tokens"] = sum(len(q) + len(a) for _, _, q, a in pending)
    if not pending:
        return reports
    engine = get_engine()
    scoring_started = time.monotonic()
    for (key, job, q, a), result in _probe_results(engine, pending, params, batch_size=batch_size, metrics=metrics):
        report = reports[key]
        if report["status"] != "pending":
            continue  # Another view already invalidated this task, not its neighbours.
        try:
            if getattr(result, "num_cached_tokens", 0) or list(result.prompt_token_ids) != q + a:
                raise ValueError("Unexpected cache hit or mismatched probe prompt")
            if result.prompt_logprobs is None or len(result.prompt_logprobs) != len(q) + len(a):
                raise ValueError("Incomplete/misaligned probe prompt logprobs")
            scores = score_cells(a, [logprobs_at_position(result.prompt_logprobs, len(q) + i)
                                    for i in range(len(a))], job["output_grid"])
            report["rows"].append(dict(held_out=job["held_out"], view=job["view"],
                                       query_tokens=len(q), answer_tokens=len(a), scores=scores))
        except ValueError as error:
            report.update(status="invalid-probe", error=str(error))
            publish(key)
            continue
        if len(report["rows"]) == expected[key]:
            report["rows"].sort(key=lambda row: (row["held_out"], row["view"]))
            # Equal weight per pair/view, not pooled cells or mixed-task means.
            report.update(status="complete", scores={
                name: statistics.fmean(row["scores"][name] for row in report["rows"])
                for name in ("nll", "entropy")})
            publish(key)
    metrics["scoring_seconds"] = time.monotonic() - scoring_started
    metrics["prompt_tokens_per_second"] = metrics["prompt_tokens"] / max(metrics["scoring_seconds"], 1e-9)
    metrics["seconds_including_startup"] = time.monotonic() - started
    return reports


def probe_task(task, tokenizer, get_engine, params, *, context, batch_size):
    """Single-task compatibility helper; production uses probe_tasks()."""
    report = probe_tasks({"task": task}, tokenizer, get_engine, params,
                         context=context, batch_size=batch_size)["task"]
    if report["status"] == "invalid-probe":
        raise ValueError(report["error"])
    return report


def base_engine_kwargs(model_path, *, context, batch_size, quantization,
                       gpu_memory_utilization=.89, max_num_batched_tokens=16384):
    config = validate_probe_model(model_path)
    kwargs = dict(model=model_path, tokenizer=model_path, skip_tokenizer_init=True,
                  dtype="bfloat16", quantization=None if quantization == "none" else quantization,
                  trust_remote_code=True, max_model_len=context, tensor_parallel_size=1,
                  enable_prefix_caching=False, enable_lora=False, gpu_memory_utilization=gpu_memory_utilization,
                  max_num_seqs=batch_size, max_num_batched_tokens=max_num_batched_tokens,
                  enable_chunked_prefill=True,
                  enforce_eager=True, async_scheduling=False, disable_log_stats=True, seed=42)
    if config.get("unsloth_delta_mhar") or (Path(model_path) / "delta_mhar_config.json").is_file():
        from arc_vllm_delta_mhar import PLUGIN_NAME, QWEN3_ARCHITECTURE, register
        from arc_vllm_delta_mhar.config import read_delta_mhar_spec

        plugins = [p.strip() for p in os.environ.get("VLLM_PLUGINS", "").split(",") if p.strip()]
        if plugins and PLUGIN_NAME not in plugins:
            raise ValueError(f"Early-routing base requires VLLM_PLUGINS={PLUGIN_NAME}")
        # Also register in EngineCore spawned children through its entry point.
        os.environ.setdefault("VLLM_PLUGINS", PLUGIN_NAME)
        spec = read_delta_mhar_spec(model_path)
        if spec.family != "qwen3":
            raise ValueError("Early routing only supports Qwen3-4B")
        register()
        kwargs["hf_overrides"] = {"architectures": [QWEN3_ARCHITECTURE]}
        kwargs["worker_cls"] = "arc_vllm_delta_mhar.worker.ArcDeltaMHARWorker"
    from vllm.config.kernel import KernelConfig

    if "enable_cutedsl_warmup" in inspect.signature(KernelConfig).parameters:
        kwargs["kernel_config"] = KernelConfig(enable_cutedsl_warmup=False)
    return kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.89)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--quantization", choices=("none", "fp8"), default="none")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or args.batch_size < 1:
        parser.error("Invalid shard/batch-size")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.max_num_batched_tokens < args.batch_size:
        parser.error("--max-num-batched-tokens must be at least --batch-size")
    input_data = json.loads(args.input.read_text())
    tasks = {k: visible_task(t) for k, t in input_data["tasks"].items()}
    keys = input_data["task_keys"][args.shard::args.shards]
    payload = dict(recipe=RECIPE, tasks={}, status="running", task_keys=keys, model=model_manifest(args.model_path),
                   quantization=args.quantization, adapter=None, prefix_cache=False, eager=True,
                   max_model_len=args.max_model_len, batch_size=args.batch_size, startup_seconds=None,
                   gpu_memory_utilization=args.gpu_memory_utilization,
                   max_num_batched_tokens=args.max_num_batched_tokens, workload={})
    atomic_json(args.report, payload)
    try:
        from runtime_bootstrap import bootstrap_transformer_engine_cpp_runtime, patch_vllm_compact_vocab_sampler_warmup

        bootstrap_transformer_engine_cpp_runtime()
        patch_vllm_compact_vocab_sampler_warmup()
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from runtime_bootstrap import install_vllm_token_only_renderer_warmup
        from arc_vllm_infer import ensure_arc_tokenizer_backend

        install_vllm_token_only_renderer_warmup()
        torch.set_num_threads(1)
        payload["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")}
        tokenizer = ensure_arc_tokenizer_backend(AutoTokenizer.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=True), args.model_path)
        sampling_kwargs = dict(temperature=0, max_tokens=1, ignore_eos=True, detokenize=False,
                               prompt_logprobs=-1, output_kind=RequestOutputKind.FINAL_ONLY)
        if "flat_logprobs" in inspect.signature(SamplingParams).parameters:
            sampling_kwargs["flat_logprobs"] = True
        params = SamplingParams(**sampling_kwargs)
        engine = None

        def get_engine():
            nonlocal engine
            if engine is None:
                started = time.monotonic()
                try:
                    engine = LLM(**base_engine_kwargs(args.model_path, context=args.max_model_len,
                                                     batch_size=args.batch_size, quantization=args.quantization,
                                                     gpu_memory_utilization=args.gpu_memory_utilization,
                                                     max_num_batched_tokens=args.max_num_batched_tokens))
                except Exception as error:
                    # Startup ValueErrors are fatal to this worker, not a bad
                    # task score. Do not repeatedly allocate partially started
                    # engines for every remaining task in the shard.
                    raise RuntimeError(f"Base probe engine startup failed: {error}") from error
                payload["startup_seconds"] = time.monotonic() - started
            return engine

        def save_report(key, row):
            payload["tasks"][key] = row
            atomic_json(args.report, payload)
            print(f"[early routing] {key}: {row['status']}; scores={row.get('scores')}", flush=True)

        probe_tasks({key: tasks[key] for key in keys}, tokenizer, get_engine, params,
                    context=args.max_model_len, batch_size=args.batch_size,
                    on_report=save_report, metrics=payload["workload"])
        payload["status"] = "complete"
        atomic_json(args.report, payload)
        print(f"[early routing] shard {args.shard} workload: {payload['workload']}", flush=True)
    except Exception as error:
        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(args.report, payload)
        raise


if __name__ == "__main__":
    main()
