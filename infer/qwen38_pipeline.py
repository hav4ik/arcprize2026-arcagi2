"""Deadline-safe, label-blind Qwen3.8, Qwen3.5-9B, DeepSeek-V4-Flash or dots3-note ARC
submission writer.

Standalone by default; an optional primary submission/routing manifest turns
this into the second stage of the 4B cascade. No experiments-directory imports.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time
import uuid

import httpx

import dots3_llama
import dsv41_havu
import dsv41_repl
import dsv41_vllm
import dsv4_llama
import qwen38_llama
import qwen38_repl_vision
import qwen38_server
import qwen38_live
import qwen38_loop
import qwen38_repl
import qwen38_repl_cot
import qwen38_recipe
import qwen38_tir
import qwen38_vision
from qwen38_server import stop_owned_process
from qwen38_recipe import RecipeParser as ProfileParser
from run_config import resolve_challenge_path, resolve_evaluation_solution_path, select_task_keys
from run_status import (STATUS_SUCCESS, extract_run_status_path, run_entrypoint_with_status)


ATTEMPTS = ("attempt_1", "attempt_2")
FINAL_PATH = None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with temp.open("x", encoding="utf-8") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def cleanup_completed_stream(output, result, *, keep=False):
    """Drop only a finished direct request's raw log AFTER its results are committed.

    Interrupted/error streams remain available for diagnostics and llama.cpp
    resume. Completion is independent of whether the generated grid was valid.
    Do not walk nested TIR rounds or previously completed/resumed directories.
    """
    if keep or result.get("error") is not None or result.get("cancelled"):
        return
    path = Path(output) / "stream.jsonl"
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        # A logging cleanup failure must not invalidate a committed answer.
        print(f"WARNING {path.parent.name}: could not remove completed stream {path}: {exc}", flush=True)


def grid_valid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
            and len({len(row) for row in grid}) == 1
            and all(type(cell) is int and 0 <= cell <= 9 for row in grid for cell in row))


def parse_final_grid(content):
    """Accept exactly one distinct JSON grid, never infer cells from prose."""
    grids, offset, decoder = [], 0, json.JSONDecoder()
    while (offset := content.find("[", offset)) >= 0:
        try:
            value, consumed = decoder.raw_decode(content[offset:])
        except ValueError:
            offset += 1
            continue
        if grid_valid(value) and value not in grids:
            grids.append(value)
        offset += consumed
    return grids[0] if len(grids) == 1 else None


def empty_submission(queries):
    return {key: [{name: [[0]] for name in ATTEMPTS} for _ in task["test"]]
            for key, task in queries.items()}


def validate_submission(submission, queries):
    if not isinstance(submission, dict) or set(submission) != set(queries):
        raise ValueError("Submission keys must exactly match the challenge JSON")
    for key, task in queries.items():
        outputs = submission[key]
        if not isinstance(outputs, list) or len(outputs) != len(task["test"]):
            raise ValueError(f"Wrong submission output count for {key}")
        for output in outputs:
            if (not isinstance(output, dict) or set(output) != set(ATTEMPTS)
                    or not all(grid_valid(output[name]) for name in ATTEMPTS)):
                raise ValueError(f"Malformed submission grid for {key}")


def merge_output(primary, secondary, *, primary_count=None, policy="one-each", alternative=None):
    """No successful secondary answer means byte-for-byte equivalent attempts. alternative: the secondary
    model's second-ranked grid (REPL-TIR voting); one-each places it after the first 4B attempt."""
    if policy not in ("one-each", "prefer-4b", "prefer-qwen38"):
        raise ValueError(f"Unknown merge policy: {policy}")
    if secondary is None:
        return deepcopy(primary), ["4b", "4b"]
    if not grid_valid(secondary) or alternative is not None and not grid_valid(alternative):
        raise ValueError("Cannot merge a malformed secondary grid")
    originals = [primary[name] for name in ATTEMPTS]
    if primary_count is None:
        primary_count = 0 if all(grid == [[0]] for grid in originals) else 2
    originals = originals[:max(0, min(2, int(primary_count)))]
    # With one secondary response, one-each and prefer-qwen38 coincide.
    ours = [("qwen38", secondary)] + ([("qwen38", alternative)] if alternative is not None else [])
    theirs = [("4b", g) for g in originals]
    ranked = (theirs + ours if policy == "prefer-4b" else
              ours[:1] + theirs[:1] + ours[1:] + theirs[1:] if policy == "one-each" else ours + theirs)
    chosen, sources = [], []
    for source, grid in ranked:
        if grid not in chosen:
            chosen.append(deepcopy(grid))
            sources.append(source)
        if len(chosen) == 2:
            break
    while len(chosen) < 2:
        chosen.append(deepcopy(chosen[-1]))
        sources.append(sources[-1])
    return dict(zip(ATTEMPTS, chosen)), sources


def make_prompt(template, task, test_index):
    examples = ""
    for index, pair in enumerate(task["train"]):
        examples += f"--Example {index}-- \n\n INPUT: \n\n"
        examples += json.dumps(pair["input"]) + "\n\nOUTPUT: \n\n"
        examples += json.dumps(pair["output"]) + "\n\n"
    # Only these challenge fields enter the prompt. Test output, labels,
    # 4B guesses, routing confidence and evaluation results are not consulted.
    return template.rstrip("\n").format(
        training_examples=examples, test_input=json.dumps(task["test"][test_index]["input"]))


def output_budget(prompt_tokens, context):
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("Prompt token count must be a non-negative integer")
    budget = context - prompt_tokens - 16
    if budget <= 0:
        raise ValueError("Prompt leaves no generation budget; no truncation allowed")
    return budget


def make_payload(prompt, prompt_tokens, context, *, images=None, mm_processor_kwargs=None,
                 solve_mode="direct", tir_interface="text", model="qwen38-27b", image_note=None):
    content = prompt
    if images is not None:
        # image_note describes how the images are drawn (Renderer.note); the default is the labelled sheets.
        content = [{"type": "text", "text": prompt}, {"type": "text", "text": (
            (image_note or qwen38_vision.TEXT_NOTE) + " " + (
                ("Follow the Python/tool instructions above, not grids or images." if tir_interface == "python"
                 else "Return the Python solver module as instructed, not grids or images.")
                if solve_mode == "tir" else "Return the output as a JSON numerical grid, not an image."))}]
        for image in images:
            content.extend(image.content_parts())
    payload = dict(model=model, messages=[dict(role="user", content=content)],
                   max_tokens=output_budget(prompt_tokens, context), reasoning_effort="xhigh",
                   chat_template_kwargs=dict(enable_thinking=True, reasoning_effort="xhigh"),
                   stream=True, stream_options=dict(include_usage=True))
    if mm_processor_kwargs is not None:
        payload["mm_processor_kwargs"] = mm_processor_kwargs
    return payload


def deadline_from_limits(end_time, max_seconds, reserve=0, *, now=None):
    now = time.time() if now is None else now
    if max_seconds == 0:
        # With a notebook deadline, let stage 2 consume all remaining time.
        # Standalone runs without a global deadline still have a finite cap.
        return end_time - reserve if end_time > 0 else now + 21600
    return min(now + max_seconds, end_time - reserve if end_time > 0 else math.inf)


def load_labels(args, queries):
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        return None
    path = args.solutions
    if path is None and Path(args.challenges).name == "arc-agi_evaluation_challenges.json":
        path = Path(resolve_evaluation_solution_path())
    if path is None or not Path(path).is_file():
        return None
    try:
        labels = json.loads(Path(path).read_text())
        if not all(key in labels and len(labels[key]) == len(task["test"])
                   and all(grid_valid(g) for g in labels[key]) for key, task in queries.items()):
            raise ValueError("Label keys/counts/grids do not match challenges")
        return labels
    except (OSError, ValueError, TypeError) as exc:
        print(f"*** Evaluation scoring disabled: {exc}", flush=True)
        return None


def evaluation_score(submission, labels, keys, *, generated=None):
    if labels is None:
        return None
    score = 0.0
    correct = 0
    for key in keys:
        for i, truth in enumerate(labels[key]):
            if generated is not None and (key, i) not in generated:
                continue  # A [[0]] placeholder is not a Qwen3.8 prediction.
            if any(submission[key][i][name] == truth for name in ATTEMPTS):
                correct += 1
                score += 1 / len(labels[key])
    return dict(task_weighted_score=score, max_score=len(keys), correct_outputs=correct)


async def wait_for_health(client, url, deadline, owned=None, *, log_path=None):
    with qwen38_server.StartupLogTail(log_path) as log:
        while time.monotonic() < deadline:
            log.emit()
            if owned is not None and owned.poll() is not None:
                raise RuntimeError(f"Qwen3.8 server exited with code {owned.returncode}")
            try:
                response = await client.get(url + "/health", timeout=max(0.001, min(10, deadline-time.monotonic())))
                response.raise_for_status()
                return
            except httpx.HTTPError:
                await asyncio.sleep(min(2, max(0, deadline-time.monotonic())))
        raise TimeoutError("Server health/deadline budget exhausted")


async def stream_answer(client, url, payload, output, timeout, *, parse_grid=True,
                        loop_guard=None, loop_action="off"):
    if loop_guard is not None and loop_action not in ("observe", "stop"):
        raise ValueError("A gzip guard needs observe or stop mode")
    if loop_guard is not None:
        payload = dict(payload, return_token_ids=True)
    output.mkdir()
    atomic_json(output / "request.json", payload)
    atomic_json(output / "messages.json", payload["messages"])
    start = time.monotonic()
    content, reasoning, usage = [], [], {}
    finish, error, first, last = None, None, None, None
    done = False
    cancelled = False
    loop_stopped = False
    calls = qwen38_tir.ToolCallAccumulator()
    try:
        async with asyncio.timeout(timeout):
            async with client.stream("POST", url + "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                with (output / "stream.jsonl").open("x", buffering=1) as raw:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        event = json.loads(data)
                        elapsed = time.monotonic()-start
                        raw.write(json.dumps(dict(elapsed_s=elapsed, event=event)) + "\n")
                        if event.get("error"):
                            raise RuntimeError(str(event["error"]))
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices", []):
                            if choice.get("index", 0) != 0:
                                raise ValueError("Unexpected extra completion sample")
                            delta = choice.get("delta") or {}
                            answer = delta.get("content") or ""
                            thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                            content.append(answer)
                            reasoning.append(thought)
                            if delta.get("tool_calls"):
                                calls.add(delta["tool_calls"])
                            if answer or thought or delta.get("tool_calls"):
                                first = elapsed if first is None else first
                                last = elapsed
                            finish = choice.get("finish_reason") or finish
                            if loop_guard is not None:
                                ids = choice.get("token_ids")
                                # Do not silently substitute a character window on old servers.
                                if ids is None and (answer or thought or delta.get("tool_calls")) and not loop_guard.disabled_reason:
                                    loop_guard.disabled_reason = "Server omitted delta token_ids; gzip guard disabled"
                                    print(f"WARNING {output.name}: {loop_guard.disabled_reason}", flush=True)
                                hit = loop_guard.feed(ids) if ids is not None else None
                                if hit:
                                    atomic_json(output / "loop_detection.json", dict(action=loop_action, **hit))
                                    print(f"GZIP {output.name}: ratio={hit['ratio']:.2f}, action={loop_action}, "
                                          f"token={hit['observed_completion_tokens']}", flush=True)
                                    # Do not discard a response that already finished naturally.
                                    if loop_action == "stop" and finish is None:
                                        loop_stopped = True
                        if loop_stopped:
                            finish = "loop_detected"
                            break  # Exiting the stream context closes this request, not the server.
                if not loop_stopped and (not done or not usage or finish is None):
                    raise RuntimeError("Incomplete SSE: missing DONE, usage or finish reason")
                if calls.calls and not loop_stopped:
                    calls.validate_complete()
    except asyncio.CancelledError:
        error, cancelled = "Cancelled: pipeline stopped or server exited", True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    text = "".join(content)
    grid = parse_final_grid(text) if parse_grid and error is None and finish == "stop" else None
    result = dict(content=text, reasoning="".join(reasoning), usage=usage, finish_reason=finish,
                  tool_calls=calls.snapshot(),
                  stream_done=done, error=error, cancelled=cancelled, final_grid=grid,
                  elapsed_s=time.monotonic()-start, first_content_s=first, last_content_s=last)
    if loop_guard is not None:
        result["loop_detection"] = dict(action=loop_action, stopped=loop_stopped, **loop_guard.snapshot())
        # An aborted SSE may never deliver provider usage; do not invent a billable count.
        result["usage_complete"] = done and bool(usage)
    result["decode_tokens_per_s_estimate"] = (
        usage.get("completion_tokens", 0) / (last-first)
        if first is not None and last is not None and last > first and usage else None)
    atomic_json(output / "result.json", result)
    return result


def add_qwen38_arguments(parser):
    qwen38_recipe.add_arguments(parser)
    qwen38_vision.add_arguments(parser)
    qwen38_tir.add_arguments(parser)
    variants = ("27b", "flash-next", "qwen35-9b", "dsv4", "dsv4-vllm", "flash-llama", "dots3")
    qwen38_server.add_backend_arguments(parser, "--qwen38-variant", variants)
    dsv4_llama.add_arguments(parser)
    dsv41_vllm.add_arguments(parser)
    dsv41_havu.add_arguments(parser)
    qwen38_llama.add_arguments(parser)
    dots3_llama.add_arguments(parser)
    parser.add_argument("--model-qwen38", default=qwen38_server.TARGET)
    parser.add_argument("--qwen38-draft-model", default=qwen38_server.DRAFT)
    parser.add_argument("--context", type=int, default=81920)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--server-max-num-seqs", type=int, default=None,
                        help="Optional server capacity override; defaults to --concurrency.")
    parser.add_argument("--qwen38-max-seconds", type=float, default=0,
                        help="Stage cap including startup; 0 uses the remaining global budget (6h without --end-time).")
    parser.add_argument("--health-timeout-seconds", type=float, default=2400)
    parser.add_argument("--server-startup-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Mirror the owned server's startup log to the notebook/console until healthy; "
                             "the complete server/server.log is still retained.")
    parser.add_argument("--request-timeout-seconds", type=float, default=21600)
    parser.add_argument("--keep-completed-streams", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep direct-output stream.jsonl debug logs after completion. By default remove "
                             "each log after result/submission/score/progress are saved; retain interrupted "
                             "or error streams and TIR round logs.")
    parser.add_argument("--qwen38-port", type=int, default=18083)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--model-loader-threads", type=int, default=32)
    parser.add_argument("--temp-root", type=Path, default=Path("/kaggle/working/q27_tmp"))
    parser.add_argument("--eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-url", help="Use an already running compatible server; do not start/stop it.")
    parser.add_argument("--queue-order", choices=("small-area", "short-prompt", "input"), default="small-area",
                        help="small-area: ascending test-input cells, then prompt tokens; no tasks dropped.")
    parser.add_argument("--loop-gzip-action", choices=("off", "observe", "stop"), default="off",
                        help="Opt-in heuristic: repeated ARC grids can be highly compressible even when correct.")
    parser.add_argument("--loop-gzip-window-tokens", type=int, default=1000)
    parser.add_argument("--loop-gzip-ratio", type=float, default=8.0)
    parser.add_argument("--loop-gzip-check-tokens", type=int, default=1000,
                        help="1000 with a 1000-token window checks disjoint windows.")
    parser.add_argument("--loop-gzip-confirmations", type=int, default=3,
                        help="Consecutive above-threshold checks before flagging; avoid a single repetitive grid.")
    parser.add_argument("--loop-tokenizer", type=Path,
                        help="Local target tokenizer.json or directory; defaults to --model-qwen38. No downloads.")
    parser.add_argument("--min-request-seconds", type=float, default=60,
                        help="Do not admit new requests below this remaining wall-time budget.")
    parser.add_argument("--min-request-tokens", type=int, default=60000,
                        help="Do not admit new requests once the remaining time, at the median measured "
                             "per-request decode rate, cannot reach this many tokens (or the request's "
                             "own budget if smaller). 0 disables.")
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).with_name("qwen38_system_prompt.txt"))
    parser.add_argument("--require-qwen38-success", action="store_true")


def build_parser():
    parser = ProfileParser(description=__doc__)
    add_qwen38_arguments(parser)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path,
                        help="Output root of an interrupted run with the same settings (llama.cpp backends, direct "
                             "mode): keep its finished outputs, continue each unfinished one from its streamed "
                             "tokens (same seed and total budget), run the rest. --qwen38-max-seconds still caps "
                             "this run alone.")
    parser.add_argument("--final-submission", type=Path, default=Path("/kaggle/working/submission.json"))
    parser.add_argument("--challenges", type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--task-keys", help="Comma-separated exact task keys; otherwise use ARC_TASK_KEY/INDEX.")
    parser.add_argument("--primary-submission", type=Path)
    parser.add_argument("--routing-manifest", type=Path)
    parser.add_argument("--merge-policy", choices=("one-each", "prefer-qwen38", "prefer-4b"), default="one-each")
    parser.add_argument("--end-time", type=float, default=0)
    parser.add_argument("--finalization-reserve-seconds", type=float, default=120)
    parser.add_argument("--run-status-file")
    return parser


def validate_options(args):
    qwen38_recipe.validate(args)
    dsv41_havu.validate_options(args)
    if args.server_max_num_seqs is None:
        args.server_max_num_seqs = args.concurrency
    qwen38_vision.validate_options(args)
    qwen38_tir.validate_options(args)
    qwen38_server.validate_backend_options(args)
    dsv4_llama.validate_options(args)
    dsv41_vllm.validate_options(args)
    qwen38_llama.validate_options(args)
    dots3_llama.validate_options(args)
    qwen38_loop.GzipLoopGuard(None, window_tokens=args.loop_gzip_window_tokens,
        ratio=args.loop_gzip_ratio, check_tokens=args.loop_gzip_check_tokens,
        confirmations=args.loop_gzip_confirmations)
    for name in ("context", "concurrency", "server_max_num_seqs",
                 "health_timeout_seconds", "request_timeout_seconds", "max_num_batched_tokens", "model_loader_threads",
                 "live_interval_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("draft_tokens", "end_time", "finalization_reserve_seconds", "min_request_seconds", "min_request_tokens",
                 "qwen38_max_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if args.concurrency > args.server_max_num_seqs:
        raise ValueError("Client concurrency exceeds the explicit --server-max-num-seqs; "
                         "omit --server-max-num-seqs to follow --concurrency, or increase it")
    if getattr(args, "resume_from", None) is not None and (args.qwen38_variant not in ("dsv4", "flash-llama", "dots3")
                                         or args.solve_mode != "direct"):
        raise ValueError("--resume-from needs a llama.cpp backend (dsv4, flash-llama or dots3) in direct mode")
    if not 0 < args.gpu_memory_utilization < 1 or not 1 <= args.qwen38_port <= 65535:
        raise ValueError("Invalid GPU utilization/port")


def prompt_ledger(args):
    """The REPL prompt ledger sends exact ids; multimodal prompts need vLLM's token-in/token-out route."""
    return (args.solve_mode in ("tir", "hybrid") and getattr(args, "tir_interface", None) == "repl"
            and getattr(args, "tir_repl_prompt_ids", None) == "ledger")


def server_options(args):
    batch = args.concurrency if args.server_max_num_seqs is None else args.server_max_num_seqs
    ledger = prompt_ledger(args)
    return qwen38_server.parser().parse_args([
        "--variant", args.qwen38_variant, "--mtp-tokens", str(args.mtp_tokens),
        "--flash-next-profile", args.flash_next_profile,
        "--flash-next-host-pool-gib", str(args.flash_next_host_pool_gib),
        *(["--flash-next-prefill-graphs"] if args.flash_next_prefill_graphs else []),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        *(["--gpu-ids", args.gpu_ids] if args.gpu_ids is not None else []),
        "--target-quantization", args.target_quantization,
        "--load-format", args.load_format,
        "--fastsafetensors-max-gpu-shard-gib", str(args.fastsafetensors_max_gpu_shard_gib),
        *(["--draft-stage-dir", str(args.draft_stage_dir)] if args.draft_stage_dir is not None else []),
        "--mtp-moe-backend", args.mtp_moe_backend, "--moe-backend", args.moe_backend,
        "--expert-parallel" if args.expert_parallel else "--no-expert-parallel",
        "--enable-eplb" if args.enable_eplb else "--no-enable-eplb",
        *(["--eplb-config", json.dumps(args.eplb_config)] if args.eplb_config is not None else []),
        "--ple-cpu-offload" if args.ple_cpu_offload else "--no-ple-cpu-offload",
        "--rust-frontend" if args.rust_frontend else "--no-rust-frontend",
        "--cpu-offload-gb", str(args.cpu_offload_gb),
        "--cpu-offload-params", *args.cpu_offload_params,
        "--kv-offload-gib-per-rank", str(args.kv_offload_gib_per_rank),
        "--kv-offload-mode", args.kv_offload_mode,
        "--model", str(args.model_qwen38), "--draft-model", str(args.qwen38_draft_model),
        "--output", str(args.output_root / "server"), "--temp-root", str(args.temp_root),
        "--context", str(args.context), "--batch", str(batch),
        "--draft-tokens", str(args.draft_tokens), "--port", str(args.qwen38_port),
        "--draft-quantization", args.draft_quantization,
        "--draft-rejection-method", args.draft_rejection_method,
        "--draft-sample-method", args.draft_sample_method,
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--model-loader-threads", str(args.model_loader_threads),
        *(["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
          if args.solve_mode == "tir" and qwen38_tir.tool_mode(args) else []),
        "--input-mode", args.input_mode,
        *(["--vision-cell-pixels", str(args.vision_cell_pixels)] if args.vision_cell_pixels else []),
        "--vision-max-pixels", str(args.vision_max_pixels),
        "--vision-max-images", str(args.vision_max_images),
        "--scale-out-endpoints" if ledger and args.input_mode == "text-image" else "--no-scale-out-endpoints",
        "--prompt-tokens-details" if ledger else "--no-prompt-tokens-details",
        *(f"--vllm-arg={a}" for a in args.vllm_arg),
        "--eager" if args.eager else "--no-eager", "--launch"])


def load_plan(args, queries):
    selected = select_task_keys(queries, task_keys=args.task_keys)
    primary = (json.loads(args.primary_submission.read_text()) if args.primary_submission
               else empty_submission(queries))
    validate_submission(primary, queries)
    counts, routed = {}, list(selected)
    if args.routing_manifest:
        if not args.primary_submission:
            raise ValueError("Routing manifest requires the matching primary submission")
        manifest = json.loads(args.routing_manifest.read_text())
        if not isinstance(manifest.get("routed_task_keys"), list):
            raise ValueError("Routing manifest must contain routed_task_keys")
        if any(k not in queries for k in manifest["routed_task_keys"]):
            raise ValueError("Routing manifest contains unknown tasks")
        routed = [k for k in selected if k in manifest["routed_task_keys"]]
        counts = manifest.get("nvarc_output_candidate_counts", {})
    # Filter only generation, not the scoring cohort or primary submission.
    # This preserves all 4B answers (standalone: [[0]]) for skipped tasks.
    filtered_keys = []
    for key in routed:
        task = queries[key]
        if len(task["train"]) != 2:
            pass
            # print("skip", key)
            # continue
        filtered_keys.append(key)
    routed = filtered_keys
    return selected, routed, primary, counts


def job_test_area(task, index, solve_mode="direct"):
    # TIR shares a single program across all test inputs: account for all of them,
    # rather than depend on the arbitrary index of the first input.
    tests = task["test"] if solve_mode == "tir" else [task["test"][index]]
    return sum(len(pair["input"]) * len(pair["input"][0]) for pair in tests)


def order_jobs(jobs, queries, policy, solve_mode="direct", *, task_modes=None):
    if policy == "small-area":
        return sorted(jobs, key=lambda job: (job_test_area(queries[job[0]], job[1],
            task_modes[job[0]] if task_modes is not None else solve_mode), job[2]))
    if policy == "short-prompt":
        return sorted(jobs, key=lambda job: job[2])
    if policy == "input":
        return list(jobs)
    raise ValueError(f"Unknown queue policy: {policy}")


# Options that change which outputs run or what they generate; a resumed run must keep them.
RESUME_SAME_OPTIONS = ("qwen38_variant", "context", "solve_mode", "input_mode", "challenges", "solutions",
                       "task_keys", "primary_submission", "routing_manifest", "merge_policy", "template",
                       "task_token_budget",
                       "qf_model", "dsv4_model", "dsv4_profile", "d3_model", "loop_gzip_action", "loop_gzip_window_tokens", "loop_gzip_ratio",
                       "loop_gzip_check_tokens", "loop_gzip_confirmations")


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (TypeError, ProcessLookupError):
        return False
    except PermissionError:
        pass
    return True


def load_resume(args, planned, template_path, sampling, effort):
    """An earlier run's finished rows (kept) and the streamed prefix of each unfinished output (continued)."""
    root = args.resume_from
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if summary.get("status") in ("preparing", "waiting-for-server", "running") and pid_alive(summary.get("pid")):
        raise ValueError(f"--resume-from {root} is still running (pid {summary['pid']})")
    old = manifest["options"]
    now = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    changed = [k for k in RESUME_SAME_OPTIONS if old.get(k) != now.get(k)]
    changed += [name for name, a, b in (("sampling", manifest.get("sampling"), json.loads(json.dumps(sampling))),
                                        ("reasoning effort", manifest.get("reasoning_effort"), effort),
                                        ("planned outputs", summary.get("planned_outputs"), len(planned))) if a != b]
    changed += [str(path) for path in (args.challenges, template_path)
                if manifest["source_sha256"].get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest()]
    if changed:
        raise ValueError(f"--resume-from {root} ran with different settings: {', '.join(changed)}")
    raw = json.loads((root / "qwen38_submission.json").read_text())
    kept, partial, fresh_notes = {}, {}, []
    for row in summary.get("results", []):
        if row.get("error") is None:
            grid = raw[row["task"]][row["test_index"]]["attempt_1"] if row.get("valid") else None
            kept[f"{row['task']}_{row['test_index']}"] = (row, grid)
    for key, index in planned:
        label = f"{key}_{index}"
        folder = root / "outputs" / label
        if label in kept or not any((folder / name).exists() for name in ("stream.jsonl", "resume.json")):
            continue
        try:
            prefix = dsv4_llama.read_partial(folder)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            fresh_notes.append(f"{label}: {exc}")
            continue
        if prefix["tokens"]:
            partial[label] = prefix
    return dict(source=str(root.resolve()), kept=kept, partial=partial, fresh_notes=fresh_notes,
                prior_elapsed_s=summary.get("prior_elapsed_s", 0) + summary.get("elapsed_s", 0))


async def run_inference(args):
    """The parent never imports Torch or shares CUDA state with either model."""
    global FINAL_PATH
    validate_options(args)
    recipe = qwen38_recipe.describe(args)
    if recipe["name"] != "manual":
        print(f"*** Qwen3.8 recipe {recipe['name']} v{recipe['version']}: "
              f"context={args.context}, concurrency={args.concurrency}, "
              f"task_tokens={args.task_token_budget}; overrides={recipe['overrides']}", flush=True)
    args.challenges = args.challenges or Path(resolve_challenge_path(os.getenv("KAGGLE_IS_COMPETITION_RERUN")))
    queries = json.loads(args.challenges.read_text())
    selected, routed, primary, counts = load_plan(args, queries)
    task_modes = {key: qwen38_tir.task_solve_mode(args, queries[key]) for key in routed}
    has_tir = "tir" in task_modes.values()
    template_path = qwen38_tir.template_path(args) if args.solve_mode == "tir" else args.template
    template = template_path.read_text()
    tir_template_path = qwen38_tir.template_path(args) if has_tir else None
    tir_template = tir_template_path.read_text() if has_tir else None
    labels = load_labels(args, queries)
    dsv4, flash = args.qwen38_variant == "dsv4", args.qwen38_variant == "flash-llama"
    dots3 = args.qwen38_variant == "dots3"
    dsv4v = args.qwen38_variant == "dsv4-vllm"  # DeepSeek-V4.1 on our vLLM fork, the dsv41_repl encoder.
    repl = has_tir and args.tir_interface == "repl"
    queued_repl = getattr(args, "tir_repl_trajectory_schedule", "sequential") == "queued"
    cot_second = getattr(args, "tir_repl_second_attempt", "repl") == "cot"
    cot_template = args.template.read_text() if cot_second else None
    cot_context = min(args.context, getattr(args, "tir_repl_cot_context", 102400))
    cot_states = {}
    trajectory_states = {}
    trajectory_queue = dict(first_waiting=0, repeat_waiting=0, started=0, completed=0, planned=0)
    if repl:
        tir_template = qwen38_repl.experiment_prompt(tir_template, args)
    # The llama.cpp backends send token-id prompts to /completion and share its stream and live view.
    llama = dsv4_llama if dsv4 else qwen38_llama if flash else dots3_llama if dots3 else None
    repl_effort = args.dsv4_reasoning_effort if dsv4 or dsv4v else args.qf_reasoning_effort if flash else "xhigh"
    planned = [(key, i) for key in routed for i in range(len(queries[key]["test"]))]
    # TIR synthesizes one program per task and reuses it for all test inputs.
    planned_jobs = [(key, i) for key, i in planned if task_modes[key] == "direct" or i == 0]
    direct_job_numbers = {job: n for n, job in enumerate(planned)}
    resume = (load_resume(args, planned, template_path, llama.sampling(args),
                          args.dsv4_reasoning_effort if dsv4 else args.qf_reasoning_effort if flash else None)
              if getattr(args, "resume_from", None) is not None else None)
    args.output_root.mkdir(parents=True, exist_ok=False)
    FINAL_PATH = args.final_submission
    merged, raw_submission = deepcopy(primary), empty_submission(queries)
    active, results = {}, []
    if resume is not None:
        # Finished outputs keep their answer; their folders are linked, not copied.
        (args.output_root / "outputs").mkdir()
        for label, (row, grid) in resume["kept"].items():
            key, index = row["task"], row["test_index"]
            row = dict(row, resumed_from=row.get("resumed_from", resume["source"]))
            if grid is not None:
                merged[key][index], row["selected_sources"] = merge_output(
                    primary[key][index], grid, primary_count=counts.get(label), policy=args.merge_policy)
                raw_submission[key][index] = {name: deepcopy(grid) for name in ATTEMPTS}
            if labels is not None:
                row["correct"] = grid is not None and grid == labels[key][index]
            if (args.resume_from / "outputs" / label).is_dir():
                (args.output_root / "outputs" / label).symlink_to((args.resume_from / "outputs" / label).resolve())
            results.append(row)
        print(f"RESUME {resume['source']}: kept {len(results)} finished outputs "
              f"({sum(bool(r.get('correct')) for r in results)} correct), continuing {len(resume['partial'])} "
              f"from {sum(p['tokens'] for p in resume['partial'].values()):,} streamed tokens, "
              f"{len(planned) - len(results) - len(resume['partial'])} fresh; earlier runs took "
              f"{resume['prior_elapsed_s']:.0f} s", flush=True)
        for note in resume["fresh_notes"]:
            print(f"RESUME fresh {note}", flush=True)
    atomic_json(args.final_submission, merged)  # Safe before model loading.
    atomic_json(args.output_root / "primary_submission.json", primary)
    atomic_json(args.output_root / "qwen38_submission.json", raw_submission)
    deadline_epoch = deadline_from_limits(args.end_time, args.qwen38_max_seconds, args.finalization_reserve_seconds)
    deadline = time.monotonic() + max(0, deadline_epoch-time.time())

    def job_indices(key, index):
        return list(range(len(queries[key]["test"]))) if task_modes[key] == "tir" else [index]

    tir_verified = set()
    started = time.monotonic()
    state = dict(status="preparing", error=None, selected_task_keys=selected, routed_task_keys=routed,
                 planned_outputs=len(planned), context=args.context, concurrency=args.concurrency,
                 input_mode=args.input_mode, solve_mode=args.solve_mode, planned_jobs=len(planned_jobs),
                 task_solve_modes=task_modes, task_token_budget=args.task_token_budget,
                 variant=args.qwen38_variant, solver_recipe=recipe,
                 draft_tokens={"flash-next": args.mtp_tokens, "dsv4": 0, "dsv4-vllm": 0, "flash-llama": args.qf_mtp_draft_max,
                               "dots3": args.d3_mtp_draft_max}.get(args.qwen38_variant, args.draft_tokens),
                 deadline_epoch=deadline_epoch,
                 final_submission=str(args.final_submission), pid=os.getpid(), server_pid=None)
    if resume is not None:
        state.update(resumed_from=resume["source"], prior_elapsed_s=resume["prior_elapsed_s"],
                     resumed_kept=len(resume["kept"]), resumed_continued=sorted(resume["partial"]))
    if cot_second:
        state["task_token_budget_scope"] = "repl-only"

    def save():
        generated = {(r["task"], r["test_index"]) for r in results if r.get("valid")} | tir_verified
        covered = {f"{r['task']}_{r['test_index']}" for r in results} | set(active)
        state.update(elapsed_s=time.monotonic()-started, active=dict(active), results=list(results),
                     finished_outputs=len(results), queued_outputs=max(0, len(planned)-len(covered)),
                     valid_outputs=len(generated),
                     primary_score=evaluation_score(primary, labels, selected),
                     qwen38_raw_score=evaluation_score(raw_submission, labels, selected, generated=generated),
                     qwen38_routed_score=evaluation_score(raw_submission, labels, routed, generated=generated),
                     merged_score=evaluation_score(merged, labels, selected))
        if queued_repl:
            state["trajectory_queue"] = dict(trajectory_queue, schedule="queued", order=args.queue_order)
        atomic_json(args.output_root / "summary.json", state)

    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    solver_recipe=recipe,
                    options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    selected_task_keys=selected, routed_task_keys=routed,
                    reasoning_effort=(args.dsv4_reasoning_effort if dsv4 or dsv4v else
                                      args.qf_reasoning_effort if flash else None if dots3 else "xhigh"),
                    sampling=(llama.sampling(args) if llama else dsv41_vllm.sampling(args) if dsv4v else
                              "server/model defaults"),
                    prompt_encoding=(dsv4_llama.encoder_info(args) if dsv4 else
                                     dict(dsv4_llama.encoder_info(args), transport=dsv41_vllm.__doc__.split("\n\n")[1])
                                     if dsv4v else
                                     "GGUF chat template (/apply-template) for the system/tools/task prefix, later "
                                     "turns by qwen38_repl (Flash-Next template), sent as token ids" if repl and flash else
                                     "checkpoint chat template for the messages after the previous turn's exact served "
                                     "ids, images on every REPL turn via /inference/v1/generate"
                                     if repl and args.input_mode == "text-image" and prompt_ledger(args) else
                                     "checkpoint chat template and images on every REPL turn via /v1/chat/completions"
                                     if repl and args.input_mode == "text-image" else
                                     "checkpoint chat template (/tokenize chat form) for every REPL turn, "
                                     "sent as token ids to /v1/completions" if repl and args.qwen38_variant in ("qwen35-9b", "flash-next") else
                                     "server chat template (/tokenize chat form) for the system/tools/task prefix, "
                                     "later turns by qwen38_repl (Flash-Next template), sent as token ids to "
                                     "/v1/completions" if repl else
                                     "GGUF chat template (/apply-template), sent as token ids" if flash else
                                     "GGUF chat template (/apply-template, enable_thinking), sent as token ids"
                                     if dots3 else
                                     "server chat template"),
                    request_order_policy=args.queue_order,
                    resumed_from=resume["source"] if resume is not None else None,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (args.challenges, template_path, Path(__file__),
                                             Path(qwen38_vision.__file__), Path(qwen38_server.__file__), Path(qwen38_loop.__file__),
                                             Path(qwen38_server.flash_profile.__file__),
                                             Path(qwen38_live.__file__), Path(qwen38_repl_vision.__file__),
                                             Path(qwen38_recipe.__file__),
                                             Path(dsv4_llama.__file__), *dsv4_llama.source_paths(args),
                                             *dsv41_vllm.source_paths(args),
                                             Path(qwen38_llama.__file__),
                                             Path(dots3_llama.__file__),
                                             *qwen38_tir.source_paths())})
    if tir_template_path is not None:
        manifest["source_sha256"][str(tir_template_path)] = hashlib.sha256(tir_template_path.read_bytes()).hexdigest()
    if cot_second:
        manifest["source_sha256"][str(args.template)] = hashlib.sha256(args.template.read_bytes()).hexdigest()
        manifest["second_attempt"] = dict(mode="cot", context=cot_context,
            requests="one per test input; fresh public task only", budget="independent per-request context",
            task_token_budget_scope="repl-only",
            selection="preserve REPL top-1; distinct valid CoT in slot 2")
    manifest["task_solve_modes"] = task_modes
    if args.solve_mode == "hybrid":
        manifest["prompt_encoding"] = dict(
            direct="GGUF chat template" if llama else "server chat template",
            tir=manifest["prompt_encoding"])
    atomic_json(args.output_root / "manifest.json", manifest)
    save()
    if not planned or deadline-time.monotonic() <= args.min_request_seconds:
        state["status"] = "not-routed" if not planned else "skipped-deadline"
        save()
        if not planned:
            print("*** Qwen3.8 not-routed: no task was routed to Qwen3.8", flush=True)
        else:
            print(f"*** Qwen3.8 skipped-deadline: {deadline_epoch-time.time():.0f} s left before the stage "
                  f"deadline, need over {args.min_request_seconds:.0f} s (now {time.time():.0f}, --end-time "
                  f"{args.end_time:.0f}, reserve {args.finalization_reserve_seconds:.0f} s, --qwen38-max-seconds "
                  f"{args.qwen38_max_seconds:.0f}). Re-set the end time and run again.", flush=True)
        return state
    owned = None
    workers = []
    monitor = None
    fatal = False
    renderer, vision_metadata, vision_manifest = None, None, None
    prompt_details = {}
    tir_images = {}
    request_files, stream_progress, live = {}, {}, None
    prefixes = {}  # Label -> streamed prefix of a resumed output.
    repl_prefixes = {}  # Label -> served rendering of the REPL system/tools/task turns.
    repl_slots = asyncio.Semaphore(args.tir_repl_exec_slots) if repl else None
    repl_cpus = qwen38_llama.free_cpus(args) if repl and flash else []
    repl_server = None
    url = (args.base_url or f"http://127.0.0.1:{args.qwen38_port}").rstrip("/")
    try:
        loop_tokenizer = None
        if args.loop_gzip_action != "off" and llama:
            # Same token windows, decoded by the server itself: no separate tokenizer to match.
            manifest["loop_tokenizer"] = dict(source="llama.cpp /completion stream text per token")
            atomic_json(args.output_root / "manifest.json", manifest)
        elif args.loop_gzip_action != "off":
            loop_tokenizer, manifest["loop_tokenizer"] = qwen38_loop.load_tokenizer(args.model_qwen38, args.loop_tokenizer)
            atomic_json(args.output_root / "manifest.json", manifest)

        def llama_guard():
            if args.loop_gzip_action == "off":
                return {}
            return dict(loop_action=args.loop_gzip_action, loop_guard=qwen38_loop.StreamTextLoopGuard(
                window_tokens=args.loop_gzip_window_tokens, ratio=args.loop_gzip_ratio,
                check_tokens=args.loop_gzip_check_tokens, confirmations=args.loop_gzip_confirmations))

        def token_guard():
            if loop_tokenizer is None:
                return {}
            return dict(loop_action=args.loop_gzip_action, loop_guard=qwen38_loop.GzipLoopGuard(loop_tokenizer,
                window_tokens=args.loop_gzip_window_tokens, ratio=args.loop_gzip_ratio,
                check_tokens=args.loop_gzip_check_tokens, confirmations=args.loop_gzip_confirmations))

        async def guarded_stream(*a, **kw):
            return await stream_answer(*a, **kw, **token_guard())

        if has_tir:
            sandbox = await qwen38_tir.preflight(args, deadline)
            atomic_json(args.output_root / "tir_runtime.json", sandbox)
        if args.input_mode == "text-image":
            vision_metadata = (dsv41_repl.read_metadata(args.dsv4_mmproj) if dsv4 else
                               dsv41_vllm.read_metadata(args.dsv41_vllm_model) if dsv4v else
                               qwen38_vision.read_metadata(args.model_qwen38))
            renderer = qwen38_vision.Renderer(args, vision_metadata)
            vision_manifest = dict(schema_version=1, renderer=renderer.settings,
                                   checkpoint=vision_metadata, requests={})
            atomic_json(args.output_root / "vision_manifest.json", vision_manifest)
        if not args.base_url:
            owned, receipt = (llama.launch(args) if llama else dsv41_vllm.launch(args) if dsv4v else
                              qwen38_server.launch(server_options(args)))
            state["server_pid"] = owned.pid
            manifest["server_launch"] = receipt
            atomic_json(args.output_root / "manifest.json", manifest)
        state["status"] = "waiting-for-server"
        save()
        timeout = httpx.Timeout(connect=20, read=None, write=60, pool=60)
        # --concurrency bounds the requests; httpx's default pool of 100 failed the 101st on with PoolTimeout.
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
            await wait_for_health(client, url, min(deadline, time.monotonic()+args.health_timeout_seconds), owned,
                                  log_path=(args.output_root / "server" / "server.log"
                                            if owned is not None and args.server_startup_log else None))
            async with asyncio.timeout(max(0.001, deadline-time.monotonic())):
                if llama:
                    # Slot context/count, special-token vocabulary, then one short completion.
                    served = await llama.check_server(client, url, args)
                    served["smoke"] = await llama.smoke(client, url, served["special_token_ids"],
                                                        args.output_root / "smoke")
                else:
                    response = await client.get(url + "/v1/models", timeout=20)
                    response.raise_for_status()
                    model_name = (dsv41_vllm.SERVED_NAME if dsv4v else
                                  qwen38_server.served_model_name(args.qwen38_variant))
                    served = next((x for x in response.json()["data"] if x["id"] == model_name), None)
                    if served is None or served.get("max_model_len", 0) < args.context:
                        raise ValueError("Server does not advertise the requested model/context")
                if repl:
                    repl_server = (dsv41_repl.Dsv41ReplServer(client, url, root=args.output_root,
                                       metadata=vision_metadata, max_images=args.vision_max_images) if dsv4 else
                                   dsv41_vllm.Dsv41VllmServer(client, url, model_name, root=args.output_root,
                                       metadata=vision_metadata, max_images=args.vision_max_images) if dsv4v else
                                   qwen38_repl_vision.VllmVisionServer(client, url, model_name,
                                       root=args.output_root, metadata=vision_metadata,
                                       max_images=args.vision_max_images,
                                       token_transport=prompt_ledger(args)) if renderer is not None else
                                   qwen38_repl.LlamaServer(client, url) if llama else
                                   qwen38_repl.VllmServer(client, url, model_name,
                                                        native_template=args.qwen38_variant in ("qwen35-9b", "flash-next")
                                                        or getattr(args, "tir_repl_test_reveal", "all") != "all"))
                    if not llama:
                        served = dict(served, special_token_ids=await repl_server.specials())
                    if getattr(repl_server, "token_transport", False):
                        await repl_server.check_transport(served["special_token_ids"]["im_start"])
                    if (dsv4 or dsv4v) and renderer is not None:
                        # The smallest (padded) and the largest test image: the server must count our ids.
                        served["vision_transport"] = await repl_server.check_transport([
                            renderer.test_image([[0]], 0, "Transport check A."),
                            renderer.test_image([[(r + c) % 10 for c in range(30)] for r in range(30)], 0,
                                                "Transport check B.")])
                    # 9B/FN and vision use native per-turn rendering; legacy text profiles keep their checked renderer.
                    (args.output_root / "repl_system_prompt.txt").write_text(tir_template)
                    check = await qwen38_repl.check_template(repl_server, tir_template, repl_effort,
                                                            tools=qwen38_repl.tool_schemas(args))
                    if args.tir_repl_keep_reasoning != 0 and not check.get("reasoning_preserved"):
                        raise ValueError("REPL reasoning retention requested, but the served chat template drops it")
                    atomic_json(args.output_root / "repl_template_check.json", dict(check, sandbox_cpus=repl_cpus))
                    served["repl_template"] = {k: v for k, v in check.items() if not k.endswith("_excerpt")}
                    if check.get("renderer") != "checkpoint-every-turn" and not check.get("turns_exact"):
                        print("WARNING REPL turns render differently from the served template; see "
                              "repl_template_check.json", flush=True)
                if llama:
                    print(f"Server ready: {json.dumps(served)}", flush=True)
                if loop_tokenizer is not None:
                    probe = "ARC grid: [[0, 1, 2], [3, 4, 5]]\nCoordinates (12, 29); reasoning → answer."
                    response = await client.post(url + "/tokenize", json=dict(model=model_name,
                        prompt=probe, add_special_tokens=False), timeout=30)
                    response.raise_for_status()
                    ids = response.json().get("tokens")
                    if ids != loop_tokenizer.encode(probe, add_special_tokens=False).ids:
                        raise ValueError("Gzip guard tokenizer does not match server probe; use the target checkpoint tokenizer")
                    manifest["loop_tokenizer"]["server_probe_matched"] = True
                manifest["served_model"] = served
                atomic_json(args.output_root / "manifest.json", manifest)
                jobs = []
                for job_number, (key, index) in enumerate(planned_jobs):  # Seeds count every planned job.
                    label = f"{key}_{index}"
                    mode = task_modes[key]
                    job_repl = repl and mode == "tir"
                    if args.solve_mode == "hybrid":
                        # Removing sibling jobs for REPL must not change CoT's seeded backends.
                        job_number = direct_job_numbers[(key, index)]
                    if resume is not None and label in resume["kept"]:
                        continue
                    prompt = (None if job_repl else qwen38_tir.make_prompt(tir_template, queries[key]) if mode == "tir"
                              else make_prompt(template, queries[key], index))
                    images, mm_kwargs = None, None
                    if renderer is not None:
                        try:
                            if mode == "tir":
                                visible = (qwen38_repl.split_holdout(queries[key],
                                    getattr(args, "tir_repl_holdout_first", False))[0] if job_repl else queries[key])
                                images = qwen38_repl_vision.task_images(renderer, visible,
                                    test_reveal=getattr(args, "tir_repl_test_reveal", "all") if job_repl else "all",
                                    visual_tools=job_repl and args.tir_repl_visual_tools)
                            else:
                                images = renderer.render(queries[key], index)
                        except qwen38_vision.RenderLimitError as exc:
                            for failed_index in job_indices(key, index):
                                results.append(dict(task=key, test_index=failed_index, valid=False,
                                                    status="vision-render-limit", error=str(exc)))
                            save()
                            continue
                        vision_manifest["requests"][label] = dict(images=[im.save(args.output_root) for im in images])
                        atomic_json(args.output_root / "vision_manifest.json", vision_manifest)
                        mm_kwargs = qwen38_vision.processor_kwargs(args.vision_max_pixels)
                    if dsv4 and not job_repl:
                        # The release's official encoder (0731 or V4.1-1017), sent as token ids: the server chat
                        # template is not used.
                        encoded = dsv4_llama.encode(prompt, args.dsv4_reasoning_effort)
                        ids = await dsv4_llama.tokenize_prompt(client, url, encoded, served["special_token_ids"])
                        payload, count = dsv4_llama.make_payload(args, ids, args.dsv4_seed + job_number), len(ids)
                        request_files[label] = {"messages.json": dsv4_llama.messages_for(prompt), "prompt.txt": encoded}
                    elif job_repl:
                        # The template file is the system prompt; the served template renders it with the tools and task.
                        repl_prefixes[label], ids = await qwen38_repl.render_prefix(
                            repl_server, tir_template, queries[key], repl_effort, served["special_token_ids"],
                            holdout_first=getattr(args, "tir_repl_holdout_first", False),
                            tools=qwen38_repl.tool_schemas(args), images=images, mm_kwargs=mm_kwargs,
                            image_note=renderer.note if renderer is not None else None,
                            test_reveal=getattr(args, "tir_repl_test_reveal", "all"),
                            grid_sep=qwen38_repl.grid_sep(args))
                        # vLLM samples unseeded with the model's generation config, as the direct 27b requests do.
                        payload = (dsv4_llama.make_payload(args, ids, args.dsv4_seed + job_number) if dsv4 else
                                   dsv41_vllm.make_payload(args, ids) if dsv4v else
                                   qwen38_llama.make_payload(args, ids, args.qf_seed + job_number) if llama else
                                   dict(prompt=ids, max_tokens=0))
                        count = len(ids)
                        if images is not None:
                            tokenization = ids.tokenization
                    elif flash:
                        # The served GGUF template renders the prompt once; the checked token ids are sent.
                        rendered = await qwen38_llama.render(client, url, prompt, args.qf_reasoning_effort)
                        ids = await qwen38_llama.tokenize_prompt(client, url, rendered, served["special_token_ids"])
                        payload, count = qwen38_llama.make_payload(args, ids, args.qf_seed + job_number), len(ids)
                        request_files[label] = {"messages.json": qwen38_llama.messages_for(prompt), "prompt.txt": rendered}
                    elif dots3:
                        # As flash-llama: the served GGUF template renders the prompt, the checked token ids are sent.
                        rendered = await dots3_llama.render(client, url, prompt)
                        ids = await dots3_llama.tokenize_prompt(client, url, rendered, served["special_token_ids"])
                        payload, count = dots3_llama.make_payload(args, ids, args.d3_seed + job_number), len(ids)
                        request_files[label] = {"messages.json": dots3_llama.messages_for(prompt), "prompt.txt": rendered}
                    else:
                        payload = make_payload(prompt, 0, args.context, images=images, mm_processor_kwargs=mm_kwargs,
                                               solve_mode=mode, tir_interface=args.tir_interface, model=model_name,
                                               image_note=renderer.note if renderer is not None else None)
                        if mode == "tir" and qwen38_tir.tool_mode(args):
                            qwen38_tir.add_verifier_tool(payload, args.tir_interface)
                        response = await client.post(url + "/tokenize", json=qwen38_tir.tokenization_payload(payload), timeout=120)
                        response.raise_for_status()
                        tokenization = response.json()
                        count = tokenization["count"]
                    if images is not None:
                        details = (dsv41_repl.audit_tokenization if dsv4 or dsv4v else qwen38_vision.audit_tokenization)(
                            tokenization, images, vision_metadata)
                        if mode == "tir":
                            tir_images[label] = images
                        prompt_details[label] = details
                        vision_manifest["requests"][label].update(prompt_tokens=count, **details)
                        atomic_json(args.output_root / "vision_manifest.json", vision_manifest)
                    if resume is not None and label in resume["partial"]:
                        # Continue the output: the same prompt and seed, then its streamed tokens.
                        prefix = resume["partial"][label]
                        tokens = [t for event_tokens, _ in prefix["events"] for t in event_tokens]
                        sent = json.loads((Path(prefix["source"]) / "request.json").read_text())
                        if sent["prompt"] != ids + tokens[:len(sent["prompt"]) - len(ids)] or sent["seed"] != payload["seed"]:
                            raise ValueError(f"{label}: the prompt or seed differs from {prefix['source']}")
                        payload["prompt"], prefixes[label] = ids + tokens, prefix
                    try:
                        # Retain the identical tokenized message/images. Only the
                        # generation budget changes; no second rendering pass.
                        payload["max_tokens"] = output_budget(count + prefixes.get(label, {}).get("tokens", 0), args.context)
                        if mode == "direct":
                            payload["max_tokens"] = qwen38_tir.direct_token_allowance(
                                args.task_token_budget, len(queries[key]["test"]), index, payload["max_tokens"])
                            if label in prefixes:
                                # A resumed output cannot receive a second share of its task budget.
                                payload["max_tokens"] = min(payload["max_tokens"],
                                    qwen38_tir.direct_token_allowance(args.task_token_budget,
                                        len(queries[key]["test"]), index, output_budget(count, args.context))
                                    - prefixes[label]["tokens"])
                                if payload["max_tokens"] <= 0:
                                    raise ValueError("Resumed output has exhausted its completion-token allowance")
                        if job_repl:  # The first turn's limit; the trajectory sets each turn's own.
                            payload["max_tokens"] = min(payload["max_tokens"], args.tir_repl_turn_tokens)
                            if args.task_token_budget is not None:
                                payload["max_tokens"] = min(payload["max_tokens"], args.task_token_budget)
                    except ValueError as exc:
                        for failed_index in job_indices(key, index):
                            results.append(dict(task=key, test_index=failed_index, valid=False,
                                                status="prompt-too-long", error=str(exc), prompt_tokens=count,
                                                **prompt_details.get(label, {})))
                        save()
                        continue
                    jobs.append((key, index, count, payload))
                # Only public test-input geometry and exact prompt token counts.
                # No test labels, per-task outcomes or 4B placeholder penalty.
                jobs = order_jobs(jobs, queries, args.queue_order, task_modes=task_modes)
                # Continued outputs first: they were running when the earlier run stopped.
                jobs.sort(key=lambda job: f"{job[0]}_{job[1]}" not in prefixes)
                atomic_json(args.output_root / "queue.json", [dict(task=k, test_index=i, test_indices=job_indices(k, i),
                    solve_mode=task_modes[k], task_token_budget=args.task_token_budget,
                    test_input_area=job_test_area(queries[k], i, task_modes[k]),
                    prompt_tokens=n, max_output_tokens=p["max_tokens"], **prompt_details.get(f"{k}_{i}", {}),
                    **({"resumed_tokens": prefixes[f"{k}_{i}"]["tokens"]} if f"{k}_{i}" in prefixes else {}))
                    for k, i, n, p in jobs])
            queue = asyncio.PriorityQueue() if queued_repl else asyncio.Queue()
            for priority, job in enumerate(jobs):
                queue.put_nowait((0, priority, job) if queued_repl else job)
            if queued_repl:
                trajectory_queue.update(first_waiting=len(jobs), planned=(
                    len(jobs) + sum(len(queries[k]["test"]) for k, _, _, _ in jobs)
                    if cot_second else len(jobs)*args.tir_repl_trajectories))
                if cot_second:
                    trajectory_queue.update(second_attempt="cot", repeat_unit="test-input request",
                                            cot_context=cot_context, task_token_budget_scope="repl-only")
                print(f"REPL queued trajectories: max={args.tir_repl_trajectories}/task, "
                      f"per_trajectory_tokens={args.tir_repl_budget}, total_task_tokens={args.task_token_budget}; "
                      f"first attempts before retries, order={args.queue_order}; no completion barrier; "
                      f"second_attempt={'cot' if cot_second else 'repl'}"
                      + (f"; token caps above apply to REPL only; independent CoT context={cot_context}/request"
                         if cot_second else ""), flush=True)
            state["status"] = "running"
            save()
            queue_changed = asyncio.Event()
            busy_workers = 0

            def reachable_tokens():
                """Tokens one new request could decode before the deadline, once a rate is measured."""
                # A TIR task produces several output rows from one conversation.
                # Count its measured rate once, not once per test input.
                rates_by_job = {r.get("usage_shared_task", f"{r['task']}_{r['test_index']}"):
                                r["decode_tokens_per_s_estimate"] for r in results if r.get("decode_tokens_per_s_estimate")}
                rates = list(rates_by_job.values())
                return (deadline-time.monotonic()) * statistics.median(rates) if rates else math.inf

            async def cot_repeat(key, index, allowance):
                """One independent raw CoT request per test input; no Python session/history."""
                label = f"{key}_{index}"
                second = cot_states[key]
                now = time.monotonic()
                if second.get("started") is None:
                    second["started"] = now
                # Raw CoT gets a fresh per-request timeout, not TIR's remainder
                # or a sibling's time window. The global deadline still wins.
                end = min(deadline, now + args.request_timeout_seconds)
                folder = args.output_root / "outputs" / label / "cot_1"
                request = dict(final_grid=None, usage={}, usage_complete=False, error=None,
                    finish_reason=None, cancelled=False, elapsed_s=0, prompt_tokens=0,
                    max_tokens=allowance, context=cot_context, output=str(folder))
                active[label] = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    prompt_tokens=0, trajectory=1, solve_mode="direct", context=cot_context)
                save()
                try:
                    if end - now <= args.min_request_seconds or allowance <= 0:
                        raise TimeoutError("No CoT request/deadline budget remains")
                    async with asyncio.timeout(end - now):
                        # Whitelist challenge fields, even when evaluation outputs are embedded.
                        public = dict(train=[dict(input=p["input"], output=p["output"])
                                             for p in queries[key]["train"]],
                                      test=[dict(input=p["input"]) for p in queries[key]["test"]])
                        prompt = make_prompt(cot_template, public, index)
                        images = renderer.render(public, index) if renderer is not None else None
                        # As VllmVisionServer: current servers pin processor limits
                        # globally and reject per-request mm_processor_kwargs.
                        # Audit actual image tokens below; never silently resize.
                        payload = make_payload(prompt, 0, cot_context, model=model_name, images=images,
                            image_note=renderer.note if renderer is not None else None)
                        if dsv4v:
                            payload.update(dsv41_vllm.chat_overrides(args))
                        response = await client.post(url + "/tokenize",
                            json=qwen38_tir.tokenization_payload(payload), timeout=min(120, max(.001, end-time.monotonic())))
                        response.raise_for_status()
                        tokenization = response.json()
                        count = tokenization["count"]
                        request["prompt_tokens"] = count
                        payload["max_tokens"] = output_budget(count, cot_context)
                        request["max_tokens"] = payload["max_tokens"]
                        details = {}
                        if images is not None:
                            details = (dsv41_repl.audit_tokenization(dict(tokenization,
                                           placeholder_id=vision_metadata["image_token_id"]), images, vision_metadata)
                                       if dsv4v else qwen38_vision.audit_tokenization(tokenization, images, vision_metadata))
                            vision_manifest["requests"][label + "/cot_1"] = dict(
                                images=[im.save(args.output_root) for im in images], prompt_tokens=count, **details)
                            atomic_json(args.output_root / "vision_manifest.json", vision_manifest)
                        active[label].update(prompt_tokens=count, max_tokens=payload["max_tokens"], **details)
                        save()
                        print(f"START CoT {label}: context={cot_context}, prompt={count}, "
                              f"max_output={payload['max_tokens']}, trajectory=2/2", flush=True)
                        request.update(await guarded_stream(client, url, payload, folder,
                            max(.001, end-time.monotonic())))
                except asyncio.CancelledError:
                    request.update(error="Cancelled: pipeline stopped or server exited", cancelled=True)
                except Exception as exc:
                    request["error"] = f"{type(exc).__name__}: {exc}"
                if request.get("cancelled") and time.monotonic() >= end:
                    # stream_answer records cancellation instead of propagating it.
                    # A local timeout must not retire a worker needed by other tasks.
                    request.update(cancelled=False, finish_reason="timeout", timed_out=True,
                        error=f"TimeoutError: CoT {'global' if end == deadline else 'request'} deadline exhausted")
                request["elapsed_s"] = time.monotonic() - now
                request["usage_complete"] = bool(request.get("stream_done") and request.get("usage"))
                atomic_json(folder / "result.json", request)
                second["requests"][index] = {k: v for k, v in request.items()
                    if k not in ("reasoning", "content", "tool_calls")}  # Full traces stay on disk.
                combined = qwen38_repl_cot.combine(second["first"], second["requests"],
                    allowances=second["allowances"], context=cot_context,
                    elapsed_s=time.monotonic() - second["started"])
                atomic_json(args.output_root / "programs" / key / "combined_result.json", combined)
                # Refresh all sibling rows with the SAME shared-task usage, not one
                # copy of the REPL tokens per CoT request. No label-based selection.
                for row in results:
                    if row["task"] != key:
                        continue
                    i = row["test_index"]
                    grid, alt = combined["final_grids"][i], combined["alternative_grids"][i]
                    output_label = f"{key}_{i}"
                    output_folder = args.output_root / "outputs" / output_label
                    row.update(valid=grid is not None, usage=combined["usage"],
                        task_token_budget_scope="repl-only", repl_usage=combined["repl_usage"],
                        cot_usage=combined["cot_usage"],
                        usage_complete=combined["usage_complete"], elapsed_s=combined["elapsed_s"],
                        streamed_tokens=combined["streamed_tokens"],
                        decode_tokens_per_s_estimate=combined["decode_tokens_per_s_estimate"],
                        solver_path=combined["solver_paths"][i],
                        second_attempt=combined["second_attempt"],
                        prediction_sources=combined["second_attempt"]["prediction_sources"][i])
                    if grid is not None:
                        row["error"] = None
                        merged[key][i], row["selected_sources"] = merge_output(
                            primary[key][i], grid, primary_count=counts.get(output_label),
                            policy=args.merge_policy, alternative=alt)
                        raw_submission[key][i] = dict(zip(ATTEMPTS, deepcopy([grid, grid if alt is None else alt])))
                    if labels is not None:
                        row["correct"] = grid is not None and labels[key][i] in (grid, alt)
                        row["correct_top1"] = grid is not None and grid == labels[key][i]
                    atomic_json(output_folder / "result.json", dict(combined, final_grid=grid, test_index=i))
                    atomic_json(output_folder / "score.json", row)
                atomic_json(args.final_submission, merged)
                atomic_json(args.output_root / "qwen38_submission.json", raw_submission)
                active.pop(label, None)
                trajectory_queue["completed"] += 1
                save()
                cleanup_completed_stream(folder, request, keep=getattr(args, "keep_completed_streams", False))
                print(f"DONE CoT {label}: valid={request.get('final_grid') is not None}, "
                      f"finish={request.get('finish_reason')}, error={request.get('error')}", flush=True)
                return request.get("cancelled", False)

            async def one_worker():
                nonlocal busy_workers
                while not fatal and deadline-time.monotonic() > args.min_request_seconds:
                    if queue.empty():
                        if not cot_second or not busy_workers:
                            break
                        # Keep idle workers for multi-input CoT fan-out, even if
                        # there are fewer first-pass task jobs than worker slots.
                        queue_changed.clear()
                        try:
                            await asyncio.wait_for(queue_changed.wait(), max(.001, deadline-time.monotonic()))
                        except TimeoutError:
                            break
                        continue
                    item = queue.get_nowait()
                    trajectory_number, priority, job = item if queued_repl else (0, 0, item)
                    key, index, count, payload = job
                    label = f"{key}_{index}"
                    mode = task_modes[key]
                    job_repl = repl and mode == "tir"
                    # A late start rarely finishes; the idle slot speeds up the requests still running.
                    if (reachable := reachable_tokens()) < min(args.min_request_tokens, payload["max_tokens"]):
                        if queued_repl:
                            queue.put_nowait(item)
                        print(f"SKIP {label}: ~{reachable:.0f} tokens reachable before the deadline", flush=True)
                        break
                    if queued_repl:
                        trajectory_queue["repeat_waiting" if trajectory_number else "first_waiting"] -= 1
                        trajectory_queue["started"] += 1
                    busy_workers += 1
                    if cot_second and trajectory_number == 1:
                        try:
                            cancelled = await cot_repeat(key, index, payload["max_tokens"])
                        finally:
                            busy_workers -= 1
                            queue_changed.set()
                        if cancelled:
                            break
                        continue
                    for output_index in job_indices(key, index):
                        active[f"{key}_{output_index}"] = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                            prompt_tokens=count, **prompt_details.get(label, {}),
                            **({"trajectory": trajectory_number} if queued_repl else {}))
                    save()
                    vision_log = (f", images={prompt_details[label]['image_count']}, "
                                  f"image_tokens={prompt_details[label]['image_tokens']}"
                                  if label in prompt_details else "")
                    resumed = f", resumed +{prefixes[label]['tokens']}" if label in prefixes else ""
                    print(f"START {label}: prompt={count}{resumed}, max_output={payload['max_tokens']}{vision_log}"
                          + (f", trajectory={trajectory_number + 1}/{args.tir_repl_trajectories}" if queued_repl else ""),
                          flush=True)
                    (args.output_root / "outputs").mkdir(exist_ok=True)
                    if mode == "tir":
                        folder = args.output_root / "programs" / key
                        folder.parent.mkdir(exist_ok=True)

                        def commit_tir_predictions(predictions, alternatives=None):
                            # Preserve passing outputs before a revision for a
                            # different test input can consume the remaining time.
                            for output_index, grid in enumerate(predictions):
                                if grid is None:
                                    continue
                                output_label = f"{key}_{output_index}"
                                alt = alternatives[output_index] if alternatives else None
                                merged[key][output_index], _ = merge_output(
                                    primary[key][output_index], grid, primary_count=counts.get(output_label),
                                    policy=args.merge_policy, alternative=alt)
                                raw_submission[key][output_index] = dict(zip(ATTEMPTS, deepcopy([grid, grid if alt is None else alt])))
                                tir_verified.add((key, output_index))
                            atomic_json(args.final_submission, merged)
                            atomic_json(args.output_root / "qwen38_submission.json", raw_submission)
                            save()

                        if job_repl:
                            now = time.monotonic()
                            continuation = trajectory_states.setdefault(key, {}) if queued_repl else None
                            # Completed trajectories already contribute through results; only
                            # this dispatch's new tokens belong in the live progress counter.
                            stream_progress[label] = dict(prompt_tokens=count, decoded=0, decoded_before=0,
                                                          started=now, task_started=now)
                            result = await qwen38_repl.solve_task(args, queries[key], payload, count, repl_server,
                                folder, deadline, prefix=repl_prefixes[label] if queued_repl else repl_prefixes.pop(label),
                                specials=served["special_token_ids"],
                                progress=stream_progress[label], guard=llama_guard if llama else token_guard,
                                slots=repl_slots,
                                write_json=atomic_json, on_predictions=commit_tir_predictions, cpus=repl_cpus,
                                visualizer=qwen38_repl_vision.Visualizer(renderer, args.output_root)
                                    if renderer is not None and args.tir_repl_visual_tools else None,
                                **({"trajectory_state": continuation} if queued_repl else {}))
                        else:
                            result = await qwen38_tir.solve_task(args, queries[key], payload, count, client, url,
                                folder, deadline, stream_answer=guarded_stream, write_json=atomic_json,
                                check_tokenization=(lambda value: qwen38_vision.audit_tokenization(
                                    value, tir_images[label], vision_metadata)) if label in tir_images else None,
                                on_predictions=commit_tir_predictions)
                    else:
                        folder = args.output_root / "outputs" / label
                        request_timeout = min(args.request_timeout_seconds, max(0.001, deadline-time.monotonic()))
                        if llama:
                            stream_progress[label] = {}
                            result = await dsv4_llama.stream_completion(client, url, payload, folder, request_timeout,
                                parse_grid=parse_final_grid, specials=served["special_token_ids"],
                                progress=stream_progress[label], files=request_files.pop(label),
                                marker="</think>" if flash or dots3 else None, prefix=prefixes.pop(label, None), **llama_guard())
                        else:
                            result = await guarded_stream(client, url, payload, folder, request_timeout)
                    for output_index in job_indices(key, index):
                        output_label = f"{key}_{output_index}"
                        output_folder = args.output_root / "outputs" / output_label
                        grid = result["final_grids"][output_index] if mode == "tir" else result["final_grid"]
                        alt = result["alternative_grids"][output_index] if result.get("alternative_grids") else None
                        row = dict(task=key, test_index=output_index, prompt_tokens=count, valid=grid is not None,
                                   solve_mode=mode, task_token_budget=args.task_token_budget,
                                   **prompt_details.get(label, {}),
                                   **{k: result[k] for k in ("error", "finish_reason", "usage", "elapsed_s",
                                                            "decode_tokens_per_s_estimate")})
                        if result.get("loop_detection") is not None:
                            row["loop_detection"] = result["loop_detection"]
                        if llama:
                            row.update({k: result[k] for k in ("stop_type", "think_closed", "streamed_tokens")},
                                       server_tokens_per_s=(result["timings"] or {}).get("predicted_per_second"))
                            if result.get("resumed"):
                                row["resumed"] = result["resumed"]
                        elif job_repl:
                            row.update({k: result[k] for k in ("stop_type", "think_closed", "streamed_tokens")})
                        if mode == "tir":
                            row.update(solve_mode="tir", tir_status=result["tir_status"], tir_rounds=result["tir_rounds"],
                                       solver_path=result["solver_paths"][output_index], usage_shared_task=key)
                            if "trajectory_stop_reason" in result:
                                row["trajectory_stop_reason"] = result["trajectory_stop_reason"]
                            output_folder.mkdir(exist_ok=queued_repl and trajectory_number > 0)
                            atomic_json(output_folder / "result.json", dict(result, final_grid=grid, test_index=output_index))
                        if grid is not None:
                            merged[key][output_index], row["selected_sources"] = merge_output(
                                primary[key][output_index], grid, primary_count=counts.get(output_label),
                                policy=args.merge_policy, alternative=alt)
                            raw_submission[key][output_index] = dict(zip(ATTEMPTS, deepcopy([grid, grid if alt is None else alt])))
                            # Commit the competition artifact before auxiliary stats.
                            atomic_json(args.final_submission, merged)
                            atomic_json(args.output_root / "qwen38_submission.json", raw_submission)
                        if labels is not None:
                            # The Qwen attempts as submitted: the top grid, and REPL-TIR's runner-up if any.
                            row["correct"] = grid is not None and labels[key][output_index] in (grid, alt)
                            if job_repl:
                                row["correct_top1"] = grid is not None and grid == labels[key][output_index]
                        atomic_json(output_folder / "score.json", row)
                        prior_row = next((i for i, old in enumerate(results) if queued_repl
                                          and (old["task"], old["test_index"]) == (key, output_index)), None)
                        if prior_row is None:
                            results.append(row)
                        else:
                            results[prior_row] = row  # Aggregate usage/candidates; never duplicate score rows.
                        stream_progress.pop(output_label, None)  # Same step as the append: no double count.
                        del active[output_label]
                        save()
                        if mode == "direct":
                            cleanup_completed_stream(output_folder, result,
                                                     keep=getattr(args, "keep_completed_streams", False))
                        print("DONE " + json.dumps(row), flush=True)
                        if live is not None:
                            print(live.result_line(row, results), flush=True)
                    if queued_repl:
                        trajectory_queue["completed"] += 1
                        if not result["cancelled"] and (
                                (cot_second and deadline-time.monotonic() > args.min_request_seconds)
                                or (not cot_second and trajectory_states[key].get("pending"))):
                            if cot_second:
                                # Independent CoT is eligible even if TIR exhausted
                                # its token or local time cap. Never run TIR again.
                                allowances = qwen38_repl_cot.allocate(cot_context, len(queries[key]["test"]))
                                cot_states[key] = dict(first=deepcopy(result), allowances=allowances, requests={})
                                for test_index, allowance in enumerate(allowances):
                                    if allowance > 0:
                                        queue.put_nowait((1, (priority, test_index),
                                            (key, test_index, 0, dict(max_tokens=allowance))))
                                        trajectory_queue["repeat_waiting"] += 1
                                atomic_json(args.output_root / "programs" / key / "cot_queue.json",
                                    dict(context=cot_context, allowances=allowances, mode="cot",
                                         order=priority, test_indices=list(range(len(allowances)))))
                            else:
                                queue.put_nowait((trajectory_number + 1, priority, job))
                                trajectory_queue["repeat_waiting"] += 1
                        if cot_second or not trajectory_states[key].get("pending") or result["cancelled"]:
                            trajectory_states.pop(key)
                            repl_prefixes.pop(label)
                        save()
                    busy_workers -= 1
                    queue_changed.set()
                    if result["cancelled"]:
                        break

            async def monitor_progress():
                nonlocal fatal
                next_live = time.monotonic() + args.live_interval_seconds
                with (args.output_root / "metrics.jsonl").open("x", buffering=1) as log:
                    while True:
                        if owned is not None and owned.poll() is not None:
                            fatal = True
                            state["error"] = f"Server exited: {owned.returncode}"
                            for task in workers:
                                task.cancel()
                            return
                        record = dict(utc=datetime.now(timezone.utc).isoformat())
                        metrics_text = None
                        try:
                            response = await client.get(url + "/metrics", timeout=5)
                            response.raise_for_status()
                            metrics_text = response.text
                            record["metrics"] = [l for l in metrics_text.splitlines()
                                if not l.startswith("#") and (l.startswith("llamacpp:") or any(k in l for k in
                                ("spec_decode", "num_requests_running", "num_requests_waiting",
                                 "kv_cache_usage", "preemptions", "generation_tokens")))]
                        except Exception as exc:
                            record["error"] = str(exc)
                        if live is not None and time.monotonic() >= next_live:
                            try:
                                record["live"] = await live.tick(results, stream_progress if llama else active, queue.qsize(),
                                                                 **({} if llama else {"metrics": metrics_text}))
                            except Exception as exc:  # Observability must never stop the run.
                                record["live_error"] = f"{type(exc).__name__}: {exc}"
                            next_live = time.monotonic() + args.live_interval_seconds
                        log.write(json.dumps(record) + "\n")
                        save()
                        await asyncio.sleep(min(15, max(0.001, next_live - time.monotonic())))

            if llama:
                prior = {} if resume is None else {
                    **{label: dsv4_llama.generated(row) or 0 for label, (row, _) in resume["kept"].items()},
                    **{label: prefix["tokens"] for label, prefix in prefixes.items()}}
                live = dsv4_llama.LiveView(args.output_root, len(planned),
                                           {key: len(queries[key]["test"]) for key in routed}, len(routed),
                                           labels is not None,
                                           title=dsv4_llama.profile_of(args)["release"] if dsv4 else
                                           "dots3-note llama.cpp" if dots3 else
                                           "Qwen3.8-Flash llama.cpp", prior=prior)
            else:
                live = qwen38_live.VllmLiveView(args.output_root, len(planned),
                    {key: len(queries[key]["test"]) for key in routed}, len(routed), labels is not None,
                    title=f"{model_name} vLLM", model=model_name)
                # Read a baseline before requests start: a reused server may
                # already have millions of generated tokens from earlier work.
                try:
                    response = await client.get(url + "/metrics", timeout=5)
                    response.raise_for_status()
                    live.measure(response.text)
                except Exception as exc:
                    print(f"LIVE metrics baseline unavailable (generation continues): {type(exc).__name__}", flush=True)
            workers = [asyncio.create_task(one_worker()) for _ in range(min(args.concurrency,
                trajectory_queue["planned"] if cot_second else len(jobs)))]
            monitor = asyncio.create_task(monitor_progress())
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
                if live is not None:  # Final totals: average throughput and accuracy of the whole run.
                    try:
                        metrics_text = None
                        if not llama:
                            try:
                                response = await client.get(url + "/metrics", timeout=5)
                                response.raise_for_status()
                                metrics_text = response.text
                            except Exception:
                                pass
                        await live.tick(results, stream_progress if llama else active, queue.qsize(),
                                        **({} if llama else {"metrics": metrics_text}))
                    except Exception as exc:
                        print(f"LIVE final summary failed: {type(exc).__name__}: {exc}", flush=True)
            state["status"] = ("server-failed-fallback" if fatal else
                               "deadline" if len(results) < len(planned) or time.monotonic() >= deadline else
                               "complete-with-fallbacks" if any(not r.get("valid") for r in results) else "complete")
            if args.require_qwen38_success and state["status"] != "complete":
                raise RuntimeError(f"Qwen3.8 did not complete every planned output: {state['status']}")
    except (asyncio.CancelledError, KeyboardInterrupt):
        state["status"] = "interrupted"
        raise
    except Exception as exc:
        state.update(status="failed-fallback", error=f"{type(exc).__name__}: {exc}")
        print(f"*** Qwen3.8 failure; durable primary/completed answers retained: {exc}", flush=True)
        if args.require_qwen38_success:
            raise
    finally:
        # This also preserves a schema-valid file after startup/stream errors.
        try:
            validate_submission(merged, queries)
            atomic_json(args.final_submission, merged)
            save()
        finally:
            if owned is not None:
                stop_owned_process(owned, grace=min(15, max(0, args.end_time-time.time())) if args.end_time else 15)
    print(f"*** Qwen3.8 {state['status']}: valid={state['valid_outputs']}/{len(planned)}; "
          f"submission={args.final_submission}; score={state['merged_score']}", flush=True)
    return state


def run_havu(args):
    """--dsv41-harness havu: Vu's B32 solver on our supervised dsv41 server (dsv41_havu)."""
    global FINAL_PATH
    validate_options(args)
    args.challenges = args.challenges or Path(resolve_challenge_path(os.getenv("KAGGLE_IS_COMPETITION_RERUN")))
    queries = json.loads(args.challenges.read_text())
    keys = select_task_keys(queries, task_keys=args.task_keys)
    labels = load_labels(args, queries)
    args.output_root.mkdir(parents=True, exist_ok=False)
    FINAL_PATH = args.final_submission
    deadline_epoch = deadline_from_limits(args.end_time, args.qwen38_max_seconds, args.finalization_reserve_seconds)
    summary = dsv41_havu.run(args, queries, keys, deadline_epoch, validate=lambda s: validate_submission(s, queries),
                             write=atomic_json, score=lambda s: evaluation_score(s, labels, keys))
    if args.require_qwen38_success and summary["status"] != "complete":
        raise RuntimeError(f"The havu harness did not complete: {summary['status']}")


def main():
    args = build_parser().parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    if args.dsv41_harness == "havu":
        return run_havu(args)
    asyncio.run(run_inference(args))


if __name__ == "__main__":
    run_entrypoint_with_status(main, path=extract_run_status_path(), phase="qwen38_pipeline",
                             success_status=STATUS_SUCCESS,
                             success_check=lambda: FINAL_PATH is not None and FINAL_PATH.is_file())
