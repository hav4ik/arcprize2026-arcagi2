"""Opt-in, label-blind Qwen3-4B pre-TTT routing from frozen-base probabilities.

No fitted predictor, historical outcomes, test answers, or task-ID rules. GPU
probes run in disposable subprocesses before any production workers start.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

from pipeline_runtime import atomic_json


MANIFEST_NAME = "early_routing.json"
RECIPE = "base-loo-v1: all train pairs; identity + rot90/palette/reverse; mean cell score"


def add_early_routing_arguments(parser):
    group = parser.add_argument_group("label-blind Qwen3-4B early routing (default off)")
    group.add_argument("--early-route-metric", choices=("off", "nll", "entropy"), default="off",
                       help="Skip 4B before TTT using mean frozen-base leave-one-out score per output cell.")
    cutoff = group.add_mutually_exclusive_group()
    cutoff.add_argument("--early-route-threshold", type=float,
                        help="Skip when score >= this fixed cutoff, in nats/cell. No learned calibration.")
    cutoff.add_argument("--early-route-fraction", type=float,
                        help="Instead use an unlabelled cohort quantile, e.g. .1 for its highest-risk tail. Cutoff ties stay in 4B.")
    group.add_argument("--early-route-gpus", type=int, default=0,
                       help="Probe workers: 0 = auto, up to 4 visible GPUs. Models exit before training.")
    group.add_argument("--early-route-batch-size", type=int, default=24,
                       help="Maximum active probe requests per GPU, batched across tasks (default: 24).")
    group.add_argument("--early-route-gpu-memory-utilization", type=float,
                       help="Probe GPU memory reservation; defaults to the production vLLM setting.")
    group.add_argument("--early-route-max-num-batched-tokens", type=int, default=16384,
                       help="Per-GPU probe token budget per scheduler step (default: 16384).")
    group.add_argument("--early-route-quantization", choices=("inherit", "none", "fp8"), default="inherit",
                       help="Probe-only precision; inherit uses the production vLLM setting. Does not change training/serving.")
    group.add_argument("--early-route-max-seconds", type=float, default=600,
                       help="Whole probe-phase budget including startup; also bounded by --end-time.")


def validate_early_routing_arguments(args):
    enabled = args.early_route_metric != "off"
    has_threshold = args.early_route_threshold is not None
    has_fraction = args.early_route_fraction is not None
    if enabled and has_threshold == has_fraction:
        raise ValueError("Early routing requires exactly one of --early-route-threshold / --early-route-fraction")
    if not enabled and (has_threshold or has_fraction):
        raise ValueError("A cutoff requires --early-route-metric nll or entropy")
    if has_threshold and (not math.isfinite(args.early_route_threshold) or args.early_route_threshold < 0):
        raise ValueError("--early-route-threshold must be finite and nonnegative")
    if has_fraction and (not math.isfinite(args.early_route_fraction) or not 0 < args.early_route_fraction < 1):
        raise ValueError("--early-route-fraction must be strictly between 0 and 1")
    if not 0 <= args.early_route_gpus <= 4 or args.early_route_batch_size < 1:
        raise ValueError("Early routing requires 0..4 GPUs and a positive batch size")
    if args.early_route_max_num_batched_tokens < args.early_route_batch_size:
        raise ValueError("--early-route-max-num-batched-tokens must be at least --early-route-batch-size")
    memory = args.early_route_gpu_memory_utilization
    if memory is not None and (not math.isfinite(memory) or not 0 < memory <= 1):
        raise ValueError("--early-route-gpu-memory-utilization must be in (0, 1]")
    if not math.isfinite(args.early_route_max_seconds) or args.early_route_max_seconds <= 0:
        raise ValueError("--early-route-max-seconds must be positive and finite")


def early_routing_starter_args(args, starter_args):
    """Accept flags above OR below the cascade's passthrough marker, not both."""
    validate_early_routing_arguments(args)
    result = list(starter_args)
    if args.early_route_metric == "off":
        return result
    if any(arg.split("=", 1)[0].startswith("--early-route-") for arg in result):
        raise ValueError("Specify early-routing flags either before or after --stage1-starter-args, not both")
    for name in ("metric", "threshold", "fraction", "gpus", "batch-size", "gpu-memory-utilization",
                 "max-num-batched-tokens", "quantization", "max-seconds"):
        value = getattr(args, "early_route_" + name.replace("-", "_"))
        if value is not None:
            result.extend(("--early-route-" + name, str(value)))
    return result


def resolve_probe_runtime(args):
    """Resolve probe overrides without changing the production configuration."""
    memory = args.early_route_gpu_memory_utilization
    if memory is None:
        memory = getattr(args, "vllm_gpu_memory_utilization", None)
    if memory is None:
        memory = float(os.environ.get("ARC_VLLM_GPU_MEMORY_UTILIZATION", "0.89"))
    if not math.isfinite(memory) or not 0 < memory <= 1:
        raise ValueError("Inherited early-routing GPU memory utilization must be in (0, 1]")
    quantization = args.early_route_quantization
    if quantization == "inherit":
        quantization = (getattr(args, "vllm_quantization", None)
                        or os.environ.get("ARC_VLLM_QUANTIZATION", "none"))
    if quantization not in ("none", "fp8"):
        raise ValueError(f"Unsupported early-routing quantization: {quantization!r}")
    return dict(batch_size=args.early_route_batch_size, gpu_memory_utilization=memory,
                max_num_batched_tokens=args.early_route_max_num_batched_tokens,
                quantization=quantization)


def visible_task(task):
    return {"train": [{"input": p["input"], "output": p["output"]} for p in task["train"]],
            "test": [{"input": p["input"]} for p in task["test"]]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _grid_text(grid):
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Empty or ragged ARC grid")
    if any(type(c) is not int or not 0 <= c <= 9 for row in grid for c in row):
        raise ValueError("ARC cells must be integer colors 0..9")
    return "\n".join("".join(map(str, row)) for row in grid)


def _transform(grid, view):
    _grid_text(grid)  # Validate before zip can truncate a ragged grid.
    if view == 0:
        return [list(row) for row in grid]
    return [[(c + 1) % 10 for c in row] for row in zip(*grid)][::-1]


def probe_jobs(task):
    """Only known training answers are targets; never a real test output."""
    train = task["train"]
    if len(train) < 2:
        return []

    def query(grid):
        return "<|im_start|>user\n" + _grid_text(grid) + "<|im_end|><|im_start|>assistant\n"

    jobs = []
    for held_out in range(len(train)):
        for view in (0, 1):
            pairs = [{k: _transform(p[k], view) for k in ("input", "output")} for p in train]
            demos = [p for i, p in enumerate(pairs) if i != held_out]
            if view:
                demos.reverse()
            held = pairs[held_out]
            prompt = "".join(query(p["input"]) + _grid_text(p["output"]) + "<|im_end|>" for p in demos)
            jobs.append(dict(held_out=held_out, view=view, query=prompt + query(held["input"]),
                             answer=_grid_text(held["output"]) + "<|im_end|>", output_grid=held["output"]))
    return jobs


def encode_job(job, tokenizer):
    q = tokenizer.encode(job["query"], add_special_tokens=False)
    a = tokenizer.encode(job["answer"], add_special_tokens=False)
    if not q or tokenizer.encode(job["query"] + job["answer"], add_special_tokens=False) != q + a:
        raise ValueError("Unstable query/answer token boundary")
    expected = []
    for i, row in enumerate(job["output_grid"]):
        if i:
            expected.append(10)
        expected.extend(row)
    if a != expected + [15]:
        raise ValueError("Early routing requires ARC digit/newline/EOS token IDs 0..9/10/15")
    return q, a


def score_cells(answer, distributions, output_grid):
    """Raw full-vocabulary entropy/NLL at digit positions, in natural-log units.

    Row separators and EOS are checked but excluded from cell means. Do not
    renormalize over legal grid tokens: that would change the measured score.
    """
    if len(answer) != len(distributions):
        raise ValueError("Incomplete answer logprobs")
    cells = [c for row in output_grid for c in row]
    expected = []
    for i, row in enumerate(output_grid):
        if i:
            expected.append(10)
        expected.extend(row)
    if list(answer) != expected + [15] or not cells:
        raise ValueError("Answer/grid alignment mismatch")
    nll, entropy = [], []
    for token, lp in zip(answer, distributions):
        if set(lp) != set(range(16)) or not all(math.isfinite(v) for v in lp.values()):
            raise ValueError("Incomplete/nonfinite full-vocabulary logprobs")
        if max(lp.values()) > .005:
            raise ValueError("Invalid logprob normalization")
        probabilities = [math.exp(v) for v in lp.values()]
        if abs(sum(probabilities) - 1) > .005:
            raise ValueError("Invalid logprob normalization")
        if token < 10:
            nll.append(-lp[token])
            entropy.append(-sum(p * v for p, v in zip(probabilities, lp.values())))
    return dict(nll=statistics.fmean(nll), entropy=statistics.fmean(entropy), cells=len(cells))


def decide_routes(task_keys, reports, *, metric, threshold=None, fraction=None):
    """Raw cutoff only. Failed/incomplete probes always retain the 4B path."""
    scores = {}
    for key in task_keys:
        report = reports.get(key, {})
        values = report.get("scores")
        score = values.get(metric) if isinstance(values, dict) else None
        if report.get("status") == "complete" and isinstance(score, (int, float)) and math.isfinite(score):
            scores[key] = float(score)
    cutoff = threshold
    # Do not estimate a cohort quantile from only the easy/completed tasks.
    quantile_incomplete = fraction is not None and len(scores) != len(task_keys)
    if fraction is not None and scores and not quantile_incomplete:
        values = sorted(scores.values())
        index = (len(values) - 1) * (1 - fraction)
        lo, hi = math.floor(index), math.ceil(index)
        cutoff = values[lo] + (values[hi] - values[lo]) * (index - lo)
    decisions = {}
    for key in task_keys:
        score = scores.get(key)
        skip = (score is not None and cutoff is not None and not quantile_incomplete
                and (score > cutoff if fraction is not None else score >= cutoff))
        decisions[key] = dict(skip=skip, score=score, reason=(
            "incomplete-cohort-retain" if quantile_incomplete else
            "probe-unavailable-retain" if score is None else
            "high-base-score" if skip else "below-cutoff-or-tie"))
    return cutoff, decisions


def validate_probe_model(model_path):
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    expected = dict(model_type="qwen3", hidden_size=2560, num_hidden_layers=36, vocab_size=16)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Early routing currently supports only Qwen3-4B ARC vocab16 checkpoints")
    if (root / "arc_pissa_manifest.json").exists() or config.get("arc_pissa_shared_base"):
        raise ValueError("Early routing needs the original base, not a PiSSA shared residual base")
    if (root / "arc_rope_3d.json").exists() or config.get("arc_rope_3d"):
        raise ValueError("Early-routing probes have not been validated with 3D RoPE")
    if config.get("quantization_config"):
        raise ValueError("Early routing expects an unquantized checkpoint; use --early-route-quantization fp8 for online quantization")
    return config


def read_early_routing_manifest(output_dir, tasks, selected_keys):
    """Ignore test labels in identity checks, but never apply a stale skip list."""
    path = Path(output_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported early-routing manifest: {path}")
    if not manifest.get("enabled"):
        return None
    if manifest.get("status") != "complete":
        raise ValueError(f"Early routing is not finalized: {path}; finish/restart starter.py before submission")
    fingerprints = manifest.get("task_sha256", {})
    for key in selected_keys:
        # A later submission may select only a subset, never an unrelated set.
        if fingerprints.get(key) != digest(visible_task(tasks[key])):
            raise ValueError(f"Early-routing task identity mismatch for {key}; use the matching challenge/output directory")
    skipped = manifest.get("skipped_task_keys", [])
    decisions = manifest.get("decisions", {})
    if (set(skipped) - fingerprints.keys()
            or set(skipped) != {k for k, d in decisions.items() if d.get("skip")}):
        raise ValueError(f"Inconsistent early-routing skip list: {path}")
    return manifest


def early_routed_task_keys(output_dir, tasks, selected_keys):
    manifest = read_early_routing_manifest(output_dir, tasks, selected_keys)
    return set(manifest["skipped_task_keys"]).intersection(selected_keys) if manifest else set()


def run_early_routing(args, tasks, task_keys, model_path, output_dir, *, gpu_count=None, runtime_specs=None,
                      identity_task_keys=None):
    """Return retained keys; preserve the original selection in a durable manifest.

    No global model/environment changes, no production candidates written here.
    Runtime failures retain tasks; unsupported configurations fail explicitly.
    """
    validate_early_routing_arguments(args)
    root = Path(output_dir)
    active_path = root / MANIFEST_NAME
    if args.early_route_metric == "off":
        if active_path.exists():
            atomic_json(active_path, dict(schema_version=1, enabled=False, status="disabled"))
        return list(task_keys)
    validate_probe_model(model_path)
    if getattr(args, "bidirectional_prompt_attention", False):
        raise ValueError("Early-routing probes currently require causal Qwen3 attention")
    if getattr(args, "fill_train_to_tokens", None):
        raise ValueError("Early routing cannot be combined with synthetic --fill-train-to-tokens")
    if getattr(args, "pissa_base_mode", "original") == "shared-residual":
        raise ValueError("Early routing requires the original base, not PiSSA shared-residual mode")
    runtime = resolve_probe_runtime(args)
    started = time.monotonic()
    visible = {k: visible_task(tasks[k]) for k in task_keys}
    report_dir = root / "early_routing_runs" / uuid.uuid4().hex
    report_dir.mkdir(parents=True, exist_ok=False)
    input_path = report_dir / "visible_tasks.json"
    # atomic_json sorts dictionaries; carry queue order explicitly for sharding.
    atomic_json(input_path, dict(task_keys=list(task_keys), tasks=visible))
    manifest = dict(schema_version=1, enabled=True, status="probing", recipe=RECIPE,
                    metric=args.early_route_metric, threshold=args.early_route_threshold,
                    fraction=args.early_route_fraction, quantization=runtime["quantization"],
                    probe_runtime=runtime,
                    model_path=str(Path(model_path).resolve()), report_dir=str(report_dir.resolve()),
                    selected_task_keys=list(identity_task_keys or task_keys), probed_task_keys=list(task_keys),
                    task_sha256={k: digest(visible_task(tasks[k])) for k in (identity_task_keys or task_keys)},
                    skipped_task_keys=[], decisions={})
    atomic_json(active_path, manifest)
    atomic_json(report_dir / "manifest.json", manifest)
    if gpu_count is None:
        import torch
        gpu_count = torch.cuda.device_count()
    gpu_tokens = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_tokens = [s.strip() for s in gpu_tokens.split(",") if s.strip()] if gpu_tokens is not None else [str(i) for i in range(gpu_count)]
    workers = min(args.early_route_gpus or 4, gpu_count, len(gpu_tokens), len(task_keys))
    limit = started + args.early_route_max_seconds
    if args.end_time:
        # Leave time for owned EngineCore cleanup and publishing the decisions
        # before a cascade parent enforces this same stage deadline.
        limit = min(limit, started + max(0, args.end_time - time.time() - 5))
    processes, reports, failures = [], {}, []
    from pipeline_runtime_launcher import stop_processes

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Early routing interrupted by signal {signum}")

    old_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        with ExitStack() as stack:
            for shard in range(workers):
                if time.monotonic() >= limit:
                    break
                environment = dict(os.environ)
                python = sys.executable
                if runtime_specs:
                    from pipeline_runtime import build_runtime_environment
                    spec = runtime_specs["vllm"]
                    python = spec.python
                    environment = build_runtime_environment(spec, "vllm", other_specs=runtime_specs.values())
                environment.update(CUDA_VISIBLE_DEVICES=gpu_tokens[shard], OMP_NUM_THREADS="1",
                                   VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_BATCH_INVARIANT="0")
                for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                    environment.pop(name, None)
                log_path = report_dir / f"shard_{shard}.log"
                command = [python, "-u", str(Path(__file__).with_name("early_routing_worker.py")),
                           "--input", str(input_path.resolve()), "--report", str((report_dir / f"shard_{shard}.json").resolve()),
                           "--model-path", str(Path(model_path).resolve()), "--shard", str(shard), "--shards", str(workers),
                           "--batch-size", str(runtime["batch_size"]), "--max-model-len", str(args.max_seq_length),
                           "--gpu-memory-utilization", str(runtime["gpu_memory_utilization"]),
                           "--max-num-batched-tokens", str(runtime["max_num_batched_tokens"]),
                           "--quantization", runtime["quantization"]]
                print(f"[early routing] base probe GPU={gpu_tokens[shard]}; cross-task queue; "
                      f"max_num_seqs={runtime['batch_size']}, "
                      f"max_num_batched_tokens={runtime['max_num_batched_tokens']}, "
                      f"gpu_memory_utilization={runtime['gpu_memory_utilization']}, "
                      f"quantization={runtime['quantization']}; log={log_path}", flush=True)
                log = stack.enter_context(log_path.open("xb"))
                try:
                    processes.append(subprocess.Popen(command, env=environment, stdout=log,
                                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                      start_new_session=True))
                except OSError as error:
                    failures.append(f"shard {shard}: {error}")
            last_progress = time.monotonic()
            while any(p.poll() is None for p in processes) and time.monotonic() < limit:
                if time.monotonic() - last_progress >= 30:
                    print(f"[early routing] probing; elapsed={time.monotonic()-started:.0f}s; reports={report_dir}", flush=True)
                    last_progress = time.monotonic()
                time.sleep(min(.2, max(0, limit - time.monotonic())))
    finally:
        try:
            stop_processes(processes, grace=3)
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
    for shard in range(workers):
        try:
            payload = json.loads((report_dir / f"shard_{shard}.json").read_text())
            if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
                raise ValueError("Invalid probe report structure")
            for key in task_keys[shard::workers]:
                row = payload.get("tasks", {}).get(key, {})
                if isinstance(row, dict) and row.get("visible_task_sha256") == manifest["task_sha256"][key]:
                    reports[key] = row
            if payload.get("error"):
                failures.append(f"shard {shard}: {payload['error']}")
        except (OSError, ValueError, TypeError) as error:
            failures.append(f"shard {shard}: no usable report ({error})")
    cutoff, decisions = decide_routes(task_keys, reports, metric=args.early_route_metric,
                                     threshold=args.early_route_threshold, fraction=args.early_route_fraction)
    skipped = [k for k in task_keys if decisions[k]["skip"]]
    manifest.update(status="complete", cutoff=cutoff, skipped_task_keys=skipped, decisions=decisions,
                    probe_status={k: reports.get(k, {}).get("status", "unavailable") for k in task_keys},
                    failures=failures, probe_workers=workers, seconds=time.monotonic() - started)
    atomic_json(report_dir / "manifest.json", manifest)
    atomic_json(active_path, manifest)
    for message in failures:
        print(f"[early routing] warning: {message}", flush=True)
    for key in task_keys:
        d = decisions[key]
        print(f"[early routing] {key}: {args.early_route_metric}={d['score']}; "
              f"{'SKIP 4B' if d['skip'] else 'keep 4B'} ({d['reason']})", flush=True)
    print(f"[early routing] skipped {len(skipped)}/{len(task_keys)}; cutoff={cutoff}; "
          f"probe={manifest['seconds']:.1f}s; manifest={active_path}", flush=True)
    skipped_set = set(skipped)
    return [k for k in task_keys if k not in skipped_set]
