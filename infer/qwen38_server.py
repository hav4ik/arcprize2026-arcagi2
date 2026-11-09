"""Offline Qwen serving: Qwen3.8 NVFP4 or Qwen3.5-9B FP8/DFlash.

Promoted from the validated L4 experiment; no runtime downloads. Draft weights
default to checkpoint-declared precision; online FP8 is opt-in. ``launch`` returns
the owned process for pipeline cleanup.
"""

import argparse
import codecs
from copy import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import gpu_pair_bench
import qwen38_vision
import qwen38_flash_profile as flash_profile


TARGET = "/kaggle/input/qwen3-8-27b-nvfp4"
DRAFT = "/kaggle/input/models/dangkhoa2016/z-lab-qwen3-8-27b-dflash2/transformers/default/1"
FLASH_NEXT = "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


class StartupLogTail:
    """Mirror the owned server's durable log while its health check is pending.

    The child still writes straight to disk, not to a pipe that can fill up.
    Polling is bounded; the final read drains a fixed snapshot, including any
    traceback written just before an exit, timeout or cancellation.
    """

    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self.file = None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.disabled = False

    def __enter__(self):
        return self

    def emit(self, *, final=False):
        if self.path is None or self.disabled:
            return
        try:
            if self.file is None:
                try:
                    self.file = self.path.open("rb")
                except FileNotFoundError:
                    return  # The launcher may not have created the log yet.
            remaining = (max(0, os.fstat(self.file.fileno()).st_size - self.file.tell())
                         if final else 65536)
            while remaining:
                chunk = self.file.read(min(65536, remaining))
                if not chunk:
                    break
                sys.stdout.write(self.decoder.decode(chunk))
                remaining -= len(chunk)
            if final:
                sys.stdout.write(self.decoder.decode(b"", final=True))
            sys.stdout.flush()
        except OSError:
            # A notebook output/file error must not interrupt model startup.
            # The server keeps its independent durable log either way.
            self.disabled = True

    def __exit__(self, *unused):
        try:
            self.emit(final=True)
        finally:
            if self.file is not None:
                self.file.close()
            self.disabled = True


def served_model_name(variant):
    return {"27b": "qwen38-27b", "flash-next": "qwen38-flash-next", "qwen35-9b": "qwen35-9b"}[variant]


def json_object(value):
    try:
        result = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Expected a JSON object") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return result


def add_backend_arguments(p, variant_flag="--variant", variants=("27b", "flash-next", "qwen35-9b")):
    flash_profile.add_arguments(p)
    # The pipeline also offers "dsv4", which dsv4_llama serves with llama.cpp instead of vLLM.
    p.add_argument(variant_flag, dest="qwen38_variant", choices=variants, default="27b")
    p.add_argument("--tensor-parallel-size", "--tp", type=int, default=4,
                   help="vLLM target and DFlash TP; use 1 for the single-GPU benchmark. Default remains 4.")
    p.add_argument("--gpu-ids", default=None,
                   help="vLLM CUDA device indices, one per TP rank; defaults to 0 through TP-1. auto-bench picks "
                        "them from a GPU link bench run once per server (gpu_pair_bench.py).")
    p.add_argument("--target-quantization", choices=("auto", "fp8", "fp8_per_tensor"), default="auto",
                   help="Qwen3.5-9B only: auto selects online fp8_per_tensor; fp8 loads serialized FP8. "
                        "Existing Qwen3.8 profiles retain checkpoint-declared quantization.")
    p.add_argument("--draft-quantization", choices=("auto", "none", "nvfp4", "fp8", "fp8_per_tensor"), default="auto",
                   help="Dense DFlash/DFlash2 weights only; separate from FP8 KV. auto preserves checkpoint precision "
                        "(including ModelOpt/compressed-tensors NVFP4); none is a legacy alias for auto. "
                        "nvfp4 additionally requires an NVFP4 checkpoint, not online conversion. "
                        "fp8_per_tensor enables online FP8; fp8 supports older vLLM or serialized FP8.")
    p.add_argument("--draft-rejection-method", choices=("standard", "block"), default="standard",
                   help="DFlash draft verification. block verifies the drafts jointly (Sun et al.): still "
                        "lossless, and accepts at least as many tokens as per-token rejection.")
    p.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"), default="greedy",
                   help="DFlash draft sampling. probabilistic samples the drafts from the drafter's distribution "
                        "and verifies against it, which can accept more under sampled target decoding.")
    p.add_argument("--load-format", choices=("auto", "safetensors", "fastsafetensors"), default="auto",
                   help="fastsafetensors uses its native loader instead of multithread safetensors loading; requires the installed offline package.")
    p.add_argument("--fastsafetensors-max-gpu-shard-gib", type=float, default=0,
                   help="Opt-in fast loader guard: files larger than this use CPU safetensors loading. 0 disables; no change to inference-time parameter placement.")
    p.add_argument("--draft-stage-dir", type=Path, default=None,
                   help="Opt-in verified CPU copy of the draft checkpoint before CUDA startup. "
                        "Use a fresh local destination (e.g. /dev/shm/qwen38_draft); avoids mounted-file "
                        "fastsafetensors stalls. Preserves all bytes and keeps the copy for reuse.")
    p.add_argument("--mtp-tokens", type=int, default=3, help="Flash-Next built-in MTP tokens; 0 disables.")
    p.add_argument("--mtp-moe-backend", default="triton", help="Flash-Next's unquantized MTP experts.")
    p.add_argument("--moe-backend", default="marlin", help="Flash-Next NVFP4 experts on L4.")
    p.add_argument("--expert-parallel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-eplb", action=argparse.BooleanOptionalAction, default=False,
                   help="Opt-in native Flash-Next expert load balancing across GPU ranks; not hot/cold CPU expert caching.")
    p.add_argument("--eplb-config", type=json_object, default=None,
                   help="JSON object passed to native vLLM with --enable-eplb; installed vLLM validates supported fields.")
    p.add_argument("--ple-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu-offload-gb", type=float, default=0, help="Selective weight offload GiB per GPU, separate from PLE/KV.")
    p.add_argument("--cpu-offload-params", nargs="+", default=["w2_weight"])
    p.add_argument("--kv-offload-gib-per-rank", type=float, default=0,
                   help="CPU KV cache per TP rank; 2 means 8 GiB total on four GPUs. Not additional resident GPU KV.")
    p.add_argument("--kv-offload-mode", choices=("eager", "lazy"), default="eager",
                   help="Native SimpleCPUOffload policy: eager saves computed blocks; lazy saves cached GPU blocks near eviction. Requires a positive KV offload budget; eager remains the default.")
    p.add_argument("--rust-frontend", action=argparse.BooleanOptionalAction, default=False,
                   help="Opt-in installed vllm-rs frontend; requires compatible /tokenize and streaming APIs.")
    p.add_argument("--scale-out-endpoints", action=argparse.BooleanOptionalAction, default=False,
                   help="Register vLLM's token-in/token-out routes (/inference/v1/generate, /v1/*/render); the "
                        "multimodal REPL prompt ledger sends its exact ids there.")
    p.add_argument("--prompt-tokens-details", action=argparse.BooleanOptionalAction, default=False,
                   help="Report usage.prompt_tokens_details.cached_tokens, the prefix cache hit of each request.")
    p.add_argument("--vllm-arg", action="append", default=[],
                   help="Extra raw `vllm serve` argument appended last; repeat it and write --vllm-arg=--flag=value.")


def validate_backend_options(args):
    flash_profile.validate(args)
    if getattr(args, "draft_stage_dir", None) is not None:
        if args.qwen38_variant not in ("27b", "qwen35-9b") or not args.draft_tokens:
            raise ValueError("--draft-stage-dir requires an enabled DFlash draft model")
    if args.draft_quantization not in ("auto", "none") and args.qwen38_variant not in ("27b", "qwen35-9b"):
        raise ValueError("--draft-quantization requires the 27b DFlash2 or qwen35-9b DFlash profile")
    if args.target_quantization != "auto" and args.qwen38_variant != "qwen35-9b":
        raise ValueError("--target-quantization is only supported by the qwen35-9b profile")
    selected_gpu_ids(args, bench=False)
    if args.mtp_tokens < 0:
        raise ValueError("MTP tokens must be non-negative")
    if args.enable_eplb and (args.qwen38_variant != "flash-next" or not args.expert_parallel):
        raise ValueError("EPLB requires the Flash-Next profile and expert parallel")
    if args.eplb_config is not None and not args.enable_eplb:
        raise ValueError("--eplb-config requires --enable-eplb")
    for field in ("cpu_offload_gb", "kv_offload_gib_per_rank", "fastsafetensors_max_gpu_shard_gib"):
        value = getattr(args, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    if args.cpu_offload_gb and (not args.cpu_offload_params or not all(p.strip() for p in args.cpu_offload_params)):
        raise ValueError("Selective offload requires non-empty parameter names")
    if 0 < args.kv_offload_gib_per_rank * 1024**3 < 1:
        raise ValueError("CPU KV offload must allocate at least one byte per rank")
    if args.kv_offload_mode == "lazy" and not args.kv_offload_gib_per_rank:
        raise ValueError("Lazy KV offload requires --kv-offload-gib-per-rank > 0")
    if args.fastsafetensors_max_gpu_shard_gib:
        if args.load_format != "fastsafetensors":
            raise ValueError("Fast shard guard requires --load-format fastsafetensors")
        if args.fastsafetensors_max_gpu_shard_gib * 1024**3 < 1:
            raise ValueError("Fast GPU shard cap must be at least one byte")


def validate_flash_next(config, context, tp):
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("Flash-Next profile requires a qwen4_exp checkpoint; use --qwen38-variant 27b for dense 27B")
    text = config["text_config"]
    if not text.get("ple_layer_ids") or not text.get("num_experts") or not text.get("mtp_num_hidden_layers"):
        raise ValueError("Flash-Next checkpoint must declare PLE, experts and built-in MTP")
    if context > text["max_position_embeddings"]:
        raise ValueError("Requested context exceeds checkpoint positional limit")
    # GQA heads can be replicated when TP exceeds the KV-head count.
    kv_heads = text["num_key_value_heads"]
    if text["num_attention_heads"] % tp or (kv_heads % tp if kv_heads >= tp else tp % kv_heads):
        raise ValueError("Requested TP is incompatible with attention head counts")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "modelopt" or quant.get("quant_algo") not in ("NVFP4", "MIXED_PRECISION"):
        raise ValueError("Flash-Next profile expects checkpoint-declared ModelOpt NVFP4 weights")
    check_nvfp4_quantization(quant)


def stop_owned_process(process, grace=15):
    """Only a group created by our Popen(start_new_session=True) is touched."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # The parent can exit before CUDA children. Reap the owned group as well.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def read_checkpoint(path, *, allow_flattened=False):
    path = Path(path).resolve()
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    index = path / "model.safetensors.index.json"
    names = (set(json.loads(index.read_text())["weight_map"].values())
             if index.exists() else {p.name for p in path.glob("*.safetensors")})
    if not names:
        raise ValueError(f"No safetensors weights found in {path}")
    shards, flattened = flash_profile.resolve_shards(path, names, allow_flattened=allow_flattened)
    weights = {name: weight.stat().st_size for name, weight in shards.items()}
    return {"path": str(path), "config": config, "weights_bytes": weights,
            "resolved_shards": {name: str(weight) for name, weight in shards.items()},
            "flattened_shards": flattened,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}


def draft_quantization_plan(args):
    """Inspect local draft metadata only; let vLLM select its native quant loader.

    No weight reads, package imports, downloads or checkpoint edits. Missing
    config is allowed for command-only previews; launch still validates the
    complete checkpoint. ``nvfp4`` is a checkpoint assertion, not a vLLM online
    quantization name. Never inherit the target's quantization for DFlash.
    """
    requested = args.draft_quantization
    plan = {"requested": requested,
            "active": args.qwen38_variant in ("27b", "qwen35-9b") and args.draft_tokens > 0,
            "checkpoint_quant_method": None, "checkpoint_has_nvfp4": False,
            "vllm_quantization": None, "quantized_context": False, "metadata_sha256": {}}
    if not plan["active"]:
        return plan

    def read_json(path):
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Draft metadata must contain a JSON object: {path}")
        plan["metadata_sha256"][path.name] = hashlib.sha256(raw).hexdigest()
        return value

    config_path = args.draft_model / "config.json"
    config = read_json(config_path) if config_path.is_file() else {}
    quant = config.get("quantization_config")
    sidecar = args.draft_model / "hf_quant_config.json"
    # vLLM imports legacy ModelOpt sidecars into hf_config before inspecting
    # compression_config/text_config. Newer exports embed the quant config.
    if quant is None and sidecar.is_file():
        quant = read_json(sidecar)
    if quant is None:
        quant = config.get("compression_config")
    if quant is None:
        text = config.get("text_config") or {}
        quant = text.get("quantization_config", text.get("compression_config"))
    if quant is not None and not isinstance(quant, dict):
        raise ValueError("Draft quantization metadata must be a JSON object")
    quant = quant or {}
    method = str(quant.get("quant_method", "")).lower().replace("compressed_tensors", "compressed-tensors")
    recipe = quant.get("quantization", quant)
    if not isinstance(recipe, dict):
        raise ValueError("Draft ModelOpt quantization recipe must be a JSON object")
    if not method and (quant.get("producer", {}).get("name") == "modelopt"
                       or "modelopt_quant_config" in recipe):
        method = "modelopt"
    # Older MIXED_PRECISION exports keep the layer map only in the sidecar.
    if (method.startswith("modelopt") and recipe.get("quant_algo") == "MIXED_PRECISION"
            and "quantized_layers" not in recipe and sidecar.is_file()):
        extra = read_json(sidecar)
        recipe = extra.get("quantization", extra)
    if method.startswith("modelopt"):
        algos = {str(recipe.get("quant_algo", "")).upper()} | {
            str(layer.get("quant_algo", "")).upper()
            for layer in recipe.get("quantized_layers", {}).values()}
        has_nvfp4 = bool(algos & {"NVFP4", "W4A16_NVFP4"})
    elif method == "compressed-tensors":
        groups = list(quant.get("config_groups", {}).values())
        formats = {quant.get("format")} | {
            group.get("format") for group in groups}
        # Mixed exports may only identify NVFP4 through their weight scheme.
        # Match vLLM's tensor_group/16/float4 scheme, not MXFP4 or integer INT4.
        has_nvfp4 = "nvfp4-pack-quantized" in formats or any(
            w.get("num_bits") == 4 and w.get("type") == "float"
            and w.get("strategy") == "tensor_group" and w.get("group_size") == 16
            and w.get("symmetric", True)
            for group in groups if (w := group.get("weights")) is not None)
    else:
        has_nvfp4 = False

    if has_nvfp4 and requested in ("fp8", "fp8_per_tensor"):
        raise ValueError(f"Draft checkpoint declares NVFP4 ({method}), but --draft-quantization {requested} "
                         "requests FP8. Use --draft-quantization auto (or nvfp4); packed NVFP4 weights "
                         "must use their checkpoint-declared quantization.")
    if requested == "nvfp4" and not has_nvfp4:
        raise ValueError("--draft-quantization nvfp4 requires a checkpoint declaring ModelOpt or "
                         "compressed-tensors NVFP4. It does not convert BF16 weights to NVFP4.")
    override = requested if requested in ("fp8", "fp8_per_tensor") else None
    plan.update(checkpoint_quant_method=method or None, checkpoint_has_nvfp4=has_nvfp4,
                vllm_quantization=override, quantized_context=bool(quant) or override is not None)
    return plan


def validate_pair(target, draft, context, tp, *, variant="27b", target_quantization="auto", draft_tokens=None):
    small = variant == "qwen35-9b"
    if target.get("model_type") not in (("qwen3_5", "qwen3_5_text") if small else ("qwen3_5",)):
        raise ValueError("Expected a dense qwen3_5 target")
    allowed = {"DFlashDraftModel", "DFlash2DraftModel"} if small else {"DFlash2DraftModel"}
    if not allowed.intersection(draft.get("architectures", [])):
        raise ValueError(f"Expected a compatible draft architecture: {sorted(allowed)}")
    text = target.get("text_config", target)
    if small and (text.get("num_hidden_layers"), text.get("hidden_size")) != (32, 4096):
        raise ValueError("Qwen3.5-9B profile expects 32 text layers and hidden size 4096")
    for field in ("hidden_size", "vocab_size"):
        if text[field] != draft[field]:
            raise ValueError(f"Target/draft {field} mismatch")
    if draft.get("num_target_layers") != text["num_hidden_layers"]:
        raise ValueError("Draft target-layer count does not match target")
    layers = draft["dflash_config"]["target_layer_ids"]
    if not layers or any(i < 0 or i >= text["num_hidden_layers"] for i in layers):
        raise ValueError("Invalid DFlash2 target hidden-state taps")
    for config in (text, draft):
        if context > config["max_position_embeddings"]:
            raise ValueError("Requested context exceeds checkpoint positional limit")
        if any(config[field] % tp for field in ("num_attention_heads", "num_key_value_heads")):
            raise ValueError("Requested TP does not divide attention head counts")
    quant = target.get("quantization_config", {})
    if small:
        method = "fp8_per_tensor" if target_quantization == "auto" else target_quantization
        if quant and quant.get("quant_method") != method:
            raise ValueError(f"Target checkpoint quantization {quant.get('quant_method')!r} conflicts with {method!r}; "
                             "use --target-quantization fp8 for a serialized FP8 checkpoint, not FP4/NVFP4 weights")
        block = draft["dflash_config"].get("block_size", draft.get("block_size"))
        if draft_tokens is not None and block is not None and draft_tokens + 1 > block:
            raise ValueError(f"Draft tokens + anchor ({draft_tokens + 1}) exceed draft block size {block}")
        return
    check_nvfp4_quantization(quant)


def check_nvfp4_quantization(quant):
    """A compressed-tensors or ModelOpt NVFP4 export; vLLM loads either directly."""
    if quant.get("quant_method") == "modelopt":
        layers = quant.get("quantized_layers", {})
        if "NVFP4" not in {quant.get("quant_algo")} | {v.get("quant_algo") for v in layers.values()}:
            raise ValueError("No NVFP4 layers found in the ModelOpt target checkpoint")
        # The DFlash2 draft-vocab head reads an FP8 or BF16 lm_head, not NVFP4.
        if layers.get("lm_head", {}).get("quant_algo") == "NVFP4":
            raise ValueError("ModelOpt NVFP4 lm_head is unsupported; export it as FP8")
        return
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("Expected checkpoint-declared compressed-tensors or ModelOpt quantization")
    if not any(g.get("format") == "nvfp4-pack-quantized"
               for g in quant.get("config_groups", {}).values()):
        raise ValueError("No NVFP4 group found in target checkpoint")


def parser():
    p = flash_profile.ProfileParser(description=__doc__)
    qwen38_vision.add_arguments(p)
    add_backend_arguments(p)
    p.add_argument("--model", type=Path, default=Path(TARGET))
    p.add_argument("--draft-model", type=Path, default=Path(DRAFT))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--temp-root", type=Path, default=Path("/kaggle/working/q27_tmp"))
    p.add_argument("--context", type=int, default=81920)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--draft-tokens", type=int, default=4, help="0 serves the target alone, without the drafter")
    p.add_argument("--port", type=int, default=18083)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--model-loader-threads", type=int, default=32)
    p.add_argument("--eager", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--enable-auto-tool-choice", action="store_true")
    p.add_argument("--tool-call-parser", default="qwen3_coder")
    p.add_argument("--launch", action="store_true")
    add_supervise_arguments(p)
    return p


def command_for(args, *, draft_plan=None):
    qwen38_vision.validate_options(args)
    validate_backend_options(args)
    if min(args.context, args.batch, args.model_loader_threads) < 1 or args.draft_tokens < 0:
        raise ValueError("Context, batch, loader threads must be positive and draft tokens not negative")
    if not 0 < args.gpu_memory_utilization < 1 or not 1 <= args.port <= 65535:
        raise ValueError("Invalid GPU utilization or port")
    flash = args.qwen38_variant == "flash-next"
    tokens = args.mtp_tokens if flash else args.draft_tokens
    if args.max_num_batched_tokens < args.batch * (1 + tokens):
        raise ValueError("Token batch must fit verification tokens for all sequences")
    speculative = {"method": "dflash", "model": str(args.draft_model.resolve()),
                   "num_speculative_tokens": args.draft_tokens,
                   "draft_tensor_parallel_size": args.tensor_parallel_size, "max_model_len": args.context,
                   "kv_cache_dtype": "fp8", "attention_backend": "TRITON_ATTN",
                   "rejection_sample_method": args.draft_rejection_method,
                   "draft_sample_method": args.draft_sample_method}
    if draft_plan is None:
        draft_plan = draft_quantization_plan(args)
    if draft_plan["vllm_quantization"] is not None:
        speculative["quantization"] = draft_plan["vllm_quantization"]
    if flash:
        speculative = {"method": "mtp", "num_speculative_tokens": tokens,
                       "moe_backend": args.mtp_moe_backend, "max_model_len": args.context}
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
               str(args.model.resolve()), "--served-model-name", served_model_name(args.qwen38_variant),
               "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", str(args.tensor_parallel_size), "--dtype", "bfloat16",
               "--kv-cache-dtype", "fp8",
               "--max-model-len", str(args.context), "--max-num-seqs", str(args.batch),
               "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--no-enable-flashinfer-autotune", "--reasoning-parser", "qwen3"]
    if args.qwen38_variant == "qwen35-9b":
        command += ["--quantization", "fp8_per_tensor" if args.target_quantization == "auto" else args.target_quantization]
    if args.load_format != "auto":
        command += ["--load-format", args.load_format]
    if args.load_format != "fastsafetensors":
        command += ["--model-loader-extra-config", json.dumps({
            "enable_multithread_load": not flash_profile.active(args), "num_threads": args.model_loader_threads})]
    if draft_plan["quantized_context"]:
        command += ["--worker-cls", "qwen38_dflash_worker.QuantizedDFlashWorker"]
    elif args.fastsafetensors_max_gpu_shard_gib:
        command += ["--worker-cls", "qwen38_fastsafetensors_worker.GuardedFastSafetensorsWorker"]
    if tokens:
        command += ["--speculative-config", json.dumps(speculative)]
    if flash:
        # Let checkpoint metadata select plain NVFP4 or FP4-Local's mixed
        # NVFP4/MXFP8 recipe; forcing modelopt_fp4 loses the per-layer recipe.
        command += ["--moe-backend", args.moe_backend,
                    "--engram-config", json.dumps({"cpu_offload": args.ple_cpu_offload}),
                    "--enable-prefix-caching"]
        if args.expert_parallel:
            command.append("--enable-expert-parallel")
        if args.enable_eplb:
            command.append("--enable-eplb")
            if args.eplb_config is not None:
                command += ["--eplb-config", json.dumps(args.eplb_config)]
    if args.cpu_offload_gb:
        command += ["--cpu-offload-gb", str(args.cpu_offload_gb),
                    "--cpu-offload-params", *args.cpu_offload_params]
    if args.kv_offload_gib_per_rank:
        if "--enable-prefix-caching" not in command:
            command.append("--enable-prefix-caching")
        command += ["--kv-transfer-config", json.dumps({
            "kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use_per_rank": int(args.kv_offload_gib_per_rank * 1024**3),
                                           "lazy_offload": args.kv_offload_mode == "lazy"}})]
    if args.input_mode == "text-image":
        # FP4-Local's vision MLP has K=4304: TP4 gives K=1076 and
        # splits NVFP4 scale groups (16). Keep each vision encoder whole;
        # native encoder DP distributes images, without changing language TP/EP.
        vision_tp_mode = "data" if flash_profile.active(args) else "weights"
        command += ["--limit-mm-per-prompt", json.dumps({"image": args.vision_max_images, "video": 0}),
                    "--mm-processor-kwargs", json.dumps(qwen38_vision.processor_kwargs(args.vision_max_pixels)),
                    "--mm-encoder-tp-mode", vision_tp_mode]
    else:
        command.append("--language-model-only")
    if args.scale_out_endpoints:
        command.append("--enable-scale-out")
    if args.prompt_tokens_details:
        command.append("--enable-prompt-tokens-details")
    if args.eager:
        command.append("--enforce-eager")
    if args.enable_auto_tool_choice:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser]
    return command + flash_profile.command_args(args) + args.vllm_arg


def dcp_size(vllm_args):
    size = 1
    for i, arg in enumerate(vllm_args):
        name, eq, value = arg.partition("=")
        if name in ("--decode-context-parallel-size", "-dcp"):
            size = int(value if eq else vllm_args[i + 1])
    return size


def selected_gpu_ids(args, *, bench=True):
    """bench=False only validates: auto-bench then runs at launch, when the GPUs are idle."""
    if args.tensor_parallel_size < 1:
        raise ValueError("Tensor parallel size must be positive")
    if args.gpu_ids == gpu_pair_bench.AUTO:
        if not bench:
            return args.gpu_ids
        return gpu_pair_bench.choose(args.tensor_parallel_size, dcp_size(args.vllm_arg))
    ids = ([x.strip() for x in args.gpu_ids.split(",")] if args.gpu_ids is not None else
           [str(i) for i in range(args.tensor_parallel_size)])
    if (len(ids) != args.tensor_parallel_size or len(set(ids)) != len(ids)
            or any(not x.isdigit() for x in ids)):
        raise ValueError("GPU IDs must be distinct nonnegative indices, one per TP rank")
    return ",".join(ids)


def environment(temp, args=None, *, draft_plan=None):
    env = os.environ.copy()
    if args is not None and flash_profile.active(args):
        for key in flash_profile.cleared_environment_keys(env):
            env.pop(key)
    overrides = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "4",
                 "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_PLUGINS": "",
                 "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                 "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1",
                 "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp),
                 "TRITON_CACHE_DIR": str(temp / "triton"),
                 "TORCHINDUCTOR_CACHE_DIR": str(temp / "inductor"),
                 "VLLM_CACHE_ROOT": str(temp / "vllm")}
    if args is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = selected_gpu_ids(args)
        overrides["VLLM_USE_RUST_FRONTEND"] = "1" if args.rust_frontend else "0"
        # The supported native implementation uses --engram-config; the old
        # PLE worker environment switch is unrecognized by modern vLLM.
        env.pop("VLLM_PLE_CPU_OFFLOAD", None)
        env.pop("ARC_FASTSAFETENSORS_MAX_GPU_SHARD_BYTES", None)
        env.pop("VLLM_FASTSAFETENSORS_MAX_GPU_SHARD_BYTES", None)
        if draft_plan is None:
            draft_plan = draft_quantization_plan(args)
        if draft_plan["quantized_context"] or args.fastsafetensors_max_gpu_shard_gib:
            overrides["PYTHONPATH"] = os.pathsep.join(filter(None, [
                str(Path(__file__).resolve().parent), env.get("PYTHONPATH")]))
        if args.load_format == "fastsafetensors":
            # Use buffered I/O on Kaggle's FUSE mounts; preserve an explicit
            # direct-I/O opt-in. This does not fix all cold remote-read stalls.
            overrides["FASTSAFETENSORS_ODIRECT"] = env.get("FASTSAFETENSORS_ODIRECT", "0")
        if args.fastsafetensors_max_gpu_shard_gib:
            overrides.update({
                "ARC_FASTSAFETENSORS_MAX_GPU_SHARD_BYTES": str(int(args.fastsafetensors_max_gpu_shard_gib * 1024**3)),
                # Newer forks implement the same CPU spill inside the native iterator.
                # Keep both entry points bounded (including paths not using the worker monkeypatch).
                "VLLM_FASTSAFETENSORS_MAX_GPU_SHARD_BYTES": str(int(args.fastsafetensors_max_gpu_shard_gib * 1024**3)),
                "VLLM_FASTSAFETENSORS_QUEUE_SIZE": "0"})
        if flash_profile.active(args):
            overrides.update(flash_profile.environment())
    env.update(overrides)
    return env, overrides


def stage_draft_for_launch(args):
    """No-copy default/dry run; the verified new destination is never overwritten."""
    destination = getattr(args, "draft_stage_dir", None)
    if destination is None or not args.launch:
        return args, None
    from qwen38_stage_checkpoint import stage_checkpoint
    receipt = stage_checkpoint(args.draft_model, destination)
    staged = copy(args)
    staged.draft_model = Path(receipt["destination"])
    return staged, receipt


def launch(args):
    args.model, args.draft_model = args.model.resolve(), args.draft_model.resolve()
    args.output, args.temp_root = args.output.resolve(), args.temp_root.resolve()
    try:
        draft_plan = draft_quantization_plan(args)
        command = command_for(args, draft_plan=draft_plan)
        target = read_checkpoint(args.model, allow_flattened=flash_profile.active(args))
        draft = None
        if args.qwen38_variant == "flash-next":
            validate_flash_next(target["config"], args.context, args.tensor_parallel_size)
            if flash_profile.active(args):
                flash_profile.validate_checkpoint(target["config"])
        else:
            draft = read_checkpoint(args.draft_model)
            validate_pair(target["config"], draft["config"], args.context, args.tensor_parallel_size,
                          variant=args.qwen38_variant, target_quantization=args.target_quantization,
                          draft_tokens=args.draft_tokens)
        if not (args.model / "tokenizer.json").is_file():
            raise ValueError("Local target tokenizer.json is required; no downloads")
        vision = None
        if args.input_mode == "text-image":
            vision = qwen38_vision.read_metadata(args.model)
            vision.update(qwen38_vision.validate_weights(args.model, vision,
                          resolved_shards=target["resolved_shards"]))
        if len(os.fsencode(args.temp_root / ("q27-" + "x" * 8) / ("x" * 36))) > 107:
            raise ValueError("Temp root too long for vLLM Unix sockets")
        if args.output.exists():
            raise ValueError(f"Use a fresh server output directory: {args.output}")
        if args.launch:
            if args.load_format == "fastsafetensors" and importlib.util.find_spec("fastsafetensors") is None:
                raise ValueError("fastsafetensors is not installed; this offline launcher will not download packages")
            if args.fastsafetensors_max_gpu_shard_gib:
                for helper in ("qwen38_fastsafetensors.py", "qwen38_fastsafetensors_worker.py"):
                    if not Path(__file__).with_name(helper).is_file():
                        raise ValueError(f"Missing fast loader guard helper: {helper}")
            if draft_plan["quantized_context"]:
                for helper in ("qwen38_dflash_quantization.py", "qwen38_dflash_worker.py"):
                    if not Path(__file__).with_name(helper).is_file():
                        raise ValueError(f"Missing draft quantization helper: {helper}")
            vp = Path(importlib.util.find_spec("vllm").origin).parent
            if flash_profile.active(args):
                flash_profile.check_runtime(vp)
            draft_type = ("dflash2" if draft and "DFlash2DraftModel" in draft["config"]["architectures"] else "dflash")
            required = (["models/qwen4_exp/nvidia/model.py", "config/engram.py"]
                        if args.qwen38_variant == "flash-next" else
                        [f"model_executor/models/qwen3_{draft_type}.py", f"v1/worker/gpu/spec_decode/{draft_type}/speculator.py"])
            layers = target["config"].get("quantization_config", {}).get("quantized_layers", {})
            if any(info.get("weight_scale_encoding") == "csf" for info in layers.values()):
                required.append("model_executor/model_loader/csf_scales.py")
            for relative in required:
                if not (vp / relative).is_file():
                    raise ValueError(f"Installed vLLM is missing {args.qwen38_variant} support: {relative}")
            if args.rust_frontend:
                raw = os.environ.get("VLLM_RUST_FRONTEND_PATH", "auto")
                binary = vp / "vllm-rs" if raw.lower() in ("auto", "1", "true") else Path(raw)
                if not binary.is_file() or not os.access(binary, os.X_OK):
                    raise ValueError("Rust frontend requested but no installed executable vllm-rs found")
            processes = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=10)
            if processes.strip():
                raise ValueError(f"GPU processes are active; will not stop them: {processes.strip()}")
            with socket.socket() as check:
                # Match the HTTP server's restart behavior: TIME_WAIT sockets
                # are harmless, but an active listener must still reject bind.
                check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                check.bind(("127.0.0.1", args.port))
    except (ValueError, OSError, KeyError, AttributeError) as exc:
        raise RuntimeError(str(exc)) from exc
    args.output.mkdir(parents=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="q27-", dir=args.temp_root))
    server_args, draft_staging = stage_draft_for_launch(args)
    checkpoint_view = None
    if target["flattened_shards"]:
        view = args.output / "model"
        checkpoint_view = flash_profile.make_checkpoint_view(args.model, view, target["resolved_shards"])
        server_args = copy(server_args)
        server_args.model = view
    if checkpoint_view is not None or draft_staging is not None:
        command = command_for(server_args, draft_plan=draft_plan)
    env, overrides = environment(temp, args, draft_plan=draft_plan)
    text = target["config"].get("text_config", target["config"])
    full_layers = text["layer_types"].count("full_attention")
    receipt = {"status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
               "command": command, "environment_overrides": overrides,
               "target": target, "draft": draft,
               "draft_staging": draft_staging,
               "checkpoint_view": checkpoint_view,
               "flash_next_profile": args.flash_next_profile,
               "flash_next_prefill_graphs": args.flash_next_prefill_graphs,
               "input_mode": args.input_mode, "vision": vision,
               "vision_encoder_tp_mode": command[command.index("--mm-encoder-tp-mode") + 1]
                   if args.input_mode == "text-image" else None,
               "variant": args.qwen38_variant, "served_model_name": served_model_name(args.qwen38_variant),
               "tensor_parallel_size": args.tensor_parallel_size,
               "target_quantization": ("fp8_per_tensor" if args.target_quantization == "auto" else args.target_quantization)
                   if args.qwen38_variant == "qwen35-9b" else "checkpoint",
               "draft_quantization": args.draft_quantization,
               "draft_quantization_plan": draft_plan,
               "load_format": args.load_format,
               "fastsafetensors_max_gpu_shard_gib": args.fastsafetensors_max_gpu_shard_gib,
               "ple_cpu_offload": args.ple_cpu_offload if args.qwen38_variant == "flash-next" else None,
               "environment_removed": flash_profile.cleared_environment_keys(os.environ) if flash_profile.active(args)
                   else ["VLLM_PLE_CPU_OFFLOAD"] if "VLLM_PLE_CPU_OFFLOAD" in os.environ else [],
               "versions": {}, "draft_tokens": args.mtp_tokens if draft is None else args.draft_tokens,
               "query_positions_per_request": (args.mtp_tokens if draft is None else args.draft_tokens) + 1,
               "draft_training_block_size": draft["config"]["dflash_config"].get("block_size", draft["config"].get("block_size")) if draft else None,
               "target_full_attention_kv_gib_per_rank_at_full_batch":
                   args.batch * args.context * 2 * full_layers * text["num_key_value_heads"]
                   * text["head_dim"] / args.tensor_parallel_size / 1024**3 if draft else None,
               "cpu_kv_offload_gib_total": args.tensor_parallel_size * args.kv_offload_gib_per_rank,
               "hisparse_logical_host_pool_gib": args.flash_next_host_pool_gib if flash_profile.active(args) else None,
               "capacity_note": "Use measured vLLM capacity. Dense arithmetic excludes draft/state/padding; Flash-Next uses QSA compression and replicated KV heads. CPU KV offload does not guarantee full-batch GPU residency."}
    helpers = []
    if flash_profile.active(args):
        helpers += ["qwen38_flash_profile.py"]
        receipt["speed_profile_evidence"] = {
            "decode_tps": 379.7, "batch": 24, "prompt_tokens": 65536, "output_tokens": 4096,
            "hardware": "4xL4", "context_limit": 102400,
            "input_mode": "text", "vision_validated": False,
            "scope": "One completed bounded serving probe plus short greedy checks; not full ARC accuracy. "
                     "The merged fork wheel and fresh ARC workload need revalidation. "
                     "Vision/TIR speed and memory capacity are not covered by this text-only probe. "
                     "24 simultaneous 102400-token requests are not guaranteed to fit."}
        if args.input_mode == "text-image":
            receipt["capacity_note"] += " Native encoder DP replicates vision weights on each GPU; encoder buffers also consume VRAM. Remeasure KV capacity."
            print("*** FP4-Local vision uses --mm-encoder-tp-mode data to preserve NVFP4 scale groups "
                  "(vision MLP 4304 / TP4 = 1076, not divisible by 16). Language model stays TP4/EP4. "
                  "Replicated vision weights/encoder buffers use additional VRAM. "
                  "The B24 text-only benchmark does not establish vision capacity or speed.", flush=True)
    if args.fastsafetensors_max_gpu_shard_gib:
        helpers += ["qwen38_fastsafetensors.py", "qwen38_fastsafetensors_worker.py"]
    if draft_plan["quantized_context"]:
        helpers += ["qwen38_dflash_quantization.py", "qwen38_dflash_worker.py"]
    receipt["runtime_helpers_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in helpers}
    for package in ("vllm", "torch", "transformers", "triton", "compressed-tensors", "flashinfer-python"):
        try:
            receipt["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            receipt["versions"][package] = None
    receipt_path = args.output / "launch.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    process = None
    try:
        if args.launch:
            with (args.output / "server.log").open("xb") as log:
                process = subprocess.Popen(command, env=env, cwd=args.output,
                                           stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(status="launched_not_yet_ready", pid=process.pid)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if draft_plan["active"]:
            print("Draft quantization:", json.dumps(draft_plan), flush=True)
        print("Command:", shlex.join(command), flush=True)
        print(json.dumps({k: receipt[k] for k in ("status", "draft_tokens", "query_positions_per_request")}))
        if "pid" in receipt:
            print("PID:", receipt["pid"], "Log:", args.output / "server.log", flush=True)
    except BaseException:
        stop_owned_process(process)
        raise

    return process, receipt


# Restart supervisor (--supervise N). A detached copy of this script owns the server: before each start it stops
# whatever listens on the port and waits for the GPUs to be free; after an exit it stops what is left of the server's
# session and starts again, N times at most; so does a start that stops making tokens (--supervise-stall-s). Its
# status JSON tells clients (the S26 solver) which start is serving.
SUPERVISOR_CHILD = "ARC_SUPERVISOR_CHILD"  # set in the detached copy: the protected session ids, comma-separated
SUPERVISOR_FIRST_PID_S = 600


def add_supervise_arguments(p):
    g = p.add_argument_group("restart supervisor")
    g.add_argument("--supervise", type=int, metavar="N",
                   help="Start the server from a detached supervisor that restarts it up to N times after it exits")
    g.add_argument("--supervise-status", type=Path, help="Supervisor status JSON (default: next to the server output)")
    g.add_argument("--supervise-deadline", type=float, help="Unix time: no restart that would not be up by then")
    g.add_argument("--supervise-up-timeout", type=float, default=3600.0,
                   help="Seconds a start may take to list the model on /v1/models before it is stopped")
    g.add_argument("--supervise-stall-s", type=float, default=1500.0,
                   help="Seconds a start that is up may run or queue requests without a new token (or without "
                        "answering /metrics) before it is stopped and restarted; 0: never")


def proc_stat(pid):
    """(state, ppid, pgrp, session, starttime) from /proc/<pid>/stat, or None once the process is gone."""
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        return f[0], int(f[1]), int(f[2]), int(f[3]), int(f[19])
    except (OSError, ValueError, IndexError):
        return None


def alive(pid):
    stat = proc_stat(pid)
    return stat is not None and stat[0] not in ("Z", "X")


def ancestor_pids(pid=None):
    """pid (default: ours) and its ancestors up to the namespace's init."""
    pids, pid = [], pid or os.getpid()
    while pid > 0 and pid not in pids and (stat := proc_stat(pid)) is not None:
        pids.append(pid)
        pid = stat[1]
    return pids


def session_pids(sid):
    return [int(d.name) for d in Path("/proc").iterdir()
            if d.name.isdigit() and (stat := proc_stat(d.name)) is not None and stat[3] == sid]


def port_holders(port):
    """Pids holding a TCP socket that listens on port (any address), from /proc/net/tcp{,6} and /proc/*/fd."""
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            f = row.split()
            if len(f) > 9 and f[3] == "0A" and int(f[1].rpartition(":")[2], 16) == port:
                inodes.add(f"socket:[{f[9]}]")
    pids = set()
    for d in Path("/proc").iterdir() if inodes else ():
        if not d.name.isdigit():
            continue
        try:
            fds = list((d / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(fd) in inodes:
                    pids.add(int(d.name))
                    break
            except OSError:
                continue
    return sorted(pids)


def kill_pids(pids, grace=10.0):
    """SIGTERM, then SIGKILL whatever is still alive after grace seconds."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        t = time.monotonic()
        while any(map(alive, pids)) and time.monotonic() - t < grace:
            time.sleep(0.2)


def free_port(port, protect=(), grace=10.0):
    """Stop what listens on port: the holder's whole session, or only the holder when that session is protected
    (the notebook kernel's, ours). Never this process or an ancestor. Returns the pids it signalled."""
    holders = port_holders(port)
    if not holders:
        return []
    mine = ancestor_pids()
    if set(holders) & set(mine):
        raise RuntimeError(f"Port {port} is held by this process or an ancestor: {holders}")
    protect, targets = set(protect) | {stat[3] for p in mine if (stat := proc_stat(p))}, set()
    for pid in holders:
        if (stat := proc_stat(pid)) is not None:
            targets.update([pid] if stat[3] in protect else session_pids(stat[3]))
    targets = sorted(targets - set(mine))
    kill_pids(targets, grace)
    t = time.monotonic()
    while held := port_holders(port):
        if time.monotonic() - t > grace:
            raise RuntimeError(f"Port {port} is still held after stopping {targets}: {held}")
        time.sleep(0.2)
    return targets


def gpu_processes():
    """Compute-app pids nvidia-smi lists; [] without nvidia-smi (the launch's own check then reports it)."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                      text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def wait_gpu_free(timeout, stop, probe=gpu_processes):
    t = time.monotonic()
    while (pids := probe()) and not stop.is_set():
        if time.monotonic() - t > timeout:
            raise RuntimeError(f"GPU processes are still active after {timeout:.0f}s: {pids}")
        stop.wait(2)


def served_models(port, timeout=5.0):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            return [m.get("id") for m in json.loads(r.read())["data"]]
    except Exception:
        return []


def server_metrics(port, timeout=10.0):
    """(generated tokens so far, requests running or waiting) from the vLLM server's /metrics, or None."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tokens = busy = 0.0
    try:
        with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=timeout) as r:
            for line in r.read().decode().splitlines():
                name, value = line.partition("{")[0].partition(" ")[0], line.rpartition(" ")[2]
                if name == "vllm:generation_tokens_total":
                    tokens += float(value)
                elif name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
                    busy += float(value)
    except Exception:
        return None
    return tokens, busy


def write_status(path, status):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(status, updated=time.time()), indent=2) + "\n")
    os.replace(tmp, path)


def read_status(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def run_supervisor(start, *, port, model, restarts, status_path, deadline=None, up_timeout=3600.0, stall_s=0.0,
                   protect=(), stop=None, poll_s=5.0, backoff_s=10.0, gpu_timeout=180.0,
                   free=free_port, gpu_free=wait_gpu_free, probe=served_models, metrics=server_metrics):
    """start(n) -> (process, log) starts the server (n = 0, 1, ...). After each exit, a start that does not list
    model within up_timeout, or (stall_s > 0) one that is up but makes no new token for stall_s while requests run or
    wait, or does not answer /metrics for that long (a hung engine), the rest of its session is stopped and it starts
    again, restarts times at most, but not once the deadline (Unix time) is nearer than the startup the last start
    took. A start 0 that fails before its process exists (launch checks, port, GPUs) ends it at once. Status: start,
    state (starting, up, restarting, failed, stopped), pid, log, up_s, error and the earlier starts in history.
    Returns 0 once stopped, 1 when it gives up."""
    stop = stop or threading.Event()
    status = dict(supervisor_pid=os.getpid(), port=port, model=model, max_restarts=restarts, deadline=deadline,
                  start=0, state="starting", pid=None, log=None, up_s=None, error=None, history=[])

    def say(message):
        print(f"[supervisor {time.strftime('%H:%M:%S')}] {message}", flush=True)

    n, last_up, process = 0, 0.0, None
    try:
        while True:
            status.update(start=n, state="starting", pid=None, log=None, up_s=None, error=None)
            write_status(status_path, status)
            t, process, log, error = time.time(), None, None, None
            try:
                if freed := free(port, protect):
                    say(f"port {port}: stopped pids {freed}")
                gpu_free(gpu_timeout, stop)
                if not stop.is_set():
                    process, log = start(n)
            except Exception as exc:  # e.g. launch's checks; a failed start counts as one
                error = f"{type(exc).__name__}: {exc}"
            if process is not None:
                status.update(pid=process.pid, log=str(log))
                write_status(status_path, status)
                say(f"start {n}: pid {process.pid}, log {log}")
                while not stop.is_set():
                    if process.poll() is not None:
                        error = f"server exited with code {process.returncode}"
                        break
                    if status["up_s"] is None:
                        if model in probe(port):
                            status.update(state="up", up_s=round(time.time() - t, 1))
                            last_up = status["up_s"]
                            write_status(status_path, status)
                            say(f"start {n}: up after {last_up:.0f}s")
                            progress_t, tokens = time.time(), -1.0
                        elif time.time() - t > up_timeout:
                            error = f"server not up after {up_timeout:.0f}s"
                            break
                    elif stall_s > 0:  # E066: a hung engine keeps its process and port; idle is progress
                        if (m := metrics(port)) is not None and (m[0] > tokens or m[1] == 0):
                            progress_t, tokens = time.time(), m[0]
                        elif time.time() - progress_t > stall_s:
                            error = (f"server stalled: no new token for {stall_s:.0f}s with {m[1]:.0f} requests "
                                     "running or waiting" if m else f"/metrics unreadable for {stall_s:.0f}s")
                            break
                    stop.wait(poll_s)
                stop_owned_process(process)
                kill_pids(session_pids(process.pid), grace=5.0)
            status["history"].append(dict(start=n, pid=process and process.pid, log=log and str(log),
                                          code=process and process.returncode, up_s=status["up_s"],
                                          ran_s=round(time.time() - t, 1), error=error))
            if stop.is_set():
                status.update(state="stopped", error=None)
                write_status(status_path, status)
                say("stopped")
                return 0
            say(f"start {n}: {error}")
            if n == 0 and process is None:
                give_up = "start 0 failed"
            elif n >= restarts:
                give_up = f"no restart left ({restarts})"
            elif deadline is not None and time.time() + backoff_s + last_up > deadline:
                give_up = f"a restart would not be up before the deadline ({last_up:.0f}s startup)"
            else:
                give_up = None
            if give_up:
                status.update(state="failed", error=f"{error}; {give_up}")
                write_status(status_path, status)
                say(f"giving up: {give_up}")
                return 1
            status.update(state="restarting", error=error)
            write_status(status_path, status)
            stop.wait(backoff_s)
            n += 1
    except BaseException as exc:
        stop_owned_process(process)
        status.update(state="failed", error=f"supervisor: {type(exc).__name__}: {exc}")
        write_status(status_path, status)
        raise


def stop_previous_supervisor(port, grace=60.0):
    """Stop the supervisor an earlier run left on this port (it stops its server first). Found by the pid and start
    time it recorded, never by name; the record is ours from then on."""
    record = Path(tempfile.gettempdir()) / f"arc-supervisor-{port}.json"
    old = read_status(record)
    if old.get("pid") not in (None, os.getpid()) and (stat := proc_stat(old["pid"])) and stat[4] == old.get("start"):
        print(f"[supervisor] stopping the previous supervisor of port {port}: pid {old['pid']}", flush=True)
        kill_pids([old["pid"]], grace)
    write_status(record, dict(pid=os.getpid(), start=proc_stat(os.getpid())[4]))


def supervise(args, start, *, port, model, status_path):
    """--supervise N: run this script again in its own session (stdout to supervisor.log next to the status file) as
    run_supervisor(start, ...), then print its pid and the first server's pid and log once that process exists."""
    status_path = Path(status_path).resolve()
    protect = os.environ.pop(SUPERVISOR_CHILD, None)
    if protect is not None:
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *unused: stop.set())
        stop_previous_supervisor(port)
        sids = {int(s) for s in protect.split(",") if s}
        sys.exit(run_supervisor(start, port=port, model=model, restarts=args.supervise, status_path=status_path,
                                deadline=args.supervise_deadline, up_timeout=args.supervise_up_timeout,
                                stall_s=args.supervise_stall_s, protect=sids, stop=stop,
                                poll_s=float(os.environ.get("ARC_SUPERVISOR_POLL_S", 5)),
                                backoff_s=float(os.environ.get("ARC_SUPERVISOR_BACKOFF_S", 10))))
    if status_path.exists():
        raise RuntimeError(f"Use a fresh supervisor status file: {status_path}")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log = status_path.with_suffix(".log")
    sids = sorted({stat[3] for pid in ancestor_pids() if (stat := proc_stat(pid))})
    with log.open("xb") as out:
        child = subprocess.Popen([sys.executable, "-u", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
                                 env=dict(os.environ, **{SUPERVISOR_CHILD: ",".join(map(str, sids))}),
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print("Supervisor PID:", child.pid, "Status:", status_path, "Log:", log, flush=True)
    t, status = time.monotonic(), {}
    while not (status := read_status(status_path)).get("pid") and status.get("state") not in ("failed", "stopped"):
        if child.poll() is not None or time.monotonic() - t > SUPERVISOR_FIRST_PID_S:
            break
        time.sleep(0.5)
    if not status.get("pid"):
        error = status.get("error") or (f"supervisor exited with code {child.returncode}" if child.poll() is not None
                                        else f"no server process after {SUPERVISOR_FIRST_PID_S}s")
        kill_pids([child.pid], grace=60.0)  # SIGTERM first: it stops its server before it exits
        child.wait()
        raise RuntimeError(f"Supervised start failed: {error}; see {log}")
    print("PID:", status["pid"], "Log:", status["log"], flush=True)
    return child.pid, status


def main():
    args = parser().parse_args()
    if args.supervise is None:
        launch(args)
        return
    if not args.launch:
        raise SystemExit("--supervise needs --launch")
    output = args.output.resolve()

    def start(n):
        run = copy(args)
        run.output = output.with_name(f"{output.name}_restart{n}") if n else output
        return launch(run)[0], run.output / "server.log"

    supervise(args, start, port=args.port, model=served_model_name(args.qwen38_variant),
              status_path=args.supervise_status or output.with_name(output.name + "_supervisor.json"))


if __name__ == "__main__":
    main()
