"""Opt-in four-L4 FP4-Local serving recipe, with no experiment-folder imports.

Reproduces the completed B24/no-draft recipe, not unvalidated kernel screens.
Only child-server environment is changed; the 4B cascade stage is untouched.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath
import sys


NAME = "fp4-local-l4"
MODEL = "/kaggle/input/models/boristown/qwen3-8-flash-next-fp4-local/transformers/fp4-local/1"


def active(args):
    return getattr(args, "flash_next_profile", "manual") == NAME


class ProfileParser(argparse.ArgumentParser):
    """Apply profile defaults after parsing, without changing parser reuse/order.

    Explicit options always win; validation rejects incompatible combinations.
    All three public entrypoints use this parser, including command previews.
    """

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        explicit = set(vars(namespace)) if namespace is not None else set()
        result = super().parse_args(argv, namespace)
        if not active(result):
            return result
        for token in argv:
            option = token.split("=", 1)[0]
            action = self._option_string_actions.get(option)
            if action is None and self.allow_abbrev and option.startswith("--"):
                matches = [a for name, a in self._option_string_actions.items()
                           if name.startswith(option)]
                if matches and len({a.dest for a in matches}) == 1:
                    action = matches[0]
            if action is not None:
                explicit.add(action.dest)
        defaults = dict(qwen38_variant="flash-next", model=MODEL, model_qwen38=MODEL,
                        context=102400, concurrency=24, batch=24,
                        draft_tokens=0, mtp_tokens=0, gpu_memory_utilization=0.96,
                        max_num_batched_tokens=512, model_loader_threads=1,
                        load_format="safetensors", cpu_offload_gb=0.25,
                        cpu_offload_params=["embed_tokens"])
        for action in self._actions:
            if action.dest in defaults and action.dest not in explicit:
                value = deepcopy(defaults[action.dest])
                if action.type is not None and isinstance(value, str):
                    value = action.type(value)
                setattr(result, action.dest, value)
        return result


def add_arguments(parser):
    parser.add_argument("--flash-next-profile", choices=("manual", NAME), default="manual",
                        help="Opt-in tested FP4-Local/4xL4 recipe: TP4+EP4, no draft, HiSparse, "
                             "embedding-only UVA, HC TP/prefetch and lossless host reduction. "
                             "Defaults to B24/102400; dense 27B defaults remain unchanged.")
    parser.add_argument("--flash-next-host-pool-gib", type=float, default=24,
                        help="FP4-Local profile's logical HiSparse host KV pool, NOT per rank. "
                             "Separate from PLE and pinned input-embedding RAM. Default 24 GiB.")


def validate(args):
    if not active(args):
        return
    expected = dict(qwen38_variant="flash-next", tensor_parallel_size=4,
                    expert_parallel=True, ple_cpu_offload=True, moe_backend="marlin",
                    mtp_tokens=0, draft_tokens=0, draft_quantization="auto",
                    input_mode="text", eager=False, rust_frontend=False, enable_eplb=False,
                    cpu_offload_gb=0.25, cpu_offload_params=["embed_tokens"],
                    kv_offload_gib_per_rank=0, model_loader_threads=1,
                    max_num_batched_tokens=512)
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise ValueError(f"{NAME} requires {field}={value!r}; use --flash-next-profile manual "
                             "for a different, unbenchmarked configuration")
    batch = getattr(args, "batch", getattr(args, "server_max_num_seqs", None))
    if batch is None:
        batch = args.concurrency
    if not 1 <= batch <= 24 or not 1 <= args.context <= 102400:
        raise ValueError(f"{NAME} supports 1..24 slots and context <=102400; "
                         "24 simultaneous full-length requests are not guaranteed to fit")
    if not 0 < args.gpu_memory_utilization <= 0.96:
        raise ValueError(f"{NAME} requires GPU utilization <=0.96")
    pool = args.flash_next_host_pool_gib
    if not math.isfinite(pool) or pool <= 0:
        raise ValueError("flash-next-host-pool-gib must be positive and finite")
    # Raw flags must not silently negate the recipe's capacity/correctness guards.
    protected = ("--speculative-config", "--attention-config", "--kv-transfer-config",
                 "--compilation-config", "--cuda-graph", "--cudagraph", "--quantization",
                 "--tensor-parallel-size", "--decode-context-parallel-size", "--pipeline-parallel-size",
                 "--data-parallel-size", "--max-num-seqs", "--max-num-batched-tokens", "--max-model-len",
                 "--cpu-offload", "--offload-", "--enable-hc-sp", "--enforce-eager", "--worker-cls",
                 "--moe-backend", "--enable-expert-parallel", "--no-enable-expert-parallel",
                 "--no-enable-ep-weight-filter", "--disable-custom-all-reduce", "--enable-eplb",
                 "--engram-config", "--kv-cache-dtype", "--dtype", "--load-format", "--model-loader-",
                 "--gpu-memory-utilization", "--no-enable-batch-sharded-sampling",
                 "--no-enable-prefix-caching", "--enable-flashinfer-autotune", "--model", "--served-model-name")
    for raw in args.vllm_arg:
        if raw.split("=", 1)[0].startswith(protected):
            raise ValueError(f"Raw vLLM argument conflicts with {NAME}: {raw.split('=', 1)[0]}")


def validate_checkpoint(config):
    text = config.get("text_config", {})
    quant = config.get("quantization_config", {})
    algos = {v.get("quant_algo") for v in quant.get("quantized_layers", {}).values()}
    if (config.get("model_type") != "qwen4_exp" or quant.get("quant_algo") != "MIXED_PRECISION"
            or not {"NVFP4", "MXFP8"}.issubset(algos)):
        raise ValueError(f"{NAME} requires the mixed NVFP4/MXFP8 FP4-Local checkpoint, not plain NVFP4")
    for field, expected in (("hidden_size", 2560), ("num_hidden_layers", 48), ("num_experts", 512)):
        if text.get(field) != expected:
            raise ValueError(f"{NAME} requires text_config.{field}={expected}")
    if text.get("tie_word_embeddings", config.get("tie_word_embeddings", False)):
        raise ValueError("Input-embedding UVA profile requires an untied output head")


def command_args(args):
    if not active(args):
        return []
    sizes = [n for n in (1, 2, 4, 8, 16) if n < args.batch] + [args.batch]
    return ["--offload-backend", "uva", "--enable-ep-weight-filter",
            "--decode-context-parallel-size", "1", "--enable-batch-sharded-sampling",
            "--attention-config", json.dumps({"indexer_kv_dtype": "fp8",
                                               "hisparse_config": {"device_buffer_size": 3136}}),
            "--kv-transfer-config", json.dumps({"kv_connector": "HiSparseConnector", "kv_role": "kv_both",
                "kv_connector_extra_config": {"host_pool_gib": args.flash_next_host_pool_gib}}),
            "--compilation-config", json.dumps({"cudagraph_capture_sizes": sizes})]


def environment():
    return {"OMP_NUM_THREADS": "1", "VLLM_HOST_ALLREDUCE": "1", "VLLM_HOST_ALLREDUCE_MODE": "44",
            "VLLM_HOST_ALLREDUCE_PREFETCH_BLOCKS": "16", "VLLM_QWEN4_EXP_HC_AR_PREFETCH": "1",
            "VLLM_QWEN4_EXP_HC_DOWN_TP": "1", "VLLM_BATCH_SHARD_TOPK": "32",
            "VLLM_GDN_DEFERRED_STATE_COMMIT": "off", "VLLM_GDN_SINGLE_STATE_BLOCK": "0"}


def cleared_environment_keys(environ):
    # Do not inherit dense-27B tile/vocab/GDN/quantized-AR experiments into FN.
    return sorted(k for k in environ if k.startswith(("VLLM_", "ARC_FP4_")))


def check_runtime(vllm_path):
    required = {
        "envs.py": ("VLLM_QWEN4_EXP_HC_DOWN_TP", "VLLM_QWEN4_EXP_HC_AR_PREFETCH"),
        "model_executor/model_loader/csf_scales.py": (),
        "distributed/kv_transfer/kv_connector/v1/hisparse/connector.py": (),
        "models/qwen4_exp/nvidia/model.py": ("embed_tokens", "wrap_input_embedding"),
    }
    for relative, markers in required.items():
        path = vllm_path / relative
        if not path.is_file() or any(marker not in path.read_text() for marker in markers):
            raise ValueError(f"{NAME} needs the matching vLLM fork wheel; missing support in {relative}")


def resolve_shards(source, names, *, allow_flattened=False):
    """Validate index paths before creating a read-only symlink layout."""
    result, flattened = {}, []
    for name in sorted(names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
            raise ValueError(f"Invalid checkpoint shard path: {name!r}")
        shard = source / name
        if not shard.is_file() and allow_flattened:
            shard = source / relative.name
            flattened.append(name)
        if not shard.is_file() or not shard.stat().st_size:
            raise ValueError(f"Missing or empty weight shard: {source / name}")
        resolved = shard.resolve()
        if resolved in result.values():
            raise ValueError(f"Ambiguous flattened shard: {name}")
        result[name] = resolved
    return result, flattened


def make_checkpoint_view(source, destination, shards):
    """Never edit/copy checkpoint weights or replace an existing output path."""
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            (destination / item.name).symlink_to(item.resolve())
    for name, original in shards.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(original)
    return {"source": str(source), "view": str(destination), "shards": len(shards),
            "weights_copied": False, "source_modified": False}
