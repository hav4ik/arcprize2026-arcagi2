"""Immutable, shared PiSSA residual checkpoints and adapter provenance.

Only file/metadata operations run at import time. Preparation streams HF
safetensors shards and initializes one projection at a time; no Unsloth/vLLM
engine or full GPU model is needed. A completed directory is never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from run_config import (
    is_pissa_init, lora_effective_scaling, normalize_lora_init,
    normalize_pissa_base_mode, resolve_lora_method,
)


SHARED_CONFIG_KEY = "arc_pissa_shared_base"
MANIFEST_FILE = "arc_pissa_manifest.json"
INITIAL_ADAPTER_DIR = "arc_pissa_initial_adapter"
FALLBACK_ADAPTER_ID = 1_000_000_000
PROJECTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_PROJECTION_RE = re.compile(
    r"model\.layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight"
)


def _json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"PiSSA manifest path escapes artifact: {relative!r}")
    return path


def read_shared_manifest(model_path, *, verify_files=False):
    """Read the completion marker, configuration identity, and file inventory."""
    root = Path(model_path)
    config = _json(root / "config.json")
    manifest = _json(root / MANIFEST_FILE)
    identity = manifest.get("base_id", "")
    if (
        manifest.get("format_version") != 1
        or not re.fullmatch(r"[0-9a-f]{64}", identity)
        or config.get(SHARED_CONFIG_KEY) != identity
        or manifest.get("mode") != "shared-residual"
    ):
        raise ValueError("Invalid/incomplete shared PiSSA residual identity")
    files = manifest.get("files", {})
    required = {
        "config.json", f"{INITIAL_ADAPTER_DIR}/adapter_config.json",
        f"{INITIAL_ADAPTER_DIR}/adapter_model.safetensors",
    } | set(manifest.get("weight_files", ()))
    if not manifest.get("weight_files") or not required <= files.keys():
        raise ValueError("Shared PiSSA manifest has an incomplete file inventory")
    for name, info in files.items():
        path = _inside(root, name)
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Shared PiSSA artifact missing/size mismatch: {name}")
        # Full weights are hashed at launcher preflight; tiny metadata is
        # checked again by every reader. Never rewrite a validated artifact.
        if (verify_files or name.endswith(".json")) and file_sha256(path) != info["sha256"]:
            raise ValueError(f"Shared PiSSA artifact checksum mismatch: {name}")
    return manifest


def validate_runtime_base(
    model_path, *, mode="original", initialization="default", method="rslora",
    rank=128, alpha=32.0, verify_files=False,
):
    """Reject both an ordinary adapter on W_res and a residual adapter on W."""
    mode = normalize_pissa_base_mode(mode)
    config_path = Path(model_path) / "config.json"
    config = _json(config_path) if config_path.is_file() else {}
    marked = SHARED_CONFIG_KEY in config or (Path(model_path) / MANIFEST_FILE).exists()
    if mode == "original":
        if marked:
            raise ValueError("Residual checkpoint requires --pissa-base-mode shared-residual; never rerun PiSSA SVD on it")
        return None
    if not marked or not is_pissa_init(normalize_lora_init(initialization)):
        raise ValueError("shared-residual mode requires a prepared PiSSA checkpoint and --lora-init pissa[_niter_N]")
    manifest = read_shared_manifest(model_path, verify_files=verify_files)
    expected = dict(
        initialization=normalize_lora_init(initialization), method=str(method).lower(),
        rank=int(rank), alpha=float(alpha),
    )
    if any(manifest["settings"].get(k) != v for k, v in expected.items()):
        raise ValueError(f"Shared PiSSA settings mismatch: requested={expected}, prepared={manifest['settings']}")
    return manifest


def validate_adapter_base(
    adapter_path, *, mode="original", base_id=None, rank=None, alpha=None, method=None,
):
    """Validate before vLLM sees the artifact, including inference-only replay."""
    mode = normalize_pissa_base_mode(mode)
    root = Path(adapter_path)
    manifest_path = root / "arc_adapter_manifest.json"
    if not manifest_path.is_file():
        if mode == "shared-residual":
            raise ValueError("Residual-base serving requires an adapter provenance manifest")
        return
    conversion = _json(manifest_path).get("conversion", {})
    artifact_mode = conversion.get("pissa_base_mode", "original")
    if artifact_mode != mode:
        raise ValueError(f"PiSSA adapter/base mode mismatch: adapter={artifact_mode}, engine={mode}")
    if mode == "shared-residual":
        config = _json(root / "adapter_config.json")
        if (
            not base_id or conversion.get("pissa_base_id") != base_id
            or config.get("r") != rank or conversion.get("export_rank") != rank
            or conversion.get("training_rank") != rank
            or conversion.get("pissa_converted")
            or config.get("init_lora_weights") is not True
            or config.get("rank_pattern") or config.get("alpha_pattern")
            or config.get("use_dora")
        ):
            raise ValueError("PiSSA adapter does not match the shared residual base/rank")
        config_method = "rslora" if config.get("use_rslora") else "lora"
        scaling = lora_effective_scaling(config_method, rank, config["lora_alpha"])
        if (
            conversion.get("source_lora_method") != config_method
            or not math.isclose(conversion.get("scaling", float("nan")), scaling, rel_tol=1e-12)
            or (alpha is not None and config["lora_alpha"] != alpha)
            or (method is not None and config_method != method)
        ):
            raise ValueError("PiSSA adapter scaling differs from the shared residual initialization")


def _source_files(root):
    """Use the authoritative HF index, not a glob that includes sidecars."""
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = _json(index)["weight_map"]
        weights = sorted(set(mapping.values()))
    elif (root / "model.safetensors").is_file():
        weights, mapping = ["model.safetensors"], None
    else:
        raise ValueError("PiSSA preparation requires an HF safetensors checkpoint")
    for name in weights:
        if Path(name).name != name or not (root / name).is_file():
            raise ValueError(f"Invalid/missing HF weight shard: {name}")
    # Preserve tokenizer/custom-code/routing metadata, not optimizer state or
    # unrelated training artifacts. Subdirectories are not model inputs here.
    files = set(weights)
    for path in root.iterdir():
        if path.is_file() and (
            path.suffix in {".json", ".txt", ".py", ".model", ".tiktoken", ".jinja", ".jinja2"}
            or path.name == "delta_mhar.safetensors"
        ):
            if path.name not in {"trainer_state.json", "adapter_config.json"}:
                files.add(path.name)
    return sorted(files), weights, mapping


def prepare_shared_base(
    source, destination, *, rank=128, alpha=32.0, method="rslora",
    initialization="pissa_niter_4", seed=42, device="cuda:0",
):
    """Prepare once, with PEFT's SVD, and publish only a complete artifact."""
    import torch
    import peft
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, destination = Path(source).resolve(), Path(destination).resolve()
    initialization = normalize_lora_init(initialization)
    method = resolve_lora_method(method, initialization)["method"]
    if not is_pissa_init(initialization):
        raise ValueError("Shared residual preparation requires PiSSA")
    scaling = lora_effective_scaling(method, rank, alpha)
    if not math.isfinite(scaling):
        raise ValueError("PiSSA scaling must be finite")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Use a separate sibling directory for the residual checkpoint")
    config = _json(source / "config.json")
    if config.get("model_type") != "qwen3" or config.get("quantization_config"):
        raise ValueError("Shared PiSSA preparation currently supports unquantized Qwen3 safetensors only")
    validate_runtime_base(source)
    files, weight_files, mapping = _source_files(source)
    source_hashes = {name: file_sha256(source / name) for name in files}
    settings = dict(rank=int(rank), alpha=float(alpha), method=method, initialization=initialization, seed=int(seed))
    if destination.exists():
        existing = read_shared_manifest(destination, verify_files=True)
        if existing["settings"] != settings or existing["source_sha256"] != source_hashes:
            raise ValueError("Existing residual checkpoint is different; choose a new output directory")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        factors, projections, seen = {}, [], set()
        for filename in files:
            if filename not in weight_files:
                shutil.copy2(source / filename, temporary / filename)
        for filename in weight_files:
            shard = {}
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for key in sorted(handle.keys()):
                    if key in seen or (mapping is not None and mapping.get(key) != filename):
                        raise ValueError(f"Duplicate/index-mismatched weight: {key}")
                    seen.add(key)
                    value = handle.get_tensor(key)
                    if _PROJECTION_RE.fullmatch(key):
                        if value.dtype != torch.bfloat16 or value.ndim != 2 or min(value.shape) < rank:
                            raise ValueError(f"PiSSA requires a BF16 matrix with dimensions >= rank: {key}")
                        if not torch.isfinite(value).all():
                            raise ValueError(f"Nonfinite base projection: {key}")
                        target_device = torch.device(device)
                        gpu_ids = [] if target_device.type == "cpu" else [target_device.index or 0]
                        # A stable per-projection seed makes shard order irrelevant.
                        tensor_seed = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") % (2**31)
                        with torch.random.fork_rng(devices=gpu_ids), torch.no_grad():
                            torch.manual_seed(tensor_seed)
                            layer = torch.nn.Linear(value.shape[1], value.shape[0], bias=False, device="meta", dtype=value.dtype)
                            layer.weight = torch.nn.Parameter(value.to(target_device), requires_grad=False)
                            container = torch.nn.Module()
                            container.add_module("projection", layer)
                            wrapped = peft.get_peft_model(
                                container, peft.LoraConfig(
                                    r=rank, lora_alpha=alpha, target_modules=["projection"],
                                    use_rslora=method == "rslora", init_lora_weights=initialization,
                                ), autocast_adapter_dtype=False,
                            )
                            adapter = wrapped.base_model.model.projection
                            module = key.removesuffix(".weight")
                            for side in ("A", "B"):
                                tensor = getattr(adapter, f"lora_{side}")["default"].weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                                if not torch.isfinite(tensor).all():
                                    raise ValueError(f"Nonfinite PiSSA initial factor: {key}/{side}")
                                factors[f"base_model.model.{module}.lora_{side}.weight"] = tensor
                            value = layer.weight.detach().cpu().contiguous()
                            if not torch.isfinite(value).all():
                                raise ValueError(f"Nonfinite PiSSA residual: {key}")
                            del adapter, layer, wrapped, container
                        projections.append(module)
                        print(f"PiSSA prepared {len(projections)}: {module}", flush=True)
                    shard[key] = value
            save_file(shard, temporary / filename, metadata=metadata or {"format": "pt"})
            del shard
        expected = {
            f"model.layers.{i}.{site}.{target}"
            for i in range(int(config["num_hidden_layers"]))
            for site, targets in (("self_attn", PROJECTION_TARGETS[:4]), ("mlp", PROJECTION_TARGETS[4:]))
            for target in targets
        }
        if set(projections) != expected or (mapping is not None and seen != mapping.keys()):
            raise ValueError("Shared PiSSA preparation did not cover the exact Qwen3 projection/index set")
        initial_root = temporary / INITIAL_ADAPTER_DIR
        initial_root.mkdir()
        save_file(factors, initial_root / "adapter_model.safetensors")
        _write_json(initial_root / "adapter_config.json", {
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank,
            "lora_alpha": alpha, "use_rslora": method == "rslora",
            "target_modules": list(PROJECTION_TARGETS), "bias": "none",
            "lora_dropout": 0.0, "init_lora_weights": True,
            "inference_mode": True, "modules_to_save": None,
        })
        residual_hashes = {name: file_sha256(temporary / name) for name in weight_files}
        initial_hash = file_sha256(initial_root / "adapter_model.safetensors")
        identity = hashlib.sha256(json.dumps(
            [settings, source_hashes, residual_hashes, initial_hash], sort_keys=True,
        ).encode()).hexdigest()
        config[SHARED_CONFIG_KEY] = identity
        _write_json(temporary / "config.json", config)
        inventory = {
            str(path.relative_to(temporary)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(temporary.rglob("*")) if path.is_file()
        }
        manifest = dict(
            format_version=1, mode="shared-residual", base_id=identity,
            settings=settings, source_path=str(source), source_sha256=source_hashes,
            files=inventory, weight_files=weight_files, projections=sorted(projections),
            dtype="bfloat16", torch_version=torch.__version__, peft_version=peft.__version__,
        )
        # Detect a source modified during preprocessing rather than bind mixed weights.
        if any(file_sha256(source / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("Source checkpoint changed during PiSSA preparation")
        _write_json(temporary / MANIFEST_FILE, manifest)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Residual destination appeared during preparation: {destination}")
        os.rename(temporary, destination)
        return manifest
    except BaseException:
        # Only remove the unique temporary directory created by this call.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
