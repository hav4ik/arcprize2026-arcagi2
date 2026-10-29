"""Strict checkpoint contract for the vLLM Delta-MH executors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


CONFIG_FILENAME = "delta_mhar_config.json"
WEIGHTS_FILENAME = "delta_mhar.safetensors"
FORMAT_VERSION = 1
SINGLE_SITE_MODE = "delta_mh_single_site"
SUPPORTED_MODEL_TYPES = {"qwen3", "qwen3_5_text"}


@dataclass(frozen=True)
class DeltaMHARSpec:
    model_path: Path
    model_type: str
    architecture: str
    mode: str
    num_heads: int
    block_size: int
    fused: bool
    format_version: int
    num_hidden_layers: int
    hidden_size: int
    rms_norm_eps: float
    routing_weight_paths: tuple[Path, ...]
    routing_weights_embedded: bool
    tensor_count: int

    @property
    def family(self) -> str:
        return "qwen3_5" if self.model_type == "qwen3_5_text" else "qwen3"

    @property
    def max_completed_blocks(self) -> int:
        return self.num_hidden_layers // self.block_size

    def declaration(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "num_heads": self.num_heads,
            "block_size": self.block_size,
            "fused": self.fused,
            "format_version": self.format_version,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read Delta-MH JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _parse_declaration(values: Mapping[str, Any], *, source: Path) -> dict[str, Any]:
    allowed = {"mode", "num_heads", "block_size", "fused", "format_version"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown Delta-MH fields in {source}: {sorted(unknown)}")
    parsed = {
        "mode": str(values.get("mode", "")),
        "num_heads": int(values.get("num_heads", 0)),
        "block_size": int(values.get("block_size", 0)),
        "fused": bool(values.get("fused", True)),
        "format_version": int(values.get("format_version", 0)),
    }
    if parsed["mode"] != SINGLE_SITE_MODE:
        raise ValueError(
            "vLLM routed inference currently supports only "
            f"{SINGLE_SITE_MODE!r}; {source} declares {parsed['mode']!r}"
        )
    if parsed["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Delta-MH format version in {source}: "
            f"{parsed['format_version']}"
        )
    if parsed["num_heads"] <= 0 or parsed["block_size"] <= 0:
        raise ValueError(f"Delta-MH heads and block size must be positive in {source}")
    return parsed


def expected_sidecar_shapes(
    *,
    num_hidden_layers: int,
    hidden_size: int,
    num_heads: int,
) -> dict[str, tuple[int, ...]]:
    if hidden_size % num_heads:
        raise ValueError(
            f"hidden size {hidden_size} is not divisible by {num_heads} routing heads"
        )
    head_size = hidden_size // num_heads
    expected: dict[str, tuple[int, ...]] = {}
    for layer_index in range(num_hidden_layers):
        prefix = f"model.layers.{layer_index}._unsloth_delta_mhar"
        expected.update(
            {
                f"{prefix}.attention.query": (num_heads, head_size),
                f"{prefix}.attention.norm_weight": (hidden_size,),
                f"{prefix}.attention.null_source": (hidden_size,),
                f"{prefix}.attention.alpha": (1,),
                f"{prefix}.carry.delta": (hidden_size,),
            }
        )
    return expected


def _validate_routing_weights(
    paths: tuple[Path, ...],
    *,
    num_hidden_layers: int,
    hidden_size: int,
    num_heads: int,
) -> int:
    import torch

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Delta-MH preflight requires safetensors") from error

    expected = expected_sidecar_shapes(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
    )
    actual: dict[str, Path] = {}
    invalid = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if "._unsloth_delta_mhar." not in name:
                    continue
                if name in actual:
                    raise RuntimeError(
                        f"Duplicate Delta-MH tensor {name!r} in "
                        f"{actual[name]} and {path}"
                    )
                actual[name] = path
                expected_shape = expected.get(name)
                if expected_shape is None:
                    continue
                tensor = checkpoint.get_slice(name)
                actual_shape = tuple(tensor.get_shape())
                dtype = str(tensor.get_dtype())
                if actual_shape != expected_shape or dtype != "F32":
                    invalid.append((name, actual_shape, dtype, expected_shape))
                    continue
                value = checkpoint.get_tensor(name)
                if not bool(torch.isfinite(value).all().item()):
                    invalid.append((name, actual_shape, "non-finite", expected_shape))
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            "Delta-MH routing-weight key mismatch; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if invalid:
        raise RuntimeError(
            f"Delta-MH routing weights contain invalid tensor metadata: {invalid[:8]}"
        )
    return len(expected)


def _embedded_routing_weight_paths(model_path: Path) -> tuple[Path, ...]:
    """Locate ordinary HF safetensor shards containing embedded router weights."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise RuntimeError(f"{index_path} must contain a weight_map object")
        filenames = {
            str(filename)
            for name, filename in weight_map.items()
            if "._unsloth_delta_mhar." in str(name)
        }
        paths = tuple(sorted(model_path / filename for filename in filenames))
    else:
        paths = tuple(sorted(model_path.glob("model*.safetensors")))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing routed HF safetensor shard(s): {missing}")
    if not paths:
        raise FileNotFoundError(
            f"Routed checkpoint requires {model_path / WEIGHTS_FILENAME} or "
            "embedded router weights in model*.safetensors"
        )
    return paths


def read_delta_mhar_spec(model_path: str | Path) -> DeltaMHARSpec:
    """Read and fully validate one local routed checkpoint."""
    model_path = Path(model_path).expanduser().resolve()
    hf_config_path = model_path / "config.json"
    sidecar_config_path = model_path / CONFIG_FILENAME
    sidecar_path = model_path / WEIGHTS_FILENAME
    if not hf_config_path.is_file():
        raise FileNotFoundError(hf_config_path)
    hf_config = _read_json(hf_config_path)
    hf_declaration_value = hf_config.get("unsloth_delta_mhar")
    if not isinstance(hf_declaration_value, Mapping):
        raise RuntimeError(f"{hf_config_path} must declare unsloth_delta_mhar")
    hf_declaration = _parse_declaration(hf_declaration_value, source=hf_config_path)
    if sidecar_config_path.is_file():
        sidecar_declaration = _parse_declaration(
            _read_json(sidecar_config_path), source=sidecar_config_path
        )
        if hf_declaration != sidecar_declaration:
            raise RuntimeError(
                "Delta-MH HF/sidecar declarations disagree: "
                f"hf={hf_declaration}, sidecar={sidecar_declaration}"
            )

    model_type = str(hf_config.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported routed model_type {model_type!r}; expected one of "
            f"{sorted(SUPPORTED_MODEL_TYPES)}"
        )
    architectures = hf_config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError(
            f"Expected one architecture in {hf_config_path}, got {architectures!r}"
        )
    architecture = str(architectures[0])
    expected_architecture = (
        "Qwen3_5ForCausalLM" if model_type == "qwen3_5_text" else "Qwen3ForCausalLM"
    )
    if architecture != expected_architecture:
        raise ValueError(
            f"Routed {model_type} checkpoint must use {expected_architecture}, "
            f"got {architecture}"
        )
    num_hidden_layers = int(hf_config.get("num_hidden_layers", 0))
    hidden_size = int(hf_config.get("hidden_size", 0))
    rms_norm_eps = float(hf_config.get("rms_norm_eps", 0.0))
    if num_hidden_layers <= 0 or hidden_size <= 0 or rms_norm_eps <= 0:
        raise ValueError(
            "Invalid routed model dimensions: "
            f"layers={num_hidden_layers}, hidden={hidden_size}, eps={rms_norm_eps}"
        )
    routing_weights_embedded = not sidecar_path.is_file()
    routing_weight_paths = (
        _embedded_routing_weight_paths(model_path)
        if routing_weights_embedded
        else (sidecar_path,)
    )
    tensor_count = _validate_routing_weights(
        routing_weight_paths,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_heads=hf_declaration["num_heads"],
    )
    return DeltaMHARSpec(
        model_path=model_path,
        model_type=model_type,
        architecture=architecture,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        rms_norm_eps=rms_norm_eps,
        routing_weight_paths=routing_weight_paths,
        routing_weights_embedded=routing_weights_embedded,
        tensor_count=tensor_count,
        **hf_declaration,
    )


def spec_from_hf_config(config: Any) -> dict[str, Any]:
    """Validate the already-loaded HF declaration used by model layers."""
    declaration = getattr(config, "unsloth_delta_mhar", None)
    if declaration is None and hasattr(config, "to_dict"):
        declaration = config.to_dict().get("unsloth_delta_mhar")
    if not isinstance(declaration, Mapping):
        raise RuntimeError("vLLM routed model received no Delta-MH declaration")
    return _parse_declaration(declaration, source=Path("config.json"))


__all__ = [
    "CONFIG_FILENAME",
    "DeltaMHARSpec",
    "FORMAT_VERSION",
    "SINGLE_SITE_MODE",
    "WEIGHTS_FILENAME",
    "expected_sidecar_shapes",
    "read_delta_mhar_spec",
    "spec_from_hf_config",
]
