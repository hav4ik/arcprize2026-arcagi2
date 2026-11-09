"""Process compatibility helpers; safe to import before Torch."""

from __future__ import annotations

import json
import os
import importlib.util
from functools import wraps
from pathlib import Path
import sys
from typing import Callable, Mapping, MutableMapping, Sequence


TE_CPP_RUNTIME_MARKER = "ARC_TE_CPP_RUNTIME_BOOTSTRAPPED"
TE_CPP_RUNTIME_OVERRIDE = "ARC_TE_CPP_RUNTIME_DIR"
DEFAULT_TE_CPP_RUNTIME_DIRS = (
    "/usr/local/lib/julia",
    "/opt/nvidia/nsight-compute/2025.3.1/host/linux-desktop-glibc_2_11_3-x64",
)
VLLM_COMPACT_VOCAB_WARMUP_MARKER = "ARC compact-vocab sampler warmup"
VLLM_DIFFUSION_GEMMA_CAUSAL_MARKER = (
    "ARC DiffusionGemma CUDA-graph causal-buffer hotfix"
)


def install_vllm_token_only_renderer_warmup(renderer_class=None) -> bool:
    """Skip unused chat-template warmup when vLLM has no tokenizer.

    This is a process-local patch: no installed files are rewritten, and GPU
    model warmup/capture are untouched. Keep modern background/MM warmup and
    its cleanup; unknown or legacy multimodal layouts use the original method.
    """
    if renderer_class is None:
        try:
            from vllm.renderers.base import BaseRenderer as renderer_class
        except ModuleNotFoundError as exc:
            if exc.name in {"vllm.renderers", "vllm.renderers.base"}:
                return False  # Older vLLM has no separate renderer warmup.
            raise
    original = getattr(renderer_class, "warmup", None)
    if original is None or getattr(original, "_arc_token_only_warmup", False):
        return False

    @wraps(original)
    def warmup(self, *args, **kwargs):
        if hasattr(self, "tokenizer") and self.tokenizer is None:
            join_mm = getattr(self, "_join_mm_warmup", None)
            warmup_mm = getattr(self, "warmup_mm", None)
            if callable(join_mm) and callable(warmup_mm):
                join_mm()
                return warmup_mm()
            # Legacy vLLM has inline MM warmup. Skip the method only when
            # both known processors and any background task are absent.
            if (hasattr(self, "mm_processor")
                    and self.mm_processor is None
                    and hasattr(self, "_readonly_mm_processor")
                    and self._readonly_mm_processor is None
                    and getattr(self, "_mm_warmup_future", None) is None):
                return None
        return original(self, *args, **kwargs)

    warmup._arc_token_only_warmup = True
    renderer_class.warmup = warmup
    return True


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def fp8_frozen_linear_requested(
    argv: Sequence[str],
    environment: Mapping[str, str],
) -> bool:
    """Resolve the early FP8 flag without importing the full CLI module."""
    requested = _enabled(environment.get("ARC_FP8_FROZEN_LINEAR"))
    if "--fp8-frozen-linear" in argv:
        requested = True
    if "--no-fp8-frozen-linear" in argv:
        requested = False
    # Compact DiffusionGemma always uses the frozen-FP8 TTT path. This family
    # invariant is applied by starter.py after argument parsing, which is too
    # late for the dynamic loader: recent Transformer Engine wheels need the
    # compatible C++ runtime before Torch is imported. Detect the lightweight
    # config here so users do not need a redundant ARC_FP8_FROZEN_LINEAR=1.
    model_path = environment.get("ARC_MODEL_PATH")
    if model_path:
        config_path = Path(model_path).expanduser() / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            config = {}
        architectures = {str(value) for value in config.get("architectures", ())}
        model_types = {
            str(config.get("model_type", "")),
            str((config.get("text_config") or {}).get("model_type", "")),
        }
        if (
            "DiffusionGemmaForBlockDiffusion" in architectures
            or "DiffusionGemmaForConditionalGeneration" in architectures
            or "diffusion_gemma" in model_types
            or "diffusion_gemma_text" in model_types
        ):
            requested = True
    return requested


def _compact_diffusion_gemma_vocab_size(
    environment: Mapping[str, str],
) -> int | None:
    """Return the compact text vocabulary size without importing Transformers."""
    model_path = environment.get("ARC_MODEL_PATH")
    if not model_path:
        return None
    try:
        config = json.loads(
            (Path(model_path).expanduser() / "config.json").read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None
    architectures = {str(value) for value in config.get("architectures", ())}
    text_config = config.get("text_config") or {}
    model_types = {
        str(config.get("model_type", "")),
        str(text_config.get("model_type", "")),
    }
    is_diffusion_gemma = (
        "DiffusionGemmaForBlockDiffusion" in architectures
        or "DiffusionGemmaForConditionalGeneration" in architectures
        or "diffusion_gemma" in model_types
        or "diffusion_gemma_text" in model_types
    )
    if not is_diffusion_gemma:
        return None
    vocab_size = int(text_config.get("vocab_size", 0) or 0)
    return vocab_size if 0 < vocab_size < 1024 else None


def _diffusion_gemma_requested(environment: Mapping[str, str]) -> bool:
    """Detect a DiffusionGemma model without importing its runtime."""
    model_path = environment.get("ARC_MODEL_PATH")
    if not model_path:
        return False
    try:
        config = json.loads(
            (Path(model_path).expanduser() / "config.json").read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    architectures = {str(value) for value in config.get("architectures", ())}
    text_config = config.get("text_config") or {}
    model_types = {
        str(config.get("model_type", "")),
        str(text_config.get("model_type", "")),
    }
    return (
        "DiffusionGemmaForBlockDiffusion" in architectures
        or "DiffusionGemmaForConditionalGeneration" in architectures
        or "diffusion_gemma" in model_types
        or "diffusion_gemma_text" in model_types
    )


def _resolve_vllm_package_root(package_root: str | Path | None) -> Path:
    if package_root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot locate the installed vLLM package")
        return Path(spec.origin).resolve().parent
    resolved = Path(package_root).resolve()
    if resolved.name != "vllm" and (resolved / "vllm").is_dir():
        resolved /= "vllm"
    return resolved


def _write_runtime_sources(sources: Mapping[Path, str]) -> None:
    """Validate and atomically replace a small set of Python sources."""
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for path, source in sources.items():
            compile(source, str(path), "exec")
            temporary = path.with_suffix(path.suffix + ".arc-tmp")
            temporary.write_text(source, encoding="utf-8")
            os.chmod(temporary, path.stat().st_mode)
            temporary_paths.append((path, temporary))
        for path, temporary in temporary_paths:
            temporary.replace(path)
    except OSError as exc:
        for _, temporary in temporary_paths:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise RuntimeError("Cannot install vLLM runtime source hotfix") from exc


def patch_vllm_diffusion_gemma_cudagraph_causal_buffer(
    *,
    environment: Mapping[str, str] | None = None,
    package_root: str | Path | None = None,
) -> bool:
    """Backport vLLM PR #51994 before importing vLLM.

    DiffusionGemma alternates causal encoder passes and bidirectional denoising
    passes through a persistent per-request tensor. vLLM currently allocates
    that tensor as bool, then FlashAttention casts it out of place to int32.
    FULL CUDA graphs retain the cast tensor's capture-time address, silently
    freezing the attention mode during replay. Keep the producer int32 and
    reject future out-of-place casts, matching the reviewed upstream fix.
    """
    environment = os.environ if environment is None else environment
    if not _diffusion_gemma_requested(environment):
        return False

    package = _resolve_vllm_package_root(package_root)
    model_path = package / "model_executor/models/diffusion_gemma.py"
    attention_path = package / "v1/attention/backends/flash_attn.py"
    try:
        original_model = model_path.read_text(encoding="utf-8")
        original_attention = attention_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Cannot read vLLM DiffusionGemma CUDA-graph hotfix targets"
        ) from exc
    model_source = original_model
    attention_source = original_attention

    changed: dict[Path, str] = {}
    bool_buffer = "self.max_num_reqs, dtype=torch.bool, device=device"
    int32_buffer = "self.max_num_reqs, dtype=torch.int32, device=device"
    if bool_buffer in model_source:
        model_source = model_source.replace(bool_buffer, int32_buffer, 1)
        marker_anchor = (
            "        # Persistent buffer for per-request causal flags, updated "
            "in-place\n"
        )
        if marker_anchor in model_source:
            model_source = model_source.replace(
                marker_anchor,
                marker_anchor
                + "        # "
                + VLLM_DIFFUSION_GEMMA_CAUSAL_MARKER
                + ".\n",
                1,
            )
    elif int32_buffer not in model_source:
        raise RuntimeError(
            "Unsupported vLLM DiffusionGemma causal-buffer allocation; "
            f"refusing to patch unknown source: {model_path}"
        )

    vulnerable_update = (
        "        self._causal_buf[:actual_num_reqs] = "
        "self.diffusion_states.is_encoder_phase[\n"
        "            slots\n"
        "        ]\n"
    )
    safe_update = (
        "        self._causal_buf[:actual_num_reqs].copy_(\n"
        "            self.diffusion_states.is_encoder_phase[slots]\n"
        "        )\n"
    )
    if vulnerable_update in model_source:
        model_source = model_source.replace(vulnerable_update, safe_update, 1)
    elif safe_update not in model_source:
        raise RuntimeError(
            "Unsupported vLLM DiffusionGemma causal-buffer update; "
            f"refusing to patch unknown source: {model_path}"
        )
    vulnerable_padding = (
        "self._causal_buf[actual_num_reqs:num_reqs] = False"
    )
    if vulnerable_padding in model_source:
        model_source = model_source.replace(
            vulnerable_padding,
            "self._causal_buf[actual_num_reqs:num_reqs] = 0",
            1,
        )

    vulnerable_cast = (
        "        if isinstance(causal, torch.Tensor) and causal.dtype != "
        "torch.int32:\n"
        "            causal = causal.to(torch.int32)\n"
    )
    safe_cast = (
        "        if isinstance(causal, torch.Tensor) and causal.dtype != "
        "torch.int32:\n"
        "            raise ValueError(\n"
        "                f\"Per-request causal tensor must be int32, got "
        "{causal.dtype}. \"\n"
        "                \"Casting here would allocate a fresh tensor each "
        "build() and \"\n"
        "                \"break FULL CUDA graph replay; allocate the buffer "
        "as int32.\"\n"
        "            )\n"
    )
    if vulnerable_cast in attention_source:
        attention_source = attention_source.replace(
            vulnerable_cast,
            safe_cast,
            1,
        )
    elif "Per-request causal tensor must be int32" not in attention_source:
        raise RuntimeError(
            "Unsupported vLLM FlashAttention causal-tensor conversion; "
            f"refusing to patch unknown source: {attention_path}"
        )

    if model_source != original_model:
        changed[model_path] = model_source
    if attention_source != original_attention:
        changed[attention_path] = attention_source
    if not changed:
        return False

    _write_runtime_sources(changed)
    print(
        "[starter] patched vLLM DiffusionGemma CUDA-graph causal buffer "
        "(upstream PR #51994)",
        flush=True,
    )
    return True


def patch_vllm_compact_vocab_sampler_warmup(
    *,
    environment: Mapping[str, str] | None = None,
    package_root: str | Path | None = None,
) -> bool:
    """Keep vLLM's synthetic warmup IDs inside a compact model vocabulary.

    Recent vLLM sampler warmup constructs ``range(decode_query_len + 1)`` as
    dummy token IDs. DiffusionGemma uses a 256-token diffusion canvas, so a
    deliberately compact 31-token ARC vocabulary receives IDs up through 256
    and fails in its embedding gather before the engine can accept a request.
    Patch only that synthetic list; real request validation and logits remain
    unchanged.
    """
    environment = os.environ if environment is None else environment
    if _compact_diffusion_gemma_vocab_size(environment) is None:
        return False

    package_root = _resolve_vllm_package_root(package_root)

    warmup_path = package_root / "v1/worker/gpu/warmup.py"
    try:
        source = warmup_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read vLLM sampler warmup source: {warmup_path}"
        ) from exc
    if VLLM_COMPACT_VOCAB_WARMUP_MARKER in source:
        return False

    vulnerable = "    prompt_token_ids = list(range(prompt_len))\n"
    replacement = (
        "    # ARC compact-vocab sampler warmup: synthetic IDs must remain in "
        "the model vocabulary.\n"
        "    warmup_vocab_size = model_runner.model_config.get_vocab_size()\n"
        "    prompt_token_ids = [\n"
        "        token_id % warmup_vocab_size for token_id in range(prompt_len)\n"
        "    ]\n"
    )
    if source.count(vulnerable) != 1:
        # Accept an equivalent upstream repair rather than rewriting unknown
        # source. The modulo expression is intentionally specific enough not
        # to confuse unrelated warmup code with this correctness guard.
        if "token_id %" in source and "range(prompt_len)" in source:
            return False
        raise RuntimeError(
            "Unsupported vLLM sampler warmup source for a compact "
            f"DiffusionGemma checkpoint: {warmup_path}"
        )
    patched = source.replace(vulnerable, replacement, 1)
    temporary = warmup_path.with_suffix(".py.arc-tmp")
    try:
        temporary.write_text(patched, encoding="utf-8")
        os.chmod(temporary, warmup_path.stat().st_mode)
        temporary.replace(warmup_path)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Cannot install compact-vocab vLLM warmup patch: {warmup_path}"
        ) from exc
    print(
        "[starter] patched vLLM sampler warmup for compact DiffusionGemma vocab",
        flush=True,
    )
    return True


def find_te_cpp_runtime_dir(
    environment: Mapping[str, str],
    candidates: Sequence[str] = DEFAULT_TE_CPP_RUNTIME_DIRS,
) -> str | None:
    """Find an image-provided libstdc++ suitable for recent TE wheels."""
    override = environment.get(TE_CPP_RUNTIME_OVERRIDE)
    search = (override, *candidates) if override else tuple(candidates)
    for candidate in search:
        directory = Path(candidate).expanduser()
        if (directory / "libstdc++.so.6").is_file():
            return str(directory.resolve())
    return None


def find_te_cuda_wheel_runtime_dirs(
    search_paths: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Find CUDA component libraries installed as split NVIDIA wheels.

    Recent Kaggle images can carry an older NCCL/cuDNN in the image-wide CUDA
    path while Torch and Transformer Engine were upgraded from wheels.  The
    wheel directories must precede the image directories at process start;
    changing ``LD_LIBRARY_PATH`` after importing Torch is too late.
    """
    roots = tuple(sys.path if search_paths is None else search_paths)
    by_component: dict[str, str] = {}
    for root in roots:
        if not root:
            continue
        nvidia = Path(root).expanduser() / "nvidia"
        if not nvidia.is_dir():
            continue
        for library_dir in nvidia.glob("*/lib"):
            if library_dir.is_dir():
                by_component.setdefault(
                    library_dir.parent.name,
                    str(library_dir.resolve()),
                )
    # Torch fails at import when an old NCCL wins symbol resolution; TE then
    # fails when mixed cuDNN component versions are selected.  Put both first,
    # followed by the remaining matching wheel components in stable order.
    priority = ("nccl", "cudnn")
    ordered = [by_component[name] for name in priority if name in by_component]
    ordered.extend(
        by_component[name]
        for name in sorted(by_component)
        if name not in priority
    )
    return tuple(dict.fromkeys(ordered))


def bootstrap_transformer_engine_cpp_runtime(
    *,
    argv: Sequence[str] | None = None,
    environment: MutableMapping[str, str] | None = None,
    executable: str | None = None,
    execvpe: Callable = os.execvpe,
    force: bool = False,
    cuda_wheel_runtime_dirs: Sequence[str] | None = None,
) -> bool:
    """Re-exec FP8 runs with a TE-compatible C++ runtime when available.

    Newer prebuilt Transformer Engine extensions can require a newer GLIBCXX
    symbol set than Kaggle's Ubuntu default.  Some Kaggle images already ship
    a compatible runtime with Julia.  ``LD_LIBRARY_PATH`` is consumed by the
    dynamic loader at process start, so the safe fix is one early re-exec,
    before Torch or any C++ extension has been imported.
    """
    argv = tuple(sys.argv if argv is None else argv)
    environment = os.environ if environment is None else environment
    if not force and not fp8_frozen_linear_requested(argv, environment):
        return False
    if _enabled(environment.get(TE_CPP_RUNTIME_MARKER)):
        return False

    runtime_dir = find_te_cpp_runtime_dir(environment)
    if runtime_dir is None:
        return False

    wheel_runtime_dirs = tuple(
        find_te_cuda_wheel_runtime_dirs()
        if cuda_wheel_runtime_dirs is None
        else cuda_wheel_runtime_dirs
    )
    desired_prefix = tuple(
        dict.fromkeys((runtime_dir, *wheel_runtime_dirs))
    )
    existing = tuple(
        entry for entry in environment.get("LD_LIBRARY_PATH", "").split(":") if entry
    )
    if existing[: len(desired_prefix)] == desired_prefix:
        environment[TE_CPP_RUNTIME_MARKER] = "1"
        return False

    relaunched_environment = dict(environment)
    relaunched_environment["LD_LIBRARY_PATH"] = ":".join(
        (
            *desired_prefix,
            *(entry for entry in existing if entry not in desired_prefix),
        )
    )
    relaunched_environment[TE_CPP_RUNTIME_MARKER] = "1"
    python = executable or sys.executable
    print(
        "[starter] re-executing FP8 run with Transformer Engine C++ runtime "
        f"from {runtime_dir}",
        flush=True,
    )
    execvpe(python, [python, *argv], relaunched_environment)
    return True
