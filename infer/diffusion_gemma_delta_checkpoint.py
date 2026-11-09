#!/usr/bin/env python3
"""Pack and reconstruct exact DiffusionGemma Trainer checkpoints.

The base-model portion is represented as a byte-shuffled XOR against the
immutable Hugging Face base artifact.  XOR operates on tensor bit patterns,
not floating-point values, so reconstruction is exact even for BF16 tensors.
Optimizer, scheduler, Trainer, RNG, and ARC data-state files are copied without
loss into the compressed archive.

Version 2 archives may reference a previous delta checkpoint. Supply every
ancestor oldest-first; identities bind the chain and every tensor is verified
at each hop. Ancestors contribute weights only, never optimizer or data state.

Direct archives use a single continuous compressor stream. Optional directory
staging encodes once on disk and reuses that whole delta for codec retries.
Packing uses one temporary model shard, and reconstruction consumes one delta
shard at a time instead of first
extracting a second full checkpoint.  Gzip inputs use parallel ``pigz``
decompression when available, while Zstandard inputs use the external
``zstd`` codec, including Kaggle-safe ``.tar.gz.xx`` and ``.tar.zst.xx`` names.
Reconstruction can also emit a weight-only Hugging Face artifact while
verifying and discarding all saved run state, which supports fresh optimizer
and data schedules from archived model weights.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DELTA_FORMAT = "nvarc-exact-model-delta-v1"
CHAINED_DELTA_FORMAT = "nvarc-exact-model-delta-v2"
DELTA_ROOT = "_arc_exact_model_delta"
DELTA_MANIFEST_NAME = f"{DELTA_ROOT}/manifest.json"
DELTA_STAGING_NAME = f"{DELTA_ROOT}/staging.json"
MODEL_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.safetensors.index.json",
)
SINGLE_MODEL_NAMES = ("model.safetensors", "pytorch_model.safetensors")
# Repository transport metadata is not part of the model's runtime identity.
# Keep this exception narrow: tokenizer/config/vocabulary files remain exact.
IGNORED_BASE_AUXILIARY_FILES = frozenset({".gitattributes"})
COPY_CHUNK_BYTES = 8 * 1024 * 1024
ARCHIVE_COPY_CHUNK_BYTES = 32 * 1024 * 1024
ARCHIVE_COPY_FREE_MARGIN_BYTES = 64 * 1024 * 1024
DEFAULT_DELTA_SHARD_BYTES = 1024 * 1024 * 1024
DEFAULT_RECONSTRUCT_WORKERS = 1
XZ_PRESETS = tuple(str(level) for level in range(10)) + tuple(
    f"{level}e" for level in range(10)
)
DELTA_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tgz",
    ".tar.zst",
    ".tar.zstd",
    ".tzst",
    ".tar.xz",
    ".txz",
    ".tar",
)


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(COPY_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _remove_directory_contents(
    directory: Path,
    *,
    expected_directory: Path,
) -> list[str]:
    """Remove one explicitly confirmed directory's children without following links."""
    directory = directory.expanduser().resolve()
    expected_directory = expected_directory.expanduser().resolve()
    if directory != expected_directory:
        raise RuntimeError(
            "refusing cleanup because the resolved directory changed: "
            f"requested={directory}, expected={expected_directory}"
        )
    if directory in {Path("/"), Path.home().resolve()}:
        raise RuntimeError(f"refusing broad filesystem cleanup: {directory}")
    if not directory.is_dir():
        raise FileNotFoundError(f"cleanup directory not found: {directory}")

    removed: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda path: path.name):
        if child.is_mount():
            raise RuntimeError(f"refusing to remove mounted path: {child}")
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise RuntimeError(f"refusing to remove special filesystem entry: {child}")
        removed.append(child.name)
    return removed


def cleanup_delta_export_dataset_cache(
    cache_root: str | Path,
    shuffle_file: str | Path,
    *,
    protected_paths: list[str | Path],
    allowed_roots: tuple[str | Path, ...] = ("/dev/shm", "/kaggle/tmp"),
) -> dict[str, Any]:
    """Remove only a validated disposable ARC dataset cache after training ends."""
    allowed = [Path(path).expanduser().resolve() for path in allowed_roots]
    protected = [Path(path).expanduser().resolve() for path in protected_paths if path]

    def validate(path: Path) -> Path:
        absolute = Path(os.path.abspath(path.expanduser()))
        resolved = absolute.resolve()
        if resolved != absolute:
            raise ValueError(f"refusing cache cleanup through a symlink: {absolute}")
        if not any(resolved != root and resolved.is_relative_to(root) for root in allowed):
            raise ValueError(f"disposable cache must be strictly below a temporary root: {resolved}")
        for preserve in protected:
            if resolved.is_relative_to(preserve) or preserve.is_relative_to(resolved):
                raise ValueError(f"cache cleanup overlaps protected path: {resolved}, {preserve}")
        return resolved

    root = validate(Path(cache_root))
    shuffle = validate(Path(shuffle_file))
    targets = []
    if root.exists():
        if not root.is_dir() or not any(
            child.name.startswith("diffusion-gemma-arc") and child.is_dir() and not child.is_symlink()
            for child in root.iterdir()
        ):
            raise ValueError(f"refusing cleanup of unrecognized DiffusionGemma cache: {root}")
        targets.append(root)
    if not shuffle.is_relative_to(root) and shuffle.exists():
        if shuffle.suffix != ".arrow" or not shuffle.is_file():
            raise ValueError(f"refusing cleanup of unrecognized shuffle file: {shuffle}")
        targets.append(shuffle)
    # Validate all targets before deleting anything. Never follow child symlinks.
    for target in targets:
        if target.is_dir() and any(child.is_mount() for child in target.rglob("*")):
            raise ValueError(f"refusing cleanup of cache with mounted children: {target}")
    removed = []
    for target in targets:
        size = (
            sum(child.stat().st_size for child in target.rglob("*") if child.is_file() and not child.is_symlink())
            if target.is_dir() else target.stat().st_size
        )
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append({"path": str(target), "bytes": size})
        print(f"[DiffusionGemma] removed disposable dataset cache: {target} ({size:,} bytes; regenerated on resume)", flush=True)
    return {"removed": removed, "removed_bytes": sum(item["bytes"] for item in removed)}


def publish_staged_delta_archive(
    staged_archive: str | Path,
    final_archive: str | Path,
    *,
    clean_directory: str | Path,
    expected_clean_directory: str | Path,
    free_margin_bytes: int = ARCHIVE_COPY_FREE_MARGIN_BYTES,
) -> dict[str, Any]:
    """Clean an approved output volume and atomically publish a staged archive.

    The complete compressed archive must already exist outside ``clean_directory``.
    This ordering keeps the staged recovery artifact intact if cleanup or copying
    fails.  A read-back SHA-256 validates the cross-filesystem copy before the
    hidden partial file is atomically renamed to its final name.
    """
    staged_archive = Path(staged_archive).expanduser().resolve()
    final_archive = Path(final_archive).expanduser().resolve()
    clean_directory = Path(clean_directory).expanduser().resolve()
    expected_clean_directory = Path(expected_clean_directory).expanduser().resolve()
    free_margin_bytes = int(free_margin_bytes)
    if free_margin_bytes < 0:
        raise ValueError("archive copy free-space margin cannot be negative")
    if not staged_archive.is_file():
        raise FileNotFoundError(f"staged delta archive not found: {staged_archive}")
    if staged_archive.is_relative_to(clean_directory):
        raise ValueError(
            "staged delta archive must be outside the directory being cleaned"
        )
    if final_archive.parent != clean_directory:
        raise ValueError(
            "final delta archive must be a direct child of the cleaned directory"
        )
    if final_archive.exists():
        raise FileExistsError(f"refusing to overwrite delta archive: {final_archive}")

    partial = final_archive.with_name(
        f".{final_archive.name}.{os.getpid()}.partial"
    )
    if partial.exists():
        raise FileExistsError(f"partial archive already exists: {partial}")
    archive_bytes = staged_archive.stat().st_size
    usage_before = shutil.disk_usage(clean_directory)
    free_before = usage_before.free
    required_bytes = archive_bytes + free_margin_bytes
    if usage_before.total < required_bytes:
        raise OSError(
            28,
            "staged delta archive cannot fit even on an empty output volume: "
            f"archive={archive_bytes / 1024**3:.2f}GiB, "
            f"volume={usage_before.total / 1024**3:.2f}GiB, "
            f"margin={free_margin_bytes / 1024**2:.0f}MiB; "
            f"working directory preserved and staged archive retained at "
            f"{staged_archive}",
            str(final_archive),
        )

    try:
        current_directory = Path.cwd().resolve()
    except FileNotFoundError:
        current_directory = None
    if current_directory is not None and (
        current_directory == clean_directory
        or current_directory.is_relative_to(clean_directory)
    ):
        os.chdir(clean_directory.parent)

    removed = _remove_directory_contents(
        clean_directory,
        expected_directory=expected_clean_directory,
    )
    free_after_cleanup = shutil.disk_usage(clean_directory).free
    if free_after_cleanup < required_bytes:
        raise OSError(
            28,
            "clean output volume is too small for staged delta archive: "
            f"archive={archive_bytes / 1024**3:.2f}GiB, "
            f"free={free_after_cleanup / 1024**3:.2f}GiB, "
            f"margin={free_margin_bytes / 1024**2:.0f}MiB; "
            f"staged archive retained at {staged_archive}",
            str(final_archive),
        )

    source_digest = hashlib.sha256()
    try:
        with staged_archive.open("rb") as source, partial.open("xb") as output:
            while chunk := source.read(ARCHIVE_COPY_CHUNK_BYTES):
                source_digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        copied_bytes = partial.stat().st_size
        if copied_bytes != archive_bytes:
            raise RuntimeError(
                "staged delta archive copy size mismatch: "
                f"source={archive_bytes}, copied={copied_bytes}"
            )
        source_sha256 = source_digest.hexdigest()
        copied_sha256 = _sha256_file(partial)
        if copied_sha256 != source_sha256:
            raise RuntimeError(
                "staged delta archive copy checksum mismatch: "
                f"source={source_sha256}, copied={copied_sha256}"
            )
        os.replace(partial, final_archive)
        directory_fd = os.open(clean_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return {
        "staged_archive": str(staged_archive),
        "archive": str(final_archive),
        "archive_bytes": archive_bytes,
        "archive_sha256": source_sha256,
        "clean_directory": str(clean_directory),
        "removed_entries": removed,
        "free_bytes_before_cleanup": free_before,
        "free_bytes_after_cleanup": free_after_cleanup,
        "free_margin_bytes": free_margin_bytes,
    }


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).reshape(-1)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    raw = _tensor_bytes(tensor)
    digest = hashlib.sha256()
    digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise TypeError(f"unsupported tensor dtype name: {name}")
    return name.removeprefix("torch.")


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported checkpoint tensor dtype: {name!r}")
    return dtype


def _byte_shuffle(raw: torch.Tensor, element_size: int) -> torch.Tensor:
    """Group byte lanes together so gzip/xz can exploit zero high bytes."""
    raw = raw.contiguous().reshape(-1)
    if element_size <= 1:
        return raw.clone()
    if raw.numel() % element_size:
        raise ValueError(
            f"raw byte count {raw.numel()} is not divisible by {element_size}"
        )
    return (
        raw.reshape(-1, element_size)
        .transpose(0, 1)
        .contiguous()
        .reshape(-1)
    )


def _byte_unshuffle(encoded: torch.Tensor, element_size: int) -> torch.Tensor:
    encoded = encoded.contiguous().reshape(-1)
    if element_size <= 1:
        return encoded.clone()
    if encoded.numel() % element_size:
        raise ValueError(
            f"encoded byte count {encoded.numel()} is not divisible by "
            f"{element_size}"
        )
    return (
        encoded.reshape(element_size, -1)
        .transpose(0, 1)
        .contiguous()
        .reshape(-1)
    )


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise RuntimeError(f"unsafe archive path: {value!r}")
    return Path(*pure.parts)


class SafeTensorLayout:
    """Tensor-to-shard mapping for a Hugging Face safetensors directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"safetensors directory not found: {self.root}")
        self.index_name: str | None = None
        self.weight_map: dict[str, str] = {}
        for name in MODEL_INDEX_NAMES:
            path = self.root / name
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                weight_map = payload.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise RuntimeError(f"invalid weight map: {path}")
                self.index_name = name
                self.weight_map = {
                    str(key): str(value) for key, value in weight_map.items()
                }
                break
        if not self.weight_map:
            for name in SINGLE_MODEL_NAMES:
                path = self.root / name
                if path.is_file():
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        self.weight_map = {key: name for key in handle.keys()}
                    break
        if not self.weight_map:
            raise FileNotFoundError(
                f"no Hugging Face model safetensors found in {self.root}"
            )
        self.files = sorted(set(self.weight_map.values()))
        missing = [name for name in self.files if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing safetensors shard(s) below {self.root}: {missing}"
            )

    def keys_for_file(self, filename: str) -> list[str]:
        return sorted(
            key for key, shard in self.weight_map.items() if shard == filename
        )


class SafeTensorReader:
    def __init__(self, layout: SafeTensorLayout, *, max_open_files: int = 2):
        self.layout = layout
        self.max_open_files = int(max_open_files)
        if self.max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self._handles: OrderedDict[str, tuple[Any, Any]] = OrderedDict()

    def __enter__(self) -> "SafeTensorReader":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        while self._handles:
            _, (context, _) = self._handles.popitem(last=False)
            context.__exit__(*exc_info)

    def _handle(self, filename: str):
        existing = self._handles.pop(filename, None)
        if existing is not None:
            self._handles[filename] = existing
            return existing[1]
        context = safe_open(
            self.layout.root / filename,
            framework="pt",
            device="cpu",
        )
        handle = context.__enter__()
        self._handles[filename] = (context, handle)
        while len(self._handles) > self.max_open_files:
            _, (old_context, _) = self._handles.popitem(last=False)
            old_context.__exit__(None, None, None)
        return handle

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handle(self.layout.weight_map[key]).get_tensor(key)

    def metadata(self, filename: str) -> dict[str, str] | None:
        metadata = self._handle(filename).metadata()
        return dict(metadata) if metadata is not None else None


def _artifact_identity(root: Path, layout: SafeTensorLayout) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "safetensors_files": [
            {
                "name": filename,
                "bytes": (root / filename).stat().st_size,
            }
            for filename in layout.files
        ],
        "tensor_count": len(layout.weight_map),
        "auxiliary_files": [
            {
                "name": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path, relative in _base_auxiliary_files(root, layout)
        ],
    }
    for name in ("config.json", *MODEL_INDEX_NAMES):
        path = root / name
        if path.is_file():
            identity[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return identity


def _non_model_files(
    checkpoint: Path, model_files: set[str]
) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"checkpoint symlinks are not supported: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(checkpoint).as_posix()
        if (
            relative in model_files
            or relative.startswith(f"{DELTA_ROOT}/")
            or relative == "arc_delta_reconstruction.json"
        ):
            continue
        result.append((path, relative))
    return result


def _base_auxiliary_files(
    base_model: Path,
    layout: SafeTensorLayout,
) -> list[tuple[Path, str]]:
    """Return immutable runtime files that are not model-weight layout files.

    A resumed Trainer checkpoint is not necessarily a complete Hugging Face
    artifact. In particular, our compact DiffusionGemma base carries the ARC
    vocabulary contract beside the tokenizer, while ``Trainer.save_model``
    does not reproduce that manifest. These files must follow the immutable
    base into a reconstruction unless the checkpoint archive explicitly
    replaces the same relative path.

    Base safetensors and their index are deliberately excluded. Copying the
    base index beside a reconstructed consolidated ``model.safetensors`` would
    make Transformers resolve the removed base shards instead of the restored
    trained file. Root ``.gitattributes`` is Git transport metadata, not a
    runtime file, and may be absent when the same model is mounted on Kaggle.
    """
    model_files = {
        *layout.files,
        *MODEL_INDEX_NAMES,
        *SINGLE_MODEL_NAMES,
    }
    result: list[tuple[Path, str]] = []
    for path in sorted(base_model.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"base-model symlinks are not supported: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(base_model).as_posix()
        if (
            relative in model_files
            or relative in IGNORED_BASE_AUXILIARY_FILES
            or path.suffix == ".safetensors"
            or relative.startswith(f"{DELTA_ROOT}/")
            or relative == "arc_delta_reconstruction.json"
        ):
            continue
        result.append((path, relative))
    return result


def _copy_base_auxiliary_files(
    base_model: Path,
    layout: SafeTensorLayout,
    destination: Path,
    *,
    replaced_paths: set[str],
) -> list[str]:
    copied: list[str] = []
    for source, relative in _base_auxiliary_files(base_model, layout):
        if relative in replaced_paths:
            continue
        target = destination / _safe_relative_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_handle, target.open("xb") as output:
            shutil.copyfileobj(input_handle, output, COPY_CHUNK_BYTES)
        if _sha256_file(target) != _sha256_file(source):
            raise RuntimeError(
                f"base auxiliary file changed while copying: {relative}"
            )
        copied.append(relative)
    return copied


def _build_manifest(
    checkpoint: Path,
    base_model: Path,
    *,
    max_delta_shard_bytes: int,
) -> tuple[dict[str, Any], SafeTensorLayout, SafeTensorLayout]:
    trained_layout = SafeTensorLayout(checkpoint)
    base_layout = SafeTensorLayout(base_model)
    model_files = set(trained_layout.files)
    state_files = _non_model_files(checkpoint, model_files)
    state_manifest = {
        relative: {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path, relative in state_files
    }
    tensor_manifest: dict[str, Any] = {}
    trained_file_manifest: list[dict[str, Any]] = []
    with SafeTensorReader(trained_layout) as trained, SafeTensorReader(
        base_layout
    ) as base:
        for filename in trained_layout.files:
            keys = trained_layout.keys_for_file(filename)
            trained_file_manifest.append(
                {
                    "trained_file": filename,
                    "metadata": trained.metadata(filename),
                    "tensors": keys,
                }
            )
            for key in keys:
                tensor = trained.get_tensor(key)
                dtype = _dtype_name(tensor.dtype)
                shape = list(tensor.shape)
                element_size = tensor.element_size()
                entry: dict[str, Any] = {
                    "shape": shape,
                    "dtype": dtype,
                    "element_size": element_size,
                    "bytes": tensor.numel() * element_size,
                    "trained_sha256": _sha256_tensor(tensor),
                    "encoding": "raw-byte-shuffle",
                    "trained_file": filename,
                }
                if key in base_layout.weight_map:
                    base_tensor = base.get_tensor(key)
                    if (
                        base_tensor.dtype == tensor.dtype
                        and tuple(base_tensor.shape) == tuple(tensor.shape)
                    ):
                        entry["encoding"] = "base-xor-byte-shuffle"
                        entry["base_sha256"] = _sha256_tensor(base_tensor)
                tensor_manifest[key] = entry
                del tensor
    delta_shards: list[dict[str, Any]] = []
    current_keys: list[str] = []
    current_bytes = 0
    ordered_keys = [
        key
        for item in trained_file_manifest
        for key in item["tensors"]
    ]
    for key in ordered_keys:
        tensor_bytes = int(tensor_manifest[key]["bytes"])
        if current_keys and current_bytes + tensor_bytes > max_delta_shard_bytes:
            delta_shards.append({"tensors": current_keys})
            current_keys = []
            current_bytes = 0
        current_keys.append(key)
        current_bytes += tensor_bytes
    if current_keys:
        delta_shards.append({"tensors": current_keys})
    for index, item in enumerate(delta_shards, start=1):
        item["delta_file"] = (
            f"{DELTA_ROOT}/model_delta_{index:05d}.safetensors"
        )
        item["uncompressed_tensor_bytes"] = sum(
            int(tensor_manifest[key]["bytes"]) for key in item["tensors"]
        )
    manifest = {
        "format": DELTA_FORMAT,
        "endianness": sys.byteorder,
        "checkpoint_name": checkpoint.name,
        "base_identity": _artifact_identity(base_model, base_layout),
        "trained_layout": {
            "index_name": trained_layout.index_name,
            "files": trained_file_manifest,
            "delta_shards": delta_shards,
            "max_delta_shard_bytes": max_delta_shard_bytes,
        },
        "tensors": tensor_manifest,
        "state_files": state_manifest,
    }
    return manifest, trained_layout, base_layout


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    ).encode("utf-8") + b"\n"


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


def _compression_from_path(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):
        return "gzip"
    if name.endswith((".tar.zst", ".tar.zstd", ".tzst")):
        return "zstd"
    if name.endswith((".tar.xz", ".txz")):
        return "xz"
    if name.endswith(".tar"):
        return "none"
    raise ValueError(
        "cannot infer compression; use .tar.gz, .tar.zst, .tar.xz, or "
        "--compression"
    )


def _input_compression_from_path(path: Path) -> str:
    """Infer an input codec, ignoring Kaggle's inert ``.xx`` escape suffix."""
    name = path.name.lower()
    if name.endswith(".xx"):
        name = name[: -len(".xx")]
    if name.endswith((".tar.gz", ".tgz")):
        return "gzip"
    if name.endswith((".tar.zst", ".tar.zstd", ".tzst")):
        return "zstd"
    if name.endswith((".tar.xz", ".txz")):
        return "xz"
    if name.endswith(".tar"):
        return "none"
    return "auto"


def is_delta_archive_path(path: str | Path) -> bool:
    """Return whether a path has a supported delta-archive suffix.

    Kaggle-safe archive names append an inert ``.xx`` suffix to prevent the
    dataset service from extracting the tarball.  Input handling strips only
    that final suffix, matching :func:`_input_compression_from_path`.
    """
    name = Path(path).name.lower()
    if name.endswith(".xx"):
        name = name[: -len(".xx")]
    return name.endswith(DELTA_ARCHIVE_SUFFIXES)


@contextmanager
def _streaming_tar_writer(
    final_path: Path,
    *,
    compression: str,
    xz_preset: str = "9e",
) -> Iterator[tarfile.TarFile]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"partial archive already exists: {temporary}")
    output = temporary.open("xb")
    process: subprocess.Popen[bytes] | None = None
    stream: BinaryIO = output
    try:
        if compression == "gzip":
            executable = shutil.which("pigz") or shutil.which("gzip")
            if executable is None:
                raise RuntimeError("gzip/pigz is required for .tar.gz output")
            process = subprocess.Popen(
                [executable, "-9"],
                stdin=subprocess.PIPE,
                stdout=output,
            )
            assert process.stdin is not None
            stream = process.stdin
        elif compression == "zstd":
            executable = shutil.which("zstd")
            if executable is None:
                raise RuntimeError("zstd is required for .tar.zst output")
            process = subprocess.Popen(
                [executable, "-T0", "--ultra", "-22", "-c"],
                stdin=subprocess.PIPE,
                stdout=output,
            )
            assert process.stdin is not None
            stream = process.stdin
        elif compression == "xz":
            executable = shutil.which("xz")
            if executable is None:
                raise RuntimeError("xz is required for .tar.xz output")
            process = subprocess.Popen(
                [executable, f"-{xz_preset}", "-T0", "-c"],
                stdin=subprocess.PIPE,
                stdout=output,
            )
            assert process.stdin is not None
            stream = process.stdin
        elif compression != "none":
            raise ValueError(f"unsupported compression: {compression}")
        with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            yield tar
        if process is not None:
            stream.close()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f"{compression} compressor exited with status {return_code}"
                )
        output.flush()
        os.fsync(output.fileno())
        output.close()
        os.replace(temporary, final_path)
    except BaseException:
        if process is not None:
            try:
                stream.close()
            except OSError:
                pass
            if process.poll() is None:
                process.terminate()
            process.wait()
        output.close()
        temporary.unlink(missing_ok=True)
        raise


def _encode_delta_shard(
    *,
    delta_shard: dict[str, Any],
    tensor_manifest: dict[str, Any],
    trained: SafeTensorReader,
    base: SafeTensorReader,
    output: Path,
) -> None:
    encoded: dict[str, torch.Tensor] = {}
    for key in delta_shard["tensors"]:
        entry = tensor_manifest[key]
        trained_tensor = trained.get_tensor(key)
        trained_raw = _tensor_bytes(trained_tensor)
        if entry["encoding"] == "base-xor-byte-shuffle":
            base_tensor = base.get_tensor(key)
            if _sha256_tensor(base_tensor) != entry["base_sha256"]:
                raise RuntimeError(f"base tensor changed while packing: {key}")
            raw = torch.bitwise_xor(trained_raw, _tensor_bytes(base_tensor))
            del base_tensor
        else:
            raw = trained_raw
        encoded[key] = _byte_shuffle(raw, int(entry["element_size"]))
        del trained_tensor, trained_raw, raw
    save_file(
        encoded,
        output,
        metadata={"format": DELTA_FORMAT, "codec": "byte-shuffled-xor"},
    )


class _DeltaDirectoryWriter:
    """Minimal tar-writer interface for a complete, checksummed delta folder."""

    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, dict[str, Any]] = {}

    def addfile(self, info: tarfile.TarInfo, stream: BinaryIO) -> None:
        name = str(_safe_relative_path(info.name))
        if name in self.files or name == DELTA_STAGING_NAME or not info.isfile():
            raise ValueError(f"invalid or duplicate delta payload: {name}")
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        remaining = info.size
        with path.open("xb") as output:
            while remaining:
                chunk = stream.read(min(remaining, COPY_CHUNK_BYTES))
                if not chunk:
                    raise RuntimeError(f"truncated delta payload while staging: {name}")
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            if stream.read(1):
                raise RuntimeError(f"delta payload grew while staging: {name}")
        self.files[name] = {"bytes": info.size, "sha256": digest.hexdigest()}

    def add(self, path: Path, *, arcname: str, recursive: bool = False) -> None:
        if recursive or path.is_symlink() or not path.is_file():
            raise ValueError(f"delta payload must be a regular file: {path}")
        info = tarfile.TarInfo(arcname)
        info.size = path.stat().st_size
        with path.open("rb") as stream:
            self.addfile(info, stream)


@contextmanager
def _directory_delta_writer(final_path: Path) -> Iterator[_DeltaDirectoryWriter]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite delta directory: {final_path}")
    # Only this uniquely owned partial directory is removed on failure. A
    # published directory remains reusable after a failed compression/copy.
    temporary = Path(tempfile.mkdtemp(prefix=".arc-delta-stage-", dir=final_path.parent))
    try:
        writer = _DeltaDirectoryWriter(temporary)
        yield writer
        (temporary / DELTA_STAGING_NAME).write_bytes(_json_bytes({
            "format": "nvarc-exact-delta-staging-v1", "files": writer.files,
        }))
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite delta directory: {final_path}")
        os.rename(temporary, final_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _pack_delta_checkpoint_single(
    checkpoint: str | Path,
    base_model: str | Path,
    archive: str | Path,
    *,
    work_dir: str | Path | None = None,
    compression: str = "auto",
    xz_preset: str = "9e",
    max_delta_shard_bytes: int = DEFAULT_DELTA_SHARD_BYTES,
    reference: dict[str, Any] | None = None,
    directory: bool = False,
) -> dict[str, Any]:
    """Stream an exact restart checkpoint delta into a compressed archive."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    base_model = Path(base_model).expanduser().resolve()
    archive = Path(archive).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite delta archive: {archive}")
    try:
        archive.relative_to(checkpoint)
    except ValueError:
        pass
    else:
        raise ValueError("delta archive must be outside the source checkpoint")
    max_delta_shard_bytes = int(max_delta_shard_bytes)
    if max_delta_shard_bytes <= 0:
        raise ValueError("max_delta_shard_bytes must be positive")
    selected_compression = "directory" if directory else _compression_from_path(archive, compression)
    if xz_preset not in XZ_PRESETS:
        raise ValueError(f"unsupported XZ preset: {xz_preset!r}")
    print(
        "[exact-delta] inventory and checksum pass: "
        f"checkpoint={checkpoint}, base={base_model}",
        flush=True,
    )
    manifest, trained_layout, base_layout = _build_manifest(
        checkpoint,
        base_model,
        max_delta_shard_bytes=max_delta_shard_bytes,
    )
    if reference is not None:
        manifest["format"] = CHAINED_DELTA_FORMAT
        manifest["reference"] = reference
    manifest_payload = _json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    model_files = set(trained_layout.files)
    state_files = _non_model_files(checkpoint, model_files)
    if directory:
        archive.parent.mkdir(parents=True, exist_ok=True)
        # XOR keeps the original tensor byte count. Account for state, shard
        # headers and a safety margin before starting a multi-GB disk stage.
        required_bytes = (
            sum(int(entry["bytes"]) for entry in manifest["tensors"].values())
            + sum(int(entry["bytes"]) for entry in manifest["state_files"].values())
            + len(manifest_payload) + len(manifest["tensors"]) * 1024
            + ARCHIVE_COPY_FREE_MARGIN_BYTES
        )
        free_bytes = shutil.disk_usage(archive.parent).free
        if required_bytes > free_bytes:
            raise OSError(28, f"uncompressed delta stage needs about {required_bytes:,} bytes; "
                          f"only {free_bytes:,} free", str(archive))
    temp_parent = (
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else checkpoint.parent
    )
    temp_parent.mkdir(parents=True, exist_ok=True)
    writer_context = (
        _directory_delta_writer(archive) if directory else
        _streaming_tar_writer(archive, compression=selected_compression, xz_preset=xz_preset)
    )
    started = __import__("time").perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="arc-delta-shard-", dir=temp_parent
    ) as temporary_dir, SafeTensorReader(trained_layout) as trained, SafeTensorReader(
        base_layout
    ) as base, writer_context as tar:
        _add_bytes(tar, DELTA_MANIFEST_NAME, manifest_payload)
        for path, relative in state_files:
            tar.add(path, arcname=relative, recursive=False)
        shard_count = len(manifest["trained_layout"]["delta_shards"])
        for shard_index, delta_shard in enumerate(
            manifest["trained_layout"]["delta_shards"], start=1
        ):
            print(
                "[exact-delta] encoding bounded delta shard "
                f"{shard_index}/{shard_count}: "
                f"{delta_shard['uncompressed_tensor_bytes'] / 1024**2:.1f}MiB",
                flush=True,
            )
            delta_path = Path(temporary_dir) / Path(
                delta_shard["delta_file"]
            ).name
            _encode_delta_shard(
                delta_shard=delta_shard,
                tensor_manifest=manifest["tensors"],
                trained=trained,
                base=base,
                output=delta_path,
            )
            tar.add(
                delta_path,
                arcname=delta_shard["delta_file"],
                recursive=False,
            )
            delta_path.unlink()
    elapsed = __import__("time").perf_counter() - started
    source_bytes = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    )
    archive_bytes = (
        sum(path.stat().st_size for path in archive.rglob("*") if path.is_file())
        if directory else archive.stat().st_size
    )
    return {
        "format": manifest["format"],
        "reference": manifest.get("reference"),
        "checkpoint": str(checkpoint),
        "base_model": str(base_model),
        "archive": str(archive),
        "compression": selected_compression,
        "xz_preset": xz_preset if selected_compression == "xz" else None,
        "manifest_sha256": manifest_sha256,
        "source_bytes": source_bytes,
        "archive_bytes": archive_bytes,
        "compression_ratio": archive_bytes / max(source_bytes, 1),
        "tensor_count": len(manifest["tensors"]),
        "model_shards": len(manifest["trained_layout"]["files"]),
        "delta_shards": len(manifest["trained_layout"]["delta_shards"]),
        "wall_seconds": elapsed,
    }


@contextmanager
def _open_streaming_tar(
    path: Path,
    *,
    parallel_gzip: bool = False,
) -> Iterator[tarfile.TarFile]:
    """Open an archive sequentially, optionally streaming through ``pigz``.

    Reconstruction consumes the complete archive and benefits from parallel
    decompression.  Manifest-only reads intentionally use Python's gzip reader
    because they stop after the first tar member and should not launch or drain
    a multi-gigabyte decompression stream.
    """
    compression = _input_compression_from_path(path)
    decompressor: list[str] | None = None
    decompressor_name = ""
    if parallel_gzip and compression == "gzip":
        pigz = shutil.which("pigz")
        if pigz is not None:
            decompressor = [pigz, "-dc", str(path)]
            decompressor_name = "pigz"
    elif compression == "zstd":
        zstd = shutil.which("zstd")
        if zstd is None:
            raise RuntimeError("zstd is required to read .tar.zst inputs")
        decompressor = [zstd, "-dc", str(path)]
        decompressor_name = "zstd"
    if decompressor is not None:
        process = subprocess.Popen(
            decompressor,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        stdout = process.stdout
        stderr = process.stderr
        try:
            print(
                "[exact-delta] streaming decompression with "
                f"{decompressor_name}",
                flush=True,
            )
            with tarfile.open(
                fileobj=stdout,
                mode="r|",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                yield tar
            # Tar readers may stop at the end-of-archive blocks before the gzip
            # footer. Drain the bounded pipe so pigz can verify and exit cleanly.
            while stdout.read(COPY_CHUNK_BYTES):
                pass
            error_output = stderr.read()
            return_code = process.wait()
            if return_code:
                detail = error_output.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"{decompressor_name} decompressor exited with status "
                    f"{return_code}: {detail or 'no diagnostic'}"
                )
        except BaseException:
            try:
                stdout.close()
            except OSError:
                pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            stdout.close()
            stderr.close()
        return

    mode = {
        "gzip": "r|gz",
        "xz": "r|xz",
        "none": "r|",
        "auto": "r|*",
    }[compression]
    with tarfile.open(path, mode=mode) as tar:
        yield tar


def _validate_delta_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("format") not in (DELTA_FORMAT, CHAINED_DELTA_FORMAT):
        raise RuntimeError(
            f"unsupported delta format: {manifest.get('format')!r}"
        )
    reference = manifest.get("reference")
    if manifest["format"] == DELTA_FORMAT and reference is not None:
        raise RuntimeError("v1 delta cannot declare a parent reference")
    if manifest["format"] == CHAINED_DELTA_FORMAT:
        if not isinstance(reference, dict) or reference.get("kind") != "parent-delta":
            raise RuntimeError("chained delta is missing its parent reference")
        digest = reference.get("manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise RuntimeError("invalid parent manifest checksum")
        if type(reference.get("depth")) is not int or reference["depth"] < 1:
            raise RuntimeError("invalid parent reference depth")
    return manifest


def _directory_payload_names(manifest: dict[str, Any]) -> list[str]:
    return [
        *[
            str(item["delta_file"])
            for item in manifest["trained_layout"]["delta_shards"]
        ],
        *[str(name) for name in manifest["state_files"]],
    ]


def _resolve_directory_payload(root: Path, name: str) -> Path:
    relative = _safe_relative_path(name)
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"delta directory is missing required file: {name}"
        ) from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"delta directory member escapes its root: {name}"
        ) from error
    if path.is_symlink() or not resolved.is_file():
        raise RuntimeError(
            f"delta directory member must be a regular file: {name}"
        )
    return resolved


@contextmanager
def _open_delta_payloads(
    source: Path,
) -> Iterator[tuple[dict[str, Any], Iterator[tuple[str, BinaryIO]]]]:
    """Yield a manifest and bounded payload streams from an archive or directory."""
    if source.is_dir():
        manifest = read_delta_manifest(source)

        def directory_payloads() -> Iterator[tuple[str, BinaryIO]]:
            for name in _directory_payload_names(manifest):
                path = _resolve_directory_payload(source, name)
                with path.open("rb") as handle:
                    yield name, handle

        yield manifest, directory_payloads()
        return
    if not source.is_file():
        raise FileNotFoundError(f"delta checkpoint source not found: {source}")
    with _open_streaming_tar(source, parallel_gzip=True) as tar:
        iterator = iter(tar)
        member = next(iterator, None)
        if member is None or member.name != DELTA_MANIFEST_NAME or not member.isfile():
            raise RuntimeError(
                f"delta archive must begin with {DELTA_MANIFEST_NAME}"
            )
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"cannot read {DELTA_MANIFEST_NAME}")
        manifest = _validate_delta_manifest(json.load(handle))

        def archive_payloads() -> Iterator[tuple[str, BinaryIO]]:
            for payload in iterator:
                _safe_relative_path(payload.name)
                if payload.isdir():
                    continue
                if not payload.isfile():
                    raise RuntimeError(
                        f"unsupported non-regular archive member: {payload.name}"
                    )
                payload_handle = tar.extractfile(payload)
                if payload_handle is None:
                    raise RuntimeError(
                        f"cannot read archive member: {payload.name}"
                    )
                yield payload.name, payload_handle

        yield manifest, archive_payloads()


def _delta_source_bytes(source: Path, manifest: dict[str, Any]) -> int:
    if source.is_file():
        return source.stat().st_size
    return sum(
        _resolve_directory_payload(source, name).stat().st_size
        for name in (DELTA_MANIFEST_NAME, *_directory_payload_names(manifest))
    )


def read_delta_manifest(source: str | Path) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        manifest_path = _resolve_directory_payload(source, DELTA_MANIFEST_NAME)
        with manifest_path.open("r", encoding="utf-8") as handle:
            return _validate_delta_manifest(json.load(handle))
    if not source.is_file():
        raise FileNotFoundError(f"delta checkpoint source not found: {source}")
    with _open_streaming_tar(source) as tar:
        member = next(iter(tar), None)
        if member is None or member.name != DELTA_MANIFEST_NAME or not member.isfile():
            raise RuntimeError(
                f"delta archive must begin with {DELTA_MANIFEST_NAME}"
            )
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"cannot read {DELTA_MANIFEST_NAME}")
        return _validate_delta_manifest(json.load(handle))


def _runtime_auxiliary_identity(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Old manifests included .gitattributes. Filter for comparison only; never
    # mutate their serialized identity, which may bind a parent delta chain.
    return [
        item for item in files
        if item["name"] not in IGNORED_BASE_AUXILIARY_FILES
    ]


def _validate_base_identity(
    base_model: Path,
    base_layout: SafeTensorLayout,
    expected: dict[str, Any],
) -> None:
    actual = _artifact_identity(base_model, base_layout)
    for name in ("config.json", *MODEL_INDEX_NAMES):
        if name in expected and actual.get(name) != expected[name]:
            raise RuntimeError(f"base-model identity mismatch for {name}")
    if actual.get("tensor_count") != expected.get("tensor_count"):
        raise RuntimeError("base-model tensor count does not match delta archive")
    if (
        "auxiliary_files" in expected
        and _runtime_auxiliary_identity(actual["auxiliary_files"])
        != _runtime_auxiliary_identity(expected["auxiliary_files"])
    ):
        raise RuntimeError("base-model auxiliary-file identity mismatch")


def _decode_delta_shard(
    *,
    delta_path: Path,
    delta_shard: dict[str, Any],
    tensor_manifest: dict[str, Any],
    base: SafeTensorReader,
) -> dict[str, torch.Tensor]:
    reconstructed: dict[str, torch.Tensor] = {}
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        actual_keys = sorted(delta.keys())
        expected_keys = sorted(delta_shard["tensors"])
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"delta shard key mismatch for {delta_path}: "
                f"expected {len(expected_keys)}, got {len(actual_keys)}"
            )
        for key in expected_keys:
            entry = tensor_manifest[key]
            encoded = delta.get_tensor(key)
            raw = _byte_unshuffle(encoded, int(entry["element_size"]))
            if entry["encoding"] == "base-xor-byte-shuffle":
                base_tensor = base.get_tensor(key)
                if _sha256_tensor(base_tensor) != entry["base_sha256"]:
                    raise RuntimeError(f"wrong base tensor while reconstructing: {key}")
                raw = torch.bitwise_xor(raw, _tensor_bytes(base_tensor))
                del base_tensor
            elif entry["encoding"] != "raw-byte-shuffle":
                raise RuntimeError(
                    f"unsupported tensor encoding for {key}: {entry['encoding']}"
                )
            dtype = _torch_dtype(entry["dtype"])
            expected_bytes = int(entry["bytes"])
            if raw.numel() != expected_bytes:
                raise RuntimeError(
                    f"decoded byte count mismatch for {key}: "
                    f"{raw.numel()} != {expected_bytes}"
                )
            tensor = raw.view(dtype).reshape(tuple(entry["shape"])).clone()
            if _sha256_tensor(tensor) != entry["trained_sha256"]:
                raise RuntimeError(f"reconstructed tensor checksum mismatch: {key}")
            reconstructed[key] = tensor
    return reconstructed


def _decode_delta_shard_with_reader(
    *,
    delta_path: Path,
    delta_shard: dict[str, Any],
    tensor_manifest: dict[str, Any],
    base_layout: SafeTensorLayout,
) -> dict[str, torch.Tensor]:
    """Decode one shard with thread-local safetensor handles."""
    with SafeTensorReader(base_layout) as base:
        return _decode_delta_shard(
            delta_path=delta_path,
            delta_shard=delta_shard,
            tensor_manifest=tensor_manifest,
            base=base,
        )


def _save_reconstructed_model_file(
    *,
    tensors: dict[str, torch.Tensor],
    trained_file: dict[str, Any],
    tensor_manifest: dict[str, Any],
    destination: Path,
) -> None:
    expected_keys = sorted(trained_file["tensors"])
    if sorted(tensors) != expected_keys:
        raise RuntimeError(
            f"reconstructed model file is incomplete: {trained_file['trained_file']}"
        )
    output = destination / _safe_relative_path(trained_file["trained_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata=trained_file.get("metadata"))
    with safe_open(output, framework="pt", device="cpu") as saved:
        for key in expected_keys:
            if _sha256_tensor(saved.get_tensor(key)) != tensor_manifest[key][
                "trained_sha256"
            ]:
                raise RuntimeError(f"saved tensor verification failed: {key}")


def _reconstruct_delta_checkpoint_single(
    archive: str | Path,
    base_model: str | Path,
    destination: str | Path,
    *,
    work_dir: str | Path | None = None,
    workers: int = DEFAULT_RECONSTRUCT_WORKERS,
    model_only: bool = False,
    expected_manifest_sha256: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct a Trainer checkpoint or a weight-only HF model artifact.

    ``model_only=True`` still streams and checks every archived state file so
    archive corruption cannot be hidden, but it writes only the trained model
    tensors and the model index.  Runtime configuration/tokenizer files are
    copied from the verified immutable base.  Optimizer, scheduler, Trainer,
    RNG, and ARC data state are deliberately discarded.
    """
    archive = Path(archive).expanduser().resolve()
    base_model = Path(base_model).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not (archive.is_file() or archive.is_dir()):
        raise FileNotFoundError(f"delta checkpoint source not found: {archive}")
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite reconstruction destination: {destination}"
        )
    workers = int(workers)
    if workers < 1:
        raise ValueError("delta reconstruction workers must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{os.getpid()}.reconstruct"
    if staging.exists():
        raise FileExistsError(f"stale reconstruction directory exists: {staging}")
    temp_parent = (
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else destination.parent
    )
    temp_parent.mkdir(parents=True, exist_ok=True)
    base_layout = SafeTensorLayout(base_model)
    started = __import__("time").perf_counter()
    staging.mkdir()
    try:
        with _open_delta_payloads(archive) as (
            manifest,
            payloads,
        ), SafeTensorReader(
            base_layout
        ) as base, tempfile.TemporaryDirectory(
            prefix="arc-delta-read-", dir=temp_parent
        ) as temporary_dir:
            if expected_manifest_sha256 is None:
                if manifest["format"] != DELTA_FORMAT:
                    raise RuntimeError("chained delta requires explicit parent checkpoints")
            elif _manifest_sha256(manifest) != expected_manifest_sha256:
                raise RuntimeError("delta manifest changed after chain validation")
            if manifest.get("endianness") != sys.byteorder:
                raise RuntimeError(
                    "delta endianness differs from the current runtime: "
                    f"{manifest.get('endianness')} != {sys.byteorder}"
                )
            _validate_base_identity(
                base_model, base_layout, manifest["base_identity"]
            )
            delta_shards = {
                item["delta_file"]: item
                for item in manifest["trained_layout"]["delta_shards"]
            }
            trained_files = {
                item["trained_file"]: item
                for item in manifest["trained_layout"]["files"]
            }
            pending_tensors: dict[str, dict[str, torch.Tensor]] = {
                filename: {} for filename in trained_files
            }
            expected_state = manifest["state_files"]
            trained_index_name = manifest["trained_layout"].get("index_name")
            model_state_files = (
                {str(trained_index_name)} if trained_index_name else set()
            )
            copied_base_auxiliary = _copy_base_auxiliary_files(
                base_model,
                base_layout,
                staging,
                replaced_paths=(
                    {*trained_files, *model_state_files}
                    if model_only
                    else {*expected_state, *trained_files}
                ),
            )
            seen_state: set[str] = set()
            restored_state: set[str] = set()
            seen_delta_shards: set[str] = set()
            submitted_delta_shards: set[str] = set()
            saved_model_files: set[str] = set()
            pending_decodes: deque[
                tuple[str, Path, Future[dict[str, torch.Tensor]]]
            ] = deque()
            executor = (
                ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="arc-delta-decode",
                )
                if workers > 1
                else None
            )

            def merge_decoded(
                member_name: str,
                decoded: dict[str, torch.Tensor],
            ) -> None:
                touched_files: set[str] = set()
                for key, tensor in decoded.items():
                    trained_filename = manifest["tensors"][key]["trained_file"]
                    pending_tensors[trained_filename][key] = tensor
                    touched_files.add(trained_filename)
                for trained_filename in sorted(touched_files):
                    trained_file = trained_files[trained_filename]
                    if len(pending_tensors[trained_filename]) == len(
                        trained_file["tensors"]
                    ):
                        print(
                            "[exact-delta] writing reconstructed model file: "
                            f"{trained_filename}",
                            flush=True,
                        )
                        _save_reconstructed_model_file(
                            tensors=pending_tensors.pop(trained_filename),
                            trained_file=trained_file,
                            tensor_manifest=manifest["tensors"],
                            destination=staging,
                        )
                        saved_model_files.add(trained_filename)
                seen_delta_shards.add(member_name)

            def consume_oldest_decode() -> None:
                member_name, temporary, future = pending_decodes.popleft()
                try:
                    decoded = future.result()
                finally:
                    temporary.unlink(missing_ok=True)
                merge_decoded(member_name, decoded)

            try:
                for member_name, source in payloads:
                    if member_name in delta_shards:
                        if member_name in submitted_delta_shards:
                            raise RuntimeError(
                                f"duplicate delta shard: {member_name}"
                            )
                        submitted_delta_shards.add(member_name)
                        temporary = Path(temporary_dir) / Path(member_name).name
                        with temporary.open("xb") as output:
                            shutil.copyfileobj(source, output, COPY_CHUNK_BYTES)
                        print(
                            "[exact-delta] decoding bounded delta shard "
                            f"{len(submitted_delta_shards)}/{len(delta_shards)} "
                            f"(workers={workers})",
                            flush=True,
                        )
                        if executor is None:
                            try:
                                decoded = _decode_delta_shard(
                                    delta_path=temporary,
                                    delta_shard=delta_shards[member_name],
                                    tensor_manifest=manifest["tensors"],
                                    base=base,
                                )
                            finally:
                                temporary.unlink(missing_ok=True)
                            merge_decoded(member_name, decoded)
                        else:
                            future = executor.submit(
                                _decode_delta_shard_with_reader,
                                delta_path=temporary,
                                delta_shard=delta_shards[member_name],
                                tensor_manifest=manifest["tensors"],
                                base_layout=base_layout,
                            )
                            pending_decodes.append(
                                (member_name, temporary, future)
                            )
                            if len(pending_decodes) >= workers:
                                consume_oldest_decode()
                        continue
                    if member_name not in expected_state:
                        raise RuntimeError(
                            "unexpected file in delta checkpoint source: "
                            f"{member_name}"
                        )
                    if member_name in seen_state:
                        raise RuntimeError(f"duplicate state file: {member_name}")
                    restore_member = not model_only or member_name in model_state_files
                    target = None
                    output = None
                    if restore_member:
                        relative = _safe_relative_path(member_name)
                        target = staging / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        output = target.open("xb")
                    digest = hashlib.sha256()
                    written = 0
                    try:
                        while chunk := source.read(COPY_CHUNK_BYTES):
                            if output is not None:
                                output.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                    finally:
                        if output is not None:
                            output.close()
                    expected = expected_state[member_name]
                    if written != int(expected["bytes"]):
                        raise RuntimeError(
                            f"state-file size mismatch: {member_name}"
                        )
                    if digest.hexdigest() != expected["sha256"]:
                        raise RuntimeError(
                            f"state-file checksum mismatch: {member_name}"
                        )
                    seen_state.add(member_name)
                    if restore_member:
                        restored_state.add(member_name)
                while pending_decodes:
                    consume_oldest_decode()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                while pending_decodes:
                    _, temporary, _ = pending_decodes.popleft()
                    temporary.unlink(missing_ok=True)
            missing_state = sorted(set(expected_state) - seen_state)
            missing_delta_shards = sorted(
                set(delta_shards) - seen_delta_shards
            )
            missing_model_files = sorted(set(trained_files) - saved_model_files)
            if missing_state or missing_delta_shards or missing_model_files:
                raise RuntimeError(
                    "incomplete delta checkpoint source: "
                    f"missing_state={missing_state}, "
                    f"missing_delta_shards={missing_delta_shards}, "
                    f"missing_model_files={missing_model_files}"
                )
        reconstruction_record = {
            "format": manifest["format"],
            "manifest_sha256": _manifest_sha256(manifest),
            "reference": manifest.get("reference"),
            "archive": str(archive),
            "archive_bytes": _delta_source_bytes(archive, manifest),
            "source_kind": "directory" if archive.is_dir() else "archive",
            "base_model": str(base_model),
            "tensor_count": len(manifest["tensors"]),
            "model_shards": len(trained_files),
            "delta_shards": len(delta_shards),
            "reconstruct_workers": workers,
            "base_auxiliary_files": copied_base_auxiliary,
            "model_only": bool(model_only),
            "state_files_total": len(expected_state),
            "state_files_restored": sorted(restored_state),
            "state_files_discarded": sorted(set(expected_state) - restored_state),
            "verified": True,
        }
        reconstruction_record.update(provenance or {})
        record_path = staging / "arc_delta_reconstruction.json"
        record_path.write_text(
            json.dumps(reconstruction_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    elapsed = __import__("time").perf_counter() - started
    return {
        **reconstruction_record,
        "checkpoint": str(destination),
        "elapsed_seconds": elapsed,
        "wall_seconds": elapsed,
        "checkpoint_bytes": sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        ),
    }


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    # Identity is independent of archive name, codec, or .xx suffix. Tensor and
    # state checksums inside the manifest bind the actual checkpoint contents.
    return hashlib.sha256(_json_bytes(manifest)).hexdigest()


def restore_delta_training_state(
    archive: str | Path,
    model: str | Path,
    destination: str | Path,
    *,
    workers: int = DEFAULT_RECONSTRUCT_WORKERS,
) -> dict[str, Any]:
    """Restore run state against an already materialized, exact trained model.

    No parent archive is read and no XOR is decoded. Every supplied model
    tensor must match the target manifest (not its parent). Runtime metadata
    must still match the archive's immutable base metadata. All run-state
    payloads are checksum verified; unused delta payloads are drained, not
    decoded/verified. Read-only weight symlinks let the ordinary Trainer resume
    loader operate without making another model-sized copy. Cleanup of the
    generated checkpoint never follows those symlinks.
    """
    archive = Path(archive).expanduser().resolve()
    model = Path(model).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"resume destination already exists: {destination}")
    if destination == model or destination.is_relative_to(model):
        raise ValueError("state-only destination must be outside the immutable model")
    if archive.is_dir() and destination.is_relative_to(archive):
        raise ValueError("state-only destination must be outside the input delta directory")
    if workers <= 0:
        raise ValueError("state-only verification workers must be positive")
    layout = SafeTensorLayout(model)
    if layout.index_name and any((model / name).exists() for name in SINGLE_MODEL_NAMES):
        raise RuntimeError("state-only model has ambiguous indexed and single-file weights")
    for filename in layout.files:
        _safe_relative_path(filename)
        with safe_open(model / filename, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != set(layout.keys_for_file(filename)):
                raise RuntimeError("state-only model shard keys disagree with its index")
    identity = _artifact_identity(model, layout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = __import__("time").perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=".arc-state-resume-", dir=destination.parent,
    ) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        with _open_delta_payloads(archive) as (manifest, payloads):
            if manifest.get("endianness") != sys.byteorder:
                raise RuntimeError("state-only model/archive endianness mismatch")
            tensors = manifest["tensors"]
            if set(layout.weight_map) != set(tensors):
                raise RuntimeError("state-only model tensor keys do not match target checkpoint")
            expected_auxiliary = manifest["base_identity"].get("auxiliary_files")
            if (
                expected_auxiliary is None
                or _runtime_auxiliary_identity(identity["auxiliary_files"])
                != _runtime_auxiliary_identity(expected_auxiliary)
            ):
                raise RuntimeError("state-only model runtime metadata does not match archive base")

            def verify_keys(keys):
                with SafeTensorReader(layout) as reader:
                    for key in keys:
                        tensor = reader.get_tensor(key)
                        entry = tensors[key]
                        if (
                            list(tensor.shape) != entry["shape"]
                            or _dtype_name(tensor.dtype) != entry["dtype"]
                            or tensor.numel() * tensor.element_size() != entry["bytes"]
                            or _sha256_tensor(tensor) != entry["trained_sha256"]
                        ):
                            raise RuntimeError(f"state-only model tensor mismatch: {key}")

            keys = sorted(tensors)
            partitions = [keys[i::workers] for i in range(min(workers, len(keys)))]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(verify_keys, partitions))
            print(
                f"[exact-delta] state-only resume: verified {len(keys):,} target "
                "model tensors; restoring run state without parent reconstruction",
                flush=True,
            )
            expected_state = manifest["state_files"]
            required = {
                "trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt",
                "arc_diffusion_restart.json", "arc_diffusion_collator_state.pt",
            }
            if not required.issubset(expected_state) or not any(
                Path(name).match("rng_state*.pth") for name in expected_state
            ):
                raise RuntimeError("state-only archive is missing required training state")
            delta_names = {
                item["delta_file"] for item in manifest["trained_layout"]["delta_shards"]
            }
            if delta_names & set(expected_state):
                raise RuntimeError("state-only archive has overlapping model/state members")
            excluded = {
                *MODEL_INDEX_NAMES, *SINGLE_MODEL_NAMES, *layout.files,
                *(item["trained_file"] for item in manifest["trained_layout"]["files"]),
            }
            state_bytes = sum(
                int(entry["bytes"]) for name, entry in expected_state.items()
                if name not in excluded
            )
            if shutil.disk_usage(destination.parent).free < state_bytes + ARCHIVE_COPY_FREE_MARGIN_BYTES:
                raise OSError(28, f"state-only resume needs {state_bytes:,} bytes plus safety margin")
            _copy_base_auxiliary_files(
                model, layout, staging, replaced_paths=set(expected_state) | excluded,
            )
            seen: set[str] = set()
            restored: set[str] = set()
            for name, source in payloads:
                if name in seen:
                    raise RuntimeError(f"duplicate state-only archive member: {name}")
                seen.add(name)
                if name in delta_names:
                    while source.read(COPY_CHUNK_BYTES):
                        pass
                    continue
                if name not in expected_state or name == "arc_delta_reconstruction.json":
                    raise RuntimeError(f"unexpected state-only archive member: {name}")
                output = None
                if name not in excluded:
                    target = staging / _safe_relative_path(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = target.open("xb")
                digest = hashlib.sha256()
                size = 0
                try:
                    while chunk := source.read(COPY_CHUNK_BYTES):
                        if output is not None:
                            output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                finally:
                    if output is not None:
                        output.close()
                entry = expected_state[name]
                if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"state-only state-file checksum/size mismatch: {name}")
                if name not in excluded:
                    restored.add(name)
            if seen != delta_names | set(expected_state):
                raise RuntimeError("state-only archive is incomplete")
        # Never copy the archived index: a promoted HF artifact may be resharded.
        linked_weights = [*layout.files, *([layout.index_name] if layout.index_name else [])]
        for name in linked_weights:
            relative = _safe_relative_path(name)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(model / relative)
        record = {
            "format": manifest["format"],
            "manifest_sha256": _manifest_sha256(manifest),
            "archive": str(archive),
            "base_model": str(model),
            "promoted_model_identity": identity,
            "checkpoint": str(destination),
            "checkpoint_name": manifest.get("checkpoint_name"),
            "tensor_count": len(tensors),
            "verified": True,
            "state_only": True,
            "model_only": False,
            "parent_checkpoints": [],
            "delta_payloads_decoded": False,
            "state_files_restored": sorted(restored),
            "linked_model_files": linked_weights,
            "state_bytes": sum(expected_state[name]["bytes"] for name in restored),
            "elapsed_seconds": __import__("time").perf_counter() - started,
        }
        (staging / "arc_delta_reconstruction.json").write_bytes(_json_bytes(record))
        os.replace(staging, destination)
    return record


def _validated_delta_chain(sources) -> list[tuple[Path, dict[str, Any]]]:
    chain = []
    seen = set()
    for index, source in enumerate(sources):
        source = Path(source).expanduser().resolve()
        manifest = read_delta_manifest(source)
        digest = _manifest_sha256(manifest)
        if digest in seen:
            raise RuntimeError("duplicate/cyclic delta checkpoint in parent chain")
        if manifest["format"] == DELTA_FORMAT:
            if index:
                raise RuntimeError("base-relative delta must be first; unexpected parent checkpoints")
        else:
            if not chain:
                raise RuntimeError("chained delta requires missing parent checkpoints (oldest first)")
            reference = manifest["reference"]
            if reference["manifest_sha256"] != _manifest_sha256(chain[-1][1]):
                raise RuntimeError("delta parent identity mismatch: wrong, missing, or misordered parent")
            if reference.get("depth") != index:
                raise RuntimeError("delta parent depth mismatch")
        chain.append((source, manifest))
        seen.add(digest)
    return chain


def select_delta_export_anchor(sources, anchor_checkpoint: str | Path) -> tuple[str, ...]:
    """Select a verified manifest-chain prefix, without decoding any weights.

    The export reference is independent of the checkpoint whose optimizer/data
    state resumes training. Match manifests, not filenames or step numbers, so
    an identical repacked/renamed archive can select the same fixed anchor.
    Actual payload checksums are still verified when decoding the references.
    """
    chain = _validated_delta_chain(sources)
    anchor_digest = _manifest_sha256(read_delta_manifest(anchor_checkpoint))
    for index, (_, manifest) in enumerate(chain):
        if _manifest_sha256(manifest) == anchor_digest:
            return tuple(str(source) for source, _ in chain[:index + 1])
    raise ValueError(
        "delta export anchor is not in the resume chain: supply an identical "
        "manifest from --delta-parent-checkpoint or --resume-from-delta-checkpoint"
    )


@contextmanager
def _resolved_delta_reference(base_model, chain, *, work_dir, workers):
    """Materialize only successive weight-only parents, not all run states.

    At most the previous and next parent model coexist while advancing. Final
    reconstruction then needs one reference model plus the target checkpoint.
    TemporaryDirectory owns every directory removed here; input archives and
    the immutable base are never cleanup targets.
    """
    base_model = Path(base_model).expanduser().resolve()
    if not chain:
        yield base_model, None
        return
    temp_parent = Path(work_dir).expanduser().resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-delta-parents-", dir=temp_parent) as temporary:
        reference_model = base_model
        previous = None
        for index, (source, manifest) in enumerate(chain):
            destination = Path(temporary) / f"parent-{index:04d}"
            print(f"[exact-delta] verifying weight-only parent {index + 1}/{len(chain)}: {source}", flush=True)
            _reconstruct_delta_checkpoint_single(
                source, reference_model, destination,
                work_dir=temp_parent, workers=workers, model_only=True,
                expected_manifest_sha256=_manifest_sha256(manifest),
            )
            if previous is not None:
                shutil.rmtree(previous)
            previous = reference_model = destination
        parent_manifest = chain[-1][1]
        yield reference_model, {
            "kind": "parent-delta",
            "manifest_sha256": _manifest_sha256(parent_manifest),
            "checkpoint_name": parent_manifest.get("checkpoint_name"),
            "depth": len(chain),
        }


def pack_delta_checkpoint(
    checkpoint: str | Path,
    base_model: str | Path,
    archive: str | Path,
    *,
    work_dir: str | Path | None = None,
    compression: str = "auto",
    xz_preset: str = "9e",
    max_delta_shard_bytes: int = DEFAULT_DELTA_SHARD_BYTES,
    parent_checkpoints=(),
    workers: int = DEFAULT_RECONSTRUCT_WORKERS,
    directory: bool = False,
) -> dict[str, Any]:
    """Pack against the immutable base, or the last verified parent in a chain."""
    archive = Path(archive).expanduser().resolve()
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite delta archive: {archive}")
    if int(workers) < 1:
        raise ValueError("delta reconstruction workers must be positive")
    chain = _validated_delta_chain(parent_checkpoints)
    temp_parent = work_dir or Path(checkpoint).expanduser().resolve().parent
    if directory and Path(temp_parent).expanduser().resolve().is_relative_to(archive):
        raise ValueError("delta work directory must be outside the new staging directory")
    with _resolved_delta_reference(
        base_model, chain, work_dir=temp_parent, workers=workers,
    ) as (reference_model, reference):
        result = _pack_delta_checkpoint_single(
            checkpoint, reference_model, archive,
            work_dir=work_dir, compression=compression, xz_preset=xz_preset,
            max_delta_shard_bytes=max_delta_shard_bytes, reference=reference,
            directory=directory,
        )
    # Report the stable root, not a deleted temporary reference directory.
    result["base_model"] = str(Path(base_model).expanduser().resolve())
    result["parent_checkpoints"] = [str(source) for source, _ in chain]
    return result


def stage_delta_checkpoint(
    checkpoint: str | Path,
    base_model: str | Path,
    directory: str | Path,
    *,
    work_dir: str | Path | None = None,
    parent_checkpoints=(),
    workers: int = DEFAULT_RECONSTRUCT_WORKERS,
    max_delta_shard_bytes: int = DEFAULT_DELTA_SHARD_BYTES,
) -> dict[str, Any]:
    """Encode once to a new complete delta folder, without compressing it."""
    result = pack_delta_checkpoint(
        checkpoint, base_model, directory, directory=True,
        work_dir=work_dir, parent_checkpoints=parent_checkpoints, workers=workers,
        max_delta_shard_bytes=max_delta_shard_bytes,
    )
    result["delta_directory"] = result.pop("archive")
    result["directory_bytes"] = result.pop("archive_bytes")
    return result


class _CheckedPayloadReader:
    """Hash as tar reads, avoiding an extra full-delta disk pass."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.size += len(chunk)
        return chunk


def compress_delta_directory(
    directory: str | Path,
    archive: str | Path,
    *,
    compression: str = "auto",
    xz_preset: str = "9e",
) -> dict[str, Any]:
    """Compress one whole staged folder; preserve it on success and failure.

    This is one continuous tar/compressor stream, like direct packing. Staging
    does not reset/expand the codec dictionary or guarantee a better ratio.
    All payloads are checked during the copy before the archive is published.
    """
    directory = Path(directory).expanduser().resolve()
    archive = Path(archive).expanduser().resolve()
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite delta archive: {archive}")
    if archive.is_relative_to(directory):
        raise ValueError("compressed archive must be outside the delta directory")
    compression = _compression_from_path(archive, compression)
    if xz_preset not in XZ_PRESETS:
        raise ValueError(f"unsupported XZ preset: {xz_preset!r}")
    manifest = read_delta_manifest(directory)
    stage = json.loads(_resolve_directory_payload(directory, DELTA_STAGING_NAME).read_bytes())
    if stage.get("format") != "nvarc-exact-delta-staging-v1":
        raise RuntimeError("unsupported/incomplete delta directory staging manifest")
    inventory = stage["files"]
    # Manifest first is part of the archive reader contract; state next and
    # delta shards last match the direct export order.
    names = [DELTA_MANIFEST_NAME, *manifest["state_files"], *[
        item["delta_file"] for item in manifest["trained_layout"]["delta_shards"]
    ]]
    if len(names) != len(set(names)) or set(inventory) != set(names):
        raise RuntimeError("incomplete or duplicate staged delta payload inventory")
    for name in names:
        _safe_relative_path(name)
        entry = inventory[name]
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise RuntimeError(f"invalid staged delta payload size: {name}")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise RuntimeError(f"invalid staged delta payload checksum: {name}")
        if name in manifest["state_files"] and entry != manifest["state_files"][name]:
            raise RuntimeError(f"state-file changed while staging: {name}")
        path = _resolve_directory_payload(directory, name)
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"staged delta payload size mismatch: {name}")
    started = __import__("time").perf_counter()
    with _streaming_tar_writer(archive, compression=compression, xz_preset=xz_preset) as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = inventory[name]["bytes"]
            info.mode = 0o644
            with _resolve_directory_payload(directory, name).open("rb") as stream:
                reader = _CheckedPayloadReader(stream)
                tar.addfile(info, reader)
                if stream.read(1) or reader.size != info.size or reader.digest.hexdigest() != inventory[name]["sha256"]:
                    raise RuntimeError(f"staged delta payload checksum mismatch: {name}")
    archive_bytes = archive.stat().st_size
    source_bytes = sum(entry["bytes"] for entry in inventory.values())
    return {
        "format": manifest["format"], "reference": manifest.get("reference"),
        "manifest_sha256": _manifest_sha256(manifest),
        "delta_directory": str(directory), "archive": str(archive),
        "compression": compression, "xz_preset": xz_preset if compression == "xz" else None,
        "archive_bytes": archive_bytes, "source_bytes": source_bytes,
        "compression_ratio": archive_bytes / max(source_bytes, 1),
        "wall_seconds": __import__("time").perf_counter() - started,
        "payload_checksums_verified": True,
    }


def reconstruct_delta_checkpoint(
    archive: str | Path,
    base_model: str | Path,
    destination: str | Path,
    *,
    work_dir: str | Path | None = None,
    workers: int = DEFAULT_RECONSTRUCT_WORKERS,
    model_only: bool = False,
    parent_checkpoints=(),
) -> dict[str, Any]:
    """Verify oldest-first ancestors; restore run state only from the target."""
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite reconstruction destination: {destination}")
    if int(workers) < 1:
        raise ValueError("delta reconstruction workers must be positive")
    # Validate the entire chain before extracting anything. In particular a v2
    # child is never guessed to be an old base-relative archive.
    chain = _validated_delta_chain([*parent_checkpoints, archive])
    with _resolved_delta_reference(
        base_model, chain[:-1], work_dir=work_dir or destination.parent, workers=workers,
    ) as (reference_model, _):
        result = _reconstruct_delta_checkpoint_single(
            archive, reference_model, destination, work_dir=work_dir,
            workers=workers, model_only=model_only,
            expected_manifest_sha256=_manifest_sha256(chain[-1][1]),
            provenance={
                "base_model": str(Path(base_model).expanduser().resolve()),
                "parent_checkpoints": [str(source) for source, _ in chain[:-1]],
            },
        )
    return result


def rebase_delta_checkpoint(
    source: str | Path,
    base_model: str | Path,
    archive: str | Path,
    *,
    work_dir: str | Path,
    source_parent_checkpoints=(),
    parent_checkpoints=(),
    workers: int = DEFAULT_RECONSTRUCT_WORKERS,
    compression: str = "auto",
    xz_preset: str = "9e",
    max_delta_shard_bytes: int = DEFAULT_DELTA_SHARD_BYTES,
) -> dict[str, Any]:
    """Repack a retained archive without training or changing any saved state."""
    if Path(archive).expanduser().resolve().exists():
        raise FileExistsError(f"refusing to overwrite delta archive: {archive}")
    _validated_delta_chain([*source_parent_checkpoints, source])
    _validated_delta_chain(parent_checkpoints)
    temp_parent = Path(work_dir).expanduser().resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-delta-rebase-", dir=temp_parent) as temporary:
        manifest = read_delta_manifest(source)
        name = str(manifest.get("checkpoint_name") or "checkpoint")
        if name in (".", "..") or Path(name).name != name:
            raise ValueError("invalid source checkpoint name")
        checkpoint = Path(temporary) / name
        reconstruct_delta_checkpoint(
            source, base_model, checkpoint, work_dir=temp_parent, workers=workers,
            parent_checkpoints=source_parent_checkpoints,
        )
        result = pack_delta_checkpoint(
            checkpoint, base_model, archive, work_dir=temp_parent, workers=workers,
            parent_checkpoints=parent_checkpoints, compression=compression,
            xz_preset=xz_preset, max_delta_shard_bytes=max_delta_shard_bytes,
        )
    result["source_archive"] = str(Path(source).expanduser().resolve())
    result["source_parent_checkpoints"] = [str(Path(p).expanduser().resolve()) for p in source_parent_checkpoints]
    result["rebase_only"] = True
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="create an exact delta archive")
    pack.add_argument("--checkpoint", required=True)
    pack.add_argument("--base-model", required=True)
    pack.add_argument("--archive", required=True)
    pack.add_argument("--work-dir", default=None)
    pack.add_argument("--parent-checkpoint", action="append", default=[],
                      help="repeat oldest-first; export relative to the last parent")
    pack.add_argument("--workers", type=int, default=DEFAULT_RECONSTRUCT_WORKERS)
    pack.add_argument(
        "--compression",
        choices=("auto", "gzip", "zstd", "xz", "none"),
        default="auto",
    )
    pack.add_argument(
        "--xz-preset",
        choices=XZ_PRESETS,
        default="9e",
        help="XZ compression preset; lower levels trade archive size for speed",
    )
    pack.add_argument(
        "--max-delta-shard-mib",
        type=int,
        default=DEFAULT_DELTA_SHARD_BYTES // 1024**2,
    )
    stage = subparsers.add_parser("stage", help="encode a complete uncompressed delta folder once")
    stage.add_argument("--checkpoint", required=True)
    stage.add_argument("--base-model", required=True)
    stage.add_argument("--directory", required=True)
    stage.add_argument("--work-dir", default=None)
    stage.add_argument("--parent-checkpoint", action="append", default=[])
    stage.add_argument("--workers", type=int, default=DEFAULT_RECONSTRUCT_WORKERS)
    stage.add_argument("--max-delta-shard-mib", type=int, default=DEFAULT_DELTA_SHARD_BYTES // 1024**2)
    compress = subparsers.add_parser("compress", help="compress a complete staged delta folder; no re-encoding")
    compress.add_argument("--directory", required=True)
    compress.add_argument("--archive", required=True)
    compress.add_argument("--compression", choices=("auto", "gzip", "zstd", "xz", "none"), default="auto")
    compress.add_argument("--xz-preset", choices=XZ_PRESETS, default="9e")
    unpack = subparsers.add_parser("reconstruct", help="rebuild a checkpoint")
    unpack.add_argument("--archive", required=True)
    unpack.add_argument("--base-model", required=True)
    unpack.add_argument("--destination", required=True)
    unpack.add_argument("--work-dir", default=None)
    unpack.add_argument("--parent-checkpoint", action="append", default=[],
                        help="repeat ancestors oldest-first, excluding --archive")
    unpack.add_argument(
        "--workers",
        type=int,
        default=int(
            os.getenv(
                "ARC_DIFFUSION_GEMMA_DELTA_WORKERS",
                str(DEFAULT_RECONSTRUCT_WORKERS),
            )
        ),
        help=(
            "Bounded delta-shard decode workers (default: 1; environment: "
            "ARC_DIFFUSION_GEMMA_DELTA_WORKERS)."
        ),
    )
    unpack.add_argument(
        "--model-only",
        action="store_true",
        help=(
            "reconstruct only verified model weights plus base runtime files; "
            "discard optimizer, scheduler, Trainer, RNG, and ARC data state"
        ),
    )
    rebase = subparsers.add_parser("rebase", help="repack an existing delta against a new reference; no training")
    rebase.add_argument("--source", required=True)
    rebase.add_argument("--base-model", required=True)
    rebase.add_argument("--archive", required=True)
    rebase.add_argument("--work-dir", required=True)
    rebase.add_argument("--source-parent-checkpoint", action="append", default=[],
                        help="oldest-first dependencies of --source (none for a v1 archive)")
    rebase.add_argument("--parent-checkpoint", action="append", default=[],
                        help="new oldest-first reference chain; omit to flatten to the original base")
    rebase.add_argument("--workers", type=int, default=DEFAULT_RECONSTRUCT_WORKERS)
    rebase.add_argument("--compression", choices=("auto", "gzip", "zstd", "xz", "none"), default="auto")
    rebase.add_argument("--xz-preset", choices=XZ_PRESETS, default="9e")
    rebase.add_argument("--max-delta-shard-mib", type=int, default=DEFAULT_DELTA_SHARD_BYTES // 1024**2)
    inspect = subparsers.add_parser("inspect", help="print archive metadata")
    inspect.add_argument("--archive", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "pack":
        result = pack_delta_checkpoint(
            args.checkpoint,
            args.base_model,
            args.archive,
            work_dir=args.work_dir,
            compression=args.compression,
            xz_preset=args.xz_preset,
            max_delta_shard_bytes=args.max_delta_shard_mib * 1024**2,
            parent_checkpoints=args.parent_checkpoint,
            workers=args.workers,
        )
    elif args.command == "reconstruct":
        result = reconstruct_delta_checkpoint(
            args.archive,
            args.base_model,
            args.destination,
            work_dir=args.work_dir,
            workers=args.workers,
            model_only=args.model_only,
            parent_checkpoints=args.parent_checkpoint,
        )
    elif args.command == "stage":
        result = stage_delta_checkpoint(
            args.checkpoint, args.base_model, args.directory, work_dir=args.work_dir,
            parent_checkpoints=args.parent_checkpoint, workers=args.workers,
            max_delta_shard_bytes=args.max_delta_shard_mib * 1024**2,
        )
    elif args.command == "compress":
        result = compress_delta_directory(
            args.directory, args.archive, compression=args.compression, xz_preset=args.xz_preset,
        )
    elif args.command == "rebase":
        result = rebase_delta_checkpoint(
            args.source, args.base_model, args.archive,
            work_dir=args.work_dir, workers=args.workers,
            source_parent_checkpoints=args.source_parent_checkpoint,
            parent_checkpoints=args.parent_checkpoint,
            compression=args.compression, xz_preset=args.xz_preset,
            max_delta_shard_bytes=args.max_delta_shard_mib * 1024**2,
        )
    else:
        result = read_delta_manifest(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
