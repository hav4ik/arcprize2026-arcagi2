"""Select a runtime-matched SM89 FA3 wheel before Unsloth imports its dispatcher.

Enabled only by UNSLOTH_FLASH_ATTENTION_3=1. No compiler, pip, network, or
writable site-packages is needed. Native extensions must be extracted to an
executable filesystem, so the default intentionally avoids /dev/shm.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tempfile
import zipfile


def inspect_flash_attention3_wheel(wheel: Path) -> dict:
    manifest = json.loads(Path(str(wheel) + ".build.json").read_text())
    if manifest.get("artifact") != wheel.name:
        raise ValueError("FA3 manifest artifact does not match the wheel filename")
    if manifest.get("bytes") != wheel.stat().st_size:
        raise ValueError("FA3 wheel size does not match its build manifest")
    hasher = hashlib.sha256()
    with wheel.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    checksum = Path(str(wheel) + ".sha256").read_text().split()[0]
    if digest != manifest.get("sha256") or digest != checksum:
        raise ValueError("FA3 wheel SHA-256 does not match its manifest/checksum")
    return manifest


def validate_flash_attention3_runtime(manifest: dict, runtime: dict) -> None:
    """Check the tested binary ABI instead of guessing compatibility by version."""
    fields = (
        "torch",
        "torch_cuda",
        "cxx11_abi",
        "platform",
        "machine",
        "target_compute_capability",
        "python_implementation",
    )
    mismatches = {
        key: {"built": manifest.get(key), "runtime": runtime.get(key)}
        for key in fields
        if manifest.get(key) != runtime.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Bundled FA3 wheel is incompatible with this runtime: "
            f"{mismatches}. Use a matching SM89 wheel or disable "
            "UNSLOTH_FLASH_ATTENTION_3."
        )
    if sys.version_info < (3, 10):
        raise RuntimeError("The bundled FA3 abi3 wheel requires Python >=3.10")


def _runtime_identity() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("UNSLOTH_FLASH_ATTENTION_3 requires a CUDA GPU")
    return {
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "target_compute_capability": ".".join(
            map(str, torch.cuda.get_device_capability())
        ),
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
    }


def _find_wheel(runtime: dict) -> Path:
    """Select by recorded ABI, never by filename order or Torch major version."""
    override = os.getenv("ARC_FA3_WHEEL")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parent / "third_party" / "flash_attention3"
    wheels = sorted(root.glob("flash_attn_3-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            f"No bundled FA3 wheels found in {root}. "
            "Copy third_party/flash_attention3 with kaggle_submit, or set "
            "ARC_FA3_WHEEL to a wheel with .build.json and .sha256 sidecars."
        )
    matches = []
    rejected = {}
    for wheel in wheels:
        try:
            manifest = json.loads(Path(str(wheel) + ".build.json").read_text())
            if not isinstance(manifest, dict) or manifest.get("artifact") != wheel.name:
                raise ValueError("invalid manifest or mismatched artifact filename")
            validate_flash_attention3_runtime(manifest, runtime)
        except (OSError, ValueError, RuntimeError) as error:
            rejected[wheel.name] = str(error)
        else:
            matches.append(wheel)
    if not matches:
        raise RuntimeError(
            f"No bundled FA3 wheel matches this runtime: {runtime}. "
            f"Rejected candidates: {rejected}. Copy the matching wheel and its "
            "sidecars, or disable UNSLOTH_FLASH_ATTENTION_3."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple bundled FA3 wheels match this runtime: {[w.name for w in matches]}. "
            "Set ARC_FA3_WHEEL to the intended artifact; refusing an ambiguous selection."
        )
    # The selected artifact's bytes and SHA-256 are checked by the bootstrap
    # before extraction. Incompatible wheels are never loaded or imported.
    return matches[0]


def _extract_wheel(wheel: Path, manifest: dict, root: Path) -> Path:
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    if os.statvfs(root).f_flag & getattr(os, "ST_NOEXEC", 8):
        raise RuntimeError(
            f"FA3 cannot load native extensions from noexec filesystem {root}; "
            "set ARC_FA3_RUNTIME_DIR to a directory under /kaggle/working."
        )
    target = root / manifest["sha256"]
    marker = {"sha256": manifest["sha256"], "artifact": wheel.name}
    with (root / ".extract.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.exists():
            ready = target / ".arc_fa3_ready.json"
            if (
                ready.is_file()
                and json.loads(ready.read_text()) == marker
                and (target / "flash_attn_3/flash_attn_interface.py").is_file()
                and any((target / "flash_attn_3").glob("_C.*.so"))
            ):
                return target
            raise RuntimeError(
                f"Incomplete FA3 extraction at {target}; use a fresh ARC_FA3_RUNTIME_DIR"
            )
        temporary = Path(tempfile.mkdtemp(prefix=".fa3-", dir=root))
        try:
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                for name in names:
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                        raise ValueError(f"Unsafe FA3 wheel member: {name!r}")
                if "flash_attn_3/flash_attn_interface.py" not in names or not any(
                    name.startswith("flash_attn_3/_C.") and name.endswith(".so")
                    for name in names
                ):
                    raise ValueError(
                        "FA3 wheel lacks its Python interface or native extension"
                    )
                archive.extractall(temporary)
            (temporary / ".arc_fa3_ready.json").write_text(json.dumps(marker) + "\n")
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def bootstrap_flash_attention3_runtime() -> Path | None:
    """Load the exact bundled extension once per trainer, before importing Unsloth."""
    if os.getenv("UNSLOTH_FLASH_ATTENTION_3", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    current_runtime = _runtime_identity()
    wheel = _find_wheel(current_runtime)
    manifest = inspect_flash_attention3_wheel(wheel)
    validate_flash_attention3_runtime(manifest, current_runtime)
    default_root = (
        Path("/kaggle/working")
        if Path("/kaggle/working").is_dir()
        else Path(tempfile.gettempdir())
    ) / "arc_flash_attention3_runtime"
    root = (
        Path(os.getenv("ARC_FA3_RUNTIME_DIR", str(default_root))).expanduser().resolve()
    )
    target = _extract_wheel(wheel, manifest, root)
    loaded = sys.modules.get("flash_attn_3")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(target):
            raise RuntimeError(
                "A different FA3 package was imported before the bundled wheel. "
                "Start a fresh training process so native operators are registered once."
            )
    target_text = str(target)
    if target_text not in sys.path:
        sys.path.insert(0, target_text)
    importlib.invalidate_caches()
    try:
        importlib.import_module("flash_attn_3.flash_attn_interface")
    except Exception as error:
        raise RuntimeError(
            f"Bundled FA3 native import failed ({type(error).__name__}: {error}); "
            "check the wheel/runtime compatibility before starting training."
        ) from error
    if loaded is None:
        print(
            f"[FA3] bundled SM89 wheel ready: {wheel.name}; "
            f"sha256={manifest['sha256'][:16]}; runtime={target}",
            flush=True,
        )
    return target
