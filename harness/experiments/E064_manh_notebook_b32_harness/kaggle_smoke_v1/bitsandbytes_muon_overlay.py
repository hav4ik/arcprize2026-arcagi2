"""Install the pinned pure-Python bitsandbytes Muon implementation offline.

The Kaggle runtime provides the compiled bitsandbytes CUDA extension, but the
stock wheel may predate the experimental Muon optimizer.  These helpers copy
only the pinned Python optimizer and Triton kernels bundled with this project;
the compiled wheel and its CUDA libraries are left untouched.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys


BITSANDBYTES_MUON_COMMIT = "4779028943707ea6aced4669effc9ebfba51981c"
ARC_PER_HEAD_MUON_VERSION = 1
VENDOR_ROOT = (
    Path(__file__).resolve().parent
    / "third_party"
    / "bitsandbytes_muon"
    / BITSANDBYTES_MUON_COMMIT
    / "bitsandbytes"
)
VENDORED_FILES = {
    Path("optim/muon.py"): (
        "004c3dc4bafeced12f8577649acb4543a59c89f8cac75a239835e305ffbbf5da"
    ),
    Path("backends/triton/kernels_muon.py"): (
        "a54aa46dc46617c553c265767beae44bcf2edc7e7f8358c322578c8c605d90a8"
    ),
    Path("backends/triton/kernels_sym_gemm.py"): (
        "b2fc753bc06f7b2731a0db7807fac304fc1344460c98f5a65f9f51da139608e1"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _muon_available(*, require_precision=False) -> bool:
    """Probe in-process so a freshly installed module is immediately usable."""
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("bitsandbytes.optim.muon")
        kernels = importlib.import_module("bitsandbytes.backends.triton.kernels_muon")
    except (ImportError, AttributeError):
        return False
    return (
        callable(getattr(module, "Muon8bit", None))
        and callable(getattr(module, "MuonBF16", None))
        and getattr(module, "ARC_BF16_MUON_VERSION", None) == 1
        and callable(getattr(kernels, "muon_momentum_8bit_fused", None))
        and callable(getattr(kernels, "muon_momentum_bf16_fused", None))
        and getattr(module, "ARC_PER_HEAD_MUON_VERSION", None)
        == ARC_PER_HEAD_MUON_VERSION
        and (not require_precision or getattr(module, "ARC_PARAMETER_UPDATE_VERSION", None) == 1)
    )


def install_bitsandbytes_muon_overlay(
    package_root: Path,
    *,
    backup_root: Path | None = None,
) -> list[Path]:
    """Atomically overlay checksum-verified Python files onto bitsandbytes."""
    package_root = Path(package_root)
    backup_root = backup_root or (
        Path("/tmp") / "nvarc_bitsandbytes_backup" / BITSANDBYTES_MUON_COMMIT
    )
    installed: list[Path] = []
    for relative_path, expected_sha256 in VENDORED_FILES.items():
        source = VENDOR_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Missing vendored Muon file: {source}")
        actual_sha256 = _sha256(source)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Vendored Muon checksum mismatch for {source}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        destination = package_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and _sha256(destination) != expected_sha256:
            backup = Path(backup_root) / relative_path
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)

        temporary = destination.with_name(
            f".{destination.name}.arc-muon-{os.getpid()}.tmp"
        )
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        installed.append(destination)
    importlib.invalidate_caches()
    return installed


def ensure_bitsandbytes_muon(*, require_precision=False) -> bool:
    """Ensure ARC Muon precision/routing support, returning whether installed."""
    if _muon_available(require_precision=require_precision):
        return False

    spec = importlib.util.find_spec("bitsandbytes")
    locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if len(locations) != 1:
        raise RuntimeError(
            "Cannot locate one installed bitsandbytes package for the pinned "
            f"Muon overlay; locations={locations!r}"
        )
    try:
        installed = install_bitsandbytes_muon_overlay(Path(locations[0]))
    except OSError as error:
        raise RuntimeError(
            "The installed bitsandbytes package is not writable. Install "
            "bitsandbytes in a writable environment and apply the complete "
            "ARC Muon overlay before using --optimizer muon8bit, muonbf16, "
            "or muon32bit."
        ) from error

    # A previous failed import may have left only the package initializer in
    # sys.modules. Remove Muon children without disturbing the compiled wheel.
    for name in tuple(sys.modules):
        if name in {
            "bitsandbytes.optim.muon",
            "bitsandbytes.backends.triton.kernels_muon",
            "bitsandbytes.backends.triton.kernels_sym_gemm",
        }:
            sys.modules.pop(name, None)
    if not _muon_available(require_precision=require_precision):
        raise RuntimeError(
            "Pinned bitsandbytes Muon overlay failed its import probe after "
            f"installing {len(installed)} files from {BITSANDBYTES_MUON_COMMIT}."
        )
    return True
