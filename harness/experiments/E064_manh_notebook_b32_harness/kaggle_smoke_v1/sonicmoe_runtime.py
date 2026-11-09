"""Load the pinned SonicMoE source snapshot without a notebook-time clone.

SonicMoE is a small pure-Python frontend over QuACK.  The ARC training path
benchmarks a specific upstream-main commit, while the published package still
uses the same version string as an older snapshot.  A repository-vendored
wheel plus this extractor makes that distinction explicit and keeps
``PYTHONPATH=.`` sufficient on a fresh offline Kaggle runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Iterable, Mapping, MutableSequence
import zipfile


SONICMOE_UPSTREAM_COMMIT = "7396f3e604827d8186c2e16e64b28ee33d3defd0"
SONICMOE_PACK_VERSION = "0.1.2.post1+arc.7396f3e"
SONICMOE_PACK_SCHEMA = 1
SONICMOE_WHEEL_ENV = "ARC_SONICMOE_WHEEL"
SONICMOE_RUNTIME_DIR_ENV = "ARC_SONICMOE_RUNTIME_DIR"
SONICMOE_PACK_GLOB = "sonic_moe-0.1.2.post1+arc.7396f3e-*.whl"


@dataclass(frozen=True)
class SonicMoEPack:
    wheel: Path
    sha256: str
    upstream_commit: str
    version: str
    runtime_dir: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_sonicmoe_wheel(path: str | os.PathLike[str]) -> SonicMoEPack:
    """Validate and describe an ARC-pinned SonicMoE wheel."""

    wheel = Path(path).expanduser().resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"SonicMoE wheel does not exist: {wheel}")
    try:
        with zipfile.ZipFile(wheel) as archive:
            raw_manifest = archive.read("sonicmoe/ARC_PACK.json")
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ValueError(
                    "expected exactly one .dist-info/METADATA entry, found "
                    f"{len(metadata_names)}"
                )
            metadata = archive.read(metadata_names[0]).decode("utf-8")
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid ARC SonicMoE wheel {wheel}: {error}") from error

    try:
        manifest = json.loads(raw_manifest)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid SonicMoE ARC_PACK.json in {wheel}") from error

    expected = {
        "schema": SONICMOE_PACK_SCHEMA,
        "upstream_commit": SONICMOE_UPSTREAM_COMMIT,
        "version": SONICMOE_PACK_VERSION,
    }
    observed = {key: manifest.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"SonicMoE pack identity mismatch in {wheel}: "
            f"expected {expected}, found {observed}"
        )
    if "Name: sonic-moe\n" not in metadata:
        raise ValueError(f"Wheel metadata is not for sonic-moe: {wheel}")
    if f"Version: {SONICMOE_PACK_VERSION}\n" not in metadata:
        raise ValueError(
            f"Wheel metadata does not contain version {SONICMOE_PACK_VERSION}: "
            f"{wheel}"
        )
    sha256 = _sha256(wheel)
    adjacent_manifest = wheel.parent / "manifest.json"
    if adjacent_manifest.is_file():
        try:
            external = json.loads(adjacent_manifest.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid adjacent SonicMoE manifest {adjacent_manifest}"
            ) from error
        if external.get("wheel") == wheel.name:
            external_identity = {
                "pack_schema": external.get("pack_schema"),
                "sha256": external.get("sha256"),
                "upstream_commit": external.get("upstream_commit"),
                "version": external.get("version"),
            }
            expected_external = {
                "pack_schema": SONICMOE_PACK_SCHEMA,
                "sha256": sha256,
                "upstream_commit": SONICMOE_UPSTREAM_COMMIT,
                "version": SONICMOE_PACK_VERSION,
            }
            if external_identity != expected_external:
                raise ValueError(
                    f"SonicMoE manifest mismatch in {adjacent_manifest}: "
                    f"expected {expected_external}, found {external_identity}"
                )
    return SonicMoEPack(
        wheel=wheel,
        sha256=sha256,
        upstream_commit=SONICMOE_UPSTREAM_COMMIT,
        version=SONICMOE_PACK_VERSION,
    )


def _local_find_links(environment: Mapping[str, str]) -> Iterable[Path]:
    for variable in ("PIP_FIND_LINKS", "UV_FIND_LINKS"):
        for entry in environment.get(variable, "").split():
            if entry.startswith("file://"):
                entry = entry[7:]
            if "://" not in entry:
                yield Path(entry).expanduser()


def _candidate_wheels(environment: Mapping[str, str]) -> Iterable[Path]:
    override = environment.get(SONICMOE_WHEEL_ENV)
    if override:
        path = Path(override).expanduser()
        if path.is_dir():
            yield from sorted(path.glob(SONICMOE_PACK_GLOB))
        else:
            yield path
        return

    bundled = Path(__file__).resolve().parent / "third_party" / "sonic_moe"
    yield from sorted(bundled.glob(SONICMOE_PACK_GLOB))
    for directory in _local_find_links(environment):
        if directory.is_dir():
            yield from sorted(directory.glob(SONICMOE_PACK_GLOB))

    # Do not bake a Kaggle account name into the runtime.  This catches the
    # user's normal offline wheel datasets if the small wheel is stored there
    # instead of committed beside this module.
    kaggle_datasets = Path("/kaggle/input/datasets")
    if kaggle_datasets.is_dir():
        yield from sorted(
            kaggle_datasets.glob(f"*/wheel-arc/wheels/wheels/{SONICMOE_PACK_GLOB}")
        )


def find_sonicmoe_wheel(
    environment: Mapping[str, str] | None = None,
) -> SonicMoEPack:
    """Return the first valid exact pack from repository or offline wheels."""

    environment = os.environ if environment is None else environment
    override = environment.get(SONICMOE_WHEEL_ENV)
    errors: list[str] = []
    seen: set[Path] = set()
    for candidate in _candidate_wheels(environment):
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            return inspect_sonicmoe_wheel(resolved)
        except (FileNotFoundError, ValueError) as error:
            errors.append(str(error))

    suffix = f" Checked: {'; '.join(errors)}" if errors else ""
    if override:
        raise RuntimeError(
            f"{SONICMOE_WHEEL_ENV} did not resolve to the pinned SonicMoE pack."
            + suffix
        )
    raise FileNotFoundError(
        "Missing pinned SonicMoE pack. Keep third_party/sonic_moe in the "
        "repository checkout, add the wheel to wheel-arc, or set "
        f"{SONICMOE_WHEEL_ENV}. Expected commit {SONICMOE_UPSTREAM_COMMIT}."
        + suffix
    )


def _runtime_root(environment: Mapping[str, str]) -> Path:
    override = environment.get(SONICMOE_RUNTIME_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and os.access(shared_memory, os.W_OK):
        return shared_memory / "arc_sonicmoe_runtime"
    return Path(tempfile.gettempdir()) / "arc_sonicmoe_runtime"


def _safe_extract_wheel(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Unsafe member {member.filename!r} in SonicMoE wheel {wheel}"
                )
        archive.extractall(destination)


def _extracted_pack_matches(target: Path, pack: SonicMoEPack) -> bool:
    marker = target / ".arc_sonicmoe_ready.json"
    try:
        observed = json.loads(marker.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return observed == {
        "schema": SONICMOE_PACK_SCHEMA,
        "sha256": pack.sha256,
        "upstream_commit": pack.upstream_commit,
        "version": pack.version,
    }


def _extract_pack(pack: SonicMoEPack, environment: Mapping[str, str]) -> Path:
    root = _runtime_root(environment)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"sonic_moe-{pack.sha256[:16]}"
    if _extracted_pack_matches(target, pack):
        return target

    # Trainer workers can enter this path together.  A process lock keeps the
    # extraction atomic without requiring pip or writable site-packages.
    import fcntl

    lock_path = root / ".extract.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _extracted_pack_matches(target, pack):
            return target
        temporary = Path(tempfile.mkdtemp(prefix=".sonic-moe-", dir=root))
        try:
            _safe_extract_wheel(pack.wheel, temporary)
            marker = {
                "schema": SONICMOE_PACK_SCHEMA,
                "sha256": pack.sha256,
                "upstream_commit": pack.upstream_commit,
                "version": pack.version,
            }
            (temporary / ".arc_sonicmoe_ready.json").write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n"
            )
            if target.exists():
                shutil.rmtree(target)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def bootstrap_sonicmoe_runtime(
    *,
    environment: Mapping[str, str] | None = None,
    sys_path: MutableSequence[str] | None = None,
) -> SonicMoEPack:
    """Expose the pinned wheel on ``sys.path`` and validate prior imports."""

    environment = os.environ if environment is None else environment
    sys_path = sys.path if sys_path is None else sys_path
    pack = find_sonicmoe_wheel(environment)
    loaded = sys.modules.get("sonicmoe")
    if loaded is not None:
        loaded_commit = getattr(loaded, "__arc_upstream_commit__", None)
        if loaded_commit != SONICMOE_UPSTREAM_COMMIT:
            raise RuntimeError(
                "A different SonicMoE package was imported before the ARC "
                f"bootstrap (commit={loaded_commit!r}); remove the manual "
                "sonic-moe PYTHONPATH entry and start a fresh process."
            )

    target = _extract_pack(pack, environment)
    target_text = str(target)
    if target_text in sys_path:
        sys_path.remove(target_text)
    sys_path.insert(0, target_text)
    importlib.invalidate_caches()
    return SonicMoEPack(
        wheel=pack.wheel,
        sha256=pack.sha256,
        upstream_commit=pack.upstream_commit,
        version=pack.version,
        runtime_dir=target,
    )
