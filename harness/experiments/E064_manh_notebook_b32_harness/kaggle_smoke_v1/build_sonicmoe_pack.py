#!/usr/bin/env python3
"""Build the audited upstream SonicMoE snapshot as a portable pure-Python wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from sonicmoe_runtime import (
    SONICMOE_PACK_SCHEMA,
    SONICMOE_PACK_VERSION,
    SONICMOE_UPSTREAM_COMMIT,
    inspect_sonicmoe_wheel,
)


UPSTREAM_URL = "https://github.com/Dao-AILab/sonic-moe"


def _run(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_identity(source: Path, *, allow_dirty: bool) -> int:
    commit = _run("git", "rev-parse", "HEAD", cwd=source)
    if commit != SONICMOE_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Expected SonicMoE commit {SONICMOE_UPSTREAM_COMMIT}, found {commit}"
        )
    dirty = _run("git", "status", "--porcelain", "--untracked-files=no", cwd=source)
    if dirty and not allow_dirty:
        raise RuntimeError("SonicMoE source has tracked modifications; refusing to pack")
    return int(_run("git", "show", "-s", "--format=%ct", "HEAD", cwd=source))


def _pyproject() -> str:
    return f'''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sonic-moe"
version = "{SONICMOE_PACK_VERSION}"
description = "Pinned upstream SonicMoE snapshot for offline ARC training"
readme = "README.md"
requires-python = ">=3.12"
license = "Apache-2.0"
license-files = ["LICENSE"]
dependencies = [
    "torch>=2.11.0",
    "quack-kernels>=0.6.4",
]

[project.urls]
Repository = "{UPSTREAM_URL}"

[tool.setuptools.packages.find]
where = ["."]
include = ["sonicmoe*"]

[tool.setuptools.package-data]
sonicmoe = ["ARC_PACK.json"]
'''


def build_pack(
    source: Path,
    output_dir: Path,
    *,
    allow_dirty: bool = False,
) -> Path:
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not (source / "sonicmoe" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not a SonicMoE source checkout: {source}")
    source_date_epoch = _source_identity(source, allow_dirty=allow_dirty)

    output_dir.mkdir(parents=True, exist_ok=True)
    # The artifact hash changes with wheel ZIP timestamps. Remove only the
    # generated adjacent manifest before validating and replacing it below.
    generated_manifest = output_dir / "manifest.json"
    if generated_manifest.exists():
        generated_manifest.unlink()
    with tempfile.TemporaryDirectory(prefix="arc-sonicmoe-pack-") as temporary:
        staging = Path(temporary)
        shutil.copytree(source / "sonicmoe", staging / "sonicmoe")
        shutil.copy2(source / "README.md", staging / "README.md")
        shutil.copy2(source / "LICENSE", staging / "LICENSE")
        (staging / "pyproject.toml").write_text(_pyproject())

        package_init = staging / "sonicmoe" / "__init__.py"
        init_text = package_init.read_text()
        init_text = init_text.replace(
            '__version__ = "0.1.2.post1"',
            f'__version__ = "{SONICMOE_PACK_VERSION}"',
            1,
        )
        if SONICMOE_PACK_VERSION not in init_text:
            raise RuntimeError("Could not stamp SonicMoE package version")
        init_text += (
            "\n# Added only by the ARC packaging step; kernel source is unchanged.\n"
            f'__arc_upstream_commit__ = "{SONICMOE_UPSTREAM_COMMIT}"\n'
            f"__arc_pack_schema__ = {SONICMOE_PACK_SCHEMA}\n"
        )
        package_init.write_text(init_text)
        internal_manifest = {
            "schema": SONICMOE_PACK_SCHEMA,
            "upstream_commit": SONICMOE_UPSTREAM_COMMIT,
            "upstream_url": UPSTREAM_URL,
            "version": SONICMOE_PACK_VERSION,
        }
        (staging / "sonicmoe" / "ARC_PACK.json").write_text(
            json.dumps(internal_manifest, indent=2, sort_keys=True) + "\n"
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output_dir),
            ],
            cwd=staging,
            check=True,
            env={**os.environ, "SOURCE_DATE_EPOCH": str(source_date_epoch)},
        )

    wheels = sorted(output_dir.glob("sonic_moe-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"Wheel build produced no artifact in {output_dir}")
    wheel = wheels[-1]
    pack = inspect_sonicmoe_wheel(wheel)
    manifest = {
        "pack_schema": SONICMOE_PACK_SCHEMA,
        "sha256": pack.sha256,
        "upstream_commit": pack.upstream_commit,
        "upstream_url": UPSTREAM_URL,
        "version": pack.version,
        "wheel": wheel.name,
    }
    generated_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "third_party" / "sonic_moe",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    wheel = build_pack(args.source, args.output_dir, allow_dirty=args.allow_dirty)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    print(f"Built {wheel}")
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
