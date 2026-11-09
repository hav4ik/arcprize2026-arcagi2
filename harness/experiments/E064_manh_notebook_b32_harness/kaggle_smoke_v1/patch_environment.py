"""Resolve patch destinations without importing Torch, vLLM or Unsloth.

An explicit site root is a boundary, not another import search path. Missing
packages must fail instead of falling back to a different Python environment.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


PATCH_SITE_ROOT_ENV = "ARC_PATCH_SITE_ROOT"


def require_file_within(root: Path, relative: str) -> Path:
    """Reject missing files and symlinks/path traversal outside the target."""
    root = root.expanduser().resolve()
    candidate = root / relative
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Patch target escapes {root}: {candidate} -> {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing patch target: {candidate}")
    return resolved


def resolve_package_dir(package_name: str, site_root: Path | None = None) -> Path:
    """CLI root > ARC_PATCH_SITE_ROOT > the launching interpreter's imports.

    ``site_root`` is the directory containing package folders: either a flat
    ``pip install --target`` directory or a venv's actual site-packages directory.
    No package initializer or editable-install .pth file is executed here.
    """
    if not package_name.isidentifier():
        raise ValueError(f"Expected a top-level package name: {package_name!r}")
    configured_root = site_root
    if configured_root is None:
        configured_root = os.environ.get(PATCH_SITE_ROOT_ENV) or None
    if configured_root is not None:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Patch site root is not a directory: {root}")
        try:
            init = require_file_within(root, f"{package_name}/__init__.py")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No {package_name}/ package in requested site root {root}; "
                "refusing to fall back to global packages. Install it into this "
                "directory, or select its intended source explicitly."
            ) from exc
        return init.parent

    spec = importlib.util.find_spec(package_name)
    locations = () if spec is None else spec.submodule_search_locations or ()
    for location in locations:
        package = Path(location).resolve()
        if (package / "__init__.py").is_file():
            return package
    raise FileNotFoundError(
        f"Cannot locate installed {package_name}; pass --site-root or set "
        f"{PATCH_SITE_ROOT_ENV} explicitly"
    )
