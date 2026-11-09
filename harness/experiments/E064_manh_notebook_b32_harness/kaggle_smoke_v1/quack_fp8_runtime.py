"""Narrow, fail-closed SM120 plain-FP8 B-load fix for the audited QuACK build.

The upstream kernel selects its byte-transpose B load only for blockscaled
FP8. Plain E4M3 needs that same load and fragment retile. Generated source is
isolated in a runtime cache; the installed package is never edited.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

UPSTREAM_SHA256 = "d14e5c3b7cf2722b6b236b1ffcf270e74cb871b9ef081e7f5c4c8db8967ff0b0"
PATCHED_SHA256 = "c5c9d3a0d2e9aa2bbb5e36ba5a557f9dec825ae196fbb132045dfc91f2515040"
_OLD = "self.blockscaled and self.b_dtype.width == 8 and self.b_layout.is_n_major_b()"
_NEW = "self.b_dtype.width == 8 and self.b_layout.is_n_major_b()"


def patched_quack_sm120_source(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == PATCHED_SHA256:
        return source
    if digest != UPSTREAM_SHA256 or source.count(_OLD) != 2:
        raise RuntimeError(
            "FP8 SonicMoE requires the audited QuACK SM120 source; "
            f"got sha256={digest}. Refusing to patch an unknown kernel build."
        )
    patched = source.replace(_OLD, _NEW)
    if hashlib.sha256(patched.encode()).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("QuACK FP8 patched source checksum mismatch")
    return patched


def bootstrap_quack_sm120_fp8():
    name = "quack.gemm_sm120"
    existing = sys.modules.get(name)
    if existing is not None:
        path = Path(existing.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() == PATCHED_SHA256:
            return {"source_sha256": PATCHED_SHA256, "already_loaded": True}
        raise RuntimeError("Enable FP8 SonicMoE before importing QuACK GEMMs; restart this Python process")
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError("FP8 SonicMoE requires quack-kernels")
    source = Path(spec.origin).read_text()
    patched = patched_quack_sm120_source(source)
    root = Path(os.environ.get("ARC_QUACK_FP8_RUNTIME_DIR", tempfile.gettempdir()))
    root = root / f"arc_quack_fp8_{PATCHED_SHA256[:16]}"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "gemm_sm120.py"
    if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != PATCHED_SHA256:
        fd, temporary = tempfile.mkstemp(prefix="gemm_sm120-", suffix=".py", dir=root)
        with os.fdopen(fd, "w") as handle:
            handle.write(patched)
        os.replace(temporary, target)
    patched_spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(patched_spec)
    sys.modules[name] = module
    try:
        patched_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return {"source_sha256": PATCHED_SHA256, "already_loaded": False}
