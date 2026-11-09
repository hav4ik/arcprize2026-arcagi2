#!/usr/bin/env python3
"""Install the native Qwen3 decode optimization without line-number hunks.

Use instead of ``patch ... qwen3.py qwen3.patch`` before importing Unsloth.
Training and vLLM are unchanged. Masked decode retains the original path;
unmasked single-token decode uses FA2 after the existing sliding-window crop.
"""

from __future__ import annotations

import argparse
import ast
import textwrap
from pathlib import Path

from patch_environment import (
    PATCH_SITE_ROOT_ENV,
    require_file_within,
    resolve_package_dir,
)

FUNCTION = "Qwen3Attention_fast_forward_inference"
MARKER = "# ARC native Qwen3 FA2 decode v1 (mask/window safe)."
IMPORT = """try:
    from flash_attn import flash_attn_func as _arc_flash_attn_func
except ImportError:
    _arc_flash_attn_func = None
"""
FAST_BRANCH = """    # ARC native Qwen3 FA2 decode v1 (mask/window safe).
    if (
        _arc_flash_attn_func is not None
        and attention_mask is None
        and Qn.shape[-2] == 1
    ):
        A = _arc_flash_attn_func(
            Qn.transpose(1, 2), Knn.transpose(1, 2), Vnn.transpose(1, 2),
            causal=False,
        )
    else:
"""


def _assignment(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _legacy_attention_body() -> str:
    """Recover the original fallback when the old raw patch fully succeeded."""
    lines = Path(__file__).with_name("qwen3.patch").read_text().splitlines(True)
    removed = "".join(
        line[1:]
        for line in lines
        if line.startswith("-") and not line.startswith("---")
    )
    if (
        "    # Handle sliding windows\n" not in removed
        or "    A = A.transpose(1, 2)\n" not in removed
    ):
        raise RuntimeError(
            "Unrecognized legacy qwen3.patch; cannot recover its fallback"
        )
    return removed


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION
    ]
    if len(functions) != 1:
        raise RuntimeError(f"Expected exactly one {FUNCTION}")
    function = functions[0]
    lines = source.splitlines(True)
    body = "".join(lines[function.lineno - 1 : function.end_lineno])
    if MARKER in body:
        if (
            FAST_BRANCH not in body
            or IMPORT not in source
            or "sliding_window =" not in body
        ):
            raise RuntimeError("Incomplete ARC Qwen3 decode patch; refusing to skip it")
        return source

    # Recover only the exact former unguarded FA2 branch, never an unknown
    # attention implementation. This also handles the user's partial import hunk.
    legacy = """    Qnn = Qn.transpose(1, 2)
    Knn = Kn.transpose(1, 2)
    Vnn = Vn.transpose(1, 2)

    A = flash_attn_func(Qnn, Knn, Vnn)

"""
    if legacy in body:
        source = source.replace(legacy, _legacy_attention_body() + "\n", 1)
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == FUNCTION
        )
        lines = source.splitlines(True)

    q_lengths = [node for node in function.body if _assignment(node, "Q_len")]
    transposes = [
        node
        for node in function.body
        if _assignment(node, "A")
        and ast.dump(node.value)
        == ast.dump(ast.parse("A.transpose(1, 2)", mode="eval").body)
    ]
    windows = [node for node in function.body if _assignment(node, "sliding_window")]
    if len(q_lengths) != 1 or len(transposes) != 1 or len(windows) != 1:
        raise RuntimeError(
            "Unsupported Qwen3 decode attention layout; installed file left unchanged"
        )
    start, end = q_lengths[0].lineno - 1, transposes[0].end_lineno
    if not windows[0].lineno < start < end:
        raise RuntimeError("Unsupported Qwen3 attention block order")
    fallback = "".join(lines[start:end])
    if (
        "scaled_dot_product_attention(" not in fallback
        or "attention_mask" not in fallback
    ):
        raise RuntimeError("Qwen3 decode fallback has no recognized mask handling")
    # The original bsz=1 manual matmul ignores masks. Use the existing SDPA
    # branch for that case too; never silently attend to padded/blocked keys.
    fallback = fallback.replace(
        "    if bsz == 1:\n", "    if bsz == 1 and attention_mask is None:\n"
    )
    lines[start:end] = [FAST_BRANCH + textwrap.indent(fallback, "    ")]
    updated = "".join(lines)

    old_import = "from flash_attn import flash_attn_func\n"
    if old_import in updated:
        updated = updated.replace(old_import, IMPORT, 1)
    else:
        # Insert before this top-level function; do not depend on the imports'
        # formatting or on symbols re-exported through llama's wildcard import.
        updated = updated.replace(
            f"def {FUNCTION}(", IMPORT + f"\n\ndef {FUNCTION}(", 1
        )
    compile(updated, "qwen3.py", "exec")
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root", type=Path, help=f"package root; overrides {PATCH_SITE_ROOT_ENV}"
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    target = require_file_within(
        resolve_package_dir("unsloth", args.site_root), "models/qwen3.py"
    )
    source = target.read_text()
    updated = patch_source(source)
    if updated == source:
        status = "already-present"
    elif args.check_only:
        status = "applicable"
    else:
        backup = target.with_name(target.name + ".arc-before-fa2")
        if not backup.exists():
            backup.write_text(source)
        if target.read_text() != source:
            raise RuntimeError("Qwen3 source changed during preflight; rerun installer")
        target.write_text(updated)
        status = "applied"
    print(f"Unsloth Qwen3 native decode patch: {status}: {target}")
    # Keep .rej/.orig from old manual attempts as diagnostic evidence. They
    # are not imported and are not a sign that this helper failed.


if __name__ == "__main__":
    main()
