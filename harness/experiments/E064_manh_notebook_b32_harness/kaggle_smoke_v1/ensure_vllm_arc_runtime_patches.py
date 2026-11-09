#!/usr/bin/env python3
"""Apply the ARC vLLM runtime patches across supported vLLM layouts.

The original runtime patch targets commit c6fe94b4d.  Newer vLLM already
contains the uniform-decode/prefill correctness fix that was backported in one
hunk of that patch, while the remaining ARC containment and LoRA fixes still
apply.  Applying one hunk at a time lets this installer:

* skip only that obsolete backport on modern vLLM;
* retain the backport on the c6 runtime;
* recover cleanly when a previous ``patch`` command applied only some hunks;
* remain idempotent when run more than once.

Run this before importing vLLM in the notebook kernel.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from patch_environment import (
    PATCH_SITE_ROOT_ENV,
    require_file_within,
    resolve_package_dir,
)

LEGACY_UNIFORM_GUARD_MARKER = "no-spec backport of upstream vLLM #51865"
MODERN_UNIFORM_HELPER = "get_uniform_decode_token_count"

# Commit 8c1557a79c539ffe82d004d2a0c8d7b5e71159ce reformatted the
# _get_lora_b_ptr docstring and removed a blank line immediately before the
# cache-key assignment.  The original unified hunk used that blank line as
# interior context, so GNU patch cannot fuzz across the otherwise harmless
# formatting-only change.  Handle this one tiny source transformation
# semantically so the installer works both before and after that commit.
LORA_B_PTR_TARGET = "vllm/lora/ops/triton_ops/utils.py"
LORA_B_CACHE_KEY_OLD = (
    "    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)\n"
)
LORA_B_CACHE_WEIGHT_PTRS = (
    "    weight_ptrs = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)\n"
)
LORA_B_CACHE_KEY_NEW = "    key = (weight_ptrs, offset_start)\n"
LORA_B_CACHE_KEY_NEW_BLOCK = LORA_B_CACHE_WEIGHT_PTRS + LORA_B_CACHE_KEY_NEW


@dataclass(frozen=True)
class PatchHunk:
    patch_name: str
    target: str
    text: str

    @property
    def description(self) -> str:
        header = next(
            (line for line in self.text.splitlines() if line.startswith("@@")),
            "hunk",
        )
        return f"{self.patch_name}:{self.target}:{header}"


def _default_site_root(site_root: Path | None = None) -> Path:
    package_dir = resolve_package_dir("vllm", site_root)
    if package_dir.name != "vllm":
        raise RuntimeError(f"Unexpected vLLM package location: {package_dir}")
    return package_dir.parent


def _split_patch_hunks(patch_path: Path) -> list[PatchHunk]:
    """Split a text git patch into independently applicable unified hunks."""
    lines = patch_path.read_text(encoding="utf-8").splitlines(keepends=True)
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)

    result: list[PatchHunk] = []
    for section in sections:
        try:
            first_hunk = next(
                i for i, line in enumerate(section) if line.startswith("@@")
            )
        except StopIteration as exc:
            raise RuntimeError(f"Patch section has no hunk: {patch_path}") from exc
        prefix = section[:first_hunk]
        target_line = next(
            (line for line in prefix if line.startswith("+++ b/")),
            None,
        )
        if target_line is None:
            raise RuntimeError(f"Patch section has no target path: {patch_path}")
        target = target_line.removeprefix("+++ b/").strip()

        starts = [
            i for i in range(first_hunk, len(section)) if section[i].startswith("@@")
        ]
        starts.append(len(section))
        for start, stop in zip(starts, starts[1:]):
            result.append(
                PatchHunk(
                    patch_name=patch_path.name,
                    target=target,
                    text="".join(prefix + section[start:stop]),
                )
            )
    return result


def _run_patch(site_root: Path, hunk: PatchHunk, *, reverse: bool, dry_run: bool):
    require_file_within(site_root, hunk.target)
    command = [
        "patch",
        "--binary",
        "--batch",
        "-p1",
        "-d",
        str(site_root),
    ]
    if dry_run:
        command.append("--dry-run")
    command.append("--reverse" if reverse else "--forward")
    return subprocess.run(
        command,
        input=hunk.text,
        text=True,
        capture_output=True,
        check=False,
    )


def _apply_lora_b_cache_key_hunk(
    site_root: Path, hunk: PatchHunk, *, apply: bool
) -> str | None:
    """Apply the LoRA-B cache-key hunk without docstring-sensitive context.

    Returns ``None`` for unrelated hunks so they continue through the normal
    unified-patch path.  This keeps the workaround tightly scoped while making
    this hunk idempotent across both pre- and post-pydocstyle vLLM sources.
    """
    if (
        hunk.target != LORA_B_PTR_TARGET
        or LORA_B_CACHE_KEY_OLD not in hunk.text
        or LORA_B_CACHE_KEY_NEW not in hunk.text
    ):
        return None

    path = require_file_within(site_root, hunk.target)
    source = path.read_text(encoding="utf-8")

    # The generic reverse dry-run also fails on post-8c1557 sources because its
    # surrounding docstring context changed, so detect the applied form first.
    if LORA_B_CACHE_KEY_NEW_BLOCK in source:
        return "already-present"

    count = source.count(LORA_B_CACHE_KEY_OLD)
    if count != 1:
        raise RuntimeError(
            f"Unsupported vLLM source around {hunk.description}: expected exactly "
            f"one legacy LoRA-B cache-key assignment, found {count}."
        )

    if apply:
        path.write_text(
            source.replace(
                LORA_B_CACHE_KEY_OLD,
                LORA_B_CACHE_KEY_NEW_BLOCK,
                1,
            ),
            encoding="utf-8",
        )
    return "applied" if apply else "applicable"


def _modern_uniform_guard_present(site_root: Path) -> bool:
    source = site_root / "vllm/v1/worker/gpu/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for function in ast.walk(tree):
        if (
            not isinstance(function, ast.FunctionDef)
            or function.name != "gather_batch_req_state"
        ):
            continue
        eligibility_from_prefill = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "decode_graph_eligible"
                for t in node.targets
            )
            and ast.dump(node.value)
            == ast.dump(ast.parse("not has_prefill", mode="eval").body)
            for node in ast.walk(function)
        )
        for node in ast.walk(function):
            if not isinstance(node, ast.Return) or not isinstance(
                node.value, ast.Tuple
            ):
                continue
            for call in node.value.elts:
                if not isinstance(call, ast.Call) or not isinstance(
                    call.func, ast.Name
                ):
                    continue
                if call.func.id != MODERN_UNIFORM_HELPER:
                    continue
                guard = call.args[3] if len(call.args) > 3 else None
                if isinstance(guard, ast.Attribute) and guard.attr == "has_prefill":
                    return True
                # f700178f9 renamed/reworked this input to allow validated
                # one-token extensions. Do not replace its newer policy with
                # the old no-spec backport. An import/dummy call alone is not
                # evidence that real batches use the upstream guard.
                if (
                    eligibility_from_prefill
                    and isinstance(guard, ast.Name)
                    and guard.id == "decode_graph_eligible"
                ):
                    return True
    return False


def _apply_hunk(site_root: Path, hunk: PatchHunk, *, apply: bool) -> str:
    semantic_status = _apply_lora_b_cache_key_hunk(site_root, hunk, apply=apply)
    if semantic_status is not None:
        return semantic_status

    forward = _run_patch(site_root, hunk, reverse=False, dry_run=True)
    if forward.returncode == 0:
        if apply:
            result = _run_patch(site_root, hunk, reverse=False, dry_run=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Patch changed between dry-run and apply: {hunk.description}\n"
                    f"{result.stdout}{result.stderr}"
                )
        return "applied" if apply else "applicable"

    reverse = _run_patch(site_root, hunk, reverse=True, dry_run=True)
    if reverse.returncode == 0:
        return "already-present"

    raise RuntimeError(
        f"Unsupported vLLM source around {hunk.description}\n"
        f"forward check:\n{forward.stdout}{forward.stderr}\n"
        f"reverse check:\n{reverse.stdout}{reverse.stderr}"
    )


def _clean_obsolete_reject(site_root: Path, modern_uniform_guard: bool) -> None:
    if not modern_uniform_guard:
        return
    reject = site_root / "vllm/v1/worker/gpu/model_runner.py.rej"
    if not reject.exists():
        return
    text = reject.read_text(encoding="utf-8", errors="replace")
    hunk_headers = [line for line in text.splitlines() if line.startswith("@@")]
    if LEGACY_UNIFORM_GUARD_MARKER not in text or len(hunk_headers) != 1:
        raise RuntimeError(f"Refusing to remove an unrecognized reject file: {reject}")
    reject.unlink()
    print(f"removed obsolete legacy-only reject: {reject}")


def _validate_sources(site_root: Path, modern_uniform_guard: bool) -> None:
    checks = [
        ("vllm/lora/model_manager.py", "def get_lm_punica_wrapper("),
        ("vllm/lora/ops/triton_ops/utils.py", "key = (weight_ptrs, offset_start)"),
        (
            "vllm/lora/worker_manager.py",
            "return self._adapter_manager.get_lm_punica_wrapper()",
        ),
        ("vllm/v1/worker/gpu/input_batch.py", "VOCAB_SIZE=vocab_size"),
        (
            "vllm/v1/worker/gpu/sample/logprob.py",
            "safe_topk_ids = tl.where(valid_id, topk_ids, 0)",
        ),
        ("vllm/v1/worker/gpu/model_runner.py", "prompt_logits_fn"),
        (
            "vllm/v1/worker/gpu_model_runner.py",
            "prompt_mapping_meta.prepare_tensors(token_slice)",
        ),
        (
            "vllm/v1/worker/gpu_model_runner.py",
            "compiled sampler may return tensors backed by reusable graph",
        ),
    ]
    if modern_uniform_guard:
        if not _modern_uniform_guard_present(site_root):
            raise RuntimeError("Post-patch validation lost the upstream decode guard")
    else:
        checks.append(
            ("vllm/v1/worker/gpu/model_runner.py", LEGACY_UNIFORM_GUARD_MARKER)
        )

    missing = []
    for relative, marker in checks:
        path = site_root / relative
        if not path.is_file() or marker not in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            missing.append(f"{relative}: {marker!r}")
    if missing:
        raise RuntimeError("Post-patch validation failed:\n  " + "\n  ".join(missing))


def _byte_compile(site_root: Path) -> None:
    paths = [
        site_root / "vllm/v1/worker/gpu/model_runner.py",
        site_root / "vllm/v1/worker/gpu_model_runner.py",
        site_root / "vllm/v1/worker/gpu/input_batch.py",
        site_root / "vllm/v1/worker/gpu/sample/logprob.py",
    ]
    subprocess.run(
        [sys.executable, "-m", "py_compile", *(str(path) for path in paths)],
        check=True,
    )


def _commit_staged_sources(site_root, staged_root, originals):
    patched = {
        relative: (staged_root / relative).read_bytes() for relative in originals
    }
    for relative, original in originals.items():
        current = (site_root / relative).read_bytes()
        # Another worker may have completed the same installer during our
        # preflight. Identical final bytes are safe, unrelated edits are not.
        if current not in (original, patched[relative]):
            raise RuntimeError(
                f"Patch target changed during preflight: {relative}; rerun installer"
            )
    for relative, content in patched.items():
        target = site_root / relative
        if target.read_bytes() == content:
            continue
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(content)
                handle.flush()
                shutil.copymode(target, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root",
        type=Path,
        default=None,
        help=(
            "directory containing vllm/; overrides "
            f"{PATCH_SITE_ROOT_ENV}. Missing packages never fall back to global."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify applicability without modifying files",
    )
    args = parser.parse_args(argv)

    site_root = _default_site_root(args.site_root)
    print(f"vLLM ARC patch site root: {site_root}")

    patch_dir = Path(__file__).resolve().parent / "patches"
    patch_paths = [
        patch_dir / "vllm_c6_arc_stability_runtime.patch",
        patch_dir / "vllm_c6_async_sampler_snapshot_runtime.patch",
    ]
    hunks = [hunk for path in patch_paths for hunk in _split_patch_hunks(path)]
    # Validate every destination before patching any file. In particular, a
    # package copied with links to global files must not mutate that install.
    for hunk in hunks:
        require_file_within(site_root, hunk.target)
    modern_uniform_guard = _modern_uniform_guard_present(site_root)
    print(
        "vLLM ARC patch mode: "
        + ("modern-integrated-uniform-guard" if modern_uniform_guard else "legacy-c6")
    )

    # Resolve the entire sequential patch on isolated copies first. In older
    # installers an unsupported late hunk left the live package half-patched.
    originals = {
        hunk.target: require_file_within(site_root, hunk.target).read_bytes()
        for hunk in hunks
    }
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="arc-vllm-patch-") as temporary:
        staged_root = Path(temporary)
        for relative in originals:
            target = staged_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(site_root / relative, target)
        for hunk in hunks:
            if modern_uniform_guard and LEGACY_UNIFORM_GUARD_MARKER in hunk.text:
                status = "skipped-upstream"
            else:
                status = _apply_hunk(staged_root, hunk, apply=True)
            if status == "applied":
                status = "applicable" if args.check_only else "staged"
            counts[status] = counts.get(status, 0) + 1
            print(f"{status:>15}  {hunk.description}")
        _validate_sources(staged_root, modern_uniform_guard)
        for relative in originals:
            compile((staged_root / relative).read_bytes(), relative, "exec")
        if not args.check_only:
            _commit_staged_sources(site_root, staged_root, originals)
            if "staged" in counts:
                counts["applied"] = counts.pop("staged")

    if not args.check_only:
        _clean_obsolete_reject(site_root, modern_uniform_guard)
        _validate_sources(site_root, modern_uniform_guard)
        _byte_compile(site_root)
    print("vLLM ARC runtime patch summary:", counts)


if __name__ == "__main__":
    main()
