#!/usr/bin/env python3
"""Idempotently install the pre-tokenized ARC loss-mask compatibility fix."""

import argparse
from pathlib import Path
import subprocess

from patch_environment import (
    PATCH_SITE_ROOT_ENV,
    require_file_within,
    resolve_package_dir,
)


PATCH_MARKERS = (
    "loss_mask_columns = [",
    'for name in ("completion_mask", "assistant_masks")',
    "used_column_names.extend(loss_mask_columns)",
)


def find_dataset_utils(site_root=None):
    package = resolve_package_dir("unsloth_zoo", site_root)
    return require_file_within(package, "dataset_utils.py")


def patch_is_applied(target):
    source = target.read_text()
    return all(marker in source for marker in PATCH_MARKERS)


def remove_generated_reject(target):
    reject = Path(f"{target}.rej")
    if reject.is_file():
        reject.unlink()
        print(f"Removed stale patch reject: {reject}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target", type=Path, help="explicit dataset_utils.py file (legacy option)"
    )
    target_group.add_argument(
        "--site-root",
        type=Path,
        help=(
            "directory containing unsloth_zoo/; overrides "
            f"{PATCH_SITE_ROOT_ENV}. Missing packages never fall back to global."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify applicability without modifying files",
    )
    parser.add_argument(
        "--patch-file",
        type=Path,
        default=Path(__file__).with_name("unsloth_pretokenized_loss_mask.patch"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    target = (
        args.target.expanduser().resolve()
        if args.target is not None
        else find_dataset_utils(args.site_root)
    )
    if not target.is_file():
        raise FileNotFoundError(target)
    if not args.patch_file.is_file():
        raise FileNotFoundError(args.patch_file)

    print(f"Unsloth Zoo loss-mask patch target: {target}")
    if patch_is_applied(target):
        if not args.check_only:
            remove_generated_reject(target)
        print(f"Unsloth Zoo loss-mask fix already present: {target}")
        return

    command = [
        "patch", "--binary", "--batch", "--forward", str(target), str(args.patch_file)
    ]
    result = subprocess.run(
        [*command, "--dry-run"], text=True, capture_output=True
    )
    if result.returncode == 0:
        if args.check_only:
            print(f"Unsloth Zoo loss-mask fix applicable (check only): {target}")
            return
        result = subprocess.run(command, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    # A concurrent rank/process may have installed it between our check and
    # patch(1). Treat the verified resulting source as success either way.
    if patch_is_applied(target):
        if not args.check_only:
            remove_generated_reject(target)
        print(f"Verified Unsloth Zoo loss-mask fix: {target}")
        return
    raise RuntimeError(
        f"Could not apply {args.patch_file} to {target} "
        f"(patch exit status {result.returncode})"
    )


if __name__ == "__main__":
    main()
