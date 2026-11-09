"""Fresh-interpreter bootstrap for a trainer, serving frontend, or CPU probe."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import traceback

from pipeline_runtime import (
    activate_runtime,
    atomic_json,
    collect_runtime_manifest,
    verify_trainer_cache_support,
    verify_runtime_manifest,
)


def probe(report, *, check_only):
    role = os.environ["ARC_PIPELINE_ROLE"]
    patch_args = ["--check-only"] if check_only else []
    if role == "trainer":
        from ensure_unsloth_loss_mask_patch import main as ensure_patch
        verify_trainer_cache_support()
    else:
        from ensure_vllm_arc_runtime_patches import main as ensure_patch
    ensure_patch(patch_args)
    # Verify that the selected interpreter can load these packages, not just
    # locate their installed metadata.
    import torch  # noqa: F401
    import transformers  # noqa: F401

    atomic_json(report, collect_runtime_manifest(role))


def execute_job(path):
    job = json.loads(Path(path).read_text())
    if job["schema_version"] != 1 or job["role"] != os.environ["ARC_PIPELINE_ROLE"]:
        raise ValueError("Invalid split-runtime launch schema/role")
    stop_watch = threading.Event()

    def watch_parent():
        while not stop_watch.wait(1):
            if os.getppid() != job["parent_pid"]:
                # Own session only; descendants should not survive a dead
                # coordinator and continue consuming notebook GPUs.
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return

    threading.Thread(target=watch_parent, daemon=True).start()

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        from pipeline_runtime_ipc import decode

        credentials = json.loads(Path(job["credentials"]).read_text())
        kwargs = decode(
            job["kwargs"],
            address=credentials["address"],
            authkey=bytes.fromhex(credentials["authkey"]),
        )
        module, name = job["target"].split(":", 1)
        target = getattr(importlib.import_module(module), name)
        manifest = collect_runtime_manifest(job["role"])
        expected = json.loads(
            Path(os.environ["ARC_PIPELINE_EXPECTED_MANIFEST"]).read_text()
        )
        verify_runtime_manifest(manifest, expected)
        atomic_json(
            Path(os.environ["ARC_PIPELINE_REPORT_DIR"]) / f"worker-{os.getpid()}.json",
            manifest,
        )
        target(job["rank"], **kwargs)
        atomic_json(
            job["exit_path"],
            {"status": "completed", "pid": os.getpid(), "role": job["role"]},
        )
    except BaseException as error:
        atomic_json(
            job["error_path"],
            {
                "pid": os.getpid(),
                "role": job["role"],
                "python": sys.executable,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        stop_watch.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--probe", action="store_true")
    action.add_argument("--job")
    parser.add_argument("--report")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    activate_runtime()
    if args.probe:
        if not args.report:
            parser.error("--probe requires --report")
        probe(args.report, check_only=args.check_only)
    else:
        execute_job(args.job)


if __name__ == "__main__":
    main()
