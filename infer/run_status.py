"""Atomic terminal-status reporting for Kaggle private reruns."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from typing import Any


RUN_STATUS_ENV = "ARC_RUN_STATUS_PATH"
STATUS_RUNNING = "running"
STATUS_STARTER_SUCCESS = "starter_success"
STATUS_OUT_OF_MEMORY = "out_of_memory"
STATUS_EXCEPTION = "exception"
STATUS_SUBMISSION_MISSING = "submission_missing"
STATUS_SUCCESS = "success"

TERMINAL_STATUSES = frozenset(
    {
        STATUS_STARTER_SUCCESS,
        STATUS_OUT_OF_MEMORY,
        STATUS_EXCEPTION,
        STATUS_SUBMISSION_MISSING,
        STATUS_SUCCESS,
    }
)

_OOM_TEXT_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "defaultcpuallocator: can't allocate memory",
    "cannot allocate memory",
)


def extract_run_status_path(argv: list[str] | None = None) -> str | None:
    """Read the status path without requiring the full application parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-status-file")
    args, _ = parser.parse_known_args(argv)
    value = args.run_status_file or os.getenv(RUN_STATUS_ENV)
    if value is None or not str(value).strip():
        return None
    return os.path.abspath(os.path.expanduser(str(value).strip()))


def iter_exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit/implicit causes once each."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_out_of_memory_error(error: BaseException) -> bool:
    """Recognize direct and multiprocessing-wrapped CPU/CUDA OOM failures."""
    for chained in iter_exception_chain(error):
        if isinstance(chained, MemoryError):
            return True
        class_name = type(chained).__name__.lower()
        message = str(chained).lower()
        if "outofmemory" in class_name:
            return True
        if any(marker in message for marker in _OOM_TEXT_MARKERS):
            return True
    return False


def classify_exception(error: BaseException) -> str:
    return STATUS_OUT_OF_MEMORY if is_out_of_memory_error(error) else STATUS_EXCEPTION


def write_run_status(
    path: str | None,
    status: str,
    *,
    phase: str,
    error: BaseException | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Atomically replace the JSON status file, returning its payload."""
    if path is None:
        return None
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": str(status),
        "phase": str(phase),
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)[:4000]
    if details:
        payload["details"] = dict(details)

    temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "x", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
    return payload


def read_run_status(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise ValueError(f"Invalid ARC run-status payload in {path}")
    return payload


def run_entrypoint_with_status(
    entrypoint: Callable[[], Any],
    *,
    path: str | None,
    phase: str,
    success_status: str,
    success_check: Callable[[], bool] | None = None,
) -> Any:
    """Run one CLI phase and preserve its terminal class before re-raising."""
    write_run_status(path, STATUS_RUNNING, phase=phase)
    try:
        result = entrypoint()
    except BaseException as error:
        write_run_status(
            path,
            classify_exception(error),
            phase=phase,
            error=error,
        )
        raise

    status = success_status
    if success_check is not None and not success_check():
        status = STATUS_SUBMISSION_MISSING
    write_run_status(path, status, phase=phase)
    return result
