"""One unassigned adapter pool shared by Qwen vLLM services.

All counters use the same pipeline lock. The pending count includes messages
still in a multiprocessing.Queue feeder: queue.empty()/qsize() cannot establish
completion. Per-service counts cover claimed work, including recovery replays.
Only metadata travels through the queue; artifacts stay in their staging tier.
"""

from __future__ import annotations

from queue import Empty
from typing import Any

from arc_vllm_router import SERVICE_FINISHED


def publish_pending_adapter(queue: Any, count: Any, lock: Any, job: dict) -> int:
    """Publish once, accounting for the handoff before it becomes visible."""
    message = dict(job, service_index=None)
    with lock:
        count.value += 1
        try:
            queue.put(message)
        except BaseException:
            count.value -= 1
            raise
        return int(count.value)


def claim_pending_adapter(
    queue: Any, count: Any, service_index: int, outstanding: Any, lock: Any,
) -> dict | None:
    """Claim one job only when the caller has a free resident adapter slot."""
    with lock:
        if int(count.value) == 0:
            return None
        try:
            job = queue.get_nowait()
        except Empty:
            # A producer has reserved a count but its feeder has not delivered.
            return None
        if job.get("kind") != "adapter":
            raise RuntimeError("The shared adapter pool received a non-adapter job")
        count.value -= 1
        outstanding[service_index] += 1
        return dict(job, service_index=service_index)


def return_pending_adapter(
    queue: Any, count: Any, service_index: int, outstanding: Any,
    ready: Any, lock: Any, job: dict,
) -> bool:
    """Return a retired engine's unfinished job, preserving slot ownership.

    Starting/recovering/future services (ready=0) may receive it later. Normally
    closed or dead services may not. If none remain, the caller must fail the
    pipeline instead of leaving a pool that can never drain.
    """
    message = dict(job, service_index=None, failover_source_service_index=service_index)
    message.pop("_arc_vllm_admission_in_progress", None)
    with lock:
        if not any(i != service_index and int(value) >= 0 for i, value in enumerate(ready)):
            return False
        if int(outstanding[service_index]) < 1:
            raise RuntimeError("Cannot return an unowned adapter to the shared pool")
        outstanding[service_index] -= 1
        count.value += 1
        try:
            queue.put(message)
        except BaseException:
            outstanding[service_index] += 1
            count.value -= 1
            raise
        return True


def pending_pool_snapshot(count: Any, outstanding: Any, lock: Any) -> tuple[int, tuple[int, ...]]:
    with lock:
        return int(count.value), tuple(int(n) for n in outstanding)


def finish_service_if_pool_drained(
    service_index: int, count: Any, outstanding: Any, ready: Any, lock: Any,
) -> bool:
    """Close only after *all* work drains; caller first checks all trainers done."""
    with lock:
        if int(count.value) != 0 or any(int(n) != 0 for n in outstanding):
            return False
        if ready is not None and int(ready[service_index]) >= 0:
            ready[service_index] = SERVICE_FINISHED
        return True
