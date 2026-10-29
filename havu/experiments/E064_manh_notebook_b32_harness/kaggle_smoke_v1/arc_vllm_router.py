"""Small shared-state helpers for routing ARC tasks to vLLM services."""

from __future__ import annotations

from typing import Any


# ``service_ready`` is 0 while a service starts or recovers, 1 while it can
# serve, and negative once permanently closed (failed or normally finished).
SERVICE_DEAD = -1
SERVICE_FINISHED = -2


class NoReadyVllmServiceError(RuntimeError):
    """No inference service has completed engine startup or recovery."""


def _snapshot_unlocked(service_outstanding: Any) -> tuple[int, ...]:
    return tuple(
        int(service_outstanding[index])
        for index in range(len(service_outstanding))
    )


def service_load_snapshot(
    service_outstanding: Any,
    service_load_lock: Any,
) -> tuple[int, ...]:
    """Snapshot owned tasks (claimed/active; also queued in legacy routing)."""
    with service_load_lock:
        return _snapshot_unlocked(service_outstanding)


def reserve_least_loaded_service(
    service_outstanding: Any,
    service_load_lock: Any,
    preferred_service_index: int | None = None,
    service_active: Any | None = None,
    service_ready: Any | None = None,
) -> tuple[int, tuple[int, ...]]:
    """Reserve the least-loaded ready service, optionally preferring one tie.

    When an adaptive pipeline starts with zero inference services, no active
    candidate exists yet. In that one case reservations are balanced across
    all future service queues so exported adapters can wait for the ramp.
    """
    if len(service_outstanding) < 1:
        raise ValueError("At least one vLLM service is required")
    if service_active is not None and len(service_active) != len(
        service_outstanding
    ):
        raise ValueError(
            "service_active and service_outstanding must have equal length"
        )
    if service_ready is not None and len(service_ready) != len(
        service_outstanding
    ):
        raise ValueError(
            "service_ready and service_outstanding must have equal length"
        )
    if preferred_service_index is not None:
        preferred_service_index = int(preferred_service_index)
        if not 0 <= preferred_service_index < len(service_outstanding):
            raise IndexError(
                f"Preferred vLLM service {preferred_service_index} is outside "
                f"0..{len(service_outstanding) - 1}"
            )
    with service_load_lock:
        loads = _snapshot_unlocked(service_outstanding)
        candidate_indices = list(range(len(loads)))
        if service_active is not None:
            active_indices = [
                index
                for index in candidate_indices
                if bool(service_active[index])
            ]
            if active_indices or service_ready is not None:
                candidate_indices = active_indices
        if service_ready is not None:
            candidate_indices = [
                index
                for index in candidate_indices
                if int(service_ready[index]) > 0
            ]
            if not candidate_indices:
                raise NoReadyVllmServiceError(
                    "No active vLLM service has completed engine startup"
                )
        service_index = min(
            candidate_indices,
            key=lambda index: (
                loads[index],
                preferred_service_index is None
                or index != preferred_service_index,
                index,
            ),
        )
        service_outstanding[service_index] += 1
        return service_index, _snapshot_unlocked(service_outstanding)


def mark_service_dead(
    service_ready: Any,
    service_load_lock: Any,
    service_index: int,
) -> None:
    """Permanently exclude a service from routing after an unrecoverable failure."""
    with service_load_lock:
        service_ready[int(service_index)] = SERVICE_DEAD


def finish_service_if_idle(
    service_index: int,
    service_outstanding: Any,
    service_ready: Any,
    service_load_lock: Any,
) -> bool:
    """Close routing atomically with the final outstanding-count check.

    Call only after local work and upstream producers have finished. A peer
    reserves the destination count before enqueueing a handoff, so a locally
    empty queue is not sufficient evidence that this service may exit.
    """
    with service_load_lock:
        if int(service_outstanding[service_index]) != 0:
            return False
        if service_ready is not None and int(service_ready[service_index]) >= 0:
            service_ready[service_index] = SERVICE_FINISHED
        return True


def reassign_task_to_live_service(
    source_service_index: int,
    service_outstanding: Any,
    service_active: Any,
    service_ready: Any,
    service_load_lock: Any,
    *,
    include_unstarted: bool = True,
) -> tuple[int, tuple[int, ...]] | None:
    """Move one outstanding task from a retired service to the least-loaded peer.

    Peers are active, not permanently closed, and either ready, holding committed
    work, or (``include_unstarted``) not yet started.  Once the
    post-training rebalance is done a future service that skipped its cold
    start must not be chosen, so callers pass ``include_unstarted=False``.
    Returns ``None`` when no peer exists and leaves the counters unchanged.
    """
    source_service_index = int(source_service_index)
    with service_load_lock:
        loads = _snapshot_unlocked(service_outstanding)
        if not 0 <= source_service_index < len(loads):
            raise IndexError(
                f"Source vLLM service {source_service_index} is outside "
                f"0..{len(loads) - 1}"
            )
        if loads[source_service_index] < 1:
            raise RuntimeError(
                f"vLLM service {source_service_index} has no outstanding "
                "task to reassign"
            )
        candidates = [
            index
            for index in range(len(loads))
            if index != source_service_index
            and bool(service_active[index])
            and int(service_ready[index]) >= 0
            and (
                int(service_ready[index]) > 0
                or loads[index] > 0
                or include_unstarted
            )
        ]
        if not candidates:
            return None
        target_service_index = min(
            candidates,
            key=lambda index: (loads[index], index),
        )
        service_outstanding[source_service_index] = (
            loads[source_service_index] - 1
        )
        service_outstanding[target_service_index] += 1
        return target_service_index, _snapshot_unlocked(service_outstanding)


def activate_inactive_service(
    service_outstanding: Any,
    service_active: Any,
    service_load_lock: Any,
    *,
    service_limit: int | None = None,
) -> tuple[int | None, tuple[int, ...], tuple[bool, ...]]:
    """Activate a ramp service, excluding slots reserved for final scale-out."""
    if len(service_active) != len(service_outstanding):
        raise ValueError(
            "service_active and service_outstanding must have equal length"
        )
    limit = len(service_active) if service_limit is None else int(service_limit)
    if not 1 <= limit <= len(service_active):
        raise ValueError("service_limit must be within the service topology")
    with service_load_lock:
        inactive_indices = [
            index
            for index in range(limit)
            if not bool(service_active[index])
        ]
        if not inactive_indices:
            return (
                None,
                _snapshot_unlocked(service_outstanding),
                tuple(bool(value) for value in service_active),
            )
        service_index = max(
            inactive_indices,
            key=lambda index: (
                int(service_outstanding[index]),
                -index,
            ),
        )
        service_active[service_index] = 1
        return (
            service_index,
            _snapshot_unlocked(service_outstanding),
            tuple(bool(value) for value in service_active),
        )


def rebalance_pending_service_task(
    source_service_index: int,
    eligible_service_indices: Any,
    service_outstanding: Any,
    service_load_lock: Any,
) -> tuple[int, tuple[int, ...]]:
    """Reassign one queued task to the least-loaded eligible service.

    The source count already includes this task.  Remove it before choosing a
    destination so the comparison reflects the loads that would remain after
    migration.  This helper is intended only for jobs that have not yet been
    admitted into an inference engine; active searches must stay resident.
    """
    source_service_index = int(source_service_index)
    eligible = tuple(dict.fromkeys(map(int, eligible_service_indices)))
    if not 0 <= source_service_index < len(service_outstanding):
        raise IndexError(
            f"Source vLLM service {source_service_index} is outside "
            f"0..{len(service_outstanding) - 1}"
        )
    if source_service_index not in eligible:
        raise ValueError("The source service must remain an eligible destination")
    if not eligible:
        raise ValueError("At least one eligible service is required")
    if any(
        index < 0 or index >= len(service_outstanding)
        for index in eligible
    ):
        raise IndexError("An eligible vLLM service index is outside the topology")

    with service_load_lock:
        source_load = int(service_outstanding[source_service_index])
        if source_load < 1:
            raise RuntimeError(
                f"vLLM service {source_service_index} has no outstanding "
                "queued task to rebalance"
            )
        service_outstanding[source_service_index] = source_load - 1
        target_service_index = min(
            eligible,
            key=lambda index: (int(service_outstanding[index]), index),
        )
        service_outstanding[target_service_index] += 1
        return target_service_index, _snapshot_unlocked(service_outstanding)


def transfer_service_task_assignment(
    source_service_index: int,
    target_service_index: int,
    service_outstanding: Any,
    service_load_lock: Any,
) -> tuple[int, ...]:
    """Move one outstanding-task count between explicit services.

    This is primarily the rollback path if enqueueing a rebalanced adapter
    fails after :func:`rebalance_pending_service_task` reserved its target.
    """
    source_service_index = int(source_service_index)
    target_service_index = int(target_service_index)
    with service_load_lock:
        for label, index in (
            ("Source", source_service_index),
            ("Target", target_service_index),
        ):
            if not 0 <= index < len(service_outstanding):
                raise IndexError(
                    f"{label} vLLM service {index} is outside "
                    f"0..{len(service_outstanding) - 1}"
                )
        source_load = int(service_outstanding[source_service_index])
        if source_load < 1:
            raise RuntimeError(
                f"vLLM service {source_service_index} has no outstanding "
                "task to transfer"
            )
        service_outstanding[source_service_index] = source_load - 1
        service_outstanding[target_service_index] += 1
        return _snapshot_unlocked(service_outstanding)


def record_adapter_publication_and_maybe_activate(
    published_adapter_count: Any,
    ramp_after_adapters: int,
    service_outstanding: Any,
    service_active: Any,
    service_load_lock: Any,
    ramp_ready_event: Any,
    *,
    service_limit: int | None = None,
) -> tuple[int, int | None, tuple[int, ...], tuple[bool, ...]]:
    """Count one durable handoff and let its trainer claim a ramp service."""
    threshold = int(ramp_after_adapters)
    if threshold < 1:
        raise ValueError("ramp_after_adapters must be positive")
    if len(service_active) != len(service_outstanding):
        raise ValueError(
            "service_active and service_outstanding must have equal length"
        )
    limit = len(service_active) if service_limit is None else int(service_limit)
    if not 1 <= limit <= len(service_active):
        raise ValueError("service_limit must be within the service topology")
    with service_load_lock:
        published_adapter_count.value = int(published_adapter_count.value) + 1
        count = int(published_adapter_count.value)
        service_index = None
        if count >= threshold:
            inactive_indices = [
                index
                for index in range(limit)
                if not bool(service_active[index])
            ]
            if inactive_indices:
                service_index = max(
                    inactive_indices,
                    key=lambda index: (
                        int(service_outstanding[index]),
                        -index,
                    ),
                )
                service_active[service_index] = 1
        loads = _snapshot_unlocked(service_outstanding)
        active = tuple(bool(value) for value in service_active)
    if count >= threshold:
        ramp_ready_event.set()
    return count, service_index, loads, active


def release_service_task(
    service_index: int,
    service_outstanding: Any,
    service_load_lock: Any,
) -> tuple[int, ...]:
    """Release one completed/discarded task from its assigned service."""
    service_index = int(service_index)
    with service_load_lock:
        if not 0 <= service_index < len(service_outstanding):
            raise IndexError(
                f"vLLM service index {service_index} is outside "
                f"0..{len(service_outstanding) - 1}"
            )
        current = int(service_outstanding[service_index])
        if current < 1:
            raise RuntimeError(
                f"vLLM service {service_index} has no outstanding task to release"
            )
        service_outstanding[service_index] = current - 1
        return _snapshot_unlocked(service_outstanding)
