"""Standard-library control-plane IPC for independently exec'd Python workers.

No Torch objects cross this boundary. The parent retains all proxies until
workers exit; JSON launch files contain authenticated manager references, not
pickled application objects. Both interpreters must use the same Python minor.
"""

from __future__ import annotations

import math
import multiprocessing
from multiprocessing.managers import (
    AcquirerProxy,
    BaseProxy,
    EventProxy,
    ListProxy,
    SyncManager,
    Token,
    ValueProxy,
)
import queue
import secrets
import tempfile
import threading


def validate_metadata(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in (list, tuple):
        for item in value:
            validate_metadata(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            validate_metadata(item)
        return
    raise TypeError(
        f"Pipeline IPC only accepts plain metadata, not {type(value).__name__}"
    )


class MetadataQueueProxy(BaseProxy):
    _exposed_ = ("put", "get", "empty", "qsize", "full")

    def put(self, item, block=True, timeout=None):
        validate_metadata(item)  # Check before serialization, not on the server.
        return self._callmethod("put", (item, block, timeout))

    def get(self, block=True, timeout=None):
        return self._callmethod("get", (block, timeout))

    def put_nowait(self, item):
        return self.put(item, False)

    def get_nowait(self):
        return self.get(False)

    def empty(self):
        return self._callmethod("empty")

    def qsize(self):
        return self._callmethod("qsize")

    def full(self):
        return self._callmethod("full")

    # There is no client-side feeder thread: put() is acknowledged by the
    # server. The parent closes the broker after all workers have exited.
    def close(self):
        pass

    def join_thread(self):
        pass

    def cancel_join_thread(self):
        pass


class ProcessAcquirerProxy(AcquirerProxy):
    """Match multiprocessing's block= API, not threading's blocking= API."""

    def acquire(self, block=True, timeout=None):
        # multiprocessing ignores timeout for nonblocking acquisitions and
        # treats a negative timeout as zero; threading uses different rules.
        if not block:
            timeout = None
        elif timeout is not None:
            timeout = max(0.0, timeout)
        return super().acquire(blocking=block, timeout=timeout)


class PipelineManager(SyncManager):
    pass


PipelineManager.register("MetadataQueue", queue.Queue, MetadataQueueProxy)
PipelineManager.register("Lock", threading.Lock, ProcessAcquirerProxy)
PipelineManager.register("RLock", threading.RLock, ProcessAcquirerProxy)
PipelineManager.register(
    "BoundedSemaphore", threading.BoundedSemaphore, ProcessAcquirerProxy
)


class Counter:
    """Retain Value.value/get_lock semantics with an explicit remote lock."""

    def __init__(self, value, lock):
        self.remote_value = value
        self.lock = lock

    @property
    def value(self):
        return self.remote_value.value

    @value.setter
    def value(self, value):
        self.remote_value.value = value

    def get_lock(self):
        return self.lock


PROXY_TYPES = {
    "MetadataQueue": MetadataQueueProxy,
    "list": ListProxy,
    "Value": ValueProxy,
    "Lock": ProcessAcquirerProxy,
    "RLock": ProcessAcquirerProxy,
    "BoundedSemaphore": ProcessAcquirerProxy,
    "Event": EventProxy,
}


def encode(value):
    """Encode only known IPC handles plus JSON-compatible launch metadata."""
    if isinstance(value, Counter):
        return {"__arc_counter__": [encode(value.remote_value), encode(value.lock)]}
    if isinstance(value, BaseProxy):
        token = value._token
        if token.typeid not in PROXY_TYPES:
            raise TypeError(f"Unsupported pipeline proxy {token.typeid}")
        return {"__arc_proxy__": token.typeid, "id": token.id}
    if isinstance(value, tuple):
        return {"__arc_tuple__": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key.startswith("__arc_") for key in value):
            raise TypeError("Pipeline launch dictionary keys must be plain strings")
        return {key: encode(item) for key, item in value.items()}
    validate_metadata(value)
    return value


def decode(value, *, address, authkey):
    if isinstance(value, list):
        return [decode(item, address=address, authkey=authkey) for item in value]
    if isinstance(value, dict):
        if "__arc_counter__" in value:
            return Counter(
                *decode(value["__arc_counter__"], address=address, authkey=authkey)
            )
        if "__arc_tuple__" in value:
            return tuple(
                decode(value["__arc_tuple__"], address=address, authkey=authkey)
            )
        if "__arc_proxy__" in value:
            kind = value["__arc_proxy__"]
            proxy = PROXY_TYPES[kind]
            return proxy(Token(kind, address, value["id"]), "pickle", authkey=authkey)
        return {
            key: decode(item, address=address, authkey=authkey)
            for key, item in value.items()
        }
    validate_metadata(value)
    return value


class PipelineBroker:
    """Private local manager using a spawn process, never a CUDA fork.

    The manager uses the coordinator interpreter and only handles metadata.
    It is not a trainer or engine; GPU workers are separately exec'd.
    """

    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="arc-ipc-")
        self.address = self.directory.name + "/socket"
        self.authkey = secrets.token_bytes(32)
        self.manager = PipelineManager(
            address=self.address,
            authkey=self.authkey,
            ctx=multiprocessing.get_context("spawn"),
            shutdown_timeout=5,
        )
        try:
            self.manager.start()
        except BaseException:
            self.directory.cleanup()
            raise
        self.references = []

    def is_alive(self):
        return self.manager._process.is_alive()

    def _keep(self, value):
        self.references.append(value)
        return value

    def Queue(self):
        return self._keep(self.manager.MetadataQueue())

    def Lock(self):
        return self._keep(self.manager.Lock())

    def Event(self):
        return self._keep(self.manager.Event())

    def BoundedSemaphore(self, value):
        return self._keep(self.manager.BoundedSemaphore(value))

    def Array(self, typecode, values, *, lock=False):
        if lock:
            raise ValueError("Pipeline arrays require explicit service_load_lock")
        return self._keep(self.manager.list(values))

    def Value(self, typecode, value, *, lock=True):
        return self._keep(
            Counter(self.manager.Value(typecode, value), self.manager.RLock())
        )

    def close(self):
        if self.directory is None:
            return
        # Callers must have reaped clients first. Release client references
        # while the server can still process their final decref messages.
        self.references.clear()
        self.manager.shutdown()
        self.directory.cleanup()
        self.directory = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
