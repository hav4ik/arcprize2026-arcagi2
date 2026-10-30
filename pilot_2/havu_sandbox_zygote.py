"""Pre-TPU zygote fork server for isolated Python sandboxes inside hermetic google3 binaries.

On Borg, google3 `py_binary` targets run via an embedded launcher where `sys.executable is None`
and no external `python3` with `numpy` / `scipy` exists, so `subprocess.Popen([sys.executable, ...])`
cannot be used to spawn Havu's `LocalExecutor` child processes. Instead, `ensure_started()` forks a
tiny single-threaded zygote process **before** `jax` / `vllm` / `libtpu` are imported (so it carries
all hermetic runfiles on `sys.path` but zero TPU / gRPC / thread state), and `spawn()` asks that zygote
to fork a fresh process-group-isolated worker whose stdin / response channel is a `socketpair`.

This module is intentionally stdlib-only so the zygote stays lightweight.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import sys
import threading
from typing import Any

_LOCK = threading.Lock()
_SOCK: socket.socket | None = None
_PID: int | None = None
_STARTED_BY_PID: int | None = None


def _worker_main(conn: socket.socket, req: dict[str, Any]) -> None:
  """Grandchild: own session/process group; fd 0 <- socket, responses -> dup of socket, 1/2 -> devnull."""
  os.setsid()
  signal.signal(signal.SIGCHLD, signal.SIG_DFL)
  signal.signal(signal.SIGINT, signal.default_int_handler)
  signal.signal(signal.SIGPIPE, signal.SIG_DFL)
  fd = conn.fileno()
  devnull = os.open(os.devnull, os.O_RDWR)
  os.dup2(fd, 0)
  os.dup2(devnull, 1)
  os.dup2(devnull, 2)
  resp_fd = os.dup(fd)
  if devnull > 2:
    os.close(devnull)
  if fd > 2 and fd != resp_fd:
    os.close(fd)
  cwd = req.get("cwd") or os.environ.get("TEST_TMPDIR") or "/tmp"
  try:
    os.chdir(cwd)
  except OSError:
    pass
  for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
  sys.stdin = os.fdopen(0, "r", encoding="utf-8", errors="replace")
  sys.argv = [
      "-c",
      str(resp_fd),
      str(int(req["max_output"])),
      str(int(req["mem_bytes"])),
      str(int(req["cpu_seconds"])),
      str(int(req["fsize_bytes"])),
      str(int(req["nice"])),
      str(req["style"]),
      "[]",
  ]
  exec(req["child_source"], {"__name__": "__main__"})  # pylint: disable=exec-used


def _server_main(zsock: socket.socket) -> None:
  """Zygote loop: one JSON request line per worker; replies with `(pid, worker_socket_fd)` via SCM_RIGHTS."""
  for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
  try:
    os.setsid()
  except OSError:
    pass
  # Auto-reap exited sandbox workers so no zombies accumulate.
  signal.signal(signal.SIGCHLD, signal.SIG_IGN)
  reader = zsock.makefile("r", encoding="utf-8")
  for line in reader:
    if not line.strip():
      continue
    try:
      req = json.loads(line)
    except json.JSONDecodeError:
      continue
    p_end, w_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:
      try:
        reader.close()
        zsock.close()
        p_end.close()
        _worker_main(w_end, req)
      except BaseException:  # pylint: disable=broad-except
        pass
      finally:
        os._exit(0)  # pylint: disable=protected-access
    w_end.close()
    try:
      socket.send_fds(zsock, [struct.pack("<i", pid)], [p_end.fileno()])
    finally:
      p_end.close()
  os._exit(0)  # pylint: disable=protected-access


def ensure_started() -> None:
  """Forks the zygote once (idempotent per process). Call BEFORE importing jax / vllm / torch."""
  global _SOCK, _PID, _STARTED_BY_PID
  with _LOCK:
    if _SOCK is not None and _STARTED_BY_PID == os.getpid():
      return
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:
      try:
        parent.close()
        _server_main(child)
      except BaseException:  # pylint: disable=broad-except
        pass
      finally:
        os._exit(0)  # pylint: disable=protected-access
    child.close()
    _SOCK, _PID, _STARTED_BY_PID = parent, pid, os.getpid()


def is_started() -> bool:
  return _SOCK is not None and _STARTED_BY_PID == os.getpid()


def spawn(
    child_source: str,
    *,
    max_output: int,
    mem_bytes: int,
    cpu_seconds: int,
    fsize_bytes: int,
    nice: int,
    style: str,
    cwd: str | None = None,
) -> tuple[int, socket.socket]:
  """Returns `(pid, socket)` of a fresh isolated worker running `child_source` (see `havu_b32_harness._CHILD`)."""
  ensure_started()
  req = json.dumps({
      "child_source": child_source,
      "max_output": int(max_output),
      "mem_bytes": int(mem_bytes),
      "cpu_seconds": int(cpu_seconds),
      "fsize_bytes": int(fsize_bytes),
      "nice": int(nice),
      "style": str(style),
      "cwd": cwd or "",
  })
  with _LOCK:
    assert _SOCK is not None
    _SOCK.sendall((req + "\n").encode("utf-8"))
    msg, fds, _, _ = socket.recv_fds(_SOCK, 4, 1)
  if len(msg) < 4 or not fds:
    raise RuntimeError("sandbox zygote did not return a worker pid/socket")
  (pid,) = struct.unpack("<i", msg[:4])
  sock = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM, fileno=fds[0])
  return int(pid), sock
