# Cell 2s (E066, 2026-10-29): restart supervisor for Manh's server (cell 2m). Without it, one engine death ends the
# run's new attempts (B32-M's 46.5 fits a stop around Kaggle hour 4-6). A daemon thread watches MANH_SERVER_PID; when it
# exits, or makes no progress for E066_STALL_S while requests are running, the thread:
#   1. writes the status "restarting" (the solver's S26 then stops launching, and re-queues in-flight attempts);
#   2. stops the server's process group (TERM, then KILL) and waits until it and every GPU compute process are gone, so
#      two engines' host RAM never overlap;
#   3. reruns cell 2m's exact launcher command (_cmd, _env) with a fresh --output-root, dsv41_logs/r<N>;
#   4. waits for /v1/models to list dsv41, then writes the status "up" with the new start's pid, log and startup time.
# The status file uses the format of Manh's --supervise (PR #2), which the pinned solver's S26 follows
# (ARC_EXTERNAL_SERVER_STATUS, cell 3). It gives up, with the status "failed", after E066_MAX_RESTARTS restarts or when
# less than E066_RESTART_MIN_LEFT_S remain before stop-new. Cell 3s stops it before cell 4 stops the server. Uses cell
# 0's T0, WORK, MODE and cell 2m's _cmd, _env, MANH_BASE_URL, MANH_SERVER_PID, MANH_SERVER_LOG, _manh_alive, _t.
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request

SUPERVISOR_STATUS = f"{WORK}/dsv41_logs/e066_supervisor.json"
SUPERVISOR = {"stop": False, "restarts": 0, "events": []}
E066_MAX_RESTARTS = int(os.environ.get("E066_MAX_RESTARTS", "2"))
E066_RESTART_MIN_LEFT_S = float(os.environ.get("E066_RESTART_MIN_LEFT_S", "3600"))
E066_STALL_S = float(os.environ.get("E066_STALL_S", "1500"))
E066_UP_TIMEOUT_S = float(os.environ.get("E066_UP_TIMEOUT_S", "3000"))


def _sup_log(msg):
    line = f"[+{time.time() - T0:.0f}s] E066 SUPERVISOR: {msg}"
    SUPERVISOR["events"].append(line)
    print(line, flush=True)


def _sup_write(**status):
    tmp = SUPERVISOR_STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({**status, "t": round(time.time() - T0)}, f)
    os.replace(tmp, SUPERVISOR_STATUS)


def _sup_group_alive(pgid):
    """Any non-zombie process left in process group pgid."""
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    rest = f.read().rsplit(")", 1)[1].split()
                if int(rest[2]) == pgid and rest[0] not in ("Z", "X"):
                    return True
            except (OSError, ValueError, IndexError):
                pass
    return False


def _sup_gpu_apps():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception as exc:
        return [f"nvidia-smi failed: {type(exc).__name__}"]


def _sup_stop(pgid):
    """Stop the old server's group; True once it and every GPU compute process are gone."""
    for sig, wait_s in ((signal.SIGTERM, 45), (signal.SIGKILL, 180)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except Exception as exc:
            _sup_log(f"killpg({pgid}, {sig.name}): {type(exc).__name__}: {exc}")
        t = time.time()
        while time.time() - t < wait_s:
            if not _sup_group_alive(pgid) and not _sup_gpu_apps():
                return True
            time.sleep(3)
    return False


def _sup_models_ok():
    try:
        with urllib.request.urlopen(f"{MANH_BASE_URL}/models", timeout=5) as u:
            return any(m.get("id") == "dsv41" for m in json.loads(u.read())["data"])
    except Exception:
        return False


def _sup_metrics():
    """(generation tokens total, running + waiting requests) from /metrics, or None if unreadable."""
    try:
        with urllib.request.urlopen(MANH_BASE_URL.removesuffix("/v1") + "/metrics", timeout=10) as u:
            text = u.read().decode("utf-8", "replace")
    except Exception:
        return None
    gen = busy = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            gen += float(line.rsplit(" ", 1)[1])
        elif line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            busy += float(line.rsplit(" ", 1)[1])
    return gen, busy


def _sup_relaunch(n):
    root = f"{WORK}/dsv41_logs/r{n}"
    cmd = list(_cmd)
    cmd[cmd.index("--output-root") + 1] = root
    t = time.time()
    r = subprocess.run(cmd, cwd=WORK, env=_env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       errors="replace", timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"launcher exited {r.returncode}: {(r.stdout + r.stderr)[-1500:]}")
    pid = int(json.load(open(f"{root}/server/launch.json"))["pid"])
    log = f"{root}/server/server.log"
    while time.time() - t < E066_UP_TIMEOUT_S:
        if _sup_models_ok():
            return pid, log, time.time() - t
        if not _manh_alive(pid):
            raise RuntimeError(f"server exited during startup; {log}")
        time.sleep(5)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass
    raise RuntimeError(f"server not up after {time.time() - t:.0f}s; {log}")


def _sup_loop():
    global MANH_SERVER_PID, MANH_SERVER_LOG
    start, prog_t, prog_gen = 0, time.time(), -1.0
    while not SUPERVISOR["stop"]:
        time.sleep(5)
        pid = MANH_SERVER_PID
        why = None
        if not pid or not _manh_alive(pid):
            why = f"server pid {pid} exited"
        elif E066_STALL_S > 0:
            m = _sup_metrics()
            if m is not None and (m[0] > prog_gen or m[1] == 0):  # progress, or idle (nothing to generate)
                prog_t, prog_gen = time.time(), m[0]
            elif time.time() - prog_t > E066_STALL_S:
                why = (f"no generated tokens for {time.time() - prog_t:.0f}s with requests running"
                       if m is not None else f"/metrics unreadable for {time.time() - prog_t:.0f}s")
        if why is None or SUPERVISOR["stop"]:
            continue
        left = T0 + float(os.environ.get("ARC_STOP_NEW_S", 39000)) - time.time()
        if SUPERVISOR["restarts"] >= E066_MAX_RESTARTS or left < E066_RESTART_MIN_LEFT_S:
            _sup_log(f"{why}; no restart ({SUPERVISOR['restarts']}/{E066_MAX_RESTARTS} used, {left / 60:.0f} min to "
                     f"stop-new)")
            _sup_write(start=start, state="failed", error=f"{why}; no restart left")
            return
        start += 1
        SUPERVISOR["restarts"] += 1
        _sup_write(start=start, state="restarting", error=why)
        _sup_log(f"{why}; restart {start}/{E066_MAX_RESTARTS} ({left / 60:.0f} min to stop-new)")
        if not _sup_stop(pid):
            _sup_log(f"old server group {pid} or GPU processes still there: {_sup_gpu_apps()}; giving up")
            _sup_write(start=start, state="failed", error="old server would not stop")
            return
        try:
            new_pid, log, up_s = _sup_relaunch(start)
        except Exception as exc:
            _sup_log(f"relaunch {start} failed: {type(exc).__name__}: {exc}")
            MANH_SERVER_PID = None
            continue  # counts as another death: retried while restarts and time are left
        MANH_SERVER_PID, MANH_SERVER_LOG = new_pid, log
        prog_t, prog_gen = time.time(), -1.0
        _sup_write(start=start, state="up", pid=new_pid, log=log, up_s=round(up_s))
        _sup_log(f"start {start} up after {up_s:.0f}s: pid {new_pid}, log {log}")


if MODE != "fallback" and globals().get("MANH_SERVER_PID"):
    os.makedirs(os.path.dirname(SUPERVISOR_STATUS), exist_ok=True)
    _sup_write(start=0, state="up", pid=MANH_SERVER_PID, log=MANH_SERVER_LOG, up_s=round(time.time() - _t))
    SUPERVISOR["thread"] = threading.Thread(target=_sup_loop, daemon=True)
    SUPERVISOR["thread"].start()
    _sup_log(f"watching pid {MANH_SERVER_PID} (max {E066_MAX_RESTARTS} restarts, stall {E066_STALL_S:.0f}s, "
             f"status {SUPERVISOR_STATUS})")
else:
    SUPERVISOR_STATUS = ""  # cell 3: no status file, the solver keeps S25 (no restarts)
