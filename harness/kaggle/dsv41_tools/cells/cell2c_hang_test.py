# Cell 2c (smoke, commit runs only): a deliberate engine hang, to exercise the failure handling on Kaggle. A background
# thread waits for "server up", then freezes this notebook's own VLLM::EngineCore process (SIGSTOP; found among this
# kernel's descendants, never by name alone). The API server keeps answering while nothing is generated, which is the
# hang vLLM itself does not detect. The solver's progress watchdog (S20) must mark the server dead and restart it; the
# re-launch hold (S21) and the post-restart ramp (S22) then run. The smoke shortens the watchdog to ARC_SMOKE_STALL_S and
# lets a restart happen late in the short run (RESTART_MIN_LEFT_S). If no restart happened after ARC_SMOKE_HANG_MAX_S,
# the engine is resumed (SIGCONT) so the smoke still ends, and the report cell fails the commit run. The competition
# rerun skips this cell. Uses cell 0's T0, WORK, MODE and RERUN and cell 2's BASE_ENV.
import json
import os
import signal
import threading
import time

HANG_OUT = f"{WORK}/smoke_hang.json"


def hang_save(state):
    with open(HANG_OUT + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(HANG_OUT + ".tmp", HANG_OUT)


def hang_descendants(pid):
    kids = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def hang_cmd(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def hang_log_has(path, text):
    try:
        with open(path, errors="replace") as f:
            return text in f.read()
    except OSError:
        return False


def hang_worker(state, after_s, max_s):
    log = f"{WORK}/dsv41_logs/solver.log"
    try:
        while time.time() - T0 < 3600:
            if hang_log_has(log, "server up"):
                break
            time.sleep(5)
        time.sleep(after_s)
        cores = [p for p in hang_descendants(os.getpid()) if hang_cmd(p).startswith("VLLM::EngineCore")]
        if not cores:
            state["error"] = "no VLLM::EngineCore among this kernel's descendants"
            return
        pid = cores[0]
        os.kill(pid, signal.SIGSTOP)
        state.update(engine_pid=pid, frozen_min=round((time.time() - T0) / 60, 2))
        hang_save(state)
        t0 = time.time()
        while time.time() - t0 < max_s:
            if hang_log_has(log, "server start (restart"):
                state["restart_seen_after_s"] = round(time.time() - t0)
                break
            time.sleep(5)
        else:
            os.kill(pid, signal.SIGCONT)  # the watchdog did not act: resume so the smoke can still end
            state["resumed_without_restart"] = True
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    hang_save(state)


if not RERUN and MODE != "fallback":
    BASE_ENV["ARC_STALL_RESTART_S"] = os.environ.get("ARC_SMOKE_STALL_S", "60")
    BASE_ENV["ARC_RESTART_MIN_LEFT_S"] = os.environ.get("ARC_SMOKE_RESTART_MIN_LEFT_S", "300")
    _hang_state = {"after_s": int(os.environ.get("ARC_SMOKE_HANG_AFTER_S", "20"))}
    hang_save(_hang_state)
    threading.Thread(target=hang_worker, args=(_hang_state, _hang_state["after_s"],
                                               int(os.environ.get("ARC_SMOKE_HANG_MAX_S", "420"))), daemon=True).start()
    print(f"hang test armed: freeze VLLM::EngineCore {_hang_state['after_s']} s after the server is up; watchdog "
          f"{BASE_ENV['ARC_STALL_RESTART_S']} s", flush=True)
