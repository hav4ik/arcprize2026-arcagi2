#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 tool solver for the ARC Prize 2026 Kaggle rerun on 4xL4 (E042 RW, E043 tool E128 / E112).

One process does everything after the vLLM install (the notebook installs the fork and its overlay first):

1. Writes a valid fallback submission.json at once (every test pair: the test input, twice), then rewrites it
   atomically after every finished attempt, so the file is valid whenever the notebook dies.
2. Starts the vLLM server (the E042 RW flags: TP4, fp8 KV, util 0.93, 160K, prefix caching, retention 128) in its
   own session and waits for it. A failed start stops the server's process group before any retry; if it never
   comes up, the fallback submission stays and the solver exits 0. A failed checkpoint pre-flight is retried with
   backoff for ARC_PREFLIGHT_RETRY_S, then the server starts anyway.
3. Runs E043's tool harness (`run_deepseek_tools.solve`, the sub80s40 + minimal-header config, --token-exact,
   --rescue-code, --train-check) on every test pair, attempt 0 for all pairs first (smallest tasks first), then
   attempt 1, with ARC_WORKERS attempts at once. Tool code runs in `arc_harness.executor.LocalExecutor` (one child
   Python per attempt, IPython-style output like the Docker sandbox the configs were measured with).
4. Deadlines, measured from ARC_T0 (the notebook's first cell):
   - ARC_STOP_NEW_S: no new attempts after this;
   - ARC_FORCE_AT_S: every open generation stream is cut (its socket shut down, so a stalled stream ends too) and
     reported as "output budget spent", so each running attempt ends through the harness's own fallback (the
     session's `answer`), not through an error;
   - ARC_HARD_STOP_S: final write, server stopped, exit, whatever is still running.
5. Loop guard (Nguyen's gzip monitor): while a turn is still thinking, the last ARC_LOOP_WINDOW tokens are
   gzip-compressed every ARC_LOOP_EVERY tokens. ARC_LOOP_CONFIRM consecutive windows at ratio >= ARC_LOOP_RATIO end the attempt
   like a budget cut (the session's `answer` is still used).
6. Watchdog: if the server process dies, running attempts error out; the harness still reads each session's
   `answer`, which counts until the attempt's re-run brings a grid. The server is restarted (at most
   ARC_MAX_SERVER_RESTARTS times, while the server can be up before stop-new) and errored attempts are queued again
   (each job at most ARC_JOB_TRIES times). An exception in one main-loop iteration is logged and the loop goes on.
7. SIGTERM / SIGINT: write the submission, stop the server, exit.
7b. Wrap-up: within ARC_WRAPUP_S of the force deadline, tool outputs carry a "N min left: submit now" note.
8. Second attempts: once every pair's attempt 0 has started, attempt-1 jobs go to pairs without a train-passing
   answer first and to train-passed pairs last (`pick`).
9. Admission (Nguyen's min-request-tokens): a new attempt starts only if (force deadline - now) x the median
   per-stream rate of finished attempts is at least ARC_MIN_ATTEMPT_TOKENS; late starts rarely finish.
10. Before writing, the submission is checked against the challenge file (every task, one row per test input,
    attempt_1 and attempt_2, grids 1-30 x 1-30 of ints 0-9); a failing build is not written.

Answers per test pair: `train_check.ranked_answers` (train-passing attempts first, then attempt order; two
distinct grids), padded with the fallback. Logs: ARC_OUT/dsv41_logs (solver.log, attempts.jsonl, status.json,
server.log, traces/). Local scoring when ARC_SOLUTIONS is set.
"""
import gzip
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

E = os.environ.get
ENV_WARNINGS = []  # S12: bad environment values (defaults used instead); logged loudly by main()


def env_num(name, default, cast=float):
    """S12: a numeric setting from the environment. An empty or invalid value (e.g. ARC_KV_ADMIT_MAX="", which raised
    at import in V6) gives the default and a warning instead of an exception before main's handlers."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        if raw is not None:
            ENV_WARNINGS.append(f"{name} is empty: using {default!r}")
        return cast(default)
    try:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("not finite")
        return int(value) if cast is int else value
    except ValueError:
        ENV_WARNINGS.append(f"{name}={raw[:40]!r} is not a number: using {default!r}")
        return cast(default)


def env_json(name, default):
    """S12: a JSON setting from the environment (same type as `default`); invalid JSON gives the default."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        value = json.loads(raw)
        if isinstance(value, type(default)):
            return value
    except ValueError:
        pass
    ENV_WARNINGS.append(f"{name}={raw[:80]!r} is not a JSON {type(default).__name__}: using {default!r}")
    return default


def env_arg(name, cast, low, high=math.inf):
    """S12: an optional harness flag value (ARC_MULTI_BUDGET, ARC_CONTEXT_LIMIT, ARC_BEST_GUESS_*): "" when unset,
    empty or invalid, i.e. not a finite number in [low, high) (argparse's SystemExit escaped main's `except Exception`
    in V6), else the normalized number as a string."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ""
    try:
        value = float(raw)
        if math.isfinite(value) and low <= value < high:
            return str(int(value) if cast is int else value)
    except ValueError:
        pass
    ENV_WARNINGS.append(f"{name}={raw[:40]!r} is invalid: the option stays off")
    return ""


T0 = env_num("ARC_T0", time.time())
KIT = Path(E("ARC_KIT", Path(__file__).resolve().parent))
OUT = Path(E("ARC_OUT", "/kaggle/working"))
LOG = OUT / "dsv41_logs"
SUB = OUT / "submission.json"
CHALLENGES = E("ARC_CHALLENGES", "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
SOLUTIONS = E("ARC_SOLUTIONS", "")
VLLM = E("ARC_VLLM", "/tmp/dsv41/venv/bin/vllm")
TOOL_PYTHON = E("ARC_TOOL_PYTHON", sys.executable)
PORT = env_num("ARC_PORT", 8100, int)
WORKERS = env_num("ARC_WORKERS", 48, int)  # E128 x 56 on pod B: KV 0.64 after 57 min and rising; 48 keeps headroom
MAX_TURNS = env_num("ARC_MAX_TURNS", 40, int)
STOP_NEW_S = env_num("ARC_STOP_NEW_S", 10 * 3600 + 50 * 60)
FORCE_AT_S = env_num("ARC_FORCE_AT_S", 11 * 3600 + 10 * 60)
HARD_STOP_S = env_num("ARC_HARD_STOP_S", 11 * 3600 + 30 * 60)
UP_TIMEOUT_S = env_num("ARC_SERVER_UP_TIMEOUT_S", 7200)  # /kaggle/input is FUSE: 131 GB at ~50 MB/s is ~45 min
# S6: up to 3 server starts after the first (startup retries and restarts after a crash share the count), each only
# while restart_allowed(): RESTART_MIN_LEFT_S before the hard stop and the server expected up before stop-new
MAX_RESTARTS = env_num("ARC_MAX_SERVER_RESTARTS", 3, int)
RESTART_MIN_LEFT_S = env_num("ARC_RESTART_MIN_LEFT_S", 2700)
RESTART_UP_S = env_num("ARC_RESTART_UP_S", 600)  # S6: expected server start time, at least (the first start's if longer)
JOB_TRIES = env_num("ARC_JOB_TRIES", 2, int)
CONN_TRIES = env_num("ARC_CONN_TRIES", 4, int)  # re-queues for attempts lost to a server outage (not in JOB_TRIES)
CONN_ERRORS = ("Connection", "ConnectError", "RemoteProtocolError", "ReadError", "Server disconnected",
               "Incomplete stream", "Connection refused", "ReadTimeout",
               # E055: a dying engine answers in-flight requests with these before the port closes; without them every
               # server death used up one ARC_JOB_TRIES of each in-flight job (43 jobs lost after two deaths)
               "EngineCore encountered an issue", "EngineDeadError")
TASK_LIMIT = env_num("ARC_TASK_LIMIT", 0, int)
TASK_IDS = [t for t in E("ARC_TASK_IDS", "").split(",") if t]  # smoke tests: these tasks instead of the first N
# E046 (opt-in): one attempt per task for tasks with several test inputs (all test inputs in one rollout, the answer
# is a list of grids). On Flash-1017 it ties per-pair attempts (pass@2 82.6% vs 85.9%) at -41% tokens per task.
MULTI_TEST = E("ARC_MULTI_TEST", "0") == "1"
CONTEXT_LIMIT = env_arg("ARC_CONTEXT_LIMIT", int, 1)  # E046 --context-limit: the server's max-model-len (163840)
MULTI_BUDGET = env_arg("ARC_MULTI_BUDGET", float, 0)  # E046 --multi-budget f: (1 + f (n - 1)) x budget and turns
# I5 (opt-in): the harness's --auto-repair (retry a run_python cell once with an identifier typo fixed). Needs
# evals/auto_repair.py + auto_repair_names.json in the kit's repo/evals (not in the V5 kit): without them it is off.
AUTO_REPAIR = E("ARC_AUTO_REPAIR", "0") == "1"
# I7 (opt-in, untested on GPU): the harness's --best-guess-at F / --best-guess-turns K, one "stop exploring, submit your
# best answer" note per attempt at F x the output budget or K turns before the turn cap (recommended 0.8 / 3; E128
# runs 32 of 91 never-train-passing attempts to a cap without an answer). Invalid values leave the option off.
BEST_GUESS_AT = env_arg("ARC_BEST_GUESS_AT", float, 0, 1)
BEST_GUESS_TURNS = env_arg("ARC_BEST_GUESS_TURNS", int, 0, MAX_TURNS)
# I8 (opt-in, untested on GPU): the harness's --post-pass-note, one "[All training pairs pass] check the test input's
# uncovered assumptions with code, then submit_answer" note per attempt on the first tool result that shows a program
# reproducing every training pair (E128 at effort 50 spends 14% of its output after that point). Only "1" turns it on.
POST_PASS_NOTE = E("ARC_POST_PASS_NOTE", "0") == "1"
ATTEMPTS = env_num("ARC_ATTEMPTS", 2, int)
LOOP_WINDOW = env_num("ARC_LOOP_WINDOW", 1000, int)
LOOP_EVERY = env_num("ARC_LOOP_EVERY", 1000, int)
LOOP_RATIO = env_num("ARC_LOOP_RATIO", 8)
# Nguyen uses 3 confirmations; on E043's traces that would cut normal terse reasoning (E112 streaks of 7-8 windows
# in correct-looking notes; E128 has none in 5,699 turns, and its only 3+ streak is grid literals in tool code).
# Thinking only, 10 windows: catches true doom loops (E112 "thermometer thermometer ...", 18 windows).
LOOP_CONFIRM = env_num("ARC_LOOP_CONFIRM", 10, int)
EXEC_MEM_MB = env_num("ARC_EXEC_MEM_MB", 4096, int)
SALT = E("ARC_CACHE_SALT", "kaggle")
WRAPUP_S = env_num("ARC_WRAPUP_S", 900)  # wall-clock note in tool outputs this long before the force deadline
MIN_ATTEMPT_TOKENS = env_num("ARC_MIN_ATTEMPT_TOKENS", 15000)  # admission: time left x per-stream rate
# E042 KV-gated admission (opt-in, 0 = off): above ARC_KV_ADMIT_MIN_WORKERS running, start an attempt only while
# vLLM's KV usage is below ARC_KV_ADMIT_MAX. Pod B, E128 x 56: at KV ~0.89 the cached prefixes of streams between
# turns are evicted and every turn re-prefills (381 -> 196 tok/s, uncached 130 -> 438 tok/s).
KV_ADMIT_MAX = env_num("ARC_KV_ADMIT_MAX", 0)
KV_ADMIT_MIN_WORKERS = env_num("ARC_KV_ADMIT_MIN_WORKERS", 16, int)
# S1: when /metrics gives no KV usage for more than ARC_KV_MISSING_POLLS readings in a row (unreachable, renamed,
# garbled, NaN), the gate fails safe: no new attempt above ARC_KV_MISSING_WORKERS running (V6 failed open)
KV_MISSING_POLLS = env_num("ARC_KV_MISSING_POLLS", 2, int)
KV_MISSING_WORKERS = env_num("ARC_KV_MISSING_WORKERS", 32, int)
# S17 (E055 candB_x2): preempting requests frees KV, so the usage the gate reads dips (0.99 -> 0.6) while the server
# still queues requests, and the gate admitted more attempts into an oversubscribed pool (832 preemptions, server
# queue 11-22). With ARC_ADMIT_MAX_WAITING=N > 0 the gate also holds while the server has >= N waiting requests.
ADMIT_MAX_WAITING = env_num("ARC_ADMIT_MAX_WAITING", 0, int)
# S20 (2026-10-27 debugging of B32): the watchdog only noticed a server whose process EXITED. A server that hangs (process
# alive, e.g. a stuck collective; or an API server that stops answering) would stall the run until the deadlines. With
# ARC_STALL_RESTART_S=N > 0, no generated tokens (server counter) for N s while attempts run counts as a dead server:
# its process group is stopped and the normal restart path runs (in-flight attempts are re-queued as server-side errors).
STALL_RESTART_S = env_num("ARC_STALL_RESTART_S", 0)
# S21 (E059 B16 run): jobs re-queued after a server-side error were launched again within the same second against the
# dying server, each spending one of its CONN_TRIES. ARC_CONN_HOLD=1: after a server-side error, no new launch until the
# server answers /v1/models again (checked at most every 5 s).
CONN_HOLD = env_num("ARC_CONN_HOLD", 0, int)
# S22 (E059 B16 run): after a restart the gate admitted every worker at once into the empty KV pool, mostly long re-queued
# attempts that then grew together (KV > 0.95 half of the next hour, 464 preemptions). ARC_RESTART_RAMP="start,per_min":
# after a restart at most start + per_min x minutes attempts run until WORKERS is reached.
_ramp = [int(x) for x in os.environ.get("ARC_RESTART_RAMP", "").split(",") if x.strip()]
RESTART_RAMP = tuple(_ramp) if len(_ramp) == 2 else None
STAGE_DONE = E("ARC_STAGE_DONE", "")  # marker file written by the notebook's model-staging copy
STAGE_WAIT_S = env_num("ARC_STAGE_WAIT_S", 2400)
PREFLIGHT_RETRY_S = env_num("ARC_PREFLIGHT_RETRY_S", 300)  # S7: re-run a failed checkpoint pre-flight this long
PREFLIGHT_WAIT_S = 15.0  # S7: first wait between pre-flight runs (doubles, at most 120 s)
MAX_MODEL_LEN = str(env_num("ARC_MAX_MODEL_LEN", 163840, int))
MAX_NUM_SEQS = str(env_num("ARC_MAX_NUM_SEQS", 128, int))
GPU_UTIL = str(env_num("ARC_GPU_UTIL", 0.93))
SERVER_ENV_JSON = env_json("ARC_SERVER_ENV_JSON", {})
SERVER_EXTRA_ARGS = env_json("ARC_SERVER_EXTRA_ARGS", [])
BIG = 10 ** 9  # completion tokens reported for a cut stream: the harness then stops on its output budget

URL = f"http://127.0.0.1:{PORT}"
HARNESS = [
    "--mode", "grid", "--effort", "low", "--prompt", "lean2sub", "--answer-tool", "--final-mode", "session",
    "--lenient-parse", "--max-tokens", "64000", "--max-total-output", "80000", "--budget-soft", "40000",
    "--budget-notes", "0.01", "--budget-warn", "0.75", "--effort-header", "minimal", "--turns-note", "5",
    "--train-check", "--rescue-code", "--token-exact", "--model", "dsv41",
    "--fallback-solve",  # only when `answer` is not a valid grid at the end (e.g. after a cut): can only add answers
]
# E050: V4.1's numeric reasoning effort in [1, 100]; DeepSeek's encoder maps "low" to 50 (the fork maps it to 25).
REASONING_EFFORT = int(env_arg("ARC_REASONING_EFFORT", int, 1, 101) or 50)
EXTRA_BODY = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": REASONING_EFFORT}, "temperature": 1.0,
              "top_p": 1.0, "cache_salt": SALT}

LOCK = threading.RLock()
STATE = {"forced": False, "server": None, "restarts": 0, "server_ok": False, "phase": "start"}
RECORDS = {}  # (task, pair) -> list of finished attempt records
RUNS = {}  # (task, pair, attempt) -> runs started (T3: trace file suffix)
STREAMS = set()  # open cut-able streams


def elapsed():
    return time.time() - T0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())} +{elapsed() / 3600:.2f}h] {msg}"
    print(line, flush=True)
    try:
        with open(LOG / "solver.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- submission

def valid_grid(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_submission(tasks):
    from train_check import ranked_answers
    sub = {}
    for t, task in tasks.items():
        rows = []
        for p, test in enumerate(task["test"]):
            fallback = test["input"] if valid_grid(test["input"]) else [[0]]
            with LOCK:
                recs = [r for r in RECORDS.get((t, p), []) if valid_grid(r.get("pred"))]
            preds = [r["pred"] for r in ranked_answers(recs)] if recs else []
            a1 = preds[0] if preds else fallback
            a2 = preds[1] if len(preds) > 1 else fallback
            rows.append({"attempt_1": a1, "attempt_2": a2})
        sub[t] = rows
    return sub


def check_submission(sub, tasks):
    assert set(sub) == set(tasks), "task ids differ from the challenge file"
    for t, task in tasks.items():
        assert isinstance(sub[t], list) and len(sub[t]) == len(task["test"]), f"{t}: wrong number of rows"
        for row in sub[t]:
            assert set(row) == {"attempt_1", "attempt_2"}, f"{t}: wrong keys"
            assert valid_grid(row["attempt_1"]) and valid_grid(row["attempt_2"]), f"{t}: invalid grid"


def write_submission(tasks, why=""):
    try:
        with LOCK:
            sub = build_submission(tasks)
            check_submission(sub, tasks)
            atomic_json(SUB, sub)
        if why:
            log(f"submission written ({why})")
    except Exception as exc:  # never let a write failure kill the run; the previous file stays valid
        log(f"submission write failed: {type(exc).__name__}: {exc}")


def write_fallback(tasks):
    sub = {t: [{"attempt_1": (x["input"] if valid_grid(x["input"]) else [[0]]),
                "attempt_2": (x["input"] if valid_grid(x["input"]) else [[0]])} for x in task["test"]]
           for t, task in tasks.items()}
    check_submission(sub, tasks)
    atomic_json(SUB, sub)


# ---------------------------------------------------------------- server

def server_env():
    env = dict(os.environ)
    model = STATE["model"]
    env.update({
        "VLLM_USE_V2_MODEL_RUNNER": "1", "FLASHINFER_CUDA_ARCH_LIST": "8.9", "TORCH_CUDA_ARCH_LIST": "8.9",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DSV41_INDEXER_Q_PREDECODE": "1",
        "DSV41_EXPERT_HUMMING": E("ARC_HUMMING", '{"quant_method": "humming", "weight_dtype": "uint2", '
                                  '"weight_scale_dtype": "bfloat16", "group_size": 128, "has_zero_point": true}'),
        "DSV41_EXPERT_HUMMING_W2": E("ARC_HUMMING_W2", '{"group_size": 64}'),
        "E043_GPTQ_DIR": str(Path(model) / "gptq"), "VLLM_NO_USAGE_STATS": "1", "TOKENIZERS_PARALLELISM": "false",
        "MAX_JOBS": E("MAX_JOBS", "8"),  # JIT compiles at the first forward, after ~90 GiB is pinned
    })
    for k in ("NCCL_P2P_LEVEL", "VLLM_MARLIN_INPUT_DTYPE"):  # never on Kaggle (E027: P2P there is ~12x slower)
        env.pop(k, None)
    env.update({str(k): str(v) for k, v in SERVER_ENV_JSON.items()})
    return env


def server_cmd():
    return [VLLM, "serve", STATE["model"], "--served-model-name", "dsv41", "--host", "127.0.0.1", "--port", str(PORT),
            "--trust-remote-code", "--tensor-parallel-size", "4", "--distributed-executor-backend", "mp",
            "--kv-cache-dtype", "fp8", "--block-size", "128", "--max-model-len", MAX_MODEL_LEN,
            "--max-num-seqs", MAX_NUM_SEQS, "--max-num-batched-tokens", "2048",
            "--gpu-memory-utilization", GPU_UTIL, "--enable-prefix-caching",
            "--prefix-cache-retention-interval", "128", "--engram-config", '{"cpu_offload":true}',
            "--tokenizer-mode", "deepseek_v41", "--language-model-only", "--reasoning-parser", "deepseek_v41",
            "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice", "--enable-prompt-tokens-details",
            "--compilation-config", '{"cudagraph_mode":"FULL"}'] + [str(x) for x in SERVER_EXTRA_ARGS]


def server_alive():
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server():
    """Start vLLM in its own session; True once /v1/models answers, False if it died or timed out."""
    n = STATE["restarts"]
    if server_alive():
        log(f"warning: something already answers on port {PORT}")
    path = LOG / (f"server.log" if n == 0 else f"server_restart{n}.log")
    log(f"server start ({'first' if n == 0 else f'restart {n}'}): model {STATE['model']}")
    with open(LOG / "server_cmd.json", "w") as f:  # credentials a box image sets (e.g. HF_TOKEN) are never logged
        json.dump({"cmd": server_cmd(), "env": {k: "<redacted>" if any(s in k for s in ("TOKEN", "KEY", "SECRET", "PASS"))
                                                else v for k, v in server_env().items()
                                                if k.startswith(("VLLM", "DSV41", "E043", "FLASHINFER", "TORCH_CUDA",
                                                                 "CUDA", "NCCL", "HF_", "PYTORCH"))}}, f, indent=1)
    with open(path, "ab") as fh:  # the server keeps its own copy of the descriptor
        proc = subprocess.Popen(server_cmd(), stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=server_env(), start_new_session=True, cwd=E("ARC_SCRATCH", "/tmp"))
    STATE["server"], STATE["server_log"] = proc, path
    t = time.time()
    while time.time() - t < UP_TIMEOUT_S and elapsed() < STOP_NEW_S:
        if proc.poll() is not None:
            log(f"server exited during startup (code {proc.returncode}); tail: {tail(path)}")
            # S5: its TP workers / engine core can outlive the leader and hold GPU memory, so a retry would fail too
            stop_server()
            return False
        if server_alive():
            log(f"server up after {time.time() - t:.0f}s; {grep_kv(path)}")
            STATE["server_ok"] = True
            STATE.setdefault("up_s", time.time() - t)  # the first start's duration (restart_allowed)
            return True
        time.sleep(5)
    log(f"server not up after {time.time() - t:.0f}s; stopping it")
    stop_server()
    return False


def restart_allowed(now=None):
    """S6: one more server start (a startup retry or a restart after a crash) while restarts are left, more than
    RESTART_MIN_LEFT_S remain before the hard stop, and the server can be up before stop-new (the first start's
    duration, at least RESTART_UP_S; a server that comes up later cannot start any attempt)."""
    now = elapsed() if now is None else now
    return (STATE["restarts"] < MAX_RESTARTS and HARD_STOP_S - now > RESTART_MIN_LEFT_S
            and STOP_NEW_S - now > max(RESTART_UP_S, STATE.get("up_s") or 0.0))


def preflight_retry(model):
    """S7: the staging pre-flight failed. A transient unreadable safetensors header (a model-mount error; submission
    57534139) must not end the run before the server starts: re-run the kit's pre-flight with backoff for up to
    PREFLIGHT_RETRY_S, then start the server anyway (a truly partial checkpoint scores no worse than the fallback)."""
    t, wait_s = time.time(), PREFLIGHT_WAIT_S
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("stage_model", KIT / "stage_model.py")
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
    except Exception as exc:
        log(f"WARNING pre-flight retry unavailable ({type(exc).__name__}: {exc}); starting the server anyway")
        return
    while True:
        try:
            bad = stage.preflight(model)
        except Exception as exc:
            bad = f"{type(exc).__name__}: {exc}"
        if not bad:
            log(f"checkpoint pre-flight passed on retry after {time.time() - t:.0f}s")
            return
        if time.time() - t + wait_s > PREFLIGHT_RETRY_S:
            log(f"WARNING checkpoint pre-flight still failing after {time.time() - t:.0f}s ({bad[:200]}); starting "
                "the server anyway")
            return
        log(f"checkpoint pre-flight failed ({bad[:200]}); retry in {wait_s:.0f}s")
        time.sleep(wait_s)
        wait_s = min(2 * wait_s, 120.0)


def stop_server():
    """Stop the server's whole process group (also when its leader already exited: TP workers or the engine core
    can outlive it and hold GPU memory), then wait up to 120 s for the GPUs to free."""
    proc = STATE.get("server")
    STATE["server_ok"] = False
    if proc is None:
        return
    for sig, wait_s in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        try:
            os.killpg(proc.pid, sig)  # the group id outlives its leader
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    t = time.time()
    while time.time() - t < 120:
        try:
            out = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            used, _ = out.communicate(timeout=20)
            if used and max(int(x) for x in used.split()) < 1500:
                return
        except Exception:
            try:
                out.kill()  # no wait(): nvidia-smi can hang in D state after a GPU fault
            except Exception:
                pass
        time.sleep(5)
    log("warning: GPU memory not freed 120 s after stopping the server")


def tail(path, n=600):
    try:
        return Path(path).read_bytes()[-n:].decode(errors="replace").replace("\n", " | ")
    except OSError:
        return ""


def grep_kv(path):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    import re
    kv = re.findall(r"GPU KV cache size: [0-9,]+ tokens", text)
    w = re.findall(r"Model loading took [0-9.]+ GiB", text)
    return "; ".join(x for x in (kv[-1] if kv else "", w[0] if w else "") if x)


# ---------------------------------------------------------------- client wrapper: deadlines and loop guard

class _Usage:
    def __init__(self, completion):
        self.completion = completion

    def model_dump(self):
        return {"prompt_tokens": 0, "completion_tokens": self.completion, "total_tokens": self.completion}


def tokenizer():
    """The model's tokenizer.json (tokenizers library), for decoding loop-guard windows; None if unavailable."""
    if "tok" not in STATE:
        try:
            from tokenizers import Tokenizer
            STATE["tok"] = Tokenizer.from_file(str(Path(STATE["model"]) / "tokenizer.json"))
        except Exception as exc:
            log(f"loop guard: no tokenizer ({type(exc).__name__}); using streamed text")
            STATE["tok"] = None
    return STATE["tok"]


def gzip_ratio(text):
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(gzip.compress(raw, compresslevel=6, mtime=0)))


def _cut_chunk():
    choice = SimpleNamespace(index=0, text="", finish_reason="length", model_extra={},
                             delta=SimpleNamespace(content="", reasoning_content=None, tool_calls=None, model_extra={}))
    return SimpleNamespace(usage=_Usage(BIG), choices=[choice], model_extra={})


class GuardedStream:
    """Iterates a streaming response; past the force deadline or on a detected loop it closes the real stream and
    yields one final chunk with finish_reason "length" and an exhausted output budget."""

    def __init__(self, factory, stats):
        self.stats, self.real, self.cut = stats, None, None
        self.text, self.ids, self.ntok, self.next_check, self.hits = [], [], 0, LOOP_WINDOW, 0
        self.thinking = True  # loop checks stop once the turn leaves its thinking block
        self.finished = False  # a choice carried finish_reason: the turn is complete, never cut it
        if STATE["forced"]:
            self.cut = "deadline"
        else:
            self.real = factory()
            with LOCK:
                STREAMS.add(self)
            if STATE["forced"]:  # S14: the sweep ran between the check above and the add
                self.close(hard=True)

    def _loop(self):
        """Disjoint windows of the last LOOP_WINDOW tokens (decoded ids when the stream carries them)."""
        tok = tokenizer() if self.ids else None
        if tok is not None:
            window = tok.decode(self.ids[-LOOP_WINDOW:], skip_special_tokens=False)
            self.ids = self.ids[-LOOP_WINDOW:]
        else:
            window = "".join(self.text)[-LOOP_WINDOW * 3:]
            self.text = [window]
        self.hits = self.hits + 1 if gzip_ratio(window) > LOOP_RATIO else 0
        return self.hits >= LOOP_CONFIRM

    def __iter__(self):
        if self.cut is None:
            try:
                for chunk in self.real:
                    if STATE["forced"] and not self.finished:
                        self.cut = "deadline"
                        break
                    for ch in chunk.choices or []:
                        piece = getattr(ch, "text", None)
                        if piece is None and getattr(ch, "delta", None) is not None:
                            d = ch.delta
                            piece = (d.content or "") + (getattr(d, "reasoning_content", None)
                                                         or (d.model_extra or {}).get("reasoning") or "")
                        ids = (ch.model_extra or {}).get("token_ids")
                        if getattr(ch, "finish_reason", None):
                            self.finished = True
                        if getattr(ch, "delta", None) is not None and (ch.delta.content or ch.delta.tool_calls):
                            self.thinking = False  # chat path: content or a tool call means the thinking ended
                        if piece and "</think>" in piece:
                            self.thinking = False
                        if not self.thinking:
                            pass
                        elif ids:
                            self.ids.extend(ids)
                            self.ntok += len(ids)
                            if piece:
                                self.text.append(piece)  # used if the tokenizer is unavailable
                        else:
                            self.ntok += max(1, len(piece or "") // 3)
                            if piece:
                                self.text.append(piece)
                    yield chunk
                    if self.thinking and not self.finished and LOOP_RATIO > 0 and self.ntok >= self.next_check:
                        self.next_check = self.ntok + LOOP_EVERY
                        if self._loop():
                            self.cut = "loop"
                            break
                if self.cut is None and STATE["forced"] and not self.finished:
                    self.cut = "deadline"  # closed by the force sweep and ended without an exception
            except Exception:
                if not STATE["forced"] or self.finished:
                    raise
                self.cut = "deadline"  # the stream was closed under us by the force-deadline sweep
            finally:
                with LOCK:
                    STREAMS.discard(self)
        if self.cut is not None:
            self.stats[self.cut] = self.stats.get(self.cut, 0) + 1
            self.close()
            yield _cut_chunk()

    def close(self, hard=False):
        """hard (S14, the force sweep): shut the socket down first, so a reader blocked on a stalled stream wakes at
        once; closing the response from another thread does not wake it (e2e close_test.py: still blocked after
        10 s, vs 1 s with the shutdown)."""
        if self.real is not None:
            if hard:
                shutdown_socket(self.real)
            try:
                self.real.close()
            except Exception:
                pass


def shutdown_socket(stream):
    """S14: shut down the socket under an open streaming response (openai Stream -> httpx response -> httpcore network
    stream); True if one was shut down. Never raises."""
    try:
        ns = (getattr(getattr(stream, "response", None), "extensions", None) or {}).get("network_stream")
        sock = ns.get_extra_info("socket") if ns is not None else None
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
            return True
    except Exception:
        pass
    return False


class GuardedClient:
    """The harness's OpenAI client with every streaming call routed through GuardedStream."""

    def __init__(self, real):
        self.real, self.base_url, self.stats = real, real.base_url, {}
        outer = self

        class _Comp:
            @staticmethod
            def create(**kw):
                return GuardedStream(lambda: real.completions.create(**kw), outer.stats)

        class _Chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return GuardedStream(lambda: real.chat.completions.create(**kw), outer.stats)

        self.completions, self.chat = _Comp(), _Chat()


def force_sweep():
    """Force deadline: from now on new streams are cut at once, and open ones are closed (S14: socket shutdown, so
    stalled streams end too). Each cut attempt then ends through the harness's fallback (the session's answer)."""
    STATE["forced"] = True
    with LOCK:
        open_streams = list(STREAMS)
    for s in open_streams:
        s.close(hard=True)
    log(f"force deadline: {len(open_streams)} open streams cut")


# ---------------------------------------------------------------- attempts

def make_args():
    import run_deepseek_tools as h
    # Wall-clock wrap-up (Nguyen's finalization window): within WRAPUP_S of the force deadline, every tool output
    # also says how long is left, so running attempts submit before the cut instead of falling back to `answer`.
    orig = h.budget_note

    def budget_note(args, used):
        note = orig(args, used)
        left = FORCE_AT_S - elapsed()
        if 0 < left < WRAPUP_S:
            note += (f"\n[time: about {max(1, round(left / 60))} min left for this task: submit your best answer now "
                     "with submit_answer, or at least set `answer`.]")
        return note
    h.budget_note = budget_note
    # After the force deadline, skip the train check (no model tokens, but up to minutes of probes per attempt).
    orig_tc = h.train_check

    def train_check(*a, **kw):
        if STATE["forced"]:
            return {"train_pass": None, "train_frac": None, "train_trace": "skipped_deadline"}
        return orig_tc(*a, **kw)
    h.train_check = train_check
    return h.parser().parse_args(HARNESS + (["--multi-test"] if MULTI_TEST else [])
                                 + (["--auto-repair"] if AUTO_REPAIR else [])
                                 + (["--multi-budget", MULTI_BUDGET] if MULTI_TEST and MULTI_BUDGET else [])
                                 + (["--context-limit", CONTEXT_LIMIT] if CONTEXT_LIMIT else [])
                                 + (["--best-guess-at", BEST_GUESS_AT] if BEST_GUESS_AT else [])
                                 + (["--best-guess-turns", BEST_GUESS_TURNS] if BEST_GUESS_TURNS else [])
                                 + (["--post-pass-note"] if POST_PASS_NOTE else []) + [
                                            "--max-turns", str(MAX_TURNS), "--base-url", URL + "/v1",
                                            "--extra-body", json.dumps(EXTRA_BODY), "--sandbox-url", "local",
                                            "--attempts", str(ATTEMPTS)])


def session_factory(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    from arc_harness.executor import LocalExecutor
    return LocalExecutor(train, test_input, timeout, max_output, memory_mb=EXEC_MEM_MB, python=TOOL_PYTHON,
                         prelude_extra=prelude_extra, style="ipython", cpu_seconds=7200)


def run_attempt(args, task, t, p, a):
    import run_deepseek_tools as h
    from openai import OpenAI
    import httpx
    client = GuardedClient(OpenAI(api_key="none", base_url=URL + "/v1", max_retries=0,
                                  timeout=httpx.Timeout(7200, connect=30, read=900)))
    t1 = time.time()
    with LOCK:  # T3: a re-run of a re-queued attempt gets its own trace file (_r1, _r2, ...)
        run = RUNS[(t, p, a)] = RUNS.get((t, p, a), -1) + 1
    trace_file = f"{t}_p{p}_a{a}" + (f"_r{run}" if run else "") + ".json.gz"
    multi = MULTI_TEST and len(task["test"]) > 1
    trace, result = h.solve(client, args, task, task["test"][p]["input"], session_factory=session_factory,
                            **({"test_inputs": [x["input"] for x in task["test"]]} if multi else {}))
    result.update({"task_id": t, "pair": p, "attempt": a, "wall_s": round(time.time() - t1, 1),
                   "cuts": client.stats, "t_end_h": round(elapsed() / 3600, 3), "trace_file": trace_file})
    if client.stats:
        result["usage"]["completion_reported_cut"] = True  # BIG was added for the cut; real tokens are lower
    try:
        with gzip.open(LOG / "traces" / trace_file, "wt") as f:
            json.dump(trace, f)
    except Exception:
        pass
    return result


def pick(queue):
    """Index of the next job. Attempt-0 jobs go first, in queue order (smallest tasks first). Among attempt-1 jobs:
    pairs whose attempt 0 ended without a train-passing answer first, then pairs whose attempt 0 is still running,
    then pairs that already have a train-passing answer (on E043's E128 run, skipping those would cost 8 of 167
    pairs, so they still run, but last: a short time budget drops them first)."""
    best, key = 0, None
    for i, (t, p, a, _) in enumerate(queue):
        if a == 0:
            return i
        with LOCK:
            recs = [r for r in RECORDS.get((t, p), []) if r.get("attempt") == 0]
        k = 1 if not recs else (2 if any(r.get("train_pass") is True for r in recs) else 0)
        if key is None or k < key:
            best, key = i, k
            if k == 0:
                break
    return best


def split_multi(rec, task):
    """E046: a --multi-test attempt (pred = list of grids, one per test input) as one record per pair; the train
    label (the program behind test input 0) applies to every pair."""
    n = rec.get("multi_test")
    if not n:
        return [rec]
    preds = rec["pred"] if isinstance(rec.get("pred"), list) and len(rec["pred"]) == n else [None] * n
    return [dict(rec, pair=q, pred=preds[q], **({"multi_share": True} if q else {})) for q in range(n)]


def keep(rec, task, provisional=False):
    """Add an attempt's records (one per pair) to RECORDS (caller holds LOCK). S6: a provisional record (the session
    answer harvested from an attempt that an error ended and that was re-queued) counts until a later record of the
    same attempt brings a valid grid for that pair."""
    for r in split_multi(rec, task):
        recs = RECORDS.setdefault((r["task_id"], r["pair"]), [])
        if valid_grid(r.get("pred")):
            recs[:] = [x for x in recs if not (x.get("provisional") and x.get("attempt") == r.get("attempt"))]
        recs.append(dict(r, provisional=True) if provisional else r)


def score(tasks):
    if not SOLUTIONS or not Path(SOLUTIONS).exists():
        return None
    sol = json.loads(Path(SOLUTIONS).read_text())
    sub = json.loads(SUB.read_text())
    pairs = hit = 0
    for t in tasks:
        for p, row in enumerate(sub[t]):
            pairs += 1
            hit += sol[t][p] in (row["attempt_1"], row["attempt_2"])
    return hit, pairs


METRIC_KEYS = ("generation_tokens_total", "prefix_cache_hits_total", "prefix_cache_queries_total",
               "kv_cache_usage_perc", "num_requests_running", "num_requests_waiting", "num_preemptions_total",
               "external_prefix_cache_hits_total")
METRICS_EVERY_S = float(E("ARC_METRICS_EVERY_S", "300"))


def server_metrics():
    """The server's Prometheus counters we track (first finite sample of each name), or None if unreachable. S13:
    never raises (V6: a tracked `vllm:` line without a value raised IndexError through the main loop's KV gate and
    ended the run); a line that does not parse, or a NaN / inf value, is skipped."""
    try:
        with urllib.request.urlopen(URL + "/metrics", timeout=5) as r:
            text = r.read().decode(errors="replace")
        m = {"t": time.time()}
        for line in text.splitlines():
            if line.startswith("vllm:"):
                try:
                    name = line.split("{", 1)[0].split(" ", 1)[0][5:]
                    if name in METRIC_KEYS and name not in m:
                        value = float(line.rsplit(" ", 1)[1])
                        if math.isfinite(value):
                            m[name] = value
                except (ValueError, IndexError):
                    pass
        return m
    except Exception:
        return None


def kv_admits(n_running):
    """KV-gated admission: True if one more attempt may start now. KV usage is re-read at most every 10 s; a launch
    near the threshold waits for a fresh reading, since the new attempt's prompt is not in KV yet."""
    if KV_ADMIT_MAX <= 0 or n_running < KV_ADMIT_MIN_WORKERS:
        return True
    now = time.time()
    if now - STATE.get("kv_t", 0) >= 10:
        m = server_metrics()
        STATE["kv_t"], STATE["kv"] = now, (m or {}).get("kv_cache_usage_perc")
        STATE["kv_read"], STATE["srv_waiting"] = STATE["kv"], (m or {}).get("num_requests_waiting")
        STATE["kv_missing"] = 0 if STATE["kv"] is not None else STATE.get("kv_missing", 0) + 1
    kv = STATE.get("kv")
    if kv is None:
        # S1: no KV usage. A brief gap admits as before; after KV_MISSING_POLLS readings in a row the gate fails safe
        # (no new attempt above KV_MISSING_WORKERS running) instead of open
        if STATE.get("kv_missing", 0) <= KV_MISSING_POLLS:
            return True
        ok = n_running < KV_MISSING_WORKERS
        if not ok and now - STATE.get("kv_gate_logged", 0) >= 300:
            STATE["kv_gate_logged"] = now
            log(f"WARNING KV gate: no KV usage from /metrics in {STATE['kv_missing']} readings; holding new attempts "
                f"at {n_running} running (fail-safe cap {KV_MISSING_WORKERS})")
        return ok
    waiting = STATE.get("srv_waiting")
    if ADMIT_MAX_WAITING > 0 and waiting is not None and waiting >= ADMIT_MAX_WAITING:
        if now - STATE.get("kv_gate_logged", 0) >= 300:
            STATE["kv_gate_logged"] = now
            log(f"KV gate: server has {waiting:.0f} waiting requests (limit {ADMIT_MAX_WAITING}), KV {kv:.3f}; holding "
                f"new attempts at {n_running} running")
        return False
    ok = kv < KV_ADMIT_MAX
    if ok and kv >= KV_ADMIT_MAX - 0.1:
        STATE["kv"] = KV_ADMIT_MAX  # one launch per reading in the last 10 points below the threshold
    if not ok and now - STATE.get("kv_gate_logged", 0) >= 300:
        STATE["kv_gate_logged"] = now
        log(f"KV gate: usage {STATE.get('kv_read') or kv:.3f} (limit {KV_ADMIT_MAX}); holding new attempts at "
            f"{n_running} running")
    return ok


def conn_hold():
    """S21: True while a recent server-side error is not yet followed by a server that answers /v1/models."""
    if not CONN_HOLD or not STATE.get("conn_error_t"):
        return False
    if time.time() - STATE.get("conn_probe_t", 0) < 5:
        return STATE.get("conn_hold", True)
    STATE["conn_probe_t"] = time.time()
    STATE["conn_hold"] = not server_alive()
    if not STATE["conn_hold"]:
        STATE["conn_error_t"] = None
    return STATE["conn_hold"]


def worker_cap():
    """S22: WORKERS, or the post-restart ramp start + per_min x minutes since the restart."""
    if not RESTART_RAMP or STATE.get("ramp_t0") is None:
        return WORKERS
    cap = RESTART_RAMP[0] + RESTART_RAMP[1] * int((elapsed() - STATE["ramp_t0"]) / 60)
    if cap >= WORKERS:
        STATE["ramp_t0"] = None
        return WORKERS
    return cap


def stall_check(now, n_running):
    """S20: mark the server dead when its generated-token counter has not moved for STALL_RESTART_S while attempts run
    (an unreadable /metrics counts as no progress). Checked every min(30 s, STALL_RESTART_S / 4)."""
    if now - STATE.get("stall_check_t", -1e9) < max(1.0, min(30.0, STALL_RESTART_S / 4)):
        return
    STATE["stall_check_t"] = now
    STATE.setdefault("stall_t", now)
    m = server_metrics()
    gen = (m or {}).get("generation_tokens_total")
    # idle (nothing to generate) only when no request is running OR waiting: requests stuck on KV-offload loads count
    # as waiting (vllm v1/metrics/loggers.py), and those must still trip the watchdog
    idle = not ((m or {}).get("num_requests_running") or (m or {}).get("num_requests_waiting"))
    if gen is not None and (gen > STATE.get("stall_gen", -1.0) or idle):
        STATE["stall_gen"], STATE["stall_t"] = gen, now
    elif now - STATE["stall_t"] >= STALL_RESTART_S:
        log(f"server stalled: no generated tokens for {now - STATE['stall_t']:.0f} s with {n_running} attempts running"
            f" ({'no /metrics' if m is None else 'counter flat'}); treating it as dead")
        STATE["server_ok"] = False
        STATE["stall_t"] = now


def throughput():
    """Generated tok/s and prefix-cache hit rate since the previous call (TPS log every METRICS_EVERY_S). S16: with KV
    offload, tokens the connector restores from host (`external_prefix_cache_hits_total`) are not GPU prefix hits but
    are not recomputed either, so `uncached_tok_s` leaves them out and `host_tok_s` reports them."""
    cur = server_metrics()
    if cur is None:
        return None
    prev, STATE["metrics"] = STATE.get("metrics"), cur
    if prev is None or cur["t"] - prev["t"] < 1:
        return None
    dt = cur["t"] - prev["t"]
    d = lambda k: cur.get(k, 0.0) - prev.get(k, 0.0)
    q, host = d("prefix_cache_queries_total"), d("external_prefix_cache_hits_total")
    tp = {"window_s": round(dt), "gen_tok_s": round(d("generation_tokens_total") / dt, 1),
          "prefix_hit": round(d("prefix_cache_hits_total") / q, 3) if q > 0 else None,
          "uncached_tok_s": round((q - d("prefix_cache_hits_total") - host) / dt, 1),
          "kv": round(cur.get("kv_cache_usage_perc", 0.0), 3), "srv_running": cur.get("num_requests_running"),
          "srv_waiting": cur.get("num_requests_waiting"), "preemptions": cur.get("num_preemptions_total")}
    if "external_prefix_cache_hits_total" in cur:
        tp["host_tok_s"] = round(host / dt, 1)
    return tp


def status(tasks, queue, running, done, errors):
    try:
        st = {"elapsed_h": round(elapsed() / 3600, 3), "phase": STATE["phase"], "server_ok": STATE["server_ok"],
              "restarts": STATE["restarts"], "queued": len(queue), "running": len(running), "done": done,
              "errors": errors, "pairs_with_answer": sum(1 for v in RECORDS.values() if any(r.get("pred") for r in v)),
              "pairs": sum(len(x["test"]) for x in tasks.values())}
        if STATE["server_ok"] and time.time() - STATE.get("metrics_logged", 0) >= METRICS_EVERY_S:
            tp = throughput()
            if tp:
                STATE["metrics_logged"] = time.time()
                STATE["last_tp"] = tp
                tp["elapsed_h"], tp["done"], tp["running"] = st["elapsed_h"], done, len(running)
                with open(LOG / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(tp) + "\n")
                log(f"server: {tp['gen_tok_s']} gen tok/s, prefix hits {tp['prefix_hit']}, uncached "
                    f"{tp['uncached_tok_s']} tok/s, KV {tp['kv']}, running {tp['srv_running']}, waiting "
                    f"{tp['srv_waiting']}, preemptions {tp['preemptions']}"
                    + (f", host restored {tp['host_tok_s']} tok/s" if "host_tok_s" in tp else "")
                    + f" (last {tp['window_s']} s); {done} attempts done")
        st["server"] = STATE.get("last_tp")
        atomic_json(LOG / "status.json", st)
    except Exception:
        pass


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # inherited by the server and executors
    except Exception:
        pass
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "traces").mkdir(exist_ok=True)
    sys.path[:0] = [str(KIT / "repo" / "evals")]
    # The submission always covers every task of the challenge file; ARC_TASK_LIMIT (smoke tests) only limits the
    # tasks that get attempts.
    all_tasks = json.loads(Path(CHALLENGES).read_text())
    tasks = dict(list(sorted(all_tasks.items()))[:TASK_LIMIT]) if TASK_LIMIT else all_tasks
    if TASK_IDS:
        tasks = {t: all_tasks[t] for t in TASK_IDS if t in all_tasks}
    write_fallback(all_tasks)
    n_pairs = sum(len(x["test"]) for x in tasks.values())
    log(f"fallback submission written ({len(all_tasks)} tasks); attempting {len(tasks)} tasks, {n_pairs} pairs; deadlines: stop-new "
        f"{STOP_NEW_S / 3600:.2f}h, force {FORCE_AT_S / 3600:.2f}h, hard stop {HARD_STOP_S / 3600:.2f}h")
    for w in ENV_WARNINGS:  # S12
        log("WARNING environment: " + w)
    if AUTO_REPAIR and not (KIT / "repo" / "evals" / "auto_repair.py").exists():
        log("WARNING ARC_AUTO_REPAIR=1 but the kit has no evals/auto_repair.py: auto-repair stays off")

    def on_signal(sig, _frame):
        log(f"signal {sig}: final write and exit")
        write_submission(all_tasks, "signal")
        stop_server()
        os._exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    model = E("ARC_MODEL_DIR") or next((str(p.parent) for p in sorted(Path("/kaggle/input").rglob("e043_manifest.json"))),
                                       "")
    if not model or not (Path(model) / "config.json").exists():
        log(f"no model dir found ({model!r}); keeping the fallback submission")
        return
    if STAGE_DONE:
        t = time.time()
        while not Path(STAGE_DONE).exists() and time.time() - t < STAGE_WAIT_S:
            time.sleep(5)
        staged = Path(STAGE_DONE).read_text().strip() if Path(STAGE_DONE).exists() else ""
        log(f"model staging: {staged or 'not finished'} after {time.time() - t:.0f}s wait")
        if staged.startswith("ok "):
            model = staged.split(" ", 1)[1]
        elif staged.startswith("fail preflight"):
            preflight_retry(model)  # S7: V6 kept the fallback submission here
    STATE["model"] = model
    args = make_args()

    STATE["phase"] = "server"
    while not start_server():  # S5: a failed start stops its process group before any retry
        if not restart_allowed():
            log(f"server failed to start ({STATE['restarts']} retries); keeping the fallback submission")
            return
        STATE["restarts"] += 1

    size = {t: len(json.dumps(x["train"])) + max(len(json.dumps(y["input"])) for y in x["test"]) for t, x in tasks.items()}
    order = sorted(tasks, key=lambda t: size[t])
    # E046 --multi-test: pair 0's job answers every test input of its task
    queue = [(t, p, a, 0) for a in range(ATTEMPTS) for t in order
             for p in range(1 if MULTI_TEST else len(tasks[t]["test"]))]
    running, done, errors = {}, 0, 0
    conn_retries = {}  # (task, pair, attempt) -> re-queues after server-side connection errors
    rates = []  # completion tokens per wall second of finished, uncut attempts
    STATE["phase"] = "attempts"
    pool = ThreadPoolExecutor(WORKERS)
    last_status, loop_errors = 0, 0
    try:
        while True:
            now = elapsed()
            if now >= HARD_STOP_S:
                log("hard stop")
                break
            try:
                if now >= FORCE_AT_S and not STATE["forced"]:
                    force_sweep()
                # Server watchdog.
                proc = STATE["server"]
                if STATE["server_ok"] and proc is not None and proc.poll() is not None:
                    STATE["server_ok"] = False
                    log(f"server died (code {proc.returncode}); "
                        f"tail: {tail(STATE.get('server_log') or LOG / 'server.log')}")
                if STALL_RESTART_S and STATE["server_ok"] and not STATE["forced"] and running:
                    stall_check(now, len(running))
                if not STATE["server_ok"] and not STATE["forced"]:
                    if restart_allowed(now):  # S6: up to MAX_RESTARTS, while the time left allows
                        STATE["restarts"] += 1
                        stop_server()
                        start_server()
                        STATE["stall_t"], STATE["stall_gen"] = elapsed(), -1.0
                        STATE["ramp_t0"] = elapsed()
                    elif not running:
                        log("server down and no restart left; finishing with what we have")
                        break
                # Launch.
                rate = sorted(rates)[len(rates) // 2] if len(rates) >= 5 else None
                if rate is not None and queue and (FORCE_AT_S - elapsed()) * rate < MIN_ATTEMPT_TOKENS:
                    if not STATE.get("admission_closed"):
                        STATE["admission_closed"] = True
                        log(f"admission closed: {(FORCE_AT_S - elapsed()) / 60:.0f} min x {rate:.1f} tok/s per stream "
                            f"< {MIN_ATTEMPT_TOKENS:.0f} tokens; {len(queue)} jobs left unstarted")
                # Only while the server process is alive: an attempt started against a dead server fails in 0.2 s with
                # "connection refused" (T2, 2026-10-24).
                while (queue and len(running) < worker_cap() and STATE["server_ok"] and not STATE["forced"]
                       and not conn_hold()
                       and STATE["server"] is not None and STATE["server"].poll() is None
                       and elapsed() < STOP_NEW_S and not STATE.get("admission_closed") and kv_admits(len(running))):
                    t, p, a, tries = queue.pop(pick(queue))
                    running[pool.submit(run_attempt, args, tasks[t], t, p, a)] = (t, p, a, tries)
                if not running:
                    if not queue or elapsed() >= STOP_NEW_S or STATE["forced"] or STATE.get("admission_closed"):
                        break
                    time.sleep(5)
                    continue
                finished, _ = wait(list(running), timeout=10, return_when=FIRST_COMPLETED)
                for fut in finished:
                    t, p, a, tries = running.pop(fut)
                    try:
                        try:
                            rec = fut.result()
                        except Exception as exc:
                            rec = {"task_id": t, "pair": p, "attempt": a, "pred": None,
                                   "error": f"driver: {type(exc).__name__}: {exc}"[:500],
                                   "tb": traceback.format_exc()[-1500:]}
                        with open(LOG / "attempts.jsonl", "a") as f:
                            f.write(json.dumps({k: v for k, v in rec.items() if k != "content"}) + "\n")
                        key = (t, p, a)
                        requeue = None
                        if rec.get("error") and not STATE["forced"]:
                            if any(c in rec["error"] for c in CONN_ERRORS) and conn_retries.get(key, 0) < CONN_TRIES:
                                # Lost to the server (died, restarting, or dropped the stream): re-queue without using
                                # a try.
                                conn_retries[key] = conn_retries.get(key, 0) + 1
                                STATE["conn_error_t"] = time.time()  # S21: hold launches until the server answers
                                requeue = ((t, p, a, tries),
                                           f"re-queue {t}.{p}#{a} after a server-side error ({conn_retries[key]})")
                            elif tries + 1 < JOB_TRIES:
                                requeue = (t, p, a, tries + 1), f"retry {t}.{p}#{a}"
                        if requeue:
                            errors += 1
                            queue.insert(0, requeue[0])
                            # S6 / A4: the harness read the session's answer before the error ended the attempt;
                            # it counts until the re-run brings a grid
                            harvested = rec.get("pred") is not None and "task_id" in rec
                            if harvested:
                                with LOCK:
                                    keep(rec, tasks[t], provisional=True)
                                write_submission(all_tasks)
                            log(f"{requeue[1]}: {rec['error'][:160]}"
                                + ("; its session answer is kept" if harvested else ""))
                            continue
                        done += 1
                        if not rec.get("cuts") and rec.get("wall_s") and rec.get("usage", {}).get("completion"):
                            rates.append(rec["usage"]["completion"] / rec["wall_s"])
                        with LOCK:
                            keep(rec, tasks[t])
                        write_submission(all_tasks)
                        log(f"[{done}] {t}.{p}#{a} pred={'yes' if rec.get('pred') else 'no'} "
                            f"stop={rec.get('stop_reason')} turns={rec.get('turns')} "
                            f"out={rec.get('usage', {}).get('completion')} "
                            f"train_pass={rec.get('train_pass')} cuts={rec.get('cuts')} {rec.get('wall_s')}s")
                    except Exception as exc:  # one bad record (disk, encoding) must not end the run
                        log(f"result handling failed for {t}.{p}#{a}: {type(exc).__name__}: {exc}"[:400])
                if time.time() - last_status > 60:
                    last_status = time.time()
                    status(tasks, queue, running, done, errors)
            except Exception as exc:
                # S13: an exception in one iteration (V6: a garbled /metrics line) ended the run silently with
                # fallback rows, because `finally` exits before the crash handler. Log it loudly and go on.
                loop_errors += 1
                if loop_errors <= 20 or loop_errors % 100 == 0:
                    log(f"ERROR main loop iteration failed ({loop_errors}): {type(exc).__name__}: {exc} | "
                        + traceback.format_exc()[-1500:].replace("\n", " | "))
                time.sleep(5)
    finally:
        STATE["phase"] = "final"
        write_submission(all_tasks, "final")
        status(tasks, queue, running, done, errors)
        try:
            s = score(tasks)
            if s:
                log(f"local score: {s[0]}/{s[1]} pairs")
        except Exception as exc:
            log(f"local score failed: {type(exc).__name__}: {exc}")
        stop_server()
        log("done")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    LOG.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception:
        log("solver crashed: " + traceback.format_exc()[-2000:])
        stop_server()
        os._exit(0)
