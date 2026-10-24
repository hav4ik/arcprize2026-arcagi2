#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 tool solver for the ARC Prize 2026 Kaggle rerun on 4xL4 (E042 RW, E043 tool E128 / E112).

One process does everything after the vLLM install (the notebook installs the fork and its overlay first):

1. Writes a valid fallback submission.json at once (every test pair: the test input, twice), then rewrites it
   atomically after every finished attempt, so the file is valid whenever the notebook dies.
2. Starts the vLLM server (the E042 RW flags: TP4, fp8 KV, util 0.93, 160K, prefix caching, retention 128) in its
   own session and waits for it. If it never comes up, the fallback submission stays and the solver exits 0.
3. Runs E043's tool harness (`run_deepseek_tools.solve`, the sub80s40 + minimal-header config, --token-exact,
   --rescue-code, --train-check) on every test pair, attempt 0 for all pairs first (smallest tasks first), then
   attempt 1, with ARC_WORKERS attempts at once. Tool code runs in `arc_harness.executor.LocalExecutor` (one child
   Python per attempt, IPython-style output like the Docker sandbox the configs were measured with).
4. Deadlines, measured from ARC_T0 (the notebook's first cell):
   - ARC_STOP_NEW_S: no new attempts after this;
   - ARC_FORCE_AT_S: every open generation stream is cut and reported as "output budget spent", so each running
     attempt ends through the harness's own fallback (the session's `answer`), not through an error;
   - ARC_HARD_STOP_S: final write, server stopped, exit, whatever is still running.
5. Loop guard (Nguyen's gzip monitor): while a turn is still thinking, the last ARC_LOOP_WINDOW tokens are
   gzip-compressed every ARC_LOOP_EVERY tokens. ARC_LOOP_CONFIRM consecutive windows at ratio >= ARC_LOOP_RATIO end the attempt
   like a budget cut (the session's `answer` is still used).
6. Watchdog: if the server process dies, running attempts error out. The server is restarted (at most
   ARC_MAX_SERVER_RESTARTS times, if enough time is left) and errored attempts are queued again (each job at most
   ARC_JOB_TRIES times).
7. SIGTERM / SIGINT: write the submission, stop the server, exit.
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
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

T0 = float(os.environ.get("ARC_T0") or time.time())
E = os.environ.get
KIT = Path(E("ARC_KIT", Path(__file__).resolve().parent))
OUT = Path(E("ARC_OUT", "/kaggle/working"))
LOG = OUT / "dsv41_logs"
SUB = OUT / "submission.json"
CHALLENGES = E("ARC_CHALLENGES", "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
SOLUTIONS = E("ARC_SOLUTIONS", "")
VLLM = E("ARC_VLLM", "/tmp/dsv41/venv/bin/vllm")
TOOL_PYTHON = E("ARC_TOOL_PYTHON", sys.executable)
PORT = int(E("ARC_PORT", "8100"))
WORKERS = int(E("ARC_WORKERS", "40"))
MAX_TURNS = int(E("ARC_MAX_TURNS", "40"))
STOP_NEW_S = float(E("ARC_STOP_NEW_S", 10 * 3600 + 50 * 60))
FORCE_AT_S = float(E("ARC_FORCE_AT_S", 11 * 3600 + 10 * 60))
HARD_STOP_S = float(E("ARC_HARD_STOP_S", 11 * 3600 + 30 * 60))
UP_TIMEOUT_S = float(E("ARC_SERVER_UP_TIMEOUT_S", 3000))
MAX_RESTARTS = int(E("ARC_MAX_SERVER_RESTARTS", "1"))
RESTART_MIN_LEFT_S = float(E("ARC_RESTART_MIN_LEFT_S", 2700))
JOB_TRIES = int(E("ARC_JOB_TRIES", "2"))
TASK_LIMIT = int(E("ARC_TASK_LIMIT", "0"))
ATTEMPTS = int(E("ARC_ATTEMPTS", "2"))
LOOP_WINDOW = int(E("ARC_LOOP_WINDOW", "1000"))
LOOP_EVERY = int(E("ARC_LOOP_EVERY", "1000"))
LOOP_RATIO = float(E("ARC_LOOP_RATIO", "8"))
# Nguyen uses 3 confirmations; on E043's traces that would cut normal terse reasoning (E112 streaks of 7-8 windows
# in correct-looking notes; E128 has none in 5,699 turns, and its only 3+ streak is grid literals in tool code).
# Thinking only, 10 windows: catches true doom loops (E112 "thermometer thermometer ...", 18 windows).
LOOP_CONFIRM = int(E("ARC_LOOP_CONFIRM", "10"))
EXEC_MEM_MB = int(E("ARC_EXEC_MEM_MB", "4096"))
SALT = E("ARC_CACHE_SALT", "kaggle")
MIN_ATTEMPT_TOKENS = float(E("ARC_MIN_ATTEMPT_TOKENS", "30000"))  # admission: time left x per-stream rate
STAGE_DONE = E("ARC_STAGE_DONE", "")  # marker file written by the notebook's model-staging copy
STAGE_WAIT_S = float(E("ARC_STAGE_WAIT_S", "2400"))
BIG = 10 ** 9  # completion tokens reported for a cut stream: the harness then stops on its output budget

URL = f"http://127.0.0.1:{PORT}"
HARNESS = [
    "--mode", "grid", "--effort", "low", "--prompt", "lean2sub", "--answer-tool", "--final-mode", "session",
    "--lenient-parse", "--max-tokens", "64000", "--max-total-output", "80000", "--budget-soft", "40000",
    "--budget-notes", "0.01", "--budget-warn", "0.75", "--effort-header", "minimal", "--turns-note", "5",
    "--train-check", "--rescue-code", "--token-exact", "--model", "dsv41",
]
EXTRA_BODY = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "temperature": 1.0, "top_p": 1.0,
              "cache_salt": SALT}

LOCK = threading.RLock()
STATE = {"forced": False, "server": None, "restarts": 0, "server_ok": False, "phase": "start"}
RECORDS = {}  # (task, pair) -> list of finished attempt records
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
    })
    for k in ("NCCL_P2P_LEVEL", "VLLM_MARLIN_INPUT_DTYPE"):  # never on Kaggle (E027: P2P there is ~12x slower)
        env.pop(k, None)
    env.update(json.loads(E("ARC_SERVER_ENV_JSON", "{}")))
    return env


def server_cmd():
    return [VLLM, "serve", STATE["model"], "--served-model-name", "dsv41", "--host", "127.0.0.1", "--port", str(PORT),
            "--trust-remote-code", "--tensor-parallel-size", "4", "--distributed-executor-backend", "mp",
            "--kv-cache-dtype", "fp8", "--block-size", "128", "--max-model-len", E("ARC_MAX_MODEL_LEN", "163840"),
            "--max-num-seqs", E("ARC_MAX_NUM_SEQS", "128"), "--max-num-batched-tokens", "2048",
            "--gpu-memory-utilization", E("ARC_GPU_UTIL", "0.93"), "--enable-prefix-caching",
            "--prefix-cache-retention-interval", "128", "--engram-config", '{"cpu_offload":true}',
            "--tokenizer-mode", "deepseek_v41", "--language-model-only", "--reasoning-parser", "deepseek_v41",
            "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice", "--enable-prompt-tokens-details",
            "--compilation-config", '{"cudagraph_mode":"FULL"}'] + json.loads(E("ARC_SERVER_EXTRA_ARGS", "[]"))


def server_alive():
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server():
    """Start vLLM in its own session; True once /v1/models answers, False if it died or timed out."""
    n = STATE["restarts"]
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        if apps:
            log(f"warning: GPU processes before server start: {apps[:300]}")
    except Exception as exc:
        log(f"nvidia-smi check failed: {exc}")
    if server_alive():
        log(f"warning: something already answers on port {PORT}")
    path = LOG / (f"server.log" if n == 0 else f"server_restart{n}.log")
    log(f"server start ({'first' if n == 0 else f'restart {n}'}): model {STATE['model']}")
    with open(LOG / "server_cmd.json", "w") as f:
        json.dump({"cmd": server_cmd(), "env": {k: v for k, v in server_env().items()
                                                if k.startswith(("VLLM", "DSV41", "E043", "FLASHINFER", "TORCH_CUDA",
                                                                 "CUDA", "NCCL", "HF_"))}}, f, indent=1)
    fh = open(path, "ab")
    proc = subprocess.Popen(server_cmd(), stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=server_env(), start_new_session=True, cwd=str(LOG))
    STATE["server"] = proc
    t = time.time()
    while time.time() - t < UP_TIMEOUT_S and elapsed() < STOP_NEW_S:
        if proc.poll() is not None:
            log(f"server exited during startup (code {proc.returncode}); tail: {tail(path)}")
            return False
        if server_alive():
            log(f"server up after {time.time() - t:.0f}s; {grep_kv(path)}")
            STATE["server_ok"] = True
            return True
        time.sleep(5)
    log(f"server not up after {time.time() - t:.0f}s; stopping it")
    stop_server()
    return False


def stop_server():
    proc = STATE.get("server")
    STATE["server_ok"] = False
    if proc is None or proc.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        try:
            os.killpg(proc.pid, sig)  # the server leads its own session: this reaches its TP workers too
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


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
        if STATE["forced"]:
            self.cut = "deadline"
        else:
            self.real = factory()
            with LOCK:
                STREAMS.add(self)

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
                    if STATE["forced"]:
                        self.cut = "deadline"
                        break
                    for ch in chunk.choices or []:
                        piece = getattr(ch, "text", None)
                        if piece is None and getattr(ch, "delta", None) is not None:
                            d = ch.delta
                            piece = (d.content or "") + (getattr(d, "reasoning_content", None)
                                                         or (d.model_extra or {}).get("reasoning") or "")
                        ids = (ch.model_extra or {}).get("token_ids")
                        if getattr(ch, "delta", None) is not None and (ch.delta.content or ch.delta.tool_calls):
                            self.thinking = False  # chat path: content or a tool call means the thinking ended
                        if piece and "</think>" in piece:
                            self.thinking = False
                        if not self.thinking:
                            pass
                        elif ids:
                            self.ids.extend(ids)
                            self.ntok += len(ids)
                        else:
                            self.ntok += max(1, len(piece or "") // 3)
                            if piece:
                                self.text.append(piece)
                    yield chunk
                    if self.thinking and LOOP_RATIO > 0 and self.ntok >= self.next_check:
                        self.next_check = self.ntok + LOOP_EVERY
                        if self._loop():
                            self.cut = "loop"
                            break
            except Exception:
                if not STATE["forced"]:
                    raise
                self.cut = "deadline"  # the stream was closed under us by the force-deadline sweep
            finally:
                with LOCK:
                    STREAMS.discard(self)
        if self.cut is not None:
            self.stats[self.cut] = self.stats.get(self.cut, 0) + 1
            self.close()
            yield _cut_chunk()

    def close(self):
        if self.real is not None:
            try:
                self.real.close()
            except Exception:
                pass


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
    """Force deadline: from now on new streams are cut at once, and open ones are closed."""
    STATE["forced"] = True
    with LOCK:
        open_streams = list(STREAMS)
    for s in open_streams:
        s.close()
    log(f"force deadline: {len(open_streams)} open streams cut")


# ---------------------------------------------------------------- attempts

def make_args():
    import run_deepseek_tools as h
    return h.parser().parse_args(HARNESS + ["--max-turns", str(MAX_TURNS), "--base-url", URL + "/v1",
                                            "--extra-body", json.dumps(EXTRA_BODY), "--sandbox-url", "local",
                                            "--attempts", str(ATTEMPTS)])


def session_factory(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    from arc_harness.executor import LocalExecutor
    return LocalExecutor(train, test_input, timeout, max_output, memory_mb=EXEC_MEM_MB, python=TOOL_PYTHON,
                         prelude_extra=prelude_extra, style="ipython")


def run_attempt(args, task, t, p, a):
    import run_deepseek_tools as h
    from openai import OpenAI
    import httpx
    client = GuardedClient(OpenAI(api_key="none", base_url=URL + "/v1", max_retries=0,
                                  timeout=httpx.Timeout(7200, connect=30, read=1800)))
    t1 = time.time()
    trace, result = h.solve(client, args, task, task["test"][p]["input"], session_factory=session_factory)
    result.update({"task_id": t, "pair": p, "attempt": a, "wall_s": round(time.time() - t1, 1),
                   "cuts": client.stats, "t_end_h": round(elapsed() / 3600, 3)})
    if client.stats:
        result["usage"]["completion_reported_cut"] = True  # BIG was added for the cut; real tokens are lower
    try:
        with gzip.open(LOG / "traces" / f"{t}_p{p}_a{a}.json.gz", "wt") as f:
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


def status(tasks, queue, running, done, errors):
    try:
        atomic_json(LOG / "status.json", {
            "elapsed_h": round(elapsed() / 3600, 3), "phase": STATE["phase"], "server_ok": STATE["server_ok"],
            "restarts": STATE["restarts"], "queued": len(queue), "running": len(running), "done": done,
            "errors": errors, "pairs_with_answer": sum(1 for v in RECORDS.values() if any(r.get("pred") for r in v)),
            "pairs": sum(len(x["test"]) for x in tasks.values())})
    except Exception:
        pass


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "traces").mkdir(exist_ok=True)
    sys.path[:0] = [str(KIT / "repo" / "evals")]
    tasks = json.loads(Path(CHALLENGES).read_text())
    if TASK_LIMIT:
        tasks = dict(list(sorted(tasks.items()))[:TASK_LIMIT])
    write_fallback(tasks)
    n_pairs = sum(len(x["test"]) for x in tasks.values())
    log(f"fallback submission written: {len(tasks)} tasks, {n_pairs} pairs; deadlines: stop-new "
        f"{STOP_NEW_S / 3600:.2f}h, force {FORCE_AT_S / 3600:.2f}h, hard stop {HARD_STOP_S / 3600:.2f}h")

    def on_signal(sig, _frame):
        log(f"signal {sig}: final write and exit")
        write_submission(tasks, "signal")
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
    STATE["model"] = model
    args = make_args()

    STATE["phase"] = "server"
    if not start_server():
        if STATE["restarts"] < MAX_RESTARTS and HARD_STOP_S - elapsed() > RESTART_MIN_LEFT_S:
            STATE["restarts"] += 1
            if not start_server():
                log("server failed twice; keeping the fallback submission")
                return
        else:
            log("server failed; keeping the fallback submission")
            return

    size = {t: len(json.dumps(x["train"])) + max(len(json.dumps(y["input"])) for y in x["test"]) for t, x in tasks.items()}
    order = sorted(tasks, key=lambda t: size[t])
    queue = [(t, p, a, 0) for a in range(ATTEMPTS) for t in order for p in range(len(tasks[t]["test"]))]
    running, done, errors = {}, 0, 0
    rates = []  # completion tokens per wall second of finished, uncut attempts
    STATE["phase"] = "attempts"
    pool = ThreadPoolExecutor(WORKERS)
    last_status = 0
    try:
        while True:
            now = elapsed()
            if now >= HARD_STOP_S:
                log("hard stop")
                break
            if now >= FORCE_AT_S and not STATE["forced"]:
                force_sweep()
            # Server watchdog.
            proc = STATE["server"]
            if STATE["server_ok"] and proc is not None and proc.poll() is not None:
                STATE["server_ok"] = False
                log(f"server died (code {proc.returncode}); tail: {tail(LOG / 'server.log')}")
            if not STATE["server_ok"] and not STATE["forced"]:
                if STATE["restarts"] < MAX_RESTARTS and HARD_STOP_S - now > RESTART_MIN_LEFT_S:
                    STATE["restarts"] += 1
                    stop_server()
                    start_server()
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
            while (queue and len(running) < WORKERS and STATE["server_ok"] and not STATE["forced"]
                   and elapsed() < STOP_NEW_S and not STATE.get("admission_closed")):
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
                    rec = fut.result()
                except Exception as exc:
                    rec = {"task_id": t, "pair": p, "attempt": a, "error": f"driver: {type(exc).__name__}: {exc}"[:500],
                           "pred": None, "tb": traceback.format_exc()[-1500:]}
                with open(LOG / "attempts.jsonl", "a") as f:
                    f.write(json.dumps({k: v for k, v in rec.items() if k != "content"}) + "\n")
                if rec.get("error") and rec.get("pred") is None and tries + 1 < JOB_TRIES and not STATE["forced"]:
                    errors += 1
                    queue.insert(0, (t, p, a, tries + 1))
                    log(f"retry {t}.{p}#{a}: {rec['error'][:200]}")
                    continue
                done += 1
                if not rec.get("cuts") and rec.get("wall_s") and rec.get("usage", {}).get("completion"):
                    rates.append(rec["usage"]["completion"] / rec["wall_s"])
                with LOCK:
                    RECORDS.setdefault((t, p), []).append(rec)
                write_submission(tasks)
                log(f"[{done}] {t}.{p}#{a} pred={'yes' if rec.get('pred') else 'no'} stop={rec.get('stop_reason')} "
                    f"turns={rec.get('turns')} out={rec.get('usage', {}).get('completion')} "
                    f"train_pass={rec.get('train_pass')} cuts={rec.get('cuts')} {rec.get('wall_s')}s")
            if time.time() - last_status > 60:
                last_status = time.time()
                status(tasks, queue, running, done, errors)
    finally:
        STATE["phase"] = "final"
        write_submission(tasks, "final")
        status(tasks, queue, running, done, errors)
        s = score(tasks)
        if s:
            log(f"local score: {s[0]}/{s[1]} pairs")
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
