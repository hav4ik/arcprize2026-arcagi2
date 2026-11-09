"""ARC Prize 2026 (ARC-AGI-2) offline solver: Qwen3.8-27B-FP8 on vLLM (TP=4, 4xL4), no tools.

One self-contained file; kaggle/build_notebook.py embeds it in the Kaggle notebook. Modes (env ARC_MODE):

- submit:   full run on the competition test file (used automatically on the competition rerun).
- fallback: write a valid fallback submission.json and exit (used for the commit run before submitting).
- bench:    test run: provenance hashes, server start, long-context decode benchmark, short end-to-end run on
            evaluation tasks with a near deadline, scoring against the evaluation solutions.
            ARC_HASH=0 / ARC_BENCH_DECODE=0 skip the first two (validation run: ARC_E2E_TASKS, ARC_E2E_MIN).

Robustness rules: a complete, valid submission.json is written before anything else and rewritten atomically
after every finished attempt; every failure path keeps the last valid file.

Prompt, parser and Qwen raw-template rendering follow evals/arc_harness/arcprize.py and evals/run_qwen_generate.py
(ARC Prize official prompt, tier xhigh, temp 1.0, top_p 0.95, top_k 20, forced final answer at the budget).
"""
import asyncio
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

T0 = float(os.environ.get("ARC_T0", time.time()))  # notebook start (set by the first cell)
MODE = os.environ.get("ARC_MODE", "submit")
WORK = "/kaggle/working"
OUT_SUB = os.path.join(WORK, "submission.json")
LOG_DIR = os.path.join(WORK, "arc_logs")
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
SERVED = "qwen3.8-27b"

# ---------------------------------------------------------------- settings (env overridable)

def envf(name, default):
    return type(default)(os.environ.get(name, default))

FORCE_AT_S = envf("ARC_FORCE_AT_S", 11.0 * 3600)    # force final answers for all in-flight attempts
HARD_STOP_S = envf("ARC_HARD_STOP_S", 11.6 * 3600)  # stop everything, final write
B_MAX1 = envf("ARC_B_MAX1", 96000)                  # attempt-1 budget cap (thinking + answer)
B_MIN1 = envf("ARC_B_MIN1", 16000)
B_MAX2 = envf("ARC_B_MAX2", 64000)
B_MIN2 = envf("ARC_B_MIN2", 32000)                  # attempt 2 is launched only if it can get this much
FINAL_TOKENS = envf("ARC_FINAL_TOKENS", 3000)
RATE_PRIOR = envf("ARC_RATE_PRIOR", 350.0)          # tok/s before the run measures its own rate
MAX_MODEL_LEN = envf("ARC_MAX_MODEL_LEN", 131072)
MAX_SEQS = envf("ARC_MAX_SEQS", 64)
CONCURRENCY = envf("ARC_CONCURRENCY", 0)            # 0 = from KV capacity
TOKENS_PER_SEQ_KV = envf("ARC_KV_PER_SEQ", 30000)   # KV tokens assumed per sequence when sizing concurrency
MTP = envf("ARC_MTP", 0)                            # speculative tokens (0 = off)
TIER = os.environ.get("ARC_TIER", "xhigh")
TEMPERATURE = envf("ARC_TEMPERATURE", 1.0)
GPU_UTIL = envf("ARC_GPU_UTIL", 0.95)
SERVER = os.environ.get("ARC_SERVER", "sglang")      # sglang (our wheel source) | vllm (user's vllm-0.23.0 source)
AGENT = os.environ.get("ARC_AGENT", "tools")         # tools (E012 agent_qwen) | plain (v1 no-tool prompt)
TOOLS_CAP = envf("ARC_TOOLS_CAP", 48000)             # E012: hard cap per attempt (min with the adaptive share)
TOOLS_SOFT = envf("ARC_TOOLS_SOFT", 24000)
TOOLS_TIER = os.environ.get("ARC_TOOLS_TIER", "medium")
TOOL_TIMEOUT = envf("ARC_TOOL_TIMEOUT", 30.0)
LOCAL_FORCE_MIN = envf("ARC_LOCAL_FORCE_MIN", 40.0)  # local (non-rerun) validation: force answers at T0 + this
PKG = os.path.join(WORK, "pkg")  # bundled evals/arc_harness + notes/prompts, written by the notebook

PROMPT_TEMPLATE = "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.\n\nRespond in the format of the training output examples\n\n--Training Examples--\n{training_examples}\n--End of Training Examples--\n\n--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:"  # noqa: F821  (ARC Prize official prompt, injected by build_notebook.py)

CLOSE = "</think>"
FORCE_TEXT = "\n\nI have to give my final answer now.\n</think>\n\n"
LOW_YIELD = 20
TIER_SYSTEM = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
             "answer.",
    "medium": None,
}


def log(*a):
    el = time.time() - T0
    print(f"[{el / 3600:5.2f}h {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- prompt / parse (verbatim from arc_harness)

def build_prompt(train, test_input):
    ex = ""
    for i, pair in enumerate(train):
        ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(pair["input"]) + "\n\n"
        ex += "OUTPUT: \n\n" + json.dumps(pair["output"]) + "\n\n"
    return PROMPT_TEMPLATE.format(training_examples=ex, test_input=json.dumps(test_input))


def render_prompt(user_content, tier=TIER):
    s = ""
    if TIER_SYSTEM.get(tier):
        s += f"<|im_start|>system\n{TIER_SYSTEM[tier]}<|im_end|>\n"
    return s + f"<|im_start|>user\n{user_content.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def answer_section(gen):
    i = gen.rfind(CLOSE)
    return None if i < 0 else gen[i + len(CLOSE):]


def is_grid(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(r, list) for r in x)


def extract_from_boxed(s):
    m = re.search(r"\\boxed\{(.*?)\}", s, re.DOTALL)
    if m:
        try:
            g = json.loads(m.group(1).strip())
            return g if is_grid(g) else None
        except json.JSONDecodeError:
            return None
    return None


def backscan_json_parser(s):
    end = max(s.rfind("]"), s.rfind("}"))
    if end < 0:
        return None
    close = s[end]
    open_ = "[" if close == "]" else "{"
    depth = 1
    for i in range(end - 1, -1, -1):
        depth += (s[i] == close) - (s[i] == open_)
        if depth == 0:
            try:
                g = json.loads(s[i:end + 1])
                return g if is_grid(g) else None
            except json.JSONDecodeError:
                return None
    return None


def parse(s):
    return extract_from_boxed(s or "") or backscan_json_parser(s or "")


def valid_grid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
            and len({len(row) for row in grid}) == 1
            and all(type(v) is int and 0 <= v <= 9 for row in grid for v in row))


# ---------------------------------------------------------------- paths

def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def comp_file(name):
    return (find_one(f"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{name}")
            or find_one(f"/kaggle/input/arc-prize-2026-arc-agi-2/{name}")
            or find_one(f"/kaggle/input/**/{name}"))


def model_dir():
    for pat in ["/kaggle/input/models/mikedan7/qwen3-8-27b-fp8-official/**/config.json",
                "/kaggle/input/qwen3-8-27b-fp8-official/**/config.json",
                "/kaggle/input/**/qwen3-8-27b-fp8*/**/config.json"]:
        p = find_one(pat)
        if p:
            return os.path.dirname(p)
    raise FileNotFoundError("model dir")


def wheel_dir():
    """Our own wheel sources (attached as kernel sources): SGLang 0.5.20 cu130 or the user's vLLM 0.23.0 cu128."""
    pat = "sglang-0.5.20-*.whl" if SERVER == "sglang" else "vllm-0.23.0*.whl"
    p = find_one(f"/kaggle/input/notebooks/chankhavu/**/{pat}") or find_one(f"/kaggle/input/**/{pat}")
    if not p:
        raise FileNotFoundError(f"wheel source for {SERVER}")
    return os.path.dirname(p)


# ---------------------------------------------------------------- submission

class Submission:
    def __init__(self, tasks):
        self.tasks = tasks
        self.answers = {}  # (tid, pair) -> {attempt_index: grid}

    def fallback(self, tid, p):
        g = self.tasks[tid]["test"][p]["input"]
        return g if valid_grid(g) else [[0]]

    def build(self):
        sub = {}
        for tid, task in self.tasks.items():
            lst = []
            for p in range(len(task["test"])):
                got = [g for _, g in sorted(self.answers.get((tid, p), {}).items()) if valid_grid(g)]
                a1 = got[0] if got else self.fallback(tid, p)
                a2 = next((g for g in got[1:] if g != a1), None)
                if a2 is None:
                    a2 = self.fallback(tid, p) if a1 != self.fallback(tid, p) else a1
                lst.append({"attempt_1": a1, "attempt_2": a2})
            sub[tid] = lst
        return sub

    def write(self):
        sub = self.build()
        tmp = OUT_SUB + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sub, f)
        os.replace(tmp, OUT_SUB)


def check_submission(path, tasks):
    sub = json.load(open(path))
    assert set(sub) == set(tasks), "task ids differ"
    for tid, task in tasks.items():
        assert len(sub[tid]) == len(task["test"]), tid
        for slot in sub[tid]:
            assert valid_grid(slot["attempt_1"]) and valid_grid(slot["attempt_2"]), tid
    return True


# ---------------------------------------------------------------- environment

def sh(cmd, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def driver_env():
    """Host driver libcuda first (images also carry an older compat libcuda); set before any CUDA process starts."""
    real = "/usr/local/nvidia/lib64"
    if os.path.exists(real + "/libcuda.so"):
        os.environ["LD_LIBRARY_PATH"] = real + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_PRELOAD"] = real + "/libcuda.so"


def install_server():
    wd = wheel_dir()
    t = time.time()
    if SERVER == "sglang":
        cmd = f"{sys.executable} -m pip install --no-index --find-links {wd} -r {wd}/requirements.lock"
    else:
        cmd = (f"{sys.executable} -m pip install --no-index --find-links {wd} vllm==0.23.0 flashinfer-python "
               f"flashinfer-jit-cache")
    code, out = sh(cmd, timeout=2400)
    log(f"offline install of {SERVER} from {wd}: rc={code} in {time.time() - t:.0f}s")
    if code != 0:
        print(out[-4000:])
        raise RuntimeError("pip install failed")
    mod = "sglang" if SERVER == "sglang" else "vllm"
    code, out = sh(f"{sys.executable} -c 'import torch, {mod}; print(torch.__version__, torch.version.cuda, "
                   f"torch.cuda.is_available(), torch.cuda.device_count(), {mod}.__version__)'", timeout=600)
    log("torch/server:", out.strip().splitlines()[-1] if out.strip() else out)


def file_hashes(paths, workers=16):
    def h(p):
        s256, s1 = hashlib.sha256(), hashlib.sha1()
        size = os.path.getsize(p)
        s1.update(f"blob {size}\0".encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                s256.update(b)
                s1.update(b)
        return p, {"size": size, "sha256": s256.hexdigest(), "git_sha1": s1.hexdigest()}
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(h, paths))


def preload(mdir, workers=16):
    files = [os.path.join(r, f) for r, _, fs in os.walk(mdir) for f in fs]

    def rd(p):
        with open(p, "rb") as f:
            while f.read(1 << 26):
                pass
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(rd, files))
    log(f"preloaded {len(files)} model files in {time.time() - t:.0f}s")


def launch_server(mdir):
    env = os.environ.copy()
    env.update(PYTORCH_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    if SERVER == "sglang":
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", mdir, "--served-model-name", SERVED,
               "--host", "127.0.0.1", "--port", str(PORT), "--tp-size", "4", "--kv-cache-dtype", "fp8_e4m3",
               "--context-length", str(MAX_MODEL_LEN), "--mem-fraction-static", os.environ.get("ARC_SGL_MEM", "0.85"),
               "--max-running-requests", str(MAX_SEQS), "--trust-remote-code",
               "--attention-backend", os.environ.get("ARC_SGL_ATTN", "triton"), "--log-level", "info"]
        if os.environ.get("ARC_RADIX", "1") != "1":
            cmd.append("--disable-radix-cache")
    else:
        attn = "TRITON_ATTN"  # FlashInfer fp8-KV kernels need SM90+; L4 is SM89
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", mdir,
               "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", str(PORT),
               "--tensor-parallel-size", "4", "--gpu-memory-utilization", str(GPU_UTIL),
               "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_SEQS),
               "--kv-cache-dtype", "fp8", "--attention-backend", attn, "--enable-prefix-caching",
               "--trust-remote-code", "--no-async-scheduling"]
        if MTP > 0:
            cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": MTP})]
        env.update(VLLM_FORCE_ATTN_BACKEND=attn, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1")
    log("server cmd:", " ".join(cmd))
    path = os.path.join(LOG_DIR, f"{SERVER}_server.log")
    fh = open(path, "a")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=env)
    return proc, path


def kill_server():
    sh("pkill -9 -f 'vllm.entrypoints' ; pkill -9 -f 'sglang.launch_server' ; pkill -9 -f 'sglang::' ; sleep 5",
       timeout=60)


def server_log_tail(path, n=60):
    try:
        return "".join(open(path, errors="ignore").readlines()[-n:])
    except Exception:
        return ""


def wait_ready(proc, path, timeout):
    import urllib.request
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            log(f"server exited rc={proc.returncode}\n" + server_log_tail(path))
            return False
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    log(f"server ready in {time.time() - t:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    log("server not ready in time\n" + server_log_tail(path))
    return False


def server_capacity(path):
    """(kv_tokens, max_running) from the server log."""
    txt = open(path, errors="ignore").read()
    if SERVER == "sglang":
        kv = re.findall(r"max_total_num_tokens\W{1,3}(\d+)", txt)
        mr = re.findall(r"max_running_requests\W{1,3}(\d+)", txt)
        out = (int(kv[-1]) if kv else None, int(mr[-1]) if mr else None)
    else:
        m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
        out = (int(m[-1].replace(",", "")) if m else None, None)
    log("server capacity (kv tokens, max running):", out)
    return out


# ---------------------------------------------------------------- client

class Client:
    def __init__(self):
        import httpx
        self.last_prompt_tokens = None
        self.http = httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(None, connect=30.0),
                                      limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))

    async def count(self, prompt):
        try:
            r = await self.http.post("/tokenize", json={"model": SERVED, "prompt": prompt})
            return int(r.json()["count"])
        except Exception:
            return int(len(prompt) / 2.5)

    async def stream(self, prompt, max_tokens, deadline, on_tokens, stop=None, info=None):
        """Streamed completion. Returns (text, completion_tokens, finish_reason); finish 'deadline' if cut off.

        Breaking out of the stream closes the connection, which makes vLLM abort the request.
        """
        body = {"model": SERVED, "prompt": prompt, "max_tokens": max_tokens, "temperature": TEMPERATURE,
                "top_p": 0.95, "top_k": 20, "stream": True, "skip_special_tokens": False,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        if stop:
            body["stop"] = stop
        text, ntok, finish = [], 0, None
        async with self.http.stream("POST", "/v1/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                d = json.loads(data)
                u = d.get("usage")
                got_text = False
                for c in d.get("choices") or []:
                    text.append(c.get("text") or "")
                    got_text = got_text or bool(c.get("text"))
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
                if u and u.get("completion_tokens") is not None:
                    self.last_prompt_tokens = u.get("prompt_tokens") or self.last_prompt_tokens
                    if info is not None and u.get("prompt_tokens"):
                        info["prompt_tokens"] = u["prompt_tokens"]
                    on_tokens(u["completion_tokens"] - ntok)
                    ntok = u["completion_tokens"]
                elif got_text:  # server without continuous usage stats: ~1 token per chunk until usage arrives
                    on_tokens(1)
                    ntok += 1
                if time.time() > deadline:
                    finish = "deadline"
                    break
        return "".join(text), ntok, finish or "stop"


# ---------------------------------------------------------------- scheduler

class Scheduler:
    def __init__(self, tasks, sub, client, force_at, hard_stop, concurrency, attempts=2, tag="run"):
        self.tasks, self.sub, self.client = tasks, sub, client
        self.force_at, self.hard_stop = force_at, hard_stop
        self.concurrency, self.attempts = concurrency, attempts
        self.pending1 = []
        self.pending2 = []
        self.inflight = {}  # job -> [budget, used]
        self.tokens = 0
        self.rate_samples = []  # (time, total tokens)
        self.gen_start = None
        self.records = []
        self.rec_path = os.path.join(LOG_DIR, f"attempts_{tag}.jsonl")
        self.launch_margin = 120  # no new attempts in the last 2 min before forcing
        self.retries = {}
        self.server_down = lambda: False  # set by main(): True while the vLLM server is dead or restarting

    def on_tokens(self, n):
        self.tokens += n

    def rate(self):
        now = time.time()
        self.rate_samples.append((now, self.tokens))
        self.rate_samples = [s for s in self.rate_samples if now - s[0] < 1800] or [(now, self.tokens)]
        t_old, k_old = self.rate_samples[0]
        if self.gen_start is None or now - self.gen_start < 900 or now - t_old < 300:
            return RATE_PRIOR
        return max(50.0, (self.tokens - k_old) / (now - t_old))

    def capacity_left(self):
        cap = self.rate() * max(0.0, self.force_at - time.time())
        return cap - sum(max(0, b - u) for b, u in self.inflight.values())

    def next_job(self):
        """(job, budget) or None. Attempt-1 jobs first; attempt 2 only with spare capacity."""
        if self.pending1:
            job = self.pending1.pop(0)
            share = self.capacity_left() / (len(self.pending1) + 1)
            return job, int(min(B_MAX1, max(B_MIN1, share)))
        while self.pending2:
            job = self.pending2.pop(0)
            share = self.capacity_left() / (len(self.pending2) + 1)
            if share < B_MIN2:
                continue
            return job, int(min(B_MAX2, share))
        return None

    async def attempt(self, job, budget):
        if AGENT == "tools":
            return await self.attempt_tools(job, budget)
        return await self.attempt_plain(job, budget)

    async def attempt_tools(self, job, budget):
        """One E012 tool-agent attempt (agent_qwen.solve + LocalExecutor) with deadline forcing.

        DeadlineLLM streams each completion; past force_at it cuts the stream and reports the attempt's budget as
        used, so the agent goes straight to its forced submit_answer (then the session `answer`). The forced call
        (stop=["</parameter>"]) may run until the hard stop."""
        import dataclasses
        from arc_harness import agent_qwen as AQ
        from arc_harness.executor import local_factory
        tid, p, a = job
        task = self.tasks[tid]
        cap = int(min(TOOLS_CAP, budget))
        cfg = AQ.AgentConfig(tier=TOOLS_TIER, max_total_output=cap, budget_soft=min(TOOLS_SOFT, cap),
                             budget_notes=0.01, budget_warn=0.75, verify_submit=False, max_turns=60,
                             chunk_tokens=16384, force_submit_tokens=256, tool_timeout=TOOL_TIMEOUT,
                             temperature=TEMPERATURE, top_p=0.95, top_k=20)
        sch, client = self, self.client
        stats = {"tokens": 0, "requests": 0, "deadline_cuts": 0}

        class DeadlineLLM:
            async def complete(self, prompt, max_tokens, stop=None):
                info = {}
                stats["requests"] += 1
                if stop is None and time.time() >= sch.force_at:
                    stats["deadline_cuts"] += 1
                    return AQ.Completion("", cap, 0, "length")
                dl = sch.hard_stop if stop else sch.force_at

                def cb(n):
                    stats["tokens"] += n
                    sch.on_tokens(n)
                    if job in sch.inflight:
                        sch.inflight[job][1] = stats["tokens"]
                text, ntok, finish = await client.stream(prompt, max_tokens, dl, cb, stop=stop, info=info)
                if finish == "deadline":
                    stats["deadline_cuts"] += 1
                    return AQ.Completion(text, ntok + cap, info.get("prompt_tokens", 0), "length")
                return AQ.Completion(text, ntok, info.get("prompt_tokens", 0),
                                     "length" if finish == "length" else "stop")

        self.inflight[job] = [cap, 0]
        rec = {"task_id": tid, "pair": p, "attempt": a, "budget": cap, "agent": "tools"}
        t0 = time.time()
        try:
            transcript, res = await AQ.solve(DeadlineLLM(), local_factory(timeout=TOOL_TIMEOUT), task["train"],
                                             task["test"][p]["input"], cfg)
        except Exception as exc:
            transcript, res = "", {"pred": None, "error": f"{type(exc).__name__}: {exc}"[:400]}
        finally:
            self.inflight.pop(job, None)
        pred = res.get("pred")
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        rec.update(tokens=stats["tokens"], requests=stats["requests"], deadline_cuts=stats["deadline_cuts"],
                   answered=pred is not None and valid_grid(pred), pred=pred,
                   forced=res.get("answer_source") in ("forced", "session") or stats["deadline_cuts"] > 0,
                   force_reason=("deadline" if stats["deadline_cuts"] else res.get("stop_reason")),
                   elapsed_s=round(time.time() - t0),
                   **{k: res.get(k) for k in ("answer_source", "stop_reason", "turns", "tool_calls", "tool_errors",
                                              "parse_failures", "eos_continues", "chunks", "seconds", "turn_log")})
        if res.get("error"):
            rec["error"] = res["error"]
        try:
            with open(os.path.join(LOG_DIR, f"trace_{tid}_{p}_{a}.txt"), "w") as f:
                f.write(transcript)
        except Exception:
            pass
        return rec

    async def attempt_plain(self, job, budget):
        tid, p, a = job
        task = self.tasks[tid]
        raw = render_prompt(build_prompt(task["train"], task["test"][p]["input"]))
        ptok = await self.client.count(raw)
        rec = {"task_id": tid, "pair": p, "attempt": a, "budget": budget, "prompt_tokens": ptok}
        self.inflight[job] = [budget, 0]
        gen, used, conts, streak, forced, force_reason, after_eos, chunks = "", 0, 0, 0, False, None, False, []
        t0 = time.time()
        try:
            while True:
                room = MAX_MODEL_LEN - ptok - used - 64
                if forced:
                    mt, dl = min(FINAL_TOKENS, room), self.hard_stop
                else:
                    mt, dl = min(budget - used, room - FINAL_TOKENS), self.force_at
                if mt <= 0 and not forced:
                    forced, force_reason = True, "budget"
                    gen += FORCE_TEXT
                    continue
                if mt <= 0:
                    break
                text, ntok, finish = await self.client.stream(raw + gen, mt, dl, self.on_tokens)
                gen += text
                used += ntok
                self.inflight[job][1] = used
                chunks.append([ntok, finish])
                if after_eos:
                    streak = streak + 1 if ntok < LOW_YIELD else 0
                if forced:
                    break
                if answer_section(gen) is not None:
                    if finish == "length" and used < budget and room - ntok > 64:
                        continue
                    break
                if finish == "deadline" or used >= budget or finish == "length":
                    forced, force_reason = True, ("deadline" if finish == "deadline" else "budget")
                    gen += FORCE_TEXT
                    continue
                # EOS mid-thought: continue, force after two low-yield continuations in a row
                if streak >= 2:
                    forced, force_reason = True, "low_yield_eos"
                    gen += FORCE_TEXT
                    continue
                conts += 1
                after_eos = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            self.inflight.pop(job, None)
        content = answer_section(gen)
        pred = parse(content) if content is not None else None
        rec.update(tokens=used, continuations=conts, forced=forced, force_reason=force_reason, chunks=chunks,
                   elapsed_s=round(time.time() - t0), answered=pred is not None and valid_grid(pred), pred=pred,
                   tail=(content or gen[-300:])[-300:])
        with open(os.path.join(LOG_DIR, f"trace_{tid}_{p}_{a}.txt"), "w") as f:
            f.write(gen)
        return rec

    async def worker(self, wid):
        while time.time() < self.force_at - self.launch_margin:
            nj = self.next_job()
            if nj is None:
                return
            job, budget = nj
            rec = await self.attempt(job, budget)
            if "error" in rec and not rec["answered"] and self.server_down():
                # server crashed: requeue the job (at most twice) and wait for the watchdog to restart it
                self.retries[job] = self.retries.get(job, 0) + 1
                if self.retries[job] <= 2:
                    (self.pending1 if job[2] == 0 else self.pending2).insert(0, job)
                    log(f"server down; requeued {job}, waiting")
                    while self.server_down() and time.time() < self.force_at - self.launch_margin:
                        await asyncio.sleep(15)
                    continue
            self.records.append(rec)
            with open(self.rec_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            tid, p, a = job
            if rec["answered"]:
                self.sub.answers.setdefault((tid, p), {})[a] = rec["pred"]
                self.sub.write()
            if a == 0 and self.attempts > 1:
                self.pending2.append((tid, p, 1))
            log(f"done {tid}.{p}#{a} answered={rec['answered']} tok={rec['tokens']} budget={budget} "
                f"forced={rec['forced']}/{rec['force_reason']} "
                f"{rec['elapsed_s']}s {rec.get('error', '')}")

    async def monitor(self):
        last_t, last_k = time.time(), 0
        while True:
            await asyncio.sleep(120)
            now = time.time()
            r = (self.tokens - last_k) / (now - last_t)
            last_t, last_k = now, self.tokens
            log(f"progress: {len(self.records)} attempts done, in flight {len(self.inflight)}, pending1 "
                f"{len(self.pending1)}, pending2 {len(self.pending2)}, tokens {self.tokens}, window {r:.0f} tok/s, "
                f"rate est {self.rate():.0f} tok/s, avg {self.tokens / max(1, now - self.gen_start):.0f} tok/s")

    async def run(self, jobs1):
        self.pending1 = list(jobs1)
        self.gen_start = time.time()
        mon = asyncio.create_task(self.monitor())
        try:
            await asyncio.wait_for(asyncio.gather(*(self.worker(i) for i in range(self.concurrency))),
                                   timeout=max(60, self.hard_stop - time.time()))
        except asyncio.TimeoutError:
            log("hard stop reached; cancelling in-flight attempts")
        mon.cancel()
        el = time.time() - self.gen_start
        log(f"scheduler finished: {len(self.records)} attempts, {self.tokens} tokens in {el / 60:.1f} min "
            f"({self.tokens / max(1, el):.0f} tok/s)")


def order_jobs(tasks):
    """Attempt-1 jobs, shortest prompts first (approximated by character length)."""
    jobs = [(tid, p, 0) for tid, t in tasks.items() for p in range(len(t["test"]))]
    size = {tid: len(json.dumps(t["train"])) + len(json.dumps(t["test"])) for tid, t in tasks.items()}
    return sorted(jobs, key=lambda j: (size[j[0]], j[0], j[1]))


# ---------------------------------------------------------------- benchmark (test runs only)

async def bench_decode(client, prefixes, n_seqs, gen_tokens, tasks, max_chars=None, n_unique=None):
    """Decode throughput at long context: real ARC prompts + real Qwen trace prefixes, n_seqs in parallel.

    n_unique: number of distinct prompts (sequence i uses prompt i % n_unique), so repeats can hit a prefix cache.
    """
    total = {"tok": 0}
    prompts = []
    for i in range(n_seqs):
        pr = prefixes[i % (n_unique or len(prefixes))]
        t = tasks[pr["task_id"]]
        prompts.append(render_prompt(build_prompt(t["train"], t["test"][pr["pair"]]["input"]))
                       + pr["text"][:max_chars])
    ctx = [await client.count(p) for p in prompts]
    log(f"bench: {n_seqs} seqs, context tokens mean {sum(ctx) / len(ctx):.0f} min {min(ctx)} max {max(ctx)}")
    first = {}
    t_start = time.time()

    def cb(n):
        total["tok"] += n

    async def one(i):
        def cbi(n):
            if i not in first:
                first[i] = time.time()
            cb(n)
        return await client.stream(prompts[i], gen_tokens, time.time() + 3600, cbi)

    samples = []

    async def sampler():
        while True:
            await asyncio.sleep(10)
            samples.append((time.time(), total["tok"]))
    s = asyncio.create_task(sampler())
    res = await asyncio.gather(*(one(i) for i in range(n_seqs)), return_exceptions=True)
    s.cancel()
    t_end = time.time()
    t_all_first = max(first.values()) if first else t_end
    dec = [x for x in samples if x[0] >= t_all_first]
    steady = (dec[-1][1] - dec[0][1]) / (dec[-1][0] - dec[0][0]) if len(dec) >= 2 else None
    errs = [repr(r)[:200] for r in res if isinstance(r, Exception)]
    out = {"n_seqs": n_seqs, "ctx_mean": sum(ctx) / len(ctx), "gen_tokens": gen_tokens, "total_tokens": total["tok"],
           "wall_s": t_end - t_start, "prefill_phase_s": t_all_first - t_start, "steady_decode_tok_s": steady,
           "overall_tok_s": total["tok"] / (t_end - t_start), "errors": errs}
    log("bench result:", json.dumps(out))
    return out


def score(sub, solutions, tasks):
    tot, pairs = 0.0, 0
    for tid, t in tasks.items():
        n = len(t["test"])
        ok = 0
        for p in range(n):
            s = sub[tid][p]
            ok += solutions[tid][p] in (s["attempt_1"], s["attempt_2"])
        tot += ok / n
        pairs += n
    return tot, len(tasks)


# ---------------------------------------------------------------- main

def setup_harness():
    """Make the bundled E012 harness importable (written by the notebook to /kaggle/working/pkg/evals)."""
    sys.path.insert(0, os.path.join(PKG, "evals"))
    from arc_harness import agent_qwen  # noqa: F401  (fail early if the bundle is broken)
    log("harness:", agent_qwen.HARNESS_VERSION)


def main():
    global OUT_SUB
    os.makedirs(LOG_DIR, exist_ok=True)
    driver_env()
    rerun = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
    mode = "submit" if rerun else MODE
    log(f"mode={mode} server={SERVER} agent={AGENT} rerun={rerun} "
        f"T0={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(T0))} UTC")
    test_tasks = json.load(open(comp_file("arc-agi_test_challenges.json")))
    Submission(test_tasks).write()  # valid fallback for the competition test file, always
    log(f"fallback submission written ({len(test_tasks)} tasks)")
    if mode == "fallback":
        return
    if mode == "local":
        # Non-rerun commit/interactive run: small local test set with known solutions, self-scored, time-boxed.
        tasks = json.load(open(find_one("/kaggle/input/**/arc-agi_local_challenges.json")))
        solutions = json.load(open(find_one("/kaggle/input/**/arc-agi_local_solutions.json")))
        force_at, hard_stop = T0 + LOCAL_FORCE_MIN * 60, T0 + (LOCAL_FORCE_MIN + 8) * 60
        OUT_SUB = os.path.join(WORK, "local_submission.json")
    else:
        tasks, solutions = test_tasks, None
        force_at, hard_stop = T0 + FORCE_AT_S, T0 + HARD_STOP_S
    sub = Submission(tasks)
    sub.write()
    log(f"{mode}: {len(tasks)} tasks, {sum(len(t['test']) for t in tasks.values())} test inputs; force at "
        f"{(force_at - T0) / 3600:.2f}h, hard stop {(hard_stop - T0) / 3600:.2f}h")
    if AGENT == "tools":
        setup_harness()

    mdir = model_dir()
    log("model dir:", mdir)
    install_server()
    preload(mdir)
    proc, slog = launch_server(mdir)
    if not wait_ready(proc, slog, timeout=1800):
        raise RuntimeError("server failed to start")
    kv, max_running = server_capacity(slog)
    conc = CONCURRENCY or min(MAX_SEQS, max_running or MAX_SEQS,
                              max(8, int(kv / TOKENS_PER_SEQ_KV)) if kv else 16)
    log(f"concurrency {conc} (max_seqs {MAX_SEQS}, server max running {max_running}, kv {kv})")
    server = {"proc": proc, "down": False, "restarts": 0}

    async def watchdog(force_at):
        """Restart the server (at most twice) if it dies while there is time left to use it."""
        while True:
            await asyncio.sleep(15)
            if server["proc"].poll() is None:
                continue
            server["down"] = True
            log(f"server died (rc={server['proc'].returncode}):\n" + server_log_tail(slog, 40))
            if server["restarts"] >= 2 or force_at - time.time() < 45 * 60:
                log("not restarting the server")
                return
            server["restarts"] += 1
            kill_server()
            server["proc"], _ = launch_server(mdir)
            ok = await asyncio.to_thread(wait_ready, server["proc"], slog, 1800)
            log(f"server restart {'ok' if ok else 'FAILED'}")
            if not ok:
                return
            server["down"] = False

    async def cpu_monitor():
        while True:
            await asyncio.sleep(120)
            try:
                la = os.getloadavg()
                log(f"cpu load avg {la[0]:.1f}/{la[1]:.1f}/{la[2]:.1f} on {os.cpu_count()} cpus")
            except Exception:
                pass

    async def go():
        from concurrent.futures import ThreadPoolExecutor
        # agent_qwen runs executor calls via asyncio.to_thread: one thread per concurrent attempt (E012)
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=conc + 8))
        client = Client()
        sch = Scheduler(tasks, sub, client, force_at, hard_stop, conc, attempts=2, tag=mode)
        sch.server_down = lambda: server["down"] or server["proc"].poll() is not None
        wd = asyncio.create_task(watchdog(sch.force_at))
        cm = asyncio.create_task(cpu_monitor())
        await sch.run(order_jobs(tasks))
        wd.cancel()
        cm.cancel()
        sub.write()
        check_submission(OUT_SUB, tasks)
        recs = sch.records
        summary = {"mode": mode, "server": SERVER, "agent": AGENT, "concurrency": conc, "kv_tokens": kv,
                   "attempts": len(recs), "tokens": sch.tokens,
                   "gen_minutes": round((time.time() - sch.gen_start) / 60, 1),
                   "answered": sum(r["answered"] for r in recs), "errors": sum("error" in r for r in recs),
                   "tool_calls": sum(r.get("tool_calls") or 0 for r in recs),
                   "tool_errors": sum(r.get("tool_errors") or 0 for r in recs),
                   "parse_failures": sum(r.get("parse_failures") or 0 for r in recs),
                   "answer_sources": {k: sum(r.get("answer_source") == k for r in recs)
                                      for k in ("submit", "forced", "session", "parse", "lenient", None)}}
        if solutions:
            sc, n = score(json.load(open(OUT_SUB)), solutions, tasks)
            summary["score"] = f"{sc:.2f}/{n} tasks"
            summary["per_pair"] = [{"task": r["task_id"], "pair": r["pair"], "attempt": r["attempt"],
                                    "correct": r["pred"] == solutions[r["task_id"]][r["pair"]],
                                    "tokens": r["tokens"], "turns": r.get("turns"), "tool_calls": r.get("tool_calls"),
                                    "tool_errors": r.get("tool_errors"), "source": r.get("answer_source"),
                                    "stop": r.get("stop_reason"), "s": r["elapsed_s"], "error": r.get("error")}
                                   for r in recs]
        json.dump(summary, open(os.path.join(WORK, f"summary_{mode}.json"), "w"), indent=1)
        log("SUMMARY", json.dumps(summary))
    asyncio.run(go())
    log("server log tail:\n" + server_log_tail(slog, 10))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # keep whatever submission.json exists
        import traceback
        traceback.print_exc()
        log(f"FATAL {type(exc).__name__}: {exc}")
    finally:
        try:
            tasks_path = comp_file("arc-agi_test_challenges.json")
            check_submission(os.path.join(WORK, "submission.json"), json.load(open(tasks_path)))
            log("submission.json valid")
        except Exception as exc:
            log(f"submission check failed ({exc}); rewriting fallback")
            OUT_SUB = os.path.join(WORK, "submission.json")
            Submission(json.load(open(comp_file("arc-agi_test_challenges.json")))).write()
        os._exit(0)
