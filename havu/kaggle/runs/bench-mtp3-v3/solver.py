"""ARC Prize 2026 (ARC-AGI-2) offline solver: Qwen3.8-27B-FP8 on vLLM (TP=4, 4xL4), no tools.

One self-contained file; kaggle/build_notebook.py embeds it in the Kaggle notebook. Modes (env ARC_MODE):

- submit:   full run on the competition test file (used automatically on the competition rerun).
- fallback: write a valid fallback submission.json and exit (used for the commit run before submitting).
- bench:    test run: provenance hashes, server start, long-context decode benchmark, short end-to-end run on
            evaluation tasks with a near deadline, scoring against the evaluation solutions.

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
    p = (find_one("/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3/**/requirements.lock")
         or find_one("/kaggle/input/**/arc3-vllm-h100-wheelhouse-v3/**/requirements.lock"))
    if not p:
        raise FileNotFoundError("wheelhouse")
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


def install_vllm():
    try:
        import vllm  # noqa: F401
        log("vllm already installed", vllm.__version__)
        return
    except Exception:
        pass
    wd = wheel_dir()
    t = time.time()
    code, out = sh(f"{sys.executable} -m pip install --no-index --find-links {wd} -r {wd}/requirements.lock",
                   timeout=1800)
    log(f"pip install rc={code} in {time.time() - t:.0f}s")
    if code != 0:
        print(out[-3000:])
        raise RuntimeError("pip install failed")
    # The pinned py3.12 image ships a CPU torch 2.10.0, which satisfies the lock's torch==2.10.0, so pip keeps it.
    # Force the wheelhouse's CUDA build (its nvidia-* deps were installed by the lock above).
    code, out = sh(f"{sys.executable} -m pip install --no-index --find-links {wd} --force-reinstall --no-deps "
                   f"torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0", timeout=1800)
    log(f"torch CUDA reinstall rc={code}")
    if code != 0:
        print(out[-3000:])
        raise RuntimeError("torch reinstall failed")
    code, out = sh(f"{sys.executable} -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, "
                   f"torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)'", timeout=600)
    log("torch/vllm:", out.strip().splitlines()[-1] if out.strip() else out)


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
    attn = "TRITON_ATTN"  # FlashInfer fp8-KV kernels need SM90+; L4 is SM89
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", mdir,
           "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", str(PORT),
           "--tensor-parallel-size", "4", "--gpu-memory-utilization", str(GPU_UTIL),
           "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_SEQS),
           "--kv-cache-dtype", "fp8", "--attention-backend", attn, "--enable-prefix-caching",
           "--trust-remote-code", "--no-async-scheduling"]
    if MTP > 0:
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": MTP})]
    env = os.environ.copy()
    env.update(PYTORCH_ALLOC_CONF="expandable_segments:True", VLLM_FORCE_ATTN_BACKEND=attn,
               VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1", TOKENIZERS_PARALLELISM="false",
               VLLM_LOGGING_LEVEL="INFO")
    real = "/usr/local/nvidia/lib64"
    if os.path.exists(real + "/libcuda.so"):
        env["LD_LIBRARY_PATH"] = real + ":" + env.get("LD_LIBRARY_PATH", "")
        env["LD_PRELOAD"] = real + "/libcuda.so"
    log("server cmd:", " ".join(cmd))
    path = os.path.join(LOG_DIR, "vllm_server.log")
    fh = open(path, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=env)
    return proc, path


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


def kv_capacity(path):
    txt = open(path, errors="ignore").read()
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
    c = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", txt)
    kv = int(m[-1].replace(",", "")) if m else None
    log("KV cache tokens:", kv, "| max concurrency lines:", c[-1:] if c else None)
    return kv


# ---------------------------------------------------------------- client

class Client:
    def __init__(self):
        import httpx
        self.http = httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(None, connect=30.0),
                                      limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))

    async def count(self, prompt):
        try:
            r = await self.http.post("/tokenize", json={"model": SERVED, "prompt": prompt})
            return int(r.json()["count"])
        except Exception:
            return int(len(prompt) / 2.5)

    async def stream(self, prompt, max_tokens, deadline, on_tokens):
        """Streamed completion. Returns (text, completion_tokens, finish_reason); finish 'deadline' if cut off.

        Breaking out of the stream closes the connection, which makes vLLM abort the request.
        """
        body = {"model": SERVED, "prompt": prompt, "max_tokens": max_tokens, "temperature": TEMPERATURE,
                "top_p": 0.95, "top_k": 20, "stream": True, "skip_special_tokens": False,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
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
                if u and u.get("completion_tokens") is not None:
                    on_tokens(u["completion_tokens"] - ntok)
                    ntok = u["completion_tokens"]
                for c in d.get("choices") or []:
                    text.append(c.get("text") or "")
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
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

async def bench_decode(client, prefixes, n_seqs, gen_tokens, tasks):
    """Decode throughput at long context: real ARC prompts + real Qwen trace prefixes, n_seqs in parallel."""
    total = {"tok": 0}
    prompts = []
    for i in range(n_seqs):
        pr = prefixes[i % len(prefixes)]
        t = tasks[pr["task_id"]]
        prompts.append(render_prompt(build_prompt(t["train"], t["test"][pr["pair"]]["input"])) + pr["text"])
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
            await asyncio.sleep(30)
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

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log(f"mode={MODE} T0={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(T0))} UTC")
    rerun = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
    if MODE == "bench":
        ch = comp_file("arc-agi_evaluation_challenges.json")
    else:
        ch = comp_file("arc-agi_test_challenges.json")
    tasks = json.load(open(ch))
    log(f"challenges {ch}: {len(tasks)} tasks, {sum(len(t['test']) for t in tasks.values())} test inputs; "
        f"rerun={rerun}")
    sub = Submission(tasks if MODE != "bench" else json.load(open(comp_file("arc-agi_test_challenges.json"))))
    sub.write()
    log("fallback submission written")
    if MODE == "fallback":
        return

    mdir = model_dir()
    log("model dir:", mdir)
    install_vllm()
    if MODE == "bench":
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(mdir) for f in fs)
        wd = wheel_dir()
        wheels = sorted(glob.glob(wd + "/**/*", recursive=True))
        t = time.time()
        hashes = {"model": {os.path.relpath(p, mdir): v for p, v in file_hashes(files).items()},
                  "wheels": {os.path.relpath(p, wd): v for p, v in file_hashes([w for w in wheels if os.path.isfile(w)]).items()}}
        json.dump(hashes, open(os.path.join(WORK, "provenance_hashes.json"), "w"), indent=1)
        log(f"hashed {len(hashes['model'])} model files + {len(hashes['wheels'])} wheel files in {time.time() - t:.0f}s")
    else:
        preload(mdir)
    proc, slog = launch_server(mdir)
    if not wait_ready(proc, slog, timeout=1800):
        raise RuntimeError("server failed to start")
    kv = kv_capacity(slog)
    conc = CONCURRENCY or (min(MAX_SEQS, max(16, int((kv or 0) / TOKENS_PER_SEQ_KV))) if kv else 32)
    log(f"concurrency {conc} (max_num_seqs {MAX_SEQS}, MTP {MTP})")

    async def go():
        client = Client()
        if MODE == "bench":
            sol = json.load(open(comp_file("arc-agi_evaluation_solutions.json")))
            prefixes = json.load(open(find_one("/kaggle/input/**/bench_prefixes.json")))
            results = {"kv_tokens": kv, "concurrency": conc, "mtp": MTP}
            results["decode_long"] = await bench_decode(client, prefixes, int(os.environ.get("ARC_BENCH_SEQS", conc)),
                                                        int(os.environ.get("ARC_BENCH_GEN", 1500)), tasks)
            json.dump(results, open(os.path.join(WORK, "bench_results.json"), "w"), indent=1)
            # end-to-end: real scheduler on eval tasks with a deadline a few minutes away
            e2e_min = float(os.environ.get("ARC_E2E_MIN", 15))
            ids = sorted(tasks)[:int(os.environ.get("ARC_E2E_TASKS", 48))]
            etasks = {t: tasks[t] for t in ids}
            esub = Submission(etasks)
            global OUT_SUB
            keep = OUT_SUB
            OUT_SUB = os.path.join(WORK, "e2e_submission.json")
            sch = Scheduler(etasks, esub, client, time.time() + e2e_min * 60, time.time() + (e2e_min + 8) * 60,
                            conc, attempts=1, tag="e2e")
            await sch.run(order_jobs(etasks))
            esub.write()
            check_submission(OUT_SUB, etasks)
            sc, n = score(json.load(open(OUT_SUB)), sol, etasks)
            OUT_SUB = keep
            results["e2e"] = {"tasks": n, "score": sc, "attempts": len(sch.records), "tokens": sch.tokens,
                              "answered": sum(r["answered"] for r in sch.records),
                              "forced": sum(bool(r["forced"]) for r in sch.records),
                              "errors": sum("error" in r for r in sch.records)}
            log("e2e:", json.dumps(results["e2e"]))
            json.dump(results, open(os.path.join(WORK, "bench_results.json"), "w"), indent=1)
        else:
            sch = Scheduler(tasks, sub, client, T0 + FORCE_AT_S, T0 + HARD_STOP_S, conc, attempts=2, tag="submit")
            await sch.run(order_jobs(tasks))
            sub.write()
    asyncio.run(go())
    log("server log tail:\n" + server_log_tail(slog, 15))


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
            check_submission(OUT_SUB, json.load(open(tasks_path)))
            log("submission.json valid")
        except Exception as exc:
            log(f"submission check failed ({exc}); rewriting fallback")
            Submission(json.load(open(comp_file("arc-agi_test_challenges.json")))).write()
        os._exit(0)
