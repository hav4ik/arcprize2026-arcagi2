# Review: `chankhavu/c2-distill-vllm018-watchdog`, and what our v2 should take from it

Reviewed 2026-10-11. Source: `kaggle/reviews/src_c2_distill_watchdog/` (pulled read-only, version of 2026-04-12).
Compared against `kaggle/src/solver.py`, `kaggle/build_notebook.py` (v2), `evals/arc_harness/{agent_qwen,executor}.py`.

**Correction up front:** this notebook is not an ARC harness. It is the user's **AIMO3** (math olympiad) notebook:
`competition_sources: ai-mathematical-olympiad-progress-prize-3`, machine `NvidiaH100` (1 GPU, TP=1), 5 h limit,
`kaggle_evaluation.aimo_3_inference_server` gateway (one problem per `predict()` call). There are no L4 workarounds
in it. The ideas still carry over (rerun switch, watchdog, preload, deadlines, streaming, voting). The newest
harness of the same family, `chankhavu/inference-baseline-dflash` (AIMO proof-pilot, vLLM 0.23, June 2026), fixes
several of this notebook's gaps. Points taken from it are marked **[dflash]**.

## 1. Design summary of the c2 notebook

| Area | What it does |
|---|---|
| Install | `uv pip install --no-index --find-links .../fork-of-vllm-0-18-0-py312-d168-flashinfer066/wheels vllm flashinfer-jit-cache flashinfer-python`. Uninstalls keras/tf/sklearn/matplotlib first. Prints `WALL TIME ELAPSED` after every setup cell (`GLOBAL_START_TIME` set in cell 0). |
| Env | `CUDA_VISIBLE_DEVICES=0`, `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, `VLLM_USE_FLASHINFER_MOE_FP8=1`, `TRANSFORMERS_NO_TF/FLAX=1`. |
| Model | `chankhavu/bogoconic1-c2-softcpy-v3-fp8`, served as `chankhavu/c2-legaldistill-30b-a3b-fp8`: a distilled Nemotron-Cascade-2-style 30B-A3B **hybrid Mamba MoE**, FP8. Weights preloaded into the page cache (16 threads, 1 GB reads). |
| vLLM 0.18 flags | `vllm serve <path> --tensor-parallel-size 1 --max-num-seqs 64 --gpu-memory-utilization 0.9 --dtype auto --max-model-len 131072 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --trust-remote-code --mamba-ssm-cache-dtype float16 --mamba-cache-mode all --mamba-block-size 512 --enable-chunked-prefill --enable-prefix-caching --async-scheduling --enable-auto-tool-choice --tool-call-parser qwen3_coder`. The verifier variant uses `--mamba-ssm-cache-dtype float32` and a 262144 context. |
| Readiness | Poll `openai_client.models.list()` once a second (180 s cap). Before every problem, `wait_for_vllm()` checks that `/v1/models` lists the served name (comment: `/health` returns 200 before weights are loaded), and skips the problem (answer 0) if the server is not ready within 120 s. |
| Watchdog | A separate Python **process** (`start_new_session=True`, its own log) polls every 30 s. It reaps its own zombies, then scans `/proc/*/stat` for live processes whose `comm` starts with `VLLM::`, falling back to cmdline `vllm` / `vllm.entrypoints`, and skips state `Z`/`X`. If it finds none, it SIGKILLs the process group of every leftover (including zombies), waits 2 s, reaps, relaunches the same `cmd` and appends `=== vllm_watchdog restart #N ===` to the server log. The HTTP probe only feeds the log, because restarting on HTTP failure would loop during the ~100 s startup. Restarts are unlimited. The lessons are in its comments: a PID file plus `os.kill(pid, 0)` reports a zombie as alive, and `setproctitle("VLLM::APIServer_0")` overwrites `/proc/PID/cmdline`, so pattern-matching `vllm.entrypoints` misses live vLLM processes. |
| In-flight recovery | None. An API error ends that attempt with no answer, and the next problem waits up to 120 s for the server. |
| Rerun switch | `if os.getenv('KAGGLE_IS_COMPETITION_RERUN'): inference_server.serve()`, else `run_local_gateway(('/kaggle/input/.../test.csv',))`. The commit run drives the same `predict()` through the real local gateway on the competition's public `test.csv` (3 sample problems). It is **not scored**: `reference_first3.csv` is written but never compared. Its duration is bounded only by the per-problem budget (below), roughly 3 × 900 s plus startup. |
| Time budget | `notebook_limit = 17600` against the 18000 s hard limit. Per problem: `budget = clamp(time_left − (problems_left − 1)·300, 300, 900)` s. `predict()` also has a panic path: return a random answer if within 30 s of the hard limit, and run the solve in a thread with `future.result(timeout=remaining)`. **That panic path does not work:** the `with ThreadPoolExecutor(...)` block's exit calls `shutdown(wait=True)` and blocks until the solve thread finishes. Do not copy this pattern. |
| Sampling | Chat API, `chat_template_kwargs={"enable_thinking": True}`, `skip_special_tokens=False`, temperature 1.0, top_p 0.95, streaming with `logprobs=True, top_logprobs=5`. 16 parallel attempts, up to 128 tool turns, Python tool with a 30 s timeout, tool output truncated to 1000 chars (600 in dflash). |
| Prompt | System: "You are a helpful assistant and a mathematical problem solver. Solve the given problem step by step using rigorous reasoning." User: problem + "The answer is a non-negative integer between 0 and 99999 … Put your final answer in \boxed{}." |
| Selection | Early stop when 4 attempts agree. Otherwise the answer with the largest Σ 1/mean-entropy wins, where entropy is per token over the top-5 logprobs. Fallback answer is 0. The `c2-distill-verifier` variant replaces this with pairwise LLM grading of answer buckets (`MatchupTracker`, ratings 1–10, temperature 0). |
| Logging | `vllm_server.log`, `vllm_watchdog.log` and `sandbox_server.log` in `/kaggle/working`, plus per-problem DataFrames (attempt, tokens, python calls/errors, entropy, answer) and a vote table. |

Our v2 is already ahead of it in several places:
- the local set is scored;
- `submission.json` is written atomically from the start and rewritten after every attempt;
- capacity is token-based;
- the final `finally` re-validates the submission and ends with `os._exit(0)`;
- jobs are requeued after a server crash.

The gaps are in hang and stall handling, process cleanup, a global kill switch, and the rerun detection.

## 2. Recommendations for v2 (prioritised)

**MUST** = small change; do it before the first submission today. **SHOULD** = soon after. **LATER** = needs a GPU
bench or an eval.

### MUST-1. Second rerun signal: the content of the test file
**Why:** if `KAGGLE_IS_COMPETITION_RERUN` were ever missing in the rerun, v2 would run `local` mode and submit the
fallback for the hidden set, which scores 0 with no error. The public `arc-agi_test_challenges.json` holds 240
**training** task IDs: 259 test inputs, 240 of them in `arc-agi_training_challenges.json`, 0 in eval, sorted-ID
sha256 prefix `4cab35dea7bde6ee`. The hidden set will not match.
**How** (notebook mode cell and `solver.main`):
```python
import hashlib, json, glob
def hidden_test_present():
    p = (glob.glob('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json')
         + glob.glob('/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json'))[0]
    ids = sorted(json.load(open(p)))
    return hashlib.sha256(','.join(ids).encode()).hexdigest()[:16] != '4cab35dea7bde6ee'
rerun = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN')) or hidden_test_present()
```
Log both signals. If they disagree, treat the run as a rerun.

### MUST-2. Finite stall timeout on the streaming client, keeping the partial text
**Why:** `Client` uses `httpx.Timeout(None, connect=30.0)`, so the read timeout is infinite. The deadline check in
`Client.stream` only runs when a line arrives. If TP=4 hangs (an NCCL hang, or a dead worker under a live
APIServer), every stream blocks until `asyncio.wait_for` at the hard stop. Tool attempts then never reach their
forced submit, and the run loses up to 11 h. dflash's comment: *"A FINITE read timeout is essential: with streaming,
`read` is the max SILENCE BETWEEN TOKENS."* It uses 300 s. Ours must be larger because the first token can wait
for prefill: the bench measured 474 s for 16 × 40k prompts.
**How:**
```python
STALL_S = envf("ARC_STALL_S", 1200.0)
self.http = httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(connect=30.0, read=STALL_S, write=60.0, pool=None), ...)
# in stream(): wrap the `async with` block
try:
    ...existing loop...
except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
    log(f"stream error {type(exc).__name__}; keeping {ntok} tokens")
    return "".join(text), ntok, "error"
```
The point, also from dflash, is to keep the text decoded this turn instead of losing it in the exception. Then:
- `attempt_plain` treats `finish == "error"` like `"deadline"` and forces the final answer if the server is up;
- `DeadlineLLM` maps it to `Completion(text, ntok, ..., "length")` and lets the agent continue in place (see SHOULD-1).

### MUST-3. Kill the whole server process tree before a restart
**Why:** `kill_server()` runs `pkill -9 -f 'vllm.entrypoints'`. After `setproctitle`, the vLLM APIServer, EngineCore
and the four `VLLM::Worker_TP*` processes no longer have that string in their cmdline (the c2 watchdog's main
lesson). A restart would then start next to orphaned TP workers that still hold GPU memory, and fail with OOM.
**How:** the server is started with `start_new_session=True`, so its process group equals `proc.pid`:
```python
def kill_server(proc=None):
    if proc is not None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass
    sh("pkill -9 -f 'VLLM::' ; pkill -9 -f 'vllm.entrypoints' ; pkill -9 -f 'sglang' ; sleep 3", timeout=60)
    for _ in range(60):  # wait until the GPUs are actually free
        _, out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=30)
        if all(int(x) < 1000 for x in out.split() if x.strip().isdigit()): break
        time.sleep(5)
```
Call `kill_server(server["proc"])` in `watchdog()`.

### MUST-4. Wall-clock guard in the notebook cell, outside the solver process
**Why:** if the notebook exceeds the 12 h limit, the submission errors even when `submission.json` is valid.
`HARD_STOP_S` is only enforced inside `Scheduler.run`. A hang in `install_server` (2400 s timeout), `wait_ready`
(1800 s, twice with restarts), or `asyncio.run` teardown (the default executor joins threads for up to 300 s) is not
covered. The c2 version (`PANIC_TIME`) is broken by the `with ThreadPoolExecutor` exit, so build ours as a separate
watcher that kills the process group.
**How** (v2 run cell):
```python
import subprocess, sys, threading, os, signal, time
T0 = float(os.environ['ARC_T0']); KILL_AT = T0 + float(os.environ.get('ARC_KILL_AT_S', 11.8 * 3600))
p = subprocess.Popen([sys.executable, '-u', '/kaggle/working/solver.py'], stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, cwd='/kaggle/working', start_new_session=True)
def guard():
    while p.poll() is None and time.time() < KILL_AT: time.sleep(10)
    if p.poll() is None:
        print('GLOBAL GUARD: killing solver', flush=True)
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
        os.system("pkill -9 -f 'VLLM::' ; pkill -9 -f 'vllm.entrypoints' ; pkill -9 -f sglang")
threading.Thread(target=guard, daemon=True).start()
for line in p.stdout: print(line, end='', flush=True)
print('solver exit code', p.wait())
```
The last cell should then **validate and, if needed, rewrite the fallback without importing the solver**: same
checks as `check_submission`, and copy each test input into both attempts on failure. Today it only loads the file
and prints counts.

### MUST-5. Harvest the session `answer` when an attempt dies on an exception
**Why:** in `agent_qwen.solve`, the `submit_answer(session, "answer")` fallback sits inside `try`. Any LLM or stream
exception jumps to `except`, which closes the session without reading `answer`, a variable the prompt tells the
model to keep updated. Server-crash and stall attempts then lose a usable guess.
**How:**
```python
    except Exception as exc:
        error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
        if pred is None and session is not None:
            try:
                out, grid = await _call(submit_answer, session, "answer")
                if grid is not None: pred, source = grid, "session"
            except Exception: pass
    finally: ...
```
`Scheduler.worker` already requeues errored attempts when the server is down, so a later attempt can still
overwrite this answer.

### MUST-6. Make the commit run a real gate for the rerun
The c2 commit run is only a smoke test. Ours is scored, which is better. Three changes make it a reliable gate:
- **Time-box from server-ready, not from T0.** With `LOCAL_FORCE_MIN = 40` from T0, a slow start (SGLang compile;
  vLLM took 14 min end to end) leaves almost no generation time and a misleading score of 0. Use
  `force_at = max(T0 + 40 min, server_ready + 20 min)`, capped at `T0 + 60 min`.
- **Record phase timings** (`install_s`, `preload_s`, `server_ready_s`, `first_token_s`, `kv_tokens`, `concurrency`)
  in `summary_local.json`. They are the only startup measurement for the rerun. Rerun logs are not visible to us,
  so nothing may depend on reading them.
- **Submit only if** `summary_local.json` shows the server ready, ≥ 90 % of attempts answered, 0 `error` records, and
  a score > 0. If `ARC_SERVER=sglang`, also require that the server actually started (see MUST-7).

### MUST-7. Server choice for today: the SGLang path does not start on L4 yet
`kaggle/runs/sglang-bench-v1` shows SGLang 0.5.20 failing at the first forward:
`/usr/local/cuda/bin/nvcc: not found` while JIT-compiling `sglang/kernels/jit` for sm_89. Both radix on and off.
The wheel source already contains `nvidia-cuda-nvcc`. dflash solves the same problem by pointing CUDA at the pip
toolkit:
```python
cu = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13"   # verify with: find / -name nvcc -path '*nvidia*'
os.environ.update(CUDA_HOME=cu, CUDA_PATH=cu, PATH=f"{cu}/bin:" + os.environ["PATH"],
                  LD_LIBRARY_PATH=f"{cu}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""), TORCH_CUDA_ARCH_LIST="8.9")
```
Set this in `launch_server`'s `env` when `SERVER == "sglang"`. Until a commit run shows SGLang serving, build the
first submission with `ARC_SERVER=vllm`. Its 0.23.0 source is also untested on L4 here, so the MUST-6 gate applies
to both servers.

### SHOULD-1. Recover in-flight work after a restart instead of discarding it
The c2 notebook recovers nothing. Ours requeues the job from scratch. Completions have no side effects, and our
`LocalExecutor` is a local child process that survives a server restart. So `DeadlineLLM.complete` (and the plain
loop) can wait for the server and repeat the same request. The transcript and the Python session are both kept:
```python
async def complete(self, prompt, max_tokens, stop=None):
    while True:
        text, ntok, finish = await client.stream(...)
        if finish != "error": return ...
        if text:  # keep what was decoded; the agent continues in place on "length"
            return AQ.Completion(text, ntok, info.get("prompt_tokens", 0), "length")
        await asyncio.sleep(10)          # let poll() notice a dying server
        while sch.server_down() and time.time() < sch.force_at: await asyncio.sleep(15)
        if time.time() >= sch.force_at: return AQ.Completion("", cap, 0, "length")
```
Also, in `worker()`, sleep about 10 s and re-check `server_down()` before deciding not to requeue. Today an attempt
that fails while the APIServer is still exiting (`poll()` still `None`) is recorded as a failure.

### SHOULD-2. Stall-aware watchdog: restart on hang, not only on exit
`watchdog()` only acts on `proc.poll() is not None`. Add a hang rule that is valid only after the server has been
ready once. This avoids the c2 restart-loop concern:
- `inflight > 0` and `Scheduler.tokens` unchanged for 600 s, **or**
- 4 consecutive failed probes of `GET /v1/models` (10 s timeout, every 30 s).

On either, run `kill_server(proc)` and the existing restart path, keeping the current limits (≤ 2 restarts, none
within 45 min of `force_at`). Global token progress is safe during the initial prefill, because chunked prefill
interleaves decode and the first streams emit within about 30 s. Log one `healthy` line every 5 min, not every
30 s like c2.

### SHOULD-3. Readiness probe that generates a token
c2 checks that `/v1/models` lists the served name. Add one real request after it: `POST /v1/completions` with
`max_tokens=1` and a 300 s timeout. This catches first-forward failures (the SGLang nvcc JIT error above, or
CUDA-graph and Triton compile errors) inside `wait_ready`, where the restart and fallback logic runs. Otherwise they
appear inside the first 16 attempts.

### SHOULD-4. Put a hard timeout around executor calls
dflash: *"Hard wall-clock cap so a wedged sandbox can't block the rollout past its deadline — the sandbox's own
timeout is only advisory."* `LocalExecutor` has its own timeout plus SIGINT and kill, but `_call` is
`asyncio.to_thread` with no outer bound. Use `await asyncio.wait_for(_call(session.execute, code), cfg.tool_timeout + 30)`
and return `[ERROR] tool call exceeded …` on timeout. Apply the same to `submit_answer`.

### SHOULD-5. Order the attempt-2 queue by attempt-1 outcome (in place of c2's early stop on agreement)
c2 stops sampling once 4 attempts agree. With our budget (about 36k tokens per pair), most pairs get one attempt,
so the analogue is deciding where spare capacity goes. Queue attempt 2 first for pairs whose attempt 1:
1. gave no answer,
2. was `forced` or deadline-cut, or
3. answered from `session` or `lenient` parsing.

Queue last the pairs whose attempt 1 was a `submit`. This is a sort key on `pending2` using the attempt-1 record.

### LATER-1. vLLM flags to bench on L4 (from c2/dflash; none is free to turn on blind)
| Flag | c2 / dflash | Ours (vLLM path) | Note |
|---|---|---|---|
| `--mamba-ssm-cache-dtype float16` | c2 float16; verifier float32 | default | Halves Mamba state memory. Qwen3.8 is hybrid (Gated DeltaNet), and our KV is capped at about 304k by Mamba `align` blocks. Needs an accuracy check: SGLang's log shows 5.17 GB ssm_state with radix on. |
| `--mamba-cache-mode` / `--mamba-block-size 512` | `all` / 512 | vLLM picks `align`, block 1600 | Controls prefix-cache granularity and memory for hybrids. Bench KV tokens and tok/s. Keep prefix caching: the tool agent re-sends the whole prompt every turn. |
| `--async-scheduling` | on (both) | `--no-async-scheduling` | Check whether vLLM 0.23 supports it with MTP; if yes, bench it. |
| `--enable-chunked-prefill --max-num-batched-tokens 8192` | on / 8192 | defaults | Set explicitly so decode does not stall behind 30k-token prefills. |
| `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | set | not set | Only needed when Triton's bundled ptxas does not know the arch (Blackwell). Not needed for SM89. |
| `VLLM_USE_FLASHINFER_MOE_FP8=1` | set | n/a | MoE only; Qwen3.8-27B is dense. |

### LATER-2. Voting or selection when there are several candidates per pair
c2 weights votes by 1/entropy, which needs `logprobs` and costs throughput. The verifier variant uses pairwise LLM
grading. ARC has an objective signal instead: whether the submitted expression reproduces the training outputs
(`check_answer_on_train` already exists). If throughput allows k ≥ 3 short tool attempts per pair, fill
`attempt_1`/`attempt_2` by:
1. train-verified first,
2. then vote count,
3. then answer source (`submit` > `forced` > `session`).

### LATER-3. Tool-output cap
c2 caps tool output at 1000 chars and dflash at 600, for context and speed. We use 16000 (`max_tool_output`). ARC
grids print large, so do not change it without an E012-style eval. A cap around 4–6k is worth testing.

### Do not copy
- `with ThreadPoolExecutor(...)` as a timeout guard: its exit waits for the thread (c2 `predict`).
- Restarting on HTTP failure alone: it loops during startup, as c2's comment warns.
- A hard-coded `problems_remaining = 50` in the budget: in c2's 3-problem commit run it reserves time for 49
  problems that do not exist.

## 3. The user's other notebooks (`kaggle kernels list --mine`, 400 kernels; ARC-relevant or infrastructure)

None of the user's own notebooks are ARC-AGI-2 work except the `arc2-*` ones built in this repo:
`arc2-qwen38-27b-bench-mtp{0,3}`, `arc2-sglang-bench`, `arc2-cpu-executor-check`, plus
`3-arc24-developed-2020-winning-solutions` from 2024. Worth reviewing:

| Notebook | What it is | Why look |
|---|---|---|
| `chankhavu/inference-baseline-dflash` (2026-06, AIMO proof-pilot, RTX Pro 6000, vLLM 0.23 from `vllm-0-23-0-py312-cu130`) | The newest harness of this family: DFlash speculative decoding (`num_speculative_tokens 15`), streaming with a finite read timeout as the stall net, partial text saved in `finally`, `asyncio.wait_for` around sandbox calls, phase backstops (`pool.shutdown(wait=False, cancel_futures=True)`), output CSV rewritten atomically after each problem, greedy dynamic per-problem budget (floor 30, ceiling 90 min), and CUDA 13 from the pip toolkit (`CUDA_HOME=.../nvidia/cu13`). | **Most useful.** Source of MUST-2, MUST-4, MUST-7 and SHOULD-4. |
| `chankhavu/vllm-0-23-0-py312-cu128` (2026-06-24) | Wheel builder: `pip download --python-version 3.12 --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com vllm==0.23.0 transformers`, plus `flashinfer-jit-cache==0.6.12` from `https://flashinfer.ai/whl/cu128`. Creates an empty `tiktoken_encodings/` (for gpt-oss). | Our vLLM source. `transformers` is unpinned (the newest at build time). There is no `requirements.lock` and no hash list, unlike our SGLang source. Install with `vllm==0.23.0 flashinfer-python flashinfer-jit-cache`, as `install_server` does. |
| `chankhavu/vllm-0-23-0-py312-cu130` | Same build for CUDA 13.0. | Alternative if cu128 torch misbehaves; the host driver 580.x supports CUDA 13. |
| `chankhavu/fork-of-vllm-0-18-0-py312-d168-flashinfer066`, `vllm-0-18-0-py312-d168-flashinfer066`, `vllm-0-19-1-cu128-py3-12`, `vllm-bleeding-edge-install(-cu129)` | Older vLLM wheel sources (0.18 is what c2 uses). | Fallback only if 0.23 fails on L4. 0.19.x is closest to the third-party 0.19.0 wheelhouse that our v1 benches proved on L4. |
| `chankhavu/c2-distill-verifier` | c2 plus pairwise LLM grading of answer buckets (`MatchupTracker`, ratings 1–10, verification at temperature 0, 8 rollouts plus 16 verifications), 262k context, `mamba_ssm_cache_dtype float32`. | Selection ideas (LATER-2). |
| `chankhavu/c2-distill-eagle3-watchdog`, `c2-distill-eagle3-cot` | c2 with EAGLE3 speculative decoding from a custom vLLM build (`vllm-cu129-pceaglehybrid-dev`). | Only if we revisit speculative decoding beyond MTP. |
| `chankhavu/gptoss88-vllm018-watchdog`, `c2-distill-vllmwatchdog` | The same watchdog harness for gpt-oss, and an earlier watchdog version. | Nothing beyond what this review covers. |
| `chankhavu/ipython-sandbox` | `sandbox_server.py` / `sandbox_client.py` (`LocalSandbox`) used by c2 and dflash. | Alternative to our `LocalExecutor`; ours is already tested on Kaggle CPU (`arc2-cpu-executor-check`). |
| `chankhavu/local-validation-benchmark` (AIMO3) | Local validation harness. | Low priority. |
