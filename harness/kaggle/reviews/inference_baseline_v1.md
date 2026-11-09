# Review: `chankhavu/inference-baseline-v1` against our v2 submission (2026-10-11)

Source pulled read-only with `kaggle kernels pull ... -m` into `kaggle/reviews/src_inference_baseline_v1/`.
For reference only, outside the repo: the attached `chankhavu/ipython-sandbox` kernel source is in
`/tmp/ipython_sandbox_src/`, and the last run's logs and CSVs are in `/tmp/ibv1_out/`.

## What the notebook actually is

It is **not an ARC notebook**. It is the user's **AIMO Proof Pilot** proof-writing harness:
- `competition_sources: ai-mathematical-olympiad-proof-pilot`
- `machine_shape: NvidiaRtxPro6000` (one 96 GB Blackwell GPU, SM120)
- model `smolmo-32b-sft-merged-proofpilot` in NVFP4
- vLLM 0.23.0 cu130

Prompts, ranking and answer format do not carry over to ARC. The infrastructure patterns do.

Design in one paragraph:
- **Install.** `uv pip uninstall` keras, matplotlib, scikit-learn and tensorflow. Then
  `uv pip install -U --no-index --find-links <kernel_source>/wheels vllm flashinfer-*`.
- **Sandbox server.** Starts the NeMo-Skills Flask IPython sandbox server on :6000. It forks one IPython shell
  process per session. On timeout it sends SIGINT, waits a 0.5 s grace, then terminates and restarts the shell.
  `RLIMIT_AS` is set to 2 x 50 GiB, which is effectively no limit. The client replays the history of successful
  cells after a restart.
- **Model.** Preloads the 22.5 GB of weights into the page cache in 27 s. Starts `vllm serve` (TP=1, fp8 KV,
  prefix caching, chunked prefill, async scheduling). The server was ready in 109 s.
- **Solver.** Writes `kaggle_solver.py` and runs it as a subprocess over `test.csv`. Per problem:
  1. 16 tool rollouts;
  2. 3 self-verify rollouts per solution, graded 0/1/6/7;
  3. one refine round of 4 candidates x 4 rollouts;
  4. ranking by mean self-grade.
- **Rollout engine.** Uses streaming chat completions with `tools=` plus `tool_choice="none"`, so the template
  renders the schema but the server does not parse calls. The harness parses `<function_calls>` itself (AST
  first, regex as fallback).
- **Deadlines.** One shared deadline per problem: a greedy budget with a reserved floor for each remaining
  problem. Closing the stream aborts the decode. There is also a pool backstop (deadline + 60 s) and a
  per-problem watchdog thread (deadline + 300 s).
- **Output.** The results CSV is rewritten atomically (tmp + fsync + `os.replace`) after every problem. The
  notebook cell then converts it to `submission.csv`.
- **Rerun detection and fallback.** Neither exists. The commit run simply solves the 3 placeholder problems in
  the visible `test.csv` (95 s, 158 s and 61 s; all self-graded 7). `check_sandbox.py` exists but is commented out.

## Side-by-side

| Area | inference-baseline-v1 | Our v2 (`kaggle/src/solver.py` + `evals/arc_harness`) | Verdict |
|---|---|---|---|
| Offline install | `uv pip install --no-index --find-links`: 141 pkgs prepared in 9.7 s, installed in 0.2 s | `pip install --no-index -r requirements.lock`: **405 s** (sglang-bench-v1), 501 s (cpu-check-v1) | Adopt uv (R6) |
| Server | `vllm serve`, TP1, `--kv-cache-dtype fp8 --enable-chunked-prefill --enable-prefix-caching --async-scheduling`, util 0.95, host 0.0.0.0 | SGLang TP4 triton attn fp8_e4m3, or vLLM TP4 `TRITON_ATTN` with `--no-async-scheduling`, host 127.0.0.1 | Ours fits L4; bench async later (R12) |
| HTTP client | `httpx.Timeout(connect=10, read=300, write=60)`, `max_retries=0` ("finite read timeout is essential") | `httpx.Timeout(None, connect=30)`: **no read timeout**; deadline checked only when a line arrives | **Gap (R1)** |
| Partial work on error | Decoded text committed to the conversation in `finally` before any exception propagates; error -> `finish_reason="error"` with the generation kept | `agent_qwen.solve`: an exception skips the session-`answer` fallback | **Gap (R2)** |
| Tool exec | Flask server, forked IPython; SIGINT, 0.5 s grace, terminate and restart; plus `asyncio.wait_for(..., python_timeout + 30)` client-side hard cap | Child process per attempt; SIGINT, 5 s grace, killpg and restart; 4 GB `RLIMIT_AS`, BLAS threads 1, `-I`, cwd /tmp; no hard cap around the thread call | Ours is stronger on limits; add the hard cap (R3) |
| Sandbox self-test | `check_sandbox.py`: exec, state, traceback, timeout, truncation (unused) | CPU-only notebook ran once; nothing at runtime | Add at startup (R4) |
| Prompt and tools | Self-parsed calls; no server tool parser ("it corrupts multi-line code= args"); byte-exact prompt literals | Self-parsed Qwen XML calls on raw `/v1/completions`; byte-identical harness bundle | Same idea; keep ours |
| Sampling | T=1.0, top_p=0.95, max_tokens=0 (fill context), 64 turns | T=1.0, top_p=0.95, top_k=20, 48k cap, 60 turns | Keep ours |
| Scheduling | Sequential problems; phase barriers (16 -> verify -> refine). The vLLM log shows the tail: running reqs 16 -> 1, gen 820 -> 61 tok/s at every phase end | Continuous worker pool over all pairs, budget from measured capacity | **Ours is better; do not adopt phases** |
| Deadline | Greedy budget with a reserved floor (`budget = clamp(time_left - n_others * min_s, min_s, max_s)`); one deadline shared by all phases; backstop and watchdog | Capacity share per attempt; force at 11h00, hard stop 11h36 via `wait_for(gather)` | Ours fits; it only needs stall handling (R1) |
| Selection / attempt 2 | Rank: `has_text` > mean self-grade > original over refined > n_verifications > length; refine only "eligible" (neither unanimous 7 nor 0); early exit if unanimous 7; never submit empty | Attempt 2 whenever at least 32k spare; the first distinct grid fills the slot; no-answer attempts fall back to a copy of the test input | Borrow the "never empty" and "eligibility" ideas (R7, R9) |
| Submission | Atomic CSV with fsync after each problem; every input row gets an output row; written by a later notebook cell (fragile) | Fallback written first, atomic rewrite after every answer, validated in `finally` | Ours is better; add fsync (R10) |
| Rerun / self-test | None. The commit run is the full pipeline on the placeholder `test.csv`, and nothing checks the result | `KAGGLE_IS_COMPETITION_RERUN` switch; local self-scored 8-task set, 40 min | Ours is better; make it a gate (R5) |
| Logging | Per-rollout trace JSON, atomic `summary.jsonl`, `DONE.txt`, server logs to file, httpx logger at WARNING | attempts jsonl, traces, summary json, progress and CPU monitors, server log | Similar; add hashes (R11) |

## Recommendations, prioritised

**MUST** means do it before the first submission today: small changes that only touch failure paths.
**SHOULD** means today if time allows, otherwise in v3. **LATER** means it needs a benchmark or eval first.

### R1 (MUST): bound every streamed request; treat a stall like the deadline
- **What.** Add a finite read timeout to our `Client`, and wrap each `client.stream` call in a wall-clock cap
  tied to its deadline.
- **Why.** Their `build_client` puts it this way: "A FINITE read timeout is essential: with streaming, `read`
  is the max SILENCE BETWEEN TOKENS ... It is the stall safety net." Ours uses `httpx.Timeout(None, connect=30.0)`.
  `Client.stream` checks `time.time() > deadline` only after a line arrives. The watchdog checks only
  `proc.poll()`.
- **Failure it prevents.** A TP=4 server that hangs without dying (for example an NCCL hang) would block every
  in-flight attempt until the hard stop. `wait_for(gather)` would then cancel them all. They would get no
  forced submit, and their sessions would close without harvesting `answer`.
- **How.**
  ```python
  # Client.__init__ (theirs: httpx.Timeout(connect=10.0, read=read, write=60.0, pool=read), max_retries=0)
  self.http = httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=None),
                                limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))
  # DeadlineLLM.complete (and the same in attempt_plain)
  try:
      text, ntok, finish = await asyncio.wait_for(
          client.stream(prompt, max_tokens, dl, cb, stop=stop, info=info), timeout=max(5.0, dl - time.time()) + 30)
  except (asyncio.TimeoutError, httpx.TimeoutException):
      stats["deadline_cuts"] += 1
      return AQ.Completion("", cap, 0, "length")   # agent goes to its forced submit / session answer
  ```
  Read=600 s leaves room for queued prefill. With `continuous_usage_stats` a healthy stream sends a chunk per
  token.
- **Later.** In `monitor()`, if the window rate is 0 tok/s for 10 min while `inflight` is non-empty, set
  `server["down"]`, then kill and restart the server.

### R2 (MUST): never lose the session answer on an error path
- **What.** In `agent_qwen.solve`, also try `submit_answer(session, "answer")` inside the `except Exception`
  branch, before the `finally` closes the session.
- **Why.** Theirs commits partial output "inside the finally, BEFORE any exception ... can propagate ...
  Otherwise the in-flight turn's reasoning+content is silently discarded".
- **The bug in ours.** The `if pred is None and session is not None:` fallback sits inside the `try`. Any
  HTTP error, a 400 or 5xx, or a stream exception therefore records `pred=None`, even when the model had set
  `answer`.
- **How.**
  ```python
  except Exception as exc:
      error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
      if pred is None and session is not None:
          try:
              out, grid = await _call(submit_answer, session, "answer")
              if grid is not None:
                  pred, source = grid, "session"
          except Exception:
              pass
  ```
  The success path is unchanged, so the E012 numbers still hold. The scheduler then records the attempt as
  answered instead of requeueing it.

### R3 (MUST): hard wall-clock cap on executor calls; robust kill
- **What and why: cap the call.**
  - Theirs: `result = await asyncio.wait_for(_execute_in_sandbox(...), timeout=python_timeout + 30.0)`, with the
    comment "a wedged sandbox ... can't block the rollout past its deadline — the sandbox's own `timeout` is
    only advisory".
  - Ours: `_call(session.execute, code)` is an unbounded `asyncio.to_thread`. `LocalExecutor._kill` calls
    `self.proc.wait()` with no timeout. If `os.killpg` raises (the child never reached `setsid`), the child is
    never killed, and that worker thread and its attempt hang until the hard stop.
- **What and why: drop `preexec_fn`.** We spawn executors from 16+ threads, and CPython documents
  `preexec_fn` as unsafe in a threaded process.
- **How.**
  ```python
  # executor.py: no preexec_fn; the child limits itself
  self.proc = subprocess.Popen([...], start_new_session=True, ...)       # replaces os.setsid in preexec
  # top of _CHILD:  import resource; m = int(sys.argv[3]); m and resource.setrlimit(resource.RLIMIT_AS, (m, m)); os.nice(5)
  def _kill(self):
      try: os.killpg(self.proc.pid, signal.SIGKILL)
      except (ProcessLookupError, PermissionError):
          try: self.proc.kill()
          except ProcessLookupError: pass
      try: self.proc.wait(timeout=10)
      except subprocess.TimeoutExpired: pass
  # agent_qwen: wrap tool calls
  result = await asyncio.wait_for(_call(session.execute, code), cfg.tool_timeout + 35)
  #   on TimeoutError -> {"process_status": "timeout", "stdout": "", "stderr": "[sandbox call exceeded ...]"} and
  #   replace the session with a fresh executor
  ```
  `os.nice(5)` keeps tool code from competing with SGLang's CPU-bound scheduler and tokenizer processes.
  `cpu_monitor` already logs the load.

### R4 (MUST): runtime executor self-test; fall back to the plain agent if it fails
- **What.** Before the server starts, spend about 10 s on the checks their `check_sandbox.py` makes:
  - basic exec;
  - state persists between cells;
  - a traceback comes back;
  - the timeout is honoured, and state survives the SIGINT;
  - output is truncated;
  - plus a `submit_answer` round trip.
- **Why.** The tool path depends on pipes, `select`, signals and rlimits in an image we cannot test until
  rerun time. If it fails, we still have the evaluated no-tool path (`ARC_AGENT=plain`) in the same file.
- **How.**
  ```python
  def executor_selftest():
      from arc_harness.executor import LocalExecutor, submit_answer
      with LocalExecutor([{"input": [[1]], "output": [[2]]}], [[1]], timeout=3) as ex:
          assert ex.execute("x = 41")["process_status"] == "completed"
          assert ex.execute("print(x + 1)")["stdout"].strip() == "42"
          assert "ZeroDivisionError" in ex.execute("1/0")["stderr"]
          t = time.time(); r = ex.execute("import time; time.sleep(10)")
          assert r["process_status"] == "timeout" and time.time() - t < 10
          assert ex.execute("print(x)")["stdout"].strip() == "41"
          assert submit_answer(ex, "[[2]]")[1] == [[2]]
  # main(): try: executor_selftest() except Exception as e: log("executor self-test FAILED", e); global AGENT; AGENT = "plain"
  ```

### R5 (MUST): turn the commit run into a gate
- **What.** In the v2 notebook's last cell, on non-rerun runs only, raise if any of these hold:
  - `summary_local.json` is missing;
  - `answered == 0`;
  - `tool_calls == 0` while `ARC_AGENT=tools`.
- **Why.** Their commit run exercises the full pipeline but checks nothing. Ours always ends green, because
  the `finally` writes the fallback and `os._exit(0)` follows. A version whose server never started therefore
  looks like a good one.
- **How.** Raise only after `submission.json` has been written, and never when `KAGGLE_IS_COMPETITION_RERUN` is
  set.
  ```python
  if not os.environ.get('KAGGLE_IS_COMPETITION_RERUN'):
      s = json.load(open('/kaggle/working/summary_local.json'))
      assert s['answered'] > 0 and (s['agent'] != 'tools' or s['tool_calls'] > 0), s
  ```

### R6 (SHOULD, today with pip fallback): install with uv
- **What.** Install the wheel source with `uv pip install --no-index --find-links`, falling back to pip.
- **Why.** Their log shows "Prepared 141 packages in 9.68s ... Installed 141 packages in 221ms" (vLLM 0.23.0,
  torch 2.11.0+cu130, flashinfer 0.6.12). Our pip install of the SGLang source took 405 s on the GPU box and
  501 s on CPU. Saving about 6.5 min at roughly 250 tok/s is about 100k tokens, or two to three attempts.
- **How.** Their cell is
  `!uv pip install -U --no-index --find-links /kaggle/input/notebooks/chankhavu/vllm-0-23-0-py312-cu130/wheels vllm flashinfer-jit-cache flashinfer-python flashinfer-cubin`.
  Ours would be:
  ```python
  uv = shutil.which("uv")
  code = 1
  if uv:
      code, out = sh(f"{uv} pip install --python {sys.executable} --no-index --find-links {wd} -r {wd}/requirements.lock",
                     timeout=1200)
  if code != 0:
      code, out = sh(f"{sys.executable} -m pip install --no-index --find-links {wd} -r {wd}/requirements.lock", timeout=2400)
  ```
  Keep the existing `import torch, sglang` check after install.
- **Also from them.** They uninstall `keras matplotlib scikit-learn tensorflow` first. That is not needed with
  transformers 5.12, which has no TF path, and scipy/numpy must stay for our tools.

### R7 (SHOULD): salvage the 11% of attempts that end with no answer
- **What.** Before falling back to the test-input copy, look for a usable function in the session.
- **Why.** Theirs never submits empty ("a NON-EMPTY solution ALWAYS outranks an empty one", then a fallback to
  the most developed raw attempt). In our E012 run `evals/results/qwen_tools_medium_48k_dev_s24.jsonl`,
  **12 of 107 attempts have `answer_source=None`**. In most of them the forced expression was a whole `def`
  with no `answer` (NameError), or `answer` was never set.
- **How.** In `agent_qwen.solve`, after the `answer` fallback fails, run one cell that:
  1. collects the session's user-defined one-argument functions (`f.__module__ == '__main__'`), newest first,
     capped at about 8;
  2. scores each on the training pairs, in-session, as `check_answer_on_train` does;
  3. applies the best scorer to `test_input`;
  4. accepts the result if `valid_grid`, with `answer_source="salvage"`.

  This is host-only work: no tokens, and one executor call.

### R8 (MUST, one line): agent context = server context
- **What.** Set `AQ.CONTEXT = MAX_MODEL_LEN` in `setup_harness()`.
- **Why.** `agent_qwen.CONTEXT = 262144` is used for `room`, but the Kaggle server runs with
  `--context-length 131072`.
  - At the current 48k cap, E012's largest prompt was 86.5k (p99 76.8k), so this is not hit today.
  - Raising `ARC_TOOLS_CAP` would make long attempts fail with HTTP 400, losing their answer unless R2 is in.
  - Their notebook avoids the mismatch by letting the server fill the context (`max_tokens=0`).

### R9 (LATER): spend attempt 2 where attempt 1 is weakest
- **What.** Order `pending2` by attempt-1 outcome: no valid answer first, then an answer that is not
  train-verified, then train-verified answers.
- **Why.** This is their "refine only eligible; early exit if unanimous 7" policy translated to ARC. Our
  host-side train check is free and more reliable than a self-grade.
- **How.**
  - Add `check_only` to `AgentConfig`: run `check_answer_on_train` on the accepted expression and record
    `submit_verified` without rejecting.
  - The scheduler sorts `pending2` by that field.
  - Do not change attempt 1.

### R10 (MUST, trivial): fsync before rename
Their `_atomic_write_text` calls `f.flush(); os.fsync(f.fileno())` before `os.replace`. `Submission.write`
should do the same.

### R11 (SHOULD, trivial): log what we actually run
- Their prompts are "byte-exact ... the model is extremely prompt-sensitive". Log the sha256 of every
  bundled harness file, plus the rendered `initial_prompt` of one fixed task, and compare them with the Vast
  E012 run.
- Also log `nvidia-smi` at start, as their cell 0 does.

### R12 (LATER, needs a bench): vLLM flags and fp8 KV accuracy
- **Flags.** Their vLLM 0.23 uses `--async-scheduling --enable-chunked-prefill`, while our vLLM path forces
  `--no-async-scheduling`. Bench async scheduling on 4xL4 if we switch to `ARC_SERVER=vllm`.
- **fp8 KV accuracy.** Their vLLM log warns: "Using KV cache scaling factor 1.0 for fp8_e4m3 ... uncalibrated
  q_scale ... may cause accuracy issues". We run fp8_e4m3 KV on both servers.
  - Confirm that the E012 Vast numbers we rely on were measured with fp8 KV too.
  - If they were measured with bf16 KV, the local accuracy may not transfer. Re-check on the local set.

### R13 (LATER): restore state after a crash by replaying cells
Their client re-executes the history of successful cells after a shell restart, and skips cells that timed
out. Ours restores only the task data and says so. Consider this only if the traces show many `state_reset`
events, because it changes the behaviour we evaluated.

### Do not adopt
- **Per-problem phase barriers.** The GPU idles in the tail.
- **The 600-character tool-output cap.** ARC grids need more.
- **The 50 GiB sandbox memory limit.**
- **`--host 0.0.0.0`.**
- **OpenAI chat with `tools=`/`tool_choice="none"`.** Our raw completions do the same job, and also keep the
  prefix cache hot.
- **`VLLM_USE_FLASHINFER_MOE_FP8`, `TORCH_CUDA_ARCH_LIST=12.0`, `TRITON_PTXAS_PATH`.** These are Blackwell- or
  MoE-specific. On L4 the matching arch would be 8.9, and only if JIT problems appear.
