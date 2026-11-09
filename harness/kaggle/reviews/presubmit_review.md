# Pre-submission robustness review: v2 (tool agent) Kaggle submission

Date 2026-10-11. Read-only review of `kaggle/src/solver.py` (version of 15:56), `kaggle/build_notebook.py`,
`kaggle/nb_v2/` (built 15:54), `evals/arc_harness/{agent_qwen,executor}.py` and the tests. Line numbers refer to
those versions. Nothing was run on Kaggle, the API or a GPU.

## Tests run locally (.venv, Python 3.11.5, no network)

| Test | Result |
|---|---|
| `kaggle/tests/fake_server_test.py` (v1 plain path: normal, deadline, server down, crash and restart, submission semantics) | ALL OK |
| `kaggle/tests/fake_tools_test.py` (v2 tool path: normal, deadline, server down) | ALL OK |
| `python -m unittest evals/tests/test_qwen_tools.py` | 17 tests OK |

Kaggle runs Python 3.12, and no 3.12 wheels are cached offline, so the tests could not be repeated on 3.12. None of
the tests runs `main()`: install, launch, `wait_ready`, watchdog, the rerun switch, local mode and the `finally`
block are untested locally.

## Blockers (process)

1. **The notebook is stale.** The solver embedded in `kaggle/nb_v2/arc2-qwen38-tools-submit.ipynb` is not the
   current `solver.py`: it is missing the CPU-torch force-reinstall block in `install_server`
   (solver.py:255-262). Rebuild the notebook before pushing, and check that the embedded `SOLVER` equals
   `build_notebook.solver_source()`.
2. **The v2 stack has never run on a Kaggle GPU.** That stack is vLLM 0.23.0 from the user's source, the tool agent
   and `LocalExecutor`. The earlier GPU benches used the third-party vLLM 0.19 wheelhouse. The SGLang bench failed
   because nvcc is missing, and the CPU check covered only the executor. The local-mode commit run is the only
   end-to-end test. Do not submit until its log shows all of the following:
   - the server is ready and torch has CUDA;
   - attempts answered, with tool calls run;
   - no `error` records and no HTTP 400s;
   - no vLLM preemption warnings;
   - a sensible KV usage and concurrency line.
3. **The default server is the broken one.** `SERVER` defaults to `"sglang"` (solver.py:58), and so does
   `build_v2` (build_notebook.py:98). SGLang crashes on Kaggle at the first forward pass because nvcc is missing
   (E013 README). The current nb_v2 passes `ARC_SERVER=vllm`, but a rebuild without `--env` silently produces a
   fallback-only submission. Change the default to `vllm`.

## Should-fix

S1. **There is no process-level kill switch for the 12 h limit.** Overrunning 12 h fails the submission even though
a valid `submission.json` exists. The only stop is `asyncio.wait_for` at the hard stop (solver.py:660). That call
waits for every cancelled attempt to finish its `finally` (`await to_thread(session.close)`). After that,
`asyncio.run` (3.12) waits up to 300 s for default-executor threads to finish. A watchdog `wait_ready` thread can run
for up to 1800 s. A process stuck in D state in `proc.wait()` (executor.py:144) can block forever. Fixes:
- In `solver.py`, add a `threading.Timer` daemon that fires at `T0 + 11.75 h`, validates `submission.json` (or
  rewrites the fallback) and calls `os._exit(0)`.
- In the notebook runner cell (build_notebook.py:115-119), kill the solver's process group at `T0 + 11.85 h`.
- Call `os._exit` right after the final write in `go()` instead of returning through `asyncio.run`.

Expected wall time when nothing hangs:
- before `T0`: a few minutes;
- startup: about 15-20 min (install timeout 40 min, `wait_ready` timeout 30 min);
- forcing starts at 11.0 h;
- hard stop at 11.6 h;
- post-processing: up to about 5 min.

The run ends at about 11.7 h, which leaves a margin of about 15 min only if nothing hangs.

S2. **Deadlines are enforced only when a chunk arrives, and the forced call runs until the hard stop.**
- `Client.stream` checks `time.time() > deadline` only after a received line (solver.py:427). The httpx read
  timeout is `None` (solver.py:380). A queued request, or a hung server that is still alive, never times out; only
  the hard-stop cancel ends it.
- `DeadlineLLM` gives the forced `submit_answer` call `dl = sch.hard_stop` (solver.py:513). That is the same instant
  `wait_for` cancels everything (solver.py:660). An attempt still in the forced phase at the hard stop therefore
  loses its forced expression and also the session `answer` fallback that runs after it.

Fixes: wrap each stream in `asyncio.timeout(deadline - now)`, and give forced calls `hard_stop - 180 s`, so that
`submit_answer` (at most 35 s) and the session fallback finish before the cancel.

S3. **An exception skips the session-answer fallback** (agent_qwen.py:409-414). The `submit_answer(session,
"answer")` fallback runs only on normal loop exit. Any exception from `llm.complete` jumps to `except`, and the
attempt returns `pred=None` even when the model has set `answer`. Examples are an HTTP 400 for context length, an
`httpx.ReadError` or a server crash. Verified with a probe: `answer=[[9,8,7]]`, then the LLM raises, and the result
is `pred None, stop_reason error`. Fixes:
- Move the fallback into the `except` / before `close` in `finally`.
- Retry transient errors once or twice in `DeadlineLLM` (solver.py:506-525). `OpenAICompletionsLLM` already has
  retries; the Kaggle wrapper has none.

S4. **The agent's context limit does not match the server's.** `CONTEXT = 262144` (agent_qwen.py:32), and the room
calculation uses it (agent_qwen.py:311). The server runs at `--max-model-len` / `--context-length 131072`
(solver.py:50). `room` also uses the previous request's `prompt_tokens` and ignores tool responses appended since
then; each can be up to 16k chars.

In E012 (medium tier, 48k cap) the `max_prompt` per attempt was p50 49.6k, p90 64.7k and max 86.5k tokens, and each
chunk asks for up to 16384 more tokens. Larger hidden tasks or chatty tool output can push prompt + `max_tokens`
past 131072. vLLM then returns 400, and through S3 the attempt is lost.

Fix: pass the context length in `AgentConfig` (set from `MAX_MODEL_LEN`), add an estimate of the appended tool
output (about chars/3), and switch to the forced-submit path instead of sending an oversized request.

S5. **`kill_server` kills its own shell** (solver.py:329-331). Verified locally:
`sh -c "pkill -9 -f A ; pkill -9 -f B ; echo after"` returns rc -9 with no output. The first `pkill -f` matches
the `sh -c` command line, which contains the pattern, so the later pkills and the `sleep 5` never run. The patterns
also miss vLLM's `VLLM::EngineCore` / worker processes. After a crash the watchdog restart (solver.py:791-810)
probably finds the GPUs still occupied, and the restart fails. Fix:
- Use `os.killpg(proc.pid, SIGKILL)`; the server is started with `start_new_session=True` (solver.py:325).
- Add bracketed patterns such as `pkill -9 -f '[V]LLM::'`.
- Wait until `nvidia-smi` shows the memory freed before relaunching.

S6. **A server start failure is fatal** (solver.py:783-784). One transient start failure (NCCL or compile flake)
makes the whole run fallback-only, scoring about 0. Retry once (kill the group, relaunch) if `T0 + 1.5 h` has not
passed.

S7. **A `LocalExecutor` start failure raises `TypeError`** (executor.py:120-121). If the prelude does not answer
within `timeout + grace`, `_run` returns `None`, and `result["process_status"]` raises `TypeError` (verified with a
probe). This can happen on a burst of `conc` concurrent numpy imports on 4 busy CPUs, or on a deadlocked fork (see
N1).
- At attempt start, the attempt errors out. The server is up, so the attempt is not requeued and attempt 1 for that
  pair is lost.
- In the restart path (executor.py:186-187), the exception leaves `execute` and, through S3, the session answer is
  lost.

Fix: treat `None` as a failure, then retry the spawn once with a longer timeout (for example 90 s).

S8. **Tool children are not protected against CPU, RAM or disk abuse** (executor.py:81-117, 164). The 4 CPUs are
shared with about 6 busy vLLM processes (API server, EngineCore, 4 TP workers) and up to `conc` (about 15) tool
children. Each child can spin a core for 30 s plus 5 s grace per cell. SIGINT goes only to the child, not its
process group (executor.py:164), so multiprocessing or subprocess grandchildren keep running until the attempt
ends. Cheap fixes inside `_limit_memory` (or the child prologue):
- `os.nice(15)`, so tool code never starves vLLM's scheduler and workers;
- `RLIMIT_CPU`, as a cumulative backstop of a few minutes per process;
- `RLIMIT_FSIZE` of about 64 MB; the cwd is `/tmp` and file writes are unbounded;
- write `1000` to `/proc/self/oom_score_adj`, so a host OOM kills a tool child and not vLLM;
- on timeout, `killpg(SIGINT)` instead of `send_signal`.

`RLIMIT_AS` 4 GB × 15 children is about 60 GB worst case, and the host RAM of the 4xL4 box is not recorded
anywhere. Log `free -g` at startup.

S9. **In-stream server errors are swallowed** (solver.py:410-424). vLLM and SGLang report some failures as
`data: {"error": ...}` on a 200 stream. `Client.stream` ignores the event and returns `finish="stop"` with empty
text, which the agent treats as an EOS in the middle of thinking. Raise on an `"error"` key so failures show up in
the records and go through S3.

S10. **KV sizing for tool contexts is not verified.** nb_v2 sets `ARC_KV_PER_SEQ=20000`, giving
`conc = kv/20000`, which is about 15 with the 304k reported for vLLM 0.19. Tool-agent contexts grow to p50 about
50k and p90 about 65k tokens. The vLLM bench measured 16 sequences at 40k context with 80% peak KV use, so 15
sequences at 50-65k context are likely at or over KV capacity, which causes preemption and recompute. Check vLLM's
`preempted` warnings and KV % in the commit run, and size from that.

## Nice-to-have

- N1. `preexec_fn` in a heavily threaded process (executor.py:81-87, 115). Python documents it as unsafe with
  threads, because the child can deadlock before `exec`. Use `start_new_session=True` and call `setrlimit` at the
  top of `_CHILD`, or `resource.prlimit(pid, ...)` after the spawn.
- N2. Hitting `max_turns` (agent_qwen.py:303-305) goes straight to the session `answer` without the forced
  `submit_answer` that the budget cap gets.
- N3. `max_eos_continues` counts per attempt, not per turn as its comment says (agent_qwen.py:85, 326).
- N4. `kill_server` blocks the event loop for about 5 s through `subprocess.run` (solver.py:803). This is minor and
  only happens on a restart.
- N5. The docstring advertises `ARC_MODE=bench` (solver.py:7), but `main()` has no bench branch, so bench mode
  falls through to a full 11 h run on the test set. The v1 `submit` / `bench` variants of `build_notebook.py` now
  break with the current defaults: sglang, and tools without the bundled pkg or kernel source. Drop or fix them.
- N6. The CPU-torch reinstall (solver.py:255-262) ignores its return code. Raise if it fails, or if
  `torch.version.cuda` is still `None` afterwards.
- N7. A stubbed test of `main()` would cover the rerun switch, local mode, `finally` and the fallback rewrite. Use
  fake `/kaggle` paths and a fake server command.
- N8. The server is left running at `os._exit` (solver.py:875). This is harmless: the bench runs ended the same way
  and completed.

## Checked and OK

- **Submission file:**
  - The fallback for the test file is written first (solver.py:757-758).
  - `build()` covers every task id and test index and always sets both attempts. Attempt 2 is a distinct answer or
    the test-input copy (solver.py:194-206).
  - `valid_grid` enforces 1-30 × 1-30, a rectangular shape and Python ints 0-9; bools are rejected.
  - Writes are atomic: tmp file, then `os.replace` (solver.py:208-213).
  - `finally` re-validates `/kaggle/working/submission.json` against the test file and rewrites the fallback if it
    is broken (solver.py:866-875).
  - Agent predictions are validated and numpy values are converted with `tolist`.
- **Rerun switch:**
  - The notebook's first cell and `main()` both force `submit` when `KAGGLE_IS_COMPETITION_RERUN` is set
    (solver.py:753-754).
  - In local mode, predictions go to `local_submission.json` (solver.py:767), only after the test-file fallback
    `submission.json` exists.
  - The final check always validates `submission.json` against `arc-agi_test_challenges.json`. A commit run cannot
    leave local predictions in `submission.json`, and the rerun regenerates it from the hidden file.
- **Thread pool:** the default executor has `conc + 8` threads (solver.py:823), and each attempt runs at most one
  `to_thread` call at a time. There is no starvation.
- **Executor:**
  - Timeout is SIGINT, then grace, then `killpg(SIGKILL)` plus `wait` (reaped).
  - Each child gets its own session, a clean environment with BLAS/OMP threads at 1, `-I`, output truncation, and
    is always closed in `finally`.
  - On Kaggle's image the CPU check verified timeout, crash restart and MemoryError.
- **Agent:**
  - Malformed calls get feedback.
  - The budget is accounted across chunks.
  - At the cap, the agent forces `submit_answer` with `stop=["</parameter>"]`, then falls back to the session
    `answer`.
  - `skip_special_tokens=False` keeps `<tool_call>` and `</think>`.
  - Sampling is temp 1.0, top_p 0.95, top_k 20, the same as E012.
