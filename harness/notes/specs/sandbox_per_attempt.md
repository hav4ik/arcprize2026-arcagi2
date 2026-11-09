# Spec: one sandbox container per attempt (replaces the shared NeMo sandbox server)

## Why: review of the current architecture
The current path: `SandboxSession` (evals/sandbox_client.py) → HTTP → nginx (hash on `X-Session-ID`) → 24 single-process uWSGI workers → `ShellManager` forks one IPython process per session (evals/sandbox_docker/main.py).

Problems seen in E004 (2026-10-02/03):

1. **Session lifetime is detached from the attempt.**
   - The end-of-attempt delete does not reach the worker that holds the session.
   - The finished sessions' processes accumulated: ~1,650 processes and ~54 GB, until the host ran out of memory and the runs failed.
   - The idle reaper added on 2026-10-04 (`NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT`) only approximates the attempt's end. Tying cleanup to a timeout is the wrong coupling.
2. **No isolation between attempts.**
   - `/tmp` is shared by every session in every concurrent run. The safety audit saw a model list other runs' scratch files.
   - Code runs as root, with no per-session memory, CPU or process limits. A single cell allocated 40 GB.
3. **Head-of-line blocking.** Each worker is single-process, so a cell queues behind other sessions' cells on the same worker. The HTTP timeout had to be widened to `2×timeout+60`.
4. **Code-mode checker sessions** go through the same shared server and inherit all of the above.
5. **Hard to reason about.** Session state lives in a forked process inside a web server behind a hashing proxy. Health, cleanup and failure handling all go through HTTP.

## Proposed architecture
**One short-lived container per attempt, owned by the harness process, with a persistent Python REPL inside.**
- The harness talks to the REPL over the container's stdin/stdout (`docker run -i`), one JSON message per line.
- The attempt's `with` block owns the container. Leaving the block (normal end, exception, Ctrl-C) stops it.
- `--rm` removes it. Session lifetime is therefore the attempt's lifetime by construction: no server, routing, reaper or timeouts.

```
harness thread (one attempt)
  with AttemptSandbox(train, test_input) as sb:        # docker run -i --rm ... python3 /repl.py
      sb.execute(code)  -> {"status": completed|error|timeout, "stdout", "stderr"}
      ...
  # __exit__: close stdin, wait briefly, then `docker kill`; --rm deletes the container
submit_solution(code):
  with AttemptSandbox(train_inputs_only, test_input) as checker:   # fresh container, no outputs inside
      run transform on inputs -> host compares                      # same v3 semantics as today
```

### Container settings, per attempt
`docker run -i --rm` with:

| Setting | Effect |
|---|---|
| `--network none` | No network at all; no shared Docker network needed |
| `--user 65534:65534` | Not root |
| `--read-only --tmpfs /tmp:size=256m` | Private, size-limited scratch space per attempt; nothing shared between attempts |
| `--memory 4g --memory-swap 4g` | A runaway cell kills only its own attempt |
| `--cpus 1` | Limits CPU per attempt |
| `--pids-limit 128` | Limits processes per attempt |
| `--cap-drop ALL --security-opt no-new-privileges` | Drops privileges |
| `--entrypoint python3`, with `repl.py` mounted read-only | Skips the image's web-server entrypoint |
| `--label arc-run=<run id>` | Lets the sweep below find the run's containers |

Image: the existing NeMo sandbox image (Python 3.10, numpy 2.2.6, scipy 1.15.3). Measured startup, including the numpy and scipy import, is about 0.9 s. That is negligible against 1–15 min attempts, and acceptable for code-mode submission checks.

### REPL protocol (`evals/sandbox_docker/repl.py`)
- **Input:** one JSON request per line, `{"code": str, "timeout": float, "max_output": int}`.
- **Output:** one JSON reply per line, `{"status", "stdout", "stderr"}`. Stdout is truncated to `max_output`.
- **Execution:** `exec` runs in one persistent namespace preloaded with `train`, `test_input` and `np`, so variables persist across cells as today.
- **Display:** like IPython, the value of the cell's final expression is printed. The tool description currently promises this.
- **Timeout:**
  - `signal.setitimer` raises an exception in the cell. State is kept, and the reply says `timeout`, matching today's `[timeout]` semantics.
  - If the REPL does not answer within `timeout + grace`, the harness kills the container and starts a fresh one. It then reloads `train`/`test_input` and marks `state_reset`, exactly as `SandboxSession.execute` does today.

### Failure handling and leaks
- **Normal exit:** the `with` block stops the container, and `--rm` deletes it.
- **Harness crash or kill:**
  - On startup and at exit (`atexit`, SIGTERM handler), the harness runs `docker rm -f $(docker ps -q --filter label=arc-run=<run id>)`.
  - A manual sweep, `docker ps -q --filter label=arc-run | xargs -r docker rm -f`, is documented.
- **Concurrency:** one container per in-flight attempt (≤ workers), each limited to 4 GB memory and 1 CPU. 30 workers means at most about 30 containers plus short-lived checkers. Nothing accumulates.

## Changes
| File | Change |
|---|---|
| `evals/sandbox_docker/repl.py` (new) | The REPL above, about 80 lines |
| `evals/sandbox_client.py` | Replace HTTP `SandboxSession` with `AttemptSandbox` (same interface: `execute`, `checker`, context manager, `train`/`test_input`, `max_output`); keep `tool_output` |
| `evals/run_deepseek_tools.py` | `--sandbox-url` → `--sandbox-image`; drop `check_health` and the HTTP timeout logic; add the run label and sweep; bump `HARNESS_VERSION` to persistent-v4 |
| `evals/tests/test_deepseek_tools.py` | Docker tests use `AttemptSandbox`. Add tests for: state persists; a timeout keeps state; a hung cell causes a reset; a container is gone after the block (also on exception); `/tmp` is not shared between two attempts; 4 GB memory limit; no network; not root |
| `evals/sandbox_docker/start.sh`, `main.py` | Kept for reference/rollback, no longer used by the harness |

## Compatibility and risk
- **Model-visible behaviour is unchanged:** same tools, same preloaded variables, same `[completed]/[error]/[timeout]` outputs.
- **One difference:** a plain Python REPL instead of IPython. Checked on all 24,532 `run_python` cells from E004:
  - 0 use `%` magics and 0 use `!` shell commands;
  - 920 (3.8%) rely on IPython printing the value of the last expression, so `repl.py` must keep that (printed with `repr`, as IPython does for plain values).
- **Results comparability:** bump the harness version. Re-run one full grid/lean2/low sample (~$5.7) to confirm accuracy is unchanged before comparing with older runs.
