# AIMO sandbox review for the ARC Python tool

Probes ran on scratch copies (system `python3` 3.11, numpy 2.1.3; 24 cores, 62 GB).

## Baseline: `evals/run_deepseek_tools.py:55-73`
Each call is a fresh `unshare -rn python3` with `RLIMIT_AS` 2 GB set in `preexec_fn`, a 30 s timeout and an 8k head-only cap. A call takes about 70 ms. Gaps measured:
- `x*2` as the last line prints `[no output]`.
- Tracebacks show `/tmp/tmpXXXX/main.py, line 6` for user line 2: the 4-line `PRELUDE` shifts line numbers and exposes the temp path.
- A `sleep` child started with `Popen` survives the timeout.
- Partial stdout is lost on a timeout.
- No state is kept between calls.

## 1. NeMo-Skills IPython server (AIMO3 main line, Kaggle-proven)
**Files:**
- `AIMO3/scripts/sandbox_server.py`. Byte-identical copies are in `src/solvers/`, `big_inference_run/scripts/`, `autoaimo/autoaimo/`, `AIMO-proof-pilot/imo-inference/evaluation/harness/sandbox/` and `chankhavu-kaggle-solution/`.
- Client: `AIMO3/src/solvers/sandbox_client.py`.
- Upstream version: `AIMO3/evals/Skills/nemo_skills/code_execution/local_sandbox/local_sandbox_server.py`, with Docker files in `evals/Skills/dockerfiles/`.

**Execution model:**
- A Flask `/execute` endpoint is keyed by an `X-Session-ID` header.
- Each session is a forked `mp.Process` holding a `TerminalInteractiveShell`, reached over an `mp.Pipe` (`shell_worker`, `:77-162`).
- State persists and the last expression is shown through IPython's displayhook.
- `postprocess_output` (`:391-412`) strips ANSI codes, `File <ipython-…>` frames and `Out[n]:`.
- The client replays successful cells after a restart (`sandbox_client.py:181-253`).

**Timeouts** (`run_cell`, `:206-353`): poll, send SIGINT, wait a 0.5 s grace, then terminate and start a fresh shell:
```python
proc.send_signal(signal.SIGINT)          # :288
proc.terminate(); proc.join(timeout=2.0) # :324-330
self.start_shell(shell_id)               # :341
```

**Isolation:**
- `RLIMIT_AS` of 2×50 GiB on the whole server (`:503-507`).
- A Python socket patch (`:82-101`), plus `libblock_network.so` through `/etc/ld.so.preload` in Docker.
- `setsid` + `killpg` is used only on the stateless `python` path (`:547-579`). IPython shells are never killed by process group.

**Output:** head-only truncation, defaulting to 1000 characters with an `<output cut>` marker (`:688-691`).

**Concurrency:**
- Docker: `NUM_WORKERS=nproc` uWSGI workers behind nginx, using `hash $http_x_session_id consistent`.
- Kaggle: plain `python sandbox_server.py` started with `Popen(start_new_session=True)` and a health poll (`last_chance/prod_notebook_export.py:205-232`), loaded from a Kaggle dataset (`:61-62`). It runs offline and needs flask, IPython and psutil.

**Probe results (scratch copy):**

| probe | result |
|---|---|
| warm call | 1.1 ms |
| 60 new sessions + numpy/scipy exec | 4.4 s; **~33 MB USS** (83 MB RSS) per shell |
| `y=5; 1/0`, then `y` | `5`: partial state kept |
| timeout of `while True` **or `time.sleep`** | shell dies, **all state lost** |
| replay after restart | **failed**: cell `y` depended on the errored cell, which is not replayed |
| child process | orphaned to PID 1 |

The shell dies because SIGINT kills `TerminalInteractiveShell` when cells run programmatically. Upstream's `signal(SIGINT, SIG_DFL)` kills the shell too.

This matters because `writeup_prize/rebuttal.md:43` attributes **88-94 % of wasted tool time to 30 s timeouts**, and the failed replay explains the "rollback" NameErrors reported there.

**Fix, verified on the scratch copy:** add `signal.signal(signal.SIGINT, signal.default_int_handler)` after creating the shell. Both timeout cases then return `KeyboardInterrupt` and `z` survives.

## 2. Jupyter-kernel pool (`AIMO3/scripts/44-50-let-me-over-cook.py`)
**Implementation:**
- `AIMO3Sandbox` (`:275-444`) runs one `KernelManager` per worker, with hand-assigned ZMQ ports starting at 50000.
- The pool is a `queue.Queue` (`:661-679`). An attempt takes a kernel at `:776` and returns it at `:882` after `%reset -f` and the preimports.

**Execution:**
- `execute` (`:343-411`) collects iopub `stream`, `error` and `execute_result` messages, and cleans tracebacks (`:329-341`).
- `_ensure_last_print` (`:467-488`) is a naive `print(` wrapper.

**Weaknesses:**
- A timeout calls `interrupt_kernel()` and **returns immediately** (`:363-366`). If the interrupt does not land, the kernel stays busy and the next attempt queues behind it. Nothing restarts it.
- No memory limit and no network block: Cascade 2 made HTTP calls in 2.4 % of traces (`writeup/AIMO3 a collection of delightful side quests.md:209`).
- Each kernel costs roughly 1 s and 60-80 MB to start (estimated, not measured).

## 3. `SafePythonSession` (`AIMO-proof-pilot/opd-image/ycchen-proof-pilot-codes/evaluation/harness/tools/`)
**Execution model:**
- `safe_session.py:27-81` keeps one long-lived `python -u _exec_driver.py` per session, speaking JSON lines over stdin/stdout. There is no server.
- The parent applies a `select` wall clock and kills on expiry (`:55-58`), then respawns lazily on the next call.

**Inside the driver:**
- `RLIMIT_AS` is set before imports.
- `SecureLightweightPythonSession` (`python_tool.py`; its docstring says it was written "for ARC-AGI") interrupts with in-process `SIGALRM` (`:206-209`) and keeps the namespace.
- Last-expression display uses `ast` (`:245-267`):
```python
if isinstance(tree.body[-1], ast.Expr):
    exec(compile(ast.Module(tree.body[:-1], []), '<stdin>', 'exec'), ns)
    result = eval(compile(ast.Expression(tree.body[-1].value), '<stdin>', 'eval'), ns)
```
- Imports such as `os`, `sys`, `subprocess` and `pandas` are blocked, and the builtins `open`, `eval`, `exec` and `getattr` are set to `None` (`:53-109`).

**Measured:**

| probe | result |
|---|---|
| warm call | 0.07 ms |
| `while True` / a long numpy `matrix_power` | both interrupted at 2 s, **state kept** |
| 3 GB `np.zeros` under a 2 GB cap | clean `MemoryError` |
| `getattr(x,'shape')` | confusing `'NoneType' object is not callable` |
| `import sys` | blocked, so `setrecursionlimit` is unavailable |

`stderr` goes to DEVNULL, and child processes are not killed by process group.

`README.md:40` records that the in-thread version once crashed the host at **concurrency 400**.

**Benchmarks:** neither repo has any beyond this.

## Recommendation
**(a) Now (API, up to ~300 attempts): combine #3's process model with #1's executor.** Write `evals/pysandbox.py` with one persistent worker per attempt, launched as `[unshare -rn] python -u driver.py` with `start_new_session=True`, speaking JSON lines (no HTTP).
- **Copy from #3:** the driver/session split, the `select` backstop, lazy respawn, and `RLIMIT_AS` set inside the driver. That replaces `preexec_fn`, which is not thread-safe in our 64-thread pool.
- **Copy from #1:** the `shell_worker` loop (IPython `InteractiveShell.run_cell`, displayhook output, `postprocess_output`) plus the SIGINT fix above. If we want to avoid the IPython dependency, use #3's `ast` display instead.
- **Drop:** #3's restricted builtins and #2's kernel pool.

**Timeouts:**
1. Send SIGINT to the worker; state is kept.
2. If it is still busy after about 2 s, `os.killpg(pgid, SIGKILL)` and respawn.
3. Replay **all** non-timeout cells, errored ones included, with output suppressed. ARC code is deterministic, so this reproduces partial state; it fixes #1's divergence.
4. Return partial stdout plus a note saying whether state was kept or reset.

**Load:**
- Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1. Otherwise 300 workers × 24 BLAS threads oversubscribe the CPU.
- Put a global semaphore of about 20 around execution, so contention does not cause spurious wall-clock timeouts.
- - Memory: about 33-50 MB per idle worker × 300 is roughly 10-15 GB, which fits. Spawn on the first tool call and kill at the end of the attempt.

**Preload once per session:**
- `train` and `test_input` as lists and as `np` arrays.
- `numpy as np`, `scipy.ndimage`, `itertools`, `collections`.
- `np.set_printoptions(linewidth=250, threshold=10**6, legacy="1.25")`. numpy 2 otherwise prints `[np.int64(3)]` and wraps grid rows.

**Output and prompt:**
- Truncate head + tail, since errors come last.
- Tell the model that state persists.

**(b) Later (Kaggle offline, 4×L4):** the same module needs only the stdlib, numpy, scipy and IPython, which ships with Kaggle images. It needs no Flask or ZMQ.
- Make `unshare` optional: user namespaces may be disabled, and the box is offline anyway.
- Size the semaphore to `os.cpu_count()` and check the container's RAM.
- Reuse #1's launch pattern: `Popen(start_new_session=True)`, a health/ready poll and a log file.
