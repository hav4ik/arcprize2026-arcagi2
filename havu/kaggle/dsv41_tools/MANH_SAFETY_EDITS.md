# Safety edits for Manh's submission (2026-10-29, for v8 = scriptVersionId 363632080)

Five small edits to `nguyennguyen599/dsv41-tools-submit-b32m-external-server` v8. None adds new code, and each
answers something measured tonight on Runpod 4xL4 pods. Checked against v8: the settings cell reaches the solver
(cell 2: `BASE_ENV = dict(os.environ)`), and none of the new keys are in cell 0b's `_MT_EXPECT`.

## 1. Cell 00 (settings): job and connection retries

Add two entries to the `for _k, _v in ((...), ...)` settings tuple, for example right after `("ARC_PORT", "18084"),`:

```python
               ("ARC_JOB_TRIES", "3"),     # a server death costs every in-flight attempt one try (was 2)
               ("ARC_CONN_TRIES", "12"),   # free re-queues across several restarts (was 4)
```

**Why (pod kill test, 22:06 UTC):** we SIGKILLed one TP worker with 63 requests running, under a restart supervisor.
- The API server exits on its own 13 s later. The supervisor relaunched the server, and it was up again in 150 s.
- In the second before it exits, the dying server answers **every in-flight request with a plain HTTP 500**:
  `APIError: The server had an error while processing your request.`
- Your pinned solver (`11bfaaaa`, PR #2) does not see that as a connection loss. It only matches `EngineDeadError` /
  "EngineCore encountered an issue", which are in server.log only. So each in-flight attempt used 1 of its 2
  `ARC_JOB_TRIES`.
- One attempt was lost: its retry went 0.3 s later into the same dying server, got another 500, and was finalized
  `stop=error`.
- On a second death, every attempt then in flight would be lost.

With 3 tries and 12 connection re-queues, attempts survive two deaths. Our solver (`1ea2b25d`, in
`out/nb_dsv41-tools-submit-b32m-fix`) fixes the cause instead: that 500 is a connection error there.

## 2. Cell 2m (`MANH_SERVER_ENV`): step timeout

```python
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",  # default 300: a slower step kills the engine
```
Deadlocks still trap at 600 s (host all-reduce / all-gather `__trap()`). This only keeps a slow but progressing
step from becoming an engine death, which costs a 3-25 min restart plus the in-flight attempts.

## 3. Cell 2m: GPU memory utilization 0.935 -> 0.92

```python
                "--concurrency", str(MANH_SEQS), "--gpu-memory-utilization", "0.92", "--kv-offload-gib-per-rank", "10",
```
- Measured free VRAM per GPU: ~1.0 GiB at 0.94, 2.0-2.5 GiB at 0.90. B32 (LB 68.89) had 2.7 GiB; B32-M (LB 46.5)
  had 1.1 GiB.
- Your re-prefill OOM finding needs that margin.
- 0.92 is ~+0.35 GiB free per GPU for ~5% of the KV pool.

## 4. kernel-metadata.json: pin the wheels

```json
  "dataset_sources": ["nguyennguyen599/arc2-2026-src", "nguyennguyen599/cu13-install", "chankhavu/dsv41-tools-kit",
                      "nguyennguyen599/unsloth-main", "nguyennguyen599/vllm-wheels/11"],
```
- Use the version you tested. v11 = vLLM `g8cf0c0664b`, 1,669 files, 9.17 GB, checked via the API.
- Kaggle attaches a dataset's latest version at push unless it is pinned.
- Also make sure the latest `arc2-2026-src` is the one whose `dsv41_vllm.py` has `--supervise`. Otherwise cell 2m
  prints a warning and runs one start with no restarts.

## 5. Don't change `ARC_WORKERS` alone

Cell 0b's `_MT_EXPECT` pins `ARC_WORKERS` (80) and `ARC_CONTEXT_LIMIT` (163840). If the settings cell differs, commit
runs raise, and **the scored rerun silently falls back to the v3 settings (36 workers)**. Change both or neither.

## After the commit: check the log for

- `E046 CHECK OK` (cell 0b: no fallback);
- the wheel cell 1 installs: `vllm-0.32.1rc1.dev0+g8cf0c0664b`;
- cell 2m: the supervisor start line and `MANH_SERVER_STATUS`;
- the solver: `server up after ...s (external, start 0); GPU KV cache size: ...`;
- cell 4: "supervisor: stopped", then the server stopped;
- the smoke verdict: OK.

## Evidence

- Kill test timeline and RAM samples: `experiments/E062_manh_vllm_fork/per_layer/pod3_runs/README.md`.
- Decode numerics (your decode flags on vs off: identical, fp noise):
  `experiments/E062_manh_vllm_fork/per_layer/phase_h/README.md`.
- B32-M postmortem (46.5 fits a run that stopped answering at Kaggle hour 4-6) and the RAM/VRAM comparison:
  `kaggle/dsv41_tools/CANDIDATE_E066_IMG.md` and `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/README.md`.
- Our full candidate on your v8: `kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m-fix/`. It has all of the above
  plus the solver fix, util 0.90, 64 workers and a light-smoke commit with fail-closed checks.
