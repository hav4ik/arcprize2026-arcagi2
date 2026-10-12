# Kaggle submission: Qwen3.8-27B-FP8, no tools (E013)

Offline code-competition notebook for ARC Prize 2026 / ARC-AGI-2 (`arc-prize-2026-arc-agi-2`, deadline 2026-11-02).
Experiment notes: `experiments/E013_kaggle_submission/README.md`.

## Files

| Path | What |
|---|---|
| `src/solver.py` | The whole pipeline (one file): fallback submission, vLLM install from wheelhouse, server launch, deadline-aware scheduler, benchmark mode. `__TEMPLATE__` is replaced by the ARC Prize prompt at build time. |
| `build_notebook.py` | Builds `nb_<variant>[_<tag>]/` (notebook + `kernel-metadata.json`) from `src/solver.py`. |
| `tests/fake_server_test.py` | Local test of client + scheduler against a fake streaming server (no GPU). |
| `provenance/` | `hf_tree.json` (HF file list + hashes, Qwen/Qwen3.8-27B-FP8 rev 017b9c7), `wheel_list.txt`, `check.py` (compares hashes computed in Kaggle against HF / PyPI / PyTorch index), `report.json`. |
| `datasets/bench_prefixes/` | Private dataset `chankhavu/arc-qwen-bench-prefixes`: 64 Qwen trace prefixes (85k chars) from E008 for the long-context decode benchmark. Used by bench notebooks only. |
| `runs/` | Downloaded outputs of Kaggle runs (bench results, logs). |
| `src/bench_sglang.py` | Offline SGLang throughput test (radix cache on vs off); notebook `nb_sglang_bench/` (`build_notebook.py --variant sglang_bench`). |
| `wheel_sources/` | Wheel-source builder notebooks (see below). |
| `src/replay_bench.py` | E014 agentic replay benchmark (concurrency sweep, prefix-cache hit rate, preemptions); notebook `nb_replay_<server>/` (`build_notebook.py --variant replay --env '{"ARC_SERVER": ..., "ARC_REPLAY_PLAN": ...}'`), dataset `chankhavu/arc2-agentic-replay` (`datasets/agentic_replay/build.py`). Test: `tests/fake_replay_test.py`. |

## Wheel sources (our own; user direction 2026-10-11: no third-party wheelhouses)

Builder notebooks run on CPU with internet and download wheels into their output; GPU notebooks attach them as
`kernel_sources` and install offline (`pip install --no-index --find-links <dir>`), with internet off.

| Source notebook | Contents | Builder |
|---|---|---|
| `chankhavu/vllm-0-23-0-py312-cu128` (user's) | vLLM 0.23.0, torch 2.11.0+cu128, flashinfer 0.6.12 (+ jit-cache cu128) | copy in `wheel_sources/vllm-0-23-0-py312-cu128.ipynb` |
| `chankhavu/sglang-0-5-20-py312-cu130` (v3, E014) | v3 = same 208 wheels (SHA256 identical to v1) + `sglang_jit_cache/sm89` + `toolchain_check.json` (nvcc JIT for sm_89 verified on CPU). SGLang 0.5.20 + 208 wheels (6.66 GB): sglang-kernel 0.4.7, torch 2.13.0 (CUDA 13.0 libs), flashinfer 0.6.18 + cubin + jit-cache cu130, triton 3.7.1, transformers 5.12.1; cuda-tile 1.6.0rc5 built from its sdist. `requirements.lock`, `SHA256SUMS`, `manifest.json`, offline dry-run resolve verified in the builder. | `wheel_sources/build_sglang_source.py` |

Both use the Python 3.12 image `gcr.io/kaggle-images/python@sha256:e5452ce6...` (the user's vLLM source image).
The host driver (580.x) supports CUDA 13.0, which SGLang 0.5.20's torch 2.13 needs.

## Kaggle inputs (first bench round used a third-party wheelhouse; superseded by the wheel sources above)

- Model `mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`: mirror of `Qwen/Qwen3.8-27B-FP8` at revision 017b9c7; verified against HF hashes (see provenance).
- Dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`: vLLM 0.19.0 + torch 2.10.0 cp312 wheels, `requirements.lock`.
- Docker image pinned to a Python 3.12 Kaggle GPU image (`build_notebook.DOCKER`). The current default image is Python 3.13, and the cp312 wheelhouse fails to install there.

## How it runs

1. First cell records `ARC_T0` (all deadlines are relative to it) and picks the mode: `submit` on the competition rerun (`KAGGLE_IS_COMPETITION_RERUN`), else the variant default (`fallback` for the submit notebook, so the commit run takes seconds; `bench` for test notebooks).
2. A complete fallback `submission.json` (test input copied into both attempts) is written before anything else, then rewritten atomically after every answered attempt. A `finally` block re-validates it and rewrites the fallback if it is broken.
3. vLLM: TP=4, FP8 weights, FP8 KV cache with `TRITON_ATTN` (FlashInfer's fp8 KV kernels need SM90+; L4 is SM89), prefix caching, `max_model_len` 131072, optional MTP speculative decoding (`ARC_MTP`).
4. Prompts: ARC Prize official prompt, Qwen chat template rendered by hand (tier xhigh system message, `<think>` opened), `/v1/completions` streaming, temp 1.0 / top_p 0.95 / top_k 20.
5. Scheduler: concurrency from the KV-cache size; attempt-1 jobs for every test pair (shortest prompts first), each with a budget = remaining capacity / remaining jobs, clamped to [16k, `ARC_B_MAX1`] (solver default 96k; the submit notebook sets 64k); capacity = measured tok/s × time to the force deadline. Attempt 2 is queued per pair only if it can still get >= 32k tokens. Mid-thought EOS: continue; after 2 low-yield continuations or at the budget/deadline, append "I have to give my final answer now.\n</think>" and allow 3k tokens.
6. Deadlines (from notebook start): no new attempts after 10h58m, force final answers at 11h00m (`ARC_FORCE_AT_S`), hard stop at 11h36m (`ARC_HARD_STOP_S`).

## Commands

```bash
export KAGGLE_API_TOKEN="$(sed -n 's/^kaggle_api_token:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")"
.venv/bin/python kaggle/tests/fake_server_test.py
.venv/bin/python kaggle/build_notebook.py --variant bench --mtp 3 --tag mtp3 && .venv/bin/kaggle kernels push -p kaggle/nb_bench_mtp3
.venv/bin/python kaggle/build_notebook.py --variant submit --env '{"ARC_MTP": 3, "ARC_RATE_PRIOR": 250, "ARC_B_MAX1": 64000}'
.venv/bin/kaggle kernels push -p kaggle/nb_submit   # commit run = fallback only; needs a GPU session (approval)
.venv/bin/kaggle kernels output chankhavu/<slug> -p kaggle/runs/<slug>
.venv/bin/python kaggle/provenance/check.py kaggle/runs/<slug>/provenance_hashes.json
# competition submission (uses a daily slot; needs approval):
.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-2 -k chankhavu/arc2-qwen38-27b-submit -v <version> -f submission.json -m "<msg>"
```

## Measured on Kaggle 4xL4 (2026-10-11)
Decode 227 tok/s at 40k context x 16 sequences with MTP=3 (195 without), ~295 tok/s in the end-to-end scheduler at
shorter contexts, prefill ~1.4k tok/s, KV cache ~304k tokens (prefix caching puts the hybrid model in Mamba `align`
mode), server start 8.5 min, total startup ~14 min. Details: `experiments/E013_kaggle_submission/README.md`.
Kaggle GPU quota is 6 h/week for this account (API: `get_accelerator_quota_statistics`); benches used ~2.9 h.
Kernel sessions cannot be cancelled through the API (403); stop them in the web UI.

## v2: tool agent (E012) + v1 machinery (2026-10-11)

- Notebook `nb_v2/` (slug `chankhavu/arc2-qwen38-tools-submit`), built by
  `build_notebook.py --variant v2 --env '{"ARC_SERVER": "sglang"}'` (or `"vllm"` for the user's vLLM 0.23.0 source).
  The E012 harness (`evals/arc_harness/{agent_qwen,executor,arcprize,grids,prompts}.py` + the ARC Prize prompt) is
  embedded in the notebook and written to `/kaggle/working/pkg/` (same relative layout); no extra dataset.
- **Rerun switch** (pattern from the user's `c2-distill-vllm018-watchdog`): `KAGGLE_IS_COMPETITION_RERUN` set ->
  hidden test set with the full 11 h budget; otherwise (commit / interactive) -> the private local test set
  `chankhavu/arc2-local-testcases` (8 public eval tasks / 13 pairs with solutions: 1818057f, 2ba387bc, dd6b8c4b,
  9aaea919, 1ae2feb7 solved by Qwen tools on the Vast box; 58f5dbd5, 78332cb0, 221dfab4 not), self-scored and
  time-boxed (`ARC_LOCAL_FORCE_MIN`, default 40 min from notebook start), so the commit run is the e2e validation.
  It still writes a valid fallback `submission.json` for the public test file.
- **Agent:** `agent_qwen.solve` with E012's Kaggle config: tier medium, soft target 24k, hard cap
  min(48k, adaptive per-pair share), budget notes on every tool output, warn at 75%, verify_submit off, 60 turns,
  16k chunks, 256-token forced submit, temp 1.0 / top_p 0.95 / top_k 20, `LocalExecutor` (child process, 30 s
  timeout with SIGINT then kill/restart, 4 GB RLIMIT_AS, BLAS threads 1), one executor per attempt, always closed.
  The loop's default thread pool is set to concurrency + 8 threads (executor calls go through `asyncio.to_thread`).
- **Deadline forcing for tools:** `DeadlineLLM` wraps our streaming client. Past the force time it cuts the stream
  and reports the attempt budget as spent, so the agent does its forced `submit_answer` (allowed until the hard
  stop), then falls back to the session `answer`, then parsed text, then the fallback copy of the test input.
- Server: SGLang 0.5.20 or vLLM 0.23.0, both from our own wheel sources, chosen by `ARC_SERVER`. The client is
  server-agnostic (`/v1/completions`, streaming, `stop`, continuous usage when available).
- Tests (no GPU): `tests/fake_tools_test.py` drives the tool agent end to end against a scripted fake server
  (submit, executor crash + timeout, malformed call -> parsed text, cap -> forced submit, deadline cuts, server
  down); `tests/fake_server_test.py` covers the v1 plain path. CPU-only Kaggle check of LocalExecutor + offline
  SGLang install: notebook `chankhavu/arc2-cpu-executor-check`.

## Status (2026-10-11 23:50 UTC)

- **Submittable:** `chankhavu/arc2-qwen38-tools-submit` v2 (tools; commit gate PASS, local 2/4) and backup
  `chankhavu/arc2-qwen38-plain-submit` v1 (plain; gate PASS, local 1/4). Both use vLLM 0.23.0 from
  `chankhavu/vllm-0-23-0-py312-cu130`. The cu128 source is not usable: its vLLM wheel needs `libcudart.so.13`.
  Both have internet off.
- Build + check: `build_notebook.py --variant v2 --env '{...}'` (settings in `experiments/E013_kaggle_submission`),
  then `tests/check_notebook.py --extract-to DIR` (nbformat, nbconvert, sources == repo, imports, fake tests on the
  extracted modules). Off-Kaggle run of the same modules: `ARC_EXTERNAL_SERVER=http://host:port/v1 ARC_MODEL_ID=...
  ARC_WORK=... ARC_INPUT=...`.
- Submit (needs the user's OK; uses a daily slot):
  `.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-2 -k chankhavu/arc2-qwen38-tools-submit -v 2 -f submission.json -m "..."`
- SGLang 0.5.20 (own source) did not start on Kaggle: sglang-kernel JIT needs nvcc (`/usr/local/cuda/bin/nvcc`
  missing). Fixed in E014 without new wheels: `solver.cuda_toolkit_env()` points CUDA_HOME at a `/tmp/cuda13` shim
  over the pip CUDA 13 toolkit already in the source (verified on CPU: the failing kernel compiles for sm_89).
  GPU replay runs pending (sessions pushed during the competition rerun got 2x T4; see E014).
- Solver: TP from the GPUs present (`choose_tp`, `ARC_TP`), cache hit rate / KV usage / preemptions in the progress
  lines (vLLM `/metrics`, SGLang `/v1/loads`).

## Submission watcher (`monitor_submissions.py`)

Kaggle does not show how long a code-competition rerun has been going, so `monitor_submissions.py` polls our
submission list (read-only; it never submits or cancels) and times each one from its Kaggle submission time.
Auth: `KAGGLE_API_TOKEN` from the environment, else read from `SECRETS.yaml` in-process (never printed).

```bash
.venv/bin/python kaggle/monitor_submissions.py --once            # current table: id, submitted_at, elapsed, status, scores
nohup .venv/bin/python -u kaggle/monitor_submissions.py --until-done --interval 300 \
    >> kaggle/runs/submission_watch.log 2>&1 < /dev/null &  echo $! > kaggle/runs/submission_watch.pid
tail -f kaggle/runs/submission_watch.log                          # NEW / DONE / WARNING lines, heartbeat every 30 min
kill "$(cat kaggle/runs/submission_watch.pid)"                    # stop
.venv/bin/python kaggle/tests/test_monitor_submissions.py         # offline unit tests
```

- `kaggle/runs/submission_watch.jsonl`: one record per observed status change (id, description, submitted_at,
  status, observed_at, elapsed_h, scores, error). A finished run gets `wall_h_lower`/`wall_h_upper`: the
  completion happened between the last PENDING poll and the first non-pending poll (one interval of uncertainty;
  after a watcher restart the lower bound is looser).
- Flags: `--interval` (s, min 60, default 300), `--until-done`, `--warn-hours` (default 11.0; WARNING once when a
  pending run exceeds it), `--guard-hours` (11.75, message only), `--heartbeat-min` (30). API errors retry with
  backoff 30 s doubling to 30 min.
- Elapsed is measured from submission, so it includes queueing before the notebook's first cell; the notebook's
  11.75 h self-stop therefore fires at ~11.75 h elapsed or a bit later.
