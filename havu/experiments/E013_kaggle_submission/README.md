# E013: Kaggle submission v1, Qwen3.8-27B-FP8 on 4xL4, no tools

Code: `kaggle/` (see `kaggle/README.md`). Started 2026-10-11.

## Goal
A first valid offline submission to `arc-prize-2026-arc-agi-2` with Qwen3.8-27B (official FP8 weights), using the
E008 no-tool setup (ARC Prize prompt, tier xhigh, temp 1.0, top_p 0.95, top_k 20) under the Kaggle limits:
4xL4 (24 GB, SM89), 12 h, no internet, 240 tasks / 259 test inputs.

## Inputs and provenance (2026-10-11)
- Model: `mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1`. All 81 files of `Qwen/Qwen3.8-27B-FP8` rev 017b9c7
  match HF (sha256 for LFS files, git blob sha1 for the rest), hashed inside Kaggle. One extra file
  (`mirror_manifest.json`), not read by vLLM.
- Wheels: `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` (vLLM 0.19.0, torch 2.10.0, cp312). All 174 wheels match
  PyPI's sha256; no custom builds. Report: `kaggle/provenance/report.json`.
- Docker: Kaggle's default image is now Python 3.13, so the notebook pins a Python 3.12 image (CPU torch preinstalled;
  the CUDA torch from the wheelhouse is force-reinstalled).

## Token budget reasoning
From E008 (xhigh, 240k budget), accuracy vs token cap: 48k -> pass@1 21%, 64k -> 28.5%, 96k -> 34.5%, 128k -> 35.8%.
Two attempts at 28k (10.8% pass@2) are much worse than one at 64k, so the scheduler gives every pair one attempt with
budget = remaining capacity / remaining pairs (clamped to 16k-64k in the submit notebook), and adds a second attempt only with spare capacity.

## Runs

### Bench runs on Kaggle 4xL4 (2026-10-11, notebooks `chankhavu/arc2-qwen38-27b-bench-mtp{3,0}` v3)
vLLM 0.19.0, TP=4, FP8 weights, FP8 KV (`TRITON_ATTN`), prefix caching (vLLM sets Mamba cache mode `align`,
attention block size 1600, "experimental"), `max_model_len` 131072, `max_num_seqs` 64, gpu util 0.95.
Outputs in `kaggle/runs/bench-mtp{3,0}-v3/`. Each bench used ~40 min wall time.

| | MTP=3 | MTP=0 |
|---|---|---|
| Startup: pip install wheelhouse + CUDA torch reinstall | 3.7 min | 3.6 min |
| Hash 31 GB model + wheels | 1.5 min | 1.3 min |
| vLLM server ready (load + compile + CUDA graphs) | 8.5 min | 6.9 min |
| KV cache (reported) | 304k tokens | 376k tokens |
| Concurrency used (KV / 30k) | 16 | 16 |
| Prefill, 16 prompts x 40k tokens | 474 s (1.36k tok/s) | 437 s (1.47k tok/s) |
| Decode at 40k context, 16 seqs, steady | **227 tok/s** | **195 tok/s** |
| E2E scheduler, 16 seqs, real ARC prompts, contexts 2-20k | 300-366 tok/s window, 295 tok/s avg over 16 min | 213-262 window, 233 avg |
| Peak server-reported generation throughput | 442 tok/s | 269 tok/s |
| MTP acceptance | 68-99% per draft (mean accepted length 3.0-3.97 of 4) | - |
| Errors / crashes / preemptions | 0 / 0 / 0 (peak KV use 80%) | 0 / 0 / 0 (peak KV 50%) |
| E2E (15-min deadline, all forced, 48 eval tasks / 73 pairs) | 21 attempts, 21 parsed, 2 pairs correct | 16 attempts, 15 parsed, 0 correct |

Takeaways:
- Throughput is ~250 tok/s in a real run (MTP=3), i.e. ~9-10M generated tokens in the ~10.7 h left after startup:
  about 36k tokens per test pair for one attempt. E008's cap table puts that at pass@1 ~10-14% (28k: 6.9%, 48k: 21%).
- Concurrency is capped by the KV cache: Mamba `align` prefix caching uses 1600-token blocks that each hold a Mamba
  state, so only ~300k tokens fit. Disabling prefix caching (not tested on Kaggle) should give several times more KV
  and allow much higher concurrency; it is the main untested lever. Speed work was stopped by user direction.
- Provenance check passed (see above). Submission files from all runs validate against the test challenge file.

### Submission notebook (built, not pushed; awaiting go-ahead)
`kaggle/nb_submit/` (slug `chankhavu/arc2-qwen38-27b-submit`): MTP=3, concurrency 16 from KV, `ARC_RATE_PRIOR` 250,
attempt-1 budget = capacity share clamped to [16k, 64k], attempt 2 only with >= 32k spare per pair, force at 11h00m,
hard stop 11h36m. Commit run = fallback only (seconds); full run only on the competition rerun.

### Own wheel sources + SGLang attempt (2026-10-11, user direction: no third-party wheelhouses)
- Built `chankhavu/sglang-0-5-20-py312-cu130` (CPU builder, internet on): 208 wheels, 6.66 GB, offline dry-run
  resolve OK. vLLM path switched to the user's `chankhavu/vllm-0-23-0-py312-cu128`.
- Offline SGLang test `chankhavu/arc2-sglang-bench` (internet off; ~13 GPU-min): install 405 s; torch 2.13.0+cu130
  sees 4 L4s (driver 580.178.04). The **server fails at the first forward pass**: sglang-kernel JIT-compiles the
  fp8 `per_token_group_quant` kernel with nvcc, and the image has no `/usr/local/cuda/bin/nvcc`. Same with radix
  off. Pools reported before the crash (TP=4, fp8_e4m3 KV, mem fraction 0.85):
  - radix on: 772k KV tokens + 146 Mamba slots (5.2 GB SSM state); radix off: 1.16M KV tokens + 64 Mamba slots.
  - vLLM 0.19 with prefix caching: 304k KV tokens. SGLang would allow far more concurrency once it runs.
  - Fix for later: add an nvcc / CUDA 13 toolkit wheel (`nvidia-cuda-nvcc`, headers) to the source and set
    `CUDA_HOME`, or pre-build the JIT kernels in a GPU builder. Not pursued for the first submission.
- Kaggle queueing: the 4xL4 session started ~28 min after the push.

### v2 = tool agent (E012) + v1 machinery
See `kaggle/README.md` (v2 section). Config from E012: tier medium, soft 24k, hard min(48k, adaptive share),
LocalExecutor, rerun switch to a local self-scored test set (`chankhavu/arc2-local-testcases`, 8 tasks / 13 pairs).
CPU check on the Kaggle image (`chankhavu/arc2-cpu-executor-check`): LocalExecutor timeout / crash restart /
6 GB MemoryError / submit all behave; 16 parallel sessions in 1.4 s; load avg 1.5 on 4 CPUs.

### v2 commit run on Kaggle 4xL4 (2026-10-11 23:18–23:34 UTC): `chankhavu/arc2-qwen38-tools-submit` v2, gate PASS
- Readable notebook (`%%writefile` cells from the repo; `kaggle/tests/check_notebook.py` verifies it). Offline:
  vLLM 0.23.0 from the user's `chankhavu/vllm-0-23-0-py312-cu130` (the cu128 source failed: its vLLM wheel needs
  `libcudart.so.13`), qwen-tools-3 agent + LocalExecutor, verified model mirror.
- Phases (s from the first cell): harness self-test 4, uv install 93 (68 s install), model preload 147, server ready
  562, first token 570. **KV cache 1.40M tokens on vLLM 0.23** (304k on 0.19), concurrency 35 (KV / 40k).
- Local test (4 tasks / 5 pairs, forced at server-ready + 6 min): 2/4 tasks correct (2ba387bc, 1818057f via
  submit_answer at ~2k tokens); the other 3 attempts were cut by the deadline at ~6.9k tokens (no answer -> fallback).
  21 tool calls, 2 tool errors, 0 parse failures, 0 attempt/stream errors, 0 preemptions, peak KV 4.5%.
  Generation ~65 tok/s with 3-5 live sequences (~20 tok/s per sequence).
- submission.json valid (240 tasks / 259 inputs, 0 slots repaired). Backup kernel with the plain agent:
  `chankhavu/arc2-qwen38-plain-submit`.
- Local 2x3090 smoke (SGLang AWQ-INT4, same extracted notebook modules via `ARC_EXTERNAL_SERVER`, 8 tasks / 13
  pairs, concurrency 8, forced at 25 min): 3.67/8 tasks; 11/13 answered (submit 7, forced 3, salvage 1, none 2);
  67 tool calls, 10 tool errors, 0 attempt/stream errors; 202k tokens in 26.9 min (125 tok/s); no leftover
  processes. Guard (scaled clock: process group killed, no children left) and emergency timer (valid file, exit at
  12 s) tested separately.
