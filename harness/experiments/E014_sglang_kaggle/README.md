# E014: SGLang on Kaggle 4xL4 with prefix caching, KV offload, and concurrency vs. cache hit rate

Started 2026-10-12. Code: `kaggle/src/replay_bench.py`, `kaggle/src/solver.py` (SGLang path),
`kaggle/wheel_sources/build_sglang_source.py`, `kaggle/datasets/agentic_replay/build.py`, `kv_working_set.py` (here).

## Hypothesis
SGLang 0.5.20 (own wheel source) runs Qwen3.8-27B-FP8 on Kaggle 4xL4 once its JIT kernels can find nvcc, and with
the radix (prefix) cache on it serves the E012 tool agent at least as fast as vLLM 0.23 (the current submission).
Host-memory KV offload (SGLang hierarchical cache) may let us run more agents without eviction thrash.

## Constraints (user, 2026-10-12)
- Prefix caching is mandatory (each tool turn re-sends the whole transcript); radix OFF is not a configuration.
- Concurrency must keep the prefix-cache hit rate high and stable over a long run (AGENTS.md "Self-hosted serving
  for tool use").
- GPU budget: at most 2 h of 4xL4 this week for E014; offline GPU runs only, own wheel sources, hash-verified model.
  Builder notebooks run on CPU. Do not touch `chankhavu/arc2-qwen38-tools-submit`.

## Setup

### 1. Why SGLang died on L4, and the fix (no GPU needed to find it)
- First L4 run (`chankhavu/arc2-sglang-bench`, E013): sglang-kernel JIT-compiles the fp8 `per_token_group_quant`
  kernel with `/usr/local/cuda/bin/nvcc`, which the image lacks.
- Our wheel source already contains the CUDA 13 pip toolkit: nvidia-cuda-nvcc 13.4.92, nvidia-cuda-crt 13.4.92,
  nvidia-nvvm 13.4.92, nvidia-cuda-cccl 13.3.4, nvidia-cuda-runtime 13.0.96 (headers + libcudart.so.13), all under
  `site-packages/nvidia/cu13`. SGLang's JIT resolves `$CUDA_HOME/bin/nvcc` and links with
  `-L$CUDA_HOME/lib64 -lcudart`, but the wheels ship `lib/` with only `libcudart.so.13`. So the fix is an
  environment shim, not new wheels: `solver.cuda_toolkit_env()` builds `/tmp/cuda13` (symlinks `bin`, `include`,
  `nvvm`, `lib`, and `lib64/` with `libcudart.so -> libcudart.so.13`) and sets `CUDA_HOME`/`CUDA_PATH`/`PATH`/
  `LD_LIBRARY_PATH`, `TORCH_CUDA_ARCH_LIST=8.9`.
- JIT cache: `SGLANG_JIT_CACHE_DIR=/tmp/sglang_jit`, seeded from `sglang_jit_cache/sm89` of the wheel source if
  attached. The cache key is target + `nvcc --version` + `c++ --version` + package versions + anchor-normalized
  build file, so a cache built in the same image and install layout is reusable.
- Wheel source v2 (`chankhavu/sglang-0-5-20-py312-cu130` version 2, CPU builder): same 208-package lock (download
  constrained to the v1 lock, `kaggle/wheel_sources/sglang-0-5-20-v1.requirements.lock`), then installs offline
  into the CPU image and compiles the failing kernel for sm_89 through SGLang's own loader
  (`override_jit_cuda_arch(8, 9)`); result in its `toolchain_check.json`.

### 2. Hybrid-model memory on SGLang (from the first L4 log)
Qwen3.8-27B: 64 layers, 16 full-attention (fp8 KV: 32 KB/token over TP=4, 7.6 KB/token/GPU) and 48 Gated-DeltaNet
layers whose recurrent state is fp32 (`mamba_ssm_dtype: float32` in the model config): 35 MB per state slot per GPU.
- Radix ON, defaults (mem fraction 0.85): 772k KV tokens + 146 Mamba slots, and **max_running_requests capped to
  29** ("5 state slots per request": extra_buffer strategy with the overlap scheduler).
- Slots per running request: 5 (extra_buffer, overlap), 4 (`extra_buffer_lazy`, or
  `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`), 3 (both). Fewer slots per request leave more memory for KV.
- Hierarchical cache works with hybrid models in 0.5.20 (UnifiedRadixCache with a MAMBA component + `init_hicache`).

### 3. Offline working-set estimate (`kv_working_set.py`)
From the 341 replay transcripts (E012 medium 48k; mean 33.5k generated tokens, 14 turns, final context 48k, p90
67k): a live agent's context, weighted by time (~generated tokens), has mean 32.2k, p90 54.7k, p99 68.3k tokens.
The KV working set of C desynchronised agents (sum of C draws):

| C | mean | p99 | p99.9 | P(> 1.40M, vLLM 0.23) | P(> 772k, SGLang default) |
|---|---|---|---|---|---|
| 16 | 515k | 670k | 728k | 0 | 0 |
| 20 | 645k | 812k | 874k | 0 | 0.04 |
| 24 | 772k | 958k | 1.02M | 0 | 0.50 |
| 28 | 902k | 1.10M | 1.17M | 0 | 0.94 |
| 32 | 1.03M | 1.25M | 1.32M | 0 | 1.00 |
| 35 | 1.13M | 1.35M | 1.42M | 0.002 | 1.00 |
| 40 | 1.29M | 1.53M | 1.61M | 0.14 | 1.00 |
| 48 | 1.55M | 1.81M | 1.89M | 0.91 | 1.00 |

So the current submission (vLLM, 35 = 1.40M / 40k) sits at ~80 % of its reported KV on average, if vLLM's
reported token count is what the cache can really hold for this hybrid model (vLLM uses 1568-token blocks with
Mamba states in the same pool; to be checked on GPU). SGLang with default sizing can only hold ~24 agents.

### 4. Agentic replay benchmark (`kaggle/src/replay_bench.py`, dataset `chankhavu/arc2-agentic-replay`)
- Every agent replays one recorded attempt: turn t sends prompt_t and asks for exactly the recorded number of
  tokens (ignore_eos, temp 1.0 / top_p 0.95 / top_k 20, 16k chunks); prompt_{t+1} = prompt_t + the server's own
  generated text + the recorded tool output and next assistant header, i.e. what the real agent sends. 1 s tool
  delay. A finished attempt is followed by the next transcript.
- Concurrency ramps within one server start (16 -> 24 -> 35 -> 48 agents, 5 min each); agents joining a level are
  warm-started at a random turn (time-weighted, start context <= 20k) to approach the long-run working set.
- Per level (first 90 s skipped): decode tok/s, new-prefill tok/s, hit rate (client: cached / prompt tokens from
  `usage.prompt_tokens_details`; server: vLLM `/metrics` prefix-cache counters, SGLang `/v1/loads` uncached prefill),
  hit vs. ideal on continuation requests (ideal = previous request's prompt + completion), running / waiting,
  KV usage, vLLM preemptions, SGLang retracted queue, TTFT.
- Tests: `kaggle/tests/fake_replay_test.py` (fake server with a toy prefix cache; no GPU).

### 5. Solver changes (behind `ARC_SERVER`, vLLM default unchanged)
- SGLang launch: CUDA shim + JIT cache, `--enable-cache-report`, `ARC_SGL_MAMBA_SLOTS` / `ARC_SGL_MAMBA_STRATEGY` /
  `ARC_SGL_SKIP_LOCK` / `ARC_SGL_HICACHE` (+ `ARC_SGL_HICACHE_POLICY`), `ARC_SERVER_EXTRA` (extra flags, both servers).
- SGLang capacity from `/get_server_info` (token capacity and the effective running cap after the Mamba limit).
- Progress lines (every 2 min) now include the window's prefix-cache hit rate, running / waiting, KV usage and
  preemptions (vLLM `/metrics`) or retracted queue (SGLang `/v1/loads`); the summary keeps the last counters.

## Runs and GPU budget

| Run | Kernel | Planned min | Used min | Status |
|---|---|---|---|---|
| Wheel source v2 (CPU) | `chankhavu/sglang-0-5-20-py312-cu130` v2 | 0 (CPU) | 0 | lock + SHA256 identical to v1; toolchain check script bug (imported the function, not the module) |
| Wheel source v3 (CPU) | `chankhavu/sglang-0-5-20-py312-cu130` v3 | 0 (CPU) | 0 | **toolchain PASS**: nvcc 13.4.92 via `/tmp/cuda13` compiles, links and loads `per_token_group_quant` (bf16 -> fp8, group 128, row- and column-major scales) for sm_89 in 13 s each; cache in `sglang_jit_cache/sm89`; uv offline install 112 s on CPU. `kaggle/runs/sglang-source-v3/` |
| vLLM replay v1 | `chankhavu/arc2-e014-replay-vllm` v1 | 34 | ~3 | no data: Kaggle stores uploaded `.gz` decompressed (`agentic_replay.json`), bench did not find it; session had 2 GPUs / 31 GB host |
| vLLM replay v2 | `chankhavu/arc2-e014-replay-vllm` v2 | 34 | ~1 | aborted by the new 4xL4 pre-check: **2x Tesla T4 (15 GB), 31 GB host, 4 CPUs** |
| SGLang replay v1 | `chankhavu/arc2-e014-replay-sglang` v1 | 51 | ~1 | aborted by the pre-check: 2x Tesla T4 again (00:54 UTC). GPU pushes paused until rerun 57088016 finishes |

GPU quota: 4.61 h at the start -> 4.66 h after the three aborted sessions (E014 total ~5 GPU-min).

### Machine fallback observed (2026-10-12 00:34, 00:50 and 00:54 UTC)
Both GPU sessions pushed while our competition rerun (submission 57088016, started 23:51 UTC) was running came up
on **2x Tesla T4 + 31 GB RAM + 4 CPUs**, although `kernel-metadata.json` says `machine_shape: NvidiaL4` (the same
metadata gave 4x L4 / 188 GB / 48 CPUs before 23:51). Likely the account's 4xL4 slot is held by the rerun and Kaggle
silently falls back. The replay bench now checks `nvidia-smi` first and exits within a minute unless it sees 4x L4.
Robustness item for the next submission (do not change the submitted kernel): the solver assumes TP=4. It should
read the GPUs actually present and adapt: TP = GPU count; on 2x L4 (48 GB) a TP=2 FP8 config (27 GB of weights
leaves ~14 GB/GPU at util 0.95 for KV + Mamba state, i.e. roughly a third of the 4xL4 KV, so concurrency ~10-12 by
the 40k/agent rule); on T4 (SM75, 2x 15 GB) Qwen3.8-27B-FP8 does not fit and FP8 has no kernel support, so the only
option there is the fallback submission (log it loudly rather than crash-looping the server).
Implemented in `kaggle/src/solver.py` (not in the submitted kernel): `gpu_inventory()` + `choose_tp()` pick TP 4/2/1
from the GPUs present (SM89+, weight shard + 6 GB per GPU; `ARC_TP` overrides), the launch commands use it, and with
no usable GPU `main()` keeps the fallback submission and returns (tested with a faked 2x T4 inventory: no install,
valid 240-task fallback).

Kaggle GPU quota at the start of E014: 4.61 h used of 30 h (resets 2026-10-17).

## Results
- Toolchain: fixed and verified on CPU (wheel source v3). No GPU measurement yet: every GPU session pushed during
  the competition rerun came up on 2x T4 and was aborted by the pre-check (~5 GPU-min used in total).
- Offline working-set estimate: see Setup 3. vLLM's 1.40M reported KV covers concurrency 35 at the mean (80 %)
  and roughly at p99.9; SGLang's default radix sizing (772k KV, 29 running) only covers ~24 agents. To run 35 on
  SGLang, Mamba slots must drop to ~3 per request (`extra_buffer_lazy` + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`,
  `--max-mamba-cache-size 108`, ~1.0M KV expected), which the replay will test first (fallbacks: skip-lock only
  with 144 slots, then defaults).

## Conclusions
Pending GPU runs. Expected from the analysis: host offload (hicache) mostly helps when the device pool overflows;
the tool agents spend ~99 % of their time generating (tool calls take ~1 s against 50-100 s of decoding per turn),
so the live working set must fit on the device anyway and hicache is at best a cheap safety net against
retraction re-prefill at the top concurrency level.

## Next (commands)
1. When rerun 57088016 is no longer PENDING (`kaggle/monitor_submissions.py --once`):
   `P=$(python3 experiments/E014_sglang_kaggle/plans.py)`, build `--variant replay` with `["sgl"]`, then
   `.venv/bin/kaggle kernels push -p kaggle/nb_replay_sglang -t 3780` (planned 51 GPU-min).
2. Then the vLLM replay: build with `["vllm"]`, `kernels push -p kaggle/nb_replay_vllm -t 2520` (planned 34 min).
3. `kaggle kernels output chankhavu/arc2-e014-replay-<server> -p kaggle/runs/e014-replay-<server>-vN`, then
   `.venv/bin/python experiments/E014_sglang_kaggle/report.py kaggle/runs/e014-replay-*-vN`.
