# vLLM lab: GPU memory dynamics (util, startup profile, CUDA graphs, caching allocator)

2026-10-30. Hands-on lab, part of E067 (`experiments/E067_vllm_lab/memory/`). Upstream vLLM 0.20.1rc1.dev31
(torch 2.11 cu129), one RTX 3090 (24 GB, `total` = 23.56 GiB), dense Qwen2 7B in bf16, FLASH_ATTN (FA2),
max_model_len 65,536, max_num_seqs 64, prefix caching on, CUDA graphs on (PIECEWISE <= 128 tokens, FULL decode <= 64).

Why: B32-M (DeepSeek-V4.1 on 4xL4, Manh's fork) most likely died of a GPU OOM a few hours in: its allocator cap sat
below a peak that grows as long-context prefills arrive. We wanted to see the mechanism first-hand.

Model: the weights of `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and `chankhavu/Qwen2.5-Math-1.5B-...` are **not** in the
local HF cache (only their tokenizers/refs are). I used `~/.cache/kagglehub/models/yeoyunsianggeremie/aimo-sft-7b/...`,
an SFT of R1-Distill-Qwen-7B with the identical architecture (14.27 GiB of weights, KV 56 KiB/token). Memory behaviour
depends on shapes, not on weight values, so the numbers carry over to the requested 7B.

## Method

- `hook/sitecustomize.py` (on `PYTHONPATH`, active only with `MEMLAB_LOG`) patches `vllm.v1.worker.gpu_worker.Worker`
  in the engine-core process. It logs startup events (after load, profile, KV allocation, warmup) and **one line per
  engine step**: the step's shape (scheduled tokens, requests, max context) and torch `memory_stats` (allocated now and
  peak *within that step*, reserved, inactive split bytes, allocator retries, OOMs, cudaMalloc count) plus
  `cudaMemGetInfo` free. Every 400 steps it walks `torch.cuda.memory_snapshot()` for fragmentation (segments, cached
  free bytes, largest free block). It catches `torch.OutOfMemoryError` and logs the state.
- Options emulate the fork: `MEMLAB_CAP_MB=512` caps the allocator after warmup at `reserved + free - 512 MiB` (exactly
  what `VLLM_GPU_DRIVER_RESERVE_MB` does in `gpu_worker._reserve_driver_memory`), `MEMLAB_EXPANDABLE_AFTER=1` switches to
  expandable segments after the cap (the fork's commit 46a2cb33de); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  turns them on from launch.
- `drive.py`: random-token prompts (no prefix hits), forced output lengths. Workloads: a ladder of single requests at
  2k, 4k, 8k, 16k, 24k, 32k, 48k, 61k prompt tokens; a burst of 6 x 30k prefills at once (KV-bound, preemptions);
  mixed traffic with 16-48 concurrent clients, prompts 256..32k (log-uniform), outputs 32..512.
- `run_case.sh` (start, short workload, stop, summary), `analyze.py` (summary / timeline / transient by bucket),
  `table.py` (the table below). Raw logs are in the scratchpad (`.../scratchpad/vllmlab/memory/logs/`).

## How vLLM 0.20 sizes the KV pool (code reading)

`gpu_worker.py` + `utils/mem_utils.py` + `gpu_model_runner.profile_run`:

1. `init_device`: CUDA context and NCCL are created, `empty_cache()`, then `init_snapshot` (free/total/used).
   `requested = ceil(util x total)`; startup fails if `free < requested`.
2. `load_model`: `weights = torch allocated during load` (14.27 GiB).
3. `determine_available_memory` -> `memory_profiling`: `empty_cache`, reset the peak, then `profile_run()` =
   `_dummy_run(max_num_batched_tokens, is_profile=True)` (**eager**, and with `attn_metadata=None`, so every attention
   backend returns early: **attention kernels and their workspaces are not profiled**), then `_dummy_sampler_run` over
   `min(max_num_seqs, mbt)` rows. `peak_act = allocated peak - allocated before`. Then `profile_cudagraph_memory()`
   captures the first two graphs of each mode into a temporary pool and extrapolates (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`
   is on by default in this build). `non_torch = growth of (used - torch reserved) since init_snapshot`.
4. **`KV = util x total - weights - peak_act - non_torch - cudagraph_estimate`.**
5. After the profile, `empty_cache()` gives the activation memory back to the driver. At runtime it is re-reserved lazily
   by the first big steps, so nvidia-smi at "ready" is below the target and rises later.
6. Memory in use at `init_snapshot` (the CUDA context, 0.26 GiB here) is **outside** the budget: the device target is
   `init_used + util x total`.
7. **Upstream never caps the PyTorch allocator.** `gpu_memory_utilization` only sizes the KV pool. The caching
   allocator can grow into the `(1 - util)` remainder until the device is full. When `cudaMalloc` fails it frees every
   *fully free* cached segment and retries (`num_alloc_retries`), and only then raises OOM. Allocations outside torch
   (cuBLAS workspaces, Triton/"humming" module loads, NCCL) have no such fallback, which is why the fork adds
   `VLLM_GPU_DRIVER_RESERVE_MB`: after warmup, `set_per_process_memory_fraction((reserved + free - 512 MiB) / total)`.
   That cap is "the whole device minus non-torch minus 512 MiB" and **does not depend on util**: a higher util only
   moves memory from the runtime margin into the KV pool.

## Results

### Startup accounting (GiB unless noted)

| case | util | mbt | compile cache | weights | peak_act | non_torch | CUDA graphs est / actual | KV GiB | KV tokens | ready alloc / reserved / free | nvidia-smi ready |
|---|---|---|---|---|---|---|---|---|---|---|---|
| u90_b2048 | 0.90 | 2048 | cold | 14.27 | 1.043 | 0.092 | 0.193 / 0.16 | 5.60 | 104,944 | 19.95 / 20.29 / 2.81 | 21,254 MiB |
| cold_u90_b2048 (fresh caches, repeat) | 0.90 | 2048 | cold | 14.27 | 1.043 | 0.092 | 0.193 / 0.16 | 5.60 | 104,944 | 19.95 / 20.29 / 2.81 | 21,254 MiB |
| u90_b8192 | 0.90 | 8192 | cold | 14.27 | 1.125 | 0.092 | 0.193 / 0.16 | 5.52 | 103,408 | 19.91 / 20.16 / 2.93 | 21,122 MiB |
| cap_u95_b8192 | 0.95 | 8192 | warm | 14.27 | 1.040 | 0.092 | 0.193 / 0.16 | 6.79 | 127,056 | 21.18 / 21.48 / 1.62 | 22,470 MiB |
| cap_u96_b8192 | 0.96 | 8192 | warm | 14.27 | 1.040 | 0.092 | 0.193 / 0.16 | 7.02 | 131,456 | 21.41 / 21.70 / 1.40 | 22,694 MiB |
| cap_u96_b2048 | 0.96 | 2048 | warm | 14.27 | **0.267** | 0.092 | 0.193 / 0.16 | 7.79 | 145,952 | 22.17 / 22.44 / 0.66 | 23,454 MiB |
| nocap_u98_b8192 | 0.98 | 8192 | warm | 14.27 | 1.040 | 0.092 | 0.193 / 0.16 | 7.49 | 140,288 | 21.88 / 22.13 / 0.96 | 23,142 MiB |
| cap_u95_b8192_expl (expandable from launch) | 0.95 | 8192 | warm | 14.22 | 1.040 | 0.092 | 0.195 / 0.16 | 6.84 | 128,048 | 21.17 / 21.39 / 1.71 | 22,378 MiB |

Arithmetic check (u90_b2048): 21.20 requested - 14.27 - 1.043 - 0.092 - 0.193 = 5.60. Each +0.01 util adds
0.236 GiB (= 0.01 x total) to the KV pool and nothing else.

**Cold vs warm torch.compile changes the profile.** A cold compile (empty vLLM/Inductor/Triton caches; reproduced with
fresh cache dirs) inflates `peak_act` at mbt 2048 from 0.267 to 1.043 GiB (presumably Inductor/Triton compile and
autotune allocations inside the profile window). With a warm cache the profile matches the real activation peak almost exactly (0.267 vs 0.258 measured
at runtime; 1.040 vs 1.031 at mbt 8192). So a cold start hands you ~0.78 GiB of hidden margin that a warm start
spends on KV. At mbt 8192 the real activations dominate, so cold and warm differ by only 0.09.

### What grows during a run (uncapped, util 0.90, ~10 min of mixed traffic each)

Per-step transient = allocated peak within the step minus allocated at its start:

| step shape | mbt 2048: steps, p50 / max transient | mbt 8192: steps, p50 / max transient |
|---|---|---|
| decode only (1 token/request) | 2,930: 0.006-0.03 / 0.095 | 3,503: 0.006-0.03 / 0.095 |
| 256-1k tokens | 44: 0.098 / 0.129 | 21: 0.098 / 0.122 |
| 1k-2k tokens | 52: 0.193 / 0.255 | 27: 0.205 / 0.258 |
| full 2048 chunk | 762: 0.258 / 0.258 | 35: 0.399 / 0.515 (2k-4k) |
| 4k-8k tokens | — | 57: 0.785 / 1.018 |
| full 8192 chunk | — | 145: 1.031 / 1.031 |

- The transient is a function of the **step's token count** only, about 0.126 GiB per 1k scheduled tokens for this 7B
  (MLP gate/up/act intermediates dominate). It is **identical at 2k and 61k context** (the max over every context bucket
  is 0.258 / 1.031) and does not depend on concurrency (1..48 requests). FA2 has no context-sized workspace, and the
  sampler is sized by the number of requests.
- **Reserved memory jumps once and then stays flat.** mbt 8192: reserved 20.16 -> 21.68 GiB on the first 8192-token
  chunk (the first minute), then flat for 9 more minutes; cudaMalloc count 227 -> 233; 0 retries. mbt 2048: 20.29 ->
  20.44, two cudaMallocs. Allocated "live" memory never moved (19.954 / 19.914 GiB flat).
- **Reserved minus peak allocated is fragmentation, and it is large:** 0.74 GiB at mbt 8192 (72% of the 1.03 GiB
  transient; 1.77 GiB of cached free blocks, largest 0.58 GiB) and 0.23 GiB at mbt 2048 (89% of the transient).
  Blocks freed by one step size stay cached in segments of that size; the next step size gets new segments.
- No OOM is possible here: free never went below 1.42 GiB (mbt 8192) / 2.66 GiB (mbt 2048).

### Making it OOM: the fork's cap (keep 512 MiB) at tight util

Same short workload for each case: the ladder, then a burst of 6 x 30k, then 150 s of 32-way mixed traffic.
`cap` is the fork's `reserved + free - 512 MiB` after warmup, which comes out at 22.60 GiB in every case.

| case | util | mbt | segments | KV GiB | cap - (live + 1.03 transient), GiB | result | stranded at OOM (reserved - allocated) | retries | requests OK |
|---|---|---|---|---|---|---|---|---|---|
| cap_u95_b8192 | 0.95 | 8192 | native | 6.79 | +0.39 | **OOM** at the first 8192-token chunk (asked 296 MiB) | 0.53 GiB (67 split segments with free space) | 2 | 2 |
| cap_u95_b8192_expl | 0.95 | 8192 | expandable from launch | 6.84 | +0.39 | **survived** the whole workload (ladder to 61k, burst, mix) | inactive split 0.00 | **2,775** (about one per full chunk) | 80 |
| cap_u96_b8192 | 0.96 | 8192 | native | 7.02 | +0.16 | **OOM**, same step | 0.22 GiB | 2 | 2 |
| cap_u96_b8192_exp | 0.96 | 8192 | expandable after warmup (fork style) | 7.02 | +0.16 | **OOM**, same step | 0.19 GiB | 2 | 2 |
| cap_u96_b8192_expl | 0.96 | 8192 | expandable from launch | 7.07 | +0.16 | **OOM**, same step | 0.17 GiB (12 segments, largest free block 34 MiB) | 28 | 2 |
| cap_u96_b2048 | 0.96 | 2048 | native | 7.79 | +0.17 (transient 0.26) | **OOM** at step 1 (asked 74 MiB) | 0.19 GiB | 2 | 0 |
| nocap_u98_b8192 (upstream, no cap) | 0.98 | 8192 | native | 7.49 | device: 0.19 left by bytes | **OOM** at the first 8192 chunk; the device itself was full (0.28 GiB free) | 0.19 GiB | 2 | 2 |

Every OOM killed the engine (`EngineDeadError`), and every later request failed with HTTP 500 or a dropped
connection. Full table: `experiments/E067_vllm_lab/memory/results/runs.md`.


- With the cap, OOM comes from **memory stranded in split segments**, not from live memory reaching the cap.
  cap_u95_b8192: the first 8192-token chunk asked for 296 MiB with 21.92 GiB allocated and a 22.60 GiB cap, so 0.39 GiB
  was free *by allocated bytes*, but 0.53 GiB sat in cached blocks inside segments that also hold live tensors (67 split
  segments with free space). The retry can only release fully free segments, so it failed. cap_u96_b8192: the same with
  0.22 GiB stranded.
- **Switching to expandable segments after warmup did not help** (cap_u96_b8192_exp: same OOM at the same step):
  the holes are in the native segments created before the switch (0.14 GiB of inactive split blocks already at "ready").
- **Smaller chunks alone do not help** (cap_u96_b2048 OOMed at step 1): with a warm cache the profile is exact, so the
  KV pool simply grows (7.02 -> 7.79 GiB) into whatever the smaller transient freed, and the margin to the cap stays ~0.
- **Expandable segments from launch turn "stranded -> OOM" into "flush and retry".** At util 0.95 they survived the
  whole workload, where native segments died on the first full chunk, but the allocator was at the cap on every full
  8192-token chunk: 2,775 retries, and 4,350 page maps against 206 at ready. Each retry synchronises and unmaps or remaps
  pages. Long prefills were 11-19% slower than in the uncapped util 0.90 runs (16k: 5.55 vs 4.66 s; 32k: 12.81 vs
  11.16 s; 61k: 30.01 vs 26.95 s). At util 0.96, with 0.16 GiB of byte margin, they OOMed anyway. They remove the
  fragmentation; they do not replace the margin.
- **Upstream (no cap) OOMs the same way once util leaves no room on the device:** at 0.98 the first full chunk needed
  296 MiB, 0.28 GiB was free on the device, and 0.19 GiB was cached but not releasable.

## The mechanism, plainly

1. At startup vLLM measures one dummy step (max tokens, eager, **no attention**), adds weights, non-torch memory and a
   CUDA-graph estimate, and gives everything else under `util x total` to the KV pool.
2. At runtime, memory = **live** (weights + KV + persistent buffers, flat for the whole run) + **the transient of the
   current step** + **cached free blocks** the caching allocator keeps for reuse.
3. The transient depends on the step's shape. For a dense FA model the shape is just the token count, so the worst case
   appears in the first minute and the profile covers it. Anything whose size scales with **context length** (or with
   rarer combinations such as a long-context chunk together with many decodes) is not in the profile, and its worst case
   arrives only when the longest contexts arrive. The running maximum then keeps creeping up for hours.
4. Every new, larger transient needs new segments (reserved grows: "reserved creep"). The old ones stay cached and are
   partly pinned by long-lived tensors allocated into them, so their free space can never be returned (inactive split
   bytes).
5. Uncapped (upstream), the allocator may use the whole `(1 - util)` remainder, and an OOM needs the device itself to be
   full. With a cap (the fork), an allocation fails as soon as **live + transient + stranded > cap**, even when live +
   transient alone fits.

## Relation to the DSv4.1 / B32-M observations

- **Allocator cap.** The fork's cap is `reserved + free - 512 MiB` after warmup, independent of util. Our cap cases
  reproduce the failure signature of B32-M: OOM while "cap - peak allocated" is still positive, with the difference held
  in split segments (B32-M: 0.11 GiB left, 0.26 GiB stranded; here 0.39 GiB left, 0.53 stranded).
- **"Cap minus peak" shrinking ~0.2 GiB/h with live allocation flat.** On a dense FA model this cannot happen: the
  peak saturates within a minute, because the transient depends only on the step's token count. A peak that keeps
  rising for hours needs per-step tensors whose size depends on something that keeps getting rarer and bigger. DSv4.1
  has such tensors: the sparse-attention indexer's prefill logits (`M x N x 4` bytes per chunk, N = compressed context,
  chunked to `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` = 512 MiB), its K gathers for the chunk's context, and decode logits
  `rows x max_model_len` that piecewise/eager steps allocate outside the CUDA-graph pools. The profile skips attention,
  and the fork only started reserving the decode indexer logits (317081fb6d, 10-28) and the Engram activations
  (7ab756940a, 10-29) in the profile; per tonight's audit both post-date the build B32-M ran. Data-dependent sizes
  (for example per-expert padding in the MoE kernels, which varies with routing) behave the same way. Agent
  transcripts grow over the run, so the longest contexts, and the worst steps, arrive hours in. The pod's running peak
  (+0.45 GiB from 10 min to 4.4 h, then flat) looks like a running maximum over a heavy-tailed distribution of step
  shapes, not a leak (live allocation stays flat).
- **Reserved creep.** The same thing seen from the allocator: each new largest transient adds segments, and under a cap
  the allocator survives by flushing its cache (B32-M pods: 54-73 retries by the end; each retry is a sync and a slow
  step) until a request no longer fits next to the stranded holes.
- **Expandable segments after warmup** (fork 46a2cb33de) cannot recover the holes already in startup-era segments
  (our cap_u96_b8192_exp). B32-M's 0.26 GiB of inactive split bytes is exactly that kind of memory: with expandable
  segments from launch it would read 0.00, as in our `_expl` runs.
- **Compile cache.** A cold compile inflates the profiled activation peak (here +0.78 GiB at mbt 2048), a warm one does
  not. Check whether Kaggle (fresh container) and pod replicas start with the same cache state: a pod with a warm cache
  gets a bigger KV pool and a smaller margin than Kaggle, or the reverse.

## Practical rules for util and margins on 4xL4

On an L4 (22.49 GiB), 0.01 util = 0.225 GiB per GPU.

1. **Budget the margin explicitly, per GPU:** `cap - (live + worst transient at max context and full chunk) >= stranded
   + 0.3 GiB`. Measure "worst transient" with a long-context replay (max_model_len prompts, full chunks, max decode
   batch at once), not from the first hour of traffic. Take "stranded" from `inactive_split_bytes` late in a replay
   (0.26 GiB on B32-M; 50-90% of the largest transient with native segments here).
2. **Pin the KV pool instead of letting the profile size it:** `--kv-cache-memory-bytes` (upstream) makes KV, and hence
   the margin, identical across cold/warm compile caches, pods and Kaggle. Size it from a measured replay, then let util
   only check that it fits.
3. **Lower util is the only knob that adds margin.** A smaller `max_num_batched_tokens` shrinks the transient (~linear
   in chunk tokens), but with a profile-sized pool the KV just grows into the freed space. Pair smaller chunks with a
   pinned KV size if you want the margin.
4. **Expandable segments must be on before the KV cache and warmup, or not at all.** Turning them on after warmup leaves
   the startup segments' holes in place. From launch
   (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) they remove the stranded memory (inactive split 0.00). They do
   not remove the need for margin, and running at the cap costs a cache flush per big step (here 11-19% slower long
   prefills). Treat them as a backstop. Check that the fork's custom kernels and CUDA graphs behave with them before
   relying on it (upstream 0.20 ran fine here).
5. **Every context-dependent buffer must be in the profile at max_model_len** (the indexer logits budget, gathers,
   decode logits for `max_num_seqs` rows). The profile run skips attention, so assume anything attention-side is
   missing until a test shows the profiled peak >= a long-context replay's peak.
6. **Monitor and alarm per worker:** peak allocated, reserved, inactive split bytes, `num_alloc_retries`, and
   `cap - peak`. Retries climbing means the allocator is already flushing its cache to survive; treat it as an early
   OOM warning, not as noise.
7. For B32-M's numbers (cap - final peak about 0.11 GiB, about 0.26 GiB stranded, +0.45 GiB of peak growth after the
   first 10 min), staying safe needs roughly 0.5 GiB more margin per GPU than it had: util about 0.02 lower (0.945 ->
   ~0.925, costing ~0.45 GiB/GPU, about 13% of its KV pool), or about 0.01 lower plus expandable segments from launch to
   remove the stranded part (accepting retries near the cap). With native segments, our lab needed margin for
   fragmentation of about 70% of the largest transient (0.74 GiB at a 1.03 GiB transient): 0.39 GiB of byte margin
   was not enough.

## Files

- `experiments/E067_vllm_lab/memory/hook/sitecustomize.py`: per-step memory logger and fork-cap emulation.
- `experiments/E067_vllm_lab/memory/{run_server.sh,stop_server.sh,run_case.sh,drive.py,analyze.py,table.py}`.
- `experiments/E067_vllm_lab/memory/results/`: the tables of this note (`runs.md`, `transient_by_ntok.txt`).
