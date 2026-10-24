# E042: synthetic DeepSeek-V4.1-Flash E64 on 4×L4, first inference and TPS vs batch size

**Goal (user, 2026-10-21):** see whether the pruned-shape DeepSeek-V4.1-Flash-1017 (E64/top-6, all 40 layers)
runs at all on the Vast 4×L4 box with random weights, and measure tokens/s at several batch sizes.
The user's full plan (synthetic-first, pruning in parallel) is in `plan/` (unzipped upload; `plan/START_HERE.md`).
This experiment is the first, deliberately narrow slice of it: inference + TPS only. No accuracy, no pruning.

## Summary (2026-10-22, first session; random weights, so no quality claims)

- **Inference works.** The E64/top-6 synthetic model with all 40 layers runs end to end on 4×L4 with the SM89
  vLLM fork: TP4, native MXFP4 experts, MXFP8 dense layers, FP8 KV, sparse indexer attention, mHC, and a compact
  synthetic Engram table in pinned host memory. Weights take 15.06 GiB per GPU, leaving KV for 1.02M reported
  tokens at util 0.85. A stream uses ≈ 1.6K + 0.355 × context of them (S4), so ≈ 41 concurrent 64K streams fit.
  Routing is diverse at every depth.
- **Steady decode** (FULL graphs, which run as FULL_DECODE_ONLY): about **35–40 tok/s at C = 1**, 120–126 at
  C = 8, 255 at C = 32, **371 at C = 64 and 465 tok/s at C = 128** (1K context; S4). It barely drops with
  context: 257 at 8.6K × 32, 348 at 8.6K × 64, 127 at 33K × 8. 100 streams held 320–400 tok/s for 65 min
  while their context grew to 14.5K.
- **Prefill is the weak spot:** about 700–900 tok/s on the whole box. Half of a 2048-token prefill step is the
  TP all-reduce over host shared memory (1.5 GB/s on this box, about 2× faster on Kaggle). An 8K prompt takes about
  11 s.
- **Hardware caveat:** GPU2 on this Vast box is thermally throttled (88 °C, about 0.7–1.4 GHz vs 1.5–2.0 GHz on the
  others) and gates every TP step. Treat these numbers as a floor.
- **Where decode time goes** (P3, C = 64): all-reduce 34% of rank 0's step (about 60% of that is waiting for the
  throttled GPU2), MXFP4 Marlin experts 28% (bandwidth-bound), indexer 12%, dense MXFP8 10%, attention 5%.
  The estimate is ≈ 450 tok/s at C = 64 on a well-cooled box, with a ceiling of about 600.
- **E72** costs +1.50 GiB/GPU of weights and 4× less KV (255,744 reported tokens, ≈ 10 × 64K streams). The
  speed cost is ≈ 4% from bandwidth; the 8–20% measured was mostly GPU2 heat.
- **Prefix caching works on the fork** (P4) and is required for tool use. On a 6-turn tool-shaped replay it
  makes the run 2.5× (C = 16) to 3.0× (C = 32) faster, with a 0.74 hit rate (0.765 reusable). Decode speed is
  unchanged. Random weights cannot verify the logits after a cached prefix: the output distribution is so flat
  that two cold runs already diverge.
- Expert parallel ties with TP. A larger prefill chunk (8192) costs 85% of the KV capacity for a small gain. NCCL
  P2P is box-only.

## Setup

- **Box:** Vast 55889595, 4× L4 (22.0 GiB usable each, SM89, PCIe, no P2P), driver 595.84, 128 vCPU,
  cgroup RAM 107.9 GiB, 64 GB disk. Handed over by session `arc-agi-2` (E041 stopped at 23:30 UTC).
- **Software (publisher wheel route, not the editable pin):** `yhfgyyf/vllm-deepseek-v4-sm89` release
  `v0.28.1rc1-vision9-sm89-sm120-cu130`, assets `vllm-0.28.1rc1.dev517+glm53.dsv41.vision11` (sha256 `e1c8313e…`,
  Python source 86d35745) and `flashinfer_python-0.6.18+glm53.dsv41.vision2` (sha256 `078aa768…`); torch
  2.13.0+cu130, triton 3.7.1, transformers 5.16.1, Python 3.12. Venv `/root/dsv41/venv` (7.6 GB). Hashes match
  `plan/software_lock.json`.
- **Model metadata:** `plan/tools/prepare_synthetic_metadata.py --download-metadata` → `metadata/`. Only
  `text_config.n_routed_experts` changes (384 → 64); tokenizer files from revision 1bae0e10. No checkpoint shard
  was downloaded.
- **Patches** (`patches/apply_patches.py` on the installed wheel; env-gated except the router whitelist):
  1. `router.py`: accept (64, 6) and (72, 6). The Triton top-k kernel already masks lanes ≥ E.
     `DSV41_ROUTE_STATS=N` logs per-layer expert load (eager only).
  2. `engram.py`: `DSV41_ENGRAM_SYNTH_ROWS=N` allocates N physical rows per rank and layer **inside the
     constructor**, so the original ~189 GiB pinned table is never allocated. Hash geometry, head ownership and
     the lookup/Wkv/gate path are unchanged; owned ids are folded into the compact table with `% N`.
     So this is a synthetic **capacity/traffic** fixture, not the plan's compact map with zero-on-miss.
     Runs use 4M rows = 0.98 GiB per rank and layer (7.9 GiB aggregate), pinned host memory read over UVA.
  3. `dummy_loader.py` + `synth_init.py`: `DSV41_SYNTH_INIT=1` with `--load-format dummy` does one stock dummy
     pass and then format-valid synthetic values before the normal finalizers (Marlin repack, MXFP8 prep,
     wo_a → BF16, mHC):
     random E2M1 bytes plus fan-in UE8M0 scales for experts, N(0,1) FP8 plus 1/√fan-in scales for dense layers,
     norms = 1, router bias = 0, router/embedding/LM-head N(0, 1/fan-in), real sinks = 0 and padded sinks = −inf.
     The stock dummy fill would leave the experts zero and the router at ±1e-3, so the bias alone would pick the
     same 6 experts for every token. Values are keyed by (seed, name), not by global coordinates.
- **Serving flags** (`scripts/serve.sh`): TP4, EP off, `--kv-cache-dtype fp8 --block-size 128
  --max-model-len 65536 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.85`, prefix caching off,
  Engram CPU offload, `--language-model-only`, no speculative decoding, no CED, Model Runner V2.
  The run is either eager or FULL CUDA graphs (the fork's recommended mode). No `NCCL_P2P_LEVEL`, as on Kaggle.
- **Benchmark** (`scripts/bench.sh`): `vllm bench serve`, random token prompts, `--ignore-eos`, temperature 0.
  For concurrency C it sends 2C prompts. Decode tok/s = C × 1000 / mean TPOT.

## Results

### S0: eager smoke (B8)

- Startup: model built at full depth. Weights take 15.06 GiB per GPU, and loading plus synthetic init took 20 s.
  KV memory left at 0.85 utilization is 2.58 GiB per rank = 1,102,947 tokens, enough for 16.8 concurrent 64K
  requests. The API was ready about 3.5 min after launch, mostly TileLang/FlashInfer JIT.
- Synthetic-init manifest (`runs/s0_eager/synth.rank0.json`): 40 distinct tensors per layer pattern. Routed
  experts per rank: w13 7,200 MiB + w2 3,600 MiB packed E2M1 (64 experts × local I 576) plus 675 MiB scales.
  Shared experts are separate MXFP8 linears. Engram pinned host memory is 1.97 GiB per rank (2 layers). No vision
  tower; only three `image_*` placeholders on meta.
- **First request works end to end** (prefill + 48 decode tokens, greedy): token ids are valid, logprobs are finite
  and near-uniform (−11.4 vs ln(1/129280) = −11.8), as expected for random weights.
- **Routing is diverse at every depth** (`DSV41_ROUTE_STATS`, 1,499 router calls per layer): decode steps with
  8 tokens use 34–36 distinct experts of 64 (uniform-random expectation 34.9). Max/mean expert load is 1.9–3.0
  per layer and the top-6 share is 0.16–0.22 (uniform = 0.094). That is moderate skew, with no collapse through
  the 40 layers.
- Eager TPS (1024 in / 256 out, with the routing-stats syncs on, so slightly pessimistic): C=1 TPOT 146.5 ms =
  6.8 tok/s; C=8 TPOT 166.9 ms = 47.9 tok/s aggregate decode (40.8 tok/s including prefill). Mean TTFT is
  7.5–8.4 s, inflated by first-shape JIT. Eager is not the fork's intended mode, which is FULL CUDA graphs.

### S1: FULL CUDA graphs (`runs/s1_full`, TP4, EP off, max-num-seqs 64, MBT 2048, util 0.85)

Startup took 120 s, of which graph capture took 7 s and 0.20 GiB. KV cache is 1,021,072 tokens. The smoke
request is identical to eager.

**Sweep A: 1024 in / 512 out** (2C prompts per point; decode = C × 1000 / mean TPOT; out tok/s includes prefill):

| C | out tok/s | decode tok/s | per stream | TPOT mean / p95 (ms) | TTFT mean / p95 (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.6 | **40.0** | 40.0 | 25.0 / 25.2 | 1.2 / 1.5 |
| 2 | 53.6 | 59.3 | 29.6 | 33.7 / 35.4 | 1.8 / 2.8 |
| 4 | 78.8 | 88.8 | 22.2 | 45.0 / 48.1 | 3.0 / 4.6 |
| 8 | 98.6 | **115.5** | 14.4 | 69.2 / 79.7 | 6.1 / 9.3 |
| 16 | 114.5 | 134.3 | 8.4 | 119.2 / 134.2 | 10.4 / 20.6 |
| 32 | 146.1 | 172.0 | 5.4 | 186.0 / 208.8 | 16.7 / 42.0 |
| 64 | 180.2 | **213.8** | 3.3 | 299.4 / 338.5 | 27.8 / 80.8 |

FULL graphs are 5.9× faster than eager at C=1. But step time grows almost linearly with batch, about +4 ms per
extra sequence from C=8 to C=64, so aggregate decode only goes 40 → 115 → 214 tok/s. A bandwidth estimate puts
reading every routed expert once per step (all 64 are touched at C ≥ 16) at under 60 ms per step; 299 ms is
measured, so something else dominates. Prefill is also slow: TTFT 1.2 s for 1K tokens at C=1, and 12.3 s for 8K
tokens (≈ 670 tok/s). P1 below profiles this.

**Sweep B: 8192 in / 512 out.** These numbers are prefill-bound and the TPOT is contaminated. Each 2048-token
prefill chunk takes about 2.5–3 s and stalls every decoding stream (ITL p95 2.7–2.8 s at C ≥ 16), so the
C × 1000 / TPOT figures (34.0 / 44.6 / 49.6 / 53.2 tok/s at C = 1 / 8 / 16 / 32) do **not** measure decode.
Single-stream prefill: 8K tokens in 12.3 s ≈ 670 tok/s; 1K in 1.2 s ≈ 840 tok/s. Total (in+out) tok/s levels off
near 680. Sweep C (32K) was stopped at its start for the same reason and replaced by `decode_window.py`
(clean steady-decode windows; see S1-DW below).

**S1 facts (monitor, from `runs/s1_full/server.log.gz` and `gpu.csv`):**
- `cudagraph_mode=FULL` is downgraded to **FULL_DECODE_ONLY** on every rank, because the DeepseekV41IndexerBackend
  supports only UNIFORM_BATCH. Pure decode steps replay graphs (11 sizes, 1–64, 7 s, 0.20 GiB); prefill and mixed
  steps run eagerly.
- KV is 2.39 GiB per rank = 1,021,072 reported tokens (≈ 41 concurrent 64K streams by the S4 fit). Peak GPU memory was 20,188 MiB on every rank (19,540
  after load), the same across ranks. Host peak was about 38 GB, out of a 108 GB cgroup.
- Custom all-reduce is off (more than 2 PCIe GPUs), so NCCL is used. KV usage peaked at 15.9% (A c64) and 18.4%
  (B c32), with no preemptions.
- The server's own 10 s metrics at A c64 (64 running, no prefill) show about **345 tok/s** of generation, versus 214
  from TPOT. The bench's TPOT includes wave-2 prefill stalls.
- **Thermals: GPU2 (and partly GPU1) are thermally throttled.** Mean busy SM clock: GPU2 ≈ 1.0 GHz at 86–88 °C
  (SW thermal slowdown, only 53–56 W), GPU1 ≈ 1.6 GHz at 84–87 °C, GPU0 ≈ 1.75 GHz, GPU3 ≈ 1.9 GHz at 69 °C.
  GPU0, GPU1 and GPU3 also hit the 72 W L4 power cap. Each TP4 step waits for the slowest rank, so these numbers
  are probably pessimistic for a well-cooled 4×L4 (for example Kaggle's) by an amount still to be measured
  (see P1 per-rank kernel times).

### S1-DW: clean steady decode (`scripts/decode_window.py`, same S1 server)

C requests with distinct random-token prompts are sent together. Once all C are decoding (no prefill left), the
client skips 16 steps and counts all tokens for 30 s, then aborts the requests. Decode tok/s is aggregate;
"ctx" is the context at the window end. Files: `runs/s1_full/dw/`.

| prompt | C | decode tok/s | per stream | ITL median (ms) | ctx at end | prefill of all C (s) | prefill tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1 | 36.6 | 36.6 | 28.0 | 2.1K | 1.1 | 904 |
| 1K | 8 | 126.4 | 15.8 | 64.9 | 1.4K | 9.8 | 834 |
| 1K | 32 | 254.9 | 8.0 | 121.3 | 1.3K | 43.6 | 752 |
| 1K | 64 | **371.2** | 5.8 | 167.3 | 1.2K | 86.8 | 755 |
| 8K | 1 | 34.9 | 34.9 | 28.8 | 9.3K | 11.4 | 718 |
| 8K | 8 | 118.7 | 14.8 | 60.9 | 8.7K | 92.2 | 711 |
| 8K | 16 | 180.8 | 11.3 | 86.9 | 8.6K | 179.4 | 731 |
| 8K | 32 | **257.1** | 8.0 | 117.7 | 8.6K | 351.8 | 745 |
| 32K | 1 | 34.9 | 34.9 | 29.0 | 33.8K | 48.7 | 673 |
| 32K | 4 | 82.9 | 20.7 | 45.4 | 33.5K | 193.8 | 676 |
| 32K | 8 | **126.7** | 15.8 | 63.1 | 33.4K | 380.2 | 689 |

- Decode throughput barely depends on context: C = 8 gives 126 / 119 / 127 tok/s at 1.4K / 8.7K / 33K, so the sparse
  indexer attention holds up. Batch scaling is sublinear: per-step time is 28 ms at C = 1 and 167 ms at C = 64,
  about +2.2 ms per extra sequence.
- Prefill is the weak spot: about 700–900 tok/s on the whole box at every length, roughly 2.4 s per 2048-token
  chunk. An 8K ARC prompt costs about 11 s; 32 of them cost about 6 min before decode starts.
- The bench-serve TPOT figures in sweep A understate decode at large C (214 vs 371 tok/s at C = 64), because
  their windows include wave-2 prefill.

### P1: torch profiler (`runs/p1_prof`, same S1 flags plus `--profiler-config`)

The profiler took one capture: 10 active steps during the prefill phase of 32 × 1K prompts, i.e. 2048-token
prefill chunks plus a few decode tokens. A second /start_profile, meant for steady decode, produced no trace.
Self CUDA time over the capture (`runs/p1_prof/trace/profiler_out_<rank>.txt`, 23.2 s on rank 0):

| kernel | rank 0 | share | calls | avg |
|---|---:|---:|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 11.54 s | **49.7%** | 846 | 13.6 ms |
| `_ada_mxfp8_k32_linear_kernel` (dense MXFP8, Triton) | 3.22 s | 13.9% | | 1.47 ms |
| `_mxfp4_indexer_logits_kernel` (sparse-attention indexer) | 2.49 s | 10.7% | | 14.8 ms |
| `moe_wna16_marlin_gemm` (routed experts) | 1.77 s | 7.6% | | 2.2 ms |
| `mhc_post_tilelang_kernel` + `_prenorm_project_kernel` + `mhc_pre_big_fuse` | 1.41 s | 6.1% | | |
| all-gather (Engram heads, LL) | 0.18 s | 0.8% | 31 | 5.9 ms |

- **Prefill is half communication.** All four ranks spend 46–51% of their GPU time inside the TP all-reduce,
  about 84 all-reduces per step (2 per layer) at 12.6–13.9 ms each. Even the slowest rank (GPU2, rank 2, 46%)
  spends that long, so this is mostly real transfer time, not waiting for the throttled GPU. Waiting accounts for
  about 1 s of the 11.5 s on rank 0.
- A [2048, 5120] bf16 activation is 20 MiB, so the effective algbw is ≈ 1.6 GB/s. NCCL picked `RING_LL`, the
  low-latency protocol, for these large messages over SHM (no P2P on this box). The NCCL protocol sweep below
  tests whether `NCCL_PROTO=Simple` or `LL128` fixes this.
- Compute is spread over dense MXFP8 GEMMs (Triton fallback on Ada), the FP4 indexer and Marlin MoE.
  No single kernel dominates.

### EP1: expert parallel (`--enable-expert-parallel`, 16 full-width experts per rank; `runs/ep1`)

KV is 989,711 tokens. Clean decode tok/s, EP vs TP (S1-DW):

| prompt | C | EP decode | TP decode | EP prefill tok/s | TP prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 1K | 1 | 34.7 | 36.6 | 756 | 904 |
| 1K | 8 | 125.6 | 126.4 | 884 | 834 |
| 1K | 32 | 259.2 | 254.9 | 787 | 752 |
| 1K | 64 | 364.8 | 371.2 | 770 | 755 |
| 8K | 8 | 118.7 | 118.7 | 732 | 711 |
| 8K | 32 | 251.7 | 257.1 | 740 | 745 |

**A tie** within ±2–5%. EP does not help here, since the MoE GEMMs are not the bottleneck. TP stays the default.

### P2: `--max-num-batched-tokens 8192` (`runs/p2_mbt8k`)

KV falls to **152,404 tokens** (from 1.02M), because the larger step budget's activation and indexer workspaces
eat the KV pool. One 8K prompt prefills in 9.7 s vs 11.4 s (842 vs 718 tok/s); 8 × 8K prefill is 736 vs 711 tok/s.
Decode is unchanged (8K C8 126.7, 1K C32 263.5 tok/s). **Not worth it:** a small prefill gain for 85% of the KV
capacity.

### NCCL all-reduce micro-benchmark (`scripts/nccl_allreduce_bench.py`, `runs/nccl/`)

bf16 [T, 5120] all-reduce on all 4 GPUs (NCCL 2.28 from torch 2.13). The default transport is `SHM/direct` on
every ring channel.

| T | MiB | default (SHM, auto proto) | Simple | LL128 | LL | `NCCL_P2P_LEVEL=SYS` (box only) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.08 | – | 0.103 ms | 0.061 | 0.065 | 0.042 ms |
| 32 | 0.31 | – | 0.196 | 0.246 | 0.216 | 0.065 |
| 64 | 0.63 | – | 0.349 | 0.507 | 0.416 | 0.105 |
| 256 | 2.5 | 1.443 | 1.588 | 1.739 | 1.705 | 0.243 |
| 2048 | 20 | 13.908 (1.51 GB/s) | 13.898 | 14.059 | 13.860 | **1.620 (12.9 GB/s)** |
| 8192 | 80 | – | 56.9 | 57.4 | 54.2 | 6.49 |

- **The protocol is not the problem:** every protocol tops out at about 1.5 GB/s algbw through host shared memory.
  This matches E027 (`experiments/E027_l4_throughput/README.md`): 14.2 ms per 2048-token all-reduce over this
  box's SHM, 1.6 ms with P2P forced.
- Prefill: 84 all-reduces per 2048-token chunk × 13.9 ms ≈ 1.17 s of a ~2.4 s step, matching P1's 46–51%.
  Decode at C = 32–64: 84 × 0.2–0.4 ms ≈ 17–35 ms of a 120–167 ms step (≈ 14–21%).
- **Kaggle projection (from E027's Kaggle measurement):** Kaggle's default SHM all-reduce is about 2× faster
  (6.4 ms at 2048 tokens), and P2P there is 12× *slower*, so keep the default. Prefill communication would be
  ≈ 0.54 s per chunk, so roughly 1.1–1.3K tok/s prefill if compute stays the same. Decode communication would be
  ≈ 8–17 ms per step.

### S3: `NCCL_P2P_LEVEL=SYS` decode grid (box-only; `runs/s3_p2p`) and the thermal confound

A 60 s bf16 matmul burn on all 4 GPUs at once (`scripts/gpu_burn_check.py`), starting from idle, gives
GPU0–3 = 52.4 / 52.9 / **49.2** / 55.1 TFLOPS in the last 5 s. That is too short to reach thermal steady state;
idle temperatures afterwards were 70 / 75 / **81** / 60 °C.

Clean decode, P2P vs SHM, with the mean SM clock in each window (`gpu.csv`):

| prompt | C | SHM decode | GPU2 clock | P2P decode | GPU2 clock |
|---:|---:|---:|---:|---:|---:|
| 1K | 1 | 36.6 | 882 MHz | 40.3 | 1448 MHz |
| 1K | 32 | 254.9 | 1361 | 276.3 | 1394 |
| 1K | 64 | 371.2 | 994 | 362.7 | 1001 |
| 8K | 8 | 118.7 | 1139 | 111.2 | 1007 |
| 8K | 32 | 257.1 | 902 | 229.3 | 724 |
| 32K | 1 | 34.9 | 884 | 31.3 | 672 |
| 32K | 8 | 126.7 | 992 | 103.5 | 925 |

- **GPU2 sets the pace.** It sits at its 88 °C slowdown limit in every sustained window, at 0.7–1.4 GHz, while
  GPU0, GPU1 and GPU3 run at 1.5–2.0 GHz. Decode tok/s follows GPU2's clock, not the NCCL setting. At matched
  GPU2 clocks, P2P shows no decode gain (1K C64: 371 vs 363 at ≈1.0 GHz). P2P did speed up early 1K prefill
  (834 → 1343 tok/s at C8, while the GPUs were still cool), but 8K and 32K prefill stayed at 740–770 tok/s.
- So every number in this README is a **floor for this box**, gated by one badly cooled GPU. A well-cooled 4×L4,
  for example Kaggle's (E027 saw this box about 10% slower than Kaggle for Qwen), should do better. By how much is
  unmeasured. A crude sensitivity from the pairs above is about +0.5% tok/s per +1% GPU2 clock.
- NCCL P2P is irrelevant for Kaggle (12× slower there, E027), so it is not pursued further.

### P3: decode-phase profile (C = 64, 1K context, default NCCL; `runs/p3_prof_decode`)

About 11 steady decode steps were captured. Rank 0 (≈181 ms of GPU time per step with the profiler on):

| kernel | per step | share | note |
|---|---:|---:|---|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 62 ms | 34% | rank 2 (the throttled GPU) spends only 24 ms; the other ~38 ms on ranks 0/3 is waiting for it |
| `marlin_moe_wna16::Marlin` (MXFP4 routed experts) | 50 ms | 28% | all 64 experts are read each step (≈11.5 GB/rank) ≈ 240 GB/s, near the L4's 300 GB/s |
| `_mxfp4_indexer_logits_kernel` | 21 ms | 12% | 8 index layers × 2.7 ms; 2× slower on throttled rank 2 |
| `_ada_mxfp8_k32_linear_kernel` (dense MXFP8, Triton) | 17 ms | 10% | 230 calls per step |
| `flashinfer::deepseek_v41_prefill_kernel` (attention) | 9 ms | 5% | |
| Engram all-gather | 6 ms | 3% | |

- At C = 64 the MoE is bandwidth-bound, as expected: every expert is touched every step. Reading the expert
  bank alone sets a floor of about 38 ms per step.
- Removing the GPU2 wait (~38 ms) would make the step ≈ 143 ms, so ≈ 450 tok/s at C = 64 on a balanced box,
  before counting GPU2's own slower compute. A rough ceiling for 4×L4 at C = 64 is about 100–110 ms per step
  (≈ 600 tok/s): MoE 38 + dense 10 + attention/indexer about 30 + communication about 12 (Kaggle) to 24 (box).

### C2: E72 (`runs/c2_e72`, `model-e72`, otherwise the S1 flags)

Weights are **16.56 GiB per GPU** (+1.50 vs E64). **KV falls to 255,744 tokens** (E64: 1,021,072) at util 0.85, ≈ 10
concurrent 64K streams by the S4 fit.

| prompt | C | E72 decode | E64 decode | GPU2 clock E72 / E64 |
|---:|---:|---:|---:|---:|
| 1K | 1 | 38.5 | 36.6 | 1905 / 882 MHz |
| 1K | 8 | 116.3 | 126.4 | 1531 / – |
| 1K | 32 | 216.5 | 254.9 | 843 / 1361 |
| 1K | 64 | 313.6 | 371.2 | 730 / 994 |
| 8K | 8 | 97.3 | 118.7 | 745 / 1139 |
| 8K | 32 | 205.9 | 257.1 | 656 / 902 |

- E72 measured 8–20% slower, but GPU2 ran 26–40% slower during the E72 windows, so most of the gap is thermal.
  From bandwidth alone, the extra 12.5% of expert bytes should cost about 4% per step at C = 64.
- **The real E72 cost is memory:** 4× less KV at the same utilization. By the S4 per-stream fit that is about
  34 × 16K or 10 × 64K concurrent streams (E64: about 138 and 41). That is far less headroom for long ARC
  reasoning traces with many concurrent attempts.

### S4: batch ceiling (`runs/s4_c128`, `--max-num-seqs 128`, util 0.88, otherwise the S1 flags)

KV is 1,267,921 tokens at util 0.88. Clean decode with the mean GPU2 clock in each window:

| prompt | C | decode tok/s | per stream | ITL median (ms) | step ≈ (ms) | prefill tok/s | GPU2 clock |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 64 (S1) | 371.2 | 5.8 | 167 | 172 | 755 | 994 MHz |
| 1K | 96 | 393.6 | 4.1 | 246 | 244 | 785 | 775 |
| 1K | 128 | **465.1** | 3.6 | 268 | 275 | 761 | 1075 |
| 8K | 32 (S1) | 257.1 | 8.0 | 118 | 124 | 745 | 902 |
| 8K | 64 | **347.7** | 5.4 | 177 | 184 | 728 | 863 |

- **Throughput keeps rising past C = 64, but slowly:** 371 → 465 tok/s from C = 64 to 128 (+25% for 2× the
  streams), and each stream drops to 3.6 tok/s. The routed-expert reads are already saturated at C = 64 (P3);
  each extra sequence adds about 1.6–2.4 ms per step (attention, indexer, dense layers, larger all-reduces).
  The C = 96 point ran with GPU2 at 775 MHz, so it sits low. The ceiling on this box is ≈ 450–500 tok/s at 1K.
- **8K × 64 gives 348 tok/s**, close to 1K × 64. Context still barely matters for decode. Filling 64 × 8K
  prompts took 12 min at 728 tok/s.
- **Long-run C = 100 (incidental, 65 min).** The first C = 128 attempt opened only 100 streams: aiohttp caps a
  session at 100 connections. `decode_window.py` now passes `TCPConnector(limit=0)`; only this point was
  affected. Those 100 streams decoded for 65 min while their context grew from 1.2K to 14.5K. The server
  logged **320–400 tok/s throughout, rising with GPU2's clock (650 → 810 MHz), not falling with context.**
- **KV capacity is larger than the reported token figure.** Over the 65 min, block usage fits
  **≈ 1.6K + 0.355 × context** reported tokens per stream, with the compressed and indexer caches growing
  sub-linearly. Extrapolating linearly, 64K-token streams need ≈ 24.8K each, so **≈ 41 concurrent 64K streams
  fit at util 0.85** (1.02M) and ≈ 51 at 0.88. That is not 15.6 or 19.4, the reported size ÷ 64K.
  The extrapolation is untested beyond 14.5K. For E72 (255,744) it gives ≈ 10 × 64K, not 3.9.

### P4: prefix caching on vs off (`runs/p4_pc_on`, `p4_pc_on2`, `p4_pc_off`; S1 flags, max-num-seqs 64)

The plan's Task 8 order is CUDA graphs first, then prefix caching. Our serving rule (AGENTS.md) makes prefix
caching mandatory for multi-turn tool use. `scripts/prefix_probe.py` runs two probes against each server.

**Startup and KV.** `--enable-prefix-caching` starts normally with FULL graphs. KV is unchanged at 1,021,072
tokens. The server prints no warning that the cache is disabled for any layer group.

**Parity** (3,000-token random prompt P, greedy 24 tokens):

- Cache on: P_cold takes 4.4 s. P_again hits 2,944 cached tokens (23 full 128-token blocks) and takes 0.73 s.
  P+S (300 more tokens) hits the same 2,944 tokens and takes 1.42 s.
- Cache off: 4.2 / 3.9 / 4.7 s, with 0 hits.
- The test is too weak to show a fine error. On random weights the output distribution is nearly flat
  (chosen-token logprob ≈ −11.4; uniform is −11.77). Greedy tokens diverge at position 1 **even between two cold
  runs on the cache-off server**.
- Cold vs cached P differs by 0.022 nats in the first token's logprob, about the same as cold vs cold on the
  cache-off server (0.018). So no caching-specific deviation shows at this resolution. That includes the Engram
  n-gram lookback across the cached boundary.
- A real check of cached-prefix continuation needs peaked logits, i.e. real weights or a small real fixture.

**Tool-shaped replay.** C attempts × 6 turns. Each turn re-sends the whole history: a 2,048-token base, every
earlier 192-token output and a 384-token "tool result" per turn. 76.5% of prompt tokens are reusable.

| C | cache | wall (s) | generated tok/s | hit rate | mean turn latency, turn 0 → turns 1–5 (s) |
|---:|---|---:|---:|---:|---|
| 16 | off | 532.3 | 34.6 | – | 69 → 75, 89, 103, 112, 83 |
| 16 | on | 208.8 | **88.3** | 0.740 | 58 → 30, 30, 32, 32, 27 |
| 32 | off | 1037.3 | 35.5 | – | 151 → 148, 172, 198, 218, 148 |
| 32 | on (seed 7) | 343.7 | **107.3** | 0.740 | 105 → 47, 50, 50, 51, 40 |

- **Prefix caching makes the tool-shaped run 2.5× (C = 16) to 3.0× (C = 32) faster.** Without it, every turn
  re-prefills the whole history at about 750 tok/s, so turn latency grows with the transcript. With it, latency
  stays flat.
- The hit rate is 0.740 against 0.765 reusable. The gap is block granularity: the partial last block of each
  history is recomputed.
- The first cache-on C = 32 run (`p4_pc_on/replay_c32.json`, hit rate 0.786) is void. Its attempts 0–15 reused
  the seeds of the C = 16 run just before it, so they hit that run's cache. The seed-7 rerun replaces it.
- Generated tok/s is far below decode-only numbers (107 vs 255 at C = 32), because each turn still prefills
  576 new tokens per attempt at the box's slow prefill rate. Turns with more generated tokens (ARC traces) shift
  the balance back toward decode (P5).
- Decode is not slowed by the cache: C = 32 × 1K gives 238.9 tok/s with the cache on, at a GPU2 clock of
  806 MHz. S1 gave 254.9 at 1,361 MHz.

### P5: 12-turn agentic replay (`runs/p5_agentic`; prefix caching on, max-num-seqs 64, util 0.85)

`prefix_probe.py replay`: a 3,000-token base, then 12 turns of 1,024 generated tokens plus a 400-token tool result.
Every turn re-sends the whole history. Seeds 42 (C = 32) and 74 (C = 64).

| C | wall (s) | generated tok/s | hit rate | reusable if outputs are new tokens |
|---:|---:|---:|---:|---:|
| 32 | 2,609.7 | 150.7 | 0.851 | 0.856 |
| 64 | 4,073.0 | **193.1** | 0.851 | 0.856 |

- 193 generated tok/s is 52% of pure decode at C = 64 (371, S4).
- The hit rate equals the reusable fraction *counting the previous turns' outputs as new*. So each turn
  re-prefills the previous turn's 1,024 generated tokens plus the 400 tool tokens: ≈ 1,616 uncached tokens per
  turn per attempt at ~750 tok/s. P7 confirms the cause.
- Estimated cost ≈ 30% of wall time, so reusing generated blocks would make this replay ≈ 1.5× faster.

### C3: E48 (`runs/c3_e48`, `model-e48`: the E64 config with `n_routed_experts` 48; S4 flags)

- Weights 12.35 GiB/GPU (E64: 15.06). KV **2,451,609 tokens** at util 0.88, 1.93× E64's 1,267,921.
  That fits ≈ 100 concurrent 64K streams.
- Decode, 1K context: C1 39.3, C8 146.1, C32 297.6, C64 366.9, C128 380.4; 8K × 32: 236.8 tok/s. GPU2 ran at a
  990 MHz median.
- Same speed as E64 within the box's thermal noise; the box's 1K comparison in P8 gives E48 +3–6%.
  Experts are ≈ 30% of a step, so 25% fewer experts should save only ≈ 9%.

### P6: PR #154, PCIe IPC all-reduce, profiles at C = 16 / 128 (`runs/p6_*`, E64, S4 flags)

| server | 1K × 64 | 1K × 128 | 8K × 32 | GPU2 busy-clock median |
|---|---:|---:|---:|---:|
| base | 345.6 | 396.8 | 224.0 | 900 MHz |
| PR #154 (`DSV41_INDEXER_Q_PREDECODE=1`) | 403.2 | 507.7 | 233.6 | 1,035 MHz |
| PCIe IPC (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) | 360.5 | 418.1 | 230.4 | 1,110 MHz |

- The PR #154 gain was confounded by GPU2's clock (P8 resolves it: +5–15%). The IPC all-reduce gains no more
  than its higher clock explains, and its server takes 590 s to start (vs 70–80 s).
- Profiles (base, 1K context; `p6_prof_c*/summary.json`), per-step ms on rank 0 / GPU2:

| C | ITL | all-reduce | Marlin MoE | indexer | dense MXFP8 | attention | other | all-gather |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 106.5 | 22.5 / 7.1 | 39.3 / 40.7 | 5.6 / 10.7 | 18.7 / 21.0 | 3.4 / 5.9 | 12.3 / 16.5 | 1.5 / 1.4 |
| 128 | 310.5 | 136.6 / 45.3 | 52.9 / 58.5 | 42.4 / 90.6 | 30.7 / 64.7 | 18.2 / 37.3 | 17.7 / 25.9 | 11.4 / 11.0 |

- At C = 128 the cool ranks spend 120–143 ms of a 310 ms step in all-reduce, mostly waiting for GPU2, which runs
  its indexer, dense and attention kernels about 2× slower. Experts read the whole bank from C = 16 on
  (39 → 53 ms). Indexer, dense and attention grow with C (≈ 0.33, 0.11 and 0.13 ms per stream at 1K context).

### P7: are generated tokens cached? (`runs/p7_gencache`, E64, prefix caching on)

A 3,000-token prompt P, then greedy 512 or 1,024 tokens. Then P + those tokens + 16 new ones is re-sent.

| generated | hits | if only the prompt is cached | if the output is cached too |
|---:|---:|---:|---:|
| 512 | 2,944 | 2,944 | 3,456 |
| 1,024 | 2,944 | 2,944 | 3,968 |

**The fork does not reuse KV blocks of generated tokens**, only the prompt's. In tool use, every turn
re-prefills the previous turn's output (P5: ≈ 30% of wall time). Fixing this in the fork, or keeping the box's
prefill fast, matters for the tool mode.

### P8: back-to-back check of PR #154, and stacks (`runs/p8_*`, 1K context, S4 flags)

| run (order) | 1K × 64 | 1K × 128 |
|---|---:|---:|
| E64 base 1 | 443.7 | 520.5 |
| E64 PR #154 1 | 465.1 | 576.0 |
| E64 base 2 | 398.9 | 473.6 |
| E64 PR #154 2 | 450.1 | 546.1 |
| E64 PR #154 + PCIe IPC | 471.5 | 563.2 |
| E48 PR #154 | 484.2 | 584.5 |

- **PR #154 is a real +5–13% at C = 64 and +11–15% at C = 128**, consistent with Kaggle (+11 / +15%). It is
  bit-identical in the kernel bench (K4). It is env-gated to SM120 upstream; our gate enables it on SM89.
- PCIe IPC on top: tie. E48 over E64 (both with PR #154): +3–7%.
- The box ran faster in this hour than in P6 (base 1K × 128: 474–521 vs 397), GPU2 again.

### K4: Kaggle 4×L4, run 4 (`kaggle/runs/gpu_v4`; E64, S4 flags, internet off)

Setup: `kaggle/build_kaggle.py`. A CPU notebook (`chankhavu/dsv41-sm89-fork-py312-cu130`) downloads the box venv's
198 wheels (3.82 GB, sha256-checked fork wheels). The GPU notebook (`chankhavu/e042-dsv41-l4-synthetic`, competition
attached for 4×L4) installs them offline in 43 s, applies the box patches and runs from the kit dataset
(`chankhavu/e042-dsv41-kit`). Runs 1–3 failed at server start, which cost ≈ 45 min of quota:

1. FlashInfer re-JITs the attention module (paths differ from the box) and the link fails on `-lcuda`. Fix: a
   `stubs/libcuda.so` symlink to the host driver.
2. The FlashInfer `sampling` JIT fails the CCCL check (pip nvcc 13.4 vs CUDA 13.0 headers). Fix:
   `VLLM_USE_FLASHINFER_SAMPLER=0`; greedy decoding never uses top-k/top-p.
3. Box-built `.so` files as FlashInfer AOT modules need `GLIBCXX_3.4.32`, which the image's libstdc++ lacks.
   Dropped; the attention module JIT-compiles fine on Kaggle (server up in 350 s).

**Timing:** Kaggle showed run 4 as RUNNING from 06:21 UTC, but its first cell ran at 08:59 (`env.txt`). It sat
waiting for a 4×L4 machine for 2 h 38 min. The user cancelled it at 09:43, 44 min in, during the last profile.
All other outputs were saved.

**Hardware:** GPUs at 1,545–1,785 MHz under load (72 W power cap, 78–83 °C), even across the four.
NCCL all-reduce is 2.2× faster than the box at 2,048 tokens (run 1).

| point | Kaggle base | Kaggle PR #154 | box base (P6 / P8) |
|---|---:|---:|---:|
| 1K × 1 | 39.4 | | 39.3 (C3) |
| 1K × 8 | 140.8 | | 146 |
| 1K × 32 | 302.9 | | 298 |
| 1K × 64 | 460.8 | **509.9** | 346–444 |
| 1K × 128 | 618.6 | **708.3** | 397–521 |
| 8K × 32 | 305.1 | 326.4 | 224 |
| 32K × 8 | 146.4 | | 127 (S1) |

- Balanced GPUs give +33–56% at C ≥ 64 and nothing at C ≤ 32: the box's GPU2 costs it most at large batches.
- **At 1K context, PR #154 reaches 510 tok/s at C = 64 and 708 tok/s at C = 128.** Long context at high C is
  not measured yet (P9, K5).
- Profile at C = 64 (base, `prof_c64/summary.json`), per-step ms: Marlin MoE 49.9, all-reduce 22–28 (box: 62),
  indexer 22–25, dense MXFP8 17, other 14, attention 10–11, all-gather 3.6. That sums to ≈ 140 ms = ITL. On
  balanced GPUs the experts are the largest cost (36%), near the HBM roofline (12 GB/GPU read per step).
- Indexer kernel bench (PR #154, Kaggle L4): bit-identical at all 15 shapes, 1.4–1.74× faster at width ≥ 4K
  (e.g. 128 rows × 32K width: 8.19 → 4.71 ms).

### P9: long-context grid on the box (`runs/p9_e48`; E48, PR #154, S4 flags)

| point | tok/s | ITL | GPU2 busy median |
|---|---:|---:|---:|
| 1K × 32 | 346.1 | 92.5 ms | 1,140 MHz |
| 32K × 32 | 271.5 (−22%) | 106.4 ms | |

Stopped after these two points for the trace-shaped runs (R). `decode_window.py --profile-s` works, but **the fork
records only the first profiler capture per server**: later captures rewrite the first one's text table and write
no trace. So each server is profiled once.

### K5: Kaggle run 5, long context (`kaggle/runs/gpu_v5`; E48, PR #154, max-model-len 64K, no prefix caching)

Started at once (09:57:45, no machine wait) and ran 1 h 46 min, with a 2 h hard stop (watchdog) and a
fits-before-the-stop check per point. GPUs at 1,620–1,665 MHz.

| point | tok/s | ITL | prefill |
|---|---:|---:|---:|
| 1K × 128 | 761.6 | 168.1 ms | 1,095 tok/s |
| 32K × 64 | 552.5 | 115.8 ms | 1,190 tok/s |
| 32K × 128 | **736.0** | 173.7 ms | 1,170 tok/s |

- **On balanced GPUs 32K context costs 3.4% at C = 128.** The box loses 22% at 32K × 32 (P9) and 20–30% on the
  trace-shaped snapshot (R1), because its throttled GPU2 runs the context-dependent kernels (indexer, attention)
  about 1.7× slower and the other ranks wait in the all-reduce.
- Profile at 1K × 128 (per step, rank 0): all-reduce 40.7, Marlin MoE 40.3, dense MXFP8 27.4, indexer 21.4,
  attention 20.8, other 18.3, all-gather 7.1 ms.

### R: trace-shaped load from the real Flash-1017 runs (`scripts/trace_shapes.py`, `run_r.sh`)

Shapes (`metadata/trace_shapes/`, token counts only):

- **No tools:** E034 (ARC Prize prompt, max effort, 334 calls). Prompt p50 6.5K, mean 7.2K. Output p50 45K, mean
  50K, p90 93K, max 146K.
- **Tools:** E030 hdrmin (sub80s40 + header, 334 attempts). 12 turns at the median (p90 27, max 40). First prompt
  7.2K. Generated 23K per attempt (mean 27K), ≈ 955 per turn. Tool output ≈ 300 per turn. Final context p50 35K,
  p90 81K, max 97K.

In a closed loop each slot sits in a trace with probability ∝ its length, at a uniform position. Steady-state slot
context: no tools mean 40.6K (p90 79K, p99 120K), tools mean 33.7K. With the S4 KV fit, E64 holds C = 64 and E48
C = 128 at p95 ≤ 91% of KV.

Server: PR #154, prefix caching on, max-model-len 160K, SEQS 128, util 0.88. At 160K, vLLM reports the E64 KV as
2,069,257 tokens (12.6 × 160K requests); the unit depends on max-model-len.

**R1, no tools, E64 × 64** (`runs/r_e64/r1_notools_c64.json`): snapshot contexts mean 41.9K, p50 36K, p90 77K, max
111K. 2.68M tokens prefilled in 62 min (723 tok/s), then a 120 s window: **324.3 tok/s**, ITL 193 ms
(1K × 64 with PR #154 on the box: 403–465). Profile (per step): cool ranks all-reduce 58, MoE 49, indexer 24,
dense 17, attention 10 ms; GPU2 indexer 39, attention 18, dense 24 ms.

**Generated-token caching (P11, 2026-10-22).** By default the fork keeps sliding-window KV only at the prompt end
(`prefix_cache_retention_interval` 0), so a re-sent transcript hits only the previous prompt: P7 hit 2,944 tokens
of 3,000 + 512 / 3,000 + 1,024 re-sent. With `--prefix-cache-retention-interval 128`, P11 hit **3,456 / 3,968**,
the whole re-sent text up to block granularity. Retention controls which blocks get hashed; it does not pin them.

**R2, tools, E64 × 64, flag off** (`runs/r_e64/kvmon.log`; the replay client was stopped, server metrics only):
- Cold setup: 1.88M history tokens in 45 min.
- Steady state 12:23–13:03 UTC: **151 generated tok/s**, prefix hit rate 81% (72% after 12:38), 433 uncached
  prompt tok/s. That is 2.9 re-prefilled tokens per generated token, against 1.5 expected from the shapes. The
  server spent most of its time prefilling: box prefill tops out near 720 tok/s.
- Unverified guess for the extra misses: the sliding-window blocks at the prompt end are freed after the window
  passes them, early in a turn, and long turns outlive them in the free queue.
- Expected hit rate with the flag on: 96.4%, about 0.5 re-prefilled tokens per generated token.

`tool_replay.py` bugs fixed on 2026-10-22:
1. Setup waited for each slot's whole first turn, not its first token. Length-biased first turns reach 36K tokens,
   hours of decode.
2. Random ids were appended in place of the generated tokens, so generated-token blocks could never hit. The
   script now sends back the generated ids (`return_token_ids`).
3. With `return_token_ids`, the first SSE chunk echoes the prompt ids. Past ~85K tokens that line exceeds aiohttp's
   512 KB readline limit. This crashed the box E64 run with the flag (`r_e64ret`, after 30 min of setup) and would
   have crashed Kaggle run 6. The script now splits lines from `iter_any()`, tested with a 200K-token prompt.

**Kaggle run 6** (`kaggle/runs/gpu_v6`): E48 × 128 tools with the flag. Cancelled after 32 min (0.54 h) because of
bug 3; no throughput numbers. Data points:
- At max-model-len 160K with prefix caching, the KV cache is 3,988,970 tokens (box: 4,001,043).
- Cold prefill ran at about 1,190 tok/s, so the 3.62M history tokens need about 51 min.

Run 7 (the same run, fixed script, `HARD_STOP_S` 5600 to fit the 4 h cap) pushed 2026-10-22 18:00 UTC.

**R1, no tools, E48 × 128** (`runs/r_e48ret/r1_notools_c128.json`, flag on, PR #154):
- Snapshot contexts: mean 43.0K, p50 40K, p90 80K, max 144K. 5.50M tokens prefilled in 128 min (714 tok/s).
- 120 s window: **407.5 tok/s**, ITL 308 ms, 3.2 tok/s per stream (E64 × 64 on the same snapshot type: 324.3).
- Profile per step, ms (`trace/summary.json`):

  | Rank | Indexer | MoE | Dense | Attention | All-reduce |
  |---|---:|---:|---:|---:|---:|
  | Cool ranks | 46–52 | 40–41 | 27–31 | 18–22 | 132–149 |
  | GPU2 (throttled) | 92 | 49 | 50 | 42 | 45 |

  The cool ranks spend their all-reduce time waiting for GPU2. With GPU2 matching the others, the step would be
  about 167 ms of compute plus about 45 ms of communication, so balanced (Kaggle-class) GPUs would reach about
  600 tok/s. That is an inference, not a measurement; Kaggle measured 736 at a uniform 32K × 128.

**R2, tools, E48 × 128, flag on** (`runs/r_e48ret/r2_tools_c128.json`, `kvmon.log`):
- Cold setup: 3.62M history tokens in 87 min (setup_s 5213).
- The 40-min window average is **188.8 tok/s at an 80% hit rate**, but its first 19 minutes were still recovering
  from the cold start. Turns that slots began during setup waited behind the cold prefills (TTFT up to 58 min), the
  prefill churn had evicted their cached prefixes, and they re-prefilled at 0% hits.
- Steady part from server metrics:

  | Window (UTC) | Generated tok/s | Hit rate | Re-prefilled per generated token |
  |---|---:|---:|---:|
  | 17:53–18:03 | 284 | 91% | 0.81 |
  | 18:03–18:13 | **366** | **97%** | 0.34 |

  Turns that started after 17:52 had a TTFT p50 of 1.5–1.9 s. With the flag off (E64 × 64): 151 tok/s, 81% hits,
  2.9 re-prefilled per generated token.
- The 18:03–18:13 stretch still carried the tail of the first turns (first-turn end p90 at 18:08), which are decode
  only, so true steady state is probably 330–370 tok/s on the box.

`tool_replay.py` v4:
- The first turn starts at a uniform point inside its completion, matching steady state (before, it generated the
  whole length-biased completion).
- After setup, the turns queued behind the cold prefills get their first token before the warm-up starts
  (`drain_s`).

R3 (`run_r3.sh`, after P10): a clean A/B of the flag on E64 × 64 with v4 (results after run 8 below).

**Kaggle run 7, tools, E48 × 128, flag on** (`kaggle/runs/gpu_v7`; 18:01:22 to 19:30:50 UTC, 1.49 h; E042 Kaggle
total about 3.8 of the 4 approved hours). Uses `tool_replay.py` v3, without the v4 drain.
- Server up at 410 s (KV 3,988,970 tokens). Cold setup 3,171 s at about 1,150 tok/s.
- Window average (1,478 s): 342.3 tok/s, 86.3% hits. Like the box, it includes the cold-start recovery: 0% hits
  until t ≈ 4,300 s.
- Settled part from `kvmon.log`, t 4,612–5,333 s (12 min, three 5-min bins at 494 / 508 / 505):

  | Generated tok/s | Hit rate | Re-prefilled per generated token | Per stream | TTFT p50 / p90 |
  |---:|---:|---:|---:|---:|
  | **505.5** | **96.8%** | 0.37 | 3.9 tok/s | 1.3 / 3.6 s (104 turns, prompt p50 21K) |

- GPU clocks were even: median 1,575–1,665 MHz, 73–76 °C. Box settled (18:03–18:13): 366 tok/s, so box / Kaggle =
  0.72.
- The settled part still holds some first turns (first-turn end p90 at 5,127 s), which are decode only. At the
  steady-state 0.53 re-prefilled per generated token, a decode-plus-prefill model (1,190 tok/s prefill) gives about
  **470–505 tok/s**.
- Reference, Qwen 3.8 27B FP8 (E027) on real tool traffic: 224–246 tok/s on Kaggle.

**P10: box twins of Kaggle run 5** (`runs/p10_e48`; E48, PR #154, max-model-len 64K, no prefix caching; 18:14–20:37
UTC). During the 32K × 128 window GPU2 ran at about 1.0 GHz (88 °C), the others at 1.5–2.0 GHz.

| point | box tok/s | Kaggle tok/s | box / Kaggle | box prefill | Kaggle prefill |
|---|---:|---:|---:|---:|---:|
| 1K × 128 | 629.3 (ITL 201 ms) | 761.6 | 0.83 | 912 tok/s | 1,095 tok/s |
| 32K × 64 | 466.1 (ITL 137 ms) | 552.5 | 0.84 | 799 tok/s | 1,190 tok/s |
| 32K × 128 | 450.1 (ITL 277 ms) | 736.0 | **0.61** | 764 tok/s | 1,170 tok/s |

- 32K context costs the box 28% at C = 128 (Kaggle: 3.4%). The throttled GPU2 runs the context-dependent kernels
  and the others wait on it; the cost grows with both context and batch.
- Box prefill drops with prompt length (912 → 764 tok/s); Kaggle's does not (1,095 → 1,170).

**Kaggle run 8, tools, E64 × 64, flag on** (`kaggle/runs/gpu_v8`; 20:35:24 to 22:01:29 UTC, 1.44 h; `tool_replay.py`
v4, `HARD_STOP_S` 5400). The Kaggle twin of R3's flag-on half. Quota before the push: 15.09 of 30 h left this week.
- Server up at 415 s (KV 2,057,184 tokens). Setup 1,797 s (all 64 slots), drain 355 s, window 2,292 s.
- Window average: 339.2 tok/s, 95.8% hits, 0 preemptions, `ids_back_ok` 1.0. With v4 the window is nearly all
  settled: 0% hits only during the cold prefill (first 40 min of the replay).
- Settled stretch from `kvmon.log` (replay minutes 45–79, 34 min):

  | Generated tok/s | Hit rate | Re-prefilled per generated token | Per stream | TTFT p50 / p90 | KV used |
  |---:|---:|---:|---:|---:|---:|
  | **349.0** (55–79 min: 355.6) | **96.2%** | 0.63 | 5.5 tok/s | 1.33 / 10.4 s (402 turns, prompt p50 24.6K) | 66% mean, 72.6% peak |

- GPUs power-capped (≈ 71 of 72 W) at 1,500–1,650 MHz median, 72–77 °C.
- **Compared with run 7 (E48 × 128, 505.5):** −31%, half the streams. Run 8's re-prefill (0.63 per generated token)
  matches the steady-state expectation (0.53); run 7's settled part (0.37) still held decode-only first turns, so its
  true steady state is nearer the low end of 470–505.
- KV peaked at 73%, so E64 has room for more than 64 slots in tool mode (about 80–85 by this margin, untested).
- 12 h budget model (`budget12h.py` in the session scratchpad; E030 attempt shapes, measured per-concurrency speeds,
  late attempts given a budget that fits the time left): with a 30K generated-token cap per attempt, E64 × 64 at
  349 tok/s gives ≈ 735 rollouts (attempt median 1.0 h, max 1.6 h). E48 × 128 at 470–505 gives ≈ 1,010–1,090
  (median 1.3 h, max 2.2–2.3 h). With E030's own 80K budget: ≈ 580 and ≈ 800–860. A 30K cap truncates 35% of the
  E030 attempts; its accuracy cost is unmeasured.

**R3: box A/B of the flag, E64 × 64 tools** (`runs/r3_e64ret`, `runs/r3_e64off`; `tool_replay.py` v4, seed 1, window
2,400 s each; flag on 20:37–22:22, off 22:23–23:58 UTC):

| | Generated tok/s | Hit rate | Uncached prompt tok/s | Re-prefilled per generated token | TTFT median | Turn median | Turns in window |
|---|---:|---:|---:|---:|---:|---:|---:|
| flag on (`--prefix-cache-retention-interval 128`) | **209.0** | 94.4% | 194.6 | 0.93 | 2.01 s | 275 s | 294 |
| flag off | 103.2 | 72.1% | 536.3 | 5.2 | 6.06 s | 496 s | 157 |

- The flag doubles box tool-mode throughput (2.03×). Setup 2,957 / 2,975 s, 0 preemptions, `ids_back_ok` 1.0 in both.
- Box vs Kaggle run 8 (same load, flag on): 209.0 vs 339.2 window (349 settled), **0.60–0.62×**, the same ratio as
  P10's 32K × 128 (0.61). The box understates long-context tool mode by ≈ 40%; use it for A/Bs.
- The flag costs no KV on this fork: through setup both halves used the same KV at every 5-min mark (e.g. 0.4514 vs
  0.4525 at 35 min). Retained SWA blocks only compete in the evictable cache. So encoder-side SWA bounded replay
  (vLLM, 2026-10-07: cache only global KV, re-run the last 128 tokens on a hit) would not free streams here.

### P12: kernel and prefill A/B from the vLLM V4.1 and Humming posts (`run_p12.sh`, `run_p12b.sh`; started 23:59 UTC)

vLLM's V4.1-Flash post (2026-10-07) and Humming (2026-10-21) list many optimizations; most need SM90/SM100, NVLink or
multi-pool deployments. The ones that run on SM89 and are already in the installed fork:
- Humming kernels (`humming_kernels` 0.1.12 in the venv; SM75+, NVRTC JIT): `--moe-backend humming` (MXFP4 experts,
  instead of Marlin), `--linear-backend humming` (MXFP8 dense, instead of `AdaMxfp8LinearKernel`), FP8 activations
  for the experts via `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"quant_method":"humming","dtype":"float8e4m3"}'`.
- W4A8 Marlin: `VLLM_MARLIN_INPUT_DTYPE=fp8` (`serve.sh` `MARLIN_IN=fp8`).
- CED prefill: `--hf-overrides '{"ced_prefill": true}'` (decoder-side bounded replay; fork: model runner V2, no
  PP/DP/DBO).
- Not in the fork: Engram THP (`--engram-config use_thp`) and async Engram prefetch. They matter only for a full-size
  table (≈ 130–150 GiB on Kaggle); the 7.9 GiB synthetic table's 3% decode cost will not transfer.
- Async scheduling is already on (the fork enables it by default with the mp executor).

**NCCL copy-engine SHM all-reduce** (`runs/nccl/p12_default`, `shm_ce1`, `shm_ce3`; GPUs idle, before P12's servers),
ms per bf16 all-reduce of T × 5120:

| T (tokens) | 1 | 8 | 32 | 64 | 256 | 1024 | 2048 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.043 | 0.073 | 0.220 | 0.297 | 1.444 | 6.92 | 13.96 | 57.1 |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` | 0.049 | 0.081 | 0.219 | **0.635** | 0.671 | 1.85 | **3.53** | 13.9 |
| + `NCCL_SHM_MEMCPY_MODE=3` | 0.049 | 0.066 | 0.221 | 1.177 | 0.840 | 1.91 | 3.61 | 14.0 |

- The copy engine makes prefill-size all-reduces 4× faster (box SHM 1.5 → 6.0 GB/s) but 64-token ones 2× slower.
  Prefill has 84 all-reduces per 2,048-token chunk (≈ 1.17 → 0.30 s of a ≈ 2.4 s step); a C = 64 decode step pays
  ≈ +0.34 ms per all-reduce. Net effect is measured end to end in P12b, then in a tool replay (R4) against R3's 209.0.
- Kaggle's default SHM is ≈ 3× faster than the box's (4.9 GB/s), so the box overstates the prefill gain there.

**P12 results** (`runs/p12_*`; E64, PR #154, 128 seqs, MBT 2048, util 0.88, max-model-len 64K, no prefix caching;
decode tok/s / prefill tok/s, % vs the mean of the three runs on the default kernels; `scripts/dw_table.py`):

| run | 1K × 64 | 1K × 128 | 8K × 16 | 32K × 4 | KV tokens | weights / GPU |
|---|---:|---:|---:|---:|---:|---:|
| base mean (base, hlin, base2) | 468 / 936 | 561 / 858 | 178 / 826 | 84 / 770 | 1,267,921 | 15.06 GiB |
| base (00:00) | 464 / 934 | 561 / 850 | 179 / 827 | 85 / 771 | | |
| hlin (fell back to Ada, so a base run; 00:15) | 461 / 933 | 548 / 841 | 176 / 821 | 83 / 775 | | |
| base2 (01:02) | 480 / 940 | 574 / 882 | 180 / 830 | 83 / 764 | | |
| **Humming experts + dense** (hboth) | 469 / 1,043 (+0% / +11%) | **610 / 988 (+9% / +15%)** | 180 / 979 (+1% / +18%) | 86 / 922 (+3% / +20%) | **610,701** | 16.36 GiB |
| Humming experts, FP8 per-token activations, Ada dense (hw4a8) | 447 / 963 (−5% / +3%) | 570 / 873 (+2% / +2%) | 189 / 875 (+6% / +6%) | 87 / 818 (+3% / +6%) | 658,945 | |

- The three default-kernel runs agree within ±3%, so GPU2's drift was small tonight.
- **Humming (experts + dense):** prefill +11–20%, decode +9% at 1K × 128, flat at C ≤ 64 (experts are
  bandwidth-bound there). It costs **1.30 GiB more weights per GPU**, which halves the KV (2.96 → 1.43 GiB). The
  experts account for most of it (hw4a8, Humming experts only: 659K KV). E48 has the KV to spare; E64 × 64 tool mode
  (73% peak of 2.07M) would be tight; E72 would not fit.
- FP8 activations for the experts (hw4a8) add nothing; the prefill gain comes with the Humming dense kernel (P12b
  separates experts-only and dense-only).
- Needed `patches/humming_v41.py` (applied on the box at 00:16 UTC, opt-in, defaults unchanged): the fork whitelists
  V4.1's experts to auto/b12x/marlin (hmoe failed at start, 00:14) and its dense MXFP8 to FlashInfer CUTLASS / Ada
  (`--linear-backend humming` silently fell back to Ada).
- **Marlin W4A8 is not available for V4.1:** the fork refuses `VLLM_MARLIN_INPUT_DTYPE` ("V4.1 uses packed Marlin
  W4A16 after its official K32 MXFP8 quantize/dequantize boundary"). The default path already quantizes expert
  activations to MXFP8 and dequantizes them before a BF16 Marlin GEMM.
- **CED needs `max_num_batched_tokens >= 128 × max_num_seqs`** (the tail replay of every running request must fit
  one step): 16,384 at 128 seqs, 8,192 at 64. P2's MBT 8192 alone cost 85% of E64's KV, so CED at E64's
  concurrency does not fit 24 GB as implemented. P12b measures it at 16 seqs × MBT 2048 and 64 seqs × MBT 8192.

**P12b, part 0: Humming split and CED** (`runs/p12b_*`; same flags as P12 unless noted; % vs P12's base mean):

| run | 1K × 16 | 1K × 64 | 1K × 128 | 8K × 16 | 32K × 4 | KV tokens | weights / GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| base mean (P12) | | 468 / 936 | 561 / 858 | 178 / 826 | 84 / 770 | 1,267,921 | 15.06 GiB |
| **Humming dense only** (hlin) | | 475 / 1,046 (+1% / +12%) | **616 / 970 (+10% / +13%)** | 194 / 966 (+9% / +17%) | 90 / 893 (+7% / +16%) | 1,226,939 | 15.11 GiB |
| Humming experts only (hmoe) | | 429 / 934 (−8% / −0%) | 531 / 823 (−5% / −4%) | 162 / 817 (−9% / −1%) | 78 / 769 (−7% / −0%) | 687,492 | 16.31 GiB |
| **CED**, 16 seqs × MBT 2048 (ced16) | 206 / 1,244 | | | 180 / **1,554 (+1% / +88%)** | 81 / **1,487 (−3% / +93%)** | 852,698 | 15.76 GiB |
| no CED, 64 seqs × MBT 8192 (mbt8k64) | | 435 / 821 (−7% / −12%) | | 174 / 709 (−3% / −14%) | 82 / 689 (−2% / −11%) | 251,231 | 15.15 GiB |
| CED, 64 seqs × MBT 8192 (ced64) | failed: no memory for KV | | | | | 0 | 17.97 GiB |

- **The Humming dense MXFP8 kernel is the win:** decode +7–10% (except 1K × 64, +1%), prefill +12–17%, for +0.05 GiB
  and −3% KV. The Humming expert kernel is slower than Marlin (−5 to −9% decode) and is what costs the 1.25 GiB.
  Caveat (found in R5): hlin's slowest GPU ran ≈ 4% faster than the base runs' (1,198 vs 1,121–1,180 MHz) and
  hmoe's ≈ 8% slower (1,059), so part of both effects is thermal; R7 repeats the A/B with clock control.
- **CED nearly doubles prefill** (1,554 / 1,487 vs 826 / 770 tok/s) with unchanged decode, but its per-request tail
  state is preallocated for every one of `max_num_seqs`: ≈ 45 MB per sequence per GPU (+0.70 GiB at 16 seqs,
  +2.91 GiB at 64), plus MBT ≥ 128 × seqs. At 64 seqs nothing is left for KV. Usable at our concurrency only with a
  fork change that keeps tail state for the requests in prefill (a small slot pool) and drops the MBT rule to
  128 × slots.
- MBT 8192 at 64 seqs costs 80% of the KV and is slower even for prefill here (P2's single-prompt gain did not hold).

**P12b, part 1: NCCL copy-engine variants** (`runs/nccl/b_*`; ms per bf16 all-reduce of T × 5120):

| T | 64 | 96 | 128 | 192 | 256 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.310 | 0.412 | 0.611 | 1.193 | 1.581 | 3.318 | 13.88 | 57.2 |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` | 0.641 | 0.593 | 0.525 | 0.519 | 0.600 | 0.996 | 3.53 | 14.0 |
| + `NCCL_SHM_MEMCPY_MODE=2` | 0.707 | 0.696 | 0.873 | 1.213 | 1.625 | 3.331 | 14.02 | 56.9 |
| + `NCCL_PROTO=LL` | 0.434 | 0.655 | 0.879 | 1.339 | 1.780 | 3.611 | 14.22 | 56.8 |
| + `NCCL_PROTO=LL128` | 0.515 | 0.665 | 0.914 | 1.315 | 1.735 | 3.486 | 14.07 | 57.3 |
| + `NCCL_PROTO=LL,Simple` | 0.598 | 0.602 | 0.520 | 0.533 | 0.602 | 0.995 | 3.52 | 14.0 |

- The copy engine has a ≈ 0.5–0.6 ms floor: it loses below ≈ 110 tokens (decode at C ≤ 96) and wins from 128 up
  (4× at prefill sizes). Restricting the protocol does not remove the floor; mode 2 and LL/LL128 lose the gain.
  It is a per-process setting, so it cannot be applied to prefill-size messages only.
- **End to end, the copy engine hangs vLLM's startup** (`runs/p12b_ce1`, `NCCL_SHM_USE_CUDA_MEMCPY=1`): weights
  load (02:07), then no progress at the first forward / memory profile until the 2,700 s timeout (02:51). Not
  debugged: with its decode-size floor it would lose in decode-heavy tool mode anyway. **Dropped**, so R4 (CE tool
  replay) did not run; `run_r5.sh` (Humming dense tool replay, A/B vs R3's 209.0) started at 02:52.

**R5: Humming dense in tool mode** (`runs/r5_e64hlin`; otherwise R3's r3_e64ret: E64 × 64, PR #154, prefix caching,
retention 128, max-model-len 160K, util 0.88, tool_replay seed 1):

| run | gen tok/s | hit rate | uncached prompt tok/s | TTFT median | turn median | setup | slowest-GPU clock in window |
|---|---:|---:|---:|---:|---:|---:|---:|
| r3_e64ret (Ada dense) | 209.0 | 94.4% | 194.6 | 2.01 s | 275 s | 2,957 s | 721 MHz |
| r5_e64hlin (Humming dense) | **289.4 (+38%)** | 95.0% | 239.9 | 1.32 s | 195 s | 2,306 s | 1,038 MHz (+44%) |

KV 2,002,373 tokens vs 2,069,257 (−3%); turn prompts p50/p90 24.8K / 57.8K.

- **Not a clean A/B.** GPU2 sits at its 87 °C limit in both runs and throttles (R3 window mean 869 MHz, R5 1,198),
  and in TP4 every step waits for the slowest GPU. R5 ran in cooler hours (02:54–04:24 UTC vs 20:39–22:22), and its
  slowest GPU was 44% faster, about as much as the throughput gain. Per-30-s intervals cannot separate the two:
  prefill-heavy intervals run at higher clocks (the GPUs wait on PCIe all-reduce), so clock and phase are confounded.
- `nvidia-smi -lgc` (clock locking) is not permitted in the container. **R7** therefore interleaves short decode
  benches B H H B E B H (`run_r7.sh`; `decode_window.py` now records `t0_unix` / `window_start_s`) and fits
  log tok/s = a + b · log(slowest-GPU clock) + arm per point (`scripts/clock_dw.py`).

**R6: Engram 8× table, OOM-killed** (`runs/r6_engram8x`). With ENGRAM_ROWS 32M (7.87 GiB per rank and layer, 63 GiB
pinned) the workers were killed during startup (cgroup `memory.events`: oom_kill 2). The container's cgroup limit is
107.9 GiB (`memory.max`), though `free` shows 125 GiB total and 112 available. So the box cannot hold a full-size
table (130–190 GiB on Kaggle); R7 tries 4× (16M rows, 31.5 GiB) inside its interleaved A/B. R6's E48 × 128 tool
replays moved behind R7's A/B (R6 stopped 2 min into r6_e48base's setup).

**R7, part 1: clock-controlled A/B of Humming dense** (`run_r7.sh`, `runs/r7_*`; 04:31–05:47 UTC; P12's decode-bench
server: E64, 128 seqs, util 0.88, MBT 2048, 64K, no prefix caching; order B H H B E B H; decode / prefill tok/s):

| run | 1K × 128 | 8K × 32 | KV tokens |
|---|---:|---:|---:|
| base mean (b1–b3) | 587 / 899 | 282 / 848 | 1,267,921 |
| r7_b1, r7_b2, r7_b3 | 608 / 917, 576 / 890, 576 / 890 | 290 / 866, 275 / 840, 280 / 838 | |
| Humming dense h1, h2, h3 | 646 / 1,027, 644 / 1,030, 636 / 1,024 | 302 / 995, 299 / 992, 288 / 984 | 1,226,939 |
| Engram 4× table (e4x) | 580 / 901 | 277 / 842 | 1,262,929 |

Fit per point and phase: log tok/s = a + b · log(slowest-GPU clock) + arm (`scripts/clock_dw.py`; slowest-GPU clock
875–1,264 MHz in decode windows, 1,068–1,712 in prefill):

| point | phase | clock elasticity b | Humming dense, clock-adjusted | raw | Engram 4×, clock-adjusted | raw |
|---|---|---:|---:|---:|---:|---:|
| 1K × 128 | decode | 0.30 | **+8.4%** | +9.5% | −0.6% | −1.1% |
| 1K × 128 | prefill | 0.21 | **+11.5%** | +14.2% | +0.6% | +0.2% |
| 8K × 32 | decode | 0.27 | **+4.4%** | +5.3% | +0.3% | −1.6% |
| 8K × 32 | prefill | 0.20 | **+13.4%** | +16.8% | −1.5% | −0.7% |

- **Humming dense is a real gain:** decode +4–8%, prefill +11–13% after the clock adjustment (the raw figures are an
  upper bound, the adjusted ones a lower bound if the kernel itself lets the GPUs clock higher). The base runs agree
  within ±4%; the three Humming runs within ±2%. KV −3%.
- Throughput is not very clock-sensitive in this range (elasticity 0.2–0.3: memory- and PCIe-bound). R5's +44% clock
  explains about +10% at that elasticity, but R3's 721 MHz is below the fitted range, so R5 cannot be adjusted
  reliably. **Tool-mode expectation** from the decode-plus-prefill model (r = 0.93 re-prefilled per generated token,
  box): **+6–9%**, not R5's +38%.
- **Engram 4× (16M rows per rank and layer, 3.93 GiB, 31.5 GiB pinned in all): no measurable cost** (within ±1.6%,
  inside the base runs' spread). Lookups do not slow down from 7.9 to 31.5 GiB of table; 8× did not fit the cgroup.
- Part 2 (E48 × 128 tool replays, base then Humming dense) started at 05:47. **The box went offline around 07:40 UTC**
  (r7_e48base mid-replay; ssh and the Vast proxy unreachable, ping fails, Vast shows `actual_status` offline with a
  stale 100% GPU util; still offline at 17:47). Part 2 results are unknown until it returns; if it rebooted, re-run
  part 2 (the replays need ~2 h each).

**Kaggle run 9: E64 at higher concurrency** (`kaggle/runs/gpu_v9`; pushed 17:55 UTC on user request, "we're not
comparing apples to apples"; `HARD_STOP_S` 11,400 ≈ 3.2 h of the ≈ 13.6 h left before the 2026-10-24 reset). Run 8's
E64 × 64 peaked at 73% KV while run 7's E48 × 128 had more slots, so run 9 replays E64 × 96 (wall cap 5,200 s), then
E64 × 80 on a fresh server, both with run 8's settings (Ada dense, Marlin, PR #154, flag on, `tool_replay.py` v4,
seed 1). 64 / 0.726 ≈ 88 slots fill the KV at run 8's peak, so 96 should show whether preemptions cost more than
the extra streams give. Done 21:06 UTC (11,164 s ≈ 3.1 h; ≈ 10.5 h of the week left before the reset). Window
figures from `tool_replay.py`; settled = the 5-min `kvmon.log` bins once every slot runs with no queue:

| | E64 × 64 (run 8) | E64 × 80 | E64 × 96 |
|---|---:|---:|---:|
| setup / drain / window (s) | 1,797 / 355 / 2,292 | 2,110 / 523 / 2,280 | 2,688 / 802 / 1,410 |
| generated tok/s, window | 339.2 | 348.4 | **57.9** |
| generated tok/s, settled | 349 (55–79 min: 356) | **355** (55–85 min) | never settles |
| hit rate (window) | 95.8% | 94.1% | **0%** |
| TTFT median / turn median | 1.33 s / – | 1.39 s / 216 s | 603 s / 1,766 s |
| per stream | 5.5 tok/s | 4.4 tok/s | – |
| KV peak | 0.73 | 0.88 | 0.99 |

- **E64 saturates at ≈ 350–355 tok/s by 64 slots.** 80 slots tie with 64 and only slow each attempt down by 25%,
  so for the 12 h budget 64 stays the better setting (≈ 735 rollouts).
- **96 slots collapse:** live context fills the KV (0.99 from 70 min on), so every finished turn's blocks are taken
  at once by waiting prefills and the next turn re-prefills its whole transcript at ≈ 1,100 tok/s. The cold
  setup never ends: at 85 min 91 run and 9–13 wait, 0% hits throughout, 0 preemptions (the scheduler admits only
  what fits). This is the cache-health failure of the serving rule (AGENTS.md): concurrency needs KV headroom.
- So the E48 × 128 vs E64 × 64 comparison was not under-filling E64: its throughput is flat beyond 64. (E48's
  470–505 is from run 7's v3 replay and was not re-measured with v4.)

### T72: E72 tool mode on a Runpod 4×L4 (`scripts/run_e72t.sh`, `runs/t72_*`; started 2026-10-24 00:52 UTC)

The user rented a Runpod 4×L4 pod (pod d1jzls3yrcfet9, $2.36/h, 250 GB RAM, 50 GB disk, driver 595.91, CUDA 13.0
in the image; `nvidia-smi topo -p2p r` all OK, unlike the Vast box). `setup_box.sh` + `box_install.sh` rebuilt the
venv in ~2 min ("INSTALL OK" 00:23). Default NCCL (SHM, as on Kaggle). The question: user experiments put E72 at
≈ 90% of the full model's quality, so what does it give in tool mode?
- Kaggle tool config (128 seqs, util 0.88, 160K, prefix caching, retention 128, PR #154 flag on, `tool_replay.py`
  v4, seed 1) at E72 × 32, × 40, × 24, then E64 × 64 as the anchor against Kaggle's 349 tok/s.
- **E72 KV: 821,422 tokens** (1.18 GiB per GPU; weights 16.56 GiB), vs E64's 2,057,184 on Kaggle. By E64's KV per
  slot at run 9's healthy peaks (23.5K at 0.73, 22.6K at 0.88) that is ≈ 25–32 healthy slots.
- GPUs at 64–66 °C and 1.5–1.6 GHz under load at the start (the Vast box's GPU2 throttled to 0.7–1.0 GHz).
- E72 × 32, early 5-min intervals after the cold setup (server metrics, not the replay window): 129–145 tok/s
  while recovering, then 230–254 tok/s at 91–95% hits, one dip to 196 at 86%; KV 0.79–0.91.
- **E72 × 32 tools (40-min window): 205.3 tok/s, 87.8% hits**, uncached prefill 421 tok/s, TTFT median 0.91 s,
  turn median 138 s, 21 attempts finished in the window, 0 preemptions; setup 807 s. Below the 250–290 estimate:
  at the KV edge the hit rate drops 8 points under E64 × 64's 95.8% (run 9), and every miss re-prefills.
- 01:50: `run_dspark.sh` took the box after E72 × 32 (user: "let's try that dspark model out").
- E043 (2026-10-24): E72 with no-tool keep statistics (`keep_contrib_dec_sp0_w10_e72`, Engram 48 GiB) keeps 57.5%
  pass@1 / 66.9% pass@2 in the no-tool harness (full 64.7% pass@1); that is the user's "≈ 90% of the full model".
  Its 334 calls (`evals/results/e043nt_e72_w10_eng48.jsonl`; low effort, minimal header, task summary): prompt mean
  9.4K, output mean 31.0K, p50 26.7K, p90 59K, max 118K → `metadata/trace_shapes/notools_e043_e72.json`. Steady-state
  slot context mean 32.1K (p90 58K); by the S4 fit in 160K units E72 holds C = 28 at p95 85% of its KV, C = 32 at
  97%. So the resumed T72 points are no-tool snapshots (`nt:e72:28`, `nt:e72:32`, `nt:e72:24`, R1-style decode
  windows of 120 s; `runs/t72.points`), then the E64 × 64 tool anchor. A snapshot window has no prompt turnover:
  new attempts' 9.4K prompts add ≈ 0.3 prefilled tokens per generated token, roughly −9% at ~330 tok/s.
- **E72 no-tool snapshot, C = 28: 270.7 tok/s decode**, 9.7 tok/s per stream, ITL 103 ms (contexts mean 34.9K, p50
  25K, p90 68K, max 124K; need 639K of 821K KV); prefill 1,276 tok/s. With prompt turnover (≈ 0.30 prefilled per
  generated token, before prefix hits between a task's two attempts) ≈ 254 tok/s, ≈ 350 attempts of 31K in 12 h.
- **E72 no-tool snapshot, C = 32: does not fit.** Need 759K by the S4 fit (92% of KV), but the first streams decode
  through the ≈ 15-min prefill and outgrow the KV: 28 running, 4–5 waiting, KV 0.99–1.00, 4 preemptions in 42 min,
  so the window never starts (stopped at 03:01). With these shapes **E72 tops out at ≈ 28 streams**. The DS1
  snapshot guard now uses 85% of KV.
- **E72 no-tool snapshot, C = 24: 266.8 tok/s**, 11.1 tok/s per stream, ITL 89.9 ms (contexts mean 33.4K, max
  124K; need 528K of 821K). **A tie with C = 28's 270.7**, at 15% faster attempts and more KV headroom, so C = 24 is
  the setting. At these contexts each extra stream adds ≈ 3.4 ms per step (89.9 → 103.4 ms for +4), far more than
  the 0.9 ms per batch token of Kaggle's uniform 32K grid: long-context streams (indexer over all keys) dominate.
- Health (`scripts/run_health.py` over gpu.csv / kvmon.log): every run so far at 71–73 °C max on all four GPUs,
  busy SM clock p50 1.47–1.55 GHz on each (p5 1.07–1.34 GHz in prefill phases), so no thermal gating as on the
  Vast box. KV peak / preemptions: E72 tools × 32 0.93 / 0, no-tool × 28 0.86 / 0, no-tool × 32 1.00 / 4.
- **Real E72 no-tool checkpoint** (E043, 2026-10-24): `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E72` on HF
  (private, 118.5 GB, 54 files): no-tool keep set ×10, experts renumbered ascending (`e043_manifest.json`), its own
  compact Engram 48 GiB (`model-engram-*`, 107.8M / 87.4M rows on layers 1 / 14 + row_map,
  `text_config.engram_compact_kept`), text only (no `mtp.*`, no vision). Needs E043's
  `patches/apply_compact_engram.py` and the router entry (72, 6); 16.64 GiB per GPU at TP4+EP4 on the RTX box.
  Export NLL 0.69195 vs runtime masks 0.6918 on 64 dev traces. The Runpod pod has 35 GB disk free but a 116 GiB
  /dev/shm (110.4 GiB needed), 251 GB RAM.
- 02:02: the user confirmed the no-tool harness is the main focus, so `run_seq.sh` runs the no-tool E72 snapshots
  first, then DS1, then the E64 × 64 tool anchor. DS1's first base server was stopped mid-run (kept as
  `runs/ds_e64_base.first`: 8K × 32 decode 339.2 tok/s, ITL 94.2 ms, prefill 1,382 tok/s; the Vast box: 257.1).

### DS1: DSpark drafter cost (`scripts/run_dspark.sh`, `runs/ds_*`, `runs/ds.log`; Runpod)

The 1017 config ships DeepSeek's own drafter: `num_nextn_predict_layers` 3, `dspark_n_routed_experts` 128 (top-3),
`dspark_block_size` 5 (one pass drafts a 5-token block from target hidden states at `dspark_target_layer_ids`; the
paper adds a confidence head for the verification length). The fork registers it as `DSparkV41DraftModel` (method
`dspark`, model runner V2, weights from the target checkpoint); there is no DFlash drafter for DeepSeek in the fork,
and RedHat's DFlash is for V4-Flash. Random weights accept nothing, so DS1 uses the fork's synthetic rejection
sampler (`rejection_sample_method: synthetic`, `synthetic_acceptance_length` L): the step cost does not depend on L,
so tok/s at another L' scales by L'/L. Real acceptance (full and E72) was requested from E043, which has real weights.
- P12 decode server (E64, 128 seqs, util 0.88, MBT 2048, 64K, no prefix caching, PR #154 flag) at 8K × 32,
  8K × 64, 32K × 16: no drafter, then n = 1 / 2 / 4 drafts at L = 1.8 / 2.4 / 3.0; then whether E72 + drafter
  leaves any KV. Points whose KV does not fit are skipped.
- E043 measured the drafter in the checkpoint: 2,401 `mtp.0–2` tensors, 7.38 GiB (routed experts 6.72: MXFP4 6.33
  + E8M0 scales 0.40; dense 0.66), ≈ 1.85 GiB per GPU at TP4, more than E72's 1.18 GiB of KV;
  `dspark_target_layer_ids` [37, 38, 39]. So the E72 step (user: "let's try it anyways") uses drafters cut to 32
  and 16 experts as well (`model-e72-d32`, `-d16`: `text_config.dspark_n_routed_experts`; router shapes (64, 3),
  (32, 3), (16, 3) added to `apply_patches.py`), est. 0.59 / 0.38 GiB per GPU. Each runs the no-tool snapshot at
  the largest C in 28…4 that fits, against `t72nt_e72_c*`.
- **DS1 E64 results: the full drafter does not fit even E64.** No drafter (rerun 03:16, P12 server): 8K × 32
  339.7 tok/s (ITL 94.1 ms), 8K × 64 538.0 (118.8 ms), 32K × 16 230.1 (69.5 ms); 15.06 GiB weights, 2.96 GiB KV.
  With the drafter, n = 1: weights 17.14 GiB per GPU (+2.08, above E043's 1.85 GiB estimate), KV 0.48 GiB
  (186,384 tokens), so every point was skipped. n = 2: 0.12 GiB of KV, the server refuses a 64K request; n = 4 the
  same. E72 + full drafter (n = 2): "No available memory for the cache blocks". Only the cut drafters (d32, d16)
  remain.
- **Real acceptance (E043, `experiments/E043_dsv41_real_pruning/results/dspark_probe.jsonl`):** one N = 5 serve
  per config, standard rejection sampling, T = 1, 15-min windows of real ARC traffic, drafter's 128 experts unmasked.
  Mean acceptance length at N = 1 / 2 / 3 / 4 / 5:

  | config | N=1 | N=2 | N=3 | N=4 | N=5 | drafts |
  |---|---:|---:|---:|---:|---:|---:|
  | full, tool harness | 1.66 | 2.11 | 2.42 | 2.64 | 2.81 | 224k |
  | full, no-tool | 1.69 | 2.19 | 2.55 | 2.83 | 3.05 | 267k |
  | **E72 no-tool** (x10 keep, Engram 48 GiB) | 1.59 | **1.99** | 2.27 | **2.49** | 2.66 | 314k |
  | E72 tools (w30, Engram 64, rescue) | 1.39 | 1.62 | 1.77 | 1.88 | 1.97 | 292k |

  DS1's synthetic L therefore uses E043's values: E64 points at full no-tool 1.69 / 2.19 / 2.83 (n = 1 / 2 / 4),
  E72 points at E72 no-tool 1.99 (n = 2) and 2.49 (n = 4). A cut drafter accepts less by an unmeasured amount, so
  the d32 / d16 points are upper bounds. E043 can test a pruned drafter (needs drafter-side masks), held until DS1
  shows it could pay off.
- Expectation from Kaggle's decode data (step ≈ 58 ms + 0.9 ms per batch token at 32K): about +20–25% at 32
  streams, +5–10% at 64, nothing at 128, if acceptance matches the full model. Drafter memory (estimated ~2 GiB per
  GPU at MXFP4) is the risk on L4.

### LB: more experts at fewer bits (`patches/humming_experts.py`, `scripts/run_lowbit.sh`, `runs/lb_*`; Runpod)

User (2026-10-24): "is there any way to bring [Maya's very low quantization] to vllm and make it run properly on
batched inference? … do a virtual random pruning down to 128 experts and see the throughput?" The original model has
**384** routed experts (top-6); E72 keeps 19%. Maya's formats (IQ1_S, IQ2_XXS, EXL3) have no batched MoE kernels in
vLLM, but the fork ships Humming (humming-kernels 0.1.12), whose vLLM MoE path (`fused_humming_moe`, `HummingMoEMethod`)
takes integer weights at any bit width with group scales (GPTQ schema). `humming_experts.py` (env-gated
`DSV41_EXPERT_HUMMING=<schema JSON>`) gives V4.1's routed experts that method instead of `DeepseekV41Mxfp4MoEMethod`,
and adds router shapes (96, 6), (128, 6), (160, 6). Random weights, so speed and memory only; quality at 2–3 bits
would need a real quantization (GPTQ/AutoRound-style calibration) and an eval.
- Expected per-GPU weights (0.18 GiB per expert at 4.25 bits; dense ≈ 3.5 GiB): E128 at 2 bits ≈ 15.1 GiB (≈ E64),
  E96 at 3 bits ≈ 16.8 (≈ E72), E160 at 2 bits ≈ 18.0 (the ceiling). All 384 at Maya-XS's ≈ 1.6 bits ≈ 29 GiB:
  does not fit 4×L4.
- Runs (no-tool server, as T72's nt points): E128 uint2, then controls E64 Humming uint4 and E64 Marlin MXFP4 at the
  same config, then E96 uint3 and E160 uint2. Each: decode 8K × 32 and × 64, then the E043 no-tool snapshot at the
  largest C that fits 85% of KV. Queued after DS1 by `run_seq2.sh`, before the E64 × 64 tool anchor.
- DS1's E72 + d32 drafter (n = 2, 128 seqs) also failed: "No available memory for the cache blocks". The d16 points
  (`run_ds16.sh`) now use 32 seqs: spec decode's CUDA graphs at 128 × (n + 1) tokens cost KV, and E72 holds ≤ 28
  no-tool streams anyway.
- First LB start (03:44): E128 uint2 OOMed in `HummingMoEMethod.create_weights` because the dying d32 server still
  held 15 GiB per GPU (the GPU-free check counted compute apps; it now waits for memory < 1 GiB). E64 Humming uint4
  came up: **15.97 GiB per GPU (Marlin MXFP4: 15.06), KV 1.96 GiB = 1,369,160 tokens** at 160K, "Using
  HummingIndexedExperts Humming MoE backend" ("indexed gemm"). Restarted at 03:47 with E128 first (`run_seq4.sh`).
- **E128 uint2 (Humming): 16.34 GiB per GPU, KV 1.41 GiB = 987,780 tokens (E72 MXFP4: 821,422, +20%). Decode
  8K × 32: 337.1 tok/s, ITL 94.9 ms, prefill 1,394 tok/s**, the same as E64 on Marlin (339.7 / 94.1 ms / 1,401) on
  the P12 server. Twice the experts at half the bits cost no speed. 8K × 64 skipped (needs 1.01M).
- **E128 uint2 no-tool snapshot, C = 32: 341.3 tok/s**, 10.7 per stream, ITL 93.7 ms (need 759K of 988K; KV peak
  0.84, 0 preemptions, 71–72 °C). E72 MXFP4 on Marlin: 270.7 at C = 28 (ITL 103.4 ms), and C = 32 did not fit.
  So +26% at the same memory, more than the 4 extra streams explain (ITL is lower at more streams). Unexplained
  yet: at 8K × 32 decode Humming E128 and Marlin E64 tie, so the gap appears with long contexts. `sh_e72_util93`
  (E72 Marlin at C = 32, same snapshot) and the E64 Humming / Marlin controls should separate kernel from load.
- **E104 mixed 2/3 bits (layers 0–19 at 3): 16.46 GiB per GPU, KV 1.34 GiB = 937,561 tokens**: the `dynamic`
  overrides are honored (all-2-bit E104 would be ≈ 13.9 GiB).
  **With the 30% Engram table (56.6 GiB pinned: 7.08 GiB per rank and layer): decode 8K × 32 327.5 tok/s (ITL
  97.6 ms); no-tool snapshot C = 32: 330.9 tok/s, 10.3 per stream, ITL 96.7 ms**, KV peak 0.89, 0 preemptions,
  71–73 °C; host 107 GiB used / 143 available. Startup 190 s (pinning; 80 s with the 7.9 GiB table). Within 3% of
  E128 uint2 (341.3 at C = 32, small table): the user's target shape runs at ≈ 330 tok/s on this pod.
- **Low-bit experts + full DSpark** (user: "maybe we can afford dspark with this lower bit experts?"): the
  more-experts shapes use E72's memory, so no room; spending the bits on the drafter instead does fit. Est. per
  GPU: E72 at 3 bits ≈ 14.0 GiB + drafter at 3 bits ≈ 1.4 (its 128 experts take the same Humming schema) + spec
  buffers ≈ 0.4 (32 seqs) → KV ≈ 1.9 GiB (E72 MXFP4 today: 1.18); E96 at 2 bits ≈ 13.1 + 1.05 + 0.4 → KV ≈ 2.8.
  Queued in `run_ds16.sh` after `sh_e72_util93` (`run_cut.sh`): E72 uint3 without / with n = 2 (L 1.99) / n = 4
  (L 2.49), E96 uint2 without / with n = 2, then the d16 points, then `run_shape_rest.sh` (E96 uint3, E160 uint2,
  E64 controls). L is E043's MXFP4 acceptance, an upper bound for low-bit target and drafter.
- **E043's real-weight candidates (04:45):** (1) E128, w13 uint2 g128 + w2 uint2 g64, asymmetric, bf16 scales,
  Engram 30%, no DSpark (≈ 12.8 GiB routed per GPU by the padded-shard arithmetic); (2) E96, w13 uint2 g128 + w2
  uint3 g64 + unpruned DSpark (≈ 11.1 + ≈ 1.1 drafter + ≈ 0.17 dense; KV ≈ 1.0–1.2M). E043 decodes the MXFP4
  experts to exact BF16 and runs per-expert GPTQ into native Humming params; their
  `experiments/E043_dsv41_real_pruning/patches/apply_e043_humming.py` (+ `e043_humming.py`) carries the same
  `DSV41_EXPERT_HUMMING` hook plus `DSV41_EXPERT_HUMMING_W2` (down projection override) and humming 0.1.12 API fixes
  (no `WeightScaleType.GROUP_TENSOR`; `quantize_weight(weight_scale_2_type=)`). On the pod, `humming_experts.py`'s
  hook was removed from quant_config.py (backup `bundle/quant_config.e042lb.bak`) and E043's patch applied; the
  router shapes stay. `run_ds16.sh` now runs both candidates with dummy weights (`ds_x128`, `ds_x96`, `ds_x96_n2`,
  `ds_x96_n4`; 48 seqs, Engram 30%) instead of the E72 uint3 / E96 gptq DSpark and d16 points.
- **Target shape (user, 04:00): "30% of the original engrams (keep it at original precision), and having more
  experts at 2–3 bits".** Engram geometry: `engram_num_embeddings` [384,006,168, 384,016,682] rows on layers 1 / 14,
  264 B FP8 rows; 30% = 230.4M rows = 56.6 GiB pinned (+ ≈ 2.9 GiB row map), host RAM only. GPU budget at util 0.88
  with healthy KV ≈ 17 GiB of weights − 3.5 dense − 0.9 Humming overhead ≈ 12.6 GiB for experts: ≈ E128 at 2 bits,
  ≈ E88 at 3 bits, ≈ E104 at mixed 2/3 by layer (Humming `dynamic` overrides; one type per MoE layer).
  `run_shape.sh` (via `run_seq5.sh`, after E128's snapshot): E104 with 3 bits on layers 0–19 and 2 bits on 20–39
  plus 28.8M synthetic Engram rows per rank and layer (56.6 GiB), then E128 uint2 with the same table, then LB's
  E96 uint3, E160 uint2 and the E64 Humming / Marlin decode controls.
- **Alignment (found for E043's low-bit plan):** under TP4 the expert intermediate shard is 2304 / 4 = 576. Humming
  pads N to a multiple of 256 and K to 128 (`humming.py` `pad_n_to_multiple=256`, `pad_k_to_multiple=128`), so
  w13 N 1152 → 1280 and w2 K 576 → 640: ≈ +11% expert bytes, most of Humming's +0.9 GiB at E64 uint4. With real
  weights, g128 groups over K = 2304 straddle the 576 shard boundaries: use g64 (2304 = 36 × 64, 576 = 9 × 64),
  quantize per TP shard, or serve the experts EP4 (whole experts per GPU, no padding). Added to `run_shape.sh`:
  E128 uint2 with `--enable-expert-parallel`, and E72 Marlin at util 0.93 (nvidia-smi shows ≈ 2.2 GiB per GPU free
  at 0.88: 20.3 of 22.5 GiB used, ≈ 0.9 GiB of it outside torch).
- E72 memory ledger per GPU (160K config): weights 16.56 GiB = routed 13.2 (0.183 GiB per expert at MXFP4: 35.4M
  params per expert-layer × 40 layers / 4) + ≈ 3.35 other; KV 1.18 GiB; activation peak + CUDA-graph estimate
  ≈ 1.66 GiB inside util 0.88 × 22.04 GiB = 19.40 GiB. Routed budget at today's KV ≈ 13.2 GiB per GPU in kernel
  format ≈ 306 "expert × bits" units (E72 × 4.25, E96 × ~3.2, E104 × ~3.0, E128 × ~2.4 effective).
- **E128 uint2 + 30% Engram (`sh_e128_g2_eng30`): 16.34 GiB, KV 967,337; decode 8K × 32 336.5 tok/s; no-tool C = 32
  340.5 tok/s, ITL 93.9 ms**, KV peak 0.86, 0 preemptions over 22 min, 71–72 °C; host 109 GiB used / 141 available.
  Same as with the 7.9 GiB table (341.3): the Engram table's size does not change speed.
- **E128 uint2 with `--enable-expert-parallel` (`sh_e128_g2_ep4`): failed at startup at util 0.88, "No available
  memory for the cache blocks"**. EP4 costs more GPU memory than TP's padded shards here. Not investigated.
- **E72 Marlin at util 0.93 (`sh_e72_util93`): 16.56 GiB, KV 2.28 GiB = 1,590,625 tokens (×1.94 vs 0.88); no OOM
  over 31 min.** Decode 8K × 32 305.6 tok/s (ITL 104.6 ms), 8K × 64 471.5 (135.6 ms); **no-tool C = 36: 311.1 tok/s,
  ITL 115.6 ms**, KV peak 0.58 (C = 36 was the largest in its list; ≈ 50 streams would fit), 0 preemptions, 71–73 °C.
  This explains LB's +26%: at the same 8K × 32 decode, 2-bit E128 beats 4-bit E72 by 10% (336.5 vs 305.6). Both banks
  hold ≈ 13 GiB, but a step of 32 × 6 picks touches ≈ 78% of 128 experts and ≈ 93% of 72, so E128 reads fewer bytes.
  The rest is 32 streams vs 28.
- **E043 candidate 1 with dummy weights (`ds_x128`: w13 uint2 g128 + w2 uint2 g64, asymmetric, bf16 scales,
  Engram 30%, 48 seqs, util 0.88): the SM89 Humming kernels accept the schema. 16.87 GiB per GPU, KV 1.11 GiB =
  772,536 tokens** (the g64 w2 scales and the zero points cost ≈ 0.5 GiB vs symmetric g128). **No-tool C = 28:
  298.7 tok/s, ITL 93.7 ms**, KV peak 0.92, 0 preemptions over 15 min, 71–73 °C. C = 32 failed the guard.
  KV peaks run ≈ 1.07–1.11× the snapshot estimate (0.83 → 0.92 here, 0.785 → 0.86, 0.54 → 0.58).
- **Replan (06:00):** every no-tool point so far was KV-bound, and util 0.93 is safe on E72. So `run_shape_rest.sh`
  now runs `ds_x128_u93`, `ds_x96_n4_u93` (E043 candidate 2 + DSpark n = 4, L 2.49, 64 seqs) and `lb_e96_g3_u93`, all
  with Engram 30%, at util 0.93, with snapshots up to C = 60 and an 80% guard. The E64 Humming / Marlin decode
  controls follow at 0.88. Dropped: E160 uint2 (≈ 19.5 GiB of weights does not fit) and the E64 × 64 tool anchor
  (no-tool is the focus). `run_seq6.sh` replaced `run_seq5.sh`'s tail.
- **E043 real-weight quality (06:15, dev split, teacher-forced NLL on 64 no-tool traces vs the full model; GPTQ per
  expert on the pruned model's own routing, Engram 30%):**

  | shape | dev NLL vs full |
  |---|---:|
  | E128 native MXFP4 | +8.3% |
  | E128 w13 u2 g128 + w2 u2 g64, GPTQ | **+12.1%** |
  | E128 same schema, RTN | +22.0% |
  | E96 native MXFP4 | +12.7% |
  | E96 w13 u2 g128 + w2 u3 g64, GPTQ | +15.2% |
  | E72 ×10 native MXFP4 (ARC pass@1 57.5% / pass@2 66.9%) | +22.2% |

  Both GPTQ shapes write the answer correctly in 97.4% of a dev probe (E72: 99.5%). ARC 120-task evals are running
  on E043's box.
- **Export size problem (sent to E043):** E043 planned an MXFP4 base plus a GPTQ dir. Its loader reaches the
  precomputed tensors only through the MXFP4 → BF16 path. That layout would be ≈ 96 GB of routed MXFP4 + ≈ 13 dense
  + ≈ 61 Engram 30% + ≈ 50 GPTQ (2.19 bits average) ≈ 220 GB, which does not fit the pod's 50 GB disk / 116 GiB
  /dev/shm. A GPTQ-only export with placeholder expert tensors would be ≈ 124 GB.
- **E043 candidate 2 with dummy weights (`ds_*`: E96, w13 uint2 g128 + w2 uint3 g64, asymmetric, Engram 30%, util
  0.88, 48 seqs; DSpark's 128 drafter experts get the same schema; synthetic acceptance at E043's E72 no-tool L):**

  | run | weights / GPU | KV tokens | C | tok/s | per stream | ITL ms | KV peak |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `ds_x96`, no drafter | 15.18 GiB | 2,033,852 | 40 (list max) | **380.3** | 9.5 | 105.1 | 0.50 |
  | `ds_x96_n2`, L 1.99 | 16.68 | 618,609 | 20 | 300.6 | 15.0 | 66.2 | 0.79 |
  | `ds_x96_n4`, L 2.49 | 16.68 | 464,760 | 16 | 272.2 | 17.0 | 71.9 | 0.87 |

  The drafter costs 1.50 GiB of weights per GPU, plus 0.47 / 0.70 GiB of spec-decode buffers and graphs. KV memory
  drops from 2.91 to 0.94 / 0.71 GiB, a 3.3–4.4× cut, which halves the streams that fit. Per-stream speed rises
  1.6–1.8×, but total throughput falls 21–28%. **For batched no-tool rollouts on 4×L4, DSpark loses.** Real
  acceptance on a 2/3-bit target and drafter is likely lower than these MXFP4 values, which would widen the loss.
  It could still pay when the batch is latency-bound (few long rollouts left), but turning it on means a server
  restart. All 0 preemptions, 71–73 °C. E96 without the drafter had KV for ≈ 60+ streams, so 380 is a floor.
- **Replan 2 (06:45):** `run_cut2.sh` stops `run_shape_rest.sh` after `ds_x96_n4_u93`. `run_seq7.sh` then runs
  `run_rest2.sh`: `ds_x96_u93` (candidate 2, no drafter, util 0.93, snapshots up to C = 80) and the E64 controls. The
  gptq uint3 E96 point is dropped, since E043 did not pick it.
- **E043 candidate 1 at util 0.93 (`ds_x128_u93`, 128 seqs): 16.87 GiB, KV 2.10 GiB = 1,469,523 tokens (×1.90 vs
  0.88). Decode 8K × 32 334.9 tok/s, 8K × 64 487.5 (ITL 131.3 ms); no-tool snapshot C = 48: 418.8 tok/s**, 8.7 per
  stream, ITL 114.6 ms, KV peak 0.81, 0 preemptions over 36 min, 71–73 °C. Against E72 MXFP4 at util 0.88 (270.7 at
  C = 28) that is +55%, with better dev NLL (+12.1% vs +22.2% over the full model). C = 52 failed the 80% guard.
- **E043 partial ARC (07:30, no-tool, 225/334 attempts; the early-finishing pairs skew easy):**
  - pass@1 on 111 pairs: E128 2-bit GPTQ 76, full model 77, E72 ×10 70.
  - pass@2 on 106 pairs: 82 / 81 / 77.
  - The GPTQ-only export is built on E043's box: `notool_e128_x10_eng56_gptq`, 123.1 GB. It carries no MXFP4 experts;
    `gptq/` holds compact-id q/s/z; env: `DSV41_EXPERT_HUMMING` uint2 g128 asymmetric, `DSV41_EXPERT_HUMMING_W2`
    `{"group_size": 64}`, `E043_GPTQ_DIR=<model>/gptq`.
- **Pod prep for real weights (dry run on a scratch copy, `/root/dsv41/vllm_dry`):**
  - E043's updated `apply_e043_humming.py` applies; only H4 (zero-point loading) is new on the pod.
  - `apply_compact_engram.py` conflicts with our synthetic-fold edit of the same lookup line. `unpatch_engram.py`
    (in `/root/dsv41/e043p_new/`) reverts our 8 `engram.py` edits from `apply_patches.py`'s own pair list. After that,
    E043C applies and the file parses.
  - A real run swaps that file in and keeps the synthetic one.
  - Export size: 123,143,357,239 B = 114.69 GiB. It splits into compact Engram ≈ 63.8 GB with row maps, `gptq/`
    ≈ 50 GB, and dense / router / shared ≈ 9 GB.
  - On the pod, put the ≈ 9 GB dense part on the root disk (35 GB free) and the rest in /dev/shm (117 GiB). That
    leaves ≈ 11 GiB of shm spare instead of ≈ 2.
- **E96 at util 0.93 (E043 candidate 2 schema, E72 output shapes):**
  - `ds_x96_n4_u93` (DSpark n = 4, L 2.49, 64 seqs): 16.68 GiB, KV 1,106,426; decode 8K × 32 334.0; no-tool C = 36:
    337.2 tok/s, ITL 131.9 ms, KV peak 0.82.
  - **`ds_x96_u93` (no drafter, 128 seqs): 15.18 GiB, KV 3.9 GiB = 2,721,210 tokens; decode 8K × 32 340.8, 8K × 64
    506.7; no-tool C = 80 (list max): 558.0 tok/s**, 7.0 per stream, ITL 143.3 ms, KV peak 0.67.
  - DSpark costs 40% of throughput at 0.93 as well. Both runs: 0 preemptions, 71–73 °C.
  - Throughput keeps rising with C (40 → 380, 80 → 558), so on 4×L4 the KV budget sets the rate.
- **E043 final ARC (no-tool, 120 tasks, 2 attempts, task-weighted):**

  | model | pass@1 | pass@2 | mean output |
  |---|---:|---:|---:|
  | E128 2-bit GPTQ | 59.6% | 72.0% | 41.6K |
  | full model | 64.7% | 76.8% | 31.4K |
  | E72 ×10 | 57.5% | 66.9% | 31.0K |

  E128 hedges more ("maybe" 1.5×) but doesn't loop. 3 of its 334 attempts hit a client timeout and will be rerun.
  - HF: `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E128-2bit` (private), 123,143,896,963 B.
  - Export parity: dev NLL 0.63486, vs 0.63499 with runtime masks.
  - E043's FP4 Engram (MXFP4 rows) is free on dev NLL (0.63507) and would halve the pinned table. Host RAM does not
    bind here (pod 251 GiB, Kaggle 173 GiB), so FP8 stays.
  - E192 at 2 bits cannot fit (≈ 23.6 GiB per GPU). The 2-bit ceiling with healthy KV at 0.93 is ≈ E132.
- **Attempts per hour (`run_rest3.sh`, 08:40):** E128's longer answers raise the steady-state in-flight context from
  32.3K to 37.4K (length-biased mean, `metadata/trace_shapes/notools_e043_e128.json`). Two runs follow the E64
  controls, each at util 0.93 with its own shapes and its largest C under the 80% guard, without decode benches:
  - `ds_x128_u93_l`: E128 with E128 shapes.
  - `sh_e72_u93_l`: E72 Marlin up to C = 64.
- **Attempts per hour, results (util 0.93, each with its own ARC answer lengths, largest C under the 80% guard):**

  | model | C | no-tool tok/s | ITL ms | KV peak | mean output | attempts/h* | 334 attempts* | pass@2 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | E72 MXFP4 Marlin (`sh_e72_u93_l`) | 52 | 399.5 | 130.1 | 0.80 | 31.0K | ≈ 42 | ≈ 8.0 h | 66.9% |
  | E128 2-bit (`ds_x128_u93_l`) | 48 | 419.2 | 114.5 | 0.84 | 41.6K | ≈ 34 | ≈ 9.9 h | 72.0% |
  | E96 2/3-bit (`ds_x96_u93`, E72 shapes) | 80 | 558.0 | 143.3 | 0.67 | 26.8K (partial) | ≈ 57–65 | ≈ 5–6 h | due ~11:00 |

  \*Per attempt: output / decode tok/s + 9.4K prompt / ≈ 1,270 tok/s prefill, done serially. This is crude: it
  ignores mixed batches and the low-concurrency tail at the end of a run.
  - Both runs: 0 preemptions over 26 min, 71–73 °C.
  - E72 at util 0.93 gains +48% over util 0.88 (270.7 at C = 28). Kaggle's 4×L4 baseline should move to 0.93, after
    checking that Kaggle leaves the same GPU memory free.
  - E128's longer answers cost about 20% of attempts/h against E72, for +5.1 pp pass@2.
- **E043 final no-tool ARC for E96 u2/u3 GPTQ (11:15; 120 tasks, 2 attempts, 0 errors):**

  | model | pass@1 tw | pass@2 tw | pass@2 pairs | output mean / p90 |
  |---|---:|---:|---:|---:|
  | E96 u2/u3 | 57.8% | **75.3%** | 125/167 | 37.4K / 71.4K |
  | E128 u2 | 59.9% | 72.0% | 118/164 | 41.6K / 78.4K |
  | E72 MXFP4 | 57.4% | 66.9% | 110/167 | 31.0K / 59.2K |
  | full | 67.9% | 76.8% | 125/167 | 31.4K / 58.4K |

  - E96 ties the full model on pass@2 pairs, and it is the fastest shape on L4. At its 37.4K outputs (steady-state
    context 36.1K vs E72's 32.3K) it should make ≈ 47–50 attempts/h, ≈ 1.15× E72 and ≈ 1.4× E128.
  - `ds_x96_u93_l` (`run_rest4.sh`, via `run_seq9.sh` after T2) measures it at its own shapes
    (`metadata/trace_shapes/notools_e043_e96.json`, 334 calls), up to C = 96.
  - **Result (13:16):** C = 88 (the largest under the 80% guard): **585.9 tok/s** decode, prefill 1284 tok/s,
    snapshot context mean 37.4K (p50 / p90 / max 32.9K / 76.2K / 104.5K). KV peak 0.83, p50 0.48, 0 preemptions over
    45 min, 72–73 °C. Attempts/h ≈ 3600 / (37.4K / 585.9 + 9.4K / 1284) ≈ **50.6/h, ≈ 6.6 h for 334 attempts**:
    1.19× E72 (42.4/h) and 1.50× E128 (33.8/h), at pass@2 75.3% vs 66.9% and 72.0%.
    The peak ran past the guard (0.83 vs 0.80 planned), so a real run should use C ≈ 80.
- **Direction change (10:10, the user via E043): tool-integrated reasoning with a pruned + quantized model, targets
  E128 and E96.** E043 dropped the no-tool full-384 2-bit ARC run and is building tool E128 (eval ~15:00) and tool E96
  (~16:30), using tool keep sets, the 64 GiB tool Engram table and tool GPTQ calibration.
- **T2 (`scripts/run_tool2.sh`, `runs/t2_*`, `runs/t2.log`; started 10:15):** a tool-mode proxy until E043's tool
  turn shapes arrive.
  - Traffic: T72's multi-turn replay (`tools_e030_hdrmin.json`), with a 30-min window after 5 min of warm-up.
  - Server: dummy E043 schemas, 64 GiB tool Engram (32.6M synthetic rows per rank and layer), util 0.93, 128 seqs,
    prefix caching, retention 128.
  - Points: E128 × 48 (KV 1.47M), then E96 × 80 (KV 2.72M), sized by KV per healthy slot (≈ 23K at E64 × 64's
    0.73 peak). Reference: E72 × 32 at util 0.88 gave 205.3 tok/s at 87.8% hits.
  - **Results (12:30; 71–73 °C, clocks p50 1.50–1.58 GHz, 0 preemptions in both):**

    | point | KV | gen tok/s | prefix hits | uncached prefill tok/s | KV peak / p50 | window |
    |---|---:|---:|---:|---:|---:|---:|
    | E72 × 32, util 0.88 (T72) | 821K | 205.3 | 87.8% | — | — | 30 min |
    | E128 × 48 | 1.47M | **340.5** | 96.1% | 221.7 | 0.77 / 0.68 | 30 min |
    | E96 × 80 | 2.72M | **453.5** | 96.5% | 244.6 | 0.64 / 0.60 | 17 min |

    - E128 × 48 (replay JSON `runs/t2_x128_c48/r2_tools_c48.json`): 349 turns and 27 attempts finished in the window,
      TTFT median 0.84 s, turn median 117.2 s, turn prompt p50 / p90 24.9K / 56.3K. Host 111 GiB used / 140 GiB
      available.
    - E96 × 80 wrote no replay JSON: the shell `timeout 4200` killed it 17 min into the window, because setup (80 cold
      mid-attempt histories prefilled at 0% hits, ≈ 600 tok/s next to decode) took 47.5 min. Its numbers come from
      `kvmon.log` deltas over the post-warm-up window. Checked on E128 × 48 with the same method: 342.0 tok/s and
      96.1%, against the replay's 340.5 and 96.1%. In 5-min chunks: 470.7 / 401.7 / 454.5 tok/s at 95.6–96.6% hits.
      Host 102 GiB used / 148 GiB available. The 47.5 min of setup is an artifact of the replay: a real run starts every
      attempt at turn 0.
    - E96 × 80 is 1.33× E128 × 48 and 2.2× E72 × 32, with KV to spare: 20.4K tokens per slot at p50, so C = 96 sits
      near E128 × 48's 0.77 peak.
    - Caveat: these are the full model's turn shapes. E043's pruned models wrote 1.2–1.3× longer no-tool answers, and
      tool turns may also grow.
  - **T3 (`scripts/run_tool3.sh` via `run_seq10.sh` after `run_rest4.sh`; `runs/t3.log`):** E96 × 96 with the same
    server and traffic, capped with the replay's own `--max-wall-s 5400` instead of a shell timeout. Question: does the
    hit rate hold, and how much throughput does C = 96 add?
  - **T3 result (14:45): C = 96 is past the knee.** 291.3 tok/s at 85.8% hits (C = 80: 453.5 at 96.5%), uncached
    prefill 736.5 tok/s (3× C = 80's), TTFT median 1.54 s, turn median 250 s. KV peak 0.80 / p50 0.60, 0 preemptions
    over 86 min, 72–73 °C. The cache thrashes before anything is preempted: once active sequences fill ~0.8 of KV, the
    blocks a slot leaves cached during its 1-s tool call get evicted, and its next turn re-prefills. Setup 2370 s,
    drain 667 s, full 30-min window.
  - For sizing: E96 holds at C = 80 and drops 36% at C = 96 on these shapes. Watch the hit rate, not preemptions: KV
    peak ≈ 0.8 marks the edge, and E128 × 48 (peak 0.77) sat just under it.
- **E043 interim tool ARC (12:50; 8× RTX, sub80s40 + header, temperature 1.0, vs the API full model E030 hdrmin on the
  same attempts):**

  | model | attempts done | correct (model vs full) | mean output (model vs full) | turns/attempt | Python errors |
  |---|---:|---:|---:|---:|---:|
  | E128 2-bit | 281/334 | 229 vs 223 | 28.8K vs 22.8K | 16.0 | 13.7% |
  | E96 2/3-bit | 127/334 | 100 vs 110 | 16.1K | 18.3 (15 hit max_turns) | 35.8% |

  - The full model has 2.3% Python errors. The pruned models' errors are almost all character-level typos (`coms` for
    `comps`, `scpy`, `zzip`, `dequeue`). Native E72 MXFP4 has them too (44.6%), so they come from expert pruning, not
    the 2-bit quantization. E043 is testing min_p 0.05 and top_p 0.95 against plain temperature 1.0.
  - Crude attempts/h on L4 from T2's full-shape throughput: E128 ≈ 340.5 × 3600 / 28.8K ≈ 43/h, ≈ 7.8 h for 334
    attempts. E96's partial subset is not comparable yet. Its extra turns re-prefill more tool output per generated
    token, so its rate needs its own replay.
  - Per-turn shapes from both tool evals will follow (E128 ~14:00, E96 ~16:00, self-hosted full model ~18:00).
- **E043 final tool E128 (13:15; sub80s40 + header, 2 attempts, 120 tasks, no rescues):** pass@1 72.8%, **pass@2
  82.5%**, mean output 35.2K, against the API full model's 74.5% / 84.9% / 27.4K. Tool E96 at 250/334 is weaker: 176 vs
  the full model's 200 correct on the same attempts, 21.5 turns per attempt.
  - GPTQ-only export on E043's box: `tool_e128_w10_eng64_gptq`, 131,090,957,662 B (128 experts u2 g128 w13 / u2 g64
    w2, 64 GiB tool Engram). Env as the no-tool E128. HF upload after a parity check (~16:00).
  - Turn shapes (`metadata/trace_shapes/tools_e043_e128t.json`, converted from E043's
    `results/tool_shapes_e128t_w10_u2_gptq_eng64_sub80.json`): 17.1 turns per attempt (E030 13.9), turn prompt p50 / p90
    27.7K / 67.9K (24.3K / 57.9K), tool tokens per turn p50 / p90 310 / 1268 (300 / 1157), 35.2K generated per attempt
    (27.4K). Stops: 296 submitted, 27 output budget, 11 max turns.
  - **T4 (`scripts/run_tool4.sh` via `run_seq11.sh` after T3; `runs/t4.log`):** E128 × 40 and × 48 at these shapes,
    `--max-wall-s 4800` each. KV per slot grows ~15% over T2, so × 48 should peak near 0.88 and × 40 near 0.75.
    Attempts/h = gen tok/s × 3600 / 35.2K.
  - **T4 result (15:48), E128 × 40 at its own tool shapes: 318.6 gen tok/s at 96.7% hits**, uncached prefill 175.6
    tok/s, KV peak 0.76 / p50 0.68, 0 preemptions over 60 min, 72–73 °C. 298 turns and 16 attempts finished in the
    30-min window, TTFT median 0.76 s, turn median 114.9 s, turn prompt p50 / p90 24.5K / 72.7K. Host 192 GiB used,
    including the downloaded checkpoint in /dev/shm.
  - **≈ 32.6 attempts/h (318.6 × 3600 / 35.2K), ≈ 10.3 h for 334 attempts.** Tool E128 sits at no-tool E128's rate
    (33.8/h): fewer tok/s, but shorter attempts (35.2K vs 41.6K). The KV peak sits just under the ~0.8 edge from T3, so
    × 40 is about the limit at these shapes. The × 48 point was dropped (the user).
- **E043 final tool E96 u2/u3 (sub80s40 + header, 2 attempts, 40-turn cap):** pass@1 64.9%, **pass@2 75.8%**, against
  E128's 72.8% / 82.5% and full's 74.5% / 84.9%. It is turn-limited: 58 of 334 attempts hit the 40-turn cap at a mean
  40.4K of their 80K budget, because Python typos cost turns. E043 reruns it with `--max-turns 64`.
  - Shapes (`metadata/trace_shapes/tools_e043_e96t_t40.json`, from `tool_shapes_e96t_w10_u2u3_gptq_eng64_sub80.json`;
    8 negative tool-token values clamped): 23.6 turns per attempt (p90 40), turn prompt p50 / p90 25.1K / 54.2K,
    completion per turn mean / p50 / p90 1,290 / 455 / 3,218, 30.5K generated per attempt. Tool tokens per generated
    token are the same as E128's (0.256 vs 0.255). E96 has shorter turns, so prefill and per-turn overhead weigh more.
  - Not replayed yet: E128 is ahead by 6.7 pp pass@2, so E96 gets a replay only if the 64-turn rerun closes the gap.
  - Engram on the 2-bit E128 (E043 dev NLL): 0.79679 at 64 GiB, 0.79647 at 112 GiB, 0.79650 full. The 64 GiB tool
    table stays.
- **Tool E128 on HF (14:00):** `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit` (private), 131,090,957,662 B =
  122.1 GiB, 122 files.
  - Layout: backbone without routed experts 9.74 GB, 64 GiB tool Engram in `model-engram-0000{1,2,3}`, and
    `gptq/layer_LL.safetensors` (128 experts, compact ids). `gptq/layer_LL.json` holds error stats the loader ignores.
  - Env: as the no-tool E128, plus the router shape (128, 6) from `apply_e043_humming.py`.
  - Parity on 64 tool dev traces: NLL 0.79640 (export) vs 0.79652 (runtime masks), tool-block NLL 0.1490 vs 0.1494.
  - **Fitting it on the pod:** 122.1 GiB is more than `/dev/shm` (117 GiB) or the free disk (34 GB) alone. One
    Engram file (~21 GiB) plus ~10 GiB of `gptq/` can go on disk and the rest (~91 GiB) in `/dev/shm`, symlinked into
    one model dir. Host RAM while serving ≈ 91 GiB tmpfs + 64 GiB pinned Engram + ~40 GiB processes ≈ 195 of 251 GiB.
- **RW plan (the user, 15:00: "Yes, drop the 48 replay"):** T4 runs C = 40 only (`run_cut3.sh` stops `run_tool4.sh`
  after it), then a real-weight check of the tool E128 checkpoint, done ~19:00.
  - `dl_e128t.sh` (started 15:08, during T4's setup; ~200 MB/s). Disk part in `/root/dsv41/model-e128t`, /dev/shm
    part in `/dev/shm/e128t`. The pod's cgroup limit is 232.8 GiB (`memory.max` 249,999,998,976 B).
    `hf download` reads extra patterns after one `--include` as file names, so the script passes one flag per pattern.
  - `run_rw.sh`: waits for `T4_CUT_DONE` and `DL_DONE`, then switches the venv's vLLM from E042's synthetic Engram
    fold to E043's compact Engram loader. The files are backed up to `/root/dsv41/vllm_backup_e042synth` first; the
    switch is `unpatch_engram.py` (dry-run checked again), then `apply_compact_engram.py` and `apply_e043_humming.py`.
    It then serves with `serve_real.sh` (`serve.sh` without the synthetic init and dummy load) at the T2 config and
    checks one chat answer.
  - Download: 15:08–15:17 (≈ 230 MB/s), 40 GPTQ layers and 3 Engram files. Disk free afterwards 4.1 GiB, /dev/shm
    free 24 GiB. Pod memory (cgroup v1) during T4: 174 of 232 GiB used, 159.7 GiB of it shmem (94 GiB checkpoint +
    the synthetic 64 GiB Engram table). The Engram loader `narrow()`s and `copy_()`s into the pinned shard, so a real
    load peaks near the synthetic one, plus ~15 GiB of row-map temporaries.
  - E043 caveat: its ARC run used runtime masks on the full checkpoint. The export layout was checked only by NLL
    parity and a 25-prompt first-turn probe, so this is the first ARC run on the export itself. Its noise:
    E128 vs full differed by 28 vs 35 one-sided attempt wins out of 334.
  - E043's third patch, `apply_e043_patches.py` (its runtime-mask and REAP observation hooks), is left out on purpose.
    Without `$E043_CTRL` it only adds a zero bias and an all-ones Engram bitmap ("i.e. the original model"), the
    other two patches don't use it, and a Kaggle deployment wouldn't carry per-step observation hooks. Outputs
    should still pair with E043's run.
  - `eval_rw.sh`: E043's eval with the same harness flags (config checked against E043's jsonl, plus `--rescue-code`,
    which that jsonl's `rescues` field shows it used), 2 attempts, 40 workers. It runs from the workstation through an SSH tunnel, using E042's own sandbox (`arc-sandbox-e042`,
    http://172.18.0.6:6000). Time-boxed to ~18:45: real tok/s, hits, KV and temperatures over the steady part,
    plus correctness paired with E043's attempts. The finished attempts are biased toward short ones.
  - **Server `rw_e128t2`** (with the parsers): up in 120 s, 16.87 GiB/GPU, KV 1,447,673 tokens (2.13 GiB). Per GPU,
    out of 22.04 GiB usable (ECC on: 23,034 MiB total, 469 MiB reserved): ~1.0 GiB non-torch, 0.51 activations,
    0.31 CUDA graphs. nvidia-smi shows ~21.4 GiB used.
  - **Real-weight decode, 8K random-token context, no DSpark** (`rw_dw.sh`, `runs/rw_e128t2/dw_8192_c*.json`):

    | C | tok/s | per stream | ITL | dummy (ds_x128_u93) |
    |---:|---:|---:|---:|---:|
    | 32 | 464.0 | 14.5 | 68.8 ms | 334.9 |
    | 64 | 625.1 | 9.77 | 102.3 ms | 487.5 |
    | 128 | 750.9 | 5.87 | 170.1 ms | — |

    Prefill ~1330–1376 tok/s; at C128, GPUs 71–73 °C, KV peak 0.71, 0 preemptions. Real routing is more concentrated
    than the dummy weights' routing, so the real weights decode 28–39% faster.
  - **Real ARC tool run** (`eval_rw.sh rw_e128t_l4`, 40 workers, 17:25–18:59 UTC; stopped by PID at the time box,
    `evals/results/tools_persistent_grid_lean2sub_dsv41_low_e042_rw_e128t_l4.jsonl`). From `kvmon.log`:

    | window (UTC) | gen tok/s | prefix hits | uncached prefill tok/s | KV p50 / max |
    |---|---:|---:|---:|---:|
    | 17:25–17:40 | 263.5 | 84.7% | 435 | 0.21 / 0.31 |
    | 17:40–18:00 | 247.5 | 74.7% | 521 | 0.35 / 0.45 |
    | 18:00–18:20 | 180.4 | 66.5% | 712 | 0.49 / 0.57 |
    | 18:20–18:40 | 176.4 | 64.7% | 728 | 0.59 / 0.63 |
    | 18:40–19:00 | 130.9 | 54.8% | 893 | 0.66 / 0.69 |
    | **17:45–19:00** | **176.8** | **63.8%** | **736** | 0.55 / 0.69 |

    - 0 preemptions; 38–40 running; GPUs max 71–73 °C at ~99% utilisation.
    - The T4 replay at the same C40 gave 318.6 tok/s and 96.7% hits at KV peak 0.76.
    - At 35.2K output per attempt, 176.8 tok/s is ≈ 18 attempts/h (≈ 18.5 h for 334), vs 32.6 from the replay.
    - Only 18 attempts finished: one attempt takes ~75 min at ~4.4 tok/s per stream. On the same attempts this run had
      17 correct and E043 18, too few to judge.
    - E043's 8× box, same harness and model, got ~92% cumulative hits at KV ≤ 17% (`E043 results/cache_watch.tsv`).
      So the low hit rate comes from L4's small cache pool, not from the harness alone.
    - **Hypothesis (unverified):** the replay sends token-exact prefixes. The real harness re-renders the transcript
      through the chat template, which likely changes the last assistant turn (thinking or DSML tool call), so the
      cache has to resume before it. The blocks there, probably the sliding-window state freed while the 15–20 min
      thinking decode ran, have been evicted from the ~450K-token free pool on L4, so the turn re-prefills from much
      earlier. The miss grows with context: uncached prefill rose from 435 to 893 tok/s.
    - **Lesson:** token-exact replays overstate tool throughput for this model on L4. Validate on real traffic.
  - **Where the prefix breaks (the user, 19:05: "please figure out where is the mismatch"):**
    - Offline (`scripts/prefix_offline.py`, server `/tokenize` on 25 turns of 3 saved real-run transcripts): each
      prompt is an exact token prefix of the next one. With tools in the request, the V4.1 encoder keeps every
      earlier turn's thinking (`effective_drop_thinking = False`), so the old part of the prompt never changes.
    - Live (`scripts/prefix_probe.py`, real harness with `return_token_ids`, 3 tasks): 6 of 6 short turns (162–688
      sampled tokens) were exact extensions of prompt + sampled ids.
    - E043 (`render_parity.py` on its 8× E112 server): 1 of 11 tool turns broke 11 tokens into a 1,032-token turn.
      The model had sampled a non-canonical split ("fil|ls"), and re-tokenizing the text gives the canonical split.
      On 78,992 API turns, 13% re-tokenize to a different length.
    - Under real 40-stream load (`scripts/prefix_hitlog.py`, server `rw_e128t3` with
      `--enable-prompt-tokens-details`, first 12 min; `hitlog_e128t_w40_report.txt`): **20 of 93 turns (21%)
      broke**.
      - Eviction loss (cache that existed but was gone) was 3.1% of queried tokens: 2.1 points on broken turns,
        1.1 on exact ones.
      - The break itself cost only 0.6%, but it moves the resume point back into turn-k blocks freed during the
        decode.
      - Early in a run the misses are mostly first prompts (22%), so this window understates the late-run loss.
    - Fix (E043, 19:43): `--token-exact` in `evals/run_deepseek_tools.py` (opt-in, `--base-url` only). Each attempt
      keeps its exact token history; only newly appended messages are rendered, through the server's
      `/tokenize`, and the turn is generated via `/v1/completions` with the ids.
  - **E128 with `--token-exact`** (`eval_rw.sh rw_e128t_te --token-exact`, 40 workers, 19:28–19:44 UTC, server
    `rw_e128t3`; stopped for E112):

    | window | gen tok/s | hits | uncached tok/s | KV p50 / max |
    |---|---:|---:|---:|---:|
    | 19:28–19:32 (first prompts) | 196.9 | 83.4% | 665 | 0.14 / 0.18 |
    | 19:32–19:36 | 337.5 | 91.8% | 194 | 0.19 / 0.21 |
    | 19:36–19:40 | 353.0 | 93.0% | 146 | 0.23 / 0.25 |
    | 19:40–19:44 | 347.1 | 95.6% | 140 | 0.26 / 0.32 |

    - At the same run age (0–15 min), the unfixed run made 263.5 tok/s at 84.7%.
    - 0 preemptions; GPUs max 71–73 °C.
    - 5 attempts finished, all correct; 0 re-renders, 0 parse fallbacks. Their `breaks` counter (6 in 19 turns) is
      the old cumulative one: it counts every turn after an attempt's first break (E043 fixed it at ~20:10).
    - Caveat: only the early phase (KV ≤ 0.32) was seen. The unfixed run lost most of its speed after 35 min, when
      KV passed 0.5.
  - **E112** (the user, 19:25: "please continue with E112. We want to have a decision which model to use on Kaggle
    within the next 1 hour"; `scripts/run_rw112.sh`):
    - Setup: deleted the E128 files, then downloaded `chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E112-2bit`
      19:44–19:54 (124.9 GB). Same split as E128: 90 GiB in /dev/shm, 28 GiB on disk, 6.5 GiB disk free afterwards.
    - E043's row-chunk Engram loader (`apply_compact_engram.py` v2, which only adds the `nvidia/model.py` edit) was
      applied over the E128 patches. Its first real-server load passed the full-coverage asserts.
    - Server `rw_e112t` (E128 flags plus `--enable-prompt-tokens-details`): up in 160 s, **15.29 GiB/GPU, KV
      2,571,147 tokens** (+78% over E128). The smoke chat answers correctly.
    - Pod memory (cgroup v1) after load: 219 of 233 GiB used (180 GiB shmem), failcnt 0.
    - **ARC run with `--token-exact`, 60 workers** (`eval_rw.sh rw_e112t_te --token-exact`, from 19:57 UTC):

      | window | gen tok/s | hits | uncached tok/s | KV p50 / max |
      |---|---:|---:|---:|---:|
      | 19:57–20:01 (first prompts) | 223.3 | 85.1% | 705 | 0.10 / 0.13 |
      | 20:01–20:05 | 395.4 | 91.8% | 245 | 0.13 / 0.15 |
      | 20:05–20:10 | 425.3 | 94.8% | 158 | 0.17 / 0.18 |

      0 preemptions; GPUs max 71–73 °C; 3 attempts finished, all correct (old cumulative `breaks` counter: 3 in 12 turns).
  - **Decision inputs at 20:10** (E128 vs E112, both on 4×L4 with `--token-exact`, minutes 4–15 of a run):

    | | E128 × 40 | E112 × 60 |
    |---|---:|---:|
    | gen tok/s | ~345 | ~410–425 |
    | KV max so far | 0.32 | 0.18 |
    | tokens/attempt (E043, same 190 attempts) | 22.0K (all 334: 35.2K) | 19.3K (0.88× E128) |
    | ≈ attempts/h at the full-run mean | ≈ 35 (35.2K) | ≈ 48 (≈ 31K est.) |
    | E043 ARC on the same 190 attempts | 163 correct | 157 correct (one-sided 5 vs 11) |
    | E043 pass@2, 74 pairs with both attempts | 69 | 68 |
    | Python tool errors | 12.7% | 27.6% |

    - E043's E112 run finished the easier attempts first, so its 190 are biased toward short ones.
    - The late-run speed with the fix is not yet measured.
- **E043 follow-ups (09:00):**
  - E96 2/3-bit GPTQ ARC now goes first, from a GPTQ-only E96 export. E043's user wants E96 only if it is much
    faster. Partial (51 easy-first pairs): pass@1 31 vs full 37, mean output 26.8K (1.12× full).
  - E192 2-bit dev NLL +5.5%: more experts help, but it cannot fit L4. E043's next levers within the L4 budget are
    non-uniform per-layer expert counts and 3 bits on sensitive layers.
  - Non-uniform per-layer expert counts would need a loader/allocation patch here: `n_routed_experts` is global, and
    the router whitelist is per shape.
  - E96 GPTQ-only export: HF `chankhavu/DeepSeek-V4.1-Flash-1017-arc-notool-E96-2bit3bit` (private), 116,494,276,575
    B = 108.5 GiB, fits /dev/shm. Same layout and env as E128, but with `DSV41_EXPERT_HUMMING_W2`
    `{"weight_dtype": "uint3", "group_size": 64}`. Export parity: dev NLL 0.65215 vs 0.65213 with runtime masks. ARC
    (2 attempts) due ~11:00 UTC.
  - **A teammate's 4×L4 setup (relayed by E043 from the user, 09:30; their own engine, code not available):**
    - All 384 experts as unsloth UD-IQ2_XXS (156 GB, ≈ 2.3 bpw) held in host RAM, with a GPU expert cache: misses
      are copied to the GPU, ≈ 97% hit rate. Full Engram at Q4 (110 GB) split across disk and /dev/shm. Q8 KV cache.
    - Q8 host-side all-reduce: ≈ 25 GB/s, vs ≈ 6 GB/s NCCL P2P (Kaggle NCCL SHM here: ≈ 4.9 GB/s).
    - Their numbers: ≈ 440 tok/s no-tool at batch 60 and 56K context, ≈ 360 at batch 32. E288 gives 460.
    - Correction: I had told the user that offloading experts to CPU would collapse batched throughput, because I
      assumed every step streams the whole bank. A GPU-resident hot-expert cache copies only the misses, so offload
      is competitive with all-GPU E128 2-bit (419 @ 48) while keeping every expert.
    - The deciding number is E043's full-384 2-bit GPTQ ARC, the closest proxy for their model, due ~13:30 UTC.
  - Queued at E043: E128 2-bit for the tool harness. It uses a tool-statistics keep set, the 64 GiB tool Engram table
    and its own GPTQ calibration. Memory and KV match the no-tool E128, but tool-mode throughput needs a T72-style
    multi-turn replay with prefix caching.

### Pruned-checkpoint handoff format (agreed with session dsv4-pruning, 2026-10-23)

The real-weight pruning runs in another session on an 8× RTX PRO 6000 box (Vast 57370044); it evaluates candidates on
the full model with runtime masks (pruned experts get bias −1e9 before the native top-6; dropped Engram rows map to
id −1 → zero row) and a REAP observer on `MarlinExperts.moe_sum`. The export it will hand over:
- **Experts:** HF-style checkpoint, `text_config.n_routed_experts` = E, top-6 unchanged, experts renumbered 0..E−1 in
  ascending original-id order; `gate.weight`, `e_score_correction_bias` and `bias_vl` sliced; native MXFP4 bytes and
  scales copied unchanged; a keep manifest with the original ids. Needs the router whitelist entry (E, 6).
- **Engram, per layer (1 and 14):** compact FP8 e4m3 values [R, 256] + uint8 UE8M0 scales [R, 8] (264 B/row, the
  fork's native row format), rows in original-id order; an int32 map from original hashed row id to compact row
  (−1 = miss = zero embedding before Wkv). The map is ≈ 2.86 GiB over all original rows.
- **Budget:** Kaggle cgroup 173.0 GiB; non-table host use ≈ 26 GiB steady / 30 GiB at load, so table + map ≤ ≈ 125 GiB
  (full table ≈ 189 GiB). Load in bounded chunks into the pinned buffer (the synthetic 8× fill had a whole-table
  transient). A GPU-side map would come out of KV (2.86 GiB replicated = all of E64's; 0.72 GiB/rank sharded).
- **E043's actual export (2026-10-23):** compact rows as `layers.{1,14}.engram.embed.weight` / `.scale`, map
  `layers.{1,14}.engram.embed_tokens.row_map` (int32), `text_config.engram_compact_kept` (24 kept counts per layer,
  one per hash column). Loader: `experiments/E043_dsv41_real_pruning/patches/apply_compact_engram.py` (per-rank
  contiguous compact slice; rank-local map in pinned host memory, read by `_engram_lookup_kernel` with `HAS_MAP`).
  E043's Engram choice: top-frequency rows, 64 GiB (34% of the table), held-out NLL +0.09%, hit rate 94%.
- **First real export (E043, 2026-10-23):** rule one-shot "contribution total" (sum of native weighted-output
  norms; decoder layers 20–39 scored on generation targets; iterative re-observation lost at every size). NLL on 32
  eval-task traces: full 0.664, E72 0.822, E64 0.857, E48 0.962. The physical E64 export (15.17 GiB/GPU at TP4) matches
  the runtime masks (NLL 0.85694 vs 0.85723, identical top-1). Private HF: `chankhavu/DeepSeek-V4.1-Flash-1017-arc-
  engram64` (shared `model-engram-0000{1,2}` files, ~71 GB, + `engram_index.json`), `chankhavu/DeepSeek-V4.1-Flash-
  1017-arc-E64` (backbone, config, tokenizer, `e043_manifest.json`); E48/E72 to follow; per-row Engram counts in
  dataset `chankhavu/dsv41-flash1017-e043-pruning-artifacts` (`engram_counts/`). The Vast L4 box cannot hold it
  (64 GB disk, 108 GiB cgroup); the L4 test must run on Kaggle (/tmp ≈ 1 TB free, 173 GiB RAM).
- **Hold (E043, 2026-10-23):** E64 and E48 exports break in free generation (ARC harness 0/11 and 0/5: one turn, no
  valid tool call; a trivial tool prompt at temperature 0 gives "run_ppython" and repeated identical calls), although
  teacher-forced NLL was only +29%. E043 is isolating Engram mask vs expert pruning. Suggested from here: decode vs
  teacher-forced argmax consistency, free generation of the runtime-masked full checkpoint, and an in-distribution
  Engram miss policy (each token gathers 24 rows into `wkv`, so a zeroed row shifts key and value; the gate is not 0).
  No Kaggle time on real weights until this is resolved.
- **Diagnosis (E043):** not Engram and not a decode bug. Greedy decode matches the teacher-forced argmax except fp8
  near-ties, as for the full model. At ≤ E72 the model loses rare DSML structure: after `</｜DSML｜ invoke>` it opens
  another invoke ("run_p_python") instead of `</｜DSML｜ calls>` and runs to max tokens. First-turn valid tool calls:
  full 9/10, Engram 64 GiB only 4/4, E192 9/10, E128 8/10, E96 10/10, **E72 0/10**, E64/E48 broken. Tool-call blocks
  are ≈ 6% of positions, which all-token NLL and contribution keep sets underweight. Also: switching runtime masks on
  a prefix-caching server reuses KV from the previous mask; use a per-config `cache_salt`. E043 now tests keep sets
  that protect experts by contribution on tool-block positions (E72 sp16/sp32, E64 sp32, E48 sp32). Suggested from
  here: a teacher-forced metric at DSML structural positions, and a harness-side stop at `</｜DSML｜ invoke>` that
  closes the block client-side (one run_python call per turn). E96 (≈ 21 GiB/GPU) does not fit 4×L4; E72 is the max.
- **First ARC numbers (E043, 167 eval pairs, 1 attempt, self-hosted):** full model 110/167 (65.9%); **E72 w30**
  (keep set = contribution total + tool-block positions × 30, which fixes the DSML structure: valid first-turn calls
  0/10 → 10/10) + Engram 64 GiB + `--rescue-code`: **49/167 (29.3%)**. E64 w30 ≈ 25% partway, with real semantic
  damage (57% of tool calls error); E48 w30 running. HF repos `-arc-E72` and `-arc-E64` now hold the w30 keep sets
  (the one-shot E64 was overwritten). Against E042's 12 h budgets (E72 ≈ 400–450 estimated; E64 735 at C 64, run 9
  pending; E48 1,010–1,090 at C 128, not E48's ceiling), E72 has ≈ 2.5 attempts per eval-sized pair set, E48 ≈ 6.
  A real-weight Kaggle harness needs E043's `--rescue-code` safety net.
- **From E043's jsonl files** (`evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_<tag>.jsonl`):

  | | full | E72 w30 | E64 w30 |
  |---|---:|---:|---:|
  | correct | 110 (65.9%) | 49 (29.3%) | 30 (18.0%) |
  | P(correct \| train_pass) | 88/107 (82%) | 37/60 (62%) | 21/29 (72%) |
  | tool calls that error | 2.5% | 45% | 53% |
  | output tokens mean / p50 | 23.9K / 23.9K | 26.1K / 22.3K | 32.2K / 25.0K |
  | attempts past 30K (correct among them) | 55 (19) | 40 (2) | 67 (2) |

  A 30K output cap would cost E72 2 of its 49 correct answers (the budget model's assumption holds). Kaggle-pace
  Qwen3.8-27B for scale (other pair sets): 33–40% pass@1 (E023 v4b 33.3%, E041 Kaggle 19/48; E020 K3 37.4%).
- **To do here:** (1) for real weights, apply only the router whitelist, PR #154 and Humming from E042, plus E043's
  loader: E042's `DSV41_ENGRAM_SYNTH_ROWS` edit uses the same `engram.py` anchors and conflicts with it.
  (2) After R7: measure the map's dependent UVA read on L4 (synthetic 34%-kept map and table through E043's patch,
  decode/prefill vs no map). (3) Load RAM: not a problem for whole-layer tensors. Tested with safetensors 0.8.0 (the fork's pin):
  `get_tensor` is a memory map (RSS unchanged for 2.86 GiB), and copying a ¼ slice touches only that slice's file
  pages (reclaimable page cache) plus the destination. vLLM's default load strategy is lazy mmap (prefetch on a
  network FS); only an explicit `--safetensors-load-strategy eager` reads whole files into anonymous memory.
  An earlier warning to E043 about per-worker full copies was retracted.
