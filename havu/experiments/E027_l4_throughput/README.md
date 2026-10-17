# E027: faster inference for Qwen3.8-27B on 4×L4 (weight and KV formats, MTP, interconnect, parallelism)

Started 2026-10-16. The goal is more decode throughput (and KV room) on the Kaggle-like 4×L4 box than the FP8 checkpoint gives, without losing ARC accuracy.

(First numbered E025; renamed because `E025_train_rollouts` already existed. The Kaggle notebooks pushed on 2026-10-16/17 still say "E025".)

## Summary (2026-10-17 03:00 UTC)

**Goal:** 1.5–2× v4b's tool-traffic throughput on Kaggle 4×L4 at unchanged accuracy. v4b reference: 224 tok/s on Kaggle, 201 on the Vast L4 box, 13–19 of 48 probe pairs.

| Lever | Verdict | Evidence |
|---|---|---|
| INT4 GPTQ / AWQ weights | **rejected** (accuracy) | decode +17–66%, real traffic only +15%; probe 9 and 10 of 48 vs FP8 13–19 (INT4 vs FP8 McNemar p = 0.039) |
| NVFP4 weights | not better than INT4 | Marlin FP4 W4A16 on L4, ~5% slower than INT4 |
| GGUF Q4_K_M, llama.cpp ×4 | **rejected** (speed) | 97–206 tok/s; vLLM 0.23 cannot load Qwen3.8 GGUFs |
| Ternary Bonsai 2 27B, llama.cpp ×4 | not pursued | ties INT4 up to c16, plateaus near 287 tok/s; ARC quality unknown |
| Triton block-FP8 W8A8 instead of Marlin, async scheduling | no gain | −2–12% and −5–15% decode |
| Larger `max_num_batched_tokens` | no gain | prefill 916 → 888 → 870 tok/s (2048 → 4096 → 8192) |
| NCCL P2P (`NCCL_P2P_LEVEL=SYS`) | box only; **harmful on Kaggle** | box: all-reduce 8–9× faster, prefill 3×; Kaggle: P2P 12× slower than SHM (v4c commit run several times slower) |
| Pipeline parallelism (PP4, PP2×TP2) | **rejected** | prefill 3.2–3.3× faster, decode 14–77% slower |
| TurboQuant KV (`k8v4`, `4bit_nc`, `3bit_nc`) | KV +26/+84/+135%, decode −4–18% | `k8v4` full-eval accuracy so far equal to baseline (26 vs 27 of 56) |
| **MTP2 on vLLM 0.31** | **promising** | decode +26–46% on L4, clean output; full-eval interim 45 vs 50 of 112 (p = 0.46), same tokens per attempt |
| MTP2 + TurboQuant | **broken** in vLLM 0.31 | 19 of 64 degenerate outputs |
| Transcript compaction (`ARC_COMPACT_AT`) | implemented, untested on accuracy | evals queued on the RTX 6000 boxes |
| Tuned FP8 GEMM configs for L4 | queued | Kaggle vLLM warns "Using default W8A8 Block FP8 kernel config"; GEMM is 24–31% of decode |

**Where time goes on 4×L4:**
- The box profile of TP4 decode at ~25k context: NCCL all-reduce 40–52%, FP8 GEMM 24–31%, attention 16–19%, DeltaNet 3–4%.
- Kaggle's default all-reduce is ~2× faster than the box's, so comm is ~19% of a decode step there.
- Kaggle v4b decodes at 336 tok/s with short contexts, falling to ~200–256 as transcripts grow.

**Next:** finish the MTP2 accuracy eval (box A vs B) and its L4 tool probe; tune the FP8 GEMM configs; evaluate compaction. A candidate (v4b + vLLM 0.31 + MTP2, concurrency ~20, maybe compaction or tuned configs) needs one Kaggle validation probe before submission.

**Box:** Vast 55889595 (4× L4 24 GB, PCIe; same box as E023), restarted 2026-10-16 11:22 UTC.

**What L4 can and can't do:**
- L4 is Ada (SM89), so it has FP8 tensor cores but no FP4 ones.
- In vLLM, FP8 checkpoints run as Marlin W8A16 (E023), and INT4/NVFP4 run as Marlin W4A16 weight-only kernels.
- vLLM 0.23 cannot load Qwen3.8 GGUFs: its GGUF loader has no `qwen3_5` → `qwen35` mapping, and its GGUF path is documented as experimental and unoptimized. GGUFs therefore run on llama.cpp.

**Decode bench:** `bench_decode.py` (vLLM, one TP4 server) or `bench_dp.py` (four single-GPU servers, round-robin).
- Prompts: real ARC tasks, ~2.4–2.7k tokens, each starting with a unique nonce so prefixes are not reused.
- Output: 3,000 tokens per request with `ignore_eos`; levels are total concurrent requests on the box.
- vLLM servers run with `--no-enable-prefix-caching --language-model-only`, fp8 KV unless noted; llama.cpp with q8_0 KV, 12 slots per GPU and a unified KV pool.

## Results: box decode tok/s

| Setup | Weights on disk | KV capacity | c=8 | c=16 | c=32 | c=48 |
|---|---|---|---|---|---|---|
| vLLM 0.23 TP4, FP8 (`Qwen/Qwen3.8-27B-FP8`, Marlin W8A16), E023 | 29 GB | 1.40M tokens | 109 | 167 | 253 | 307 |
| vLLM 0.23 TP4, **INT4** (`RedHatAI/Qwen3.8-27B-INT4`, GPTQ W4A16 g128) | 19.5 GB | **1.99M tokens** | 181 | **259** | **337** | **358** |
| llama.cpp `f336359`, 4 × 1 GPU, unsloth `UD-Q4_K_M` | 16.5 GB | 12 × 72k slots per GPU (q8_0) | 97 | 103 | 128 | 206 |
| PrismML llama.cpp fork `5ae2e16`, 4 × 1 GPU, **Ternary Bonsai 2 27B** `PTQ1_0` (1.75 bits/weight) | 5.95 GB | 12 × 72k slots per GPU (q8_0); ~17 GB free per GPU | **190** | **264** | 271 | 287 |
| vLLM 0.23 TP4, NVFP4 (`nvidia/Qwen3.8-27B-NVFP4`, `MarlinNvFp4LinearKernel`) | 21.9 GB | 1.86M tokens | 169 | 246 | 317 | 340 |
| vLLM 0.23 TP4, INT4 + TurboQuant KV `turboquant_k8v4` (backend TURBOQUANT) | 19.5 GB | 2.51M (+26% vs fp8 KV) | 174 | 240 | 306 | 323 |
| vLLM 0.23 TP4, INT4 + `turboquant_4bit_nc` | 19.5 GB | **3.65M (+84%)** | 172 | 238 | 302 | 317 |
| vLLM 0.23 TP4, INT4 + `turboquant_3bit_nc` | 19.5 GB | 4.67M (+135%) | 164 | 225 | 281 | 294 |

**TurboQuant** works with the hybrid Qwen3.8; vLLM compresses only the full-attention layers. It costs 4–18% decode speed and buys 26–135% KV capacity, so it helps only when KV-bound (high concurrency with long tool contexts). Quality is not tested yet: published results show reasoning drops at 3 bits on small models, so `4bit_nc` is the preset to probe first.

**Notes:**
- The E023 claim that INT4 is slower than FP8 was wrong. That bench used ~8k prompts against FP8's ~2.6k. With matched prompts, INT4 is 17–66% faster and has 42% more KV.
- llama.cpp Q4_K_M batches poorly on L4: ~52 tok/s per GPU even at 12 concurrent.
- Bonsai matches INT4 up to 16 concurrent, then plateaus near 270–290. Its appeal is memory: one copy per GPU leaves ~17 GB for KV, and there is no cross-GPU traffic.
- Bonsai's quality on ARC is unknown. Its card claims 98.2% of FP16 on 14 thinking benchmarks, but Nemotron Bolt (also ~2-bit) collapsed on ARC (E024).
- Prefix reuse on llama.cpp's hybrid models needs `--ctx-checkpoints` and `--cache-ram` (Bonsai-demo `PROMPT-CACHE.md`). That is not tested yet; tool use needs it.

**Failed attempts (re-queued):**
- **INT4 as 2 × TP2:** crashed in CUDA-graph profiling with `custom_all_reduce.cuh:455 'invalid argument'`, probably GPU peer access unavailable on this PCIe box. Rerun with `--disable-custom-all-reduce`.
- **INT4 TP4 with `--async-scheduling`:** never benchmarked. The chain's wait helper returned the status of a failed `grep`, so it treated the live server as down. Rerun.

**Tool-traffic probes (v4b settings, 48 pairs, forced at ready + 120 min; `probe.sh`):**

| Probe | Weights | KV | Concurrency | Score (35 tasks) | tok/s | Notes |
|---|---|---|---|---|---|---|
| E023 run B (reference) | FP8 | fp8 1.40M | 24 | 13.33 (33.3% of pairs 16/48) | 201 | cache hit 86.7% |
| p_int4_c24 | INT4 (RedHatAI GPTQ g128) | fp8 1.79M (prefix caching on) | 24 | **8.00** (9/48) | **232** (+15%) | 34/48 hit the 48k cap (FP8 25); 14 train-passing submissions (FP8 23); 34.9k tokens per attempt (FP8 29.4k); cache hit 87.1%, no preemptions |
| p_int4_c32 | INT4 | fp8 | 32 | stopped at 35 min (INT4 already out on quality; AWQ answers the same throughput question) | | |
| p_awq_c32 | AWQ (`cyankiwi`, g32 asym, DeltaNet BF16) | fp8 1.74M | 32 | **7.83** (10/48) | 210 | 31/48 hit the 48k cap; 17 train-passing submissions; 31.6k tokens per attempt; cache hit 83.7%; vs FP8 run B: 10 FP8-only, 4 AWQ-only (p = 0.18) |

**INT4 GPTQ loses accuracy on tool use.** On the same 48 first attempts, FP8 alone solved 8 and INT4 alone 1 (exact McNemar p = 0.039). FP8 has scored 13, 16 and 19 of 48 on this probe; INT4 got 9. INT4 rambles more and submits fewer train-passing programs. Its real-traffic speedup (+15%) is far below the decode bench's +55% at c16, because tool traffic adds prefill and long-context attention that weight compression doesn't speed up.

**Next quality candidate:** `cyankiwi/Qwen3.8-27B-AWQ-INT4` (21 GB): AWQ g32 asymmetric with an MSE observer, calibrated on STEM/agentic data, with the DeltaNet linear-attention layers left in BF16. It runs on the same Marlin W4A16 path.

**Both 4-bit quants lose accuracy.** FP8 has solved 13–19 of these 48 pairs (three runs); INT4 solved 9 and AWQ 10. Both think longer and hit the token cap more.

**c32 is not a fair throughput test on this probe.** It has only 48 pairs, sized for c24, so at c32 the queue empties mid-run and effective concurrency falls (11 running at the end). Throughput at higher concurrency needs a saturating workload.

**Change of direction (20:05 UTC): keep FP8 weights, use the FP8 tensor cores.**
- On L4, vLLM runs the official block-FP8 checkpoint as Marlin W8A16 (bf16 activations), although Ada has FP8 tensor cores. The checkpoint declares `activation_scheme: dynamic`, i.e. block-FP8 W8A8 is its intended mode.
- `chain_w8a8.sh` benches Marlin against the Triton block-FP8 W8A8 kernel (`VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel`) for decode and prefill (16 × ~12k-token prompts), then the winner with `--async-scheduling`.
- Alternatives:
  - `huginnfork/Qwen3.8-27B-FP8`: FP8-dynamic W8A8 on the MLPs only (attention and DeltaNet in BF16), 38.5 GB, CUTLASS on SM89.
  - `Avesed/Qwen3.8-27B-INT8-W8A8`: INT8 W8A8 per channel/per token, 31.3 GB.
- AWQ and Bonsai were deleted for disk.

### FP8 kernels and the prefill bottleneck (official `Qwen/Qwen3.8-27B-FP8`, vLLM 0.23 TP4, KV 1.59M with prefix caching off)

| Variant | c=8 | c=16 | c=32 | c=48 | Prefill (16 × 12.3k prompts) |
|---|---|---|---|---|---|
| Marlin W8A16 (v4b) | 131 | 171 | 256 | 302 | 916 tok/s (215 s) |
| Triton block-FP8 W8A8 (Marlin/Humming disabled) | 115 | 158 | 243 | 295 | 915 tok/s |
| Marlin + `--async-scheduling` | 111 | 154 | 232 | 284 | 912 tok/s |

**W8A8 and async scheduling don't help on L4.** Prefill is the striking number, and it is identical across GEMM kernels:
- The server prefills only 1–4 prompts at a time, at a steady ~1,240 tok/s, with 15 waiting.
- 4 × L4 FP16 peak would allow roughly 5–9k tok/s.
- **Cause found:** vLLM's default `max_num_batched_tokens` is **2048** for API servers on GPUs under 70 GB (`arg_utils.py`; H100-class cards get 8192). v4b does not override it (`kaggle/src/solver.py`), but the solver's `ARC_SERVER_EXTRA` can pass `--max-num-batched-tokens` without code changes.
- Tool traffic has ~2M uncached prompt tokens per 2-hour probe (15–17M prompt tokens at 84–87% cache hits). At ~1.2k tok/s that is ~30 min of prefill per probe.
- Sweep 4096/8192/16384, then a v4b probe with the best value: `chain_mnbt.sh`.

**The chunk size is not it.** `--max-num-batched-tokens` 4096 gave 888 tok/s and 8192 gave 870, against 916 at 2048.

### Root cause: NCCL all-reduce goes through host shared memory (`allreduce_bench.py`, 4 GPUs, bf16 [tokens, 5120])

- `nvidia-smi topo -m`: GPU0–GPU1 PHB, all other pairs NODE. P2P read reports OK for every pair.
- NCCL 2.28 nevertheless picks `SHM/direct` (its default P2P level excludes NODE). Forcing `NCCL_P2P_LEVEL=SYS` switches it to `P2P/CUMEM`.
- **Correctness:** `allreduce_check.py` found 0 mismatches in 4,800 checks with P2P forced.

| Tokens per all-reduce | Default (SHM) | P2P forced | × 128 all-reduces per TP4 step |
|---|---|---|---|
| 24 (decode at c24) | 0.18 ms, 2.1 GB/s | 0.06 ms, 6.2 GB/s | 0.022 s → 0.008 s |
| 48 (decode at c48) | 0.28 ms | 0.09 ms | 0.036 s → 0.012 s |
| 2,048 (one prefill chunk) | 14.2 ms, 2.2 GB/s | 1.62 ms, 19.4 GB/s | **1.82 s → 0.21 s** |
| 8,192 | 57.5 ms | 6.5 ms | 7.4 s → 0.84 s |

- With SHM, a 2,048-token prefill chunk spends ~1.8 s in all-reduce. That matches the ~1.65 s per chunk seen in vLLM, so **prefill on this box is almost all communication**.
- Decode steps (~0.1–0.16 s) also spend 20–30% of their time in it.
- `chain_p2p.sh` measures vLLM with `NCCL_P2P_LEVEL=SYS` (benches, then a v4b probe at c24).
- **Kaggle:** whether its 4×L4 VM has the same topology and allows P2P is unknown. It is checkable with a short 4×L4 session (`nvidia-smi topo -m`, `allreduce_bench.py`, `allreduce_check.py`; a few minutes of GPU quota, which needs the user's approval).

## Next

1. Finish NVFP4 and the TurboQuant presets.
2. Run an accuracy probe on the fastest candidates. **INT4 has never had one:** the E023 `k3_int4_*` probes died at startup during a failed download.

### MTP and TurboQuant on vLLM 0.31.0 (L4 box, FP8, TP4; 2026-10-16 22:45–23:55 UTC)

Decode bench: `bench_decode.py`, ~2.5k-token prompts, 3k out, NCCL_P2P_LEVEL=SYS.

| Variant | KV capacity | c=16 | c=24 | c=32 | MTP acceptance | Degenerate outputs (`mtp_check.py`, 64) |
|---|---|---|---|---|---|---|
| a: no MTP, fp8 KV | 1.56M | 195 | 230 | 268 | — | 0 |
| b: **MTP2**, fp8 KV | 1.25M (−19%) | **286 (+46%)** | **305 (+32%)** | **338 (+26%)** | 2.81 | 0 |
| c: MTP2 + `turboquant_k8v4` | 1.49M | 271 | 305 | 336 | 2.17 | **19 (broken)** |
| d: `turboquant_k8v4`, no MTP | 1.99M | — | — | — | — | 0 |

- **MTP2 + TurboQuant is broken in vLLM 0.31.** Even with a cold cache, c's text degenerates: `"apply to some \" \" \" \" ma"`, an early `</think>` followed by blank lines, and `"The user says: \" \"Find..."`. In addition, all 16 second turns diverge at the same character as soon as the prefix is cached.
- The TurboQuant backend treats any batch with `max_query_len > 1` as prefill, so every MTP verify step goes through that path. MTP2 alone and TurboQuant alone look clean.
- **The greedy consistency check cannot test the MTP cache bug.** Without MTP the stack is not bit-deterministic either (variant a: cold vs cached identical in 2 of 16). Also, turn 2 builds on generated turn 1, so turn-2 prompts differ between runs. The bug question is settled by downstream accuracy instead (RTX 6000 full evals).

### Full-eval accuracy on 1× RTX PRO 6000 boxes (v4b agent recipe, attempt 1 + free attempt_2, vLLM 0.31, FP8)

| Run | Box | Lever | Concurrency (KV ÷ 55k) | Status |
|---|---|---|---|---|
| e6_base | B 56310184 (WS) | none | 32 (1.81M KV) | running from 23:41 |
| e6_mtp2 | A 56310182 (Max-Q) | MTP2 | 28 (1.56M) | running from 23:10 |
| e6_tq | C 56310968 (Max-Q) | `turboquant_k8v4` | 40 (2.31M) | running from 23:40 |
| e6_compact | B, after e6_base | compaction at 24k tokens (`ARC_COMPACT_AT=24000`) | — | queued |
| e6_mtp2_compact | A, after e6_mtp2 | MTP2 + compaction | — | queued (replaces MTP2 + TurboQuant) |

Results are copied to `e6/<label>/`, and each box is destroyed by `box_pipeline.sh` once its last run is fetched.

### Profile of steady decode (vLLM 0.23 FP8 TP4, 24 requests at ~24.6k-token context, 20-s torch-profiler window, rank 0)

`prof_run.sh` / `prof_parse.py`; traces in `/root/arc/runs/prof_{shm,p2p}/trace` on the L4 box. The FP8 GEMM ran as `_w8a8_triton_block_scaled_mm` here, not Marlin.

| GPU time share | NCCL default (SHM) | NCCL_P2P_LEVEL=SYS |
|---|---|---|
| NCCL all-reduce (`AllReduce_Sum_bf16_RING_LL`), including waits for other ranks | **52%** | **40%** |
| FP8 GEMM (Triton block W8A8) | 24% | 31% |
| Full attention (`kernel_unified_attention`) | 16% | 19% |
| DeltaNet decode (FLA `fused_recurrent_gated_delta_rule_packed_decode`) | 3% | 4% |
| BF16 GEMMs, norms, activations, sampling | ~5% | ~5% |

GPU busy 96–97% of the window. **On this box, TP4 decode spends more time in all-reduce (transfer plus synchronization) than in GEMM and attention combined.**
- This motivates pipeline parallelism: only one hidden-state send per stage boundary per step (~0.25 MB at 24 requests), against 128 all-reduces per step under TP4, and no P2P needed.
- `chain_pp.sh` benches PP4 and PP2×TP2 on vLLM 0.23.

### Kaggle 4×L4 interconnect (`chankhavu/arc2-l4-p2p-check2` v2, 2026-10-17 ~01:30 UTC, ~2 min of GPU)

- **Hardware:** 4 × NVIDIA L4, driver 580.178.04, **PCIe Gen3 max** (Gen1 at idle), x16. All GPU pairs PHB; `nvidia-smi` reports P2P read/write OK.
- **v1 of the check got 2×T4:** without the ARC-AGI-2 competition attached, Kaggle silently uses the default accelerator.

| bf16 [tokens, 5120] all-reduce | Kaggle default (SHM) | Kaggle `NCCL_P2P_LEVEL=SYS` | Vast box default (SHM) | Vast box P2P |
|---|---|---|---|---|
| 24 tokens | 0.14 ms (×128 = 18 ms) | 0.53 ms | 0.18 ms | 0.06 ms |
| 2,048 tokens | 6.4 ms, 4.9 GB/s (×128 = 0.82 s) | ~70+ ms, 0.4 GB/s | 14.2 ms, 2.2 GB/s | 1.6 ms, 19.4 GB/s |

- **Kaggle's P2P is ~12× slower than its SHM path,** which explains v4c. Keep the NCCL default on Kaggle.
- **Kaggle's default all-reduce is ~2× faster than the box's,** so communication is ~19% of a c24 decode step there (vs 40–52% on the box). The box overstates comm-bound effects.
- **Prefill is still communication-heavy on Kaggle:** ~0.8 s of all-reduce per 2,048-token chunk, a ceiling near 2.5k tok/s (v4b measured 1.4–2.2k). So pipeline parallelism's prefill gain should transfer, but its decode trade-off is less favourable than on the box.

### Pipeline parallelism (vLLM 0.23, FP8, NCCL default, L4 box) — rejected

| Layout | KV | c=8 | c=16 | c=32 | c=48 | Prefill (16 × 12.3k) |
|---|---|---|---|---|---|---|
| TP4 (v4b) | 1.59M | 131 | 171 | 256 | 302 | 916 tok/s |
| PP2 × TP2 (custom AR off) | 1.50M | 65 | 108 | 193 | 259 | **3,007 tok/s** |
| PP4 | 1.24M | 30 | 53 | 102 | 149 | 2,896 tok/s |

- **Prefill 3.2–3.3× faster, decode much slower.**
- Under PP4, per-request speed is flat at ~3.2 tok/s at every batch size: each decode step takes ~0.31 s regardless of load, so stages don't overlap across micro-batches in vLLM 0.23.
- Decode dominates tool traffic, so PP loses. A prefill/decode split (disaggregated serving) would be needed to use the prefill gain.

### vLLM 0.31 prefix caching: lower hit rate on tool traffic (cause open)

- **Observation:** the first real-traffic MTP2 probe on L4 (`p31_mtp2_c20`: vLLM 0.31, MTP2, concurrency 20), first ~65 min:
  - mean generation **147 tok/s**;
  - computed prompt tokens 412 tok/s;
  - prefix-cache hit rate **61.6%**;
  - MTP acceptance 2.3–2.5.
- **Comparison:** Kaggle v4b (vLLM 0.23), same window: generation 246 tok/s, computed prompt 220 tok/s, hits ~85–87%. The 0.31 run recomputed ~2× the prompt tokens, which swallowed MTP's per-step gain.
- **Retracted hypothesis (03:15 → 04:35 UTC):** I first blamed `mamba_cache_mode` defaulting to `"none"` in 0.31. That was wrong.
  - vLLM 0.31 logs `Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled`. My log search used 0.23's wording and missed it.
  - A re-run with explicit `--mamba-cache-mode align` (`p31_mtp2_align_c20`) gave the same numbers: hits 62.5%, generation 152 tok/s, prompt 402 tok/s, acceptance 2.37.
- **Open: why 0.31 hits less than 0.23 on this workload.** The same-version no-MTP probe (`p31_base_align_c24`, queued) will separate a vLLM-version effect from an MTP effect (e.g., speculative tokens making fewer blocks cacheable). E023 already saw worse cache hits on 0.31 without MTP.
- The RTX 6000 full evals (e6_base, e6_mtp2, e6_tq) used align by default: accuracy comparisons are valid; their low hit rates (44–61%) come from the same open issue.

### Full-eval results: MTP2 vs baseline on vLLM 0.31 (1× RTX PRO 6000 each, 172 pairs, 2026-10-17 04:36 UTC)

| Run | Score (120 tasks) | Pairs correct | Box tok/s | Concurrency | Per-attempt tok/s | Prefix-cache hits | Tokens per attempt | Cap hits |
|---|---|---|---|---|---|---|---|---|
| e6_base (no MTP) | **50.50 (42.1%)** | 70 | **362** | 32 | 12.96 | 58.5% | 37.0k | 82 |
| e6_mtp2 | 48.00 (40.0%) | 64 | 334 | 28 | 13.72 | 47.8% | 36.6k | 79 |

**MTP2 accuracy is a statistical tie:** paired MTP-only 17 vs baseline-only 23, McNemar p = 0.43, no extra tokens. The vLLM 0.23 MTP failure (14.6% vs 39.6% on the probe) does not reproduce on 0.31.

**MTP2 does not raise throughput on real tool traffic:**
- Per-attempt speed +6%, but the box is 8% slower overall: KV for the draft forces lower concurrency (28 vs 32), and prefix hits drop a further ~11 points.
- The L4 probe agrees: ~150 tok/s with MTP2 on 0.31, against v4b's ~200 on 0.23.

**vLLM 0.31's prefix caching is the bottleneck, with or without MTP.** The no-MTP baseline reaches only 58.5% hits, against ~86% on vLLM 0.23 for the same agent (E023 run B; Kaggle v4b 85.4%).

**Decision:** keep v4b on vLLM 0.23 without MTP. Revisit MTP on a vLLM version with both the MTP fix and 0.23-level prefix reuse, after finding what changed between 0.23 and 0.31.

### Tuned W8A8 block-FP8 Triton configs for NVIDIA L4 (vLLM 0.23, 2026-10-17 05:24–06:22 UTC)

- **Why:** on L4, vLLM 0.23 runs the official FP8 checkpoint's GEMMs with `_w8a8_triton_block_scaled_mm` and warns `Using default W8A8 Block FP8 kernel config ... Config file not found` for all five Qwen3.8-27B TP4 shapes: (N, K) = (3584, 5120), (4096, 5120), (5120, 1536), (5120, 4352), (8704, 5120). The same warning appears in Kaggle's v4b and v4c logs.
- **How:** vLLM's `benchmarks/kernels/benchmark_w8a8_block_fp8.py` (v0.23.0), with Qwen's shapes, BF16 output, batch sizes 1–2048, on 4 GPUs; 30 min. Config files: `fp8_configs_L4/`, installed into `vllm/model_executor/layers/quantization/utils/configs/`.
- **Bug in run 1:** all four GPU workers wrote the same per-shape files, so only the batch sizes of the last writer (256–2048) survived. Run 2 (`chain_retune.sh`) writes per-GPU folders and merges them.

| Decode tok/s | c=8 | c=16 | c=32 | c=48 | Prefill |
|---|---|---|---|---|---|
| untuned (default config) | 131 | 171 | 256 | 302 | 916 |
| run 1 (configs for M ≥ 256 only) | 133.5 (+2%) | 199 (+16%) | 283 (+11%) | 319 (+6%) | 877 |
| **run 2 (merged, M = 1–2048)** | **144 (+10%)** | **214 (+25%)** | **294 (+15%)** | **324 (+7%)** | 904 |

Tuning changes kernel tiling only, not the arithmetic, so accuracy is unaffected. The configs can ship in the Kaggle notebook (copied into vLLM's config folder before the server starts).

- **Run 2 merge bug:** the first merge of run 2 used `glob` on the file names, and their `[128,128]` part acts as a character class, so the merged files came out empty (vLLM then found configs and stopped warning, but used defaults). It was redone with explicit paths (`remerge.sh`). `fp8_configs_L4/` holds the complete files (15 batch sizes × 5 shapes).
- **Solver:** opt-in `ARC_FP8_CONFIGS=<dir>`. `install_fp8_configs()` runs right after the vLLM install and never fails the run. It was tested with a fake vLLM package and `fake_v3_test.py`.
- **Kaggle validation notebook v4d** (`kaggle/nb_probe_v4d`, settings `probe_v4d_settings.json`): v4b plus `ARC_FP8_CONFIGS` and a cell that writes the five configs. `check_notebook.py` (patched to this path) passes: NOTEBOOK CHECK OK. Not pushed to Kaggle yet: it needs ~2.3 GPU-h and the user's approval.

### TurboQuant `k8v4` full-eval accuracy (RTX box C, vLLM 0.31, concurrency 40; stopped at 129 of 172 pairs)

56 vs 57 correct on 129 shared pairs against e6_base (TurboQuant-only 17, baseline-only 18, p = 1.00); 35.0k vs 35.9k tokens per attempt. **No accuracy cost.** It was not pursued further, because it only mattered as KV room for MTP, and on Kaggle KV is not the binding constraint at concurrency 24. Box C was destroyed with partial results.

### Transcript compaction full eval (RTX box B, vLLM 0.31, concurrency 32, `ARC_COMPACT_AT=24000`, keep 2, tool outputs cut to 1,500 chars)

| Run | Score (120 tasks) | Pairs correct | Box tok/s | Prompt tokens (total) | Prefix hits | Tokens per attempt |
|---|---|---|---|---|---|---|
| e6_base | 50.50 | 70 | 362 | 69.2M | 58.5% | 37.0k |
| e6_compact | **44.50** | **60** | **329** | 56.5M (−18%) | 49.3% | 37.8k |

Paired: compact-only 16, baseline-only 26, McNemar p = 0.16. Compaction sent 18% fewer prompt tokens, but each compaction forces a full re-prefill (the hit rate drops), the model spends slightly more tokens, and accuracy trends down. **Rejected in this form.** The solver's attempt records do not carry the agent's `compactions` counter; the prompt-token drop shows it ran.

### vLLM 0.31 no-MTP tool probe on L4 (`p31_base_align_c24`, concurrency 24, NCCL default)

| L4 probe (48 pairs, 120 min) | Score | tok/s | Prefix hits |
|---|---|---|---|
| vLLM 0.23 v4b (E023 run B) | 13.33 / 35 | **201** | 86.7% |
| vLLM 0.31, no MTP | 8.83 / 35 | 154 | 70.9% |
| vLLM 0.31, MTP2 (c20) | 6.50 / 35 | 134 | 58.6% |

**vLLM 0.31 itself is ~23% slower than 0.23 on this tool traffic** (lower prefix reuse), and MTP2 lowers it further. Stay on vLLM 0.23. Finding the 0.23 → 0.31 caching regression is an open item for any later upgrade.

## Final status (2026-10-17 18:30 UTC)

- **Kept:** tuned FP8 GEMM configs (decode +7–25% on L4, accuracy-neutral) → Kaggle notebook v4d (`kaggle/nb_probe_v4d`), awaiting a Kaggle probe.
- **Rejected:** 4-bit weights, GGUF/llama.cpp, ternary Bonsai, Triton W8A8 instead of Marlin, async scheduling, bigger prefill chunks, NCCL P2P on Kaggle, pipeline parallelism, MTP2 on vLLM 0.31 (accuracy tie, slower on real traffic), MTP2 + TurboQuant (broken), transcript compaction (accuracy trend down, no throughput gain).
- **Neutral:** TurboQuant `k8v4` (accuracy tie; KV +26%, only useful if KV becomes binding).
- **Boxes:** RTX A/B/C/D destroyed; L4 box 55889595 stopped (not destroyed) at 18:25 UTC. It was idle ~9 h after the last probe (~$12).
