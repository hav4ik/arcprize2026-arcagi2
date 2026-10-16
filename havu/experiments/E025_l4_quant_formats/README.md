# E025: faster weight and KV formats for Qwen3.8-27B on 4×L4

Started 2026-10-16. The goal is more decode throughput (and KV room) on the Kaggle-like 4×L4 box than the FP8 checkpoint gives, without losing ARC accuracy.

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
