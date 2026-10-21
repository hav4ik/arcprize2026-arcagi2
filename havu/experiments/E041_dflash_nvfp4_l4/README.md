# E041: NVFP4 / INT4 / FP8 with the DFlash 2 draft on 4×L4 (vLLM 0.32) — stopped, box handed over

Run 2026-10-21 on Vast 4×L4 (55889595) after a report that "NVFP4 + DFlash gets 500 tps at long context on 4×L4".
**Status: stopped 2026-10-21 23:30 UTC; the box was handed to session dsv4-infer at the user's request.** Only the NVFP4 phase finished; FP8+DFlash, FP8 on 0.32 and INT4 AutoRound were not run.

## Setup
- vLLM 0.32.0 (torch 2.13 cu130) in `/root/vllm32` (deleted at handover). TP4, fp8 KV, TRITON_ATTN, prefix caching, max-num-seqs 64, max-model-len 131072, `--language-model-only`. Kaggle-style flags: no tool or reasoning parsers (E036).
- Target `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt MIXED_PRECISION). On SM89 vLLM uses `MarlinNvFp4LinearKernel` (weight-only FP4, "may degrade performance") for the FP4 layers and CUTLASS FP8 for the rest. KV 1.85–1.87M tokens.
- Draft `z-lab/Qwen3.8-27B-DFlash2` (2B, BF16): `--speculative-config '{"method": "dflash", "model": ..., "num_speculative_tokens": 7}'`. It needs `--gpu-memory-utilization 0.88` (OOM at 0.95). KV drops to 812k tokens.
- Probe: the 48-pair Kaggle-pace probe (35 tasks, v4b agent, forced at server-ready + 120 min, c24), as in E021/E023/E027.

## Results
| Config | Probe (48 pairs) | Real-traffic gen | Notes |
|---|---|---|---|
| **NVFP4** (0.32) | **9/48** (6.50/35 tasks) | 178 tok/s over the run; 293 tok/s at ≥20 running in the first 8 min | 19 answered (program_pass 18, session_solve 1), cache hit 81% |
| FP8 v4b (0.23, E023 B, reference) | 16/48 (13.33/35) | 201 tok/s | |
| base v4b on Kaggle (E021, reference) | 19/48 | — | |

Paired with base v4b on Kaggle: both 7, NVFP4-only 2, base-only 12.

Long-context decode bench (`bench_decode2.py ... 8000 1000 1 8 24`, i.e. ~24k-token prompts, 1000 output tokens; tok/s includes the prefill of all prompts, so it understates decode):

| Concurrency | NVFP4 | NVFP4 + DFlash |
|---|---|---|
| 1 | 17.7 tok/s | **23.0 (+30%)** |
| 8 | 30.2 | 31.2 |
| 24 | 32.5 | 32.7 |

DFlash: draft acceptance 63.8%, "accepted throughput" 240 tok/s in the server's spec-decode log. Greedy output identical to plain NVFP4 (lossless, as documented).

(A first bench with parameter 24000 made 74k-token prompts and hit its 50-min timeout; `bench_decode2.py`'s length unit gives ~3× tokens.)

## Findings
- **NVFP4 loses accuracy like the other 4-bit formats** (E027: INT4 GPTQ 9/48, AWQ 10/48): 9/48 vs 16/48 for FP8 on the same box. On L4 there are no FP4 tensor cores, so it runs weight-only through Marlin and is not faster on real traffic (178 vs 201 tok/s).
- **DFlash helps a single stream (+30%) but not at Kaggle's concurrency** (c8 +3%, c24 +1%), consistent with its published H200 curve (3.4× at c1, 1.0–1.45× at c32). It also halves KV capacity. The "500 tps" claim does not reproduce with NVFP4 at c24.
- Not run (handed over): FP8 + DFlash (the only combination that keeps FP8 accuracy), FP8 on 0.32, INT4 AutoRound (`CySpiegel/Qwen3.8-27B-Int4-AutoRound`). Scripts: `scripts/e037_chain_v3.sh` runs exactly these three probes.
- Process note: an 8.7-h gap (14:38–23:22 UTC) came from uploading v3 without the executable bit; the watcher only looked for success markers. Watchers now also check that the chain process is alive.

Files: `results/` (chain.log, per-config server logs, bench JSONs, greedy checks, the NVFP4 probe's summary and attempts), `scripts/`.
