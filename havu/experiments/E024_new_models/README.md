# E024: new open models on ARC-AGI-2 (MiniCPM-V-4.7-35B-A3B, LFM2.5-24B-A2B, Nemotron-3.5-Bolt-74B-A9B-24GB)

Started 2026-10-16.

**Box:** Vast instance 56197430, 4× RTX PRO 6000 Blackwell Max-Q (96 GB each, SM120). PCIe 4.0, no NVLink; GPU3 sits on the other NUMA node. 48 vCPUs, 1 TB RAM, $5.33/h.

**Serving:**
- vLLM nightly `0.32.0rc1.dev277+g642c18d64` (transformers 5.19), in `/root/vllmn`. vLLM 0.31 lacks MiniCPM-V 4.7.
- **Data parallel, no TP:** one server per GPU on ports 8001–8004 (`/root/arc/serve_dp.sh`), each with `--max-model-len 131072 --max-num-seqs 32 --enable-prefix-caching`.
- **Client side:** the harnesses pin each attempt to one server by a hash of (task, pair, attempt), so tool turns keep their prefix cache. Local tunnels on 18101–18104.

**Harnesses:**
- **No tools:** `evals/run_openai_arcprize.py` (new). ARC Prize prompt and parser, one chat call, thinking on via the chat template, `max_tokens` 120000. It reads Kaggle's eval copy (172 pairs); score on the 167 pairs common with our reports.
- **Tools:** `evals/run_deepseek_tools.py` with the new `--base-url` (comma-separated, sticky per attempt) and `--extra-body`, using DeepSeek's sub80s40 config: grid mode, lean2sub, `submit_answer`, session final, lenient parse, 64k per turn, 80k total, 40k soft target, notes from 1%, warn at 75%.

**References (Qwen3.8-27B, full public eval, pass@1, Kaggle metric):** no tools xhigh 240k 35.9%; tools K3 (v4b recipe, 48k) 37.4%; x96pc8 (tools, 96k) 55.7%.

## Models

| Model | Format | Size | Parsers / template | Sampling used |
|---|---|---|---|---|
| openbmb/MiniCPM-V-4.7-35B-A3B-FP8 | FP8 compressed-tensors; text backbone qwen3_5_moe_text (Qwen3.6-35B-A3B-like hybrid), 256 experts | 38 GB | reasoning `qwen3`, tools `qwen3_coder`; thinking on by default; `--language-model-only` | temp 1.0, top_p 0.95, top_k 20 (generation_config) |
| LiquidAI/LFM2.5-24B-A2B-FP8 | FP8 128×128 blocks + dynamic activations; conv weights, routers, embeddings unquantized; 64 experts | 24.8 GB | `<think>` in the template (preserve_thinking); tools via `<\|tool_call_start\|>` pythonic | generation_config: temp 0.1, top_k 50, repetition 1.05 |
| nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB | see below | 24.1 GB | reasoning `ultra_v3` (plugin file from the repo), tools `qwen3_coder` | temp 1.0, top_p 0.95 |

### How Nemotron-3.5-Bolt fits in 24 GB (from the model card)

1. **Architecture compression ("Star Elastic"):** a budget-conditioned router picks routed-expert counts and MoE widths per layer to fit under 24 GB at batch 1 and 128K context. Expert top-K drops from 22 to 9; recovery training runs 120B tokens. The parent is Nemotron-3.5-Super-VL-120B-A12B, a Mamba-2 + attention + LatentMoE hybrid; the result has 74B total and ~9B active parameters.
2. **2-bit experts:** routed and shared experts are quantized to **IQ2-XXS (~2.06 bits per weight)**, with quantization-aware distillation in NVIDIA ModelOpt.
3. **Mixed precision elsewhere:** Mamba layers stay in **NVFP4**, embedding and output layers in **Q8_0**. SSM state and the Mamba cache are kept in bf16.
4. **Serving:** NVIDIA's `b12x` kernels (`pip install b12x`, `vllm download-kernels`, `--linear-backend b12x --moe-backend b12x`) target Blackwell SM120/121. The card also uses FlashInfer attention and the Mamba backend, `--mamba-cache-mode align`, and an optional LiLiCorr `dflash` drafter with 7 tokens.

Vendor benchmarks: GPQA-Diamond 86.4, HLE 25.3, Terminal-Bench 2.1 51.0. The card gives no comparison with BF16 Bolt itself.

## Runs

### MiniCPM-V-4.7-35B-A3B-FP8 (2026-10-16 04:50–05:30 UTC)

- **Servers:** 4 × 1 GPU, 2.59M KV tokens each.

| Mode (167 pairs, 1 attempt) | pass@1 | Mean output tokens | Notes |
|---|---|---|---|
| No tools, thinking on, max 120k | **1.2%** (2/167) | 21.2k | All 167 ended by stop; every answer parsed (114/172 had the correct output shape); median 47k characters of reasoning |
| Tools (sub80s40) | **3.6%** (6/167) | 16.8k | 165 submitted via `submit_answer`, 1 hit the 80k cap; 2,447 tool calls |

- **Verdict:** far below Qwen3.8-27B (35.9% / 37.4%), in the range of Qwen3.6-35B-A3B (6%, its backbone). The model's 4.7 additions are vision-side. Files: `evals/results/e024_minicpm_{notools,tools}.jsonl`.

### LFM2.5-24B-A2B-FP8 (2026-10-16 05:48–07:40 UTC)

- **Servers:** 4 × 1 GPU, 3.31M KV tokens each, ~1,000 tok/s generation per GPU at 29 running. Reasoning parser `deepseek_r1`, tools `lfm2`. Sampling from generation_config: temp 0.1, top_k 50, repetition 1.05.

| Mode | Done | pass@1 | Mean output tokens | Notes |
|---|---|---|---|---|
| No tools, max 120k | 166/172 at check | **~1.2%** (2/166) | 42.3k | |
| Tools (sub80s40) | stopped at 82/167 | **1.2%** (1/82) | 74.5k | Most attempts think to the 80k cap with only 2–5 tool calls. Stopped to save box time; the outcome was clear |

- **Verdict:** ~1% in both modes, far below Qwen3.8-27B. Files: `evals/results/e024_lfm_{notools,tools}.jsonl`.

### Nemotron-3.5-Bolt-74B-A9B-24GB (started 2026-10-16 08:01 UTC)

- **Install:** `pip install b12x` (1.5.1, adds only that package) and `vllm download-kernels` (flashinfer-jit-cache) on top of the vLLM nightly. The nightly already registers `NemotronH_Omni_Reasoning_V3`.
- **Launch:** NVIDIA's flags, minus vision (`--limit-mm-per-prompt '{"video":0,"image":0}'`) and the dflash drafter, plus batching; script `$JOB/box6000/bolt_serve2.sh`. KV is 10.0M tokens per GPU with prefix caching and 12.7M without.
- **Stability: the b12x stack is not stable under batched serving.** NVIDIA validates only `--max-num-seqs 1`.

| Try | Config | What happened |
|---|---|---|
| 1 | 16 seqs, prefix cache on | GPU0 and GPU3 segfaulted ~1 min in, during the first big prefill, right after the CuTeDSL JIT of `W4A16FusedMoeKernel` |
| 2 | Same, after JIT caches are warm | A synthetic stress test passed on all four (16 × 24.5k-token prompts, 2k out, ~290 tok/s per GPU incl. prefill). Real eval traffic then killed **all four** at once ~3.5 min in with `CUDA_ERROR_ILLEGAL_ADDRESS` |
| 3 | 16 seqs, **prefix cache off** | GPU0 and GPU1 segfaulted, GPU3 hit an illegal address; GPU2 survived (~770 tok/s at 16 running). Prefix caching is not the trigger |
| 4 | GPU2 at 16 seqs, GPUs 0/1/3 restarted at **4 seqs** | GPU2 crashed at 08:45 (30 min in). At 4 seqs, 3 of 4 crashed within ~15 min |
| 5 | **1 seq per GPU** (NVIDIA's validated setting), prefix cache off, supervisor restarts dead servers | stable from 09:11; ~140–147 tok/s per GPU (~575 tok/s per box) |

- The stress test differs from real traffic: it starts all requests together and they never finish early. Real traffic mixes new prefills into running decodes. So the likely trigger is the b12x kernels on mixed prefill/decode batches.
- **Throughput at 1 seq:** ~4 h for 167 no-tool attempts at ~50k tokens each. The dflash drafter (7 tokens) is NVIDIA's intended speed-up at batch 1 and is untried here.
- **Client:** `run_openai_arcprize.py --slots-per-server 2` (new) sends each no-tool attempt to the least-loaded server. With hash pinning and few workers, one GPU sat idle. A retry loop (`$JOB/box6000/bolt_client_loop.sh`) re-runs errored attempts.
- Error records from crashed tries: `evals/results/failed/e024_bolt_*`.

#### Bolt result (stopped 2026-10-16 10:49 UTC)

- **No tools, 1 seq per GPU, max 120k:** **0/30** answered pairs.
  - On the first 28 of these pairs: Qwen3.8-27B (no tools, xhigh 240k) averages **42.9%** per attempt; LFM2.5 got 2/28 and MiniCPM 1/28.
  - If Bolt's true pass@1 were 20%, P(0/30) = 0.8^30 ≈ 0.1%. It was stopped at the user's request.
- **Failure modes:** 9 of the first 25 hit the 120k cap with no answer, and finished attempts averaged ~89k tokens. Most finished answers have the wrong output shape; 2 had the right shape with ≥99% of cells right.
- **Tools run:** not done.
- **Caveat:** whether the IQ2 (2-bit) compression or the model itself is to blame can't be separated here.
- **Files:** `evals/results/e024_bolt_notools.jsonl` (error records from the crashed tries are in `evals/results/failed/`); server, supervisor and stress logs in `box_logs/bolt_logs.tgz`.

## Conclusion

None of the three is useful for ARC-AGI-2: all are at the floor (0–4% pass@1), versus 36–37% for Qwen3.8-27B. Qwen3.8-27B stays the model.

Box 56197430 was destroyed on 2026-10-16 at ~10:52 UTC, after ~6.3 h (~$34).
