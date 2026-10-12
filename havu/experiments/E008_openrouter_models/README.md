# E008: Other models via OpenRouter, no tools (E001d setup)

## Setup
- **Script:** `evals/run_openrouter_arcprize.py`. ARC Prize prompt and parser, one call per attempt, same record format as E001d.
- **Router:**
  - Routes between a model's free and paid endpoints: free first, adaptive capacity (AIMD), with the floor and gentle back-off added 2026-10-10.
  - A paid-spend cap, using OpenRouter's reported per-call cost.
  - Streaming calls; full reasoning saved per attempt.
- **Paid endpoints are pinned to one full-precision provider,** because providers differ in quantization and output caps (some cap at 32k).
- **Scoring:** ARC Prize task-weighted, on the official public eval (120 tasks, 167 pairs).

## Free vs paid endpoints (Nemotron 3.5 Lightning, 2026-10-10)
- **Free (`:free`):** NVIDIA's endpoint, nvfp4 quantization, 65k output cap.
  - Slow: about 14–19 tokens per second.
  - Non-streaming requests were cut at 600 s.
  - Not used for the evaluation (user decision: paid only).
- **Paid, pinned to CoreWeave (bf16):** about 420 tokens per second per stream. It throttles (HTTP 429) at about 10–14 concurrent requests.

## Results

| Model | Provider (precision) | Effort | pass@1 | Output tokens / attempt | Cost |
|---|---|---|---|---|---|
| DeepSeek V4.1-Flash (E001d, 3 samples) | DeepSeek API | max | 59.9% | 55.7k | ~$10 per sample |
| **nvidia/nemotron-3.5-lightning** | CoreWeave (bf16) | high | **0.0% (0/167)** | 68.0k (median 64.9k) | $2.35 |
| qwen/qwen3.8-27b, pilot (first 15 tasks, 26 pairs) | DeepInfra (bf16) | high | 20.0% (6/26); Flash 57.4% on the same tasks | 35.2k | $1.75 |

**Nemotron 3.5 Lightning:**
- Every one of 167 attempts ended in a well-formed answer: 164 stopped normally, 3 hit the cap, and only 4 had no parseable grid.
- Only 28/167 had the right output size, even on tasks Flash always solves.
- This is a capability gap, not a harness problem. A second attempt (pass@2) was skipped as not worth the cost.
- **Parser re-check (2026-10-10):** re-parsing all 167 answers reproduces every stored flag. No JSON grid in any answer or reasoning trace equals the solution. Answers: 87 wrong shape, 52 jagged, 24 right shape but wrong cells, 4 unparseable.
- **Manual trace reading (2026-10-10).** Four sub-agents each read 10 traces, grouped by failure type; notes and quotes are in `nemotron_traces/notes_{A,B,C,D}.md`. Across all 40:
  - **Correct rule found: 0/40.** At most a partial rule in about 10/40.
  - **Every answer is an admitted guess or give-up: 40/40.** Fallbacks include copying the test input, a constant fill, or a half-rule ("I give up", "I surrender. I'll output all zeros").
  - **Invented time pressure: nearly all.** "Given the time…" or "out of time" appears 4–14× per trace, although the prompt sets no limit. It quits at 50–95k tokens of a 200k budget.
  - **Grid sizes misread from the prompt: about 30/40.** For example, 27×27 read as 25×25. Miscounts are never re-checked, and they derail its comparisons.
  - **Writes grids that don't match its own plan.** Copies drop 3–29% of cells or whole rows; this is the source of the jagged answers. Even with perfect copying, 0/10 of the jagged sample would be correct.
  - **Shallow search.** 30–270 "maybe…" ideas per trace, often arithmetic formulas on rows, each tried on one example and dropped. It never checks a rule against all training pairs before answering.
  - **Verdict:** this is a capability gap. Format fixes or answering through code would produce well-formed wrong answers. A Python tool might fix perception for 1–2 near misses (28a6681f, dd6b8c4b), but not rule discovery.

**Qwen 3.8 27B pilot (2026-10-10):**
- **14/26 attempts ended with empty content.** The reasoning stops mid-sentence, yet the call reports `stop`.
  - The cut-offs vary widely: 22–55k tokens and 353–1075 s. That overlaps the attempts that answered, so it is not a fixed limit.
  - The cause is unknown: either provider-side truncation (DeepInfra) or the model stopping mid-thought.
- **Of the 12 attempts that answered, 6 were correct (50%).** The 20% is therefore likely understated.
- **Speed and cost:** about 55 tokens/s and $0.067 per attempt.
- **Next:** a diagnostic re-run of a few cut-off pairs on another full-length provider (CoreWeave), then Qwen's recommended sampling settings if needed.

**Qwen 3.8 27B provider diagnostics (2026-10-10).** The re-run covered 5 pairs that DeepInfra had cut off (0934a4d8, 136b0064, 16b78196, 13e47133 ×2).

| Provider | Thinking allowed | Result | Cost |
|---|---|---|---|
| DeepInfra (bf16), effort=high | long (22–55k observed) | 14/26 pilot attempts cut off mid-reasoning (empty content, finish `stop`) | $1.75 |
| Reka (fp8), effort=high | **2,047 tokens**, fixed | 0/5, short answers | $0.04 |
| Reka, `reasoning.max_tokens=150000` | still ~2k (budget ignored) | 0/1 | $0.004 |
| Alibaba, `reasoning.max_tokens=100000` | **16,384 tokens**, fixed (budget ignored) | 0/5; 2 unparsed (answered before finishing) | $0.29 |

**Conclusion so far:** OpenRouter providers map Qwen's thinking very differently, and none tried gives a clean full-length evaluation. Next candidates: CoreWeave, Parasail or Mancer (fp8, 236k caps), or self-hosting with vLLM. `--reasoning-tokens` was added to the script for explicit thinking budgets.

## Nemotron 3 Super 120B-A12B pilot on OpenRouter (2026-10-10, incomplete)

- **Settings:** `nvidia/nemotron-3-super-120b-a12b` via DekaLLM (fp8; OpenRouter marked it "degraded"). Effort high, model-card sampling (temperature 1.0, top_p 0.95, added as `--temperature`/`--top-p`), 200k max tokens, first 15 tasks (26 pairs), 1 attempt.
- **Other providers:** DeepInfra caps output at 16k, which is unusable; the only other is NVIDIA's rate-limited free endpoint.
- **Result: inconclusive. 21/26 attempts failed on the provider side** ("Network connection lost" or DekaLLM upstream connection errors), typically on 15–30-minute streams.
- **The 5 that completed were all wrong:** 1ae2feb7 ×2, 0934a4d8, 1818057f, 16b78196. They used 19–87k tokens and stopped normally, at ~45 tok/s.
- **Cost:** $0.11 recorded.
- **Next:** self-host on the Vast 4×RTX 6000 box and use the generation-API harness, instead of retrying on DekaLLM.

## Qwen3.8-27B NVFP4 on 4×RTX 6000, generation API (2026-10-10)

**Setup**
- **Server:** `nvidia/Qwen3.8-27B-NVFP4` on a rented vast.ai box (`ENDPOINTS.yaml` entry `vastai-4xrtx6000-qwen27b`). SGLang 0.5.20, TP4, 262k context, radix cache on. API cost $0 (box rental only).
- **Script:** `evals/run_qwen_generate.py`. Tests: `evals/tests/test_qwen_generate.py`. Stats and the Flash comparison: `evals/qwen_gen_stats.py`.
- **Prompt and scoring:** the ARC Prize prompt and parser, task-weighted, as in E001d.
- **Chat template:** rendered by hand and sent to `/v1/completions`. A `--verify-template` check found `prompt_tokens` identical to `/v1/chat/completions` with `chat_template_kwargs.reasoning_effort` for 2 tasks × 3 tiers (xhigh, medium, low).
- **Run settings:**
  - Tier `medium` (no system message), temperature 0.6, top_p 0.95, top_k 20.
  - 48k generated-token budget per attempt, sent in chunks of at most 16k; each chunk continues the text so far verbatim.
  - **Mid-thought EOS:** the attempt is continued. After 2 consecutive continuations that add fewer than 20 tokens, the model is forced to answer.
  - **Budget exhausted while thinking:** the harness appends `I have to give my final answer now.\n</think>` and allows 4k more tokens (forced final).
- **Run size:** 2 attempts × 167 pairs at concurrency 64.
- **Results:** `evals/results/qwen_nvfp4_gen_medium_48k.jsonl`; gzipped full traces in `..._traces/`. Smoke run: `..._smoke.jsonl`.

**Results**

| Run | Pairs | pass@1 | pass@2 | Tokens / attempt (mean / median) |
|---|---|---|---|---|
| Qwen NVFP4, medium, 48k, pilot (first 15 tasks) | 26 | 12.8% | 14.4% | 40.0k / 46.8k |
| Qwen NVFP4, medium, 48k, full | 167 | **9.7%** (9.7%, 9.6%) | **13.3%** | 39.1k / 44.8k |
| Qwen via OpenRouter DeepInfra (bf16), high, pilot, 1 attempt | 26 | 20.0% | – | 35.2k |
| Flash E001d (max), pilot tasks | 26 | 57.4% | 80.0% (a0+a1); 72.6% (mean over attempt pairs) | – |
| Flash E001d (max), full | 167 | 59.9% | 79.3% (a0+a1); 74.6% (mean over attempt pairs) | 55.7k / 49.2k |

- **Paired pass@2 against Flash a0+a1 (pairs):**
  - Full set: both 23, only Qwen **0**, only Flash 108, neither 36 (of 167).
  - Pilot: both 4, only Qwen 0, only Flash 16, neither 6 (of 26).
- **Attempt-level numbers:**
  - 32 of 334 attempts were correct, and 96.7% gave a parseable grid. There were no errors and no transient HTTP failures.
  - 46.7% of attempts closed `</think>` on their own; 19.2% of those (30/156) were correct.
  - 53.3% were forced to answer: 139 ran out of budget and 39 hit repeated EOS. Only 2/178 forced answers were correct, so forced answers are essentially guesses.
- **Mid-thought EOS is frequent and sticky:**
  - 142/334 attempts (42.5%) emitted EOS while still thinking, with 360 events in total. The EOS often lands mid-grid-row or mid-word.
  - A continuation often yields exactly 1 token and then EOS again; 100 of the 360 continuation chunks were under 20 tokens.
  - The first EOS never came before about 17k tokens of total context (median about 36k, prompt included). This hints at long-context degradation; NVFP4 quantization is a possible cause, but this is not established.
  - The DeepInfra (bf16) pilot showed the same symptom: a mid-sentence `stop` with empty content. So it is model behaviour, not provider truncation. It may be made worse by quantization.
  - Of the 142 attempts with EOS, 33 later closed thinking naturally (4 correct), and 109 ended forced.
- **Throughput:** 13.07M generated tokens in 115.8 min, averaging 1,881 tok/s at 64 concurrent requests. Each attempt took 20.8 min on average (max 28.5).

**Conclusion:** at medium effort with a 48k budget, the NVFP4 model is far behind Flash (9.7% vs 59.9% pass@1) and solves no pair that Flash misses. Accuracy is limited by two things: more than half the attempts need more than 48k tokens to finish thinking, and frequent mid-thought EOS. Possible next steps:
- A larger budget (96k or more) with a subset of the medium tier.
- A bf16/fp8 checkpoint, to test whether the EOS rate is caused by quantization.

### Official settings (xhigh, temp 1.0, top_p 0.95, top_k 20, 240k budget) (2026-10-10/11)

**Settings**
- Same server and harness (`evals/run_qwen_generate.py`, `qwen-gen-1`). Tier `xhigh`, temperature 1.0, top_p 0.95, top_k 20 (Qwen's recommended thinking settings).
- 240k generated-token budget per attempt, in chunks of at most 32k; 4k forced-final allowance. 2 attempts × 167 pairs at concurrency 48.
- Command: `run_qwen_generate.py --tier xhigh --temperature 1.0 --budget 240000 --chunk-tokens 32768 --attempts 2 --concurrency 48 --tag t1`.
- Results: `evals/results/qwen_nvfp4_gen_xhigh_240k_t1.jsonl` (334 records; the first 52 were copied from the pilot `..._t1_pilot.jsonl`); traces in `..._t1_traces/` and `..._t1_pilot_traces/`. Logs: `log_qwen_gen_xhigh_240k_t1{_pilot,}.txt`.

**Pilot (first 15 tasks, 26 pairs):** pass@1 40.0%, pass@2 46.7%, 68.5k tokens/attempt, 0 EOS-continuations, 0 forced finals.

**Results**

| Run | Pairs | pass@1 task-w (pair-w) | pass@2 task-w (pair-w) | Tokens / attempt mean / median / p90 / max |
|---|---|---|---|---|
| Qwen NVFP4, xhigh, 240k, temp 1.0, pilot | 26 | 40.0% (42.3%) | 46.7% (50.0%) | 68.5k / 63.2k / 109.5k / 157.2k |
| Qwen NVFP4, xhigh, 240k, temp 1.0, full | 167 | **36.4%** (35.9%); per attempt 35.8%, 36.9% | **48.2%** (47.9%; 80 pairs) | 73.6k / 67.5k / 119.9k / 241.2k |
| Qwen NVFP4, medium, 48k, temp 0.6, full | 167 | 9.7% (9.6%) | 13.3% (13.8%) | 39.1k / 44.8k / 50.6k / 52.1k |
| Flash E001d (max), full | 167 | 59.9% | 79.3% (a0+a1); 74.6% (mean over attempt pairs) | 55.7k / 49.2k |
| ARC Prize reported, Qwen3.8-27B | semi-private | – | 42.4% (verified) | – |

- **Health of the run:** 0 errors, 0 transient failures, 100% of attempts gave a parseable grid. 331/334 (99.1%) closed `</think>` on their own; 3 were forced at the 240k budget (1 correct). **No mid-thought EOS at all** (0 continuations), against 42.5% of attempts at medium/temp 0.6. Changing the settings (tier, temperature 1.0, 32k chunks) removed the EOS problem on the same NVFP4 checkpoint, so quantization is not its main cause. Which setting matters was not isolated.
- **By whether Flash solves the pair (Flash a0+a1 pass@2):**
  - Flash solves (131 pairs): Qwen pass@1 45.0%, pass@2 60.3% (79).
  - Flash fails (36 pairs): Qwen pass@1 2.8%, pass@2 2.8% (1: `dd6b8c4b` pair 0, which Flash attempt 2 solves). Against any of Flash's 3 attempts, Qwen solves nothing that Flash misses.
  - Paired pass@2: both 79, only Qwen 1, only Flash 52, neither 35.
- **vs ARC Prize's 42.4%:** our public-eval pass@2 of 48.2% is in the same range. The sets differ (semi-private vs public eval), so this is not a like-for-like comparison, but there is no sign that NVFP4 or our harness loses much accuracy.
- **Accuracy vs token cap** (an attempt counts only if it was correct and finished within the cap; forced answers are close to guesses, see the medium run):

  | Cap | Attempts finished within cap | pass@1 | pass@2 |
  |---|---|---|---|
  | 28k | 8% | 6.9% | 10.8% |
  | 48k | 30% | 21.2% | 29.0% |
  | 64k | 48% | 28.5% | 36.9% |
  | 96k | 74% | 34.5% | 46.1% |
  | 128k | 93% | 35.8% | 47.4% |
  | 240k | 100% | 36.4% | 48.2% |

- **Throughput and wall time:** the full run generated 21.0M tokens for 282 attempts in 251.1 min, averaging 1,395 tok/s at 48 concurrent requests. The server reported about 1.4–1.9k tok/s in steady state. The pilot generated 3.56M tokens in 48.9 min. Each attempt took 36.9 min on average (median 34.3, p90 60.4, max 132.9). The tail was long: the last 40 attempts took about 40 min with fewer than 48 requests in flight.
- **Rental cost:** about $5.42/h × 5.0 h (pilot 0.8 h + full 4.2 h) ≈ **$27** (≈ $23 for the full-run part). API cost $0.
- **Kaggle implication:** Kaggle has 240 tasks / 259 pairs, 4×L4 and 12 h. At a projected ~330 tok/s on 4×L4, 12 h gives about 14.3M tokens, or **~27k tokens per attempt** at pass@2 (518 attempts). This setting needs 73.6k on average (≈ 38M tokens, ~32 h at 330 tok/s). At a ~27k cap only 8% of attempts finish thinking, and pass@2 drops to about 11%. One attempt per pair (~55k each) would give a pass@1 of about 25%. So this model can be competitive on Kaggle only with much faster inference or a much shorter reasoning mode.

## Qwen3.6-35B-A3B NVFP4 on 4×RTX PRO 6000 Server (2026-10-11)

**Setup**
- **Server:** `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE, about 3B active parameters, hybrid linear/full attention) on a rented vast.ai box, instance 55388103. The box has 4× RTX PRO 6000 Blackwell Server and costs $5.93/h. `ENDPOINTS.yaml` entry: `vastai-4xrtx6000s-qwen36-35b-a3b`. SGLang 0.5.20, TP4.
- **MoE backend fix:** SGLang's default NVFP4 MoE runner crashed. The server was restarted with `--moe-runner-backend flashinfer_cutlass`, set in `/etc/sglang-args.conf` on the box.
- **Harness:** `evals/run_qwen_generate.py` (`qwen-gen-1`), with the ARC Prize prompt and parser, task-weighted as in E001d.
- **Sampling:** Qwen3.6 has no effort tiers, so `--tier none` with thinking on. Model-card sampling: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5.
- **Budget:** 240k tokens in chunks of at most 32k, plus a 4k forced-final allowance. 2 attempts × 167 pairs.
- **Command:** `run_qwen_generate.py --endpoint vastai-4xrtx6000s-qwen36-35b-a3b --model nvidia/Qwen3.6-35B-A3B-NVFP4 --tier none --temperature 1.0 --presence-penalty 1.5 --budget 240000 --chunk-tokens 32768 --attempts 2 --concurrency 128 --tag t1pp15`.
- **Concurrency restart:** the run started at concurrency 64 and decoded only about 1,170 tok/s. After 30 min (14 attempts done) it was stopped and resumed at 128, where the server decoded about 3,600 tok/s at first. KV usage was 19% at that point and peaked at 74%; there were no retractions and no queue.
- **Results:** `evals/results/nvidia-qwen3-6-35b-a3b-nvfp4_gen_none_240k_t1pp15.jsonl` (334 records), with gzipped traces in `..._t1pp15_traces/`. Logs: `log_qwen36_gen_none_240k_t1pp15{_c64,}.txt`. Answer-format audit: `qwen_format_audit.md`.

**Results** (task-weighted, with pair-weighted in brackets)

| Run | Pairs | pass@1 | pass@2 | Tokens / attempt mean / median / p90 / max |
|---|---|---|---|---|
| Qwen3.6-35B-A3B NVFP4, pilot (first 15 tasks) | 26 | 10.0% (per attempt 6.7%, 13.3%) | 13.3% | 47.9k / 44.8k / – / – |
| Qwen3.6-35B-A3B NVFP4, full | 167 | **6.2%** (6.0%); per attempt 6.2%, 6.2% | **7.5%** (7.2%; 12 pairs) | 53.0k / 56.6k / 81.3k / 111.2k |
| Qwen3.8-27B NVFP4, xhigh, 240k, temp 1.0, full | 167 | 36.4% (35.9%) | 48.2% (47.9%; 80 pairs) | 73.6k / 67.5k / 119.9k / 241.2k |
| Flash E001d (max), full | 167 | 59.9% | 79.3% (a0+a1); 74.6% (mean over attempt pairs) | 55.7k / 49.2k |

- **Health of the run:** clean. 0 errors, 0 transient failures, 0 mid-thought EOS continuations and 0 forced finals. Every attempt closed `</think>` on its own, and the longest used 111k of the 240k budget. 333/334 attempts gave a parseable grid; the exception is `446ef5d2` pair 1 attempt 0. Only 20/334 attempts were correct.
- **By whether Flash solves the pair (Flash a0+a1 pass@2):**
  - Flash solves (131 pairs): Qwen3.6 pass@1 7.6%, pass@2 9.2% (12 pairs).
  - Flash fails (36 pairs): 0%.
  - Paired pass@2: both 12, only Qwen3.6 0, only Flash 119, neither 36. Qwen3.6 also solves nothing that any of Flash's 3 attempts misses.
- **Accuracy vs token cap** (an attempt counts only if it was correct and finished within the cap):

  | Cap | Attempts finished within cap | pass@1 | pass@2 |
  |---|---|---|---|
  | 28k | 23% | 1.3% | 1.8% |
  | 48k | 38% | 4.0% | 5.4% |
  | 64k | 63% | 6.2% | 7.5% |
  | 96k | 98% | 6.2% | 7.5% |
  | 240k | 100% | 6.2% | 7.5% |

- **Official vs lenient ceilings** (offline, from the saved answers and traces; the rules are cumulative):
  - L0 (official): exact match.
  - L1: also counts an answer that equals the solution after inserting one missing row or deleting one extra row.
  - L2: also counts attempts where the solution grid appears verbatim in the last 20% of the thinking.
  - L3 (*optimistic, not formatting*): also counts answers with at most 2 wrong cells after the best one-row alignment.

  | Level | Qwen3.6 attempts correct | Qwen3.6 pass@1 / pass@2 | Qwen3.8 xhigh attempts correct | Qwen3.8 pass@1 / pass@2 |
  |---|---|---|---|---|
  | L0 official | 20 | 6.2% / 7.5% | 120 | 36.4% / 48.2% |
  | L1 + one-row transcription | 21 | 6.5% / 7.9% | 125 | 38.1% / 51.1% |
  | L2 + solution in late thinking | 21 (+0) | 6.5% / 7.9% | 125 (+0) | 38.1% / 51.1% |
  | L3 + ≤2 cells off (optimistic) | 28 | 8.7% / 11.1% | 143 | 43.4% / 56.5% |

  - **L1 recoveries for Qwen3.6:** `3e6067c3` p1 a1, which is missing row 22 of 30.
  - **L1 recoveries for Qwen3.8:**
    - `3e6067c3` p0 a0: missing row 24 of 29.
    - `64efde09` p0 a1: missing row 17 of 29.
    - `80a900e0` p0 a1: missing row 21 of 22.
    - `800d221b` p0 a1: missing row 27 of 30.
    - `cb2d8a2c` p1 a1: one duplicated row at 18.
  - **L2:** 0 for both models. An incorrect attempt never had the solution in its late thinking, which agrees with `qwen_format_audit.md`.
  - **Takeaway:** formatting costs Qwen3.8 about 3 points of pass@2 (dropped rows in large grids) and Qwen3.6 under 1 point. L3 is near-miss reasoning, not formatting.
- **Throughput and wall time:**
  - At concurrency 128 the run generated 17.4M tokens for 320 attempts in 132.9 min. That averages 2,181 tok/s measured by the harness; the server reported 3.6k tok/s at first and 2.5–2.9k tok/s in steady state.
  - **Tail slowdown:** throughput fell abruptly at about 93 requests in flight. It went from about 2.4k to about 1.3k tok/s, then to about 1.0k at 60–80 in flight with about 55–60k context per request. The GPUs stayed at 100% utilisation and about 196 W, with no throttling. Long-context attention seems to dominate decode once contexts are long.
  - Each attempt took 42.6 min on average (median 45.7, p90 67.2, max 94.5).
  - The whole eval took about 164 min of wall time: 30 min at concurrency 64 plus 133 min at 128.
  - Compared with Qwen3.8-27B (1,395 tok/s at concurrency 48 on 4×RTX PRO 6000 WS), the MoE decodes about 1.6× faster on average and about 2.6× faster at its peak.
- **Rental cost:** the box has been up since 12:21 UTC. That is about 3.25 h × $5.93 ≈ **$19** up to the end of the run at 15:33 UTC, including about 0.5 h of server setup and the MoE-backend fix. API cost is $0.
- **Kaggle implication:** with about 27k tokens per attempt at pass@2 on 4×L4, Qwen3.6 would score about 2% pass@2. Even unlimited budget only reaches 7.5%, since every attempt finished by 111k. Faster decoding does not make up for the accuracy gap; Qwen3.8-27B is 48.2% at full budget and about 11% at 27k. Qwen3.6-35B-A3B is not a viable Kaggle base model for ARC-AGI-2 without fine-tuning.
