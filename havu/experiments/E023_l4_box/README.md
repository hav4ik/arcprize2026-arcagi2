# E023: peak capability of Qwen3.8-27B on Kaggle-equivalent hardware (Vast 4×L4)

Started 2026-10-14. Box: Vast instance 55889595, 4× L4 23 GB, driver 595.84 (CUDA 13.2), 102 vCPU, 128 GB RAM, 64 GB disk, $1.39/h. Image `vastai/sglang:v0.5.20-cuda-13.0`. Model: `Qwen/Qwen3.8-27B-FP8`, the same checkpoint as the Kaggle mirror `mikedan7/qwen3-8-27b-fp8-official`.

**Goal (user, 2026-10-14):** a harness better than v4b, maximum throughput on 4×L4, and an understanding of the model's peak capability under the Kaggle budget.

## Reference points

| Run | Where | Result |
|---|---|---|
| v2 submission (medium 48k, submit_answer, concurrency 35) | Kaggle LB | 11.94 |
| v3 submission (v2 agent, concurrency 29) | Kaggle LB | 18 |
| External "direct" notebook (no tools, xhigh, 98k thinking budget, MTP2, conc 16, 2 samples, sample-major) | Kaggle LB | 21, and 13 on a rerun with a small change |
| Probe v2 config, 48 eval pairs, 120 min | Kaggle 4×L4, vLLM 0.23 | 10.4% pass@1, 138 tok/s, KV peak 98%, cache hit collapsed |
| Probe v4b config (K3 recipe, conc 24, prior 250), same 48 pairs | Kaggle 4×L4, vLLM 0.23 | **39.6%** pass@1, 224 tok/s average (202 late in the run), KV 65%, cache hit 92–98% |
| Probe v4b + MTP2, same pairs | Kaggle 4×L4 | 14.6%, KV capacity 1.17M, cache collapsed |

v4b was submitted 2026-10-13 22:39 UTC (submission 57157836); its score is pending.

## Protocol

- The solver bundle from the v4 notebook (`kaggle/src/solver.py` + `evals/arc_harness`) runs in local mode with `ARC_LOCAL_SOURCE=eval`, on the same 48 pairs as the Kaggle probes (`$CLAUDE_JOB_DIR/tmp/probe_tasks.txt`, also in `probe_v4b_settings.json`), with `ARC_LOCAL_READY_MIN=120`, against an external server (`ARC_EXTERNAL_SERVER`).
- Throughput-only probes may use shorter windows.
- 48 pairs are noisy (±6–7 pp). Accuracy claims need larger gaps or repeats.

## Plan (overnight 2026-10-14)

Each run is the 48-pair, 120-min probe unless noted (~2.3 h, ~$3.2):

1. **A, SGLang 0.5.20:** tuned server (mem 0.88, triton, extra_buffer_lazy + skip-lock, 108 Mamba slots; 996k KV, 32 running), v4b settings at concurrency 24. Compare with Kaggle vLLM v4b.
2. **B, vLLM 0.23 cu130 on the box:** v4b settings, to calibrate box vs Kaggle (same stack). This also gives a second v4b accuracy sample.
3. **C, vLLM + MTP2 + `--language-model-only`** at concurrency ~18–20, so KV stays under ~70%.
4. **D, two TP2 servers (data parallel 2):** two solver shards of 24 pairs each, 12 concurrent per server. The total pairs-per-hour equals the rerun's, so per-pair budgets match. Tests whether PCIe all-reduce at TP4 costs throughput.
5. **E, harness:** if B–D raise throughput, try a 64k hard cap. Optionally run the external "direct" no-tool recipe on the same pairs to place its LB 21 / 13 on our scale.

**Launch gotcha:** start the solver with SIGINT reset to default. A background `&` launch in a non-interactive shell ignores SIGINT, and the executor's cell timeout (SIGINT) then kills the kernel instead of interrupting it, so the self-test fails and the solver falls back to the plain agent. Wrapper: `python -c "import signal,os,sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(...)"`. Kaggle runs in the foreground and is not affected. A robust fix for later: have the executor child restore `signal.default_int_handler` itself.

## Runs

### A. SGLang 0.5.20, v4b settings, concurrency 24 (08:38–10:33 UTC)

- **Server:** `--mem-fraction-static 0.88 --attention-backend triton --kv-cache-dtype fp8_e4m3 --context-length 131072 --max-running-requests 32 --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 108` + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`. This gives **996k KV tokens**, against 1.40M for vLLM. The template's defaults gave 390k KV and 14 running.
- **Throughput:** 228 tok/s average over the whole run, 250–255 tok/s in the first 40 min; per-attempt speed 10.0 tok/s.
- **KV and cache:** peak KV 81%, prefix-cache hit ~100% throughout, 0 retractions.
- **Result:** pass@1 **13/48 = 27.1%** (11.0/35 = 31.4% task-weighted). Mean 31.0k tokens per attempt; 28 output_budget stops, 20 submitted. Answer sources: program_pass 20, session_solve 7, none 21.
- **Comparison with vLLM v4b on Kaggle, same pairs:** 19/48 = 39.6%, 224 tok/s, 32.7k tokens. Throughput ties. Accuracy is 6 pairs lower, inside the noise for 48 pairs (±7 pp), but worth a repeat before adopting SGLang. Possible causes: FP8 KV/attention numerics (triton, e4m3) or sampling differences. Verdict: no throughput reason to switch to SGLang.

### C. vLLM 0.23 + MTP2 + `--language-model-only`, concurrency 20 (10:44–12:45 UTC)

- **KV capacity:** 1,289,804 tokens. MTP2 alone on Kaggle gave 1.17M, and no MTP 1.40M, so `--language-model-only` recovers ~120k.
- **MTP acceptance:** mean acceptance length 2.33–2.35; per-position rates 0.76 / 0.57–0.59.
- **Throughput:** ~211 tok/s average (1.517M tokens in 120 min), ~247 tok/s in the first 25 min. Per-attempt speed 11.7–12.4 tok/s, against 10.4 for v4b. Total throughput is the same as v4b with 4 fewer slots.
- **KV and cache:** KV stayed well under the collapse point; cache hit 78% overall; 0 preemptions.
- **Result:** pass@1 **8/48 = 16.7%** (6.5/35 = 18.6% task-weighted). Only 11 attempts submitted; 37 of 48 stopped at output_budget (v4b: 26 stopped, 22 submitted). Mean 31.6k tokens per attempt.
- **Interpretation:** this is the second MTP run with a very low score, after Kaggle v4b-mtp2 (14.6%), and this time the KV cache did not collapse. **MTP in vLLM 0.23 appears to hurt this model's output quality at temp 1.0 / top_p 0.95 / top_k 20.** Attempts meander and run out of budget instead of submitting. A plausible cause is the speculative-decoding rejection sampler not matching the top-k/top-p distribution exactly; not verified. Verdict: **no MTP for this model with vLLM 0.23**, whatever the KV headroom. The external notebook that uses MTP (LB 21/13) may be paying the same penalty.
- **Process note:** the box sat idle 12:45–16:00 UTC because the next run was not queued.

### B. vLLM 0.23 (Kaggle stack) v4b on the box, concurrency 24 (16:05–18:02)

- **Server:** KV 1,449,073 tokens.
- **Throughput:** ~201 tok/s average. The box runs about 10% slower than Kaggle: its L4s were at 75–83 °C with SM clocks of 1.7–2.0 GHz against a 2.04 GHz maximum (thermal throttling).
- **KV and cache:** peak KV ~63%, cache hit 94–98%.
- **Result:** **16/48 = 33.3%** pass@1 (13.33/35 = 38.1% task-weighted). Mean 29.4k tokens per attempt; 25 output_budget stops.
- **Pair-by-pair vs Kaggle v4b (19/48):** +6.2 pp [−9.1, +22.9], i.e. noise. Two runs of the same config differ by 3 pairs; their mean of 36.5% matches the pool's 37.5% on these pairs.
- **SGLang (A, 13/48) vs B:** −6.2 [−22.2, +10.0], a tie.
- **MTP vs v4b (Kaggle) by pair:** Kaggle mtp2 −25.0 [−40.5, −12.2]; box C −22.9 [−37.0, −10.2]. Both solve 0–1 pairs v4b missed. **MTP's quality loss is real.**

### Cache-consistency test (greedy cold vs warm), 18:03–18:30

- **Cold vs warm prompt:** identical on vLLM 0.23 no-MTP, vLLM 0.23 MTP2 and vLLM 0.31 MTP2.
- **The continuation check was flawed:** it split the output string mid-token, so the text was re-tokenized differently, and it diverged even without MTP. A token-level version (send token ids) is still needed to test the generated-block MTP bug.

### vLLM 0.31.0 (released after the user's remembered June fix; 0.23.0 is from 2026-06-15) + MTP2 + LMO + 8k batched tokens, concurrency 24 (18:27–20:28)

- **Server:** KV 1,229,845 tokens, less than 0.23's 1.29M.
- **KV and cache:** KV climbed to 78–89% and prefix-cache hits fell to ~0% from ~1 h in.
- **Throughput:** 120 tok/s average; only 11 of 48 attempts finished.
- **Interpretation:** another cache collapse. MTP needs more KV per agent, so the quality question on 0.31 is unanswered and needs concurrency ~16.

### Leaderboard update (2026-10-14)

v4b scored **19** (v3 18, v2 11.94; the external no-tool notebook 21/13), well below my estimate of 25–35. The ~+10 pp public-eval gain of K3 over the v2 agent did not show up on the hidden set. Likely causes:
- selection overfitting to the 120 public tasks;
- leaderboard noise (±4);
- possibly long-run effects that a 2 h probe can't show.

Proposed check: K0 vs K3 on 48 held-out ARC-AGI-2 *training* pairs.

### Queued (2026-10-14 ~20:40 UTC)

1. **Throughput, vLLM 0.31 no MTP, concurrency 24:** triton attention, 40 min.
2. **The same with the FlashInfer attention backend,** 40 min.
3. **No-tool (plain agent, xhigh, ARC_B_MAX1 98k, 2nd sample from leftover capacity), concurrency 16:** without MTP, and with MTP2 + LMO. Full 120-min probes. No-tool attempts don't reuse generated tokens through the prefix cache, so MTP's bug, if it is the cache-state one, should not apply.

### Results of the queue and later runs (2026-10-14 to 2026-10-16)

| Run | Result |
|---|---|
| vLLM 0.31, no MTP, c24 | ~185 tok/s with triton attention, ~180 with FlashInfer. Prefix-cache hits are worse than on 0.23, so **0.23 stays** |
| No tools + MTP2 + LMO, c16, 120-min probe | **14.6%** (48 pairs) at 167 tok/s. Not a candidate |
| K3 at the 340-input pairs/hour ratio | 12.5%, 18.9k tokens/attempt, 162 tok/s |
| K0 at the same ratio | 6.2% |
| FP8, no prefix cache, ~2.6k-token prompts, 3000 out (vLLM 0.23) | no MTP: 109 / 167 / 253 / 307 tok/s at 8 / 16 / 32 / 48 concurrent; MTP2: 139 / 258 / 334 at 8 / 16 / 32 (48 crashed) |
| INT4 (`RedHatAI/Qwen3.8-27B-INT4`, GPTQ W4A16), no MTP, ~8k prompts | 135 / 143 / 149 / 156 tok/s at 16 / 32 / 48 / 64. KV 1.94M tokens (vs 1.40M FP8), but **slower decode than FP8** |
| INT4, no MTP, ~99k prompts | 7.7 / 8.6 tok/s at 8 / 16 (prefill-dominated); 19 of 24 and 26 of 32 requests failed at 24 / 32 concurrent |
| INT4 + MTP2, ~8k prompts | 163 / 196 tok/s at 16 / 32; crashed at 48, so the long-context pass never ran (acceptance length 2.67) |

INT4 result files: `results/int4/`. **Verdict:** INT4 W4A16 fits more KV but decodes slower than FP8 (Marlin W8A16) on L4, so FP8 stays.

**Leaderboard:** v4b resubmitted unchanged scored **26** (first run 19). Run-to-run noise is about ±4.5–5 per submission; v4b's expected score is ~22–23.

**Box 55889595** was idle after 04:16 UTC on 2026-10-16. It was **stopped** (not destroyed; disk and models kept) at ~08:45 UTC.
