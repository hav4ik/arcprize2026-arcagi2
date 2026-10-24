# E028 report: LoRA SFT of Qwen3.8-27B on DeepSeek-V4.1-Flash tool-use rollouts

**Dates:** 2026-10-16 to 2026-10-18. **Status:** trained, exported, quantized, evaluated and audited. A Kaggle
notebook (v2) is ready as a submission experiment, but a submission-pace test (§6.3) puts it well below base. **Detailed lab notes:** `README.md` in this folder.

## TL;DR

**What we built.** We fine-tuned Qwen3.8-27B (LoRA rank 128, Tinker, Dimuon) on 2,500 multi-turn ARC tool-use rollouts
by DeepSeek-V4.1-Flash. That is 70.9M tokens, 44.5M of them trained (assistant turns only), with correct and
incorrect rollouts both included.

**The pipeline works end to end.**
- Training NLL on unseen batches fell from 0.764 to 0.688.
- The merged BF16, ModelOpt FP8 and official-layout block-FP8 exports all match Tinker within ±0.002 trained-token
  NLL, including at 33–62.5k tokens of context.
- Kaggle notebooks for both FP8 variants passed their commit runs offline, and the right model was verified byte for
  byte.

**The model doesn't beat base yet.**
- On dev56, running the teacher's own harness and limits, it scores **21/56 (37.5%) pass@1**. Base Qwen3.8-27B scores
  22/56 (39.3%) and the DeepSeek teacher 39/56 (69.6%).
- SFT − base = −1.8 pp, 95% CI [−16, +12.5], so the difference is inconclusive.
- The reasoning is healthy and teacher-like, but the model uses more tokens (60k vs 55k for base and 39k for the
  teacher) and often runs to the cap without answering.

**At Kaggle pace it is clearly worse than base** (§6.3). The exact v1 submission setup on our 4×L4, paced like Kaggle,
scored **23/93 pairs (24.7%)** on the 93 non-dev eval pairs. Base with the v4b solver, run the same way, scored
**35/93 (37.6%)** and **38/93 (40.9%)** in two samples: paired SFT − mean base = **−14.5 pp, 95% CI [−22.7, −6.5]**. The adaptive per-attempt budget was a
median 28.8k: 70 of 93 SFT attempts hit it, and only 1 of those was correct. When the model does submit, it is right
(22/23).

**Caveat (2026-10-19, E040):** the SFT run's vLLM server had `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`. The Kaggle notebooks don't pass these
flags, and the base v4b runs didn't use them. On the same weights and agent they cut throughput from about 192 to
143 tok/s, so the SFT attempts got fewer tokens than they would on Kaggle. Treat −14.5 pp as an upper bound on the gap.

**Main cause: our data.**
- The 64K training context cut every teacher wrap-up. DeepSeek submitted after the 60k "Budget nearly spent" note in
  87% of its rollouts, but that note appears in **0 of 2,500** training sequences.
- The model never saw how a long attempt ends.
- Fix for E028b: train with 256K context so full rollouts survive, plus more data or epochs.

> **Teacher provenance (2026-10-17):** all E028 training data (E025) and the dev56 teacher comparison (E017 v3all) come from the *original* DeepSeek-V4.1-Flash. From ~2026-10-17 the API name `deepseek-flash` (and the legacy names) serves **DeepSeek-V4.1-Flash-1017**, and the old model is retired. New rollouts for E028b therefore come from a different teacher. Record the request date and `system_fingerprint` per rollout, and don't mix the two silently.

## 1. Goal

Distil DeepSeek-V4.1-Flash's agentic ARC reasoning into Qwen3.8-27B, the model we can run offline on Kaggle (4×L4).
The target was the reasoning process itself:
- the full thinking of every assistant turn;
- tool use: `run_python` in a persistent IPython session, and `submit_answer`;
- verifying `solve()` on the train pairs;
- ROLES notes.

Incorrect rollouts were included on purpose, so the model learns general reasoning patterns and not only lucky solves.

## 2. Data

### 2.1 Source: E025 rollouts

- **Teacher:** DeepSeek-V4.1-Flash, effort low.
- **Harness:** `evals/run_deepseek_tools.py`, persistent-v5, in grid mode with:
  - `--answer-tool --final-mode session --lenient-parse`;
  - budgets: 64k output per turn, 80k per attempt, soft target 40k, notes from 1% of the cap, warning at 75%;
  - `--prompt lean2roles --task-summary --helpers --train-check`.
- **Thinking is preserved in full:** the stored `reasoning_content` matches the API-reported reasoning tokens for every
  one of 34,712 turns.

| pool | tasks / pairs | teacher pass@1 | mean output tokens |
|---|---|---|---|
| arc2_train (Kaggle ARC-AGI-2 training) | 1,000 / 1,076 | 94.0% | 12.1k |
| nvarc_b0 (NVARC synthetic, training parents) | 4,000 / 4,000 | 68.2% | 30.0k |

### 2.2 Cleaning and selection (`build_dataset.py`)

- **Rendering.**
  - Uses tinker-cookbook's `qwen3_8_xhigh_reasoning` renderer, which is the Qwen3.8 HF chat template at xhigh
    reasoning effort.
  - Thinking is kept in every assistant turn, and tool specs are in OpenAI-wrapped form.
  - Each rollout is one sequence with `ALL_ASSISTANT_MESSAGES`.
  - The output is token-identical to the HF template; verified on all 2,500 sequences by three separate audits.
- **Masking.**
  - Weight 1 only on assistant tokens: thinking, text, tool-call XML and `<|im_end|>`.
  - Weight 0 on the system prompt and tools, the task prompt, tool outputs, headers and separators.
  - `check_mask()` runs on every sequence, plus independent text-level re-verification.
- **Cuts and drops.**
  - Over-long rollouts are cut after the last complete assistant turn that fits in 65,536 tokens, never mid-turn.
  - A rollout is cut before an unusable turn: a per-turn length stop or broken/empty tool JSON.
  - Dropped (counts are before sampling): no clean turns (9), usage events that don't line up with the turns (5),
    leaked chat markup (4), first turn longer than 64K (6), and eval near-matches 2753e76c / 81c0276b (13).
- **Selection.** Every usable arc2_train rollout (1,070 of 1,076) plus 1,430 random nvarc_b0 rollouts (seed 0).

| final set (2,500 rollouts) | value |
|---|---|
| arc2_train / nvarc_b0 | 1,070 / 1,430 |
| correct / incorrect | 1,992 / 508 |
| truncated at 64K / cut before a bad turn | 385 / 67 |
| total tokens | 70,868,248 |
| trained tokens (assistant) | **44,470,697 (62.8%)** |
| masked: task prompt / tool outputs / system and tools / headers | 18.80M / 5.41M / 2.07M / 0.12M |
| sequence length mean / p50 / p90 / max | 28.3k / 22.8k / 59.7k / 65.5k |
| fingerprint | `2500:cc16865ac6586a04` |

### 2.3 Contamination

- No dev56 or ARC-AGI-2 eval task id appears in the SFT data, and no eval id is an NVARC parent.
- No grid matches exactly; that covers 21.5k SFT grids, including solutions.
- One near-match was missed: training task `b7999b51`'s test output matches eval `89565ca0` up to D8 + colour. It is
  not in dev56; it should be excluded in future builds.

## 3. Training (Tinker)

### 3.1 Setup

- **Model and loop:** `Qwen/Qwen3.8-27B` with 64K context and LoRA rank 128, using our own training loop. The cookbook
  loop is Adam-only, and we wanted Tinker's Dimuon optimizer.
- **Optimizer:** Dimuon with β1 0.9, no clipping, LR decaying linearly to 0, no warmup.
- **Batches:** greedy token-budget batches of 300k total tokens (about 9–10 rollouts and 180–235k trained tokens);
  nothing is concatenated.
- **Loss:** the per-token mean over each batch's trained tokens.
- **Run:** 1 epoch, 252 steps, 2,498 rollouts; the final short batch is dropped.
- **Reviews:** the code was reviewed by independent sub-agents twice and tested against a fake Tinker client.

### 3.2 LR pilots

Each pilot is the first 20 steps of the real 252-step schedule, on identical batches. Values are train NLL on unseen
batches.

| peak LR (Dimuon) | NLL steps 10–19 | Δ vs next lower LR |
|---|---|---|
| 1e-3 | 0.7409 | — |
| **3e-3 (chosen)** | 0.7259 | −0.0150 ± 0.0011, lower at every step |
| 6e-3 | 0.7143 | −0.0117 ± 0.0007, but the gap narrowed from −0.0136 to −0.0097 |

The rule, fixed before launch, required a gap that isn't narrowing, so 3e-3 was chosen. The full run resumed from the
3e-3 pilot's step-20 checkpoint.

### 3.3 Full run

The run took 59 minutes, at 15.2 s/step, with no incidents.

| steps | 0 (base) | 0–19 | 20–62 | 63–125 | 126–188 | 189–251 |
|---|---|---|---|---|---|---|
| train NLL (unseen batches) | 0.764 | 0.732 | 0.700 | 0.697 | 0.693 | **0.688** |

First 50 vs last 50 steps:

| slice | first 50 | last 50 | change |
|---|---|---|---|
| correct | 0.702 | 0.667 | −0.035 |
| incorrect | 0.742 | 0.727 | −0.015 |
| arc2_train | 0.688 | 0.653 | −0.036 |
| nvarc_b0 | 0.729 | 0.702 | −0.026 |

### 3.4 Billing

- **Every datum token is billed at the train price**, masked tokens included (minus one per datum). A pilot billed
  5,623,335 tokens at $4.103/M = $23.07; trained-only billing would have been $15.06.
- **Total:** 3 pilots ($69.2) + the full run ($267.25) ≈ **$336.5**, plus checkpoint storage at $0.10/GB-month
  (each checkpoint is 5.8 GB of state plus 3.9 GB of sampler weights).

## 4. Export, merge, quantization, parity

- **Export.**
  - The deprecated archive download (`get_checkpoint_archive_url`) timed out twice.
  - `TrainingClient.save_weights_external()` + `get_external_weights_urls()` delivered the adapter in **82 s**: r128,
    alpha 32, fp32, 994 tensors, 3.86 GB.
- **Merge.** `tinker_cookbook.weights.build_hf_model` (Qwen3.5-family merger), shard by shard: **127 s**, 52 GB BF16.
- **Quantization.**
  - **ModelOpt FP8 (v1).** Static per-tensor E4M3 weights and activations on every language-model linear layer, the
    same layer set NVIDIA quantizes in its NVFP4 checkpoint; lm_head, in_proj_a/b, conv1d, vision and MTP stay BF16.
    Calibrated with max on 96 × 2,048 training-token windows (196,608 tokens). 29 GB.
  - **Block FP8 (v2).** The exact official `Qwen/Qwen3.8-27B-FP8` layout: 1,606 tensors, 128×128 e4m3 blocks, BF16
    `weight_scale_inv`, dynamic activations, and an identical config. No calibration; took 22 s. The scale convention
    was verified against a real official shard.

| mean trained-token NLL, 8 training sequences | Tinker (adapter) | merged BF16 | ModelOpt FP8 | block FP8 |
|---|---|---|---|---|
| value | 0.5269 | 0.5273 | 0.5272 | within ±0.002 per sequence |

**Long context** (4 sequences of 33–62.5k tokens), versus BF16:
- block FP8: +0.001 to +0.002 NLL;
- ModelOpt FP8 with fp8 KV: +0.002 to +0.004;
- no growth with length.

The Tinker reference (`tinker_reference.py`, under $1) reproduces the training logs: base matches step-0 NLL to 4
decimals, and SFT matches step 251 within 0.0003.

**Published (all private):**

| artifact | HF (`livobereshka/…`) | Kaggle (`chankhavu/qwen3-8-27b-e028-sft/transformers/…`) |
|---|---|---|
| LoRA adapter | `Qwen3.8-27B-e028-sft-lora` | `lora` |
| merged BF16 | `Qwen3.8-27B-e028-sft` | `bf16` |
| ModelOpt FP8 | `Qwen3.8-27B-e028-sft-FP8` | `fp8` |
| block FP8 (official layout) | `Qwen3.8-27B-e028-sft-FP8block` **(incomplete, see below)** | `fp8block` (complete) |

> **HF block-FP8 upload failed (found 2026-10-18).** The repo holds only 8 of the 18 shards (9.95 GB of ~31 GB). The
> upload stopped with "Private storage limit reached for org livobereshka": the org's private repos now total exactly
> 100 GB. The failure was in `box_artifacts/e028/up_fp8block_hf.log` but was missed at the time. The Kaggle `fp8block`
> variation is complete (all 18 shards) and is what notebook v2 uses. **vLLM 0.23 loads the 8-shard directory without
> any error**, so a missing shard won't fail loudly anywhere: check every shard in `model.safetensors.index.json`
> before serving a downloaded model.

## 5. Evaluation (dev56)

### 5.1 Setup

- **Pairs:** 56 ARC-AGI-2 evaluation pairs (`experiments/E009_tool_token_efficiency/dev_pairs.txt`), 1 attempt each.
- **Harness:** **the teacher's own harness and limits** (E025 flags). Sampling at temperature 1.0, top_p 0.95,
  top_k 20, xhigh.
- **Serving:** both Qwen arms in ModelOpt FP8 with the same recipe, on SGLang 0.5.20 with fp8 KV and radix cache. One
  server per GPU, with each attempt pinned to one server: 93–99% prefix-cache hits. DP round-robin had broken reuse.
- **Prompt parity:** stock SGLang re-serializes tool specs, adding `"strict": false` and `"defer_loading": null` and
  reordering keys (+22 tokens). We patched it in the container, after which prompt token counts matched training
  12/12. vLLM 0.23 needs no patch.

### 5.2 Results

| dev56, 1 attempt per pair, same harness and limits | SFT | base | DeepSeek-Flash teacher |
|---|---|---|---|
| pass@1 | **21/56 = 37.5%** | 22/56 = 39.3% | 39/56 = 69.6% |
| with an offline `solve(test_input)` fallback on capped attempts | 22/56 (1 of 8 rescued) | 22/56 (0 of 5) | — |
| mean output tokens per attempt | 60.0k | 55.0k | 39.1k |
| cap hits (80k) | 23 | 19 | — |
| correct among submitted | 20/32 (62%) | 22/37 (59%) | — |
| submitted after the 60k "Budget nearly spent" note | 10/28 | 8/16 | 16/17 |
| extra output after the 40k "Past the target" note | 30k | 21k | 18k |

- **Statistics:** SFT − base −1.8 pp, paired-bootstrap 95% CI [−16, +12.5], McNemar p = 1, so inconclusive.
- **Pair overlap:** SFT-only 7, base-only 8, both 14. This split is consistent with sampling noise: the teacher's own
  attempts 0 and 1 disagree on 11 pairs.
- **On the teacher's 39 solved pairs:** SFT 18, base 19.
- **Reasoning quality** (trace monitor and audit):
  - no loops, no language drift, thinking always closes, valid tool calls;
  - 17/21 early attempts verify `solve()` on the train pairs and write ROLES notes, as the teacher does;
  - it inherited the teacher's "Let me…" end-of-turn dithering;
  - it works in short iterative turns (largest turn p50 20k vs 33k for base) and never has base's failure mode of a
    64k-token pure-thinking turn.

### 5.3 Why the student underperforms the teacher

- **No wrap-ups in the data.**
  - E025 has 713 rollouts that saw the 60k warning, and DeepSeek submitted after it in 623 of them (87%).
  - 252 of those 713 are in the SFT set, but none kept a turn after the warning: a 60k-output rollout plus prompt and
    tool outputs always exceeds 64K, so the turn-boundary cut always lands before it.
  - The warning text appears in **0/2,500** training sequences.
  - The 385 truncated sequences (34% of trained tokens, 253 of them incorrect) never show a wrap-up.
- **Too little data for one epoch.** One epoch of 2.5k rollouts lowered NLL by 0.08 but didn't transfer the teacher's
  extra problem-solving: SFT solves 18 of the teacher's 39, base 19.

## 6. Kaggle

### 6.1 Notebooks

The notebook is `chankhavu/arc2-qwen38-e028-sft-submit`, private.
- **Build:** `kaggle/build_notebook.py --variant v5_sft` plus a settings JSON.
- **Model sources:** the SFT model plus the official FP8 mirror, used for the tokenizer.
- **Machinery:** the v3/v4 solver with the harness in **E025 mode**:
  - lean2roles, tool descriptions `e025`, executor style ipython, solve_fallback, max_turns 40;
  - replay parity against the training tokens: 70/70 rollouts token-identical.
- **Guards:**
  - internet probe ("INTERNET: OFF");
  - `MODEL IDENTITY` logged before vLLM starts, and a fail-loud "WRONG MODEL" guard on `e028_quant_info.json`;
  - `SERVER IDENTITY` logged after startup.
- **Budget:** 64k ceiling, soft 32k, with an adaptive per-attempt share. At ~200 tok/s over 11 h that comes to ~30k
  per pair.

| version | checkpoint | commit run (local mode) | L4 kernel | gen tok/s at ~13 running |
|---|---|---|---|---|
| 1 | ModelOpt FP8 | PASS (20-min window): 4.67/8 tasks, 9/13 answered | CUTLASS W8A8 | 143–169 |
| **2** | block FP8 | PASS (12-min window): 2.00/8, 3/13 answered* | default W8A8 block-FP8 (as official) | **198–207** |

\*The short commit window cut most attempts, so this score is not an accuracy signal.

- **Both commit runs:** internet off, model identity matched the uploaded files exactly (config md5 and shard sizes),
  1.49M KV tokens, 88–95% cache hits.
- **Box run of the v2 solver** on its submit path with ~60k budgets: 4/6 on random non-dev eval tasks. Not
  representative: no scipy or IPython in that executor, and a 65k server context.
- **Recommendation (made before §6.3):** submit v2 as an experiment. The leaderboard keeps the best score (v4b scored
  19 / 26 / 23). Expect SFT at or below v4b, because at a ~30k cap SFT finished only 8 of 56 dev56 attempts. The
  paced 4×L4 test in §6.3 makes "below v4b" the likely outcome.

### 6.2 Adversarial audit findings (no blockers)

1. Budget notes on Kaggle show the adaptive cap ("15k (hard limit 30k)") where training always showed 40k/80k.
   Fix: show the notes on the training scale.
2. **Bug:** `executor.py` `salvage_answer` / `check_answer_on_train` lose answers on large grids, because their output
   line is truncated at 16,000 characters. This affects v4b too.
3. Forced final submits (FORCE_THINK) are mostly useless: of 19, 4 copied test_input and 14 raised NameError. An
   accepted copy also skips the fallbacks. Fix: `force_submit_tokens=0`.
4. The free attempt_2 found no alternatives in any run.
5. Minor:
   - a hung vLLM that is still alive isn't detected;
   - late attempts get budgets they can't finish;
   - the 2-chars-per-token estimate undercounts digit-heavy grids;
   - the self-test should check that scipy and IPython are installed.

### 6.3 Submission-pace test on 4×L4 (E027 session, finished 2026-10-18 03:25 UTC)

**Setup:**
- Run by the E027 session on the Vast 4×L4 box: notebook v5_sft solver (md5 cad74873) with
  `kaggle_v5_sft_settings.json`, v1 ModelOpt FP8 on CUTLASS, concurrency 24.
- All 71 non-dev eval tasks (93 pairs), 1 attempt each.
- Only the local window was changed, to FORCE 225 / CAP 235 / WALL 250 min, which matches Kaggle's 11 h / 259 pairs.
- Logs: `experiments/E027_l4_throughput/sft_nb_kaggle/`. The numbers below were re-checked against
  `arc_logs/attempts_local.jsonl` and the eval solutions.

**Result:** 23/93 pairs = **24.7%** (17.00/71 tasks = 23.9%).

| stop / answer source | attempts | correct |
|---|---|---|
| submitted (`submit`) | 23 | **22** |
| hit the output budget | 70 | 1 |
| ↳ forced final answer | 31 | 1 |
| ↳ no answer | 37 | 0 |
| ↳ salvage / session `solve` | 1 / 1 | 0 / 0 |

- **Budget:**
  - The adaptive per-attempt share was a median 28.8k tokens, not the 64k ceiling.
  - Tokens per attempt: median 25.0k, p95 31.9k, max 39.5k.
  - Wall time per attempt: median 52 min.
- **Serving:** 159 gen tok/s at ≥20 running (6.65 tok/s per stream), 75% prefix-cache hits, 0 preemptions.
  - **Caveat (2026-10-19, E040):** this server had `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`, which the Kaggle notebooks don't pass and the base
    runs below didn't use. On E032 weights with the same agent, they cut throughput from 192 to 143 tok/s. The SFT
    arm was therefore slower than on Kaggle, and its gap to base is an upper bound.
**Paired base run** (same box, pairs, pace and concurrency): official Qwen3.8-27B-FP8 with the v4b solver,
`experiments/E027_l4_throughput/base93_v4b/`, re-checked the same way.

| | score | submitted (correct) | output budget (correct) | mean tokens | gen tok/s | cache hits |
|---|---|---|---|---|---|---|
| base (v4b), sample 1 | **35/93 = 37.6%** (28.00/71 tasks) | 41 (32) | 52 (3) | 27.9k | 192 | 82% |
| base (v4b), sample 2 | **38/93 = 40.9%** (30.50/71 tasks) | 39 (35) | 54 (3) | 29.2k | n/a | 88% |
| SFT (v5_sft v1) | **23/93 = 24.7%** (17.00/71 tasks) | 23 (22) | 70 (1) | 22.5k | 159 | 75% |

Paired differences, 95% CIs from a task-cluster bootstrap:

| comparison | difference |
|---|---|
| base sample 2 − sample 1 (run-to-run noise) | +3.2 pp [−7.1, +13.0] |
| SFT − base sample 1 | −12.9 pp [−23.3, −2.2] |
| SFT − base sample 2 | −16.1 pp [−25.0, −7.9] |
| **SFT − mean of both base samples** | **−14.5 pp [−22.7, −6.5]** |

- Against base sample 1, both solved 18 pairs; base alone 17, SFT alone 5.
- Base sample 2 is in `experiments/E027_l4_throughput/base93_v4b_s2/`. The score, tokens and CIs were re-checked here.
- **Other base references** on these pairs, unpaced: E027 base on vLLM 0.31, 47.3%; E020 K3 attempt 0, 37.8% vs SFT
  24.4% on 90 pairs.
- **Kaggle v4b probe** (paced, base FP8): 14/29 vs SFT 4/29 on the common pairs.

**Reading:** the model's answers are accurate when it finishes, but at Kaggle pace it rarely finishes inside ~29k. This
matches §5.3: the 64K training cut removed the teacher wrap-ups, so the model ignores budget notes. v2 (block FP8)
runs ~25% faster, so its budget would be about 36k, still well short of where this model finishes. **Expect v5
(either build) below v4b on the leaderboard,** unless the per-attempt share grows a lot.

## 7. Costs and compute

| item | cost |
|---|---|
| Tinker training (3 pilots + full run; all datum tokens billed) | ~$336.5 |
| Tinker sampling (reference logprobs + greedy) | < $1 |
| Tinker checkpoint storage | ~$4/month while kept (pilot and final checkpoints) |
| Vast 4× RTX PRO 6000 WS (18:33 → ~23:58 UTC; merge, parity, quantization, dev56, audit) | ~5.4 h × $4.49 ≈ $24 |
| Kaggle 4×L4 commit runs (v1 + v2) | ~50 session-min |

## 8. Lessons

1. **Long-horizon behaviour has to be in the training sequences.** Cutting at the context limit removes exactly the
   endings: wrap-up, final submit, reaction to budget warnings. Train with a context that holds full rollouts
   (Tinker's `:peft:262144`), or cut in the middle, not at the end.
2. **Check the serving format token by token.** Stock SGLang changes tool JSON (+22 tokens). Chat-completion harnesses
   re-serialize tool calls and lose prefix-cache hits (36% vs ~90%), so append verbatim.
3. **On Kaggle L4, use the official block-FP8 layout.** Per-tensor ModelOpt FP8 lands on CUTLASS, which is slower,
   and needs calibration; the official layout runs ~25% faster and needs none.
4. **Tinker bills every token of each datum.** Masking saves nothing; only shorter sequences do.
5. **Small evals are noisy.** dev56 with 1 attempt has CI ±14 pp, so use paired, repeated samples before claiming
   gains or complementarity.

## 9. Next steps (E028b)

- Rebuild the data on the 256K model, keeping full rollouts with the wrap-ups. Weight the rollouts where the teacher
  finishes efficiently; drop the never-submitting truncated ones. Add more NVARC and ARC-2 rollouts and/or a second
  epoch.
- Harness: no forced submit, budget notes on the training scale, the executor salvage fix (which helps v4b too),
  scipy/IPython in the self-test, exclude `b7999b51`.
- Evaluate at the Kaggle budget (~30k) with 2+ samples per arm, and compare against K3/v4b, not only base. §6.3 is the
  first such test, and SFT lost clearly. Budget-aware finishing has to come from the data (full rollouts, wrap-ups
  after the budget notes), not from the harness.

## Files

**This folder:**
- `README.md`: lab notebook.
- `build_dataset.py`, `train.py`, `tools_spec.py`, `e025_tools.json`.
- `tinker_reference.py`, `parity_check.py`, `format_check.py`, `quantize_fp8.py`.
- `eval_status.py`, `solve_fallback_rescue.py`.
- `HARNESS_E025_MODE.md`, `KAGGLE_V5_SFT.md`, the `kaggle_v5*_settings.json` files.
- `runs/`: Tinker metrics, per-rollout NLL, checkpoints.
- `parity/`, `box_artifacts/`: box scripts and logs, including `quantize_blockfp8.py` and the audit outputs.

**Elsewhere:**
- Data: `data/sft/e028/` (not synced).
- Results: `evals/results/tools_persistent_grid_lean2roles_{sft,base}_low_e028_dev56.jsonl` plus traces.
- Kaggle outputs: `kaggle/runs/v5-sft-commit{,-v2}/`.
- Harness E025 mode: `evals/arc_harness/{agent_qwen,executor}.py`, `evals/tests/test_qwen_e025.py`.
- Kaggle: `kaggle/src/solver.py`, `kaggle/build_notebook.py` (`v5_sft`), `kaggle/nb_v5_sft/`.
