# Epoch-2 Plan: Pruned DeepSeek-V4.1 (`Image E128` 2-bit) QAT SFT — What Worked, What the Rescue Job Produced, and How to Continue

**Date**: 2026-11-01 · **Warm-start checkpoint**: `round2_v14_16xb200_fast/wu_1/step-00193-epoch_1p00` (resumable FP32 delta) · **Eval for epoch 2**: Havu Candidate B32, **text-only** (`--grid_format=json`)

Supporting artifacts:
- Rescue-job analysis (full tables + printed trajectories): [rescue_analysis.md](file:///usr/local/google/home/arazdai/.gemini/jetski/brain/c3ceefa2-c7fc-4e0f-b392-0b99ead3ba64/scratch/rescue/rescue_analysis.md) · machine-readable: [rescue_analysis.json](file:///usr/local/google/home/arazdai/.gemini/jetski/brain/c3ceefa2-c7fc-4e0f-b392-0b99ead3ba64/scratch/rescue/rescue_analysis.json)
- New reusable tool (blaze target `//experimental/users/arazdai/kaggle/sft:analyze_e065_rescued_rollouts`): [analyze_e065_rescued_rollouts.py](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/analyze_e065_rescued_rollouts.py)
- Epoch-1 report: [dsv41_qat_image_v14_experiment_report.md](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/reports/dsv41_qat_image_v14_experiment_report.md)

---

## 0. TL;DR

| Topic | Bottom line |
| :--- | :--- |
| **What made epoch 1 (v14) work** | (1) *Trainable 2-bit group scales* (86% of 1.89B scales moved; v12 had moved 0%), (2) *strict data hygiene* (0 identifier-typo turns, ≤1 error turn, no post-`train_pass` dithering) + *token loss weights* (WIC-alnum 10×, submit 5×, tool 2×) → tool-call error rate −57%, tokens/attempt −28%, (3) *cosine anneal to the end* — the model only beat the base after LR dropped below ~4e-6 (0.6–1.0 epoch). |
| **New finding** | The routed-expert **LoRA (r=32, 3.02B params) was forward-inert**: `routed_codes_changed = 0 / 181,193,932,800` at every export. All routed-expert adaptation came from the scales. LoRA deltas (~1e-4) never reach half a quantization step (~0.014). Either drop it (saves memory/compute) or raise its LR ≥20× in a pilot. |
| **Rescue job (overnight)** | 32/32 shards, **848 / 1,429 tasks processed (59%), 533 solved + rewritten (62.9%)**. Demo-overfit failures rescue at 64.6%, train-fail at 46.4%. Rescued trajectories are **~11× shorter** (median 1.3–1.7k output tokens vs 19k) and 100% define `solve`. **Caveats**: 22% of model rewrites leak the hidden rewrite instructions ("verified implementation", "instructions say…"), 57% of the 105 sandbox-synthesised rows reference the previous attempt, 23% contain a self-corrected error turn. **~360 (68%) are clean as-is; ~300 survive the epoch-1 filters.** The CNS output dir has not been updated since 00:14 UTC — check the XM job (581 tasks still pending). |
| **Previous-run data (Top-10K v8)** | **8,006 verified-clean rows / 6,777 unique tasks across 22 ARC-type datasets** (NVARC 1,776, Nemotron-RL 1,359, ARC-GEN 1,327, diva 1,140, tama 648, RE-ARC 437, ConceptARC 314, …); tiers: 4,986 frontier / 964 hard / 2,056 anchors; median 2.3k completion tokens, 5 turns; text-only. |
| **Epoch-2 recipe (recommended)** | Warm-start from `step-00193`, **fresh schedule**, peak LR **5e-6** (0.5×), scales LR 3e-6, warmup 3%, cosine→10%, bs 32, 1 epoch over ≈9–10k shuffled trajectories (E065 text + Top-10K + clean rescued, arc2 ×2, rescued ×2, tier2 ×1.5/tier3 ×0.5), **text-majority (≤15–20% images)**, same loss weights, milestones every 0.25, **model-soup** the last two milestones. ≈9–11 h on 16×B200. |
| **Blockers before launch** | 3 small code changes: `--init_from_checkpoint_dir` (fresh step counter — the current `resume_from_checkpoint_dir` fast-forwards 193 batches of the *new* dataset and keeps the annealed LR), a combined `e065_all_with_top10k_v8` subset + modality-aware dedupe + meta-leak scrub in `audit_and_prepare_data.py`, and **pinning the epoch-1 val parents** (otherwise 32 rescued parents / re-shuffled parents leak between epoch-1 train and epoch-2 val). |
| **Eval first** | Run the text-only B32 harness on `epoch_1p00` **and** the base image model now (no text-only number exists for either); compare with v12's text-only base (71.9% P@1 / 80.3% P@2). |

---

## 1. Epoch 1 (v14) recap — which tricks improved performance

### 1.1 Setup in one paragraph
`round2_v14_16xb200_fast`: 16×B200 (DP=2 × EP=8), 6,149 trajectories (3,050 image / 3,099 text, 79.7M tokens, 24.7M supervised), 193 steps at bs 32, AdamW lr 1e-5 cosine (warmup 5%, min 0.1×), WD 0.01, global grad-clip 1.0, 32k max seq. Trainable: 1.89B 2-bit group scales (FP32), 3.02B routed-expert LoRA (r=32), 1.42B shared experts (FP8 block-32 STE), 73M vision aligner. QAD KL weight was effectively **0** (`qad_kl = 0.0` in every logged step). Wall time ≈ 7.1 h of training (+ 12 min per 2-bit export).

### 1.2 What moved the needle (ranked, plain language)

| # | Trick | Evidence | Keep for epoch 2? |
| :-: | :--- | :--- | :--- |
| 1 | **Make the 2-bit experts trainable via FP32 group scales** (`--enable_qat --qat_trainable_expert_scales`) | v12 (no QAT) exported **0.00%** changed routed experts; v14 exports changed **86.3%** of scales. val_loss 0.7112 → 0.6715 (v12 plateaued at 0.66 on a different val set but gains were all in shared experts). | **Yes** — this is the core mechanism. |
| 2 | **Data hygiene filters** (0 NameError/SyntaxError turns, ≤1 Python-error turn, ≤1 wasted turn / ≤2.5k tokens after `train_pass`, ≤3 per parent×modality, arc2 ×2) | Tool-call error rate 15.8% → 6.4% (−57%), auto-repairs 101 → 1, mean tokens/attempt 26.2k → 19.0k (−27.6%), `train_pass` 86.5% → 92.3%. | **Yes**, apply the *same* filters to Top-10K and rescued rows (r2_curated_v2 had them off for Top-10K). |
| 3 | **Token loss weights** (WIC-alnum 10×, final submit 5×, tool call 2×) | `val_wic_alnum_loss` 0.2252 → 0.1198 (−46.8%) — directly the BPE split-identifier typo pathology of 2-bit MoE. | **Yes**, unchanged. |
| 4 | **Cosine anneal / late checkpoints** | At lr ≈ 9.5e-6 (0.2–0.4 ep) the model was **−13 pp vs base** on common tasks; at 2e-6 → 1e-6 (0.8–1.0 ep) it was **+7.1 / +3.9 pp**. Large simultaneous scale moves perturb routing; it recovers only once LR is small. | **Yes** → start epoch 2 at a *lower* peak LR (no reason to re-perturb). |
| 5 | **Exact global grad clipping + tiered LRs** | Zero DP-replica divergence over 193 steps; grad-norm 0.8 → 0.2. | Yes (unchanged). |
| 6 | **Vision aligner training + E058 image prompts** | Needed for the image harness; irrelevant for a text-only eval. | Optional (see §4.5). |
| 7 | Routed-expert **LoRA r=32** | **Inert**: 0 code flips at every milestone (needs |ΔW| > s/2 ≈ 0.014; actual ΔW ≈ 1e-4). Costs ≈ 4.5 GB/rank (params + Adam) and a B·A matmul per expert per step. | **Drop or fix** (§4.3). |

### 1.3 Refreshed evaluation status (text+image B32, N=2; 4-shard runs, 2026-11-01 01:10 UTC)

| Milestone | Tasks done (of 120) | Official Pass@1 | Official Pass@2 | Pair `train_pass` | Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `epoch_0p20` | **120** | 64.31% (77.2) | 73.47% (88.2) | 84.4% | complete |
| `epoch_0p40` | **120** | 68.75% (82.5) | 79.03% (94.8) | 83.2% | complete |
| `epoch_0p60` | 38 | 83.8% | 86.4% | 90.4% | running |
| `epoch_0p80` | 9 | — | — | — | just started |
| `epoch_1p00` | 41 | 80.1% | 83.7% | 98.1% | running |
| Base image model (apples-to-apples, from report) | 49–52 common | 80.6–82.7% | — | 85.7–86.5% | `+7.14 pp` (0p80) / `+3.85 pp` (1p00) for v14 on the same tasks |
| **Reference: text-only** base `dsv41_e128_2bit` (v12 eval) | 120 | 71.94% | 80.28% | 87.4% | the number epoch 2 must beat under the text harness |

> [!IMPORTANT]
> There is **no text-only (`json`) B32 number yet for `epoch_1p00`, nor for the base *image* model** (`dsv41_img_e128_2bit_causal`). Both are needed as the epoch-2 baseline. Launch them first (command in §8.1).

### 1.4 Training-curve sanity (from `metrics.jsonl`)
Train loss by 39-step window: 0.412 → 0.407 → 0.404 → 0.376 → 0.383; val_loss decreased monotonically at every milestone; no overfitting signal at the end of epoch 1 → a second pass over the same data at lower LR is justified (standard SFT practice is 2–3 epochs; see Muennighoff et al. 2023 on data repetition being nearly free up to ~4 epochs).

---

## 2. The overnight rescue job (`deepseek_v41_e065_correct_and_rewrite`)

### 2.1 What the job does (the "trick")
For each E065 rollout that passed the demos but failed the hidden test (or failed the demos), the **full** DeepSeek-V4.1-Flash (384 experts, BF16→FP32 on TPU) is prompted with the puzzle **plus a diagnostic block** — *"your previous rule passed all training pairs but failed the test input (shape mismatch / N cells wrong) — look for the hidden assumption that breaks on `test_input`"* — together with the previous (incorrect) code and any prior failed retries ("do NOT repeat this hypothesis"). Up to 4 correction rounds; success = `train_pass` **and** exact match on **all** test pairs. Then a **clean prefix-free rewrite**: a fresh session where the model is handed the verified `solve` and asked to produce a direct 2-turn trajectory with no mention of prior attempts. If the rewrite leaks or needs nudges, a canned 2-turn trajectory is synthesised in a sandbox (`sandbox_verified_clean_synthesis`).

### 2.2 Progress & solve rates (32/32 shards, 1,429 tasks; snapshot 01:30 UTC)

| Metric | Value |
| :--- | ---: |
| Completed tasks | **848 / 1,429 (59.3%)** — 581 pending; CNS dir last modified **00:14 UTC** → verify the XM job is alive |
| Solved + rewritten | **533 (62.9% of completed)**; 535 image rows + 535 text-twin rows written |
| Solved on attempt 1 / 2 / 3 / 4 | 248 / 153 / 95 / 37 (cumulative 29% → 47% → 58% → 63%) — attempts 3–4 still add ¼ of the yield |
| By original failure | demo-overfit **501/776 = 64.6%**, train-fail-submitted 32/69 = 46.4%, budget-exhausted 0/3 |
| By source | nvarc_training 518/830 = 62.4%, kaggle arc2_train 15/18 = 83.3% |
| Unsolved (315) | all used 4 attempts; top diagnostics are **near-misses**: 1–6 wrong cells in 70% of final attempts |
| Rewrite source | model rewrite try-1 **355**, try-2 **73**, sandbox synthesis **105** |
| Cost | 8.7M decode tokens, 138 tok/s/slice, ≤1.3 h wall per shard so far |

### 2.3 Anatomy: original failed rollout vs rescued trajectory (same 533 tasks)

| | Original (failed) | Rescued (rewritten) |
| :--- | :---: | :---: |
| Assistant turns (median / p90 / max) | 7 / 16 / 39 | **3 / 7.8 / 31** (2-turn: 165, 3-turn: 109, 4-turn: 76) |
| Output tokens (median / mean) | 18,977 / 22,100 | **≈1,350–1,700 / 2,650–3,100** (≈11× shorter) |
| Defines `solve(g)` | — | **100%** |
| Contains a self-corrected error turn | — | 125 (23%) |
| Text twin of the task already test-passing in E065 | 162 (30%) | → these tasks already have one trajectory in epoch-1 train (52 survived the caps) |

**Independent re-execution** (plain `exec` of the rewritten cells, GT train + all test pairs, 533/533): **88.0% pass train & test** (469); 396 (74%) pass with *zero* erroring cells; 36 fail on names only defined in a failed cell (`label`, `ndimage`, `blocks` — the Havu session preloads only `np`/`scipy`, so these are genuine fragile trajectories); 1 timeout. The pipeline's own gate is stricter at the session level, so treat the ≈12% as "drop candidates", not as a pipeline bug.

### 2.4 Quality problems you must handle before training on them

| Issue | Where | Size | Fix |
| :--- | :--- | :---: | :--- |
| **Meta-instruction leaks** — reasoning mentions "the verified implementation", "instructions say…", "clean solution trajectory", "already discovered", "key reasoning insight" | model rewrites | **96 / 428 (22%)** with the widened lexicon; the pipeline's own list caught only ~11%. One printed "clean" sample still says *"Based reference insight…"* → the true rate is somewhat higher | Scrub sentences matching the lexicon (add `reference insight/solve/code`, `hint`); drop the row if >20% of turn-1 reasoning is removed. Better: re-run the rewrite stage for these tasks with the verified code moved to the *system* message and the widened lexicon as the acceptance gate (cheap: ≈1 h/shard). |
| **Prior-attempt references** — "diff vs prev…", "explains why previous failed", "accuracy 60%" copied from the diagnostic block | sandbox synthesis (turn-1 think = last 2,000 chars of the *correction* thought) | **60 / 105 (57%)**; also 52/105 have an erroring verify cell | Drop the 105 synthesised rows (or regenerate with a real rewrite). They are also stylistically odd (mid-thought opening + canned turn 2). |
| Self-corrected error turns (NameError/typos then fixed) | both | 125 | Already removed by the epoch-1 filters (`max_ident_syntax_error_turns=0`, `max_py_error_turns=1`). |
| Debug artefacts in code (`prev=solve(...)`, `print('diff vs old rule', …)`) | synthesis + some rewrites | tens | Covered by dropping synthesis rows; otherwise harmless. |

**Net usable now: ≈362 clean (no strong leak, no prior-attempt reference) → ≈300 after the epoch-1 filters; ≈500 if the job finishes and leaking rows are regenerated.**

### 2.5 Are they teaching the right thing? (printed samples, see [rescue_analysis.md](file:///usr/local/google/home/arazdai/.gemini/jetski/brain/c3ceefa2-c7fc-4e0f-b392-0b99ead3ba64/scratch/rescue/rescue_analysis.md#sample-rescued-trajectories))
- `d4b1c2b1_f8be4b64` (demo-overfit → 3 turns): labels components, derives "square size = number of distinct colours, anchored at centre − s//2", verifies all train pairs in one cell, inspects the test output, submits. This is exactly the compact verify-then-submit behaviour that cut tokens by 28% in epoch 1.
- `642d658d_310f3251` (7 turns): the hint code was incomplete (`extract_self_contained_solve_code` only keeps the last `def solve` cell, dropping helper cells), so the model genuinely re-derived the rule — good exploratory data, but it says *"given 'verified implementation' in prompt has undefined pluses2"* → leak.
- `29623171_bd4472b8` (synthesis, 2 turns): opens with *"diff vs prev: … explains why previous failed"* → must not be trained on as-is.
- What the rescues capture that epoch-1 data lacked: **edge-case generalisation** (densest vs. right-most segment, centre anchoring for even sizes, multi-test consistency) on the tasks the teacher itself got wrong first — i.e., the hardest 25% of the distribution.

### 2.6 Leakage vs the epoch-1 split (why the val set must be pinned)
`e065_img_text_v1`: 6,149 train records (2,857 unique tasks; arc2_train 2,830 / nvarc_b1 2,353 / nvarc_b0 966), 508 val records (313 tasks). Rescued tasks: 52 already in epoch-1 train (via text twins), **9 in epoch-1 val, 32 share a parent with epoch-1 val parents**, 350 share a parent with train parents, 7 new parents. `split_by_parent_puzzle_id` reshuffles parents with a fixed seed → adding Top-10K/rescued parents **changes the permutation**, so epoch-2 val would contain parents the model trained on in epoch 1 (optimistic, non-comparable val loss). → Pin epoch-1 val parents (code change §7.2d) and route rescued/Top-10K rows whose parent is in that set to val (or drop them).

---

## 3. Data for epoch 2

### 3.1 Sources (text-only eval ⇒ text-first mix)

| Source | Unique verified trajectories | Notes |
| :--- | ---: | :--- |
| **E065 epoch-1 pool** (`e065_all`, both modalities) | 4,734 train / 508 val (3,050 image + 3,099 text records after arc2 ×2) | Harness "E046 best (text)" twins + E058 image rollouts of the same tasks. Reuse the identical filters. |
| **Top-10K v8 Pass@1** (`deepseek_v41_rollouts_top10k_v8_pass1`, 64 shards) | **8,006 rows / 6,777 tasks** (was 5,188 on 10/27 → re-audit at prep time) | Text-only, v8/E030 prompt; 22 datasets; 3-level verified (`verified_clean_sft`); `code_rescues`/`cap_nudges` = 0. Tier1 4,986 / Tier2 964 / Tier3 2,056. Median 2.3k completion tokens, 5 turns. |
| **Rescued** (`corrected_rewritten/shards`, text twins) | 533 now (→ ~900 when done), **≈300 clean after filters** | Hardest tasks; 10× shorter; upweight ×2. |
| E031–E038 (older DeepSeek text rollouts) | 6,825 | Same task families as E065 with an older harness → mostly duplicates; **skip** (or only tasks absent from E065 ∪ Top-10K). |

### 3.2 Proposed recipe (`audit_and_prepare_data.py`, `--model_family=deepseek_v41`)

| Knob | Epoch 1 | **Epoch 2** | Why |
| :--- | :--- | :--- | :--- |
| subset | `e065_all` | **`e065_all_with_top10k_v8` + `--e065_extra_rollouts_dir=…/corrected_rewritten/shards`** (new subset, §7.2a) | all three sources, one shuffle |
| modality | 50/50 | **text twins for all tasks; images only for arc2_train + the 301 `images_pass_text_fail` tasks** (≈15–20% of records) or `--text_only` | text-only harness; keeps a multimodal hedge cheaply |
| row filters | py_err ≤1, ident/syntax 0, post-pass ≤1 turn/≤2.5k tok, ≤32 images | **same, applied to Top-10K & rescued too** (+ `--drop_rescued_or_nudged`) | hygiene was a top-3 win |
| caps | 3 per parent×modality | 3 per parent×modality (nvarc), **2 per Top-10K task** | prevents easy synthetic families dominating |
| up/down-sampling | arc2 ×2 | **arc2 ×2, rescued ×2, tier2 ×1.5, tier3 ×0.5** | hard/frontier emphasis (r2_curated_v2 precedent) |
| val | 10% parents (seeded) | **pin epoch-1 val parents** + 10% of *new* parents; keep `max_eval_samples` on the epoch-1 subset for comparability | §2.6 |
| length | ≤32,768 | ≤32,768 | fits 16×B200 |
| seed | 20260405 | **20260406** for the batch order (trainer `--seed`), data split seed unchanged | fresh shuffle |

**Expected size**: ≈9–10k unique trajectories → ≈10.5–11.5k records after up-sampling → ≈100–120M tokens (Top-10K rows are short) → ≈330 steps at bs 32 → **≈9–11 h on 16×B200** (text rows skip the ViT).

### 3.3 Prompt-consistency check (do before prep)
E065 text rows carry the E046 prompt; Top-10K rows carry the v8 (E030 `sub80s40` + `lean2sub` + `minimal`) prompt; the B32 `json` harness is E043+E046+E055. Print `messages[0:2]` from one row of each source and diff against `havu_b32_harness.py`'s system/user template. If they differ beyond whitespace, add a `--canonicalize_prompt_to_b32_text` step (re-render only the system prompt + tool header; never touch grids).

---

## 4. Training recipe for epoch 2

### 4.1 Warm start (code change required)
`--resume_from_checkpoint_dir` restores weights but **keeps `global_step=193`, fast-forwards 193 batches of the new dataset and resumes the cosine schedule at lr≈1e-6**. Add `--init_from_checkpoint_dir=<step-00193-epoch_1p00>`: load trainable weights (tolerating/skipping `lora_*` tensors if `--routed_expert_lora_r=0`), `resume_step=0`, fresh AdamW + fresh schedule (§7.1).

### 4.2 Learning rate & schedule
- **Peak `--learning_rate=5e-6`** (0.5× epoch 1), `--warmup_ratio=0.03`, cosine `--min_lr_ratio=0.1`. Rationale: the scales are already adapted (86% moved); epoch 1 showed the high-LR phase *hurts* (−13 pp) and recovery only below ≈4e-6; 60% of the tokens are a second exposure → lower LR = less memorisation. Literature: continued SFT / second stage typically uses 0.3–0.5× the first-stage peak with a short re-warmup.
- **`--scale_learning_rate=3e-6`** (slightly lower tier for the 1.89B scales — they caused the routing perturbation), shared experts at the base LR.
- Keep `--weight_decay=0.01`, `--max_grad_norm=1.0`, `--global_grad_clip=true`, bs 32 (`grad_accum=2` on 16 GPUs).
- Milestones `0.25,0.50,0.75,1.00` with 2-bit export; eval ≥3 of them (epoch-1 lesson: pick by harness score, not val loss).
- Cheap A/B if time permits: a 0.25-epoch pilot at 1e-5 vs 5e-6 (≈2.5 h each) — compare `val_loss` and `train_pass` on one eval shard.

### 4.3 LoRA decision (new finding)
Because no 2-bit code flipped, epoch 1 was effectively **scale-only QAT + shared experts + aligner**. Options:
1. **Recommended for the main run:** `--routed_expert_lora_r=0` → identical forward behaviour, −4.5 GB/rank, faster steps (use the headroom for `--max_eval_samples=128` or more images).
2. **Side pilot:** keep r=32 but `--lora_learning_rate=2e-4` (20–40×) for 0.25 epoch and read `routed_codes_changed_frac` in `EXPORT_COMPLETED.json`. Real bit-flip QAT could unlock capacity, but STE flips are noisy — only adopt if val/eval improve.

### 4.4 Regularisation (what is worth it)
| Option | Cost | Verdict |
| :--- | :--- | :--- |
| Lower peak LR + short warmup | free | **Do it** (main lever against overfitting here). |
| **Model soup** of the last two milestones (average `backbone_trainable.pt` + `expert_lora_rank_XX.pt` FP32 tensors, then export) | ≈15 min + 1 eval | **Do it** — scales are continuous, codes fixed, so the soup is exactly representable; Llama-3 / WiSE-FT style gains are typically +0.5–2 pp and free. Also try (epoch-1 final, epoch-2 final). |
| Weight decay 0.01 on 2-D tensors | already on | keep. |
| QAD forward-KL to a teacher (`--qat_kl_weight`) | +≈40% step time (teacher forward) | **Skip for the main run.** Note the teacher snapshot is taken *before* the checkpoint load (base model, shared experts/vision only). "KL to epoch-1" needs the snapshot moved after loading (§7.3) — keep as a fallback if epoch-2 val diverges. |
| Dropout / label smoothing / NEFTune | code + risk with QAT | not worth it here. |

### 4.5 Include image data?
The submission harness is text-only, so images do not help the target metric and cost ≈11% of tokens (8.6M image tokens in epoch 1) plus ViT compute. But v14's multimodal gains are real and a text-only epoch would let the experts drift away from the image path. **Recommendation: text-majority with ≤15–20% image records (arc2_train images + tasks only solvable with images), `--train_vision=aligner`.** If compute-constrained or the Kaggle harness is final: text-only (`--disable_vision` path, `--train_vision=none`).

### 4.6 Data augmentation — what is safe
- **Do not** D8/colour-augment existing trajectories: the reasoning text and code hard-code colours/orientations; augmented grids would make the trajectories inconsistent.
- Safe: the synthetic families already *are* augmentations (NVARC variants `parent_variant`, RE-ARC, ARC-GEN); prefer sampling more verified variants of hard parents over any text surgery.
- Multi-test tasks: the B32 harness shows all test inputs in one rollout; E065 multi-pair rows already share one rollout — keep them.
- The best "augmentation" for epoch 3 is **on-policy rejection sampling** (STaR / ReST-EM / Llama-3-style): run the epoch-2 2-bit student with the B32 text harness on the 1,429 failed E065 tasks + Top-10K tier-2 + arc2 training tasks, keep 3-level-verified rollouts. Student-style data reduces the teacher→2-bit-student style gap (DeepSeek-R1-Distill and "s1/LIMO" show small amounts of high-quality on-distribution traces dominate).

### 4.7 Other cheap tricks from epoch 1 / literature
- **Short-trajectory bias is good, but cap it**: rescued 2-turn trajectories should stay ≤20% of records so the student keeps exploring when the first hypothesis fails (epoch-1 `train_pass` rose *because* it still iterates).
- **Rescue pass #2** on the 315 near-miss tasks: 6 attempts, temperature 0.8, and reveal mismatched *coordinates* (not values) — optional; each shard is ≈1 h.
- **Checkpoint selection by harness, not loss** (epoch 1: lowest val loss ≠ best score at 0.2–0.4).
- **Verify export delta** at each milestone (`routed_scales_changed_frac`, `backbone_changed_frac`) — the free regression test that caught v12.

---

## 5. Evaluation plan (text-only Candidate B32)
1. `launch_dsv41_b32_eval_xm.py --grid_format=json` (default) with `--eval_base_model=true` on the v14 root, `--milestones=1.00` → gives **base image model (text)** and **epoch_1p00 (text)**; 4 shards, N=2. This is the epoch-2 baseline.
2. Before launching, pull Havu's repo `main` and diff `kaggle/dsv41_tools/CANDIDATE_*.md` against the port at `91ecbcc1` (B32); port any new A-fixes into `havu_b32_harness.py` (both training-time verification and eval use it).
3. Epoch-2 milestones: evaluate 0.50 / 0.75 / 1.00 (+ the soup). Report Official Pass@1/2 on all 120 tasks and, for early reads, the common-task delta vs `epoch_1p00` (same deterministic shard order).
4. Keep the per-attempt telemetry that made epoch 1 interpretable: tool-call error %, auto-repairs, `train_pass`, median tokens/attempt.

---

## 6. Risks & open questions
- **Rescue job health**: no CNS writes since 00:14 UTC with 581 tasks pending — check `xid` status; relaunch `--start_shard/--end_shard` for stalled shards.
- **Second exposure to epoch-1 data**: mitigated by 0.5× LR and ≈40% new data; watch val loss on the pinned epoch-1 val subset — it should not go *up* at 0.25.
- **Prompt heterogeneity** (E046 vs v8 vs B32) — §3.3.
- **Image share**: decide text-only vs 15–20% images (affects wall time ≈10%).
- **LoRA**: drop (default) vs. high-LR pilot.
- **Eval cost**: a 4-shard, N=2 text eval takes ≈8–10 h/milestone on 8×B200 — budget 3 milestones + soup.

---

## 7. Required code changes (small, all in `//experimental/users/arazdai/kaggle/sft`)

### 7.1 [train_dsv41_sft.py](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/train_dsv41_sft.py)
- `--init_from_checkpoint_dir`: call `load_dsv41_resumable_checkpoint` but set `resume_step = 0` (no batch fast-forward, fresh schedule/AdamW); allow `strict=False` for `lora_*` keys when `routed_expert_lora_r=0`; log "warm-started from …" in `metrics.jsonl` row 0. Keep `auto_resume` working *within* the epoch-2 output dir (it must not pick up the epoch-1 dir).
- Optional: move `model.snapshot_teacher_weights()` after the checkpoint load and include `qat_scale` in the snapshot (enables "KL to epoch-1").
- Optional: `--save_fp32_soup_of=<ckptA>,<ckptB>` or a tiny standalone `soup_and_export_dsv41.py` (tensor-wise mean of the two resumable checkpoints → `export_dsv41_e128_2bit_checkpoint`).

### 7.2 [audit_and_prepare_data.py](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/audit_and_prepare_data.py) / [data_pipeline.py](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/sft/data_pipeline.py)
- (a) New subset `e065_all_with_top10k_v8` (load Top-10K shards when `subset` ∈ E065 subsets ∪ {new}); `selected_records = e065_verified + top10k_verified`.
- (b) Extra-shard dedupe keyed on `(task_id, modality)` instead of `task_id` (today the text twin of a rescued task is dropped because the image row is discovered first); add `--e065_modality_filter=text|image|both` and `--max_image_fraction`.
- (c) `--scrub_meta_leaks=true` + `--drop_meta_leak_rows=true` with the widened lexicon (§2.4) applied to `reasoning_content` of `corrected_rewritten` rows; `--drop_rewrite_source=sandbox_verified_clean_synthesis` (needs `rewrite_source` joined from `correction_diagnostics_*.jsonl`).
- (d) `--pin_val_parents_manifest=/cns/lu-d/home/arazdai/kaggle/sft/data/e065_img_text_v1/val_manifest.jsonl`: force those parents to val; split only the *new* parents.
- (e) `--batch_weights=corrected_rewritten:2` (generic per-batch up-sampling, like `tier_weights`).

### 7.3 Rescue pipeline (optional, for a cleaner re-run) — [deepseek_v41_e065_correct_and_rewrite.py](file:///google/src/cloud/arazdai/test_workspace/google3/experimental/users/arazdai/kaggle/arc_agi_2/pilot_1/setB/tpu_vllm/deepseek_v41_e065_correct_and_rewrite.py)
- Widen `FORBIDDEN_META_PHRASES`; put the verified code in the system message; make `extract_self_contained_solve_code` concatenate *all* cells that define functions used by `solve` (the `pluses2` case); drop the sandbox-synthesis fallback in favour of a third rewrite try; add `--only_task_ids` to regenerate the 96 leaking tasks.

---

## 8. Next steps (in order)

1. **Now** — check the rescue XM job; text-only baseline evals:
   ```bash
   /google/bin/releases/xmanager/cli/xmanager.par launch //experimental/users/arazdai/kaggle/sft:launch_dsv41_b32_eval_xm -- \
     --noxm_monitor_on_launch --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks --xm_skip_launch_confirmation \
     --exp_name=dsv41_qat_img_v14_b32_TEXT_eval \
     --sft_output_root=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1 \
     --eval_output_root=/cns/lu-d/home/arazdai/kaggle/sft/evals/v14_text_b32_eval \
     --base_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal \
     --base_engram_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
     --eval_base_model=true --milestones=1.00 --grid_format=json --num_shards=4 --num_attempts=2
   ```
2. **Code changes** §7.1–7.2 (≈half a day), unit tests in `sft_test.py` for the warm-start path and the pinned split.
3. **Data prep** (after the rescue job finishes or at a cut-off):
   ```bash
   blaze run //experimental/users/arazdai/kaggle/sft:audit_and_prepare_data -- \
     --model_family=deepseek_v41 --subset=e065_all_with_top10k_v8 \
     --e065_extra_rollouts_dir=/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten/shards \
     --drop_rewrite_source=sandbox_verified_clean_synthesis --scrub_meta_leaks=true --drop_meta_leak_rows=true \
     --e065_modality_filter=both --max_image_fraction=0.2 \
     --max_seq_len=32768 --max_py_error_turns=1 --max_ident_syntax_error_turns=0 \
     --max_wasted_post_pass_turns=1 --max_post_pass_tokens=2500 --max_images_per_sample=32 \
     --max_per_parent_vu=3 --max_per_task_top10k=2 --drop_rescued_or_nudged=true --top10k_require_verified_clean_sft=true \
     --upsample_arc2_train=2 --batch_weights=corrected_rewritten:2 --tier_weights=tier1:1,tier2:1.5,tier3:0.5 \
     --pin_val_parents_manifest=/cns/lu-d/home/arazdai/kaggle/sft/data/e065_img_text_v1/val_manifest.jsonl \
     --output_cns_dir=/cns/lu-d/home/arazdai/kaggle/sft/data/r3_e065_top10k_rescued_v1
   ```
   Then run `inspect_e065_rollouts`-style sanity on the manifest (source mix, token totals, image share) and §3.3's prompt diff.
4. **Launch epoch 2** (16×B200):
   ```bash
   /google/bin/releases/xmanager/cli/xmanager.par launch //experimental/users/arazdai/kaggle/sft:launch_dsv41_sft_b200_xm -- \
     --noxm_monitor_on_launch --xm_resource_alloc=group:research-mm/genai-lti-lh-tasks --xm_skip_launch_confirmation \
     --exp_name=dsv41_img_e128_qat_sft_16xb200_v15_epoch2 \
     --model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_img_e128_2bit_causal \
     --engram_model_dir=/cns/lu-d/home/arazdai/kaggle/models/dsv41_e128_2bit \
     --data_dir=/cns/lu-d/home/arazdai/kaggle/sft/data/r3_e065_top10k_rescued_v1 \
     --output_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round3_v15_epoch2/wu_1 \
     --init_from_checkpoint_dir=/cns/lu-d/home/arazdai/kaggle/sft/checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1/step-00193-epoch_1p00 \
     --num_nodes=2 --train_moe_only=true --train_vision=aligner --enable_qat=true --qat_trainable_expert_scales=true \
     --routed_expert_lora_r=0 --qat_kl_weight=0 --global_grad_clip=true --load_compact_engram=true \
     --learning_rate=5e-6 --scale_learning_rate=3e-6 --warmup_ratio=0.03 --min_lr_ratio=0.1 --weight_decay=0.01 \
     --num_epochs=1 --grad_accum_steps=2 --max_seq_len=32768 --seed=20260406 \
     --export_2bit_milestones=true --milestone_fractions=0.25,0.50,0.75,1.00 --auto_resume=true
   ```
5. **Evaluate** 0.50 / 0.75 / 1.00 + soup(0.75, 1.00) with the text-only B32 harness; compare against the §8.1 baselines and v12's 80.28% P@2 text reference.
6. **Epoch 3 (if epoch 2 helps)**: on-policy rejection sampling with the epoch-2 student (§4.6) + rescue pass #2 on the near-misses.

---

### Appendix A — Numbers behind the rescue analysis (snapshot 01:35 UTC)
- Leaks by rewrite source: try-1 355 rows → 71 strong meta leaks, 18 prior-attempt refs (85 either), 57 with error turns; try-2 73 → 25 / 3 (26), 16 error turns; synthesis 105 → 0 / 60 (60), 52 error turns. Clean candidates (no strong leak, no prior ref): **362**.
- Re-execution: 469/533 train∧test pass (88.0%); 396 with zero error cells; 124 with ≥1 self-corrected error cell; 36 NameError-on-helper failures; 1 timeout.
- Epoch-1 manifest: 79.7M tokens / 24.7M supervised; text 3,099 / image 3,050 records.
- Top-10K: 8,006 verified rows; tiers 4,986 / 964 / 2,056; `completion_tokens` p50 2,296 / p90 13,507; turns p50 5 / p90 13.

### Appendix B — Where things live
- Rescue shards: `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/e065_hardest_rollouts/corrected_rewritten/shards/` (`summary_*.json`, `correction_diagnostics_*.jsonl`, `e065_{images,text}_corrected_rewritten_*.parquet`)
- Failure records + GT: `.../e065_hardest_rollouts/non_test_passing/e065_images_non_test_passing_tasks.jsonl`
- Top-10K rollouts: `/cns/lu-d/home/arazdai/kaggle/arc_sft_data/sft_prep_pipeline/deepseek_v41_rollouts_top10k_v8_pass1/verified_rollouts_shard_*.parquet`
- Epoch-1 data / checkpoints / evals: `/cns/lu-d/home/arazdai/kaggle/sft/data/e065_img_text_v1`, `.../checkpoints_dsv41_e128_qat_iter2/round2_v14_16xb200_fast/wu_1`, `.../evals/v14_e065_img_b32_eval`
