# E028: LoRA SFT of Qwen3.8-27B on Tinker, on E025 DeepSeek-Flash tool-use rollouts

Status: trained, merged, quantized (ModelOpt FP8 and official-layout block FP8), evaluated on dev56 (2026-10-17): ties base; Kaggle notebooks v1/v2 built. Started 2026-10-16. Originally named E026_qwen38_sft;
renamed because E026 and E027 were already taken.

Goal: SFT Qwen3.8-27B on DeepSeek-V4.1-Flash agentic rollouts (E025), training every assistant turn
(thinking, text, tool calls) so the model learns the reasoning process, not only solved cases. Incorrect rollouts
are included on purpose.

## Data: `build_dataset.py` → `data/sft/e028/`

- **Source.** E025 rollouts `tools_persistent_grid_lean2roles_deepseek-flash_low_e025_{arc2_train,nvarc_b0}`, from harness
  `persistent-v5` with prompt `lean2roles`, `--task-summary --helpers --answer-tool`, grid mode, and budget notes of
  40k soft / 80k hard.
- **Rendering.**
  - Uses the tinker-cookbook renderer `qwen3_8_xhigh_reasoning`, which matches the HF Qwen3.8 template token for
    token (verified).
  - Thinking is kept in every turn, and there is one sequence per rollout with `ALL_ASSISTANT_MESSAGES`.
  - Tool specs come from `tools_spec.py` / `e025_tools.json` in OpenAI-wrapped form, as the harness sent them.
- **Masking.** Weight 1 covers only assistant tokens, from after the header through `<|im_end|>`. The system prompt
  with tools, the task prompt, tool outputs, headers and separators all have weight 0. An independent mask check runs
  on every sequence (`check_mask`) and was re-verified by three separate audits.
- **Cleaning.**
  - A rollout is cut just before its first unusable assistant turn: a per-turn length stop, or tool-call JSON that is
    broken or empty.
  - Dropped entirely:
    - rollouts with no clean turn left;
    - usage events that don't line up with the turns;
    - DeepSeek or Qwen chat markup leaked into the text;
    - text that encodes to a Qwen special token;
    - tasks `2753e76c` and `81c0276b`, whose train outputs match eval `89565ca0` up to rotation, flip and colour.
- **Over-long rollouts.** Anything over 65,536 tokens is cut after the last complete assistant turn that fits, never
  mid-turn. The prefix equals the rendering of the shortened conversation.
- **Trailing tool replies are kept,** for example `[answer accepted]`. They are untrained, about 0.09% of tokens. If
  the conversation ended on an assistant turn, the renderer would format that last turn as a generation rather than as
  history.
- **Selection (`--sample 2500`).** Every usable arc2_train rollout (1,070 of 1,076) plus 1,430 random nvarc_b0
  rollouts (seed 0).
- **Result:**
  - 2,500 rollouts (1,992 correct). 385 were truncated at 64K and 67 cut before a bad turn.
  - 70.87M tokens, of which 44.47M are trained and 26.40M masked. The masked part is 18.8M task prompt, 5.4M tool
    outputs, 2.1M system and tools, and 0.1M headers.
  - Dataset fingerprint `2500:cc16865ac6586a04`.
- **Earlier builds:**
  - `data/sft/e028_v1_correct_only/` holds only rollouts that are correct, pass the train check and were submitted.
  - `data/sft/e028_v2_all_drop64k/` holds all rollouts, with over-long ones dropped.

## Training: `train.py`

- **Loop.** Our own loop, because the cookbook's SFT loop is Adam-only.
- **Optimizer.** Dimuon (https://tinker-docs.thinkingmachines.ai/tinker/optimizers/) with β1 0.9 and no clipping.
  LoRA rank 128, 64K context (`Qwen/Qwen3.8-27B`), 1 epoch, LR decaying linearly to 0, no warmup.
- **Batches.** Each batch is filled greedily up to 300k total tokens (about 9–10 rollouts and 180–235k trained
  tokens). Every rollout stays its own sequence; nothing is concatenated. The full run is 252 steps over 2,498
  rollouts; the final short batch is dropped.
- **Loss.** Per-token mean over each batch's trained tokens: the weights are 1 divided by the number of trained tokens
  in the batch, so every step's weights sum to 1.
- **Logs.** `runs/<name>/`:
  - `metrics.jsonl`: per step, `train_nll` measured on the batch before training on it.
  - `rollout_nll.jsonl`: per rollout; `idx` is the row in `train_manifest.jsonl`.
  - `checkpoints.jsonl`: state and sampler paths.
  - `launches.jsonl`, `train.log`, `stdout.log`.
- **Pilots.** `--stop-after N` runs the first N steps of the full schedule and saves a resumable checkpoint that is
  kept indefinitely. Resume with the same `--name` and args to continue on the same schedule. A resume guard refuses
  changed args or data, and `--max-cost` refuses launches whose estimate is too high (default $50).
- **Reviews.** Code was reviewed twice by independent sub-agents, and the whole flow was tested offline against a fake
  Tinker client.

## LR pilots (2026-10-17): Dimuon 1e-3 vs 3e-3, first 20 of 252 steps, identical batches

| | 1e-3 | 3e-3 |
|---|---|---|
| train NLL, steps 0 to 9 (mean) | | −0.0053 ± 0.0013 vs 1e-3 |
| train NLL, steps 10 to 19 (mean) | 0.7409 | **0.7259** (−0.0150 ± 0.0011, lower at every step from 1 to 19) |
| Steps 10 to 19 by slice (1e-3 → 3e-3) | correct .7284 → .7125; incorrect .7677 → .7547; arc2 .7465 → .7323; nvarc .7413 → .7263; truncated .7482 → .7345 | |

- **Result.** The step-0 NLL (base model) is 0.764, and neither run spiked. 3e-3 wins under the rule fixed before
  launch. That rule required a clear win over steps 10–19, because short pilots favour higher LRs.
- **Caveat.** This measures early progress, not final loss. The next check would be a 6e-3 pilot (about $23).
- **Speed.** About 14 s per step (median), so the full run should take roughly 1–1.4 h.
- **Cost.** $23.07 per pilot, confirmed by billing: every datum token is billed. - **Follow-up 6e-3 pilot (same batches):**
  - Steps 10–19 mean NLL is 0.7143, which is −0.0117 ± 0.0007 vs 3e-3 and lower at every step.
  - The gap is narrowing, from −0.0136 at steps 10–14 to −0.0097 at steps 15–19 (slope +0.0007 per step, t = 7.2).
  - There is no instability.
  - The pre-registered rule requires a win that isn't narrowing, so it picks **3e-3**. 6e-3 checkpoint:
    `tinker://30f6c030-9223-5847-a0fb-18041a8431e4:train:0/{weights,sampler_weights}/step00020`.
- **Checkpoints (step00020, resumable):**
  - 1e-3: `tinker://129f7d2d-0619-5adc-a7e5-2ec278961953:train:0/{weights,sampler_weights}/step00020`
  - 3e-3: `tinker://1d7fa654-7547-54b4-91bf-022f4b3bda2b:train:0/{weights,sampler_weights}/step00020`

## Full run (2026-10-17): `full2500_dimuon_3e-3`, resumed from the pilot's step 20 to step 252

- **Clean run.** 15:49 → 16:48 UTC (59 min, 15.2 s/step). One launch, no incidents, no NLL flags.
- **Volume.** 2,498 rollouts trained once. 70.76M tokens sent over the whole run, 65.14M of them in this launch.

| steps | 0–19 | 20–62 | 63–125 | 126–188 | 189–251 |
|---|---|---|---|---|---|
| mean train NLL (unseen batches) | 0.7323 | 0.7000 | 0.6966 | 0.6934 | 0.6880 |

- **Trend.** The base model starts at 0.764 (step 0). Most of the drop happens within about 60 steps, then the NLL
  keeps falling slowly.
- **First 50 vs last 50 steps** (token-weighted NLL):

  | slice | first 50 | last 50 | change |
  |---|---|---|---|
  | correct | .702 | .667 | −0.035 |
  | incorrect | .742 | .727 | −0.015 (least) |
  | arc2_train | .688 | .653 | −0.036 |
  | nvarc_b0 | .729 | .702 | −0.026 |
  | truncated | .738 | .703 | −0.035 |
  | all | .717 | .688 | −0.029 |

- **Checkpoints** (`tinker://ccafea71-06e8-5045-b0f5-1b93f24e3c5e:train:0/{weights,sampler_weights}/<name>`):
  - `step00063`, `step00126`, `step00189` (7-day TTL, until 2026-10-24).
  - **`final`** (kept). The sampler for inference is `.../sampler_weights/final`.
  - Each checkpoint is 5.8 GB of training state plus 3.9 GB of sampler weights.

**Billing:**
- **Every datum token is billed at the train price,** masked weight-0 prompt and tool tokens included (minus one per
  datum, from the shift). A pilot billed 5,623,335 tokens = $23.07, where trained-only billing would have been $15.06.
- **Total E028 Tinker spend:** 3 pilots × $23.07 + full run $267.25 ≈ **$336.5**, plus storage.

## Export, merge, parity and FP8 (2026-10-17, on a rented 4× RTX PRO 6000 WS box)

**Export.**
- The deprecated `get_checkpoint_archive_url` path (server builds a .tar) timed out twice, after 20+ minutes each.
- `TrainingClient.save_weights_external()` + `RestClient.get_external_weights_urls()` worked in 82 s. Load the client
  from `.../weights/final`, save the external weights, then fetch one signed URL per file. Only the signed URLs
  reached the box, never the API key.
- Result: `adapter_model.safetensors` (994 tensors, fp32, 3.86 GB) plus `adapter_config.json` (r 128, alpha 32).
  Keys are in Tinker's native naming: split `in_proj_q/k/v`, `unembed_tokens`.

**Merge.**
- `tinker_cookbook.weights.build_hf_model(base_model=Qwen/Qwen3.8-27B, adapter_path=...)`, shard by shard: 127 s,
  52 GB BF16.
- The merger doesn't copy `generation_config.json`; copy it from the base model.
- The box environment is a uv venv with `zz_sglang_site.pth` pointing at `/opt/sglang` site-packages, plus
  `flash-linear-attention`. The cookbook is installed with `SETUPTOOLS_SCM_PRETEND_VERSION`.

**Tinker reference** (`tinker_reference.py`, under $1, 28 s). Tinker sampling client with the adapter applied, and
the base model:
- `compute_logprobs` on 8 training sequences: 6.6k–12.6k tokens; correct and incorrect; arc2 and nvarc.
- Greedy first-turn generation (top_k=1, 1,024 tokens) on 4 task prompts.

Tinker reproduces the training logs: base matches the step-0 NLL to 4 decimals, the SFT adapter matches step 251
within 0.0003. SFT lowers trained-token NLL on every sequence by 0.05–0.17. Base and SFT greedy outputs diverge
within 3–8 tokens.

**Parity** (`parity_check.py`, SGLang 0.5.20, one model copy per GPU, BF16 KV, radix off). Mean NLL over the 8
sequences:

| | Tinker | merged BF16 | merged FP8 (ModelOpt) |
|---|---|---|---|
| trained (assistant) tokens | 0.5269 | 0.5273 (per-sequence Δ ≤ ±0.0013) | 0.5272 (Δ ≤ ±0.0020) |
| untrained tokens (prompt grids, tool outputs) | 5.172 | 5.211 | 5.105 |
| greedy tokens identical to Tinker's (of 1,024) | – | 18–317 | 5–184 |

- **Untrained positions** are very high-NLL and overconfident on task grids (4–8 nats per token, for every engine).
  Individual positions there swing by up to ±29 nats between engines, but there is no systematic bias.
- **Greedy divergence** is ordinary kernel-level noise: base BF16 diverges from Tinker base after 28–696 tokens too.
- **Conclusion:** both the merge and the FP8 quantization are faithful.

**FP8 (`quantize_fp8.py`, ModelOpt 0.46.1, `FP8_DEFAULT_CFG`, max calibration).**
- **Format.** Static per-tensor E4M3 weights and activations on every language-model linear layer: self_attn
  q/k/v/o, linear_attn in_proj_qkv/in_proj_z/out_proj, and mlp gate/up/down. That is the same layer set NVIDIA
  quantizes in `nvidia/Qwen3.8-27B-NVFP4`.
- **Kept in BF16:** lm_head, embeddings, linear_attn in_proj_a/b and conv1d, and vision.
- **Calibration.** 96 windows of 2,048 tokens of E028 training sequences (seed 0; a quarter start at the beginning of
  the sequence). This took 92 s; the export took 140 s in total.
- **Output.** 29 GB, `quant_method: modelopt`, loaded by SGLang as `modelopt_fp8`. Weights take 28.5 GB of GPU
  memory, which leaves 444k KV tokens per RTX PRO 6000 in BF16 KV (195k for the BF16 model).
- **The MTP head is not in the FP8 export** (the HF class drops `mtp.*`; the merged BF16 model has 15 MTP tensors).
  It only matters for speculative decoding and was never trained.
- **Artifacts.** Locally: `data/sft/e028_parity/` (the reference with full logprob arrays, the parity data, the
  calibration windows) and `parity/bf16_merged_vs_tinker.json`. The models (`Qwen3.8-27B-e028-sft`, BF16, and
  `-FP8`) are only on the box, which is not persistent.

## dev56 evaluation (2026-10-17, 20:12 → 23:20 UTC, rented 4× RTX PRO 6000)

**Setup.**
- **Harness:** the DeepSeek harness itself (`evals/run_deepseek_tools.py`) with E025's exact flags: lean2roles, task
  summary, helpers, `submit_answer`, budget notes at 40k soft / 80k hard, 64k per turn, sampling at temperature 1.0,
  top_p 0.95, top_k 20, xhigh.
- **Arms:** both models in ModelOpt FP8 (same recipe and calibration), SGLang 0.5.20, FP8 KV, radix cache, one server
  per GPU. Each attempt is pinned to one server, which gave 93–99% prefix-cache hits; DP round-robin had broken cache
  reuse.
- **Prompt format check:** stock SGLang re-serializes tools, adding `"strict": false` and `"defer_loading": null` and
  reordering keys: +22 tokens. Patched in the container (protocol.py). After that, `format_check.py` shows 12/12
  prompts token-identical to training. vLLM 0.23 needs no patch.

| dev56, 1 attempt per pair | SFT | base | DeepSeek-Flash teacher (E017 v3all, same harness and limits) |
|---|---|---|---|
| pass@1 | 21/56 = 37.5% | 22/56 = 39.3% | 39/56 = 69.6% |
| plus offline `solve(test_input)` fallback on capped attempts | 22/56 = 39.3% (1 of 8 with a solve() rescued) | 22/56 (0 of 5) | — |
| mean output tokens | 60.0k | 55.0k | 39.1k |
| cap hits (80k) | 23 | 19 | — |
| submitted answers that were correct | 20/32 (62%) | 22/37 (59%) | — |

- **Paired comparison:** both solve 14, SFT-only 7, base-only 8. They are complementary: the union is 29/56.
- **On the 37 pairs the teacher solved,** SFT and base each solve 19.
- **Reasoning quality is healthy** (trace monitor): no loops or language drift, thinking closes, tool calls are valid,
  and it verifies solve() on the train pairs and writes ROLES notes like the teacher. It inherited the "Let me…" end-of-turn
  dithering.
- **Main failure: budget adherence.** After the 40k note the SFT model spends a median of 29k more tokens (base 15k,
  teacher 6k), and it ignored every "Budget nearly spent" (60k) note.
- **Root cause, in the data:**
  - E025 has 713 rollouts that saw the 60k warning, and DeepSeek submitted after it in 623 of them (87%).
  - None of the 252 such rollouts in our 2,500-rollout SFT set kept a turn after the warning, because the 64K
    turn-boundary cut always lands before it: a rollout with 60k output plus prompt plus tool outputs exceeds 64K.
- **For E028b:**
  - train on the 256K-context model (`:peft:262144`) so the wrap-up and submit stay in the data;
  - add more data and/or epochs;
  - consider emphasizing efficient teacher rollouts;
  - test the SFT+base ensemble for Kaggle's two answer slots.
- **Files:** `evals/results/tools_persistent_grid_lean2roles_{sft,base}_low_e028_dev56.jsonl` (+ traces),
  `solve_fallback_rescue.py`, `eval_status.py`, `format_check.py`.

## Adversarial audit (2026-10-17 ~23:30 UTC, 4 read-only sub-agents) and corrections

**Kaggle rerun path: sound, no blocker.**
- What ran on Kaggle is byte-identical to the repo.
- Rerun detection, the 240-task / 259-input fallback with atomic rewrites, the repair cell and the deadlines are all
  the same as v2/v4, whose reruns scored.
- 188 GB RAM fits 24 sessions; the ipython executor self-test passed on Kaggle.

**Block-FP8 checkpoint: CORRECT.**
- The scale convention was verified against a real official shard: dequantization is fp8 × scale_inv, with relative
  error 0.026.
- All 1,606 tensors were checked. BF16 tensors are bit-identical to the merged model; config, tokenizer and chat
  template are identical to the official files.
- Long context (33–62.5k tokens), versus BF16: block FP8 +0.001 to +0.002 NLL, ModelOpt +0.002 to +0.004, with no
  growth with length.
- Sampled generation: 0/48 loops.
- Not tested: vLLM logprobs. The Kaggle commit runs load and generate fine.

**Harness interface: faithful in format, but:**
- **(material) budget notes on Kaggle use the adaptive cap,** e.g. "15k (hard limit 30k)" against training's
  "40k (80k)". Fix: show the notes on the 80k scale.
- **(bug) `executor.py` `salvage_answer` / `check_answer_on_train` print all grids on one line,** which is truncated at
  16,000 characters, so salvage and train_pass are lost on large tasks (e.g. 4 × 30×30 train pairs). This affects v4b
  as well.
- **FORCE_THINK forced submits are mostly useless:** of 19, 4 copied test_input, 1 was a truncated grid and 14 raised
  NameError. An accepted copy also skips the solve/salvage fallbacks. Fix: `force_submit_tokens=0`, or reject the input
  copy.
- **The free attempt_2 found no alternatives in any run.**
- **The off-Kaggle box run had no scipy and no IPython** (3/6 traces hit ModuleNotFoundError). Kaggle has both; add
  them to the self-test.

**Evals and hygiene.**
- **Contamination:** no dev56 contamination.
- **Missed near-match:** training task `b7999b51`'s test output matches eval `89565ca0` up to D8 + colour. It is not
  in dev56; add it to `EXCLUDE_TASKS`.
- **Secrets:** none in the repo, the snapshot or its history, and none left on the box. The HF and Kaggle artifacts
  are private.

**Corrections to statements above.**
- SFT − base on dev56 is −1.8 pp with a paired-bootstrap 95% CI of [−16, +12.5] (McNemar p = 1): **inconclusive**,
  not "ties".
- The 7/8 SFT-only / base-only split is consistent with sampling noise: the teacher's own attempts 0 and 1 disagree
  on 11 pairs.
- SFT submitted after the 60k note in 10 of 28 warned attempts, not 0 (base 8/16, teacher 16/17).
- Extra tokens after the 40k note: SFT 30k, base 21k, teacher 18k (not 29k / 15k / 6k).
- The teacher solved 39 dev56 pairs; on those, SFT solves 18 and base 19.
- `format_check.py` compares prompt token counts, not ids.
- The eval used fp8 KV, while parity used bf16 KV.
- SFT pair 981571dc ended with an error at SGLang's 131,072-token context limit.

**Kaggle cap context.** At about a 30k cap, SFT finished only 8 of 56 dev56 attempts (7 correct), against 9 correct
for base, so the expected leaderboard score is at or below v4b's (19 / 26 / 23).

**Queued fixes for the next build:**
- `force_submit_tokens=0`;
- budget notes on the training scale;
- the executor salvage / train-check fix (also helps v4b);
- scipy and IPython in the self-test;
- exclude `b7999b51`;
- record extra_body and the launch command in eval rows.

Box artifacts (scripts, logs, audit outputs) are in `box_artifacts/`.

## Commands

```bash
PY=<venv with tinker-cookbook>/bin/python   # see "Environment"
cd experiments/E028_qwen38_sft
$PY -I build_dataset.py --sample 2500          # → data/sft/e028 (deterministic; fingerprint above)
$PY -I train.py --dry-run --lr 3e-3            # cost of what this launch would run
$PY -I train.py --name full2500_dimuon_3e-3 --lr 3e-3 --max-cost 300 --yes   # resume pilot → step 252 (~$267)
```

## Environment

- Tinker SDK 0.33.1, tinker-cookbook at commit `f6e01684117801cd49b23e9e3c24b75385371cc5`, torch 2.14.1+cpu.
- The venv lives outside the project `.venv` so the project's torch is left alone. To recreate it:
  `uv venv V && uv pip install --python V torch --index-url https://download.pytorch.org/whl/cpu && uv pip install
  --python V -e <tinker-cookbook clone>`.
- The Tinker key is `tinker_api_key` in `SECRETS.yaml`, which `train.py` reads at runtime.

## Open items

- **Before evaluating:** the Qwen harness (`evals/arc_harness/agent_qwen.py`) has no `lean2roles` prompt. It has to
  reproduce the training interface (`lean2roles` prompt, task summary, helpers, `submit_answer`, budget notes, xhigh
  system line); check by diffing tokens against `train.npz`.
- **Data weighting:** 235 incorrect NVARC rollouts truncated at 64K hold about 18% of trained tokens, and most never
  reach an answer. They are kept for now, as the user chose.
- **Evaluation plan:** base vs SFT in the same harness on the 56 dev pairs (pass@1, pass@2, tokens per attempt), then
  on all pairs. Also test the step-126 checkpoint.
