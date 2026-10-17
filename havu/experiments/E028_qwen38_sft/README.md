# E028: LoRA SFT of Qwen3.8-27B on Tinker, on E025 DeepSeek-Flash tool-use rollouts

Status: training done (2026-10-17); evaluation pending. Started 2026-10-16. Originally named E026_qwen38_sft;
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
