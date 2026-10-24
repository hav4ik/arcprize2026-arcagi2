# E032: LoRA SFT of Qwen3.8-27B on the correct E031 Flash-1017 rollouts

Started 2026-10-18. This is the E028 follow-up ("E028b"). E028 trained on E025 traces cut at 64K, which removed every
teacher wrap-up, and at Kaggle pace it scored well below base (E028 REPORT §6.3).

E031 re-collected the same 2,500 pairs with DeepSeek-V4.1-Flash-1017. It used s20x60 (soft target 20k, hard limit
60k), the `--effort-header minimal` system message, lean2sub and the answer tool, with no helpers and no task
summary. The traces are about half as long, so almost all of them fit in 64K whole.

## Data (`build_dataset.py` → `data/sft/e032/`)

`build_dataset.py` is a copy of E028's builder. Changes:
- It reads the E031 results.
- It uses E031's own tool specs (`e031_tools.json`, a snapshot of `run_deepseek_tools.tools_for()` with the E031
  config).
- It renders the E031 system message after the tools block in the system turn, as the HF template does.

The renderer (`qwen3_8_xhigh_reasoning`), mask check and filters are unchanged from E028.

The system turn therefore holds Qwen's "Reasoning effort is set to xhigh…" line, then the tools, then DeepSeek's
"Reasoning Effort: Minimal…" header. The student harness must reproduce exactly this.

**Filter (user's choice): correct only.** Kept rollouts are correct, train_pass and submitted (`--correct-only`).
Over-long ones are dropped, not cut (`--overlong drop`). Why only correct ones, from two read-only audits of the
incorrect traces on 2026-10-18:
- The traces that submitted with a failing training pair (192) mostly did so right after the 20k "Past the target"
  note, by hand-typing grids, using test-only parameters or copying the test input.
- The traces that never answered (22) are thinking loops.
- The traces that passed the train check but were wrong on the test (256) have a clean process. Their recovery
  moves (diff, fix, re-verify) already appear in the correct traces.

```
S=<tokvenv>; $S/bin/python -I build_dataset.py --correct-only --overlong drop --max-tokens 65536
```

| | value |
|---|---|
| input rollouts | 2,500 |
| rejected | not_correct 593, over_max_tokens 7 |
| kept | **1,900** (971 arc2_train + 929 nvarc_b0; 943 parent tasks) |
| tokens | 29.50M total, **13.71M trained** (assistant turns only), 15.79M masked |
| token categories | system 1.10M, user 9.88M, tool_response 4.76M, assistant trained 13.71M |
| seq len | mean 15.5k, p50 12.4k, p90 32.3k, max 65.1k |
| empty-think turns | 836, mostly a bare final submit_answer, rendered as untrained header |

Of the kept traces, 161 cross the 20k "Past the target" note and then submit a verified answer. E028 had 0 sequences
with a budget wrap-up.

## Training (`train.py`)

`train.py` is a copy of E028's trainer. It uses `data/sft/e032`, and adds `--lr-final`: linear decay from `--lr` to
`--lr-final`, reached on the last step.

- **Setup:** Dimuon, LoRA rank 128, `Qwen/Qwen3.8-27B` (64K), batches of up to 300k tokens, 1 epoch.
- **LR (user's choice):** 3e-3 → 1e-4, linear. 3e-3 is E028's pilot winner.
- **Size:** 103 steps, 18.4 rollouts per batch on average. 29.50M tokens billed, est. **$121.04** at $4.103/M.

```
$S/bin/python -I train.py --name correct1900_dimuon_3e-3_to_1e-4 --lr 3e-3 --lr-final 1e-4 --max-cost 130 --yes
```

Launched 2026-10-18 19:56 UTC.

- **Paused at step 25 by mistake, then resumed (20:01 → 20:02 UTC).**
  - Per-step train NLL on unseen batches looked flat over steps 0–24 (0.824, 0.845, 0.856, 0.849 in 5-step blocks), so
    I stopped the run at the step-25 checkpoint.
  - A paired check (`paired_nll.py`: compute_logprobs on 40 sequences from not-yet-trained batches, base vs step-25
    sampler) showed the run was learning: **NLL 0.840 → 0.794 (−0.045), lower on 40/40 sequences**. Per sequence:
    −0.064 ± 0.005; arc2 −0.071, nvarc −0.054.
  - The flat curve was batch-to-batch difficulty variance (per-batch base NLL ranges from about 0.76 to 0.95), which
    hides a gain of this size.
  - The run resumed from `step00025` with identical args, so the schedule is unchanged. The interruption cost at most
    one discarded in-flight step (~$1.2).
  - **Lesson:** judge early progress with the paired check, not the per-batch curve.

### Result (finished 2026-10-18 20:16 UTC)

- **Run:** 103/103 steps, final LR 1e-4. Billed ~$121–122: $29.6 for steps 0–24 in the first launch, $91.46 for steps
  25–102, plus at most one discarded in-flight step.
- **Per step:** 286k tokens on average (181k–300k, budget 300k incl. masked), 133k of them trained (70k–179k), with
  18 rollouts per batch (10–27).
- **Checkpoints** (`tinker://99c4c413-7a89-5c0b-9d9a-30bcfa9a9e1d:train:0/{weights,sampler_weights}/<name>`): `step00050`,
  `step00075`, `step00100` (7-day TTL) and **`final`** (kept). The step-25 checkpoint is from the first session
  (`tinker://6e5e1a54-acc5-5697-8d87-2517aefef4aa:train:0/...`).
- **Train NLL** on unseen batches, token-weighted in 20-step blocks: 0.844, 0.819, 0.803, 0.801, 0.811, and 0.768 for
  the last 3 steps. This is noisy, because per-batch difficulty varies by about ±0.05.
- **Paired NLL vs base (compute_logprobs, identical sequences):**

  | sequences | base | step 25 | step 50 | final |
  |---|---|---|---|---|
  | 40 not-yet-trained batches (≥ step 51) | 0.898 | 0.850 (−0.047) | 0.838 (−0.060) | n/a (trained on them) |
  | 40 held-out E031 rollouts never in the set (incorrect but train_pass; 39 nvarc) | 0.958 | | 0.906 (−0.052) | **0.900 (−0.058)** |

  Every checkpoint beats base on every sequence (40/40). Most of the gain comes in the first 25–50 steps; from step 50
  to final, the held-out NLL drops a further 0.006. Scripts: `paired_nll.py N SEEN_STEPS CKPT...` and `heldout_nll.py
  N CKPT...`.

## Export, merge, block FP8, serving (2026-10-18, Vast 56613651, 2× RTX PRO 6000 WS)

- **Export:** `export_adapter.py` (save_weights_external, then signed URLs; only the URLs reach the box). The adapter
  is 3.86 GB (rank 128).
- **Merge:** `merge.py` (tinker_cookbook `build_hf_model`) took 117 s and gives 52 GB BF16. The box venv is `cbenv`,
  with a `.pth` pointing at the /opt/sglang site-packages.
- **Block FP8:** `quantize_blockfp8.py`, E028's v2 recipe in the official Qwen3.8-27B-FP8 layout: 128×128 e4m3 blocks,
  BF16 `weight_scale_inv`, dynamic activations. Took 18 s, 30.87 GB, max per-tensor relative error 0.0265 (as E028
  v2). `quantization_config` is identical to the official one; written with `e032_quant_info.json`.
- **Serving:** SGLang 0.5.20 with E028's tool-JSON patch to `protocol.py` (the box's file matched E028's base
  exactly). One server per GPU (GPU0 `e032` block FP8, GPU1 `base` official FP8). FP8 KV, radix cache,
  context 131072, 939k KV tokens each; tunnelled to local ports 32002 (e032) and 32001 (base).
- **Prompt format** (`format_check.py`): the training render is token-identical to the HF chat template, and both
  servers report the same prompt token counts. 12/12 for each arm.
- **Parity** (`parity_check.py`): trained-token NLL served vs Tinker on 6 sequences (3 train, 3 held-out):
  - e032: +0.002 to +0.006;
  - base: +0.001 to +0.006, the same offset, attributable to the FP8 KV cache;
  - the SFT gain over base, about −0.10 on these sequences, is preserved when served.

## Eval (`eval_chain.sh`, `eval_status.py`)

- **Harness:** the DeepSeek harness with E031's exact config. That is `--prompt lean2sub --answer-tool --final-mode
  session --lenient-parse --max-tokens 64000 --max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75
  --budget-soft 20000 --soft-wording strong --effort-header minimal --turns-note 5 --train-check`, 1 attempt.
- **Sampling:** temperature 1.0, top_p 0.95, top_k 20, reasoning_effort xhigh. 20 workers per arm (one GPU each).
- **Pair sets, in order:** dev56, then the 93 E027 Kaggle-pace pairs (`pairs_k93.txt`), then the remaining 23 of the
  172.
- **Launched** 2026-10-18 20:47 UTC. Results: `evals/results/tools_persistent_grid_lean2sub_{e032,base}_low_e032_*.jsonl`.

## Published weights (block FP8, private)

- **Kaggle:** `chankhavu/qwen3-8-27b-e032-sft/transformers/fp8block/1`, 30 files. Names and sizes are identical to the box
  copy; a per-file pull onto box 56598413 verified all 18 index shards.
- **HF:** `livobereshka/Qwen3.8-27B-e032-sft-FP8block`, 30.89 GB. Every file size matches the box copy and no index
  shard is missing.
  - To make room in the org's 100 GB private storage, the user had the E028 BF16 repo
    `livobereshka/Qwen3.8-27B-e028-sft` deleted (55.6 GB). It is still on Kaggle as `.../e028-sft/transformers/bf16`.
  - The upload must run with /opt/sglang's huggingface_hub 1.32: cbenv's 2.2.0 has no `upload_large_folder`.

## Results (2026-10-19 01:47 UTC, RTX boxes released)

First attempt, DeepSeek harness at E031 settings (60k cap), xhigh, all 172 public eval pairs, paired:
- **E032 69/172 = 40.1% vs base 62/172 = 36.0%,** +4.1 pp [−3.9, +12.0] (task-cluster bootstrap); task-weighted
  42.9% vs 38.3%. E032-only 25, base-only 18.
- **Correct within 29k output tokens** (≈ Kaggle per-attempt share): **47 vs 35**, +7.0 pp [0.0, +13.8]. Within 20k:
  31 vs 15. Pairs both solve: 21.4k vs 26.4k mean tokens. Cap hits 32 vs 26.
- **Minimal tier vs xhigh (E032, 149 pairs):** 53 vs 60, tied within 29k (41 vs 41), −4% tokens, cap hits 19 vs 28.
  No gain; xhigh is the Kaggle pick (`chankhavu/arc2-qwen38-e032-sft-submit` v1).
- **Medium (no tier line), partial:** worse for both (E032 9 vs 14 on 25 pairs; base 8 vs 14 on 34).
- **pass@2 partial:** 93-pair set, both attempts on 20 pairs: E032 15 vs base 16.
- **Boxes:** 56613651, 56598413 and 56630374 were destroyed at 01:47 UTC (tokens checked: 0 left), about $35 in total.
- **Report:** `REPORT.html` (published at https://claude.ai/artifact/VGvuVqNoW5Ey6jpe715A5P).
- **Kaggle-pace 4×L4 (E027 session; slowdown diagnosis E040), 2026-10-19 02:50 UTC:**
  - The first 48-pair probe is void. Its vLLM server had `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`, which the Kaggle notebooks don't pass. On the
    same E032 weights and E031-mode agent, those flags give 143 tok/s against 192 without them.
  - The slowdown is not from the prefix cache: prompts are 100% token-prefix-stable, there are 0 preemptions, and the
    v4b agent was just as fast at the same 50% hit rate.
  - **Rerun with Kaggle's exact server flags and `FORCE_SUBMIT_LAST` (the submitted config): 6/48 = 12.5%** (6.00/35
    tasks), against 19/48 for base v4b at Kaggle pace (both 6, base only 13, E032 only 0). 152 gen tok/s (vs ~210 for
    base), 21.8k tokens per attempt, 37/48 out of budget; answers: submit 11, forced 7, salvage 2, none 28. Records:
    `experiments/E027_l4_throughput/e032_probe48_kflags/`.
  - **The notebook reproduces the RTX eval:** on these 48 pairs E032 solved 15 at the 60k cap on RTX, 7 within 29k,
    and 6 within the tokens each paced attempt actually got, exactly the paced result. The loss is finishing in budget.
  - **Correction:** the "within 29k" edge (47 vs 35) is against base in the E031 harness. Base's own v4b solver is far
    better at that budget: 35–38/93 at Kaggle pace, where E032 has 25/93 within 29k and E031-harness base 17/93. So
    E032 is not expected to beat the v4b submission on the leaderboard. Next: fit the student to a ~29k share.
  - E028's paced SFT run (24.7%) had the same flags (159 tok/s), so its −14.5 pp gap to base is an upper bound.
