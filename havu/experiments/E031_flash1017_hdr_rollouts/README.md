# E031: Flash-1017 rollouts with s20x60 + minimal effort header on the E028 SFT pairs

Started 2026-10-18 (Sunday, off-peak). Goal: re-collect DeepSeek teacher traces for the same 2,500 pairs as the E028
SFT set (1,070 arc2_train + 1,430 nvarc_b0, from `data/sft/e028/train_manifest.jsonl`) using the new model
(V4.1-Flash-1017) and the cheapest config within 5 pp of the best on eval (E030: s20x60 + `--effort-header minimal`,
80.3% pass@2 at 23.3k tokens). 1 attempt per pair. Approved at an estimate of ~$22–25 (E025 spent $37.9 on these pairs).

## Setup
`launch.sh` (persistent-v6): `--prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000
--max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 20000 --soft-wording strong
--effort-header minimal`, grid mode, low effort. Additions to the E030 eval config:
- `--train-check`: host-side only, no model tokens; records `train_pass` for SFT filtering.
- `--turns-note 5`: the header raised turn-cap stops in E030; the note is a safety net (E029).

No `--task-summary` and no `--helpers`. Pair lists: `pairs_{arc2_train,nvarc_b0}.txt`. Results:
`evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_e031_{arc2_train,nvarc_b0}.jsonl` (+ trace dirs),
logs `evals/results/log_e031_*.txt`.

## Results
Finished 2026-10-18 18:52Z (78 min, 60 workers), 0 errors, recorded cost **$21.45** (E025: $37.94 on the same pairs).
Scratch analysis: `scripts/analyze.py`.

| Batch | n | errors | pass@1 | train_pass | P(correct \| train_pass) | P(correct \| no train_pass) | out tokens mean / median / p90 | correct ≤40k | cost |
|---|---|---|---|---|---|---|---|---|---|
| arc2_train | 1,070 | 0 | 95.8% | 93.5% | 97.3% | 74.3% | 6.0k / 3.0k / 15.6k | 99.4% | $5.02 |
| nvarc_b0 | 1,430 | 0 | 70.3% | 81.3% | 80.3% | 27.0% | 15.6k / 9.7k / 37.1k | 98.4% | $16.43 |
| **total** | 2,500 | 0 | **81.2%** | 86.5% | 88.2% | 36.8% | **11.5k / 5.9k / 31.3k** | 98.9% | **$21.45** |

Stop reasons: 2,478 submitted, 17 output_budget, 5 max_turns (session fallback).

Paired with E025 (lean2roles + helpers + train-check, old Flash) on the same (task_id, pair):

| Batch | both | only E025 | only E031 | neither | pass@1 E025 → E031 | train_pass E025 → E031 | tokens/attempt E025 → E031 | cost ratio |
|---|---|---|---|---|---|---|---|---|
| arc2_train | 984 | 24 | 41 | 21 | 94.2% → 95.8% | 97.4% → 93.5% | 12.0k → 6.0k (0.50×) | 0.55× |
| nvarc_b0 | 902 | 82 | 104 | 342 | 68.8% → 70.3% | 86.8% → 81.3% | 29.3k → 15.6k (0.53×) | 0.57× |
| **total** | 1,886 | 106 | 145 | 363 | **79.7% → 81.2%** | 91.3% → 86.5% | **21.9k → 11.5k (0.53×)** | **0.57×** |

- **Same accuracy, half the tokens.** pass@1 +1.5 pp (a tie; 145 vs 106 discordant pairs) at 0.53× output tokens and 0.57× cost. These traces are much shorter teacher targets for SFT: correct rollouts that are also train_pass have a median of 2.6k vs 5.6k tokens (arc2_train) and 6.1k vs 13.0k (nvarc_b0).
- **train_pass is lower (−4.8 pp) while correctness is not.** E031 more often submits test answers whose program does not reproduce every train pair (cause not investigated; plausibly the minimal header's "submit as soon as" wording, or no `--helpers`). P(correct | no train_pass) is 37%, so the train_pass filter discards real solutions. Strict SFT filtering (correct and train_pass) keeps 1,907 traces vs 1,968 for E025 (−3%).
- **The two runs complement each other.** Either run is correct on 2,137 of 2,500 pairs (85.5%). Combining E025 and E031 traces gives more pairs with at least one correct trace than either run alone.
- Budget pressure barely binds on these easier pairs: only 0.7% of attempts hit the output budget, and 98.9% of correct rollouts used ≤40k tokens.

On Hugging Face (private): `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036`.
