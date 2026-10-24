---
pretty_name: ARC-AGI-2 DeepSeek Flash-1017 tool-use rollouts (s20x60 + minimal header; E031/E036/E038)
license: other
task_categories:
- text-generation
tags:
- arc-agi
- arc-agi-2
- reasoning
- tool-use
- agent-trajectories
configs:
- config_name: e031_arc2_train
  data_files: rollouts/e031_arc2_train.parquet
- config_name: e031_nvarc_b0
  data_files: rollouts/e031_nvarc_b0.parquet
- config_name: e036_nvarc_b1
  data_files: rollouts/e036_nvarc_b1.parquet
- config_name: e038_openrouter_deepinfra_b1r
  data_files: rollouts/e038_openrouter_deepinfra_b1r.parquet
- config_name: e037_openrouter_deepinfra
  data_files: rollouts/e037_openrouter_deepinfra.parquet
---

# ARC-AGI-2 DeepSeek Flash-1017 tool-use rollouts (E031 / E036 / E038)

This dataset holds **10,000 successful agentic rollouts** by **DeepSeek V4.1-Flash-1017** (`deepseek-flash` from
2026-10-17, low reasoning effort) on ARC-style training tasks: 8,754 through the DeepSeek API (`e031_*`, `e036_*`) and
1,246 through OpenRouter pinned to DeepInfra (`e038_*`), all with the same harness. A further 103 rollouts
(`e037_*`) are a paired DeepInfra-vs-DeepSeek probe.
The model works in a persistent Python sandbox, so each rollout is a full multi-turn trajectory: thinking, Python tool
calls, tool outputs and the final answer. Every attempt is scored against the ground truth and checked against the
task's training pairs.

All rollouts use one harness, chosen as the cheapest config within 5 pp of the best on the ARC-AGI-2 public eval:
80.3% pass@2 at 23.3k output tokens per attempt.

Companion dataset: `livobereshka/arc-agi2-deepseek-v3-rollouts-e025`. It uses the old V4.1-Flash and a different
harness (with task summary and helpers), on the same 2,500 pairs as `e031_*`.

## Contents

| Config | Tasks / rollouts | Source |
|---|---|---|
| `e031_arc2_train` | 994 / 1,070 | Kaggle `arc-prize-2026-arc-agi-2` training tasks; the arc2_train pairs of the E028 SFT set |
| `e031_nvarc_b0` | 1,430 / 1,430 | Random NVARC `nvarc_training` puzzles (E025 batch `nvarc_b0`); the NVARC pairs of the E028 SFT set |
| `e036_nvarc_b1` | 6,255 / 6,255 | **Parent-balanced** NVARC puzzles (batch `nvarc_b1`, see below) |
| `e038_openrouter_deepinfra_b1r` | 1,246 / 1,246 | The rest of `nvarc_b1` (tasks E036 did not run, plus its 1 errored task), same harness via OpenRouter → DeepInfra (fp8) |
| `e037_openrouter_deepinfra` | 103 / 103 | Same harness via OpenRouter, provider DeepInfra (fp8), on 103 `nvarc_b1` pairs that also appear in `e036_nvarc_b1` (paired comparison) |

1 attempt per test pair; one row per attempt.

## Results

| Config | pass@1 | train_pass | P(correct \| train_pass) | Output tokens mean / median | Correct & train_pass | Recorded cost |
|---|---|---|---|---|---|---|
| e031_arc2_train | 95.8% | 93.5% | 97.3% | 6.0k / 3.0k | 973 | $5.02 |
| e031_nvarc_b0 | 70.3% | 81.3% | 80.3% | 15.6k / 9.7k | 934 | $16.43 |
| e036_nvarc_b1 | 70.9% | 81.5% | 80.7% | 15.6k / 10.1k | 4,112 | $72.36 |
| e038_openrouter_deepinfra_b1r | 70.9% | 80.8% | 81.5% | 15.0k / 9.7k | 821 | $9.77 |
| e037_openrouter_deepinfra | 74.5% | 83.3% | 83.5% | 13.0k / 6.4k | 71 | $0.63 |

97.6% of correct E036 rollouts use 40k output tokens or fewer. Errors: 1 API connection error each in `e036_nvarc_b1` and
`e037_openrouter_deepinfra` (rows kept, `error` set; the `e036` one was re-run in `e038`). `e038` has no errors (13
connection errors were retried; the table keeps the last record per attempt).

**Correct & train_pass across the 10,000:** 973 + 934 + 4,112 + 821 = **6,840**.

**Caution:** about 19% of NVARC answers that reproduce every training pair are still wrong. Some generated puzzles are
underdetermined or noisy. The unused pairs of each puzzle are listed in `tasks/manifest.jsonl` and can be used to
re-check an answer program before treating a failure as a hard negative.

## DeepInfra quality check (`e038` vs DeepSeek API)

The `e038` tasks are different tasks from `e036`, so accuracy is compared with a parent-matched expectation (each task's
parent's E036 correct rate). The text checks matter for SFT.

| | DeepSeek API (800 random `e036` rows) | DeepInfra `e038` (1,246) |
|---|---|---|
| correct (parent-matched expectation) | 71.6% | 70.9% (expected 71.0%) |
| train_pass / train_pass but wrong | 82.9% / 18.7% | 80.8% / 18.5% |
| output tokens mean / median / p95 | 15.3k / 10.1k / 44.3k | 15.0k / 9.7k / 42.8k |
| length-stopped turns | 4 | 2 |
| assistant turns without thinking / attempts without any thinking | 4.7% / 0 | 4.7% / 0 |
| repetition (zlib ratio of thinking, median / min) | 0.350 / 0.300 | 0.351 / 0.301 |
| attempts with CJK characters (>50) | 0 (0) | 2 (0) |
| tool errors per call | 0.015 | 0.017 |

The 99-pair paired probe (`e037`) agrees: both correct 69, only DeepInfra 4, only DeepSeek 2.

## Harness (persistent-v6, "s20x60 + minimal header")

- Grid mode, with a `run_python` tool (a persistent session) and `submit_answer(expr)`. The answer grid is read from
  the session.
- Prompt `lean2sub`: the ARC Prize prompt, JSON grids, plus a short tool-use suffix. There is **no task summary and
  there are no preloaded helpers**.
- System message `--effort-header minimal`: "Reasoning Effort: Minimal. Keep your thinking short and terse …". It is
  the first message of every transcript.
- Budget: 60k hard and 20k soft output tokens per attempt (strong wording), 64k per turn, at most 40 turns, and a
  turns note 5 turns before the cap.
- Host-side train check after each attempt (`train_pass`, `train_frac`). It uses no model tokens.

The exact config per config name is in `run_config.json`.

## Schema (`rollouts/*.parquet`, one row per attempt)

| Column | Meaning |
|---|---|
| `task_id`, `pair`, `attempt` | Task id, test-pair index, attempt index |
| `batch`, `source`, `parent` | Pool batch, data source, NVARC parent folder (ARC task id) |
| `correct` | Whether the predicted grid equals the ground truth |
| `train_pass`, `train_frac` | Whether, and on what fraction, the answer program reproduces the training pairs |
| `pred` | The predicted grid (JSON) |
| `answer_source`, `stop_reason`, `error` | How the answer was obtained, why the attempt ended, and any API error |
| `turns`, `tool_calls` | Counts per attempt |
| `prompt_tokens`, `cache_hit_tokens`, `completion_tokens`, `seconds`, `est_cost_usd` | Usage per attempt |
| `model`, `provider` | Model id; `DeepSeek API` or `DeepInfra (OpenRouter, fp8)` |
| `messages` | The full transcript as a JSON string: OpenAI chat format with `reasoning_content`, `tool_calls` and tool messages |
| `events` | Per-turn metadata (usage, finish_reason, tool status), as a JSON string |

## Task pool (`tasks/`): reproducible and extendable

- `manifest.jsonl` has 12,500 rows: arc2_train (1,000), nvarc_b0 (4,000) and nvarc_b1 (7,500). Each row gives the
  source file and sha1, and the train / test / unused pair indices.
- `nvarc_order.txt` lists all 47,337 `nvarc_training` puzzles in a fixed seeded order (seed 2026).
- `build_dataset.py`:
  - `--batch NAME --n N` takes the next eligible puzzles in that order.
  - `--balance-parents --count-pairs FILE` fills a batch parent by parent: each pick goes to the parent folder with the
    fewest tasks so far, counting the tasks in FILE.
  - `nvarc_b1` was built with `--n 7500 --balance-parents --count-pairs e031_pairs_nvarc_b0.txt`.
- `e036_pairs_nvarc_b1.txt` holds the first 5,600 `nvarc_b1` tasks, all run. `e036_pairs_nvarc_b1x.txt` holds the next
  1,200, of which 655 were run before the credit ran out. `e038_pairs_nvarc_b1r.txt` holds the remaining 1,246
  (including the 1 errored E036 task). All 7,500 `nvarc_b1` tasks now have a successful rollout.
- **Coverage:** together with `e031_nvarc_b0`, every one of the 716 NVARC parents has 4–13 successful rollouts, median 13 (parents with few puzzles have fewer).
- `*_challenges.json` / `*_solutions.json` are Kaggle-format task files. ARC-2 training tasks are referenced by id; use
  the official Kaggle or GitHub files.

## About NVARC parents

All 716 `nvarc_training` parent folders are Kaggle ARC-AGI-2 training tasks, and they come from ARC-AGI-1. Each NVARC
puzzle is an LLM mix of two seed-task summaries (NVARC paper, section 2.2), filed under one parent. So the parents are
not pure rule families.

## Contamination

NVARC puzzles with any grid matching an ARC-AGI-2 evaluation grid (canonical hash, invariant to D8 and colour
relabelling) were skipped when the pool was built. `nvarc_full`, which is derived from the eval tasks, is not used.

## Licence

Task data follows the licences of its sources (ARC Prize / Kaggle, NVARC). The model outputs were generated with
DeepSeek's API and DeepInfra (via OpenRouter).
