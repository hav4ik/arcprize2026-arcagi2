# E025: DeepSeek Flash V3 rollouts on a 5000-task training pool

## Goal
Collect tool-use rollouts, plus per-task difficulty, for SFT and data selection. The model is DeepSeek Flash with the E017 V3 harness (`lean2roles` + `--task-summary` + `--helpers` + `--train-check`, sub80s40 budgets), 1 attempt per test pair.
- Batch `arc2_train`: the 1000 Kaggle ARC-AGI-2 training tasks (1076 test pairs).
- Batch `nvarc_b0`: 4000 tasks sampled from NVARC synthetic `nvarc_training`.
  - Each task has n_train pairs drawn from the ARC-2 training distribution of train-pair counts, plus 1 test pair.
  - All 716 parents of `nvarc_training` are ARC-AGI-1 tasks.

## Task pool (`build_dataset.py`, deterministic, extendable)
- `data/nvarc_order.txt`: all 47,337 nvarc_training puzzles in one fixed order (seed 2026). Each new batch takes the next eligible puzzles: `build_dataset.py --batch nvarc_b1 --n N`.
- `data/manifest.jsonl`: one row per task with:
  - batch, source, source file and sha1;
  - `order_rank`;
  - `train_idx` / `test_idx` / `unused_idx` into the puzzle's pair list (about 30 pairs per puzzle). The unused pairs can serve as extra test pairs later.
- `data/<batch>_{challenges,solutions}.json`: Kaggle-format input for the harness.
- Skips:
  - any grid matching an ARC-AGI-2 eval grid (D8 + colour invariant; 5 puzzles in nvarc_b0);
  - grids larger than 30×30;
  - repeated inputs among the chosen pairs.

## Run
`launch.sh [batch ...]` runs the batches in order, and re-running resumes them. The kill-guard stops launching below a $40 balance; 24 workers.
Outputs: `evals/results/tools_persistent_grid_lean2roles_deepseek-flash_low_e025_<batch>.jsonl`, with transcripts in the folder of the same name and the log in `evals/results/log_e025_<batch>.txt`.

## Cost
- **Estimate:** 5076 attempts × $0.027 (V3 off-peak per attempt on eval) ≈ $137: about $29 for arc2_train and $108 for nvarc_b0. Training tasks should be cheaper.
- **Approved by the user:** 2026-10-16. Balance at launch: $192.37.
- **Launched:** 2026-10-16 13:45 UTC (Friday, off-peak).

## Results

### arc2_train (done 2026-10-16 14:57 UTC, 72 min, all off-peak, 24 workers)
- 1076 attempts, 1 API error (APITimeoutError on 4c5c2cf0.0, counted as wrong), 7 attempts with no answer, 5189 tool calls. Recorded cost **$9.32** ($0.0087/attempt, about a third of the $29 estimate). Balance went from $192.37 to about $182.9.
- ARC-1 origin = the 767 task ids that are also in ARC-AGI-1 training/evaluation; ARC-2 only = the other 233.
- Columns: pass@1 (pair) is per test pair; pass@1 (task) averages over pairs within a task first; 'all pairs solved' is the strict task score. P(correct given train_pass) is precision of the host train check.

| subset | tasks | pairs | pass@1 (pair) | pass@1 (task) | all pairs solved | train_pass | P(correct given train_pass) | correct w/o train_pass | out-tok mean | out-tok median | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1000 | 1076 | 94.0% | 94.1% | 93.8% | 97.0% | 95.6% | 13 | 12,119 | 6,170 | $9.32 |
| ARC-1 origin | 767 | 802 | 95.4% | 95.3% | 95.2% | 97.9% | 96.7% | 6 | 9,516 | 4,392 | $5.62 |
| ARC-2 only | 233 | 274 | 89.8% | 90.2% | 89.3% | 94.5% | 92.3% | 7 | 19,738 | 13,246 | $3.70 |

Pair-level difficulty by output tokens (correct / train_pass):

| out-tokens | pairs | correct+train_pass | correct only | train_pass only (wrong) | neither | pass@1 |
|---|---|---|---|---|---|---|
| <5k | 464 | 457 | 0 | 5 | 2 | 98.5% |
| 5-10k | 216 | 207 | 1 | 8 | 0 | 96.3% |
| 10-20k | 209 | 196 | 1 | 12 | 0 | 94.3% |
| 20-40k | 115 | 101 | 2 | 12 | 0 | 89.6% |
| >=40k | 72 | 37 | 9 | 9 | 17 | 63.9% |

Task-level difficulty tiers (mean output tokens over the task's pairs):

| tier | all | ARC-1 origin | ARC-2 only |
|---|---|---|---|
| easy (correct+tp, <10k tok) | 624 (62%) | 549 (72%) | 75 (32%) |
| medium (correct+tp, 10-40k) | 271 (27%) | 159 (21%) | 112 (48%) |
| hard (correct+tp, >=40k) | 31 (3%) | 16 (2%) | 15 (6%) |
| correct, train_check fails | 12 (1%) | 6 (1%) | 6 (3%) |
| partial (some pairs wrong) | 6 (1%) | 2 (0%) | 4 (2%) |
| unsolved | 56 (6%) | 35 (5%) | 21 (9%) |

- Takeaways: training tasks are easy for V3: 94% pass@1 at 12k output tokens per attempt, against 69% at 39k on the 120 eval tasks (E017). ARC-2-only tasks cost 2x the tokens of ARC-1-origin tasks and fail about twice as often. Correctness drops sharply past 40k output tokens (64% there vs 98% below 5k), so token count is a usable difficulty signal. Only 13 correct answers lacked train_pass, and 46 train-passing answers were wrong.

### nvarc_b0 (done 2026-10-16 21:04 UTC, all off-peak)
- Ran 14:57–17:34 UTC with 24 workers (1342 attempts). The coordinator then restarted it, with the user's approval, at 64 workers, which gave about 1.8x the throughput (about 14 attempts/min against 8). Per-stream API latency rose from about 6.8 to 9.7 s per 1k output tokens, with no throttling or 429s. Sandbox memory peaked at about 4.2 GiB.
- 4000 attempts with 4 API errors (APIConnectionError, counted as wrong, 0.1%) and 170 attempts with no answer (4.3%; 156 hit the 80k output budget, 19 hit max_turns). 29,376 tool calls. Recorded cost **$82.65** ($0.0207/attempt), against an estimate of $108.
- Parents: 696 distinct nvarc_training parents, 5.7 children each on average. All of them are ARC-1-origin tasks that are also in arc2_train.

| subset | tasks | pairs | pass@1 (pair) | pass@1 (task) | all pairs solved | train_pass | P(correct given train_pass) | correct w/o train_pass | out-tok mean | out-tok median | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 4000 | 4000 | 68.2% | 68.2% | 68.2% | 85.8% | 77.7% | 62 | 30,028 | 21,492 | $82.65 |

| out-tokens | pairs | correct+train_pass | correct only | train_pass only (wrong) | neither | pass@1 |
|---|---|---|---|---|---|---|
| <5k | 581 | 543 | 1 | 35 | 2 | 93.6% |
| 5-10k | 606 | 551 | 1 | 50 | 4 | 91.1% |
| 10-20k | 743 | 634 | 0 | 108 | 1 | 85.3% |
| 20-40k | 785 | 586 | 3 | 188 | 8 | 75.0% |
| >=40k | 1285 | 353 | 57 | 384 | 491 | 31.9% |

| tier | all |
|---|---|
| easy (correct+tp, <10k tok) | 1094 (27%) |
| medium (correct+tp, 10-40k) | 1220 (30%) |
| hard (correct+tp, >=40k) | 353 (9%) |
| correct, train_check fails | 62 (2%) |
| unsolved | 1271 (32%) |

- Parents with at least 3 children (589), by the share of children solved (0–20 / 20–40 / 40–60 / 60–80 / 80–100%): 5 / 54 / 124 / 217 / 189.
- Children of the 665 parents that V3 solved in arc2_train score 68.1%. Children of the 31 parents it failed score 72.2%. NVARC children are therefore much harder than their ARC-1 parents (68% vs 95%), and parent success does not predict child success.
- **Takeaways:**
  - nvarc_b0 is far harder than arc2_train: 68.2% pass@1 at 30k output tokens per attempt (median 21k), and 32% of tasks are unsolved.
  - The train check is much weaker here. P(correct given train_pass) is 77.7%, and 765 train-passing answers are wrong (19% of attempts). Likely causes are synthetic tasks whose test pair is under-determined by the train pairs, or noisy generator outputs. Treat a wrong-but-train-passing child as a possible label issue before using it as a negative in SFT.
  - Above 40k output tokens, pass@1 is only 32%: 1285 attempts there cost $53.5 of the $82.65 (65%).

### Totals
- 5076 attempts, recorded **$91.97** ($9.32 + $82.65), all off-peak, against the $137 estimate. Balance went from $192.37 to $100.02 (delta $92.35). The extra ~$0.4 is mostly the in-flight attempts re-billed at the 64-worker restart, plus the errored attempts.
- 5 attempts failed with API errors (1 in arc2_train, 4 in nvarc_b0). They are not retried yet. Re-running `launch.sh` retries them, because resume re-queues rows that carry `error`; that costs about $0.1.

- **Hugging Face (2026-10-16):** private dataset `livobereshka/arc-agi2-deepseek-v3-rollouts-e025`. It holds rollouts as parquet (one row per attempt, with the transcript in `messages`), the task pool files, `run_config.json` and a dataset card.

## Notes for using this data (E026, 2026-10-17)
- **The summary is baked in.** These rollouts were collected with `--task-summary`, and the traces quote it about 23 times per attempt. A Qwen model trained on them must get the same summary at inference.
- **Next batches drop it.** Later batches are planned without `--task-summary`; see `notes/2026-10-17_distillation_data_findings.md`.
- **Suggested filters:**
  - keep answers from train-passing programs;
  - prefer a first tool call at ≤5k tokens;
  - down-weight turns of 16k+ tokens;
  - re-check NVARC answers that pass train but are wrong, using the unused pairs in the manifest.
