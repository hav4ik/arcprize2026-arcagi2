# E036: parent-balanced NVARC rollouts (Flash-1017, s20x60 + minimal header)

Started 2026-10-20 19:29 UTC (Tuesday, off-peak). Goal: a diverse distillation / pruning-calibration set. This tops up
E031's 2,500 rollouts (same harness) with new NVARC tasks spread evenly over NVARC's 716 parent rules. Budget: all
current DeepSeek credit ($75.39 at launch), with `--max-balance-stop 4`.

- **Task pool:** `nvarc_b1` in `experiments/E025_train_rollouts/data/`. 7,500 tasks were built with
  `build_dataset.py --batch nvarc_b1 --n 7500 --balance-parents --count-pairs <E031 nvarc pair list>`: each pick goes
  to the parent with the fewest tasks so far, counting E031's. 28 puzzles were skipped for eval-grid matches.
- **This run:** the first 5,600 of `nvarc_b1` in manifest (balance) order, in `pairs_nvarc_b1.txt`, sized to the
  credit at ~$0.0115 per rollout (E031 NVARC). With E031, every parent reaches 10 NVARC rollouts, except the 4 parents
  with fewer than 10 puzzles. The other 1,900 `nvarc_b1` tasks are kept for a later top-up.
- **Harness:** identical to E031 (`launch.sh`): lean2sub, s20x60 (60k hard / 20k soft, strong wording), `--effort-header
  minimal`, `--turns-note 5`, `--train-check`, grid mode, low effort, 1 attempt, 64 workers.
- **Results:** `evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_e036_nvarc_b1.jsonl`, log
  `evals/results/log_e036_nvarc_b1.txt`.
- **Diversity note:** all 716 NVARC parent folders are Kaggle ARC-2 training tasks (verified). Each NVARC puzzle is an
  LLM mix of two seed-task summaries (NVARC paper, section 2.2, Instruct-SkillMix), filed under one parent; the paper does
  not say which seed names the folder. So balancing by parent spreads over one side of each pair, not over pure
  families. Mini-ARC (147) and ConceptARC (160) were in NVARC's pretraining data but never used as mixing seeds; none
  of our rollouts cover them yet.
- **Workers:** started at 64, restarted at 128 at 19:33 UTC (94 attempts done; the in-flight ones were re-billed).
- **Top-up `nvarc_b1x`:** the next 1,200 balanced tasks (manifest positions 5600–6800, `pairs_nvarc_b1x.txt`, tag
  `e036_nvarc_b1x`). It started at 20:46 with 32 workers next to b1 and was paused after 21 attempts, so that b1 could finish whole. `resume_b1x.sh` resumed it
  at 64 workers once b1 ended (23:22). The guard fired at 23:47; the process was killed at 23:48 (the real balance was already
  below $4, and the in-flight partials were dropped).

## Results

Analysis: `scripts/analyze.py` (reuses E031's `stats`). b1 ran 19:29–23:22 UTC, b1x 23:22–23:48; all off-peak.

| run | attempts | errors | pass@1 | train_pass | P(c\|tp) | P(c\|¬tp) | out tok mean / median / p90 | recorded $ |
|---|---|---|---|---|---|---|---|---|
| nvarc_b1 | 5,600 / 5,600 | 1 | 70.7% | 80.9% | 80.7% | 28.1% | 15.8k / 10.2k / 38.5k | 65.24 |
| nvarc_b1x | 655 / 1,200 | 0 | 72.7% | 86.0% | 80.5% | 25.0% | 14.5k / 9.3k / 35.1k | 7.12 |
| **E036 total** | **6,255** | 1 | **70.9%** (4,435) | 81.5% | 80.7% | 27.9% | 15.6k / 10.1k / 38.1k | **72.36** ($0.0116/att) |

- **Errors:** 1 (`APIConnectionError`, fe9372f3_60c09cac), not retried. 0 rate-limit errors at 128 workers.
- **Stops:** 6,211 submitted, 32 output_budget, 10 max_turns, 1 final. 97.6% of correct attempts used ≤40k output tokens.
- **Training traces:** 4,112 E036 attempts are both correct and train_pass (E031: 1,907).
- **Cost:** the recorded total is $72.36. The balance went from $75.39 to $1.53, stable from 23:51, so $73.86 was billed. The ~$1.50
  gap is the in-flight attempts re-billed at the 19:33 restart and those killed at 23:48.
- **Combined dataset:** E031 2,500 (1,070 arc2_train + 1,430 nvarc_b0) + E036 6,255 = **8,755 rollouts**; correct + train_pass
  6,019 (973 arc2_train + 934 nvarc_b0 + 4,112 E036), of which 5,046 are NVARC.
- **Parent coverage (716 NVARC parents):**
  - E036 alone covers all 716; rollouts per parent are min 1 / median 9 / max 12.
  - E031-nvarc + E036: min 4 / median 11 / max 12. Histogram {4:1, 5:1, 8:2, 9:56, 10:255, 11:207, 12:194}; the low ones are parents with few
    puzzles. Before E036, E031 covered only 591 parents (median 2).
  - Per-parent accuracy (combined): mean 0.708, median 0.727. **No parent has 0 correct** (E031 alone had 67), 59 parents are
    below 50%, and 31 are all correct.
- **Unrun tasks:** b1 ran every task. b1x left 545 of 1,200 unrun: the task-id tail 74dd1130_… to ff72ca3e_…, 330 parents, at most
  2 per parent. Parents with IDs from 74dd1130 up average 10.1 combined rollouts, against 11.5 below. The skew is mild,
  and every parent has ≥9 except the 4 parents with few puzzles.
  To finish b1x later, re-run `resume_b1x.sh`-style (same tag; it resumes).
- **128 vs 64 workers:** no speed-up. Output tokens/s per attempt dropped from ~100 (64 workers; E031 at 60: 108) to 50, so
  pooled throughput stayed ~6.4k tok/s for the account. Median wall time per attempt was 203 s at 128 workers, against 90 s in E031. The b1x
  attempts at 64 workers alone were back at 102 tok/s. DeepSeek appears to cap total throughput per account, so more
  workers only add in-flight work that is lost if a run is killed.
- **Balance-guard lag:** the guard polls every 5 min and the balance it reads lags. When it fired, the real balance was
  already ~$2.6, against the $4 floor. For a near-zero budget, set the stop ~$3 higher, or kill by recorded cost.

On Hugging Face (private): `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036`.
