# E011: Pareto search over accuracy and tokens for the Flash tool harness

## Goal
Find the cheapest config that is also the most accurate. This is a Pareto search over (accuracy, output tokens per attempt) on the full public eval (120 tasks / 167 pairs). Tokens count twice: they are the API cost and they decide Kaggle feasibility.

- **Primary metrics:**
  - pass@2: the mean over attempt pairs, comparable with E004-GL's 76.4%;
  - pass@1.
- **Rule:** frontier points need at least 2 full-set samples.
- **Recommendations wanted:**
  - (a) best accuracy overall;
  - (b) best accuracy at about 35k tokens per attempt or less;
  - (c) the cheapest config within about 1 pp of E004-GL.

Earlier results come from E009 (`experiments/E009_tool_token_efficiency/README.md`) and E010 (`experiments/E010_half_reasoning/README.md`). This phase tested levers meant to buy accuracy with the tokens saved:
1. verified submit;
2. high effort with the sub80s40 budgets;
3. diverse second attempts, i.e. mixed-config attempt pairs.

## New harness options (harness `persistent-v4`, opt-in)
- `--verify-submit`: when `submit_answer` gets an expression that uses `test_input` (e.g. `solve(test_input)`), the harness checks it on the training pairs before accepting it:
  - it re-evaluates the expression with `test_input` bound to each training input, using host copies of the inputs;
  - it compares the results with the host's training outputs;
  - on a mismatch it rejects once, saying which pairs failed and how (shape or number of wrong cells). Resubmitting exactly the same code accepts it as a best guess.
  - Records gain `submit_verified` and `submit_rejections`.
- Prompt `lean2subv` = `lean2` plus: put the rule in `solve(grid)` and submit exactly `solve(test_input)`; the harness checks `solve` on every training input first.

Tests: `ARC_SANDBOX_URL=<url> .venv/bin/python -m unittest discover -s evals/tests` (44 tests, including the train check against the real sandbox).

Tools:
- `evals/frontier.py`: frontier table and mixed-config pass@2;
- `evals/paired_vs_baseline.py`: paired diffs with bootstrap CIs, pass@2, long-tail breakdown, `--table`, `--attempts`.

## Runs in this phase (all full set, 2 samples each, off-peak 2026-10-11 20:30–23:42 UTC)
Estimate before launch: 334 attempts × ~42k (low effort) or ~48k (high effort) × $0.60/M, plus ~15% input, ≈ $9.5 / $12 per 2-sample run.

| Run | Config | Recorded cost |
|---|---|---|
| sub96n sample 2 (E009 file) | 96k hard cap, budget notes from 50% used, `lean2sub` | $5.5 |
| sub80s40v | sub80s40 + `--verify-submit` (`lean2sub` prompt) | $9.24 |
| subv80s40 | sub80s40 + `--verify-submit` + `lean2subv` prompt (submit `solve(test_input)`) | $9.15 |
| sub80s40h | sub80s40 at **high** effort | $10.36 |

Plus the second samples of E010's s20x60 and s25x80; their costs are in E010.

## Frontier (full set, 167 pairs; `evals/frontier.py`, `evals/paired_vs_baseline.py`)
- **pass@1** is the mean over samples.
- **pass@2** is the mean over attempt pairs, the same convention as E004-GL's 76.4%.
- **Diffs** are paired vs E004-GL, with bootstrap over tasks.
- **Tokens** are output tokens per attempt, all turns.

| Config | n | pass@1 (per sample) | Δ pass@1 [95% CI] | pass@2 | Δ pass@2 [95% CI] | tok/attempt mean / median | Correct rollouts p50 / p90 | $/sample |
|---|---|---|---|---|---|---|---|---|
| E004-GL (baseline) | 4 | 66.0% (66.7 / 70.0 / 64.4 / 62.8) | — | 76.4% | — | 49.5k / 41.4k | 28.7k / 72.1k | $5.59 |
| E004 grid/lean2/high | 2 | 68.3% (67.1 / 69.4) | +2.3 | 78.2% | +1.8 | 55.2k / 48.5k | 35.0k / 79.8k | $6.15 |
| **sub96n** | 2 | **68.3%** (69.9 / 66.8) | +2.4 [−2.2, +6.9] | **78.5%** | +2.1 [−2.6, +6.9] | 45.7k / 39.0k | 29.4k / 68.2k | $5.06 |
| **sub80s40** (E009) | 3 | 67.8% (67.8 / 70.7 / 64.9) | +1.9 [−2.5, +6.1] | 77.5% | +0.9 [−3.7, +5.5] | **40.7k** / 37.6k | 25.4k / 62.8k | $4.50 |
| sub80s40v | 2 | 66.5% (67.4 / 65.6) | +0.5 [−4.2, +4.8] | 77.2% | +0.8 [−3.8, +5.2] | 41.9k / 37.6k | 26.8k / 66.6k | $4.62 |
| subv80s40 | 2 | 65.7% (62.8 / 68.6) | −0.3 [−4.8, +4.3] | 76.5% | +0.1 [−4.5, +4.9] | 41.4k / 37.4k | 26.7k / 63.4k | $4.58 |
| sub80s40h (high effort) | 2 | 66.0% (64.3 / 67.6) | +0.0 [−4.5, +4.4] | 77.5% | +1.1 [−4.2, +6.6] | 47.1k / 48.0k | 31.3k / 65.6k | $5.18 |
| s25x80 (E010) | 2 | 64.7% (61.9 / 67.4) | −1.3 [−6.6, +3.7] | 75.3% | −1.1 [−6.9, +4.7] | 35.9k / 34.1k | 25.3k / 58.1k | $3.94 |
| **s20x60** (E010) | 2 | 62.4% (63.6 / 61.3) | −3.5 [−8.3, +1.2] | **76.5%** | +0.1 [−4.9, +5.0] | **32.9k** / 30.4k | 24.1k / 47.8k | $3.58 |
| s20x40 (E010) | 1 | 47.8% | −17.8 [−25.2, −10.7] | — | — | 29.3k / 34.2k | 21.1k / 36.7k | $3.08 |
| sub64a (E009) | 1 | 58.2% | −7.8 [−14.3, −1.5] | — | — | 38.7k / 40.7k | 24.7k / 49.3k | $4.27 |

**Mixed-config attempt pairs** (pass@2 of one attempt from each config, averaged over all sample combinations):

| Pair | pass@2 | Tokens per pair |
|---|---|---|
| sub80s40 + sub96n | 78.4% | 86.4k |
| sub96n + sub80s40h | 77.9% | 92.8k |
| s25x80 + sub80s40h | 76.7% | 83.0k |
| sub80s40 + sub80s40h | 76.4% | 87.8k |
| s20x60 + sub80s40 | 75.3% | 73.5k |
| s20x60 + sub80s40h | 74.4% | 79.9k |

None of these beats two attempts of the same config; for comparison, E004-GL + E004-high gives 78.2% at 104.6k.

**The pass@2 Pareto front** (tokens per attempt → pass@2) is s20x60 (32.9k → 76.5%), then sub80s40 (40.7k → 77.5%), then sub96n (45.7k → 78.5%). Every point on it uses fewer tokens than E004-GL and is at or above its pass@2.
- On pass@1 the front is s20x60 (62.4%), s25x80 (64.7%), sub80s40 (67.8%) and sub96n (68.3%).
- One sample swings by ±3 pp, so the pass@1 gaps between sub80s40, sub96n and E004-high are within noise. None of the frontier diffs vs E004-GL is significant except the losses of s20x40 and sub64a.

## What the levers did
1. **Verified submit buys nothing.**
   - With the `lean2sub` prompt the model submits `answer` (a variable) 90% of the time, so the check never runs: 3 rejections in 334 attempts.
   - With `lean2subv` the check runs on 76% of submissions (233 of 308). It rejected 16 attempts; 5 of those ended correct after revision.
   - Of the verified submissions, 41 of 224 are wrong. Their `solve` reproduces every training pair but not the test, the out-of-distribution test case.
   - Net: pass@1 65.7% and pass@2 76.5%, at the same tokens as sub80s40. The model already checks train before submitting (as in E004 code mode), so a gate adds little. Verified submits are 82% precise (183/224) and unverified ones 39% (29/75), which may be useful as a confidence signal for selection or SFT filtering, but not as a gate.
2. **High effort with the sub80s40 budgets buys nothing.**
   - It costs 16% more tokens: 47.1k vs 40.7k.
   - It leaves 16% of attempts unanswered (55/334), against 6% at low effort (30/498), because the 80k cap binds more often.
   - pass@1 66.0%, pass@2 77.5%.
3. **A diverse second attempt (mixed config or effort) does not beat a duplicate.** The best mixed pair, sub80s40 + sub96n, gives 78.4%, equal to sub96n twice (78.5%).
4. **What does buy accuracy is budget headroom for the long tail.**
   - sub96n (96k cap, notes only after 50% used) has the best pass@1 and pass@2 and still uses 7.5% fewer tokens than E004-GL.
   - The `submit_answer` hand-off is what lets it be both more accurate and cheaper.

## Recommendation
All configs use grid mode, low effort, `--prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000`.

| Goal | Pick | pass@1 / pass@2 | Tokens / attempt | Extra flags |
|---|---|---|---|---|
| (a) best accuracy overall | **sub96n** | 68.3% / 78.5% | 45.7k (−7.5%) | `--max-total-output 96000 --budget-notes 0.5` |
| (b) best accuracy at ≲35k | **s20x60** for pass@2; s25x80 if pass@1 matters | s20x60 62.4% / 76.5% at 32.9k; s25x80 64.7% / 75.3% at 35.9k | 32.9k (−34%) | `--max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 20000 --soft-wording strong` |
| (c) cheapest within ~1 pp of E004-GL | on pass@2 (the Kaggle metric): **s20x60**, +0.1 pp at −34% tokens; on pass@1: **sub80s40**, +1.9 pp at −18% | — | 32.9k / 40.7k | sub80s40: `--max-total-output 80000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 40000` |

```bash
cd evals
# (a) sub96n
../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool --final-mode session \
  --lenient-parse --max-tokens 64000 --max-total-output 96000 --budget-notes 0.5 --price-multiplier 1 --attempts 2
# (b)/(c) s20x60
../.venv/bin/python run_deepseek_tools.py --mode grid --effort low --prompt lean2sub --answer-tool --final-mode session \
  --lenient-parse --max-tokens 64000 --max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 \
  --budget-soft 20000 --soft-wording strong --price-multiplier 1 --attempts 2
```

**For SFT data:**
- s20x60 gives the most correct rollouts per dollar: $3.58 per sample, and 53% of its correct rollouts are ≤25k tokens (p90 47.8k).
- sub96n gives the most correct rollouts per task.
- Pooling samples from several configs and keeping the shortest correct rollout per task is the efficient way to build a ≤22k-token training set.
- In any config, about half of the correct rollouts already fit a 25k budget.

## Spend and balance
DeepSeek list prices, off-peak; nothing was in flight during a peak window.

| Item | Recorded cost |
|---|---|
| E009 | $54.97 (includes the sub96n second sample, $5.5, run in this phase) |
| E010 | $24.75 |
| E011 runs | $28.75 |
| **This phase in total** | **$34.2** (E011 runs + sub96n sample 2), against the $60 approved |
| **Total recorded** | **$108.47** |

**Balance reconciliation:**
- Balance went from $92.37 (14:50) to $25.07 (20:14), a drop of $67.30.
- The user topped up to $123.79 (20:22); the balance was $82.92 at 23:43, a drop of $40.87.
- Total drop: $108.17, against $108.47 recorded.

The balance lags in-flight calls by minutes, which is why interim projections looked off. The recorded `est_cost_usd` matches billing, so the earlier "balance fell faster than projected" was lag plus my underestimate of in-flight work, not extra billing. Partial outputs of killed or errored attempts (the stopped sub64a and thinking-off runs, API connection errors) are not in the recorded total but are within that ~$0.3 difference.
