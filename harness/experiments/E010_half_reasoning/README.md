# E010: Half reasoning (≤25k output tokens per attempt)

## Goal
Find the best accuracy at **≤25k mean output tokens per attempt**, about half of E004-GL's 49.5k, and map the accuracy-vs-tokens trade-off (Pareto front) between about 20k and 40k.
- **Why 25k:** this is the Kaggle pass@2 budget for a 27B student on 4×L4 (about 22k tokens per attempt).
- **Baseline E004-GL:** pass@1 66.0% over 4 samples, 49.5k tokens per attempt.
- **Accuracy loss** is acceptable if it is the price of halving the tokens; the aim is the smallest loss.
- **Budget:** ≤$22, balance kept ≥$15 (`--max-balance-stop 16`), off-peak only.

It builds on E009 (`experiments/E009_tool_token_efficiency/README.md`), which found three things:
- `submit_answer` with a session fallback and always-on budget notes is the efficient harness.
- Hard caps cut off the long tail of slow but solvable pairs: sub64a scored −7.8 pp on the full set.
- A soft target nudges attempts to finish without truncating them.

## New harness options (harness `persistent-v4`, opt-in)
- `--think-turns N`: thinking is enabled only on the first N assistant turns. Later turns send `thinking: disabled`.
  - DeepSeek accepts the toggle mid tool-loop; this was checked with one tiny call.
  - `-1`, the default, keeps thinking on throughout.
- `--soft-wording strong`: once usage passes `--budget-soft`, the note in each tool output reads "Past the target: submit your best answer now with submit_answer, unless one more experiment will very likely make every training pair pass". The E009 wording was "submit as soon as a rule fits every training pair".

Tests: `ARC_SANDBOX_URL=<url> .venv/bin/python -m unittest discover -s evals/tests` (41 tests).

## Common settings
- Grid mode at low effort, with a 64k per-turn cap.
- Harness flags: `--prompt lean2sub --answer-tool --final-mode session --lenient-parse --budget-notes 0.01 --budget-warn 0.75`.
- Variant names:
  - `sXXxYY` = soft target XX k with strong wording (`--budget-soft`), hard total YY k (`--max-total-output`);
  - `ntN` = `--think-turns N`.
- Dev set: E009's 56 stratified pairs (`dev_pairs.txt`). Each dev run is 1 sample, compared paired against E004-GL's 4-sample mean on the same pairs.
- Launcher: `experiments/E009_tool_token_efficiency/run_variant.sh`.

## Dev screening (56 pairs, 1 sample each, paired vs E004-GL's 4-sample mean on the same pairs)
The three thinking-off runs were stopped after 20–25 pairs because they were far behind.

| Variant | Pairs | pass@1 | E004-GL | diff [95% CI] | tok/attempt mean / median | vs E004-GL | Correct rollouts p50 / p90 | Recorded cost |
|---|---|---|---|---|---|---|---|---|
| nt0s20x60 (no thinking at all) | 25 | 27.3% | 72.7% | −45.5 [−65.9, −23.3] | 18.5k / 20.2k | −64% | 13.8k / 23.7k | $0.38 |
| nt1s40x80 (thinking on turn 1 only) | 20 | 36.1% | 72.9% | −36.8 [−56.9, −17.4] | 23.3k / 24.1k | −50% | 17.4k / 24.6k | $0.37 |
| nt3s20x60 (thinking on turns 1–3) | 23 | 40.5% | 68.5% | −28.0 [−47.6, −9.5] | 19.9k / 19.1k | −56% | 19.0k / 31.0k | $0.41 |
| s10x40 | 51 | 50.7% | 67.6% | −16.9 [−29.6, −5.4] | 26.3k / 24.5k | −42% | 17.2k / 37.1k | $0.89 |
| s15x45 | 54 | 48.0% | 65.4% | −17.4 [−30.4, −4.6] | 31.0k / 31.9k | −35% | 26.0k / 42.2k | $1.11 |
| s20x40 | 54 | 55.0% | 66.6% | −11.6 [−21.5, −2.6] | 26.9k / 24.8k | −42% | 20.5k / 34.6k | $0.97 |
| s20x60 | 51 | 60.9% | 67.4% | −6.5 [−19.3, +6.0] | 32.7k / 29.9k | −29% | 22.6k / 48.5k | $1.16 |
| s25x80 | 53 | 67.0% | 63.7% | +3.4 [−4.7, +11.0] | 35.9k / 29.4k | −25% | 28.5k / 58.4k | $1.27 |

**Thinking off does not work for Flash.**
- Without reasoning between calls, the model makes many tiny tool calls: 40 turns of one cell each in nt1.
- It hits the 40-turn limit without an answer (9 of 20 attempts in nt1) and rarely converges.
- Tokens drop, but accuracy falls by 28–45 pp.

## Full-set results (120 tasks / 167 pairs; paired vs E004-GL, bootstrap over tasks)
Estimates before launch were 167 × ~30k × $0.60/M ≈ $3.0 output + ~20% input ≈ $3.6 per sample. Recorded costs per sample were $3.0–3.9.

| Config | Samples | pass@1 (per sample) | diff [95% CI] | pass@2 (mean over attempt pairs) | diff [95% CI] | tok/attempt mean / median | vs 49.5k | Correct rollouts p50 / p90 | Long-tail pairs* | $/sample |
|---|---|---|---|---|---|---|---|---|---|---|
| E004-GL | 4 | 66.0% | — | 76.4% | — | 49.5k / 41.4k | — | 28.7k / 72.1k | 71.4% | $5.59 |
| E009 sub80s40 | 3 | 67.8% (67.8 / 70.7 / 64.9) | +1.9 [−2.5, +6.1] | 77.5% | +0.9 [−3.7, +5.5] | 40.7k / 37.6k | −18% | 25.4k / 62.8k | 63.1% | $4.50 |
| s25x80 | 2 | 64.7% (61.9 / 67.4) | −1.3 [−6.6, +3.7] | 75.3% | −1.1 [−6.9, +4.7] | 35.9k / 34.1k | −27% | 25.3k / 58.1k | 56.2% | $3.94 |
| **s20x60** | 2 | 62.4% (63.6 / 61.3) | −3.5 [−8.3, +1.2] | **76.5%** | **+0.1 [−4.9, +5.0]** | **32.9k** / 30.4k | **−34%** | 24.1k / 47.8k | 57.1% | $3.58 |
| s20x40 | 1 (8 API errors not retried) | 47.8% | −17.8 [−25.2, −10.7] | — | — | 29.3k / 34.2k | −41% | 21.1k / 36.7k | 24% | $3.08 |
| E009 sub64a (hard cap, for reference) | 1 | 58.2% | −7.8 [−14.3, −1.5] | — | — | 38.7k / 40.7k | −22% | 24.7k / 49.3k | 44.6% | $4.27 |

\* Long-tail pairs are the 56 pairs whose E004-GL correct rollouts averaged more than 40k tokens; the column is pair accuracy.

For comparison, simple hard caps simulated on E004-GL's records (an attempt over the cap counts as unanswered) give:

| Cap | 25k | 30k | 35k | 40k | 45k | 50k | 60k |
|---|---|---|---|---|---|---|---|
| pass@1 | 30.5% | 36.4% | 41.1% | 43.1% | 47.7% | 50.5% | 55.3% |
| tokens per attempt | 21.5k | 24.7k | 27.5k | 30.2k | 32.6k | 34.7k | 38.6k |

The budget-note harness lies far above that curve: s20x60 scores 62.4% at 32.9k, where a bare cap gives about 47%.

## Findings
1. **≤25k mean tokens per attempt is not reachable without a large loss.**
   - The cheapest config that still works is s20x40, at 29k tokens and −18 pp.
   - The 40k hard cap is what kills it: long-tail pairs drop from 71% to 24%.
   - Thinking off reaches 19–23k but at −28 to −45 pp.
2. **Between 33k and 41k the trade-off is gentle.**
   - Per 1k tokens saved, pass@1 drops about 0.7 pp: 67.8% at 40.7k, 64.7% at 35.9k, 62.4% at 32.9k.
   - pass@2 barely moves: 77.5%, 75.3%, 76.5%. At lower budgets the two attempts diverge more, so pass@2 holds while pass@1 falls.
   - **s20x60 matches E004-GL's pass@2 (76.5% vs 76.4%) at −34% tokens.**
3. **Correct rollouts are short in every config.** Their p50 is 24–25k everywhere, and the p90 shrinks with the budget: 62.8k for sub80s40, 47.8k for s20x60, 36.7k for s20x40.
   - For SFT under a ~22k per-attempt Kaggle budget, about half of the correct rollouts from any of these configs already fit.
   - s20x60 gives the most of them per dollar.

**Spend:** recorded $24.7 (dev $6.6, full $18.2), against a $22 budget plus a $2.4 overrun that was approved for the s25x80 second sample.
