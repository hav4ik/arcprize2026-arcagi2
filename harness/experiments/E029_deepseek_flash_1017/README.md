# E029: DeepSeek-V4.1-Flash-1017 on the tool harness (pass@2, full public eval)

Run 2026-10-18 (Sunday, off-peak). Total cost about $13.3 (recorded estimates $7.18 + $6.10).

## The model
- DeepSeek released V4.1-Flash-1017 on 2026-10-17. The API docs (`api-docs.deepseek.com/news`) say `deepseek-flash` is now served by it; the old V4.1-Flash is retired and legacy names route to 1017.
- There is no separate API id: `/models` lists only `deepseek-flash` and `deepseek-v4-pro`, and names like `deepseek-v4.1-1017` are rejected. `system_fingerprint` on 2026-10-18: `e72d3c7f663d3e1383b00b3884cb4501` (not recorded for older runs).
- **Consequence:** every `deepseek-flash` run from 2026-10-17 on uses the new model. Older results (E001–E026) and the E025 distillation rollouts are from the old model.

## Setup
Unchanged harness (`persistent-v4`), both 167 pairs / 120 tasks × 2 attempts, `--effort low --prompt lean2sub --answer-tool --final-mode session --lenient-parse --max-tokens 64000`, `--auto-peak-price --price-multiplier 1`:
- **sub80s40** (E009/E011 pass@1 pick): `--max-total-output 80000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 40000`, 12 workers. Baseline: E009, 3 samples, old model.
- **s20x60** (E011 economical pick): `--max-total-output 60000 --budget-notes 0.01 --budget-warn 0.75 --budget-soft 20000 --soft-wording strong`, 36 workers. Baseline: E010, 2 samples, old model.

Results: `evals/results/tools_persistent_grid_lean2sub_deepseek-flash_low_e029_flash1017_{sub80s40,s20x60}.jsonl` and logs `log_e029_*.txt`. Comparison: `evals/paired_vs_baseline.py --baseline <old> <new>`.

## Results (task-weighted, ARC Prize scoring)

| Config | Model | pass@1 | Δ pass@1 [95% CI] | pass@2 | Δ pass@2 [95% CI] | tok/attempt mean / median | $ (2 attempts) |
|---|---|---|---|---|---|---|---|
| sub80s40 | old (E009, 3 samples) | 67.8% | — | 77.5% | — | 40.7k / 37.6k | ~$9.0 |
| sub80s40 | **1017** | **74.3%** (75.4 / 73.2) | **+6.5 [+1.9, +11.3]** | **82.4%** | **+4.9 [+0.1, +9.9]** | **31.7k / 27.5k** (−22%) | $7.18 |
| s20x60 | old (E010, 2 samples) | 62.4% | — | 76.5% | — | 32.9k / 30.4k | — |
| s20x60 | **1017** | 65.8% (63.2 / 68.3) | +3.3 [−1.5, +7.9] | 77.8% | +1.2 [−4.6, +7.5] | **26.7k / 24.5k** (−19%) | $6.10 |

Output tokens per attempt (mean / median / p75 / p95 / max):

| Config, model | All attempts | Correct | Failed | Wall s (mean / median / p95) |
|---|---|---|---|---|
| sub80s40 old | 40.7k / 37.6k / 64.3k / 80.0k / 80.0k | 30.7k / 25.4k / 42.9k / 67.6k / 79.9k | 60.6k / 67.5k / 74.6k / 80.0k / 80.0k | 226 / 211 / 439 |
| sub80s40 1017 | 31.7k / 27.5k / 47.2k / 69.1k / 80.0k | 25.9k / 21.9k / 36.7k / 57.8k / 75.7k | 47.5k / 48.0k / 64.9k / 73.0k / 80.0k | 156 / 135 / 335 |
| s20x60 old | 32.9k / 30.4k / 48.7k / 60.0k / 60.0k | 26.2k / 24.1k / 37.8k / 51.2k / 59.6k | 43.6k / 48.9k / 57.5k / 60.0k / 60.0k | 183 / 172 / 343 |
| s20x60 1017 | 26.7k / 24.5k / 38.1k / 56.3k / 60.0k | 20.3k / 18.1k / 27.7k / 46.9k / 59.8k | 38.2k / 38.1k / 51.3k / 60.0k / 60.0k | 141 / 124 / 289 |

## Findings
- **1017 is better and cheaper.** On sub80s40 both pass@1 (+6.5 pp) and pass@2 (+4.9 pp) gains have CIs above zero, at 22% fewer output tokens. 1 sample of the new model, so the size is uncertain; the direction is not.
- **The gain comes from the long tail.** On the 42 pairs whose old correct rollouts needed >40k tokens, pair accuracy went 69.8% → 79.8%. The 80k budget lets 1017 use its efficiency on hard pairs; at 60k (s20x60) the gain is within noise (+1.2 pp pass@2).
- **sub80s40 is now the pick for accuracy per dollar:** 82.4% pass@2 for $3.6 per sample, above the old model's best (sub96n 78.5% at 45.7k tokens).
- Run health: 0 errors in 668 attempts; 36 workers ran without rate limits (s20x60 took 25 min).

## A/B: lean2code + turns note + solve fallback (harness persistent-v6), 2026-10-18
Based on `strategy_quant.md` / `strategy_reading.md` (prose rule comparison ~7k and hand re-reading of printed grids ~5k per trace; 4 of 5 turn-cap stops left no answer). Same sub80s40 flags plus `--prompt lean2code --turns-note 5 --fallback-solve`, 36 workers, 2 attempts × 167 pairs, $7.44 recorded.

| | 1017 sub80s40 | 1017 lean2code arm | Δ [95% CI] |
|---|---|---|---|
| pass@1 (task-wt.) | 74.3% | 73.8% (73.2 / 74.4) | −0.5 [−5.6, +4.7] |
| pass@2 | 82.4% | 81.8% | −0.6 [−6.0, +4.9] |
| tokens mean / median / p75 / p95 | 31.7k / 27.5k / 47.2k / 69.1k | 32.9k / 29.5k / 48.4k / 72.1k | +3.6% |
| turns = tool calls (mean) | 13.6 | 14.5 | |
| stops: max_turns / output_budget / no answer | 5 / 3 / 7 | 1 / 3 / 3 | |

- **No token saving:** the "decide in code" line moved work into more tool calls (+0.9 per attempt) instead of removing prose; tokens +3.6%, accuracy a tie. Do not adopt `lean2code`.
- **The turns note works as a safety net:** turn-cap stops 5 → 1 and attempts without an answer 7 → 3, at no cost. `--fallback-solve` fired once usefully (session answer 1, correct). Too few cases to move accuracy; harmless to keep on.
- Conclusion: prompt wording alone does not shorten 1017's traces further; the remaining lever is structural (per-turn thinking cap, needs harness work for the DeepSeek API).

## Next
- A second sample of 1017 sub80s40 (~$3.6) to tighten the CI, and sub96n (old best accuracy) on 1017.
- Distillation: new rollouts are from 1017; record that in any dataset mixing them with E025.
