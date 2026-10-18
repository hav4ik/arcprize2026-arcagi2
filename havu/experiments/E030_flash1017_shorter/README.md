# E030: shorter DeepSeek-V4.1-Flash-1017 traces within 5 pp of the best pass@2

Goal (2026-10-18): fewer output tokens per attempt than E029 sub80s40 (82.4% pass@2, 31.7k tokens) while keeping task-weighted pass@2 ≥ ~77.4%. All runs: Flash-1017, `lean2sub`, `--answer-tool --final-mode session --lenient-parse`, low effort, 167 pairs × 2 attempts, off-peak, 30 workers. Scores are task-weighted (ARC Prize); Δ = paired vs E029 sub80s40 (`evals/paired_vs_baseline.py`). Spend $27.1 recorded ($12.69 round 1, $0.16 smoke, $14.23 round 2).

## Results

| Config | pass@1 | pass@2 | Δ pass@2 [95% CI] | tokens mean / median / p75 / p95 / max | Δ tokens | hard pairs* |
|---|---|---|---|---|---|---|
| E029 sub80s40 (80k / 40k soft) | 74.3% | 82.4% | — | 31.7k / 27.5k / 47.2k / 69.1k / 80.0k | — | 81.8% |
| **s20x80** (80k / 20k soft, strong wording) | 68.8% | **81.1%** | −1.2 [−6.2, +3.8] | **28.7k / 24.5k / 38.7k / 62.5k / 77.9k** | **−10%** | 52.3% |
| s15x80 (80k / 15k soft, strong) | 65.7% | 78.3% | −4.0 [−9.9, +1.7] | 27.9k / 24.3k / 37.4k / 65.9k / 80.0k | −12% | 50.0% |
| E029 s20x60 (60k / 20k soft, strong) | 65.8% | 77.8% | −4.6 [−10.8, +1.7] | 26.7k / 24.5k / 38.1k / 56.3k / 60.0k | −16% | 47.7% |
| cap12: sub80s40 + `--max-tokens 12000 --cap-nudge` | 69.4% | 79.4% | −2.9 [−8.3, +2.5] | 31.0k / 24.6k / 47.2k / 65.4k / 72.6k | −2% | 59.1% |
| cap16: sub80s40 + `--max-tokens 16000 --cap-nudge` | 71.6% | 81.5% | −0.8 [−5.8, +4.2] | 31.9k / 29.1k / 46.7k / 68.3k / 77.7k | +1% | 65.9% |

\* Pair accuracy on the 22 pairs whose baseline correct rollouts needed > 40k tokens.

Mixed-config pass@2 (attempt 0 from one run, attempt 1 from another; offline, no cost) adds nothing: every mix lies on the same curve, about −1 pp pass@2 per 1k tokens saved (lean2code + s20x60 82.4% at 29.8k; s20x80 + sub80s40 81.1% at 30.2k; s15x80 + s20x60 77.8% at 27.3k).

## Findings
- **Budget pressure saturates.** Soft targets of 15k or 20k all land at a ~24.5k median; the saving comes from the upper half (p75 47k → 38k) and costs the hard pairs (82% → ~50%). The best point within the band is **s20x80: −10% tokens at −1.2 pp pass@2**; s20x60 reaches −16% at −4.6 pp, the edge of the band.
- **A per-turn thinking cap does not shorten DeepSeek.** `--cap-nudge` (new, harness persistent-v6): when a turn hits `--max-tokens`, the harness keeps its reasoning and closes it with a synthetic `run_python` call whose output says to test the hypothesis in code now. It fired in 31% (12k) / 15% (16k) of attempts, and the API accepts it (0 errors). But total tokens stay the same: the model resumes the same deliberation in the next turn. Unlike Qwen (E020: +7.2 pp), there is no gain; 12k costs accuracy on hard pairs.
- With E029's prompt-only A/B (lean2code: +3.6% tokens), neither prompt wording nor turn structure moves 1017's token use; only the total budget does, and it trades directly against hard-pair accuracy.

## Options not yet tried (cost)
- Per-pair rather than per-attempt saving: run attempt 2 only when attempt 1 fails the host train check (`--train-check`, no model tokens). On the old model this saved ~30% of tokens per pair for −3.6 pp pass@2 (E017 phase 1). One run of 167 × 2 with `--train-check` (~$7) gives the data to simulate it.
- s25x70 / s20x70 to fill the gap between s20x80 and s20x60 (~$6 each).
