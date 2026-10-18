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

## Round 3: a "below low" effort header (2026-10-18)
DeepSeek implements its effort levels as a header at the start of the conversation (`notes/prompts/deepseek_v4_0813_reasoning_effort_prefixes.txt`): `high` and `max` prepend "Reasoning Effort: …" text, `low` adds nothing. `--effort-header minimal` (harness persistent-v6) sends a header in the same style as a system message: "Reasoning Effort: Minimal. Keep your thinking short and terse: write brief notes, not full sentences. Do not restate the task, do not re-read or re-derive what code has already printed, and do not re-verify what has already been checked. Test ideas with small code experiments instead of long deliberation, and submit as soon as the rule reproduces every training pair." Otherwise sub80s40, 60 workers, $6.50.

| Config | pass@1 | pass@2 | Δ pass@2 [95% CI] | tokens mean / median / p75 / p95 / max | Δ tokens | hard pairs |
|---|---|---|---|---|---|---|
| E029 sub80s40 | 74.3% | 82.4% | — | 31.7k / 27.5k / 47.2k / 69.1k / 80.0k | — | 81.8% |
| **sub80s40 + `--effort-header minimal`** | **74.5%** | **84.9%** | **+2.5 [−2.5, +7.5]** | **27.4k / 22.8k / 40.7k / 66.9k / 80.0k** | **−13.5%** | **84.1%** |

- **It shortens without the budget-pressure penalty.** Correct rollouts on the same pairs get shorter (median 17.4k vs 22.0k, p90 46.2k vs 52.5k), and the hard pairs are kept (84% vs 82%, against ~50% for s20x80).
- Pass@1 is level (+0.2 [−4.6, +5.0]) and pass@2 a tie or slightly better (1 sample).
- **Placement matters.** The same request as a suffix at the end of the user prompt (E009 `lean2brief`, old model) saved 3–15% and lost up to 14 pp.
- **Side effect:** 8 attempts hit the 40-turn cap (5 in the baseline); `--turns-note 5` covers it.

**s20x60 + header** (60k cap, 20k soft target, strong wording, `--effort-header minimal`; $5.34): pass@1 69.9% (68.6 / 71.2), **pass@2 80.3%**, tokens mean / median / p75 / p90 / p95 / max **23.3k / 20.6k / 33.5k / 43.8k / 50.1k / 60.0k** (−27% vs sub80s40), correct median 16.1k.
- vs sub80s40: pass@2 −2.1 [−6.7, +2.5], pass@1 −4.4 [−9.2, +0.2]; hard pairs 52% (the 60k cap still cuts the tail).
- vs plain s20x60: pass@1 +4.2 [−1.2, +9.6], pass@2 +2.5 [−4.6, +9.2] at −13% tokens. The header and the budget stack.

**Picks:** best accuracy sub80s40 + header (84.9% pass@2, −14% tokens); fewest tokens within 5 pp s20x60 + header (80.3%, −27%).

## Cross-model comparison on the Kaggle-pace pair set (2026-10-18)
The 87 pairs (68 tasks) shared by every run below: E027's 93 non-dev pairs minus the pairs that exist only in Kaggle's 172-pair eval copy. pass@ values task-weighted. The Qwen Kaggle-pace rows run one attempt per pair, forced at server-ready + 225 min (Kaggle's 11 h / 259-pair pace); their pass@2 combines two separate samples and is not reachable on Kaggle.

| Model | Harness | Time limit | Samples | pass@1 | pass@2 | tokens mean / median / p95 / max | correct median |
|---|---|---|---|---|---|---|---|
| DeepSeek old V4.1-Flash | E017 V3 (the E025 SFT-data setting) | none | 2 | 66.9% | 74.3% | 37.2k / 30.3k / 80.0k / 80.0k | 22.3k |
| DeepSeek old V4.1-Flash | sub80s40 | none | 3 | 68.1% | 78.9% | 40.2k / 37.8k / 80.0k / 80.0k | 25.6k |
| DeepSeek new 1017 | sub80s40 | none | 2 | 73.9% | 83.1% | 30.7k / 27.0k / 65.6k / 80.0k | 22.0k |
| DeepSeek new 1017 | sub80s40 + header | none | 2 | 73.5% | 84.6% | 26.7k / 20.3k / 68.9k / 80.0k | 16.9k |
| DeepSeek new 1017 | s20x60 + header | none | 2 | 68.4% | 79.4% | 23.3k / 20.4k / 51.2k / 60.0k | 15.2k |
| Qwen3.8-27B base (NVFP4) | E020 K3 (v4 recipe, 48k) | none | 2 | 40.1% | 46.3% | 36.8k / 47.7k / 48.0k / 48.0k | 22.5k |
| Qwen3.8-27B base (FP8) | v4b solver (E027, vLLM 0.31) | none | 1 | 47.1% | — | 35.0k / 39.3k / 48.0k / 48.0k | 24.2k |
| Qwen3.8-27B base (FP8) | v4b solver | Kaggle pace | 3 | 41.2% | 51.5%* | 28.4k / 33.4k / 38.3k / 47.5k | 20.8k |
| Qwen3.8-27B E028 SFT (FP8) | v5 notebook solver | Kaggle pace | 1 | 24.3% | — | 22.5k / 25.1k / 31.8k / 39.5k | 12.5k |

\* Two independent paced samples combined; Kaggle has time for only one attempt per pair at this pace.

**For future distillation data:** 1017 + header traces come from a ~10 pp stronger teacher than the E025 setting and are much shorter (correct median ~16–17k vs 22–25k; p95 51–69k vs 80k), so far more of them fit whole, wrap-up included, inside a 64K training cut (the E028 failure mode).

## Options not yet tried (cost)
- Per-pair rather than per-attempt saving: run attempt 2 only when attempt 1 fails the host train check (`--train-check`, no model tokens). On the old model this saved ~30% of tokens per pair for −3.6 pp pass@2 (E017 phase 1). One run of 167 × 2 with `--train-check` (~$7) gives the data to simulate it.
- s25x70 / s20x70 to fill the gap between s20x80 and s20x60 (~$6 each).
