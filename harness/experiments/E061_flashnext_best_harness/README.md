# E061: Qwen3.8-Flash-Next on DeepSeek's best tool harness + grid images (API)

User question 2026-10-28: why not run Flash-Next with DeepSeek's best harness, with grid images too? E048 ran
Flash-Next on the then-best harness (sub80s40), E058 added images to that; neither had the E046 improvements.

## Arms compared (120 eval tasks, GitHub eval copy, 2 attempts, task-weighted)
| run | harness | grids | file tag |
|---|---|---|---|
| E048 | sub80s40 | text | `e048_qwenflash_sub80s40` (+ `_retry`) |
| E058q | sub80s40 | text + images | `e058q_pilot30` |
| **E061** | **E046 best** (sub80s40 + `--turns-note 5 --fallback-solve --multi-test --multi-budget 0.5`) | text + images | `e061q_best_img` |
| DeepSeek reference | E046 best + `--effort-header minimal` | text / text + images | `e046p_note_all` / `e058_pilot30` |

DeepSeek's "Reasoning Effort: Minimal" header is left off: it is a DeepSeek prefix, not a Qwen one. Sampling stays at
the provider's defaults, as in E048/E058q, so only the harness changes vs E058q.

## Setup
`run.sh [TAG] [WORKERS]`: `evals/run_deepseek_tools.py` with E048's `cfg_tools.txt` (OpenRouter, provider Alibaba,
reasoning enabled, kill-guard $5) + the E046 flags + `--grid-format text+image --image-cell 0 --image-align qwen`,
sandbox `arc-sandbox` (172.18.0.2:6000), 8 workers. `retry_loop.sh` re-runs attempts that failed on API errors (4
workers, 20 min apart, up to 6 passes), as E058 did. Launched 2026-10-28 ~01:25 UTC (`chain.log`).
Cost estimate: ~$6-8 (E058q cost $8.81 for 334 per-pair attempts; multi-test makes 240 rollouts). Credit before:
$81.97.

## Results (main pass done 08:29 UTC, retries 09:32 UTC, 0 errored attempts left; $6.39)
Scored with `experiments/E058_deepseek_vision/compare.py` (latest successful record per task/pair/attempt; a task
counts only when both attempts of all its pairs exist on both sides). Task-weighted; tokens are output tokens per task
per attempt (all test inputs of a task summed).

| run (Flash-Next unless noted) | tasks | pass@1 | pass@2 | tokens | cost |
|---|---|---|---|---|---|
| **E061: E046 best harness + images** | 120 | **69.0** | **75.7** | **30.3k** | $6.19 |
| E058q: sub80s40 + images | 120 | 67.8 | 77.4 | 40.5k | $7.92 |
| E048: sub80s40, text | 118 | 59.5 | 72.5 | 48.3k | $9.04 |
| DeepSeek-1017: E046 best harness + images (E058) | 120 | 81.1 | 90.7 | 27.0k | $4.61 |

- vs E058q: pass@2 -1.7 pp [95% CI -8.3, +4.6], 8 tasks better / 12 worse; pass@1 +1.2. **A tie on accuracy at 25%
  fewer tokens** (multi-test answers all test inputs of a task in one rollout).
- vs E048 (118 tasks): pass@1 +9.4, pass@2 +3.2 [-4.0, +10.2], -38% tokens (images + harness).
- vs DeepSeek on its best harness + images: pass@2 -15.0 [-22.1, -8.8], 2 tasks better / 22 worse.
- Conclusion: DeepSeek's harness tuning (turn note, fallback, multi-test) carries over to Flash-Next only as a token
  saving; the ~13-15 point gap to DeepSeek stays. Flash-Next has never had a harness tuned for it; a Flash-Next-specific
  harness (sampling, budget, prompt) is the open question, not the DeepSeek levers.
- Interim on the first 66 finished tasks (07:00 UTC) looked worse (pass@2 72.7 vs 79.5); those were the fast, easier
  tasks, and the gap closed on the full set.
