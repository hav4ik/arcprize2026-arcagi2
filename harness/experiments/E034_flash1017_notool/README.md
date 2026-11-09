# E034: no-tool ARC Prize replica (E001d setup) on DeepSeek-V4.1-Flash-1017

Started 2026-10-18 (Sunday, off-peak). Same setup as E001d: official 120 tasks / 167 pairs, ARC Prize prompt with
JSON grids, `\boxed{}` parsing, max effort, 2 attempts per pair, max_tokens 200k. Baseline: E001d on the old
V4.1-Flash, pass@2 78.1% / pass@1 63.1% task-weighted (lower bound: 17/334 connection errors), 55.7k out tokens/call.
Estimate ~$10–11 (334 calls × ~50k tokens × $0.60/M).

```bash
cd evals && ../.venv/bin/python run_deepseek_arcprize.py --model deepseek-flash --effort max --attempts 2 --workers 40 --tag e034_flash1017
```
Results: `evals/results/arcprize_deepseek-flash_max_e034_flash1017.jsonl`, log `evals/results/log_arcprize_e034_flash1017.txt`.

## Results
Ran 2026-10-19 04:00–04:42 UTC (Monday, off-peak throughout), 334/334 calls, 0 errors, all `finish_reason=stop`,
1 unparsed answer, no truncation. Analysis: `scripts/analyze.py` (attempts 0/1 of both runs; the last non-error record
per call is used).

| | E034 Flash-1017 | E001d old Flash | Δ (paired, 95% bootstrap CI over tasks) |
|---|---|---|---|
| pass@2, task-weighted | **80.6%** | 79.3% | +1.2 pp [−5.4, +7.9] (14 tasks better / 13 worse) |
| pass@1 attempt 0, task-weighted | 66.4% | 65.1% | +1.2 pp [−6.7, +9.6] |
| pass@1 mean of attempts 0/1, task-weighted | 64.6% | 62.7% | +1.9 pp [−4.1, +7.7] |
| pass@2 / pass@1 a0, pair-weighted (167 pairs) | 79.0% / 66.5% | 78.4% / 63.5% | |
| out tokens/call mean / median / p90 / max | 50.4k / 44.8k / 93.3k / 146k | 55.3k / 49.3k / 102.8k / 152k | −9% mean |
| cost (list, off-peak) | $10.44 ($10.10 out + $0.34 in) | $11.42 | |

E001d note: its file also holds the resumed retries of its 17 connection errors, so with those included the baseline is
79.3% pass@2 / 62.7% pass@1 (mean a0/a1), not 78.1% / 63.1% (errors counted as wrong).

Conclusion: without tools, 1017 ties the old Flash (+1.2 pp pass@2, well inside the noise band and the CI) at ~9% fewer
output tokens. For reference, the best tool harnesses on 1017 reach a similar or higher pass@2 at about half the tokens: E030 sub80s40 +
header 84.9% at 27.4k tok/attempt, and E029 sub80s40 82.4% at 31.7k.
