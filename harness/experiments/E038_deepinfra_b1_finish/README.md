# E038: finish nvarc_b1 through OpenRouter → DeepInfra (10,000 rollouts total)

2026-10-21. The DeepSeek credit ran out in E036 with 1,245 `nvarc_b1` tasks unrun (plus 1 errored). This runs those
1,246 (`pairs_nvarc_b1r.txt`) with the E031/E036 harness (s20x60 + minimal header, `launch.sh`) through OpenRouter pinned
to DeepInfra (fp8), 64 workers, as validated by the E037 probe. 08:51–10:50 UTC, then 13 connection errors retried
(13 workers, done 11:20). Cost $9.84 recorded (OpenRouter usage cost). Results: `evals/results/e038_deepinfra_nvarc_b1r.jsonl`.

Together with E031 (2,500) and E036 (6,254 successful) this gives 10,000 successful rollouts; every nvarc_b1 task is run.

## Quality check (`scripts/qc.py`, output `qc_final.txt`)

New tasks differ from E036's, so accuracy is compared with a parent-matched expectation (each task's parent's E036 rate).

| | DeepSeek API (800 random E036 rows) | DeepInfra E038 (1,246) |
|---|---|---|
| correct | 71.6% | 70.9% (parent-matched expected 71.0%, z −0.08) |
| train_pass / train_pass but wrong | 82.9% / 18.7% | 80.8% / 18.5% |
| out tok mean / median / p95 | 15.3k / 10.1k / 44.3k | 15.0k / 9.7k / 42.8k |
| length-stopped turns | 4 | 2 |
| turns without thinking / attempts without thinking | 4.7% / 0 | 4.7% / 0 |
| zlib ratio of thinking median / min (loops) | 0.350 / 0.300 | 0.351 / 0.301 |
| attempts with CJK chars (>50) | 0 (0) | 2 (0) |

No quality difference. Throughput ~10.5 rollouts/min at 64 workers (median wall 191 s/attempt); 1.0% connection errors.

On HF as config `e038_openrouter_deepinfra_b1r` of `livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036`.
