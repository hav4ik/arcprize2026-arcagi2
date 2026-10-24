# E044: no-tool Flash-1017 rollouts on the training pool (for pruning / distillation)

2026-10-23. Goal: no-tool reasoning traces on the same tasks as the tool rollouts (E031 2,500 pairs, then all 7,500
`nvarc_b1` tasks), so each task has both. Harness: E035 arm D (effort low + `--effort-header minimal` + `--task-summary`,
official ARC Prize prompt), via OpenRouter → DeepInfra (fp8), 1 attempt per pair, 64 workers. `launch.sh`, `run_pilot.sh`.

New opt-in options of `evals/run_deepseek_arcprize.py` (defaults unchanged): `--challenges/--solutions/--pairs/--out`
(Kaggle-format files; jobs keep the `--pairs` order so partial runs are prefixes), `--openrouter-provider`,
`--or-reasoning`, `--context` (cuts max_tokens to fit the estimated prompt), `--max-balance-stop`; records `est_cost_usd`.

## Step 1: harness check on the 120 public-eval tasks (334 calls, 2 attempts)

| | pass@2 | pass@1 | out tok mean / median / p90 / max | truncated | cost |
|---|---|---|---|---|---|
| DeepSeek API (E035 D) | 75.4% | 62.6% | 31.9k / 28.0k / 57.6k / 114.4k | 0 | ~$6.4 (list) |
| DeepInfra (E044) | 74.6% | 62.4% | 32.2k / 28.7k / 57.3k / 104.0k | 0 | $4.76 |

Paired attempts: both 171, only DeepSeek 38, only DeepInfra 40: parity. Results `evals/results/e044_notool_deepinfra_eval120.jsonl`
(not uploaded: eval set). **Price:** DeepInfra bills ~$0.44/M output (`usage.cost`), not the listed $0.2352/M.

## Step 2: pilot on the 2,500 E031 pairs (arc2_train 1,070, nvarc_b0 1,430)

`run_pilot.sh`, `run_pilot2.sh`. DeepInfra at first; the nvarc_b0 tail and a full arc2_train repeat ran on the DeepSeek API
(the repeat came from an empty split file; kept as config `notool_e044_arc2_train_sample2`).

## Step 3: all 7,500 `nvarc_b1` tasks (`run_b1.sh`, `run_split.sh`)

Round-robin split over DeepInfra (128 workers), InferenceNet (fp8, 64) and the DeepSeek API (48), started 21:16 and
finished 23:20 UTC 2026-10-23 (off-peak). A second pass retried API errors (e.g. "Provider timed out" at 160 DeepInfra
workers). InferenceNet was first checked on 100 nvarc_b0 pairs (65 correct vs 60 for DeepInfra on the same pairs).

## Results (`scripts/analyze.py`; tool = E031 / E036 + E038 rollouts on the same pairs)

| Config | Rows | No-tool correct | Tool correct | Only no-tool / only tool | Out tok mean / median / p90 | Cost |
|---|---|---|---|---|---|---|
| arc2_train (DeepInfra) | 1,070 | 93.7% | 95.8% | 24 / 46 | 9.2k / 5.3k / 21.4k | $5.02 |
| arc2_train_sample2 (DeepSeek) | 1,076 | 93.3% | 95.8% | 27 / 52 | 9.4k / 5.5k / 22.1k | $6.92 |
| nvarc_b0 (DeepInfra 1,050 + DeepSeek 380) | 1,430 | 66.5% | 70.3% | 94 / 149 | 19.6k / 12.8k / 48.5k | $15.12 |
| nvarc_b1 (DeepInfra 2,498 + InferenceNet 2,500 + DeepSeek 2,502) | 7,500 | 65.3% | 70.9% | 354 / 772 | 19.5k / 12.8k / 47.4k | $81.34 |

- nvarc_b1 per provider: DeepInfra 65.1%, InferenceNet 66.2%, DeepSeek 64.7% (tool on the same pairs 70.4 / 71.5 /
  70.9%). All within noise, so the providers are interchangeable.
- Cost per attempt: DeepInfra ~$0.009, InferenceNet ~$0.010, DeepSeek API ~$0.013.
- No-tool is about 5 pp below tool on NVARC, at about twice the output tokens. It still solves 354 nvarc_b1 tasks that the
  tool rollout missed.
- Text QC: no empty thinking, no CJK runs, no degenerate repetition (zlib ratio min 0.246). All `finish_reason` stop;
  1 unparsed answer.
- Total about $108 for 11,076 rows (plus $4.76 eval120 and small probes).

On HF: configs `notool_e044_{arc2_train,nvarc_b0,nvarc_b1,arc2_train_sample2}` of
`livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036` (final upload 23:34 UTC, secret scan clean). Rows hold the
rebuilt messages (system header, user prompt with task summary, assistant thinking + answer).
