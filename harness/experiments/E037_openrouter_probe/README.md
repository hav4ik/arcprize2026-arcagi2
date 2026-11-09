# E037: Flash-1017 through OpenRouter (pinned provider) vs the DeepSeek API

2026-10-20. Goal: more rollout throughput than DeepSeek's per-account cap (~6.3k out tok/s, E036). Flash-1017 is
served by ~20 OpenRouter providers. New harness option (persistent-v6, opt-in): `run_deepseek_tools.py
--openrouter-provider NAME --model deepseek/deepseek-v4.1-flash-1017` pins one provider (no fallbacks). It sends
OpenRouter `reasoning` (default {"effort": --effort}; override with `--or-reasoning`), passes previous-turn thinking
back as `reasoning`, takes cost from OpenRouter usage, and uses OpenRouter credit for the kill-guard.

Probe: DeepInfra (fp8; listed $0.0784/M input, $0.2352/M output, $0.002352/M cached input, but `usage.cost` bills ~$0.44/M output, E044),
s20x60 + header (E036 harness), on 100 random E036 nvarc_b1 pairs (`pairs_probe.txt`), 50 workers, $0.62. 1 connection error.

| Same 99 pairs | DeepInfra fp8 | DeepSeek API (E036) |
|---|---|---|
| correct / train_pass | 73 / 83 | 71 / 76 |
| out tok/attempt mean / median / p95 / max | 13.0k / 6.4k / 48.2k / 58.0k | 14.8k / 7.5k / 48.9k / 60.0k |
| longest turn; length-stopped turns | 33.9k; 0 | 43.8k; 2 |
| wall s median / p95 | 77 / 541 | 151 / 941 (DeepSeek at 128 workers, capped) |
| $/attempt | 0.0062 | 0.0110 |

Paired: both 69, only DeepInfra 4, only DeepSeek 2. No thinking cap seen (listed per-request max 131k). Cache hits are
reported (`prompt_tokens_details.cached_tokens`). Compare script: `compare.py <results file>`.
Results: `evals/results/e037_openrouter_deepinfra_{smoke,probe}.jsonl`.
