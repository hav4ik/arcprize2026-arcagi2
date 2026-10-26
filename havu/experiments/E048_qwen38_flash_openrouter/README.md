# E048: Qwen3.8 Flash (Flash-Next) on ARC-AGI-2 eval via OpenRouter, with tools and without

User request 2026-10-26: benchmark Qwen3.8 Flash-Next with the best-scoring harness, tools first, then no tools.

- Model: OpenRouter `qwen/qwen3.8-flash` (HF id `Qwen/Qwen3.8-Flash-Next`), provider pinned to Alibaba (the only one),
  `reasoning: {enabled: true}` (OpenRouter lists no reasoning_effort for it). Price $0.15 / $0.016 cached / $0.47 per M.
- Tools (`cfg_tools.txt`): the best-scoring DeepSeek-1017 harness, **sub80s40** (E029: pass@1 74.3 / pass@2 82.4 on
  DeepSeek): lean2sub prompt, run_python + submit_answer, 80k hard / 40k gentle soft budget, 64k per turn, final mode
  session, lenient parse, train check, 2 attempts on 120 tasks / 167 pairs (GitHub eval dir). The minimal effort
  header (E030, 84.9, a tie) is DeepSeek-specific and was left off.
- No tools: next, ARC Prize prompt (E034 style), after the OpenRouter credit is topped up.
- Kill-guard: no new attempts below $5 OpenRouter credit (credit is shared with other sessions' OpenRouter runs).

## Smoke (3 pairs, 1 attempt)
2/3 correct (136b0064, 135a2760; 0934a4d8 wrong); 8-17 turns; 19k / 41k / 56k output tokens; $0.015 / $0.036 / $0.047;
cache hits 84-90%; 6-17 min per attempt. Reasoning is returned and round-trips; tool calls work.

## Results
(running: `evals/results/tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e048_qwenflash_sub80s40.jsonl`,
log `evals/results/log_e048_qwenflash_sub80s40.txt`; the model id's "/" puts the files in a subfolder)
