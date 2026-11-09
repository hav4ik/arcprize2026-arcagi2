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

## Results: tools (sub80s40), 2026-10-26
Files: `evals/results/tools_persistent_grid_lean2sub_qwen/qwen3.8-flash_low_e048_qwenflash_sub80s40{,_retry}.jsonl`
(the model id's "/" puts them in a subfolder); logs `log_e048_qwenflash_sub80s40{,_retry}.txt`. 32 workers, 2 h.
API errors (connection drops, then Alibaba 429 rate limits / "Provider returned error") hit 12/334 attempts; they
were re-run into the `_retry` file (`retry_api_errors.sh`, `auto_retry.sh`); scoring takes the latest successful
record per attempt. 332/334 attempts scored (2 still rate-limited: 9385bd28:0 a1, dd6b8c4b:1 a1). Cost ~$9.6.

Paired with DeepSeek V4.1-Flash-1017 on the same harness (E029 sub80s40), task-weighted, 165 pairs / 120 tasks:

| Model | pass@1 | pass@2 | out tokens/attempt mean / median | $/attempt |
|---|---|---|---|---|
| Qwen3.8 Flash (OpenRouter, Alibaba) | 59.8% | **72.9%** | 35.0k / 31.6k | $0.028 |
| DeepSeek V4.1-Flash-1017 (DeepSeek API, off-peak) | 74.3% | **82.4%** | 31.7k / 27.1k | $0.022 |

pass@2 delta -9.4 pp [95% CI -16.1, -3.2] (bootstrap over tasks). Pairs solved: both 112, only Qwen 7, only DeepSeek
23, neither 23. Before the retries (errors counted wrong) the harness summary was 58.8% / 71.2%.
Qwen3.8 Flash is a competent tool user (far above our self-hosted Qwen3.8-27B) but clearly below DeepSeek-1017, with
~10% more output tokens and ~30% higher cost per attempt (cache reads $0.016/M vs $0.003/M).

## Results: no tools (ARC Prize prompt), 2026-10-26
File `evals/results/arcprize_qwen38-flash_e048_notool.jsonl` (`cfg_notool.txt`): reasoning on, max_tokens 131072,
2 attempts, 32 workers. **Stopped early by decision (2026-10-26 ~12:40 UTC): the gap was clear.** 241/334 calls
succeeded; 94 failed on the API side (76 "Provider returned error" from Alibaba, 17 connection drops, 1 truncated
stream; rate limiting under 32 parallel calls) and the resume of those was stopped. 36 successful calls hit the 131k
cap (no answer). Cost $8.92 (failed calls record no cost).

Paired with DeepSeek-1017 no-tool (E034, max effort, same prompt), 104 pairs / 79 tasks where both attempts exist:

| Model | pass@1 | pass@2 | out tokens/call mean / median | $/call |
|---|---|---|---|---|
| Qwen3.8 Flash | 46.0% | **58.2%** | 75.5k / 71.2k | $0.036 |
| DeepSeek V4.1-Flash-1017 | 63.3% | **80.4%** | 48.3k / 41.6k | ~$0.031 (E034 $10.44 / 334) |

pass@2 delta -22.2 pp [95% CI -32.9, -10.8]. Pairs: both 57, only Qwen 5, only DeepSeek 28, neither 14.
The subset excludes pairs with API-failed attempts; if failures cluster on long calls the gap may shift, not its sign.

## Conclusion
Qwen3.8 Flash (Flash-Next) is consistently worse than DeepSeek V4.1-Flash-1017 on ARC-AGI-2 eval: -9.4 pp pass@2 with
tools, -22 pp without. Tools help it much more (+~14 pp pass@2 vs ~+2 for DeepSeek). It is not cheaper per attempt
(more tokens; cached input $0.016/M vs $0.003/M). Through OpenRouter it is served only by Alibaba, which rate-limits
and errors heavily at 32 parallel requests (4% of tool attempts, 28% of no-tool calls). Not worth switching to.

Total spend ~$19 (tools ~$9.6 incl. retries, no tools $8.9, smoke $0.1).
