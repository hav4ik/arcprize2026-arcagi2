# E012: Qwen3.8-27B with the Python tool harness

## Goal
Port DeepSeek's best tool-use harness (E009 `sub80s40`) to Qwen3.8-27B. Then measure accuracy against tokens at Kaggle-like budgets: about 22k tokens per attempt at pass@2, or about 44k at one attempt per pair, on 4×L4. The comparison is no-tool Qwen on the same pairs, both uncapped and capped at the same budget.

## Harness (`qwen-tools-1` to `qwen-tools-3`)
- **Code:**
  - `evals/arc_harness/agent_qwen.py`: the agent loop, importable for the Kaggle notebook.
  - `evals/arc_harness/executor.py`: code-execution backends.
  - `evals/run_qwen_tools.py`: runner.
  - `evals/qwen_tools_stats.py`: stats and paired CIs.
  - Tests: `.venv/bin/python -m unittest evals/tests/test_qwen_tools.py` (23 tests; no network or sandbox needed).
- **Model API:** raw `/v1/completions` with the Qwen3.8 chat template rendered by hand. It does not use chat completions with the server's `qwen3_coder` tool parser. The reasons:
  - Every past turn is appended verbatim, so the prompt only grows, and the server's prefix cache hits on every turn. The SGLang log shows, for example, 30k cached tokens and 250 new ones per turn.
  - In a tool loop the template keeps the thinking of all turns after the last real user message. Tool responses are user turns that start with `<tool_response>`, so the template keeps them all; the `preserve_thinking` flag does not matter here. Feeding back all thinking is exactly what the model was trained on.
  - A turn cut by `max_tokens` or by an EOS inside `<think>` is continued in place. Qwen sometimes emits EOS mid-thought: the run continues it up to 4 times, and after two low-yield continuations it forces the answer.
  - Tool calls are parsed here (`<tool_call><function=NAME><parameter=code>…</parameter></function></tool_call>`). Malformed calls get a `[tool error: …]` response and are counted (`parse_failures`).
  - No dependence on a particular server's reasoning or tool parser, so vLLM and SGLang behave the same. An offline vLLM engine only needs `async complete(prompt, max_tokens, stop)`.
- **Template check:** `run_qwen_tools.py --verify-template` builds a 2-turn tool conversation. Our rendering matches the server's chat-template prompt token count exactly for the xhigh, medium and low tiers. One caveat: SGLang re-serializes tool JSON through pydantic. That adds `"strict": false` and `"defer_loading": null` and moves `description` first, so the check renders that form. The agent itself uses the plain dicts, which is what `transformers.apply_chat_template` renders; the string matches the template rendered locally with jinja2.
- **Agent logic (same as E009 `sub80s40` unless noted):**
  - ARC Prize prompt plus the `lean2sub` suffix.
  - Tools `run_python` (persistent session; `train`, `test_input` and `np` preloaded) and `submit_answer(expr)`, which evaluates the expression in the session, so a grid is never retyped.
  - A budget note on every tool output with a soft target and a hard cap.
  - At the hard cap, unlike E009, the harness closes the thinking and opens a `submit_answer` call. The model writes only the expression, in at most 256 tokens. If that fails, the session's `answer` is used.
  - Exactly one scored grid per attempt. `answer_source` is one of `submit`, `forced`, `parse`, `lenient`, `session` or null.
  - `--verify-submit` (E011) is available.
- **Effort tier:** the template's system line (xhigh, medium = no line, or low), placed before the tools block.
- **Sampling:** temperature 1.0, top_p 0.95, top_k 20. No presence penalty.
- **Executors:**
  - `SandboxSession`: the Docker sandbox, with no network and no host mounts.
  - `LocalExecutor`: a persistent child Python per attempt, for Kaggle. Timeouts send SIGINT, which keeps state; a cell that does not stop is killed and the process restarted with only the task data. It also sets a memory rlimit, its own process group, and a separate response pipe, so model prints cannot corrupt the protocol. It is **not** a security boundary (same user and filesystem), which is fine on Kaggle but not on this host.
- **Versions:**
  - `qwen-tools-1`: the pilot.
  - `qwen-tools-2`: the forced-submit text names the expression to use (`answer` or `solve(test_input)`).
  - `qwen-tools-3`: robustness fixes from the Kaggle reviews (`kaggle/reviews/inference_baseline_v1.md` R2/R3/R7/R8, `presubmit_review.md` S3/S4/S7/S8):
    - the session `answer` fallback also runs after an exception;
    - a hard wall-clock cap on every executor call, replacing the session on a hit;
    - the `context` length is configurable, and room includes the tool output appended since the last request;
    - LocalExecutor without `preexec_fn`: limits set in the child (RLIMIT_AS/CPU/FSIZE, nice, oom_score_adj=1000), process-group signals, a bounded kill, and a spawn retry;
    - `salvage`: if no answer is left, the session's best-scoring one-argument function is applied to `test_input`.
    - v3 changes only failure paths; runs differ from v2 only where v2 had no answer.
  - `AgentConfig.soft_wording="strong"` (any version) replaces the line after the soft target with the E010 "submit your best answer now, unless one more experiment will very likely make every training pair pass".
- **Server:** `nvidia/Qwen3.8-27B-NVFP4` on SGLang 0.5.20, on vast.ai instance 55443640 (4× RTX PRO 6000 Server, $5.89/h). `ENDPOINTS.yaml` entry `vastai-4xrtx6000s-qwen27b-b`. API cost $0; box time only.
- **Sandboxes:**
  - Pilot and first dev minutes: the shared `arc-sandbox` (`http://172.18.0.2:6000`).
  - Everything after 21:50 UTC: a dedicated `arc-sandbox-qwen` (`http://172.18.0.3:6000`), started with `NAME=arc-sandbox-qwen SANDBOX_MEMORY=12g evals/sandbox_docker/start.sh`. It uses the same image, internal network, patched `main.py` and idle reaper; `start.sh` gained the `SANDBOX_MEMORY` option. It was removed after the runs.
  - Up to 64 sessions used about 7.7 GiB.

## Runs
All runs use tier medium unless noted, temperature 1.0, top_p 0.95, top_k 20, budget notes on every tool output (`budget_notes=0.01`, `budget_warn=0.75`), a 16k chunk, 60 max turns and verify-submit off. Scoring is ARC Prize task-weighted. pass@1 is the mean over attempts 0 and 1; pass@2 counts a pair if either attempt is correct.
- **No-tool baseline:** `evals/results/qwen_nvfp4_gen_xhigh_240k_t1.jsonl` (E008: xhigh, 240k, the same sampling). Capped no-tool rows are simulated: an attempt counts as correct only if it was correct and used no more than the cap, and its cost becomes min(tokens, cap). E008 shows a forced final answer at a cap rarely helps.
- **Stats:**
  - `experiments/E012_qwen_tools/final_stats.py` produces the numbers below.
  - `evals/qwen_tools_stats.py <runs…>` gives per-run tables, paired bootstrap CIs over tasks, and simulated lower caps (`--no-salvage` scores v3 salvage answers as wrong).
- **Box time:** about 4.3 h of the $5.89/h box (20:45–01:00 UTC), about $25. Throughput was about 1.2k tok/s at 24 concurrent requests and about 1.9k tok/s at 64. A typical attempt took about 21 min of wall time, because each stream decodes at only 30–50 tok/s.

### Full set: medium, 48k hard / 24k soft, 2 attempts (120 tasks / 167 pairs / 334 attempts)
Files:
- `evals/results/qwen_tools_medium_48k_dev_s24.jsonl`: 193 attempts, `qwen-tools-2`; the 56 dev pairs plus the first part of the remainder.
- `evals/results/qwen_tools_medium_48k_full_v3.jsonl`: 141 attempts, `qwen-tools-3`.
- Traces: `…_traces/` (raw prompt and generation, gzipped).
- Launch (remainder):

  ```bash
  ARC_SANDBOX_URL=http://172.18.0.3:6000 .venv/bin/python evals/run_qwen_tools.py --pairs-file experiments/E012_qwen_tools/nondev_pairs.txt \
    --skip-done-in evals/results/qwen_tools_medium_48k_dev_s24.jsonl --attempts 2 --tier medium --hard 48000 --soft 24000 --concurrency 48 --tag full_v3
  ```

| config | pass@1 | pass@2 | pair-weighted pass@1 / pass@2 | tokens mean / median / p90 |
|---|---|---|---|---|
| tools medium 48k/24k (v3, with salvage) | 25.8% | 34.6% | 24.6% / 33.5% | 35.6k / 40.0k / 48.0k |
| tools medium 48k/24k (v2-equivalent) | 25.8% | 34.6% | 24.6% / 33.5% | 35.6k / 40.0k / 48.0k |
| no-tool xhigh, cap 24k (simulated) | 3.4% | 5.4% | 3.6% / 5.4% | 23.7k / 24.0k / 24.0k |
| no-tool xhigh, cap 36k (simulated) | 12.1% | 16.7% | 10.8% / 14.4% | 34.7k / 36.0k / 36.0k |
| no-tool xhigh, cap 48k (simulated) | 21.2% | 29.0% | 20.4% / 28.1% | 44.2k / 48.0k / 48.0k |
| no-tool xhigh, cap 240k (simulated) | 36.0% | 47.4% | 35.6% / 47.3% | 73.6k / 67.5k / 120.2k |

Paired differences, pass@1, bootstrap over tasks (95% CI):

| tools vs | Δ pass@1 |
|---|---|
| no-tool capped at 24k | **+22.4** [+16.1, +28.9] |
| no-tool capped at 36k (about equal mean tokens) | **+13.8** [+6.7, +20.6] |
| no-tool capped at 48k (equal hard cap) | +4.7 [−2.3, +11.7] |
| no-tool uncapped (73.6k mean) | **−10.1** [−17.6, −2.4] |

Reference, DeepSeek V4.1-Flash E009 `sub80s40` (same tool harness, 80k/40k): 67.8% / 77.4% at 40.6k tokens.

- **Accuracy per token:** tools give 25.8% / 34.6% at 35.6k mean tokens. No-tool Qwen needs about 64–80k tokens per attempt for the same pass@1 (28.5% at a 64k cap, 32.4% at 80k), roughly 2× the tokens. At equal mean tokens (no-tool capped at 36k) tools are 2.1× better. At an equal hard cap the edge shrinks to about +5 points, because tool attempts finish earlier (median 40k against 48k).
- **Splits:**
  - dev (56 pairs): 20.4% / 27.6%;
  - non-dev (111 pairs): 28.6% / 38.5%;
  - pilot (26 pairs): 31.7% / 40.0%.
  - The E009 dev set is harder for Qwen with tools: on those pairs tools score 20.4% against 18.7% for no-tool at a 48k cap.
- **Where tokens go:**
  - 34% of attempts end at the 48k cap or with thinking that never closes. They use 43% of all tokens, and none is correct: the forced `submit_answer` gave 64 grids and 0 were right.
  - All 82 correct answers came through `submit_answer`. Correct rollouts used 23.1k tokens at the p50 and 39.7k at the p90.
  - The largest prompt was 86.5k tokens (p50 53k, p90 65k), well under 131k.
- **Answer sources:** submit 222, forced 64, none 46, salvage 2. v3 salvage gained nothing: 2 of 46 no-answer attempts were salvaged, both wrong, so v2 and v3 scores are identical.
- **Turns and tool calls:**
  - 14.9 turns on average; 0.73 failed Python cells per attempt (ValueError, TypeError, IndexError, NameError).
  - **One tool-call parse failure in 334 attempts.**
  - 39 EOS-in-thinking continuations; 11 attempts ended with thinking that never closed.
  - 0 executor resets; no repeated-cell loops (2 of 193 attempts re-ran an identical cell).

### Dev and pilot screens (single configurations, partial)
| run | pairs × attempts | result | tokens / attempt | notes |
|---|---|---|---|---|
| pilot xhigh 80k/40k, xhigh 48k/24k, medium 48k/24k (`…_pilot_s40`, `…_pilot_s24`) | 26 × 2, stopped at 11–17 attempts each | easy pairs only; all about 50–57% per attempt | 17–20k | stopped to respect the shared-sandbox session cap; biased to fast attempts |
| xhigh 48k/24k, dev (`qwen_tools_xhigh_48k_dev_s24.jsonl`) | 38 attempts (attempt 0) | 21.1% per attempt; medium on the same attempts did better | 40.8k | 74% of tokens in cap-hit attempts; fewer but longer turns. Dropped. |
| medium 28k/14k, dev (`qwen_tools_medium_28k_dev_s14.jsonl`) | 22 attempts | 4 correct vs 7 for 48k/24k on the same attempts (no-tool at 28k: 2) | 24.0k | half of attempts hit the cap. Dropped. |
| **medium 48k/24k, strong wording**, dev (`qwen_tools_medium_48k_dev_s24strong.jsonl`) | 90 attempts | **26 vs 18 correct** against gentle wording on the same 90 attempts; discordant 10 vs 2; task-weighted **+10.2 [+3.1, +18.4]** per attempt | 34.4k (gentle 35.4k) | cap hits 29 vs 33. Dev only (dev screens have misled 3× before); not run on the full set |

### Failure modes (trace review of cap-hit attempts plus counts)
- **Hand reasoning in one huge turn.** 54 of 193 attempts have a single turn over 15k tokens. In the 8b9c3697 trace, turn 3 is 25.6k tokens and 70k characters of reasoning over printed grids; the model then spends the rest of the budget debugging an over-complex rule. The tool is used for inspection and testing, but discovery still happens in thinking.
- **Debug cycles to the cap.** Failing attempts run 10–20 short turns of fix and re-test. Most cap hits come after a median 12 turns, with one turn over 17k tokens.
- **Forced submit is wasted.**
  - The model often writes a whole `def` or code block instead of an expression (16 of 67 forced expressions in the v2 file); v2's clearer text did not stop this.
  - Accepted forced grids are never correct.
  - Since these attempts score 0, their tokens (43% of the total) are the main target.
- **Formatting and tool calls are clean:** 1 parse failure in about 5,000 calls, and no retyped grids (the answer comes from the session).
- **EOS mid-thought:** 39 continuations; 11 attempts never closed their thinking after two low-yield continuations (stop `unclosed_think`).

## Recommendation for the Kaggle notebook
- **Use tools:** `agent_qwen.solve` with `LocalExecutor`, tier **medium**, hard cap **48k**, or min(48k, adaptive budget), and soft target **24k**, at **one attempt per pair**.
  - Expected accuracy at the Kaggle budget of about 36k tokens per pair: **about 25% pass@1** on the public eval. The full set gave 25.8% at 35.6k mean.
  - No-tool Qwen at the same budget gives about 12%, and 21% at 48k.
- **Wording:** `soft_wording="strong"`. It is +10 points on 90 dev attempts at no token cost; the full set has not confirmed it, but there is no observed downside.
- **verify_submit off** (untested on Qwen). Keep `max_turns=60`, `chunk_tokens=16384`, `force_submit_tokens=256` and `context=131072` (the server's max_model_len).
- **xhigh is worse at these budgets**, and two attempts of about 24–28k each are worse than one of 48k.
- **Spare budget:** a second attempt at about 48k on the pairs where the first had no `submit` answer is the cheapest use (pass@2 34.6% vs pass@1 25.8% here).

## Next (not run; the box is being released)
1. **Strong wording on the full set, with `check_only`** (record a host-side train check of every submitted expression without rejecting). This confirms the +10 points and measures how many wrong answers fail train, the input to item 3. Also try soft 16k + strong.
2. **Per-turn thinking cap.** After about 6–8k thinking tokens in one turn, inject `</think>` and a short "test the current hypothesis with code now" nudge, to move rule discovery into the tool and cut the 15k+ turns.
3. **Second-attempt policy.** Spend a second attempt only where attempt 1 has no `submit` answer or its answer fails the train check (R9). Order by that and stop when the budget runs out.
4. Cut cap-hit waste further: stop at the soft target when no cell has reproduced any training pair yet; those attempts are almost never solved.
