# E012: Qwen3.8-27B with the Python tool harness

## Goal
Port DeepSeek's best tool-use harness (E009 `sub80s40`) to Qwen3.8-27B. Then measure accuracy against tokens at Kaggle-like budgets: about 22k tokens per attempt at pass@2, or about 44k at one attempt per pair, on 4×L4. The comparison is no-tool Qwen on the same pairs, both uncapped and capped at the same budget.

## Harness (`qwen-tools-2`)
- **Code:**
  - `evals/arc_harness/agent_qwen.py`: the agent loop, importable for the Kaggle notebook.
  - `evals/arc_harness/executor.py`: code-execution backends.
  - `evals/run_qwen_tools.py`: runner.
  - `evals/qwen_tools_stats.py`: stats and paired CIs.
  - Tests: `.venv/bin/python -m unittest evals/tests/test_qwen_tools.py` (17 tests; no network or sandbox needed).
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
- **Server:** `nvidia/Qwen3.8-27B-NVFP4` on SGLang 0.5.20, on vast.ai instance 55443640 (4× RTX PRO 6000 Server, $5.89/h). `ENDPOINTS.yaml` entry `vastai-4xrtx6000s-qwen27b-b`. API cost $0; box time only.

## Runs
RESULTS_PLACEHOLDER
