# Qwen harness: e025 mode (the interface the E028 SFT model was trained on)

These options are opt-in. With every option at its default, the harness is byte-identical to before: the same prompts,
tool specs, `config_dict`, `harness_version` (qwen-tools-4) and LocalExecutor output. Setting any of them records
qwen-tools-5.

| Option | Where | Effect |
|---|---|---|
| `prompt="lean2roles"` | `agent_qwen.PROMPTS` | Same text as `run_deepseek_tools.PROMPT_SUFFIXES["lean2roles"]` (lean2 + submit + roles). With `helpers=True, task_summary=True`, the user message is byte-equal to E025's. |
| `tool_desc="e025"` | `AgentConfig` | `tool_specs` returns `e025_tools.json` exactly (`run_python` with HELPERS_DOC, plus `submit_answer`). It also uses run_deepseek_tools strings for `[tool error: Unknown tool: X]`, `[tool error: Expected an object with a string code field]` and `[skipped: solution already accepted]`. When a call hits the executor cap, the output becomes the sandbox's `Execution timed out after 30 seconds` with state lost. |
| `solve_fallback=True` | `AgentConfig` | When an attempt ends with no answer and no valid `answer`, the harness evaluates `solve(test_input)` in the session before salvage. The result is recorded as `answer_source="session_solve"` (already a weak source in `selection.py` and `solver.py`). |
| `style="ipython"` | `LocalExecutor(...)` / `local_factory(..., style="ipython")` | Cell output in the Docker sandbox's text format: IPython display (pretty-printed, with a blank line before multi-line values after prints), Plain tracebacks in stdout (cell frames show only the source line), `KeyboardInterrupt` tracebacks on timeouts, the sandbox's output post-processing, and `<output cut>` truncation. It uses `IPython.lib.pretty` when IPython is installed and falls back to pprint otherwise. |

Matching E025 config: `prompt="lean2roles", helpers=True, task_summary=True, tool_desc="e025", answer_mode="submit",
tier="xhigh", max_total_output=80000, budget_soft=40000, budget_notes=0.01, budget_warn=0.75, tool_timeout=30,
max_tool_output=16000, max_turns=40`, with executor `style="ipython"` and the helpers prelude. `budget_note` was
already byte-identical to DeepSeek's for these settings.

## Parity (2026-10-17)
- `tool_specs(e025)` matches `e025_tools.json` byte for byte. The `render_system("xhigh")` block matches the first
  831 tokens of the training sequences.
- **Replay:** 70 training rollouts (30 picked for errors, timeouts, past-target and nearly-spent notes, rejected
  answers, output truncation, multi-call turns and 65k cuts; 40 random) were rebuilt with `initial_prompt`,
  `render_assistant` and `render_tool_responses` and tokenized as one string. The user message is byte-equal in
  70/70, and all 70 sequences match token for token.
  - The only difference: training sequences end at the final `<|im_end|>`, without the separator `\n`.
  - Training data never contains `[tool error: ...]` turns, because `build_dataset.clean` cuts at them.
- **Re-execution:** 221 recorded `run_python` cells from 40 rollouts were re-run with `style="ipython"`. 215/221
  outputs are identical to the sandbox text; plain style gets 203/221. The 6 that differ come from the environment:
  the line a timeout interrupts, Python 3.11 `^^^^` carets against 3.10's, and scipy-version behaviour. Paths and line
  numbers in library frames also depend on the environment.
- Scripts are in the session scratchpad (`e025mode/replay.py`, `reexec.py`). The unit tests are in
  `evals/tests/test_qwen_e025.py`.

## Not like training
- **Hard cap:** at the 80k cap, the context limit, or when thinking never closes, the harness appends FORCE_THINK and
  opens a `submit_answer` call. E025 had no forced turn: it stopped and took the session's `answer`.
  `force_submit_tokens=0` reproduces that, with the fallback order answer, then solve (if `solve_fallback`), then
  salvage.
- **Budget units:** budgets count Qwen tokens, while E025 counted DeepSeek tokens.
- **Assistant turns:** turns are appended exactly as the model generated them, not re-rendered.
