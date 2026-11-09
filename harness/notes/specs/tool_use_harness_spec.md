# Tech spec: Python tool-use harness for DeepSeek on ARC-AGI-2 (`evals/harness.py`)

## 1. Goal
Let DeepSeek V4.1-Flash (`deepseek-flash`, thinking mode) call a Python tool while solving ARC-AGI-2 tasks, and score it exactly like our baseline.

The research question: does tool use cut output tokens per attempt (baseline ~53k) at equal or better accuracy (baseline 63.1% pass@1, 78.1% pass@2)?

The harness must be:
- resumable;
- safe to run with ~200 concurrent attempts;
- fully logged, so every model-written cell can be audited.

## 2. Repository context (read these first)
Repo root: `/home/vu/Workspace/ARC-AGI-2`. Python env: `.venv` (managed with `uv`; add packages with `uv pip install --python .venv <pkg>`). Read `AGENTS.md`.

Reuse; do not duplicate:

| Need | Existing code |
|---|---|
| API key | `api_key()` in `evals/run_deepseek_arc2.py` (reads `SECRETS.yaml`; never print it) |
| One streamed DeepSeek turn with tool calls | `turn(client, model, effort, messages, max_tokens)` in `evals/run_deepseek_tools.py`. Returns `(assistant_message_dict, usage_dict, finish_reason)`. The message dict already contains `content`, `reasoning_content` and `tool_calls` (OpenAI format). Its module-level `TOOL` constant is the old stateless tool; define new tool schemas in the harness. `PRICES` there holds off-peak $/1M (cache hit, cache miss, output). |
| Task files | `TASKS` in `evals/run_deepseek_arcprize.py` → `data/arc-agi-2-github/data/evaluation/*.json` (120 tasks, 167 test pairs; each test pair has `input` and `output`) |
| ARC Prize prompt | `build_prompt(train, test_input)` in `evals/run_deepseek_arcprize.py`. Its text is byte-identical to ARC Prize's and ends with `Your response:` |
| Answer parser | `parse(text)` in the same file (`\boxed{json}`, else the last list of lists) |
| Run summary | `evals/report.py` reads `evals/results/*.jsonl`; keep the record fields in §7 so it works |

DeepSeek thinking-mode tool rules (https://api-docs.deepseek.com/guides/thinking_mode):
- When the request carries `tools`, every previous assistant message must be sent back **with its `reasoning_content`**. Omitting it returns a 400 error.
- `temperature` has no effect.
- Effort is set with `reasoning_effort` (`low` / `high` / `max`).

## 3. Sandbox (already running; do not build another)
Start it with `evals/sandbox_docker/start.sh`, which prints the base URL (e.g. `http://172.18.0.2:6000`). It is the NeMo-Skills IPython sandbox server:
- runs on an `--internal` Docker network (no internet);
- no host directories mounted;
- patched so state survives timeouts.

The harness only talks HTTP to it.

### Execute a cell
- Request: `POST {url}/execute`
  - header `Content-Type: application/json`
  - header `X-Session-ID: <session id>`
  - JSON body: `{"generated_code": <str>, "std_input": "", "timeout": <seconds>, "language": "ipython", "max_output_characters": 6000, "traceback_verbosity": "Plain"}`
- Response JSON: `process_status` (`completed` / `error` / `timeout`), `stdout`, `stderr`, `new_session_created` (bool).
- Sessions are IPython: state persists within a session id, and the value of a trailing expression is displayed.
- A cell that exceeds `timeout` returns `error` with a `KeyboardInterrupt` traceback, and the session keeps its variables.
- Use an HTTP client timeout of `timeout + 15` seconds.

### Delete a session
`DELETE {url}/sessions/<session id>`. Always do this when the attempt ends, including on errors.

### Restart detection
- If `new_session_created` is true on any call after the first, the session was restarted and state is lost.
- Then re-run the init cell (§5) and every previously **completed** model cell, in order.
- Prefix the tool result with: `[note: the Python session was restarted; your earlier cells were re-run to restore state]`.

## 4. Command line
```
cd evals && ../.venv/bin/python harness.py --variant {plain,code} --sandbox-url URL
    [--model deepseek-flash] [--effort