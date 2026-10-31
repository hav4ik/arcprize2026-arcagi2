"""Havu Candidate B32 ARC-AGI-2 harness for Qwen3.8-27B on continuous TPU vLLM.

Faithfully implements Havu's latest best-performing harness (`Candidate B32` from
`hav4ik/arcprize2026-arcagi2` @ `main` `91ecbcc1`):
  1. Task-level multi-test rollouts (`ARC_MULTI_TEST=1`, `ARC_MULTI_BUDGET=0.5`, Fix A5 `inputs_prelude`).
  2. Retry-once AST-safe identifier typo auto-repair (`ARC_AUTO_REPAIR=1`, `havu_auto_repair.py`, Fix I5).
  3. Token-exact prompt history (`--token-exact` via `TokensPrompt`, zero mid-run `<think>` compaction).
  4. Uninterrupted thinking (`max_tokens=64000` per turn, `--cap-nudge` OFF, `--final-mode session`,
     plus up to 4 seamless continuations on spurious EOS inside open `<think>`).
  5. Candidate B32 budget & pacing (`80k` hard / `40k` soft `"gentle"`, `budget_notes=0.01`,
     `budget_warn=0.75`, `turns_note=5`, `max_turns=40`, `max_tool_output=16000`, `tool_timeout=30s`,
     context-aware `context_budget` Fix A9 + `ContextFull`).
  6. Immediate `submit_answer` acceptance (`--verify-submit` OFF).
  7. Isolated child-process Python sandbox per attempt (`LocalExecutor`, `style="ipython"`,
     `memory_mb=4096`, `fsize_mb=64`, `oom_score_adj=1000`, `OMP_NUM_THREADS=1`, 30s timeout with
     `SIGINT` + 5s grace then `SIGKILL` & clean respawn).
  8. All Candidate B32 session/fallback fixes (`A1–A11`: `restore_inputs`, shadow-proof `builtins as _b`,
     nested ndarray/0-d bool `_plain`, chunked `ANSWER_SLICE`, per-item partial multi-test recovery,
     `solve(list)` then `solve(np.array)`, `cut_parse`, error-path harvest).
  9. Code-outside-tool rescue (`--rescue-code` + `RESCUE_NUDGE` + `--lenient-parse` + `rescue_name`).
 10. 4-path zero-token post-attempt `train_check` (`expr` -> `cell` -> `fn` -> `transcript` fresh replay,
     Fix A6 `multi_check_code`) + `ranked_answers` selection.
 11. Runtime thinking gzip doom-loop guard (`LOOP_WINDOW=1000`, `LOOP_RATIO=8.0`, `LOOP_CONFIRM=10`)
     and smallest-prompt-first queue scheduling (`attempt=0` first, then `attempt>=1` prioritizing
     non-`train_pass` tasks).
"""

from __future__ import annotations

import ast
from concurrent import futures
import copy
import dataclasses
import gzip
import hashlib
import json
import os
import re
import select
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Callable

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import grid_image
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_auto_repair
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_sandbox_zygote
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector
from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens
from google3.pyglib import gfile


Grid = list[list[int]]

# ==============================================================================
# 1. CANDIDATE B32 CONSTANTS & PROMPTS
# ==============================================================================

CONTEXT_MIN_ROOM = 1024
LOOP_WINDOW = 1000
LOOP_EVERY = 1000
LOOP_RATIO = 8.0
LOOP_CONFIRM = 10
MAX_EOS_CONTINUES = 4

ARC_PRIZE_TEMPLATE = prompts_and_selector.E031_ARC_PRIZE_TEMPLATE
LEAN2SUB_SUFFIX = prompts_and_selector.E031_LEAN2SUB_SUFFIX
TIER_SYSTEM_XHIGH = prompts_and_selector.TIER_SYSTEM_XHIGH
EFFORT_HEADER_MINIMAL = prompts_and_selector.E031_EFFORT_HEADER_MINIMAL
TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_FOOTER = prompts_and_selector.E031_TOOLS_FOOTER
GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
THINK_CLOSE = "</think>"

MULTI_NOTE = (
    "\n\nThis task has {n} test inputs, shown above in order and preloaded in"
    " the session as `test_inputs` (a list of {n} grids; test_input is"
    " test_inputs[0]). One rule must work for all of them: check its"
    " assumptions on every test input. Submit a list with one output grid per"
    " test input, in order, e.g. submit_answer with `[solve(t) for t in"
    " test_inputs]`, and keep `answer` as that list."
)

RESCUE_NUDGE = (
    "[harness] No valid tool call was found in your last message. To run code,"
    " call the run_python tool (do not write code as plain text); to finish,"
    " call submit_answer."
)

PRELOADED = {"train", "test_input", "np", "json"}


def tools_spec_b32(multi_test: bool = True, tool_timeout: int = 30) -> list[dict[str, Any]]:
  """Exact `tools_for(args)` tool schema from Havu's `run_deepseek_tools.py`."""
  run_desc = (
      "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
      "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
      + (
          "test_inputs: list of every test input grid of the task (test_input is test_inputs[0]). "
          if multi_test
          else ""
      )
      + f"scipy is available. No internet or host files. Timeout {tool_timeout}s. "
      "Use print() to inspect results. Check transformations on all training pairs."
  )
  sub_desc = (
      "End this attempt with your test output. `code` is a Python expression evaluated in your session, "
      "e.g. solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or numpy array); "
      "statements that set a variable `answer` also work. The harness reads the grid directly: do not "
      "retype it. Invalid grids return an error and the attempt continues."
      + (
          " When the task has several test inputs, the expression must give a list with one grid per test "
          "input, in order, e.g. [solve(t) for t in test_inputs]."
          if multi_test
          else ""
      )
  )
  return [
      {
          "type": "function",
          "function": {
              "name": "run_python",
              "description": run_desc,
              "parameters": {
                  "type": "object",
                  "properties": {"code": {"type": "string"}},
                  "required": ["code"],
                  "additionalProperties": False,
              },
          },
      },
      {
          "type": "function",
          "function": {
              "name": "submit_answer",
              "description": sub_desc,
              "parameters": {
                  "type": "object",
                  "properties": {"code": {"type": "string"}},
                  "required": ["code"],
                  "additionalProperties": False,
              },
          },
      },
  ]


def build_prompt_single(train: list[dict[str, Any]], test_input: Grid) -> str:
  """Exact `arc_harness.arcprize.build_prompt`."""
  ex = ""
  for i, pair in enumerate(train):
    ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(pair["input"]) + "\n\n"
    ex += "OUTPUT: \n\n" + json.dumps(pair["output"]) + "\n\n"
  return ARC_PRIZE_TEMPLATE.format(
      training_examples=ex, test_input=json.dumps(test_input)
  )


def build_prompt_multi(train: list[dict[str, Any]], test_inputs: list[Grid]) -> str:
  """Exact `run_deepseek_tools.build_prompt_multi`."""
  tests = "\n\n".join(
      f"--Test input {i}--\n\n{json.dumps(g)}" for i, g in enumerate(test_inputs)
  )
  return build_prompt_single(train, [[0]]).replace(
      json.dumps([[0]]) + "\n--End of Test Input--",
      tests + "\n--End of Test Input--",
  )


def render_system_qwen(
    tools: list[dict[str, Any]],
    tier_text: str = TIER_SYSTEM_XHIGH,
    effort_header: str = EFFORT_HEADER_MINIMAL,
) -> str:
  """Exact `agent_qwen.render_system` for Qwen3.8 ChatML."""
  s = "<|im_start|>system\n"
  if tier_text:
    s += tier_text + "\n\n"
  s += (
      TOOLS_HEADER
      + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools)
      + TOOLS_FOOTER
  )
  if effort_header:
    s += "\n\n" + effort_header.strip()
  return s + "<|im_end|>\n"


def render_user_qwen(content: str) -> str:
  return f"<|im_start|>user\n{content.strip()}<|im_end|>\n"


def render_tool_responses_qwen(outputs: list[str]) -> str:
  return (
      "<|im_start|>user"
      + "".join(f"\n<tool_response>\n{o.strip()}\n</tool_response>" for o in outputs)
      + "<|im_end|>\n"
  )


def render_assistant_qwen(
    reasoning: str,
    content: str,
    calls: list[dict[str, Any]],
) -> str:
  """Canonical Qwen3.8 ChatML assistant turn rendering (used on resync)."""
  s = f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
  for i, call in enumerate(calls):
    name = call["function"]["name"]
    params = json.loads(call["function"]["arguments"])
    s += ("\n\n" if content.strip() else "") if i == 0 else "\n"
    s += f"<tool_call>\n<function={name}>\n"
    s += "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    s += "</function>\n</tool_call>"
  return s + "<|im_end|>\n"


def render_full_messages_qwen(
    system_block: str,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> str:
  """Render structured `messages` (`user` / `assistant` / `tool`) to Qwen3.8 ChatML."""
  chunks = [system_block]
  i = 0
  while i < len(messages):
    m = messages[i]
    role = m["role"]
    if role == "user":
      chunks.append(render_user_qwen(m["content"]))
      i += 1
    elif role == "assistant":
      chunks.append(
          render_assistant_qwen(
              m.get("reasoning_content", ""),
              m.get("content", ""),
              m.get("tool_calls", []),
          )
      )
      i += 1
    elif role == "tool":
      tool_outs = []
      while i < len(messages) and messages[i]["role"] == "tool":
        tool_outs.append(messages[i]["content"])
        i += 1
      chunks.append(render_tool_responses_qwen(tool_outs))
    else:
      i += 1
  if add_generation_prompt:
    chunks.append(GEN_PROMPT)
  return "".join(chunks)


# DeepSeek-V4.1 DSML formatting (`encoding/encoding.py` + `run_deepseek_tools.py`)
DS41_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DS41_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DS41_SYSTEM_SP_TOKEN = "<｜System｜>"
DS41_USER_SP_TOKEN = "<｜User｜>"
DS41_ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
DS41_THINK_START = "<think>"
DS41_GEN_PROMPT = f"{DS41_ASSISTANT_SP_TOKEN}{DS41_THINK_START}"
DS41_TOKEN = "｜DSML｜"
DS41_TC_OPEN = f"<{DS41_TOKEN} calls>"
DS41_TC_CLOSE = f"</{DS41_TOKEN} calls>"

DS41_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def render_system_dsv41(
    tools: list[dict[str, Any]],
    tier_text: str = "",
    reasoning_effort: int = 50,
    effort_header: str = "",
) -> str:
  """Exact `encoding/encoding.py` DeepSeek-V4.1 DSML system block."""
  s = DS41_BOS_TOKEN + DS41_SYSTEM_SP_TOKEN
  if reasoning_effort > 0:
    s += (
        f"Reasoning Effort: {reasoning_effort} "
        "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    )
  if tier_text:
    s += tier_text
  if tools:
    fn_schemas = [json.dumps(dict(t["function"]), ensure_ascii=False) for t in tools]
    s += "\n\n" + DS41_TOOLS_TEMPLATE.format(tool_schemas="\n".join(fn_schemas))
  if effort_header:
    s += "\n\n" + effort_header.strip()
  return s


def user_content_to_dsml_text(content: Any) -> str:
  """Plain text of a user message for DSML rendering.

  Strings pass through. OpenAI-style content parts (E058 text+image prompts) are rendered exactly
  like DeepSeek's `encoding.py::process_image_messages`: every `image_url` part becomes the
  `<｜deepseek_image｜>` placeholder (one token; expanded to its image span at tokenization time)
  and the text parts are joined with `"\\n\\n"`.
  """
  if isinstance(content, str):
    return content
  texts: list[str] = []
  for part in content:
    ptype = part.get("type")
    if ptype == "text":
      texts.append(part.get("text", ""))
    elif ptype in ("image_url", "image"):
      texts.append(dsv41_image_tokens.IMAGE_PLACEHOLDER)
    else:
      raise ValueError(f"Unsupported user content part type: {ptype!r}")
  return "\n\n".join(texts)


def render_user_dsv41(content: Any) -> str:
  return f"{DS41_USER_SP_TOKEN}{user_content_to_dsml_text(content)}"


def render_tool_responses_dsv41(outputs: list[str]) -> str:
  return DS41_USER_SP_TOKEN + "\n\n".join(
      f"<tool_result>{o}</tool_result>" for o in outputs
  )


def render_assistant_dsv41(
    reasoning: str,
    content: str,
    calls: list[dict[str, Any]],
) -> str:
  """Canonical DeepSeek-V4.1 DSML assistant turn rendering (`encoding/encoding.py`)."""
  s = f"{DS41_ASSISTANT_SP_TOKEN}{DS41_THINK_START}{reasoning}{THINK_CLOSE}{content}"
  if calls:
    tc_list = []
    for call in calls:
      name = call["function"]["name"]
      params = json.loads(call["function"]["arguments"])
      param_parts = []
      for k, v in params.items():
        is_str = "true" if isinstance(v, str) else "false"
        val_s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        param_parts.append(
            f'<{DS41_TOKEN} parameter name="{k}" string="{is_str}">{val_s}</{DS41_TOKEN} parameter>'
        )
      arg_str = "\n".join(param_parts)
      tc_list.append(
          f'<{DS41_TOKEN} invoke name="{name}">\n{arg_str}\n</{DS41_TOKEN} invoke>'
      )
    tc_joined = "\n".join(tc_list)
    s += f"\n\n<{DS41_TOKEN} calls>\n{tc_joined}\n</{DS41_TOKEN} calls>"
  return s + DS41_EOS_TOKEN


def render_full_messages_dsv41(
    system_block: str,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> str:
  """Render structured `messages` (`user` / `assistant` / `tool`) to DeepSeek-V4.1 DSML."""
  chunks = [system_block]
  i = 0
  while i < len(messages):
    m = messages[i]
    role = m["role"]
    if role == "user":
      chunks.append(render_user_dsv41(m["content"]))
      i += 1
    elif role == "assistant":
      chunks.append(
          render_assistant_dsv41(
              m.get("reasoning_content", ""),
              m.get("content", ""),
              m.get("tool_calls", []),
          )
      )
      i += 1
    elif role == "tool":
      tool_outs = []
      while i < len(messages) and messages[i]["role"] == "tool":
        tool_outs.append(messages[i]["content"])
        i += 1
      # If immediately followed by a user message (e.g. rescue nudge), merge into same User turn
      while i < len(messages) and messages[i]["role"] == "user":
        tool_outs.append(messages[i]["content"])
        i += 1
      chunks.append(render_tool_responses_dsv41(tool_outs))
    else:
      i += 1
  if add_generation_prompt:
    chunks.append(DS41_GEN_PROMPT)
  return "".join(chunks)


# ==============================================================================
# 2. ISOLATED CHILD-PROCESS PYTHON SANDBOX (`LocalExecutor`, `style="ipython"`)
# ==============================================================================

_CHILD = r'''
import ast, contextlib, io, json, os, resource, signal, sys, traceback
resp = os.fdopen(int(sys.argv[1]), "w")
limit = int(sys.argv[2])
mem, cpu, fsize, nice = (int(x) for x in sys.argv[3:7])
style = sys.argv[7]
extra_paths = json.loads(sys.argv[8]) if len(sys.argv) > 8 else []
if extra_paths:
    seen = set(sys.path)
    sys.path.extend([p for p in extra_paths if p and p not in seen])
if cpu:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
if fsize:
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
if nice:
    try:
        os.nice(nice)
    except OSError:
        pass
try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
except OSError:
    pass
ns = {"__name__": "__main__"}

def run(code):
    tree = ast.parse(code, "<cell>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), ns)
    if last is not None:
        value = eval(compile(last, "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))

def plain_traceback(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename != "<string>"]
    text = "".join(traceback.format_list(frames)) if frames else ""
    return "Traceback (most recent call last):\n" + text + "".join(traceback.format_exception_only(type(exc), exc))

if style == "ipython":
    import hashlib, linecache, pprint, re
    try:
        from IPython.lib.pretty import pretty as _pretty
    except Exception:
        _pretty = None

    def show(value):
        if _pretty is not None:
            text = _pretty(value, max_width=79)
        elif isinstance(value, (list, tuple, dict, set, frozenset)):
            text = pprint.pformat(value, width=79, sort_dicts=False)
        else:
            text = repr(value)
        print("\n" + text if "\n" in text else text)

    def run(code):
        cell = code if code.endswith("\n") else code + "\n"
        name = "<ipython-input-1-%s>" % hashlib.sha1(cell.encode()).hexdigest()[:12]
        linecache.cache[name] = (len(code), None, code.splitlines(True), name)
        tree = ast.parse(code, name, "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, name, "exec"), ns)
        if last is not None:
            value = eval(compile(last, name, "eval"), ns)
            if value is not None:
                show(value)

    def ipython_traceback(exc):
        if isinstance(exc, SyntaxError):
            lines = traceback.format_exception_only(type(exc), exc)
            text = "".join(lines)
            return text.split("\n", 1)[1] if text.startswith("  File ") else text
        s = "Traceback (most recent call last):\n"
        for f in traceback.extract_tb(exc.__traceback__):
            if f.filename == "<string>":
                continue
            if not f.filename.startswith("<ipython-input-"):
                s += "  File %s:%d in %s\n" % (f.filename, f.lineno, f.name)
            if f.line:
                s += "    " + f.line.strip() + "\n"
        msg = str(exc)
        return s + type(exc).__name__ + (": " + msg if msg else "") + "\n"

    def post(text):
        lines = [re.sub(r"^Out\[\d+\]:\s*", "", l) for l in text.split("\n")
                 if not l.strip().startswith("File <ipython-")]
        while lines and lines[0] == "":
            lines.pop(0)
        text = "\n".join(lines).rstrip()
        text = text + ("\n" if text else "")
        return text[:limit] + "<output cut>" if len(text) > limit else text

for line in sys.stdin:
    req = json.loads(line)
    out, err = io.StringIO(), io.StringIO()
    status = "completed"
    if style == "ipython":
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    run(req["code"])
                except KeyboardInterrupt as exc:
                    status = "timeout"
                    out.write(ipython_traceback(exc))
                except BaseException as exc:
                    status = "error"
                    out.write(ipython_traceback(exc))
        except KeyboardInterrupt:
            status = "timeout"
        except BaseException as exc:
            status = "error"
            out.write("%s: %s\n" % (type(exc).__name__, exc))
        resp.write(json.dumps({"process_status": status, "stdout": post(out.getvalue()),
                               "stderr": post(err.getvalue())}) + "\n")
        resp.flush()
        continue
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run(req["code"])
    except KeyboardInterrupt:
        status = "timeout"
        err.write("KeyboardInterrupt: cell timed out (state kept)\n")
    except BaseException as exc:
        status = "error"
        try:
            err.write(plain_traceback(exc))
        except BaseException:
            err.write(f"{type(exc).__name__}: {exc}\n")
    o, e = out.getvalue(), err.getvalue()
    if len(o) > limit:
        o = o[:limit] + "\n[output truncated]"
    if len(e) > limit:
        e = e[:limit] + "\n[output truncated]"
    resp.write(json.dumps({"process_status": status, "stdout": o, "stderr": e}) + "\n")
    resp.flush()
'''

_SESSION_MEM_CAP = """
def _arc_memcap():
    import resource
    try:
        with open('/proc/self/status') as f:
            vm = next(int(l.split()[1]) * 1024 for l in f if l.startswith('VmSize:'))
        cap = vm + {mb} * 2 ** 20
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY or cap < hard:
            resource.setrlimit(resource.RLIMIT_AS, (cap, hard if hard != resource.RLIM_INFINITY else cap))
    except Exception:
        pass
_arc_memcap()
del _arc_memcap
"""


class LocalExecutor:
  """Persistent Python session in an isolated child process (`arc_harness.executor.LocalExecutor`).

  Workers are forked on demand by `havu_sandbox_zygote` (a pre-TPU fork server) because hermetic
  google3 binaries have `sys.executable is None` on Borg; each worker runs `_CHILD` in its own
  session / process group with stdin and the response channel bound to one `socketpair`.
  """

  def __init__(
      self,
      train: list[dict[str, Any]],
      test_input: Grid,
      timeout: float = 30.0,
      max_output: int = 16000,
      memory_mb: int = 4096,
      grace: float = 5.0,
      python: str | None = None,
      cpu_seconds: int = 7200,
      fsize_mb: int = 64,
      nice: int = 10,
      start_timeout: float = 90.0,
      prelude_extra: str = "",
      style: str = "ipython",
  ) -> None:
    del python  # Workers are forked from the zygote; no interpreter path is needed.
    self.train = train
    self.test_input = test_input
    self.style = style
    self.timeout = timeout
    self.max_output = max_output
    self.memory_mb = memory_mb
    self.grace = grace
    self.cpu_seconds = cpu_seconds
    self.fsize_mb = fsize_mb
    self.nice = nice
    self.start_timeout = start_timeout
    self.pid: int | None = None
    self.sock: socket.socket | None = None
    self.resp: int | None = None
    self.buf = b""
    self.closed = False
    self.last_failure = "error"
    data = json.dumps({"train": train, "test_input": test_input})
    mem_prelude = _SESSION_MEM_CAP.format(mb=memory_mb) if memory_mb > 0 else ""
    self.prelude = (
        "import json, numpy as np\n"
        f"_arc_data = json.loads({data!r})\n"
        "train, test_input = _arc_data['train'], _arc_data['test_input']\n"
        "del _arc_data\n"
        + mem_prelude
        + prelude_extra
    )

  @property
  def proc(self) -> int | None:
    """Backwards-compatible liveness handle (worker pid or None)."""
    return self.pid

  def _spawn_once(self, timeout: float) -> dict[str, Any] | None:
    work_cwd = os.environ.get("TEST_TMPDIR") or tempfile.gettempdir()
    self.pid, self.sock = havu_sandbox_zygote.spawn(
        _CHILD,
        max_output=self.max_output,
        mem_bytes=self.memory_mb * 2**20,
        cpu_seconds=int(self.cpu_seconds),
        fsize_bytes=self.fsize_mb * 2**20,
        nice=self.nice,
        style=self.style,
        cwd=work_cwd,
    )
    self.resp = self.sock.fileno()
    self.buf = b""
    return self._run(self.prelude, timeout)

  def _spawn(self) -> None:
    result = None
    for t_lim in (max(self.timeout, 30.0), self.start_timeout):
      result = self._spawn_once(t_lim)
      if result is not None and result.get("process_status") == "completed":
        return
      self._kill()
    detail = "" if result is None else (result.get("stderr", "") or result.get("stdout", ""))[:500]
    raise RuntimeError(f"Local executor initialization failed {detail}".strip())

  def _alive(self) -> bool:
    if self.pid is None:
      return False
    try:
      os.kill(self.pid, 0)
      return True
    except ProcessLookupError:
      return False
    except PermissionError:
      return True

  def _read(self, deadline: float) -> dict[str, Any] | str | None:
    assert self.resp is not None
    # NOTE: poll(), not select(): the vLLM/TPU host process holds thousands of fds, and
    # select() raises `ValueError: filedescriptor out of range` for fds >= 1024.
    poller = select.poll()
    poller.register(self.resp, select.POLLIN | select.POLLHUP | select.POLLERR)
    while b"\n" not in self.buf:
      wait_s = deadline - time.monotonic()
      if wait_s <= 0:
        return None
      try:
        events = poller.poll(max(1, int(wait_s * 1000.0)))
      except (OSError, ValueError):
        return "eof"
      if events:
        try:
          chunk = os.read(self.resp, 1 << 16)
        except OSError:
          return "eof"
        if not chunk:
          return "eof"
        self.buf += chunk
    line, self.buf = self.buf.split(b"\n", 1)
    return json.loads(line)

  def _signal_group(self, sig: int) -> None:
    if self.pid is None:
      return
    try:
      os.killpg(self.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
      try:
        os.kill(self.pid, sig)
      except (ProcessLookupError, OSError):
        pass

  def _kill(self) -> None:
    if self.pid is not None:
      self._signal_group(signal.SIGKILL)
      deadline = time.monotonic() + 10.0
      while self._alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if self.sock is not None:
      try:
        self.sock.close()
      except OSError:
        pass
    self.pid = None
    self.sock = None
    self.resp = None
    self.buf = b""

  def _run(self, code: str, timeout: float | None = None) -> dict[str, Any] | None:
    self.last_failure = "error"
    if self.pid is None or self.sock is None:
      return None
    try:
      self.sock.sendall((json.dumps({"code": code}) + "\n").encode("utf-8"))
    except (BrokenPipeError, OSError):
      return None
    result = self._read(time.monotonic() + (timeout or self.timeout))
    if result is None:
      self._signal_group(signal.SIGINT)
      self.last_failure = "timeout"
      result = self._read(time.monotonic() + self.grace)
      if isinstance(result, dict):
        result["process_status"] = "timeout"
        return result
      return None
    return result if isinstance(result, dict) else None

  def initialize(self) -> None:
    self._spawn()

  def execute(self, code: str) -> dict[str, Any]:
    if self.closed:
      raise RuntimeError("Executor is closed")
    if self.pid is None:
      self._spawn()
    result = self._run(code)
    if result is None:
      failure = self.last_failure  # `_spawn()` below re-runs the prelude and would reset it.
      self._kill()
      try:
        self._spawn()
      except RuntimeError as exc:
        return {
            "process_status": "error",
            "stdout": "",
            "state_reset": True,
            "stderr": f"Python process died and could not be restarted: {exc}",
        }
      result = {
          "process_status": failure,
          "stdout": "",
          "state_reset": True,
          "stderr": (
              "Cell did not stop after the timeout; process restarted."
              if failure == "timeout"
              else "Python process died (crash, exit or memory limit); process restarted."
          ),
      }
      if self.style == "ipython" and failure == "timeout":
        result["stderr"] = f"Execution timed out after {self.timeout:g} seconds\n"
    return result

  def close(self) -> None:
    if not self.closed:
      self.closed = True
      self._kill()

  def __enter__(self) -> LocalExecutor:
    try:
      self.initialize()
    except BaseException:
      self.close()
      raise
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()


# ==============================================================================
# 3. ANSWER SUBMISSION, FALLBACKS (A1-A11), AUTO-REPAIR (I5) & RESCUE HELPERS
# ==============================================================================


def valid_grid(grid: Any) -> bool:
  return (
      isinstance(grid, list)
      and 1 <= len(grid) <= 30
      and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
      and len({len(row) for row in grid}) == 1
      and all(type(value) is int and 0 <= value <= 9 for row in grid for value in row)  # pylint: disable=unidiomatic-typecheck
  )


def grid_problem(grid: Any) -> str | None:
  """Diagnostic description of why `grid` is invalid (`A11` feedback)."""
  if not isinstance(grid, list):
    return type(grid).__name__
  if not 1 <= len(grid) <= 30:
    return f"{len(grid)} rows"
  if not all(isinstance(row, list) for row in grid):
    return "its rows are not lists"
  lengths = sorted({len(row) for row in grid})
  if len(lengths) > 1 or not 1 <= lengths[0] <= 30:
    return f"{len(grid)} rows, row lengths {lengths[:6]}"
  for row in grid:
    for value in row:
      if not (type(value) is int and 0 <= value <= 9):  # pylint: disable=unidiomatic-typecheck
        return f"value {json.dumps(value)[:20]} is not an int 0-9"
  return None


def tool_output(result: dict[str, Any], limit: int = 16000) -> str:
  text = result.get("stdout", "")
  if result.get("stderr"):
    text += "\n[stderr]\n" + result["stderr"]
  if result.get("state_reset"):
    text = (
        "[Python state lost. train/test_input/np restored; redefine other variables.]\n"
        + text
    )
  text = f"[{result['process_status']}]\n" + (text or "[no output; use print()]")
  return text if len(text) <= limit else text[:limit] + "\n[output truncated]"


_ANSWER_PLAIN = """
    def _plain(v):
        if _b.getattr(v, 'ndim', None) == 0 and _b.getattr(_b.getattr(v, 'dtype', None), 'kind', '') == 'b':
            return _b.int(v)
        if _b.hasattr(v, 'tolist'):
            return v.tolist()
        return [_plain(x) for x in v] if _b.isinstance(v, (_b.list, _b.tuple)) else v"""

_ANSWER_CELL = """
def _arc_answer():
    import builtins as _b, contextlib, io, json{plain}
{body}
    if _b.len(text) > {chunk}:
        _b.globals()[{name!r}] = text
    _b.print({marker!r} + _b.str(_b.len(text)) + ':' + text[:{chunk}])
try:
    _arc_answer()
finally:
    del _arc_answer
"""

_ANSWER_SLICE = """
def _arc_slice():
    import builtins as _b
    _b.print({marker!r} + _b.globals()[{name!r}][{start}:{end}])
try:
    _arc_slice()
finally:
    del _arc_slice
"""

_ANSWER_DROP = """
def _arc_drop():
    import builtins as _b
    _b.globals().pop({name!r}, None)
try:
    _arc_drop()
finally:
    del _arc_drop
"""


def _answer_value(session: LocalExecutor, body: str) -> tuple[dict[str, Any], Any, bool]:
  """Execute an answer-extraction cell with shadow-proof builtins (`A7`) and chunked slicing (`A11`)."""
  marker = f"ARC_ANSWER_{uuid.uuid4().hex}:"
  name = f"_arc_text_{uuid.uuid4().hex[:12]}"
  chunk = max(
      1000,
      int(getattr(session, "max_output", 16000) or 16000) - len(marker) - 64,
  )
  cell = _ANSWER_CELL.format(
      plain=_ANSWER_PLAIN, body=body, chunk=chunk, name=name, marker=marker
  )
  result = session.execute(cell)
  if result["process_status"] != "completed":
    return result, None, False

  def marked(res: dict[str, Any]) -> str | None:
    for line in reversed(res.get("stdout", "").splitlines()):
      if line.startswith(marker):
        return line[len(marker) :]
    return None

  head = marked(result)
  if head is None:
    return result, None, False
  size_s, _, text = head.partition(":")
  try:
    size = int(size_s)
    while len(text) < size:
      part = session.execute(
          _ANSWER_SLICE.format(
              marker=marker,
              name=name,
              start=len(text),
              end=min(size, len(text) + chunk),
          )
      )
      piece = marked(part) if part["process_status"] == "completed" else None
      if not piece:
        break
      text += piece
    if size > chunk:
      session.execute(_ANSWER_DROP.format(name=name))
    return result, json.loads(text) if len(text) == size else None, True
  except (ValueError, KeyError):
    return result, None, True


def submit_answer(
    session: LocalExecutor, code: str, n: int | None = None
) -> tuple[str, Any]:
  """Evaluate `submit_answer(code)` in `session` (`A7`, `A10`, `A11`)."""
  body = f"""    _src = {code!r}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            _expr = _b.compile(_src, '<answer>', 'eval')
        except _b.SyntaxError:
            _expr = None
        if _expr is not None:
            value = _b.eval(_expr, _b.globals())
        else:
            _b.exec(_b.compile(_src, '<answer>', 'exec'), _b.globals())
            if 'answer' not in _b.globals():
                raise _b.NameError('code did not define answer')
            value = _b.globals()['answer']
        if _b.hasattr(value, 'tolist'):
            value = value.tolist()
        value = _plain(value)
        text = json.dumps(value, default=_b.int, separators=(',', ':'))"""
  result, grid, found = _answer_value(session, body)
  if result["process_status"] != "completed":
    return tool_output(result, session.max_output)[:2000] + "\n[answer not accepted]", None
  if not found:
    return "[answer not accepted] no value returned", None
  if n is not None:
    if not (
        isinstance(grid, list)
        and len(grid) == n
        and all(valid_grid(g) for g in grid)
    ):
      bad = (
          f"a list of {len(grid)} items"
          if isinstance(grid, list) and not valid_grid(grid)
          else "one grid" if valid_grid(grid) else type(grid).__name__
      )
      if isinstance(grid, list) and len(grid) == n and not valid_grid(grid):
        probs = [(i, grid_problem(g)) for i, g in enumerate(grid)]
        bad += "".join(
            f"; grid {i} (test input {i}): {p}" for i, p in probs if p
        )
      return (
          f"[answer not accepted] this task has {n} test inputs: give a list of"
          f" {n} rectangular 1-30 by 1-30 grids of ints 0-9, one per test input"
          " in order (e.g. [solve(t) for t in test_inputs]); got "
          + bad
      ), None
    return (
        f"[answer accepted: {n} grids, "
        + ", ".join(f"{len(g)}x{len(g[0])}" for g in grid)
        + "]"
    ), grid
  if not valid_grid(grid):
    shape = (
        f"{len(grid)} rows, row lengths"
        f" {sorted({len(r) for r in grid if isinstance(r, list)})}"
        if isinstance(grid, list)
        else type(grid).__name__
    )
    return (
        "[answer not accepted] must be a rectangular 1-30 by 1-30 grid of ints"
        " 0-9; got "
        + shape
    ), None
  return f"[answer accepted: {len(grid)}x{len(grid[0])} grid]", grid


def _items_result(
    result: dict[str, Any], items: Any, found: bool, n: int, what: str
) -> tuple[str, list[Grid | None] | None]:
  if result["process_status"] != "completed" or not found:
    text = (
        tool_output(result, 2000)[:300]
        if result["process_status"] != "completed"
        else "no value returned"
    )
    return f"[{what}: {text}]", None
  if not isinstance(items, list) or len(items) != n:
    return f"[{what}: not a list of {n} items]", None
  grids = [g if valid_grid(g) else None for g in items]
  probs = "".join(
      f"; grid {i}: {grid_problem(g)}"
      for i, g in enumerate(items)
      if not valid_grid(g)
  )
  return (
      f"[{what}: {n - grids.count(None)} of {n} grids valid{probs}]",
      grids if any(grids) else None,
  )


def answer_items(
    session: LocalExecutor, code: str, n: int
) -> tuple[str, list[Grid | None] | None]:
  """Fix `A1`: validate each item of a multi-test `answer` independently."""
  body = f"""    with contextlib.redirect_stdout(io.StringIO()):
        value = _b.eval(_b.compile({code!r}, '<answer>', 'eval'), _b.globals())
        if _b.hasattr(value, 'tolist'):
            value = value.tolist()
        parts = []
        if _b.isinstance(value, (_b.list, _b.tuple)) and _b.len(value) == {n}:
            for x in value:
                try:
                    parts.append(json.dumps(_plain(x), default=_b.int, separators=(',', ':')))
                except _b.Exception:
                    parts.append('null')
        text = '[' + ','.join(parts) + ']' if parts else 'null'"""
  result, items, found = _answer_value(session, body)
  return _items_result(result, items, found, n, "session answer")


def solve_items(
    session: LocalExecutor, inputs: list[Grid]
) -> tuple[str, list[Grid | None] | None]:
  """Fixes `A1` / `A3` / `A8`: evaluate `solve(t)` per input (trying `list` then `np.array`)."""
  per = max(1.0, float(getattr(session, "timeout", 30.0) or 30.0))
  body = f"""    import copy, signal
    class _T(_b.Exception):
        pass
    def _h(*a):
        raise _T()
    _solve, parts = _b.globals().get('solve'), []
    old = signal.signal(signal.SIGALRM, _h)
    try:
        for t in json.loads({json.dumps(inputs)!r}):
            v = None
            if _b.callable(_solve):
                for arg in ('list', 'numpy'):
                    signal.setitimer(signal.ITIMER_REAL, {per})
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            if arg == 'list':
                                v = _solve(copy.deepcopy(t))
                            else:
                                import numpy
                                v = _solve(numpy.array(t))
                        break
                    except (_b.Exception, _T):
                        v = None
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                if _b.hasattr(v, 'tolist'):
                    v = v.tolist()
                parts.append(json.dumps(_plain(v), default=_b.int, separators=(',', ':')))
            except _b.Exception:
                parts.append('null')
    finally:
        signal.signal(signal.SIGALRM, old)
    text = '[' + ','.join(parts) + ']'"""
  old_timeout = getattr(session, "timeout", None)
  if old_timeout is not None:
    session.timeout = per * 2 * len(inputs) + old_timeout
  try:
    result, items, found = _answer_value(session, body)
  finally:
    if old_timeout is not None:
      session.timeout = old_timeout
  return _items_result(result, items, found, len(inputs), "session solve")


def restore_inputs(
    session: LocalExecutor, test_inputs: list[Grid], with_inputs: bool
) -> None:
  """Fix `A8`: restore host copies of `test_input` / `test_inputs` before fallbacks."""
  names: dict[str, Any] = {"test_input": test_inputs[0]}
  if with_inputs:
    names["test_inputs"] = test_inputs
  session.execute(f"""
def _arc_restore():
    import builtins as _b, json
    _g = _b.globals()
    for _k, _v in json.loads({json.dumps(names)!r}).items():
        _g[_k] = _v
try:
    _arc_restore()
finally:
    del _arc_restore
""")


def multi_check_code(code: str | None, test_inputs: list[Grid]) -> str | None:
  """Fix `A6`: wrap a multi-test submission into a 1-grid program of `test_input` for `train_check`."""
  if code is None:
    return None
  if code == "answer":
    return "answer[0]"
  rest = json.dumps(test_inputs[1:])
  try:
    compile(code, "<answer>", "eval")
  except SyntaxError:
    return f"test_inputs = [test_input] + {rest}\n{code}\nanswer = answer[0]"
  return f"(lambda test_inputs: ({code})[0])([test_input] + {rest})"


def inputs_prelude(test_inputs: list[Grid]) -> str:
  """Fix `A5`: preload `test_inputs` in all sessions when `multi_test=True`."""
  return f"\nimport json as _arc_json\ntest_inputs = _arc_json.loads({json.dumps(test_inputs)!r})\n"


_NAMES_CELL = """
def _arc_names():
    import builtins as _b, json
    _b.print({marker!r} + json.dumps(_b.sorted(k for k in _b.globals() if not k.startswith('_'))))
try:
    _arc_names()
finally:
    del _arc_names
"""


def session_names(session: LocalExecutor) -> set[str] | None:
  """Fix `I5`: read public variable/function names currently defined in `session`."""
  marker = f"ARC_NAMES_{uuid.uuid4().hex}:"
  result = session.execute(_NAMES_CELL.format(marker=marker))
  lines = (
      result.get("stdout", "").splitlines()
      if result.get("process_status") == "completed"
      else []
  )
  for line in reversed(lines):
    if line.startswith(marker):
      try:
        return set(json.loads(line[len(marker) :]))
      except ValueError:
        return None
  return None


def auto_repair_cell(
    session: LocalExecutor,
    code: str,
    output: str,
    names: set[str],
    limit: int,
    stats: dict[str, int],
) -> tuple[str, str, str] | None:
  """Fix `I5`: retry a failed `run_python` cell once if `havu_auto_repair.propose` finds a safe unique fix."""
  fix = havu_auto_repair.propose(code, output, (session_names(session) or set()) | names)
  if fix is None:
    return None
  new_code, note = fix
  stats["proposed"] += 1
  result = session.execute(new_code)
  stats["applied"] += 1
  stats["retry_ok"] += int(result["process_status"] == "completed")
  return note + "\n" + tool_output(result, limit), result["process_status"], new_code


# ==============================================================================
# 4. QWEN XML COMPLETION PARSER, CODE RESCUE (`--rescue-code`), & GRID PARSERS
# ==============================================================================

_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC_BLOCK_RE = re.compile(
    r"^\s*<function=([^>\n]+)>(.*?)</function>\s*$", re.DOTALL
)
_PARAM_BLOCK_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL
)

_RESCUE_XML_PARAM = re.compile(
    r"<parameter=code>\n?(.*?)(?:\n?</parameter>|$)", re.DOTALL
)
_RESCUE_DSML_CODE = re.compile(
    r'parameter name="code"[^>]*>(.*?)(?:</[^>]*parameter>|$)', re.DOTALL
)
_RESCUE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.DOTALL)
_RESCUE_JSON_CODE = re.compile(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_RESCUE_TAG = re.compile(
    r"</?(?:tool_call|function[^>\n]*|parameter[^>\n]*|[^<>\n]*DSML[^<>\n]*)>|<br[^>]*>"
)
_RESCUE_CODE_LINE = re.compile(
    r"^\s*(import |from \w+ import |print\(|for |def |if |while |np\.|[A-Za-z_]\w*\s*(\[[^\]]*\])?\s*=[^=])"
)


def parse_qwen_completion(
    raw_text: str,
) -> tuple[str, str, list[dict[str, Any]], bool]:
  """Parse a Qwen3.8 completion (after `<|im_start|>assistant\\n<think>\\n`) into `(reasoning, content, calls, clean)`."""
  has_im_end = raw_text.endswith("<|im_end|>")
  text = raw_text[:-10] if has_im_end else raw_text
  think_pos = text.find(THINK_CLOSE)
  if think_pos < 0:
    return text, "", [], False
  reasoning = text[:think_pos]
  rest = text[think_pos + len(THINK_CLOSE) :]
  calls: list[dict[str, Any]] = []
  clean = has_im_end
  for block in _CALL_BLOCK_RE.findall(rest):
    m = _FUNC_BLOCK_RE.match(block)
    if not m:
      clean = False
      continue
    fn_name = m.group(1).strip()
    params = {k.strip(): v for k, v in _PARAM_BLOCK_RE.findall(m.group(2))}
    calls.append({
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": fn_name,
            "arguments": json.dumps(params, ensure_ascii=False),
        },
    })
  content_wo_calls = _CALL_BLOCK_RE.sub("", rest)
  if "<tool_call>" in content_wo_calls or "<function=" in content_wo_calls:
    clean = False
  if any(tok in reasoning for tok in ("<|im_end|>", THINK_CLOSE)):
    clean = False
  content_clean = content_wo_calls.strip()
  return reasoning, content_clean, calls, clean


_DSML_CALLS_BLOCK_RE = re.compile(
    rf"<{DS41_TOKEN}\s+calls>\n?(.*?)\n?</{DS41_TOKEN}\s+calls>", re.DOTALL
)
_DSML_INVOKE_RE = re.compile(
    rf'<{DS41_TOKEN}\s+invoke\s+name="([^"]+)">\n?(.*?)\n?</{DS41_TOKEN}\s+invoke>',
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    rf'<{DS41_TOKEN}\s+parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)</{DS41_TOKEN}\s+parameter>',
    re.DOTALL,
)


def parse_dsv41_completion(
    raw_text: str,
) -> tuple[str, str, list[dict[str, Any]], bool]:
  """Parse a DeepSeek-V4.1 DSML completion (after `<｜Assistant｜><think>`) into `(reasoning, content, calls, clean)`."""
  has_eos = raw_text.endswith(DS41_EOS_TOKEN)
  text = raw_text[: -len(DS41_EOS_TOKEN)] if has_eos else raw_text
  think_pos = text.find(THINK_CLOSE)
  if think_pos < 0:
    return text, "", [], False
  reasoning = text[:think_pos]
  rest = text[think_pos + len(THINK_CLOSE) :]
  calls: list[dict[str, Any]] = []
  clean = has_eos
  for calls_block in _DSML_CALLS_CALLS_FINDALL(rest):
    invokes = _DSML_INVOKE_RE.findall(calls_block)
    if not invokes:
      clean = False
      continue
    for fn_name, inv_body in invokes:
      params: dict[str, Any] = {}
      for p_name, is_str_flag, p_val in _DSML_PARAM_RE.findall(inv_body):
        if is_str_flag == "true":
          params[p_name] = p_val
        else:
          try:
            params[p_name] = json.loads(p_val)
          except ValueError:
            params[p_name] = p_val
            clean = False
      calls.append({
          "id": f"call_{uuid.uuid4().hex[:24]}",
          "type": "function",
          "function": {
              "name": fn_name.strip(),
              "arguments": json.dumps(params, ensure_ascii=False),
          },
      })
  content_wo_calls = _DSML_CALLS_BLOCK_RE.sub("", rest)
  if DS41_TOKEN in content_wo_calls:
    clean = False
  if any(tok in reasoning for tok in (DS41_EOS_TOKEN, THINK_CLOSE)):
    clean = False
  content_clean = content_wo_calls.strip()
  return reasoning, content_clean, calls, clean


def _DSML_CALLS_CALLS_FINDALL(text: str) -> list[str]:
  return _DSML_CALLS_BLOCK_RE.findall(text)


def rescue_code(content: str | None) -> str | None:
  """Exact `run_deepseek_tools.rescue_code` extended with Qwen `<parameter=code>` fragments."""
  text = content or ""
  m = _RESCUE_XML_PARAM.findall(text)
  if m and m[-1].strip():
    return _RESCUE_TAG.sub("", m[-1]).strip() or None
  m = _RESCUE_DSML_CODE.findall(text)
  if m and m[-1].strip():
    return _RESCUE_TAG.sub("", m[-1]).strip() or None
  m = _RESCUE_JSON_CODE.findall(text)
  if m:
    try:
      val = json.loads('"' + m[-1] + '"')
      if val and val.strip():
        return val
    except ValueError:
      pass
  f = [b for b in _RESCUE_FENCE.findall(text) if b.strip()]
  if f:
    return _RESCUE_TAG.sub("", f[-1]).strip() or None
  lines = _RESCUE_TAG.sub("", text).splitlines()
  start = next(
      (i for i, l in enumerate(lines) if _RESCUE_CODE_LINE.match(l)), None
  )
  if start is None:
    return None
  code = "\n".join(lines[start:]).strip()
  return code or None


def rescue_tool_call(content: str | None, turn_num: int) -> dict[str, Any] | None:
  code = rescue_code(content)
  if not code or not code.strip():
    return None
  return {
      "id": f"rescue_{turn_num}",
      "type": "function",
      "function": {
          "name": "run_python",
          "arguments": json.dumps({"code": code}),
      },
  }


def rescue_name(name: str | None) -> str:
  low = (name or "").lower()
  if "submit" in low:
    return "submit_answer"
  if "python" in low or "run" in low:
    return "run_python"
  return name or ""


def parse_arcprize_grid(s: str | None) -> Grid | None:
  """Exact `arc_harness.arcprize.parse` (`\\boxed{...}` then `backscan_json_parser`)."""
  text = s or ""
  m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
  if m:
    try:
      g = json.loads(m.group(1).strip())
      if isinstance(g, list) and g and all(isinstance(r, list) for r in g):
        return g
    except json.JSONDecodeError:
      pass
  end = max(text.rfind("]"), text.rfind("}"))
  if end < 0:
    return None
  close_ch = text[end]
  open_ch = "[" if close_ch == "]" else "{"
  depth = 1
  for i in range(end - 1, -1, -1):
    depth += (text[i] == close_ch) - (text[i] == open_ch)
    if depth == 0:
      try:
        g = json.loads(text[i : end + 1])
        return (
            g
            if isinstance(g, list) and g and all(isinstance(r, list) for r in g)
            else None
        )
      except json.JSONDecodeError:
        return None
  return None


def lenient_parse(text: str | None) -> Grid | None:
  """Exact `run_deepseek_tools.lenient_parse`."""
  text = text or ""
  found = None
  for start in [i for i, ch in enumerate(text) if ch == "["][::-1]:
    if not text.startswith("[[", start) and not re.match(
        r"\[\s*\[", text[start : start + 50]
    ):
      continue
    depth = 0
    for j in range(start, len(text)):
      depth += (text[j] == "[") - (text[j] == "]")
      if depth == 0:
        block = text[start : j + 1]
        try:
          grid = json.loads(block)
        except json.JSONDecodeError:
          try:
            grid = json.loads(re.sub(r",\s*\]", "]", block))
          except json.JSONDecodeError:
            grid = None
        if valid_grid(grid):
          found = grid
        break
    if found is not None:
      return found
  rows: list[list[int]] = []
  best = None
  for line in text.splitlines() + [""]:
    line = line.strip().rstrip(",")
    try:
      row = json.loads(line) if line.startswith("[") else None
    except json.JSONDecodeError:
      row = None
    if isinstance(row, list) and row and all(type(v) is int for v in row):  # pylint: disable=unidiomatic-typecheck
      rows.append(row)
      continue
    if rows and valid_grid(rows):
      best = rows
    rows = []
  return best


def budget_note_b32(
    used: int,
    max_total_output: int,
    budget_soft: int,
    budget_notes: float = 0.01,
    budget_warn: float = 0.75,
    soft_wording: str = "gentle",
) -> str:
  """Exact `run_deepseek_tools.budget_note`."""
  if not budget_notes or used < budget_notes * max_total_output:
    return ""
  if budget_soft:
    note = (
        f"\n[budget: {used / 1000:.0f}k output tokens used; aim to finish within"
        f" {budget_soft / 1000:.0f}k (hard limit {max_total_output / 1000:.0f}k)]"
    )
    if budget_soft <= used < budget_warn * max_total_output:
      note += (
          " Past the target: submit your best answer now with submit_answer,"
          " unless one more experiment will very likely make every training"
          " pair pass."
          if soft_wording == "strong"
          else " Past the target: submit as soon as a rule fits every training pair."
      )
  else:
    note = f"\n[budget: {used / 1000:.0f}k of {max_total_output / 1000:.0f}k output tokens used]"
  if used >= budget_warn * max_total_output:
    note += (
        " Budget nearly spent: submit your best answer now with submit_answer"
        " (or at least set `answer`)."
    )
  return note


def context_adjusted_budgets(
    max_total_output: int,
    budget_soft: int,
    context_limit: int,
    prompt_plus_comp_tokens: int,
    used_completion_tokens: int,
) -> tuple[int, int]:
  """Fix `A9`: cap effective hard/soft budget at remaining context room before `context_limit`."""
  if not context_limit or prompt_plus_comp_tokens <= 0:
    return max_total_output, budget_soft
  cap = used_completion_tokens + max(
      0, context_limit - prompt_plus_comp_tokens - 16 - CONTEXT_MIN_ROOM
  )
  if cap >= max_total_output:
    return max_total_output, budget_soft
  eff_hard = max(1, cap)
  eff_soft = 0 if (budget_soft and budget_soft >= eff_hard) else budget_soft
  return eff_hard, eff_soft


def turns_note_b32(max_turns: int, turn_index_0based: int, threshold: int = 5) -> str:
  """Exact `run_deepseek_tools.turns_note`."""
  left = max_turns - turn_index_0based - 1
  if not threshold or left > threshold or left <= 0:
    return ""
  return (
      f"\n[{left} turn{'s' if left != 1 else ''} left: call submit_answer with"
      " your best answer now, or at least set `answer`.]"
  )


def gzip_ratio(text: str) -> float:
  raw = text.encode("utf-8")
  if not raw:
    return 1.0
  return len(raw) / max(1, len(gzip.compress(raw, compresslevel=6, mtime=0)))


# ==============================================================================
# 5. 4-PATH ZERO-TOKEN POST-ATTEMPT `train_check` & `ranked_answers` (`E017`)
# ==============================================================================

_TC_PROBE = r'''
def _arc_probe():
    import builtins as _b, contextlib, io, json, inspect, signal, types, copy, hashlib, time
    import numpy as _np
    class _T(_b.Exception):
        pass
    def _h(*a):
        raise _T()
    old = signal.signal(signal.SIGALRM, _h)
    g = _b.globals()
    inputs = json.loads(INPUTS)
    code = CODE
    out = {}
    def hsh(v):
        if _b.hasattr(v, "tolist"):
            v = v.tolist()
        v = json.loads(json.dumps(v, default=_b.int))
        return hashlib.sha1(json.dumps(v, separators=(",", ":")).encode()).hexdigest()[:16]
    def run(fn, sec):
        signal.setitimer(signal.ITIMER_REAL, sec)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return hsh(fn())
        except (_b.Exception, _T) as exc:
            return "ERR:" + _b.type(exc).__name__
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        if DO_EXPR and code is not None:
            try:
                expr, mode = _b.compile(code, "<answer>", "eval"), "eval"
            except _b.SyntaxError:
                try:
                    expr, mode = _b.compile(code, "<answer>", "exec"), "exec"
                except _b.SyntaxError:
                    expr, mode = None, None
            if expr is not None and "test_input" in expr.co_names:
                preds = []
                for grid in inputs:
                    def f(grid=grid):
                        env = {**g, "test_input": copy.deepcopy(grid)}
                        if mode == "eval":
                            return _b.eval(expr, env)
                        _b.exec(expr, env)
                        return env["answer"]
                    preds.append(run(f, 10.0))
                out["expr"] = preds
        if DO_FNS:
            t0 = time.monotonic()
            names = [n for n, v in _b.list(g.items()) if _b.isinstance(v, types.FunctionType) and not n.startswith("_")
                     and _b.getattr(v, "__module__", None) == "__main__" and not _b.getattr(v, "_arc_helper", False)]
            fns = []
            for n in names[-40:]:
                if time.monotonic() - t0 > 120:
                    break
                f = g[n]
                try:
                    inspect.signature(f).bind(None)
                except (_b.TypeError, _b.ValueError):
                    continue
                preds = []
                for grid in [inputs[-1]] + inputs[:-1]:
                    p = run(lambda grid=grid: f(copy.deepcopy(grid)), 3.0)
                    if p.startswith("ERR:") and p != "ERR:_T":
                        p2 = run(lambda grid=grid: f(_np.array(grid)), 3.0)
                        if not p2.startswith("ERR:"):
                            p = p2
                    preds.append(p)
                    if _b.len(preds) == 1 and p != PH:
                        break
                if preds and preds[0] == PH:
                    fns.append({"name": n, "preds": preds[1:] + preds[:1]})
            out["fns"] = fns
        vs = []
        for n, v in _b.list(g.items()):
            if n.startswith("_") or _b.isinstance(v, (types.FunctionType, types.ModuleType, _b.type)):
                continue
            if _b.isinstance(v, (_b.list, _np.ndarray)):
                try:
                    if hsh(v) == PH:
                        vs.append(n)
                except _b.Exception:
                    pass
        out["vars"] = vs
    finally:
        signal.signal(signal.SIGALRM, old)
    _b.print(MARKER + json.dumps(out))
try:
    _arc_probe()
finally:
    del _arc_probe
'''

_TC_CELL = r'''
def _arc_cell():
    import builtins as _b, contextlib, io, json, signal, copy, hashlib
    class _T(_b.Exception):
        pass
    def _h(*a):
        raise _T()
    old = signal.signal(signal.SIGALRM, _h)
    g = _b.globals()
    inputs = json.loads(INPUTS)
    src, var = SRC, VAR
    code = _b.compile(src, "<cell>", "exec")
    preds = []
    try:
        for grid in [inputs[-1]] + inputs[:-1]:
            env = {**g, "test_input": copy.deepcopy(grid)}
            env.pop(var, None)
            signal.setitimer(signal.ITIMER_REAL, 20.0)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _b.exec(code, env)
                    v = env[var]
                    if _b.hasattr(v, "tolist"):
                        v = v.tolist()
                    v = json.loads(json.dumps(v, default=_b.int))
                    preds.append(hashlib.sha1(json.dumps(v, separators=(",", ":")).encode()).hexdigest()[:16])
            except (_b.Exception, _T) as exc:
                preds.append("ERR:" + _b.type(exc).__name__)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if _b.len(preds) == 1 and preds[0] != PH:
                break
    finally:
        signal.signal(signal.SIGALRM, old)
    _b.print(MARKER + json.dumps(preds[1:] + preds[:1]))
try:
    _arc_cell()
finally:
    del _arc_cell
'''


def grid_hash(grid: Any) -> str:
  return hashlib.sha1(
      json.dumps(grid, separators=(",", ":")).encode()
  ).hexdigest()[:16]


def _fill_template(template: str, **values: str) -> str:
  for k, v in values.items():
    template = template.replace(k, v)
  return template


def _run_marked_tc(
    session: LocalExecutor, cell: str, marker: str, timeout: float
) -> Any:
  old = session.timeout, session.max_output
  session.timeout, session.max_output = timeout, 5_000_000
  try:
    result = session.execute(cell)
  finally:
    session.timeout, session.max_output = old
  for line in reversed(result.get("stdout", "").splitlines()):
    if line.startswith(marker):
      return json.loads(line[len(marker) :])
  raise RuntimeError(
      "train-check cell: "
      + result["process_status"]
      + " "
      + (result.get("stderr") or result.get("stdout") or "")[-300:]
  )


def _stored_and_loaded_names(src: str) -> tuple[set[str], set[str]]:
  try:
    tree = ast.parse(src)
  except SyntaxError:
    return set(), set()
  stores = {
      n.id
      for n in ast.walk(tree)
      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
  }
  loads = {
      n.id
      for n in ast.walk(tree)
      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
  }
  return stores, loads


def train_check(
    session: LocalExecutor,
    executed: list[tuple[str, str]],
    pred: Grid,
    code: str | None,
    replay_session: Callable[[Grid], LocalExecutor] | None = None,
    transcript_limit: float = 300.0,
) -> dict[str, Any]:
  """Exact 4-path post-attempt `train_check` (`expr` -> `cell` -> `fn` -> `transcript`)."""
  t0 = time.monotonic()
  outs = [grid_hash(p["output"]) for p in session.train]
  n_train = len(outs)
  ph = grid_hash(pred)
  paths: list[tuple[int, int, str]] = []
  inputs_repr = repr(
      json.dumps([p["input"] for p in session.train] + [session.test_input])
  )

  def score_preds(preds: list[str]) -> int:
    return sum(a == b for a, b in zip(preds[:-1], outs))

  def done() -> bool:
    return any(k == n_train for k, _, _ in paths)

  marker = f"ARC_TC_{uuid.uuid4().hex}:"
  data = _run_marked_tc(
      session,
      _fill_template(
          _TC_PROBE,
          INPUTS=inputs_repr,
          MARKER=repr(marker),
          DO_EXPR="True",
          DO_FNS="False",
          PH=repr(ph),
          CODE=repr(code),
      ),
      marker,
      timeout=300.0,
  )
  if data.get("expr") is not None and data["expr"][-1] == ph:
    paths.append((score_preds(data["expr"]), 3, "expr"))

  if not done():
    cells = [c for name, c in executed if name == "run_python"]
    for var in sorted(data.get("vars", []), key=lambda v: v != "answer")[:3]:
      for src in reversed(cells):
        stores, loads = _stored_and_loaded_names(src)
        if var in stores:
          if "test_input" in loads:
            c_marker = f"ARC_TC_{uuid.uuid4().hex}:"
            res = _run_marked_tc(
                session,
                _fill_template(
                    _TC_CELL,
                    INPUTS=inputs_repr,
                    MARKER=repr(c_marker),
                    VAR=repr(var),
                    PH=repr(ph),
                    SRC=repr(src),
                ),
                c_marker,
                timeout=20.0 * (n_train + 1) + 30.0,
            )
            if res and res[-1] == ph:
              paths.append((score_preds(res), 2, "cell"))
          break
      if any(p[2] == "cell" for p in paths):
        break

  if not done():
    f_marker = f"ARC_TC_{uuid.uuid4().hex}:"
    fns = _run_marked_tc(
        session,
        _fill_template(
            _TC_PROBE,
            INPUTS=inputs_repr,
            MARKER=repr(f_marker),
            DO_EXPR="False",
            DO_FNS="True",
            PH=repr(ph),
            CODE="None",
        ),
        f_marker,
        timeout=300.0,
    ).get("fns", [])
    if fns:
      paths.append((max(score_preds(f["preds"]) for f in fns), 1, "fn"))

  label: dict[str, Any] = {}
  if not paths and replay_session is not None:
    read_expr = (
        code
        if code not in (None, "answer")
        else (
            "answer"
            if "answer" in data.get("vars", [])
            else (data.get("vars") or [None])[0]
        )
    )
    if read_expr and len(executed) <= 120:
      preds = []
      for pair in session.train:
        if time.monotonic() - t0 > transcript_limit:
          break
        with replay_session(pair["input"]) as s:
          for name, c in executed:
            if time.monotonic() - t0 > transcript_limit:
              break
            if name == "run_python":
              s.execute(c)
            elif name == "submit_answer":
              submit_answer(s, c)
          _, grid = submit_answer(s, read_expr)
          preds.append(grid_hash(grid) if grid is not None else "ERR")
      if len(preds) == n_train:
        paths.append((score_preds(preds + [ph]), 0, "transcript"))
      else:
        label["train_check_note"] = "transcript replay over time limit"

  if paths:
    k, _, best_name = max(paths)
    label.update(
        train_pass=(k == n_train),
        train_frac=round(k / n_train, 4) if n_train else 1.0,
        train_trace=best_name,
        train_paths=[p[2] for p in paths],
    )
  else:
    label.update(
        train_pass=None,
        train_frac=None,
        train_trace="untraceable",
        train_paths=[],
    )
  label["train_check_seconds"] = round(time.monotonic() - t0, 1)
  return label


def ranked_answers(
    recs: list[dict[str, Any]], k: int = 2
) -> list[dict[str, Any]]:
  """Exact `train_check.ranked_answers`: `train_pass=True` first, then `attempt` order, top-k distinct grids."""
  order = sorted(
      (r for r in recs if valid_grid(r.get("pred"))),
      key=lambda r: (r.get("train_pass") is not True, r["attempt"]),
  )
  out: list[dict[str, Any]] = []
  for r in order:
    if r["pred"] not in [o["pred"] for o in out]:
      out.append(r)
    if len(out) == k:
      break
  return out


def majority_vote_ranked_answers(
    recs: list[dict[str, Any]], k: int = 2
) -> list[Grid]:
  """Diagnostic N>2 selector: group distinct valid grids by `(any_train_pass, vote_count, -first_attempt)`."""
  valid_recs = [r for r in recs if valid_grid(r.get("pred"))]
  by_hash: dict[str, dict[str, Any]] = {}
  for r in valid_recs:
    h = grid_hash(r["pred"])
    entry = by_hash.setdefault(
        h,
        {
            "pred": r["pred"],
            "any_tp": False,
            "tp_votes": 0,
            "total_votes": 0,
            "first_attempt": r["attempt"],
        },
    )
    is_tp = r.get("train_pass") is True
    entry["any_tp"] = entry["any_tp"] or is_tp
    entry["tp_votes"] += int(is_tp)
    entry["total_votes"] += 1
    entry["first_attempt"] = min(entry["first_attempt"], r["attempt"])
  ranked = sorted(
      by_hash.values(),
      key=lambda e: (
          not e["any_tp"],
          -e["tp_votes"],
          -e["total_votes"],
          e["first_attempt"],
      ),
  )
  return [e["pred"] for e in ranked[:k]]


# ==============================================================================
# 6. CONTINUOUS TPU vLLM MULTI-TEST RUNNER (`run_havu_b32_shard`)
# ==============================================================================


@dataclasses.dataclass
class EvalTaskSpec:
  """One ARC-AGI-2 evaluation task (with 1 or more test pairs)."""

  task_idx: int
  task_id: str
  train_pairs: list[dict[str, Any]]
  test_pairs: list[dict[str, Any]]
  global_pair_indices: list[int]
  prompt_size_bytes: int


@dataclasses.dataclass
class HavuAttemptState:
  """Live state of one `(task_id, attempt_idx)` Candidate B32 rollout."""

  task_spec: EvalTaskSpec
  attempt_idx: int
  seed: int
  max_turns: int
  max_tokens_per_turn: int
  max_total_output: int
  budget_soft: int
  system_block: str
  messages: list[dict[str, Any]]
  session: LocalExecutor
  all_inputs: list[Grid]
  multi: int | None
  prelude_fn: Callable[[list[Grid]], str]

  # Token-exact state (`--token-exact`)
  prompt_token_ids: list[int] | None = None
  resync: bool = False
  tok_stats: dict[str, int] = dataclasses.field(
      default_factory=lambda: {
          "turns": 0,
          "resyncs": 0,
          "parse_fallbacks": 0,
          "eos_continues": 0,
      }
  )
  # E058 text+image prompts: PNG bytes of the first user message's images (prompt order) and
  # the per-position image-span types of `prompt_token_ids` (`dsv41_image_tokens`; positions past
  # the end of the list are text). Empty / None for the text harness.
  image_blobs: list[bytes] = dataclasses.field(default_factory=list)
  token_types: list[int] | None = None

  # Turn & tool tracking
  n_turns: int = 0
  n_tools: int = 0
  n_errors: int = 0
  n_rescues: int = 0
  n_nudge_rescues: int = 0
  eos_continues_this_turn: int = 0
  partial_turn_text: str = ""
  partial_turn_token_ids: list[int] = dataclasses.field(default_factory=list)
  last_prompt_plus_comp: int = 0
  cum_prompt_tokens: int = 0
  cum_completion_tokens: int = 0
  cut_parse: bool = False
  pred: Any = None
  answer_source: str | None = None
  submit_code: str | None = None
  finish_reason: str | None = None
  stop_reason: str | None = None
  error: str | None = None
  executed: list[tuple[str, str]] = dataclasses.field(default_factory=list)
  events: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  turn_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  repairs: dict[str, int] = dataclasses.field(
      default_factory=lambda: {"proposed": 0, "applied": 0, "retry_ok": 0}
  )
  cuts: dict[str, int] = dataclasses.field(default_factory=dict)
  labels: dict[str, Any] = dataclasses.field(default_factory=dict)
  t0_mono: float = dataclasses.field(default_factory=time.monotonic)
  wall_s: float = 0.0
  finished: bool = False

  # Runtime thinking loop guard (`GuardedStream` inside `engine.step()`)
  loop_still_thinking: bool = True
  loop_checked_chars: int = 0
  loop_next_check: int = LOOP_WINDOW
  loop_hits: int = 0
  active_internal_rid: str = ""
  active_external_rid: str = ""


def load_eval_tasks(
    dataset_dir: str,
    shard_index: int,
    num_shards: int,
    max_tasks: int = 0,
) -> tuple[list[EvalTaskSpec], int]:
  """Load ARC-AGI-2 tasks and shard deterministically by task so multi-test pairs stay together."""
  resolved_dir = dataset_dir
  if not gfile.Exists(resolved_dir) and "/arc_agi_2/" in resolved_dir:
    cand_dir = resolved_dir.replace("/arc_agi_2/", "/ARC-AGI-2/")
    if gfile.Exists(cand_dir):
      resolved_dir = cand_dir
  filenames = (
      sorted(f for f in gfile.ListDir(resolved_dir) if f.endswith(".json"))
      if gfile.Exists(resolved_dir)
      else []
  )
  if not filenames and "/arc_agi_2/" in dataset_dir:
    cand_dir = dataset_dir.replace("/arc_agi_2/", "/ARC-AGI-2/")
    if gfile.Exists(cand_dir):
      resolved_dir = cand_dir
      filenames = sorted(
          f for f in gfile.ListDir(resolved_dir) if f.endswith(".json")
      )
  dataset_dir = resolved_dir
  if max_tasks > 0:
    filenames = filenames[:max_tasks]

  all_tasks: list[EvalTaskSpec] = []
  global_pair_idx = 0
  for task_idx, fname in enumerate(filenames):
    task_id = fname[:-5]
    fpath = os.path.join(dataset_dir, fname)
    with gfile.Open(fpath, "r") as f:
      task_data = json.load(f)
    train_pairs = task_data.get("train", [])
    test_pairs = task_data.get("test", [])
    pair_indices = list(range(global_pair_idx, global_pair_idx + len(test_pairs)))
    global_pair_idx += len(test_pairs)
    p_size = len(json.dumps(train_pairs)) + max(
        (len(json.dumps(tp["input"])) for tp in test_pairs), default=0
    )
    all_tasks.append(
        EvalTaskSpec(
            task_idx=task_idx,
            task_id=task_id,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            global_pair_indices=pair_indices,
            prompt_size_bytes=p_size,
        )
    )
  shard_tasks = [
      t for t in all_tasks if (t.task_idx % max(1, num_shards)) == shard_index
  ]
  return shard_tasks, sum(len(t.test_pairs) for t in shard_tasks)


def _grids_equal(a: Grid | None, b: Grid | None) -> bool:
  if a is None or b is None or len(a) != len(b):
    return False
  for ra, rb in zip(a, b):
    if len(ra) != len(rb) or any(int(x) != int(y) for x, y in zip(ra, rb)):
      return False
  return True


def _safe_cns_write_json(path: str, payload: Any, indent: int | None = None) -> None:
  tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
  with gfile.Open(tmp_path, "w") as f:
    json.dump(payload, f, indent=indent)
  gfile.Rename(tmp_path, path, overwrite=True)


def _load_existing_pair_records(
    pairs_path: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
  """Load already-finalized pair records on shard resume."""
  by_pair_idx: dict[int, dict[str, Any]] = {}
  submission_dict: dict[str, list[dict[str, Any]]] = {}
  if not gfile.Exists(pairs_path):
    return by_pair_idx, submission_dict
  with gfile.Open(pairs_path, "r") as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
      except json.JSONDecodeError:
        continue
      gp_idx = int(rec["global_pair_idx"])
      by_pair_idx[gp_idx] = rec
  # Rebuild submission_dict sorted by test_idx per task_id
  by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {}
  for rec in by_pair_idx.values():
    by_task.setdefault(rec["task_id"], []).append((int(rec["test_idx"]), rec))
  for t_id, items in by_task.items():
    items.sort(key=lambda x: x[0])
    submission_dict[t_id] = [
        {"attempt_1": r["attempt_1"], "attempt_2": r["attempt_2"]}
        for _, r in items
    ]
  return by_pair_idx, submission_dict


def _create_attempt_state(
    task_spec: EvalTaskSpec,
    attempt_idx: int,
    cfg: prompts_and_selector.Pilot2HypothesisConfig,
    tools_spec: list[dict[str, Any]],
    system_block: str,
    start_session: bool = True,
) -> HavuAttemptState:
  """Create a fresh Candidate B32 attempt state and (by default) spawn its isolated `LocalExecutor`."""
  all_inputs = [tp["input"] for tp in task_spec.test_pairs]
  multi = len(all_inputs) if len(all_inputs) > 1 else None
  scale = 1.0 + 0.5 * (multi - 1) if multi else 1.0
  max_total_output = int(cfg.total_token_budget * scale)
  budget_soft = int(cfg.soft_token_target * scale)
  max_turns = int(round(cfg.max_turns * scale))
  max_tokens_per_turn = int(cfg.turn_think_cap)

  if multi:
    user_prompt = (
        build_prompt_multi(task_spec.train_pairs, all_inputs)
        + LEAN2SUB_SUFFIX
        + MULTI_NOTE.format(n=multi)
    )
  else:
    user_prompt = (
        build_prompt_single(task_spec.train_pairs, all_inputs[0])
        + LEAN2SUB_SUFFIX
    )
  del tools_spec

  # E058 `--grid-format text+image --image-cell 0`: same JSON prompt + IMAGE_NOTE, then one
  # labelled image per grid (train inputs/outputs, then test inputs) as OpenAI content parts.
  grid_format = getattr(cfg, "grid_format", "json")
  if grid_format == "text+image":
    user_content: Any = grid_image.text_image_content(
        user_prompt,
        task_spec.train_pairs,
        all_inputs,
        cell=int(getattr(cfg, "image_cell", 0)),
        note=grid_image.IMAGE_NOTES[getattr(cfg, "image_note", "")],
    )
    image_blobs = grid_image.content_images(user_content)
  elif grid_format == "json":
    user_content = user_prompt
    image_blobs = []
  else:
    raise ValueError(f"Unsupported grid_format: {grid_format!r}")

  def prelude_fn(inps: list[Grid]) -> str:
    return inputs_prelude(inps)

  session = LocalExecutor(
      train=task_spec.train_pairs,
      test_input=all_inputs[0],
      timeout=30.0,
      max_output=16000,
      memory_mb=4096,
      prelude_extra=prelude_fn(all_inputs),
      style="ipython",
      cpu_seconds=7200,
  )
  if start_session:
    session.initialize()
  else:
    session.closed = True  # placeholder: never spawned, never usable
  seed = (
      cfg.base_seed
      + attempt_idx * cfg.attempt_seed_stride
      + task_spec.task_idx
  )
  return HavuAttemptState(
      task_spec=task_spec,
      attempt_idx=attempt_idx,
      seed=seed,
      max_turns=max_turns,
      max_tokens_per_turn=max_tokens_per_turn,
      max_total_output=max_total_output,
      budget_soft=budget_soft,
      system_block=system_block,
      messages=[{"role": "user", "content": user_content}],
      session=session,
      all_inputs=all_inputs,
      multi=multi,
      prelude_fn=prelude_fn,
      image_blobs=image_blobs,
  )


def _create_attempt_state_without_session(
    task_spec: EvalTaskSpec,
    attempt_idx: int,
    cfg: prompts_and_selector.Pilot2HypothesisConfig,
    system_block: str,
) -> HavuAttemptState:
  """Attempt state with a closed placeholder session (used to record a failed sandbox spawn)."""
  return _create_attempt_state(
      task_spec=task_spec,
      attempt_idx=attempt_idx,
      cfg=cfg,
      tools_spec=[],
      system_block=system_block,
      start_session=False,
  )


def _finalize_attempt_session_and_train_check(
    st: HavuAttemptState,
    stop_reason: str,
) -> None:
  """Run Candidate B32 session `answer` / `solve()` fallbacks (`A1-A11`) and 4-path `train_check`, then close session."""
  st.stop_reason = stop_reason
  try:
    if (st.pred is None or st.cut_parse) and st.stop_reason in {
        "output_budget",
        "max_turns",
        "length",
        "final",
        "context",
        "error",
    }:
      try:
        restore_inputs(st.session, st.all_inputs, with_inputs=True)
        if st.multi:
          out_txt, grids = answer_items(st.session, "answer", st.multi)
          st.events.append({
              "fallback": "session_answer",
              "accepted": grids is not None,
              "output": out_txt[:300],
          })
          grids_list: list[Grid | None] = grids or [None] * st.multi
          codes: list[str | None] = [
              "answer" if g is not None else None for g in grids_list
          ]
          missing = [i for i, g in enumerate(grids_list) if g is None]
          if missing:
            s_out, got = solve_items(
                st.session, [st.all_inputs[i] for i in missing]
            )
            st.events.append({
                "fallback": "session_solve",
                "accepted": got is not None,
                "output": s_out[:300],
            })
            for i, g in zip(missing, got or []):
              if g is not None:
                grids_list[i] = g
                codes[i] = "[solve(t) for t in test_inputs]"
          if any(g is not None for g in grids_list):
            st.pred = grids_list
            st.submit_code = codes[0]
            st.answer_source = (
                "session" if "answer" in codes else "session_solve"
            )
            st.cut_parse = False
        else:
          out_txt, grid = submit_answer(st.session, "answer")
          st.events.append({
              "fallback": "session_answer",
              "accepted": grid is not None,
              "output": out_txt[:300],
          })
          if grid is not None:
            st.pred = grid
            st.answer_source = "session"
            st.submit_code = "answer"
            st.cut_parse = False
          else:
            expr = "solve(test_input)"
            s_out, grid = submit_answer(st.session, expr)
            if grid is None:
              more, got = solve_items(st.session, st.all_inputs)
              grid = (got or [None])[0]
              s_out = s_out[:150] + " " + more
            st.events.append({
                "fallback": "session_solve",
                "accepted": grid is not None,
                "output": s_out[:300],
            })
            if grid is not None:
              st.pred = grid
              st.answer_source = "session_solve"
              st.submit_code = expr
              st.cut_parse = False
      except Exception as exc:  # pylint: disable=broad-except
        st.events.append({
            "fallback": "failed",
            "output": f"{type(exc).__name__}: {exc}"[:300],
        })

    if st.pred is not None:
      try:
        if st.multi and st.pred[0] is None:
          st.labels = {
              "train_pass": None,
              "train_frac": None,
              "train_trace": "no_answer_0",
          }
        else:
          target_pred = st.pred[0] if st.multi else st.pred
          tc_code = (
              multi_check_code(st.submit_code, st.all_inputs)
              if st.multi
              else st.submit_code
          )
          st.labels = train_check(
              session=st.session,
              executed=st.executed,
              pred=target_pred,
              code=tc_code,
              replay_session=lambda grid: LocalExecutor(
                  train=st.task_spec.train_pairs,
                  test_input=grid,
                  timeout=30.0,
                  max_output=16000,
                  memory_mb=4096,
                  prelude_extra=st.prelude_fn([grid] + st.all_inputs[1:]),
                  style="ipython",
                  cpu_seconds=7200,
              ),
          )
      except Exception as exc:  # pylint: disable=broad-except
        st.labels = {
            "train_pass": None,
            "train_frac": None,
            "train_trace": "check_error",
            "train_check_error": f"{type(exc).__name__}: {exc}"[:300],
        }
    else:
      st.labels = {
          "train_pass": None,
          "train_frac": None,
          "train_trace": "no_answer",
      }
  finally:
    st.session.close()
    st.wall_s = round(time.monotonic() - st.t0_mono, 1)
    st.finished = True


def _process_completed_turn(
    st: HavuAttemptState,
    raw_turn_text: str,
    turn_token_ids: list[int],
    finish_reason: str,
    context_limit: int,
    encode_fn: Callable[[str], list[int]],
) -> str | None:
  """Process one finished LLM turn in a worker thread.

  Returns `None` if the attempt should enqueue another LLM turn, or a `stop_reason`
  string (with `_finalize_attempt_session_and_train_check` already executed!) when done.
  """
  turn_index_0based = st.n_turns - 1
  is_dsv41 = st.system_block.startswith(DS41_BOS_TOKEN)
  if is_dsv41:
    reasoning, content, calls, clean = parse_dsv41_completion(raw_turn_text)
  else:
    reasoning, content, calls, clean = parse_qwen_completion(raw_turn_text)
  st.tok_stats["parse_fallbacks"] += int(not clean and finish_reason != "length")
  st.resync = (not clean) or (finish_reason == "length")
  st.tok_stats["turns"] += 1

  msg: dict[str, Any] = {
      "role": "assistant",
      "content": content,
      "reasoning_content": reasoning,
  }
  if calls:
    msg["tool_calls"] = calls
    eff_finish = "tool_calls" if finish_reason == "stop" else finish_reason
  else:
    eff_finish = finish_reason
  st.finish_reason = eff_finish
  st.messages.append(msg)
  st.events.append({
      "turn": st.n_turns,
      "completion_tokens": len(turn_token_ids),
      "cum_completion_tokens": st.cum_completion_tokens,
      "finish_reason": eff_finish,
  })

  # `--rescue-code`: if stopped with no tool call, try extracting code or nudging up to 3x
  if not calls and eff_finish == "stop":
    rescued = rescue_tool_call(content, st.n_turns)
    if rescued is not None:
      msg = {**msg, "content": "", "tool_calls": [rescued]}
      st.messages[-1] = msg
      calls = [rescued]
      st.n_rescues += 1
      st.resync = True
      st.events.append({"turn": st.n_turns, "rescued": "code"})
    elif st.n_nudge_rescues < 3 and not valid_grid(parse_arcprize_grid(content)):
      st.messages.append({"role": "user", "content": RESCUE_NUDGE})
      st.n_rescues += 1
      st.n_nudge_rescues += 1
      st.resync = True
      st.events.append({"turn": st.n_turns, "rescued": "nudge"})
      st.turn_records.append({
          "turn_idx": turn_index_0based,
          "gen_tokens_this_turn": len(turn_token_ids),
          "cum_gen_tokens": st.cum_completion_tokens,
          "assistant_output": raw_turn_text,
          "parsed_tool_name": "rescue_nudge",
          "tool_response": RESCUE_NUDGE,
      })
      if st.n_turns >= st.max_turns or st.cum_completion_tokens >= st.max_total_output:
        stop_r = (
            "max_turns"
            if st.n_turns >= st.max_turns
            else "output_budget"
        )
        _finalize_attempt_session_and_train_check(st, stop_r)
        return stop_r
      return None

  # No tool calls: check for direct text grid answer or length-cut turn
  if not calls:
    cand = parse_arcprize_grid(content)
    parsed_grid = cand if valid_grid(cand) else None
    src = "parse" if parsed_grid is not None else None
    if st.multi:
      parsed_grid, src = None, None
    if parsed_grid is None and not st.multi:
      parsed_grid = lenient_parse(content)
      src = "lenient" if parsed_grid is not None else None
    st.pred = parsed_grid
    st.answer_source = src
    st.turn_records.append({
        "turn_idx": turn_index_0based,
        "gen_tokens_this_turn": len(turn_token_ids),
        "cum_gen_tokens": st.cum_completion_tokens,
        "assistant_output": raw_turn_text,
        "parsed_tool_name": None,
        "tool_response": f"[no_tool_call source={src} finish={eff_finish}]",
    })
    if eff_finish == "length":
      st.cut_parse = parsed_grid is not None  # Fix A2
      if st.n_turns >= st.max_turns or st.cum_completion_tokens >= st.max_total_output:
        stop_r = (
            "max_turns"
            if st.n_turns >= st.max_turns
            else "output_budget"
        )
        _finalize_attempt_session_and_train_check(st, stop_r)
        return stop_r
      return None
    st.cut_parse = False
    stop_r = "final" if eff_finish == "stop" else eff_finish
    _finalize_attempt_session_and_train_check(st, stop_r)
    return stop_r

  # Execute tool call(s)
  accepted = None
  accepted_code = None
  tool_outputs_for_turn: list[str] = []
  for call in calls:
    st.n_tools += 1
    status = "completed"
    t_name = ""
    code_str = ""
    try:
      if eff_finish == "length":
        raise ValueError("Tool skipped: truncated response or final-turn budget reached")
      arguments = json.loads(call["function"]["arguments"])
      if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
        raise ValueError("Expected an object with a string code field")
      t_name = call["function"]["name"]
      code_str = arguments["code"]
      if t_name not in ("run_python", "submit_answer"):
        fixed_name = rescue_name(t_name)
        if fixed_name != t_name:
          st.n_rescues += 1
          st.events.append({"turn": st.n_turns, "rescued": f"name {t_name}"})
          t_name = fixed_name
          call["function"]["name"] = fixed_name
          st.resync = True
      if accepted is not None:
        output = "[skipped: solution already accepted]"
      elif t_name == "run_python":
        st.executed.append((t_name, code_str))
        res = st.session.execute(code_str)
        status = res["process_status"]
        output = tool_output(res, 16000)
        if status == "error":
          fixed = auto_repair_cell(
              session=st.session,
              code=code_str,
              output=output,
              names=PRELOADED | {"test_inputs"},
              limit=16000,
              stats=st.repairs,
          )
          if fixed is not None:
            output, status, repaired_code = fixed
            st.executed[-1] = (t_name, repaired_code)
            code_str = repaired_code
            st.events.append({"turn": st.n_turns, "auto_repair": status})
      elif t_name == "submit_answer":
        st.executed.append((t_name, code_str))
        output, accepted = (
            submit_answer(st.session, code_str, st.multi)
            if st.multi
            else submit_answer(st.session, code_str)
        )
        status = "completed" if accepted is not None else "rejected"
        if accepted is not None:
          accepted_code = code_str
      else:
        raise ValueError(f"Unknown tool: {t_name}")
    except (ValueError, TypeError, KeyError) as exc:
      output = f"[tool error: {exc}]"
      status = "error"
    st.n_errors += int(status != "completed")
    st.messages.append({
        "role": "tool",
        "tool_call_id": call["id"],
        "content": output,
    })
    tool_outputs_for_turn.append(output)
    st.events.append({
        "turn": st.n_turns,
        "tool_call_id": call["id"],
        "status": status,
    })
    st.turn_records.append({
        "turn_idx": turn_index_0based,
        "gen_tokens_this_turn": len(turn_token_ids),
        "cum_gen_tokens": st.cum_completion_tokens,
        "assistant_output": raw_turn_text,
        "parsed_tool_name": t_name,
        "code": code_str,
        "status": status,
        "tool_response": output,
    })

  if accepted is not None:
    st.pred = accepted
    st.answer_source = "submit"
    st.submit_code = accepted_code
    st.cut_parse = False
    _finalize_attempt_session_and_train_check(st, "submitted")
    return "submitted"

  # Budget + turns-left note appended to last tool response (including Fix A9 context_budget)
  eff_hard, eff_soft = context_adjusted_budgets(
      max_total_output=st.max_total_output,
      budget_soft=st.budget_soft,
      context_limit=context_limit,
      prompt_plus_comp_tokens=st.last_prompt_plus_comp,
      used_completion_tokens=st.cum_completion_tokens,
  )
  note = budget_note_b32(
      used=st.cum_completion_tokens,
      max_total_output=eff_hard,
      budget_soft=eff_soft,
      budget_notes=0.01,
      budget_warn=0.75,
      soft_wording="gentle",
  ) + turns_note_b32(st.max_turns, turn_index_0based, threshold=5)
  if note and st.messages[-1]["role"] == "tool":
    st.messages[-1]["content"] += note
    tool_outputs_for_turn[-1] += note

  if st.n_turns >= st.max_turns or st.cum_completion_tokens >= st.max_total_output:
    stop_r = (
        "max_turns"
        if st.n_turns >= st.max_turns
        else "output_budget"
    )
    _finalize_attempt_session_and_train_check(st, stop_r)
    return stop_r

  # Update `--token-exact` prompt_token_ids without re-tokenizing the assistant turn!
  if not st.resync and st.prompt_token_ids is not None:
    if is_dsv41:
      suffix_str = render_tool_responses_dsv41(tool_outputs_for_turn) + DS41_GEN_PROMPT
    else:
      suffix_str = "\n" + render_tool_responses_qwen(tool_outputs_for_turn) + GEN_PROMPT
    st.prompt_token_ids = st.prompt_token_ids + turn_token_ids + encode_fn(suffix_str)
  else:
    st.prompt_token_ids = None
  return None


def _pick_next_queued_job(
    queue: list[tuple[EvalTaskSpec, int]],
    finished_by_task: dict[str, list[HavuAttemptState]],
) -> int:
  """Exact `solver_dsv41.pick(queue)`: all attempt-0s first (smallest tasks first); then attempt>=1 prioritizing tasks without `train_pass=True`."""
  best_idx = 0
  best_key: tuple[int, int] | None = None
  for idx, (t_spec, att_idx) in enumerate(queue):
    if att_idx == 0:
      return idx
    done_atts = finished_by_task.get(t_spec.task_id, [])
    if not done_atts:
      prio = 1  # earlier attempt still running
    elif any(s.labels.get("train_pass") is True for s in done_atts):
      prio = 2  # already has a train-passing prediction -> lowest priority
    else:
      prio = 0  # finished earlier attempt(s) without train_pass -> highest priority!
    cand_key = (att_idx, prio)
    if best_key is None or cand_key < best_key:
      best_idx, best_key = idx, cand_key
      if att_idx == 1 and prio == 0:
        break
  return best_idx


def run_havu_b32_shard(
    llm: Any,
    cfg: prompts_and_selector.Pilot2HypothesisConfig,
    dataset_dir: str,
    out_dir: str,
    shard_index: int,
    num_shards: int,
    max_tasks: int = 0,
    model_path: str = "",
) -> list[dict[str, Any]]:
  """Run Havu's Candidate B32 harness on a shard of ARC-AGI-2 tasks and flush all CNS artifacts."""
  import vllm  # pylint: disable=g-import-not-at-top
  if not hasattr(vllm.__dict__, "get") or "sampling_params" not in vllm.__dict__:
    try:
      _ = vllm.sampling_params
    except AttributeError:
      import types as _types

      @dataclasses.dataclass
      class _FallbackSamplingParams:
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        max_tokens: int = 64000
        stop: list[str] | None = None
        include_stop_str_in_output: bool = True
        skip_special_tokens: bool = False
        spaces_between_special_tokens: bool = False
        output_kind: str = "CUMULATIVE"
        stream_interval: int = 250

      _sp_mod = _types.ModuleType("vllm.sampling_params")
      _sp_mod.RequestOutputKind = type("RequestOutputKind", (), {"CUMULATIVE": "CUMULATIVE"})  # type: ignore[attr-defined]
      _sp_mod.SamplingParams = _FallbackSamplingParams  # type: ignore[attr-defined]
      sys.modules["vllm.sampling_params"] = _sp_mod
      vllm.__dict__["sampling_params"] = _sp_mod
      vllm.__dict__["SamplingParams"] = _FallbackSamplingParams

  fb = prompts_and_selector.FROZEN_BASELINE_CONFIG
  # E058 prompt format (text harness: json / 0 / ""), written to every per-shard artifact.
  prompt_format = {
      "grid_format": getattr(cfg, "grid_format", "json"),
      "image_cell": int(getattr(cfg, "image_cell", 0)),
      "image_note": getattr(cfg, "image_note", ""),
  }
  gfile.MakeDirs(out_dir)
  shard_tag = f"shard_{shard_index:02d}_of_{num_shards:02d}"
  config_path = os.path.join(out_dir, f"config_{shard_tag}.json")
  transcripts_path = os.path.join(out_dir, f"transcripts_{shard_tag}.jsonl")
  code_traces_path = os.path.join(out_dir, f"code_traces_{shard_tag}.jsonl")
  attempts_path = os.path.join(out_dir, f"attempts_{shard_tag}.jsonl")
  pairs_path = os.path.join(out_dir, f"pairs_{shard_tag}.jsonl")
  submission_path = os.path.join(out_dir, f"submission_{shard_tag}.json")
  summary_path = os.path.join(out_dir, f"summary_{shard_tag}.json")

  shard_tasks, total_pairs_in_shard = load_eval_tasks(
      dataset_dir=dataset_dir,
      shard_index=shard_index,
      num_shards=num_shards,
      max_tasks=max_tasks,
  )

  _safe_cns_write_json(
      config_path,
      {
          "harness": "havu_candidate_b32",
          "hypothesis_config": dataclasses.asdict(cfg),
          "prompt_format": prompt_format,
          "b32_params": {
              "multi_test": True,
              "multi_budget": 0.5,
              "auto_repair": True,
              "token_exact": True,
              "rescue_code": True,
              "lenient_parse": True,
              "final_mode": "session",
              "fallback_solve": True,
              "train_check": True,
              "verify_submit": False,
              "cap_nudge": False,
              "transcript_compaction": False,
              "max_tokens_per_turn": cfg.turn_think_cap,
              "max_total_output": cfg.total_token_budget,
              "budget_soft": cfg.soft_token_target,
              "soft_wording": "gentle",
              "budget_notes": 0.01,
              "budget_warn": 0.75,
              "turns_note": 5,
              "tool_timeout_s": 30.0,
              "max_tool_output_chars": 16000,
              "loop_window": LOOP_WINDOW,
              "loop_ratio": LOOP_RATIO,
              "loop_confirm": LOOP_CONFIRM,
              **prompt_format,
          },
          "shard_index": shard_index,
          "num_shards": num_shards,
          "model_path": model_path,
          "dataset_dir": dataset_dir,
      },
      indent=2,
  )

  existing_pairs, submission_dict = _load_existing_pair_records(pairs_path)
  todo_tasks = [
      t
      for t in shard_tasks
      if not all(gp in existing_pairs for gp in t.global_pair_indices)
  ]
  # Order tasks smallest-prompt-first (`solver_dsv41.py` line 988)
  todo_tasks_sorted = sorted(todo_tasks, key=lambda t: t.prompt_size_bytes)

  print(
      f"[havu_b32] Starting {cfg.exp_id} ({cfg.title}) | {shard_tag} | "
      f"shard_tasks={len(shard_tasks)} ({total_pairs_in_shard} pairs), "
      f"todo_tasks={len(todo_tasks_sorted)} | num_attempts={cfg.num_attempts} | "
      f"prompt_format={prompt_format}",
      flush=True,
  )
  if not todo_tasks_sorted:
    print(f"[havu_b32] All {len(shard_tasks)} tasks already completed for {shard_tag}.", flush=True)
    return list(existing_pairs.values())

  engine = llm.llm_engine
  tok = llm.get_tokenizer()
  try:
    engine_max_len = int(engine.model_config.max_model_len)
  except Exception:  # pylint: disable=broad-except
    engine_max_len = int(fb["max_model_len"])

  def encode_fn(s: str) -> list[int]:
    return list(tok.encode(s, add_special_tokens=False))

  def decode_fn(ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)

  dsv41_eos_probe = encode_fn(DS41_EOS_TOKEN)
  qwen_eos_probe = encode_fn("<|im_end|>")
  is_dsv41_model = (
      "dsv41" in model_path.lower()
      or "dsv41" in cfg.exp_id.lower()
      or (len(dsv41_eos_probe) == 1 and len(qwen_eos_probe) > 1)
  )
  if prompt_format["grid_format"] == "text+image" and not is_dsv41_model:
    raise ValueError(
        "grid_format='text+image' (E058) is only implemented for DeepSeek-V4.1 DSML prompts "
        "(`<｜deepseek_image｜>` placeholder expansion); the Qwen ChatML path has no image route."
    )
  stop_str = DS41_EOS_TOKEN if is_dsv41_model else "<|im_end|>"
  tc_open_str = DS41_TC_OPEN if is_dsv41_model else "<tool_call>"

  im_end_ids = encode_fn(stop_str)
  im_end_id = im_end_ids[0] if len(im_end_ids) == 1 else getattr(tok, "eos_token_id", None)
  eos_special_ids = set(im_end_ids)
  if getattr(tok, "eos_token_id", None) is not None:
    eos_special_ids.add(int(tok.eos_token_id))

  tools_spec = tools_spec_b32(multi_test=True, tool_timeout=30)
  if is_dsv41_model:
    system_block = render_system_dsv41(
        tools=tools_spec,
        tier_text="",
        reasoning_effort=50,
    )
  else:
    system_block = render_system_qwen(
        tools=tools_spec,
        tier_text=TIER_SYSTEM_XHIGH,
        effort_header=EFFORT_HEADER_MINIMAL,
    )

  # Build job queue: all attempt-0s first (smallest-first), then attempt-1, ...
  job_queue: list[tuple[EvalTaskSpec, int]] = [
      (t_spec, att_idx)
      for att_idx in range(cfg.num_attempts)
      for t_spec in todo_tasks_sorted
  ]

  max_concurrent = min(int(fb["max_num_seqs"]), 24)
  active_requests: dict[str, HavuAttemptState] = {}
  rid_alias: dict[str, str] = {}
  pending_turn_futures: dict[futures.Future[str | None], HavuAttemptState] = {}
  finished_by_task: dict[str, list[HavuAttemptState]] = {}
  finalized_task_ids: set[str] = {
      t.task_id
      for t in shard_tasks
      if all(gp in existing_pairs for gp in t.global_pair_indices)
  }
  finalized_pair_records: dict[int, dict[str, Any]] = dict(existing_pairs)

  worker_pool = futures.ThreadPoolExecutor(max_workers=24)
  cumulative_kind = vllm.sampling_params.RequestOutputKind.CUMULATIVE
  start_time = time.time()
  last_heartbeat_time = start_time
  total_gen_tokens_all = 0

  def _pop_active_request(rid: str) -> HavuAttemptState | None:
    st = active_requests.pop(rid, None)
    other = rid_alias.pop(rid, None)
    if other is not None:
      active_requests.pop(other, None)
      rid_alias.pop(other, None)
    return st

  def _enqueue_attempt_turn(st: HavuAttemptState, is_eos_continue: bool = False) -> None:
    if not is_eos_continue:
      remaining = st.max_total_output - (st.cum_completion_tokens + len(st.partial_turn_token_ids))
      if remaining <= 0 or st.n_turns >= st.max_turns:
        stop_r = "output_budget" if remaining <= 0 else "max_turns"
        fut = worker_pool.submit(_finalize_and_return, st, stop_r)
        pending_turn_futures[fut] = st
        return
      if st.prompt_token_ids is None or st.resync:
        if is_dsv41_model:
          full_chatml = render_full_messages_dsv41(
              st.system_block, st.messages, add_generation_prompt=True
          )
        else:
          full_chatml = render_full_messages_qwen(
              st.system_block, st.messages, add_generation_prompt=True
          )
        if st.prompt_token_ids is not None and st.resync:
          st.tok_stats["resyncs"] += 1
        st.prompt_token_ids = encode_fn(full_chatml)
        st.token_types = None
        if st.image_blobs:
          # E058: each `<｜deepseek_image｜>` placeholder -> its image span (`dsv41_image_tokens`).
          ids_np, types_np, _, _ = dsv41_image_tokens.expand_image_placeholders(
              st.prompt_token_ids, st.image_blobs
          )
          st.prompt_token_ids = ids_np.tolist()
          st.token_types = types_np.tolist()
        st.resync = False
      st.n_turns += 1
      st.eos_continues_this_turn = 0
      st.partial_turn_text = ""
      st.partial_turn_token_ids = []
      st.loop_still_thinking = True
      st.loop_checked_chars = 0
      st.loop_next_check = LOOP_WINDOW
      st.loop_hits = 0
      send_ids = st.prompt_token_ids
    else:
      send_ids = (st.prompt_token_ids or []) + st.partial_turn_token_ids

    room = engine_max_len - len(send_ids) - 16
    if room < CONTEXT_MIN_ROOM:
      st.events.append({
          "turn": st.n_turns,
          "context_full": f"prompt {len(send_ids)} tokens, limit {engine_max_len}",
      })
      if not is_eos_continue:
        st.n_turns -= 1
      fut = worker_pool.submit(_finalize_and_return, st, "context")
      pending_turn_futures[fut] = st
      return

    rem_budget = st.max_total_output - (
        st.cum_completion_tokens + len(st.partial_turn_token_ids)
    )
    cap = max(
        1,
        min(
            st.max_tokens_per_turn - len(st.partial_turn_token_ids),
            rem_budget,
            room,
        ),
    )
    # NOTE: no `seed=`: the TPU/JAX vLLM backend rejects per-request seeds
    # (`tpu_platform.validate_request` -> "JAX does not support per-request seed").
    # Sampling is temperature/top_p/top_k only, exactly like `pilot2_eval.py`.
    sp = vllm.SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_tokens=cap,
        stop=[stop_str],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    sp.output_kind = cumulative_kind
    if hasattr(sp, "stream_interval"):
      sp.stream_interval = 250
    prompt_req: dict[str, Any] = {"prompt_token_ids": send_ids}
    if st.image_blobs and st.token_types is not None:
      prompt_req["token_types"] = st.token_types
      prompt_req["images"] = st.image_blobs
    rids = llm.enqueue([prompt_req], [sp], use_tqdm=False)
    internal = str(rids[0])
    external = internal.rsplit("-", 1)[0] if "-" in internal else internal
    st.active_internal_rid = internal
    st.active_external_rid = external
    active_requests[external] = st
    if internal != external:
      active_requests[internal] = st
      rid_alias[internal] = external
      rid_alias[external] = internal

  def _finalize_and_return(st: HavuAttemptState, stop_r: str) -> str:
    _finalize_attempt_session_and_train_check(st, stop_r)
    return stop_r

  def _write_summary(status_str: str) -> None:
    elapsed = time.time() - start_time
    tok_s = total_gen_tokens_all / max(1.0, elapsed)
    done_pairs = len(finalized_pair_records)
    denom_p = max(1, done_pairs)
    p1_hits = sum(int(bool(r["pass1"])) for r in finalized_pair_records.values())
    p2_hits = sum(int(bool(r["pass2"])) for r in finalized_pair_records.values())
    or_hits = sum(int(bool(r["sel_oracle"])) for r in finalized_pair_records.values())
    ver_hits = sum(int(bool(r["any_verified"])) for r in finalized_pair_records.values())

    # Compute official task-weighted Pass@1 and Pass@2 across completed tasks
    done_tasks = len(finalized_task_ids)
    task_p1_sum = 0.0
    task_p2_sum = 0.0
    for t_spec in shard_tasks:
      if t_spec.task_id not in finalized_task_ids:
        continue
      n_tp = len(t_spec.global_pair_indices)
      task_p1_sum += sum(
          1.0 for gp in t_spec.global_pair_indices if finalized_pair_records[gp]["pass1"]
      ) / max(1, n_tp)
      task_p2_sum += sum(
          1.0 for gp in t_spec.global_pair_indices if finalized_pair_records[gp]["pass2"]
      ) / max(1, n_tp)

    active_cnt = len({id(x) for x in active_requests.values()}) + len(
        pending_turn_futures
    )
    rem_tasks = max(0, len(shard_tasks) - done_tasks)
    eta_min = (
        round((elapsed / done_tasks) * rem_tasks / 60.0, 1)
        if done_tasks > 0 and status_str == "running"
        else 0.0
    )
    _safe_cns_write_json(
        summary_path,
        {
            "exp_id": cfg.exp_id,
            "title": cfg.title,
            "harness": "havu_candidate_b32",
            "prompt_format": prompt_format,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "status": status_str,
            "completed_tasks": done_tasks,
            "total_tasks_in_shard": len(shard_tasks),
            "completed_pairs": done_pairs,
            "total_pairs_in_shard": total_pairs_in_shard,
            "active_streams": active_cnt,
            "queued_attempts": len(job_queue),
            "pass1_count": p1_hits,
            "pass1_pct": round(100.0 * p1_hits / denom_p, 2) if done_pairs else 0.0,
            "pass2_count": p2_hits,
            "pass2_pct": round(100.0 * p2_hits / denom_p, 2) if done_pairs else 0.0,
            "official_task_pass1": round(task_p1_sum, 4),
            "official_task_pass1_pct": (
                round(100.0 * task_p1_sum / max(1, done_tasks), 2)
                if done_tasks
                else 0.0
            ),
            "official_task_pass2": round(task_p2_sum, 4),
            "official_task_pass2_pct": (
                round(100.0 * task_p2_sum / max(1, done_tasks), 2)
                if done_tasks
                else 0.0
            ),
            "sel_oracle_count": or_hits,
            "sel_oracle_pct": round(100.0 * or_hits / denom_p, 2) if done_pairs else 0.0,
            "diag_oracle_count": or_hits,
            "diag_oracle_pct": round(100.0 * or_hits / denom_p, 2) if done_pairs else 0.0,
            "verified_pairs_count": ver_hits,
            "verified_pairs_pct": round(100.0 * ver_hits / denom_p, 2) if done_pairs else 0.0,
            "total_gen_tokens": total_gen_tokens_all,
            "elapsed_sec": round(elapsed, 1),
            "eta_minutes": eta_min,
            "gen_tok_per_sec": round(tok_s, 1),
        },
        indent=2,
    )

  def _record_finished_attempt(st: HavuAttemptState) -> None:
    t_spec = st.task_spec
    finished_by_task.setdefault(t_spec.task_id, []).append(st)
    # Write Havu's per-pair attempt rows (`split_multi`) + transcripts + code traces immediately
    n_pairs = len(t_spec.test_pairs)
    if st.multi:
      preds_by_pair = (
          st.pred
          if isinstance(st.pred, list) and len(st.pred) == n_pairs
          else [None] * n_pairs
      )
    else:
      preds_by_pair = [st.pred]

    with gfile.Open(attempts_path, "a") as fa:
      for p_idx in range(n_pairs):
        gt_p = t_spec.test_pairs[p_idx].get("output")
        pred_p = preds_by_pair[p_idx]
        att_row = {
            "task_id": t_spec.task_id,
            "pair": p_idx,
            "attempt": st.attempt_idx,
            "pred": pred_p,
            "correct": _grids_equal(pred_p, gt_p) if gt_p is not None else None,
            "answer_source": st.answer_source if pred_p is not None else None,
            "stop_reason": st.stop_reason,
            "finish_reason": st.finish_reason,
            "turns": st.n_turns,
            "tool_calls": st.n_tools,
            "tool_errors": st.n_errors,
            "rescues": st.n_rescues,
            "auto_repairs": st.repairs,
            "token_exact": st.tok_stats,
            "cuts": st.cuts,
            "usage": {
                "prompt": st.cum_prompt_tokens,
                "completion": st.cum_completion_tokens,
            },
            "wall_s": st.wall_s,
            "grid_format": prompt_format["grid_format"],
            "prompt_images": len(st.image_blobs),
            **st.labels,
        }
        if st.multi:
          att_row["multi_test"] = st.multi
          if p_idx > 0:
            att_row["multi_share"] = True
        fa.write(json.dumps(att_row) + "\n")

    with gfile.Open(transcripts_path, "a") as ft:
      ft.write(
          json.dumps({
              "exp_id": cfg.exp_id,
              "task_id": t_spec.task_id,
              "attempt_idx": st.attempt_idx,
              "multi_test": st.multi,
              "prompt_format": prompt_format,
              "prompt_images": len(st.image_blobs),
              "stop_reason": st.stop_reason,
              "turns": st.n_turns,
              "cum_gen_tokens": st.cum_completion_tokens,
              "answer_source": st.answer_source,
              "labels": st.labels,
              "auto_repairs": st.repairs,
              "rescues": st.n_rescues,
              "cuts": st.cuts,
              "events": st.events,
              "messages": st.messages,
          })
          + "\n"
      )
    with gfile.Open(code_traces_path, "a") as fc:
      fc.write(
          json.dumps({
              "exp_id": cfg.exp_id,
              "task_id": t_spec.task_id,
              "attempt_idx": st.attempt_idx,
              "executed": st.executed,
              "submit_code": st.submit_code,
              "turn_records": st.turn_records,
          })
          + "\n"
      )

    done_atts = finished_by_task[t_spec.task_id]
    if (
        len(done_atts) == cfg.num_attempts
        and t_spec.task_id not in finalized_task_ids
    ):
      _finalize_task(t_spec, done_atts)

  def _finalize_task(
      t_spec: EvalTaskSpec, done_atts: list[HavuAttemptState]
  ) -> None:
    finalized_task_ids.add(t_spec.task_id)
    done_atts_sorted = sorted(done_atts, key=lambda x: x.attempt_idx)
    n_pairs = len(t_spec.test_pairs)
    sub_rows: list[dict[str, Any]] = []

    with gfile.Open(pairs_path, "a") as fp:
      for p_idx, (gp_idx, test_pair) in enumerate(
          zip(t_spec.global_pair_indices, t_spec.test_pairs)
      ):
        test_in = test_pair["input"]
        gt = test_pair.get("output")
        fallback = test_in if valid_grid(test_in) else [[0]]
        pair_recs = []
        for s in done_atts_sorted:
          pred_p = (
              s.pred[p_idx]
              if (s.multi and isinstance(s.pred, list) and len(s.pred) == n_pairs)
              else (s.pred if not s.multi else None)
          )
          pair_recs.append({
              "attempt": s.attempt_idx,
              "pred": pred_p if valid_grid(pred_p) else None,
              "train_pass": s.labels.get("train_pass"),
              "train_frac": s.labels.get("train_frac"),
              "train_trace": s.labels.get("train_trace"),
              "answer_source": s.answer_source,
              "stop_reason": s.stop_reason,
              "turns": s.n_turns,
              "tokens": s.cum_completion_tokens,
              "auto_repairs": s.repairs,
              "rescues": s.n_rescues,
              "cuts": s.cuts,
          })
        top_recs = ranked_answers(pair_recs, k=2)
        a1 = top_recs[0]["pred"] if top_recs else fallback
        a2 = top_recs[1]["pred"] if len(top_recs) > 1 else fallback
        mv_top = majority_vote_ranked_answers(pair_recs, k=2)
        mv_a1 = mv_top[0] if mv_top else fallback
        mv_a2 = mv_top[1] if len(mv_top) > 1 else fallback

        hit1 = _grids_equal(a1, gt)
        hit2 = hit1 or _grids_equal(a2, gt)
        mv_hit1 = _grids_equal(mv_a1, gt)
        mv_hit2 = mv_hit1 or _grids_equal(mv_a2, gt)
        oracle = any(
            _grids_equal(r["pred"], gt) for r in pair_recs if r["pred"] is not None
        )
        any_ver = any(r["train_pass"] is True for r in pair_recs)

        sub_rows.append({"attempt_1": a1, "attempt_2": a2})
        pair_record = {
            "exp_id": cfg.exp_id,
            "global_pair_idx": gp_idx,
            "task_id": t_spec.task_id,
            "test_idx": p_idx,
            "pass1": hit1,
            "pass2": hit2,
            "mv_pass1": mv_hit1,
            "mv_pass2": mv_hit2,
            "sel_oracle": oracle,
            "any_verified": any_ver,
            "attempt_1": a1,
            "attempt_2": a2,
            "ground_truth": gt,
            "selector_diagnostics": {
                "selector": "havu_train_check_ranked_answers",
                "multi_test": len(t_spec.test_pairs) if len(t_spec.test_pairs) > 1 else None,
                "ranked_attempts": [
                    {
                        "attempt": r["attempt"],
                        "train_pass": r["train_pass"],
                        "train_trace": r["train_trace"],
                        "answer_source": r["answer_source"],
                    }
                    for r in top_recs
                ],
                "attempt_records": [
                    {k: v for k, v in r.items() if k != "pred"}
                    for r in pair_recs
                ],
            },
        }
        finalized_pair_records[gp_idx] = pair_record
        fp.write(json.dumps(pair_record) + "\n")

    submission_dict[t_spec.task_id] = sub_rows
    _safe_cns_write_json(submission_path, submission_dict, indent=2)
    _write_summary("running")
    p_tags = [
        f"p{i}:{'P1' if finalized_pair_records[gp]['pass1'] else ('P2' if finalized_pair_records[gp]['pass2'] else 'MISS')}"
        for i, gp in enumerate(t_spec.global_pair_indices)
    ]
    print(
        f"[havu_b32 {cfg.exp_id} | {shard_tag}] Task {t_spec.task_id} "
        f"({len(finalized_task_ids)}/{len(shard_tasks)}) -> {', '.join(p_tags)}",
        flush=True,
    )

  try:
    while (
        job_queue
        or engine.has_unfinished_requests()
        or active_requests
        or pending_turn_futures
    ):
      # 1. Drain completed worker futures (tool calls, auto-repair, or attempt finalization)
      done_futs = [f for f in list(pending_turn_futures.keys()) if f.done()]
      for fut in done_futs:
        st = pending_turn_futures.pop(fut)
        try:
          stop_r = fut.result()
        except Exception as exc:  # pylint: disable=broad-except
          st.error = f"{type(exc).__name__}: {exc}"[:500]
          _finalize_attempt_session_and_train_check(st, "error")
          stop_r = "error"
        if stop_r is None and not st.finished:
          _enqueue_attempt_turn(st, is_eos_continue=False)
        else:
          _record_finished_attempt(st)

      # 2. Admit queued `(task, attempt)` jobs up to `max_concurrent` using `pick(queue)`
      active_cnt = len({id(x) for x in active_requests.values()}) + len(
          pending_turn_futures
      )
      while job_queue and active_cnt < max_concurrent:
        pick_i = _pick_next_queued_job(job_queue, finished_by_task)
        t_spec, att_idx = job_queue.pop(pick_i)
        st = None
        spawn_err = ""
        for _spawn_try in range(2):
          try:
            st = _create_attempt_state(
                task_spec=t_spec,
                attempt_idx=att_idx,
                cfg=cfg,
                tools_spec=tools_spec,
                system_block=system_block,
            )
            break
          except Exception as exc:  # pylint: disable=broad-except
            spawn_err = f"{type(exc).__name__}: {exc}"[:300]
            time.sleep(1.0)
        if st is None:
          # Sandbox could not be created even after a retry: record an empty errored attempt
          # instead of crashing the whole shard (keeps the other 23 streams + resume intact).
          print(
              f"[havu_b32 {cfg.exp_id} | {shard_tag}] sandbox spawn failed for "
              f"{t_spec.task_id} attempt {att_idx}: {spawn_err}",
              flush=True,
          )
          st = _create_attempt_state_without_session(
              task_spec=t_spec,
              attempt_idx=att_idx,
              cfg=cfg,
              system_block=system_block,
          )
          st.error = f"sandbox spawn failed: {spawn_err}"
          st.labels = {"train_pass": None, "train_frac": None, "train_trace": "no_answer"}
          st.stop_reason = "error"
          st.finished = True
          _record_finished_attempt(st)
          active_cnt = len({id(x) for x in active_requests.values()}) + len(
              pending_turn_futures
          )
          continue
        _enqueue_attempt_turn(st, is_eos_continue=False)
        active_cnt = len({id(x) for x in active_requests.values()}) + len(
            pending_turn_futures
        )

      # 3. Step vLLM engine + run `GuardedStream` thinking gzip loop monitor
      if engine.has_unfinished_requests():
        step_outputs = engine.step()
        for ro in step_outputs:
          rid = str(ro.request_id)
          st = active_requests.get(rid)
          if st is None:
            continue
          comp = ro.outputs[0]
          cur_text = comp.text or ""
          cur_ids = list(comp.token_ids)

          if not getattr(ro, "finished", False):
            # Live streaming chunk: run Candidate B32 `GuardedStream` loop monitor while inside `<think>`
            if st.loop_still_thinking:
              scan_start = max(0, st.loop_checked_chars - 16)
              tail_slice = cur_text[scan_start:]
              if THINK_CLOSE in tail_slice or tc_open_str in tail_slice:
                st.loop_still_thinking = False
              st.loop_checked_chars = len(cur_text)
            while (
                st.loop_still_thinking
                and len(cur_ids) >= st.loop_next_check
            ):
              win_ids = cur_ids[st.loop_next_check - LOOP_WINDOW : st.loop_next_check]
              st.loop_next_check += LOOP_EVERY
              win_text = decode_fn(win_ids)
              st.loop_hits = (
                  st.loop_hits + 1 if gzip_ratio(win_text) > LOOP_RATIO else 0
              )
              if st.loop_hits >= LOOP_CONFIRM:
                _pop_active_request(rid)
                for abort_id in {st.active_internal_rid, st.active_external_rid, rid}:
                  if abort_id:
                    try:
                      engine.abort_request([abort_id])
                    except Exception:  # pylint: disable=broad-except
                      pass
                st.cuts["loop"] = st.cuts.get("loop", 0) + 1
                gen_n = len(cur_ids)
                total_gen_tokens_all += gen_n
                st.cum_completion_tokens += gen_n
                st.events.append({
                    "turn": st.n_turns,
                    "loop_guard_cut_tokens": len( st.partial_turn_token_ids) + gen_n,
                })
                fut = worker_pool.submit(_finalize_and_return, st, "output_budget")
                pending_turn_futures[fut] = st
                break
            continue

          # Request finished!
          _pop_active_request(rid)
          gen_n = len(cur_ids)
          total_gen_tokens_all += gen_n
          st.cum_completion_tokens += gen_n
          prompt_len = len(getattr(ro, "prompt_token_ids", None) or st.prompt_token_ids or ())
          st.cum_prompt_tokens += prompt_len
          st.last_prompt_plus_comp = prompt_len + gen_n
          fin_reason = str(comp.finish_reason or "stop")

          combined_text = st.partial_turn_text + cur_text
          combined_ids = st.partial_turn_token_ids + cur_ids

          # Spurious EOS inside open `<think>`: seamlessly continue up to `MAX_EOS_CONTINUES` (`agent_qwen.py`)
          if (
              fin_reason == "stop"
              and THINK_CLOSE not in combined_text
              and tc_open_str not in combined_text
              and st.eos_continues_this_turn < MAX_EOS_CONTINUES
              and st.cum_completion_tokens + 256 < st.max_total_output
          ):
            while combined_ids and combined_ids[-1] in eos_special_ids:
              combined_ids.pop()
            for eos_s in (stop_str, "<|im_end|>", "<|endoftext|>"):
              if combined_text.endswith(eos_s):
                combined_text = combined_text[: -len(eos_s)]
            st.partial_turn_text = combined_text
            st.partial_turn_token_ids = combined_ids
            st.eos_continues_this_turn += 1
            st.tok_stats["eos_continues"] += 1
            _enqueue_attempt_turn(st, is_eos_continue=True)
            continue

          if (
              fin_reason == "stop"
              and im_end_id is not None
              and (not combined_ids or combined_ids[-1] != im_end_id)
          ):
            combined_ids.append(int(im_end_id))
          if fin_reason == "stop" and not combined_text.endswith(stop_str):
            combined_text += stop_str

          fut = worker_pool.submit(
              _process_completed_turn,
              st,
              combined_text,
              combined_ids,
              fin_reason,
              engine_max_len,
              encode_fn,
          )
          pending_turn_futures[fut] = st
      else:
        time.sleep(0.02)

      now = time.time()
      if now - last_heartbeat_time >= 60.0:
        last_heartbeat_time = now
        elapsed = now - start_time
        tok_s = total_gen_tokens_all / max(1.0, elapsed)
        active_cnt = len({id(x) for x in active_requests.values()}) + len(
            pending_turn_futures
        )
        print(
            f"[STATUS {cfg.exp_id} | {shard_tag}] "
            f"done_tasks={len(finalized_task_ids)}/{len(shard_tasks)} "
            f"(pairs={len(finalized_pair_records)}/{total_pairs_in_shard}) | "
            f"active={active_cnt} | queued={len(job_queue)} | "
            f"gen_tokens={total_gen_tokens_all} ({tok_s:.1f} tok/s) | "
            f"elapsed={elapsed/60:.1f}m",
            flush=True,
        )
        _write_summary("running")

    _write_summary("completed")
    print(
        f"[havu_b32] COMPLETED {cfg.exp_id} ({shard_tag}). "
        f"Outputs written to {out_dir}",
        flush=True,
    )
  finally:
    worker_pool.shutdown(wait=True)

  return [
      finalized_pair_records[gp]
      for gp in sorted(finalized_pair_records.keys())
  ]
