"""Multi-TPU Full-Precision Agentic Evaluation Runner for DeepSeek-V4.1-Flash-1017 on ARC-AGI-2 (120 Tasks, 167 Test Pairs).

Written from scratch to match collaborator Vu (`havu`)'s `E030` / `E031` / `E029` (`sub80s40` + `effort_header="minimal"`)
setup and his Kaggle notebook (`arc2-qwen38-e032-sft-minimal.ipynb` / `havu/evals/arc_harness/executor.py`):

Six Root-Cause Fixes vs. `v4` (`http://xid/303385938`, which scored `47.4%` Pass@1 / `52.4%` Pass@2):
1. Isolated Persistent Child-Process IPython Sandbox (`LocalExecutor(style="ipython")`):
   - Replaces `v4`'s in-process `threading.Thread(daemon=True)` sandbox (which leaked unkillable zombie threads on
     30s timeouts that held the Python GIL, starved the TPU host loop, and only printed `ErrorType: msg` without
     line numbers).
   - Spawns a dedicated persistent `python3 -I -u` child process (`start_new_session=True`, `OMP/OPENBLAS/MKL=1`,
     `RLIMIT_AS=4096MB`) per attempt with exact NeMo-Skills IPython displayhook (`IPython.lib.pretty.pretty(..., max_width=79)`
     with leading newline for multiline outputs), IPython plain tracebacks (`<ipython-input-N-hash>` with failing source
     lines in `stdout`), `<output cut>` at `16,000` chars, and **two-stage SIGINT -> SIGKILL timeout recovery**
     (`os.killpg(pid, SIGINT)` interrupts a runaway loop via `KeyboardInterrupt` while **preserving all earlier session
     variables and functions**; only falls back to `SIGKILL` + respawn if `SIGINT` is ignored for `grace=5s`).
2. Exact `E030` System Header Without Conflicting `"Reasoning Effort: 50..."` Prefix:
   - In `v4`, `ds4_encoding.py` prepended `"Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough
     the reasoning)\\n\\n"` BEFORE `HAVU_EFFORT_HEADER_MINIMAL` (`"Reasoning Effort: Minimal. Keep your thinking short and terse..."`),
     giving the model two contradictory `Reasoning Effort:` headers!
   - In Vu's `E029`/`E030` (`--effort low`), `REASONING_EFFORT_PROMPTS["low"] == ""` (empty string). Passing
     `reasoning_effort=0` (`"none"`) renders ONLY `HAVU_EFFORT_HEADER_MINIMAL` on both Attempt 0 and Attempt 1.
3. Stochastic Top-K + Top-P Gumbel-Max Sampling (`temperature=1.0, top_k=20, top_p=0.95`) in the TPU Megakernel:
   - In `v4`, `decoder_macrokernel` used deterministic greedy `jnp.argmax` (`temperature=0.0`), causing `44.8%` of
     attempts to get trapped in deterministic prose repetition loops (`27k–32k` tokens in a single turn, where
     accuracy collapsed from `64.4%` -> `9.9%`) and forcing Attempt 0 and Attempt 1 to use different prompts.
   - Now uses sharded `lax.top_k` + `top_p=0.95` + deterministic per-request/per-position Gumbel-max sampling inside
     `decoder_macrokernel`, so both Attempt 0 and Attempt 1 run Vu's #1 best `E030` configuration with independent
     random seeds (`seed = attempt_idx * 10007 + turn_idx * 97 + pair_seed`).
4. Per-Turn Thinking Cap (`max_turn_tokens = 16000`) + Synthetic `run_python` `CAP_NOTE` (`--cap-nudge`):
   - Matches `havu/evals/run_deepseek_tools.py:L562-L575`: if a turn reaches `16,000` tokens without calling a tool,
     closes the turn with a synthetic `run_python` call (`pass  # per-turn thinking limit`) and a `"role": "tool"`
     message (`CAP_NOTE`) instead of appending a `"role": "user"` message, and trims the repetitive prose tail so
     it never evicts earlier valid code turns from the `49,152`-token context window.
5. Non-Destructive Transcript Compaction (`compact(messages, keep=2, tool_chars=1500)`):
   - Matches `agent_qwen.py:L286-L312`: never deletes `(assistant, tool)` turns (`del work_msgs[2:4]`), which caused
     `v4` to forget variables already defined in the IPython session. Instead, strips `reasoning_content` and trims
     `tool` outputs to `1,500` chars on turns older than the last `keep=2` turns when prompt tokens exceed `36,000`.
6. Complete 3-Tier Session Fallback (`answer` -> `solve(test_input)` -> `salvage_answer(session)`) + `train_check`:
   - Replicates `executor.py:L480-L548` (`salvage_answer` scoring all 1-arg user functions on `train` with
     `max_output=5_000_000` and applying the best one to `test_input`) + `train_check.py` (`expr`, `cell`, `fn`
     verification paths with `max_output=5_000_000`).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from concurrent import futures
import contextlib
import copy
import hashlib
import io
import json
import linecache
import os
import pprint
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
import uuid


# ==============================================================================
# 0. Pre-TPU Hermetic Zygote Fork Server for Google3 Embedded Binaries
# ==============================================================================
# On Borg, Google3 `py_binary` targets run via an embedded C++ launcher where
# `sys.executable` is empty and no external `/usr/bin/python3` with `numpy`,
# `scipy`, and `IPython` exists on `$PATH`.
# Forking a lightweight Zygote server HERE — BEFORE `jax`, `vllm`, `torch`, or
# `libtpu` are imported — gives us a tiny single-threaded pure-Python process
# with all hermetic `.runfiles` on `sys.path` and zero TPU/gRPC state.

_ZYGOTE_LOCK = threading.Lock()
_ZYGOTE_SOCK: socket.socket | None = None
_ZYGOTE_PID: int | None = None


def _apply_worker_limits(mem_mb: int) -> None:
  with contextlib.suppress(Exception):
    os.nice(10)
  with contextlib.suppress(Exception):
    import resource  # pylint: disable=g-import-not-at-top

    if mem_mb > 0:
      base_vm_bytes = 0
      with contextlib.suppress(Exception):
        with open("/proc/self/status", "r", encoding="utf-8") as sf:
          for sline in sf:
            if sline.startswith("VmSize:"):
              base_vm_bytes = int(sline.split()[1]) * 1024
              break
      limit_bytes = base_vm_bytes + mem_mb * 1024 * 1024
      resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    fsize_bytes = 64 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))


def _worker_ipython_loop(
    conn_sock: socket.socket,
    style: str,
    out_limit: int,
    pretty_fn: Any,
    np_mod: Any,
    scipy_mod: Any,
) -> None:
  """Runs the isolated per-attempt IPython execution loop over `conn_sock`."""
  cut_suffix = "\n<output cut>\n"
  # Isolate fd 0, 1, 2 to /dev/null while communicating exclusively over conn_sock
  with contextlib.suppress(Exception):
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    for target_fd in (0, 1, 2):
      if conn_sock.fileno() != target_fd:
        os.dup2(devnull_fd, target_fd)
    if devnull_fd > 2 and devnull_fd != conn_sock.fileno():
      os.close(devnull_fd)

  proto_f = conn_sock.makefile(
      "rw", buffering=1, encoding="utf-8", errors="replace"
  )
  cell_counter = 0

  def _cell_name(code: str) -> str:
    nonlocal cell_counter
    cell_counter += 1
    if style != "ipython":
      return "<cell>"
    h = hashlib.sha1(code.encode("utf-8", "replace")).hexdigest()[:12]
    name = f"<ipython-input-{cell_counter}-{h}>"
    linecache.cache[name] = (
        len(code),
        None,
        [line + "\n" for line in code.splitlines()],
        name,
    )
    return name

  def _exec_cell(code: str, ns: dict[str, Any]) -> Any:
    name = _cell_name(code)
    try:
      tree = ast.parse(code, name, "exec")
    except SyntaxError:
      exec(compile(code, name, "exec"), ns)  # pylint: disable=exec-used
      return None
    if not tree.body:
      return None
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
      if len(tree.body) > 1:
        mod = ast.Module(body=tree.body[:-1], type_ignores=[])
        exec(compile(mod, name, "exec"), ns)  # pylint: disable=exec-used
      expr = ast.Expression(body=last.value)
      ast.copy_location(expr, last)
      return eval(compile(expr, name, "eval"), ns)  # pylint: disable=eval-used
    exec(compile(tree, name, "exec"), ns)  # pylint: disable=exec-used
    return None

  def _show(value: Any) -> None:
    if value is None:
      return
    if style == "ipython":
      text = (
          pretty_fn(value, max_width=79)
          if pretty_fn is not None
          else pprint.pformat(value, width=79, sort_dicts=False)
      )
      print("\n" + text if "\n" in text else text)
    else:
      print(repr(value))

  def _ipython_traceback(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
      lines = traceback.format_exception_only(type(exc), exc)
      return "Traceback (most recent call last):\n" + "".join(
          "  " + l if l.startswith("File ") else l for l in lines
      )
    frames = []
    tb = exc.__traceback__
    while tb is not None:
      if tb.tb_frame.f_code.co_filename != "<string>" and not tb.tb_frame.f_code.co_filename.endswith(
          "deepseek_v41_arc2_eval.py"
      ):
        frames.append(tb)
      tb = tb.tb_next
    body = []
    for tb in frames:
      f, lineno = tb.tb_frame, tb.tb_lineno
      fn, name = f.f_code.co_filename, f.f_code.co_name
      src = linecache.getline(fn, lineno).strip()
      if fn.startswith("<ipython-input-") and name == "<module>":
        if src:
          body.append(f"    {src}\n")
      else:
        loc = fn if name == "<module>" else f"{fn} in {name}"
        body.append(f"  File {loc}:{lineno}\n")
        if src:
          body.append(f"    {src}\n")
    only = "".join(traceback.format_exception_only(type(exc), exc))
    return "Traceback (most recent call last):\n" + "".join(body) + only

  def _post(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n").rstrip("\n")
    if limit and len(text) > limit:
      return text[:limit] + cut_suffix
    return text

  ns: dict[str, Any] = {"__name__": "__main__", "np": np_mod}
  if scipy_mod is not None:
    ns["scipy"] = scipy_mod

  for raw in proto_f:
    if not raw:
      break
    req = json.loads(raw)
    code = req.get("code", "")
    req_limit = int(req.get("out_limit", out_limit))
    out_buf, err_buf = io.StringIO(), io.StringIO()
    status = "completed"
    try:
      with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(
          err_buf
      ):
        val = _exec_cell(code, ns)
        _show(val)
    except KeyboardInterrupt:
      status = "timeout"
      if style == "ipython":
        out_buf.write("KeyboardInterrupt\n")
      else:
        err_buf.write("KeyboardInterrupt\n")
    except SystemExit as e:
      status = "error"
      target = out_buf if style == "ipython" else err_buf
      target.write(f"SystemExit: {e}\n")
    except BaseException as e:  # pylint: disable=broad-exception-caught
      status = "error"
      if style == "ipython":
        out_buf.write(_ipython_traceback(e))
      else:
        traceback.print_exc(file=err_buf)
    if style == "ipython":
      stdout = _post(out_buf.getvalue() + err_buf.getvalue(), req_limit)
      stderr = ""
    else:
      stdout = out_buf.getvalue()
      stderr = err_buf.getvalue()
    proto_f.write(
        json.dumps(
            {"process_status": status, "stdout": stdout, "stderr": stderr}
        )
        + "\n"
    )
    proto_f.flush()


def _zygote_server_main(zygote_sock: socket.socket) -> None:
  """Runs in the pre-TPU Zygote child process and forks isolated IPython workers on demand."""
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  os.environ["HOME"] = "/tmp"
  os.environ["TMPDIR"] = "/tmp"
  with contextlib.suppress(Exception):
    os.setsid()
  with contextlib.suppress(Exception):
    os.chdir("/tmp")
  # Auto-reap exited sandbox workers so no zombie processes accumulate
  signal.signal(signal.SIGCHLD, signal.SIG_IGN)

  import numpy as _np_pre  # pylint: disable=g-import-not-at-top

  _scipy_pre = None
  with contextlib.suppress(Exception):
    import scipy as _sp  # pylint: disable=g-import-not-at-top
    import scipy.ndimage  # pylint: disable=g-import-not-at-top,unused-import

    _scipy_pre = _sp

  _pretty_pre = None
  with contextlib.suppress(Exception):
    from IPython.lib.pretty import pretty as _p_fn  # pylint: disable=g-import-not-at-top

    _pretty_pre = _p_fn

  z_file = zygote_sock.makefile("r", encoding="utf-8")
  for line in z_file:
    if not line:
      break
    req = json.loads(line)
    style = str(req.get("style", "ipython"))
    mem_mb = int(req.get("mem_mb", 4096))
    out_limit = int(req.get("out_limit", 16000))

    p_end, w_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    w_pid = os.fork()
    if w_pid == 0:
      # Grandchild sandbox worker process
      try:
        z_file.close()
        zygote_sock.close()
        p_end.close()
        os.setsid()
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        _apply_worker_limits(mem_mb)
        _worker_ipython_loop(
            w_end, style, out_limit, _pretty_pre, _np_pre, _scipy_pre
        )
      except BaseException:  # pylint: disable=broad-exception-caught
        pass
      finally:
        os._exit(0)
    else:
      w_end.close()
      socket.send_fds(
          zygote_sock, [struct.pack("<i", w_pid)], [p_end.fileno()]
      )
      p_end.close()
  os._exit(0)


def _ensure_sandbox_zygote() -> None:
  """Starts the pre-TPU Zygote fork server once before JAX/libtpu is imported."""
  global _ZYGOTE_SOCK, _ZYGOTE_PID
  if _ZYGOTE_SOCK is not None:
    return
  parent_sock, child_sock = socket.socketpair(
      socket.AF_UNIX, socket.SOCK_STREAM
  )
  pid = os.fork()
  if pid == 0:
    parent_sock.close()
    _zygote_server_main(child_sock)
    os._exit(0)
  else:
    child_sock.close()
    _ZYGOTE_SOCK = parent_sock
    _ZYGOTE_PID = pid


_ensure_sandbox_zygote()

from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from vllm.g3_integration import app_runner

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_tpu_vllm_traces as ds_traces
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_encoding
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import ds4_engine_fns as engine_lib
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import prompts_and_voting
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import qwen_tpu_vllm_eval as runner_lib
from google3.pyglib import gfile

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "enable_wave2_evolutionary_repair",
    False,
    "Unused in Vu E030 replication (kept for flag compatibility).",
)
flags.DEFINE_boolean(
    "drop_prior_turn_thinking",
    False,
    "If False (default, matching ds4_encoding.py and Vu E030), keeps <think>...</think> across turns until compact_at.",
)
flags.DEFINE_boolean(
    "compact_grid_prompts",
    False,
    "If False (default, matching Vu ARC Prize prompt), formats grids with standard json.dumps(grid).",
)
flags.DEFINE_integer(
    "max_turn_tokens",
    64000,
    "Maximum output tokens per single assistant turn (Vu E029/E030 sub80s40 = 64000).",
)
flags.DEFINE_integer(
    "max_reasoning_tokens",
    0,
    "Optional separate cap on <think> tokens (0 = disabled, governed by total_token_budget=80000 matching Vu E030 sub80s40).",
)
flags.DEFINE_boolean(
    "token_exact",
    True,
    "Preserve exact sampled token_ids across multi-turn tool calls (matching Vu E043 --token-exact).",
)
flags.DEFINE_boolean(
    "rescue_code",
    True,
    "Rescue Python code written outside a valid DSML tool call or nudge when no tool/grid is emitted (matching Vu E043 --rescue-code).",
)
flags.DEFINE_integer(
    "turns_note_threshold",
    5,
    "Append '[N turns left: call submit_answer...]' when N or fewer turns remain (Vu E029/E030 --turns-note 5).",
)
flags.DEFINE_float(
    "sandbox_timeout_s",
    30.0,
    "Per-cell execution timeout in seconds for LocalExecutor(style='ipython') (matching Vu's 30s timeout).",
)
flags.DEFINE_integer(
    "compact_at_tokens",
    60000,
    "Prompt token threshold at which non-destructive transcript compaction triggers (60000 for 65536 context).",
)


# ==============================================================================
# 1. Collaborator Vu (`havu`) Exact Prompts, Tool Schemas, and Notes (`E030`)
# ==============================================================================

ARC_PRIZE_TEMPLATE = (
    "You are participating in a puzzle solving competition. You are an expert at"
    " solving puzzles.\n\nBelow is a list of input and output pairs with a"
    " pattern. Your goal is to identify the pattern or transformation in the"
    " training examples that maps the input to the output, then apply that"
    " pattern to the test input to give a final output.\n\nRespond in the"
    " format of the training output examples\n\n--Training"
    " Examples--\n{training_examples}\n--End of Training Examples--\n\n--Test"
    " Input--\n{test_input}\n--End of Test Input--\n\nYour response:"
)

HAVU_LEAN2_SUFFIX = (
    "\n\nYou have a Python tool. The variables `train` (list of {'input',"
    " 'output'} grids) and `test_input` are already loaded in it: use them"
    " directly and never retype or reassign them. Work efficiently: inspect the"
    " grids with code instead of reading them by hand, form a hypothesis, and"
    " write code that computes the output. Never construct, copy or count grids"
    " by hand. Check your rule on every training pair with code before"
    " answering, and if a pair fails, fix the rule rather than explaining the"
    " mismatch away. Before answering, check that the rule's assumptions also"
    " hold on test_input. Keep your reasoning brief."
)

HAVU_SUBMIT_SUFFIX = (
    "\n\nWhen your code reproduces every training output and its assumptions"
    " hold on test_input, call submit_answer with an expression such as"
    " `solve(test_input)`: the harness evaluates it in your session and that"
    " grid is your answer, so never retype the grid. Keep your current best"
    " guess for the test output in a variable `answer`; if you run out of"
    " budget it is submitted automatically."
)

HAVU_LEAN2SUB_SUFFIX = HAVU_LEAN2_SUFFIX + HAVU_SUBMIT_SUFFIX

HAVU_LEAN2CODE_ADDENDUM = (
    "\n\nLet code do the perceiving and the checking. Get coordinates, counts,"
    " object lists, distances, paths and mismatches by printing them from code"
    " instead of reading them off the grids or working them out cell by cell in"
    " your reasoning, and do not re-read grids that code has already analysed."
    " When you are choosing between candidate rules, implement each one and"
    " compare them on the training pairs in code rather than arguing it out in"
    " prose."
)

HAVU_EFFORT_HEADER_MINIMAL = (
    "Reasoning Effort: Minimal. Keep your thinking short and terse: write brief"
    " notes, not full sentences. Do not restate the task, do not re-read or"
    " re-derive what code has already printed, and do not re-verify what has"
    " already been checked. Test ideas with small code experiments instead of"
    " long deliberation, and submit as soon as the rule reproduces every"
    " training pair."
)

HAVU_CAP_NOTE = (
    "[per-turn thinking limit: this turn reached {k}k tokens without running"
    " code, so it was stopped here. Do not keep deliberating: write the"
    " smallest code experiment that tests your current best hypothesis on the"
    " training pairs and run it now.]"
)

HAVU_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python cell in your persistent IPython session."
                " Variables/functions persist. Preloaded: train (list of"
                " {'input': grid, 'output': grid}), test_input (grid), numpy as"
                " np. scipy is available. No internet or host files. Timeout"
                " 30s. Use print() to inspect results. Check transformations on"
                " all training pairs."
            ),
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
            "description": (
                "End this attempt with your test output. `code` is a Python"
                " expression evaluated in your session, e.g. solve(test_input)"
                " or answer, giving a rectangular grid of ints 0-9 (list or"
                " numpy array); statements that set a variable `answer` also"
                " work. The harness reads the grid directly: do not retype it."
                " Invalid grids return an error and the attempt continues."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]

DS41_STOP_STRINGS: list[str] = [
    "</｜DSML｜ calls>",
    "</|DSML| calls>",
    "</｜DSML｜tool_calls>",
    "</|DSML|tool_calls>",
    "</tool_call>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
]

DS41_KEEP_STOP_SUFFIXES = (
    "</｜DSML｜ calls>",
    "</|DSML| calls>",
    "</｜DSML｜tool_calls>",
    "</|DSML|tool_calls>",
    "</tool_call>",
)


def build_havu_arcprize_prompt(
    train: list[dict[str, Any]],
    test_input: list[list[int]],
    suffix: str = HAVU_LEAN2SUB_SUFFIX,
    compact_grids: bool = False,
) -> str:
  """Builds the exact ARC Prize prompt + `lean2sub` suffix from `havu/evals/run_deepseek_tools.py`."""
  ex = ""
  for i, pair in enumerate(train):
    inp_str = (
        json.dumps(pair["input"], separators=(",", ":"))
        if compact_grids
        else json.dumps(pair["input"])
    )
    out_str = (
        json.dumps(pair["output"], separators=(",", ":"))
        if compact_grids
        else json.dumps(pair["output"])
    )
    ex += f"--Example {i}-- \n\n INPUT: \n\n{inp_str}\n\nOUTPUT: \n\n{out_str}\n\n"
  test_str = (
      json.dumps(test_input, separators=(",", ":"))
      if compact_grids
      else json.dumps(test_input)
  )
  return ARC_PRIZE_TEMPLATE.format(training_examples=ex, test_input=test_str) + suffix


def build_havu_initial_messages(
    task: dict[str, Any],
    test_input: list[list[int]],
    attempt_idx: int,
    compact_grids: bool = False,
) -> tuple[list[dict[str, Any]], int]:
  """Builds initial OpenAI-format messages and `reasoning_effort=0` (`low` in Vu's E030 adds no numeric prefix).

  With stochastic `temperature=1.0` Gumbel-max sampling active in the TPU megakernel,
  BOTH Attempt 0 and Attempt 1 run Vu's #1 highest-scoring configuration (`E030`:
  `sub80s40` + `HAVU_EFFORT_HEADER_MINIMAL` system message, `74.5%` Pass@1 / `84.9%` Pass@2)
  with independent random seeds!
  """
  suffix = (
      HAVU_LEAN2SUB_SUFFIX
      if attempt_idx < 2
      else (HAVU_LEAN2SUB_SUFFIX + HAVU_LEAN2CODE_ADDENDUM)
  )
  prompt_text = build_havu_arcprize_prompt(
      task["train"], test_input, suffix=suffix, compact_grids=compact_grids
  )
  messages: list[dict[str, Any]] = [
      {
          "role": "system",
          "content": HAVU_EFFORT_HEADER_MINIMAL,
          "tools": copy.deepcopy(HAVU_TOOLS_SPEC),
      },
      {"role": "user", "content": prompt_text},
  ]
  # `effort=0` omits the conflicting "Reasoning Effort: 50 (range 1-100...)" prefix so that
  # `HAVU_EFFORT_HEADER_MINIMAL` is the sole Reasoning Effort header, matching Vu's E030!
  return messages, 0


def format_havu_budget_note(
    used_tokens: int,
    soft_target: int,
    hard_limit: int,
    budget_notes_frac: float = 0.01,
    budget_warn_frac: float = 0.75,
) -> str:
  """Exact `budget_note` from `havu/evals/run_deepseek_tools.py` (`sub80s40`: soft=40000, hard=80000, warn=0.75)."""
  if used_tokens < budget_notes_frac * hard_limit:
    return ""
  if soft_target > 0:
    note = (
        f"\n[budget: {used_tokens / 1000:.0f}k output tokens used; aim to"
        f" finish within {soft_target / 1000:.0f}k (hard limit"
        f" {hard_limit / 1000:.0f}k)]"
    )
    if soft_target <= used_tokens < budget_warn_frac * hard_limit:
      note += " Past the target: submit as soon as a rule fits every training pair."
  else:
    note = (
        f"\n[budget: {used_tokens / 1000:.0f}k of {hard_limit / 1000:.0f}k"
        " output tokens used]"
    )
  if used_tokens >= budget_warn_frac * hard_limit:
    note += (
        " Budget nearly spent: submit your best answer now with submit_answer"
        " (or at least set `answer`)."
    )
  return note


def format_havu_turns_note(turn_index: int, max_turns: int, threshold: int = 5) -> str:
  """Exact `turns_note` from `havu/evals/run_deepseek_tools.py` (`--turns-note 5`)."""
  left = max_turns - turn_index - 1
  if threshold <= 0 or left > threshold or left <= 0:
    return ""
  plural = "s" if left != 1 else ""
  return (
      f"\n[{left} turn{plural} left: call submit_answer with your best answer"
      " now, or at least set `answer`.]"
  )


# ==============================================================================
# 2. DSML Completion Parser & ARC Prize / Lenient Grid Parser
# ==============================================================================

_DSML_INVOKE_BLOCK_RE = re.compile(
    r"<[｜|]DSML[｜|]\s*invoke\s+name=[\"']?([a-zA-Z0-9_]+)[\"']?\s*>(.*?)(?:</[｜|]DSML[｜|]\s*invoke>|</function>|</tool_call>|$)",
    re.DOTALL,
)
_DSML_CODE_PARAM_RE = re.compile(
    r"<[｜|]DSML[｜|]\s*parameter\s+name=[\"']?code[\"']?[^>]*>(.*?)(?:</[｜|]DSML[｜|]\s*parameter>|</parameter>|$)",
    re.DOTALL,
)
_XML_TOOL_RE = re.compile(
    r"<function=([a-zA-Z0-9_]+)>\s*<parameter=code>\n?(.*?)(?:</parameter>|</function>|</tool_call>|$)",
    re.DOTALL,
)


def _clean_extracted_code(code: str) -> str:
  """Strips any accidental trailing XML/DSML closing tags from extracted Python code."""
  for tag in (
      "</｜DSML｜ parameter>",
      "</|DSML| parameter>",
      "</｜DSML｜parameter>",
      "</|DSML|parameter>",
      "</｜DSML｜ invoke>",
      "</|DSML| invoke>",
      "</｜DSML｜invoke>",
      "</|DSML|invoke>",
      "</｜DSML｜ calls>",
      "</|DSML| calls>",
      "</｜DSML｜tool_calls>",
      "</|DSML|tool_calls>",
      "</parameter>",
      "</function>",
      "</tool_call>",
  ):
    if tag in code:
      code = code.split(tag, 1)[0]
  return code.strip()


_RESCUE_DSML_CODE_RE = re.compile(
    r'parameter name="code"[^>]*>(.*?)(?:</[^>]*parameter>|$)', re.DOTALL
)
_RESCUE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.DOTALL)
_RESCUE_JSON_CODE_RE = re.compile(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_RESCUE_TAG_RE = re.compile(r"</?[^<>\n]*DSML[^<>\n]*>?|<br[^>]*>")
_RESCUE_CODE_LINE_RE = re.compile(
    r"^\s*(import |from \w+ import |print\(|for |def |if |while |np\.|[A-Za-z_]\w*\s*(\[[^\]]*\])?\s*=[^=])"
)
HAVU_RESCUE_NUDGE = (
    "[harness] No valid tool call was found in your last message. To run code,"
    " call the run_python tool (do not write code as plain text); to finish,"
    " call submit_answer."
)


def rescue_havu_tool_name(name: str) -> str:
  """Exact `rescue_name` from `havu/evals/run_deepseek_tools.py:L371-L378` (`E043 --rescue-code`)."""
  low = (name or "").lower()
  if "submit" in low:
    return "submit_answer"
  if "python" in low or "run" in low:
    return "run_python"
  return name


def rescue_havu_code(content: str | None) -> str | None:
  """Exact `rescue_code` from `havu/evals/run_deepseek_tools.py:L337-L359` (`E043 --rescue-code`)."""
  text = content or ""
  m = _RESCUE_DSML_CODE_RE.findall(text)
  if m and m[-1].strip():
    return _RESCUE_TAG_RE.sub("", m[-1])
  m = _RESCUE_JSON_CODE_RE.findall(text)
  if m:
    try:
      return json.loads('"' + m[-1] + '"')
    except ValueError:
      pass
  f = [b for b in _RESCUE_FENCE_RE.findall(text) if b.strip()]
  if f:
    return _RESCUE_TAG_RE.sub("", f[-1])
  lines = _RESCUE_TAG_RE.sub("", text).splitlines()
  start = next(
      (i for i, l in enumerate(lines) if _RESCUE_CODE_LINE_RE.match(l)), None
  )
  if start is None:
    return None
  code = "\n".join(lines[start:]).strip()
  return code or None


def rescue_havu_tool_call(
    content: str | None, turn_idx: int
) -> dict[str, Any] | None:
  """Exact `rescue_tool_call` from `havu/evals/run_deepseek_tools.py:L361-L369` (`E043 --rescue-code`)."""
  code = rescue_havu_code(content)
  if not code or not code.strip():
    return None
  return {
      "id": f"rescue_{turn_idx}",
      "type": "function",
      "function": {
          "name": "run_python",
          "arguments": json.dumps({"code": code}),
      },
  }


def parse_ds41_assistant_turn(
    raw_text: str, turn_idx: int, finish_reason: str = "stop"
) -> tuple[str, str, list[dict[str, Any]], bool]:
  """Parses a DeepSeek-V4.1 completion into `(reasoning_content, content, openai_tool_calls, is_canonical)`.

  First attempts strict canonical parsing via `ds4_encoding.parse_message_from_completion_text`
  (matching Vu's `_parse_ds_strict` in `E043 --token-exact`). If canonical, returns `is_canonical=True`
  so the exact sampled `token_ids` can be continued on the next turn without re-tokenization breaks.
  Otherwise falls back to lenient parsing with `is_canonical=False`.
  """
  text = raw_text or ""
  if text.startswith("<think>"):
    text = text[len("<think>") :]

  # 1. Try strict canonical V4.1 DSML parse (`E043 --token-exact`)
  if finish_reason == "stop":
    try:
      check_text = (
          text
          if text.endswith(ds4_encoding.eos_token)
          else text + ds4_encoding.eos_token
      )
      parsed_msg = ds4_encoding.parse_message_from_completion_text(
          check_text, thinking_mode="thinking"
      )
      strict_rc = parsed_msg.get("reasoning_content") or ""
      strict_content = parsed_msg.get("content") or ""
      raw_calls = parsed_msg.get("tool_calls") or []
      strict_calls: list[dict[str, Any]] = []
      names_canonical = True
      for call_i, tc in enumerate(raw_calls):
        fn_obj = tc.get("function") or {}
        orig_name = fn_obj.get("name") or ""
        mapped_name = rescue_havu_tool_name(orig_name)
        if mapped_name != orig_name or mapped_name not in (
            "run_python",
            "submit_answer",
        ):
          names_canonical = False
        strict_calls.append({
            "id": f"call_{turn_idx}_{call_i}",
            "type": "function",
            "function": {
                "name": (
                    mapped_name
                    if mapped_name in ("run_python", "submit_answer")
                    else "run_python"
                ),
                "arguments": fn_obj.get("arguments") or "{}",
            },
        })
      return strict_rc, strict_content, strict_calls, names_canonical
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  reasoning_content = ""
  remainder = ""
  if "</think>" in text:
    reasoning_content, remainder = text.split("</think>", 1)
  else:
    tc_pos = -1
    for marker in (
        "<｜DSML｜ calls>",
        "<|DSML| calls>",
        "<｜DSML｜tool_calls>",
        "<|DSML|tool_calls>",
        "<｜DSML｜ invoke",
        "<｜DSML｜invoke",
        "<tool_call>",
    ):
      idx = text.find(marker)
      if idx != -1 and (tc_pos == -1 or idx < tc_pos):
        tc_pos = idx
    if tc_pos != -1:
      reasoning_content = text[:tc_pos]
      remainder = text[tc_pos:]
    else:
      reasoning_content = text
      remainder = ""

  tool_calls: list[dict[str, Any]] = []
  visible_content = remainder

  for marker in (
      "<｜DSML｜ calls>",
      "<|DSML| calls>",
      "<｜DSML｜tool_calls>",
      "<|DSML|tool_calls>",
      "<｜DSML｜ invoke",
      "<｜DSML｜invoke",
      "<tool_call>",
  ):
    m_idx = visible_content.find(marker)
    if m_idx != -1:
      visible_content = visible_content[:m_idx]
      break
  visible_content = visible_content.replace(ds4_encoding.eos_token, "").strip()

  for call_i, m in enumerate(_DSML_INVOKE_BLOCK_RE.finditer(remainder)):
    raw_name = rescue_havu_tool_name(m.group(1).strip())
    fn_name = (
        "submit_answer"
        if raw_name in ("submit_answer", "submit_program", "submit_solution")
        else "run_python"
    )
    body = m.group(2)
    pm = _DSML_CODE_PARAM_RE.search(body)
    if pm:
      code_val = _clean_extracted_code(pm.group(1))
    else:
      code_val = _clean_extracted_code(body)
    if code_val:
      tool_calls.append({
          "id": f"call_{turn_idx}_{call_i}",
          "type": "function",
          "function": {
              "name": fn_name,
              "arguments": json.dumps({"code": code_val}),
          },
      })

  if not tool_calls:
    for call_i, m in enumerate(_XML_TOOL_RE.finditer(remainder)):
      raw_name = rescue_havu_tool_name(m.group(1).strip())
      fn_name = (
          "submit_answer"
          if raw_name in ("submit_answer", "submit_program", "submit_solution")
          else "run_python"
      )
      code_val = _clean_extracted_code(m.group(2))
      if code_val:
        tool_calls.append({
            "id": f"call_{turn_idx}_{call_i}",
            "type": "function",
            "function": {
                "name": fn_name,
                "arguments": json.dumps({"code": code_val}),
            },
        })

  return reasoning_content, visible_content, tool_calls, False


def is_valid_havu_grid(grid: Any) -> bool:
  """Exact `valid_grid` check from `havu/evals/run_deepseek_tools.py`."""
  return (
      isinstance(grid, list)
      and 1 <= len(grid) <= 30
      and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
      and len({len(row) for row in grid}) == 1
      and all(
          type(value) is int and 0 <= value <= 9
          for row in grid
          for value in row
      )
  )


def havu_arcprize_parse(text: str) -> list[list[int]] | None:
  """Exact `parse` + `lenient_parse` from `havu/evals/arc_harness/arcprize.py` and `run_deepseek_tools.py`."""
  if not text:
    return None
  m = re.search(r"\\boxed\{(.*?)\}", text, re.DOTALL)
  if m:
    try:
      g = json.loads(m.group(1).strip())
      if is_valid_havu_grid(g):
        return g
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  end = text.rfind("]")
  if end >= 0:
    depth = 1
    for i in range(end - 1, -1, -1):
      depth += (text[i] == "]") - (text[i] == "[")
      if depth == 0:
        try:
          g = json.loads(text[i : end + 1])
          if is_valid_havu_grid(g):
            return g
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        break
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
        except Exception:  # pylint: disable=broad-exception-caught
          try:
            grid = json.loads(re.sub(r",\s*\]", "]", block))
          except Exception:  # pylint: disable=broad-exception-caught
            grid = None
        if is_valid_havu_grid(grid):
          return grid
        break
  return None


# ==============================================================================
# 3. Collaborator Vu (`havu`) Isolated Child-Process IPython Sandbox (`executor.py`)
# ==============================================================================
# Replicates `havu/evals/arc_harness/executor.py` (Cell 9 of `arc2-qwen38-e032-sft-minimal.ipynb`)
# + `havu/evals/train_check.py` verbatim using the pre-TPU Zygote fork server.


def _spawn_worker_from_zygote(
    style: str, mem_mb: int, out_limit: int
) -> tuple[int, socket.socket]:
  """Requests a freshly forked, process-group-isolated IPython worker from `_ZYGOTE_SOCK`."""
  _ensure_sandbox_zygote()
  with _ZYGOTE_LOCK:
    assert _ZYGOTE_SOCK is not None
    req = (
        json.dumps({"style": style, "mem_mb": mem_mb, "out_limit": out_limit})
        + "\n"
    )
    _ZYGOTE_SOCK.sendall(req.encode("utf-8"))
    msg, fds, _, _ = socket.recv_fds(_ZYGOTE_SOCK, 4, 1)
    if len(msg) < 4 or not fds:
      raise RuntimeError("Failed to receive worker PID and socket from Zygote")
    (w_pid,) = struct.unpack("<i", msg[:4])
    w_sock = socket.socket(
        family=socket.AF_UNIX, type=socket.SOCK_STREAM, fileno=fds[0]
    )
    return int(w_pid), w_sock


def format_tool_output(result: dict[str, Any], limit: int = 16000) -> str:
  """Exact `tool_output` formatting from `havu/evals/arc_harness/executor.py:L209-L216`."""
  text = result.get("stdout", "") or ""
  if result.get("stderr"):
    text += ("\n[stderr]\n" if text else "[stderr]\n") + result["stderr"]
  text = f"[{result['process_status']}]\n" + (
      text if text else "[no output; use print()]"
  )
  return text[:limit] + "\n[output truncated]" if len(text) > limit else text


def _grid_hash(grid: list[list[int]]) -> str:
  return hashlib.sha1(
      json.dumps(grid, separators=(",", ":")).encode()
  ).hexdigest()[:16]


class HavuSubprocessSession:
  """Isolated persistent child-process IPython session matching `havu/evals/arc_harness/executor.py:LocalSession`.

  - Spawns an isolated child process from the pre-TPU Zygote in its own process group (`os.setsid()`).
  - Preloads `train` (deep-copied training pairs), `test_input`, `numpy as np`, and `scipy`.
  - Uses `style="ipython"` (`IPython.lib.pretty.pretty`, IPython plain tracebacks with `<ipython-input-N-hash>` and
    source lines in `stdout`, `<output cut>` at `16,000` chars).
  - Two-stage timeout recovery: on timeout (`30.0s`), sends `SIGINT` to the process group (`os.killpg(pid, SIGINT)`)
    so the timed-out cell raises `KeyboardInterrupt` and **preserves all earlier session variables and functions**;
    only sends `SIGKILL` and respawns (with `train`, `test_input`, `np` restored) if `SIGINT` does not stop the cell
    within `grace=5.0s`.
  """

  def __init__(
      self,
      train: list[dict[str, Any]],
      test_input: list[list[int]],
      timeout_s: float = 30.0,
      max_output_chars: int = 16000,
      style: str = "ipython",
      mem_mb: int = 4096,
      grace_s: float = 5.0,
  ):
    self.train = copy.deepcopy(train)
    self.test_input = copy.deepcopy(test_input)
    self.timeout = float(timeout_s)
    self.max_output = int(max_output_chars)
    self.style = style
    self.mem_mb = int(mem_mb)
    self.grace = float(grace_s)
    self.executed: list[tuple[str, str]] = []
    self._pid: int | None = None
    self._sock: socket.socket | None = None
    self._file: Any = None
    self._need_restore_note = False
    self._start(preload=True)

  def _is_alive(self) -> bool:
    if self._pid is None or self._sock is None:
      return False
    try:
      os.kill(self._pid, 0)
      return True
    except ProcessLookupError:
      return False
    except PermissionError:
      return True

  def _start(self, preload: bool = True) -> None:
    self.close()
    out_limit = self.max_output if self.style == "ipython" else 0
    self._pid, self._sock = _spawn_worker_from_zygote(
        self.style, self.mem_mb, out_limit
    )
    self._file = self._sock.makefile(
        "rw", buffering=1, encoding="utf-8", errors="replace"
    )
    if preload and (self.train is not None or self.test_input is not None):
      payload = json.dumps({"train": self.train, "test_input": self.test_input})
      setup = (
          f"import json as _json\n_d = _json.loads({payload!r})\n"
          "train = _d['train']\ntest_input = _d['test_input']\ndel _d, _json\n"
      )
      res = self._raw_execute(setup, timeout=self.timeout)
      if res["process_status"] != "completed":
        raise RuntimeError(f"Sandbox preload failed: {res}")

  def _read_until(self, deadline: float) -> str | None:
    if self._sock is None or self._file is None:
      return None
    fd = self._sock.fileno()
    poller = select.poll()
    poller.register(fd, select.POLLIN | select.POLLHUP | select.POLLERR)
    while True:
      rem = deadline - time.monotonic()
      if rem <= 0:
        return None
      events = poller.poll(max(1, int(rem * 1000.0)))
      if not events:
        return None
      line = self._file.readline()
      return line if line else None

  def _kill_group(self, sig: int) -> None:
    if self._pid is not None:
      with contextlib.suppress(Exception):
        os.killpg(self._pid, sig)

  def _raw_execute(
      self, code: str, timeout: float | None = None
  ) -> dict[str, Any]:
    timeout_val = self.timeout if timeout is None else float(timeout)
    if not self._is_alive() or self._file is None:
      self._start(preload=True)
      self._need_restore_note = True
    assert self._file is not None
    req_limit = self.max_output if self.style == "ipython" else 0
    payload_str = json.dumps({"code": code, "out_limit": req_limit}) + "\n"
    try:
      self._file.write(payload_str)
      self._file.flush()
    except (BrokenPipeError, OSError):
      self._start(preload=True)
      self._need_restore_note = True
      assert self._file is not None
      self._file.write(payload_str)
      self._file.flush()

    line = self._read_until(time.monotonic() + timeout_val)
    if line is not None:
      try:
        return json.loads(line)
      except json.JSONDecodeError:
        pass

    if self._is_alive():
      # Stage 1: Send SIGINT to the process group so KeyboardInterrupt stops the cell while keeping session state!
      self._kill_group(signal.SIGINT)
      line = self._read_until(time.monotonic() + self.grace)
      if line is not None:
        try:
          res = json.loads(line)
          res["process_status"] = "timeout"
          return res
        except json.JSONDecodeError:
          pass
      # Stage 2: Hard SIGKILL if SIGINT was ignored for `grace` seconds
      self._kill_group(signal.SIGKILL)
      status, msg = (
          "timeout",
          f"Timed out after {timeout_val:g}s (process killed)",
      )
    else:
      status, msg = "error", "Python process exited (likely OOM)"

    self._start(preload=True)
    self._need_restore_note = True
    if self.style == "ipython":
      return {"process_status": status, "stdout": msg, "stderr": ""}
    return {"process_status": status, "stdout": "", "stderr": msg}

  def execute(
      self,
      code: str,
      timeout: float | None = None,
      record: str | None = None,
  ) -> dict[str, Any]:
    if record:
      self.executed.append((record, code))
    res = self._raw_execute(code, timeout=timeout)
    if self._need_restore_note:
      self._need_restore_note = False
      note = "[Python state lost. train/test_input/np restored; redefine other variables.]\n"
      res["stdout"] = note + (res.get("stdout") or "")
    for k in ("stdout", "stderr"):
      if len(res.get(k, "")) > self.max_output:
        res[k] = res[k][: self.max_output] + "\n[output truncated]"
    return res

  def run_python(self, code: str) -> tuple[str, str]:
    """Executes `code` via `run_python` and returns `(process_status, formatted_tool_output)`."""
    res = self.execute(code, record="run_python")
    return res["process_status"], format_tool_output(res, self.max_output)

  def submit_answer(
      self, code: str, record_execution: bool = True
  ) -> tuple[str, list[list[int]] | None]:
    """Exact `submit_answer` from `havu/evals/arc_harness/executor.py:L433-L477`."""
    marker = f"ARC_ANSWER_{uuid.uuid4().hex}:"
    cell = f"""
def _arc_answer():
    import contextlib, io, json
    _src = {code!r}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            _expr = compile(_src, '<answer>', 'eval')
        except SyntaxError:
            _expr = None
        if _expr is not None:
            value = eval(_expr, globals())
        else:
            exec(compile(_src, '<answer>', 'exec'), globals())
            if 'answer' not in globals():
                raise NameError('code did not define answer')
            value = globals()['answer']
        if hasattr(value, 'tolist'):
            value = value.tolist()
        text = json.dumps(value, default=int)
    print({marker!r} + text)
try:
    _arc_answer()
finally:
    globals().pop('_arc_answer', None)
"""
    res = self.execute(
        cell, record="submit_answer" if record_execution else None
    )
    grid = None
    if res["process_status"] == "completed":
      for line in reversed(res.get("stdout", "").splitlines()):
        if line.startswith(marker):
          try:
            grid = json.loads(line[len(marker) :])
          except json.JSONDecodeError:
            pass
          break
      else:
        return "[answer not accepted] no value returned", None
    else:
      return (
          format_tool_output(res, self.max_output)[:2000]
          + "\n[answer not accepted]",
          None,
      )
    if not is_valid_havu_grid(grid):
      shape = (
          f"{len(grid)} rows, row lengths"
          f" {sorted({len(r) for r in grid if isinstance(r, list)})}"
          if isinstance(grid, list)
          else type(grid).__name__
      )
      return (
          "[answer not accepted] must be a rectangular 1-30 by 1-30 grid of"
          f" ints 0-9; got {shape}",
          None,
      )
    return f"[answer accepted: {len(grid)}x{len(grid[0])} grid]", grid

  def salvage_answer(
      self, max_functions: int = 8
  ) -> tuple[list[list[int]] | None, str | None]:
    """Exact `salvage_answer` from `havu/evals/arc_harness/executor.py:L480-L548` (with `max_output` lifted during marker read)."""
    marker = f"ARC_SALVAGE_{uuid.uuid4().hex}:"
    inputs = [pair["input"] for pair in self.train] + [self.test_input]
    cell = f"""
def _arc_salvage():
    import contextlib, copy, inspect, io, json, signal, types
    import numpy as _np
    def _alarm(s, f): raise TimeoutError()
    old_h = signal.signal(signal.SIGALRM, _alarm)
    cands = []
    for name, obj in list(globals().items()):
        if name.startswith('_') or not isinstance(obj, types.FunctionType):
            continue
        try:
            params = [p for p in inspect.signature(obj).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            req = [p for p in params if p.default is p.empty]
            if len(req) == 1 or (len(req) == 0 and len(params) >= 1):
                cands.append(name)
        except Exception:
            pass
    cands = [n for n in cands if n != 'solve'][{-max_functions}:]
    if 'solve' in globals() and callable(globals()['solve']):
        cands.append('solve')
    inps = json.loads({json.dumps(inputs)!r})
    out = {{}}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for name in reversed(cands):
                fn = globals()[name]
                for as_np in (False, True):
                    preds = []
                    for g in inps:
                        signal.setitimer(signal.ITIMER_REAL, 2.0)
                        try:
                            arg = _np.array(g) if as_np else copy.deepcopy(g)
                            v = fn(arg)
                            if hasattr(v, 'tolist'): v = v.tolist()
                            preds.append(json.loads(json.dumps(v, default=int)))
                        except BaseException:
                            preds.append(None)
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                    if preds[-1] is not None:
                        out[name + ('(np)' if as_np else '')] = preds
                        break
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_h)
    print({marker!r} + json.dumps(out))
try:
    _arc_salvage()
finally:
    globals().pop('_arc_salvage', None)
"""
    old_max = self.max_output
    self.max_output = max(old_max, 5_000_000)
    try:
      res = self.execute(cell, timeout=15.0)
    finally:
      self.max_output = old_max
    if res["process_status"] != "completed":
      return None, None
    data = None
    for line in reversed(res.get("stdout", "").splitlines()):
      if line.startswith(marker):
        try:
          data = json.loads(line[len(marker) :])
        except json.JSONDecodeError:
          pass
        break
    if not isinstance(data, dict):
      return None, None
    best_score = 0
    best_grid: list[list[int]] | None = None
    best_name: str | None = None
    for name, preds in data.items():
      if not isinstance(preds, list) or len(preds) != len(inputs):
        continue
      test_pred = preds[-1]
      if not is_valid_havu_grid(test_pred):
        continue
      score = sum(
          p == pair["output"] for p, pair in zip(preds[:-1], self.train)
      )
      if score > best_score:
        best_score, best_grid, best_name = score, test_pred, name
    return best_grid, best_name

  def _run_marked(
      self, body: str, payload: Any, timeout: float = 15.0
  ) -> Any:
    """Exact `_run_marked` from `havu/evals/train_check.py:L149-L185`."""
    marker = f"ARC_TC_{uuid.uuid4().hex}:"
    cell = (
        "def _arc_tc():\n"
        f"    import ast, contextlib, copy, hashlib, inspect, io, json, signal, types\n"
        f"    import numpy as _np\n"
        f"    _payload = json.loads({json.dumps(payload)!r})\n"
        f"    _g = globals()\n"
        + "\n".join("    " + line for line in body.strip("\n").splitlines())
        + f"\n    print({marker!r} + json.dumps(_out))\n"
        "try:\n    _arc_tc()\nfinally:\n    globals().pop('_arc_tc', None)\n"
    )
    old_max = self.max_output
    self.max_output = max(old_max, 5_000_000)
    try:
      res = self.execute(cell, timeout=timeout)
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    finally:
      self.max_output = old_max
    for line in reversed(res.get("stdout", "").splitlines()):
      if line.startswith(marker):
        try:
          return json.loads(line[len(marker) :])
        except json.JSONDecodeError:
          return None
    return None

  def run_train_check(
      self,
      pred: list[list[int]] | None,
      code: str | None,
      per_call_timeout: float = 3.0,
  ) -> dict[str, Any]:
    """Exact zero-token host `train_check` from `havu/evals/train_check.py:L15-L146`."""
    t0 = time.monotonic()
    if pred is None or not is_valid_havu_grid(pred):
      return {
          "train_pass": None,
          "train_frac": None,
          "train_trace": "no_answer",
          "train_paths": [],
          "train_check_seconds": 0.0,
      }
    outs = [_grid_hash(p["output"]) for p in self.train]
    n = len(outs)
    ph = _grid_hash(pred)
    inputs = [p["input"] for p in self.train] + [self.test_input]
    cells = [c for name, c in self.executed if name == "run_python"]

    if code is None:
      for name, c in reversed(self.executed):
        if name == "submit_answer":
          code = c
          break

    body = r"""
def _alarm(s, f): raise TimeoutError()
def _valid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30
            and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))
def _hash(v):
    if hasattr(v, 'tolist'): v = v.tolist()
    v = json.loads(json.dumps(v, default=int))
    return hashlib.sha1(json.dumps(v, separators=(',', ':')).encode()).hexdigest()[:16] if _valid(v) else 'INVALID'
def _eval_with(fn, sec):
    signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return _hash(fn())
    except BaseException as e:
        return 'ERR:' + type(e).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

inps = _payload['inputs']
code = _payload['code']
cells = _payload['cells']
ph = _payload['ph']
sec = _payload['sec']
_out = {}
old_h = signal.signal(signal.SIGALRM, _alarm)
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if code is not None:
            try:
                c, mode = compile(code, '<tc>', 'eval'), 'eval'
            except SyntaxError:
                try:
                    c, mode = compile(code, '<tc>', 'exec'), 'exec'
                except SyntaxError:
                    c, mode = None, None
            if c is not None and 'test_input' in c.co_names:
                preds = []
                for g in inps:
                    def _run(g=g):
                        env = {**_g, 'test_input': copy.deepcopy(g)}
                        if mode == 'eval':
                            return eval(c, env)
                        exec(c, env)
                        return env['answer']
                    preds.append(_eval_with(_run, sec))
                _out['expr'] = preds

        expr_ok = _out.get('expr', [None])[-1] == ph
        if not expr_ok:
            vars_matching = []
            for k, v in list(_g.items()):
                if k.startswith('_') or isinstance(v, (types.FunctionType, types.ModuleType, type)):
                    continue
                if isinstance(v, (list, _np.ndarray)):
                    try:
                        if _hash(v) == ph:
                            vars_matching.append(k)
                    except Exception:
                        pass
            for var in sorted(vars_matching, key=lambda x: x != 'answer')[:3]:
                for src in reversed(cells):
                    try:
                        tree = ast.parse(src)
                        stores = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                        loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                    except SyntaxError:
                        stores, loads = set(), set()
                    if var in stores:
                        if 'test_input' in loads:
                            c = compile(src, '<tc_cell>', 'exec')
                            preds = []
                            for g in [inps[-1]] + inps[:-1]:
                                def _run_c(g=g, var=var, c=c):
                                    env = {**_g, 'test_input': copy.deepcopy(g)}
                                    env.pop(var, None)
                                    exec(c, env)
                                    return env[var]
                                preds.append(_eval_with(_run_c, sec))
                                if len(preds) == 1 and preds[0] != ph:
                                    break
                            if preds and preds[0] == ph:
                                _out['cell'] = {'var': var, 'preds': preds[1:] + preds[:1]}
                        break
                if 'cell' in _out:
                    break

            fns = [k for k, v in _g.items() if isinstance(v, types.FunctionType) and not k.startswith('_')]
            fn_hits = {}
            for name in fns[-25:]:
                fn = _g[name]
                preds = []
                for g in [inps[-1]] + inps[:-1]:
                    h = _eval_with(lambda g=g, fn=fn: fn(copy.deepcopy(g)), sec)
                    if h.startswith('ERR:'):
                        h2 = _eval_with(lambda g=g, fn=fn: fn(_np.array(g)), sec)
                        if not h2.startswith('ERR:'):
                            h = h2
                    preds.append(h)
                    if len(preds) == 1 and h != ph:
                        break
                if preds and preds[0] == ph:
                    fn_hits[name] = preds[1:] + preds[:1]
            if fn_hits:
                _out['fns'] = fn_hits
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old_h)
"""
    data = self._run_marked(
        body,
        {
            "inputs": inputs,
            "code": code,
            "cells": cells,
            "ph": ph,
            "sec": per_call_timeout,
        },
        timeout=max(20.0, per_call_timeout * (n + 2) * 3),
    )
    elapsed = round(time.monotonic() - t0, 2)
    if not isinstance(data, dict):
      return {
          "train_pass": None,
          "train_frac": None,
          "train_trace": "check_error",
          "train_paths": [],
          "train_check_seconds": elapsed,
      }

    def _score(preds_list: list[str]) -> int:
      return sum(a == b for a, b in zip(preds_list[:-1], outs))

    paths: list[tuple[int, int, str]] = []
    if "expr" in data and data["expr"][-1] == ph:
      paths.append((_score(data["expr"]), 3, "expr"))
    if "cell" in data and data["cell"]["preds"][-1] == ph:
      paths.append((_score(data["cell"]["preds"]), 2, "cell"))
    if "fns" in data:
      best_fn = max(_score(p) for p in data["fns"].values())
      paths.append((best_fn, 1, "fn"))

    if paths:
      k, _, trace = max(paths)
      return {
          "train_pass": k == n,
          "train_frac": round(k / max(1, n), 4),
          "train_trace": trace,
          "train_paths": [p[2] for p in paths],
          "train_check_seconds": elapsed,
      }
    return {
        "train_pass": None,
        "train_frac": None,
        "train_trace": "untraceable",
        "train_paths": [],
        "train_check_seconds": elapsed,
    }

  def close(self) -> None:
    if self._pid is not None:
      self._kill_group(signal.SIGKILL)
      self._pid = None
    if self._file is not None:
      with contextlib.suppress(Exception):
        self._file.close()
      self._file = None
    if self._sock is not None:
      with contextlib.suppress(Exception):
        self._sock.close()
      self._sock = None


# ==============================================================================
# 4. Upgraded `NativeDeepSeekV41FullPrecisionLLM` (48K Context + Stochastic Gumbel-Max Sampling)
# ==============================================================================


class NativeDeepSeekV41FullPrecisionLLM(ds_traces.DeepSeekV41FullPrecisionLLM):
  """Full-Precision DeepSeek-V4.1-Flash-1017 TPU Engine with 48K+ context and stochastic top-k/top-p Gumbel-max sampling."""

  def __init__(
      self,
      raw_checkpoint_dir: str,
      ocdbt_tp8_dir: str,
      ocdbt_tp16_dir: str,
      max_seq_len: int = 49152,
      batch_size: int = 2,
      steps_per_call: int = 12,
      temperature: float = 1.0,
      top_k: int = 20,
  ):
    t0 = time.time()
    devs = sorted(jax.devices(), key=ds_traces.layout_lib.rank_sort_key)
    n_devs = len(devs)
    if n_devs not in (8, 16):
      raise ValueError(
          f"NativeDeepSeekV41FullPrecisionLLM requires 8 or 16 TensorCores, got {n_devs}"
      )
    self.mesh = ds_traces.Mesh(
        np.asarray(devs[:n_devs]), (ds_traces.layout_lib.AXIS,)
    )
    self.rep = ds_traces.NamedSharding(self.mesh, ds_traces.P())
    self.tp = ds_traces.NamedSharding(
        self.mesh, ds_traces.P(ds_traces.layout_lib.AXIS)
    )

    cfg_path = os.path.join(raw_checkpoint_dir, "config.json")
    with gfile.Open(cfg_path, "r") as f:
      self.cfg = ds_traces.ref.Ds4Config.from_dict(json.load(f))
    self.lay = ds_traces.layout_lib.build_layout(self.cfg, num_devices=n_devs)

    tok_path = os.path.join(raw_checkpoint_dir, "tokenizer.json")
    with gfile.Open(tok_path, "r") as f:
      raw_tok = f.read()
    self._raw_tokenizer = ds_traces.tokenizers.Tokenizer.from_str(raw_tok)
    self._tokenizer = ds_traces.HFTokenizerWrapper(self._raw_tokenizer)

    export_dir = ocdbt_tp8_dir if n_devs == 8 else ocdbt_tp16_dir
    print(
        f"[DS4.1-FP] Loading 71 pre-sliced full-precision weight leaves from {export_dir}"
        f" onto {n_devs} TensorCores...",
        flush=True,
    )
    sharded_w = self._load_ocdbt_weights_via_tensorstore(export_dir)
    print(
        f"[DS4.1-FP] Loaded 71 weight leaves onto {n_devs} TensorCores in {time.time() - t0:.1f}s!",
        flush=True,
    )

    self.max_seq_len = max(4096, (min(max_seq_len, 65536) // 512) * 512)
    self.batch_size = batch_size
    self.steps_per_call = steps_per_call
    self.temperature = float(temperature)
    self.top_k = int(top_k)

    t_comp0 = time.time()
    aux_np, _ = engine_lib.aux_table_arrays(self.cfg, self.max_seq_len)

    def _stack(arr: np.ndarray) -> jax.Array:
      return jax.jit(
          lambda a: jnp.broadcast_to(a[None], (n_devs, *a.shape)),
          out_shardings=self.tp,
      )(jax.device_put(arr, self.rep))

    ds4_vars = dict(sharded_w)
    ds4_vars.update({k: _stack(v) for k, v in aux_np.items()})
    self.variables = {"params": {"ds4": ds4_vars}}

    prefill_fn, generate_fn, insert_fn, state_init_fn, _, _ = (
        engine_lib.make_ds4_engine_functions(
            self.mesh,
            self.lay,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            steps_per_call=self.steps_per_call,
            temperature=self.temperature,
            top_k=self.top_k,
        )
    )

    class _GenMod(engine_lib.nn.Module):

      @engine_lib.nn.compact
      def __call__(self, t, st, fn=generate_fn):
        return fn(t, st)

    class _PrefMod(engine_lib.nn.Module):

      @engine_lib.nn.compact
      def __call__(self, pr, seed, fn=prefill_fn):
        return fn(pr, per_request_hyperparams=seed)

    class _PrefDeltaMod(engine_lib.nn.Module):

      @engine_lib.nn.compact
      def __call__(self, pr, prev_st, seed, hist_len, fn=prefill_fn):
        return fn(
            pr,
            state=prev_st,
            per_request_hyperparams=seed,
            history_length=hist_len,
        )

    self._gen_jit = jax.jit(_GenMod().apply, donate_argnums=(2,))
    self._pref_jit = jax.jit(_PrefMod().apply)
    self._pref_delta_jit = jax.jit(_PrefDeltaMod().apply, donate_argnums=(2,))
    self._ins_jit = jax.jit(insert_fn, donate_argnums=(0,))
    self._state_init_jit = jax.jit(
        lambda: state_init_fn(self.batch_size), out_shardings=self.rep
    )
    self._prefix_cache: list[tuple[list[int], Any]] = []
    print(
        f"[DS4.1-FP] Compiled static-shape Two-Phase CED Prefill + Delta KV Cache + Batched DSpark Speculative Decode "
        f"(batch_size={self.batch_size}, steps_per_call={self.steps_per_call}, max_seq_len={self.max_seq_len}, "
        f"temperature={self.temperature}, top_k={self.top_k}) in {time.time() - t_comp0:.1f}s.",
        flush=True,
    )

  def clear_prefix_cache(self) -> None:
    self._prefix_cache.clear()

  def generate(
      self,
      prompts: Sequence[Any],
      sampling_params: Sequence[Any] | Any,
      use_tqdm: bool = False,
  ) -> list[ds_traces.CompatRequestOutput]:
    if not isinstance(sampling_params, Sequence):
      params_list = [sampling_params] * len(prompts)
    else:
      params_list = list(sampling_params)
    norm_prompts: list[str | list[int]] = [
        p["prompt"]
        if isinstance(p, dict)
        else (list(p) if isinstance(p, list) else str(p))
        for p in prompts
    ]
    outs: list[ds_traces.CompatRequestOutput] = []
    rng = range(0, len(norm_prompts), self.batch_size)
    if use_tqdm:
      rng = tqdm.tqdm(rng, desc="[DS4.1-FP] Batched Generate", leave=False)
    for start in rng:
      sub_p = norm_prompts[start : start + self.batch_size]
      sub_sp = params_list[start : start + self.batch_size]
      outs.extend(self._generate_subbatch(sub_p, sub_sp))
    return outs

  def _generate_subbatch(
      self,
      sub_prompts: list[str | list[int]],
      sub_params: list[ds_traces.CompatSamplingParams],
  ) -> list[ds_traces.CompatRequestOutput]:
    b_active = len(sub_prompts)
    dec_state = self._state_init_jit()
    prompt_token_lists: list[list[int]] = []
    emitted_lists: list[list[int]] = [[] for _ in range(b_active)]
    finish_reasons: list[str] = ["length" for _ in range(b_active)]
    slot_done: list[bool] = [False for _ in range(b_active)]
    eos_ids = {self.cfg.eos_token_id, self.cfg.pad_token_id}
    effective_max_tokens: list[int] = []
    next_prefix_cache: list[tuple[list[int], Any]] = []

    for slot_idx, (p_item, sp) in enumerate(zip(sub_prompts, sub_params)):
      p_ids = (
          self._tokenizer.encode(p_item)
          if isinstance(p_item, str)
          else list(p_item)
      )
      if not p_ids or p_ids[0] != self.cfg.bos_token_id:
        p_ids = [self.cfg.bos_token_id] + p_ids
      if len(p_ids) > self.max_seq_len - 1024:
        max_prompt = self.max_seq_len - 2048
        keep_head = int(max_prompt * 0.45)
        keep_tail = max_prompt - keep_head
        p_ids = p_ids[:keep_head] + p_ids[-keep_tail:]
      prompt_token_lists.append(p_ids)
      avail_decode = max(128, self.max_seq_len - len(p_ids) - 64)
      effective_max_tokens.append(min(sp.max_tokens, avail_decode))

      seed_val = int(getattr(sp, "seed", 0) or (slot_idx + 1))
      seed_dev = jax.device_put(jnp.int32(seed_val), self.rep)

      matched_idx: int | None = None
      best_len = 0
      for c_idx, (c_ids, _) in enumerate(self._prefix_cache):
        c_len = len(c_ids)
        if c_len > best_len and len(p_ids) > c_len and p_ids[:c_len] == c_ids:
          matched_idx = c_idx
          best_len = c_len

      if matched_idx is not None:
        c_ids, c_pst = self._prefix_cache.pop(matched_idx)
        hist_len = len(c_ids)
        delta_ids = p_ids[hist_len:]
        padded_np = np.full(
            (self.max_seq_len,), engine_lib.DS4_PAD_ID, dtype=np.int32
        )
        padded_np[: len(delta_ids)] = np.asarray(delta_ids, dtype=np.int32)
        prompt_tuple = (
            jax.device_put(jnp.asarray(padded_np, dtype=np.int32), self.rep),
            jax.device_put(jnp.int32(len(delta_ids)), self.rep),
        )
        hist_dev = jax.device_put(jnp.int32(hist_len), self.rep)
        _, _, pst = self._pref_delta_jit(
            self.variables, prompt_tuple, c_pst, seed_dev, hist_dev
        )
      else:
        padded_np = np.full(
            (self.max_seq_len,), engine_lib.DS4_PAD_ID, dtype=np.int32
        )
        padded_np[: len(p_ids)] = np.asarray(p_ids, dtype=np.int32)
        prompt_tuple = (
            jax.device_put(jnp.asarray(padded_np, dtype=np.int32), self.rep),
            jax.device_put(jnp.int32(len(p_ids)), self.rep),
        )
        _, _, pst = self._pref_jit(self.variables, prompt_tuple, seed_dev)

      dec_state = self._ins_jit(dec_state, pst, jnp.int32(slot_idx))
      next_prefix_cache.append((p_ids, pst))

    self._prefix_cache.clear()
    self._prefix_cache = next_prefix_cache

    first_toks_np = np.asarray(dec_state.value["token"], dtype=np.int32)
    for slot_idx in range(b_active):
      ft = int(first_toks_np[slot_idx])
      if ft in eos_ids:
        finish_reasons[slot_idx] = "stop"
        slot_done[slot_idx] = True
      else:
        emitted_lists[slot_idx].append(ft)

    while not all(slot_done):
      seed_np = np.full(
          (self.batch_size, 1), engine_lib.DS4_PAD_ID, dtype=np.int32
      )
      for slot_idx in range(b_active):
        if not slot_done[slot_idx]:
          seed_np[slot_idx, 0] = emitted_lists[slot_idx][-1]
      seed_dev = jax.device_put(jnp.asarray(seed_np, dtype=np.int32), self.rep)
      step_out, dec_state = self._gen_jit(self.variables, seed_dev, dec_state)
      emitted_bs_np = np.asarray(step_out.emitted, dtype=np.int32)
      emit_cnt_np = np.asarray(step_out.emit_count, dtype=np.int32)
      pos_np = np.asarray(dec_state.value["position"], dtype=np.int32)

      for slot_idx in range(b_active):
        if slot_done[slot_idx]:
          continue
        sp = sub_params[slot_idx]
        slot_max_toks = effective_max_tokens[slot_idx]
        cnt = int(emit_cnt_np[slot_idx])
        row_toks = [int(x) for x in emitted_bs_np[slot_idx, 1:cnt]]
        row_toks.append(
            int(np.asarray(step_out.last_token, dtype=np.int32)[slot_idx])
        )
        for tv in row_toks:
          if tv in eos_ids:
            finish_reasons[slot_idx] = "stop"
            slot_done[slot_idx] = True
            break
          if len(emitted_lists[slot_idx]) < slot_max_toks:
            emitted_lists[slot_idx].append(tv)
          if len(emitted_lists[slot_idx]) >= slot_max_toks:
            finish_reasons[slot_idx] = "length"
            slot_done[slot_idx] = True
            break

        if (
            not slot_done[slot_idx]
            and int(pos_np[slot_idx]) + 32 >= self.max_seq_len
        ):
          finish_reasons[slot_idx] = "length"
          slot_done[slot_idx] = True

        if not slot_done[slot_idx] and sp.stop:
          tail_txt = self._tokenizer.decode(emitted_lists[slot_idx][-96:])
          if any(st in tail_txt for st in sp.stop):
            finish_reasons[slot_idx] = "stop"
            slot_done[slot_idx] = True

      if all(
          len(emitted_lists[i]) >= effective_max_tokens[i]
          for i in range(b_active)
      ):
        break

    results: list[ds_traces.CompatRequestOutput] = []
    for slot_idx in range(b_active):
      sp = sub_params[slot_idx]
      toks = list(emitted_lists[slot_idx])
      txt = self._tokenizer.decode(toks)
      orig_txt = txt
      f_reason = finish_reasons[slot_idx]
      if sp.stop:
        earliest_pos: int | None = None
        matched_st: str | None = None
        for st in sp.stop:
          idx = txt.find(st)
          if idx != -1 and (earliest_pos is None or idx < earliest_pos):
            earliest_pos = idx
            matched_st = st
        if earliest_pos is not None and matched_st is not None:
          if matched_st in DS41_KEEP_STOP_SUFFIXES:
            txt = txt[: earliest_pos + len(matched_st)]
          else:
            txt = txt[:earliest_pos]
          f_reason = "stop"
          # Only trim trailing tokens if the stop string actually shortened `txt`
          # so exact sampled token_ids (`--token-exact`) are preserved without re-tokenization!
          if len(txt) < len(orig_txt):
            while toks and len(self._tokenizer.decode(toks)) > len(txt):
              toks.pop()
            dec_prefix = self._tokenizer.decode(toks)
            if len(dec_prefix) < len(txt):
              rem_ids = self._tokenizer.encode(txt[len(dec_prefix) :])
              if rem_ids and rem_ids[0] == self.cfg.bos_token_id:
                rem_ids = rem_ids[1:]
              toks.extend(rem_ids)
      results.append(
          ds_traces.CompatRequestOutput(
              prompt_token_ids=prompt_token_lists[slot_idx],
              outputs=[
                  ds_traces.CompatCompletionOutput(
                      text=txt,
                      token_ids=toks,
                      finish_reason=f_reason,
                  )
              ],
          )
      )
    return results


# ==============================================================================
# 5. Collaborator Vu (`havu` `E030` + `E043`) Multi-Turn Execution Loop
# ==============================================================================


class HavuAttemptRuntime:
  """Tracks one `havu`-style multi-turn trajectory alongside `AttemptState` for CNS logging."""

  def __init__(
      self,
      state: runner_lib.AttemptState,
      messages: list[dict[str, Any]],
      reasoning_effort: int,
      session: HavuSubprocessSession,
      base_seed: int,
  ):
    self.state = state
    self.messages = messages
    self.reasoning_effort = reasoning_effort
    self.session = session
    self.base_seed = int(base_seed)
    self.accepted_grid: list[list[int]] | None = None
    self.answer_source: str | None = None
    self.stop_reason: str | None = None
    self.submit_code: str | None = None
    self.train_labels: dict[str, Any] = {}
    self.tool_errors: int = 0
    self.cap_nudges: int = 0
    self.code_rescues: int = 0
    self.nudge_rescues: int = 0
    self.exact_token_ids: list[int] | None = None
    self.exact_upto_msg_count: int = 0
    self.exact_resync: bool = False
    self.token_exact_breaks_saved: int = 0


def compact_havu_messages(
    messages: list[dict[str, Any]],
    keep_recent_assistant_turns: int = 2,
    old_tool_chars: int = 1500,
) -> list[dict[str, Any]]:
  """Exact non-destructive transcript compaction from Vu's `agent_qwen.py:compact(keep=2, tool_chars=1500)`.

  NEVER deletes `(assistant, tool)` turns (`del work_msgs[2:4]`), so the model never forgets
  variables/functions already defined in the persistent IPython session!
  Instead:
  - Empties `reasoning_content` on assistant turns older than the last `keep_recent_assistant_turns` turns.
  - Truncates `tool` outputs older than those turns to `old_tool_chars` characters (`[... N characters cut]`).
  """
  work_msgs = copy.deepcopy(messages)
  asst_indices = [
      i for i, m in enumerate(work_msgs) if m.get("role") == "assistant"
  ]
  if len(asst_indices) <= keep_recent_assistant_turns:
    return work_msgs
  cutoff_idx = (
      asst_indices[-keep_recent_assistant_turns]
      if keep_recent_assistant_turns > 0
      else len(work_msgs)
  )
  for i in range(2, cutoff_idx):
    m = work_msgs[i]
    role = m.get("role")
    if role == "assistant" and m.get("reasoning_content"):
      m["reasoning_content"] = ""
    elif role == "tool":
      txt = m.get("content", "")
      if isinstance(txt, str) and len(txt) > old_tool_chars:
        cut_n = len(txt) - old_tool_chars
        m["content"] = (
            txt[:old_tool_chars] + f"\n[... {cut_n} characters cut]"
        )
  return work_msgs


def encode_havu_messages_with_fit(
    messages: list[dict[str, Any]],
    reasoning_effort: int,
    tokenizer: Any,
    max_seq_len: int,
    compact_at_tokens: int = 60000,
    min_turn_headroom: int = 6144,
    drop_prior_thinking: bool = False,
) -> tuple[str, bool]:
  """Encodes `messages` via `ds4_encoding.encode_messages(..., thinking_mode='thinking')` with Vu's non-destructive `compact()`."""
  work_msgs = copy.deepcopy(messages)
  did_compact = False
  if drop_prior_thinking:
    for m in work_msgs[:-1]:
      if m.get("role") == "assistant":
        m["reasoning_content"] = ""
    did_compact = True

  prompt_str = ds4_encoding.encode_messages(
      work_msgs,
      thinking_mode="thinking",
      drop_thinking=False,
      reasoning_effort=reasoning_effort,
  )
  n_tok = len(tokenizer.encode(prompt_str))
  max_prompt_toks = min(compact_at_tokens, max_seq_len - min_turn_headroom)

  if n_tok > max_prompt_toks:
    # Tier 1: Vu's `compact(keep=2, tool_chars=1500)` persisted in-place so prefix KV cache stays valid
    work_msgs = compact_havu_messages(
        work_msgs, keep_recent_assistant_turns=2, old_tool_chars=1500
    )
    messages[:] = copy.deepcopy(work_msgs)
    did_compact = True
    prompt_str = ds4_encoding.encode_messages(
        work_msgs,
        thinking_mode="thinking",
        drop_thinking=False,
        reasoning_effort=reasoning_effort,
    )
    n_tok = len(tokenizer.encode(prompt_str))

  if n_tok > max_seq_len - min_turn_headroom:
    # Tier 2: `compact(keep=1, tool_chars=800)`
    work_msgs = compact_havu_messages(
        work_msgs, keep_recent_assistant_turns=1, old_tool_chars=800
    )
    messages[:] = copy.deepcopy(work_msgs)
    did_compact = True
    prompt_str = ds4_encoding.encode_messages(
        work_msgs,
        thinking_mode="thinking",
        drop_thinking=False,
        reasoning_effort=reasoning_effort,
    )
    n_tok = len(tokenizer.encode(prompt_str))

  if n_tok > max_seq_len - min_turn_headroom:
    # Tier 3: `compact(keep=0, tool_chars=500)` — still preserves 100% of code cells!
    work_msgs = compact_havu_messages(
        work_msgs, keep_recent_assistant_turns=0, old_tool_chars=500
    )
    messages[:] = copy.deepcopy(work_msgs)
    did_compact = True
    prompt_str = ds4_encoding.encode_messages(
        work_msgs,
        thinking_mode="thinking",
        drop_thinking=False,
        reasoning_effort=reasoning_effort,
    )

  return prompt_str, did_compact


def run_havu_attempt_wave_batched(
    llm: Any,
    vllm_mod: Any,
    runtimes: list[HavuAttemptRuntime],
    wave_desc: str,
) -> tuple[int, float]:
  """Runs a batch of `HavuAttemptRuntime` trajectories replicating `havu/evals/run_deepseek_tools.py:solve`."""
  if not runtimes:
    return 0, 0.0
  if hasattr(llm, "clear_prefix_cache"):
    llm.clear_prefix_cache()

  total_wave_gen_toks = 0
  total_wave_gen_s = 0.0
  tokenizer = llm.get_tokenizer()
  bos_id = int(getattr(getattr(llm, "cfg", None), "bos_token_id", 0))
  eos_id = int(getattr(getattr(llm, "cfg", None), "eos_token_id", 1))

  def _count_sub_tokens(txt: str) -> int:
    if not txt:
      return 0
    try:
      ids = tokenizer.encode(txt)
      if ids and ids[0] == bos_id:
        ids = ids[1:]
      return len(ids)
    except Exception:  # pylint: disable=broad-exception-caught
      return max(1, len(txt) // 3)

  turn_pbar = tqdm.tqdm(
      total=FLAGS.max_turns,
      desc=f"  {wave_desc} turns",
      leave=False,
      dynamic_ncols=True,
  )

  for turn_idx in range(FLAGS.max_turns):
    active_rts = [rt for rt in runtimes if not rt.state.done]
    if not active_rts:
      break

    prompts: list[str | list[int]] = []
    params_list: list[Any] = []
    sched_rts: list[HavuAttemptRuntime] = []

    for rt in active_rts:
      st = rt.state
      rem_output = st.hard_budget - st.output_tokens
      rem_reasoning = (
          FLAGS.max_reasoning_tokens - st.thinking_tokens
          if FLAGS.max_reasoning_tokens > 0
          else rem_output
      )
      remaining_budget = min(rem_output, rem_reasoning)
      if remaining_budget <= 0:
        rt.stop_reason = (
            "reasoning_budget" if rem_reasoning <= 0 else "output_budget"
        )
        st.done = True
        continue

      prompt_str, did_compact = encode_havu_messages_with_fit(
          messages=rt.messages,
          reasoning_effort=rt.reasoning_effort,
          tokenizer=tokenizer,
          max_seq_len=FLAGS.max_model_len,
          compact_at_tokens=FLAGS.compact_at_tokens,
          min_turn_headroom=min(8192, max(2048, remaining_budget)),
          drop_prior_thinking=FLAGS.drop_prior_turn_thinking,
      )
      st.transcript = prompt_str
      full_ids = tokenizer.encode(prompt_str)
      if not full_ids or full_ids[0] != bos_id:
        full_ids = [bos_id] + full_ids

      prompt_item: str | list[int] = full_ids
      if (
          FLAGS.token_exact
          and not did_compact
          and rt.exact_token_ids is not None
          and not rt.exact_resync
          and 0 < rt.exact_upto_msg_count <= len(rt.messages)
      ):
        upto_str = ds4_encoding.encode_messages(
            rt.messages[: rt.exact_upto_msg_count],
            thinking_mode="thinking",
            drop_thinking=False,
            reasoning_effort=rt.reasoning_effort,
        )
        if prompt_str.startswith(upto_str):
          upto_ids = tokenizer.encode(upto_str)
          if not upto_ids or upto_ids[0] != bos_id:
            upto_ids = [bos_id] + upto_ids
          if full_ids[: len(upto_ids)] == upto_ids:
            if upto_ids != rt.exact_token_ids:
              rt.token_exact_breaks_saved += 1
            prompt_item = list(rt.exact_token_ids) + full_ids[len(upto_ids) :]
          else:
            delta_ids = tokenizer.encode(prompt_str[len(upto_str) :])
            if delta_ids and delta_ids[0] == bos_id:
              delta_ids = delta_ids[1:]
            prompt_item = list(rt.exact_token_ids) + delta_ids

      p_toks = (
          len(prompt_item)
          if isinstance(prompt_item, list)
          else len(full_ids)
      )
      st.max_prompt_tokens = max(st.max_prompt_tokens, p_toks)
      ctx_headroom = FLAGS.max_model_len - p_toks - 128
      if ctx_headroom <= 256:
        st.ctx_cutoff = True
        rt.stop_reason = "ctx_cutoff"
        st.done = True
        continue

      turn_cap = min(FLAGS.max_turn_tokens, remaining_budget, ctx_headroom)
      turn_seed = int((rt.base_seed + (turn_idx + 1) * 104729) & 0x7FFFFFFF)
      sched_rts.append(rt)
      prompts.append(prompt_item)
      params_list.append(
          vllm_mod.SamplingParams(
              temperature=FLAGS.temperature,
              top_p=FLAGS.top_p,
              top_k=FLAGS.top_k,
              repetition_penalty=FLAGS.repetition_penalty,
              max_tokens=turn_cap,
              stop=DS41_STOP_STRINGS,
              seed=turn_seed,
          )
      )

    if not sched_rts:
      break

    t_gen0 = time.perf_counter()
    outputs = llm.generate(prompts, params_list, use_tqdm=False)
    gen_s = time.perf_counter() - t_gen0
    total_wave_gen_s += gen_s

    to_exec: list[tuple[HavuAttemptRuntime, list[dict[str, Any]], str]] = []

    for rt, out, sp_used in zip(sched_rts, outputs, params_list):
      st = rt.state
      gen_out = out.outputs[0]
      raw_text = gen_out.text
      n_tok = len(gen_out.token_ids)
      finish_reason = str(gen_out.finish_reason)
      st.prompt_tokens = len(out.prompt_token_ids)
      st.output_tokens += n_tok
      st.turns += 1
      total_wave_gen_toks += n_tok

      reasoning_content, visible_content, openai_calls, is_canonical = (
          parse_ds41_assistant_turn(
              raw_text, st.turns, finish_reason=finish_reason
          )
      )

      think_toks = min(n_tok, _count_sub_tokens(reasoning_content))
      if FLAGS.max_reasoning_tokens > 0:
        think_toks = min(
            think_toks, max(0, FLAGS.max_reasoning_tokens - st.thinking_tokens)
        )
      if openai_calls:
        tool_toks = max(0, n_tok - think_toks)
        ans_toks = 0
      else:
        tool_toks = 0
        ans_toks = max(0, n_tok - think_toks)

      st.thinking_tokens += think_toks
      st.tool_use_tokens += tool_toks
      st.answer_tokens += ans_toks
      if reasoning_content:
        st.thought_text = (
            f"{st.thought_text}\n\n[Turn {st.turns}]\n{reasoning_content.strip()}"
            if st.thought_text
            else f"[Turn {st.turns}]\n{reasoning_content.strip()}"
        )
      if visible_content:
        st.visible_text = (
            f"{st.visible_text}\n\n{visible_content}"
            if st.visible_text
            else visible_content
        )

      # E043 `--rescue-code`: if no DSML tool call was parsed and no valid grid is in `visible_content`,
      # check if the model wrote Python code in plain text/markdown or needs a tool-use reminder.
      if (
          not openai_calls
          and FLAGS.rescue_code
          and finish_reason == "stop"
          and havu_arcprize_parse(visible_content) is None
      ):
        rescued_call = rescue_havu_tool_call(visible_content, st.turns)
        if rescued_call is not None:
          openai_calls = [rescued_call]
          visible_content = ""
          is_canonical = False
          rt.code_rescues += 1

      st.turn_logs.append({
          "turn": st.turns,
          "output_tokens": n_tok,
          "thinking_tokens": think_toks,
          "tool_use_tokens": tool_toks,
          "answer_tokens": ans_toks,
          "finish_reason": finish_reason,
          "num_tool_calls": len(openai_calls),
          "is_canonical": is_canonical,
      })

      if not openai_calls:
        # 1. Check if visible content contains a valid grid (`parse` or `lenient_parse`)
        cand_grid = havu_arcprize_parse(visible_content)
        if cand_grid is not None:
          rt.messages.append({
              "role": "assistant",
              "content": visible_content,
              "reasoning_content": reasoning_content,
          })
          rt.accepted_grid = cand_grid
          rt.answer_source = "parse"
          rt.stop_reason = "final"
          st.done = True
          continue

        # 2. E043 `--rescue-code` nudge when the model stopped without a tool call, code, or grid (at most 3x)
        if (
            FLAGS.rescue_code
            and finish_reason == "stop"
            and rt.nudge_rescues < 3
            and turn_idx < FLAGS.max_turns - 1
        ):
          rt.code_rescues += 1
          rt.nudge_rescues += 1
          rt.exact_resync = True
          rt.messages.append({
              "role": "assistant",
              "content": visible_content,
              "reasoning_content": reasoning_content,
          })
          rt.messages.append({"role": "user", "content": HAVU_RESCUE_NUDGE})
          continue

        # 3. Exact `havu/evals/run_deepseek_tools.py:L844-L857` (`--cap-nudge`):
        # When a turn hits the per-turn thinking cap (`finish_reason == "length"`) with budget remaining,
        # keep its full `reasoning_content` (no 6,000-char truncation!) and close the turn with a synthetic
        # `run_python` call (`pass  # per-turn thinking limit`) and `HAVU_CAP_NOTE`.
        rem_out_after = st.hard_budget - st.output_tokens
        rem_think_after = (
            FLAGS.max_reasoning_tokens - st.thinking_tokens
            if FLAGS.max_reasoning_tokens > 0
            else rem_out_after
        )
        rem_after = min(rem_out_after, rem_think_after)
        if (
            finish_reason == "length"
            and rem_after > 1024
            and turn_idx < FLAGS.max_turns - 1
        ):
          rt.cap_nudges += 1
          rt.exact_resync = True
          cid = f"cap_{turn_idx}"
          rt.messages.append({
              "role": "assistant",
              "content": "",
              "reasoning_content": reasoning_content,
              "tool_calls": [{
                  "id": cid,
                  "type": "function",
                  "function": {
                      "name": "run_python",
                      "arguments": json.dumps(
                          {"code": "pass  # per-turn thinking limit"}
                      ),
                  },
              }],
          })
          cap_k = max(1, round(sp_used.max_tokens / 1000))
          used_for_note = (
              st.thinking_tokens
              if FLAGS.max_reasoning_tokens > 0
              else st.output_tokens
          )
          limit_for_note = (
              FLAGS.max_reasoning_tokens
              if FLAGS.max_reasoning_tokens > 0
              else st.hard_budget
          )
          note = format_havu_budget_note(
              used_tokens=used_for_note,
              soft_target=st.soft_target,
              hard_limit=limit_for_note,
          ) + format_havu_turns_note(
              turn_index=turn_idx,
              max_turns=FLAGS.max_turns,
              threshold=FLAGS.turns_note_threshold,
          )
          rt.messages.append({
              "role": "tool",
              "tool_call_id": cid,
              "content": HAVU_CAP_NOTE.format(k=cap_k) + note,
          })
          continue

        rt.messages.append({
            "role": "assistant",
            "content": visible_content,
            "reasoning_content": reasoning_content,
        })
        rt.stop_reason = "final" if finish_reason == "stop" else finish_reason
        st.done = True
        continue

      rt.messages.append({
          "role": "assistant",
          "content": visible_content,
          "reasoning_content": reasoning_content,
          "tool_calls": openai_calls,
      })
      if FLAGS.token_exact and is_canonical and finish_reason == "stop":
        rt.exact_token_ids = (
            list(out.prompt_token_ids) + list(gen_out.token_ids) + [eos_id]
        )
        rt.exact_upto_msg_count = len(rt.messages)
        rt.exact_resync = False
      else:
        rt.exact_resync = True
      to_exec.append((rt, openai_calls, finish_reason))

    # Execute tool calls in parallel across isolated child-process sandboxes
    if to_exec:
      cur_turn_idx = turn_idx

      def _exec_rt_tools(
          item: tuple[HavuAttemptRuntime, list[dict[str, Any]], str],
          t_idx: int = cur_turn_idx,
      ) -> None:
        rt, calls, f_reason = item
        st = rt.state
        accepted_grid: list[list[int]] | None = None
        accepted_code: str | None = None

        for call in calls:
          st.tool_calls += 1
          call_id = call["id"]
          try:
            if f_reason == "length":
              raise ValueError("Tool skipped: truncated response")
            args_dict = json.loads(call["function"]["arguments"])
            if not isinstance(args_dict, dict) or not isinstance(
                args_dict.get("code"), str
            ):
              raise ValueError("Expected an object with a string code field")
            fn_name = call["function"]["name"]
            code_str = args_dict["code"]

            if accepted_grid is not None:
              tool_out = "[skipped: solution already accepted]"
            elif fn_name == "run_python":
              status, tool_out = rt.session.run_python(code_str)
              if status != "completed":
                rt.tool_errors += 1
            elif fn_name == "submit_answer":
              tool_out, grid = rt.session.submit_answer(code_str)
              if grid is not None:
                accepted_grid = grid
                accepted_code = code_str
              else:
                rt.tool_errors += 1
            else:
              raise ValueError(f"Unknown tool: {fn_name}")
          except Exception as exc:  # pylint: disable=broad-exception-caught
            tool_out = f"[tool error: {exc}]"
            rt.tool_errors += 1

          rt.messages.append({
              "role": "tool",
              "tool_call_id": call_id,
              "content": tool_out,
          })

        if accepted_grid is not None:
          rt.accepted_grid = accepted_grid
          rt.submit_code = accepted_code
          rt.answer_source = "submit"
          rt.stop_reason = "submitted"
          st.done = True
          return

        used_for_note = (
            st.thinking_tokens
            if FLAGS.max_reasoning_tokens > 0
            else st.output_tokens
        )
        limit_for_note = (
            FLAGS.max_reasoning_tokens
            if FLAGS.max_reasoning_tokens > 0
            else st.hard_budget
        )
        note = format_havu_budget_note(
            used_tokens=used_for_note,
            soft_target=st.soft_target,
            hard_limit=limit_for_note,
        ) + format_havu_turns_note(
            turn_index=t_idx,
            max_turns=FLAGS.max_turns,
            threshold=FLAGS.turns_note_threshold,
        )
        if note and rt.messages and rt.messages[-1].get("role") == "tool":
          rt.messages[-1]["content"] += note

      with futures.ThreadPoolExecutor(
          max_workers=min(32, len(to_exec))
      ) as pool:
        list(pool.map(_exec_rt_tools, to_exec))

    rem_active = sum(1 for rt in runtimes if not rt.state.done)
    submitted_so_far = sum(
        1 for rt in runtimes if rt.accepted_grid is not None
    )
    tok_per_s = total_wave_gen_toks / max(total_wave_gen_s, 1e-6)
    turn_pbar.set_postfix(
        active=f"{rem_active}/{len(runtimes)}",
        submitted=submitted_so_far,
        tok_s=f"{tok_per_s:.0f}",
    )
    turn_pbar.update(1)
    if rem_active == 0:
      break

  turn_pbar.close()

  # Post-loop: 3-tier session fallback (`answer` -> `solve(test_input)` -> `salvage_answer`) + `train_check`
  def _finalize_one(rt: HavuAttemptRuntime) -> None:
    st = rt.state
    if rt.stop_reason is None:
      rt.stop_reason = "max_turns"

    try:
      if rt.accepted_grid is None:
        # 1. Try session `answer` variable
        _, grid = rt.session.submit_answer("answer", record_execution=False)
        if grid is not None:
          rt.accepted_grid = grid
          rt.answer_source = "session"
          rt.submit_code = "answer"
        else:
          # 2. Try session `solve(test_input)` (`--fallback-solve`)
          _, grid = rt.session.submit_answer(
              "solve(test_input)", record_execution=False
          )
          if grid is not None:
            rt.accepted_grid = grid
            rt.answer_source = "session_solve"
            rt.submit_code = "solve(test_input)"
          else:
            # 3. Try Vu's `salvage_answer(session)` (`executor.py:L480-L548`)
            salv_grid, salv_name = rt.session.salvage_answer()
            if salv_grid is not None:
              rt.accepted_grid = salv_grid
              rt.answer_source = f"salvage:{salv_name}"
              fn_base = (salv_name or "solve").replace("(np)", "")
              rt.submit_code = (
                  f"{fn_base}(np.array(test_input))"
                  if (salv_name or "").endswith("(np)")
                  else f"{fn_base}(test_input)"
              )

      # 4. Run zero-token host `train_check` (`train_check.py`) to label `train_pass` and `train_frac`
      rt.train_labels = rt.session.run_train_check(
          rt.accepted_grid, rt.submit_code
      )
      t_pass = rt.train_labels.get("train_pass")
      t_frac = rt.train_labels.get("train_frac")

      if rt.accepted_grid is not None:
        st.sandbox._record_candidate(  # pylint: disable=protected-access
            grid=rt.accepted_grid,
            source="submit_answer",
            train_pass=bool(t_pass) if t_pass is not None else False,
            train_frac=float(t_frac) if t_frac is not None else None,
            cell_frac=float(t_frac) if t_frac is not None else 0.0,
            suspect=False,
            is_primary=True,
            code=rt.submit_code,
        )
        st.sandbox.primary_grid = rt.accepted_grid
        st.sandbox.primary_train_pass = bool(t_pass)
        st.sandbox.primary_train_frac = (
            float(t_frac) if t_frac is not None else None
        )
        st.sandbox.primary_suspect = False
        st.sandbox.train_passed = bool(t_pass is True)
        if t_pass is True:
          st.sandbox.first_verified_grid = rt.accepted_grid
          st.sandbox.last_verified_grid = rt.accepted_grid

      try:
        st.transcript = ds4_encoding.encode_messages(
            rt.messages,
            thinking_mode="thinking",
            drop_thinking=False,
            reasoning_effort=rt.reasoning_effort,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    finally:
      rt.session.close()

  with futures.ThreadPoolExecutor(
      max_workers=min(32, len(runtimes))
  ) as pool:
    list(pool.map(_finalize_one, runtimes))

  return total_wave_gen_toks, total_wave_gen_s


# ==============================================================================
# 6. Summary & Documentation Builders
# ==============================================================================


def _percentile(vals: list[int | float], q: float) -> float:
  if not vals:
    return 0.0
  return round(float(np.percentile(np.asarray(vals, dtype=np.float64), q)), 1)


def build_documented_settings(
    active_model_name: str,
    active_backend: str,
    num_tpus: int,
    device_kind: str,
    total_tasks: int,
    total_pairs: int,
) -> dict[str, Any]:
  """Returns a comprehensive dictionary documenting every model, precision, prompt, sandbox, and harness setting."""
  return {
      "experiment_name": FLAGS.exp_name,
      "model_and_weights": {
          "model_name": active_model_name,
          "engine_backend": active_backend,
          "ds41_raw_checkpoint_dir": FLAGS.ds41_raw_checkpoint_dir,
          "ds41_ocdbt_tp8_dir": FLAGS.ds41_ocdbt_tp8_dir,
          "ds41_ocdbt_tp16_dir": FLAGS.ds41_ocdbt_tp16_dir,
          "architecture": (
              "DeepSeek-V4.1-Flash-1017 (40-layer Causal Encoder-Decoder + "
              "196.6B Engram conditional memory at L1/L14 + 3-layer DSpark 1+5 MTP drafter)"
          ),
      },
      "precision_and_engine": {
          "precision": "full_precision_bf16_dequant_fp32_mxu",
          "weight_dequantization": (
              "FP8 (E4M3FN) and MXFP4 (E2M1) weights dequantized by exact E8M0 block scales into bfloat16"
          ),
          "mxu_accumulation": "IEEE-754 float32 (BF16 x BF16 -> FP32)",
          "prefill_mode": (
              "Two-Phase Causal Encoder-Decoder (CED) + Bounded SWA Replay Prefill "
              "(static 512-token chunked lax.fori_loop)"
          ),
          "decode_mode": (
              "Batched DSpark 1+5 speculative verification + drafting with exact target-side "
              f"stochastic Gumbel-max sampling (temperature={FLAGS.temperature}, top_k={FLAGS.top_k}, top_p=0.95)"
          ),
          "tpu_hardware": f"{num_tpus}x {device_kind}",
          "max_model_len": FLAGS.max_model_len,
      },
      "vu_havu_e030_e043_fixes": {
          "1_isolated_subprocess_ipython_sandbox": (
              "HavuSubprocessSession (LocalExecutor(style='ipython')) runs a dedicated child process per attempt "
              "with IPython.lib.pretty displayhook, IPython plain tracebacks with <ipython-input-N-hash> and "
              "source lines in stdout, <output cut> at 16,000 chars, RLIMIT_AS=4096MB, and two-stage SIGINT -> SIGKILL "
              "timeout recovery that preserves session state on KeyboardInterrupt."
          ),
          "2_clean_e030_minimal_effort_header": (
              "Uses reasoning_effort=0 ('none', matching Vu's REASONING_EFFORT_PROMPTS['low'] == '') so that "
              "HAVU_EFFORT_HEADER_MINIMAL is not preceded by a conflicting 'Reasoning Effort: 50' header."
          ),
          "3_lossless_speculative_gumbel_sampling": (
              f"Uses exact lossless speculative rejection + residual distribution sampling "
              f"(temperature={FLAGS.temperature}, top_k={FLAGS.top_k}, top_p=0.95) in decoder_macrokernel, "
              "removing the sorted_p>=0.55 greedy clamping bug so both Attempt 0 and Attempt 1 sample "
              "losslessly from the target distribution."
          ),
          "4_uncapped_turns_with_e030_sub80s40_budget": (
              f"Restores Vu's E030 sub80s40 budget (total_token_budget={FLAGS.total_token_budget}, "
              f"soft_token_target={FLAGS.soft_token_target}, max_turn_tokens={FLAGS.max_turn_tokens}, "
              f"max_reasoning_tokens={FLAGS.max_reasoning_tokens}) with 65,536 context window "
              f"(compact_at_tokens={FLAGS.compact_at_tokens})."
          ),
          "5_e043_token_exact_and_rescue_code": (
              f"token_exact={FLAGS.token_exact} preserves exact sampled token_ids across turns to eliminate "
              f"re-tokenization breaks; rescue_code={FLAGS.rescue_code} rescues code written outside DSML tags."
          ),
          "6_three_tier_fallback_and_train_check": (
              "Falls back through answer -> solve(test_input) -> salvage_answer(session) and verifies all predictions "
              "with zero-token host train_check (expr, cell, fn paths with 5MB marker buffer)."
          ),
      },
      "sampling_and_budgets": {
          "num_attempts": FLAGS.num_attempts,
          "attempts_per_wave": FLAGS.attempts_per_wave,
          "temperature": FLAGS.temperature,
          "top_p": FLAGS.top_p,
          "top_k": FLAGS.top_k,
          "max_turns": FLAGS.max_turns,
          "max_turn_tokens": FLAGS.max_turn_tokens,
          "max_reasoning_tokens": FLAGS.max_reasoning_tokens,
          "total_token_budget": FLAGS.total_token_budget,
          "soft_token_target": FLAGS.soft_token_target,
          "turns_note_threshold": FLAGS.turns_note_threshold,
          "token_exact": FLAGS.token_exact,
          "rescue_code": FLAGS.rescue_code,
      },
      "dataset_and_sharding": {
          "cns_data_path": FLAGS.cns_data_path,
          "split": "evaluation",
          "total_eval_tasks": total_tasks,
          "total_eval_pairs": total_pairs,
          "num_shards": FLAGS.num_shards,
          "shard_idx": FLAGS.shard_idx,
          "shard_by_task": FLAGS.shard_by_task,
          "pair_batch_size": FLAGS.pair_batch_size,
          "output_dir": FLAGS.output_dir,
      },
  }


def compute_deepseek_eval_summary(
    rows: list[dict[str, Any]],
    total_eval_tasks: int,
    total_eval_pairs: int,
    shard_total_pairs: int,
    shard_done_pairs: int,
    total_gen_tokens: int,
    total_gen_seconds: float,
    wall_elapsed_s: float,
    num_tpus: int,
    device_kind: str,
    active_model_name: str,
    active_backend: str,
    documented_settings: dict[str, Any],
) -> dict[str, Any]:
  """Computes Pass@1, Pass@2 (pair & official task-weighted), reasoning token stats, and live ETA."""
  base = runner_lib.compute_summary(
      rows=rows,
      total_gen_tokens=total_gen_tokens,
      total_gen_seconds=total_gen_seconds,
      wall_elapsed_s=wall_elapsed_s,
      num_tpus=num_tpus,
      device_kind=device_kind,
  )
  all_atts = [a for r in rows for a in r.get("attempts", [])]
  think_list = [int(a.get("thinking_tokens", 0)) for a in all_atts]
  out_list = [int(a.get("output_tokens", 0)) for a in all_atts]
  tool_list = [int(a.get("tool_use_tokens", 0)) for a in all_atts]
  ans_list = [int(a.get("answer_tokens", 0)) for a in all_atts]

  att0_matches = [
      bool(a.get("exact_match"))
      for r in rows
      for a in r.get("attempts", [])
      if a.get("attempt_idx") == 0
  ]
  att1_matches = [
      bool(a.get("exact_match"))
      for r in rows
      for a in r.get("attempts", [])
      if a.get("attempt_idx") == 1
  ]
  att0_p1 = (
      round(100.0 * sum(att0_matches) / len(att0_matches), 2)
      if att0_matches
      else 0.0
  )
  att1_p1 = (
      round(100.0 * sum(att1_matches) / len(att1_matches), 2)
      if att1_matches
      else 0.0
  )
  unranked_matches = att0_matches + att1_matches
  collab_unranked_p1 = (
      round(100.0 * sum(unranked_matches) / len(unranked_matches), 2)
      if unranked_matches
      else 0.0
  )

  pair_think_totals = [
      sum(int(a.get("thinking_tokens", 0)) for a in r.get("attempts", []))
      for r in rows
  ]
  pair_out_totals = [
      sum(int(a.get("output_tokens", 0)) for a in r.get("attempts", []))
      for r in rows
  ]

  rem_shard_pairs = max(0, shard_total_pairs - shard_done_pairs)
  sec_per_pair = wall_elapsed_s / max(1, shard_done_pairs)
  eta_seconds = rem_shard_pairs * sec_per_pair

  base.update({
      "model": active_model_name,
      "engine_backend": active_backend,
      "precision": "full_precision_bf16_dequant_fp32_mxu",
      "reasoning_effort": 0,
      "collaborator_unranked_kaggle_pass_at_1_pct": collab_unranked_p1,
      "attempt_0_kaggle_pass_at_1_pct": att0_p1,
      "attempt_1_kaggle_pass_at_1_pct": att1_p1,
      "total_eval_tasks_in_benchmark": total_eval_tasks,
      "total_eval_pairs_in_benchmark": total_eval_pairs,
      "shard_idx": FLAGS.shard_idx,
      "num_shards": FLAGS.num_shards,
      "shard_pairs_completed": shard_done_pairs,
      "shard_pairs_total": shard_total_pairs,
      "shard_eta_seconds": round(eta_seconds, 1),
      "shard_eta_minutes": round(eta_seconds / 60.0, 2),
      "reasoning_token_stats": {
          "max_reasoning_tokens_cap": FLAGS.max_reasoning_tokens,
          "total_thinking_tokens": int(sum(think_list)),
          "total_output_tokens": int(sum(out_list)),
          "total_tool_use_tokens": int(sum(tool_list)),
          "total_answer_tokens": int(sum(ans_list)),
          "thinking_fraction_of_output_pct": round(
              100.0 * sum(think_list) / max(1, sum(out_list)), 2
          ),
          "per_attempt_thinking_tokens": {
              "mean": round(sum(think_list) / max(1, len(think_list)), 1),
              "median": _percentile(think_list, 50),
              "p90": _percentile(think_list, 90),
              "min": min(think_list) if think_list else 0,
              "max": max(think_list) if think_list else 0,
          },
          "per_attempt_output_tokens": {
              "mean": round(sum(out_list) / max(1, len(out_list)), 1),
              "median": _percentile(out_list, 50),
              "p90": _percentile(out_list, 90),
              "min": min(out_list) if out_list else 0,
              "max": max(out_list) if out_list else 0,
          },
          "per_pair_thinking_tokens_across_attempts": {
              "mean": round(
                  sum(pair_think_totals) / max(1, len(pair_think_totals)), 1
              ),
              "median": _percentile(pair_think_totals, 50),
              "p90": _percentile(pair_think_totals, 90),
              "max": max(pair_think_totals) if pair_think_totals else 0,
          },
          "per_pair_output_tokens_across_attempts": {
              "mean": round(
                  sum(pair_out_totals) / max(1, len(pair_out_totals)), 1
              ),
              "median": _percentile(pair_out_totals, 50),
              "p90": _percentile(pair_out_totals, 90),
              "max": max(pair_out_totals) if pair_out_totals else 0,
          },
      },
      "documented_settings": documented_settings,
  })
  return base


def save_shard_and_global_submissions(
    output_dir: str,
    shard_suffix: str,
    evalset: dict[str, dict[str, Any]],
    completed_map: dict[tuple[str, int], dict[str, Any]],
) -> None:
  """Writes both per-shard and global merged official Kaggle `submission.json` files."""
  sub: dict[str, list[dict[str, Any]]] = {}
  for tid, task in sorted(evalset.items()):
    preds_for_task: list[dict[str, Any]] = []
    for pidx, tpair in enumerate(task["test"]):
      row = completed_map.get((tid, pidx))
      if row is not None:
        a1 = (
            row.get("submission_attempt_1")
            or row.get("attempt_1")
            or copy.deepcopy(tpair["input"])
        )
        a2 = row.get("submission_attempt_2") or row.get("attempt_2") or a1
      else:
        a1 = copy.deepcopy(tpair["input"])
        a2 = copy.deepcopy(tpair["input"])
      preds_for_task.append({"attempt_1": a1, "attempt_2": a2})
    sub[tid] = preds_for_task

  if shard_suffix:
    shard_sub_path = os.path.join(output_dir, f"submission{shard_suffix}.json")
    try:
      with gfile.Open(shard_sub_path, "w") as sf:
        json.dump(sub, sf)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  global_sub_path = os.path.join(output_dir, "submission.json")
  try:
    with gfile.Open(global_sub_path, "w") as gf:
      json.dump(sub, gf)
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def main(argv: Sequence[str]) -> None:
  del argv
  t0 = time.time()

  # 1. Stage ARC-AGI-2 evaluation dataset (`120` tasks, `167` test pairs) to local `/tmp`
  local_data = f"/tmp/arc_agi_2_ds41_eval_shard_{FLAGS.shard_idx}"
  if os.path.exists(local_data):
    shutil.rmtree(local_data)
  os.makedirs(local_data, exist_ok=True)
  r = subprocess.run(
      [
          "fileutil",
          "cp",
          "-R",
          "-parallelism=32",
          f"{FLAGS.cns_data_path}/evaluation",
          f"{local_data}/evaluation",
      ],
      capture_output=True,
      text=True,
      check=False,
  )
  print(
      f"[DS4.1-ARC2-Eval] Staged evaluation data rc={r.returncode} in {time.time() - t0:.1f}s",
      flush=True,
  )
  evalset = {
      f[:-5]: json.load(open(f"{local_data}/evaluation/{f}"))
      for f in sorted(os.listdir(f"{local_data}/evaluation"))
      if f.endswith(".json")
  }
  if FLAGS.limit_tasks > 0:
    evalset = dict(list(evalset.items())[: FLAGS.limit_tasks])

  all_pairs: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
  shard_pairs: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
  for t_idx, (tid, task) in enumerate(sorted(evalset.items())):
    for pidx, tpair in enumerate(task["test"]):
      item = (tid, pidx, task, tpair)
      all_pairs.append(item)
      if FLAGS.shard_by_task and (t_idx % FLAGS.num_shards == FLAGS.shard_idx):
        shard_pairs.append(item)

  if not FLAGS.shard_by_task:
    shard_pairs = [
        item
        for idx, item in enumerate(all_pairs)
        if idx % FLAGS.num_shards == FLAGS.shard_idx
    ]

  gfile.MakeDirs(FLAGS.output_dir)
  shard_suffix = (
      f"_shard_{FLAGS.shard_idx:03d}of{FLAGS.num_shards:03d}"
      if FLAGS.num_shards > 1
      else ""
  )
  preds_path = os.path.join(FLAGS.output_dir, f"predictions{shard_suffix}.jsonl")
  transcripts_path = os.path.join(
      FLAGS.output_dir, f"transcripts{shard_suffix}.jsonl"
  )
  shard_summary_path = os.path.join(
      FLAGS.output_dir, f"summary{shard_suffix}.json"
  )
  global_summary_path = os.path.join(FLAGS.output_dir, "summary.json")
  config_settings_path = os.path.join(FLAGS.output_dir, "config_settings.json")

  completed_map = runner_lib.load_completed_rows(FLAGS.output_dir)
  todo_pairs = [
      item for item in shard_pairs if (item[0], item[1]) not in completed_map
  ]

  num_tpus = jax.device_count()
  device_kind = jax.devices()[0].device_kind if num_tpus > 0 else "unknown"
  print(
      f"[{FLAGS.exp_name}] Total tasks={len(evalset)} ({len(all_pairs)} test pairs) | "
      f"Shard {FLAGS.shard_idx}/{FLAGS.num_shards}: {len(shard_pairs)} pairs "
      f"({len(todo_pairs)} remaining) | Hardware: {num_tpus}x {device_kind}",
      flush=True,
  )

  # 2. Initialize Full-Precision DeepSeek-V4.1-Flash-1017 TPU Engine with 64K Context & Stochastic Sampling
  vllm_mod: Any = ds_traces.CompatVllmNamespace
  active_backend = FLAGS.engine_backend
  active_model_name = "DeepSeek-V4.1-Flash-1017 (Full-Precision BF16->FP32)"

  if active_backend in ("auto", "ds41_full_precision") and num_tpus in (8, 16):
    wave_bsz = max(2, FLAGS.pair_batch_size * max(1, FLAGS.attempts_per_wave))
    llm = NativeDeepSeekV41FullPrecisionLLM(
        raw_checkpoint_dir=FLAGS.ds41_raw_checkpoint_dir,
        ocdbt_tp8_dir=FLAGS.ds41_ocdbt_tp8_dir,
        ocdbt_tp16_dir=FLAGS.ds41_ocdbt_tp16_dir,
        max_seq_len=FLAGS.max_model_len,
        batch_size=wave_bsz,
        steps_per_call=12,
        temperature=FLAGS.temperature,
        top_k=FLAGS.top_k,
    )
    active_backend = "ds41_full_precision"
  else:
    raise RuntimeError(
        f"Expected 8 or 16 TensorCores for ds41_full_precision, got num_tpus={num_tpus} "
        f"(engine_backend={FLAGS.engine_backend})."
    )

  documented_settings = build_documented_settings(
      active_model_name=active_model_name,
      active_backend=active_backend,
      num_tpus=num_tpus,
      device_kind=device_kind,
      total_tasks=len(evalset),
      total_pairs=len(all_pairs),
  )
  if FLAGS.shard_idx == 0:
    try:
      with gfile.Open(config_settings_path, "w") as cf:
        json.dump(documented_settings, cf, indent=2)
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  if not todo_pairs:
    print("All pairs for this shard are already completed!", flush=True)
    return

  # 3. Group `todo_pairs` into batches of `pair_batch_size`
  batches: list[list[tuple[str, int, dict[str, Any], dict[str, Any]]]] = []
  if FLAGS.shard_by_task:
    cur_batch: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
    idx = 0
    while idx < len(todo_pairs):
      tid0 = todo_pairs[idx][0]
      task_group: list[tuple[str, int, dict[str, Any], dict[str, Any]]] = []
      while idx < len(todo_pairs) and todo_pairs[idx][0] == tid0:
        task_group.append(todo_pairs[idx])
        idx += 1
      if cur_batch and len(cur_batch) + len(task_group) > FLAGS.pair_batch_size:
        batches.append(cur_batch)
        cur_batch = []
      cur_batch.extend(task_group)
    if cur_batch:
      batches.append(cur_batch)
  else:
    for b_start in range(0, len(todo_pairs), FLAGS.pair_batch_size):
      batches.append(todo_pairs[b_start : b_start + FLAGS.pair_batch_size])

  t_eval_start = time.time()
  cum_gen_tokens = 0
  cum_gen_seconds = 0.0
  shard_done_pairs = len(shard_pairs) - len(todo_pairs)

  pbar = tqdm.tqdm(
      total=len(todo_pairs),
      desc=f"{FLAGS.exp_name} [shard {FLAGS.shard_idx}/{FLAGS.num_shards}]",
      dynamic_ncols=True,
  )

  for batch_idx, batch_items in enumerate(batches):
    attempts_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = {
        (tid, pidx): [] for tid, pidx, _, _ in batch_items
    }
    all_batch_rts: list[HavuAttemptRuntime] = []
    step_w = max(1, FLAGS.attempts_per_wave)

    for w_start in range(0, FLAGS.num_attempts, step_w):
      w_end = min(FLAGS.num_attempts, w_start + step_w)
      wave_rts: list[HavuAttemptRuntime] = []
      for tid, pidx, raw_task, tpair in batch_items:
        pair_hash_int = int(
            hashlib.sha1(f"{tid}_{pidx}".encode()).hexdigest()[:8], 16
        )
        for a_idx in range(w_start, w_end):
          st = runner_lib.AttemptState(
              task_id=tid,
              pair_idx=pidx,
              attempt_idx=a_idx,
              raw_task=raw_task,
              raw_test_input=tpair["input"],
              target_output=tpair["output"],
              use_tools=True,
              enable_tta=(FLAGS.enable_tta and a_idx >= 2),
              use_images=False,
              multi_hypothesis=False,
              soft_target=FLAGS.soft_token_target,
              hard_budget=FLAGS.total_token_budget,
              coder_next_format=False,
          )
          init_msgs, eff_val = build_havu_initial_messages(
              task=st.tta_task,
              test_input=st.tta_test_input,
              attempt_idx=a_idx,
              compact_grids=FLAGS.compact_grid_prompts,
          )
          sess = HavuSubprocessSession(
              train=st.tta_task["train"],
              test_input=st.tta_test_input,
              timeout_s=float(FLAGS.sandbox_timeout_s),
              max_output_chars=16000,
              style="ipython",
          )
          base_seed = int(
              ((pair_hash_int ^ (a_idx + 1) * 1000003) & 0x7FFFFFFF) or 1
          )
          wave_rts.append(
              HavuAttemptRuntime(
                  state=st,
                  messages=init_msgs,
                  reasoning_effort=eff_val,
                  session=sess,
                  base_seed=base_seed,
              )
          )

      w_desc = f"batch {batch_idx + 1}/{len(batches)} [attempts {w_start}..{w_end - 1}]"
      w_toks, w_secs = run_havu_attempt_wave_batched(
          llm=llm,
          vllm_mod=vllm_mod,
          runtimes=wave_rts,
          wave_desc=w_desc,
      )
      cum_gen_tokens += w_toks
      cum_gen_seconds += w_secs
      all_batch_rts.extend(wave_rts)

    for rt in all_batch_rts:
      st = rt.state
      rec = st.finalize_record()
      rec["submitted_solve_code"] = rt.submit_code or ""
      rec["answer_source"] = rt.answer_source
      rec["stop_reason"] = rt.stop_reason
      rec["train_check"] = rt.train_labels
      rec["tool_errors"] = rt.tool_errors
      rec["cap_nudges"] = rt.cap_nudges
      rec["code_rescues"] = rt.code_rescues
      rec["token_exact_breaks_saved"] = rt.token_exact_breaks_saved
      attempts_by_pair[(st.task_id, st.pair_idx)].append(rec)

    del all_batch_rts

    # Score selection policies (`train_check`-ranked + `score_selection_policies`) and flush to CNS
    transcript_rows: list[dict[str, Any]] = []
    with gfile.Open(preds_path, "a") as pf:
      for tid, pidx, task, tpair in batch_items:
        del task
        att_recs = list(attempts_by_pair[(tid, pidx)])
        target_grid = tpair["output"]
        policies = prompts_and_voting.score_selection_policies(
            att_recs, fallback_grid=tpair["input"], target=target_grid
        )

        # Exact `havu/evals/train_check.py:ranked_answers` ranking (`train_pass=True` first, then `train_frac`, then attempt order)
        ordered_atts = sorted(
            [r for r in att_recs if r.get("pred_grid") is not None],
            key=lambda r: (
                r.get("train_passed") is not True,
                -(float(r.get("train_check", {}).get("train_frac") or -1.0)),
                r["attempt_idx"],
            ),
        )
        havu_ranked_grids: list[list[list[int]]] = []
        for r in ordered_atts:
          if r["pred_grid"] not in havu_ranked_grids:
            havu_ranked_grids.append(r["pred_grid"])
          if len(havu_ranked_grids) == 2:
            break
        while len(havu_ranked_grids) < 2:
          havu_ranked_grids.append(
              havu_ranked_grids[0]
              if havu_ranked_grids
              else copy.deepcopy(tpair["input"])
          )

        att1_grid = havu_ranked_grids[0]
        att2_grid = havu_ranked_grids[1]
        pass_1 = bool(att1_grid == target_grid)
        pass_2 = bool(att1_grid == target_grid or att2_grid == target_grid)
        vote_meta = {
            "policy": "havu_train_check_ranked",
            "attempt_0_source": (
                att_recs[0].get("answer_source") if att_recs else None
            ),
            "attempt_0_train_pass": (
                att_recs[0].get("train_passed") if att_recs else None
            ),
            "attempt_1_source": (
                att_recs[1].get("answer_source") if len(att_recs) > 1 else None
            ),
            "attempt_1_train_pass": (
                att_recs[1].get("train_passed") if len(att_recs) > 1 else None
            ),
        }

        compact_atts = []
        for a in attempts_by_pair[(tid, pidx)]:
          drop = {"cross_pair_cands", "transcript"}
          if not FLAGS.keep_candidates:
            drop.add("candidates")
          compact_atts.append({k: v for k, v in a.items() if k not in drop})
          if FLAGS.save_transcripts and a.get("transcript"):
            transcript_rows.append({
                "task_id": tid,
                "pair_idx": pidx,
                "attempt_idx": a["attempt_idx"],
                "model": active_model_name,
                "precision": "full_precision_bf16_dequant_fp32_mxu",
                "reasoning_effort": 0,
                "exact_match": a.get("exact_match"),
                "train_passed": a.get("train_passed"),
                "answer_source": a.get("answer_source"),
                "stop_reason": a.get("stop_reason"),
                "train_check": a.get("train_check"),
                "turns": a.get("turns"),
                "tool_calls": a.get("tool_calls"),
                "tool_errors": a.get("tool_errors"),
                "cap_nudges": a.get("cap_nudges"),
                "output_tokens": a.get("output_tokens"),
                "thinking_tokens": a.get("thinking_tokens"),
                "tool_use_tokens": a.get("tool_use_tokens"),
                "answer_tokens": a.get("answer_tokens"),
                "submitted_solve_code": a.get("submitted_solve_code"),
                "turn_logs": a.get("turn_logs"),
                "thought_text": a.get("thought_text"),
                "visible_text": a.get("visible_text"),
                "pred_grid": a.get("pred_grid"),
                "transcript": a["transcript"],
            })

        pair_think_toks = sum(
            int(a.get("thinking_tokens", 0))
            for a in attempts_by_pair[(tid, pidx)]
        )
        pair_out_toks = sum(
            int(a.get("output_tokens", 0))
            for a in attempts_by_pair[(tid, pidx)]
        )
        row = {
            "task_id": tid,
            "pair_idx": pidx,
            "model": active_model_name,
            "precision": "full_precision_bf16_dequant_fp32_mxu",
            "reasoning_effort": 0,
            "pass_at_1": pass_1,
            "pass_at_2": pass_2,
            "attempt_1": att1_grid,
            "attempt_2": att2_grid,
            "vote_meta": vote_meta,
            "submission_policy": "havu_train_check_ranked",
            "submission_attempt_1": att1_grid,
            "submission_attempt_2": att2_grid,
            "pair_thinking_tokens": pair_think_toks,
            "pair_output_tokens": pair_out_toks,
            "policies": {
                k: {
                    "pass_at_1": bool(v["pass_at_1"]),
                    "pass_at_2": bool(v["pass_at_2"]),
                    "meta": v["meta"],
                }
                for k, v in policies.items()
            },
            "attempts": compact_atts,
        }
        pf.write(json.dumps(row) + "\n")

    if transcript_rows:
      try:
        with gfile.Open(transcripts_path, "a") as tf:
          for trow in transcript_rows:
            tf.write(json.dumps(trow) + "\n")
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[WARN] Failed to write transcripts: {exc}", flush=True)

    shard_done_pairs += len(batch_items)
    completed_map = runner_lib.load_completed_rows(FLAGS.output_dir)
    all_rows = [completed_map[k] for k in sorted(completed_map.keys())]
    shard_key_set = {(it[0], it[1]) for it in shard_pairs}
    shard_rows = [
        completed_map[k]
        for k in sorted(completed_map.keys())
        if k in shard_key_set
    ]

    wall_s = time.time() - t_eval_start
    shard_summary = compute_deepseek_eval_summary(
        rows=shard_rows,
        total_eval_tasks=len(evalset),
        total_eval_pairs=len(all_pairs),
        shard_total_pairs=len(shard_pairs),
        shard_done_pairs=shard_done_pairs,
        total_gen_tokens=cum_gen_tokens,
        total_gen_seconds=cum_gen_seconds,
        wall_elapsed_s=wall_s,
        num_tpus=num_tpus,
        device_kind=device_kind,
        active_model_name=active_model_name,
        active_backend=active_backend,
        documented_settings=documented_settings,
    )
    global_summary = compute_deepseek_eval_summary(
        rows=all_rows,
        total_eval_tasks=len(evalset),
        total_eval_pairs=len(all_pairs),
        shard_total_pairs=len(shard_pairs),
        shard_done_pairs=shard_done_pairs,
        total_gen_tokens=cum_gen_tokens,
        total_gen_seconds=cum_gen_seconds,
        wall_elapsed_s=wall_s,
        num_tpus=num_tpus,
        device_kind=device_kind,
        active_model_name=active_model_name,
        active_backend=active_backend,
        documented_settings=documented_settings,
    )

    try:
      with gfile.Open(shard_summary_path, "w") as sf:
        json.dump(shard_summary, sf, indent=2)
      with gfile.Open(global_summary_path, "w") as gf:
        json.dump(global_summary, gf, indent=2)
      save_shard_and_global_submissions(
          FLAGS.output_dir, shard_suffix, evalset, completed_map
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      print(f"[WARN] Non-fatal CNS summary lock contention: {exc}", flush=True)

    pbar.update(len(batch_items))
    pbar.set_postfix(
        done=f"{global_summary['pairs_completed']}/{len(all_pairs)}",
        p1=f"{global_summary['kaggle_pass_at_1_pct']:.1f}%",
        p2=f"{global_summary['kaggle_pass_at_2_pct']:.1f}%",
        task_p2=f"{global_summary['arc_prize_pass_at_2_pct']:.1f}%",
        think_tok=f"{global_summary['mean_thinking_tokens_per_attempt']:.0f}",
        tok_s=f"{global_summary['vllm_decode_tokens_per_sec']:.0f}",
        eta_m=f"{shard_summary['shard_eta_minutes']:.1f}m",
    )

  pbar.close()
  completed_map = runner_lib.load_completed_rows(FLAGS.output_dir)
  all_rows = [completed_map[k] for k in sorted(completed_map.keys())]
  final_summary = compute_deepseek_eval_summary(
      rows=all_rows,
      total_eval_tasks=len(evalset),
      total_eval_pairs=len(all_pairs),
      shard_total_pairs=len(shard_pairs),
      shard_done_pairs=shard_done_pairs,
      total_gen_tokens=cum_gen_tokens,
      total_gen_seconds=cum_gen_seconds,
      wall_elapsed_s=time.time() - t_eval_start,
      num_tpus=num_tpus,
      device_kind=device_kind,
      active_model_name=active_model_name,
      active_backend=active_backend,
      documented_settings=documented_settings,
  )
  print("=" * 80, flush=True)
  print(f"FINAL SUMMARY ({FLAGS.exp_name}):", flush=True)
  print(json.dumps(final_summary, indent=2), flush=True)
  print("=" * 80, flush=True)


if __name__ == "__main__":
  app_runner.run(main)
