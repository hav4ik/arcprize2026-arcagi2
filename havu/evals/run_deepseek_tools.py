"""DeepSeek ARC-AGI-2 eval with persistent sandboxed Python tool use.

Start evals/sandbox_docker/start.sh, then pass its URL with --sandbox-url.
Defaults use the ARC Prize public eval prompt/parser/scoring. Each independent
attempt owns one Python session. See experiments/E004_tool_use/README.md.
"""
import argparse
import ast
import builtins
import gzip
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from run_deepseek_arc2 import api_key
from run_deepseek_arcprize import TASKS, build_prompt, parse
from arc_harness.arcprize import TEMPLATE as ARC_TEMPLATE
from arc_harness.grid_image import aligned_cell, build_image_prompt, data_uri
from sandbox_client import SandboxSession, check_health, tool_output
from task_summary import task_summary
from train_check import ranked_answers, train_check

ROOT = Path(__file__).resolve().parents[1]
# USD / million tokens: cache hit, cache miss, output (off-peak).
PRICES = {"deepseek-flash": (0.003, 0.15, 0.60), "deepseek-v4-pro": (0.022, 0.66, 1.98)}
# Optional suffixes appended to the ARC Prize prompt (--prompt). "plain" keeps the original prompt.
PROMPT_SUFFIXES = {
    "plain": "",
    "lean": ("\n\nYou have a Python tool. Work efficiently: inspect the grids with code instead of reading them "
             "by hand, form a hypothesis, and write code that computes the output. Never construct, copy or count "
             "grids by hand. Check your rule on every training pair with code before answering, and if a pair "
             "fails, fix the rule rather than explaining the mismatch away. Keep your reasoning brief."),
    "lean2": ("\n\nYou have a Python tool. The variables `train` (list of {'input', 'output'} grids) and "
              "`test_input` are already loaded in it: use them directly and never retype or reassign them. "
              "Work efficiently: inspect the grids with code instead of reading them by hand, form a hypothesis, "
              "and write code that computes the output. Never construct, copy or count grids by hand. Check your "
              "rule on every training pair with code before answering, and if a pair fails, fix the rule rather "
              "than explaining the mismatch away. Before answering, check that the rule's assumptions also hold "
              "on test_input. Keep your reasoning brief."),
}
# E009 suffixes for --answer-tool (grid mode): hand the answer over from the session, plus brevity guidance.
_SUBMIT = ("\n\nWhen your code reproduces every training output and its assumptions hold on test_input, call "
           "submit_answer with an expression such as `solve(test_input)`: the harness evaluates it in your session "
           "and that grid is your answer, so never retype the grid. Keep your current best guess for the test "
           "output in a variable `answer`; if you run out of budget it is submitted automatically.")
_BRIEF = (" Be economical: think in short notes, test each idea with a small code experiment instead of long "
          "deliberation, and print compact views (shapes, diffs, one string per row) rather than full nested "
          "lists. Once train passes and the test checks out, submit at once without re-verifying.")
PROMPT_SUFFIXES["lean2sub"] = PROMPT_SUFFIXES["lean2"] + _SUBMIT
PROMPT_SUFFIXES["lean2brief"] = PROMPT_SUFFIXES["lean2"] + _SUBMIT + _BRIEF
# E011: require a function so --verify-submit can check it on the training pairs.
PROMPT_SUFFIXES["lean2subv"] = PROMPT_SUFFIXES["lean2"] + (
    "\n\nPut your rule in a function solve(grid) that returns the output grid. When it reproduces every training "
    "output and its assumptions hold on test_input, call submit_answer with exactly `solve(test_input)`: the "
    "harness first runs solve on every training input, then evaluates it on test_input in your session, and that "
    "grid is your answer, so never retype the grid. Keep your current best guess for the test output in a variable "
    "`answer`; if you run out of budget it is submitted automatically.")
# E017 P2: roles note before solve(), no unproven "noise" labels, test audit after train passes.
PROMPT_SUFFIXES["lean2roles"] = PROMPT_SUFFIXES["lean2sub"] + (
    "\n\nBefore writing solve(), write a short ROLES note: what each kind of object or colour does (template, "
    "marker, key/legend, obstacle, container, background). Do not call anything noise or irrelevant unless code "
    "shows it is removed or unchanged in every training output. When solve() reproduces every training output, "
    "print what is different about test_input compared with all training inputs (sizes, colours, objects clipped "
    "by the border, counts, key entries, shapes not seen) and say in one line how solve handles each difference; "
    "if your rule ignores some non-background cells of test_input, explain them before submitting. If any "
    "training pair still fails, or a test difference is unhandled, do not submit: fix the rule.")
# E029: lean2sub plus "decide in code" (Flash-1017 traces spend most tokens on prose rule comparison and hand
# reading of printed grids). Deliberately no brevity wording: E009 lean2brief lost 8-14 pp.
PROMPT_SUFFIXES["lean2code"] = PROMPT_SUFFIXES["lean2sub"] + (
    "\n\nLet code do the perceiving and the checking. Get coordinates, counts, object lists, distances, paths and "
    "mismatches by printing them from code instead of reading them off the grids or working them out cell by cell "
    "in your reasoning, and do not re-read grids that code has already analysed. When you are choosing between "
    "candidate rules, implement each one and compare them on the training pairs in code rather than arguing it out "
    "in prose.")
SUBMIT_PROMPTS = {"lean2sub", "lean2brief", "lean2subv", "lean2roles", "lean2code"}
HARNESS_VERSION = "persistent-v3"  # v3: submissions run on inputs only in a fresh session; host compares
# v4 (E009) is recorded only when a new option below is active; defaults reproduce v3 exactly.
HARNESS_VERSION_V4 = "persistent-v4"
V4_DEFAULTS = {"answer_tool": False, "final_mode": "turn", "budget_notes": 0.0, "budget_warn": 0.8,
               "lenient_parse": False, "budget_soft": 0, "soft_wording": "gentle", "think_turns": -1,
               "verify_submit": False}
# v5 (E017) is recorded when an E017 option or prompt is active; it adds the keys below to the config.
HARNESS_VERSION_V5 = "persistent-v5"
V5_DEFAULTS = {"task_summary": False, "helpers": False, "train_check": False}
V5_PROMPTS = {"lean2roles"}
# v6 (E029) is recorded when an E029 option or prompt is active; it adds the keys below to the config.
HARNESS_VERSION_V6 = "persistent-v6"
V6_DEFAULTS = {"turns_note": 0, "fallback_solve": False, "cap_nudge": False, "effort_header": "", "grid_format": "json",
               "image_cell": 32, "image_align": "deepseek", "openrouter_provider": "", "or_reasoning": "", "multi_test": False,
               "multi_budget": 0.0, "multi_prompt": "note", "context_limit": 0,
               "prior_mode": "", "alt_answer": ""}
OPENROUTER_URL = "https://openrouter.ai/api/v1"
V6_PROMPTS = {"lean2code"}
HELPERS_SRC = (Path(__file__).resolve().parent / "arc_helpers.py").read_text()
HELPERS_DOC = (
    " Also preloaded, numpy grid helpers (g = list or array; boxes (r0,c0,r1,c1) inclusive): "
    "objects(g, by='colour'|'nonbg', diag=True, bg=None) -> list of dicts {colour, colours, cells [(r,c)], bbox, "
    "size, holes, touches_border, shape_key} in reading order (equal shape_key = same shape up to "
    "rotation/reflection); bbox(mask); crop(g, box); bg_colour(g); d8(g) -> {name: array} for the 8 "
    "rotations/reflections (id, rot90, rot180, rot270, flipud, fliplr, transpose, antitranspose); match_d8(a, b) "
    "-> name of the transform mapping a to b, or None; panels(g) -> [{'bbox', 'grid', 'pos'}] sub-grids split by "
    "full separator lines ([] if none); stamp(g, pattern, at, clip=True, transparent=None) -> copy of g with "
    "pattern pasted at top-left `at` (may overhang; clipped at the edges); ray(g, start, d, stop=None) -> cells "
    "from start (exclusive) stepping by d=(dr,dc) until the edge or the first cell where stop(value) (default: "
    "non-background); show(g) prints one digit string per row; check(fn) runs fn on a copy of every training "
    "input and prints per pair OK, the shape mismatch, the error, or up to 8 differing cells as (r,c) got/exp, "
    "then ALL PASS; returns True if all pass.")
HELPERS_PROMPT = "\n\nUse check(solve) after every change to solve."
# Weekday peak pricing (2x) on the DeepSeek API, UTC hours [start, end).
PEAK_HOURS = ((1, 4), (6, 10))


def peak_multiplier(now=None):
    """2 inside a weekday peak window (UTC), else 1."""
    import datetime
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return 2.0 if now.weekday() < 5 and any(a <= now.hour < b for a, b in PEAK_HOURS) else 1.0


def harness_version(args):
    if any(getattr(args, k, v) != v for k, v in V6_DEFAULTS.items()) or getattr(args, "prompt", "") in V6_PROMPTS:
        return HARNESS_VERSION_V6
    if any(getattr(args, k, v) != v for k, v in V5_DEFAULTS.items()) or getattr(args, "prompt", "") in V5_PROMPTS:
        return HARNESS_VERSION_V5
    if any(getattr(args, k, v) != v for k, v in V4_DEFAULTS.items()):
        return HARNESS_VERSION_V4
    return HARNESS_VERSION


def tools_for(args):
    descriptions = {"run_python": (
        "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
        "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
        + ("test_inputs: list of every test input grid of the task (test_input is test_inputs[0]). "
           if getattr(args, "multi_test", False) else "") +
        f"scipy is available. No internet or host files. Timeout {args.tool_timeout}s. "
        "Use print() to inspect results. Check transformations on all training pairs."
        + (HELPERS_DOC if getattr(args, "helpers", False) else "")
    )}
    if getattr(args, "answer_tool", False):
        descriptions["submit_answer"] = (
            "End this attempt with your test output. `code` is a Python expression evaluated in your session, "
            "e.g. solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or numpy array); "
            "statements that set a variable `answer` also work. The harness reads the grid directly: do not "
            "retype it. Invalid grids return an error and the attempt continues."
            + (" When the task has several test inputs, the expression must give a list with one grid per test "
               "input, in order, e.g. [solve(t) for t in test_inputs]." if getattr(args, "multi_test", False) else "")
        )
    if args.mode == "code":
        descriptions["submit_solution"] = (
            "Submit self-contained Python code defining transform(grid), which returns a rectangular "
            "grid of ints 0-9 (or numpy array). Include imports/helpers; execution uses a fresh namespace. "
            "The harness checks every training example. Failures return feedback so you can revise. "
            "If all pass, it applies transform to test_input and ends this attempt."
        )
    return [{"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"], "additionalProperties": False},
    }} for name, description in descriptions.items()]


def valid_grid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
            and len({len(row) for row in grid}) == 1
            and all(type(value) is int and 0 <= value <= 9 for row in grid for value in row))


def submit_solution(session, code):
    """Run submitted code on the task inputs in a fresh session; compare with train outputs on the host.

    The checker session never holds any output grid, and the comparison happens outside the sandbox."""
    marker = f"ARC_SUBMIT_{uuid4().hex}:"
    inputs = [pair["input"] for pair in session.train] + [session.test_input]
    cell = f'''
def _arc_run():
    import json
    scope = {{}}
    exec({code!r}, scope)
    transform = scope.get('transform')
    if not callable(transform):
        raise ValueError('Define transform(grid)')
    outputs = []
    for grid in json.loads({json.dumps(inputs)!r}):
        value = transform(grid)
        if hasattr(value, 'tolist'):
            value = value.tolist()
        outputs.append(value)
    print({marker!r} + json.dumps(outputs, default=int))
_arc_run()
'''
    with session.checker() as checker:
        result = checker.execute(cell)
    outputs = None
    if result["process_status"] == "completed":
        for line in reversed(result.get("stdout", "").splitlines()):
            if line.startswith(marker):
                try:
                    outputs = json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    pass
                break
    if not isinstance(outputs, list) or len(outputs) != len(inputs):
        return tool_output(result, session.max_output) + "\n[submission not accepted; reduce printed output if truncated]", None
    if not all(valid_grid(o) for o in outputs):
        return "[submission not accepted] ValueError: transform must return a rectangular 1-30 by 1-30 grid of ints 0-9", None
    failures = [{"example": i, "pred": o, "expected": pair["output"]}
                for i, (o, pair) in enumerate(zip(outputs, session.train)) if o != pair["output"]]
    if failures:
        return json.dumps({"accepted": False, "training_failures": failures})[:session.max_output], None
    return json.dumps({"accepted": True, "pred": outputs[-1]}), outputs[-1]


def grid_problem(grid):
    """Why `grid` is not a rectangular 1-30 by 1-30 grid of ints 0-9, or None if it is one (A11 feedback)."""
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
            if not (type(value) is int and 0 <= value <= 9):
                return f"value {json.dumps(value)[:20]} is not an int 0-9"
    return None


# Nested numpy rows / grids (e.g. [solve(t) for t in test_inputs], or a solve that returns a list of numpy rows) have no
# .tolist() at the top and json.dumps(default=int) then fails ("only length-1 arrays"): convert nested values first.
# Values that already serialize are unchanged (E046 run 7: a single-test fallback failed on this). A10: a nested numpy
# bool scalar becomes 0/1, as v3's json.dumps(default=int) did (Python bools and bool arrays stay rejected, as in v3).
ANSWER_PLAIN = '''
    def _plain(v):
        if _b.getattr(v, 'ndim', None) == 0 and _b.getattr(_b.getattr(v, 'dtype', None), 'kind', '') == 'b':
            return _b.int(v)
        if _b.hasattr(v, 'tolist'):
            return v.tolist()
        return [_plain(x) for x in v] if _b.isinstance(v, (_b.list, _b.tuple)) else v'''
# Built-ins via the builtins module: the model's session may define its own eval / compile / exec / print (E046 run 3:
# a session function named eval(grid, H, bg) broke the answer and fallback-solve paths of 4e34c42c; A7: print).
# {body} sets `text`. A11: a value longer than the session's output limit is kept in a session variable and read back
# in slices (one 30x30 grid is ~1.9k characters; 6 of them used to be cut and rejected as "got NoneType").
ANSWER_CELL = '''
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
'''
ANSWER_SLICE = '''
def _arc_slice():
    import builtins as _b
    _b.print({marker!r} + _b.globals()[{name!r}][{start}:{end}])
try:
    _arc_slice()
finally:
    del _arc_slice
'''
ANSWER_DROP = '''
def _arc_drop():
    import builtins as _b
    _b.globals().pop({name!r}, None)
try:
    _arc_drop()
finally:
    del _arc_drop
'''


def _answer_value(session, body):
    """Run an answer cell (`body` sets `text`, a JSON string) in the session; -> (result, value, found). `found` is
    False when no marker line came back (error, timeout, or the cell printed nothing)."""
    marker, name = f"ARC_ANSWER_{uuid4().hex}:", f"_arc_text_{uuid4().hex[:12]}"
    chunk = max(1000, int(getattr(session, "max_output", 16000) or 16000) - len(marker) - 64)
    cell = ANSWER_CELL.format(plain=ANSWER_PLAIN, body=body, chunk=chunk, name=name, marker=marker)
    result = session.execute(cell)
    if result["process_status"] != "completed":
        return result, None, False

    def marked(res):
        for line in reversed(res.get("stdout", "").splitlines()):
            if line.startswith(marker):
                return line[len(marker):]
        return None
    head = marked(result)
    if head is None:
        return result, None, False
    size, _, text = head.partition(":")
    try:
        size = int(size)
        while len(text) < size:  # A11: the rest of a long value, in slices below the output limit
            part = session.execute(ANSWER_SLICE.format(marker=marker, name=name, start=len(text),
                                                       end=min(size, len(text) + chunk)))
            piece = marked(part) if part["process_status"] == "completed" else None
            if not piece:
                break
            text += piece
        if size > chunk:
            session.execute(ANSWER_DROP.format(name=name))
        return result, json.loads(text) if len(text) == size else None, True
    except (ValueError, KeyError):
        return result, None, True


def submit_answer(session, code, n=None):
    """Evaluate the model's answer expression in its own persistent session and return (feedback, grid|None).
    n (E046 --multi-test): the value must be a list of n grids, one per test input; returns that list.

    `code` is an expression (e.g. solve(test_input)) or statements that set `answer`. Printed output from
    the evaluation is captured so only the marker line comes back. The session never holds test answers."""
    body = f'''    _src = {code!r}
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
        text = json.dumps(value, default=_b.int, separators=(',', ':'))'''
    result, grid, found = _answer_value(session, body)
    if result["process_status"] != "completed":
        return tool_output(result, session.max_output)[:2000] + "\n[answer not accepted]", None
    if not found:
        return "[answer not accepted] no value returned", None
    if n is not None:
        if not (isinstance(grid, list) and len(grid) == n and all(valid_grid(g) for g in grid)):
            bad = (f"a list of {len(grid)} items" if isinstance(grid, list) and not valid_grid(grid)
                   else "one grid" if valid_grid(grid) else type(grid).__name__)
            if isinstance(grid, list) and len(grid) == n and not valid_grid(grid):  # A11: name the bad grids
                probs = [(i, grid_problem(g)) for i, g in enumerate(grid)]
                bad += "".join(f"; grid {i} (test input {i}): {p}" for i, p in probs if p)
            return (f"[answer not accepted] this task has {n} test inputs: give a list of {n} rectangular 1-30 by "
                    f"1-30 grids of ints 0-9, one per test input in order (e.g. [solve(t) for t in test_inputs]); "
                    f"got {bad}"), None
        return f"[answer accepted: {n} grids, " + ", ".join(f"{len(g)}x{len(g[0])}" for g in grid) + "]", grid
    if not valid_grid(grid):
        shape = (f"{len(grid)} rows, row lengths {sorted({len(r) for r in grid if isinstance(r, list)})}"
                 if isinstance(grid, list) else type(grid).__name__)
        return ("[answer not accepted] must be a rectangular 1-30 by 1-30 grid of ints 0-9; got " + shape), None
    return f"[answer accepted: {len(grid)}x{len(grid[0])} grid]", grid


def _items_result(result, items, found, n, what):
    """(feedback, [grid or None] * n, or None when no grid is valid) from an items cell's JSON list."""
    if result["process_status"] != "completed" or not found:
        text = tool_output(result, 2000)[:300] if result["process_status"] != "completed" else "no value returned"
        return f"[{what}: {text}]", None
    if not isinstance(items, list) or len(items) != n:
        return f"[{what}: not a list of {n} items]", None
    grids = [g if valid_grid(g) else None for g in items]
    probs = "".join(f"; grid {i}: {grid_problem(g)}" for i, g in enumerate(items) if not valid_grid(g))
    return f"[{what}: {n - grids.count(None)} of {n} grids valid{probs}]", grids if any(grids) else None


def answer_items(session, code, n):
    """A1 (--multi-test fallbacks): the value of `code` (e.g. answer) as n separately validated grids, so one bad grid
    no longer loses the other test inputs' answers. -> (feedback, [grid or None] * n, or None if none is valid)."""
    body = f'''    with contextlib.redirect_stdout(io.StringIO()):
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
        text = '[' + ','.join(parts) + ']' if parts else 'null\''''
    result, items, found = _answer_value(session, body)
    return _items_result(result, items, found, n, "session answer")


def solve_items(session, inputs):
    """A1 / A3 / A8 (--fallback-solve): the session's solve on each test input separately (host copies of the inputs),
    first as a list, then as a numpy array if the list call raises, each with its own time limit and errors caught
    per input. -> (feedback, [grid or None] per input, or None if none is valid)."""
    per = max(1.0, float(getattr(session, "timeout", 30) or 30))
    body = f'''    import copy, signal
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
    text = '[' + ','.join(parts) + ']\''''
    old_timeout = getattr(session, "timeout", None)
    if old_timeout is not None:
        session.timeout = per * 2 * len(inputs) + old_timeout
    try:
        result, items, found = _answer_value(session, body)
    finally:
        if old_timeout is not None:
            session.timeout = old_timeout
    return _items_result(result, items, found, len(inputs), "session solve")


def restore_inputs(session, test_inputs, multi):
    """A8: rebind `test_input` (and `test_inputs` when the session defines it) to the host's copies before the
    fallbacks read them: model code may have rebound them (5 rebinds in 1,779 E046 traces)."""
    names = {"test_input": test_inputs[0]}
    if multi:
        names["test_inputs"] = test_inputs
    session.execute(f'''
def _arc_restore():
    import builtins as _b, json
    _g = _b.globals()
    for _k, _v in json.loads({json.dumps(names)!r}).items():
        _g[_k] = _v
try:
    _arc_restore()
finally:
    del _arc_restore
''')


def multi_check_code(code, test_inputs):
    """A6: a --multi-test answer program as the one-grid program train_check labels (grid 0 of the list), with
    test_inputs[0] bound to the input being checked (train_check binds test_input) and the other test inputs kept."""
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


def inputs_prelude(test_inputs):
    """E046 session prelude: `test_inputs` (A5: also in single-test sessions under --multi-test, as [test_input])."""
    return f"\nimport json as _arc_json\ntest_inputs = _arc_json.loads({json.dumps(test_inputs)!r})\n"


PRELOADED = {"train", "test_input", "np", "json"}
NAMES_CELL = '''
def _arc_names():
    import builtins as _b, json
    _b.print({marker!r} + json.dumps(_b.sorted(k for k in _b.globals() if not k.startswith('_'))))
try:
    _arc_names()
finally:
    del _arc_names
'''


def session_names(session):
    """I5 --auto-repair: the public names in the session's globals (auto_repair's NameError candidates), or None."""
    marker = f"ARC_NAMES_{uuid4().hex}:"
    result = session.execute(NAMES_CELL.format(marker=marker))
    lines = result.get("stdout", "").splitlines() if result.get("process_status") == "completed" else []
    for line in reversed(lines):
        if line.startswith(marker):
            try:
                return set(json.loads(line[len(marker):]))
            except ValueError:
                return None
    return None


def auto_repair_cell(session, code, output, propose, names, limit, stats):
    """I5 --auto-repair: after a run_python cell failed, auto_repair.propose() may fix an identifier typo (a unique close
    match, and the part of the cell that ran changed nothing outside it); the fixed cell runs once in the same session
    and the model sees the note + the retry's output. -> (output, status, the code that ran) or None (no repair)."""
    fix = propose(code, output, (session_names(session) or set()) | names)
    if fix is None:
        return None
    new_code, note = fix
    stats["proposed"] += 1
    result = session.execute(new_code)
    stats["applied"] += 1
    stats["retry_ok"] += result["process_status"] == "completed"
    return note + "\n" + tool_output(result, limit), result["process_status"], new_code


def check_answer_on_train(session, code):
    """E011 --verify-submit: re-evaluate the submitted expression with `test_input` bound to each training input
    (host copies of the inputs) and compare with the host's training outputs. Returns None when the code is not
    an expression that uses test_input (nothing to check), else (failed_indices, feedback_text)."""
    try:
        expr = compile(code, "<answer>", "eval")
    except SyntaxError:
        return None
    if "test_input" not in expr.co_names:
        return None
    marker = f"ARC_CHECK_{uuid4().hex}:"
    inputs = [pair["input"] for pair in session.train]
    cell = f'''
def _arc_check():
    import builtins as _b, contextlib, io, json
    _expr = _b.compile({code!r}, '<answer>', 'eval')
    outs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _inp in json.loads({json.dumps(inputs)!r}):
            try:
                v = _b.eval(_expr, {{**_b.globals(), 'test_input': _inp}})
                if hasattr(v, 'tolist'):
                    v = v.tolist()
                outs.append(json.loads(json.dumps(v, default=int)))
            except Exception as exc:
                outs.append({{'error': type(exc).__name__ + ': ' + str(exc)[:200]}})
    print({marker!r} + json.dumps(outs))
try:
    _arc_check()
finally:
    del _arc_check
'''
    result = session.execute(cell)
    outs = None
    if result["process_status"] == "completed":
        for line in reversed(result.get("stdout", "").splitlines()):
            if line.startswith(marker):
                try:
                    outs = json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    pass
                break
    if not isinstance(outs, list) or len(outs) != len(inputs):
        return None  # the check itself failed (e.g. timeout); do not block the submission
    failed, notes = [], []
    for i, (o, pair) in enumerate(zip(outs, session.train)):
        if o == pair["output"]:
            continue
        failed.append(i)
        if isinstance(o, dict) and "error" in o:
            notes.append(f"pair {i}: {o['error']}")
        elif valid_grid(o):
            exp = pair["output"]
            if (len(o), len(o[0])) != (len(exp), len(exp[0])):
                notes.append(f"pair {i}: shape {len(o)}x{len(o[0])}, expected {len(exp)}x{len(exp[0])}")
            else:
                notes.append(f"pair {i}: {sum(a != b for ra, rb in zip(o, exp) for a, b in zip(ra, rb))} cells differ")
        else:
            notes.append(f"pair {i}: not a valid grid")
    return failed, "; ".join(notes)


RESCUE_DSML_CODE = re.compile(r'parameter name="code"[^>]*>(.*?)(?:</[^>]*parameter>|$)', re.S)
RESCUE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", re.S)
RESCUE_JSON_CODE = re.compile(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
RESCUE_TAG = re.compile(r"</?[^<>\n]*DSML[^<>\n]*>?|<br[^>]*>")
RESCUE_CODE_LINE = re.compile(r"^\s*(import |from \w+ import |print\(|for |def |if |while |np\.|[A-Za-z_]\w*\s*(\[[^\]]*\])?\s*=[^=])")
RESCUE_NUDGE = ("[harness] No valid tool call was found in your last message. To run code, call the run_python tool "
                "(do not write code as plain text); to finish, call submit_answer.")


def rescue_code(content):
    """E043: code written outside a valid tool call, from (in order) a garbled DSML code parameter, a JSON
    `"code": "..."` field, a ```python block, or plain code lines; None if there is none."""
    text = content or ""
    m = RESCUE_DSML_CODE.findall(text)
    if m and m[-1].strip():
        return RESCUE_TAG.sub("", m[-1])
    m = RESCUE_JSON_CODE.findall(text)
    if m:
        try:
            return json.loads('"' + m[-1] + '"')
        except ValueError:
            pass
    f = [b for b in RESCUE_FENCE.findall(text) if b.strip()]
    if f:
        return RESCUE_TAG.sub("", f[-1])
    lines = RESCUE_TAG.sub("", text).splitlines()
    start = next((i for i, l in enumerate(lines) if RESCUE_CODE_LINE.match(l)), None)
    if start is None:
        return None
    code = "\n".join(lines[start:]).strip()
    return code or None


def rescue_tool_call(content, n):
    """E043 --rescue-code: turn code written outside a valid tool call into a run_python call, or None.
    For pruned models that lose the DSML tool-call format while still writing code."""
    code = rescue_code(content)
    if not code or not code.strip():
        return None
    return {"id": f"rescue_{n}", "type": "function",
            "function": {"name": "run_python", "arguments": json.dumps({"code": code})}}


def rescue_name(name):
    """E043 --rescue-code: map near-miss tool names (run_ppython, run_p_python) to the real tools."""
    low = (name or "").lower()
    if "submit" in low:
        return "submit_answer"
    if "python" in low or "run" in low:
        return "run_python"
    return name


def lenient_parse(text):
    """Fallback for text answers the ARC Prize parser misses: the last valid JSON grid in the text, else the
    last block of consecutive lines that are each a JSON list of ints (one row per line). Returns one grid."""
    text = text or ""
    found = None
    for start in [i for i, ch in enumerate(text) if ch == "["][::-1]:
        if not text.startswith("[[", start) and not re.match(r"\[\s*\[", text[start:start + 50]):
            continue
        depth = 0
        for j in range(start, len(text)):
            depth += (text[j] == "[") - (text[j] == "]")
            if depth == 0:
                block = text[start:j + 1]
                try:
                    grid = json.loads(block)
                except json.JSONDecodeError:
                    try:  # tolerate trailing commas such as "],\n]"
                        grid = json.loads(re.sub(r",\s*\]", "]", block))
                    except json.JSONDecodeError:
                        grid = None
                if valid_grid(grid):
                    found = grid
                break
        if found is not None:
            return found
    rows, best = [], None
    for line in text.splitlines() + [""]:
        line = line.strip().rstrip(",")
        try:
            row = json.loads(line) if line.startswith("[") else None
        except json.JSONDecodeError:
            row = None
        if isinstance(row, list) and row and all(type(v) is int for v in row):
            rows.append(row)
            continue
        if rows and valid_grid(rows):
            best = rows
        rows = []
    return best


def openrouter_key():
    for line in (ROOT / "SECRETS.yaml").read_text().splitlines():
        if line.startswith("openrouter_api_key:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise KeyError("openrouter_api_key not found in SECRETS.yaml")


def openrouter_balance(key):
    """Remaining OpenRouter credit in USD (never logs the key)."""
    import httpx
    r = httpx.get(f"{OPENROUTER_URL}/credits", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    d = r.json()["data"]
    return float(d["total_credits"]) - float(d["total_usage"])


def openrouter_messages(messages):
    """E037 --openrouter-provider: OpenRouter takes previous-turn thinking as `reasoning`, not `reasoning_content`."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and "reasoning_content" in m:
            m = {**{k: v for k, v in m.items() if k != "reasoning_content"}, "reasoning": m["reasoning_content"]}
        out.append(m)
    return out


def compat_extra_body(args, think):
    """--base-url (OpenAI-compatible self-hosted server): request extras from --extra-body / --extra-body-nothink."""
    body = dict(json.loads(args.extra_body or "{}") if think else json.loads(args.extra_body_nothink or args.extra_body or "{}"))
    return body


class ContextFull(RuntimeError):
    """E046 --context-limit: the next turn's prompt leaves less than CONTEXT_MIN_ROOM tokens before the server's
    max-model-len; the attempt ends like an output-budget stop (session-answer fallback) instead of a 400 error."""


CONTEXT_MIN_ROOM = 1024


class TurnError(RuntimeError):
    def __init__(self, message, assistant, usage, finish):
        super().__init__(message)
        self.assistant, self.usage, self.finish = assistant, usage, finish


def turn(client, args, messages, max_tokens, final=False, think=True, tok=None):
    """Assemble streamed reasoning/content and indexed, fragmented tool calls."""
    if tok is not None and think and not final:
        return turn_token_exact(client, args, messages, max_tokens, tok)
    content, reasoning, calls, usage, finish = [], [], {}, None, None
    stream, failure = None, None
    try:
        if getattr(args, "openrouter_provider", ""):
            # E037: one pinned OpenRouter provider, no fallbacks; OpenRouter's own reasoning and usage parameters.
            or_reason = json.loads(args.or_reasoning) if args.or_reasoning else {"effort": args.effort}
            extra = {"provider": {"order": [args.openrouter_provider], "allow_fallbacks": False},
                     "reasoning": or_reason if think else {"enabled": False}, "usage": {"include": True}}
            send = openrouter_messages(messages)
        else:
            extra = (compat_extra_body(args, think) if getattr(args, "base_url", None) else
                     ({"reasoning_effort": args.effort, "thinking": {"type": "enabled"}} if think
                      else {"thinking": {"type": "disabled"}}))
            send = messages
        stream = client.chat.completions.create(
            model=args.model, messages=send, tools=tools_for(args),
            tool_choice="none" if final else "auto", max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=extra,
        )
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            d = choice.delta
            if d is not None:
                content.append(d.content or "")
                reasoning.append(getattr(d, "reasoning_content", None) or (d.model_extra or {}).get("reasoning") or "")
                for tc in d.tool_calls or []:
                    call = calls.setdefault(tc.index, {"id": "", "type": "function",
                                                      "function": {"name": "", "arguments": ""}})
                    call["id"] += tc.id or ""
                    if tc.function:
                        call["function"]["name"] += tc.function.name or ""
                        call["function"]["arguments"] += tc.function.arguments or ""
            finish = choice.finish_reason or finish
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            stream.close()
    assistant = {"role": "assistant", "content": "".join(content),
                 "reasoning_content": "".join(reasoning)}
    if calls:
        assistant["tool_calls"] = [calls[i] for i in sorted(calls)]
    if failure or finish is None or usage is None:
        raise TurnError(failure or "Incomplete stream: missing finish reason or usage", assistant, usage, finish)
    if any(not c["id"] or not c["function"]["name"] for c in calls.values()):
        raise TurnError("Incomplete streamed tool call", assistant, usage, finish)
    return assistant, usage, finish


# E043 --token-exact: DeepSeek-V4.1 completion format (the model's encoding/encoding.py, V4.1 DSML).
DS_EOS, DS_THINK_END, DSML = "<｜end▁of▁sentence｜>", "</think>", "｜DSML｜"
DS_CALLS_START, DS_CALLS_END = f"\n\n<{DSML} calls", f"</{DSML} calls>"
DS_INVOKE, DS_INVOKE_END = f"<{DSML} invoke", f"</{DSML} invoke"
DS_PARAM, DS_PARAM_END = f"<{DSML} parameter", f"/{DSML} parameter"


def _read_until(text, index, stops):
    hits = [(text.find(s, index), s) for s in stops]
    hits = [(p, s) for p, s in hits if p != -1]
    if not hits:
        return len(text), text[index:], None
    p, s = min(hits)
    return p + len(s), text[index:p], s


def _parse_ds_strict(text):
    """Raw V4.1 thinking-mode completion (after the prompt's `<think>`) -> (reasoning, content, tool_calls), like the
    model's parse_message_from_completion_text. Raises ValueError on any format error."""
    i, reasoning, stop = _read_until(text, 0, [DS_THINK_END, DS_CALLS_START])
    if stop != DS_THINK_END:
        raise ValueError("missing </think>")
    i, content, stop = _read_until(text, i, [DS_EOS, DS_CALLS_START])
    calls = []
    if stop == DS_CALLS_START:
        while True:
            i, gap, stop = _read_until(text, i, [DS_INVOKE, DS_CALLS_END])
            if gap != ">\n" or stop is None:
                raise ValueError("tool call block format")
            if stop == DS_CALLS_END:
                break
            i, head, stop = _read_until(text, i, [DS_PARAM, DS_INVOKE_END])
            name = re.findall(r'^\s*name="(.*?)">\n$', head, flags=re.DOTALL)
            if len(name) != 1:
                raise ValueError("tool name format")
            args = {}
            while stop == DS_PARAM:
                i, body, stop = _read_until(text, i, [DS_PARAM_END])
                kv = re.findall(r'^ name="(.*?)" string="(true|false)">(.*?)<$', body, flags=re.DOTALL)
                if len(kv) != 1 or kv[0][0] in args:
                    raise ValueError("parameter format")
                key, is_str, value = kv[0]
                args[key] = value if is_str == "true" else json.loads(value)
                i, gap, stop = _read_until(text, i, [DS_PARAM, DS_INVOKE_END])
                if gap != ">\n":
                    raise ValueError("parameter end format")
            calls.append({"id": f"call_{uuid4().hex[:24]}", "type": "function",
                          "function": {"name": name[0], "arguments": json.dumps(args, ensure_ascii=False)}})
        i, tail, stop = _read_until(text, i, [DS_EOS])
        if tail:
            raise ValueError("text after tool calls")
    elif stop != DS_EOS:
        raise ValueError("missing EOS")
    if i != len(text):
        raise ValueError("text after EOS")
    if any(t in s for s in (reasoning, content) for t in (DS_EOS, DS_THINK_END, DSML)):
        raise ValueError("special token inside text")
    return reasoning, content, calls


DS_TOOL_START = f"<{DSML} calls>"
DS_INVOKE_RE = re.compile(rf'<{DSML} invoke name="(.*?)">(.*?)(?=</{DSML} invoke>|$)', re.DOTALL)
DS_PARAM_RE = re.compile(rf'<{DSML} parameter\s+name="([^"]+)"\s+string="(true|false)">(.*?)'
                         rf'(?:</{DSML} parameter>|(?=<{DSML} parameter\s+name=))', re.DOTALL)


def _parse_ds_lenient(text):
    """Like the server's deepseek_v41 parser (vllm/parser/deepseek_v4.py state machine): reasoning ends at the first
    `</think>` or at a calls block opened inside the thinking; an invoke ends at `</invoke>` or its block's end; a
    parameter ends at its close tag or the next parameter; a non-JSON `string="false"` value stays a string; think tags
    in the content are dropped. Never raises."""
    text = text.replace(DS_EOS, "")
    t, e = text.find(DS_TOOL_START), text.find(DS_THINK_END)
    if e != -1 and (t == -1 or e < t):
        reasoning, rest = text[:e], text[e + len(DS_THINK_END):]
    elif t != -1:
        reasoning, rest = text[:t], text[t:]
    else:
        return text, "", []
    content, calls, i = [], [], 0
    while (t := rest.find(DS_TOOL_START, i)) != -1:
        content.append(rest[i:t])
        end = rest.find(DS_CALLS_END, t)
        block = rest[t + len(DS_TOOL_START):end if end != -1 else len(rest)]
        for name, body in DS_INVOKE_RE.findall(block):
            args = {}
            for key, is_str, value in DS_PARAM_RE.findall(body):
                try:
                    args[key] = value if is_str == "true" else json.loads(value)
                except ValueError:
                    args[key] = value
            calls.append({"id": f"call_{uuid4().hex[:24]}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        if end == -1:
            i = len(rest)
            break
        i = end + len(DS_CALLS_END)
    content.append(rest[i:])
    content = "".join(content).replace(DS_THINK_END, "").replace("<think>", "")
    return reasoning, content.rstrip("\n") if calls else content, calls


def parse_ds_completion(text):
    """-> (reasoning, content, tool_calls, clean). clean: the text is in the encoding's canonical form, so the sampled
    ids are a faithful history; otherwise it is parsed leniently like the server's parser."""
    try:
        return (*_parse_ds_strict(text), True)
    except ValueError:
        return (*_parse_ds_lenient(text), False)


def turn_token_exact(client, args, messages, max_tokens, tok):
    """E043 --token-exact (--base-url only): continue the attempt's exact token history instead of letting the server
    re-render earlier assistant turns. Re-tokenizing sampled text does not always give back the sampled ids (the
    model sometimes samples a non-canonical split; ~13% of turns), which ends the cached prefix inside that turn; on a
    small KV pool (4xL4) the state needed to resume there is evicted. Only the messages appended since the last turn
    are rendered (server /tokenize: full render minus the render through the last assistant turn), and generation
    goes to /v1/completions with the token ids. Canonical output is parsed like the model's encoding; anything else is
    parsed leniently like the server's tool parser. The next turn is rendered from the messages (as on the chat path)
    instead of the sampled ids when this turn was not canonical, was cut by max_tokens, or the harness rewrote it
    (rescue / tool-name fix: tok["resync"]), so the model never sees malformed history the chat path would have shown
    cleanly. If the render is not prefix-stable (e.g. a new user message drops earlier reasoning), the turn uses the
    full render too."""
    import httpx
    root = str(client.base_url).rstrip("/").removesuffix("/v1")
    body = dict(json.loads(args.extra_body or "{}"))
    kw = body.pop("chat_template_kwargs", {})
    tools = tools_for(args)

    def render(msgs, gen):
        r = httpx.post(f"{root}/tokenize", json={"model": args.model, "messages": msgs, "tools": tools,
                                                 "chat_template_kwargs": kw, "add_generation_prompt": gen},
                       timeout=600)
        r.raise_for_status()
        return r.json()["tokens"]

    full = render(messages, True)
    if tok["ids"] is None or tok["resync"]:
        ids = full
        tok["resyncs"] += tok["ids"] is not None
    else:
        upto = render(messages[:tok["n"]], False)
        if full[:len(upto)] == upto:
            ids = tok["ids"] + full[len(upto):]
            # a break: re-rendering the last assistant turn does not give back its sampled ids (it starts where the
            # previous turn's prompt render ended)
            tok["breaks"] += upto[tok["prev_len"]:] != tok["prev_sampled"]
        else:
            ids = full
            tok["rerenders"] += 1
    limit = getattr(args, "context_limit", 0)
    if limit:  # E046: never request past the server's max-model-len (prompt + max_tokens), which fails the attempt
        room = limit - len(ids) - 16
        if room < CONTEXT_MIN_ROOM:
            raise ContextFull(f"prompt {len(ids)} tokens, limit {limit}")
        max_tokens = min(max_tokens, room)
    text, sampled, usage, finish, failure, stream = [], [], None, None, None, None
    try:
        stream = client.completions.create(
            model=args.model, prompt=ids, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
            extra_body={**body, "return_token_ids": True, "skip_special_tokens": False,
                        "spaces_between_special_tokens": False})
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            for ch in chunk.choices or []:
                text.append(ch.text or "")
                sampled += (ch.model_extra or {}).get("token_ids") or []
                finish = ch.finish_reason or finish
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            stream.close()
    text = "".join(text)
    reasoning, content, calls, clean = parse_ds_completion(text if text.endswith(DS_EOS) or finish != "stop"
                                                           else text + DS_EOS)
    tok["parse_fallbacks"] += not clean and finish != "length"
    tok["resync"] = not clean or finish == "length"
    assistant = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        assistant["tool_calls"] = calls
        finish = "tool_calls" if finish == "stop" else finish
    if failure or finish is None or usage is None:
        raise TurnError(failure or "Incomplete stream: missing finish reason or usage", assistant, usage, finish)
    tok["ids"], tok["n"], tok["prev_len"], tok["prev_sampled"] = ids + sampled, len(messages) + 1, len(full), sampled
    tok["turns"] += 1
    return assistant, usage, finish


def add_usage(total, usage):
    total["prompt"] += usage.get("prompt_tokens", 0) or 0
    total["cache_hit"] += (usage.get("prompt_cache_hit_tokens")  # DeepSeek; OpenRouter reports cached_tokens
                           or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    total["completion"] += usage.get("completion_tokens", 0) or 0


def estimated_cost(total, model, multiplier):
    hit, miss, output = PRICES.get(model, (0.0, 0.0, 0.0))
    return multiplier * (total["cache_hit"] * hit + max(0, total["prompt"] - total["cache_hit"]) * miss
                         + total["completion"] * output) / 1e6


def add_cost(cost, usage, model, multiplier):
    """Accumulate one turn's cost at the multiplier in force when its request started (E017 --auto-peak-price)."""
    t = {"prompt": 0, "cache_hit": 0, "completion": 0}
    add_usage(t, usage)
    # OpenRouter (E037) reports the billed cost per request; DeepSeek runs are priced from the token counts.
    c = float(usage["cost"]) if usage.get("cost") is not None else estimated_cost(t, model, multiplier)
    cost["total"] += c
    if multiplier > 1:
        cost["peak"] += c
        cost["peak_turns"] += 1


def budget_note(args, used):
    """E009 --budget-notes: a status line appended to tool output once `used` passes the threshold."""
    if not args.budget_notes or used < args.budget_notes * args.max_total_output:
        return ""
    soft = getattr(args, "budget_soft", 0)
    if soft:
        note = (f"\n[budget: {used / 1000:.0f}k output tokens used; aim to finish within {soft / 1000:.0f}k "
                f"(hard limit {args.max_total_output / 1000:.0f}k)]")
        if soft <= used < args.budget_warn * args.max_total_output:
            note += (" Past the target: submit your best answer now with submit_answer, unless one more experiment "
                     "will very likely make every training pair pass." if getattr(args, "soft_wording", "gentle") == "strong"
                     else " Past the target: submit as soon as a rule fits every training pair.")
    else:
        note = f"\n[budget: {used / 1000:.0f}k of {args.max_total_output / 1000:.0f}k output tokens used]"
    if used >= args.budget_warn * args.max_total_output:
        note += (" Budget nearly spent: submit your best answer now with submit_answer (or at least set `answer`)."
                 if args.answer_tool else " Budget nearly spent: give your best answer now.")
    return note


def context_budget(args, usage, used):
    """A9 (--context-limit): the args the budget notes see, with the hard budget capped at the output that still fits
    before the context limit (this turn's prompt + output, the 16-token margin and CONTEXT_MIN_ROOM), so an attempt
    the context ends first (--multi-budget: 160k for n = 3 in a 163,840 context) still gets the "nearly spent" note."""
    limit = getattr(args, "context_limit", 0)
    if not limit or not usage:
        return args
    ctx = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    cap = used + max(0, limit - ctx - 16 - CONTEXT_MIN_ROOM)
    if cap >= args.max_total_output:
        return args
    import copy
    capped = copy.copy(args)
    capped.max_total_output = max(1, cap)
    if capped.budget_soft and capped.budget_soft >= capped.max_total_output:
        capped.budget_soft = 0
    return capped


# E030: a "below low" effort header, written like DeepSeek's own effort prefixes (notes/prompts/
# deepseek_v4_0813_reasoning_effort_prefixes.txt: `high`/`max` prepend one, `low` adds nothing), sent as a system message.
EFFORT_HEADERS = {
    "minimal": ("Reasoning Effort: Minimal. Keep your thinking short and terse: write brief notes, not full sentences. "
                "Do not restate the task, do not re-read or re-derive what code has already printed, and do not re-verify "
                "what has already been checked. Test ideas with small code experiments instead of long deliberation, and "
                "submit as soon as the rule reproduces every training pair."),
}
CAP_NOTE = ("[per-turn thinking limit: this turn reached {k}k tokens without running code, so it was stopped here. "
            "Do not keep deliberating: write the smallest code experiment that tests your current best hypothesis on "
            "the training pairs and run it now.]")


def turns_note(args, index):
    """E029 --turns-note N: after assistant turn `index` (0-based), warn once N or fewer turns remain."""
    n = getattr(args, "turns_note", 0)
    left = args.max_turns - index - 1
    if not n or left > n:
        return ""
    if left <= 0:
        return ""
    return (f"\n[{left} turn{'s' if left != 1 else ''} left: call submit_answer with your best answer now, "
            "or at least set `answer`.]")


# I7 --best-guess-at / --best-guess-turns: E128 runs 32 of its 91 never-train-passing attempts to the 80k budget or the
# 40-turn cap without an answer (0 correct) although the budget and turn notes say "submit now"; full V4.1 submits a
# best guess in 74 of 82 such attempts (43 right). One firm note, once per attempt, after the tool output.
BEST_GUESS_NOTE = ("\n[Stop exploring now] Call submit_answer with your best current answer, even if it does not "
                   "reproduce every training pair. An imperfect answer can still score; no answer scores zero.")
BEST_GUESS_MULTI = " Submit all {n} test grids."


def best_guess_due(args, budget, used, turns):
    """I7: True when the best-guess note is due on the tool result of assistant turn `turns` (1-based): `used` output
    tokens >= --best-guess-at x `budget` (the attempt's hard budget as the budget notes see it: scaled under
    --multi-budget, capped at the room left under --context-limit) or turns >= max_turns - --best-guess-turns, while
    another turn can still follow (no point in a note the model never reads)."""
    f, k = getattr(args, "best_guess_at", 0), getattr(args, "best_guess_turns", 0)
    if not (f or k) or used >= budget or turns >= args.max_turns:
        return False
    return bool((f and used >= f * budget) or (k and turns >= args.max_turns - k))


# I8 --post-pass-note: E128 (effort 50, E050) spends 14% of its output after its code first reproduces every training
# pair, 81% of it in 20 attempts with > 10K post-pass tokens (0.75 vs 0.90 for short tails: partly productive rework).
# One note, once per attempt, on the first tool result that shows a program passing every training pair.
POST_PASS_NOTE = ("\n[All training pairs pass] Check with code any assumption about the test input that the training "
                  "pairs do not cover (sizes, colours, object counts), then submit_answer.")
POST_PASS_MULTI = " Submit all {n} test grids."
_PP_SKIP = re.compile(r"Traceback|\w+Error:|\w+Exception:|\[output truncated\]|\[Python state lost")
_PP_NEG = re.compile(r"\bFalse\b|\bFAIL|\bfail(?:ed|s|ure)?\b|\bwrong\b|\bincorrect\b|✗|\bnot (?:ok|equal|match)"
                     r"|\bno match|\bdiffer(?:s|ent|ence)?\b"
                     r"|\b(?:mismatch(?:es|ed)?|errors?|bad)\b(?!\D{0,4}\b0\b)", re.I)  # "errors 0" is fine
_PP_COUNT = re.compile(r"\b(?:diffs?|ndiffs?|n_diff|diff_count|mismatch(?:es)?|errors?|wrong|nwrong|bad)\b"
                       r"[^\d\n]{0,12}?(\d+)", re.I)  # a nonzero count is a failure
_PP_WRAP = {"array", "asarray", "list", "tuple"}  # f(grid) that keeps the grid
_PP_METHOD = {"tolist", "copy", "astype"}  # grid.m() that keeps the grid
_PP_COMP = (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)


def _pp_nodes(tree):
    """The cell's AST nodes outside function, class and lambda bodies."""
    todo = [tree]
    while todo:
        node = todo.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            todo.extend(ast.iter_child_nodes(node))


def _pp_slice(node):
    return node.slice.value if type(node.slice).__name__ == "Index" else node.slice  # Python < 3.9: ast.Index


def _pp_key(node, key):
    """x['key'] or x.get('key')."""
    if isinstance(node, ast.Subscript):
        s = _pp_slice(node)
        return isinstance(s, ast.Constant) and s.value == key
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
            and bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == key)


def _pp_over_train(it):
    """train, enumerate(train), zip(train, ...) or range(len(train)): every training pair, never a slice."""
    if isinstance(it, ast.Name):
        return it.id == "train"
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.args):
        return False
    if it.func.id in ("enumerate", "zip"):
        return _pp_over_train(it.args[0])
    a = it.args[0]
    return (it.func.id == "range" and len(it.args) == 1 and isinstance(a, ast.Call) and isinstance(a.func, ast.Name)
            and a.func.id == "len" and len(a.args) == 1 and isinstance(a.args[0], ast.Name) and a.args[0].id == "train")


def _pp_per_pair(target, it):
    """Names a loop over train binds per pair besides the pair itself: enumerate / range(len(train)) indices and the
    partners of zip(train, ...)."""
    if isinstance(it, ast.Name):
        return []
    elts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
    if it.func.id == "enumerate" and len(elts) == 2:
        return _pp_targets(elts[0]) + _pp_per_pair(elts[1], it.args[0])
    if it.func.id == "zip" and len(elts) == len(it.args):
        return _pp_per_pair(elts[0], it.args[0]) + [n for e in elts[1:] for n in _pp_targets(e)]
    return [n for e in elts for n in _pp_targets(e)]  # range(len(train)), or a target shape not understood


def _pp_targets(target):
    """Names bound (or item-assigned) by an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return [target.value.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for t in target.elts for n in _pp_targets(t)]
    if isinstance(target, ast.Starred):
        return _pp_targets(target.value)
    return []


def train_pass_shown(code, output, n_train):
    """I8: True when a run_python result (`output`, as the model sees it) of cell `code` shows a program reproducing
    every one of the n_train training pairs. Conservative, since a miss only delays the note and a false alarm
    misleads: the cell completed without a traceback or truncation and no output line (lines naming the test input
    aside) has a negative verdict (False, fail, mismatch, differ, wrong, a nonzero diff count, ...); and either the
    helpers' check(fn) printed ALL PASS, or, outside any function, a loop or comprehension over all of `train` compares
    (== or array_equal) a function's result on the pair's input (a call of a non-builtin name whose arguments read
    the pair's 'input' and never an 'output') with the pair's 'output' grid, and prints it, with `True` on at least
    n_train lines (one line for a printed comprehension or all()), or asserts it."""
    try:
        return _train_pass_shown(code, output, n_train)
    except Exception:  # e.g. SyntaxError (IPython magics) or RecursionError: no note, never a broken turn
        return False


def _train_pass_shown(code, output, n_train):
    head, sep, rest = output.partition("[completed]\n")
    if not sep or (head and not head.startswith("[auto-corrected")) or _PP_SKIP.search(rest) or n_train < 1:
        return False
    lines = [ln for ln in rest.split("\n") if not re.search(r"\btest", ln, re.I)]
    if any(_PP_NEG.search(ln) or any(int(m.group(1)) for m in _PP_COUNT.finditer(ln)) for ln in lines):
        return False
    tree = ast.parse(code)
    nodes = list(_pp_nodes(tree))
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    if any(ln.strip() == "ALL PASS" for ln in lines) and any(  # the helpers' check(fn), on every training pair
            isinstance(c.func, ast.Name) and c.func.id == "check" and len(c.args) == 1 and not c.keywords
            for c in calls):
        return True
    parent, assigns = {}, {}
    for node in nodes:
        for child in ast.iter_child_nodes(node):
            parent[child] = node
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for name in _pp_targets(target):
                    assigns.setdefault(name, []).append(node)
    printed = {n.id for c in calls if isinstance(c.func, ast.Name) and c.func.id == "print"
               for a in c.args for n in ast.walk(a) if isinstance(n, ast.Name)}
    index = {n for node in nodes for target, it in (
        [(node.target, node.iter)] if isinstance(node, ast.For) else
        [(g.target, g.iter) for g in node.generators] if isinstance(node, _PP_COMP) else [])
        if _pp_over_train(it) for n in _pp_per_pair(target, it)}

    def bound(name, at):
        """What `name` holds at node `at`: its last assignment before `at` in the cell (or every one if none)."""
        stmts = assigns.get(name, [])
        before = [s for s in stmts if (s.lineno, s.col_offset) < (at.lineno, at.col_offset)]
        return [max(before, key=lambda s: (s.lineno, s.col_offset)).value] if before else [s.value for s in stmts]

    def reads(node, key):  # the expression or, through the cell's assignments, anything it was built from
        seen, todo = set(), [node]
        while todo:
            for n in ast.walk(todo.pop()):
                if _pp_key(n, key):
                    return True
                if isinstance(n, ast.Name) and n.id not in seen:
                    seen.add(n.id)
                    todo.extend(s.value for s in assigns.get(n.id, []))
        return False

    def is_output(node, depth=0):  # the pair's output grid itself (maybe as an array or a list)
        if _pp_key(node, "output"):
            return True
        if depth > 6:
            return False
        if isinstance(node, ast.Name):
            vals = bound(node.id, node)
            return bool(vals) and all(is_output(v, depth + 1) for v in vals)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr
            if name in _PP_WRAP and node.args:
                return is_output(node.args[0], depth + 1)
            return isinstance(f, ast.Attribute) and name in _PP_METHOD and is_output(f.value, depth + 1)
        return False

    def is_result(node, depth=0):  # a non-builtin function's result on the pair's input (maybe wrapped or indexed)
        if depth > 6:
            return False
        if isinstance(node, ast.Name):
            vals = bound(node.id, node)
            return bool(vals) and all(is_result(v, depth + 1) for v in vals)
        if isinstance(node, ast.Subscript):
            return is_result(node.value, depth + 1)
        if not (isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))):
            return False
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr
        if name in _PP_WRAP and node.args:
            return is_result(node.args[0], depth + 1)
        if isinstance(f, ast.Attribute):
            return name in _PP_METHOD and is_result(f.value, depth + 1)
        args = [*node.args, *(k.value for k in node.keywords)]
        return (not hasattr(builtins, name) and any(reads(a, "input") for a in args)
                and not any(reads(a, "output") or per_pair(a) for a in args))

    def per_pair(node):
        """Built from a loop index or zip partner other than as train[i], also through branches on one (hand-fit
        per-pair arguments such as [2, 2, 6][i] or `s = 7 if i == 0 else 6` are not a program)."""
        seen, todo = set(), [node]
        while todo:
            for n in ast.walk(todo.pop()):
                if not isinstance(n, ast.Name):
                    continue
                p = parent.get(n)
                if n.id in index and not (isinstance(p, ast.Subscript) and _pp_slice(p) is n
                                          and isinstance(p.value, ast.Name) and p.value.id == "train"):
                    return True
                if n.id not in seen:
                    seen.add(n.id)
                    for stmt in assigns.get(n.id, []):
                        todo.append(stmt.value)
                        x = stmt
                        while x in parent and not isinstance(x, ast.For):
                            x = parent[x]
                            if isinstance(x, (ast.If, ast.While)):
                                todo.append(x.test)
        return False

    def in_train_for(node):
        x = node
        while x in parent:
            if isinstance(parent[x], ast.For) and _pp_over_train(parent[x].iter) and x is not parent[x].iter:
                return True
            x = parent[x]
        return False

    def use(node):
        """How the comparison's value reaches the output, through f-strings, tuples, bool()/str() and at most one
        list comprehension or all(generator) over train: ("print" | "assert" | None, via such a comprehension)."""
        x, comp = node, False
        while x in parent:
            p = parent[x]
            if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id == "print" and x is not p.func:
                return "print", comp
            if isinstance(p, ast.Assert) and x is p.test:
                return "assert", comp
            if isinstance(p, (ast.Assign, ast.AnnAssign)) and x is p.value:
                targets = p.targets if isinstance(p, ast.Assign) else [p.target]
                return ("print" if any(n in printed for t in targets for n in _pp_targets(t)) else None), comp
            if (isinstance(p, (ast.ListComp, ast.GeneratorExp)) and x is p.elt and not comp
                    and any(_pp_over_train(g.iter) for g in p.generators)):
                comp = True
            elif not ((isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.args and x is p.args[0]
                       and ((p.func.id == "all" and comp and isinstance(x, (ast.GeneratorExp, ast.ListComp)))
                            or p.func.id in ("bool", "str")))
                      or isinstance(p, (ast.FormattedValue, ast.JoinedStr, ast.Tuple, ast.List))):
                return None, comp
            x = p
        return None, comp

    kinds = set()
    for node in nodes:
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            a, b = node.left, node.comparators[0]
        elif (isinstance(node, ast.Call) and len(node.args) == 2 and isinstance(node.func, (ast.Name, ast.Attribute))
              and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == "array_equal"):
            a, b = node.args
        else:
            continue
        if not ((is_result(a) and is_output(b)) or (is_result(b) and is_output(a))):
            continue
        how, comp = use(node)
        if how is None or not (comp or in_train_for(node)):
            continue
        if how == "assert":
            return True  # asserted for every training pair, and the cell raised nothing
        kinds.add("one" if comp else "for")
    true_lines = sum(bool(re.search(r"\bTrue\b", ln)) for ln in lines)
    return bool(kinds) and (true_lines >= n_train or ("one" in kinds and true_lines >= 1))


IMAGE_NOTE = ("\n\nThe images below show the SAME training examples and test input(s) as the JSON above, in the same "
              "order; they are not additional examples. Each image is one grid, drawn as flat coloured squares "
              "without lines (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 light blue, "
              "9 maroon). Use them to see shapes and layout; the JSON grids specify the exact cells.")


def text_image_content(prompt, train, test_inputs, cell=0, align="deepseek"):
    """E058 --grid-format text+image: chat content parts = the full text prompt, a note, then one labelled image per
    grid (training inputs and outputs, then test inputs). cell 0 = grid_image.aligned_cell(align): each cell exactly
    k x k model tiles (DeepSeek 42 px, Qwen 32 px), never rescaled by the API; otherwise a fixed cell size in pixels."""
    def image(label, g):
        return [{"type": "text", "text": f"\n{label} ({len(g)}x{len(g[0])}):"},
                {"type": "image_url", "image_url": {"url": data_uri(g, cell or aligned_cell(g, align))}}]
    parts = [{"type": "text", "text": prompt + IMAGE_NOTE}]
    for i, pair in enumerate(train):  # numbered from 0, like the text prompt's --Example 0--
        parts += image(f"Example {i} input", pair["input"]) + image(f"Example {i} output", pair["output"])
    for j, g in enumerate(test_inputs, 1):
        parts += image("Test input" if len(test_inputs) == 1 else f"Test input {j - 1}", g)
    return parts


def build_prompt_multi(train, test_inputs):
    """E046 --multi-test: the ARC Prize prompt with every test input of the task, numbered."""
    tests = "\n\n".join(f"--Test input {i}--\n\n{json.dumps(g)}" for i, g in enumerate(test_inputs))
    return build_prompt(train, [[0]]).replace(json.dumps([[0]]) + "\n--End of Test Input--", tests + "\n--End of Test Input--")


MULTI_NOTE = ("\n\nThis task has {n} test inputs, shown above in order and preloaded in the session as `test_inputs` "
              "(a list of {n} grids; test_input is test_inputs[0]). One rule must work for all of them: check its "
              "assumptions on every test input. Submit a list with one output grid per test input, in order, e.g. "
              "submit_answer with `[solve(t) for t in test_inputs]`, and keep `answer` as that list.")


# E046 --multi-prompt: other wordings of the multi-test note ("note" = MULTI_NOTE). "first" shows only test input 0 in
# the prompt (the others are in the session only), for a shorter prompt and less KV per attempt.
MULTI_NOTES = {
    "note": MULTI_NOTE,
    "lean": ("\n\nThis task has {n} test inputs, preloaded as `test_inputs` (test_input is test_inputs[0]). Submit "
             "`[solve(t) for t in test_inputs]`."),
    "early": MULTI_NOTE + (" Define solve(grid) as soon as you have a rule and set `answer = [solve(t) for t in "
                           "test_inputs]` after every fix, so an answer always exists."),
    "first": ("\n\nThis task has {n} test inputs. Only test input 0 is shown above; all {n} are preloaded in the session "
              "as `test_inputs` (test_input is test_inputs[0]). Print the others' sizes and grids if you need them. One "
              "rule must work for all of them. Submit a list with one output grid per test input, in order, e.g. "
              "submit_answer with `[solve(t) for t in test_inputs]`, and keep `answer` as that list."),
}


# E047 --prior-mode: attempts >= 1 see attempt 0's answer (from --prior-file). {status} is filled only when attempt 0's
# program passed the host's train check (the check misses many correct answers, so a fail is not reported).
PRIOR_NOTES = {
    "verify": ("\n\nA previous attempt at this task submitted the answer below{status}. It may be right or wrong. "
               "Check it against the training pairs and the test input{s}: if you confirm it, resubmit it (that is "
               "cheap); if you find an error, solve the task and submit your corrected answer.\n\n{grids}"),
    # verifycode: the same, plus the program behind the answer (rebuilt from the transcript; never the thinking)
    "verifycode": ("\n\nA previous attempt at this task submitted the answer below{status}. It may be right or wrong. "
                   "Its program and answer follow; the program is not loaded in your session. Check the rule against "
                   "the training pairs and the test input{s}: if you confirm it, resubmit the answer (that is cheap); if "
                   "you find an error, fix the rule and submit your corrected answer.\n\n{program}{grids}"),
    # verifytest (used by hybrid when the train check passed): the rule fits training, so audit its test extension
    "verifytest": ("\n\nA previous attempt at this task submitted the answer below. Its program, shown below, reproduces "
                   "every training pair, so any error is in how the rule extends to the test input{s}. The program is "
                   "not loaded in your session. Compare the test input{s} with the training inputs: look for anything "
                   "the training pairs never show (new colours, sizes, counts, overlaps, edge contact) where the rule "
                   "has to guess. If you find such a case, decide the most plausible behaviour from the training pairs "
                   "and submit the corrected answer; otherwise resubmit the answer (that is cheap).\n\n{program}{grids}"),
    "differ": ("\n\nThis is the second of two attempts; the task counts as solved if either answer is right. The first "
               "attempt submitted the answer below{status}. Assume it is wrong: find the most plausible rule that fits "
               "every training pair but gives a different test output, and submit that. If after checking you find "
               "no such rule, resubmit the first answer.\n\n{grids}"),
}


# E049 --alt-answer: attempt 0 may record a second guess in `alt_answer` (Kaggle scores two tries per test output).
ALT_NOTES = {
    "passive": ("\n\nOptional second guess: if you meet a different rule that also reproduces every training pair but "
                "gives a different test output, set `alt_answer` to that output (same format as your answer) before "
                "submitting. It is scored as a second try, so record it only when it is a real alternative; do not "
                "spend extra effort looking for one."),
    "active": ("\n\nYou get two tries per test output. Before submitting, briefly ask whether a different rule also "
               "reproduces every training pair but gives a different test output; if so, set `alt_answer` to that "
               "output (same format as your answer). Your main answer is still the one you submit."),
}


def prior_note(mode, preds, train_pass, program=None):
    """E047: the note for attempt >= 1; preds = attempt 0's grids, one per test input shown in this attempt."""
    if not preds or any(g is None for g in preds):
        return ""
    if mode == "hybrid":  # E047: train check passed -> audit the test extension; else check and fix the rule
        mode = "verifytest" if train_pass else "verifycode"
    if mode == "split":  # E047: train check passed -> answer only (verify); else answer + program (verifycode)
        mode = "verify" if train_pass else "verifycode"
    if mode in ("verifycode", "verifytest") and not program:
        mode = "verify"
    grids = "\n\n".join((f"--Previous answer for test input {i}--\n\n" if len(preds) > 1 else "--Previous answer--\n\n")
                         + json.dumps(g) for i, g in enumerate(preds))
    status = " (its program reproduced every training pair)" if train_pass else ""
    program = f"--Previous program--\n\n```python\n{program}\n```\n\n" if program else ""
    return PRIOR_NOTES[mode].format(status=status, s="s" if len(preds) > 1 else "", grids=grids, program=program)


def load_prior(path, programs=False):
    """E047 --prior-file: {(task, pair): (pred, train_pass, program)} from attempt 0 records (latest record wins).
    programs (verifycode): the answer program rebuilt from the pair's transcript (evals/answer_program.py)."""
    prior = {}
    for r in map(json.loads, open(path)):
        if "task_id" in r and r.get("attempt") == 0 and "error" not in r:
            prior[r["task_id"], r["pair"]] = (r.get("pred"), bool(r.get("train_pass")), r.get("trace_path"))
    if programs:
        from answer_program import answer_program
        for k, (pred, tp, trace) in prior.items():
            try:
                with gzip.open(trace, "rt") as f:
                    prior[k] = (pred, tp, answer_program(json.load(f)["messages"]))
            except (OSError, ValueError, KeyError, TypeError):
                prior[k] = (pred, tp, None)
    return {k: (v[0], v[1], v[2] if programs else None) for k, v in prior.items()}


def prior_for(args, task_id, task, pair, attempt, multi):
    """E047: attempt 0's grids for this attempt's test inputs, or None (attempt 0, option off, or no prior)."""
    if attempt < 1 or not getattr(args, "prior_mode", "") or getattr(args, "_prior", None) is None:
        return None
    ps = range(len(task["test"])) if multi else [pair]
    got = [args._prior.get((task_id, q)) for q in ps]
    if any(g is None for g in got):
        return None
    return {"preds": [g[0] for g in got], "train_pass": got[0][1], "program": got[0][2]}


def solve(client, args, task, test_input, session_factory=SandboxSession, test_inputs=None, prior=None):
    multi = len(test_inputs) if test_inputs is not None and len(test_inputs) > 1 else None  # E046 --multi-test
    if multi and getattr(args, "multi_budget", 0):
        # E046 --multi-budget f: one rollout answers n test inputs, so it gets (1 + f (n - 1)) x the output budget,
        # the soft budget and the turn cap (E128 run 1: 19 of 90 multi attempts hit the 80k budget with no answer)
        import copy
        scale = 1 + args.multi_budget * (multi - 1)
        args = copy.copy(args)
        args.max_total_output = int(args.max_total_output * scale)
        args.budget_soft = int(args.budget_soft * scale) if args.budget_soft else args.budget_soft
        args.max_turns = int(round(args.max_turns * scale))
    if multi:
        test_input = test_inputs[0]
    extra = PROMPT_SUFFIXES[args.prompt]
    if args.mode == "code":
        extra += "\n\nUse run_python to investigate; submit self-contained transform(grid) code with submit_solution."
    if getattr(args, "helpers", False):
        extra += HELPERS_PROMPT
    if getattr(args, "task_summary", False):
        extra += task_summary(task["train"], test_input)
    if getattr(args, "grid_format", "json") == "image":
        # E033: the same prompt with every JSON grid replaced by an image (one model patch token per cell).
        prompt = build_image_prompt(ARC_TEMPLATE, task["train"], test_input, args.image_cell, suffix=extra)
    else:
        mp = getattr(args, "multi_prompt", "note")
        prompt = (build_prompt_multi(task["train"], test_inputs[:1] if mp == "first" else test_inputs) + extra
                  + MULTI_NOTES[mp].format(n=multi) if multi
                  else build_prompt(task["train"], test_input) + extra)
    if getattr(args, "alt_answer", ""):  # E049
        prompt += ALT_NOTES[args.alt_answer]
    if prior:  # E047 --prior-mode
        prompt += prior_note(args.prior_mode, prior["preds"], prior["train_pass"], prior.get("program"))
    if getattr(args, "grid_format", "json") == "text+image":  # E058: JSON prompt unchanged, plus one image per grid
        prompt = text_image_content(prompt, task["train"], test_inputs if multi else [test_input], args.image_cell,
                                    getattr(args, "image_align", "deepseek"))
    messages = [{"role": "user", "content": prompt}]
    if getattr(args, "effort_header", ""):
        messages.insert(0, {"role": "system", "content": EFFORT_HEADERS[args.effort_header]})
    total = {"prompt": 0, "cache_hit": 0, "completion": 0}
    n_tools, n_errors, n_turns, finish, stop, pred = 0, 0, 0, None, None, None
    n_nudges = 0  # E030 --cap-nudge
    n_rescues = 0  # E043 --rescue-code
    alt_pred = None  # E049 --alt-answer
    n_nudge_rescues = 0
    repair, repairs = None, None  # I5 --auto-repair: auto_repair.propose and its counters
    if getattr(args, "auto_repair", False):
        repairs = {"proposed": 0, "applied": 0, "retry_ok": 0}
        try:
            from auto_repair import propose as repair
        except Exception as exc:  # e.g. a Kaggle kit without evals/auto_repair.py
            repairs["unavailable"] = f"{type(exc).__name__}: {exc}"[:200]
    # E043 --token-exact: the attempt's exact token history (prompt + sampled ids) and counters
    tok = ({"ids": None, "n": 0, "prev_len": 0, "prev_sampled": [], "resync": False, "turns": 0, "breaks": 0,
            "rerenders": 0, "resyncs": 0, "parse_fallbacks": 0}
           if getattr(args, "token_exact", False) and getattr(args, "base_url", None) else None)
    events, error, content, t0 = [], None, "", time.monotonic()
    session_final = getattr(args, "final_mode", "turn") == "session"
    source = None  # where the single scored grid came from
    forced, submit_verified = set(), None  # E011 --verify-submit state
    executed, submit_code, labels = [], None, {}  # E017 --train-check state
    auto_price = getattr(args, "auto_peak_price", False)
    cost = {"total": 0.0, "peak": 0.0, "peak_turns": 0}
    helpers = HELPERS_SRC if getattr(args, "helpers", False) else ""
    # E046: the session's `test_inputs` (A5: also [test_input] in single-test sessions under --multi-test, which the
    # tool description promises); the train check's transcript replay binds test_inputs[0] to the replayed input (A6)
    all_inputs = test_inputs if multi else [test_input]
    with_inputs = bool(multi) or getattr(args, "multi_test", False)

    def session_extra(inputs):
        text = helpers + (inputs_prelude(inputs) if with_inputs else "")
        return {"prelude_extra": text} if text else {}
    extra = session_extra(all_inputs)
    cut_parse = False  # A2: pred came from the text of a turn cut by max_tokens
    guess_on = bool(getattr(args, "best_guess_at", 0) or getattr(args, "best_guess_turns", 0))
    best_guess = None  # I7: {fired_at_tokens, fired_at_turn} once the note was given

    def guess_note(budget_args, final=False):
        """I7: the best-guess note for the tool result of the turn just made (at most once per attempt), else ""."""
        nonlocal best_guess
        if (not guess_on or best_guess is not None or final
                or not best_guess_due(args, budget_args.max_total_output, total["completion"], n_turns)):
            return ""
        best_guess = {"fired_at_tokens": total["completion"], "fired_at_turn": n_turns}
        events.append({"turn": n_turns, "best_guess": True})
        return BEST_GUESS_NOTE + (BEST_GUESS_MULTI.format(n=multi) if multi else "")
    post_on = bool(getattr(args, "post_pass_note", False))
    post_pass = None  # I8: {fired_at_turn, fired_at_tokens} once the note was given
    n_train = len(task["train"])
    try:
        with session_factory(args.sandbox_url, task["train"], test_input,
                             args.tool_timeout, args.max_tool_output, **extra) as session:
            try:
                for index in range(args.max_turns):
                    remaining = args.max_total_output - total["completion"]
                    if remaining <= 0:
                        stop = "output_budget"
                        break
                    if session_final:
                        # E009: no forced final turn; the budget-end fallback is the session's `answer`.
                        final, cap = False, min(args.max_tokens, remaining)
                    else:
                        # Reserve last turn/tokens for final grid. Keep tools= present so
                        # DeepSeek continues consuming reasoning_content from previous turns.
                        final = index == args.max_turns - 1 or remaining <= args.final_tokens
                        if final:
                            messages.append({"role": "user", "content":
                                             "Finalize now. Return only your best test output as a JSON grid."})
                        cap = min(args.max_tokens, remaining if final else remaining - args.final_tokens)
                    n_turns += 1
                    mult = peak_multiplier() if auto_price else args.price_multiplier
                    try:
                        think = args.think_turns < 0 or index < args.think_turns
                        msg, usage, finish = turn(client, args, messages, cap, final, think, tok)
                    except ContextFull as exc:
                        n_turns -= 1
                        events.append({"turn": n_turns + 1, "context_full": str(exc)})
                        stop = "context"
                        break
                    except TurnError as exc:
                        messages.append(exc.assistant)
                        if exc.usage:
                            add_usage(total, exc.usage)
                            add_cost(cost, exc.usage, args.model, mult)
                        finish = exc.finish
                        raise
                    add_cost(cost, usage, args.model, mult)
                    messages.append(msg)
                    add_usage(total, usage)
                    events.append({"turn": n_turns, "usage": usage, "finish_reason": finish,
                                   "max_tokens": cap, "final": final})
                    if not think:
                        events[-1]["thinking"] = False
                    if finish not in {"stop", "tool_calls", "length"}:
                        raise RuntimeError(f"Provider interrupted generation: {finish}")
                    if finish == "length" and not final and getattr(args, "cap_nudge", False) and cap < remaining:
                        # E030 --cap-nudge: the per-turn max_tokens cut this turn. Keep its reasoning, drop any partial
                        # text or tool call, and close it with a synthetic run_python call whose output is the nudge.
                        call_id = f"cap_{n_turns}"
                        msg = {"role": "assistant", "content": "", "reasoning_content": msg.get("reasoning_content", ""),
                               "tool_calls": [{"id": call_id, "type": "function", "function": {
                                   "name": "run_python",
                                   "arguments": json.dumps({"code": "pass  # per-turn thinking limit"})}}]}
                        messages[-1] = msg
                        budget_args = context_budget(args, usage, total["completion"])
                        note = (CAP_NOTE.format(k=round(cap / 1000)) + guess_note(budget_args)
                                + budget_note(budget_args, total["completion"]) + turns_note(args, index))
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": note})
                        events.append({"turn": n_turns, "cap_nudge": True})
                        n_nudges += 1
                        continue
                    calls = msg.get("tool_calls", [])
                    if not calls and getattr(args, "rescue_code", False) and not final and finish == "stop":
                        rescued = rescue_tool_call(msg.get("content"), n_turns)
                        if rescued is not None:
                            msg = {**msg, "content": "", "tool_calls": [rescued]}
                            messages[-1] = msg
                            calls = [rescued]
                            n_rescues += 1
                            if tok is not None:
                                tok["resync"] = True
                            events.append({"turn": n_turns, "rescued": "code"})
                        elif n_nudge_rescues < 3 and not valid_grid(parse(msg.get("content") or "")):
                            # No call, no code and no answer grid: remind the model to use the tools (at most 3x).
                            messages.append({"role": "user", "content": RESCUE_NUDGE})
                            n_rescues += 1
                            n_nudge_rescues += 1
                            events.append({"turn": n_turns, "rescued": "nudge"})
                            continue
                    if not calls:
                        content = msg["content"]
                        candidate = parse(content)
                        pred = candidate if valid_grid(candidate) else None
                        source = "parse" if pred is not None else None
                        if multi:
                            pred, source = None, None  # a text answer gives one grid; the session fallback below
                        if pred is None and getattr(args, "lenient_parse", False) and not multi:
                            pred = lenient_parse(content)
                            source = "lenient" if pred is not None else None
                        if finish == "length" and not final:
                            # Reasoning can fill a whole turn before a grid is emitted.
                            # Use the reserved budget to ask for the answer next turn.
                            cut_parse = pred is not None  # A2: a cut turn's text grid yields to a session answer
                            continue
                        cut_parse = False
                        stop = "final" if finish == "stop" else finish
                        break
                    accepted, pass_at = None, None  # I8: index of this turn's first all-train-pass tool result
                    for call in calls:
                        n_tools += 1
                        status = "completed"
                        try:
                            if finish == "length" or final:
                                raise ValueError("Tool skipped: truncated response or final-turn budget reached")
                            arguments = json.loads(call["function"]["arguments"])
                            if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
                                raise ValueError("Expected an object with a string code field")
                            name, code = call["function"]["name"], arguments["code"]
                            if getattr(args, "rescue_code", False) and name not in ("run_python", "submit_answer",
                                                                                      "submit_solution"):
                                fixed = rescue_name(name)
                                if fixed != name:
                                    n_rescues += 1
                                    events.append({"turn": n_turns, "rescued": f"name {name}"})
                                    name = fixed
                                    call["function"]["name"] = fixed
                                    if tok is not None:
                                        tok["resync"] = True
                            if accepted is not None:
                                output = "[skipped: solution already accepted]"
                            elif name == "run_python":
                                executed.append((name, code))
                                result = session.execute(code)
                                status = result["process_status"]
                                output = tool_output(result, args.max_tool_output)
                                if repair is not None and status == "error":
                                    fixed = auto_repair_cell(session, code, output, repair,
                                                             PRELOADED | ({"test_inputs"} if with_inputs else set()),
                                                             args.max_tool_output, repairs)
                                    if fixed is not None:  # the repaired cell is what the session ran last
                                        output, status, executed[-1] = fixed[0], fixed[1], (name, fixed[2])
                                        events.append({"turn": n_turns, "auto_repair": status})
                                if (post_on and post_pass is None and pass_at is None and status == "completed"
                                        and train_pass_shown(executed[-1][1], output, n_train)):
                                    pass_at = len(messages)  # the tool message appended below
                            elif name == "submit_solution" and args.mode == "code":
                                output, accepted = submit_solution(session, code)
                                status = "completed" if accepted is not None else "rejected"
                            elif name == "submit_answer" and getattr(args, "answer_tool", False):
                                executed.append((name, code))
                                output, accepted = (submit_answer(session, code, multi) if multi
                                                    else submit_answer(session, code))
                                if accepted is not None and getattr(args, "verify_submit", False) and not multi:
                                    check = check_answer_on_train(session, code)
                                    verified = None if check is None else not check[0]
                                    if check and check[0] and code not in forced:
                                        forced.add(code)
                                        output, accepted = (
                                            f"[answer not accepted yet: the same expression with test_input = each "
                                            f"training input fails training pairs {check[0]} ({check[1]}). Fix the rule, "
                                            "or call submit_answer again with exactly the same code to submit "
                                            "anyway.]"), None
                                    else:
                                        submit_verified = verified
                                status = "completed" if accepted is not None else "rejected"
                            else:
                                raise ValueError(f"Unknown tool: {name}")
                        except (ValueError, TypeError, KeyError) as exc:
                            output, status = f"[tool error: {exc}]", "error"
                        # Transport errors abort rather than repeat ambiguous executions.
                        n_errors += status != "completed"
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                        events.append({"turn": n_turns, "tool_call_id": call["id"], "status": status})
                    if accepted is not None:
                        pred, content, stop, source = accepted, json.dumps(accepted), "submitted", "submit"
                        submit_code, cut_parse = code, False
                        break
                    # I7: a valid submit ended the attempt above, so reaching here means no answer was accepted yet
                    budget_args = context_budget(args, usage, total["completion"])
                    if (pass_at is not None and not final and total["completion"] < budget_args.max_total_output
                            and n_turns < args.max_turns):  # I8, once, while another turn can follow
                        post_pass = {"fired_at_turn": n_turns, "fired_at_tokens": total["completion"]}
                        events.append({"turn": n_turns, "post_pass": True})
                        messages[pass_at]["content"] += (POST_PASS_NOTE
                                                         + (POST_PASS_MULTI.format(n=multi) if multi else ""))
                    note = ((guess_note(budget_args, final) if messages[-1]["role"] == "tool" else "")
                            + budget_note(budget_args, total["completion"]) + turns_note(args, index))
                    if note and messages[-1]["role"] == "tool":
                        messages[-1]["content"] += note
                    if final:
                        stop = "length" if finish == "length" else "max_turns"
                        break
                else:
                    stop = "max_turns"
            except Exception as exc:
                if not session_final:
                    raise
                # A4: a turn failed (stream, server or tool transport error). The session may still hold `answer`:
                # the fallbacks below run before the attempt is recorded as an error (the error is kept).
                error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
            if session_final and (pred is None or cut_parse) and stop in {"output_budget", "max_turns", "length",
                                                                           "final", "context", "error"}:
                # Budget or turns ran out without an answer: take the session's `answer` if it is a valid grid.
                try:
                    restore_inputs(session, all_inputs, with_inputs)  # A8: the host's copies, not model rebinds
                    if multi:
                        # A1: one grid per test input from `answer`, then (E029) solve() on the test inputs still
                        # without one, each checked on its own, so one bad grid no longer loses the others
                        output, grids = answer_items(session, "answer", multi)
                        events.append({"fallback": "session_answer", "accepted": grids is not None,
                                       "output": output[:300]})
                        grids = grids or [None] * multi
                        codes = ["answer" if g is not None else None for g in grids]
                        missing = [i for i, g in enumerate(grids) if g is None]
                        if missing and getattr(args, "fallback_solve", False):
                            output, got = solve_items(session, [all_inputs[i] for i in missing])
                            events.append({"fallback": "session_solve", "accepted": got is not None,
                                           "output": output[:300]})
                            for i, g in zip(missing, got or []):
                                if g is not None:
                                    grids[i], codes[i] = g, "[solve(t) for t in test_inputs]"
                        if any(g is not None for g in grids):
                            pred, content, submit_code = grids, json.dumps(grids), codes[0]
                            source, cut_parse = "session" if "answer" in codes else "session_solve", False
                    else:
                        output, grid = submit_answer(session, "answer")
                        events.append({"fallback": "session_answer", "accepted": grid is not None,
                                       "output": output[:300]})
                        if grid is not None:
                            pred, content, source, submit_code = grid, json.dumps(grid), "session", "answer"
                            cut_parse = False
                        elif getattr(args, "fallback_solve", False):
                            # E029: no valid `answer`; evaluate the session's solve on the test input, if defined.
                            expr = "solve(test_input)"
                            output, grid = submit_answer(session, expr)
                            if grid is None:  # A3: numpy-only programs (solve(np.array(test_input)))
                                more, got = solve_items(session, all_inputs)
                                grid, output = (got or [None])[0], output[:150] + " " + more
                            events.append({"fallback": "session_solve", "accepted": grid is not None,
                                           "output": output[:300]})
                            if grid is not None:
                                pred, content, source, submit_code = grid, json.dumps(grid), "session_solve", expr
                                cut_parse = False
                except Exception as exc:
                    if error is None:
                        raise
                    events.append({"fallback": "failed", "output": f"{type(exc).__name__}: {exc}"[:300]})
            if getattr(args, "alt_answer", "") and pred is not None:
                # E049: the session's optional second guess (no model tokens); kept only if valid and different
                output, grid = (submit_answer(session, "alt_answer", multi) if multi
                                else submit_answer(session, "alt_answer"))
                alt_pred = grid if grid is not None and grid != pred else None
            if getattr(args, "train_check", False) and pred is not None:
                # E017: label the answer program on the training pairs (no model tokens; host compares hashes).
                try:
                    if multi and pred[0] is None:  # A1: a partial multi answer without grid 0 is not labelled
                        labels = {"train_pass": None, "train_frac": None, "train_trace": "no_answer_0"}
                    else:
                        # multi: label the program behind test input 0's grid (session functions, cells); A6: as a
                        # one-grid program, replayed with test_inputs[0] = each training input
                        labels = train_check(session, executed, pred[0] if multi else pred,
                                             multi_check_code(submit_code, all_inputs) if multi else submit_code,
                                             submit=submit_answer,
                                             replay_session=lambda grid: session_factory(
                                                 args.sandbox_url, task["train"], grid, args.tool_timeout,
                                                 args.max_tool_output, **session_extra([grid] + all_inputs[1:])))
                except Exception as exc:
                    labels = {"train_pass": None, "train_frac": None, "train_trace": "check_error",
                              "train_check_error": f"{type(exc).__name__}: {exc}"[:300]}
    except Exception as exc:
        error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
    result = {"content": content, "pred": pred, "tool_calls": n_tools, "tool_errors": n_errors,
              "turns": n_turns, "usage": total, "finish_reason": finish, "stop_reason": stop,
              "usage_complete": error is None, "est_cost_usd": round(cost["total"], 6),
              "seconds": round(time.monotonic() - t0, 1)}
    if multi:
        result["multi_test"] = multi
    if auto_price:
        result.update(est_cost_peak_usd=round(cost["peak"], 6), peak_turns=cost["peak_turns"])
    if getattr(args, "cap_nudge", False):
        result["cap_nudges"] = n_nudges
    if getattr(args, "alt_answer", ""):
        result["alt_pred"] = alt_pred
    if getattr(args, "rescue_code", False):
        result["rescues"] = n_rescues
    if repairs is not None:
        result["auto_repairs"] = repairs
    if guess_on:  # I7: null when the note never fired
        result["best_guess"] = None if best_guess is None else {
            **best_guess, "submitted_after": source == "submit" and pred is not None,
            "turns_after": n_turns - best_guess["fired_at_turn"],
            "tokens_after": total["completion"] - best_guess["fired_at_tokens"]}
    if post_on:  # I8: null when the note never fired
        result["post_pass"] = None if post_pass is None else {
            **post_pass, "turns_after": n_turns - post_pass["fired_at_turn"],
            "tokens_after": total["completion"] - post_pass["fired_at_tokens"],
            "submitted_after": source == "submit" and pred is not None}
    if tok is not None:
        result["token_exact"] = {k: v for k, v in tok.items()
                                 if k not in ("ids", "n", "prev_len", "prev_sampled", "resync")}
    if getattr(args, "train_check", False):
        result.update({"train_pass": None, "train_frac": None, "train_trace": "no_answer"} if pred is None else labels)
    if any(getattr(args, k, v) != v for k, v in V4_DEFAULTS.items()):
        result["answer_source"] = source if pred is not None else None
    if getattr(args, "verify_submit", False):
        result["submit_verified"] = submit_verified if source == "submit" else None
        result["submit_rejections"] = len(forced)
    if error:
        result["error"] = error
    return {"messages": messages, "events": events}, result


def deepseek_balance(key):
    """Current DeepSeek balance in USD (never logs the key). Lags real usage by minutes."""
    import httpx
    response = httpx.get("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {key}"},
                         timeout=30)
    response.raise_for_status()
    return float(response.json()["balance_infos"][0]["total_balance"])


def load_tasks(challenges, solutions=None):
    challenges = Path(challenges)
    if challenges.is_dir():
        tasks = {p.stem: json.loads(p.read_text()) for p in sorted(challenges.glob("*.json"))}
    else:
        tasks = json.loads(challenges.read_text())
    if solutions:
        answers = json.loads(Path(solutions).read_text())
        for task_id, task in tasks.items():
            outputs = answers.get(task_id)
            if outputs is None:
                continue
            if len(outputs) != len(task["test"]):
                raise ValueError(f"Solution count mismatch for {task_id}")
            task["test"] = [dict(pair, output=output) for pair, output in zip(task["test"], outputs)]
    if not tasks:
        raise ValueError("No tasks found")
    return dict(sorted(tasks.items()))


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(records, tasks, attempts):
    latest = {(r["task_id"], r["pair"], r["attempt"]): r for r in records
              if r["task_id"] in tasks and r["attempt"] < attempts}
    scored = {t: task for t, task in tasks.items() if all("output" in p for p in task["test"])}
    if scored:
        def score(k):
            return sum(sum(any(latest.get((t, p, a), {}).get("correct", False) for a in range(k))
                           for p in range(len(task["test"]))) / len(task["test"])
                       for t, task in scored.items()) / len(scored)
        print(f"pass@1 {score(1):.1%}  pass@{attempts} {score(attempts):.1%} ({len(scored)} scored tasks)")
        if any("train_pass" in r for r in latest.values()) or any(a >= attempts for _, _, a in latest):
            allrecs = {(r["task_id"], r["pair"], r["attempt"]): r for r in records if r["task_id"] in tasks}
            def ranked(t, p):
                recs = [r for (u, q, _), r in allrecs.items() if (u, q) == (t, p) and "error" not in r]
                return any(r.get("correct") for r in ranked_answers(recs))
            ranked_score = sum(sum(ranked(t, p) for p in range(len(task["test"]))) / len(task["test"])
                               for t, task in scored.items()) / len(scored)
            print(f"train-check ranked pass@2 (all attempts incl. policy extras) {ranked_score:.1%}")
    print(f"{len(latest)} attempts, {sum('error' in r for r in latest.values())} errors, "
          f"{sum(r.get('tool_calls', 0) for r in latest.values())} tool calls; "
          f"estimated recorded cost ${sum(r.get('est_cost_usd', 0) for r in records):.2f}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-flash",
                    help="DeepSeek model id, or the served model name of a --base-url server")
    ap.add_argument("--base-url", help="self-hosted OpenAI-compatible server(s), comma-separated; each attempt is "
                    "pinned to one server (data parallel). Skips the DeepSeek balance/cost guards")
    ap.add_argument("--extra-body", help="JSON request extras for --base-url (e.g. sampling, chat_template_kwargs)")
    ap.add_argument("--extra-body-nothink", help="JSON request extras for non-thinking turns (default: --extra-body)")
    ap.add_argument("--effort", default="max", choices=["low", "high", "max"])
    ap.add_argument("--mode", default="grid", choices=["grid", "code"])
    ap.add_argument("--prompt", default="plain", choices=list(PROMPT_SUFFIXES))
    ap.add_argument("--sandbox-url", default=os.environ.get("ARC_SANDBOX_URL"))
    ap.add_argument("--tool-timeout", type=float, default=30)
    ap.add_argument("--max-tool-output", type=int, default=16000)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=200_000, help="per assistant turn")
    ap.add_argument("--max-total-output", type=int, default=200_000, help="per attempt, including final turn")
    ap.add_argument("--final-tokens", type=int, default=8192, help="reserve from total budget for final grid")
    ap.add_argument("--max-turns", type=int, default=40, help="includes final turn")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--challenges", type=Path, default=TASKS, help="ARC task directory or challenge JSON")
    ap.add_argument("--solutions", type=Path, help="optional host-side solutions JSON for scoring")
    ap.add_argument("--tasks", help="comma-separated ids")
    ap.add_argument("--pairs", help="comma-separated task_id:pair; only run these test pairs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--answer-tool", action="store_true",
                    help="E009: add submit_answer(expr), evaluated in the session; ends the attempt (grid mode)")
    ap.add_argument("--final-mode", default="turn", choices=["turn", "session"],
                    help="turn: forced 'Finalize now' turn (v3); session: no final turn, fall back to `answer`")
    ap.add_argument("--budget-notes", type=float, default=0.0,
                    help="E009: append token-budget status to tool output once this fraction is used (0=off)")
    ap.add_argument("--budget-soft", type=int, default=0,
                    help="E009: budget notes also state this soft target in tokens (below the hard total); 0=off")
    ap.add_argument("--soft-wording", default="gentle", choices=["gentle", "strong"],
                    help="E010: note text past --budget-soft (strong = submit your best now)")
    ap.add_argument("--verify-submit", action="store_true",
                    help="E011: re-run a submit_answer expression on the training inputs; on a mismatch reject once "
                         "with feedback (resubmitting the same code accepts it)")
    ap.add_argument("--think-turns", type=int, default=-1,
                    help="E010: thinking only on the first N assistant turns, disabled after (-1 = always think)")
    ap.add_argument("--budget-warn", type=float, default=0.8, help="fraction at which the note says to answer now")
    ap.add_argument("--only-attempts", default=None,
                    help="E043: comma-separated attempt indices to run (others are left to another process)")
    ap.add_argument("--token-exact", action="store_true",
                    help="E043 (--base-url): send each turn as the attempt's exact token ids (prompt + sampled ids + "
                         "rendered new messages) to /v1/completions, so re-tokenizing sampled text never breaks the "
                         "server's cached prefix")
    ap.add_argument("--rescue-code", action="store_true",
                    help="E043: a non-final turn without a valid tool call but with code (garbled DSML or a ```python "
                         "block) becomes a run_python call; near-miss tool names are mapped (pruned models)")
    ap.add_argument("--auto-repair", action="store_true",
                    help="I5: when a run_python cell fails on an identifier typo, evals/auto_repair.py may rewrite the "
                         "name to its unique close match; the fixed cell runs once and the model sees the note and its "
                         "output (auto_repairs per attempt; not part of the config, so files resume)")
    ap.add_argument("--best-guess-at", type=float, default=0.0,
                    help="I7: once per attempt without an accepted answer, when the output tokens reach this fraction "
                         "of the attempt's hard budget (scaled by --multi-budget, capped by --context-limit, as the "
                         "budget notes), the tool result says to stop exploring and submit the best current answer "
                         "(0=off; recommended 0.8; best_guess per attempt; not part of the config, so files resume)")
    ap.add_argument("--best-guess-turns", type=int, default=0,
                    help="I7: the same note (still once per attempt) when turns so far >= --max-turns - K "
                         "(0=off; recommended 3)")
    ap.add_argument("--post-pass-note", action="store_true",
                    help="I8: once per attempt without an accepted answer, the first run_python result that shows a "
                         "program reproducing every training pair (train_pass_shown) says to check the test input's "
                         "uncovered assumptions with code, then submit (post_pass per attempt; not part of the "
                         "config, so files resume)")
    ap.add_argument("--lenient-parse", action="store_true",
                    help="E009: if the ARC Prize parser finds no grid in the final text, take the last valid one")
    ap.add_argument("--task-summary", action="store_true",
                    help="E017 P1: append a host-computed perception pack (objects, matching, test novelty) to the prompt")
    ap.add_argument("--helpers", action="store_true",
                    help="E017 P3: preload numpy grid helpers (objects, d8, panels, stamp, ray, check) in the session")
    ap.add_argument("--train-check", action="store_true",
                    help="E017: after each attempt, re-run its answer program on the training inputs and record "
                         "train_pass / train_frac (no model tokens)")
    ap.add_argument("--extra-attempt-if-no-train-pass", action="store_true",
                    help="E017 policy: with --attempts 2 and --train-check, run attempt 2 for a pair when neither "
                         "attempt 0 nor 1 passes the train check (not part of the config, so files resume)")
    ap.add_argument("--turns-note", type=int, default=0,
                    help="E029: append a 'N turns left, submit now' note to tool output when N or fewer turns remain (0=off)")
    ap.add_argument("--grid-format", default="json", choices=["json", "image", "text+image"],
                    help="E033: image = every grid in the prompt is a PNG (ARC palette, --image-cell px per cell, no grid "
                         "lines) instead of JSON; needs a vision model behind --base-url. E058: text+image = the JSON "
                         "prompt unchanged plus one labelled image per grid (works with the DeepSeek API)")
    ap.add_argument("--image-cell", type=int, default=32,
                    help="E033: pixels per grid cell (Qwen3.8: 32 = one LLM visual token per cell, 16 = one ViT patch); "
                         "E058 text+image: 0 = DeepSeek-aligned per grid (grid_image.deepseek_cell)")
    ap.add_argument("--image-align", default="deepseek", choices=["deepseek", "qwen"],
                    help="E058 text+image with --image-cell 0: tile geometry to align cells to (deepseek 42 px, "
                         "169-1024 tiles; qwen = Qwen3.8 Flash on Alibaba, 32 px, 64-2500 tiles)")
    ap.add_argument("--openrouter-provider", default="",
                    help="E037: run through OpenRouter pinned to this provider (no fallbacks); --model is then an "
                         "OpenRouter id such as deepseek/deepseek-v4.1-flash-1017; cost comes from OpenRouter usage")
    ap.add_argument("--or-reasoning", default="",
                    help='E037: OpenRouter reasoning object as JSON (default {"effort": <--effort>})')
    ap.add_argument("--effort-header", default="", choices=["", *EFFORT_HEADERS],
                    help="E030: system message in DeepSeek's effort-prefix style (minimal = terse thinking)")
    ap.add_argument("--cap-nudge", action="store_true",
                    help="E030: when a turn hits --max-tokens, keep its reasoning and close it with a synthetic run_python "
                         "call whose output tells the model to test its hypothesis in code (per-turn thinking cap)")
    ap.add_argument("--multi-test", action="store_true",
                    help="E046: one attempt per task for tasks with several test inputs: the prompt shows all of them, "
                         "the session has test_inputs, and the answer is a list of grids (one record per pair; the "
                         "attempt's usage is on pair 0, the other pairs carry multi_share)")
    ap.add_argument("--context-limit", type=int, default=0,
                    help="E046 (--token-exact): the server's max-model-len; each turn's max_tokens is capped to the room "
                         "left, and an attempt with less than 1024 tokens of room ends like an output-budget stop "
                         "(session-answer fallback) instead of a context-length error")
    ap.add_argument("--multi-budget", type=float, default=0.0,
                    help="E046: with --multi-test, scale a multi-input task's output budget, soft budget and turn cap "
                         "by 1 + f x (n - 1)")
    ap.add_argument("--multi-prompt", choices=["note", "lean", "early", "first"], default="note",
                    help="E046: wording of the --multi-test note (note = the E046 note; lean = one line; early = note + "
                         "define solve and set answer early; first = only test input 0 in the prompt, the rest in "
                         "the session)")
    ap.add_argument("--alt-answer", choices=["", "passive", "active"], default="",
                    help="E049: the prompt invites an optional second guess in `alt_answer`, read from the session "
                         "after the attempt (alt_pred / alt_correct; passive = only if met, active = briefly check)")
    ap.add_argument("--prior-mode", choices=["", "verify", "verifycode", "hybrid", "split", "differ"], default="",
                    help="E047: attempts >= 1 see attempt 0's answer from --prior-file, never its thinking (verify = "
                         "check it, resubmit or fix; verifycode = verify + the program behind the answer; hybrid = "
                         "verifycode when attempt 0 failed the train check, else a test-extension audit with the "
                         "program; split = verifycode when attempt 0 failed the train check, else verify; differ = "
                         "assume it is wrong and look for a different rule)")
    ap.add_argument("--prior-file", type=Path,
                    help="E047: jsonl whose attempt-0 records give the prior answers (not part of the config)")
    ap.add_argument("--fallback-solve", action="store_true",
                    help="E029: with --final-mode session, if `answer` is not a valid grid, try solve(test_input)")
    ap.add_argument("--auto-peak-price", action="store_true",
                    help="E017: price each turn at 2x inside weekday peak windows (UTC 01-04, 06-10), else 1x")
    ap.add_argument("--max-balance-stop", type=float, default=20.0,
                    help="stop launching new attempts when the DeepSeek balance falls below this (USD)")
    ap.add_argument("--price-multiplier", type=float, default=2, help="2 assumes peak; 1 off-peak")
    ap.add_argument("--allow-expensive-run", action="store_true", help="explicitly allow output budget above $10")
    ap.add_argument("--dry-run", action="store_true", help="show selection/budget without API or sandbox requests")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    for name in ("attempts", "max_tokens", "max_total_output", "final_tokens", "max_turns", "workers",
                 "tool_timeout", "max_tool_output", "price_multiplier"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_tokens > 393216:
        ap.error("--max-tokens must be at most 393216")
    if args.max_total_output <= args.final_tokens:
        ap.error("--max-total-output must exceed --final-tokens")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    if args.prompt in SUBMIT_PROMPTS and not args.answer_tool:
        ap.error(f"--prompt {args.prompt} describes submit_answer; pass --answer-tool")
    if args.extra_attempt_if_no_train_pass and (args.attempts != 2 or not args.train_check):
        ap.error("--extra-attempt-if-no-train-pass needs --attempts 2 and --train-check")
    if args.grid_format == "image" and not args.base_url:
        ap.error("--grid-format image needs a vision model behind --base-url")
    if args.cap_nudge and (args.final_mode != "session" or not args.answer_tool):
        ap.error("--cap-nudge needs --final-mode session and --answer-tool")
    if args.fallback_solve and args.final_mode != "session":
        ap.error("--fallback-solve needs --final-mode session")
    if args.turns_note < 0 or args.turns_note >= args.max_turns:
        ap.error("--turns-note must be in [0, --max-turns)")
    if not 0 <= args.best_guess_at < 1 or not 0 <= args.best_guess_turns < args.max_turns:
        ap.error("--best-guess-at must be in [0, 1) and --best-guess-turns in [0, --max-turns)")
    if (args.best_guess_at or args.best_guess_turns) and not args.answer_tool:
        ap.error("--best-guess-at / --best-guess-turns need --answer-tool (the note asks for submit_answer)")
    if args.post_pass_note and not args.answer_tool:
        ap.error("--post-pass-note needs --answer-tool (the note asks for submit_answer)")
    if args.answer_tool and args.mode != "grid":
        ap.error("--answer-tool is for grid mode")
    if args.budget_soft and (not args.budget_notes or not 0 < args.budget_soft < args.max_total_output):
        ap.error("--budget-soft needs --budget-notes and must be below --max-total-output")
    if not 0 <= args.budget_notes < 1 or not 0 < args.budget_warn <= 1:
        ap.error("--budget-notes must be in [0, 1) and --budget-warn in (0, 1]")
    if bool(args.prior_mode) != bool(args.prior_file):
        ap.error("--prior-mode and --prior-file go together")
    args._prior = (load_prior(args.prior_file, args.prior_mode in ("verifycode", "hybrid", "split")) if args.prior_file
                   else None)
    tasks = load_tasks(args.challenges, args.solutions)
    pairs = None
    if args.pairs:
        pairs = {(t, int(p)) for t, p in (item.rsplit(":", 1) for item in args.pairs.split(","))}
        if {t for t, _ in pairs} - tasks.keys():
            ap.error(f"Unknown task ids in --pairs: {sorted({t for t, _ in pairs} - tasks.keys())}")
        tasks = {t: task for t, task in tasks.items() if any(t == u for u, _ in pairs)}
    if args.tasks:
        wanted = set(args.tasks.split(","))
        if wanted - tasks.keys():
            ap.error(f"Unknown task ids: {sorted(wanted - tasks.keys())}")
        tasks = {t: task for t, task in tasks.items() if t in wanted}
    if args.limit:
        tasks = dict(list(tasks.items())[:args.limit])
    tag = f"_{args.tag}" if args.tag else ""
    variant = "" if args.prompt == "plain" else f"_{args.prompt}"
    out = (args.out or ROOT / f"evals/results/tools_persistent_{args.mode}{variant}_{args.model}_{args.effort}{tag}.jsonl").resolve()
    config = {name: getattr(args, name) for name in (
        "model", "effort", "mode", "prompt", "max_tokens", "max_total_output", "final_tokens", "max_turns",
        "tool_timeout", "max_tool_output")}
    v4 = {k: getattr(args, k) for k, v in {**V4_DEFAULTS, **V5_DEFAULTS, **V6_DEFAULTS}.items()
          if getattr(args, k) != v}
    if v4:  # only new-option runs carry the extra keys, so v3 configs (and resume of old files) are unchanged
        config.update(v4)
    config.update(harness_version=harness_version(args), challenges=str(args.challenges.resolve()),
                  solutions=str(args.solutions.resolve()) if args.solutions else None)
    try:
        records = read_records(out)
    except json.JSONDecodeError:
        ap.error("Results contain an incomplete/invalid line. Repair it or choose a new --out")
    if any(r.get("config") != config for r in records):
        ap.error("Existing results use a different configuration; choose a new --tag or --out")
    latest = {(r["task_id"], r["pair"], r["attempt"]): r for r in records}
    policy = args.extra_attempt_if_no_train_pass

    def needs_extra(t, p):
        """E017 policy: attempts 0 and 1 finished (no API error), neither passes train, attempt 2 not done yet."""
        recs = [latest.get((t, p, a)) for a in (0, 1)]
        done2 = (t, p, 2) in latest and "error" not in latest[t, p, 2]
        return (not done2 and all(r is not None and "error" not in r for r in recs)
                and not any(r.get("train_pass") is True for r in recs))

    jobs = [(t, p, a) for t, task in tasks.items() for p in range(len(task["test"]))
            for a in range(args.attempts) if (t, p, a) not in latest or "error" in latest[t, p, a]]
    if args.multi_test:  # E046: pair 0's job covers every pair of its task
        jobs = [job for job in jobs if job[1] == 0 or len(tasks[job[0]]["test"]) == 1]
    if pairs is not None:
        jobs = [job for job in jobs if job[:2] in pairs]
    if policy:
        jobs += [(t, p, 2) for t, task in tasks.items() for p in range(len(task["test"]))
                 if (pairs is None or (t, p) in pairs) and needs_extra(t, p)]
    if getattr(args, "only_attempts", None):  # E043: e.g. attempt 1 in a second process beside a running attempt 0
        keep_attempts = {int(x) for x in args.only_attempts.split(",")}
        jobs = [job for job in jobs if job[2] in keep_attempts]
    price_out = PRICES.get(args.model, (0.0, 0.0, 0.0))[2]
    output_cap = len(jobs) * args.max_total_output * price_out * args.price_multiplier / 1e6
    expected = len(jobs) * 65000 * price_out * args.price_multiplier / 1e6
    print(f"{len(tasks)} tasks, {len(jobs)} remaining attempts, {args.workers} workers -> {out}", flush=True)
    print(f"Output-only estimate at 65k tokens/attempt: ${expected:.2f}; budget cap ${output_cap:.2f} "
          f"(price multiplier {args.price_multiplier:g}). Multi-turn input costs are additional.", flush=True)
    if args.dry_run:
        return
    if not jobs:
        summarize(records, tasks, args.attempts)
        return
    if output_cap > 10 and not args.allow_expensive_run:
        ap.error("Output budget exceeds $10. Review --dry-run and pass --allow-expensive-run to launch")
    if not args.sandbox_url:
        ap.error("Pass --sandbox-url or ARC_SANDBOX_URL from evals/sandbox_docker/start.sh")
    check_health(args.sandbox_url)
    if args.base_url:
        return run_compat(args, tasks, jobs, records, latest, out, config, policy, needs_extra)
    if args.openrouter_provider:
        key, base_url, balance_of, who = openrouter_key(), OPENROUTER_URL, openrouter_balance, "OpenRouter"
    else:
        key, base_url, balance_of, who = api_key(), "https://api.deepseek.com", deepseek_balance, "DeepSeek"
    balance = balance_of(key)
    print(f"{who} balance ${balance:.2f}; stop launching below ${args.max_balance_stop:.2f}", flush=True)
    if balance < args.max_balance_stop:
        ap.error("Balance below --max-balance-stop")
    low_balance = threading.Event()

    def guard():  # kill-guard: re-check the balance every 5 min; when low, no new attempts start
        while not low_balance.wait(300):
            try:
                if balance_of(key) < args.max_balance_stop:
                    print("Balance below --max-balance-stop: not starting new attempts", flush=True)
                    low_balance.set()
            except Exception as exc:
                print(f"balance check failed: {type(exc).__name__}", flush=True)
    threading.Thread(target=guard, daemon=True).start()
    trace_dir = out.with_suffix("")
    trace_dir.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size:
        with out.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            tail = f.read(1)
        if tail != b"\n":
            with out.open("a") as f:
                f.write("\n")  # valid JSON without a final newline
    lock = threading.Lock()
    # No automatic API retries: preserve partial traces and retry failed attempts on resume.
    with OpenAI(api_key=key, base_url=base_url, timeout=3600, max_retries=0) as client:
        def work(t, p, a):
            if low_balance.is_set():
                return None
            task = tasks[t]
            rec = {"task_id": t, "pair": p, "attempt": a, "model": args.model, "effort": args.effort,
                   "mode": args.mode, "config": config,
                   "price_multiplier": "auto" if args.auto_peak_price else args.price_multiplier}
            if a >= args.attempts:
                rec["policy_extra"] = True
            multi = args.multi_test and len(task["test"]) > 1
            trace, result = solve(client, args, task, task["test"][p]["input"],
                                  **({"test_inputs": [x["input"] for x in task["test"]]} if multi else {}),
                                  prior=prior_for(args, t, task, p, a, multi))
            if "error" in result:
                result["error"] = result["error"].replace(key, "[redacted]")
            path = trace_dir / f"{t}_p{p}_a{a}_{uuid4().hex[:8]}.messages.json.gz"
            with gzip.open(path, "wt") as f:
                json.dump(trace, f)
            rec.update(result, trace_path=str(path))
            recs = pair_records(rec, task, multi)
            with lock, out.open("a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
            return recs

        with ThreadPoolExecutor(args.workers) as executor:
            futures = {executor.submit(work, *job): job for job in jobs}
            index, total_jobs = 0, len(jobs)
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.pop(future)
                    recs = future.result()
                    if recs is None:
                        continue
                    for rec in recs[1:]:
                        records.append(rec)
                        latest[rec["task_id"], rec["pair"], rec["attempt"]] = rec
                    rec = recs[0]
                    index += 1
                    records.append(rec)
                    latest[rec["task_id"], rec["pair"], rec["attempt"]] = rec
                    if policy and rec["attempt"] in (0, 1) and needs_extra(rec["task_id"], rec["pair"]) \
                            and (rec["task_id"], rec["pair"], 2) not in {futures[f] for f in futures}:
                        futures[executor.submit(work, rec["task_id"], rec["pair"], 2)] = (rec["task_id"], rec["pair"], 2)
                        total_jobs += 1
                    report(rec, index, total_jobs)
    summarize(records, tasks, args.attempts)


def pair_records(rec, task, multi):
    """One scored record per test pair. E046 --multi-test: the attempt's list of grids becomes one record per pair;
    the attempt's usage and cost stay on pair 0's record, the others carry multi_share and zero usage."""
    recs = [rec]
    if multi:
        preds = rec["pred"] if isinstance(rec["pred"], list) else [None] * len(task["test"])
        recs = []
        for q in range(len(task["test"])):
            r = dict(rec, pair=q, pred=preds[q])
            if "alt_pred" in rec:  # E049: a list of n grids, or None
                r["alt_pred"] = rec["alt_pred"][q] if isinstance(rec["alt_pred"], list) else None
            if q:
                r.update(usage={"prompt": 0, "cache_hit": 0, "completion": 0}, est_cost_usd=0.0, multi_share=True)
            recs.append(r)
    for r in recs:
        if "output" in task["test"][r["pair"]]:
            r["correct"] = r["pred"] == task["test"][r["pair"]]["output"]
            if "alt_pred" in r:
                r["alt_correct"] = r["alt_pred"] is not None and r["alt_pred"] == task["test"][r["pair"]]["output"]
    return recs


def run_compat(args, tasks, jobs, records, latest, out, config, policy, needs_extra):
    """Self-hosted OpenAI-compatible servers (--base-url, comma-separated): each attempt is pinned to one server by a
    hash of (task, pair, attempt), so every turn of a conversation reuses that server's prefix cache (data parallel)."""
    import hashlib
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    clients = [OpenAI(api_key="none", base_url=u, timeout=7200, max_retries=0) for u in urls]
    trace_dir = out.with_suffix("")
    trace_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(t, p, a):
        task = tasks[t]
        client = clients[int(hashlib.md5(f"{t}:{p}:{a}".encode()).hexdigest(), 16) % len(clients)]
        rec = {"task_id": t, "pair": p, "attempt": a, "model": args.model, "effort": args.effort, "mode": args.mode,
               "config": config, "base_url": str(client.base_url)}
        multi = getattr(args, "multi_test", False) and len(task["test"]) > 1
        trace, result = solve(client, args, task, task["test"][p]["input"],
                              **({"test_inputs": [x["input"] for x in task["test"]]} if multi else {}),
                              prior=prior_for(args, t, task, p, a, multi))
        path = trace_dir / f"{t}_p{p}_a{a}_{uuid4().hex[:8]}.messages.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
        rec.update(result, trace_path=str(path))
        recs = pair_records(rec, task, multi)
        with lock, out.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
        return recs

    with ThreadPoolExecutor(args.workers) as executor:
        futures = [executor.submit(work, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            recs = fut.result()
            records.extend(recs)
            report(recs[0], i, len(jobs))
    for c in clients:
        c.close()
    summarize(records, tasks, args.attempts)


def report(rec, index, total_jobs):
    status = rec.get("error") or (
        f"correct={rec.get('correct', 'unscored')} tools={rec['tool_calls']} "
        f"{rec['usage']['completion']} out-tok ${rec['est_cost_usd']:.4f} "
        f"{rec['seconds']}s {rec['stop_reason']}"
        + (f" train_pass={rec['train_pass']}" if "train_pass" in rec else ""))
    print(f"[{index}/{total_jobs}] {rec['task_id']}.{rec['pair']}#{rec['attempt']} {status}", flush=True)


if __name__ == "__main__":
    main()
