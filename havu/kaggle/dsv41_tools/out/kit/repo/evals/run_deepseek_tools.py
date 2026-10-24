"""DeepSeek ARC-AGI-2 eval with persistent sandboxed Python tool use.

Start evals/sandbox_docker/start.sh, then pass its URL with --sandbox-url.
Defaults use the ARC Prize public eval prompt/parser/scoring. Each independent
attempt owns one Python session. See experiments/E004_tool_use/README.md.
"""
import argparse
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
from arc_harness.grid_image import build_image_prompt
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
               "image_cell": 32, "openrouter_provider": "", "or_reasoning": ""}
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


def submit_answer(session, code):
    """Evaluate the model's answer expression in its own persistent session and return (feedback, grid|None).

    `code` is an expression (e.g. solve(test_input)) or statements that set `answer`. Printed output from
    the evaluation is captured so only the marker line comes back. The session never holds test answers."""
    marker = f"ARC_ANSWER_{uuid4().hex}:"
    cell = f'''
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
    del _arc_answer
'''
    result = session.execute(cell)
    grid = None
    if result["process_status"] == "completed":
        for line in reversed(result.get("stdout", "").splitlines()):
            if line.startswith(marker):
                try:
                    grid = json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    pass
                break
        else:
            return "[answer not accepted] no value returned", None
    else:
        return tool_output(result, session.max_output)[:2000] + "\n[answer not accepted]", None
    if not valid_grid(grid):
        shape = (f"{len(grid)} rows, row lengths {sorted({len(r) for r in grid if isinstance(r, list)})}"
                 if isinstance(grid, list) else type(grid).__name__)
        return ("[answer not accepted] must be a rectangular 1-30 by 1-30 grid of ints 0-9; got " + shape), None
    return f"[answer accepted: {len(grid)}x{len(grid[0])} grid]", grid


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
    import contextlib, io, json
    _expr = compile({code!r}, '<answer>', 'eval')
    outs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for _inp in json.loads({json.dumps(inputs)!r}):
            try:
                v = eval(_expr, {{**globals(), 'test_input': _inp}})
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


def parse_ds_completion(text):
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


def turn_token_exact(client, args, messages, max_tokens, tok):
    """E043 --token-exact (--base-url only): continue the attempt's exact token history instead of letting the server
    re-render earlier assistant turns. Re-tokenizing sampled text does not always give back the sampled ids (the
    model sometimes samples a non-canonical split; ~13% of turns), which ends the cached prefix inside that turn; on a
    small KV pool (4xL4) the state needed to resume there is evicted. Only the messages appended since the last turn
    are rendered (server /tokenize: full render minus the render through the last assistant turn), and generation
    goes to /v1/completions with the token ids. The output is parsed like the model's encoding; a format error keeps
    the text as content (so --rescue-code still sees garbled calls). If the render is not prefix-stable (e.g. a new
    user message drops earlier reasoning), the turn uses the full render."""
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
    if tok["ids"] is None:
        ids = full
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
    try:
        reasoning, content, calls = parse_ds_completion(text if text.endswith(DS_EOS) or finish != "stop"
                                                        else text + DS_EOS)
    except ValueError:
        tok["parse_fallbacks"] += finish != "length"
        reasoning, sep, rest = text.partition(DS_THINK_END)
        reasoning, content, calls = (reasoning, rest, []) if sep else (text, "", [])
        content = content.removesuffix(DS_EOS)
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


def solve(client, args, task, test_input, session_factory=SandboxSession):
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
        prompt = build_prompt(task["train"], test_input) + extra
    messages = [{"role": "user", "content": prompt}]
    if getattr(args, "effort_header", ""):
        messages.insert(0, {"role": "system", "content": EFFORT_HEADERS[args.effort_header]})
    total = {"prompt": 0, "cache_hit": 0, "completion": 0}
    n_tools, n_errors, n_turns, finish, stop, pred = 0, 0, 0, None, None, None
    n_nudges = 0  # E030 --cap-nudge
    n_rescues = 0  # E043 --rescue-code
    n_nudge_rescues = 0
    # E043 --token-exact: the attempt's exact token history (prompt + sampled ids) and counters
    tok = ({"ids": None, "n": 0, "prev_len": 0, "prev_sampled": [], "turns": 0, "breaks": 0, "rerenders": 0,
            "parse_fallbacks": 0}
           if getattr(args, "token_exact", False) and getattr(args, "base_url", None) else None)
    events, error, content, t0 = [], None, "", time.monotonic()
    session_final = getattr(args, "final_mode", "turn") == "session"
    source = None  # where the single scored grid came from
    forced, submit_verified = set(), None  # E011 --verify-submit state
    executed, submit_code, labels = [], None, {}  # E017 --train-check state
    auto_price = getattr(args, "auto_peak_price", False)
    cost = {"total": 0.0, "peak": 0.0, "peak_turns": 0}
    extra = {"prelude_extra": HELPERS_SRC} if getattr(args, "helpers", False) else {}
    try:
        with session_factory(args.sandbox_url, task["train"], test_input,
                             args.tool_timeout, args.max_tool_output, **extra) as session:
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
                               "name": "run_python", "arguments": json.dumps({"code": "pass  # per-turn thinking limit"})}}]}
                    messages[-1] = msg
                    note = (CAP_NOTE.format(k=round(cap / 1000)) + budget_note(args, total["completion"])
                            + turns_note(args, index))
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
                    if pred is None and getattr(args, "lenient_parse", False):
                        pred = lenient_parse(content)
                        source = "lenient" if pred is not None else None
                    if finish == "length" and not final:
                        # Reasoning can fill a whole turn before a grid is emitted.
                        # Use the reserved budget to ask for the answer next turn.
                        continue
                    stop = "final" if finish == "stop" else finish
                    break
                accepted = None
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
                        if accepted is not None:
                            output = "[skipped: solution already accepted]"
                        elif name == "run_python":
                            executed.append((name, code))
                            result = session.execute(code)
                            status = result["process_status"]
                            output = tool_output(result, args.max_tool_output)
                        elif name == "submit_solution" and args.mode == "code":
                            output, accepted = submit_solution(session, code)
                            status = "completed" if accepted is not None else "rejected"
                        elif name == "submit_answer" and getattr(args, "answer_tool", False):
                            executed.append((name, code))
                            output, accepted = submit_answer(session, code)
                            if accepted is not None and getattr(args, "verify_submit", False):
                                check = check_answer_on_train(session, code)
                                verified = None if check is None else not check[0]
                                if check and check[0] and code not in forced:
                                    forced.add(code)
                                    output, accepted = (
                                        f"[answer not accepted yet: the same expression with test_input = each "
                                        f"training input fails training pairs {check[0]} ({check[1]}). Fix the rule, "
                                        "or call submit_answer again with exactly the same code to submit anyway.]"), None
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
                    submit_code = code
                    break
                note = budget_note(args, total["completion"]) + turns_note(args, index)
                if note and messages[-1]["role"] == "tool":
                    messages[-1]["content"] += note
                if final:
                    stop = "length" if finish == "length" else "max_turns"
                    break
            else:
                stop = "max_turns"
            if session_final and pred is None and stop in {"output_budget", "max_turns", "length", "final"}:
                # Budget or turns ran out without an answer: take the session's `answer` if it is a valid grid.
                output, grid = submit_answer(session, "answer")
                events.append({"fallback": "session_answer", "accepted": grid is not None, "output": output[:300]})
                if grid is not None:
                    pred, content, source, submit_code = grid, json.dumps(grid), "session", "answer"
                elif getattr(args, "fallback_solve", False):
                    # E029: no valid `answer`; evaluate the session's solve on the test input, if defined.
                    output, grid = submit_answer(session, "solve(test_input)")
                    events.append({"fallback": "session_solve", "accepted": grid is not None, "output": output[:300]})
                    if grid is not None:
                        pred, content, source, submit_code = grid, json.dumps(grid), "session_solve", "solve(test_input)"
            if getattr(args, "train_check", False) and pred is not None:
                # E017: label the answer program on the training pairs (no model tokens; host compares hashes).
                try:
                    labels = train_check(session, executed, pred, submit_code, submit=submit_answer,
                                         replay_session=lambda grid: session_factory(
                                             args.sandbox_url, task["train"], grid, args.tool_timeout,
                                             args.max_tool_output, **extra))
                except Exception as exc:
                    labels = {"train_pass": None, "train_frac": None, "train_trace": "check_error",
                              "train_check_error": f"{type(exc).__name__}: {exc}"[:300]}
    except Exception as exc:
        error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
    result = {"content": content, "pred": pred, "tool_calls": n_tools, "tool_errors": n_errors,
              "turns": n_turns, "usage": total, "finish_reason": finish, "stop_reason": stop,
              "usage_complete": error is None, "est_cost_usd": round(cost["total"], 6),
              "seconds": round(time.monotonic() - t0, 1)}
    if auto_price:
        result.update(est_cost_peak_usd=round(cost["peak"], 6), peak_turns=cost["peak_turns"])
    if getattr(args, "cap_nudge", False):
        result["cap_nudges"] = n_nudges
    if getattr(args, "rescue_code", False):
        result["rescues"] = n_rescues
    if tok is not None:
        result["token_exact"] = {k: v for k, v in tok.items() if k not in ("ids", "n", "prev_len", "prev_sampled")}
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
    ap.add_argument("--grid-format", default="json", choices=["json", "image"],
                    help="E033: image = every grid in the prompt is a PNG (ARC palette, --image-cell px per cell, no grid "
                         "lines) instead of JSON; needs a vision model behind --base-url")
    ap.add_argument("--image-cell", type=int, default=32,
                    help="E033: pixels per grid cell (Qwen3.8: 32 = one LLM visual token per cell, 16 = one ViT patch)")
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
    if args.answer_tool and args.mode != "grid":
        ap.error("--answer-tool is for grid mode")
    if args.budget_soft and (not args.budget_notes or not 0 < args.budget_soft < args.max_total_output):
        ap.error("--budget-soft needs --budget-notes and must be below --max-total-output")
    if not 0 <= args.budget_notes < 1 or not 0 < args.budget_warn <= 1:
        ap.error("--budget-notes must be in [0, 1) and --budget-warn in (0, 1]")
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
    if pairs is not None:
        jobs = [job for job in jobs if job[:2] in pairs]
    if getattr(args, "only_attempts", None):  # E043: e.g. attempt 1 in a second process beside a running attempt 0
        keep_attempts = {int(x) for x in args.only_attempts.split(",")}
        jobs = [job for job in jobs if job[2] in keep_attempts]
    if policy:
        jobs += [(t, p, 2) for t, task in tasks.items() for p in range(len(task["test"]))
                 if (pairs is None or (t, p) in pairs) and needs_extra(t, p)]
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
            trace, result = solve(client, args, task, task["test"][p]["input"])
            if "error" in result:
                result["error"] = result["error"].replace(key, "[redacted]")
            path = trace_dir / f"{t}_p{p}_a{a}_{uuid4().hex[:8]}.messages.json.gz"
            with gzip.open(path, "wt") as f:
                json.dump(trace, f)
            rec.update(result, trace_path=str(path))
            if "output" in task["test"][p]:
                rec["correct"] = rec["pred"] == task["test"][p]["output"]
            with lock, out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            return rec

        with ThreadPoolExecutor(args.workers) as executor:
            futures = {executor.submit(work, *job): job for job in jobs}
            index, total_jobs = 0, len(jobs)
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    futures.pop(future)
                    rec = future.result()
                    if rec is None:
                        continue
                    index += 1
                    records.append(rec)
                    latest[rec["task_id"], rec["pair"], rec["attempt"]] = rec
                    if policy and rec["attempt"] in (0, 1) and needs_extra(rec["task_id"], rec["pair"]) \
                            and (rec["task_id"], rec["pair"], 2) not in {futures[f] for f in futures}:
                        futures[executor.submit(work, rec["task_id"], rec["pair"], 2)] = (rec["task_id"], rec["pair"], 2)
                        total_jobs += 1
                    report(rec, index, total_jobs)
    summarize(records, tasks, args.attempts)


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
        trace, result = solve(client, args, task, task["test"][p]["input"])
        path = trace_dir / f"{t}_p{p}_a{a}_{uuid4().hex[:8]}.messages.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(trace, f)
        rec.update(result, trace_path=str(path))
        if "output" in task["test"][p]:
            rec["correct"] = rec["pred"] == task["test"][p]["output"]
        with lock, out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        return rec

    with ThreadPoolExecutor(args.workers) as executor:
        futures = [executor.submit(work, *job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            report(rec, i, len(jobs))
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
