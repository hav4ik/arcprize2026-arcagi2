"""Qwen3.8 tool-use agent for ARC-AGI-2 over a raw text-completion API (E012).

Backend-agnostic: the model is any `LLM` with `async complete(prompt, max_tokens, stop) -> Completion`
(`OpenAICompletionsLLM` covers SGLang / vLLM `/v1/completions`; an offline vLLM engine can implement the same
method). Code runs in an `executor.py` executor (Docker sandbox here, `LocalExecutor` on Kaggle).

Why raw completions instead of chat + server-side tool parsing:
- the prompt only ever grows (every past turn is appended verbatim), so the server's prefix cache hits on every
  turn; the Qwen template keeps the thinking of all turns after the last real user message anyway, so this is
  exactly what the chat template would render (checked by run_qwen_tools.py --verify-template);
- a turn cut by max_tokens, or an EOS in the middle of thinking, is continued in place;
- tool calls are parsed here (`parse_tool_calls`), so parse failures are visible and get feedback;
- no dependence on a particular server's tool / reasoning parser.

Harness logic follows E009 sub80s40 (DeepSeek's best): grid mode, `lean2sub` prompt, `run_python` +
`submit_answer` tools, budget note on every tool output with a soft target and hard cap, and at the hard cap a
forced `submit_answer` call (the model only writes the expression) with the session `answer` as fallback.
Exactly one scored grid per attempt; `answer_source` is submit | forced | parse | lenient | session | salvage | None.

answer_mode="program" (qwen-tools-4, E019): the answer can only be a program. `submit_answer` is replaced by
`submit_program(code)`; `code` must define solve(grid). The harness runs it in a fresh process on every training
input (host-side exact comparison) and only if all pass on the test input; failures return feedback and the attempt
continues (at most `max_submissions`). No forced typed answer at the cap: the fallback is the best submitted program
or the session's `solve`, whichever reproduces more training pairs. `answer_source` is program_pass |
program_best_fail | session_solve | None.
"""
import asyncio
import json
import re
import threading
import time
from dataclasses import asdict, dataclass

from .arcprize import build_prompt, parse
from .executor import (check_answer_on_train, program_feedback, run_program, run_session_solve, salvage_answer,
                       score_outputs, submit_answer, tool_output)
from .grids import valid_grid
from .prompts import PROMPT_SUFFIXES

HARNESS_VERSION = "qwen-tools-4"  # 2: forced-submit text names the expression; 3: robustness (R2/R3/R7/R8, S4/S7/S8)
# 4: opt-in answer_mode="program" (E019) and cache/train-check bookkeeping; answer_mode="submit" behaves as 3
# 5 (E020) is recorded only when an E020 option is active (V5_DEFAULTS); with all of them at their defaults the
# harness behaves exactly as 4 and records "qwen-tools-4", so older result files still resume.
HARNESS_VERSION_V5 = "qwen-tools-5"
V5_DEFAULTS = {"task_summary": False, "helpers": False, "turn_think_cap": 0, "turn_think_grace": 2048,
               "feedback": "basic", "confirm_test": False, "test_audit": False, "lint": False,
               "worked_example": False, "record_alternatives": False,
               "think_call_rescue": False, "tool_desc": "", "solve_fallback": False,
               "effort_header": "", "turns_note": 0, "budget_note_cap": 0}
V5_PROMPTS = {"lean2progroles", "lean2roles"}
CONTEXT = 262144
PROGRAM_MEMORY_MB = 3072  # address-space limit of a submitted program's process (runner --session-mem-mb lowers it;
# an execution-safety setting, deliberately not part of the config so result files resume)
THINK_CLOSE = "</think>"
GEN_PROMPT = "<|im_start|>assistant\n<think>\n"
TIER_SYSTEM = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
             "answer.",
    "medium": None,
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.",
}
# E009 lean2sub: lean2 + the submit_answer hand-off (byte-identical to run_deepseek_tools.py).
_SUBMIT = ("\n\nWhen your code reproduces every training output and its assumptions hold on test_input, call "
           "submit_answer with an expression such as `solve(test_input)`: the harness evaluates it in your session "
           "and that grid is your answer, so never retype the grid. Keep your current best guess for the test "
           "output in a variable `answer`; if you run out of budget it is submitted automatically.")
_PROGRAM = ("\n\nYour answer must be a program: typed grids are not accepted. When your code reproduces every "
            "training output and its assumptions hold on test_input, call submit_program with self-contained code "
            "that defines `solve(grid)`. The harness runs it on every training input and checks the outputs exactly "
            "before it runs it on the test input; if a pair fails it tells you what went wrong and you can fix the "
            "program and submit again. Keep your latest `solve` defined in your session too: if you run out of "
            "budget, the best program you submitted (or that `solve`) is used.")
# E017 P2 roles text (byte-identical to run_deepseek_tools.py lean2roles), on top of lean2prog.
_ROLES = ("\n\nBefore writing solve(), write a short ROLES note: what each kind of object or colour does (template, "
          "marker, key/legend, obstacle, container, background). Do not call anything noise or irrelevant unless code "
          "shows it is removed or unchanged in every training output. When solve() reproduces every training output, "
          "print what is different about test_input compared with all training inputs (sizes, colours, objects clipped "
          "by the border, counts, key entries, shapes not seen) and say in one line how solve handles each difference; "
          "if your rule ignores some non-background cells of test_input, explain them before submitting. If any "
          "training pair still fails, or a test difference is unhandled, do not submit: fix the rule.")
PROMPTS = {"lean2sub": PROMPT_SUFFIXES["lean2"] + _SUBMIT, "lean2prog": PROMPT_SUFFIXES["lean2"] + _PROGRAM,
           "lean2progroles": PROMPT_SUFFIXES["lean2"] + _PROGRAM + _ROLES,
           # E028: run_deepseek_tools.py lean2roles (= lean2sub + roles), the prompt of the E025 SFT rollouts
           "lean2roles": PROMPT_SUFFIXES["lean2"] + _SUBMIT + _ROLES}
# E017 P3 (--helpers): evals/arc_helpers.py preloaded in the session and in every submitted program's namespace.
_HELPERS_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "arc_helpers.py"
HELPERS_SRC = _HELPERS_PATH.read_text() if _HELPERS_PATH.exists() else ""
HELPERS_DOC = (  # byte-identical to run_deepseek_tools.py HELPERS_DOC
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
# E020 --worked-example: a short workflow demonstration on ARC *training* task 3aa6fb7a (never an eval task), with
# the real outputs of its code (objects/check are the --helpers functions).
_EXAMPLE_PATH = __import__("pathlib").Path(__file__).resolve().parent / "worked_example.txt"
WORKED_EXAMPLE = _EXAMPLE_PATH.read_text().rstrip() if _EXAMPLE_PATH.exists() else ""
HELPERS_PROGRAM_DOC = " The grid helpers listed for run_python (objects, crop, d8, ...) are preloaded here too."
# E020 per-turn thinking cap: after `turn_think_cap` tokens of thinking in one turn, this note is appended to the
# thinking (in the model's voice); if the thinking is still open `turn_think_grace` tokens later, it is closed and
# a run_python call is opened (the model writes the code).
THINK_NUDGE = ("\n\nI have been reasoning for a long time without running anything. I should stop here and test my "
               "current hypothesis with code now (and use code, not my head, for any counting or cell reading).\n")
# E032 (e031 mode): run_deepseek_tools.py --effort-header texts (byte-identical), rendered as the system message after
# the tools block in the system turn, as the HF chat template and the E032 training render place it.
EFFORT_HEADERS = {
    "minimal": ("Reasoning Effort: Minimal. Keep your thinking short and terse: write brief notes, not full sentences. "
                "Do not restate the task, do not re-read or re-derive what code has already printed, and do not re-verify "
                "what has already been checked. Test ideas with small code experiments instead of long deliberation, and "
                "submit as soon as the rule reproduces every training pair."),
}
CONFIRM_TEXT = ("[program passes all {n} training pairs. Its output for test_input ({h}x{w}):\n{grid}\n"
                "Check it critically against your rule and against what is different about test_input. If it is "
                "right, call submit_program again with the same code: that makes it your answer. If not, fix solve "
                "and submit the fixed program.]")


def _review(cfg, train, test_input, rec, code, confirm):
    """The one-time review text for the first train-passing program, or None to accept it directly
    (lint only, nothing flagged)."""
    from .audit import audit_report, lint, test_audit
    out = rec["test"]
    flags = []
    if cfg.test_audit:
        try:
            from task_summary import task_summary
            summary = task_summary(train, test_input)
        except Exception:
            summary = None
        flags = test_audit(train, test_input, out, rec.get("runtimes"), summary)
    lint_flags = lint(code, train) if cfg.lint else []
    if not (cfg.confirm_test or cfg.test_audit) and not lint_flags:
        return None
    confirm["shown"], confirm["audit"] = out, {"flags": flags, "lint": lint_flags}
    if not (cfg.test_audit or lint_flags):
        return CONFIRM_TEXT.format(n=len(train), h=len(out), w=len(out[0]), grid=_digit_rows(out))
    return audit_report(len(train), out, flags, lint_flags, _digit_rows(out))


def _digit_rows(g):
    return "\n".join("".join(str(v) for v in row) for row in g)


THINK_FORCE_ACT = "\n\nLet me test this with code now.\n</think>\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n"
# A final text reply in program mode (no tool call): typed answers are not accepted.
PROGRAM_NUDGE = ("Typed answers are not accepted. Submit your answer as a program: call submit_program with "
                 "self-contained code that defines solve(grid).")
TOOLS_HEADER = ("# Tools\n\nYou have access to the following functions:\n\n<tools>")
TOOLS_FOOTER = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>")
# At the hard cap: close the thinking and open a submit_answer call; the model writes only the expression.
FORCE_THINK = ("\n\nI am out of budget, so I must submit my best answer now as one short expression: `answer` if I "
               "set it, otherwise solve(test_input) or the name of my best function applied to test_input.\n</think>\n\n")
FORCE_CALL = "<tool_call>\n<function=submit_answer>\n<parameter=code>\n"
LOW_YIELD = 20


@dataclass
class AgentConfig:
    tier: str = "xhigh"
    prompt: str = "lean2sub"
    max_total_output: int = 80000   # hard cap: generated tokens per attempt, all turns
    budget_soft: int = 40000        # soft target stated in the budget notes (0 = none)
    budget_notes: float = 0.01      # notes on tool outputs once this fraction of the hard cap is used (0 = off)
    budget_warn: float = 0.75       # from this fraction on, the note says to submit now
    soft_wording: str = "gentle"    # past the soft target: gentle = submit once a rule fits; strong = submit now (E010)
    chunk_tokens: int = 16384       # max tokens per completion request (a turn may take several)
    max_turns: int = 60
    tool_timeout: float = 30
    max_tool_output: int = 16000
    verify_submit: bool = False     # E011: re-run the submitted expression on train inputs; reject once on mismatch
    force_submit_tokens: int = 256  # tokens for the forced submit_answer expression at the cap (0 = off)
    force_submit_last: bool = False  # E032 Kaggle: force the submit only if the session `answer`, solve() and salvage
    #                                  fallbacks all come up empty (so a forced answer never replaces them)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_eos_continues: int = 4      # EOS inside thinking: continue the turn at most this many times
    context: int = CONTEXT          # the server's context length (max_model_len); Kaggle runs 131072
    salvage: bool = True            # no answer at the end: apply the session's best-scoring function (qwen-tools-3)
    exec_margin: float = 35         # hard wall-clock cap per executor call = tool_timeout + exec_margin
    answer_mode: str = "submit"     # submit (qwen-tools-3 behaviour) | program (E019: submit_program only)
    max_submissions: int = 6        # program mode: submit_program calls per attempt
    program_timeout: float = 10     # program mode: seconds per grid when running a submitted program
    max_nudges: int = 2             # program mode: text replies answered with PROGRAM_NUDGE before the attempt ends
    record_train_check: bool = False  # submit mode: after the attempt, train-check the submitted expression (no tokens)
    # E020 (qwen-tools-5), all opt-in:
    task_summary: bool = False      # append the E017 host-computed perception pack (evals/task_summary.py)
    helpers: bool = False           # preload evals/arc_helpers.py in the session and in submitted programs
    turn_think_cap: int = 0         # thinking tokens per turn before THINK_NUDGE (0 = off)
    turn_think_grace: int = 2048    # tokens after the nudge before the thinking is closed into a run_python call
    feedback: str = "basic"         # program mode: basic | rich (adds colour confusions and small-grid diffs)
    confirm_test: bool = False      # program mode: show the first train-passing test output; accept on re-submit
    test_audit: bool = False        # program mode: like confirm_test, plus host-side audit flags (audit.py)
    lint: bool = False              # program mode: flag hardcoded constants (audit.lint) in rejected and passing programs
    worked_example: bool = False    # append the worked_example.txt workflow demo (ARC training task; needs --helpers)
    record_alternatives: bool = False  # bookkeeping: other test grids of the attempt (free attempt_2), no tokens
    think_call_rescue: bool = False  # after a think nudge, a complete <tool_call> inside open thinking is executed
    # E027 (opt-in): one-time transcript compaction to cut KV, prefill and attention cost of long attempts.
    compact_at: int = 0             # estimated prompt tokens that trigger a compaction (0 = off); again after +50%
    compact_keep: int = 2           # most recent assistant turns / tool-response messages left untouched
    compact_tool_chars: int = 1500  # older tool responses are cut to this many characters
    # E028 (opt-in): "e025" = the interface the SFT model was trained on (E025 DeepSeek rollouts): tool specs exactly
    # as experiments/E028_qwen38_sft/e025_tools.json and run_deepseek_tools.py's strings for tool errors, skips and
    # executor timeouts. Use with prompt="lean2roles", helpers, task_summary, submit mode and a LocalExecutor with
    # style="ipython" (see experiments/E028_qwen38_sft/HARNESS_E025_MODE.md).
    tool_desc: str = ""
    solve_fallback: bool = False    # no answer at the end and no valid `answer`: try solve(test_input) before salvage
    # E032 (opt-in): the E031 interface (experiments/E032_qwen38_sft_e031): tool_desc="e031" (the default run_python +
    # submit_answer texts, run_deepseek_tools.py's error/skip/timeout strings as in e025), prompt lean2sub, no helpers
    # or summary, effort_header="minimal", turns_note=5, max_turns=40, budget_soft=20000, soft_wording="strong",
    # budget_note_cap=60000 and a LocalExecutor with style="ipython" (E031_CONFIG; tests: evals/tests/test_qwen_e031.py).
    effort_header: str = ""         # EFFORT_HEADERS key: system message after the tools block ("" = none)
    turns_note: int = 0             # once this many or fewer turns remain, a note on the last tool output (0 = off)
    budget_note_cap: int = 0        # hard limit the budget notes state and scale with (0 = max_total_output); lets the
    # notes keep the training wording (e.g. 20k / 60k) when the real per-attempt cap is lower


# E032: AgentConfig fields that give the E031 interface (run_deepseek_tools.py E031 config: lean2sub, answer tool,
# final-mode session, lenient parse, s20x60 notes with strong wording, --effort-header minimal, --turns-note 5,
# 40 turns, 30 s tool timeout). The E032 model was trained on exactly this render (replay: evals/tests/test_qwen_e031.py).
# On Kaggle, max_total_output may be lower than 60000: budget_note_cap keeps the notes' training wording.
E031_CONFIG = {"tier": "xhigh", "prompt": "lean2sub", "tool_desc": "e031", "effort_header": "minimal", "turns_note": 5,
               "max_turns": 40, "max_total_output": 60000, "budget_soft": 20000, "soft_wording": "strong",
               "budget_notes": 0.01, "budget_warn": 0.75, "budget_note_cap": 60000, "tool_timeout": 30,
               "max_tool_output": 16000, "answer_mode": "submit", "helpers": False, "task_summary": False}


# ---------------------------------------------------------------- rendering (Qwen3.8 chat template, tools on)

def tool_specs(cfg):
    if cfg.tool_desc == "e025":  # run_deepseek_tools.tools_for() with E025's flags (grid, helpers, answer tool)
        return [{"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                           "required": ["code"], "additionalProperties": False},
        }} for name, description in (
            ("run_python", "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
                           "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
                           f"scipy is available. No internet or host files. Timeout {cfg.tool_timeout:g}s. "
                           "Use print() to inspect results. Check transformations on all training pairs." + HELPERS_DOC),
            ("submit_answer", "End this attempt with your test output. `code` is a Python expression evaluated in your "
                              "session, e.g. solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or "
                              "numpy array); statements that set a variable `answer` also work. The harness reads the "
                              "grid directly: do not retype it. Invalid grids return an error and the attempt "
                              "continues."))]
    descriptions = {
        "run_python": (
            "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
            "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
            f"scipy is available. No internet or host files. Timeout {cfg.tool_timeout:g}s. "
            "Use print() to inspect results. Check transformations on all training pairs."),
        "submit_answer": (
            "End this attempt with your test output. `code` is a Python expression evaluated in your session, "
            "e.g. solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or numpy array); "
            "statements that set a variable `answer` also work. The harness reads the grid directly: do not "
            "retype it. Invalid grids return an error and the attempt continues."),
    }
    if cfg.answer_mode == "program":
        del descriptions["submit_answer"]
        descriptions["submit_program"] = (
            "Submit your answer as a program; this is the only way to answer (typed grids are not accepted). "
            "`code` is self-contained Python source that defines solve(grid): grid is a list of lists of ints, "
            "and solve returns the output grid (list of lists or numpy array of ints 0-9). It runs in a fresh "
            "Python process: numpy is preloaded as np, but include every other import and helper function (your "
            "session variables are not available). The harness first runs solve on every training input and "
            "compares the results exactly with the training outputs. If any pair fails, you get feedback (failing "
            "pairs, shapes, wrong cells, errors) and the attempt continues. If all pass, it runs solve on the test "
            f"input and that grid is your answer, which ends the attempt. Limit {cfg.program_timeout:g}s per grid "
            f"and {cfg.max_submissions} submissions per attempt.")
    if cfg.helpers:
        descriptions["run_python"] += HELPERS_DOC
        if "submit_program" in descriptions:
            descriptions["submit_program"] += HELPERS_PROGRAM_DOC
    return [{"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"], "additionalProperties": False},
    }} for name, description in descriptions.items()]


def render_system(tier, tools, system=None):
    """system: a system message, placed after the tools block (where the Qwen3.8 chat template puts it)."""
    s = "<|im_start|>system\n"
    if TIER_SYSTEM[tier]:
        s += TIER_SYSTEM[tier] + "\n\n"
    s += TOOLS_HEADER + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + TOOLS_FOOTER
    if system:
        s += "\n\n" + system.strip()
    return s + "<|im_end|>\n"


def render_user(content):
    return f"<|im_start|>user\n{content.strip()}<|im_end|>\n"


def render_tool_responses(outputs):
    return "<|im_start|>user" + "".join(f"\n<tool_response>\n{o.strip()}\n</tool_response>" for o in outputs) + \
        "<|im_end|>\n"


def render_assistant(reasoning, content, calls):
    """A past assistant turn as the chat template renders it (used to check our verbatim appends)."""
    s = f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
    for i, (name, params) in enumerate(calls):
        s += ("\n\n" if content.strip() else "") if i == 0 else "\n"
        s += f"<tool_call>\n<function={name}>\n"
        s += "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
        s += "</function>\n</tool_call>"
    return s + "<|im_end|>\n"


def user_content(train, test_input, prompt, helpers=False, summary=False, example=False):
    text = build_prompt(train, test_input) + PROMPTS[prompt]
    if helpers:
        text += HELPERS_PROMPT
    if example:
        text += WORKED_EXAMPLE
    if summary:  # E017 P1, imported lazily (needs evals/ on sys.path; not used on Kaggle unless enabled)
        from task_summary import task_summary
        text += task_summary(train, test_input)
    return text


def initial_prompt(train, test_input, cfg, note=None):
    """note (E020 --memory): extra text appended to the first user message (e.g. earlier attempts' failures)."""
    return render_system(cfg.tier, tool_specs(cfg), EFFORT_HEADERS[cfg.effort_header] if cfg.effort_header else None) \
        + render_user(
        user_content(train, test_input, cfg.prompt, cfg.helpers, cfg.task_summary, cfg.worked_example)
        + (note or "")) + GEN_PROMPT


_SEG = "<|im_start|>"
_THINK_BODY = re.compile(r"(<\|im_start\|>assistant\n<think>\n).*?(\n?</think>)", re.DOTALL)
_TOOL_RESP = re.compile(r"(<tool_response>\n)(.*?)(\n</tool_response>)", re.DOTALL)


def compact(prompt, keep=2, tool_chars=1500):
    """E027: the transcript with the thinking of all but the last `keep` finished assistant turns emptied (the
    rendering Qwen's chat template uses for past turns) and tool responses in all but the last `keep` tool messages
    cut to `tool_chars` characters. The system prompt, the task message, nudges, tool calls, assistant replies
    and the open turn at the end are kept verbatim. Returns (new_prompt, characters_removed)."""
    parts = prompt.split(_SEG)
    head, segs = parts[0], [_SEG + p for p in parts[1:]]
    done_asst = [i for i, s in enumerate(segs) if s.startswith(_SEG + "assistant\n") and s.endswith("<|im_end|>\n")]
    tool_msgs = [i for i, s in enumerate(segs) if s.startswith(_SEG + "user") and "<tool_response>" in s]
    old_asst = set(done_asst[:-keep] if keep else done_asst)
    old_tool = set(tool_msgs[:-keep] if keep else tool_msgs)

    def cut(m):
        body = m.group(2)
        if len(body) <= tool_chars or body.endswith(" characters cut]"):  # short, or cut by an earlier compaction
            return m.group(0)
        return m.group(1) + body[:tool_chars] + f"\n[... {len(body) - tool_chars} characters cut]" + m.group(3)

    out = []
    for i, s in enumerate(segs):
        if i in old_asst:
            s = _THINK_BODY.sub(lambda m: m.group(1) + "\n</think>", s, count=1)
        elif i in old_tool:
            s = _TOOL_RESP.sub(cut, s)
        out.append(s)
    new = head + "".join(out)
    return new, len(prompt) - len(new)


# ---------------------------------------------------------------- parsing

_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNC = re.compile(r"^\s*<function=([^>\n]+)>(.*?)</function>\s*$", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def split_turn(text):
    """(reasoning, content) of one generated assistant turn; content is None while still thinking."""
    i = text.find(THINK_CLOSE)
    if i < 0:
        return text, None
    return text[:i], text[i + len(THINK_CLOSE):]


def parse_tool_calls(content):
    """Tool calls in the answer part of a turn: (calls, errors). calls = [(name, {param: str})].

    errors lists malformed blocks (unclosed <tool_call>, missing <function=...>)."""
    calls, errors = [], []
    for block in _CALL.findall(content):
        m = _FUNC.match(block)
        if not m:
            errors.append("malformed <tool_call> block: expected <function=NAME>...</function>")
            continue
        calls.append((m.group(1).strip(), {k.strip(): v for k, v in _PARAM.findall(m.group(2))}))
    tail = _CALL.sub("", content)
    if "<tool_call>" in tail or "<function=" in tail:
        errors.append("unclosed or malformed tool call: use <tool_call><function=NAME><parameter=code>..."
                      "</parameter></function></tool_call>")
    return calls, errors


def lenient_parse(text):
    """Last valid JSON grid in the text (trailing commas tolerated), else the last block of one-row-per-line."""
    text = text or ""
    for start in [i for i, ch in enumerate(text) if ch == "["][::-1]:
        if not re.match(r"\[\s*\[", text[start:start + 50]):
            continue
        depth = 0
        for j in range(start, len(text)):
            depth += (text[j] == "[") - (text[j] == "]")
            if depth == 0:
                block = text[start:j + 1]
                for candidate in (block, re.sub(r",\s*\]", "]", block)):
                    try:
                        grid = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if valid_grid(grid):
                        return grid
                break
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


def budget_note(cfg, used):
    """Status line appended to the last tool output of a turn (E009 wording)."""
    hard = cfg.budget_note_cap or cfg.max_total_output  # E032: the stated limit may differ from the real cap
    if not cfg.budget_notes or used < cfg.budget_notes * hard:
        return ""
    tool = "submit_program" if cfg.answer_mode == "program" else "submit_answer"
    if cfg.budget_soft:
        note = (f"\n[budget: {used / 1000:.0f}k output tokens used; aim to finish within {cfg.budget_soft / 1000:.0f}k "
                f"(hard limit {hard / 1000:.0f}k)]")
        if cfg.budget_soft <= used < cfg.budget_warn * hard:
            note += (f" Past the target: submit your best answer now with {tool}, unless one more experiment "
                     "will very likely make every training pair pass." if cfg.soft_wording == "strong"
                     else " Past the target: submit as soon as a rule fits every training pair.")
    else:
        note = f"\n[budget: {used / 1000:.0f}k of {hard / 1000:.0f}k output tokens used]"
    if used >= cfg.budget_warn * hard:
        note += (f" Budget nearly spent: submit your best answer now with {tool} "
                 + ("(or at least keep `solve` defined in your session)." if tool == "submit_program"
                    else "(or at least set `answer`)."))
    return note


def turns_note(cfg, turn):
    """run_deepseek_tools.py --turns-note N (E029/E031) after assistant turn `turn` (1-based): once N or fewer turns
    remain, a reminder to answer now."""
    left = cfg.max_turns - turn
    if not cfg.turns_note or left > cfg.turns_note or left <= 0:
        return ""
    if cfg.answer_mode == "program":
        return (f"\n[{left} turn{'s' if left != 1 else ''} left: call submit_program with your best program now, "
                "or at least keep `solve` defined in your session.]")
    return (f"\n[{left} turn{'s' if left != 1 else ''} left: call submit_answer with your best answer now, "
            "or at least set `answer`.]")


# ---------------------------------------------------------------- model backend

@dataclass
class Completion:
    text: str
    completion_tokens: int
    prompt_tokens: int
    finish: str  # "stop" (EOS or stop string) or "length"
    cached_tokens: int = None  # prompt tokens served from the server's prefix cache, when it reports them
    served_by: str = None      # worker that served the request, when a gateway reports it in a response header


class OpenAICompletionsLLM:
    """Raw /v1/completions on an OpenAI-compatible server (SGLang, vLLM). Retries transient errors only:
    a completion has no side effects, so a retry is safe."""

    def __init__(self, base_url, api_key, model, temperature=1.0, top_p=0.95, top_k=20, timeout=3600,
                 max_retries=8, extra_body=None):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key or "none", base_url=base_url, timeout=timeout, max_retries=0)
        self.model, self.temperature, self.top_p, self.top_k = model, temperature, top_p, top_k
        self.max_retries, self.extra_body = max_retries, extra_body or {}
        self.transient_errors = 0

    async def complete(self, prompt, max_tokens, stop=None, extra_headers=None):
        import openai
        errs = 0
        while True:
            try:
                raw = await self.client.completions.with_raw_response.create(
                    model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=self.temperature,
                    top_p=self.top_p, stop=stop, extra_body={"top_k": self.top_k, **self.extra_body},
                    extra_headers=extra_headers)
                r = raw.parse()
                served = next((v for k, v in raw.headers.items() if "worker" in k.lower()), None)
                c = r.choices[0]
                details = getattr(r.usage, "prompt_tokens_details", None)
                cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
                return Completion(c.text, r.usage.completion_tokens, r.usage.prompt_tokens, c.finish_reason, cached,
                                  served)
            except (openai.APIConnectionError, openai.APITimeoutError, openai.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status < 500 and status not in (408, 429):
                    raise
                errs += 1
                self.transient_errors += 1
                if errs > self.max_retries:
                    raise
                await asyncio.sleep(min(60, 5 * 2 ** min(errs - 1, 4)))

    async def close(self):
        await self.client.close()


class RoutedLLM:
    """Per-attempt view of an OpenAICompletionsLLM that sends a fixed routing header on every request (E019: the
    SGLang gateway's manual routing pins one attempt's conversation to one worker, so its prefix cache is reused)."""

    def __init__(self, llm, header, key):
        self.llm, self.headers = llm, {header: key}

    async def complete(self, prompt, max_tokens, stop=None):
        return await self.llm.complete(prompt, max_tokens, stop, extra_headers=self.headers)


# ---------------------------------------------------------------- agent loop

async def _call(fn, *args):
    # Blocking executor calls run in the loop's default thread pool. With many concurrent attempts, set a pool with
    # one thread per attempt: loop.set_default_executor(ThreadPoolExecutor(concurrency + 4)).
    return await asyncio.to_thread(fn, *args)


class ExecutorTimeout(RuntimeError):
    """An executor call did not return within its wall-clock cap; the session was replaced."""


def _quiet_close(session):
    try:
        session.close()
    except Exception:
        pass


class _Session:
    """The attempt's executor behind a hard wall-clock cap per call (the executor's own timeout is advisory: a
    wedged sandbox or child must not block the attempt). On a cap hit the session is abandoned (closed in a
    background thread) and a fresh one with only the task data takes its place."""

    def __init__(self, make_executor, train, test_input, call_limit, start_limit=180):
        self.make, self.train, self.test_input = make_executor, train, test_input
        self.call_limit, self.start_limit = call_limit, start_limit
        self.session, self.resets = None, 0

    async def start(self):
        session = self.make(self.train, self.test_input)
        try:
            await asyncio.wait_for(_call(session.__enter__), self.start_limit)
        except BaseException:
            threading.Thread(target=_quiet_close, args=(session,), daemon=True).start()
            raise
        self.session = session

    async def run(self, fn, *args):
        if self.session is None:
            raise RuntimeError("no executor session")
        try:
            return await asyncio.wait_for(_call(fn, self.session, *args), self.call_limit)
        except asyncio.TimeoutError:
            old, self.session = self.session, None
            threading.Thread(target=_quiet_close, args=(old,), daemon=True).start()
            self.resets += 1
            await self.start()
            raise ExecutorTimeout(f"executor call exceeded {self.call_limit:g}s") from None

    async def close(self):
        if self.session is not None:
            session, self.session = self.session, None
            try:
                await asyncio.wait_for(_call(session.close), 30)
            except BaseException:
                threading.Thread(target=_quiet_close, args=(session,), daemon=True).start()


def _execute(session, code):
    return session.execute(code)


def estimate_tokens(known_tokens, known_chars, text):
    """Prompt tokens of `text`, given that its first `known_chars` characters were measured as `known_tokens`.
    The unmeasured rest (tool output, the turn so far) is counted at 2 characters per token (conservative for
    JSON grids)."""
    return known_tokens + max(0, len(text) - known_chars + 1) // 2


async def solve(llm, make_executor, train, test_input, cfg, note=None):
    """One attempt. Returns (transcript, result). transcript is the full raw prompt+generation text.

    make_executor(train, test_input) -> executor (context manager with .execute(code)); train holds the
    training pairs with outputs (used host-side for --verify-submit), test_input the test grid only.
    Never raises (except on cancellation): errors are recorded, and any answer already set is still taken."""
    prompt = initial_prompt(train, test_input, cfg, note)
    used, turns, n_calls, n_tool_err, n_parse_err, n_eos, n_chunks = 0, 0, 0, 0, 0, 0, 0
    pred = source = stop = error = salvage_info = None
    deferred_force = deferred_entry = None  # force_submit_last
    max_prompt, content, submit_verified, forced_codes = 0, "", None, set()
    known_tokens, known_chars = 0, 0  # last measured prompt (tokens, chars) for the context estimate
    turn_log, t0 = [], time.monotonic()
    ex = _Session(make_executor, train, test_input, cfg.tool_timeout + cfg.exec_margin)
    program_mode = cfg.answer_mode == "program"
    answer_tool = "submit_program" if program_mode else "submit_answer"
    submissions, n_nudges, answer_code, program_info = [], 0, None, {}
    cache = {"prompt": 0, "cached": 0, "ideal": 0, "reported": 0, "last": ""}
    last_solve_cell, servers = None, {}
    think_stats = {"nudges": 0, "forced": 0, "rescued": 0}
    compact_state = {"n": 0, "removed": 0, "next": cfg.compact_at}  # E027
    confirm = {"shown": None}

    async def complete(text, max_tokens, stop_strings=None):
        nonlocal used, n_chunks, max_prompt, known_tokens, known_chars
        c = await llm.complete(text, max_tokens, stop_strings) if stop_strings else await llm.complete(text, max_tokens)
        n_chunks += 1
        used += c.completion_tokens
        max_prompt = max(max_prompt, c.prompt_tokens)
        cache["prompt"] += c.prompt_tokens
        # ideal hit: everything sent before (previous prompt + its completion) is a prefix of this prompt
        cache["ideal"] += min(known_tokens, c.prompt_tokens) if text.startswith(cache["last"]) else 0
        cache["last"] = text + c.text
        if getattr(c, "served_by", None):
            servers[c.served_by] = servers.get(c.served_by, 0) + 1
        if getattr(c, "cached_tokens", None) is not None:
            cache["cached"] += c.cached_tokens
            cache["reported"] += 1
        known_tokens, known_chars = c.prompt_tokens + c.completion_tokens, len(text) + len(c.text)
        return c

    async def run_on(code, grids):
        """Submitted program on each grid in a fresh process; executor timeouts count as errors."""
        outs = []
        for g in grids:
            t_run = time.monotonic()
            try:
                outs.append(await ex.run(run_program, code, g, cfg.program_timeout, PROGRAM_MEMORY_MB,
                                         HELPERS_SRC if cfg.helpers else ""))
            except ExecutorTimeout:
                outs.append({"error": "timeout (executor call cap; Python session restarted)", "kind": "timeout"})
            outs[-1]["seconds"] = round(time.monotonic() - t_run, 2)
            err = outs[-1].get("error") or ""
            if outs[-1].get("kind") in ("SyntaxError", "IndentationError") or "does not define solve(grid)" in err:
                outs += [outs[-1]] * (len(grids) - len(outs))  # same failure on every grid
                break
        return outs

    async def check_program(code):
        """Train-check a submitted program, then the test input if all train pairs pass. Returns a record."""
        train_outs = await run_on(code, [pair["input"] for pair in train])
        rows = score_outputs(train_outs, train)
        n_pass = sum(r["ok"] for r in rows)
        runtimes = [o.get("seconds", 0) for o in train_outs]
        rec = {"n_pass": n_pass, "n_train": len(train), "train_frac": n_pass / len(train), "code": code,
               "fail_kinds": sorted({r.get("kind") or ("shape" if "shape" in r else "cells")
                                     for r in rows if not r["ok"]}), "test": None, "test_error": None}
        rec["runtimes"] = runtimes
        if n_pass == len(train):
            out = (await run_on(code, [test_input]))[0]
            rec["runtimes"] = runtimes + [out.get("seconds", 0)]
            if "error" in out:
                rec["test_error"] = out["error"].splitlines()[-1] if out["error"] else "error"
            elif not valid_grid(out["out"]):
                from .executor import grid_shape
                rec["test_error"] = f"not a valid grid ({grid_shape(out['out'])})"
            else:
                rec["test"] = out["out"]
        return rec, rows

    e025 = cfg.tool_desc in ("e025", "e031")  # both: run_deepseek_tools.py's strings (E031 used the same harness)
    # E028 e025 mode: an executor-cap hit reads like the sandbox's killed-shell timeout (run_deepseek_tools.py)
    killed = {"process_status": "timeout", "stdout": "", "state_reset": True,
              "stderr": f"Execution timed out after {cfg.tool_timeout:g} seconds\n"}

    async def submit(code):
        try:
            return await ex.run(submit_answer, code)
        except ExecutorTimeout:
            if e025:
                return tool_output(killed, cfg.max_tool_output)[:2000] + "\n[answer not accepted]", None
            return "[answer not accepted] evaluation timed out; Python state lost (train/test_input/np restored)", None

    try:
        try:
            await ex.start()
            while True:
                if turns >= cfg.max_turns:
                    stop = "max_turns"
                    break
                turns += 1
                if cfg.compact_at and turns > 1:  # E027: compact a long transcript once per +50% growth
                    est = estimate_tokens(known_tokens, known_chars, prompt)
                    if est > compact_state["next"]:
                        new_prompt, removed = compact(prompt, cfg.compact_keep, cfg.compact_tool_chars)
                        if removed > 0:
                            ratio = len(new_prompt) / max(1, len(prompt))
                            prompt, known_tokens, known_chars = new_prompt, int(est * ratio), len(new_prompt)
                            compact_state["n"] += 1
                            compact_state["removed"] += removed
                        compact_state["next"] = max(cfg.compact_at,
                                                    estimate_tokens(known_tokens, known_chars, prompt)
                                                    + cfg.compact_at // 2)
                text, low_streak, after_eos, out_of_budget = "", 0, False, None
                turn_tokens = 0
                nudge_at, act_forced, nudge_pos = None, False, 0  # E020 per-turn thinking cap state
                while True:  # one assistant turn, possibly over several requests
                    left = cfg.max_total_output - used
                    room = cfg.context - estimate_tokens(known_tokens, known_chars, prompt + text) - 64
                    if left <= 0 or room < 256:
                        out_of_budget = "output_budget" if left <= 0 else "context"
                        break
                    want = min(cfg.chunk_tokens, left, room)
                    capped = bool(cfg.turn_think_cap) and not act_forced and THINK_CLOSE not in text
                    if capped:
                        limit = cfg.turn_think_cap if nudge_at is None else nudge_at + cfg.turn_think_grace
                        want = min(want, max(1, limit - turn_tokens))
                    c = await complete(prompt + text, want)
                    turn_tokens += c.completion_tokens
                    text += c.text
                    if after_eos:
                        low_streak = low_streak + 1 if c.completion_tokens < LOW_YIELD else 0
                    after_eos = False
                    if cfg.think_call_rescue and nudge_at is not None and THINK_CLOSE not in text:
                        # E021 finding: after the nudge the model may write its tool call inside the still-open
                        # thinking and stop; close the thinking before the call and keep the call
                        i = text.find("<tool_call>", nudge_pos)
                        j = text.find("</tool_call>", i) if i >= 0 else -1
                        if j >= 0:
                            text = text[:i] + THINK_CLOSE + "\n\n" + text[i:j + len("</tool_call>")]
                            think_stats["rescued"] += 1
                            break
                    if capped and THINK_CLOSE not in text and turn_tokens >= limit:
                        if nudge_at is None:  # first: a note in the model's voice; it may keep thinking briefly
                            text += THINK_NUDGE
                            nudge_at, nudge_pos = turn_tokens, len(text)
                            think_stats["nudges"] += 1
                        else:  # still thinking after the grace: close it and open a run_python call
                            text += THINK_FORCE_ACT
                            act_forced = True
                            think_stats["forced"] += 1
                        continue
                    if c.finish == "length":
                        continue
                    if THINK_CLOSE not in text and n_eos < cfg.max_eos_continues and low_streak < 2:
                        n_eos += 1  # EOS inside thinking: let it carry on
                        after_eos = True
                        continue
                    break
                reasoning, answer_part = split_turn(text)
                entry = {"turn": turns, "tokens": turn_tokens, "closed_think": answer_part is not None}
                if nudge_at is not None:
                    entry["think_nudge"] = nudge_at
                    if act_forced:
                        entry["think_forced"] = True
                turn_log.append(entry)
                if out_of_budget or answer_part is None:
                    # Hard cap, context limit or thinking that never closes: force a submit_answer call; the
                    # session answer and salvage follow below.
                    stop = out_of_budget or "unclosed_think"
                    if program_mode:  # no forced typed answer: the program fallbacks below decide
                        prompt += text
                        break
                    base = text + FORCE_THINK if answer_part is None else text[:len(text) - len(answer_part)] + "\n\n"
                    forced = base + FORCE_CALL
                    fits = (estimate_tokens(known_tokens, known_chars, prompt + forced) + cfg.force_submit_tokens
                            + 64 <= cfg.context)
                    if cfg.force_submit_tokens and fits and cfg.force_submit_last:
                        deferred_force, deferred_entry = prompt + forced, entry  # run after the fallbacks, if needed
                    elif cfg.force_submit_tokens and fits:
                        c = await complete(prompt + forced, cfg.force_submit_tokens, ["</parameter>"])
                        expr = c.text.strip()
                        entry["forced_expr"] = expr[:200]
                        text = forced + c.text + "\n</parameter>\n</function>\n</tool_call>"
                        if expr:
                            out, grid = await submit(expr)
                            entry["forced_result"] = out[:200]
                            if grid is not None:
                                pred, source, answer_code = grid, "forced", expr
                    prompt += text
                    break
                calls, perrs = parse_tool_calls(answer_part)
                if not calls and not perrs and program_mode:  # text reply: typed answers are not accepted
                    content = answer_part
                    if n_nudges < cfg.max_nudges:
                        n_nudges += 1
                        entry["nudge"] = True
                        prompt += text + "<|im_end|>\n" + render_user(PROGRAM_NUDGE + budget_note(cfg, used)) \
                            + GEN_PROMPT
                        continue
                    prompt += text + "<|im_end|>\n"
                    stop = "final"
                    break
                if not calls and not perrs:  # a final text reply
                    content = answer_part
                    candidate = parse(answer_part)
                    if valid_grid(candidate):
                        pred, source = candidate, "parse"
                    else:
                        candidate = lenient_parse(answer_part)
                        if candidate is not None:
                            pred, source = candidate, "lenient"
                    prompt += text + "<|im_end|>\n"
                    stop = "final"
                    break
                outputs, accepted, cap_hit = [], None, False
                n_parse_err += len(perrs)
                for name, params in calls:
                    n_calls += 1
                    code = params.get("code")
                    if accepted is not None:
                        outputs.append("[skipped: solution already accepted]" if e025 else
                                       "[skipped: answer already accepted]")
                        continue
                    if e025 and (not isinstance(code, str) or name not in ("run_python", answer_tool)):
                        n_parse_err += 1
                        n_tool_err += 1
                        outputs.append("[tool error: Expected an object with a string code field]"
                                       if not isinstance(code, str) else f"[tool error: Unknown tool: {name}]")
                        continue
                    if not isinstance(code, str) or name not in ("run_python", answer_tool):
                        n_parse_err += 1
                        n_tool_err += 1
                        outputs.append(f"[tool error: unknown tool {name!r} or missing `code` parameter; tools are "
                                       f"run_python(code) and {answer_tool}(code)]"
                                       + (" Typed grids and submit_answer are not accepted: the answer must be a "
                                          "program." if program_mode and name == "submit_answer" else ""))
                        continue
                    if cap_hit:
                        outputs.append("[skipped: submission limit reached]")
                        continue
                    if name == "run_python":
                        if "def solve" in code:
                            last_solve_cell = code
                        try:
                            result = await ex.run(_execute, code)
                        except ExecutorTimeout as exc:
                            result = dict(killed) if e025 else {
                                "process_status": "timeout", "stdout": "", "state_reset": True,
                                "stderr": f"[{exc}; the session was restarted]"}
                        n_tool_err += result["process_status"] != "completed"
                        outputs.append(tool_output(result, cfg.max_tool_output))
                        continue
                    if program_mode:
                        rec, rows = await check_program(code)
                        rec["turn"], rec["tokens_used"] = turns, used
                        submissions.append(rec)
                        if rec["test"] is not None and confirm["shown"] is None and (
                                cfg.confirm_test or cfg.test_audit or cfg.lint):
                            report = _review(cfg, train, test_input, rec, code, confirm)
                            if report:  # E020: show the test output (+ audit / lint) once before accepting
                                outputs.append(report)
                                continue
                        if rec["test"] is not None:
                            outputs.append(f"[program accepted: passes all {len(train)} training pairs; test output "
                                           f"{len(rec['test'])}x{len(rec['test'][0])}]")
                            accepted = rec["test"]
                            continue
                        n_tool_err += 1
                        outputs.append(program_feedback(rows, len(submissions), cfg.max_submissions,
                                                        test_error=rec["test_error"],
                                                        rich=cfg.feedback == "rich", train=train))
                        if cfg.lint:
                            from .audit import lint
                            flags = lint(code, train)
                            if flags:
                                rec["lint"] = flags
                                outputs[-1] += ("\nHardcoded constants in your program (often fit the training pairs "
                                                "only):\n" + "\n".join(f"- {f}" for f in flags))
                        if len(submissions) >= cfg.max_submissions:
                            cap_hit = True
                            outputs[-1] += "\n[submission limit reached: the attempt ends]"
                        continue
                    out, grid = await submit(code)
                    if grid is not None and cfg.verify_submit:
                        try:
                            check = await ex.run(check_answer_on_train, code)
                        except ExecutorTimeout:
                            check = None
                        if check and check[0] and code not in forced_codes:
                            forced_codes.add(code)
                            out, grid = (f"[answer not accepted yet: the same expression with test_input = each "
                                         f"training input fails training pairs {check[0]} ({check[1]}). Fix the "
                                         "rule, or call submit_answer again with exactly the same code to submit "
                                         "anyway.]"), None
                        else:
                            submit_verified = None if check is None else not check[0]
                    n_tool_err += grid is None
                    outputs.append(out)
                    if grid is not None:
                        accepted, answer_code = grid, code
                for e in perrs:
                    n_tool_err += 1
                    outputs.append(f"[tool error: {e}]")
                entry["calls"] = [name for name, _ in calls] + ["<malformed>"] * len(perrs)
                if accepted is not None:
                    pred, source, stop = accepted, "program_pass" if program_mode else "submit", "submitted"
                    prompt += text + "<|im_end|>\n"
                    break
                if cap_hit:
                    stop = "submission_cap"
                    prompt += text + "<|im_end|>\n" + render_tool_responses(outputs)
                    break
                outputs[-1] += budget_note(cfg, used) + turns_note(cfg, turns)
                prompt += text + "<|im_end|>\n" + render_tool_responses(outputs) + GEN_PROMPT
        except Exception as exc:
            error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
        # Program mode fallbacks (also after an error): the best submitted program or the session's `solve`,
        # whichever reproduces more training pairs (ties: the session's solve, the latest).
        if program_mode and pred is None and ex.session is not None:
            try:
                pred, source, program_info = await _program_fallback(ex, train, test_input, cfg, submissions,
                                                                      run_on)
            except Exception as exc:
                program_info = {"fallback_error": f"{type(exc).__name__}: {exc}"[:200]}
        # Fallbacks, also after an error (R2): the session's `answer`, then the session's best function (R7).
        if not program_mode and pred is None and ex.session is not None:
            try:
                out, grid = await submit("answer")
                if grid is not None:
                    pred, source = grid, "session"
                if pred is None and cfg.solve_fallback:  # E028 opt-in: the session's solve on test_input
                    out, grid = await submit("solve(test_input)")
                    if grid is not None:
                        pred, source = grid, "session_solve"
                if pred is None and cfg.salvage:
                    salvage_info, grid = await ex.run(salvage_answer)
                    if grid is not None:
                        pred, source = grid, "salvage"
            except Exception as exc:
                salvage_info = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        # force_submit_last: the forced submit_answer call only when every fallback came up empty.
        if deferred_force is not None and pred is None and ex.session is not None:
            try:
                c = await complete(deferred_force, cfg.force_submit_tokens, ["</parameter>"])
                expr = c.text.strip()
                deferred_entry["forced_expr"] = expr[:200]
                deferred_entry["forced_last"] = True
                if expr:
                    out, grid = await submit(expr)
                    deferred_entry["forced_result"] = out[:200]
                    if grid is not None:
                        pred, source, answer_code = grid, "forced", expr
            except Exception as exc:
                deferred_entry["forced_error"] = f"{type(exc).__name__}: {exc}"[:200]
        train_check = None
        if cfg.record_train_check and not program_mode and pred is not None and ex.session is not None:
            train_check = await _submit_train_check(ex, answer_code if source in ("submit", "forced") else None, train)
        alternatives = None
        if cfg.record_alternatives and ex.session is not None:
            try:
                alternatives = await _alternatives(ex, train, test_input, cfg, pred, submissions, run_on, submit,
                                                   program_mode)
            except Exception as exc:
                alternatives = [{"error": f"{type(exc).__name__}: {exc}"[:200]}]
    finally:
        await ex.close()
    result = {"pred": pred, "answer_source": source if pred is not None else None, "content": content[-2000:],
              "stop_reason": stop, "turns": turns, "tool_calls": n_calls, "tool_errors": n_tool_err,
              "parse_failures": n_parse_err, "eos_continues": n_eos, "chunks": n_chunks,
              "usage": {"completion": used, "max_prompt": max_prompt, "prompt_total": cache["prompt"],
                        "ideal_cached": cache["ideal"],
                        **({"cached": cache["cached"]} if cache["reported"] else {})}, "turn_log": turn_log,
              "executor_resets": ex.resets, "harness_version": harness_version(cfg),
              **({"served_by": servers} if servers else {}),
              "seconds": round(time.monotonic() - t0, 1)}
    if cfg.turn_think_cap:
        result["think_nudges"], result["think_forced"] = think_stats["nudges"], think_stats["forced"]
        if cfg.think_call_rescue:
            result["think_call_rescued"] = think_stats["rescued"]
    if cfg.compact_at:
        result["compactions"], result["compact_removed_chars"] = compact_state["n"], compact_state["removed"]
    if cfg.confirm_test or cfg.test_audit or cfg.lint:  # the first train-passing test output that was shown
        result["first_tp_pred"] = confirm["shown"]
        if confirm.get("audit") is not None:
            result["audit"] = confirm["audit"]
    if alternatives is not None:
        result["alternatives"] = alternatives
    if salvage_info is not None:
        result["salvage"] = salvage_info
    if cfg.verify_submit:
        result["submit_verified"] = submit_verified if source == "submit" else None
        result["submit_rejections"] = len(forced_codes)
    if program_mode:
        final = program_info.get("record") if source != "program_pass" else submissions[-1]
        if source == "program_pass":
            program_info = {"train_pass": True, "train_frac": 1.0}
        result.update({
            "train_pass": program_info.get("train_pass") if pred is not None else None,
            "train_frac": program_info.get("train_frac") if pred is not None else None,
            "n_submissions": len(submissions), "nudges": n_nudges,
            "program": ((final or {}).get("code") or (last_solve_cell if source == "session_solve" else None))
            if pred is not None else None,
            "submissions": [{k: v for k, v in r.items() if k not in ("code", "test")} | {"code_chars": len(r["code"])}
                            for r in submissions]})
        if "fallback_error" in program_info:
            result["fallback_error"] = program_info["fallback_error"]
    elif cfg.record_train_check:
        result.update(train_check or {"train_pass": None, "train_frac": None})
    if error:
        result["error"] = error
    return prompt, result


async def _program_fallback(ex, train, test_input, cfg, submissions, run_on):
    """(pred, source, info) at the end of a program-mode attempt without an accepted program."""
    cands = []  # (train_frac, order, kind, record)
    for i, rec in enumerate(submissions):
        cands.append((rec["train_frac"], i, "program_best_fail", rec))
    # the session's own solve (may use session helpers): train-check it the same way (host-side comparison)
    outs = []
    for pair in train:
        try:
            o = await ex.run(run_session_solve, pair["input"], cfg.program_timeout)
        except ExecutorTimeout:
            o = {"error": "timeout"}
        if o is None:
            break
        outs.append(o)
    if len(outs) == len(train):
        rows = score_outputs(outs, train)
        frac = sum(r["ok"] for r in rows) / len(train)
        cands.append((frac, len(submissions), "session_solve", {"train_frac": frac, "code": None}))
    for frac, _, kind, rec in sorted(cands, key=lambda c: (c[0], c[1]), reverse=True):
        if kind == "session_solve":
            try:
                out = await ex.run(run_session_solve, test_input, cfg.program_timeout)
            except ExecutorTimeout:
                out = None
        else:
            out = (await run_on(rec["code"], [test_input]))[0]
        if out and "out" in out and valid_grid(out["out"]):
            return out["out"], kind, {"train_pass": frac == 1.0, "train_frac": frac, "record": rec}
    return None, None, {"candidates": len(cands)}


async def _alternatives(ex, train, test_input, cfg, pred, submissions, run_on, submit, program_mode):
    """E020 free attempt_2 (bookkeeping only, no model tokens; the attempt's answer is unchanged): other test grids
    the attempt produced, distinct from `pred` and from each other, with their train fraction where known.
    program mode: earlier failing submissions run on test_input, and the session's solve;
    submit mode: the session's `answer`, `solve(test_input)` and the salvage function."""
    seen = [pred] if pred is not None else []
    out = []

    def add(src, grid, frac):
        if grid is not None and valid_grid(grid) and grid not in seen:
            seen.append(grid)
            out.append({"src": src, "pred": grid, "train_frac": frac})

    if program_mode:
        for i, rec in sorted(enumerate(submissions), key=lambda x: -x[1]["train_frac"]):
            if rec.get("test") is not None:
                add(f"submission_{i}", rec["test"], rec["train_frac"])
            else:
                o = (await run_on(rec["code"], [test_input]))[0]
                add(f"submission_{i}", o.get("out"), rec["train_frac"])
        outs = []
        for pair in train:
            try:
                o = await ex.run(run_session_solve, pair["input"], cfg.program_timeout)
            except ExecutorTimeout:
                o = None
            if o is None:
                break
            outs.append(o)
        if len(outs) == len(train):
            frac = sum(r["ok"] for r in score_outputs(outs, train)) / len(train)
            try:
                o = await ex.run(run_session_solve, test_input, cfg.program_timeout)
            except ExecutorTimeout:
                o = None
            add("session_solve", (o or {}).get("out"), frac)
    else:
        for expr in ("answer", "solve(test_input)"):
            _, grid = await submit(expr)
            frac = None
            if grid is not None and expr != "answer":
                chk = await _submit_train_check(ex, expr, train)
                frac = chk["train_frac"]
            add(expr, grid, frac)
        try:
            info, grid = await ex.run(salvage_answer)
            frac = info.get("train_matched") / info["train_pairs"] if info and info.get("train_pairs") else None
            add("salvage:" + str((info or {}).get("function")), grid, frac)
        except Exception:
            pass
    return out


async def _submit_train_check(ex, code, train):
    """Submit mode, bookkeeping only: re-evaluate the answer expression with test_input bound to each training
    input (E011 check). train_pass None when the expression does not read test_input (e.g. `answer`)."""
    if not code:
        return {"train_pass": None, "train_frac": None}
    try:
        check = await ex.run(check_answer_on_train, code)
    except Exception:
        check = None
    if check is None:
        return {"train_pass": None, "train_frac": None}
    return {"train_pass": not check[0], "train_frac": 1 - len(check[0]) / len(train)}


def harness_version(cfg):
    v5 = any(getattr(cfg, k) != v for k, v in V5_DEFAULTS.items()) or cfg.prompt in V5_PROMPTS
    return HARNESS_VERSION_V5 if v5 else HARNESS_VERSION


def config_dict(cfg, model):
    d = {**asdict(cfg), "model": model, "harness_version": harness_version(cfg)}
    for key, default in V5_DEFAULTS.items():  # E020 options: recorded only when set
        if d[key] == default:
            del d[key]
    if "turn_think_grace" in d and not cfg.turn_think_cap:
        del d["turn_think_grace"]
    if not cfg.compact_at:  # E027 options: recorded only when compaction is on
        for key in ("compact_at", "compact_keep", "compact_tool_chars"):
            d.pop(key, None)
    for key, default in (("soft_wording", "gentle"), ("context", CONTEXT), ("salvage", True), ("exec_margin", 35)):
        if d[key] == default:  # keep configs of runs made before the option existed unchanged
            del d[key]
    if cfg.answer_mode != "program":  # program-only options
        for key in ("max_submissions", "program_timeout", "max_nudges"):
            del d[key]
    if not cfg.record_train_check:
        del d["record_train_check"]
    return d
