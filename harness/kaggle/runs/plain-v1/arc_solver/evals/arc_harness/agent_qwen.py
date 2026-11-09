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
"""
import asyncio
import json
import re
import threading
import time
from dataclasses import asdict, dataclass

from .arcprize import build_prompt, parse
from .executor import check_answer_on_train, salvage_answer, submit_answer, tool_output
from .grids import valid_grid
from .prompts import PROMPT_SUFFIXES

HARNESS_VERSION = "qwen-tools-3"  # 2: forced-submit text names the expression; 3: robustness (R2/R3/R7/R8, S4/S7/S8)
CONTEXT = 262144
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
PROMPTS = {"lean2sub": PROMPT_SUFFIXES["lean2"] + _SUBMIT}
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
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_eos_continues: int = 4      # EOS inside thinking: continue the turn at most this many times
    context: int = CONTEXT          # the server's context length (max_model_len); Kaggle runs 131072
    salvage: bool = True            # no answer at the end: apply the session's best-scoring function (qwen-tools-3)
    exec_margin: float = 35         # hard wall-clock cap per executor call = tool_timeout + exec_margin


# ---------------------------------------------------------------- rendering (Qwen3.8 chat template, tools on)

def tool_specs(cfg):
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
    return [{"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"], "additionalProperties": False},
    }} for name, description in descriptions.items()]


def render_system(tier, tools):
    s = "<|im_start|>system\n"
    if TIER_SYSTEM[tier]:
        s += TIER_SYSTEM[tier] + "\n\n"
    s += TOOLS_HEADER + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + TOOLS_FOOTER
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


def user_content(train, test_input, prompt):
    return build_prompt(train, test_input) + PROMPTS[prompt]


def initial_prompt(train, test_input, cfg):
    return render_system(cfg.tier, tool_specs(cfg)) + render_user(user_content(train, test_input, cfg.prompt)) \
        + GEN_PROMPT


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
    if not cfg.budget_notes or used < cfg.budget_notes * cfg.max_total_output:
        return ""
    hard = cfg.max_total_output
    if cfg.budget_soft:
        note = (f"\n[budget: {used / 1000:.0f}k output tokens used; aim to finish within {cfg.budget_soft / 1000:.0f}k "
                f"(hard limit {hard / 1000:.0f}k)]")
        if cfg.budget_soft <= used < cfg.budget_warn * hard:
            note += (" Past the target: submit your best answer now with submit_answer, unless one more experiment "
                     "will very likely make every training pair pass." if cfg.soft_wording == "strong"
                     else " Past the target: submit as soon as a rule fits every training pair.")
    else:
        note = f"\n[budget: {used / 1000:.0f}k of {hard / 1000:.0f}k output tokens used]"
    if used >= cfg.budget_warn * hard:
        note += " Budget nearly spent: submit your best answer now with submit_answer (or at least set `answer`)."
    return note


# ---------------------------------------------------------------- model backend

@dataclass
class Completion:
    text: str
    completion_tokens: int
    prompt_tokens: int
    finish: str  # "stop" (EOS or stop string) or "length"


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

    async def complete(self, prompt, max_tokens, stop=None):
        import openai
        errs = 0
        while True:
            try:
                r = await self.client.completions.create(
                    model=self.model, prompt=prompt, max_tokens=max_tokens, temperature=self.temperature,
                    top_p=self.top_p, stop=stop, extra_body={"top_k": self.top_k, **self.extra_body})
                c = r.choices[0]
                return Completion(c.text, r.usage.completion_tokens, r.usage.prompt_tokens, c.finish_reason)
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


async def solve(llm, make_executor, train, test_input, cfg):
    """One attempt. Returns (transcript, result). transcript is the full raw prompt+generation text.

    make_executor(train, test_input) -> executor (context manager with .execute(code)); train holds the
    training pairs with outputs (used host-side for --verify-submit), test_input the test grid only.
    Never raises (except on cancellation): errors are recorded, and any answer already set is still taken."""
    prompt = initial_prompt(train, test_input, cfg)
    used, turns, n_calls, n_tool_err, n_parse_err, n_eos, n_chunks = 0, 0, 0, 0, 0, 0, 0
    pred = source = stop = error = salvage_info = None
    max_prompt, content, submit_verified, forced_codes = 0, "", None, set()
    known_tokens, known_chars = 0, 0  # last measured prompt (tokens, chars) for the context estimate
    turn_log, t0 = [], time.monotonic()
    ex = _Session(make_executor, train, test_input, cfg.tool_timeout + cfg.exec_margin)

    async def complete(text, max_tokens, stop_strings=None):
        nonlocal used, n_chunks, max_prompt, known_tokens, known_chars
        c = await llm.complete(text, max_tokens, stop_strings) if stop_strings else await llm.complete(text, max_tokens)
        n_chunks += 1
        used += c.completion_tokens
        max_prompt = max(max_prompt, c.prompt_tokens)
        known_tokens, known_chars = c.prompt_tokens + c.completion_tokens, len(text) + len(c.text)
        return c

    async def submit(code):
        try:
            return await ex.run(submit_answer, code)
        except ExecutorTimeout:
            return "[answer not accepted] evaluation timed out; Python state lost (train/test_input/np restored)", None

    try:
        try:
            await ex.start()
            while True:
                if turns >= cfg.max_turns:
                    stop = "max_turns"
                    break
                turns += 1
                text, low_streak, after_eos, out_of_budget = "", 0, False, None
                turn_tokens = 0
                while True:  # one assistant turn, possibly over several requests
                    left = cfg.max_total_output - used
                    room = cfg.context - estimate_tokens(known_tokens, known_chars, prompt + text) - 64
                    if left <= 0 or room < 256:
                        out_of_budget = "output_budget" if left <= 0 else "context"
                        break
                    c = await complete(prompt + text, min(cfg.chunk_tokens, left, room))
                    turn_tokens += c.completion_tokens
                    text += c.text
                    if after_eos:
                        low_streak = low_streak + 1 if c.completion_tokens < LOW_YIELD else 0
                    after_eos = False
                    if c.finish == "length":
                        continue
                    if THINK_CLOSE not in text and n_eos < cfg.max_eos_continues and low_streak < 2:
                        n_eos += 1  # EOS inside thinking: let it carry on
                        after_eos = True
                        continue
                    break
                reasoning, answer_part = split_turn(text)
                entry = {"turn": turns, "tokens": turn_tokens, "closed_think": answer_part is not None}
                turn_log.append(entry)
                if out_of_budget or answer_part is None:
                    # Hard cap, context limit or thinking that never closes: force a submit_answer call; the
                    # session answer and salvage follow below.
                    stop = out_of_budget or "unclosed_think"
                    base = text + FORCE_THINK if answer_part is None else text[:len(text) - len(answer_part)] + "\n\n"
                    forced = base + FORCE_CALL
                    fits = (estimate_tokens(known_tokens, known_chars, prompt + forced) + cfg.force_submit_tokens
                            + 64 <= cfg.context)
                    if cfg.force_submit_tokens and fits:
                        c = await complete(prompt + forced, cfg.force_submit_tokens, ["</parameter>"])
                        expr = c.text.strip()
                        entry["forced_expr"] = expr[:200]
                        text = forced + c.text + "\n</parameter>\n</function>\n</tool_call>"
                        if expr:
                            out, grid = await submit(expr)
                            entry["forced_result"] = out[:200]
                            if grid is not None:
                                pred, source = grid, "forced"
                    prompt += text
                    break
                calls, perrs = parse_tool_calls(answer_part)
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
                outputs, accepted = [], None
                n_parse_err += len(perrs)
                for name, params in calls:
                    n_calls += 1
                    code = params.get("code")
                    if accepted is not None:
                        outputs.append("[skipped: answer already accepted]")
                        continue
                    if not isinstance(code, str) or name not in ("run_python", "submit_answer"):
                        n_parse_err += 1
                        n_tool_err += 1
                        outputs.append(f"[tool error: unknown tool {name!r} or missing `code` parameter; tools are "
                                       "run_python(code) and submit_answer(code)]")
                        continue
                    if name == "run_python":
                        try:
                            result = await ex.run(_execute, code)
                        except ExecutorTimeout as exc:
                            result = {"process_status": "timeout", "stdout": "", "state_reset": True,
                                      "stderr": f"[{exc}; the session was restarted]"}
                        n_tool_err += result["process_status"] != "completed"
                        outputs.append(tool_output(result, cfg.max_tool_output))
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
                        accepted = grid
                for e in perrs:
                    n_tool_err += 1
                    outputs.append(f"[tool error: {e}]")
                entry["calls"] = [name for name, _ in calls] + ["<malformed>"] * len(perrs)
                if accepted is not None:
                    pred, source, stop = accepted, "submit", "submitted"
                    prompt += text + "<|im_end|>\n"
                    break
                outputs[-1] += budget_note(cfg, used)
                prompt += text + "<|im_end|>\n" + render_tool_responses(outputs) + GEN_PROMPT
        except Exception as exc:
            error, stop = f"{type(exc).__name__}: {exc}"[:1000], "error"
        # Fallbacks, also after an error (R2): the session's `answer`, then the session's best function (R7).
        if pred is None and ex.session is not None:
            try:
                out, grid = await submit("answer")
                if grid is not None:
                    pred, source = grid, "session"
                elif cfg.salvage:
                    salvage_info, grid = await ex.run(salvage_answer)
                    if grid is not None:
                        pred, source = grid, "salvage"
            except Exception as exc:
                salvage_info = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    finally:
        await ex.close()
    result = {"pred": pred, "answer_source": source if pred is not None else None, "content": content[-2000:],
              "stop_reason": stop, "turns": turns, "tool_calls": n_calls, "tool_errors": n_tool_err,
              "parse_failures": n_parse_err, "eos_continues": n_eos, "chunks": n_chunks,
              "usage": {"completion": used, "max_prompt": max_prompt}, "turn_log": turn_log,
              "executor_resets": ex.resets, "harness_version": HARNESS_VERSION,
              "seconds": round(time.monotonic() - t0, 1)}
    if salvage_info is not None:
        result["salvage"] = salvage_info
    if cfg.verify_submit:
        result["submit_verified"] = submit_verified if source == "submit" else None
        result["submit_rejections"] = len(forced_codes)
    if error:
        result["error"] = error
    return prompt, result


def config_dict(cfg, model):
    d = {**asdict(cfg), "model": model, "harness_version": HARNESS_VERSION}
    for key, default in (("soft_wording", "gentle"), ("context", CONTEXT), ("salvage", True), ("exec_margin", 35)):
        if d[key] == default:  # keep configs of runs made before the option existed unchanged
            del d[key]
    return d
