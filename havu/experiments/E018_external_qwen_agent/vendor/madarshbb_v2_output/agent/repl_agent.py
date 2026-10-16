"""REPL tool agent for ARC-AGI-2 tasks, modelled on the Tufa Labs duck harness.

The model gets two tools:
  python(code)  explore the task: train/test grids are preloaded as numpy
                arrays with ARC helpers; stdout comes back, capped.
  submit(code)  a program defining transform(grid). It is run on every train
                input; the reply says which examples it reproduced and shows a
                cell diff for the rest. Reproducing ALL of them verifies the
                program, records its test outputs and ends the attempt.

Many attempts run concurrently against one vLLM OpenAI server, which batches
them. Old turns are evicted when the context budget is exceeded; the system
prompt and the task stay. Answers are aggregated exactly as in agent.py.

Optional, off by default, from dfranzen's ARC-AGI-3 fork of the duck harness (see
run_attempt): reasoning-effort ladder on cut turns, function retention, kept
reasoning, hysteresis trimming, repetition-loop guard, timeout-as-cut.
"""
import json
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

from agent import aggregate
from backend import extract_code
from grids import SYSTEM, show, diff_text
from repl import run_snippet, PREAMBLE_DOC, RETAIN_DOC, new_store, with_kept
from sandbox import run_one

TOOLS = [
    {"type": "function", "function": {
        "name": "python",
        "description": "Run Python to inspect the task. " + PREAMBLE_DOC,
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code; print() what you need."}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": ("Submit a complete program defining transform(grid) -> grid. It is run on "
                        "every example input; you are told which examples it reproduces. "
                        "When it reproduces all of them it is accepted and you are done."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Full Python source defining transform(grid)."}},
            "required": ["code"]}}},
]

REPL_INSTRUCTIONS = """Solve this ARC puzzle. The examples below share one hidden transformation rule.

Work like a scientist: use the `python` tool to measure the examples (objects, colours, sizes, positions, differences between input and output), form a hypothesis, and test it in code. Then call `submit` with the full source defining `transform(grid)`. Use only the standard library and numpy.

Submit EARLY and often: submitting is free, and a failing submission tells you exactly which examples and which cells are wrong -- that is the fastest way to find the rule. Do not wait for certainty. If a submission fails, study the diff and fix the rule rather than special-casing examples."""

# probe24: every failed attempt spent 30-73k tokens exploring and never submitted
# once, so it never saw a diff. After this many python calls without a submit,
# the tool output carries a reminder.
NUDGE_EVERY = 6
NUDGE = ("\n\n[harness] You have run {n} python calls without submitting. Submitting is free "
         "and returns a cell-level diff against every example. Submit your current best "
         "transform now, then refine it from the diff.")

# probe9q38: every failed attempt had 1-3 turns cut at 16k tokens mid-thought, while
# solves used 2-3k tokens a turn. The thinking-cap treatment lowers the per-turn cap
# and, when a turn is cut, orders the model to act instead of resuming the analysis.
THINKCAP_HINT = ("[harness] You hit the per-turn thinking limit. Stop analysing. In your next "
                 "reply, call `python` to test your current best hypothesis in code, or "
                 "`submit` a transform implementing it -- a failing submit shows exactly "
                 "which cells are wrong.")


# The probe24 wording, kept verbatim so a run with the nudge off matches it exactly.
REPL_INSTRUCTIONS_V1 = """Solve this ARC puzzle. The examples below share one hidden transformation rule.

Work like a scientist: use the `python` tool to measure the examples (objects, colours, sizes, positions, differences between input and output), form a hypothesis, and test it in code. When you have a program that you believe reproduces EVERY example, call `submit` with the full source defining `transform(grid)`. Use only the standard library and numpy. If a submission fails, study the diff and fix the rule rather than special-casing examples."""


# probe9q38 transcripts: failures parse and count grids IN THE REASONING, cell by cell,
# often filling a whole 16k-token turn before any tool call ("col 7=8, 8=0, 9=0 ...").
# Tufa's harness says the same thing to its model. Optional, its own arm.
IN_PYTHON_RULE = """

Never read, count, index or transcribe grid cells in your head. Do ALL of that with the `python` tool -- the grids are already loaded as numpy arrays. Keep your own reasoning to hypotheses and plans; let code do the looking."""


def repl_prompt(task, early_submit=True, in_python=False):
    head = REPL_INSTRUCTIONS if early_submit else REPL_INSTRUCTIONS_V1
    parts = [head + (IN_PYTHON_RULE if in_python else ""), ""]
    for i, p in enumerate(task["train"], 1):
        parts += [f"## Example {i}", show(p["input"], "Input"), show(p["output"], "Output"), ""]
    for j, t in enumerate(task["test"], 1):
        parts += [f"## Test input {j}", show(t["input"], "Input"), ""]
    return "\n".join(parts).rstrip()


def verify(code, task, timeout=10.0):
    """Run a submitted program; return (n_ok, test_outputs, feedback_text)."""
    tr = task["train"]
    res = run_one(code, [p["input"] for p in tr] + [t["input"] for t in task["test"]],
                  timeout=timeout)
    n_ok, lines, shown = 0, [], 0
    for i, (r, p) in enumerate(zip(res[:len(tr)], tr), 1):
        if "output" in r and r["output"] == p["output"]:
            n_ok += 1
            continue
        if shown < 2:
            shown += 1
            lines.append(f"## Example {i} failed")
            lines.append(f"Error: {r['error']}" if "error" in r else diff_text(p["output"], r["output"]))
    test_out = [r.get("output") for r in res[len(tr):]]
    test_err = [r.get("error") for r in res[len(tr):] if "error" in r]
    if n_ok == len(tr):
        if test_err:
            return n_ok, test_out, ("Reproduces all examples but FAILS on a test input: "
                                    + test_err[0] + " -- fix it and submit again.")
        return n_ok, test_out, f"Accepted: reproduces all {len(tr)} examples."
    return n_ok, test_out, (f"Reproduced {n_ok} of {len(tr)} examples.\n" + "\n".join(lines)
                            + "\nFix the rule and submit again.")


def _est_tokens(messages):
    return len(json.dumps(messages)) // 3


def trim(messages, budget, keep_recent=6, drain=0, state=None):
    """Drop the oldest assistant/tool/user groups after the task prompt until the
    estimate fits. A group starts at an assistant message, so tool results are
    never separated from the call that produced them.

    With `drain` and a per-attempt `state` dict the cut has hysteresis (dfranzen's
    ARC-AGI-3 harness): it only moves when the estimate exceeds `budget`, and then
    drops down to `budget - drain`. Between those moments the retained history is
    byte-identical from turn to turn, so the server's prefix cache keeps serving it;
    without it, every turn past the budget shifts the cut by one group and
    re-prefills everything after the task prompt."""
    if drain and state is not None:
        return _trim_hysteresis(messages, budget, keep_recent, drain, state)
    head, rest = messages[:2], messages[2:]
    starts = [i for i, m in enumerate(rest) if m["role"] == "assistant"]
    dropped = 0
    while _est_tokens(head + rest) > budget and len(starts) > keep_recent:
        cut = starts[1]
        rest = rest[cut:]
        starts = [i - cut for i in starts[1:]]
        dropped += 1
    if dropped:
        note = {"role": "user", "content": f"[{dropped} older turns were removed to save context.]"}
        rest = [note] + rest
    return head + rest


def _trim_hysteresis(messages, budget, keep_recent, drain, state):
    head, rest = messages[:2], messages[2:]
    starts = [i for i, m in enumerate(rest) if m["role"] == "assistant"]
    max_cut = max(0, len(starts) - keep_recent)

    def view(cut):
        if cut <= 0:
            return head + rest
        note = {"role": "user", "content": f"[{cut} older turns were removed to save context.]"}
        return head + [note] + rest[starts[cut]:]

    cut = min(state.get("cut", 0), max_cut)
    if _est_tokens(view(cut)) > budget:
        while cut < max_cut and _est_tokens(view(cut)) > budget - drain:
            cut += 1
        state["evictions"] = state.get("evictions", 0) + 1
    state["cut"] = cut
    return view(cut)


def degenerate(text, min_chars=500, ratio=0.01):
    """A repetition loop rather than text: zlib compresses it below `ratio`. dfranzen saw a
    reasoning-only reply of 6,144 '!' under speculative decoding; prose compresses to
    0.2-0.3, an ASCII grid to ~0.02, so 0.01 only catches genuine loops."""
    if not text or len(text) < min_chars:
        return False
    raw = text.encode("utf-8", "replace")
    return len(zlib.compress(raw, 6)) / len(raw) < ratio


DEGENERATE_STUB = ("(previous reply was a repetition loop and has been omitted; do not continue "
                   "that pattern)")


def make_tools(retain=False):
    """The two tools; with function retention their descriptions say what persists."""
    if not retain:
        return TOOLS
    py = dict(TOOLS[0]["function"], description=(
        "Run Python to inspect the task. "
        + PREAMBLE_DOC.replace(" (each call starts fresh; nothing persists)", "") + " " + RETAIN_DOC))
    sub = dict(TOOLS[1]["function"], description=(
        TOOLS[1]["function"]["description"] + " The program may call functions kept from your "
        "python calls (they are added automatically); if it defines no transform(), the kept "
        "transform() is submitted."))
    return [{"type": "function", "function": py}, {"type": "function", "function": sub}]


class _Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.verified = {}


class Attempt:
    def __init__(self, task_id, idx):
        self.task_id, self.idx = task_id, idx
        self.programs, self.status, self.note = [], "active", ""
        self.turns = self.python_calls = self.submits = 0
        self.python_at_submit = self.nudges = self.cuts = self.forced = 0
        self.gen_tokens = self.prompt_tokens = 0
        # ARC-AGI-3 takeaways (all zero unless their option is on)
        self.effort_steps = self.timeouts = self.degenerate = self.kept_used = 0
        self.evictions = 0
        self.wall = 0.0
        # Everything the model thought, ran and saw -- so a failure can be read as
        # "right idea, out of budget" vs "never found the rule".
        self.transcript = []


TIMEOUT_HINT = ("[harness] Your previous reply took too long and was abandoned. Keep your "
                "reasoning short: call `python` to test your current best hypothesis, or `submit`.")


def run_attempt(att, task, client, shared, deadline, max_turns=40, max_gen_tokens=120000,
                context_budget=28000, stop_after=1, seed=0, snippet_timeout=30.0,
                nudge_every=NUDGE_EVERY, turn_max_tokens=None, in_python=False,
                force_submit_after=0, effort_ladder=None, retain_functions=False,
                keep_reasoning=False, reasoning_key="reasoning_content", keep_recent=6,
                trim_drain=0, degenerate_guard=False, timeout_as_cut=False):
    """One attempt at one task. The options after force_submit_after are the ARC-AGI-3
    (dfranzen / duck harness) takeaways, each off by default so the baseline is unchanged:

    effort_ladder     e.g. ["medium", "low"]: after a turn is cut at max_tokens (or times
                      out, with timeout_as_cut) the next turns run one reasoning_effort rung
                      lower; a submit that reproduces more examples than before restores
                      the client's own effort.
    retain_functions  python-defined functions persist (as source) into later python calls
                      and into submits that use them.
    keep_reasoning    send each completed turn's reasoning back under `reasoning_key`
                      (pair with chat-template kwarg preserve_thinking=True). Cut turns and
                      repetition loops are never sent back.
    trim_drain        hysteresis for the context trim (see trim()); keep_recent groups are
                      never evicted.
    degenerate_guard  a repetition-loop reply is replaced by a stub in the history.
                      Loops are COUNTED in every arm (att.degenerate).
    timeout_as_cut    a request timeout before the deadline is treated like a cut turn
                      (retry, one rung lower) instead of failing the attempt."""
    t0 = time.time()
    # The early-submit wording and the nudge are ONE treatment: off means probe24's harness.
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": repl_prompt(task, early_submit=bool(nudge_every),
                                                        in_python=in_python)}]
    tools = make_tools(retain_functions)
    store = new_store() if retain_functions else None
    trim_state = {} if trim_drain else None
    rung, best_ok = -1, -1                   # rung -1 = the client's own reasoning effort

    def step_down():
        nonlocal rung
        if effort_ladder and rung < len(effort_ladder) - 1:
            rung += 1
            att.effort_steps += 1

    while att.turns < max_turns:
        with shared.lock:
            if shared.verified.get(att.task_id, 0) >= stop_after:
                att.status, att.note = "stopped", "task verified by a sibling"
                break
        if time.time() > deadline:
            att.status, att.note = "stopped", "deadline"
            break
        if att.gen_tokens >= max_gen_tokens:
            att.status, att.note = "failed", "token budget"
            break
        att.turns += 1
        # A request may not outlive the deadline by more than a minute.
        req_timeout = max(60.0, min(3600.0, deadline - time.time() + 60.0))
        # mini2arm: the text nudge was read and ignored. The forcesubmit treatment
        # makes the NEXT call a submit after this many python calls without one.
        force = (force_submit_after
                 and att.python_calls - att.python_at_submit >= force_submit_after)
        if force:
            att.forced += 1
        extra = {}
        if rung >= 0:            # only then, so clients without the kwarg keep working
            extra["template_kwargs"] = {"reasoning_effort": effort_ladder[rung]}
        r = client.chat(trim(messages, context_budget, keep_recent, trim_drain, trim_state),
                        tools=tools, seed=seed + att.turns,
                        timeout=req_timeout, max_tokens=turn_max_tokens,
                        tool_choice="submit" if force else None, **extra)
        if trim_state is not None:
            att.evictions = trim_state.get("evictions", 0)
        if "error" in r:
            if r.get("context_overflow") and keep_reasoning:
                # Kept reasoning can make even the protected recent turns too long for
                # the model; shed it (oldest first, newest last) before shrinking.
                held = [m for m in messages if reasoning_key in m]
                if held:
                    for m in held[:max(1, len(held) - 1)]:
                        m.pop(reasoning_key)
                    continue
            if r.get("context_overflow") and context_budget > 8000:
                context_budget = int(context_budget * 0.7)
                continue
            if time.time() >= deadline:
                # Requests are timed to end a minute after the deadline: this is the run
                # ending, not the attempt failing (ab3q36 logged 21 of these as failures).
                att.status, att.note = "stopped", "deadline (request cut)"
                break
            if timeout_as_cut and r.get("timeout"):
                att.timeouts += 1
                step_down()
                if messages[-1]["role"] != "user":
                    messages.append({"role": "user", "content": TIMEOUT_HINT})
                continue
            att.status, att.note = "failed", r["error"][:200]
            break
        att.gen_tokens += r["completion_tokens"]
        att.prompt_tokens += r["prompt_tokens"]
        cut = r["finish_reason"] == "length"
        loops = [k for k in ("reasoning", "content") if degenerate(r[k])]
        att.degenerate += bool(loops)
        att.transcript.append({"turn": att.turns, "t": round(time.time() - t0, 1),
                               "reasoning": r["reasoning"], "content": r["content"],
                               "calls": [c["name"] for c in r["tool_calls"]],
                               "finish": r["finish_reason"], "tokens": r["completion_tokens"],
                               **({"effort": effort_ladder[rung]} if rung >= 0 else {}),
                               **({"loop": loops} if loops else {})})
        if cut:
            step_down()
        content = r["content"] or ""
        if degenerate_guard and "content" in loops:
            content = DEGENERATE_STUB
        msg = {"role": "assistant", "content": content}
        if keep_reasoning and r["reasoning"] and not cut and "reasoning" not in loops:
            msg[reasoning_key] = r["reasoning"]
        calls = r["tool_calls"]
        if not calls:
            code = extract_code(r["content"])
            if code:                       # answered with a code block instead of submit
                calls = [{"id": f"implicit-{att.turns}", "name": "submit",
                          "arguments": {"code": code}, "implicit": True}]
            else:
                messages.append(msg)
                if r["finish_reason"] == "length":
                    # The thinking-cap treatment answers a cut with an order to act; the
                    # baseline keeps probe24's wording.
                    hint = THINKCAP_HINT if turn_max_tokens else "Your reply was cut off. Be more concise."
                    att.cuts += 1
                else:
                    hint = "Use the `python` tool to explore, or `submit` your transform program."
                messages.append({"role": "user", "content": hint})
                continue
        if not any(c.get("implicit") for c in calls):
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"],
                                               "arguments": json.dumps(c["arguments"])}}
                                 for c in calls]
        messages.append(msg)
        done = False
        for c in calls:
            code = (c["arguments"] or {}).get("code") or ""
            if c["name"] == "python":
                att.python_calls += 1
                out = run_snippet(code, task, timeout=snippet_timeout, store=store)
                since = att.python_calls - att.python_at_submit
                if nudge_every and since and since % nudge_every == 0:
                    out += NUDGE.format(n=since)
                    att.nudges += 1
            elif c["name"] == "submit":
                att.submits += 1
                att.python_at_submit = att.python_calls
                full, added = with_kept(code, store) if store else (code, [])
                n_ok, test_out, out = verify(full, task)
                if added:
                    att.kept_used += 1
                    out += f"\n(Ran with your kept functions: {', '.join(added)}.)"
                ntr = len(task["train"])
                att.programs.append({"turn": att.turns, "code": full, "n_ok": n_ok,
                                     "n_train": ntr, "test": test_out})
                if n_ok > best_ok:
                    if best_ok >= 0 or n_ok > 0:
                        rung = -1            # progress: back to the client's own effort
                    best_ok = n_ok
                if n_ok == ntr and all(o is not None for o in test_out):
                    done = True
            else:
                out = f"Unknown tool {c['name']!r}. Tools: python, submit."
            att.transcript.append({"turn": att.turns, "tool": c["name"], "code": code,
                                   "output": out})
            if c.get("implicit"):
                messages.append({"role": "user", "content": out})
            else:
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        if done:
            att.status = "verified"
            with shared.lock:
                shared.verified[att.task_id] = shared.verified.get(att.task_id, 0) + 1
            break
    else:
        att.status, att.note = "failed", "turn budget"
    att.wall = time.time() - t0
    return att


# run_attempt options a probe or submission config may set for the whole run
# (probe arms can set any run_attempt keyword).
AGENT_OPTS = ("turn_max_tokens", "in_python", "force_submit_after", "effort_ladder",
              "retain_functions", "keep_reasoning", "reasoning_key", "keep_recent", "trim_drain",
              "degenerate_guard", "timeout_as_cut", "snippet_timeout")


def attempt_record(a, task):
    return {"task_id": a.task_id, "idx": a.idx, "status": a.status,
            "note": a.note, "turns": a.turns, "python_calls": a.python_calls,
            "submits": a.submits, "nudges": a.nudges, "cuts": a.cuts, "forced": a.forced, "gen_tokens": a.gen_tokens,
            "prompt_tokens": a.prompt_tokens, "wall": a.wall,
            "effort_steps": a.effort_steps, "timeouts": a.timeouts, "degenerate": a.degenerate,
            "kept_used": a.kept_used, "evictions": a.evictions,
            "best_ok": max((p["n_ok"] for p in a.programs), default=-1),
            "n_train": len(task["train"]),
            "programs": a.programs, "transcript": a.transcript}


def solve_repl(tasks, client, attempts=2, concurrency=32, deadline=None, stop_after=1,
               log=print, on_done=None, task_kw=None, **kw):
    """Same return shape as agent.solve: (answers, report).

    `on_done(record)` is called as each attempt finishes, so a caller can persist
    results incrementally -- probe24 lost everything to one late exception."""
    deadline = deadline or float("inf")
    shared = _Shared()
    # attempt-major order: every task gets its first attempt before any gets a second
    atts = [Attempt(tid, k) for k in range(attempts) for tid in tasks]
    t0 = time.time()
    done_n = [0]

    def work(att):
        try:
            # task_kw lets arms of an A/B share one run: key -> per-arm overrides. An arm
            # may bring its own client (different sampling / chat-template kwargs).
            over = dict(kw, **(task_kw or {}).get(att.task_id, {}))
            arm_client = over.pop("client", client)
            run_attempt(att, tasks[att.task_id], arm_client, shared, deadline,
                        stop_after=stop_after, seed=1000 * att.idx, **over)
        except Exception as e:                  # one attempt must never sink the run
            import traceback
            att.status, att.note = "crashed", f"{type(e).__name__}: {e}"[:300]
            att.transcript.append({"crash": traceback.format_exc()[-2000:]})
        if on_done:
            try:
                on_done(attempt_record(att, tasks[att.task_id]))
            except Exception as e:
                log(f"on_done failed: {e}")
        with shared.lock:
            done_n[0] += 1
            n_ver = sum(1 for v in shared.verified.values() if v)
        log(f"[{time.time() - t0:6.0f}s] {att.task_id}#{att.idx} {att.status:8s} "
            f"turns {att.turns:2d} py {att.python_calls:2d} sub {att.submits} "
            f"tok {att.gen_tokens:6d} | done {done_n[0]}/{len(atts)} tasks verified {n_ver}")
        return att

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(work, atts))

    answers = {tid: aggregate(tid, tasks[tid], atts) for tid in tasks}
    report = {"wall_s": time.time() - t0,
              "attempts": [attempt_record(a, tasks[a.task_id]) for a in atts]}
    return answers, report
