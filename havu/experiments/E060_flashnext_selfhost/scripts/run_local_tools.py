#!/usr/bin/env python3
"""E060: evals/run_deepseek_tools.py against a self-hosted server (--base-url), with tool code run in
arc_harness.executor.LocalExecutor (one child Python per attempt, IPython-style output, as the Kaggle solver
kaggle/dsv41_tools/pinned_candb1/solver_dsv41.py does) instead of the Docker sandbox. Pass `--sandbox-url local`.

Env: E060_EVALS (the evals/ copy, default ../evals next to this script's dir), E060_TOOL_PYTHON (Python for tool
code, default this interpreter), E060_EXEC_MEM_MB (default 4096, as on Kaggle).
All other arguments are the harness's own (see evals/run_deepseek_tools.py --help)."""
import os
import sys
from pathlib import Path

EVALS = Path(os.environ.get("E060_EVALS", Path(__file__).resolve().parents[1] / "evals"))
sys.path.insert(0, str(EVALS))

import run_deepseek_tools as h  # noqa: E402
from arc_harness.executor import LocalExecutor  # noqa: E402

TOOL_PYTHON = os.environ.get("E060_TOOL_PYTHON", sys.executable)
MEM_MB = int(os.environ.get("E060_EXEC_MEM_MB", "4096"))


def local_session(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    return LocalExecutor(train, test_input, timeout, max_output, memory_mb=MEM_MB, python=TOOL_PYTHON,
                         prelude_extra=prelude_extra, style="ipython", cpu_seconds=7200)


_solve = h.solve


def solve(*a, **kw):
    kw.setdefault("session_factory", local_session)
    return _solve(*a, **kw)


h.solve = solve
h.check_health = lambda url: None

# vLLM 0.32 reads an input assistant message's thinking from `reasoning` and silently ignores `reasoning_content`
# (checked on the rendered prompt with check_render.py: every earlier turn became an empty <think></think>). The
# harness stores `reasoning_content`, so convert it per request (as E048's OpenRouter path does). E060_REASONING_FIELD:
# "reasoning" (default: earlier thinking reaches the template; Flash-Next's template keeps it by default) or
# "reasoning_content" (the unpatched request: earlier thinking dropped).
REASONING_FIELD = os.environ.get("E060_REASONING_FIELD", "reasoning")
_turn = h.turn


def turn(client, args, messages, *a, **kw):
    if REASONING_FIELD == "reasoning":
        messages = h.openrouter_messages(messages)  # reasoning_content -> reasoning on assistant messages (a copy)
    return _turn(client, args, messages, *a, **kw)


h.turn = turn

if __name__ == "__main__":
    h.main()
