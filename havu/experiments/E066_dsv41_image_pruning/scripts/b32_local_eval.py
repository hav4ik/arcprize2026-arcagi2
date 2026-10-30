#!/usr/bin/env python3
"""E066: B32's pinned harness CLI with tool code in the local executor, exactly as B32's Kaggle solver runs it.

pinned_candb/solver_dsv41.py does not use the CLI's HTTP sandbox: it calls run_deepseek_tools.solve() with its own
session_factory, arc_harness.executor.LocalExecutor(memory_mb=ARC_EXEC_MEM_MB (4096), python=ARC_TOOL_PYTHON (this
interpreter), style="ipython", cpu_seconds=7200). This wrapper puts that factory in solve()'s default argument and
skips the sandbox health check, then runs the pinned CLI unchanged (its scheduling, multi-test, resume and output
records). Deadline-only solver patches (wrap-up notes, stream cuts) do not apply to a deadline-free eval.
Run from the kit's evals dir: cd <kit>/repo/evals && python b32_local_eval.py <B32 flags> --sandbox-url local ...
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import run_deepseek_tools as h  # noqa: E402
from arc_harness.executor import LocalExecutor  # noqa: E402

EXEC_MEM_MB = int(os.environ.get("ARC_EXEC_MEM_MB", "4096"))
TOOL_PYTHON = os.environ.get("ARC_TOOL_PYTHON", sys.executable)


def session_factory(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    return LocalExecutor(train, test_input, timeout, max_output, memory_mb=EXEC_MEM_MB, python=TOOL_PYTHON,
                         prelude_extra=prelude_extra, style="ipython", cpu_seconds=7200)


defaults = list(h.solve.__defaults__)
assert defaults[0] is h.SandboxSession, "unexpected solve() signature"
defaults[0] = session_factory
h.solve.__defaults__ = tuple(defaults)
h.check_health = lambda url: None
sys.argv[0] = "run_deepseek_tools.py"
h.main()
