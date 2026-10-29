"""Persistent, CPU-only ARC scratchpad. JSON-lines are a trusted host protocol.

Notebook-style state without Jupyter's host filesystem/network access. Generated
code never runs in the controller. The seccomp boundary is the same as the
single-program verifier; AST restrictions are additional defense in depth.
"""

import ast
import builtins
import collections
import contextlib
import gc
import importlib.util
import json
from pathlib import Path
import resource
import sys
import types

# -P excludes the script directory from sys.path. Load this one trusted sibling
# explicitly, rather than enabling PYTHONPATH or importing from the working dir.
_spec = importlib.util.spec_from_file_location(
    "_arc_tir_sandbox", Path(__file__).with_name("qwen38_tir_worker.py"))
safe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(safe)

MAX_MESSAGE_BYTES = 2_000_000


class Runtime:
    def __init__(self, payload):
        if set(payload) != {"task", "memory_mib", "cpu_seconds"}:
            raise ValueError("Expected public task and resource limits only")
        memory, cpu = payload["memory_mib"], payload["cpu_seconds"]
        if type(memory) is not int or not 128 <= memory <= 4096:
            raise ValueError("Invalid memory limit")
        if type(cpu) is not int or not 1 <= cpu <= 100000:
            raise ValueError("Invalid lifetime CPU limit")
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (memory * 1024**2, memory * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        import numpy as np
        self.np = np
        probe = np.array([[0, 1]], dtype=np.int64)
        np.unique(probe)
        probe.max()
        repr(probe)
        collections.Counter([0, 1, 1]).most_common(1)
        task = payload["task"]
        if not isinstance(task, dict) or set(task) != {"train", "test"}:
            raise ValueError("Invalid public task")
        if not isinstance(task["train"], list) or not 1 <= len(task["train"]) <= 64:
            raise ValueError("Expected 1..64 training pairs")
        if not isinstance(task["test"], list) or not 1 <= len(task["test"]) <= 64:
            raise ValueError("Expected 1..64 test inputs")
        if any(set(p) != {"input", "output"} for p in task["train"]):
            raise ValueError("Training pairs must contain input/output only")
        if any(set(p) != {"input"} for p in task["test"]):
            raise ValueError("Test outputs/metadata must never enter the worker")
        self.task = {
            "train": [{k: safe.normalize_grid(p[k], np) for k in ("input", "output")}
                      for p in task["train"]],
            "test": [{"input": safe.normalize_grid(p["input"], np)} for p in task["test"]],
        }
        self.templates = safe.modules_for(np)
        self.numpy_methods = {name: sys.modules[name]
                              for name in ("numpy._core._methods", "numpy.core._methods")
                              if name in sys.modules}
        safe.restrict_system_calls()  # Fail closed before constructing user state.
        self.reset()

    def reset(self):
        modules = {name: types.SimpleNamespace(**vars(module))
                   for name, module in self.templates.items()}

        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if not level and name in self.numpy_methods:
                return self.numpy_methods[name]
            if level or name not in modules:
                raise ImportError(f"Unsupported import: {name}")
            return modules[name]

        allowed = {n: getattr(builtins, n) for n in safe.BUILTINS}
        allowed["__import__"] = restricted_import
        self.namespace = {
            "__builtins__": allowed, "np": modules["numpy"], "math": modules["math"],
            "Counter": collections.Counter, "defaultdict": collections.defaultdict,
            "deque": collections.deque,
            "train": [{k: self.np.array(p[k], dtype=self.np.int64) for k in ("input", "output")}
                      for p in self.task["train"]],
            "test_inputs": [self.np.array(p["input"], dtype=self.np.int64) for p in self.task["test"]],
        }
        gc.collect()

    def cell(self, source):
        result = dict(sandbox_enabled=True, error=None, stdout="", stdout_truncated=False)
        output = safe.BoundedText()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                # Same syntax/import policy, but no requirement to define a
                # complete solver. Assignments, loops, helpers and expressions
                # are all valid cells. Never rewrite a last-line assignment.
                safe.validate_code(source, require_entrypoint=False)
                tree = ast.parse(source, filename="<tir_program>")
                expression = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<tir_program>", "exec"), self.namespace, self.namespace)
                if expression is not None:
                    value = eval(compile(ast.Expression(expression.value), "<tir_program>", "eval"),
                                 self.namespace, self.namespace)
                    if value is not None:
                        # A notebook's implicit function display must not embed
                        # a process-specific address that breaks safe replay.
                        print(f"<function {value.__name__}>" if isinstance(value, types.FunctionType)
                              else repr(value))
            except Exception as exc:
                # Like a notebook, ordinary exceptions retain assignments made
                # before the exception. Timeout/death is handled by the host.
                result["error"] = safe.error_text(exc)
        result.update(stdout=output.getvalue(), stdout_truncated=output.truncated)
        return result

    def evaluate(self, entrypoint, grid):
        if not isinstance(entrypoint, str) or not entrypoint.isidentifier() or entrypoint.startswith("_"):
            raise ValueError("Entrypoint must be a public Python identifier")
        record = dict(prediction=None, error=None)
        output = safe.BoundedText()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                fn = self.namespace.get(entrypoint)
                if not callable(fn):
                    raise ValueError(f"Define callable {entrypoint}(input_grid) in python first")
                value = fn(self.np.array(safe.normalize_grid(grid, self.np), dtype=self.np.int64))
                record["prediction"] = safe.normalize_grid(value, self.np)
            except Exception as exc:
                record["error"] = safe.error_text(exc)
        record.update(stdout=output.getvalue(), stdout_truncated=output.truncated)
        return dict(sandbox_enabled=True, error=None, cases=[record])


def read_message():
    raw = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
    if not raw:
        return None
    if len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
        raise ValueError("Oversized/incomplete host message")
    return json.loads(raw)


def main():
    runtime = None
    while True:
        try:
            request = read_message()
            if request is None:
                break
            op = request.get("op")
            if op == "init" and runtime is None and set(request) == {"op", "payload"}:
                runtime = Runtime(request["payload"])
                result = dict(sandbox_enabled=True, error=None, ready=True)
            elif runtime is None:
                raise ValueError("Worker is not initialized")
            elif op == "cell" and set(request) == {"op", "code"}:
                result = runtime.cell(request["code"])
            elif op == "reset" and set(request) == {"op"}:
                runtime.reset()
                result = dict(sandbox_enabled=True, error=None, ready=True)
            elif op == "evaluate" and set(request) == {"op", "entrypoint", "grid"}:
                result = runtime.evaluate(request["entrypoint"], request["grid"])
            else:
                raise ValueError("Unknown host operation")
        except Exception as exc:
            result = dict(sandbox_enabled=runtime is not None, error=safe.error_text(exc), fatal=True)
        print(json.dumps(result, allow_nan=False), flush=True)
        if result.get("fatal"):
            break


if __name__ == "__main__":
    main()
