"""Linux CPU-only child for generated ARC solvers; never import a model here.

Only source and input grids arrive on stdin, not expected outputs or paths.
Preload numerical libraries, install limits and a fail-closed seccomp filter,
then execute. Python restrictions are defense in depth, not the OS boundary.
"""

import ast
import builtins
import collections
import contextlib
import ctypes
import errno
import functools
import io
import itertools
import json
import math
import resource
import sys
import types


MAX_SOURCE = 100_000
ENTRYPOINTS = ("solve_puzzle", "generate_puzzle_output", "solve")
MODULE_NAMES = {"numpy", "math", "collections", "itertools", "functools"}
BUILTINS = (
    "abs all any bool dict enumerate filter float frozenset int isinstance issubclass "
    "iter len list map max min next pow range repr reversed round set slice sorted str "
    "sum tuple zip print Exception ValueError TypeError IndexError KeyError "
    "StopIteration AssertionError RuntimeError ZeroDivisionError OverflowError "
    "MemoryError NotImplementedError"
).split()
FORBIDDEN = {"open", "eval", "exec", "compile", "globals", "locals", "vars", "getattr",
             "setattr", "delattr", "dir", "type", "help", "breakpoint", "input",
             "exit", "quit", "memoryview", "bytearray", "ctypes", "tofile", "dump",
             "dumps", "fromfile", "frombuffer", "memmap", "load", "save", "savez",
             "loadtxt", "savetxt", "genfromtxt", "ctypeslib"}
NUMPY_NAMES = (
    "array asarray zeros ones full empty zeros_like ones_like full_like empty_like "
    "arange indices meshgrid where argwhere nonzero flatnonzero unique bincount "
    "argmax argmin min max amin amax sum prod any all count_nonzero isin isclose "
    "array_equal allclose logical_and logical_or logical_not logical_xor "
    "rot90 flip fliplr flipud pad tile repeat stack hstack vstack dstack concatenate "
    "transpose swapaxes moveaxis reshape squeeze expand_dims roll kron copy copyto "
    "sort argsort diff clip sign abs absolute maximum minimum eye identity diag "
    "diagonal fill_diagonal unravel_index ravel_multi_index take put ix_ outer "
    "ndarray integer signedinteger unsignedinteger number int8 int16 int32 int64 "
    "uint8 uint16 uint32 uint64 bool_ float32 float64 dtype inf nan"
).split()


def validate_code(source, *, require_entrypoint=True):
    if not isinstance(source, str) or not 0 < len(source) <= MAX_SOURCE:
        raise ValueError(f"Program must contain 1..{MAX_SOURCE} characters")
    tree = ast.parse(source, filename="<tir_program>")
    entries = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ENTRYPOINTS]
    if require_entrypoint and len(entries) != 1:
        raise ValueError("Define exactly one entrypoint: solve_puzzle or generate_puzzle_output (or solve)")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name not in MODULE_NAMES for n in node.names):
                raise ValueError("Only numpy, math, collections, itertools and functools imports are allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.level or node.module not in MODULE_NAMES
                    or any(n.name.startswith("_") or n.name == "*" for n in node.names)):
                raise ValueError("Unsupported import")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN:
                raise ValueError(f"Unsupported attribute: {node.attr}")
        elif isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            if name.startswith("__") or name in FORBIDDEN:
                raise ValueError(f"Unsupported name: {name}")
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("__"):
                raise ValueError("Dunder functions are not allowed")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    return compile(tree, "<tir_program>", "exec"), entries[0] if len(entries) == 1 else None


def restrict_system_calls(extra=()):
    """Default deny; no opens, socket APIs, exec/clone, signals to peers or ptrace. extra names more
    syscalls to allow (the REPL runtime adds its interval timer)."""
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in ("read write close exit exit_group futex brk mmap munmap mprotect mremap "
                     "madvise rt_sigreturn rt_sigaction rt_sigprocmask sigaltstack "
                     "clock_gettime gettimeofday getpid gettid sched_yield getrandom fstat lseek").split() + list(extra):
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number < 0 or library.seccomp_rule_add(context, 0x7FFF0000, number, 0) != 0:
                raise RuntimeError(f"Cannot allow sandbox syscall {name}")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed; refusing unisolated execution")
    finally:
        library.seccomp_release(context)


class BoundedText(io.StringIO):
    def __init__(self, limit=4096):
        super().__init__()
        self.limit, self.truncated = limit, False

    def write(self, text):
        remaining = max(0, self.limit - self.tell())
        self.truncated |= len(text) > remaining
        super().write(text[:remaining])
        return len(text)


def modules_for(np):
    names = {
        "numpy": (np, NUMPY_NAMES),
        "math": (math, [n for n in dir(math) if not n.startswith("_")]),
        "collections": (collections, ["Counter", "defaultdict", "deque"]),
        "itertools": (itertools, ["product", "permutations", "combinations", "combinations_with_replacement",
                                  "chain", "islice", "count", "cycle", "repeat", "groupby", "accumulate"]),
        "functools": (functools, ["reduce", "lru_cache", "cache", "partial", "cmp_to_key"]),
    }
    return {key: types.SimpleNamespace(**{n: getattr(module, n) for n in attrs})
            for key, (module, attrs) in names.items()}


def normalize_grid(value, np):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.dtype.kind not in "iu":
            raise ValueError("Output must be a 2D integer array (no floats or booleans)")
        if not all(1 <= n <= 30 for n in value.shape):
            raise ValueError(f"Output shape {value.shape} is outside ARC limits 1..30")
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 30:
        raise ValueError("Output must be a nonempty 2D grid with at most 30 rows")
    width = len(value[0]) if isinstance(value[0], (list, tuple)) else 0
    if not 1 <= width <= 30:
        raise ValueError("Output width must be 1..30")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("Output grid must be rectangular")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or not 0 <= v <= 9 for v in row):
            raise ValueError("Output cells must be integers 0..9 (no floats or booleans)")
        result.append([int(v) for v in row])
    return result


def error_text(exc):
    lines, tb = [], exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<tir_program>":
            lines.append(tb.tb_lineno)
        tb = tb.tb_next
    return f"{type(exc).__name__}: {str(exc)[:1000]}" + (f" (program lines {lines[-4:]})" if lines else "")


def main():
    report = dict(sandbox_enabled=False, error=None, cases=[])
    try:
        payload = json.loads(sys.stdin.read(2_000_000))
        if set(payload) != {"code", "grids", "cpu_seconds", "memory_mib"}:
            raise ValueError("Runner accepts source, input grids and resource limits only")
        compiled, entry = validate_code(payload["code"])
        cpu, memory = payload["cpu_seconds"], payload["memory_mib"]
        if type(cpu) is not int or not 1 <= cpu <= 300 or type(memory) is not int or not 128 <= memory <= 4096:
            raise ValueError("Invalid worker limits")
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu+1))
        resource.setrlimit(resource.RLIMIT_AS, (memory * 1024**2, memory * 1024**2))
        import numpy as np
        # NumPy 2.x lazily loads dependencies, including numpy.ma from unique().
        # Warm ordinary grid operations before blocking opens, rather than
        # permitting filesystem access after generated code starts.
        probe = np.array([[0, 1]], dtype=np.int64)
        np.unique(probe)
        probe.max()
        str(probe)
        # Counter.most_common(n) lazily imports heapq (and its C extension).
        # collections is an allowed solver API, so load this stdlib dependency
        # before seccomp denies file opens. Do not weaken the OS/import policy.
        collections.Counter([0, 1, 1]).most_common(1)
        grids = payload["grids"]
        if not isinstance(grids, list) or not 1 <= len(grids) <= 128:
            raise ValueError("Expected 1..128 input grids")
        grids = [normalize_grid(g, np) for g in grids]
        module_templates = modules_for(np)
        # ndarray reductions import these already-loaded modules through the
        # caller's __import__, even though generated source never imports them.
        # AST validation still rejects private imports/attributes in user code.
        numpy_methods = {name: sys.modules[name] for name in ("numpy._core._methods", "numpy.core._methods")
                         if name in sys.modules}

        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            if not level and name in numpy_methods:
                return numpy_methods[name]
            if level or name not in modules:
                raise ImportError(f"Unsupported import: {name}")
            return modules[name]

        restrict_system_calls()
        report["sandbox_enabled"] = True
        for grid in grids:
            modules = {name: types.SimpleNamespace(**vars(module)) for name, module in module_templates.items()}
            output = BoundedText()
            record = dict(prediction=None, error=None)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                try:
                    namespace = {"__builtins__": {n: getattr(builtins, n) for n in BUILTINS}}
                    namespace["__builtins__"]["__import__"] = restricted_import
                    # Fresh globals and a fresh input per call. No expected grids
                    # enter the child, and no previous prediction is available.
                    exec(compiled, namespace, namespace)
                    record["prediction"] = normalize_grid(namespace[entry](np.array(grid, dtype=np.int64)), np)
                except Exception as exc:
                    record["error"] = error_text(exc)
            record.update(stdout=output.getvalue(), stdout_truncated=output.truncated)
            report["cases"].append(record)
    except Exception as exc:
        report["error"] = error_text(exc)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
