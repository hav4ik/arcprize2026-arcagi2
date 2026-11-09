"""Run model-written `transform` programs in isolated subprocesses.

Each job is (code, inputs). The program runs in a fresh interpreter with a
wall-clock timeout (and, on Linux, an address-space cap), so an infinite loop,
a crash or a memory blow-up costs one job, never the run. Anything the program
prints is swallowed; the result travels back as one marked JSON line.

Per input the result is {"output": grid} or {"error": message}. An output is
only accepted if it is a 2D rectangular grid of ints 0-9, 1..30 on each side.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

MARK = "@@ARC_RESULT@@"

RUNNER = r'''
import sys, json, io, contextlib, copy, traceback
MARK = "@@ARC_RESULT@@"
job = json.loads(sys.stdin.read())

def clean(res):
    try:
        import numpy as np
        if isinstance(res, np.ndarray):
            res = res.tolist()
    except Exception:
        pass
    if isinstance(res, tuple):
        res = list(res)
    if not isinstance(res, list) or not res:
        return None, "output is not a non-empty list of rows"
    rows = []
    for row in res:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or not row:
            return None, "output rows must be non-empty lists"
        out = []
        for v in row:
            try:
                iv = int(v)
            except Exception:
                return None, f"non-integer cell {v!r}"
            if iv != v or not 0 <= iv <= 9:
                return None, f"cell value {v!r} outside 0-9"
            out.append(iv)
        rows.append(out)
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        return None, "output is not rectangular"
    if not (1 <= len(rows) <= 30 and 1 <= w <= 30):
        return None, f"output size {len(rows)}x{w} outside 1..30"
    return rows, None

results = []
sink = io.StringIO()
ns = {"__name__": "__arc_program__"}
try:
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(job["code"], "<program>", "exec"), ns)
    fn = ns.get("transform")
    if not callable(fn):
        raise NameError("no function named transform")
except Exception as e:
    msg = "".join(traceback.format_exception_only(type(e), e)).strip()
    results = [{"error": "program failed to load: " + msg[-400:]} for _ in job["inputs"]]
else:
    for grid in job["inputs"]:
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                res = fn(copy.deepcopy(grid))
            rows, err = clean(res)
            results.append({"output": rows} if err is None else {"error": err})
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()
            results.append({"error": " | ".join(tb[-3:])[-400:]})
sys.stdout.write(MARK + json.dumps(results) + "\n")
sys.stdout.flush()
'''


def _limits(mem_bytes):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
    return apply if os.name == "posix" else None


def run_one(code, inputs, timeout=10.0, mem_bytes=2 << 30):
    """Run one program on a list of input grids; never raises."""
    try:
        p = subprocess.run(
            [sys.executable, "-c", RUNNER],
            input=json.dumps({"code": code, "inputs": inputs}),
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits(mem_bytes),
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", PYTHONHASHSEED="0"))
    except subprocess.TimeoutExpired:
        return [{"error": f"timed out after {timeout:.0f}s"} for _ in inputs]
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(MARK):
            try:
                res = json.loads(line[len(MARK):])
                if len(res) == len(inputs):
                    return res
            except Exception:
                break
    tail = (p.stderr or "").strip().splitlines()[-2:]
    why = " | ".join(tail)[-300:] or f"exit code {p.returncode}"
    return [{"error": "program crashed: " + why} for _ in inputs]


def run_many(jobs, timeout=10.0, workers=None, mem_bytes=2 << 30):
    """jobs: list of (code, inputs). Returns results in the same order."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, c, i, timeout, mem_bytes) for c, i in jobs]
        return [f.result() for f in futs]
