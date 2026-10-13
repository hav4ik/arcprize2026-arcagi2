"""Code-execution backends for the tool harness, behind one small interface.

An executor is a persistent Python session for one attempt:

    with make_executor(train, test_input) as ex:     # train: [{'input', 'output'}], test_input: grid
        result = ex.execute(code)                    # {'process_status': completed|error|timeout,
                                                     #  'stdout': str, 'stderr': str, ['state_reset': True]}

It preloads `train`, `test_input`, `np` and `json`, keeps variables between cells, and shows the value of a
cell's last expression (IPython-style). Attributes used by the answer helpers: `train`, `test_input`, `max_output`.

Backends:
- `SandboxSession` (evals/sandbox_client.py): the Docker NeMo-Skills sandbox (no network, no host mounts).
- `LocalExecutor` (here): a child Python process per attempt, for Kaggle where Docker is unavailable. It is not a
  security boundary (same user, same filesystem). It adds limits (address space, cumulative CPU, file size, nice,
  oom_score_adj=1000), a per-cell timeout (SIGINT to the process group, then kill and restart) and output truncation.

`salvage_answer` (qwen-tools-3) recovers an answer from the session's own functions when the model left none.
`submit_answer` / `check_answer_on_train` / `tool_output` are kept identical to evals/run_deepseek_tools.py and
evals/sandbox_client.py (copied, so this package runs without those scripts).
"""
import json
import os
import select
import signal
import subprocess
import sys
import time
from uuid import uuid4

from .grids import valid_grid

# ---------------------------------------------------------------- local executor

_CHILD = r'''
import ast, contextlib, io, json, os, resource, signal, sys, traceback
resp = os.fdopen(int(sys.argv[1]), "w")
limit = int(sys.argv[2])
mem, cpu, fsize, nice = (int(x) for x in sys.argv[3:7])
# Self-imposed limits (no preexec_fn: the parent spawns from many threads). The child leads its own session.
if mem:
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
if cpu:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
if fsize:
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # oversized writes fail with OSError instead of killing us
if nice:
    os.nice(nice)
try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")  # under host memory pressure, kill tool code before the model server
except OSError:
    pass
ns = {"__name__": "__main__"}

def run(code):
    tree = ast.parse(code, "<cell>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), ns)
    if last is not None:
        value = eval(compile(last, "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))

def plain_traceback(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename != "<string>"]
    text = "".join(traceback.format_list(frames)) if frames else ""
    return "Traceback (most recent call last):\n" + text + "".join(traceback.format_exception_only(type(exc), exc))

for line in sys.stdin:
    req = json.loads(line)
    out, err = io.StringIO(), io.StringIO()
    status = "completed"
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run(req["code"])
    except KeyboardInterrupt:
        status = "timeout"
        err.write("KeyboardInterrupt: cell timed out (state kept)\n")
    except BaseException as exc:
        status = "error"
        try:
            err.write(plain_traceback(exc))
        except BaseException:
            err.write(f"{type(exc).__name__}: {exc}\n")
    o, e = out.getvalue(), err.getvalue()
    if len(o) > limit:
        o = o[:limit] + "\n[output truncated]"
    if len(e) > limit:
        e = e[:limit] + "\n[output truncated]"
    resp.write(json.dumps({"process_status": status, "stdout": o, "stderr": e}) + "\n")
    resp.flush()
'''


class LocalExecutor:
    """Persistent Python in a child process. On timeout the cell gets SIGINT (state kept, like the patched
    sandbox); if it does not stop within `grace` seconds the process is killed and restarted with only the
    task data (`state_reset`)."""

    def __init__(self, train, test_input, timeout=30, max_output=16000, memory_mb=4096, grace=5,
                 python=sys.executable, cpu_seconds=1200, fsize_mb=64, nice=10, start_timeout=90, prelude_extra=""):
        self.train, self.test_input = train, test_input
        self.timeout, self.max_output, self.memory_mb, self.grace = timeout, max_output, memory_mb, grace
        self.python, self.cpu_seconds, self.fsize_mb, self.nice = python, cpu_seconds, fsize_mb, nice
        self.start_timeout = start_timeout  # the prelude (numpy import) may be slow on a busy 4-CPU box
        self.proc = self.resp = None
        self.closed = False
        data = json.dumps({"train": train, "test_input": test_input})
        self.prelude = (
            "import json, numpy as np\n"
            f"_arc_data = json.loads({data!r})\n"
            "train, test_input = _arc_data['train'], _arc_data['test_input']\n"
            "del _arc_data\n"
        ) + prelude_extra  # E020 --helpers: re-run with the task data after a restart

    def _spawn_once(self, timeout):
        r, w = os.pipe()
        self.proc = subprocess.Popen(
            [self.python, "-I", "-u", "-c", _CHILD, str(w), str(self.max_output), str(self.memory_mb * 2 ** 20),
             str(int(self.cpu_seconds)), str(self.fsize_mb * 2 ** 20), str(self.nice)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(w,),
            start_new_session=True, text=True, cwd="/tmp",
            env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1"})
        os.close(w)
        self.resp, self.buf = r, b""
        return self._run(self.prelude, timeout)

    def _spawn(self):
        """Start the child and load the task data; one retry with a longer timeout, else RuntimeError."""
        for timeout in (max(self.timeout, 30), self.start_timeout):
            result = self._spawn_once(timeout)
            if result is not None and result["process_status"] == "completed":
                return
            self._kill()
        detail = "" if result is None else result.get("stderr", "")[:500]
        raise RuntimeError(f"Local executor initialization failed {detail}".strip())

    def _read(self, deadline):
        while b"\n" not in self.buf:
            wait = deadline - time.monotonic()
            if wait <= 0:
                return None
            ready, _, _ = select.select([self.resp], [], [], wait)
            if ready:
                chunk = os.read(self.resp, 1 << 16)
                if not chunk:
                    return "eof"
                self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def _signal_group(self, sig):
        try:
            os.killpg(self.proc.pid, sig)  # the child leads its own session: this also reaches grandchildren
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass

    def _kill(self):
        if self.proc is not None:
            self._signal_group(signal.SIGKILL)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        if self.resp is not None:
            os.close(self.resp)
        self.proc = self.resp = None

    def _run(self, code, timeout=None):
        """Run one cell; returns the result, or None if the process died or had to be killed."""
        self.last_failure = "error"
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return None
        result = self._read(time.monotonic() + (timeout or self.timeout))
        if result is None:  # timed out: interrupt, keep state if the cell stops
            self._signal_group(signal.SIGINT)
            self.last_failure = "timeout"
            result = self._read(time.monotonic() + self.grace)
            if isinstance(result, dict):
                result["process_status"] = "timeout"
                return result
            return None
        return result if isinstance(result, dict) else None

    def initialize(self):
        self._spawn()

    def execute(self, code):
        if self.closed:
            raise RuntimeError("Executor is closed")
        if self.proc is None:
            self._spawn()
        result = self._run(code)
        if result is None:
            # Died (memory or CPU limit, os._exit, segfault) or ignored SIGINT: restart with task data only.
            self._kill()
            try:
                self._spawn()
            except RuntimeError as exc:  # leave proc None: the next execute tries to start again
                return {"process_status": "error", "stdout": "", "state_reset": True,
                        "stderr": f"Python process died and could not be restarted: {exc}"}
            result = {"process_status": self.last_failure, "stdout": "", "state_reset": True,
                      "stderr": ("Cell did not stop after the timeout; process restarted." if self.last_failure == "timeout"
                                 else "Python process died (crash, exit or memory limit); process restarted.")}
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self._kill()

    def __enter__(self):
        try:
            self.initialize()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()


def local_factory(timeout=30, max_output=16000, memory_mb=4096, **kwargs):
    """kwargs: grace, cpu_seconds, fsize_mb, nice, start_timeout, python (see LocalExecutor)."""
    return lambda train, test_input: LocalExecutor(train, test_input, timeout, max_output, memory_mb, **kwargs)


# E021 (Kaggle v3): the evals/run_qwen_tools.py --session-mem-mb prelude (same text), importable without the
# runner. Run first in a session (prelude_extra): caps the kernel's address space at its current VmSize + {mb} MB,
# so a runaway cell raises MemoryError instead of exhausting host memory.
SESSION_MEM_CAP = """
def _arc_memcap():
    import resource
    try:
        with open('/proc/self/status') as f:
            vm = next(int(l.split()[1]) * 1024 for l in f if l.startswith('VmSize:'))
        cap = vm + {mb} * 2 ** 20
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY or cap < hard:
            resource.setrlimit(resource.RLIMIT_AS, (cap, hard if hard != resource.RLIM_INFINITY else cap))
    except Exception:
        pass
_arc_memcap()
del _arc_memcap
"""


def sandbox_factory(url, timeout=30, max_output=16000, prelude_extra=""):
    """Docker sandbox sessions (evals/sandbox_client.py must be importable, i.e. evals/ on sys.path)."""
    from sandbox_client import SandboxSession
    if prelude_extra:
        return lambda train, test_input: SandboxSession(url, train, test_input, timeout, max_output,
                                                        prelude_extra=prelude_extra)
    return lambda train, test_input: SandboxSession(url, train, test_input, timeout, max_output)


# ---------------------------------------------------------------- helpers shared with run_deepseek_tools.py

def tool_output(result, limit=16000):
    text = result.get("stdout", "")
    if result.get("stderr"):
        text += "\n[stderr]\n" + result["stderr"]
    if result.get("state_reset"):
        text = "[Python state lost. train/test_input/np restored; redefine other variables.]\n" + text
    text = f"[{result['process_status']}]\n" + (text or "[no output; use print()]")
    return text if len(text) <= limit else text[:limit] + "\n[output truncated]"


def submit_answer(session, code):
    """Evaluate the model's answer expression in its own session and return (feedback, grid|None).

    `code` is an expression (e.g. solve(test_input)) or statements that set `answer`."""
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
    """Re-evaluate the submitted expression with `test_input` bound to each training input and compare with the
    host's training outputs. None when there is nothing to check, else (failed_indices, feedback_text)."""
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
        return None
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


def salvage_answer(session, max_functions=8):
    """No answer was submitted or set: score the session's own one-argument functions on the training inputs
    (host-side comparison) and apply the best one to test_input. Returns (info, grid|None); info carries the
    function name and how many training pairs it reproduced. Costs no model tokens."""
    marker = f"ARC_SALVAGE_{uuid4().hex}:"
    inputs = [pair["input"] for pair in session.train] + [session.test_input]
    cell = f'''
def _arc_salvage():
    import contextlib, copy, inspect, io, json
    import numpy as _np
    fns = []
    for _name, _f in list(globals().items())[::-1]:
        if _name.startswith('_') or not inspect.isfunction(_f) or _f.__module__ != '__main__':
            continue
        try:
            _req = [p for p in inspect.signature(_f).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            continue
        if len(_req) == 1:
            fns.append((_name, _f))
        if len(fns) >= {max_functions}:
            break
    res = {{}}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for _name, _f in fns:
            outs = []
            for _inp in json.loads({json.dumps(inputs)!r}):
                v = None
                for arg in (copy.deepcopy(_inp), _np.array(_inp)):
                    try:
                        v = _f(arg)
                        if hasattr(v, 'tolist'):
                            v = v.tolist()
                        v = json.loads(json.dumps(v, default=int))
                        break
                    except Exception:
                        v = None
                outs.append(v)
            res[_name] = outs
    print({marker!r} + json.dumps(res))
try:
    _arc_salvage()
finally:
    del _arc_salvage
'''
    result = session.execute(cell)
    found = None
    if result["process_status"] == "completed":
        for line in reversed(result.get("stdout", "").splitlines()):
            if line.startswith(marker):
                try:
                    found = json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    pass
                break
    if not isinstance(found, dict):
        return {"status": result["process_status"]}, None
    best = None  # (train pairs matched, newest first) -> name, grid
    for rank, (name, outs) in enumerate(found.items()):
        if not isinstance(outs, list) or len(outs) != len(inputs) or not valid_grid(outs[-1]):
            continue
        score = sum(o == pair["output"] for o, pair in zip(outs, session.train))
        if best is None or score > best[0]:
            best = (score, name, outs[-1])
    if best is None:
        return {"functions": list(found)}, None
    return {"function": best[1], "train_matched": best[0], "train_pairs": len(session.train)}, best[2]


# ---------------------------------------------------------------- program answers (qwen-tools-4, answer_mode=program)

_PROGRAM_CHILD = r'''
import contextlib, io, json, linecache, resource, signal, sys, traceback
req = json.loads(sys.stdin.read())
code, timeout = req["code"], req["timeout"]
try:
    if req.get("mem"):
        resource.setrlimit(resource.RLIMIT_AS, (req["mem"], req["mem"]))
except (ValueError, OSError):
    pass

class _ProgramTimeout(BaseException):
    pass

def _alarm(*_):
    raise _ProgramTimeout()

def _plain(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return int(o)

linecache.cache["<program>"] = (len(code), None, code.splitlines(True), "<program>")
signal.signal(signal.SIGALRM, _alarm)
buf = io.StringIO()
try:
    signal.setitimer(signal.ITIMER_REAL, timeout)
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import numpy as np
        ns = {"__name__": "__arc_program__", "np": np}
        if req.get("prelude"):
            exec(compile(req["prelude"], "<helpers>", "exec"), ns)
        exec(compile(code, "<program>", "exec"), ns)
        fn = ns.get("solve")
        if not callable(fn):
            raise NameError("the program does not define solve(grid)")
        value = fn(req["grid"])
        value = json.loads(json.dumps(value, default=_plain))
    res = {"out": value}
except _ProgramTimeout:
    res = {"error": f"timeout: more than {timeout:g}s", "kind": "timeout"}
except BaseException as exc:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<program>"]
    lines = [f"  line {f.lineno}, in {f.name}: {(f.line or '').strip()[:160]}" for f in frames[-2:]]
    msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()[-400:]
    res = {"error": "\n".join(lines + [msg]), "kind": type(exc).__name__}
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
sys.stdout.write(req["marker"] + json.dumps(res) + "\n")
'''


def _marker_json(result, marker):
    if result.get("process_status") != "completed":
        return None
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return None
    return None


def run_program(session, code, grid, timeout=10, memory_mb=3072, prelude=""):
    """Run self-contained `code` (which must define solve(grid)) on one grid in a FRESH Python process started
    from the session's environment (sandbox or LocalExecutor child), so session state cannot leak in.
    Returns {'out': value} or {'error': text, 'kind': name}; per-grid wall-clock limit `timeout` seconds.
    `prelude` (E020 --helpers) is executed in the program's namespace first (e.g. the arc_helpers source)."""
    marker = f"ARC_PROGRAM_{uuid4().hex}:"
    body = {"code": code, "grid": grid, "timeout": timeout, "marker": marker, "mem": memory_mb * 2 ** 20}
    if prelude:
        body["prelude"] = prelude
    req = json.dumps(body)
    cell = f'''
def _arc_program():
    import os, shutil, subprocess, sys
    exe = sys.executable  # the Docker sandbox runs cells under uwsgi: use the python on PATH there
    if not os.path.basename(exe or "").startswith("python"):
        exe = shutil.which("python3") or shutil.which("python") or exe
    try:
        r = subprocess.run([exe, "-I", "-c", {_PROGRAM_CHILD!r}], input={req!r}, capture_output=True,
                           text=True, timeout={timeout + 10!r}, cwd="/tmp")
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ""
    for line in out.splitlines()[::-1]:
        if line.startswith({marker!r}):
            print(line)
            return
    print({marker!r} + '{{"error": "timeout or crash (no result; memory limit {memory_mb} MB)", "kind": "crash"}}')
try:
    _arc_program()
finally:
    del _arc_program
'''
    res = _marker_json(session.execute(cell), marker)
    return res if isinstance(res, dict) else {"error": "program runner failed", "kind": "runner"}


def run_session_solve(session, grid, timeout=10):
    """Apply the session's own `solve` to one grid (fallback when no submitted program gives a test output).
    Runs inside the session (it may use session helpers). Returns {'out': value} or {'error': text}; None if the
    session has no callable `solve`."""
    marker = f"ARC_SSOLVE_{uuid4().hex}:"
    cell = f'''
def _arc_ssolve():
    import contextlib, copy, io, json, signal
    if not callable(globals().get("solve")):
        print({marker!r} + "null")
        return
    class _T(BaseException):
        pass
    def _h(*_):
        raise _T()
    def _plain(o):
        return o.tolist() if hasattr(o, "tolist") else int(o)
    old = signal.signal(signal.SIGALRM, _h)
    try:
        signal.setitimer(signal.ITIMER_REAL, {timeout!r})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            v = globals()["solve"](copy.deepcopy(json.loads({json.dumps(grid)!r})))
            v = json.loads(json.dumps(v, default=_plain))
        res = {{"out": v}}
    except _T:
        res = {{"error": "timeout"}}
    except Exception as exc:
        res = {{"error": type(exc).__name__ + ": " + str(exc)[:200]}}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    print({marker!r} + json.dumps(res))
try:
    _arc_ssolve()
finally:
    del _arc_ssolve
'''
    result = session.execute(cell)
    if result.get("process_status") != "completed":
        return {"error": "session solve did not complete"}
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except json.JSONDecodeError:
                break
    return {"error": "no result"}


def grid_shape(g):
    if isinstance(g, list) and g and all(isinstance(r, list) for r in g):
        widths = sorted({len(r) for r in g})
        return f"{len(g)}x{widths[0]}" if len(widths) == 1 else f"{len(g)} rows of widths {widths[:5]}"
    return type(g).__name__


def score_outputs(outs, train):
    """Per training pair: {'ok', 'error'?, 'shape'?, 'wrong'?} comparing program outputs with the host's outputs."""
    rows = []
    for o, pair in zip(outs, train):
        exp = pair["output"]
        if "error" in o:
            rows.append({"ok": False, "error": o["error"], "kind": o.get("kind")})
            continue
        got = o["out"]
        if got == exp:
            rows.append({"ok": True})
        elif not valid_grid(got):
            rows.append({"ok": False, "error": f"not a valid grid ({grid_shape(got)}; need a rectangular 1-30 x "
                                               "1-30 grid of ints 0-9)", "kind": "invalid"})
        elif (len(got), len(got[0])) != (len(exp), len(exp[0])):
            rows.append({"ok": False, "shape": (grid_shape(got), grid_shape(exp)), "got": got})
        else:
            rows.append({"ok": False, "wrong": [(r, c, exp[r][c], got[r][c]) for r in range(len(exp))
                                                for c in range(len(exp[0])) if exp[r][c] != got[r][c]]})
    return rows


def _digits(g):
    return "\n".join("".join(str(v) for v in row) for row in g)


def rich_detail(row, pair, max_side=20):
    """E020 --feedback rich: extra lines for one failing pair (needs the host's training output)."""
    exp, lines = pair["output"], []
    if "wrong" in row:
        conf = {}
        for _, _, e, g in row["wrong"]:
            conf[(e, g)] = conf.get((e, g), 0) + 1
        top = sorted(conf.items(), key=lambda kv: -kv[1])[:8]
        lines.append("colour confusions (expected->got x count): " + ", ".join(f"{e}->{g} x{n}" for (e, g), n in top))
        if len(exp) <= max_side and len(exp[0]) <= max_side:
            bad = {(r, c): g for r, c, _, g in row["wrong"]}
            got = "\n".join("".join(str(bad[(r, c)]) if (r, c) in bad else "." for c in range(len(exp[0])))
                            for r in range(len(exp)))
            lines.append(f"expected output:\n{_digits(exp)}\nyour output ('.' = correct cell, digit = your wrong "
                         f"value):\n{got}")
    elif "shape" in row and row.get("got") is not None:
        got = row["got"]
        h, w = len(exp), len(exp[0])
        gh, gw = len(got), len(got[0])
        hint = []
        if (gh, gw) == (w, h):
            hint.append("your output is transposed in shape")
        for name, a, b in (("height", gh, h), ("width", gw, w)):
            if a != b and b and a % b == 0:
                hint.append(f"your {name} is {a // b}x the expected")
            elif a != b and a and b % a == 0:
                hint.append(f"the expected {name} is {b // a}x yours")
        if hint:
            lines.append("shape hint: " + "; ".join(hint))
        if max(h, w, gh, gw) <= 12:
            lines.append(f"expected output:\n{_digits(exp)}\nyour output:\n{_digits(got)}")
    return lines


def program_feedback(rows, k, cap, max_cells=12, test_error=None, rich=False, train=None):
    """Short tool result for a rejected program: which pairs fail and why, a capped diff for the first one.
    rich (E020): also colour confusions and a small-grid diff of the first failing pair (needs `train`)."""
    n, bad = len(rows), [i for i, r in enumerate(rows) if not r["ok"]]
    if not bad:
        head = (f"[program rejected (submission {k} of {cap}): it passes all {n} training pairs but failed on "
                f"test_input: {test_error}]")
        return head + "\nFix solve so that it also handles the test input."
    lines = [f"[program rejected (submission {k} of {cap}): {len(bad)} of {n} training pairs fail]"]
    for i, r in enumerate(rows):
        if r["ok"]:
            lines.append(f"pair {i}: OK")
        elif "error" in r:
            err = r["error"].splitlines()
            lines.append(f"pair {i}: error: " + (err[-1] if err else ""))
        elif "shape" in r:
            lines.append(f"pair {i}: wrong shape: got {r['shape'][0]}, expected {r['shape'][1]}")
        else:
            lines.append(f"pair {i}: right shape, {len(r['wrong'])} wrong cells")
    first = rows[bad[0]]
    if "error" in first and "\n" in first["error"]:
        lines.append(f"pair {bad[0]} traceback (last lines):\n{first['error']}")
    elif "wrong" in first:
        cells = "; ".join(f"({r},{c}) expected {e} got {g}" for r, c, e, g in first["wrong"][:max_cells])
        more = len(first["wrong"]) - max_cells
        lines.append(f"pair {bad[0]} first wrong cells (row,col): {cells}" + (f"; ... {more} more" if more > 0 else ""))
    if rich and train is not None:
        lines += rich_detail(first, train[bad[0]])
    return "\n".join(lines)
