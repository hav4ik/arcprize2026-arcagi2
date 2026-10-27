"""E017 --train-check: after an attempt ends, label whether its answer program reproduces the TRAINING outputs.

No model tokens: the labeller re-runs code in the attempt's own sandbox session (and, as a last resort, replays
the attempt's executed cells in fresh sessions). Method of experiments/E017_reasoning_efficiency/label_train.py,
adapted to run live. Paths, each of which must reproduce the submitted grid on the real test input:
  * expr: the submitted code reads `test_input` (e.g. solve(test_input)); re-evaluate it with test_input bound to
    each training input;
  * cell: the last executed run_python cell that assigns the answer variable (`answer`, or another session
    variable holding the submitted grid) and reads test_input, re-executed in a copy of the globals with
    test_input rebound to each training input;
  * fn: a user-defined 1-argument session function (helpers excluded);
  * transcript: all executed cells replayed in a fresh session per training input, answer read the same way.
The best train score over the reproducing paths is the label. Only hashes of outputs leave the sandbox; the test
output never enters it (the host compares).
"""
import ast
import hashlib
import json
import time
from uuid import uuid4

from sandbox_client import tool_output  # noqa: F401  (re-exported for callers)


def grid_hash(grid):
    return hashlib.sha1(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()[:16]


PROBE = r'''
def _arc_probe():
    import contextlib, io, json, inspect, signal, types, copy, hashlib, time
    import numpy as _np
    class _T(Exception):
        pass
    def _h(*a):
        raise _T()
    old = signal.signal(signal.SIGALRM, _h)
    g = globals()
    inputs = json.loads(INPUTS)
    code = CODE
    out = {}
    def hsh(v):
        if hasattr(v, "tolist"):
            v = v.tolist()
        v = json.loads(json.dumps(v, default=int))
        return hashlib.sha1(json.dumps(v, separators=(",", ":")).encode()).hexdigest()[:16]
    def run(fn, sec):
        signal.setitimer(signal.ITIMER_REAL, sec)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return hsh(fn())
        except (Exception, _T) as exc:
            return "ERR:" + type(exc).__name__
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        if DO_EXPR and code is not None:
            try:
                expr, mode = compile(code, "<answer>", "eval"), "eval"
            except SyntaxError:
                try:
                    expr, mode = compile(code, "<answer>", "exec"), "exec"
                except SyntaxError:
                    expr, mode = None, None
            if expr is not None and "test_input" in expr.co_names:
                preds = []
                for grid in inputs:
                    def f(grid=grid):
                        env = {**g, "test_input": copy.deepcopy(grid)}
                        if mode == "eval":
                            return eval(expr, env)
                        exec(expr, env)
                        return env["answer"]
                    preds.append(run(f, 10.0))
                out["expr"] = preds
        if DO_FNS:
            t0 = time.monotonic()
            names = [n for n, v in list(g.items()) if isinstance(v, types.FunctionType) and not n.startswith("_")
                     and getattr(v, "__module__", None) == "__main__" and not getattr(v, "_arc_helper", False)]
            fns = []
            for n in names[-40:]:
                if time.monotonic() - t0 > 120:
                    break
                f = g[n]
                try:
                    inspect.signature(f).bind(None)
                except (TypeError, ValueError):
                    continue
                preds = []
                for grid in [inputs[-1]] + inputs[:-1]:  # test first: skip functions that do not reproduce it
                    p = run(lambda: f(copy.deepcopy(grid)), 3.0)
                    if p.startswith("ERR:") and p != "ERR:_T":
                        p2 = run(lambda: f(_np.array(grid)), 3.0)
                        if not p2.startswith("ERR:"):
                            p = p2
                    preds.append(p)
                    if len(preds) == 1 and p != PH:
                        break
                if preds[0] == PH:
                    fns.append({"name": n, "preds": preds[1:] + preds[:1]})
            out["fns"] = fns
        vs = []
        for n, v in list(g.items()):
            if n.startswith("_") or isinstance(v, (types.FunctionType, types.ModuleType, type)):
                continue
            if isinstance(v, (list, _np.ndarray)):
                try:
                    if hsh(v) == PH:
                        vs.append(n)
                except Exception:
                    pass
        out["vars"] = vs
    finally:
        signal.signal(signal.SIGALRM, old)
    print(MARKER + json.dumps(out))
try:
    _arc_probe()
finally:
    del _arc_probe
'''

CELL = r'''
def _arc_cell():
    import contextlib, io, json, signal, copy, hashlib
    class _T(Exception):
        pass
    def _h(*a):
        raise _T()
    old = signal.signal(signal.SIGALRM, _h)
    g = globals()
    inputs = json.loads(INPUTS)
    src, var = SRC, VAR
    code = compile(src, "<cell>", "exec")
    preds = []
    try:
        for grid in [inputs[-1]] + inputs[:-1]:  # test input first (reproduction), then train inputs
            env = {**g, "test_input": copy.deepcopy(grid)}
            env.pop(var, None)
            signal.setitimer(signal.ITIMER_REAL, 20.0)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(code, env)
                    v = env[var]
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v = json.loads(json.dumps(v, default=int))
                    preds.append(hashlib.sha1(json.dumps(v, separators=(",", ":")).encode()).hexdigest()[:16])
            except (Exception, _T) as exc:
                preds.append("ERR:" + type(exc).__name__)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            if len(preds) == 1 and preds[0] != PH:
                break
    finally:
        signal.signal(signal.SIGALRM, old)
    print(MARKER + json.dumps(preds[1:] + preds[:1]))
try:
    _arc_cell()
finally:
    del _arc_cell
'''


def _fill(template, **values):
    for k, v in values.items():  # CODE/SRC are filled last so model text cannot contain a placeholder we replace
        template = template.replace(k, v)
    return template


def _run_marked(session, cell, marker, timeout):
    old = session.timeout, session.max_output
    session.timeout, session.max_output = timeout, 5_000_000
    try:
        result = session.execute(cell)
    finally:
        session.timeout, session.max_output = old
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("train-check cell: " + result["process_status"] + " " + (result.get("stderr") or "")[-300:])


def _inputs(session):
    return repr(json.dumps([p["input"] for p in session.train] + [session.test_input]))


def probe(session, code, ph, do_expr=True, do_fns=False):
    marker = f"ARC_TC_{uuid4().hex}:"
    cell = _fill(PROBE, INPUTS=_inputs(session), MARKER=repr(marker), DO_EXPR=str(bool(do_expr)),
                 DO_FNS=str(bool(do_fns)), PH=repr(ph), CODE=repr(code))
    return _run_marked(session, cell, marker, timeout=300)


def cell_eval(session, src, var, ph):
    marker = f"ARC_TC_{uuid4().hex}:"
    cell = _fill(CELL, INPUTS=_inputs(session), MARKER=repr(marker), VAR=repr(var), PH=repr(ph), SRC=repr(src))
    return _run_marked(session, cell, marker, timeout=20 * (len(session.train) + 1) + 30)


def stored_names(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), set()
    stores = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return stores, loads


def train_check(session, executed, pred, code, replay_session=None, submit=None, transcript_limit=300,
                reject_constant=False, transcript_fallback=False):
    """Label one finished attempt. `executed`: [(tool name, code)] in execution order; `pred`: the scored grid;
    `code`: the accepted submit_answer code ('answer' for the session fallback; None for a text answer).
    `replay_session(test_input)` opens a fresh session (transcript path); `submit` is run_deepseek_tools'
    submit_answer. Returns {train_pass, train_frac, train_trace, train_paths, train_check_seconds}.
    reject_constant (E042, opt-in): an expr/cell/fn path that returns the test answer for every training input
    does not read its input (e.g. a last cell `answer = cands[0]` that only prints test_input), so it is not
    counted, and the check falls through to the next path (in the end the transcript replay).
    transcript_fallback (E042, opt-in): also replay the transcript when the other paths exist but none passes every
    training pair (a last cell that reads test_input but also values earlier cells derived from it scores ~0)."""
    t0 = time.monotonic()
    outs = [grid_hash(p["output"]) for p in session.train]
    n, ph = len(outs), grid_hash(pred)
    paths = []  # (train_ok, priority, name)

    def score(preds):
        return sum(a == b for a, b in zip(preds[:-1], outs))

    def constant(preds):
        return reject_constant and all(p == ph for p in preds[:-1]) and ph not in outs

    def done():
        return any(k == n for k, _, _ in paths)

    data = probe(session, code, ph)
    if data.get("expr") is not None and data["expr"][-1] == ph and not constant(data["expr"]):
        paths.append((score(data["expr"]), 3, "expr"))
    if not done():
        cells = [c for name, c in executed if name == "run_python"]
        for var in sorted(data.get("vars", []), key=lambda v: v != "answer")[:3]:
            for src in reversed(cells):
                stores, loads = stored_names(src)
                if var in stores:
                    if "test_input" in loads:
                        res = cell_eval(session, src, var, ph)
                        if res[-1] == ph and not constant(res):
                            paths.append((score(res), 2, "cell"))
                    break  # only the last assigning cell counts
            if any(p[2] == "cell" for p in paths):
                break
    if not done():
        fns = [f for f in probe(session, None, ph, do_expr=False, do_fns=True).get("fns", [])
               if not constant(f["preds"])]
        if fns:
            paths.append((max(score(f["preds"]) for f in fns), 1, "fn"))
    label = {}
    if (not paths or (transcript_fallback and not done())) and replay_session is not None and submit is not None:
        read = code if code not in (None, "answer") else ("answer" if "answer" in data.get("vars", []) else
                                                          (data.get("vars") or [None])[0])
        if read and len(executed) <= 120:
            preds = []
            for pair in session.train:
                if time.monotonic() - t0 > transcript_limit:
                    break
                with replay_session(pair["input"]) as s:
                    for name, c in executed:
                        if time.monotonic() - t0 > transcript_limit:
                            break
                        if name == "run_python":
                            s.execute(c)
                        elif name == "submit_answer":
                            submit(s, c)
                    _, grid = submit(s, read)
                    preds.append(grid_hash(grid) if grid is not None else "ERR")
            if len(preds) == n:
                paths.append((score(preds + [ph]), 0, "transcript"))
            else:
                label["train_check_note"] = "transcript replay over time limit"
    if paths:
        k, _, name = max(paths)
        label.update(train_pass=k == n, train_frac=round(k / n, 4), train_trace=name,
                     train_paths=[p[2] for p in paths])
    else:
        label.update(train_pass=None, train_frac=None, train_trace="untraceable", train_paths=[])
    label["train_check_seconds"] = round(time.monotonic() - t0, 1)
    return label


def ranked_answers(recs, k=2):
    """Kaggle answer selection for one test pair: attempts ranked by train_pass (True first), then attempt
    order; the top-k distinct non-empty predictions."""
    order = sorted((r for r in recs if r.get("pred") is not None),
                   key=lambda r: (r.get("train_pass") is not True, r["attempt"]))
    out = []
    for r in order:
        if r["pred"] not in [o["pred"] for o in out]:
            out.append(r)
        if len(out) == k:
            break
    return out
