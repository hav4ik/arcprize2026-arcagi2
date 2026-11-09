"""E017 phase 1A: label each tool-run attempt with whether its answer program passes the TRAINING pairs (no API).

For every attempt that has an answer (pred) and a trace: open a fresh sandbox session with the task data (as the
harness does), replay every executed run_python / submit_answer cell in order (E016 replay, 0 divergences there),
then run one label cell:
  * expr path: if the accepted submit code references `test_input` (e.g. solve(test_input), or statements that set
    `answer` from test_input), re-evaluate it with test_input bound to each training input and to the test input.
  * fn path (always, unless the expr path reproduced the submitted grid): every user-defined 1-arg function is
    called on every training input and the test input (list first, np.array retry); functions whose test output
    equals the submitted grid are the answer's candidate programs.
Only hashes of the outputs come back; all comparisons with train outputs / the submitted grid happen on the host.

Labels: trace in {expr, fn, untraceable, no_answer}; train_pass (all train pairs), train_frac (best candidate).
The test output is never sent to the sandbox.

    .venv/bin/python experiments/E017_reasoning_efficiency/label_train.py --sandbox-url http://172.18.0.2:6000 \
        --runs sub80s40,sub96n,s20x60,GL [--workers 16] [--limit N]
"""
import argparse
import ast
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "experiments/E016_cutoff_continuation"))
from continue_cutoffs import executed_calls, load_trace, strip_note  # noqa: E402
from run_deepseek_tools import load_tasks, submit_answer, tool_output  # noqa: E402
from sandbox_client import SandboxSession, check_health  # noqa: E402

RES = ROOT / "evals/results"
RUNS = {
    "GL": RES / "tools_persistent_grid_lean2_deepseek-flash_low_full_a0.jsonl",
    "sub80s40": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
    "sub96n": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n.jsonl",
    "sub96s40": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96s40.jsonl",
    "sub64a": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub64a.jsonl",
    "s20x40": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x40.jsonl",
    "s20x60": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s20x60.jsonl",
    "s25x80": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e010_full_s25x80.jsonl",
    "sub80s40v": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e011_full_sub80s40v.jsonl",
    "subv80s40": RES / "tools_persistent_grid_lean2subv_deepseek-flash_low_e011_full_subv80s40.jsonl",
    "sub80s40h": RES / "tools_persistent_grid_lean2sub_deepseek-flash_high_e011_full_sub80s40h.jsonl",
}
OUT = ROOT / "experiments/E017_reasoning_efficiency/train_labels.jsonl"


class Session(SandboxSession):
    """SandboxSession whose cleanup waits longer (deletes can queue behind busy workers) and retries once."""

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for i in range(2):
                try:
                    r = self.client.delete(f"/sessions/{self.session_id}", timeout=120)
                    if r.status_code in (200, 204, 404):
                        break
                except Exception:
                    if i:
                        print(f"warning: could not delete sandbox session {self.session_id}", flush=True)
        finally:
            self.client.close()


def h(grid):
    return hashlib.sha1(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()[:16]


def load_run(path):
    """Latest record per (task, pair, attempt); API-error records dropped (as evals/frontier.py)."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return [r for r in out.values() if "error" not in r]


def accepted_submit_code(messages):
    """Code of the submit_answer call the harness accepted (last one), else None."""
    results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
    code = None
    for m in messages:
        for c in m.get("tool_calls", []) if m["role"] == "assistant" else []:
            if c["function"]["name"] == "submit_answer" and results.get(c["id"], "").startswith("[answer accepted"):
                code = json.loads(c["function"]["arguments"])["code"]
    return code


LABEL = r'''
def _arc_label():
    import contextlib, io, json, inspect, signal, types, copy, hashlib
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
        except BaseException as exc:
            return "ERR:" + type(exc).__name__
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        if code is not None:
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
                    preds.append(run(f, TSEC))
                out["expr"] = preds
        if FNS:
            names = [n for n, v in list(g.items()) if isinstance(v, types.FunctionType) and not n.startswith("_")
                     and getattr(v, "__module__", None) == "__main__"]
            fns = []
            for n in names[-60:]:
                f = g[n]
                try:
                    inspect.signature(f).bind(None)
                except (TypeError, ValueError):
                    continue
                preds = []
                for grid in inputs:
                    p = run(lambda: f(copy.deepcopy(grid)), PSEC)
                    if p.startswith("ERR:") and p != "ERR:_T":
                        p2 = run(lambda: f(_np.array(grid)), PSEC)
                        if not p2.startswith("ERR:"):
                            p = p2
                    preds.append(p)
                fns.append({"name": n, "preds": preds})
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
    _arc_label()
finally:
    del _arc_label
'''


def run_marked(session, cell, marker, timeout=600):
    old = session.timeout, session.max_output
    session.timeout, session.max_output = timeout, 5_000_000
    try:
        result = session.execute(cell)
    finally:
        session.timeout, session.max_output = old
    for line in reversed(result.get("stdout", "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("label cell: " + result["process_status"] + " " + (result.get("stderr") or "")[-300:])


def label_cell(session, code, ph):
    marker = f"ARC_LABEL_{uuid4().hex}:"
    inputs = [p["input"] for p in session.train] + [session.test_input]
    cell = LABEL
    for k, v in [("INPUTS", repr(json.dumps(inputs))), ("MARKER", repr(marker)), ("FNS", "True"), ("TSEC", "10.0"),
                 ("PSEC", "3.0"), ("PH", repr(ph)), ("CODE", repr(code))]:  # CODE last: model text is opaque
        cell = cell.replace(k, v)
    return run_marked(session, cell, marker)


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
            signal.setitimer(signal.ITIMER_REAL, 30.0)
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(code, env)
                    v = env[var]
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v = json.loads(json.dumps(v, default=int))
                    preds.append(hashlib.sha1(json.dumps(v, separators=(",", ":")).encode()).hexdigest()[:16])
            except BaseException as exc:
                preds.append("ERR:" + type(exc).__name__)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old)
    print(MARKER + json.dumps(preds[1:] + preds[:1]))
try:
    _arc_cell()
finally:
    del _arc_cell
'''


def cell_eval(session, src, var):
    marker = f"ARC_CELL_{uuid4().hex}:"
    inputs = [p["input"] for p in session.train] + [session.test_input]
    cell = CELL
    for k, v in [("INPUTS", repr(json.dumps(inputs))), ("MARKER", repr(marker)), ("VAR", repr(var)),
                 ("SRC", repr(src))]:
        cell = cell.replace(k, v)
    return run_marked(session, cell, marker, timeout=30 * (len(inputs) + 1) + 30)


def replay(session, messages):
    st = {"cells": 0, "diverged": 0, "state_lost": 0}
    for name, code, original in executed_calls(messages):
        if name == "run_python":
            result = session.execute(code)
            new = tool_output(result, session.max_output)
            st["state_lost"] += bool(result.get("state_reset"))
            st["cells"] += 1
            st["diverged"] += new != strip_note(original)
        elif name == "submit_answer":
            submit_answer(session, code)  # same side effects as the original (e.g. statements that set answer)
    return st


def stored_names(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set(), set()
    stores = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return stores, loads


def transcript_eval(url, task, rec, messages, read_code):
    """Replay the whole transcript with test_input bound to each training input; read the answer the same way."""
    cfg = rec["config"]
    preds = []
    for pair in task["train"]:
        with Session(url, task["train"], pair["input"], cfg["tool_timeout"], cfg["max_tool_output"]) as s:
            replay(s, messages)
            _, grid = submit_answer(s, read_code)
            preds.append(h(grid) if grid is not None else "ERR")
    return preds


def label_attempt(url, run, rec, task, transcript=True):
    test = task["test"][rec["pair"]]
    row = {"run": run, "task_id": rec["task_id"], "pair": rec["pair"], "attempt": rec["attempt"],
           "correct": bool(rec.get("correct")), "answer_source": rec.get("answer_source"),
           "stop_reason": rec.get("stop_reason"), "out_tokens": rec["usage"]["completion"],
           "prompt_tokens": rec["usage"]["prompt"], "cache_hit": rec["usage"].get("cache_hit", 0),
           "cost_usd": rec.get("est_cost_usd"), "turns": rec.get("turns"), "n_train": len(task["train"]),
           "submit_verified": rec.get("submit_verified")}
    pred = rec.get("pred")
    if pred is None:
        row.update(trace="no_answer", train_pass=False, train_frac=None)
        return row
    messages = load_trace(rec)["messages"]
    code = accepted_submit_code(messages)
    if code is None and rec.get("answer_source") == "session":
        code = "answer"
    row["submit_code"] = code[:300] if code else None
    cfg = rec["config"]
    outs = [h(p["output"]) for p in task["train"]]
    ph = h(pred)
    n = len(outs)
    row["pred_hash"] = ph
    with Session(url, task["train"], test["input"], cfg["tool_timeout"], cfg["max_tool_output"]) as s:
        row["replay"] = replay(s, messages)
        data = label_cell(s, code, ph)
        expr = data.get("expr")
        if expr is not None:
            row["expr_train_ok"] = sum(a == b for a, b in zip(expr[:-1], outs))
            row["expr_reproduces"] = expr[-1] == ph
        fns = data.get("fns", [])
        cands = [f for f in fns if f["preds"][-1] == ph]
        row["n_fns"], row["n_fn_cands"] = len(fns), len(cands)
        fit_all = [f for f in fns if all(a == b for a, b in zip(f["preds"][:-1], outs))]
        row["n_fns_fit_train"] = len(fit_all)
        row["fit_train_answers"] = sorted({f["preds"][-1] for f in fit_all if not f["preds"][-1].startswith("ERR")})
        row["vars"] = data.get("vars", [])[:10]
        paths = []  # (train_ok, priority, name): every program that reproduces the submitted grid
        if expr is not None and row["expr_reproduces"]:
            paths.append((row["expr_train_ok"], 3, "expr"))
        # cell path: the last executed cell that assigns a variable holding the answer and reads test_input
        var_order = sorted(row["vars"], key=lambda v: v != "answer")
        cells = [c for name, c, _ in executed_calls(messages) if name == "run_python"]
        for var in var_order[:3]:
            for src in reversed(cells):
                stores, loads = stored_names(src)
                if var in stores:
                    if "test_input" in loads:
                        res = cell_eval(s, src, var)
                        if res[-1] == ph:
                            row["cell_var"] = var
                            row["cell_train_ok"] = sum(a == b for a, b in zip(res[:-1], outs))
                            paths.append((row["cell_train_ok"], 2, "cell"))
                    break  # only the last assigning cell counts
            if "cell_var" in row:
                break
        if cands:
            best = max(cands, key=lambda f: sum(a == b for a, b in zip(f["preds"][:-1], outs)))
            row["fn_name"] = best["name"]
            row["fn_train_ok"] = sum(a == b for a, b in zip(best["preds"][:-1], outs))
            row["last_cand_train_ok"] = sum(a == b for a, b in zip(cands[-1]["preds"][:-1], outs))
            paths.append((row["fn_train_ok"], 1, "fn"))
        if paths:
            k, _, name = max(paths)
            row.update(trace=name, train_frac=k / n, train_pass=k == n, paths=[p[2] for p in paths])
            return row
    # transcript path: replay everything with test_input := each train input, read the same expression/var
    if code not in (None, "answer"):
        read_code = code
    elif row["vars"]:
        read_code = "answer" if "answer" in row["vars"] else row["vars"][0]
    else:
        read_code = None
    if transcript and read_code:
        res = transcript_eval(url, task, rec, messages, read_code)
        k = sum(a == b for a, b in zip(res, outs))
        row.update(trace="transcript", read_code=read_code[:100], train_frac=k / n, train_pass=k == n)
        return row
    row.update(trace="untraceable", train_pass=None, train_frac=None)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox-url", required=True)
    ap.add_argument("--runs", default="sub80s40,sub96n,s20x60,GL")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", help="label_sample.json: only these (task, pair) keys")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    for i in range(10):
        try:
            check_health(args.sandbox_url)
            break
        except Exception:
            time.sleep(15)
    else:
        sys.exit("sandbox unhealthy")
    tasks = load_tasks(ROOT / "data/arc-agi-2-github/data/evaluation")
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["run"], r["task_id"], r["pair"], r["attempt"]))
    jobs = []
    for run in args.runs.split(","):
        for r in load_run(RUNS[run]):
            if (run, r["task_id"], r["pair"], r["attempt"]) not in done:
                jobs.append((run, r))
    if args.sample:
        keys = {tuple(k) for v in json.load(open(args.sample))["sample"].values() for k in v}
        jobs = [j for j in jobs if (j[1]["task_id"], j[1]["pair"]) in keys]
    jobs = jobs[:args.limit]
    # longest traces first so the tail is short
    jobs.sort(key=lambda j: -(j[1].get("tool_calls") or 0))
    print(f"{len(jobs)} jobs -> {out}", flush=True)
    lock = threading.Lock()
    t0 = time.monotonic()
    count = [0]

    def work(job):
        run, rec = job
        t = time.monotonic()
        try:
            row = label_attempt(args.sandbox_url, run, rec, tasks[rec["task_id"]])
        except Exception as exc:
            row = {"run": run, "task_id": rec["task_id"], "pair": rec["pair"], "attempt": rec["attempt"],
                   "error": f"{type(exc).__name__}: {exc}"[:400]}
        row["seconds"] = round(time.monotonic() - t, 1)
        with lock:
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            count[0] += 1
            if count[0] % 25 == 0 or "error" in row:
                print(f"[{count[0]}/{len(jobs)} {time.monotonic() - t0:.0f}s] {run} {rec['task_id']}.{rec['pair']}"
                      f"#{rec['attempt']} {row.get('error') or (row['trace'], row['train_pass'], row['correct'])}",
                      flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, jobs))


if __name__ == "__main__":
    main()
