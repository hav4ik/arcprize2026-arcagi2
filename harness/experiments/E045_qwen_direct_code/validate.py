"""E045: extract each sample's program, run it on every training input and the test inputs, record the results.

Programs run in the Docker sandbox (evals/sandbox_docker: no network, no host mounts), one fresh Python process
per program with a per-grid timeout and a memory cap. Training outputs are compared inside that process (they
are in the prompt anyway); test outputs come back to the host, which alone holds the solutions (score.py).
Identical programs within a task are run once. Re-running resumes from --results.

Usage: validate.py --samples samples.jsonl --results programs.jsonl --sandbox-url http://172.18.0.5:6000
"""
import argparse
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evals"))
from sandbox_client import SandboxSession  # noqa: E402

CHILD = r'''
import contextlib, io, json, resource, signal, sys
req = json.loads(sys.stdin.read())
try:
    resource.setrlimit(resource.RLIMIT_AS, (req["mem"], req["mem"]))
except (ValueError, OSError):
    pass
class _T(BaseException):
    pass
def _alarm(*_):
    raise _T()
def _plain(o):
    return o.tolist() if hasattr(o, "tolist") else int(o)
signal.signal(signal.SIGALRM, _alarm)
res = {"compile": None, "train_ok": [], "train_err": [], "test": []}
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        signal.setitimer(signal.ITIMER_REAL, req["timeout"])
        import numpy as np
        ns = {"__name__": "__arc_program__", "np": np}
        exec(compile(req["code"], "<program>", "exec"), ns)
        signal.setitimer(signal.ITIMER_REAL, 0)
        fn = ns.get("solve")
        if not callable(fn):
            raise NameError("no solve(grid)")
    except BaseException as exc:
        res["compile"] = type(exc).__name__
        fn = None
    def run(grid):
        try:
            signal.setitimer(signal.ITIMER_REAL, req["timeout"])
            v = fn([row[:] for row in grid])
            signal.setitimer(signal.ITIMER_REAL, 0)
            return json.loads(json.dumps(v, default=_plain)), None
        except _T:
            return None, "timeout"
        except BaseException as exc:
            signal.setitimer(signal.ITIMER_REAL, 0)
            return None, type(exc).__name__
    if fn is not None:
        for pair in req["train"]:
            out, err = run(pair["input"])
            res["train_ok"].append(err is None and out == pair["output"])
            res["train_err"].append(err)
        if all(res["train_ok"]):
            for grid in req["test"]:
                out, err = run(grid)
                res["test"].append(out if err is None else {"error": err})
print(req["marker"] + json.dumps(res))
'''

CELL = r'''
def _e045():
    import os, shutil, subprocess, sys
    exe = sys.executable
    if not os.path.basename(exe or "").startswith("python"):
        exe = shutil.which("python3") or shutil.which("python") or exe
    try:
        r = subprocess.run([exe, "-I", "-c", CHILD], input=REQ, capture_output=True, text=True,
                           timeout=LIMIT, cwd="/tmp")
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ""
    for line in out.splitlines()[::-1]:
        if line.startswith(MARKER):
            print(line)
            return
    print(MARKER + '{"crash": true}')
try:
    _e045()
finally:
    del _e045
'''


def extract(text, arm):
    """(code or None, rule or None)"""
    text = text or ""
    rule = None
    if arm != "code":
        m = re.search(r"<rule>(.*?)</rule>", text, re.S)
        rule = m.group(1).strip() if m else None
        m = re.search(r"<code>(.*?)(?:</code>|$)", text, re.S)
        if m:
            body = m.group(1)
            fence = re.findall(r"```(?:python|py)?\s*\n(.*?)```", body, re.S)
            code = fence[-1] if fence else body
            if "def solve" in code:
                return code.strip(), rule
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    for b in reversed(blocks):
        if "def solve" in b:
            return b.strip(), rule
    if "def solve" in text and "```" not in text:
        return text.strip(), rule
    return None, rule


def code_id(task, code):
    return hashlib.sha1((task + "\0" + code).encode()).hexdigest()[:20]


_local = threading.local()


def session(url):
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = SandboxSession(url, [], [[0]], timeout=300, max_output=400000)
    return s


def run_one(url, task, code, timeout, mem_mb):
    marker = f"E045_{uuid4().hex}:"
    n_grids = len(task["train"]) + len(task["test"])
    req = json.dumps({"code": code, "train": task["train"], "test": [t["input"] for t in task["test"]],
                      "timeout": timeout, "mem": mem_mb * 2 ** 20, "marker": marker})
    limit = timeout * (n_grids + 1) + 15
    cell = (f"CHILD = {CHILD!r}\nREQ = {req!r}\nMARKER = {marker!r}\nLIMIT = {limit!r}\n" + CELL)
    for attempt in range(2):
        try:
            r = session(url).execute(cell)
            break
        except Exception as exc:
            _local.s = None
            if attempt:
                return {"crash": True, "runner": f"{type(exc).__name__}"}
    for line in reversed(r.get("stdout", "").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return {"crash": True, "runner": r.get("process_status")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--sandbox-url", required=True)
    ap.add_argument("--challenges", default=str(ROOT / "data/arc-agi_evaluation_challenges.json"))
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds per grid")
    ap.add_argument("--mem-mb", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    tasks = json.loads(Path(args.challenges).read_text())
    done = set()
    if args.results.exists():
        done = {json.loads(l)["id"] for l in args.results.open() if l.strip()}
    todo = {}
    for line in args.samples.open():
        rec = json.loads(line)
        for s in rec["samples"]:
            code, _ = extract(s["text"], rec["arm"])
            if code:
                cid = code_id(rec["task"], code)
                if cid not in done:
                    todo[cid] = (rec["task"], code)
    print(f"{len(todo)} unique programs to run ({len(done)} done)", flush=True)
    lock, n = threading.Lock(), 0
    with ThreadPoolExecutor(args.workers) as pool, args.results.open("a") as f:
        futs = {pool.submit(run_one, args.sandbox_url, tasks[t], c, args.timeout, args.mem_mb): (cid, t)
                for cid, (t, c) in todo.items()}
        for fut in as_completed(futs):
            cid, t = futs[fut]
            res = fut.result()
            with lock:
                f.write(json.dumps({"id": cid, "task": t, **res}) + "\n")
                f.flush()
                n += 1
                if n % 500 == 0:
                    print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
