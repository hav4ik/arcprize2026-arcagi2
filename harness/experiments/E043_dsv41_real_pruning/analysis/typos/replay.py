#!/usr/bin/env python3
"""Step 6: replay validation of the "retry once with the unique close match" repair.

For each attempt with at least one identifier error, the attempt's run_python calls are re-executed in order in a
fresh local session (evals/arc_harness/executor.py child code, plain style, rlimits, cwd = a scratch directory).
At every identifier-error call: the original code is executed (its replay status is compared with the recorded one),
then the repair is applied as a harness would do it after the error:
    unknown name / attribute / module / imported name  ->  unique candidate within OSA distance 2 (1 for <= 4 chars)
    candidates: names bound in the session so far (static) + preloaded + builtins (NameError); the attributes of the
    type named in the error message (AttributeError on an object); dir(module) (module attribute / from-import);
    importable modules (ModuleNotFoundError)
and the rewritten code is executed in the same session; up to 3 successive rewrites (a second typo in the same cell).
The replay then continues with the model's original next calls.

Python: miniconda python3 (numpy 2.1 / scipy 1.17; the eval sandbox was Python 3.10 with numpy 2.x).
Output: out/replay_<model>.pkl
"""
import json
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/evals")
from common import OUT  # noqa: E402
from ident import (BUILTINS, MODULE_ATTRS, MODULE_NAMES, PRELOADED, TYPE_ATTRS, TYPE_MAP, SCIPY_SUBMODULES,  # noqa: E402
                   loose_bound, parse_ident_error)
from arc_harness.executor import LocalExecutor, _CHILD  # noqa: E402
from textutil import osa  # noqa: E402

PY = "/home/vu/miniconda3/bin/python3"
SCRATCH = os.environ.get("REPLAY_CWD", "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/"
                         "5b3f7cae-743a-4aac-ba8b-b255a5f73280/scratchpad/replay_cwd")
DATA = json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_evaluation_challenges.json"))
DATA2 = {}
_d = "/home/vu/Workspace/ARC-AGI-2/data/arc-agi-2-github/data/evaluation"
if os.path.isdir(_d):
    for f in os.listdir(_d):
        DATA2[f[:-5]] = json.load(open(os.path.join(_d, f)))


class ScratchExecutor(LocalExecutor):
    def _spawn_once(self, timeout):
        r, w = os.pipe()
        self.proc = subprocess.Popen(
            [self.python, "-I", "-u", "-c", _CHILD, str(w), str(self.max_output), str(self.memory_mb * 2 ** 20),
             str(int(self.cpu_seconds)), str(self.fsize_mb * 2 ** 20), str(self.nice)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(w,),
            start_new_session=True, text=True, cwd=SCRATCH,
            env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1"})
        os.close(w)
        self.resp, self.buf = r, b""
        return self._run(self.prelude, timeout)


def unique_close(word, cands, prefix_ok=False, cell_names=()):
    """Unique best candidate: min OSA distance (<= 1 for <= 4 chars, else <= 2), then prefer candidates the typo is a
    deletion of (longer), then the longest common prefix, then names used in the current cell. With prefix_ok, a
    unique candidate that starts with the typo also qualifies (keyword arguments: d -> dtype)."""
    maxd = 1 if len(word) <= 4 else 2
    scored = []
    for c in cands:
        if c == word or len(c) < 2:
            continue
        if prefix_ok and c.startswith(word):
            scored.append((0, 0, -len(word), 0, c))
            continue
        if abs(len(c) - len(word)) > maxd:
            continue
        d = osa(word, c)
        if d <= maxd:
            cp = 0
            while cp < min(len(c), len(word)) and c[cp] == word[cp]:
                cp += 1
            scored.append((d, 0 if len(c) > len(word) else 1, -cp, 0 if c in cell_names else 1, c))
    if not scored:
        return None
    scored.sort()
    best = scored[0]
    ties = [s for s in scored if s[:4] == best[:4]]
    return best[4] if len(ties) == 1 else None


NP_KW = {"zeros": ["shape", "dtype", "order", "like"], "ones": ["shape", "dtype", "order", "like"],
         "empty": ["shape", "dtype", "order", "like"], "full": ["shape", "fill_value", "dtype", "order", "like"],
         "zeros_like": ["a", "dtype", "order", "subok", "shape"], "ones_like": ["a", "dtype", "order", "subok", "shape"],
         "full_like": ["a", "fill_value", "dtype", "order", "subok", "shape"],
         "array": ["object", "dtype", "copy", "order", "subok", "ndmin", "like"],
         "asarray": ["a", "dtype", "order", "like"], "arange": ["start", "stop", "step", "dtype", "like"],
         "sort": ["key", "reverse"], "sorted": ["key", "reverse"], "label": ["input", "structure", "output"],
         "unique": ["ar", "return_index", "return_inverse", "return_counts", "axis"],
         "set_printoptions": ["precision", "threshold", "edgeitems", "linewidth", "suppress", "nanstr", "infstr",
                              "formatter", "sign", "floatmode", "legacy"],
         "argwhere": ["a"], "where": ["condition", "x", "y"], "max": ["key", "default"], "min": ["key", "default"]}


def propose(code, exc_type, msg, session):
    ie = parse_ident_error(exc_type, msg)
    if not ie:
        return None
    typo, kind, extra = ie
    if kind == "name":
        cands = session | loose_bound(code) | PRELOADED | BUILTINS
    elif kind == "attr" and extra[0] == "type":
        t = TYPE_MAP.get(extra[1])
        if t is None:
            return None
        cands = TYPE_ATTRS[t]
    elif kind == "attr":
        mod = {"np": "numpy"}.get(extra[1], extra[1])
        cands = MODULE_ATTRS.get(mod, set())
    elif kind == "import":
        cands = MODULE_ATTRS.get(extra[1], set()) | (SCIPY_SUBMODULES if extra[1] == "scipy" else set())
    elif kind == "kwarg":
        m = re.match(r"(\w+)\(\) got an unexpected", msg or "") or re.search(r"argument for (\w+)\(\)", msg or "")
        fn = m.group(1) if m else None
        if fn not in NP_KW:
            return None
        r = unique_close(typo, NP_KW[fn], prefix_ok=True)
        if not r:
            return None
        new = re.sub(r"(?<![\w.])" + re.escape(typo) + r"(?=\s*=)", r, code)
        return (typo, r, kind, new) if new != code else None
    elif kind == "module":
        parts = typo.split(".")
        base = ".".join(parts[:-1])
        cands = ({m for m in MODULE_NAMES if "." not in m} if not base else
                 {m.split(".")[-1] for m in MODULE_NAMES if m.startswith(base + ".")} |
                 (SCIPY_SUBMODULES if base == "scipy" else set()))
        typo = parts[-1]
    else:
        return None
    cell = set(re.findall(r"[A-Za-z_]\w*", code))
    r = unique_close(typo, cands, cell_names=cell)
    if not r:
        return None
    pat = (r"(?<=\.)" if kind == "attr" else r"(?<![\w.])" if kind == "name" else r"(?<![\w])") + re.escape(typo) + r"(?!\w)"
    new = re.sub(pat, r, code)
    return (typo, r, kind, new) if new != code else None


TRAIN_DATA = None


def task_io(tid, pair):
    global TRAIN_DATA
    t = DATA.get(tid) or DATA2.get(tid)
    if t is None:
        if TRAIN_DATA is None:
            TRAIN_DATA = json.load(open("/home/vu/Workspace/ARC-AGI-2/data/arc-agi_training_challenges.json"))
        t = TRAIN_DATA[tid]
    return t["train"], t["test"][pair]["input"]


def replay_attempt(a):
    seq = [(ti, c) for ti, t in enumerate(a["turns"]) for c in t["calls"]]
    err_idx = [i for i, (ti, c) in enumerate(seq) if c["name"] == "run_python" and c["status"] == "error"
               and parse_ident_error(c["exc_type"], c["exc_msg"])]
    if not err_idx:
        return []
    last = err_idx[-1]
    train, test_input = task_io(a["task_id"], a["pair"])
    rows = []
    session = set()
    t0 = time.time()
    try:
        ex = ScratchExecutor(train, test_input, timeout=30, max_output=4000, memory_mb=3072, python=PY, nice=15)
        ex.__enter__()
    except Exception as e:
        return [{"key": (a["task_id"], a["pair"], a["attempt"]), "fail": repr(e)}]
    try:
        for i, (ti, c) in enumerate(seq[: last + 1]):
            if c["name"] != "run_python":
                continue
            if time.time() - t0 > 900:
                rows.append({"key": (a["task_id"], a["pair"], a["attempt"]), "fail": "attempt time cap"})
                break
            code = c["code"] or ""
            st, out = run(ex, code)
            if i in err_idx:
                row = {"key": (a["task_id"], a["pair"], a["attempt"]), "call_seq": i, "turn": ti,
                       "rec_exc": c["exc_type"], "rec_msg": c["exc_msg"], "rep_status": st,
                       "rep_exc": exc_of(out) if st == "error" else None}
                row["replay_matches"] = (st == "error" and row["rep_exc"] == c["exc_type"])
                # retry with up to 3 successive rewrites
                cur, msg_t, msg = code, c["exc_type"], c["exc_msg"]
                if not row["replay_matches"]:
                    msg_t, msg = (row["rep_exc"], exc_msg_of(out)) if st == "error" else (None, None)
                rewrites = []
                final = None
                for _ in range(3):
                    if msg_t is None:
                        break
                    p = propose(cur, msg_t, msg, session)
                    if not p:
                        break
                    rewrites.append(p[:3])
                    cur = p[3]
                    st2, out2 = run(ex, cur)
                    final = (st2, exc_of(out2) if st2 == "error" else None)
                    if st2 != "error":
                        break
                    msg_t, msg = exc_of(out2), exc_msg_of(out2)
                row["rewrites"] = rewrites
                row["final"] = final
                rows.append(row)
            session |= loose_bound(code)
    finally:
        try:
            ex.__exit__(None, None, None)
        except Exception:
            pass
    return rows


EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))(?::\s?(.*))?$")


def exc_of(out):
    for l in reversed([l for l in (out or "").split("\n") if l.strip()]):
        m = EXC.match(l.strip())
        if m:
            return m.group(1)
    return None


def exc_msg_of(out):
    for l in reversed([l for l in (out or "").split("\n") if l.strip()]):
        m = EXC.match(l.strip())
        if m:
            return m.group(2) or ""
    return ""


def run(ex, code):
    try:
        r = ex.execute(code)
    except Exception as e:
        return "harness_error", repr(e)
    st = r.get("process_status") if isinstance(r, dict) else None
    out = (r.get("stdout", "") + "\n" + r.get("stderr", "")) if isinstance(r, dict) else str(r)
    return st, out


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    for model in sys.argv[1:]:
        A = pickle.load(open(f"{OUT}/attempts_{model}.pkl", "rb"))
        A = [a for a in A if any(c["name"] == "run_python" and c["status"] == "error" and
                                 parse_ident_error(c["exc_type"], c["exc_msg"]) for t in a["turns"] for c in t["calls"])]
        with ProcessPoolExecutor(12) as pool:
            rows = [r for rr in pool.map(replay_attempt, A, chunksize=1) for r in rr]
        pickle.dump(rows, open(f"{OUT}/replay_{model}.pkl", "wb"))
        ok = [r for r in rows if "fail" not in r]
        print(model, len(A), "attempts,", len(ok), "identifier-error calls replayed;",
              sum(r["replay_matches"] for r in ok), "reproduce the recorded exception;",
              sum(1 for r in ok if r["final"] and r["final"][0] == "completed"), "complete after the retry", flush=True)


if __name__ == "__main__":
    main()
