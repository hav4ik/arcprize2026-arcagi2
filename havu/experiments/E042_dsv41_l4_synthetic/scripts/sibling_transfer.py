#!/usr/bin/env python3
"""E042: would a finished attempt's answer program also answer the OTHER test inputs of its task?

The private test set has ~335 test pairs over 240 tasks (vs 259 in the public file): ~45% of tasks have 2-3 test
inputs, and the solver runs separate attempts per pair. For each submitted attempt of a multi-pair task: rebuild
its session from the trace (as train_check_recheck.py), take the train check's answer path (expr / cell / fn /
transcript, chosen by train score then priority as in evals/train_check.py), and run it on every sibling test
input. Writes per-attempt rows {task_id, pair, attempt, train_pass, path, sib: {pair: correct}}. No model calls.
Usage: sibling_transfer.py <results.jsonl> <challenges dir with test outputs> --out rows.jsonl [--tool-python P]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_check_recheck as rc  # noqa: E402
from train_check import cell_eval, grid_hash, probe, stored_names  # noqa: E402


class View:
    """The attempt's session, with the sibling test inputs appended to the inputs the train-check cells see."""

    def __init__(self, s, extra):
        self.s, self.train, self.test_input = s, list(s.train) + [{"input": g} for g in extra], s.test_input

    def __getattr__(self, k):
        return getattr(self.s, k)

    def __setattr__(self, k, v):
        if k in ("s", "train", "test_input"):
            object.__setattr__(self, k, v)
        else:
            setattr(self.s, k, v)


def paths(s, executed, pred, code, sib_inputs, replay):
    """[(train score, priority, name, sibling hashes)] for paths that reproduce the test prediction."""
    n, ph = len(s.train), grid_hash(pred)
    outs = [grid_hash(p["output"]) for p in s.train]
    v = View(s, sib_inputs)
    res = []

    def add(preds, pri, name):  # preds: train..., siblings..., test
        if preds[-1] == ph:
            res.append((sum(a == b for a, b in zip(preds[:n], outs)), pri, name, preds[n:-1]))

    data = probe(v, code, ph)
    if data.get("expr") is not None:
        add(data["expr"], 3, "expr")
    cells = [c for name, c in executed if name == "run_python"]
    for var in sorted(data.get("vars", []), key=lambda x: x != "answer")[:3]:
        for src in reversed(cells):
            stores, loads = stored_names(src)
            if var in stores:
                if "test_input" in loads:
                    add(cell_eval(v, src, var, ph), 2, "cell")  # inputs order: train..., siblings..., test
                break
        if any(r[2] == "cell" for r in res):
            break
    for f in probe(v, None, ph, do_expr=False, do_fns=True).get("fns", []):
        add(f["preds"], 1, "fn:" + f["name"])
    if not res and replay is not None:
        res.append(replay())
    return res


def run(rec, task, trace_dir):
    t, p, a = rec["task_id"], rec["pair"], rec["attempt"]
    sibs = [q for q in range(len(task["test"])) if q != p]
    traces = sorted(Path(trace_dir).glob(f"{t}_p{p}_a{a}_*.messages.json.gz"))
    out = {"task_id": t, "pair": p, "attempt": a, "correct": bool(rec.get("correct")),
           "train_pass": rec.get("train_pass"), "train_trace": rec.get("train_trace")}
    if not traces:
        return {**out, "error": "no trace"}
    executed = rc.calls(traces[0])
    code = rec.get("submit_code") or next((c for n, c in reversed(executed) if n == "submit_answer"), "answer")
    sib_in = [task["test"][q]["input"] for q in sibs]
    sol = [grid_hash(task["test"][q]["output"]) for q in sibs]

    def replay():  # transcript path: replay every cell in a fresh session per input, read the answer the same way
        read = code if code not in (None, "answer") else "answer"
        preds = []
        for grid in [x["input"] for x in task["train"]] + sib_in:
            with rc.factory(task["train"], grid) as s2:
                for name, c in executed:
                    if name == "run_python":
                        s2.execute(c)
                    elif name == "submit_answer":
                        rc.h.submit_answer(s2, c)
                _, g = rc.h.submit_answer(s2, read)
                preds.append(grid_hash(g) if g is not None else "ERR")
        outs = [grid_hash(x["output"]) for x in task["train"]]
        return (sum(x == y for x, y in zip(preds, outs)), 0, "transcript", preds[len(outs):])

    try:
        with rc.factory(task["train"], task["test"][p]["input"]) as s:
            for name, c in executed:
                if name == "run_python":
                    s.execute(c)
            ps = paths(s, executed, rec["pred"], code, sib_in, replay if len(executed) <= 120 else None)
    except Exception as exc:
        return {**out, "error": f"{type(exc).__name__}: {exc}"[:300]}
    if not ps:
        return {**out, "path": None, "sib": {}}
    k, _, name, sp = max(ps, key=lambda r: (r[0], r[1]))
    return {**out, "path": name, "path_train_frac": round(k / len(task["train"]), 4),
            "sib": {str(q): h == o for q, h, o in zip(sibs, sp, sol)}, "sib_hash": dict(zip(map(str, sibs), sp)),
            "sib_any": {str(q): any(r[3][j] == sol[j] for r in ps) for j, q in enumerate(sibs)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("challenges")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tool-python", default=sys.executable)
    a = ap.parse_args()
    rc.TOOL_PYTHON = a.tool_python
    tasks = {f.stem: json.load(open(f)) for f in Path(a.challenges).glob("*.json")}
    recs = [r for r in map(json.loads, open(a.results)) if "task_id" in r and r.get("pred") is not None
            and r.get("answer_source") == "submit" and len(tasks[r["task_id"]]["test"]) > 1]
    trace_dir = a.results[: -len(".jsonl")]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda r: run(r, tasks[r["task_id"]], trace_dir), recs))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} attempts, {sum('error' in r for r in rows)} errors")


if __name__ == "__main__":
    main()
