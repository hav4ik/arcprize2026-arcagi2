#!/usr/bin/env python3
"""E042: re-label finished attempts offline with evals/train_check.py, with and without reject_constant.

For each submitted attempt in a results jsonl: rebuild its session from the trace (every run_python call in order,
in a LocalExecutor like the Kaggle solver's), then run train_check twice: as recorded (reject_constant=False) and
with input-independent paths rejected (True). Prints how labels move against correctness, and writes per-attempt
labels for deadline_sim.py. No model calls.
Usage: train_check_recheck.py <results.jsonl> <challenges.json|dir> --out labels.jsonl [--tool-python P] [--workers 8]
"""
import argparse
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
import run_deepseek_tools as h  # noqa: E402
from arc_harness.executor import LocalExecutor  # noqa: E402
from train_check import train_check  # noqa: E402


def calls(trace):
    out = []
    for m in json.load(gzip.open(trace))["messages"]:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            try:
                a = json.loads(args) if isinstance(args, str) else (args or {})
            except json.JSONDecodeError:
                continue
            code = a.get("code") if fn.get("name") == "run_python" else (a.get("code") or a.get("expr"))
            if fn.get("name") in ("run_python", "submit_answer") and isinstance(code, str):
                out.append((fn["name"], code))
    return out


TOOL_PYTHON = sys.executable


def factory(train, test_input):
    return LocalExecutor(train, test_input, 30, 16000, python=TOOL_PYTHON, style="ipython", cpu_seconds=1800)


def relabel(rec, task, trace_dir):
    t, p, a = rec["task_id"], rec["pair"], rec["attempt"]
    traces = sorted(Path(trace_dir).glob(f"{t}_p{p}_a{a}_*.messages.json.gz"))
    if not traces:
        return {**key(rec), "error": "no trace"}
    executed = calls(traces[0])
    test_input = task["test"][p]["input"]
    code = rec.get("submit_code") or next((c for n, c in reversed(executed) if n == "submit_answer"), "answer")
    out = {**key(rec), "correct": bool(rec.get("correct")), "old_recorded": rec.get("train_pass")}
    try:
        with factory(task["train"], test_input) as s:
            for name, c in executed:
                if name == "run_python":
                    s.execute(c)
            for flag in (False, True):
                lab = train_check(s, executed, rec["pred"], code, submit=h.submit_answer,
                                  replay_session=lambda grid: factory(task["train"], grid), reject_constant=flag,
                                  transcript_fallback=flag)
                out["new" if flag else "old"] = {k: lab.get(k) for k in ("train_pass", "train_frac", "train_trace")}
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def key(rec):
    return {"task_id": rec["task_id"], "pair": rec["pair"], "attempt": rec["attempt"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("challenges")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-false", action="store_true", help="only attempts recorded with train_pass False")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tool-python", default=sys.executable, help="interpreter for tool code (Kaggle's has scipy, skimage)")
    a = ap.parse_args()
    global TOOL_PYTHON
    TOOL_PYTHON = a.tool_python
    c = Path(a.challenges)
    tasks = ({f.stem: json.load(open(f)) for f in c.glob("*.json")} if c.is_dir() else json.load(open(c)))
    recs = [r for r in map(json.loads, open(a.results)) if "task_id" in r and r.get("pred") is not None
            and r.get("answer_source") == "submit" and (not a.only_false or r.get("train_pass") is False)]
    trace_dir = a.results[: -len(".jsonl")]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda r: relabel(r, tasks[r["task_id"]], trace_dir), recs))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if "error" not in r]
    print(f"{len(rows)} attempts, {len(rows) - len(ok)} errors")
    import collections
    c = collections.Counter((str(r["old"]["train_pass"]), str(r["new"]["train_pass"]), r["correct"]) for r in ok)
    print("old -> new label, correct: count")
    for k, v in sorted(c.items()):
        print(f"  {k[0]:5s} -> {k[1]:5s} correct={k[2]!s:5s}: {v}")
    agree = sum(str(r["old"]["train_pass"]) == str(r["old_recorded"]) for r in ok)
    print(f"replayed old label equals the recorded one: {agree}/{len(ok)}")


if __name__ == "__main__":
    main()
