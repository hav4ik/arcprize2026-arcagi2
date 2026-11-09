"""E028: offline estimate of the solve(test_input) fallback on capped attempts.

For every attempt that ended at the output cap without an answer (stop_reason output_budget), rebuild its sandbox
session exactly as the harness did (same sandbox, train/test_input preloaded, arc_helpers prelude), replay every
run_python cell in order, then evaluate `solve(test_input)` with the harness's own submit_answer parser and score
the grid against the solution. Also tries the session's `answer` variable.

  .venv/bin/python -I experiments/E028_qwen38_sft/solve_fallback_rescue.py <results.jsonl> [...] --sandbox-url URL
"""
import argparse
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/vu/Workspace/ARC-AGI-2")
sys.path.insert(0, str(ROOT / "evals"))
sys.argv, _argv = [sys.argv[0]], sys.argv
import run_deepseek_tools as h  # noqa: E402
from sandbox_client import SandboxSession  # noqa: E402
sys.argv = _argv


def rescue(row, challenges, solutions, url):
    task = challenges[row["task_id"]]
    pair = int(row["pair"])
    test_input = task["test"][pair]["input"]
    truth = solutions[row["task_id"]][pair]
    msgs = json.load(gzip.open(row["trace_path"]))["messages"]
    cells = []
    for m in msgs:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                if c["function"]["name"] == "run_python":
                    try:
                        cells.append(json.loads(c["function"]["arguments"]).get("code", ""))
                    except (json.JSONDecodeError, TypeError):
                        pass
    has_solve = any("def solve" in c for c in cells)
    out = {"key": f"{row['task_id']}:{row['pair']}", "cells": len(cells), "has_solve": has_solve,
           "solve_grid": False, "solve_correct": False, "answer_correct": False}
    if not cells:
        return out
    with SandboxSession(url, task["train"], test_input, 30, 16000, prelude_extra=h.HELPERS_SRC) as s:
        for c in cells:
            try:
                s.execute(c)
            except Exception:
                pass
        if has_solve:
            _, grid = h.submit_answer(s, "solve(test_input)")
            out["solve_grid"] = grid is not None
            out["solve_correct"] = grid == truth
        _, agrid = h.submit_answer(s, "answer")
        out["answer_correct"] = agrid is not None and agrid == truth
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--sandbox-url", default="http://172.18.0.2:6000")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    challenges = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))
    solutions = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
    for f in a.results:
        rows = [json.loads(l) for l in open(f)]
        capped = [r for r in rows if r.get("stop_reason") == "output_budget" and not r.get("correct")]
        base_correct = sum(r.get("correct") is True for r in rows)
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda r: rescue(r, challenges, solutions, a.sandbox_url), capped))
        n_solve = sum(r["has_solve"] for r in res)
        n_grid = sum(r["solve_grid"] for r in res)
        n_ok = sum(r["solve_correct"] or r["answer_correct"] for r in res)
        print(f"\n{Path(f).name}: {len(rows)} attempts, {base_correct} correct; capped {len(capped)}: "
              f"def solve {n_solve}, solve() returned a grid {n_grid}, RESCUED (correct) {n_ok} -> "
              f"{base_correct + n_ok}/{len(rows)} = {100 * (base_correct + n_ok) / len(rows):.1f}%")
        for r in res:
            print("   ", json.dumps(r))


if __name__ == "__main__":
    main()
