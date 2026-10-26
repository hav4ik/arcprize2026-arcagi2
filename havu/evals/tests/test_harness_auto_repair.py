"""I5 --auto-repair (kaggle/dsv41_tools/next_candidate_fixes.md): evals/auto_repair.py wired into run_python cells of
the harness. A fake executor (repaired NameError with the note, a refused repair, flag off = unchanged behaviour), a
real LocalExecutor session, and the solver's ARC_AUTO_REPAIR."""
import re
import unittest
from unittest.mock import patch

import dsv41_fakes as C

H = C.H
TASK = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}
TYPO = "comps = [1, 2]\nfor c in coms:\n    print(c)\n"
ERR = ("Traceback (most recent call last):\n    for c in coms:\nNameError: name 'coms' is not defined\n")


class FakeExecutor:
    """Runs nothing: a cell that uses the name `coms` fails with the sandbox's NameError text, the harness's names
    query gets `names`, every other cell completes and prints `printed`."""
    instances = []

    def __init__(self, url, train, test_input, timeout=30, max_output=16000, prelude_extra="", names=("comps",)):
        self.train, self.test_input, self.timeout, self.max_output = train, test_input, timeout, max_output
        self.calls, self.names = [], list(names)
        FakeExecutor.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def execute(self, code):
        self.calls.append(code)
        m = re.search(r"ARC_NAMES_[0-9a-f]+:", code)
        if m:
            return {"process_status": "completed", "stdout": m.group(0) + str(self.names).replace("'", '"') + "\n",
                    "stderr": ""}
        if re.search(r"\bcoms\b", code):
            return {"process_status": "error", "stdout": ERR, "stderr": ""}
        return {"process_status": "completed", "stdout": "1\n2\n", "stderr": ""}


def run(code, flag, names=("comps",)):
    FakeExecutor.instances = []
    srv = C.FakeTokenServer([C.ds_call_text(code)], completion_tokens=10 ** 6)  # one turn, then the budget ends
    args = C.harness_args(["--auto-repair"] if flag else [])
    with patch("httpx.post", srv.post):
        trace, result = H.solve(srv, args, TASK, TASK["test"][0]["input"],
                                session_factory=lambda *a, **k: FakeExecutor(*a, **k, names=names))
    tool = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
    return trace, result, tool, FakeExecutor.instances[0]


class AutoRepair(unittest.TestCase):
    def test_typo_repaired_with_note(self):
        trace, result, tool, ex = run(TYPO, True)
        self.assertTrue(tool[0].startswith("[auto-corrected: coms -> comps]\n[completed]\n1\n2\n"), tool[0])
        self.assertEqual(result["auto_repairs"], {"proposed": 1, "applied": 1, "retry_ok": 1})
        self.assertEqual(ex.calls[0], TYPO)
        self.assertEqual(ex.calls[2], TYPO.replace("coms", "comps"))  # [1] is the names query
        self.assertIn({"turn": 1, "auto_repair": "completed"}, trace["events"])
        tool_events = [e for e in trace["events"] if "tool_call_id" in e]
        self.assertEqual(tool_events[0]["status"], "completed")
        self.assertEqual(result["tool_errors"], 0)
        # the model's own call is unchanged in the transcript
        self.assertIn("coms", trace["messages"][2]["tool_calls"][0]["function"]["arguments"])

    def test_refused_repair_keeps_the_error(self):
        # the part of the cell that ran mutated a session object: no repair
        code = "res.append(1)\nprint(coms)\n"
        trace, result, tool, ex = run(code, True, names=("comps", "res"))
        self.assertEqual(tool[0].split("\n[budget")[0], H.tool_output({"process_status": "error", "stdout": ERR}))
        self.assertEqual(result["auto_repairs"], {"proposed": 0, "applied": 0, "retry_ok": 0})
        self.assertEqual(len([c for c in ex.calls if "coms" in c and "_arc" not in c]), 1)  # never re-run
        self.assertEqual(result["tool_errors"], 1)

    def test_flag_off_unchanged(self):
        with patch.dict("sys.modules", {"auto_repair": None}):  # importing it would fail: never touched
            trace, result, tool, ex = run(TYPO, False)
        self.assertEqual(tool[0].split("\n[budget")[0], H.tool_output({"process_status": "error", "stdout": ERR}))
        self.assertNotIn("auto_repairs", result)
        self.assertEqual(ex.calls[0], TYPO)
        self.assertFalse(any("ARC_NAMES_" in c for c in ex.calls))
        self.assertFalse(any("auto_repair" in e for e in trace["events"]))
        args = C.harness_args(["--auto-repair"])
        self.assertEqual(H.harness_version(args), H.harness_version(C.harness_args([])))
        self.assertNotIn("auto_repair", {**H.V4_DEFAULTS, **H.V5_DEFAULTS, **H.V6_DEFAULTS})  # not in the config

    def test_module_missing(self):
        with patch.dict("sys.modules", {"auto_repair": None}):
            trace, result, tool, ex = run(TYPO, True)
        self.assertTrue(result["auto_repairs"]["unavailable"].startswith(("ImportError", "ModuleNotFoundError")))
        self.assertTrue(tool[0].startswith("[error]"))

    def test_real_session_across_cells(self):
        srv = C.FakeTokenServer([C.ds_call_text("comps = [1, 2]"), C.ds_call_text("print(len(coms))")],
                                completion_tokens=50000)
        args = C.harness_args(["--auto-repair", "--max-total-output", "100000"])
        with patch("httpx.post", srv.post):
            trace, result = H.solve(srv, args, TASK, TASK["test"][0]["input"], session_factory=C.local_factory)
        tool = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertTrue(tool[1].startswith("[auto-corrected: coms -> comps]\n[completed]\n2\n"), tool[1])
        self.assertEqual(result["auto_repairs"], {"proposed": 1, "applied": 1, "retry_ok": 1})

    def test_solver_env(self):
        with patch.object(H, "budget_note", H.budget_note), patch.object(H, "train_check", H.train_check):
            # (make_args wraps these two harness functions; restored afterwards)
            self.assertTrue(C.load_solver_module({"ARC_AUTO_REPAIR": "1"}, "solver_ar").make_args().auto_repair)
            self.assertFalse(C.load_solver_module({}, "solver_noar").make_args().auto_repair)


if __name__ == "__main__":
    unittest.main()
