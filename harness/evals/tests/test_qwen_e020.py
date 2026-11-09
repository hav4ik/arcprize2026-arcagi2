"""Tests for the E020 options of the Qwen tool harness (qwen-tools-5; no network, no sandbox).

    .venv/bin/python -m unittest evals/tests/test_qwen_e020.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_harness import agent_qwen as aq  # noqa: E402
from arc_harness.executor import (LocalExecutor, local_factory, program_feedback, run_program,  # noqa: E402
                                  score_outputs)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen_program import GOOD  # noqa: E402
from test_qwen_tools import ANSWER, TEST, TRAIN, call  # noqa: E402


class TokLLM:
    """Scripted completions (text, finish, tokens); tokens are capped at max_tokens like a real server."""

    def __init__(self, script):
        self.script, self.prompts = list(script), []

    async def complete(self, prompt, max_tokens, stop=None):
        self.prompts.append((prompt, max_tokens))
        text, finish, n = self.script.pop(0)
        return aq.Completion(text, min(n, max_tokens), len(prompt) // 4, finish)


def cfg(**kw):
    base = dict(tool_timeout=5, answer_mode="program", prompt="lean2prog", program_timeout=3)
    return aq.AgentConfig(**{**base, **kw})


def run(llm, c, helpers=False):
    make = local_factory(5, prelude_extra=aq.HELPERS_SRC) if helpers else local_factory(5)
    return asyncio.run(aq.solve(llm, make, TRAIN, TEST, c))


class ThinkCapTest(unittest.TestCase):
    def test_nudge_then_force_act(self):
        llm = TokLLM([
            ("long thoughts", "length", 10**6),            # cut at the cap
            ("more thoughts", "length", 10**6),            # cut at the grace
            ("print(1)\n</parameter>\n</function>\n</tool_call>", "stop", 20),  # the forced run_python code
            (call("submit_program", GOOD), "stop", 50),
        ])
        _, res = run(llm, cfg(turn_think_cap=1000, turn_think_grace=300))
        self.assertEqual(llm.prompts[0][1], 1000)
        self.assertEqual(llm.prompts[1][1], 300)
        self.assertTrue(llm.prompts[1][0].endswith("long thoughts" + aq.THINK_NUDGE))
        self.assertTrue(llm.prompts[2][0].endswith("more thoughts" + aq.THINK_FORCE_ACT))
        self.assertIn("<tool_response>\n[completed]\n1", llm.prompts[3][0])
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual((res["think_nudges"], res["think_forced"]), (1, 1))
        self.assertEqual(res["turn_log"][0]["think_nudge"], 1000)
        self.assertTrue(res["turn_log"][0]["think_forced"])

    def test_closes_after_nudge(self):
        llm = TokLLM([("t", "length", 10**6), (call("submit_program", GOOD, pre="ok"), "stop", 40)])
        _, res = run(llm, cfg(turn_think_cap=500))
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual((res["think_nudges"], res["think_forced"]), (1, 0))

    def test_off_by_default(self):
        llm = TokLLM([(call("submit_program", GOOD), "stop", 40)])
        _, res = run(llm, cfg())
        self.assertEqual(llm.prompts[0][1], 16384)
        self.assertNotIn("think_nudges", res)
        self.assertEqual(res["harness_version"], "qwen-tools-4")


HELPER_PROG = ("def solve(grid):\n    objs = objects(grid)\n    assert len(objs) >= 1\n"
               "    return [row[::-1] for row in grid]")


class HelpersTest(unittest.TestCase):
    def test_program_prelude(self):
        ex = LocalExecutor(TRAIN, TEST, timeout=20)
        with ex:
            self.assertEqual(run_program(ex, HELPER_PROG, [[1, 2]], prelude=aq.HELPERS_SRC), {"out": [[2, 1]]})
            self.assertEqual(run_program(ex, HELPER_PROG, [[1, 2]])["kind"], "NameError")

    def test_agent_with_helpers(self):
        llm = TokLLM([(call("run_python", "print(check(lambda g: [r[::-1] for r in g]))"), "stop", 30),
                      (call("submit_program", HELPER_PROG), "stop", 30)])
        _, res = run(llm, cfg(helpers=True), helpers=True)
        self.assertIn("ALL PASS", llm.prompts[1][0])
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual(res["harness_version"], "qwen-tools-5")
        prompt = llm.prompts[0][0]
        self.assertIn("Also preloaded, numpy grid helpers", prompt)
        self.assertIn(aq.HELPERS_PROGRAM_DOC.strip(), prompt)
        self.assertIn(aq.HELPERS_PROMPT.strip(), prompt)


class PromptTest(unittest.TestCase):
    def test_summary_and_roles(self):
        c = cfg(task_summary=True, prompt="lean2progroles")
        p = aq.initial_prompt(TRAIN, TEST, c)
        self.assertIn("A computed summary of the grids follows", p)
        self.assertIn("ROLES note", p)
        d = aq.config_dict(c, "m")
        self.assertEqual(d["harness_version"], "qwen-tools-5")
        self.assertTrue(d["task_summary"])
        self.assertNotIn("helpers", d)
        self.assertNotIn("turn_think_grace", d)


class ConfirmTestTest(unittest.TestCase):
    def test_confirm_then_accept(self):
        llm = TokLLM([(call("submit_program", GOOD), "stop", 30), (call("submit_program", GOOD), "stop", 30)])
        _, res = run(llm, cfg(confirm_test=True))
        self.assertIn("Its output for test_input (1x3):\n987", llm.prompts[1][0])
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual(res["first_tp_pred"], ANSWER)
        self.assertEqual((res["n_submissions"], res["answer_source"]), (2, "program_pass"))

    def test_budget_out_after_confirm_uses_shown_program(self):
        llm = TokLLM([(call("submit_program", GOOD), "stop", 30), ("thinking", "length", 10**6)])
        _, res = run(llm, cfg(confirm_test=True, max_total_output=1000, budget_soft=0))
        self.assertEqual(res["pred"], ANSWER)
        self.assertTrue(res["train_pass"])


class RichFeedbackTest(unittest.TestCase):
    def test_cells_and_shape(self):
        train = [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[1]], "output": [[1, 1], [1, 1]]}]
        rows = score_outputs([{"out": [[2, 2]]}, {"out": [[1]]}], train)
        text = program_feedback(rows, 1, 6, rich=True, train=train)
        self.assertIn("colour confusions (expected->got x count): 1->2 x1", text)
        self.assertIn("expected output:\n21\nyour output ('.' = correct cell, digit = your wrong value):\n.2", text)
        rows = score_outputs([{"out": [[2, 1]]}, {"out": [[1]]}], train)
        text = program_feedback(rows, 1, 6, rich=True, train=train)
        self.assertIn("shape hint: the expected height is 2x yours; the expected width is 2x yours", text)
        self.assertNotIn("colour confusions", program_feedback(rows, 1, 6))


class AuditLintTest(unittest.TestCase):
    def test_audit_flags(self):
        from arc_harness.audit import test_audit
        train = [{"input": [[1, 0], [0, 0]], "output": [[1, 1], [0, 0]]}, {"input": [[2, 0], [0, 0]], "output": [[2, 2], [0, 0]]}]
        flags = test_audit(train, [[3, 0], [0, 0]], [[3, 3, 3]], [0.1, 0.1, 5.0])
        text = " | ".join(flags)
        self.assertNotIn("colour(s) 3", text)  # in test_input: not flagged
        self.assertIn("every training output has its input's size", text)
        self.assertIn("solve took 5.0s", text)
        self.assertIn("identical to test_input", " ".join(test_audit(train, [[1, 0], [0, 0]], [[1, 0], [0, 0]])))
        self.assertEqual(test_audit(train, [[1, 0], [0, 0]], [[1, 1], [0, 0]]), [])

    def test_lint(self):
        from arc_harness.audit import lint
        code = ("def solve(g):\n    T = {(1,2): 3, (2,3): 4, (3,4): 5, (4,5): 6, (5,6): 7, (6,7): 8}\n"
                "    if g == [[1, 2]]:\n        return g\n    if len(g) == 13:\n        g[3][5] = 1\n"
                "    x = g[0][0] + g[-1][0]\n    return g")
        text = " | ".join(lint(code, TRAIN))
        for want in ("line 2: a literal table with 6 entries", "line 3: a comparison with a literal grid",
                     "line 3: a literal copy of a training grid", "line 5: a branch on an exact grid size (13)",
                     "line 6: a hardcoded cell index [3][5]"):
            self.assertIn(want, text)
        self.assertNotIn("[0][0]", text)
        self.assertEqual(lint(GOOD, TRAIN), [])

    def test_agent_audit_turn(self):
        llm = TokLLM([(call("submit_program", GOOD), "stop", 30), (call("submit_program", GOOD), "stop", 30)])
        _, res = run(llm, cfg(test_audit=True, lint=True))
        self.assertIn("review this host-side audit", llm.prompts[1][0])
        self.assertIn("test_input is outside the training range", llm.prompts[1][0])
        self.assertIn("Your output for test_input (1x3):\n987", llm.prompts[1][0])
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual(res["audit"]["lint"], [])
        self.assertEqual(len(res["audit"]["flags"]), 1)

    def test_lint_only_accepts_clean_program(self):
        llm = TokLLM([(call("submit_program", GOOD), "stop", 30)])
        _, res = run(llm, cfg(lint=True))
        self.assertEqual((res["pred"], res["n_submissions"]), (ANSWER, 1))


if __name__ == "__main__":
    unittest.main()


class SlotPoolTest(unittest.TestCase):
    def test_balance_cap_and_dead_leases(self):
        import os
        import tempfile
        from arc_harness.slots import SlotPool
        with tempfile.TemporaryDirectory() as d:
            pool = SlotPool(os.path.join(d, "s.json"), {"a": 2, "b": 1}, run="t")
            got = [pool.try_acquire() for _ in range(4)]
            self.assertEqual(sorted(u for _, u in got[:3]), ["a", "a", "b"])
            self.assertIsNone(got[3])
            pool.release(got[0][0])
            self.assertEqual(pool.try_acquire()[1], got[0][1])
            with pool._locked() as st:  # a lease of a dead process is dropped
                st["leases"]["x"] = {"url": "b", "pid": 2 ** 22 + 12345, "run": "dead", "t": 0}
                st["slots"]["b"] = 2
            self.assertEqual(pool.try_acquire()[1], "b")
            self.assertEqual(pool.show()["b"]["used"], 2)


class AlternativesTest(unittest.TestCase):
    def test_program_alternatives(self):
        from test_qwen_program import HALF
        llm = TokLLM([(call("submit_program", HALF), "stop", 30), (call("submit_program", GOOD), "stop", 30)])
        _, res = run(llm, cfg(record_alternatives=True))
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual(res["alternatives"], [{"src": "submission_0", "pred": [[5, 4, 3]], "train_frac": 0.5}])

    def test_submit_alternatives(self):
        code = "answer = [[1, 1, 1]]\ndef solve(g):\n    return [r[::-1] for r in g]"
        llm = TokLLM([(call("run_python", code), "stop", 30), (call("submit_answer", "solve(test_input)"), "stop", 30)])
        c = aq.AgentConfig(tool_timeout=5, record_alternatives=True)
        _, res = asyncio.run(aq.solve(llm, local_factory(5), TRAIN, TEST, c))
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual(res["alternatives"][0], {"src": "answer", "pred": [[1, 1, 1]], "train_frac": None})


class ThinkCallRescueTest(unittest.TestCase):
    CALL = ("ok I'll run it\n<tool_call>\n<function=submit_program>\n<parameter=code>\n" + GOOD
            + "\n</parameter>\n</function>\n</tool_call>")

    def test_rescue(self):
        llm = TokLLM([("thinking", "length", 10**6), (self.CALL, "stop", 50)])
        _, res = run(llm, cfg(turn_think_cap=500, think_call_rescue=True))
        self.assertEqual((res["pred"], res["think_call_rescued"]), (ANSWER, 1))

    def test_without_rescue_the_call_is_lost(self):
        llm = TokLLM([("thinking", "length", 10**6), (self.CALL, "stop", 50)] + [("x", "stop", 5)] * 6)
        _, res = run(llm, cfg(turn_think_cap=500))
        self.assertEqual((res["pred"], res["stop_reason"]), (None, "unclosed_think"))
