"""Tests for the E028 e025 mode of the Qwen tool agent (prompt lean2roles, tool_desc="e025", solve_fallback) and
LocalExecutor(style="ipython"). No network, no sandbox; the DeepSeek comparisons import run_deepseek_tools.py.

    .venv/bin/python -m unittest evals/tests/test_qwen_e025.py
"""
import argparse
import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_harness import agent_qwen as aq  # noqa: E402
from arc_harness.executor import LocalExecutor, local_factory, tool_output  # noqa: E402
from test_qwen_tools import ANSWER, TEST, TRAIN, FakeLLM, call  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
E025_TOOLS = ROOT / "experiments/E028_qwen38_sft/e025_tools.json"


def e025_cfg(**kw):
    return aq.AgentConfig(prompt="lean2roles", helpers=True, task_summary=False, tool_desc="e025", tool_timeout=5,
                          max_total_output=80000, budget_soft=40000, budget_notes=0.01, budget_warn=0.75, **kw)


class E025InterfaceTest(unittest.TestCase):
    def test_prompt_matches_deepseek(self):
        import run_deepseek_tools as ds
        tasks = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
        for task_id in list(tasks)[:6]:
            t = tasks[task_id]
            ref = (ds.build_prompt(t["train"], t["test"][0]["input"]) + ds.PROMPT_SUFFIXES["lean2roles"]
                   + ds.HELPERS_PROMPT + ds.task_summary(t["train"], t["test"][0]["input"]))
            self.assertEqual(aq.user_content(t["train"], t["test"][0]["input"], "lean2roles", True, True), ref)

    def test_tool_specs(self):
        import run_deepseek_tools as ds
        cfg = e025_cfg()
        cfg.tool_timeout = 30
        ref = ds.tools_for(argparse.Namespace(tool_timeout=30, helpers=True, answer_tool=True, mode="grid"))
        self.assertEqual(json.dumps(aq.tool_specs(cfg)), json.dumps(ref))
        if E025_TOOLS.exists():
            self.assertEqual(json.dumps(aq.tool_specs(cfg)), json.dumps(json.loads(E025_TOOLS.read_text())))

    def test_budget_notes_match_deepseek(self):
        import run_deepseek_tools as ds
        args = argparse.Namespace(budget_notes=0.01, max_total_output=80000, budget_soft=40000, budget_warn=0.75,
                                  soft_wording="gentle", answer_tool=True)
        for used in range(0, 90000, 97):
            self.assertEqual(aq.budget_note(e025_cfg(), used), ds.budget_note(args, used))

    def test_defaults_unchanged(self):
        cfg = aq.AgentConfig()
        d = aq.config_dict(cfg, "m")
        self.assertNotIn("tool_desc", d)
        self.assertNotIn("solve_fallback", d)
        self.assertEqual(d["harness_version"], "qwen-tools-4")
        self.assertEqual(aq.config_dict(e025_cfg(), "m")["harness_version"], "qwen-tools-5")


class E025LoopTest(unittest.TestCase):
    def test_unknown_tool_and_skip_strings(self):
        llm = FakeLLM([(call("foo", "1"), "stop"),
                       ("ok\n</think>\n\n<tool_call>\n<function=submit_answer>\n<parameter=code>\n"
                        "[[9, 8, 7]]\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run_python>\n"
                        "<parameter=code>\n1\n</parameter>\n</function>\n</tool_call>", "stop")])
        transcript, res = asyncio.run(aq.solve(llm, local_factory(5, prelude_extra=aq.HELPERS_SRC), TRAIN, TEST,
                                               e025_cfg()))
        self.assertIn("<tool_response>\n[tool error: Unknown tool: foo]\n</tool_response>", llm.prompts[1][0])
        self.assertEqual(res["pred"], ANSWER)

    def test_solve_fallback(self):
        code = "def solve(g):\n    return [r[::-1] for r in g]\nprint('defined')"
        script = [(call("run_python", code), "stop"), ("I give up.\n</think>\n\nNo answer.", "stop")]
        _, res = asyncio.run(aq.solve(FakeLLM(script), local_factory(5), TRAIN, TEST,
                                      e025_cfg(solve_fallback=True, salvage=False)))
        self.assertEqual((res["pred"], res["answer_source"]), (ANSWER, "session_solve"))
        _, res = asyncio.run(aq.solve(FakeLLM(list(script)), local_factory(5), TRAIN, TEST,
                                      e025_cfg(salvage=False)))
        self.assertIsNone(res["pred"])


class IPythonStyleTest(unittest.TestCase):
    """Strings recorded from the Docker sandbox (IPython 8.38, Plain tracebacks) for the same cells."""

    def test_formats(self):
        cases = [
            ("x = [[1,2,3]]*3\nx", "[completed]\n[[1, 2, 3], [1, 2, 3], [1, 2, 3]]\n"),
            ("print('a')\nnp.zeros((3,3), int)",
             "[completed]\na\n\narray([[0, 0, 0],\n       [0, 0, 0],\n       [0, 0, 0]])\n"),
            ("list(range(30))", "[completed]\n[" + ",\n ".join(str(i) for i in range(30)) + "]\n"),
            ("print('pre')\nraise ValueError('boom')",
             "[error]\npre\nTraceback (most recent call last):\n    raise ValueError('boom')\nValueError: boom\n"),
            ("print(undefined_name)", "[error]\nTraceback (most recent call last):\n    print(undefined_name)\n"
                                      "NameError: name 'undefined_name' is not defined\n"),
            ("None", "[completed]\n[no output; use print()]"),
            ("print('a\\n\\n')", "[completed]\na\n"),
            ("import sys; print('err', file=sys.stderr); print('out')", "[completed]\nout\n\n[stderr]\nerr\n"),
            ("print('start')\nwhile True: pass",
             "[timeout]\nstart\nTraceback (most recent call last):\n    while True: pass\nKeyboardInterrupt\n"),
            ("x", "[completed]\n[[1, 2, 3], [1, 2, 3], [1, 2, 3]]\n"),  # state kept after the timeout
            ("print('y' * 250)", "[completed]\n" + "y" * 200 + "<output cut>"),
        ]
        with LocalExecutor(TRAIN, TEST, timeout=2, max_output=200, style="ipython") as ex:
            for code, want in cases:
                self.assertEqual(tool_output(ex.execute(code), 16000), want, code)

    def test_plain_default_unchanged(self):
        with LocalExecutor(TRAIN, TEST, timeout=5) as ex:
            r = ex.execute("print('pre')\nraise ValueError('boom')")
        self.assertEqual(r["stdout"], "pre\n")
        self.assertIn("[stderr]\nTraceback (most recent call last):\n  File \"<cell>\"", tool_output(r))


if __name__ == "__main__":
    unittest.main()
