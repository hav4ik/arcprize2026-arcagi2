"""Tests for the E019 program-only answer mode (agent_qwen answer_mode="program"; no network, no sandbox).

    .venv/bin/python -m unittest evals/tests/test_qwen_program.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_harness import agent_qwen as aq  # noqa: E402
from arc_harness.executor import (LocalExecutor, local_factory, program_feedback, run_program,  # noqa: E402
                                  run_session_solve, score_outputs)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qwen_tools import ANSWER, TEST, TRAIN, FakeLLM, call  # noqa: E402

GOOD = "def solve(grid):\n    return [row[::-1] for row in grid]"
HALF = "def solve(grid):\n    return [[5, 4, 3]]"          # passes pair 1 only
BAD = "def solve(grid):\n    return [[0] * len(grid[0])]"   # right shape, every cell wrong


def cfg(**kw):
    return aq.AgentConfig(tool_timeout=5, answer_mode="program", prompt="lean2prog", program_timeout=3, **kw)


def run(llm, c=None):
    return asyncio.run(aq.solve(llm, local_factory(5), TRAIN, TEST, c or cfg()))


class RunProgramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = LocalExecutor(TRAIN, TEST, timeout=20)
        cls.ex.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ex.close()

    def test_ok_and_numpy(self):
        self.assertEqual(run_program(self.ex, GOOD, [[1, 2]]), {"out": [[2, 1]]})
        self.assertEqual(run_program(self.ex, "def solve(g):\n    return np.array(g)[:, ::-1]", [[1, 2]]),
                         {"out": [[2, 1]]})

    def test_fresh_process(self):
        self.ex.execute("secret = 5")
        r = run_program(self.ex, "def solve(g):\n    return [[secret]]", [[1]])
        self.assertEqual(r["kind"], "NameError")

    def test_errors(self):
        r = run_program(self.ex, "def solve(g):\n    x = g[5][0]\n    return g", [[1]])
        self.assertEqual(r["kind"], "IndexError")
        self.assertIn("line 2, in solve: x = g[5][0]", r["error"])
        self.assertIn("does not define solve", run_program(self.ex, "def f(g):\n    return g", [[1]])["error"])
        self.assertEqual(run_program(self.ex, "def solve(g)\n    return g", [[1]])["kind"], "SyntaxError")

    def test_timeout_and_print(self):
        r = run_program(self.ex, "def solve(g):\n    while True:\n        pass", [[1]], timeout=1)
        self.assertEqual(r["kind"], "timeout")
        self.assertEqual(run_program(self.ex, "def solve(g):\n    print('noise')\n    return g", [[3]]), {"out": [[3]]})

    def test_session_solve(self):
        self.ex.execute("def _h(g):\n    return [r[::-1] for r in g]\ndef solve(g):\n    return np.array(_h(g))")
        self.assertEqual(run_session_solve(self.ex, [[1, 2]]), {"out": [[2, 1]]})
        self.ex.execute("del solve")
        self.assertIsNone(run_session_solve(self.ex, [[1, 2]]))


class FeedbackTest(unittest.TestCase):
    def test_rows_and_text(self):
        outs = [{"out": [[0, 0]]}, {"out": [[1]]}, {"error": "  line 2, in solve: x\nIndexError: boom", "kind": "IndexError"}]
        train = TRAIN + [{"input": [[1]], "output": [[1]]}]
        rows = score_outputs(outs, train)
        self.assertEqual(len(rows[0]["wrong"]), 2)
        self.assertEqual(rows[1]["shape"], ("1x1", "1x3"))
        self.assertEqual(rows[2]["kind"], "IndexError")
        text = program_feedback(rows, 2, 6)
        self.assertIn("submission 2 of 6", text)
        self.assertIn("3 of 3 training pairs fail", text)
        self.assertIn("pair 0: right shape, 2 wrong cells", text)
        self.assertIn("pair 1: wrong shape: got 1x1, expected 1x3", text)
        self.assertIn("pair 2: error: IndexError: boom", text)
        self.assertIn("(0,0) expected 2 got 0", text)
        self.assertIn("passes all", program_feedback([{"ok": True}], 1, 6, test_error="ValueError: x"))

    def test_invalid_grid_and_diff_cap(self):
        rows = score_outputs([{"out": [[1, 2], [3]]}], TRAIN[:1])
        self.assertEqual(rows[0]["kind"], "invalid")
        big = {"input": [[0] * 30] * 30, "output": [[1] * 30] * 30}
        text = program_feedback(score_outputs([{"out": [[0] * 30] * 30}], [big]), 1, 6)
        self.assertIn("900 wrong cells", text)
        self.assertIn("888 more", text)
        self.assertLess(len(text), 1000)


class AgentProgramTest(unittest.TestCase):
    def test_specs_prompt_and_config(self):
        c = cfg()
        names = [t["function"]["name"] for t in aq.tool_specs(c)]
        self.assertEqual(names, ["run_python", "submit_program"])
        p = aq.initial_prompt(TRAIN, TEST, c)
        self.assertIn("typed grids are not accepted", p)
        self.assertNotIn("submit_answer", p)
        self.assertIn("submit_program", aq.budget_note(cfg(max_total_output=100, budget_soft=50), 90))
        d = aq.config_dict(c, "m")
        self.assertEqual((d["answer_mode"], d["max_submissions"], d["harness_version"]), ("program", 6, "qwen-tools-4"))
        base = aq.config_dict(aq.AgentConfig(), "m")
        self.assertEqual(base["answer_mode"], "submit")
        self.assertFalse({"max_submissions", "program_timeout", "max_nudges", "record_train_check"} & set(base))
        self.assertEqual(aq.tool_specs(aq.AgentConfig())[1]["function"]["name"], "submit_answer")

    def test_pass_first_time(self):
        prompt, res = run(FakeLLM([(call("submit_program", GOOD), "stop")]))
        self.assertEqual(res["pred"], ANSWER)
        self.assertEqual((res["answer_source"], res["train_pass"], res["train_frac"]), ("program_pass", True, 1.0))
        self.assertEqual((res["n_submissions"], res["program"], res["stop_reason"]), (1, GOOD, "submitted"))

    def test_fail_feedback_then_pass(self):
        llm = FakeLLM([(call("submit_program", HALF), "stop"), (call("submit_program", GOOD), "stop")])
        prompt, res = run(llm)
        self.assertIn("1 of 2 training pairs fail", llm.prompts[1][0])
        self.assertIn("pair 0: wrong shape: got 1x3, expected 1x2", llm.prompts[1][0])
        self.assertEqual((res["answer_source"], res["n_submissions"], res["pred"]), ("program_pass", 2, ANSWER))
        self.assertEqual(res["submissions"][0]["n_pass"], 1)

    def test_train_pass_but_test_fails(self):
        code = "def solve(grid):\n    assert len(grid[0]) < 3 or grid[0][0] < 7\n    return [r[::-1] for r in grid]"
        llm = FakeLLM([(call("submit_program", code), "stop"), (call("submit_program", GOOD), "stop")])
        _, res = run(llm)
        self.assertIn("passes all 2 training pairs but failed on test_input", llm.prompts[1][0])
        self.assertEqual(res["n_submissions"], 2)
        self.assertEqual(res["answer_source"], "program_pass")

    def test_submission_cap_falls_back_to_best(self):
        llm = FakeLLM([(call("submit_program", HALF), "stop"), (call("submit_program", BAD), "stop")])
        _, res = run(llm, cfg(max_submissions=2))
        self.assertEqual(res["stop_reason"], "submission_cap")
        self.assertEqual((res["answer_source"], res["train_frac"], res["train_pass"]), ("program_best_fail", 0.5, False))
        self.assertEqual((res["pred"], res["program"]), ([[5, 4, 3]], HALF))
        self.assertEqual(len(llm.script), 0)

    def test_budget_cap_session_solve(self):
        cell = "def solve(grid):\n    return [r[::-1] for r in grid]\nprint('ok')"
        llm = FakeLLM([(call("run_python", cell), "stop"), ("still thinking", "length")], tokens=60)
        _, res = run(llm, cfg(max_total_output=100, budget_soft=50))
        self.assertEqual(res["stop_reason"], "output_budget")
        self.assertEqual((res["answer_source"], res["train_pass"], res["pred"]), ("session_solve", True, ANSWER))
        self.assertEqual(res["program"], cell)
        self.assertEqual(res["n_submissions"], 0)

    def test_session_solve_beats_worse_submission(self):
        cell = "def solve(grid):\n    return [r[::-1] for r in grid]"
        llm = FakeLLM([(call("submit_program", HALF), "stop"), (call("run_python", cell), "stop"),
                       ("thinking", "length")], tokens=40)
        _, res = run(llm, cfg(max_total_output=100, budget_soft=50))
        self.assertEqual((res["answer_source"], res["train_frac"]), ("session_solve", 1.0))

    def test_submission_beats_worse_session_solve(self):
        cell = "def solve(grid):\n    return [[0]]"
        llm = FakeLLM([(call("submit_program", HALF), "stop"), (call("run_python", cell), "stop"),
                       ("thinking", "length")], tokens=40)
        _, res = run(llm, cfg(max_total_output=100, budget_soft=50))
        self.assertEqual((res["answer_source"], res["train_frac"], res["program"]), ("program_best_fail", 0.5, HALF))

    def test_typed_answers_rejected(self):
        llm = FakeLLM([(call("submit_answer", "[[9, 8, 7]]"), "stop"), ("done\n</think>\n\n[[9, 8, 7]]", "stop"),
                       ("done\n</think>\n\n[[9, 8, 7]]", "stop"), ("done\n</think>\n\n[[9, 8, 7]]", "stop")])
        _, res = run(llm)
        self.assertIn("Typed grids and submit_answer are not accepted", llm.prompts[1][0])
        self.assertIn(aq.PROGRAM_NUDGE, llm.prompts[2][0])
        self.assertEqual((res["pred"], res["answer_source"], res["stop_reason"], res["nudges"]), (None, None, "final", 2))
        self.assertIsNone(res["train_pass"])

    def test_no_forced_typed_answer_at_cap(self):
        llm = FakeLLM([("thinking forever", "length")], tokens=100)
        _, res = run(llm, cfg(max_total_output=100, budget_soft=50))
        self.assertEqual(len(llm.prompts), 1)  # no forced submit request
        self.assertEqual((res["pred"], res["answer_source"]), (None, None))

    def test_submit_mode_train_check_bookkeeping(self):
        c = aq.AgentConfig(tool_timeout=5, record_train_check=True)
        llm = FakeLLM([(call("run_python", "def solve(g):\n    return [r[::-1] for r in g]"), "stop"),
                       (call("submit_answer", "solve(test_input)"), "stop")])
        _, res = asyncio.run(aq.solve(llm, local_factory(5), TRAIN, TEST, c))
        self.assertEqual((res["answer_source"], res["train_pass"], res["train_frac"]), ("submit", True, 1.0))
        llm = FakeLLM([(call("submit_answer", "[[9, 8, 7]]"), "stop")])
        _, res = asyncio.run(aq.solve(llm, local_factory(5), TRAIN, TEST, c))
        self.assertIsNone(res["train_pass"])


if __name__ == "__main__":
    unittest.main()


class RoutingHeaderTest(unittest.TestCase):
    def test_header_reaches_http_and_worker_recorded(self):
        import http.server
        import json
        import threading
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                seen.append(self.headers.get("x-smg-routing-key"))
                body = json.dumps({"id": "x", "object": "text_completion", "created": 0, "model": "m",
                                   "choices": [{"index": 0, "text": "hi", "finish_reason": "stop", "logprobs": None}],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6,
                                             "prompt_tokens_details": {"cached_tokens": 3}}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Worker-Id", "http://w1:8000")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        async def go():
            llm = aq.OpenAICompletionsLLM(f"http://127.0.0.1:{srv.server_port}/v1", None, "m")
            try:
                c = await aq.RoutedLLM(llm, "x-smg-routing-key", "run:t:0:0").complete("p", 4)
                c2 = await llm.complete("p", 4)
            finally:
                await llm.close()
            return c, c2
        c, c2 = asyncio.run(go())
        srv.shutdown()
        self.assertEqual(seen, ["run:t:0:0", None])
        self.assertEqual((c.text, c.cached_tokens, c.served_by), ("hi", 3, "http://w1:8000"))
