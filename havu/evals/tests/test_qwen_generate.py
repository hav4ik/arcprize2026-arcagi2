"""Pure-function tests for evals/run_qwen_generate.py (no network).

    .venv/bin/python -m unittest evals/tests/test_qwen_generate.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_qwen_generate as q  # noqa: E402
from arc_harness.arcprize import parse  # noqa: E402


class RenderTest(unittest.TestCase):
    def test_medium_has_no_system(self):
        self.assertEqual(q.render_prompt("  hi \n", "medium"),
                         "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n")

    def test_tiers_with_system(self):
        for tier in ("low", "xhigh"):
            s = q.render_prompt("hi", tier)
            self.assertTrue(s.startswith(f"<|im_start|>system\nReasoning effort is set to {tier}."))
            self.assertTrue(s.endswith("<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"))


class AnswerTest(unittest.TestCase):
    def test_answer_section(self):
        self.assertIsNone(q.answer_section("thinking [[1]]"))
        self.assertEqual(q.answer_section("a</think>b</think>\n[[1, 2]]"), "\n[[1, 2]]")
        self.assertEqual(parse(q.answer_section("x [[9]] </think>\n\n[[1, 2], [3, 4]]")), [[1, 2], [3, 4]])

    def test_forced_text_closes_think(self):
        self.assertEqual(q.answer_section("thinking" + q.FORCE_TEXT + "[[0]]"), "\n\n[[0]]")


class NextActionTest(unittest.TestCase):
    def test_done_on_stop_after_think(self):
        self.assertEqual(q.next_action("t</think>[[1]]", "stop", 100, 0, False), "done")
        self.assertEqual(q.next_action("t</think>no grid", "stop", 100, 0, False), "done")

    def test_answer_cut_by_chunk_cap_continues(self):
        self.assertEqual(q.next_action("t</think>[[1, 2", "length", 100, 0, False), "continue")
        self.assertEqual(q.next_action("t</think>[[1, 2", "length", 0, 0, False), "done")

    def test_thinking(self):
        self.assertEqual(q.next_action("thinking", "length", 100, 0, False), "continue")
        self.assertEqual(q.next_action("thinking", "length", 0, 0, False), "force_final")
        self.assertEqual(q.next_action("thinking", "stop", 0, 0, False), "force_final")

    def test_eos_mid_thought(self):
        self.assertEqual(q.next_action("thinking", "stop", 100, 0, False), "eos_continue")
        self.assertEqual(q.next_action("thinking", "stop", 100, 1, False), "eos_continue")
        self.assertEqual(q.next_action("thinking", "stop", 100, 2, False), "force_final")

    def test_forced_is_final(self):
        self.assertEqual(q.next_action("t" + q.FORCE_TEXT, "length", 0, 0, True), "done")


class SchedulingTest(unittest.TestCase):
    def test_chunk_max_tokens(self):
        self.assertEqual(q.chunk_max_tokens(48000, 16384, 5000, 0), 16384)
        self.assertEqual(q.chunk_max_tokens(1000, 16384, 5000, 47000), 1000)
        self.assertEqual(q.chunk_max_tokens(90000, 300000, 200000, 10000), q.CONTEXT - 210064)

    def test_pilot_first(self):
        ids = [f"t{i:03d}" for i in range(20)][::-1]
        jobs = q.job_order(ids, {t: 1 + (t == "t001") for t in ids}, 2, pilot=3)
        self.assertEqual(jobs[:6], [("t000", 0, 0), ("t000", 0, 1), ("t001", 0, 0), ("t001", 0, 1),
                                    ("t001", 1, 0), ("t001", 1, 1)])
        self.assertEqual(len(jobs), 2 * 21)
        self.assertEqual(jobs[8][0], "t003")


if __name__ == "__main__":
    unittest.main()
