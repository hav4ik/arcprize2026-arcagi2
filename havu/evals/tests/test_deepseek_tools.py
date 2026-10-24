"""Offline loop tests; set ARC_SANDBOX_URL to also test the real Docker backend."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai.types.chat import ChatCompletionChunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_deepseek_tools import (PROMPT_SUFFIXES, TurnError, lenient_parse, load_tasks, parser, read_records,
                               solve, submit_answer, submit_solution, summarize, turn, valid_grid,
                               check_answer_on_train)
from sandbox_client import SandboxSession, tool_output


TASK = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]], "output": [[4, 3]]}]}


def chunk(content=None, reasoning=None, calls=None, finish=None, tokens=None):
    usage = None if tokens is None else {"prompt_tokens": 30, "completion_tokens": tokens,
                                        "total_tokens": 30 + tokens, "prompt_cache_hit_tokens": 10}
    return ChatCompletionChunk.model_validate({
        "id": "test", "object": "chat.completion.chunk", "created": 0, "model": "deepseek-flash",
        "choices": [{"index": 0, "delta": {"content": content, "reasoning_content": reasoning,
                                               "tool_calls": calls}, "finish_reason": finish}],
        "usage": usage,
    })


def call(code, name="run_python", index=0):
    return {"index": index, "id": f"call_{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps({"code": code})}}


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        for item in self.chunks:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, turns):
        self.turns, self.requests, self.streams = iter(turns), [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        stream = Stream(next(self.turns))
        self.streams.append(stream)
        return stream


class FakeSession:
    instances = []

    def __init__(self, url, train, test_input, timeout, max_output):
        self.calls, self.closed, self.max_output = [], False, max_output
        self.train, self.test_input = train, test_input
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def execute(self, code):
        self.calls.append(code)
        return {"process_status": "completed", "stdout": "ok", "stderr": ""}


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.args = parser().parse_args(["--sandbox-url", "http://sandbox", "--attempts", "1"])
        FakeSession.instances = []

    def solve(self, client):
        return solve(client, self.args, TASK, TASK["test"][0]["input"], FakeSession)

    def test_fragmented_multiple_calls_and_reasoning(self):
        first = [
            chunk(reasoning="plan", calls=[{"index": 1, "id": "b", "type": "function",
                                            "function": {"name": "run_", "arguments": '{"code":'}}]),
            chunk(calls=[{"index": 1, "function": {"name": "python", "arguments": '"print(2)"}'}}]),
            chunk(calls=[call("print(1)", index=0)], finish="tool_calls", tokens=20),
        ]
        client = FakeClient([first, [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        trace, result = self.solve(client)
        self.assertEqual(FakeSession.instances[0].calls, ["print(1)", "print(2)"])
        self.assertTrue(FakeSession.instances[0].closed)
        self.assertEqual(result["pred"], [[4, 3]])
        self.assertEqual(result["usage"], {"prompt": 60, "cache_hit": 20, "completion": 25})
        self.assertEqual(client.requests[1]["messages"][1]["reasoning_content"], "plan")
        self.assertEqual([m["tool_call_id"] for m in trace["messages"] if m["role"] == "tool"], ["call_0", "b"])
        self.assertTrue(all(s.closed for s in client.streams))
        self.assertNotIn("output", FakeSession.instances[0].__dict__.keys())

    def test_token_budget_reserves_final(self):
        self.args.max_total_output, self.args.final_tokens = 100, 20
        client = FakeClient([
            [chunk(reasoning="thinking", calls=[call("print(1)")], finish="tool_calls", tokens=80)],
            [chunk(content="[[4,3]]", finish="stop", tokens=20)],
        ])
        _, result = self.solve(client)
        self.assertEqual([r["max_tokens"] for r in client.requests], [80, 20])
        self.assertEqual(client.requests[-1]["tool_choice"], "none")
        self.assertIn("tools", client.requests[-1])
        self.assertEqual(result["usage"]["completion"], 100)

    def test_length_reasoning_uses_reserved_final_turn(self):
        self.args.max_total_output, self.args.final_tokens = 100, 20
        client = FakeClient([[chunk(reasoning="long", finish="length", tokens=80)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=10)]])
        _, result = self.solve(client)
        self.assertEqual(result["pred"], [[4, 3]])
        self.assertEqual(client.requests[-1]["tool_choice"], "none")

    def test_turn_budget_forces_final_and_no_infinite_loop(self):
        self.args.max_turns = 2
        client = FakeClient([[chunk(calls=[call("print(1)")], finish="tool_calls", tokens=5)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        _, result = self.solve(client)
        self.assertEqual(result["turns"], 2)
        self.assertEqual(client.requests[-1]["tool_choice"], "none")

    def test_bad_arguments_and_unknown_tool_return_feedback(self):
        bad = call("unused")
        bad["function"]["arguments"] = "{broken"
        client = FakeClient([
            [chunk(calls=[bad, call("unused", name="shell", index=1)], finish="tool_calls", tokens=5)],
            [chunk(content="[[4,3]]", finish="stop", tokens=5)],
        ])
        _, result = self.solve(client)
        self.assertEqual(result["tool_errors"], 2)
        self.assertEqual(FakeSession.instances[0].calls, [])
        self.assertEqual(result["pred"], [[4, 3]])

    def test_partial_stream_saves_reasoning_and_cleans_up(self):
        client = FakeClient([[chunk(reasoning="partial"), httpx.ReadError("connection lost")]])
        trace, result = self.solve(client)
        self.assertIn("connection lost", result["error"])
        self.assertEqual(trace["messages"][-1]["reasoning_content"], "partial")
        self.assertFalse(result["usage_complete"])
        self.assertTrue(FakeSession.instances[0].closed)
        self.assertTrue(client.streams[0].closed)

    def test_no_usage_is_not_a_success(self):
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop")]])
        _, result = self.solve(client)
        self.assertIn("missing finish reason or usage", result["error"])

    def test_invalid_grid_is_unparsed(self):
        for grid in ([[1], [2, 3]], [[10]], [[True]], [[]], []):
            self.assertFalse(valid_grid(grid))
        _, result = self.solve(FakeClient([[chunk(content="[[10]]", finish="stop", tokens=3)]]))
        self.assertIsNone(result["pred"])

    def test_code_submission_can_fail_then_succeed(self):
        self.args.mode = "code"
        client = FakeClient([
            [chunk(calls=[call("wrong", "submit_solution")], finish="tool_calls", tokens=5)],
            [chunk(calls=[call("right", "submit_solution")], finish="tool_calls", tokens=5)],
        ])
        with patch("run_deepseek_tools.submit_solution", side_effect=[("training failed", None),
                                                                      ("accepted", [[4, 3]])]):
            _, result = self.solve(client)
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["pred"], [[4, 3]])
        self.assertEqual(result["tool_errors"], 1)

    def test_task_weighted_scoring_uses_latest_retry(self):
        tasks = {"a": {"test": [{"output": [[1]]}]},
                 "b": {"test": [{"output": [[1]]}, {"output": [[1]]}]}}
        records = [{"task_id": "a", "pair": 0, "attempt": 0, "error": "old"},
                   {"task_id": "a", "pair": 0, "attempt": 0, "correct": True},
                   {"task_id": "b", "pair": 0, "attempt": 0, "correct": True}]
        output = io.StringIO()
        with redirect_stdout(output):
            summarize(records, tasks, 2)
        self.assertIn("75.0%", output.getvalue())
        # The failed retry of a/0/0 is superseded by its successful rerun.
        self.assertIn("2 attempts, 0 errors", output.getvalue())


class E009OptionTests(unittest.TestCase):
    """--answer-tool, --final-mode session, --budget-notes, --lenient-parse (harness persistent-v4)."""

    def setUp(self):
        self.args = parser().parse_args(["--sandbox-url", "http://sandbox", "--attempts", "1", "--answer-tool",
                                         "--final-mode", "session", "--prompt", "lean2sub"])
        FakeSession.instances = []

    def solve(self, client):
        return solve(client, self.args, TASK, TASK["test"][0]["input"], FakeSession)

    def test_defaults_add_no_answer_source(self):
        args = parser().parse_args(["--sandbox-url", "http://sandbox"])
        _, result = solve(FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=3)]]), args, TASK,
                          TASK["test"][0]["input"], FakeSession)
        self.assertNotIn("answer_source", result)
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=3)]])
        solve(client, args, TASK, TASK["test"][0]["input"], FakeSession)
        self.assertEqual([t["function"]["name"] for t in client.requests[0]["tools"]], ["run_python"])

    def test_submit_answer_ends_attempt(self):
        client = FakeClient([[chunk(calls=[call("print(1)"), call("solve(test_input)", "submit_answer", 1)],
                                    finish="tool_calls", tokens=7)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer accepted: 1x2 grid]", [[4, 3]])) as sub:
            trace, result = self.solve(client)
        sub.assert_called_once()
        self.assertEqual(sub.call_args.args[1], "solve(test_input)")
        self.assertEqual((result["pred"], result["stop_reason"], result["answer_source"]), ([[4, 3]], "submitted", "submit"))
        self.assertEqual(len(client.requests), 1)
        names = [t["function"]["name"] for t in client.requests[0]["tools"]]
        self.assertEqual(names, ["run_python", "submit_answer"])
        self.assertIn("submit_answer", client.requests[0]["messages"][0]["content"])

    def test_rejected_answer_continues(self):
        client = FakeClient([[chunk(calls=[call("bad", "submit_answer")], finish="tool_calls", tokens=5)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer not accepted] x", None)):
            trace, result = self.solve(client)
        self.assertEqual(result["tool_errors"], 1)
        self.assertEqual((result["pred"], result["answer_source"]), ([[4, 3]], "parse"))
        self.assertIn("not accepted", trace["messages"][2]["content"])

    def test_session_final_mode_has_no_forced_turn_and_falls_back_to_answer(self):
        self.args.max_total_output, self.args.max_tokens = 100, 70
        client = FakeClient([[chunk(calls=[call("answer = [[4, 3]]")], finish="tool_calls", tokens=60)],
                             [chunk(reasoning="more", finish="length", tokens=40)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("ok", [[4, 3]])) as sub:
            trace, result = self.solve(client)
        self.assertEqual([r["max_tokens"] for r in client.requests], [70, 40])
        self.assertTrue(all(r["tool_choice"] == "auto" for r in client.requests))
        self.assertFalse(any(m["role"] == "user" for m in trace["messages"][1:]))
        self.assertEqual(sub.call_args.args[1], "answer")
        self.assertEqual((result["pred"], result["answer_source"], result["stop_reason"]),
                         ([[4, 3]], "session", "output_budget"))

    def test_session_fallback_missing_answer_gives_none(self):
        self.args.max_total_output = 50
        client = FakeClient([[chunk(calls=[call("x = 1")], finish="tool_calls", tokens=50)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer not accepted]", None)):
            _, result = self.solve(client)
        self.assertIsNone(result["pred"])
        self.assertIsNone(result["answer_source"])

    def test_text_answer_wins_over_session_answer(self):
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("ok", [[9, 9]])) as sub:
            _, result = self.solve(client)
        sub.assert_not_called()
        self.assertEqual((result["pred"], result["answer_source"]), ([[4, 3]], "parse"))

    def test_budget_notes(self):
        self.args.max_total_output, self.args.budget_notes, self.args.budget_warn = 100, 0.5, 0.8
        client = FakeClient([[chunk(calls=[call("1")], finish="tool_calls", tokens=40)],
                             [chunk(calls=[call("2")], finish="tool_calls", tokens=15)],
                             [chunk(calls=[call("3")], finish="tool_calls", tokens=30)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        trace, _ = self.solve(client)
        outs = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertNotIn("[budget", outs[0])
        self.assertIn("[budget: 0k of 0k", outs[1].replace("0.0", "0"))
        self.assertNotIn("nearly spent", outs[1])
        self.assertIn("nearly spent", outs[2])

    def test_budget_soft_target(self):
        self.args.max_total_output, self.args.budget_notes, self.args.budget_soft = 100000, 0.01, 40000
        client = FakeClient([[chunk(calls=[call("1")], finish="tool_calls", tokens=30000)],
                             [chunk(calls=[call("2")], finish="tool_calls", tokens=20000)],
                             [chunk(calls=[call("3")], finish="tool_calls", tokens=30000)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        trace, _ = self.solve(client)
        outs = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("[budget: 30k output tokens used; aim to finish within 40k (hard limit 100k)]", outs[0])
        self.assertNotIn("Past the target", outs[0])
        self.assertIn("Past the target", outs[1])
        self.assertIn("nearly spent", outs[2])
        self.assertNotIn("Past the target", outs[2])

    def test_turns_note_warns_near_turn_cap(self):
        self.args.max_turns, self.args.turns_note, self.args.final_mode = 4, 2, "session"
        client = FakeClient([[chunk(calls=[call(str(i))], finish="tool_calls", tokens=5)] for i in range(4)])
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer not accepted]", None)):
            trace, result = self.solve(client)
        outs = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertNotIn("turns left", outs[0])
        self.assertIn("[2 turns left: call submit_answer", outs[1])
        self.assertIn("[1 turn left: call submit_answer", outs[2])
        self.assertNotIn("left:", outs[3])
        self.assertEqual(result["stop_reason"], "max_turns")

    def test_fallback_solve_after_missing_answer(self):
        self.args.max_total_output, self.args.final_mode, self.args.fallback_solve = 50, "session", True
        client = FakeClient([[chunk(calls=[call("def solve(g): return g")], finish="tool_calls", tokens=50)]])
        replies = {"answer": ("[answer not accepted]", None), "solve(test_input)": ("ok", [[4, 3]])}
        with patch("run_deepseek_tools.submit_answer", side_effect=lambda s, code: replies[code]) as sub:
            _, result = self.solve(client)
        self.assertEqual([c.args[1] for c in sub.call_args_list], ["answer", "solve(test_input)"])
        self.assertEqual((result["pred"], result["answer_source"]), ([[4, 3]], "session_solve"))

    def test_fallback_solve_off_by_default(self):
        self.args.max_total_output, self.args.final_mode = 50, "session"
        client = FakeClient([[chunk(calls=[call("x = 1")], finish="tool_calls", tokens=50)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer not accepted]", None)) as sub:
            _, result = self.solve(client)
        self.assertEqual([c.args[1] for c in sub.call_args_list], ["answer"])
        self.assertIsNone(result["pred"])

    def test_cap_nudge_closes_cut_turn_with_synthetic_call(self):
        self.args.final_mode, self.args.answer_tool, self.args.cap_nudge = "session", True, True
        self.args.max_tokens, self.args.max_total_output = 50, 1000
        client = FakeClient([[chunk(reasoning="long thought", content="partial", finish="length", tokens=50)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        trace, result = self.solve(client)
        cut, note = trace["messages"][1], trace["messages"][2]
        self.assertEqual((cut["content"], cut["reasoning_content"]), ("", "long thought"))
        self.assertEqual(cut["tool_calls"][0]["function"]["name"], "run_python")
        self.assertEqual((note["role"], note["tool_call_id"]), ("tool", cut["tool_calls"][0]["id"]))
        self.assertIn("per-turn thinking limit", note["content"])
        self.assertEqual(client.requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(FakeSession.instances[0].calls, [])  # the synthetic call is not executed
        self.assertEqual((result["cap_nudges"], result["pred"]), (1, [[4, 3]]))

    def test_cap_nudge_not_used_when_total_budget_is_the_limit(self):
        self.args.final_mode, self.args.answer_tool, self.args.cap_nudge = "session", True, True
        self.args.max_tokens, self.args.max_total_output = 100, 60
        client = FakeClient([[chunk(reasoning="long", finish="length", tokens=60)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer not accepted]", None)):
            trace, result = self.solve(client)
        self.assertEqual(result["cap_nudges"], 0)
        self.assertEqual(result["stop_reason"], "output_budget")

    def test_effort_header_is_first_system_message(self):
        self.args.effort_header = "minimal"
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        self.solve(client)
        first = client.requests[0]["messages"][0]
        self.assertEqual(first["role"], "system")
        self.assertTrue(first["content"].startswith("Reasoning Effort: Minimal."))
        self.assertEqual(client.requests[0]["messages"][1]["role"], "user")

    def test_grid_image_cells_align_with_patches(self):
        import io
        from PIL import Image
        from arc_harness.grid_image import PALETTE, render, visual_tokens
        grid = [[0, 1, 2], [3, 4, 5]]
        img = Image.open(io.BytesIO(render(grid, cell=32)))
        self.assertEqual(img.size, (96, 64))
        for r, row in enumerate(grid):
            for c, v in enumerate(row):  # every pixel of a cell has the cell's colour, including its edges
                for y, x in ((r * 32, c * 32), (r * 32 + 31, c * 32 + 31), (r * 32 + 16, c * 32 + 7)):
                    self.assertEqual(img.getpixel((x, y)), PALETTE[v])
        self.assertEqual(visual_tokens(grid, 32), 6)    # one LLM token (2x2 merged 16-px patches) per cell
        self.assertEqual(visual_tokens([[0] * 30] * 30, 32), 900)

    def test_image_prompt_replaces_every_grid(self):
        self.args.grid_format, self.args.image_cell, self.args.base_url = "image", 32, "http://x"
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        self.solve(client)
        content = client.requests[0]["messages"][0]["content"]
        images = [p for p in content if p["type"] == "image_url"]
        self.assertEqual(len(images), 2 * len(TASK["train"]) + 1)
        text = "".join(p["text"] for p in content if p["type"] == "text")
        self.assertNotIn("[[1, 2]]", text)                # no JSON grids left in the text
        self.assertIn("Colour key", text)
        self.assertIn("1 rows x 2 columns", text)
        self.assertTrue(text.rstrip().endswith(PROMPT_SUFFIXES[self.args.prompt].strip()[-40:]))

    def test_lean2code_prompt_extends_lean2sub(self):
        self.assertTrue(PROMPT_SUFFIXES["lean2code"].startswith(PROMPT_SUFFIXES["lean2sub"]))
        self.assertIn("Let code do the perceiving", PROMPT_SUFFIXES["lean2code"])
        self.assertNotIn("brief", PROMPT_SUFFIXES["lean2code"][len(PROMPT_SUFFIXES["lean2sub"]):])

    def test_think_turns_disables_thinking_after_n(self):
        self.args.think_turns = 1
        client = FakeClient([[chunk(calls=[call("1")], finish="tool_calls", tokens=5)],
                             [chunk(calls=[call("2")], finish="tool_calls", tokens=5)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        trace, _ = self.solve(client)
        self.assertEqual(client.requests[0]["extra_body"]["thinking"], {"type": "enabled"})
        self.assertEqual(client.requests[1]["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(client.requests[2]["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual([e.get("thinking", True) for e in trace["events"] if "usage" in e], [True, False, False])

    def test_default_always_thinks_and_strong_wording(self):
        self.args.max_total_output, self.args.budget_notes, self.args.budget_soft = 100000, 0.01, 20000
        self.args.soft_wording = "strong"
        client = FakeClient([[chunk(calls=[call("1")], finish="tool_calls", tokens=30000)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=5)]])
        trace, _ = self.solve(client)
        self.assertTrue(all(r["extra_body"]["thinking"] == {"type": "enabled"} for r in client.requests))
        self.assertIn("submit your best answer now", trace["messages"][2]["content"])

    def test_verify_submit_rejects_once_then_accepts_same_code(self):
        self.args.verify_submit = True
        client = FakeClient([[chunk(calls=[call("solve(test_input)", "submit_answer")], finish="tool_calls", tokens=5)],
                             [chunk(calls=[call("solve(test_input)", "submit_answer")], finish="tool_calls", tokens=5)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("ok", [[4, 3]])), \
             patch("run_deepseek_tools.check_answer_on_train", return_value=([0], "pair 0: 2 cells differ")):
            trace, result = self.solve(client)
        tool_msgs = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("not accepted yet", tool_msgs[0])
        self.assertEqual((result["pred"], result["answer_source"], result["submit_verified"], result["submit_rejections"]),
                         ([[4, 3]], "submit", False, 1))

    def test_verify_submit_passes_through_verified(self):
        self.args.verify_submit = True
        client = FakeClient([[chunk(calls=[call("solve(test_input)", "submit_answer")], finish="tool_calls", tokens=5)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("ok", [[4, 3]])), \
             patch("run_deepseek_tools.check_answer_on_train", return_value=([], "")):
            _, result = self.solve(client)
        self.assertEqual((result["submit_verified"], result["submit_rejections"]), (True, 0))

    def test_lenient_parse(self):
        self.assertEqual(lenient_parse("[0, 1]\n[2, 3]\n"), [[0, 1], [2, 3]])
        self.assertEqual(lenient_parse("x [[1]] then [[4, 3]] and an explanation [2]"), [[4, 3]])
        self.assertEqual(lenient_parse("[[1, 2],\n [3, 4],\n]"), [[1, 2], [3, 4]])
        self.assertIsNone(lenient_parse("no grid [1, 2] here"))
        self.assertIsNone(lenient_parse("[[1, 2], [3]]"))

    def test_lenient_parse_only_when_enabled(self):
        text = "Answer:\n[4, 3]\n"
        with patch("run_deepseek_tools.submit_answer", return_value=("x", None)):
            _, result = self.solve(FakeClient([[chunk(content=text, finish="stop", tokens=5)]]))
        self.assertIsNone(result["pred"])
        self.args.lenient_parse = True
        with patch("run_deepseek_tools.submit_answer", return_value=("x", None)):
            _, result = self.solve(FakeClient([[chunk(content=text, finish="stop", tokens=5)]]))
        self.assertEqual((result["pred"], result["answer_source"]), ([[4, 3]], "lenient"))

    def test_old_prompts_unchanged(self):
        self.assertTrue(PROMPT_SUFFIXES["lean2sub"].startswith(PROMPT_SUFFIXES["lean2"]))
        self.assertNotIn("submit_answer", PROMPT_SUFFIXES["lean2"])


class TokenExactParseTests(unittest.TestCase):
    """E043 --token-exact: raw V4.1 completions parse like the model's encoding.parse_message_from_completion_text."""

    @staticmethod
    def call(name, params):
        body = "".join(f'<｜DSML｜ parameter name="{k}" string="{s}">{v}</｜DSML｜ parameter>\n' for k, (v, s) in params)
        return f'<｜DSML｜ invoke name="{name}">\n{body}</｜DSML｜ invoke>\n'

    def test_tool_calls_and_answer(self):
        from run_deepseek_tools import DS_EOS, parse_ds_completion
        text = ("look at rows</think>" + "\n\n<｜DSML｜ calls>\n"
                + self.call("run_python", [("code", ("print(1)\nx = '<a>'", "true"))])
                + self.call("submit_answer", [("code", ("answer", "true")), ("n", ("3", "false"))])
                + "</｜DSML｜ calls>" + DS_EOS)
        reasoning, content, calls = parse_ds_completion(text)
        self.assertEqual((reasoning, content), ("look at rows", ""))
        self.assertEqual([c["function"]["name"] for c in calls], ["run_python", "submit_answer"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"code": "print(1)\nx = '<a>'"})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"code": "answer", "n": 3})
        self.assertEqual(parse_ds_completion("ok</think>[[1]]" + DS_EOS), ("ok", "[[1]]", []))

    def test_format_errors_raise(self):
        from run_deepseek_tools import DS_EOS, parse_ds_completion
        for bad in ["no think end" + DS_EOS, "r</think>text without eos",
                    "r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"run_python\">\ncode</｜DSML｜ invoke>\n"
                    "</｜DSML｜ calls>" + DS_EOS]:
            with self.assertRaises(ValueError):
                parse_ds_completion(bad)


class SandboxClientTests(unittest.TestCase):
    def test_reset_restores_only_prelude_and_cleanup(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.method == "DELETE":
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"process_status": "completed", "stdout": "", "stderr": "",
                                             "new_session_created": len(requests) == 2})
        with SandboxSession("http://sandbox", TASK["train"], TASK["test"][0]["input"],
                            transport=httpx.MockTransport(handler)) as session:
            result = session.execute("print(123)")
            self.assertTrue(result["state_reset"])
            self.assertIn("state lost", tool_output(result))
        self.assertEqual(len(requests), 4)  # initialize, cell, restore prelude, delete
        self.assertEqual(json.loads(requests[0].content), json.loads(requests[2].content))
        self.assertEqual(requests[-1].method, "DELETE")

    def test_transport_error_does_not_retry_cell(self):
        requests = []
        def handler(request):
            requests.append(request)
            if len(requests) == 2:
                raise httpx.ReadTimeout("ambiguous cell")
            return httpx.Response(200, json={"process_status": "completed", "stdout": "", "stderr": ""})
        with self.assertRaises(httpx.ReadTimeout):
            with SandboxSession("http://sandbox", [], [[1]], transport=httpx.MockTransport(handler)) as session:
                session.execute("print(123)")
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[-1].method, "DELETE")

    def test_initialization_failure_also_cleans_up(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"process_status": "error", "stdout": "", "stderr": "oops"})
        with self.assertRaises(RuntimeError):
            with SandboxSession("http://sandbox", [], [[1]], transport=httpx.MockTransport(handler)):
                pass
        self.assertEqual(requests[-1].method, "DELETE")


@unittest.skipUnless(os.environ.get("ARC_SANDBOX_URL"), "Set ARC_SANDBOX_URL for Docker integration tests")
class DockerTests(unittest.TestCase):
    def session(self, timeout=5):
        return SandboxSession(os.environ["ARC_SANDBOX_URL"], TASK["train"], TASK["test"][0]["input"], timeout)

    def test_state_timeout_error_isolation_and_cleanup(self):
        with self.session() as first, self.session() as second:
            session_id = first.session_id
            self.assertEqual(first.execute("saved = 42; print(saved)")["stdout"].strip(), "42")
            self.assertEqual(second.execute("print('saved' in globals())")["stdout"].strip(), "False")
            self.assertEqual(first.execute("raise ValueError('recoverable')")["process_status"], "error")
            self.assertEqual(first.execute("print(saved)")["stdout"].strip(), "42")
            first.timeout = 0.2
            result = first.execute("import time; time.sleep(20)")
            self.assertEqual(result["process_status"], "timeout")
            self.assertFalse(result.get("state_reset", False))
            first.timeout = 5
            self.assertEqual(first.execute("print(saved)")["stdout"].strip(), "42")
            result = first.execute("import os; print(os.path.exists('/home/vu/Workspace/ARC-AGI-2/SECRETS.yaml'))")
            self.assertEqual(result["stdout"].strip(), "False")
            result = first.execute("import socket; socket.create_connection(('1.1.1.1', 443), timeout=1)")
            self.assertEqual(result["process_status"], "error")
        with httpx.Client(trust_env=False) as client:
            sessions = client.get(os.environ["ARC_SANDBOX_URL"] + "/sessions").json()["sessions"]
        self.assertNotIn(session_id, sessions)

    def test_submission_feedback_and_numpy_output(self):
        with self.session() as session:
            feedback, pred = submit_solution(session, "def transform(grid): return grid")
            self.assertIsNone(pred)
            self.assertIn("training_failures", feedback)
            feedback, pred = submit_solution(session, "import numpy as np\ndef transform(grid): return np.array(grid)[:, ::-1]")
            self.assertEqual(pred, [[4, 3]])
            feedback, pred = submit_solution(session, "def transform(grid): return [[10]]")
            self.assertIsNone(pred)
            self.assertIn("ValueError", feedback)
            # Rebinding train in the session must not let an unchecked transform through.
            session.execute("train = []")
            feedback, pred = submit_solution(session, "def transform(grid): return grid")
            self.assertIsNone(pred)
            self.assertIn("training_failures", feedback)

    def test_submit_answer_in_real_session(self):
        with self.session() as session:
            session.execute("import numpy as np\ndef solve(g):\n    print('noise' * 5000)\n    return np.array(g)[:, ::-1]")
            feedback, grid = submit_answer(session, "solve(test_input)")
            self.assertEqual(grid, [[4, 3]])
            self.assertIn("accepted", feedback)
            feedback, grid = submit_answer(session, "answer = solve(test_input).tolist()\nprint(1)")
            self.assertEqual(grid, [[4, 3]])
            feedback, grid = submit_answer(session, "answer")
            self.assertEqual(grid, [[4, 3]])
            feedback, grid = submit_answer(session, "[[10]]")
            self.assertIsNone(grid)
            self.assertIn("not accepted", feedback)
            feedback, grid = submit_answer(session, "undefined_name")
            self.assertIsNone(grid)
            self.assertIn("NameError", feedback)
            self.assertEqual(session.execute("print('_arc_answer' in globals())")["stdout"].strip(), "False")
        with self.session() as fresh:
            self.assertIsNone(submit_answer(fresh, "answer")[1])

    def test_check_answer_on_train_real_session(self):
        with self.session() as session:
            session.execute("def solve(g):\n    return [r[::-1] for r in g]\ndef bad(g):\n    return g")
            self.assertEqual(check_answer_on_train(session, "solve(test_input)"), ([], ""))
            failed, note = check_answer_on_train(session, "bad(test_input)")
            self.assertEqual(failed, [0])
            self.assertIn("cells differ", note)
            failed, note = check_answer_on_train(session, "undefined(test_input)")
            self.assertIn("NameError", note)
            self.assertIsNone(check_answer_on_train(session, "answer"))
            # the real test_input is untouched
            self.assertEqual(session.execute("print(test_input)")["stdout"].strip(), "[[3, 4]]")

    def test_mock_model_with_real_persistent_python(self):
        args = parser().parse_args(["--sandbox-url", os.environ["ARC_SANDBOX_URL"]])
        client = FakeClient([
            [chunk(calls=[call("def transform(g): return [r[::-1] for r in g]")], finish="tool_calls", tokens=5)],
            [chunk(calls=[call("print(transform(test_input))")], finish="tool_calls", tokens=5)],
            [chunk(content="[[4,3]]", finish="stop", tokens=5)],
        ])
        trace, result = solve(client, args, TASK, TASK["test"][0]["input"])
        self.assertNotIn("error", result)
        self.assertEqual(result["pred"], [[4, 3]])
        self.assertIn("[[4, 3]]", trace["messages"][-2]["content"])


if __name__ == "__main__":
    unittest.main()
