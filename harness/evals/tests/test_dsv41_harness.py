"""The dsv41 Kaggle path of the live harness (E046 paths: multi-grid submit, shadowed builtins, _plain, context limit /
ContextFull fallbacks, --multi-budget, budget notes, train-check labels) on the Kaggle runtime (LocalExecutor ipython
style, real child Python, a fake token-exact server). Ported from the 2026-10-26 deep audit
(kaggle/dsv41_tools/tests_mtf_v6/test_harness_new.py), plus one regression test per answer-path fix A1-A11 of
kaggle/dsv41_tools/next_candidate_fixes.md (each fails on the V6 harness a8a5b665) and E047 / E049 checks."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np  # noqa: F401  (only to make sure the tool python has numpy)

import dsv41_fakes as C

H = C.H
TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
         {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}]
T0, T1, T2 = [[1, 0], [0, 2]], [[3, 4, 5]], [[6], [7]]
F0, F1, F2 = [[0, 1], [2, 0]], [[5, 4, 3]], [[6], [7]]  # fliplr of T0, T1, T2
FLIP = "def solve(g):\n    return np.fliplr(np.array(g))\n"
TASK1 = {"train": TRAIN, "test": [{"input": T0}]}
TASK3 = {"train": TRAIN, "test": [{"input": T0}, {"input": T1}, {"input": T2}]}
MULTI = ["--multi-test", "--multi-budget", "0.5"]
run_solve = C.run_solve


class SubmitMulti(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = C.open_session(TRAIN, T0, [T0, T1, T2])
        r = cls.s.execute(FLIP)
        assert r["process_status"] == "completed", r

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def sub(self, code, n=3):
        return H.submit_answer(self.s, code, n)

    def test_list_of_numpy_grids(self):
        out, g = self.sub("[solve(t) for t in test_inputs]")
        self.assertEqual(g, [F0, F1, F2], out)
        self.assertIn("answer accepted: 3 grids, 2x2, 1x3, 2x1", out)

    def test_nested_numpy_rows_and_ints(self):
        out, g = self.sub("[[np.int64(1)], [np.int8(2)], [np.array([3, 4])[0:1]]]")  # not grids: rows are not lists
        self.assertIsNone(g)
        out, g = self.sub("[[list(np.array(r)) for r in t] for t in test_inputs]")  # lists of np.int64
        self.assertEqual(g, [T0, T1, T2], out)
        out, g = self.sub("[[np.array(r) for r in t] for t in test_inputs]")  # lists of numpy rows
        self.assertEqual(g, [T0, T1, T2], out)

    def test_tuple_and_statements(self):
        out, g = self.sub("tuple(solve(t) for t in test_inputs)")
        self.assertEqual(len(g), 3, out)
        self.s.execute("answer = [solve(t).tolist() for t in test_inputs]")
        out, g = self.sub("answer")
        self.assertEqual(g[0], F0, out)
        out, g = self.sub("res = [solve(t) for t in test_inputs]\nanswer = res")
        self.assertEqual(g[2], F2, out)

    def test_wrong_n_and_shapes(self):
        for code, want in (("[solve(t) for t in test_inputs[:2]]", "got a list of 2 items"),
                           ("[solve(t) for t in test_inputs] + [test_input]", "got a list of 4 items"),
                           ("solve(test_input)", "got one grid"),
                           ("[]", "got a list of 0 items"),
                           ("None", "got NoneType"),
                           ("{'a': 1}", "got dict"),
                           ("[test_input, [[1, 2], [3]], test_input]", "got a list of 3 items"),
                           ("[test_input, [[10]], test_input]", "got a list of 3 items"),
                           ("[test_input, [[-1]], test_input]", "got a list of 3 items"),
                           ("[test_input, [[True]], test_input]", "got a list of 3 items"),
                           ("[test_input, [[1.0]], test_input]", "got a list of 3 items"),
                           ("[test_input, [], test_input]", "got a list of 3 items"),
                           ("[test_input, [[1] * 31], test_input]", "got a list of 3 items")):
            out, g = self.sub(code)
            self.assertIsNone(g, code)
            self.assertIn("[answer not accepted] this task has 3 test inputs", out, code)
            self.assertIn(want, out, code)

    def test_a11_rejection_names_the_bad_grid(self):
        # A11 (fails on V6: "got a list of 3 items" only): the message names the bad grid and why
        for code, want in (("[test_input, test_input, [[1, 2], [3]]]", "grid 2 (test input 2): 2 rows, row lengths [1, 2]"),
                           ("[test_input, [[10]], test_input]", "grid 1 (test input 1): value 10 is not an int 0-9"),
                           ("[test_input, [[True]], test_input]", "grid 1 (test input 1): value true is not an int 0-9"),
                           ("[None, test_input, [[1] * 31]]", "grid 0 (test input 0): NoneType; grid 2 (test input 2): "
                                                               "1 rows, row lengths [31]")):
            out, g = self.sub(code)
            self.assertIsNone(g, code)
            self.assertIn(want, out, code)

    def test_errors_and_undefined(self):
        out, g = self.sub("undefined_fn(test_input)")
        self.assertIsNone(g)
        self.assertIn("NameError", out)
        out, g = self.sub("x = 1")  # `answer` may exist in this shared session: either way never a crash
        self.assertTrue(g is None or len(g) == 3)

    def test_a11_answers_beyond_the_tool_output_limit(self):
        # ARC-AGI-2 has at most 4 test inputs per task; 4 x 30x30 always fitted in max_tool_output (16000). A11 (fails on
        # V6: the marker line was cut and the answer rejected as "got NoneType"): 6 x 30x30 grids, and a session whose
        # output limit is smaller than one grid, are read back in slices.
        s = C.open_session(TRAIN, T0, [[[1] * 30] * 30] * 4)
        try:
            out, g = H.submit_answer(s, "[np.full((30, 30), 9) for t in test_inputs]", 4)
            self.assertIsNotNone(g, out)
            out, g = H.submit_answer(s, "[np.full((30, 30), i % 10) for i in range(6)]", 6)
            self.assertEqual(g, [[[i] * 30] * 30 for i in range(6)], out)
            r = s.execute("print([k for k in globals() if k.startswith('_arc_text')])")
            self.assertEqual(r["stdout"].strip(), "[]")  # the slice buffer is removed afterwards
        finally:
            s.close()
        s = C.open_session(TRAIN, T0, max_output=1200)
        try:
            out, g = H.submit_answer(s, "np.arange(900).reshape(30, 30) % 10")
            self.assertEqual(g, (np.arange(900).reshape(30, 30) % 10).tolist(), out)
        finally:
            s.close()


class Shadowing(unittest.TestCase):
    """E046 run 3: a session `eval` broke submit. Every builtin used by the answer cells comes from `builtins`."""

    def test_shadowed_builtins_single_and_multi(self):
        s = C.open_session(TRAIN, T0, [T0, T1])
        try:
            s.execute(FLIP + "def eval(grid, H=0, bg=0):\n    return 'model eval'\n"
                      "def compile(*a, **k):\n    raise RuntimeError('model compile')\n"
                      "def exec(*a, **k):\n    raise RuntimeError('model exec')\n"
                      "def globals():\n    return {}\n"
                      "def hasattr(*a):\n    raise RuntimeError('model hasattr')\n"
                      "def isinstance(*a):\n    raise RuntimeError('model isinstance')\n"
                      "def getattr(*a):\n    raise RuntimeError('model getattr')\n"
                      "list = tuple = None\nlen = None\nstr = int = None\nNameError = SyntaxError = KeyError\n"
                      "answer = [solve(t) for t in test_inputs]\n")
            out, g = H.submit_answer(s, "[solve(t) for t in test_inputs]", 2)
            self.assertEqual(g, [F0, F1], out)
            out, g = H.submit_answer(s, "answer", 2)
            self.assertIsNotNone(g, out)
            out, g = H.submit_answer(s, "z = 3\nanswer = [solve(t) for t in test_inputs]", 2)
            self.assertIsNotNone(g, out)
            out, g = H.submit_answer(s, "solve(test_input)")
            self.assertEqual(g, F0, out)
            s.execute("del answer")
            out, g = H.submit_answer(s, "y = 2")  # exec path without answer: NameError via _b.NameError
            self.assertIsNone(g)
            self.assertIn("code did not define answer", out)
        finally:
            s.close()

    def test_a7_shadowed_print(self):
        # A7 (fails on V6: "no value returned"): the answer cell's final print is builtins.print
        s = C.open_session(TRAIN, T0)
        try:
            s.execute(FLIP + "def print(*a, **k):\n    pass\n")
            out, g = H.submit_answer(s, "solve(test_input)")
            self.assertEqual(g, F0, out)
        finally:
            s.close()


class SingleTestPlain(unittest.TestCase):
    """V6 applied _plain to single-test answers too; compared with the v3 kit's harness 1ad5ee24 (what V5 runs)."""

    @classmethod
    def setUpClass(cls):
        cls.v3 = C.load_v3_harness()
        cls.s = C.open_session(TRAIN, T0)
        cls.s.execute(FLIP)

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_a10_same_as_v3_except_numpy_rows(self):
        # A10 (fails on V6: nested np.bool_ was rejected, v3 read it as 0/1): only lists of numpy rows differ from v3
        cases = {
            "solve(test_input)": "same",
            "solve(test_input).tolist()": "same",
            "[list(r) for r in solve(test_input)]": "same",  # lists of np.int64: default=int vs tolist
            "[r for r in solve(test_input)]": "fixed",  # list of numpy rows: v3 json error, now accepted
            "tuple(tuple(r) for r in test_input)": "same",
            "[[np.float64(1.0)]]": "same",
            "[[True]]": "same",
            "[[np.bool_(True), np.bool_(False)]]": "same",  # v3: default=int -> [[1, 0]]
            "np.array([[True]])": "same",
            "[np.array([1, 2]), [np.bool_(False), np.int8(3)]]": "fixed",
        }
        for code, kind in cases.items():
            o_new, g_new = H.submit_answer(self.s, code)
            o_old, g_old = self.v3.submit_answer(self.s, code)
            if kind == "same":
                self.assertEqual(g_new, g_old, code)
            else:
                self.assertIsNone(g_old, code)
                self.assertIsNotNone(g_new, code)
        self.assertEqual(H.submit_answer(self.s, "[[np.bool_(True), np.bool_(False)]]")[1], [[1, 0]])


class Prompts(unittest.TestCase):
    def test_build_prompt_multi(self):
        p = H.build_prompt_multi(TRAIN, [T0, T1, T2])
        for i, g in enumerate([T0, T1, T2]):
            self.assertIn(f"--Test input {i}--\n\n{json.dumps(g)}", p)
        self.assertEqual(p.count("--End of Test Input--"), 1)
        self.assertNotIn("[[0]]\n--End of Test Input--", p)
        self.assertTrue(p.startswith(H.build_prompt(TRAIN, T0).split("--Test Input--")[0]))
        p = H.build_prompt_multi(TRAIN, [[[0]], [[0]]])  # a test input that is literally [[0]] still renders
        self.assertIn("--Test input 1--\n\n[[0]]\n--End of Test Input--", p)

    def test_a5_single_test_session_defines_test_inputs(self):
        # A5 (fails on V6: NameError): with --multi-test every tool description says test_inputs is preloaded, so
        # single-test sessions define it as [test_input]; without --multi-test nothing changes
        self.assertIn("test_inputs", H.tools_for(C.harness_args(MULTI))[0]["function"]["description"])
        self.assertNotIn("test_inputs", H.tools_for(C.harness_args([]))[0]["function"]["description"])
        code = "print(len(test_inputs), test_inputs[0] == test_input, test_inputs[0] is test_input)"
        srv, trace, result = run_solve([C.ds_call_text(code), C.ds_call_text(FLIP + "answer = solve(test_input)"),
                                        C.ds_call_text("answer", name="submit_answer")], C.harness_args(MULTI), TASK1)
        tool = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("1 True False", tool[0])
        self.assertEqual((result["pred"], result["train_pass"]), (F0, True))
        srv, trace, result = run_solve([C.ds_call_text(code)], C.harness_args([]), TASK1, completion_tokens=10 ** 6)
        self.assertIn("NameError", [m["content"] for m in trace["messages"] if m["role"] == "tool"][0])


def first_prompt_len(argv, task, test_inputs=None, pad=0):
    """Prompt tokens of turn 1 (from the ContextFull message with a limit of 1)."""
    srv, trace, result = run_solve([], C.harness_args(argv + ["--context-limit", "1"]), task, test_inputs, pad=pad)
    ev = [e for e in trace["events"] if "context_full" in e][0]["context_full"]
    return int(ev.split()[1])


class ContextLimit(unittest.TestCase):
    def test_room_arithmetic(self):
        args = C.harness_args(["--context-limit", "10000"])
        tok = {"ids": None, "n": 0, "prev_len": 0, "prev_sampled": [], "resync": False, "turns": 0, "breaks": 0,
               "rerenders": 0, "resyncs": 0, "parse_fallbacks": 0}
        msgs = [{"role": "user", "content": "x"}]
        n0 = len(C.FakeTokenServer._render(msgs, True, 0))
        for pad, expect in ((0, 10000 - n0 - 16), (10000 - n0 - 16 - 5000, 5000), (10000 - n0 - 16 - 1024, 1024),
                            (10000 - n0 - 16 - 1023, None), (20000, None)):
            srv = C.FakeTokenServer([C.ds_text_answer("[[1]]")], pad=pad)
            with patch("httpx.post", srv.post):
                if expect is None:
                    with self.assertRaises(H.ContextFull):
                        H.turn(srv, args, msgs, 64000, False, True, dict(tok))
                    self.assertEqual(srv.requests, [])  # nothing sent past the limit
                else:
                    H.turn(srv, args, msgs, 64000, False, True, dict(tok))
                    self.assertEqual(srv.requests, [(n0 + pad, min(64000, expect))])
                    self.assertLessEqual(srv.requests[0][0] + srv.requests[0][1], 10000)

    def test_limit_off_keeps_max_tokens(self):
        args = C.harness_args([])
        self.assertEqual(args.context_limit, 0)
        srv, trace, result = run_solve([C.ds_call_text("answer = solve(test_input)"),
                                        C.ds_call_text("solve(test_input)", name="submit_answer")],
                                       args, TASK1, pad=200000)
        self.assertEqual(result["pred"], None)  # FLIP not defined -> errors, but requests go out unchanged
        self.assertTrue(all(m == 64000 for _, m in srv.requests), srv.requests)

    def test_context_full_takes_session_answer_single(self):
        body = C.ds_call_text(FLIP + "answer = solve(test_input)\n" + "#" * 6000)
        n1 = first_prompt_len([], TASK1, pad=500)
        args = C.harness_args(["--context-limit", str(n1 + 16 + len(body) + 200)])  # turn 1 fits, turn 2 cannot
        srv, trace, result = run_solve([body], args, TASK1, pad=500)
        self.assertEqual(len(srv.requests), 1)
        self.assertEqual(result["stop_reason"], "context", result)
        self.assertEqual(result["pred"], F0)
        self.assertEqual(result["answer_source"], "session")
        self.assertEqual(result["turns"], 1)
        self.assertNotIn("error", result)
        self.assertTrue(any("context_full" in e for e in trace["events"]))
        self.assertIn(result["train_trace"], ("cell", "fn", "expr", "transcript"))

    def test_context_full_multi_session_answer_and_fallback_solve(self):
        body = C.ds_call_text(FLIP + "answer = [solve(t) for t in test_inputs]\n" + "#" * 6000)
        n1 = first_prompt_len(MULTI, TASK3, [T0, T1, T2], pad=300)
        args = C.harness_args(MULTI + ["--context-limit", str(n1 + 16 + len(body) + 200)])
        srv, trace, result = run_solve([body], args, TASK3, [T0, T1, T2], pad=300)
        self.assertEqual(result["stop_reason"], "context")
        self.assertEqual(result["pred"], [F0, F1, F2])
        self.assertEqual(result["multi_test"], 3)
        body = C.ds_call_text(FLIP + "#" * 6000 + "\n#" + "x" * 40)  # no `answer`, only solve: --fallback-solve
        srv, trace, result = run_solve([body], args, TASK3, [T0, T1, T2], pad=300)
        self.assertEqual(result["stop_reason"], "context")
        self.assertEqual(result["answer_source"], "session_solve")
        self.assertEqual(result["pred"], [F0, F1, F2])

    def test_context_full_first_turn(self):
        args = C.harness_args(["--multi-test", "--context-limit", "4000"])
        srv, trace, result = run_solve([], args, TASK3, [T0, T1, T2], pad=5000)
        self.assertEqual(result["stop_reason"], "context")
        self.assertIsNone(result["pred"])
        self.assertEqual(result["turns"], 0)
        self.assertNotIn("error", result)
        self.assertEqual(srv.requests, [])

    def test_capped_turn_cut_then_context(self):
        # room between 1024 and max_tokens: the turn is capped, cut by length, and the next turn stops on context
        n1 = first_prompt_len([], TASK1, pad=1500)
        args = C.harness_args(["--context-limit", str(n1 + 3000)])
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = solve(test_input)\n"),
                                        lambda n, m: "x" * (m + 50)], args, TASK1, pad=1500)
        self.assertLess(srv.requests[1][1], 64000)
        self.assertEqual(srv.requests[1][1], n1 + 3000 - srv.requests[1][0] - 16)
        self.assertEqual(result["stop_reason"], "context")
        self.assertEqual(result["pred"], F0)

    def test_a9_budget_note_before_the_context_ends(self):
        # A9 (fails on V6: the note said "hard limit 160k" and never "nearly spent" although only ~6k more output
        # fit before the 163,840 context): the notes count the output that still fits
        n1 = first_prompt_len(MULTI, TASK3, [T0, T1, T2])
        limit = n1 + 5000 + 16 + H.CONTEXT_MIN_ROOM + 1000
        a = C.harness_args(MULTI + ["--context-limit", str(limit)])
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "print(1)"),
                                        C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")],
                                       a, TASK3, [T0, T1, T2], completion_tokens=5000)
        tool = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("[budget: 5k of 6k output tokens used] Budget nearly spent", tool[0])
        self.assertEqual((result["stop_reason"], result["pred"]), ("submitted", [F0, F1, F2]))
        # far from the limit the notes are unchanged
        a = C.harness_args(MULTI + ["--context-limit", "1000000"])
        srv, trace, result = run_solve([C.ds_call_text("print(1)"),
                                        C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")],
                                       a, TASK3, [T0, T1, T2], completion_tokens=5000)
        tool = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("[budget: 5k output tokens used; aim to finish within 80k (hard limit 160k)]", tool[0])


def chat_args(extra=()):
    """The solver's harness argv on the image (chat) path: --grid-format text+image instead of --token-exact."""
    a = C.harness_args(list(extra) + ["--grid-format", "text+image", "--image-cell", "0"])
    a.token_exact = False
    return a


def run_chat(turns, args, task, limit, pad=0, image_tokens=900):
    srv = C.FakeChatServer(turns, limit, pad=pad, image_tokens=image_tokens)
    trace, result = H.solve(srv, args, task, task["test"][0]["input"], session_factory=C.local_factory)
    return srv, trace, result


class ChatContextLimit(unittest.TestCase):
    """--context-limit on the chat path (images): before 2026-10-29 it was applied only on the token-exact path, so an
    image attempt whose prompt grew past max-model-len - max_tokens failed with the server's 400 (E066 full_ref: 68 of
    334 records, 19 of 120 tasks, at max-model-len 131,072)."""
    LONG = FLIP + "answer = solve(test_input)\n" + "#" * 6000  # a turn-1 call whose tool round trip adds ~6k tokens

    def first_prompt(self, pad):
        srv, _, _ = run_chat([("submit_answer", "solve(test_input)")], chat_args(), TASK1, 10 ** 9, pad=pad)
        return srv.requests[0][0]

    def test_message_bound_is_an_upper_bound(self):
        srv = C.FakeChatServer([], 0)
        for m in ({"role": "tool", "content": "x" * 5000}, {"role": "tool", "content": "é€😀" * 100},
                  {"role": "assistant", "content": "", "reasoning_content": "r" * 300,
                   "tool_calls": [{"id": "c", "type": "function", "function": {"name": "run_python",
                                                                               "arguments": json.dumps({"code": "1"})}}]},
                  {"role": "user", "content": [{"type": "text", "text": "a"},
                                               {"type": "image_url", "image_url": {"url": "data:"}}]}):
            self.assertGreaterEqual(H.message_token_bound(m), srv.prompt_tokens([m], []) - len("<assistant><think>"))

    def test_image_turn_past_the_limit_errors_without_the_flag(self):
        n1 = self.first_prompt(500)
        limit = n1 + 64000 + 3000  # turn 1 fits; turn 2 (+~6k tokens of round trip) at max_tokens 64000 does not
        srv, trace, result = run_chat([("run_python", self.LONG), ("submit_answer", "solve(test_input)")],
                                      chat_args(), TASK1, limit, pad=500)
        self.assertEqual(result["stop_reason"], "error")
        self.assertIn("maximum context length", result["error"])

    def test_image_turn_capped_to_the_room_left(self):
        n1 = self.first_prompt(500)
        limit = n1 + 64000 + 3000
        srv, trace, result = run_chat([("run_python", self.LONG), ("submit_answer", "solve(test_input)")],
                                      chat_args(["--context-limit", str(limit)]), TASK1, limit, pad=500)
        self.assertEqual((result["stop_reason"], result["pred"]), ("submitted", F0), result)
        self.assertNotIn("error", result)
        self.assertLess(srv.requests[1][1], 64000)
        self.assertTrue(all(n + m <= limit for n, m in srv.requests), srv.requests)
        slack = limit - sum(srv.requests[1]) - 16 - H.CHAT_CONTEXT_MARGIN
        self.assertLess(slack, 2 * H.CHAT_MESSAGE_TOKENS + 200, srv.requests)  # the bound is tight for ASCII text

    def test_context_full_takes_session_answer(self):
        # measure turn 1 (prompt + output tokens, its tool message), then set the limit so that the chat bound leaves
        # one token less than CONTEXT_MIN_ROOM for turn 2
        # (a short call with a long tool output, images at the bound's own size: turn 1's cap fits its output)
        code = FLIP + "answer = solve(test_input)\nprint('#' * 6000)"
        srv, trace, _ = run_chat([("run_python", code), ("submit_answer", "solve(test_input)")], chat_args(),
                                 TASK1, 10 ** 9, pad=500, image_tokens=H.CHAT_IMAGE_TOKENS)
        used = srv.requests[0][0] + len("thinking" + "run_python" + json.dumps({"code": code}))
        tool = [m for m in trace["messages"] if m["role"] == "tool"][0]
        self.assertGreater(H.message_token_bound(tool), 6000)
        limit = used + H.message_token_bound(tool) + H.CHAT_CONTEXT_MARGIN + 16 + H.CONTEXT_MIN_ROOM - 1
        srv, trace, result = run_chat([("run_python", code)], chat_args(["--context-limit", str(limit)]),
                                      TASK1, limit, pad=500, image_tokens=H.CHAT_IMAGE_TOKENS)
        self.assertEqual(len(srv.requests), 1)
        self.assertEqual((result["stop_reason"], result["pred"], result["answer_source"]), ("context", F0, "session"))
        self.assertNotIn("error", result)
        self.assertTrue(any("chat bound" in e.get("context_full", "") for e in trace["events"]))

    def test_first_turn_bounded(self):
        # pad = template text the bound cannot see (here < CHAT_TEMPLATE_TOKENS)
        n1 = self.first_prompt(1500)
        srv, trace, result = run_chat([], chat_args(["--context-limit", str(n1 + 500)]), TASK1, n1 + 500, pad=1500)
        self.assertEqual((result["stop_reason"], result["turns"], srv.requests), ("context", 0, []))
        self.assertNotIn("error", result)
        limit = n1 + 30000  # turn 1 capped below 64000 by the bound, and the request fits
        srv, trace, result = run_chat([("submit_answer", "solve(test_input)")],
                                      chat_args(["--context-limit", str(limit)]), TASK1, limit, pad=1500)
        self.assertLess(srv.requests[0][1], 64000)
        self.assertLessEqual(sum(srv.requests[0]), limit)

    def test_token_exact_with_images_is_rejected(self):
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            with patch("sys.argv", ["x", "--token-exact", "--grid-format", "text+image", "--base-url", "http://s/v1",
                                    "--sandbox-url", "http://s"]):
                H.main()


class MultiBudget(unittest.TestCase):
    def test_scaling_n1_to_n6(self):
        base = C.harness_args(MULTI)
        snapshot = copy.deepcopy(vars(base))
        seen = {}

        def fake_turn(client, args, messages, cap, final, think, tok):
            seen["a"] = (args.max_total_output, args.budget_soft, args.max_turns, cap)
            raise H.ContextFull("stop")
        for n in range(1, 7):
            for f, argv in ((0.5, ["--multi-budget", "0.5"]), (0.25, ["--multi-budget", "0.25"]), (0.0, []),
                            (0.3, ["--multi-budget", "0.3"])):
                a = C.harness_args(["--multi-test"] + argv)
                task = {"train": TRAIN, "test": [{"input": T0}] * n}
                with patch.object(H, "turn", fake_turn):
                    H.solve(None, a, task, T0, session_factory=C.local_factory,
                            **({"test_inputs": [T0] * n} if n > 1 else {}))
                scale = 1 + f * (n - 1) if n > 1 else 1
                self.assertEqual(seen["a"][0], int(80000 * scale), (n, f))
                self.assertEqual(seen["a"][1], int(40000 * scale), (n, f))
                self.assertEqual(seen["a"][2], int(round(40 * scale)), (n, f))
                self.assertEqual(seen["a"][3], 64000)  # per-turn cap unchanged
                self.assertEqual(vars(a)["max_total_output"], 80000)  # the shared args are never mutated
        self.assertEqual(vars(base), snapshot)

    def test_budget_notes_use_scaled_budget(self):
        a = C.harness_args(MULTI)
        srv, trace, result = run_solve([C.ds_call_text("print(1)"), C.ds_call_text(FLIP),
                                        C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")],
                                       a, TASK3, [T0, T1, T2], completion_tokens=50000)
        tool_msgs = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("aim to finish within 80k (hard limit 160k)", tool_msgs[0])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(len(result["pred"]), 3)


class MultiFlow(unittest.TestCase):
    def test_text_answer_in_multi_uses_session(self):
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = [solve(t) for t in test_inputs]"),
                                        C.ds_text_answer("[[0, 1], [2, 0]]")], C.harness_args(MULTI), TASK3,
                                       [T0, T1, T2])
        self.assertEqual(result["stop_reason"], "final")
        self.assertEqual(result["answer_source"], "session")
        self.assertEqual(len(result["pred"]), 3)

    def test_single_test_messages_identical_with_multi_flag(self):
        # the single-test prompt is unchanged by --multi-test; only the tool descriptions (and A5's test_inputs) differ
        outs = {}
        for flag in ([], MULTI):
            a = C.harness_args(flag + ["--context-limit", "163840"])
            srv, trace, result = run_solve([C.ds_call_text(FLIP),
                                            C.ds_call_text("solve(test_input)", name="submit_answer")], a, TASK1)
            outs[bool(flag)] = (trace["messages"][:2], result["pred"], result["turns"], result["stop_reason"],
                                srv.requests[0][1])
        self.assertEqual(outs[False], outs[True])

    def test_train_check_multi_labels(self):
        a = C.harness_args(MULTI)
        srv, trace, result = run_solve([C.ds_call_text(FLIP),
                                        C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")],
                                       a, TASK3, [T0, T1, T2])
        self.assertEqual((result["train_pass"], result["train_trace"]), (True, "expr"))  # A6: code passed

    def test_a6_train_check_replays_with_test_inputs(self):
        # A6 (fails on V6: the replay kept the real test_inputs and labelled this correct program False): the
        # transcript replay binds test_inputs[0] to each training input and labels grid 0 of the answer list
        a = C.harness_args(MULTI)
        srv, trace, result = run_solve([C.ds_call_text("out0 = [r[::-1] for r in test_inputs[0]]\n"
                                                       "answer = [[r[::-1] for r in t] for t in test_inputs]"),
                                        C.ds_call_text("answer", name="submit_answer")], a, TASK3, [T0, T1, T2])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual((result["train_trace"], result["train_pass"]), ("transcript", True))
        # the same with the program as the submitted expression (expr path, test_inputs[0] = each training input)
        srv, trace, result = run_solve([C.ds_call_text("flip = lambda g: [r[::-1] for r in g]"),
                                        C.ds_call_text("[flip(test_inputs[0])] + [flip(t) for t in test_inputs[1:]]",
                                                       name="submit_answer")], a, TASK3, [T0, T1, T2])
        self.assertEqual((result["train_trace"], result["train_pass"]), ("expr", True))
        # and a wrong rule still fails
        srv, trace, result = run_solve([C.ds_call_text("answer = [[r[:] for r in t] for t in test_inputs]"),
                                        C.ds_call_text("answer", name="submit_answer")], a, TASK3, [T0, T1, T2])
        self.assertIs(result["train_pass"], False)
        self.assertEqual(H.multi_check_code("answer", [T0, T1]), "answer[0]")
        self.assertIsNone(H.multi_check_code(None, [T0, T1]))


def budget_end(code):
    """Turn 1 runs `code` and reports 10^6 completion tokens: the attempt then ends on its output budget."""
    return [C.ds_call_text(code)]


class Fallbacks(unittest.TestCase):
    """Session fallbacks after a budget end / cut / error (A1-A4, A8)."""

    def test_a1_partial_multi_answer(self):
        # A1 (fails on V6: one invalid grid in `answer` rejected the whole list -> pred None for all 3 pairs)
        a = C.harness_args(MULTI)
        code = "answer = [np.fliplr(np.array(t)).tolist() for t in test_inputs]\nanswer[1] = [[1, 2], [3]]"
        srv, trace, result = run_solve(budget_end(code), a, TASK3, [T0, T1, T2], completion_tokens=10 ** 6)
        self.assertEqual((result["stop_reason"], result["pred"]), ("output_budget", [F0, None, F2]))
        self.assertEqual(result["answer_source"], "session")
        recs = H.pair_records({"task_id": "t", "pair": 0, "attempt": 0, **result},
                              {"test": [{"input": T0, "output": F0}, {"input": T1, "output": F1},
                                        {"input": T2, "output": F2}]}, 3)
        self.assertEqual([(r["pair"], r["pred"] is not None, r["correct"]) for r in recs],
                         [(0, True, True), (1, False, False), (2, True, True)])
        ev = [e for e in trace["events"] if e.get("fallback") == "session_answer"][0]
        self.assertIn("2 of 3 grids valid; grid 1: 2 rows, row lengths [1, 2]", ev["output"])

    def test_a1_solve_per_test_input(self):
        # A1 (fails on V6: a solve that raises on one input lost every grid): solve() runs per input, errors caught
        a = C.harness_args(MULTI)
        code = ("def solve(g):\n    assert len(g) > 1, 'one row'\n    return [r[::-1] for r in g]\n"
                "answer = [[[0]]] * 2\n")  # wrong length: not usable, solve fills in
        srv, trace, result = run_solve(budget_end(code), a, TASK3, [T0, T1, T2], completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["answer_source"]), ([F0, None, F2], "session_solve"))
        self.assertNotEqual(result["train_trace"], "no_answer_0")  # grid 0 exists: labelled
        # `answer` covers pairs 0 and 2, solve the rest
        code = ("def solve(g):\n    return [r[::-1] for r in g]\n"
                "answer = [solve(test_inputs[0]), None, [[1, 2], [3]]]\n")
        srv, trace, result = run_solve(budget_end(code), a, TASK3, [T0, T1, T2], completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["answer_source"]), ([F0, F1, F2], "session"))
        # no grid 0: kept, but not labelled
        code = "def solve(g):\n    assert len(g) != 2\n    return [r[::-1] for r in g]\n"
        srv, trace, result = run_solve(budget_end(code), a, TASK3, [T0, T1, T2], completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["train_trace"]), ([None, F1, None], "no_answer_0"))

    def test_a2_session_answer_beats_a_cut_turns_text(self):
        # A2 (fails on V6: pred [[9, 9]] from the cut turn's text; the session answer was never read)
        a = C.harness_args(["--max-total-output", "3000"])
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = solve(test_input)"),
                                        lambda n, m: "x</think>Answer: [[9, 9]]\n" + "y" * (m + 100)], a, TASK1)
        self.assertEqual([e["finish_reason"] for e in trace["events"] if "finish_reason" in e], ["tool_calls", "length"])
        self.assertEqual((result["stop_reason"], result["pred"], result["answer_source"]),
                         ("output_budget", F0, "session"))
        # without a session answer the cut turn's grid still counts
        srv, trace, result = run_solve([C.ds_call_text("x = 1"),
                                        lambda n, m: "x</think>Answer: [[9, 9]]\n" + "y" * (m + 100)], a, TASK1)
        self.assertEqual(result["pred"], [[9, 9]])
        self.assertIn(result["answer_source"], ("parse", "lenient"))

    def test_a3_numpy_only_solve(self):
        # A3 (fails on V6: solve(test_input) got a list -> AttributeError, pred None)
        code = "def solve(g):\n    h, w = g.shape\n    return g[:, ::-1]\n"
        srv, trace, result = run_solve(budget_end(code), C.harness_args([]), TASK1, completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["answer_source"]), (F0, "session_solve"))
        srv, trace, result = run_solve(budget_end(code), C.harness_args(MULTI), TASK3, [T0, T1, T2],
                                       completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["answer_source"]), ([F0, F1, F2], "session_solve"))

    def test_a4_error_after_answer(self):
        # A4 (fails on V6: an error in a turn gave pred None although the session held `answer`)
        for fail in ({"tokenize_fail_after": 1}, {"turns_extra": ConnectionError("Connection refused")}):
            turns = [C.ds_call_text(FLIP + "answer = solve(test_input)")] + (
                [fail["turns_extra"]] if "turns_extra" in fail else [])
            srv, trace, result = run_solve(turns, C.harness_args([]), TASK1,
                                           tokenize_fail_after=fail.get("tokenize_fail_after"))
            self.assertEqual((result["stop_reason"], result["pred"], result["answer_source"]),
                             ("error", F0, "session"), result)
            self.assertIn("error", result)
            self.assertFalse(result["usage_complete"])
            self.assertEqual(result["train_pass"], True)
        # multi: the error path also reads `answer` per test input
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = [solve(t) for t in test_inputs]")],
                                       C.harness_args(MULTI), TASK3, [T0, T1, T2], tokenize_fail_after=1)
        self.assertEqual((result["stop_reason"], result["pred"]), ("error", [F0, F1, F2]))
        # turn mode (no session fallback) is unchanged: the error ends the attempt with pred None
        args = C.harness_args(["--final-mode", "turn"])
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = solve(test_input)")], args, TASK1,
                                       tokenize_fail_after=1)
        self.assertEqual((result["stop_reason"], result["pred"]), ("error", None))

    def test_a8_fallbacks_use_host_inputs(self):
        # A8 (fails on V6: the fallback read the rebound test_input / test_inputs)
        code = FLIP + "test_input = [[9]]\n"
        srv, trace, result = run_solve(budget_end(code), C.harness_args([]), TASK1, completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["answer_source"]), (F0, "session_solve"))
        code = FLIP + "test_inputs = [[[9]]] * 3\ntest_input = [[8]]\n"
        srv, trace, result = run_solve(budget_end(code), C.harness_args(MULTI), TASK3, [T0, T1, T2],
                                       completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["answer_source"]), ([F0, F1, F2], "session_solve"))


class E047E049(unittest.TestCase):
    """--alt-answer (E049) and --prior-mode (E047) on the paths the fallback changes touch."""

    def test_alt_answer_after_session_fallback(self):
        a = C.harness_args(["--alt-answer", "passive"])
        srv, trace, result = run_solve(budget_end(FLIP + "answer = solve(test_input)\nalt_answer = [[5]]"), a, TASK1,
                                       completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["alt_pred"]), (F0, [[5]]))
        self.assertIn("Optional second guess", trace["messages"][1]["content"])
        srv, trace, result = run_solve(budget_end(FLIP + "answer = solve(test_input)\nalt_answer = answer"), a,
                                       TASK1, completion_tokens=10 ** 6)
        self.assertEqual((result["pred"], result["alt_pred"]), (F0, None))  # same as pred: dropped
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = solve(test_input)\nalt_answer = [[5]]")],
                                       a, TASK1, tokenize_fail_after=1)  # A4 error path
        self.assertEqual((result["stop_reason"], result["pred"], result["alt_pred"]), ("error", F0, [[5]]))

    def test_alt_answer_multi_records(self):
        a = C.harness_args(MULTI + ["--alt-answer", "active"])
        code = FLIP + "answer = [solve(t) for t in test_inputs]\nalt_answer = [[[1]], [[2]], [[3]]]"
        srv, trace, result = run_solve(budget_end(code), a, TASK3, [T0, T1, T2], completion_tokens=10 ** 6)
        self.assertEqual(result["alt_pred"], [[[1]], [[2]], [[3]]])
        task = {"test": [{"input": T0, "output": F0}, {"input": T1, "output": [[2]]}, {"input": T2, "output": F2}]}
        recs = H.pair_records({"task_id": "t", "pair": 0, "attempt": 0, **result}, task, 3)
        self.assertEqual([(r["alt_pred"], r["correct"], r["alt_correct"]) for r in recs],
                         [([[1]], True, False), ([[2]], False, True), ([[3]], True, False)])

    def test_prior_note_and_load_prior(self):
        a = C.harness_args(["--prior-mode", "verify", "--prior-file", "/dev/null"])
        prior = {"preds": [[[4, 4]]], "train_pass": True, "program": None}
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = solve(test_input)")], a, TASK1,
                                       completion_tokens=10 ** 6, prior=prior)
        prompt = trace["messages"][1]["content"]
        self.assertIn("(its program reproduced every training pair)", prompt)
        self.assertTrue(prompt.endswith("--Previous answer--\n\n[[4, 4]]"))
        self.assertEqual(result["pred"], F0)
        with tempfile.TemporaryDirectory() as d:
            trace_path = Path(d) / "t.json.gz"
            with gzip.open(trace_path, "wt") as f:
                json.dump({"messages": []}, f)
            path = Path(d) / "prior.jsonl"
            path.write_text(json.dumps({"task_id": "t", "pair": 0, "attempt": 0, "pred": [[4, 4]], "train_pass": True,
                                        "trace_path": str(trace_path)}) + "\n"
                            + json.dumps({"task_id": "t", "pair": 0, "attempt": 1, "pred": [[1]]}) + "\n")
            a._prior = H.load_prior(path)
        self.assertEqual(a._prior, {("t", 0): ([[4, 4]], True, None)})
        self.assertEqual(H.prior_for(a, "t", TASK1, 0, 1, False), {"preds": [[[4, 4]]], "train_pass": True,
                                                                   "program": None})
        self.assertIsNone(H.prior_for(a, "t", TASK1, 0, 0, False))


if __name__ == "__main__":
    unittest.main()
