"""Deep audit tests of the pinned harness a8a5b665 (new E046 paths) on the Kaggle runtime (v3 kit modules,
LocalExecutor ipython style, real child Python)."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np  # noqa: F401  (only to make sure the tool python has numpy)

import common as C

H = C.H
TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
         {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}]
T0, T1, T2 = [[1, 0], [0, 2]], [[3, 4, 5]], [[6], [7]]
FLIP = "def solve(g):\n    return np.fliplr(np.array(g))\n"


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
        self.assertEqual(g, [[[0, 1], [2, 0]], [[5, 4, 3]], [[6], [7]]], out)
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
        self.assertEqual(g[0], [[0, 1], [2, 0]], out)
        out, g = self.sub("res = [solve(t) for t in test_inputs]\nanswer = res")
        self.assertEqual(g[2], [[6], [7]], out)

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

    def test_errors_and_undefined(self):
        out, g = self.sub("undefined_fn(test_input)")
        self.assertIsNone(g)
        self.assertIn("NameError", out)
        out, g = self.sub("x = 1")  # statements that never set answer... but answer exists from another test?
        # `answer` may exist in this shared session from test_tuple_and_statements: either way never a crash
        self.assertTrue(g is None or len(g) == 3)

    def test_max_real_size_and_truncation(self):
        # ARC-AGI-2 has at most 4 test inputs per task (training/eval/test files); 4 x 30x30 fits in 16000 chars
        s = C.open_session(TRAIN, T0, [[[1] * 30] * 30] * 4)
        try:
            out, g = H.submit_answer(s, "[np.full((30, 30), 9) for t in test_inputs]", 4)
            self.assertIsNotNone(g, out)
            # 6 x 30x30 grids exceed max_tool_output (16000): the marker line is cut and the answer is rejected
            # with a misleading "got NoneType" (theoretical: no ARC-AGI-2 task has 6 test inputs)
            out, g = H.submit_answer(s, "[np.full((30, 30), 9) for _ in range(6)]", 6)
            self.assertIsNone(g)
            self.assertIn("got NoneType", out)
        finally:
            s.close()


class Shadowing(unittest.TestCase):
    """E046 run 3: a session `eval` broke submit. Now every builtin used by the submit cell comes from `builtins`."""

    def test_shadowed_builtins_single_and_multi(self):
        s = C.open_session(TRAIN, T0, [T0, T1])
        try:
            s.execute(FLIP + "def eval(grid, H=0, bg=0):\n    return 'model eval'\n"
                      "def compile(*a, **k):\n    raise RuntimeError('model compile')\n"
                      "def exec(*a, **k):\n    raise RuntimeError('model exec')\n"
                      "def globals():\n    return {}\n"
                      "def hasattr(*a):\n    raise RuntimeError('model hasattr')\n"
                      "def isinstance(*a):\n    raise RuntimeError('model isinstance')\n"
                      "list = tuple = None\nlen = None\nNameError = SyntaxError = KeyError\n"
                      "answer = [solve(t) for t in test_inputs]\n")
            out, g = H.submit_answer(s, "[solve(t) for t in test_inputs]", 2)
            self.assertEqual(g, [[[0, 1], [2, 0]], [[5, 4, 3]]], out)
            out, g = H.submit_answer(s, "answer", 2)
            self.assertIsNotNone(g, out)
            out, g = H.submit_answer(s, "z = 3\nanswer = [solve(t) for t in test_inputs]", 2)
            self.assertIsNotNone(g, out)
            out, g = H.submit_answer(s, "solve(test_input)")
            self.assertEqual(g, [[0, 1], [2, 0]], out)
            s.execute("del answer")
            out, g = H.submit_answer(s, "y = 2")  # exec path without answer: NameError via _b.NameError
            self.assertIsNone(g)
            self.assertIn("code did not define answer", out)
        finally:
            s.close()

    def test_shadowed_print_still_breaks_submit(self):
        # known (other reviewer): the final print() is not taken from builtins
        s = C.open_session(TRAIN, T0)
        try:
            s.execute(FLIP + "def print(*a, **k):\n    pass\n")
            out, g = H.submit_answer(s, "solve(test_input)")
            self.assertIsNone(g)
            self.assertIn("no value returned", out)
        finally:
            s.close()


class SingleTestPlainChange(unittest.TestCase):
    """a8a5b665 applies _plain to single-test answers too: compare with the v3 kit's harness 1ad5ee24."""

    @classmethod
    def setUpClass(cls):
        cls.v3 = C.load_v3_harness()
        cls.s = C.open_session(TRAIN, T0)
        cls.s.execute(FLIP)

    @classmethod
    def tearDownClass(cls):
        cls.s.close()

    def test_differences_vs_v3(self):
        cases = {
            "solve(test_input)": "same",
            "solve(test_input).tolist()": "same",
            "[list(r) for r in solve(test_input)]": "same",  # lists of np.int64: default=int vs tolist
            "[r for r in solve(test_input)]": "fixed",  # list of numpy rows: v3 json error, now accepted
            "tuple(tuple(r) for r in test_input)": "same",
            "[[np.float64(1.0)]]": "same",
            "[[True]]": "same",
            "[[np.bool_(True), np.bool_(False)]]": "regressed",  # v3: default=int -> [[1, 0]]; now [[true, false]]
            "np.array([[True]])": "same",
        }
        for code, kind in cases.items():
            o_new, g_new = H.submit_answer(self.s, code)
            o_old, g_old = self.v3.submit_answer(self.s, code)
            if kind == "same":
                self.assertEqual(g_new, g_old, code)
            elif kind == "fixed":
                self.assertIsNone(g_old, code)
                self.assertEqual(g_new, [[0, 1], [2, 0]], code)
            else:
                self.assertEqual(g_old, [[1, 0]], code)
                self.assertIsNone(g_new, code)


class Prompts(unittest.TestCase):
    def test_build_prompt_multi(self):
        p = H.build_prompt_multi(TRAIN, [T0, T1, T2])
        for i, g in enumerate([T0, T1, T2]):
            self.assertIn(f"--Test input {i}--\n\n{json.dumps(g)}", p)
        self.assertEqual(p.count("--End of Test Input--"), 1)
        self.assertNotIn("[[0]]\n--End of Test Input--", p)
        self.assertTrue(p.startswith(H.build_prompt(TRAIN, T0).split("--Test Input--")[0]))
        # a test input that is literally [[0]] still renders
        p = H.build_prompt_multi(TRAIN, [[[0]], [[0]]])
        self.assertIn("--Test input 1--\n\n[[0]]\n--End of Test Input--", p)

    def test_single_test_tools_mention_test_inputs(self):
        # Known (other reviewer), confirmed: with --multi-test every task's tool description says test_inputs is
        # preloaded, but single-test sessions never define it.
        a_on = C.harness_args(["--multi-test"])
        a_off = C.harness_args([])
        self.assertIn("test_inputs", H.tools_for(a_on)[0]["function"]["description"])
        self.assertNotIn("test_inputs", H.tools_for(a_off)[0]["function"]["description"])
        s = C.open_session(TRAIN, T0)  # the single-test session the harness opens (no multi prelude)
        try:
            r = s.execute("print(len(test_inputs))")
            self.assertEqual(r["process_status"], "error")
            self.assertIn("NameError", r["stdout"] + r.get("stderr", ""))
        finally:
            s.close()


def run_solve(turns, args, task, test_inputs=None, pad=0, completion_tokens=None):
    srv = C.FakeTokenServer(turns, pad=pad, completion_tokens=completion_tokens)
    with patch("httpx.post", srv.post):
        trace, result = H.solve(srv, args, task, task["test"][0]["input"], session_factory=C.local_factory,
                                **({"test_inputs": test_inputs} if test_inputs else {}))
    return srv, trace, result


TASK1 = {"train": TRAIN, "test": [{"input": T0}]}


def first_prompt_len(argv, task, test_inputs=None, pad=0):
    """Prompt tokens of turn 1 (from the ContextFull message with a limit of 1)."""
    srv, trace, result = run_solve([], C.harness_args(argv + ["--context-limit", "1"]), task, test_inputs, pad=pad)
    ev = [e for e in trace["events"] if "context_full" in e][0]["context_full"]
    return int(ev.split()[1])
TASK3 = {"train": TRAIN, "test": [{"input": T0}, {"input": T1}, {"input": T2}]}


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
        self.assertEqual(result["pred"], [[0, 1], [2, 0]])
        self.assertEqual(result["answer_source"], "session")
        self.assertEqual(result["turns"], 1)
        self.assertNotIn("error", result)
        self.assertTrue(any("context_full" in e for e in trace["events"]))
        self.assertIn(result["train_trace"], ("cell", "fn", "expr", "transcript"))

    def test_context_full_multi_session_answer_and_fallback_solve(self):
        argv = ["--multi-test", "--multi-budget", "0.5"]
        body = C.ds_call_text(FLIP + "answer = [solve(t) for t in test_inputs]\n" + "#" * 6000)
        n1 = first_prompt_len(argv, TASK3, [T0, T1, T2], pad=300)
        args = C.harness_args(argv + ["--context-limit", str(n1 + 16 + len(body) + 200)])
        srv, trace, result = run_solve([body], args, TASK3, [T0, T1, T2], pad=300)
        self.assertEqual(result["stop_reason"], "context")
        self.assertEqual(result["pred"], [[[0, 1], [2, 0]], [[5, 4, 3]], [[6], [7]]])
        self.assertEqual(result["multi_test"], 3)
        # no `answer`, only solve: --fallback-solve with the multi expression
        body = C.ds_call_text(FLIP + "#" * 6000 + "\n#" + "x" * 40)
        srv, trace, result = run_solve([body], args, TASK3, [T0, T1, T2], pad=300)
        self.assertEqual(result["stop_reason"], "context")
        self.assertEqual(result["answer_source"], "session_solve")
        self.assertEqual(len(result["pred"]), 3)

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
        self.assertEqual(result["pred"], [[0, 1], [2, 0]])


class MultiBudget(unittest.TestCase):
    def test_scaling_n1_to_n6(self):
        base = C.harness_args(["--multi-test", "--multi-budget", "0.5"])
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
        a = C.harness_args(["--multi-test", "--multi-budget", "0.5"])
        srv, trace, result = run_solve([C.ds_call_text("print(1)"), C.ds_call_text(FLIP),
                                        C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")],
                                       a, TASK3, [T0, T1, T2], completion_tokens=50000)
        tool_msgs = [m["content"] for m in trace["messages"] if m["role"] == "tool"]
        self.assertIn("aim to finish within 80k (hard limit 160k)", tool_msgs[0])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(len(result["pred"]), 3)


class MultiFlow(unittest.TestCase):
    def test_text_answer_in_multi_uses_session(self):
        a = C.harness_args(["--multi-test", "--multi-budget", "0.5"])
        srv, trace, result = run_solve([C.ds_call_text(FLIP + "answer = [solve(t) for t in test_inputs]"),
                                        C.ds_text_answer("[[0, 1], [2, 0]]")], a, TASK3, [T0, T1, T2])
        self.assertEqual(result["stop_reason"], "final")
        self.assertEqual(result["answer_source"], "session")
        self.assertEqual(len(result["pred"]), 3)

    def test_single_test_messages_identical_with_multi_flag(self):
        # the single-test prompt is unchanged by --multi-test; only the tool descriptions differ
        outs = {}
        for flag in ([], ["--multi-test", "--multi-budget", "0.5"]):
            a = C.harness_args(flag + ["--context-limit", "163840"])
            srv, trace, result = run_solve([C.ds_call_text(FLIP),
                                            C.ds_call_text("solve(test_input)", name="submit_answer")], a, TASK1)
            outs[bool(flag)] = (trace["messages"][:2], result["pred"], result["turns"], result["stop_reason"],
                                srv.requests[0][1])
        self.assertEqual(outs[False], outs[True])

    def test_train_check_multi_labels(self):
        a = C.harness_args(["--multi-test", "--multi-budget", "0.5"])
        srv, trace, result = run_solve([C.ds_call_text(FLIP),
                                        C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")],
                                       a, TASK3, [T0, T1, T2])
        self.assertEqual((result["train_pass"], result["train_trace"]), (True, "fn"))
        # wrong rule with no 1-arg function: the transcript replay keeps the real test_inputs and labels False
        srv, trace, result = run_solve([C.ds_call_text("out0 = [r[::-1] for r in test_inputs[0]]\n"
                                                       "answer = [[r[::-1] for r in t] for t in test_inputs]"),
                                        C.ds_call_text("answer", name="submit_answer")], a, TASK3, [T0, T1, T2])
        self.assertEqual(result["stop_reason"], "submitted")
        self.assertEqual(result["train_trace"], "transcript")
        self.assertIs(result["train_pass"], False)  # the program is right on train (flip), labelled False


if __name__ == "__main__":
    unittest.main()
