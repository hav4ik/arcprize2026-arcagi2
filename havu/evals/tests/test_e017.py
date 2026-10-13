"""E017 phase 2 options: --task-summary, lean2roles, --helpers, --train-check, --extra-attempt-if-no-train-pass,
--auto-peak-price. Offline tests always run; set ARC_SANDBOX_URL to also test against the real Docker sandbox."""
import datetime
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arc_helpers as H  # noqa: E402
import run_deepseek_tools as R  # noqa: E402
from task_summary import task_summary  # noqa: E402
from test_deepseek_tools import FakeClient, FakeSession, call, chunk  # noqa: E402
from train_check import grid_hash, ranked_answers, train_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = os.environ.get("ARC_SANDBOX_URL")

# A small task: objects of colour 1 move right by 1; a 2 dot is recoloured to 3; a 4 dot is removed.
G_IN = [[0, 0, 0, 0, 0],
        [1, 1, 0, 0, 2],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 4, 0]]
G_OUT = [[0, 0, 0, 0, 0],
         [0, 1, 1, 0, 3],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0]]
TASK = {"train": [{"input": G_IN, "output": G_OUT}],
        "test": [{"input": [[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 6], [0, 0, 0, 0, 0, 0]]}]}


class HelperTests(unittest.TestCase):
    def test_objects_holes_border_and_shape_keys(self):
        g = [[0, 0, 0, 0, 0, 0],
             [0, 5, 5, 5, 0, 0],
             [0, 5, 0, 5, 0, 7],
             [0, 5, 5, 5, 0, 7],
             [0, 0, 0, 0, 0, 0]]
        objs = H.objects(g)
        self.assertEqual([(o["colour"], o["size"], o["holes"], o["touches_border"]) for o in objs],
                         [(5, 8, 1, False), (7, 2, 0, True)])
        self.assertEqual(objs[0]["bbox"], (1, 1, 3, 3))
        el = [[1, 0], [1, 1]]
        keys = {H.shape_key(np.array(v, bool)) for v in H.d8(el).values()}
        self.assertEqual(len(keys), 1)  # one shape up to rotation/reflection
        self.assertNotEqual(H.shape_key(np.array([[1, 1]], bool)), keys.pop())
        multi = H.objects([[0, 1, 2], [0, 0, 0]], by="nonbg")
        self.assertEqual(multi[0]["colours"], [1, 2])

    def test_d8_panels_stamp_ray(self):
        a = [[1, 2], [3, 4]]
        self.assertEqual(H.match_d8(a, np.rot90(np.array(a))), "rot90")
        self.assertEqual(H.match_d8(a, [[1, 3], [2, 4]]), "transpose")
        self.assertIsNone(H.match_d8(a, [[9, 9], [9, 9]]))
        g = [[1, 0, 5, 2, 2], [0, 0, 5, 2, 0], [5, 5, 5, 5, 5], [3, 0, 5, 0, 4], [0, 0, 5, 0, 0]]
        ps = H.panels(g)
        self.assertEqual([p["bbox"] for p in ps], [(0, 0, 1, 1), (0, 3, 1, 4), (3, 0, 4, 1), (3, 3, 4, 4)])
        self.assertEqual(ps[1]["grid"].tolist(), [[2, 2], [2, 0]])
        self.assertEqual(H.separators(g)["colour"], 5)
        self.assertEqual(H.panels([[0, 1], [1, 0]]), [])
        s = H.stamp(np.zeros((3, 3), int), [[7, 7], [7, 0]], (-1, 2), transparent=0)
        self.assertEqual(s.tolist(), [[0, 0, 7], [0, 0, 0], [0, 0, 0]])
        with self.assertRaises(IndexError):
            H.stamp(np.zeros((2, 2), int), [[1, 1, 1]], (0, 0), clip=False)
        self.assertEqual(H.ray([[0, 0, 0, 3, 0]], (0, 0), (0, 1)), [(0, 1), (0, 2)])
        self.assertEqual(H.ray([[0, 0, 0]], (0, 0), (0, 1)), [(0, 1), (0, 2)])

    def test_check_reports_diffs_and_all_pass(self):
        pairs = [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 3, 4]], "output": [[4, 3, 3]]}]
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertTrue(H.check(lambda g: [row[::-1] for row in g], pairs))
        self.assertIn("ALL PASS", out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertFalse(H.check(lambda g: g, pairs))
            H.check(lambda g: [[0]], pairs)
            H.check(lambda g: 1 / 0, pairs)
        text = out.getvalue()
        self.assertIn("pair 0: 2 cells differ (r,c got/exp): (0,0) 1/2, (0,1) 2/1", text)
        self.assertIn("0/2 pass", text)
        self.assertIn("pair 1: shape 1x1 vs expected 1x3", text)
        self.assertIn("pair 0: error ZeroDivisionError", text)

    def test_check_does_not_let_fn_mutate_train(self):
        pairs = [{"input": [[1, 2]], "output": [[2, 1]]}]

        def bad(g):
            g[0][0] = 9
            return g
        with redirect_stdout(io.StringIO()):
            H.check(bad, pairs)
        self.assertEqual(pairs[0]["input"], [[1, 2]])


class SummaryTests(unittest.TestCase):
    def test_pair_header_matching_and_novelty(self):
        text = task_summary(TASK["train"], TASK["test"][0]["input"])
        self.assertTrue(text.startswith("\n\nA computed summary of the grids follows; trust it over reading"))
        self.assertIn("Ex0: in 4x5 bg=0 -> out 4x5 bg=0 | 6 cells changed", text)
        self.assertIn("colours in {0:15,1:3,2:1,4:1} out {0:16,1:3,3:1}", text)
        self.assertIn("#1 c1 size=3 bbox=(1,0)-(2,1) holes=0 border=yes shape=A", text)
        self.assertIn("moved #1 (+0,+1)", text)
        self.assertIn("recoloured 2->3 #2", text)
        self.assertIn("removed #3", text)
        self.assertIn("Outside train range: size 3x6", text)
        self.assertIn("new colour 6", text)
        self.assertIn("colours 2,4 absent (present in every train input)", text)
        self.assertIn("shapes not in any train input: B (#1)", text)
        self.assertEqual(text, task_summary(TASK["train"], TASK["test"][0]["input"]))  # deterministic

    def test_nothing_novel_and_resized_relations(self):
        train = [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
                 {"input": [[5, 0], [0, 0]], "output": [[5, 5, 0, 0], [5, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}]
        text = task_summary(train, [[1, 2], [3, 4]])
        self.assertIn("out = in rot270", text)
        self.assertIn("out = in scaled x2", text)
        crop = task_summary([{"input": [[0, 0, 0], [0, 7, 8], [0, 0, 0]], "output": [[7, 8]]}], [[0, 7], [0, 0]])
        self.assertIn("out = crop of in at (1,1)", crop)
        same = task_summary([{"input": [[0, 1], [0, 0]], "output": [[1, 0], [0, 0]]}], [[0, 0], [1, 0]])
        self.assertIn("Outside train range: nothing", same)

    def test_cap_on_every_eval_task(self):
        tasks = sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))[:40]
        for f in tasks:
            task = json.loads(f.read_text())
            text = task_summary(task["train"], task["test"][0]["input"])
            self.assertLessEqual(len(text), 10000, f.stem)
            self.assertIn("Outside train range:", text)


class HarnessOptionTests(unittest.TestCase):
    BASE = ["--sandbox-url", "http://sandbox", "--attempts", "1", "--answer-tool", "--final-mode", "session",
            "--prompt", "lean2sub"]

    def setUp(self):
        FakeSession.instances = []

    def test_lean2roles_prompt(self):
        text = R.PROMPT_SUFFIXES["lean2roles"]
        self.assertTrue(text.startswith(R.PROMPT_SUFFIXES["lean2sub"]))
        self.assertIn("write a short ROLES note", text)
        self.assertIn("Do not call anything noise or irrelevant unless code shows it", text)
        self.assertIn("do not submit: fix the rule.", text)
        self.assertIn("lean2roles", R.SUBMIT_PROMPTS)

    def test_versions_and_defaults_unchanged(self):
        args = R.parser().parse_args(self.BASE)
        self.assertEqual(R.harness_version(args), R.HARNESS_VERSION_V4)
        self.assertEqual(R.harness_version(R.parser().parse_args([])), R.HARNESS_VERSION)
        for extra in (["--task-summary"], ["--helpers"], ["--train-check"]):
            self.assertEqual(R.harness_version(R.parser().parse_args(self.BASE + extra)), R.HARNESS_VERSION_V5)
        args = R.parser().parse_args(self.BASE[:-1] + ["lean2roles"])
        self.assertEqual(R.harness_version(args), R.HARNESS_VERSION_V5)
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=3)]])
        _, result = R.solve(client, R.parser().parse_args(self.BASE), TASK, TASK["test"][0]["input"], FakeSession)
        self.assertNotIn("train_pass", result)
        self.assertNotIn("A computed summary", client.requests[0]["messages"][0]["content"])
        self.assertNotIn("check(fn)", client.requests[0]["tools"][0]["function"]["description"])

    def test_summary_and_helpers_reach_the_model_and_session(self):
        args = R.parser().parse_args(self.BASE + ["--task-summary", "--helpers"])
        seen = {}

        class Session(FakeSession):
            def __init__(self, *a, prelude_extra=""):
                super().__init__(*a)
                seen["prelude_extra"] = prelude_extra
        client = FakeClient([[chunk(content="[[4,3]]", finish="stop", tokens=3)]])
        R.solve(client, args, TASK, TASK["test"][0]["input"], Session)
        prompt = client.requests[0]["messages"][0]["content"]
        self.assertIn("Use check(solve) after every change to solve.", prompt)
        self.assertTrue(prompt.endswith(task_summary(TASK["train"], TASK["test"][0]["input"])))
        self.assertIn("check(fn) runs fn", client.requests[0]["tools"][0]["function"]["description"])
        self.assertIn("def check(fn", seen["prelude_extra"])

    def test_train_check_labels_are_recorded(self):
        args = R.parser().parse_args(self.BASE + ["--train-check"])
        client = FakeClient([[chunk(calls=[call("answer = f(test_input)"), call("answer", "submit_answer", 1)],
                                    finish="tool_calls", tokens=7)]])
        label = {"train_pass": True, "train_frac": 1.0, "train_trace": "cell", "train_paths": ["cell"]}
        with patch("run_deepseek_tools.submit_answer", return_value=("[answer accepted: 1x2 grid]", [[4, 3]])), \
                patch("run_deepseek_tools.train_check", return_value=label) as tc:
            _, result = R.solve(client, args, TASK, TASK["test"][0]["input"], FakeSession)
        executed, pred, code = tc.call_args.args[1:4]
        self.assertEqual(executed, [("run_python", "answer = f(test_input)"), ("submit_answer", "answer")])
        self.assertEqual((pred, code), ([[4, 3]], "answer"))
        self.assertEqual((result["train_pass"], result["train_trace"]), (True, "cell"))
        client = FakeClient([[chunk(content="no grid", finish="stop", tokens=3)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("x", None)):
            _, result = R.solve(client, args, TASK, TASK["test"][0]["input"], FakeSession)
        self.assertEqual((result["train_pass"], result["train_trace"]), (None, "no_answer"))
        client = FakeClient([[chunk(calls=[call("answer", "submit_answer")], finish="tool_calls", tokens=7)]])
        with patch("run_deepseek_tools.submit_answer", return_value=("ok", [[4, 3]])), \
                patch("run_deepseek_tools.train_check", side_effect=RuntimeError("boom")):
            _, result = R.solve(client, args, TASK, TASK["test"][0]["input"], FakeSession)
        self.assertEqual((result["pred"], result["train_trace"]), ([[4, 3]], "check_error"))

    def test_auto_peak_price(self):
        utc = datetime.timezone.utc
        self.assertEqual(R.peak_multiplier(datetime.datetime(2026, 10, 12, 6, 30, tzinfo=utc)), 2.0)  # Monday
        self.assertEqual(R.peak_multiplier(datetime.datetime(2026, 10, 12, 10, 0, tzinfo=utc)), 1.0)
        self.assertEqual(R.peak_multiplier(datetime.datetime(2026, 10, 13, 1, 59, tzinfo=utc)), 2.0)
        self.assertEqual(R.peak_multiplier(datetime.datetime(2026, 10, 11, 7, 0, tzinfo=utc)), 1.0)  # Sunday
        args = R.parser().parse_args(self.BASE + ["--auto-peak-price"])
        client = FakeClient([[chunk(calls=[call("1")], finish="tool_calls", tokens=1000)],
                             [chunk(content="[[4,3]]", finish="stop", tokens=1000)]])
        with patch("run_deepseek_tools.peak_multiplier", side_effect=[2.0, 1.0]):
            _, result = R.solve(client, args, TASK, TASK["test"][0]["input"], FakeSession)
        one = R.estimated_cost({"prompt": 30, "cache_hit": 10, "completion": 1000}, "deepseek-flash", 1)
        self.assertAlmostEqual(result["est_cost_usd"], 3 * one, places=6)
        self.assertAlmostEqual(result["est_cost_peak_usd"], 2 * one, places=6)
        self.assertEqual(result["peak_turns"], 1)


class PolicyTests(unittest.TestCase):
    def test_ranked_answers(self):
        recs = [{"attempt": 0, "pred": [[1]], "train_pass": False}, {"attempt": 1, "pred": [[1]], "train_pass": None},
                {"attempt": 2, "pred": [[2]], "train_pass": True}, {"attempt": 3, "pred": None, "train_pass": None}]
        self.assertEqual([r["attempt"] for r in ranked_answers(recs)], [2, 0])
        self.assertEqual([r["attempt"] for r in ranked_answers(recs[:2])], [0])  # duplicates collapse
        recs[0]["train_pass"] = True
        self.assertEqual([r["attempt"] for r in ranked_answers(recs)], [0, 2])

    def test_extra_attempt_only_when_neither_passes_and_resume(self):
        tasks = {"aa": {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]], "output": [[1]]}]},
                 "bb": {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[2]], "output": [[2]]}]}}
        passes = {("aa", 0): True, ("aa", 1): False, ("bb", 0): False, ("bb", 1): None, ("bb", 2): True}
        calls = []

        def fake_solve(client, args, task, test_input):
            t = "aa" if test_input == [[1]] else "bb"
            a = sum(1 for c in calls if c == t)
            calls.append(t)
            return {"messages": [], "events": []}, {
                "content": "", "pred": [[a]], "tool_calls": 0, "tool_errors": 0, "turns": 1,
                "usage": {"prompt": 1, "cache_hit": 0, "completion": 1}, "finish_reason": "stop",
                "stop_reason": "submitted", "usage_complete": True, "est_cost_usd": 0.0, "seconds": 0.0,
                "train_pass": passes[t, a]}

        with tempfile.TemporaryDirectory() as d:
            for t, task in tasks.items():
                Path(d, f"{t}.json").write_text(json.dumps(task))
            out = Path(d, "res.jsonl")
            argv = ["x", "--challenges", d, "--out", str(out), "--sandbox-url", "http://s", "--attempts", "2",
                    "--workers", "1", "--train-check", "--extra-attempt-if-no-train-pass", "--answer-tool",
                    "--prompt", "lean2sub", "--price-multiplier", "1"]
            with patch.object(sys, "argv", argv), patch("run_deepseek_tools.solve", side_effect=fake_solve), \
                    patch("run_deepseek_tools.check_health"), patch("run_deepseek_tools.api_key", return_value="k"), \
                    patch("run_deepseek_tools.deepseek_balance", return_value=100.0), \
                    redirect_stdout(io.StringIO()) as log:
                R.main()
            recs = R.read_records(out)
            self.assertEqual(sorted((r["task_id"], r["attempt"]) for r in recs),
                             [("aa", 0), ("aa", 1), ("bb", 0), ("bb", 1), ("bb", 2)])
            self.assertTrue(next(r for r in recs if r["attempt"] == 2)["policy_extra"])
            self.assertNotIn("extra_attempt_if_no_train_pass", recs[0]["config"])
            self.assertIn("train-check ranked pass@2", log.getvalue())
            # resume: nothing left to run, including no second extra attempt
            with patch.object(sys, "argv", argv), patch("run_deepseek_tools.solve", side_effect=AssertionError), \
                    redirect_stdout(io.StringIO()):
                R.main()
            # the same file can be resumed without the policy flag (it is not part of the config)
            with patch.object(sys, "argv", [a for a in argv if a != "--extra-attempt-if-no-train-pass"]), \
                    patch("run_deepseek_tools.solve", side_effect=AssertionError), redirect_stdout(io.StringIO()):
                R.main()

    def test_policy_flag_needs_train_check(self):
        with patch.object(sys, "argv", ["x", "--attempts", "2", "--extra-attempt-if-no-train-pass"]), \
                patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            R.main()


@unittest.skipUnless(SANDBOX, "set ARC_SANDBOX_URL to run Docker sandbox tests")
class DockerE017Tests(unittest.TestCase):
    TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
             {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}]
    TEST = [[1, 0, 2]]
    PRED = [[2, 0, 1]]

    def session(self, test_input=None, helpers=False):
        return R.SandboxSession(SANDBOX, self.TRAIN, test_input or self.TEST, 30, 16000,
                                prelude_extra=R.HELPERS_SRC if helpers else "")

    def label(self, cells, code, pred=None, helpers=False):
        pred = pred or self.PRED
        with self.session(helpers=helpers) as s:
            executed = []
            for name, c in cells:
                executed.append((name, c))
                s.execute(c) if name == "run_python" else R.submit_answer(s, c)
            return train_check(s, executed, pred, code, submit=R.submit_answer,
                               replay_session=lambda g: self.session(g, helpers))

    def test_expr_path(self):
        cells = [("run_python", "def solve(g):\n    return [r[::-1] for r in g]")]
        lab = self.label(cells, "solve(test_input)")
        self.assertEqual((lab["train_pass"], lab["train_frac"], lab["train_trace"]), (True, 1.0, "expr"))

    def test_cell_path_and_partial_fit(self):
        # rule fits pair 0 only (2-wide grids reversed, others unchanged) -> train_frac 0.5
        cells = [("run_python", "answer = [r[::-1] for r in test_input] if len(test_input) == 2 or "
                                "test_input[0][0] == 1 else [list(r) for r in test_input]")]
        lab = self.label(cells, "answer")
        self.assertEqual((lab["train_pass"], lab["train_frac"], lab["train_trace"]), (False, 0.5, "cell"))

    def test_fn_path_for_text_answer_and_helpers_excluded(self):
        cells = [("run_python", "def flip(g):\n    return [r[::-1] for r in g]\nprint(check(flip))")]
        lab = self.label(cells, None, helpers=True)
        self.assertEqual((lab["train_pass"], lab["train_trace"]), (True, "fn"))

    def test_transcript_path_and_untraceable(self):
        # the answer is built inline across cells, never in one cell that reads test_input and assigns answer
        cells = [("run_python", "x = test_input"), ("run_python", "answer = [r[::-1] for r in x]")]
        lab = self.label(cells, "answer")
        self.assertEqual((lab["train_pass"], lab["train_trace"]), (True, "transcript"))
        lab = self.label([("run_python", "answer = [[2, 0, 1]]")], "answer")
        self.assertEqual((lab["train_pass"], lab["train_trace"]), (False, "transcript"))
        lab = self.label([("run_python", "x = 1")], None)
        self.assertEqual((lab["train_pass"], lab["train_trace"]), (None, "untraceable"))

    def test_helpers_preloaded_and_survive_in_session(self):
        with self.session(helpers=True) as s:
            out = s.execute("def solve(g):\n    return [r[::-1] for r in g]\ncheck(solve)\n"
                            "print(len(objects(train[0]['input'])), match_d8([[1,2]], [[2,1]]))")
            self.assertEqual(out["process_status"], "completed", out)
            self.assertIn("pair 0: OK\npair 1: OK\nALL PASS", out["stdout"])
            self.assertIn("3 rot180", out["stdout"])
            out = s.execute("check(lambda g: g)")
            self.assertIn("pair 0: 4 cells differ (r,c got/exp): (0,0) 1/2", out["stdout"])
            self.assertEqual(grid_hash([[1]]), grid_hash([[1]]))


if __name__ == "__main__":
    unittest.main()
