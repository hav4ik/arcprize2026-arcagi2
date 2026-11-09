"""I7 --best-guess-at F / --best-guess-turns K (kaggle/dsv41_tools/next_candidate_fixes.md): once per attempt with no
accepted answer, a firm "submit your best answer now" note on the tool result of the turn that reaches F x the hard
output budget (scaled under --multi-budget, context-capped under --context-limit) or max_turns - K turns. Kaggle path:
the solver's harness argv, a fake token-exact server and LocalExecutor sessions. Flag off: messages and requests are
byte-identical to harness 45eb3e65 (fixtures/best_guess_off_45eb3e65.json, recorded with that harness on SCRIPTS)."""
import io
import json
import re
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import dsv41_fakes as C

H = C.H
FIXTURE = Path(__file__).resolve().parent / "fixtures/best_guess_off_45eb3e65.json"
TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
         {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}]
T0, T1, T2 = [[1, 0], [0, 2]], [[3, 4, 5]], [[6], [7]]
F0, F1, F2 = [[0, 1], [2, 0]], [[5, 4, 3]], [[6], [7]]  # fliplr of T0, T1, T2
FLIP = "def solve(g):\n    return np.fliplr(np.array(g))\n"
TASK1 = {"train": TRAIN, "test": [{"input": T0}]}
TASK3 = {"train": TRAIN, "test": [{"input": T0}, {"input": T1}, {"input": T2}]}
MULTI = ["--multi-test", "--multi-budget", "0.5"]
CTX = ["--context-limit", "163840"]
NOTE = ("\n[Stop exploring now] Call submit_answer with your best current answer, even if it does not reproduce every "
        "training pair. An imperfect answer can still score; no answer scores zero.")


def single_turns(submit_at=9):
    """9 turns at 9k output tokens each (80k budget): run_python, a rescued ```python block, an invalid submit, the
    "Budget nearly spent" note (turn 7, 63k), the "N turns left" notes (turn 5 on), then a valid submit."""
    turns = [C.ds_call_text("print(len(train))"),
             C.ds_text_answer("```python\nprint(test_input)\n```"),
             C.ds_call_text(FLIP + "print(solve(train[0]['input']).tolist())"),
             C.ds_call_text("[[1, 2], [3]]", name="submit_answer"),
             C.ds_call_text("answer = solve(test_input)\nprint(answer.tolist())"),
             C.ds_call_text("print('a')"), C.ds_call_text("print('b')"), C.ds_call_text("print('c')"),
             C.ds_call_text("print('d')")]
    turns[submit_at - 1] = C.ds_call_text("solve(test_input)", name="submit_answer")
    return turns[:submit_at]


def multi_turns():
    """6 turns at 30k each on a 3-input task (budget 160k, 20 turns under --multi-budget 0.5): an invalid 2-grid
    submit, then the 3-grid submit."""
    return [C.ds_call_text("print([len(t) for t in test_inputs])"), C.ds_call_text(FLIP),
            C.ds_call_text("[solve(t) for t in test_inputs[:2]]", name="submit_answer"),
            C.ds_call_text("print(1)"), C.ds_call_text("print(2)"),
            C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")]


# name -> (turns, argv, task, test_inputs, completion tokens per turn)
SCRIPTS = {"single": (single_turns, CTX, TASK1, None, 9000),
           "multi": (multi_turns, MULTI + CTX, TASK3, [T0, T1, T2], 30000)}


def run(name, extra=(), completion_tokens=None, turns=None, max_turns=10):
    make, argv, task, test_inputs, tokens = SCRIPTS[name]
    return C.run_solve(make() if turns is None else turns, C.harness_args(argv + list(extra), max_turns=max_turns),
                       task, test_inputs, completion_tokens=completion_tokens or tokens)


def normalize(messages):
    """The transcript with the random tool-call ids (call_<24 hex>) numbered in order of appearance."""
    ids = {}
    text = json.dumps(messages, sort_keys=True)
    return json.loads(re.sub(r"call_[0-9a-f]{24}", lambda m: ids.setdefault(m.group(0), f"call_#{len(ids)}"), text))


def tools(trace):
    return [m["content"] for m in trace["messages"] if m["role"] == "tool"]


def noted(trace):
    """1-based indices of the tool results that carry the best-guess note."""
    return [i for i, t in enumerate(tools(trace), 1) if "[Stop exploring now]" in t]


class FlagOff(unittest.TestCase):
    def test_off_matches_45eb3e65(self):
        want = json.loads(FIXTURE.read_text())
        for name in SCRIPTS:
            for extra in ([], ["--best-guess-at", "0", "--best-guess-turns", "0"]):
                srv, trace, result = run(name, extra)
                self.assertEqual(normalize(trace["messages"]), want[name]["messages"], (name, extra))
                self.assertEqual([list(r) for r in srv.requests], want[name]["requests"], (name, extra))
                self.assertEqual((result["stop_reason"], result["pred"]), tuple(want[name]["result"]), name)
                self.assertNotIn("best_guess", result)
                self.assertFalse(any("best_guess" in e for e in trace["events"]))

    def test_prompts_and_tools_unchanged_when_on(self):
        on = ["--best-guess-at", "0.8", "--best-guess-turns", "3"]
        for name in SCRIPTS:
            argv = SCRIPTS[name][1]
            self.assertEqual(H.tools_for(C.harness_args(argv + on)), H.tools_for(C.harness_args(argv)))
            off_trace, on_trace = run(name)[1], run(name, on)[1]
            self.assertEqual(on_trace["messages"][:2], off_trace["messages"][:2])  # effort header + task prompt

    def test_not_in_config(self):
        on = C.harness_args(["--best-guess-at", "0.8", "--best-guess-turns", "3"])
        self.assertEqual(H.harness_version(on), H.harness_version(C.harness_args([])))
        self.assertFalse({"best_guess_at", "best_guess_turns"} & {**H.V4_DEFAULTS, **H.V5_DEFAULTS, **H.V6_DEFAULTS}.keys())


class Fires(unittest.TestCase):
    def assert_only_note_added(self, name, on_trace, at, extra_note=""):
        """The flag-on transcript is the flag-off one with the note inserted in tool result `at`, after the tool output
        and before the budget note."""
        off = normalize(run(name)[1]["messages"])
        on = normalize(on_trace["messages"])
        k = [i for i, m in enumerate(off) if m["role"] == "tool"][at - 1]
        self.assertEqual(on[k]["content"], off[k]["content"].replace("\n[budget:", NOTE + extra_note + "\n[budget:", 1))
        self.assertEqual(on[:k] + on[k + 1:], off[:k] + off[k + 1:])

    def test_fires_once_at_the_token_threshold(self):
        srv, trace, result = run("single", ["--best-guess-at", "0.8"])
        self.assertEqual(noted(trace), [8])  # 72k >= 0.8 x 80k; never again
        self.assert_only_note_added("single", trace, 8)
        self.assertEqual(result["best_guess"], {"fired_at_tokens": 72000, "fired_at_turn": 8, "submitted_after": True,
                                                "turns_after": 1, "tokens_after": 9000})
        self.assertIn({"turn": 8, "best_guess": True}, trace["events"])
        t8 = tools(trace)[7]
        self.assertTrue(t8.startswith("[completed]\nc\n" + NOTE + "\n[budget: 72k output tokens used; aim to finish "
                                      "within 40k (hard limit 80k)] Budget nearly spent"), t8)
        # exactly at the threshold (8 x 8k = 64k): fires on that turn
        srv, trace, result = run("single", ["--best-guess-at", "0.8"], completion_tokens=8000)
        self.assertEqual((noted(trace), result["best_guess"]["fired_at_tokens"]), ([8], 64000))

    def test_fires_at_the_turn_threshold(self):
        srv, trace, result = run("single", ["--best-guess-turns", "3"], completion_tokens=100)
        self.assertEqual(noted(trace), [7])  # turns so far >= 10 - 3
        self.assertEqual(result["best_guess"], {"fired_at_tokens": 700, "fired_at_turn": 7, "submitted_after": True,
                                                "turns_after": 2, "tokens_after": 200})
        # both thresholds: the first one reached (turn 7, 63k < 64k), once
        srv, trace, result = run("single", ["--best-guess-at", "0.8", "--best-guess-turns", "3"])
        self.assertEqual(noted(trace), [7])
        self.assert_only_note_added("single", trace, 7)
        self.assertNotIn("Submit all", tools(trace)[6])

    def test_not_after_a_valid_submit(self):
        # the turn that reaches the threshold submits a valid answer: the attempt ends, no note
        for extra, at in ((["--best-guess-at", "0.8"], 8), (["--best-guess-turns", "3"], 7)):
            srv, trace, result = run("single", extra, turns=single_turns(submit_at=at))
            self.assertEqual((result["stop_reason"], result["pred"], result["turns"]), ("submitted", F0, at))
            self.assertEqual(noted(trace), [])
            self.assertIsNone(result["best_guess"])
        # an invalid submit is not an answer: its rejection carries the note
        turns = single_turns()
        turns[7] = C.ds_call_text("[[10]]", name="submit_answer")
        srv, trace, result = run("single", ["--best-guess-at", "0.8"], turns=turns)
        self.assertEqual(noted(trace), [8])
        self.assertTrue(tools(trace)[7].startswith("[answer not accepted]"))

    def test_not_when_no_turn_can_follow(self):
        # 41k then 82k: the turn that passes 0.8 x 80k also ends the budget, so the note would never be read
        srv, trace, result = run("single", ["--best-guess-at", "0.8"], completion_tokens=41000)
        self.assertEqual((result["stop_reason"], result["turns"], noted(trace)), ("output_budget", 2, []))
        self.assertIsNone(result["best_guess"])
        # K = 1: the note comes with one turn left; then the turn cap ends the attempt without a submit
        turns = [C.ds_call_text(f"print({i})") for i in range(10)]
        srv, trace, result = run("single", ["--best-guess-turns", "1"], completion_tokens=100, turns=turns)
        self.assertEqual((result["stop_reason"], noted(trace)), ("max_turns", [9]))
        self.assertEqual(result["best_guess"], {"fired_at_tokens": 900, "fired_at_turn": 9, "submitted_after": False,
                                                "turns_after": 1, "tokens_after": 100})

    def test_multi_scaled_budget_and_wording(self):
        # n = 3, --multi-budget 0.5: budget 160k (threshold 128k, turn 5 at 150k; unscaled 64k would be turn 3)
        srv, trace, result = run("multi", ["--best-guess-at", "0.8"])
        self.assertEqual(noted(trace), [5])
        self.assert_only_note_added("multi", trace, 5, " Submit all 3 test grids.")
        self.assertEqual(result["best_guess"], {"fired_at_tokens": 150000, "fired_at_turn": 5, "submitted_after": True,
                                                "turns_after": 1, "tokens_after": 30000})
        self.assertEqual((result["stop_reason"], result["pred"]), ("submitted", [F0, F1, F2]))
        # the turn cap is scaled too: 10 -> 20 turns, K = 3 -> turn 17
        turns = [C.ds_call_text(f"print({i})") for i in range(25)]
        srv, trace, result = run("multi", ["--best-guess-turns", "3"], completion_tokens=100, turns=turns)
        self.assertEqual((result["stop_reason"], result["turns"], noted(trace)), ("max_turns", 20, [17]))

    def test_context_capped_budget(self):
        # A9's case: under --context-limit the budget is the output that still fits (~6k here, 5k used >= 0.8 x 6k)
        n1 = int([e for e in run("multi", ["--context-limit", "1"], turns=[])[1]["events"]
                  if "context_full" in e][0]["context_full"].split()[1])
        limit = n1 + 5000 + 16 + H.CONTEXT_MIN_ROOM + 1000
        turns = [C.ds_call_text(FLIP + "print(1)"),
                 C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")]
        srv, trace, result = run("multi", ["--best-guess-at", "0.8", "--context-limit", str(limit)],
                                 completion_tokens=5000, turns=turns)
        self.assertEqual(noted(trace), [1])
        self.assertIn(NOTE + " Submit all 3 test grids.\n[budget: 5k of 6k output tokens used] Budget nearly spent",
                      tools(trace)[0])
        self.assertEqual(result["best_guess"]["fired_at_tokens"], 5000)
        # far from the context limit, 5k of 160k: no note
        srv, trace, result = run("multi", ["--best-guess-at", "0.8", "--context-limit", "1000000"],
                                 completion_tokens=5000, turns=turns)
        self.assertEqual(noted(trace), [])
        self.assertIsNone(result["best_guess"])

    def test_cap_nudge_result_carries_it(self):
        # --cap-nudge's synthetic tool result is a tool result too: the note goes after CAP_NOTE, before the budget
        turns = [C.ds_call_text("print(1)"), lambda n, m: "x" * (m + 10),
                 C.ds_call_text("[[0, 1], [2, 0]]", name="submit_answer")]
        srv, trace, result = run("single", ["--best-guess-turns", "8", "--cap-nudge", "--max-tokens", "500"],
                                 completion_tokens=1000, turns=turns)
        self.assertEqual(noted(trace), [2])
        self.assertTrue(tools(trace)[1].startswith(H.CAP_NOTE.format(k=0) + NOTE + "\n[budget:"), tools(trace)[1])
        self.assertEqual((result["stop_reason"], result["cap_nudges"], result["best_guess"]["fired_at_turn"]),
                         ("submitted", 1, 2))


class Cli(unittest.TestCase):
    def test_validation(self):
        base = ["run_deepseek_tools.py", "--answer-tool", "--final-mode", "session", "--dry-run", "--tasks", "x"]
        for extra, msg in ((["--best-guess-at", "1.0"], "--best-guess-at must be in [0, 1)"),
                           (["--best-guess-turns", "40"], "--best-guess-turns in [0, --max-turns)"),
                           (["--best-guess-turns", "-1"], "--best-guess-turns in [0, --max-turns)")):
            err = io.StringIO()
            with patch("sys.argv", base + extra), redirect_stderr(err), self.assertRaises(SystemExit):
                H.main()
            self.assertIn(msg, err.getvalue())
        err = io.StringIO()
        with patch("sys.argv", base[:1] + ["--best-guess-at", "0.8", "--dry-run"]), redirect_stderr(err), \
                self.assertRaises(SystemExit):
            H.main()
        self.assertIn("need --answer-tool", err.getvalue())


if __name__ == "__main__":
    unittest.main()
