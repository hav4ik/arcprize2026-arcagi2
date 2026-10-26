"""I8 --post-pass-note (kaggle/dsv41_tools/next_candidate_fixes.md): once per attempt, the first time a run_python
result shows a program reproducing every training pair (evals/run_deepseek_tools.py:train_pass_shown) and no answer
was accepted yet, the tool result gets "[All training pairs pass] Check with code any assumption about the test input
...". Kaggle path: the solver's harness argv, a fake token-exact server and LocalExecutor sessions. Flag off: messages
and requests are byte-identical to harness 13ebdb70 (fixtures/post_pass_off_13ebdb70.json, recorded with that
harness on SCRIPTS)."""
import io
import json
import re
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import dsv41_fakes as C

H = C.H
FIXTURE = Path(__file__).resolve().parent / "fixtures/post_pass_off_13ebdb70.json"
TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
         {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}]
T0, T1, T2 = [[1, 0], [0, 2]], [[3, 4, 5]], [[6], [7]]
F0, F1, F2 = [[0, 1], [2, 0]], [[5, 4, 3]], [[6], [7]]  # fliplr of T0, T1, T2
FLIP = "def solve(g):\n    return np.fliplr(np.array(g))\n"
TASK1 = {"train": TRAIN, "test": [{"input": T0}]}
TASK3 = {"train": TRAIN, "test": [{"input": T0}, {"input": T1}, {"input": T2}]}
MULTI = ["--multi-test", "--multi-budget", "0.5"]
CTX = ["--context-limit", "163840"]
ON = ["--post-pass-note"]
NOTE = ("\n[All training pairs pass] Check with code any assumption about the test input that the training pairs do "
        "not cover (sizes, colours, object counts), then submit_answer.")
PARTIAL = ("def solve(g):\n    g = np.array(g)\n    return np.fliplr(g) if g.shape[1] == 2 else g\n"
           "for p in train:\n    print(np.array_equal(solve(p['input']), p['output']))\n")  # True, False
SHAPES = "for p in train:\n    print(np.array(p['input']).shape == np.array(p['output']).shape)\n"  # True, True
PASS_LOOP = FLIP + "for i, p in enumerate(train):\n    print(i, np.array_equal(solve(p['input']), p['output']))\n"
PASS_LIST = "print([solve(p['input']).tolist() == p['output'] for p in train])\n"


def single_turns():
    """6 turns at 9k output tokens each: a partial pass (True, False), an all-True shape check (no program), the first
    all-pass (turn 4, 36k), a second all-pass, then a valid submit."""
    return [C.ds_call_text("print(len(train))"), C.ds_call_text(PARTIAL), C.ds_call_text(SHAPES),
            C.ds_call_text(PASS_LOOP), C.ds_call_text(PASS_LIST),
            C.ds_call_text("solve(test_input)", name="submit_answer")]


def multi_turns():
    """3 turns at 10k on a 3-input task: all() over the training pairs passes on turn 2, then the 3-grid submit."""
    return [C.ds_call_text("print([len(t) for t in test_inputs])"),
            C.ds_call_text(FLIP + "print(all(np.array_equal(solve(p['input']), p['output']) for p in train))"),
            C.ds_call_text("[solve(t) for t in test_inputs]", name="submit_answer")]


def helpers_turns():
    """--helpers: check(solve) fails (0/2 pass), then ALL PASS on turn 2, then a valid submit."""
    return [C.ds_call_text("def solve(g):\n    return np.array(g)\ncheck(solve)"),
            C.ds_call_text(FLIP + "check(solve)"),
            C.ds_call_text("solve(test_input)", name="submit_answer")]


# name -> (turns, argv, task, test_inputs, completion tokens per turn)
SCRIPTS = {"single": (single_turns, CTX, TASK1, None, 9000),
           "multi": (multi_turns, MULTI + CTX, TASK3, [T0, T1, T2], 10000),
           "helpers": (helpers_turns, CTX + ["--helpers"], TASK1, None, 9000)}


def run(name, extra=(), completion_tokens=None, turns=None, max_turns=10):
    make, argv, task, test_inputs, tokens = SCRIPTS[name]
    return C.run_solve(make() if turns is None else turns, C.harness_args(argv + list(extra), max_turns=max_turns),
                       task, test_inputs, completion_tokens=completion_tokens or tokens)


def normalize(messages):
    """The transcript with the random tool-call ids (call_<24 hex>) numbered in order of appearance."""
    ids = {}
    text = json.dumps(messages, sort_keys=True)
    return json.loads(re.sub(r"call_[0-9a-f]{24}", lambda m: ids.setdefault(m.group(0), f"call_#{len(ids)}"), text))


def record():
    """The flag-off fixture of SCRIPTS (run with the harness the fixture is named after)."""
    out = {}
    for name in SCRIPTS:
        srv, trace, result = run(name)
        out[name] = {"messages": normalize(trace["messages"]), "requests": [list(r) for r in srv.requests],
                     "result": [result["stop_reason"], result["pred"]]}
    return out


def two_calls(code, submit):
    """One V4.1 completion with a run_python call and then a submit_answer call."""
    d, first = C.DSML, C.ds_call_text(code)
    second = C.ds_call_text(submit, name="submit_answer")
    head = first[:first.index(f"</{d} calls>")]
    return head + second[second.index(f"<{d} invoke"):]


def tools(trace):
    return [m["content"] for m in trace["messages"] if m["role"] == "tool"]


def noted(trace):
    """1-based indices of the tool results that carry the post-pass note."""
    return [i for i, t in enumerate(tools(trace), 1) if "[All training pairs pass]" in t]


class FlagOff(unittest.TestCase):
    def test_off_matches_13ebdb70(self):
        want = json.loads(FIXTURE.read_text())
        self.assertEqual(set(want), set(SCRIPTS))
        for name in SCRIPTS:
            srv, trace, result = run(name)
            self.assertEqual(normalize(trace["messages"]), want[name]["messages"], name)
            self.assertEqual([list(r) for r in srv.requests], want[name]["requests"], name)
            self.assertEqual((result["stop_reason"], result["pred"]), tuple(want[name]["result"]), name)
            self.assertNotIn("post_pass", result)
            self.assertFalse(any("post_pass" in e for e in trace["events"]))
            self.assertEqual(noted(trace), [])

    def test_prompts_and_tools_unchanged_when_on(self):
        for name in SCRIPTS:
            argv = SCRIPTS[name][1]
            self.assertEqual(H.tools_for(C.harness_args(argv + ON)), H.tools_for(C.harness_args(argv)))
            off_trace, on_trace = run(name)[1], run(name, ON)[1]
            self.assertEqual(on_trace["messages"][:2], off_trace["messages"][:2])  # effort header + task prompt

    def test_not_in_config(self):
        self.assertEqual(H.harness_version(C.harness_args(ON)), H.harness_version(C.harness_args([])))
        self.assertNotIn("post_pass_note", {**H.V4_DEFAULTS, **H.V5_DEFAULTS, **H.V6_DEFAULTS})


class Fires(unittest.TestCase):
    def assert_only_note_added(self, name, on_trace, at, note=NOTE):
        """The flag-on transcript is the flag-off one with the note in tool result `at`, after the tool output and
        before the budget note."""
        off = normalize(run(name)[1]["messages"])
        on = normalize(on_trace["messages"])
        k = [i for i, m in enumerate(off) if m["role"] == "tool"][at - 1]
        self.assertEqual(on[k]["content"], off[k]["content"].replace("\n[budget:", note + "\n[budget:", 1))
        self.assertEqual(on[:k] + on[k + 1:], off[:k] + off[k + 1:])

    def test_fires_once_on_the_first_all_pass(self):
        srv, trace, result = run("single", ON)
        self.assertEqual(noted(trace), [4])  # not on the partial pass (2) or the shape check (3); once
        self.assert_only_note_added("single", trace, 4)
        self.assertTrue(tools(trace)[3].startswith("[completed]\n0 True\n1 True\n" + NOTE + "\n[budget: 36k"),
                        tools(trace)[3])
        self.assertEqual(result["post_pass"], {"fired_at_turn": 4, "fired_at_tokens": 36000, "turns_after": 2,
                                               "tokens_after": 18000, "submitted_after": True})
        self.assertIn({"turn": 4, "post_pass": True}, trace["events"])
        self.assertEqual((result["stop_reason"], result["pred"]), ("submitted", F0))

    def test_multi_wording(self):
        srv, trace, result = run("multi", ON)
        self.assertEqual(noted(trace), [2])
        self.assert_only_note_added("multi", trace, 2, NOTE + " Submit all 3 test grids.")
        self.assertEqual(result["post_pass"], {"fired_at_turn": 2, "fired_at_tokens": 20000, "turns_after": 1,
                                               "tokens_after": 10000, "submitted_after": True})
        self.assertEqual((result["stop_reason"], result["pred"]), ("submitted", [F0, F1, F2]))
        # a single-test attempt under --multi-test gets the single-test wording
        srv, trace, result = run("single", MULTI + ON)
        self.assertEqual(noted(trace), [4])
        self.assertNotIn("Submit all", tools(trace)[3])

    def test_helpers_check(self):
        srv, trace, result = run("helpers", ON)
        self.assertEqual(noted(trace), [2])  # 0/2 pass on turn 1, ALL PASS on turn 2
        self.assertIn("0/2 pass", tools(trace)[0])
        self.assert_only_note_added("helpers", trace, 2)

    def test_never_without_an_all_pass(self):
        turns = single_turns()
        del turns[3:5]  # partial pass, shape check, submit
        srv, trace, result = run("single", ON, turns=turns)
        self.assertEqual((noted(trace), result["post_pass"], result["stop_reason"]), ([], None, "submitted"))
        # an all-True printout followed by a traceback, or a failing pair among passing ones: no note
        for code in (PASS_LOOP + "print(undefined_name)", PASS_LOOP + "print(np.array_equal(solve(train[0]['input']), "
                     "train[1]['output']))", FLIP + "for p in train[:1]:\n    print(np.array_equal(solve(p['input']), "
                     "p['output']))\nprint(True)"):
            turns = [C.ds_call_text(code), C.ds_call_text("[[1]]", name="submit_answer")]
            srv, trace, result = run("single", ON, turns=turns)
            self.assertEqual((noted(trace), result["post_pass"]), ([], None), code)

    def test_not_after_a_submit(self):
        # the all-pass comes in the same turn as a valid submit: the attempt ends there, no note
        turns = single_turns()[:3] + [two_calls(PASS_LOOP, "solve(test_input)")]
        srv, trace, result = run("single", ON, turns=turns)
        self.assertEqual((result["stop_reason"], result["pred"], result["turns"]), ("submitted", F0, 4))
        self.assertEqual((noted(trace), result["post_pass"]), ([], None))
        # an invalid submit is not an answer: the all-pass after it gets the note
        turns = single_turns()
        turns[2] = C.ds_call_text("[[1, 2], [3]]", name="submit_answer")
        srv, trace, result = run("single", ON, turns=turns)
        self.assertTrue(tools(trace)[2].startswith("[answer not accepted]"))
        self.assertEqual(noted(trace), [4])
        # the note goes on the run_python result that passed, also when a second call follows in the turn
        turns = single_turns()
        turns[3] = two_calls(PASS_LOOP, "[[10]]")
        srv, trace, result = run("single", ON, turns=turns)
        self.assertEqual(noted(trace), [4])
        self.assertTrue(tools(trace)[4].startswith("[answer not accepted]"), tools(trace)[4])

    def test_not_when_no_turn_can_follow(self):
        # the all-pass turn spends the budget (4 x 21k > 80k) or is the last turn: the note would never be read
        srv, trace, result = run("single", ON, completion_tokens=21000)
        self.assertEqual((result["stop_reason"], result["turns"], noted(trace), result["post_pass"]),
                         ("output_budget", 4, [], None))
        srv, trace, result = run("single", ON, max_turns=4)
        self.assertEqual((result["stop_reason"], noted(trace), result["post_pass"]), ("max_turns", [], None))
        # with one turn left it fires; the turn cap then ends the attempt without a submit
        srv, trace, result = run("single", ON, max_turns=5)
        self.assertEqual((result["stop_reason"], noted(trace)), ("max_turns", [4]))
        self.assertEqual(result["post_pass"], {"fired_at_turn": 4, "fired_at_tokens": 36000, "turns_after": 1,
                                               "tokens_after": 9000, "submitted_after": False})

    def test_order_with_best_guess(self):
        # both notes due on the same tool result: post-pass, then best-guess, then the budget note
        srv, trace, result = run("single", ON + ["--best-guess-at", "0.4"])
        self.assertEqual(noted(trace), [4])
        self.assertIn(NOTE + H.BEST_GUESS_NOTE + "\n[budget: 36k", tools(trace)[3])
        self.assertEqual((result["best_guess"]["fired_at_turn"], result["post_pass"]["fired_at_turn"]), (4, 4))


class Detector(unittest.TestCase):
    """train_pass_shown(code, output, n_train) on the shapes E128 prints (and the ones it must not read as a pass)."""
    PASS = [
        ("for i,p in enumerate(train):\n    o=solve(p['input'])\n    print(i, np.array_equal(o,np.array(p['output'])))",
         "0 True\n1 True\n2 True\n"),
        ("for i,t in enumerate(train):\n    g=np.array(t['input']); o=np.array(t['output'])\n    r=solve(g)\n"
         "    print(i, np.array_equal(r,o), np.sum(r!=o))\n    if not np.array_equal(r,o): print(np.argwhere(r!=o))",
         "0 True 0\n1 True 0\n2 True 0\n"),
        ("for i,t in enumerate(train):\n    out,mp,key=solve(t['input'])\n    print(i,'map',mp,'match',"
         "np.array_equal(out,np.array(t['output'])))\nout,mp,key=solve(test_input)\nprint('test key',key)\nprint(out)",
         "0 map {3: 3} match True\n1 map {2: 6} match True\n2 map {2: 7} match True\ntest key (0, 1)\n[[0 1]\n [1 0]]\n"),
        ("for i in range(len(train)):\n    ok = np.array_equal(solve(train[i]['input']), train[i]['output'])\n"
         "    print('pair', i, ok)", "pair 0 True\npair 1 True\npair 2 True\n"),
        ("print([np.array_equal(solve_f(e['input']),np.array(e['output'])) for e in train])", "[True, True, True]\n"),
        ("print('train ok', all(np.array_equal(solve(g['input']), np.array(g['output'])) for g in train))",
         "train ok True\n"),
        ("for t in train:\n    assert np.array_equal(solve(t['input']), np.array(t['output']))\nprint('ok')", "ok\n"),
        ("check(solve)", "pair 0: OK\npair 1: OK\npair 2: OK\nALL PASS\n"),
        # a pipeline of functions on the input, one constant parameter for every pair
        ("for k,p in enumerate(train):\n    bg,blocks,a=parse(np.array(p['input']))\n    out=apply(a,blocks[0],bg)\n"
         "    print('ex',k,'match',np.array_equal(out,np.array(p['output'])))", "ex 0 match True\nex 1 match True\n"
                                                                             "ex 2 match True\n"),
        ("for ne in [ne8]:\n    for i,p in enumerate(train):\n        out=solve_grid(p['input'],ne)\n"
         "        print('EX',i,'match',np.array_equal(out,np.array(p['output'])))", "EX 0 match True\nEX 1 match True\n"
                                                                                  "EX 2 match True\n"),
    ]
    NOT = [
        # a failing pair, a nonzero diff count, an error, truncation
        (PASS[0][0], "0 True\n1 False\n2 True\n"),
        (PASS[1][0], "0 True 0\n1 True 0\n2 True 0\nmismatches 2\n"),
        (PASS[0][0], "0 True\n1 True\n2 True\nTraceback (most recent call last):\nNameError: name 'x' is not defined\n"),
        (PASS[0][0], "0 True\n1 True\n2 True\n[output truncated]"),
        # fewer True lines than pairs (a break, an except branch)
        (PASS[0][0], "0 True\n1 True\n"),
        # no program: shape / colour-set / property checks, a function of the output, hand-fit per-pair arguments
        ("for p in train:\n    a=np.array(p['input']); b=np.array(p['output'])\n    print(a.shape == b.shape)",
         "True\nTrue\nTrue\n"),
        ("for p in train:\n    print(set(np.unique(solve(p['input']))) == set(np.unique(p['output'])))",
         "True\nTrue\nTrue\n"),
        ("for i,tr in enumerate(train):\n    b=np.array(tr['output'])\n    print(i, np.array_equal(b, b.T))",
         "0 True\n1 True\n2 True\n"),
        ("for i,p in enumerate(train):\n    g=np.array(p['input']); o=np.array(p['output'])\n"
         "    mc=int(list(set(o[g!=o].flatten()))[0])\n    pred=solve_train2(g,mc)\n    print(i, np.array_equal(pred,o))",
         "0 True\n1 True\n2 True\n"),
        ("for i,t in enumerate(train):\n    o=np.array(t['output'])\n    s=shape_of(o)\n"
         "    print(i, np.array_equal(get_shape(t['input']), s))", "0 True\n1 True\n2 True\n"),
        ("print(np.array_equal(solve(train[0]['input'],[0,1]),np.array(train[0]['output'])))\n"
         "print(np.array_equal(solve(train[1]['input'],[0,4]),np.array(train[1]['output'])))\n"
         "print(np.array_equal(solve(train[2]['input'],[1,2]),np.array(train[2]['output'])))", "True\nTrue\nTrue\n"),
        # hand-fit per-pair arguments: indexed by the loop counter, a zip partner, a branch on the counter
        ("for i,p in enumerate(train):\n    sep=[2,2,6][i]\n    b=solve(p['input'],sep)\n"
         "    print('ex',i,'equal',np.array_equal(b,np.array(p['output'])))", "ex 0 equal True\nex 1 equal True\n"
                                                                             "ex 2 equal True\n"),
        ("for p, w in zip(train, walls):\n    print(np.array_equal(solve(p['input'], w), p['output']))",
         "True\nTrue\nTrue\n"),
        ("for i,p in enumerate(train):\n    g=np.array(p['input']); s=7 if i==0 else 6\n    out,fs,ref=solve(g,s)\n"
         "    print(i, np.array_equal(out,np.array(p['output'])))", "0 True\n1 True\n2 True\n"),
        ("for idx,t in enumerate(train):\n    a=np.array(t['input'])\n    if idx==0:\n        C=a[:,:7]\n    else:\n"
         "        C=a[11:,:]\n    print(idx, np.array_equal(fill(C), t['output']))", "0 True\n1 True\n2 True\n"),
        # a slice of train, any() over train, a negated assert
        ("for p in train[:2]:\n    print(np.array_equal(solve(p['input']), p['output']))\nprint(True)",
         "True\nTrue\nTrue\n"),
        ("print(any(np.array_equal(solve(p['input']), p['output']) for p in train))", "True\n"),
        ("for t in train:\n    assert not np.array_equal(solve(t['input']), t['output'])", ""),
        # the comparison is not printed (only a property of the result is)
        ("for p in train:\n    o=solve(p['input'])\n    ok=np.array_equal(o, p['output'])\n    print(o.shape == (2, 2))",
         "True\nTrue\nTrue\n"),
        # check() on a subset of the pairs, and a cell that does not parse
        ("check(solve, train[:1])", "pair 0: OK\nALL PASS\n"),
        ("%time check(solve)", "ALL PASS\n"),
    ]

    def test_cases(self):
        for code, out in self.PASS:
            self.assertTrue(H.train_pass_shown(code, "[completed]\n" + out, 3), code)
        for code, out in self.NOT:
            self.assertFalse(H.train_pass_shown(code, "[completed]\n" + out, 3), (code, out))

    def test_status_and_test_lines(self):
        code, out = self.PASS[0]
        for head in ("[error]\n", "[timeout]\n", ""):
            self.assertFalse(H.train_pass_shown(code, head + out, 3), head)
        # an auto-repair note before the output is fine; lines about the test input are not verdicts
        self.assertTrue(H.train_pass_shown(code, "[auto-corrected: slove -> solve]\n[completed]\n" + out, 3))
        self.assertTrue(H.train_pass_shown(code, "[completed]\n" + out + "test colours differ: False\n", 3))
        self.assertFalse(H.train_pass_shown(code, "[completed]\n" + out, 4))  # 3 True lines for 4 pairs


class Cli(unittest.TestCase):
    def test_needs_answer_tool(self):
        err = io.StringIO()
        with patch("sys.argv", ["run_deepseek_tools.py", "--post-pass-note", "--dry-run"]), redirect_stderr(err), \
                self.assertRaises(SystemExit):
            H.main()
        self.assertIn("--post-pass-note needs --answer-tool", err.getvalue())


if __name__ == "__main__":
    unittest.main()
