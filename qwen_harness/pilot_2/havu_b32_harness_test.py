"""vLLM-free unit tests for the Havu Candidate B32 harness port (`havu_b32_harness.py`).

Covers every pure-Python path that runs inside Borg work units without a model:
the isolated `LocalExecutor` child-process sandbox, `submit_answer` / A1-A11
fallbacks, `auto_repair` (I5), Qwen XML parsing + `--rescue-code`, budget notes
(Fix A9), the 4-path `train_check` + `ranked_answers`, task sharding / resume,
queue scheduling, and an end-to-end `_process_completed_turn` multi-test rollout
driven by scripted completions.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import struct
import tempfile

# Mirror production (`pilot2_eval.py`): fork the sandbox zygote before heavier imports.
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_sandbox_zygote  # pylint: disable=g-bad-import-order

havu_sandbox_zygote.ensure_started()

import numpy as np  # pylint: disable=g-import-not-at-top,g-bad-import-order
from PIL import Image  # pylint: disable=g-import-not-at-top,g-bad-import-order

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import grid_image  # pylint: disable=g-import-not-at-top,g-bad-import-order
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import havu_b32_harness as hb  # pylint: disable=g-import-not-at-top,g-bad-import-order
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector  # pylint: disable=g-import-not-at-top,g-bad-import-order
from google3.experimental.users.arazdai.kaggle.sft import dsv41_image_tokens  # pylint: disable=g-import-not-at-top,g-bad-import-order
from google3.testing.pybase import googletest  # pylint: disable=g-import-not-at-top,g-bad-import-order


TRAIN = [
    {"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]]},
    {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
    {"input": [[1, 1, 0]], "output": [[2, 2, 0]]},
]
TEST_INPUT = [[0, 0, 1, 0], [0, 1, 1, 1]]
TEST_OUTPUT = [[0, 0, 2, 0], [0, 2, 2, 2]]
TEST_INPUT_B = [[1, 1], [1, 0], [0, 0]]
TEST_OUTPUT_B = [[2, 2], [2, 0], [0, 0]]

SOLVE_SRC = "def solve(g):\n    return [[2 if v == 1 else v for v in row] for row in g]\n"


def _fake_encode(s: str) -> list[int]:
  return [ord(c) % 251 + 1 for c in s]


def _im(reasoning: str, body: str) -> str:
  r"""Build a raw Qwen completion (after `<|im_start|>assistant\n<think>\n`)."""
  return f"{reasoning}\n</think>\n{body}<|im_end|>"


def _call(fn: str, code: str) -> str:
  return (
      f"<tool_call>\n<function={fn}>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"
  )


class LocalExecutorTest(googletest.TestCase):

  def test_preloaded_state_persists_and_ipython_display(self) -> None:
    with hb.LocalExecutor(TRAIN, TEST_INPUT, timeout=10.0) as s:
      r = s.execute("len(train), np.array(test_input).shape")
      self.assertEqual(r["process_status"], "completed")
      self.assertIn("(3, (2, 4))", r["stdout"])
      s.execute("x = 41")
      r = s.execute("x + 1")
      self.assertEqual(r["stdout"].strip(), "42")
      r = s.execute("print('hi'); 'tail'")
      self.assertEqual(r["stdout"].strip().splitlines(), ["hi", "'tail'"])

  def test_error_traceback_is_ipython_style(self) -> None:
    with hb.LocalExecutor(TRAIN, TEST_INPUT, timeout=10.0) as s:
      r = s.execute("y = undefined_name + 1")
      self.assertEqual(r["process_status"], "error")
      self.assertIn("NameError: name 'undefined_name' is not defined", r["stdout"])
      self.assertNotIn('File "<string>"', r["stdout"])
      txt = hb.tool_output(r)
      self.assertTrue(txt.startswith("[error]\n"))

  def test_timeout_sigint_keeps_state_then_sigkill_resets(self) -> None:
    with hb.LocalExecutor(TRAIN, TEST_INPUT, timeout=1.0, grace=3.0) as s:
      s.execute("keep = 7")
      r = s.execute("import time\nwhile True: time.sleep(0.05)")
      self.assertEqual(r["process_status"], "timeout")
      self.assertFalse(r.get("state_reset"))
      self.assertEqual(s.execute("keep").get("stdout", "").strip(), "7")
      # Uninterruptible C-level sleep ignores SIGINT -> SIGKILL + respawn with fresh prelude.
      r = s.execute(
          "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(30)"
      )
      self.assertTrue(r.get("state_reset"))
      self.assertIn("timed out", r["stderr"])
      r = s.execute("'keep' in globals(), len(train)")
      self.assertIn("(False, 3)", r["stdout"])

  def test_prelude_extra_preloads_test_inputs(self) -> None:
    inputs = [TEST_INPUT, TEST_INPUT_B]
    with hb.LocalExecutor(
        TRAIN, TEST_INPUT, timeout=10.0, prelude_extra=hb.inputs_prelude(inputs)
    ) as s:
      r = s.execute("test_inputs == [test_input, [[1, 1], [1, 0], [0, 0]]]")
      self.assertEqual(r["stdout"].strip(), "True")

  def test_output_truncation(self) -> None:
    with hb.LocalExecutor(TRAIN, TEST_INPUT, timeout=10.0, max_output=200) as s:
      r = s.execute("print('x' * 1000)")
      self.assertLess(len(r["stdout"]), 260)
      self.assertIn("<output cut>", r["stdout"])


class SubmitAndFallbackTest(googletest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.s = hb.LocalExecutor(
        TRAIN,
        TEST_INPUT,
        timeout=10.0,
        prelude_extra=hb.inputs_prelude([TEST_INPUT, TEST_INPUT_B]),
    )
    self.s.initialize()
    self.addCleanup(self.s.close)

  def test_submit_expression_list_numpy_and_statement(self) -> None:
    self.s.execute(SOLVE_SRC)
    out, grid = hb.submit_answer(self.s, "solve(test_input)")
    self.assertEqual(grid, TEST_OUTPUT)
    self.assertIn("[answer accepted: 2x4 grid]", out)
    _, grid = hb.submit_answer(self.s, "np.array(solve(test_input))")
    self.assertEqual(grid, TEST_OUTPUT)
    _, grid = hb.submit_answer(self.s, "answer = solve(test_input)")
    self.assertEqual(grid, TEST_OUTPUT)
    self.assertEqual(self.s.execute("answer == solve(test_input)")["stdout"].strip(), "True")

  def test_submit_rejects_invalid_grids(self) -> None:
    out, grid = hb.submit_answer(self.s, "[[1, 2], [3]]")
    self.assertIsNone(grid)
    self.assertIn("[answer not accepted]", out)
    _, grid = hb.submit_answer(self.s, "[[10]]")
    self.assertIsNone(grid)
    _, grid = hb.submit_answer(self.s, "[[True]]")  # Python bool rejected
    self.assertIsNone(grid)
    out, grid = hb.submit_answer(self.s, "undefined_fn(test_input)")
    self.assertIsNone(grid)
    self.assertIn("NameError", out)

  def test_submit_multi_test_list_and_nested_ndarray_a10(self) -> None:
    self.s.execute(SOLVE_SRC)
    out, grids = hb.submit_answer(self.s, "[np.array(solve(t)) for t in test_inputs]", 2)
    self.assertEqual(grids, [TEST_OUTPUT, TEST_OUTPUT_B])
    self.assertIn("[answer accepted: 2 grids, 2x4, 3x2]", out)
    out, grids = hb.submit_answer(self.s, "solve(test_input)", 2)
    self.assertIsNone(grids)
    self.assertIn("this task has 2 test inputs", out)
    out, grids = hb.submit_answer(self.s, "[solve(test_input), [[1, 2], [3]]]", 2)
    self.assertIsNone(grids)
    self.assertIn("grid 1 (test input 1)", out)

  def test_submit_is_shadow_proof_a7(self) -> None:
    self.s.execute(SOLVE_SRC)
    self.s.execute("def eval(x):\n    raise RuntimeError('shadowed')\nprint = None\nlen = None")
    _, grid = hb.submit_answer(self.s, "solve(test_input)")
    self.assertEqual(grid, TEST_OUTPUT)

  def test_submit_large_grid_uses_chunked_slices_a11(self) -> None:
    self.s.max_output = 1200  # force multi-chunk ANSWER_SLICE transfer
    _, grid = hb.submit_answer(self.s, "[[7] * 30 for _ in range(30)]")
    self.assertEqual(grid, [[7] * 30] * 30)
    self.assertEqual(self.s.execute("[k for k in globals() if k.startswith('_arc_text')]")["stdout"].strip(), "[]")

  def test_answer_items_partial_validity_a1(self) -> None:
    self.s.execute("answer = [[[0, 0, 2, 0], [0, 2, 2, 2]], 'oops']")
    out, grids = hb.answer_items(self.s, "answer", 2)
    self.assertEqual(grids, [TEST_OUTPUT, None])
    self.assertIn("1 of 2 grids valid", out)
    self.assertIn("grid 1: str", out)

  def test_solve_items_numpy_retry_a3_and_restore_inputs_a8(self) -> None:
    # `list + bool` raises TypeError, so only the `np.array` retry (Fix A3) can succeed.
    self.s.execute("def solve(g):\n    return g + (g == 1)\n")
    out, grids = hb.solve_items(self.s, [TEST_INPUT, TEST_INPUT_B])
    self.assertEqual(grids, [TEST_OUTPUT, TEST_OUTPUT_B])
    self.assertIn("2 of 2 grids valid", out)
    self.s.execute("test_input = [[9]]\ntest_inputs = None")
    hb.restore_inputs(self.s, [TEST_INPUT, TEST_INPUT_B], with_inputs=True)
    r = self.s.execute("test_input == [[0, 0, 1, 0], [0, 1, 1, 1]] and len(test_inputs) == 2")
    self.assertEqual(r["stdout"].strip(), "True")

  def test_auto_repair_fixes_unique_identifier_typo_i5(self) -> None:
    self.s.execute(SOLVE_SRC)
    code = "res = solve(test_inpt)\nprint(res)"
    r = self.s.execute(code)
    self.assertEqual(r["process_status"], "error")
    stats = {"proposed": 0, "applied": 0, "retry_ok": 0}
    fixed = hb.auto_repair_cell(
        session=self.s,
        code=code,
        output=hb.tool_output(r),
        names=hb.PRELOADED | {"test_inputs"},
        limit=16000,
        stats=stats,
    )
    self.assertIsNotNone(fixed)
    out, status, new_code = fixed
    self.assertEqual(status, "completed")
    self.assertIn("test_input", new_code)
    self.assertNotIn("test_inpt", new_code)
    self.assertIn("[auto-corrected:", out)
    self.assertEqual(stats, {"proposed": 1, "applied": 1, "retry_ok": 1})

  def test_auto_repair_refuses_after_side_effects(self) -> None:
    self.s.execute("acc = []")
    code = "acc.append(1)\nprint(undefined_zz)"
    r = self.s.execute(code)
    fixed = hb.auto_repair_cell(self.s, code, hb.tool_output(r), set(), 16000, {"proposed": 0, "applied": 0, "retry_ok": 0})
    self.assertIsNone(fixed)


class ParsingAndNotesTest(googletest.TestCase):

  def test_parse_qwen_completion_with_tool_call(self) -> None:
    raw = _im("let me test", _call("run_python", "print(1)"))
    reasoning, content, calls, clean = hb.parse_qwen_completion(raw)
    self.assertEqual(reasoning.strip(), "let me test")
    self.assertEqual(content, "")
    self.assertTrue(clean)
    self.assertLen(calls, 1)
    self.assertEqual(calls[0]["function"]["name"], "run_python")
    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"code": "print(1)"})

  def test_parse_qwen_completion_unclosed_think_is_not_clean(self) -> None:
    reasoning, content, calls, clean = hb.parse_qwen_completion("still thinking...")
    self.assertEqual(reasoning, "still thinking...")
    self.assertEqual((content, calls, clean), ("", [], False))

  def test_rescue_code_variants(self) -> None:
    self.assertEqual(hb.rescue_code("blah\n```python\nprint(2)\n```"), "print(2)")
    self.assertEqual(
        hb.rescue_code("<tool_call>\n<function=run_python>\n<parameter=code>\nx = 1\nprint(x)"),
        "x = 1\nprint(x)",
    )
    self.assertEqual(hb.rescue_code('{"name": "run_python", "arguments": {"code": "print(3)\\n"}}'), "print(3)\n")
    self.assertEqual(hb.rescue_code("Let me try:\nimport numpy as np\nprint(np.eye(2))"), "import numpy as np\nprint(np.eye(2))")
    self.assertIsNone(hb.rescue_code("The answer is probably 4."))
    self.assertEqual(hb.rescue_name("submit"), "submit_answer")
    self.assertEqual(hb.rescue_name("python"), "run_python")
    call = hb.rescue_tool_call("```py\nprint(9)\n```", 3)
    self.assertEqual(call["id"], "rescue_3")
    self.assertEqual(json.loads(call["function"]["arguments"])["code"], "print(9)")

  def test_grid_parsers(self) -> None:
    self.assertEqual(hb.parse_arcprize_grid("final: \\boxed{[[1, 2], [3, 4]]}"), [[1, 2], [3, 4]])
    self.assertEqual(hb.parse_arcprize_grid("so the grid is [[0, 1], [1, 0]] done"), [[0, 1], [1, 0]])
    self.assertEqual(hb.lenient_parse("rows:\n[0, 1, 2],\n[3, 4, 5]\n"), [[0, 1, 2], [3, 4, 5]])
    self.assertEqual(hb.lenient_parse("[[1, 2,], [3, 4,]]"), [[1, 2], [3, 4]])
    self.assertIsNone(hb.lenient_parse("nothing here"))

  def test_budget_turn_and_context_notes(self) -> None:
    self.assertEqual(hb.budget_note_b32(100, 80000, 40000), "")
    gentle = hb.budget_note_b32(45000, 80000, 40000)
    self.assertIn("aim to finish within 40k (hard limit 80k)", gentle)
    self.assertIn("submit as soon as a rule fits every training pair", gentle)
    self.assertNotIn("unless one more experiment", gentle)
    self.assertIn("Budget nearly spent", hb.budget_note_b32(61000, 80000, 40000))
    self.assertEqual(hb.turns_note_b32(40, 10), "")
    self.assertIn("[1 turn left:", hb.turns_note_b32(40, 38))
    self.assertEqual(hb.turns_note_b32(40, 39), "")
    # Fix A9: remaining context room caps hard budget; soft target collapses when exceeded.
    self.assertEqual(hb.context_adjusted_budgets(80000, 40000, 131072, 2000, 1000), (80000, 40000))
    room = 131072 - 120000 - 16 - hb.CONTEXT_MIN_ROOM  # 10032 tokens of context left
    hard, soft = hb.context_adjusted_budgets(120000, 60000, 131072, 120000, 40000)
    self.assertEqual((hard, soft), (40000 + room, 0))  # soft (60k) >= eff_hard -> collapses
    hard, soft = hb.context_adjusted_budgets(120000, 60000, 131072, 120000, 50000)
    self.assertEqual((hard, soft), (50000 + room, 60000))  # soft < eff_hard -> kept
    self.assertGreater(hb.gzip_ratio("abc " * 2000), 8.0)
    self.assertLess(hb.gzip_ratio("the quick brown fox jumps over the lazy dog"), 2.0)

  def test_multi_check_code_a6(self) -> None:
    inputs = [TEST_INPUT, TEST_INPUT_B]
    self.assertEqual(hb.multi_check_code("answer", inputs), "answer[0]")
    expr = hb.multi_check_code("[solve(t) for t in test_inputs]", inputs)
    self.assertTrue(expr.startswith("(lambda test_inputs: ([solve(t) for t in test_inputs])[0])([test_input] + "))
    stmt = hb.multi_check_code("answer = [solve(t) for t in test_inputs]", inputs)
    self.assertTrue(stmt.startswith("test_inputs = [test_input] + "))
    self.assertTrue(stmt.endswith("\nanswer = answer[0]"))
    self.assertIsNone(hb.multi_check_code(None, inputs))

  def test_render_full_messages_round_trip(self) -> None:
    sys_block = hb.render_system_qwen(
        tools=hb.tools_spec_b32(),
        tier_text=hb.TIER_SYSTEM_XHIGH,
        effort_header=hb.EFFORT_HEADER_MINIMAL,
    )
    self.assertIn("<tools>", sys_block)
    self.assertIn("test_inputs", sys_block)
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "reasoning_content": "think", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_python", "arguments": json.dumps({"code": "print(1)"})}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "[completed]\n1\n"},
    ]
    text = hb.render_full_messages_qwen(sys_block, msgs, add_generation_prompt=True)
    self.assertIn("<|im_start|>user\ntask<|im_end|>", text)
    self.assertIn("<function=run_python>", text)
    self.assertIn("<tool_response>", text)
    self.assertTrue(text.endswith(hb.GEN_PROMPT))

  def test_render_full_messages_and_parse_dsv41_dsml_round_trip(self) -> None:
    sys_block = hb.render_system_dsv41(
        tools=hb.tools_spec_b32(),
        tier_text="",
        reasoning_effort=50,
    )
    self.assertTrue(sys_block.startswith(hb.DS41_BOS_TOKEN + hb.DS41_SYSTEM_SP_TOKEN))
    self.assertIn("Reasoning Effort: 50", sys_block)
    self.assertIn("<｜DSML｜ calls>", sys_block)
    msgs = [
        {"role": "user", "content": "Solve ARC puzzle."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let's test a rule.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": json.dumps({"code": "print(len(train))"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[completed]\n3\n"},
    ]
    text = hb.render_full_messages_dsv41(sys_block, msgs, add_generation_prompt=True)
    self.assertIn("<｜User｜>Solve ARC puzzle.", text)
    self.assertIn('<｜DSML｜ invoke name="run_python">', text)
    self.assertIn("<tool_result>[completed]\n3\n</tool_result>", text)
    self.assertTrue(text.endswith(hb.DS41_GEN_PROMPT))

    raw_comp = (
        "Verified on all pairs.</think>\n\n"
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="submit_answer">\n'
        '<｜DSML｜ parameter name="code" string="true">[solve(t) for t in test_inputs]</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
        + hb.DS41_EOS_TOKEN
    )
    reasoning, content, calls, clean = hb.parse_dsv41_completion(raw_comp)
    self.assertTrue(clean)
    self.assertEqual(reasoning, "Verified on all pairs.")
    self.assertEqual(content, "")
    self.assertLen(calls, 1)
    self.assertEqual(calls[0]["function"]["name"], "submit_answer")
    self.assertEqual(
        json.loads(calls[0]["function"]["arguments"]),
        {"code": "[solve(t) for t in test_inputs]"},
    )


class TrainCheckAndRankingTest(googletest.TestCase):

  def _session(self, test_input=None, inputs=None):
    test_input = TEST_INPUT if test_input is None else test_input
    s = hb.LocalExecutor(
        TRAIN,
        test_input,
        timeout=10.0,
        prelude_extra=hb.inputs_prelude(inputs or [test_input]),
    )
    s.initialize()
    self.addCleanup(s.close)
    return s

  def test_expr_path(self) -> None:
    s = self._session()
    s.execute(SOLVE_SRC)
    label = hb.train_check(s, [("run_python", SOLVE_SRC)], TEST_OUTPUT, "solve(test_input)")
    self.assertTrue(label["train_pass"])
    self.assertEqual(label["train_trace"], "expr")
    self.assertEqual(label["train_frac"], 1.0)

  def test_expr_path_partial_train_frac(self) -> None:
    s = self._session()
    bad = "def solve(g):\n    return [[2 if v == 1 else v for v in row] for row in g] if len(g) > 1 else g\n"
    s.execute(bad)
    label = hb.train_check(s, [("run_python", bad)], TEST_OUTPUT, "solve(test_input)")
    self.assertFalse(label["train_pass"])
    self.assertAlmostEqual(label["train_frac"], 2 / 3, places=3)

  def test_cell_path_for_imperative_answer(self) -> None:
    s = self._session()
    cell = "answer = [[2 if v == 1 else v for v in row] for row in test_input]\nprint(answer)"
    s.execute(cell)
    label = hb.train_check(s, [("run_python", cell), ("submit_answer", "answer")], TEST_OUTPUT, "answer")
    self.assertTrue(label["train_pass"])
    self.assertEqual(label["train_trace"], "cell")

  def test_fn_path_for_custom_function_name(self) -> None:
    s = self._session()
    fn = "def transform(grid):\n    g = np.array(grid)\n    return np.where(g == 1, 2, g)\n"
    s.execute(fn)
    # Prediction came from text (code=None); no session variable holds it.
    label = hb.train_check(s, [("run_python", fn)], TEST_OUTPUT, None)
    self.assertTrue(label["train_pass"])
    self.assertEqual(label["train_trace"], "fn")

  def test_transcript_replay_path(self) -> None:
    s = self._session()
    executed = [("run_python", SOLVE_SRC), ("submit_answer", "solve(test_input)")]
    s.execute(SOLVE_SRC)
    s.execute("del solve")  # live session can no longer reproduce; only a fresh replay can
    label = hb.train_check(
        s,
        executed,
        TEST_OUTPUT,
        "solve(test_input)",
        replay_session=lambda grid: hb.LocalExecutor(TRAIN, grid, timeout=10.0),
    )
    self.assertTrue(label["train_pass"])
    self.assertEqual(label["train_trace"], "transcript")

  def test_untraceable(self) -> None:
    s = self._session()
    label = hb.train_check(s, [], TEST_OUTPUT, None)
    self.assertIsNone(label["train_pass"])
    self.assertEqual(label["train_trace"], "untraceable")

  def test_multi_test_expr_path_via_multi_check_code(self) -> None:
    inputs = [TEST_INPUT, TEST_INPUT_B]
    s = self._session(inputs=inputs)
    s.execute(SOLVE_SRC)
    code = hb.multi_check_code("[solve(t) for t in test_inputs]", inputs)
    label = hb.train_check(s, [("run_python", SOLVE_SRC)], TEST_OUTPUT, code)
    self.assertTrue(label["train_pass"])
    self.assertEqual(label["train_trace"], "expr")

  def test_ranked_answers_and_majority_vote(self) -> None:
    g1, g2, g3 = [[1]], [[2]], [[3]]
    recs = [
        {"attempt": 0, "pred": g1, "train_pass": False},
        {"attempt": 1, "pred": g2, "train_pass": True},
        {"attempt": 2, "pred": g1, "train_pass": None},
        {"attempt": 3, "pred": g3, "train_pass": True},
        {"attempt": 4, "pred": "junk", "train_pass": True},
    ]
    top = hb.ranked_answers(recs, k=2)
    self.assertEqual([r["pred"] for r in top], [g2, g3])
    self.assertEqual([r["attempt"] for r in top], [1, 3])
    self.assertEqual(hb.majority_vote_ranked_answers(recs, k=2), [g2, g3])
    self.assertEqual(hb.majority_vote_ranked_answers(recs[:1] + recs[2:3], k=2), [g1])


class ShardingResumeAndQueueTest(googletest.TestCase):

  def _write_tasks(self, n_tests: list[int]) -> str:
    d = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    for i, k in enumerate(n_tests):
      with open(os.path.join(d, f"task{i:02d}.json"), "w") as f:
        json.dump({"train": TRAIN, "test": [{"input": TEST_INPUT, "output": TEST_OUTPUT}] * k}, f)
    return d

  def test_load_eval_tasks_keeps_pairs_together_across_shards(self) -> None:
    d = self._write_tasks([1, 3, 1, 2, 1])
    s0, n0 = hb.load_eval_tasks(d, 0, 2)
    s1, n1 = hb.load_eval_tasks(d, 1, 2)
    self.assertEqual([t.task_id for t in s0], ["task00", "task02", "task04"])
    self.assertEqual([t.task_id for t in s1], ["task01", "task03"])
    self.assertEqual((n0, n1), (3, 5))
    all_pairs = sorted(gp for t in s0 + s1 for gp in t.global_pair_indices)
    self.assertEqual(all_pairs, list(range(8)))
    self.assertEqual(s1[0].global_pair_indices, [1, 2, 3])
    lim, _ = hb.load_eval_tasks(d, 0, 1, max_tasks=2)
    self.assertLen(lim, 2)

  def test_load_existing_pair_records_rebuilds_submission(self) -> None:
    d = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    p = os.path.join(d, "pairs_shard_00_of_01.jsonl")
    recs = [
        {"global_pair_idx": 2, "task_id": "t1", "test_idx": 1, "attempt_1": [[1]], "attempt_2": [[2]]},
        {"global_pair_idx": 1, "task_id": "t1", "test_idx": 0, "attempt_1": [[3]], "attempt_2": [[4]]},
        {"global_pair_idx": 5, "task_id": "t2", "test_idx": 0, "attempt_1": [[5]], "attempt_2": [[5]]},
    ]
    with open(p, "w") as f:
      f.write("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
    by_idx, sub = hb._load_existing_pair_records(p)  # pylint: disable=protected-access
    self.assertEqual(sorted(by_idx), [1, 2, 5])
    self.assertEqual(sub["t1"], [{"attempt_1": [[3]], "attempt_2": [[4]]}, {"attempt_1": [[1]], "attempt_2": [[2]]}])
    self.assertEqual(sub["t2"], [{"attempt_1": [[5]], "attempt_2": [[5]]}])
    self.assertEqual(hb._load_existing_pair_records(os.path.join(d, "missing.jsonl")), ({}, {}))  # pylint: disable=protected-access

  def test_pick_next_queued_job_priorities(self) -> None:
    def spec(i: int) -> hb.EvalTaskSpec:
      return hb.EvalTaskSpec(i, f"t{i}", TRAIN, [{"input": TEST_INPUT, "output": TEST_OUTPUT}], [i], 10 * i)

    class _Done:

      def __init__(self, tp):
        self.labels = {"train_pass": tp}

    queue = [(spec(0), 1), (spec(1), 1), (spec(2), 0), (spec(3), 1)]
    self.assertEqual(hb._pick_next_queued_job(queue, {}), 2)  # pylint: disable=protected-access
    queue.pop(2)
    finished = {"t0": [_Done(True)], "t3": [_Done(False)]}
    self.assertEqual(hb._pick_next_queued_job(queue, finished), 2)  # t3: finished w/o train_pass  # pylint: disable=protected-access
    finished = {"t0": [_Done(True)], "t1": [_Done(True)], "t3": [_Done(True)]}
    self.assertEqual(hb._pick_next_queued_job(queue, finished), 0)  # pylint: disable=protected-access


class ProcessCompletedTurnTest(googletest.TestCase):
  """End-to-end attempt lifecycle with scripted completions (no vLLM)."""

  def setUp(self) -> None:
    super().setUp()
    self.cfg = prompts_and_selector.PILOT2_HYPOTHESES["havu_b32"]
    self.tools = hb.tools_spec_b32()
    self.system_block = hb.render_system_qwen(
        tools=self.tools, tier_text=hb.TIER_SYSTEM_XHIGH, effort_header=hb.EFFORT_HEADER_MINIMAL
    )

  def _state(self, test_pairs) -> hb.HavuAttemptState:
    spec = hb.EvalTaskSpec(
        task_idx=0,
        task_id="demo",
        train_pairs=TRAIN,
        test_pairs=test_pairs,
        global_pair_indices=list(range(len(test_pairs))),
        prompt_size_bytes=100,
    )
    st = hb._create_attempt_state(spec, 0, self.cfg, self.tools, self.system_block)  # pylint: disable=protected-access
    self.addCleanup(st.session.close)
    st.prompt_token_ids = _fake_encode(
        hb.render_full_messages_qwen(self.system_block, st.messages, add_generation_prompt=True)
    )
    return st

  def _turn(self, st, raw, finish="stop"):
    st.n_turns += 1
    ids = _fake_encode(raw)
    st.cum_completion_tokens += len(ids)
    st.last_prompt_plus_comp = len(st.prompt_token_ids or []) + len(ids)
    return hb._process_completed_turn(st, raw, ids, finish, 131072, _fake_encode)  # pylint: disable=protected-access

  def test_multi_test_scaling_prompt_and_submit_flow(self) -> None:
    pairs = [
        {"input": TEST_INPUT, "output": TEST_OUTPUT},
        {"input": TEST_INPUT_B, "output": TEST_OUTPUT_B},
    ]
    st = self._state(pairs)
    self.assertEqual(st.multi, 2)
    self.assertEqual((st.max_total_output, st.budget_soft, st.max_turns), (120000, 60000, 60))
    self.assertEqual(st.max_tokens_per_turn, 64000)
    self.assertIn("--Test input 0--", st.messages[0]["content"])
    self.assertIn("--Test input 1--", st.messages[0]["content"])
    self.assertIn("This task has 2 test inputs", st.messages[0]["content"])
    self.assertEqual(st.session.execute("len(test_inputs)")["stdout"].strip(), "2")

    # Turn 1: run_python with a typo that auto-repair fixes at zero token cost.
    code = SOLVE_SRC + "print(solve(test_inpt))"
    before = list(st.prompt_token_ids)
    self.assertIsNone(self._turn(st, _im("try", _call("run_python", code))))
    self.assertEqual(st.messages[-1]["role"], "tool")
    self.assertIn("[auto-corrected:", st.messages[-1]["content"])
    self.assertIn("[completed]", st.messages[-1]["content"])
    self.assertEqual(st.repairs["retry_ok"], 1)
    self.assertEqual(st.executed[-1][1], SOLVE_SRC + "print(solve(test_input))")
    # Token-exact: prompt grew by sampled ids + tool suffix, never re-tokenized.
    self.assertEqual(st.prompt_token_ids[: len(before)], before)
    self.assertGreater(len(st.prompt_token_ids), len(before))
    self.assertFalse(st.resync)

    # Turn 2: multi-test submit -> accepted, train_check expr path via multi_check_code.
    stop = self._turn(st, _im("done", _call("submit_answer", "[solve(t) for t in test_inputs]")))
    self.assertEqual(stop, "submitted")
    self.assertEqual(st.pred, [TEST_OUTPUT, TEST_OUTPUT_B])
    self.assertEqual(st.answer_source, "submit")
    self.assertTrue(st.labels["train_pass"])
    self.assertEqual(st.labels["train_trace"], "expr")
    self.assertTrue(st.finished)
    self.assertTrue(st.session.closed)

  def test_single_test_rescue_code_then_session_answer_fallback(self) -> None:
    st = self._state([{"input": TEST_INPUT, "output": TEST_OUTPUT}])
    self.assertIsNone(st.multi)
    self.assertEqual((st.max_total_output, st.budget_soft, st.max_turns), (80000, 40000, 40))
    # Plain fenced code (no tool call) is rescued into run_python and resyncs the prompt.
    raw = _im("oops", "```python\n" + SOLVE_SRC + "answer = solve(test_input)\n```")
    self.assertIsNone(self._turn(st, raw))
    self.assertEqual(st.n_rescues, 1)
    self.assertTrue(st.resync)
    self.assertIsNone(st.prompt_token_ids)
    self.assertEqual(st.messages[-2]["tool_calls"][0]["id"], "rescue_1")
    self.assertIn("[completed]", st.messages[-1]["content"])
    # Budget exhausted on a length-cut thinking turn -> session `answer` fallback + train_check.
    st.cum_completion_tokens = st.max_total_output
    stop = self._turn(st, "endless thinking", finish="length")
    self.assertEqual(stop, "output_budget")
    self.assertEqual(st.pred, TEST_OUTPUT)
    self.assertEqual(st.answer_source, "session")
    self.assertEqual(st.submit_code, "answer")
    self.assertTrue(st.labels["train_pass"])
    self.assertIn(st.labels["train_trace"], ("expr", "cell"))

  def test_nudge_then_solve_fallback_on_max_turns(self) -> None:
    st = self._state([{"input": TEST_INPUT, "output": TEST_OUTPUT}])
    st.session.execute(SOLVE_SRC)
    st.executed.append(("run_python", SOLVE_SRC))
    self.assertIsNone(self._turn(st, _im("hmm", "I think the rule is recolouring.")))
    self.assertEqual(st.messages[-1], {"role": "user", "content": hb.RESCUE_NUDGE})
    self.assertEqual(st.n_nudge_rescues, 1)
    st.n_turns = st.max_turns - 1
    stop = self._turn(st, _im("hmm", "Still no code."))
    self.assertEqual(stop, "max_turns")
    self.assertEqual(st.pred, TEST_OUTPUT)
    self.assertEqual(st.answer_source, "session_solve")
    self.assertTrue(st.labels["train_pass"])

  def test_text_grid_answer_is_parsed_and_fn_checked(self) -> None:
    st = self._state([{"input": TEST_INPUT, "output": TEST_OUTPUT}])
    st.session.execute(SOLVE_SRC)
    stop = self._turn(st, _im("sure", f"Final answer: \\boxed{{{json.dumps(TEST_OUTPUT)}}}"))
    self.assertEqual(stop, "final")
    self.assertEqual(st.pred, TEST_OUTPUT)
    self.assertEqual(st.answer_source, "parse")
    self.assertTrue(st.labels["train_pass"])
    self.assertEqual(st.labels["train_trace"], "fn")

  def test_run_havu_b32_shard_dsv41_end_to_end(self) -> None:
    d_in = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    d_out = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    with open(os.path.join(d_in, "task00.json"), "w") as f:
      json.dump({"train": TRAIN, "test": [{"input": TEST_INPUT, "output": TEST_OUTPUT}]}, f)

    class _FakeTok:
      eos_token_id = 1

      def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if s == hb.DS41_EOS_TOKEN:
          return [1]
        return _fake_encode(s)

      def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr((i % 26) + 97) for i in ids)

    class _FakeComp:

      def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = _fake_encode(text)
        self.finish_reason = "stop"

    class _FakeReqOut:

      def __init__(self, rid: str, text: str) -> None:
        self.request_id = rid
        self.outputs = [_FakeComp(text)]
        self.finished = True
        self.prompt_token_ids = [1, 2, 3]

    class _FakeEngine:

      def __init__(self) -> None:
        self.model_config = type("Cfg", (), {"max_model_len": 131072})()
        self.pending: list[str] = []
        self.turn_by_rid: dict[str, int] = {}
        self.counter = 0

      def has_unfinished_requests(self) -> bool:
        return bool(self.pending)

      def step(self) -> list[_FakeReqOut]:
        rid = self.pending.pop(0)
        turn = self.turn_by_rid.get(rid, 0)
        if turn == 1:
          txt = (
              "Define solve.</think>\n\n"
              "<｜DSML｜ calls>\n"
              '<｜DSML｜ invoke name="run_python">\n'
              f'<｜DSML｜ parameter name="code" string="true">{SOLVE_SRC}</｜DSML｜ parameter>\n'
              "</｜DSML｜ invoke>\n"
              "</｜DSML｜ calls>"
              + hb.DS41_EOS_TOKEN
          )
        else:
          txt = (
              "Submit solve(test_input).</think>\n\n"
              "<｜DSML｜ calls>\n"
              '<｜DSML｜ invoke name="submit_answer">\n'
              '<｜DSML｜ parameter name="code" string="true">solve(test_input)</｜DSML｜ parameter>\n'
              "</｜DSML｜ invoke>\n"
              "</｜DSML｜ calls>"
              + hb.DS41_EOS_TOKEN
          )
        return [_FakeReqOut(rid, txt)]

    class _FakeLLM:

      def __init__(self) -> None:
        self.llm_engine = _FakeEngine()
        self._tok = _FakeTok()
        self.calls_per_attempt = 0

      def get_tokenizer(self) -> _FakeTok:
        return self._tok

      def enqueue(self, prompts, sps, use_tqdm: bool = False) -> list[str]:
        del prompts, sps, use_tqdm
        self.llm_engine.counter += 1
        rid = f"req_{self.llm_engine.counter}"
        self.calls_per_attempt = (self.calls_per_attempt % 2) + 1
        self.llm_engine.turn_by_rid[rid] = self.calls_per_attempt
        self.llm_engine.pending.append(rid)
        return [rid]

    import dataclasses as _dc
    cfg = _dc.replace(
        self.cfg,
        exp_id="dsv41_unit_test",
        num_attempts=2,
        attempt_prompt_styles=("lean2sub", "lean2sub"),
        attempt_include_task_summary=(False, False),
    )
    recs = hb.run_havu_b32_shard(
        llm=_FakeLLM(),
        cfg=cfg,
        dataset_dir=d_in,
        out_dir=d_out,
        shard_index=0,
        num_shards=1,
        model_path="/cns/lu-d/home/arazdai/models/dsv41_e128",
    )
    self.assertLen(recs, 1)
    self.assertTrue(recs[0]["pass1"])
    self.assertTrue(recs[0]["any_verified"])


class TextImageContentTest(googletest.TestCase):
  """E058 `--grid-format text+image --image-cell 0` content (port of Havu's `text_image_content`)."""

  # Verbatim `run_deepseek_tools.IMAGE_NOTE` / `IMAGE_NOTES["strict"]` (Havu's repo, E058/E064).
  HAVU_IMAGE_NOTE = (
      "\n\nThe images below show the SAME training examples and test input(s) as the JSON above, in the same "
      "order; they are not additional examples. Each image is one grid, drawn as flat coloured squares "
      "without lines (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 light blue, "
      "9 maroon). Use them to see shapes and layout; the JSON grids specify the exact cells."
  )
  HAVU_STRICT_SUFFIX = (
      " The images give only an overall impression and are easy to misread: before "
      "you build on anything you see in them (a symmetry, a count, which cells "
      "connect, sizes or positions), confirm it in code on the JSON grids."
  )

  @staticmethod
  def _aligned_cell(g: list[list[int]]) -> int:
    cells = len(g) * len(g[0])
    k = 1
    while cells * k * k < 169:
      k += 1
    return 42 * k

  @staticmethod
  def _png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])

  def _encode_with_placeholders(self, text: str) -> list[int]:
    """Fake tokenizer that maps each `<｜deepseek_image｜>` to its single token id."""
    ph = dsv41_image_tokens.IMAGE_PLACEHOLDER
    ids: list[int] = []
    for i, piece in enumerate(text.split(ph)):
      if i:
        ids.append(dsv41_image_tokens.IMAGE_TOKEN_ID)
      ids.extend(_fake_encode(piece))
    return ids

  def test_notes_match_havu_wording(self) -> None:
    self.assertEqual(grid_image.IMAGE_NOTE, self.HAVU_IMAGE_NOTE)
    self.assertEqual(grid_image.IMAGE_NOTES[""], self.HAVU_IMAGE_NOTE)
    self.assertEqual(grid_image.IMAGE_NOTES["strict"], self.HAVU_IMAGE_NOTE + self.HAVU_STRICT_SUFFIX)
    self.assertEqual(sorted(grid_image.IMAGE_NOTES), ["", "strict"])

  def test_parts_order_captions_and_images(self) -> None:
    prompt = hb.build_prompt_single(TRAIN, TEST_INPUT) + hb.LEAN2SUB_SUFFIX
    parts = grid_image.text_image_content(prompt, TRAIN, [TEST_INPUT], cell=0)
    # Part 0: the unchanged JSON prompt + the note, in ONE text part.
    self.assertEqual(parts[0], {"type": "text", "text": prompt + grid_image.IMAGE_NOTE})
    # Then (label, image) pairs: train inputs/outputs in order (numbered from 0), then the test input.
    grids = []
    for pair in TRAIN:
      grids += [pair["input"], pair["output"]]
    grids.append(TEST_INPUT)
    labels = []
    for i in range(len(TRAIN)):
      labels += [f"Example {i} input", f"Example {i} output"]
    labels.append("Test input")
    self.assertLen(parts, 1 + 2 * len(grids))
    images = grid_image.content_images(parts)
    self.assertLen(images, 2 * len(TRAIN) + 1)
    for j, (label, g) in enumerate(zip(labels, grids)):
      text_part, image_part = parts[1 + 2 * j], parts[2 + 2 * j]
      self.assertEqual(text_part, {"type": "text", "text": f"\n{label} ({len(g)}x{len(g[0])}):"})
      self.assertEqual(image_part["type"], "image_url")
      url = image_part["image_url"]["url"]
      self.assertTrue(url.startswith("data:image/png;base64,"))
      png = images[j]
      self.assertEqual(png, dsv41_image_tokens.decode_data_url(url))
      # `--image-cell 0`: 42*k px cells, k = smallest with cells*k^2 >= 169 (no rescale by the model).
      cell = self._aligned_cell(g)
      self.assertEqual(cell, grid_image.deepseek_cell(g))
      self.assertEqual(cell % 42, 0)
      self.assertEqual(self._png_size(png), (len(g[0]) * cell, len(g) * cell))
      self.assertEqual(dsv41_image_tokens.png_size(png), (len(g[0]) * cell, len(g) * cell))
      # Solid palette squares, no grid lines: every cell centre and corner has the cell's colour.
      with Image.open(io.BytesIO(png)) as im:
        im = im.convert("RGB")
        for r, row in enumerate(g):
          for c, v in enumerate(row):
            for dx, dy in ((0, 0), (cell // 2, cell // 2), (cell - 1, cell - 1)):
              self.assertEqual(im.getpixel((c * cell + dx, r * cell + dy)), grid_image.PALETTE[v])
      # Token-aligned: the model's planner keeps the image size, one LLM token per 42x42 tile,
      # and Havu's `deepseek_tokens` charge equals the real span length.
      plan = dsv41_image_tokens.plan_from_bytes(png)
      self.assertEqual((plan.best_width, plan.best_height), (len(g[0]) * cell, len(g) * cell))
      self.assertEqual((plan.n_llm_h, plan.n_llm_w), (len(g) * cell // 42, len(g[0]) * cell // 42))
      self.assertEqual(plan.num_tokens, grid_image.deepseek_tokens(g))

  def test_multi_test_labels_and_strict_note(self) -> None:
    prompt = hb.build_prompt_multi(TRAIN, [TEST_INPUT, TEST_INPUT_B]) + hb.LEAN2SUB_SUFFIX + hb.MULTI_NOTE.format(n=2)
    parts = grid_image.text_image_content(
        prompt, TRAIN, [TEST_INPUT, TEST_INPUT_B], cell=0, note=grid_image.IMAGE_NOTES["strict"]
    )
    self.assertTrue(parts[0]["text"].endswith(self.HAVU_STRICT_SUFFIX))
    self.assertIn(hb.MULTI_NOTE.format(n=2) + self.HAVU_IMAGE_NOTE, parts[0]["text"])
    texts = [p["text"] for p in parts if p["type"] == "text"]
    self.assertEqual(texts[-2], "\nTest input 0 (2x4):")
    self.assertEqual(texts[-1], "\nTest input 1 (3x2):")
    self.assertLen(grid_image.content_images(parts), 2 * len(TRAIN) + 2)
    # Fixed cell size (`--image-cell 32`): no alignment.
    parts32 = grid_image.text_image_content(prompt, TRAIN, [TEST_INPUT], cell=32)
    self.assertEqual(self._png_size(grid_image.content_images(parts32)[0]), (3 * 32, 3 * 32))

  def test_dsml_rendering_matches_encoding_py_semantics(self) -> None:
    prompt = hb.build_prompt_single(TRAIN, TEST_INPUT) + hb.LEAN2SUB_SUFFIX
    parts = grid_image.text_image_content(prompt, TRAIN, [TEST_INPUT], cell=0)
    n_images = len(grid_image.content_images(parts))
    rendered = hb.render_user_dsv41(parts)
    # encoding.py `process_image_messages`: image blocks -> one `<｜deepseek_image｜>` text block each,
    # all text blocks joined with "\n\n"; nothing else is inserted.
    expected_texts = []
    for p in parts:
      expected_texts.append(p["text"] if p["type"] == "text" else dsv41_image_tokens.IMAGE_PLACEHOLDER)
    self.assertEqual(rendered, hb.DS41_USER_SP_TOKEN + "\n\n".join(expected_texts))
    self.assertEqual(rendered.count(dsv41_image_tokens.IMAGE_PLACEHOLDER), n_images)
    self.assertTrue(
        rendered.startswith(hb.DS41_USER_SP_TOKEN + prompt + grid_image.IMAGE_NOTE + "\n\n\nExample 0 input (3x3):\n\n"
                            + dsv41_image_tokens.IMAGE_PLACEHOLDER + "\n\n\nExample 0 output (3x3):\n\n")
    )
    # Plain-string user content is untouched (text harness path).
    self.assertEqual(hb.render_user_dsv41("hi"), hb.DS41_USER_SP_TOKEN + "hi")
    with self.assertRaises(ValueError):
      hb.user_content_to_dsml_text([{"type": "audio"}])

  def test_attempt_state_text_image_and_expansion(self) -> None:
    cfg = prompts_and_selector.PILOT2_HYPOTHESES["havu_b32"]
    img_cfg = dataclasses.replace(cfg, grid_format="text+image", image_cell=0, image_note="")
    tools = hb.tools_spec_b32()
    system_block = hb.render_system_dsv41(tools=tools, tier_text="", reasoning_effort=50)
    spec = hb.EvalTaskSpec(
        task_idx=0,
        task_id="demo",
        train_pairs=TRAIN,
        test_pairs=[{"input": TEST_INPUT, "output": TEST_OUTPUT}],
        global_pair_indices=[0],
        prompt_size_bytes=100,
    )
    st = hb._create_attempt_state(spec, 0, img_cfg, tools, system_block, start_session=False)  # pylint: disable=protected-access
    n_images = 2 * len(TRAIN) + 1
    self.assertLen(st.image_blobs, n_images)
    self.assertIsInstance(st.messages[0]["content"], list)
    self.assertEqual(st.messages[0]["role"], "user")
    self.assertLen(st.messages, 1)  # images only in the first user message
    # Same JSON prompt as the text harness, followed by the note.
    txt_st = hb._create_attempt_state(spec, 0, cfg, tools, system_block, start_session=False)  # pylint: disable=protected-access
    self.assertIsInstance(txt_st.messages[0]["content"], str)
    self.assertEqual(txt_st.image_blobs, [])
    self.assertEqual(st.messages[0]["content"][0]["text"], txt_st.messages[0]["content"] + grid_image.IMAGE_NOTE)

    full = hb.render_full_messages_dsv41(system_block, st.messages, add_generation_prompt=True)
    self.assertEqual(full.count(dsv41_image_tokens.IMAGE_PLACEHOLDER), n_images)
    self.assertTrue(full.endswith(hb.DS41_GEN_PROMPT))
    ids = self._encode_with_placeholders(full)
    self.assertEqual(ids.count(dsv41_image_tokens.IMAGE_TOKEN_ID), n_images)
    new_ids, types, _, plans = dsv41_image_tokens.expand_image_placeholders(ids, st.image_blobs)
    span_total = sum(p.num_tokens for p in plans)
    self.assertLen(plans, n_images)
    self.assertEqual(len(new_ids), len(ids) - n_images + span_total)
    self.assertEqual(len(types), len(new_ids))
    self.assertEqual(int((new_ids == dsv41_image_tokens.IMAGE_TOKEN_ID).sum()), span_total)
    spans = dsv41_image_tokens.image_spans(types)
    self.assertLen(spans, n_images)
    self.assertEqual([e - s for s, e in spans], [p.num_tokens for p in plans])
    self.assertEqual(int((types == dsv41_image_tokens.IMAGE).sum()), sum(p.n_llm_h * p.n_llm_w for p in plans))
    self.assertEqual(int((types == dsv41_image_tokens.TEXT).sum()), len(ids) - n_images)
    # Expected per-grid charges (Havu's `deepseek_tokens`), e.g. 3x3 grid -> 15x15 tiles -> 242 tokens.
    self.assertEqual([p.num_tokens for p in plans][:2], [grid_image.deepseek_tokens(TRAIN[0]["input"])] * 2)
    self.assertEqual(plans[0].num_tokens, 242)
    # A second attempt of the same task renders byte-identical content and images (prefix reuse).
    st2 = hb._create_attempt_state(spec, 1, img_cfg, tools, system_block, start_session=False)  # pylint: disable=protected-access
    self.assertEqual(st2.image_blobs, st.image_blobs)
    self.assertEqual(st2.messages[0]["content"], st.messages[0]["content"])
    with self.assertRaises(ValueError):
      hb._create_attempt_state(  # pylint: disable=protected-access
          spec, 0, dataclasses.replace(cfg, grid_format="image"), tools, system_block, start_session=False
      )

  def test_prompt_req_carries_token_types_and_images(self) -> None:
    """`run_havu_b32_shard` sends `token_types` + `images` with the first-turn prompt (DSV41 model)."""
    d_in = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    d_out = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    with open(os.path.join(d_in, "task00.json"), "w") as f:
      json.dump({"train": TRAIN, "test": [{"input": TEST_INPUT, "output": TEST_OUTPUT}]}, f)
    outer = self

    class _Tok:
      eos_token_id = 1

      def encode(self, s: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if s == hb.DS41_EOS_TOKEN:
          return [1]
        return outer._encode_with_placeholders(s)  # pylint: disable=protected-access

      def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr((i % 26) + 97) for i in ids)

    seen: list[dict] = []

    class _Comp:

      def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = _fake_encode(text)
        self.finish_reason = "stop"

    class _Out:

      def __init__(self, rid: str, text: str) -> None:
        self.request_id = rid
        self.outputs = [_Comp(text)]
        self.finished = True
        self.prompt_token_ids = [1, 2, 3]

    class _Engine:

      def __init__(self) -> None:
        self.model_config = type("Cfg", (), {"max_model_len": 131072})()
        self.pending: list[str] = []
        self.counter = 0

      def has_unfinished_requests(self) -> bool:
        return bool(self.pending)

      def step(self) -> list[_Out]:
        rid = self.pending.pop(0)
        txt = (
            "Submit.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"submit_answer\">\n"
            f'<｜DSML｜ parameter name="code" string="true">{json.dumps(TEST_OUTPUT)}</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>" + hb.DS41_EOS_TOKEN
        )
        return [_Out(rid, txt)]

    class _LLM:

      def __init__(self) -> None:
        self.llm_engine = _Engine()
        self._tok = _Tok()

      def get_tokenizer(self) -> _Tok:
        return self._tok

      def enqueue(self, prompts, sps, use_tqdm: bool = False) -> list[str]:
        del sps, use_tqdm
        seen.extend(prompts)
        self.llm_engine.counter += 1
        rid = f"req_{self.llm_engine.counter}"
        self.llm_engine.pending.append(rid)
        return [rid]

    cfg = dataclasses.replace(
        prompts_and_selector.PILOT2_HYPOTHESES["havu_b32"],
        exp_id="dsv41_img_unit_test",
        num_attempts=1,
        attempt_prompt_styles=("lean2sub",),
        attempt_include_task_summary=(False,),
        grid_format="text+image",
        image_cell=0,
        image_note="strict",
    )
    recs = hb.run_havu_b32_shard(
        llm=_LLM(), cfg=cfg, dataset_dir=d_in, out_dir=d_out, shard_index=0, num_shards=1,
        model_path="/tmp/dsv41_img_e128",
    )
    self.assertLen(recs, 1)
    self.assertLen(seen, 1)
    req = seen[0]
    n_images = 2 * len(TRAIN) + 1
    self.assertLen(req["images"], n_images)
    self.assertLen(req["token_types"], len(req["prompt_token_ids"]))
    self.assertLen(dsv41_image_tokens.image_spans(np.asarray(req["token_types"])), n_images)
    self.assertEqual(req["prompt_token_ids"].count(dsv41_image_tokens.IMAGE_TOKEN_ID), sum(
        dsv41_image_tokens.plan_from_bytes(b).num_tokens for b in req["images"]))
    with open(os.path.join(d_out, "config_shard_00_of_01.json")) as f:
      conf = json.load(f)
    self.assertEqual(conf["prompt_format"], {"grid_format": "text+image", "image_cell": 0, "image_note": "strict"})
    self.assertEqual(conf["hypothesis_config"]["grid_format"], "text+image")
    with open(os.path.join(d_out, "summary_shard_00_of_01.json")) as f:
      self.assertEqual(json.load(f)["prompt_format"]["grid_format"], "text+image")
    with open(os.path.join(d_out, "attempts_shard_00_of_01.jsonl")) as f:
      row = json.loads(f.readline())
    self.assertEqual((row["grid_format"], row["prompt_images"]), ("text+image", n_images))


if __name__ == "__main__":
  googletest.main()
