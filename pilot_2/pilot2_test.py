"""Unit tests for pilot_2 frozen baseline invariants, sandbox verifier, DSL summary, and end-to-end eval loop."""

from __future__ import annotations

import json
import os
import tempfile

from google3.testing.pybase import googletest
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import arc_dsl_and_summary
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import pilot2_eval
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import prompts_and_selector
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import sandbox_verifier


class Pilot2Test(googletest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.train_pairs = [
        {
            "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]],
        },
        {
            "input": [[1, 0], [0, 1]],
            "output": [[2, 0], [0, 2]],
        },
    ]
    self.test_input = [[0, 0, 1, 0], [0, 1, 1, 1]]

  def test_all_hypotheses_preserve_frozen_baseline_invariants(self) -> None:
    self.assertLen(prompts_and_selector.PILOT2_HYPOTHESES, 8)
    for exp_id, cfg in prompts_and_selector.PILOT2_HYPOTHESES.items():
      cfg.validate()
      self.assertEqual(cfg.temperature, 1.0, f"{exp_id} violated temperature=1.0")
      self.assertEqual(cfg.top_p, 0.95, f"{exp_id} violated top_p=0.95")
      self.assertEqual(cfg.top_k, 20, f"{exp_id} violated top_k=20")
      self.assertTrue(cfg.think_call_rescue)
      self.assertTrue(cfg.lock_first_verified_pred)
    self.assertTrue(
        prompts_and_selector.PILOT2_HYPOTHESES["h7_native_xml_and_grace_nudge"].use_native_xml_tools
    )
    self.assertTrue(
        prompts_and_selector.PILOT2_HYPOTHESES["h8_evolutionary_two_wave_repair"].use_evolutionary_two_wave
    )

  def test_xml_tool_parsing_and_wave2_evolutionary_seed(self) -> None:
    xml_out = (
        "Let me test in Python.\n</think>\n\n"
        "<tool_call>\n<function=run_python>\n<parameter=code>\n"
        "def solve(grid):\n    return np.where(grid == 1, 2, grid)\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    name, args, err = prompts_and_selector.parse_tool_call_from_output(xml_out)
    self.assertIsNone(err)
    self.assertEqual(name, "run_python")
    self.assertIn("def solve(grid):", args["code"])

    unclosed_xml = (
        "Let me test.\n</think>\n\n<tool_call>\n<function=submit_program>\n<parameter=code>\n"
        "def solve(grid):\n    return grid.copy()"
    )
    name2, args2, err2 = prompts_and_selector.parse_tool_call_from_output(unclosed_xml)
    self.assertIsNone(err2)
    self.assertEqual(name2, "submit_program")
    self.assertIn("def solve(grid):", args2["code"])

    sb_partial = sandbox_verifier.Pilot2AttemptSandbox(
        train_pairs=self.train_pairs,
        test_input=self.test_input,
        use_arc_helpers=True,
        auto_check_on_solve=True,
        run_equivariance_audit=True,
    )
    sb_partial.execute_run_python(
        code="def solve(grid):\n    out = grid.copy()\n    out[0, 0] = 2\n    return out",
        turn_idx=0,
        cum_gen_tokens=300,
    )
    seed_text = prompts_and_selector.build_wave2_evolutionary_seed([sb_partial], wave2_attempt_idx=3)
    self.assertIn("Wave 1 Evolutionary Program Repair Seed", seed_text)
    self.assertIn("def solve(grid):", seed_text)

    transcript = (
        "<|im_start|>system\nsys<|im_end|>\n"
        "<|im_start|>user\nusr<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n" + ("very long thought " * 200) + "\n</think>\n"
        "<tool_call>\n<function=run_python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n1\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\nturn2\n</think><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n2\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\nturn3\n</think><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n3\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    compacted = prompts_and_selector.compact_chatml_transcript(transcript, keep_recent_turns=2)
    self.assertLess(len(compacted), len(transcript))
    self.assertIn("turn2", compacted)
    self.assertIn("turn3", compacted)

  def test_cumulative_helper_replay_and_first_tp_lock(self) -> None:
    sb = sandbox_verifier.Pilot2AttemptSandbox(
        train_pairs=self.train_pairs,
        test_input=self.test_input,
        use_arc_helpers=True,
        auto_check_on_solve=True,
        run_equivariance_audit=True,
    )
    resp0, tp0 = sb.execute_run_python(
        code="def recolour_ones(g):\n    return np.where(g == 1, 2, g)\nprint('helper ready')",
        turn_idx=0,
        cum_gen_tokens=400,
    )
    self.assertFalse(tp0)
    self.assertIn("helper ready", resp0)

    resp1, tp1 = sb.execute_run_python(
        code="def solve(grid):\n    return recolour_ones(grid)",
        turn_idx=1,
        cum_gen_tokens=900,
    )
    self.assertTrue(tp1)
    self.assertNotIn("helper ready", resp1)
    self.assertIn("PASS 2/2 train pairs", resp1)
    self.assertEqual(sb.first_tp_pred, [[0, 0, 2, 0], [0, 2, 2, 2]])
    self.assertEqual(sb.first_tp_gen_tokens, 900)
    self.assertGreater(sb.first_tp_equivariance_bonus, 0.0)

    _, tp2 = sb.execute_run_python(
        code="def solve(grid):\n    return np.zeros_like(grid)",
        turn_idx=2,
        cum_gen_tokens=1500,
    )
    self.assertFalse(tp2)
    self.assertEqual(sb.first_tp_pred, [[0, 0, 2, 0], [0, 2, 2, 2]])
    self.assertEqual(sb.first_tp_gen_tokens, 900)

  def test_concurrent_24_threads_fork_free(self) -> None:
    from concurrent import futures  # pylint: disable=g-import-not-at-top

    def _run_one(idx: int) -> tuple[int, str, bool]:
      sb = sandbox_verifier.Pilot2AttemptSandbox(
          train_pairs=self.train_pairs,
          test_input=self.test_input,
          use_arc_helpers=True,
          auto_check_on_solve=True,
          run_equivariance_audit=True,
      )
      resp, ok = sb.execute_submit_program(
          code=f"print('worker_{idx}')\ndef solve(grid):\n    return np.where(grid == 1, 2, grid)",
          turn_idx=0,
          cum_gen_tokens=200 + idx,
      )
      return idx, resp, ok

    with futures.ThreadPoolExecutor(max_workers=24) as pool:
      results = list(pool.map(_run_one, range(24)))
    self.assertLen(results, 24)
    for idx, resp, ok in results:
      self.assertTrue(ok, f"Worker {idx} failed: {resp}")
      self.assertIn("Verification PASSED", resp)

  def test_attempt_stream_execution_and_resume_loader(self) -> None:
    cfg = prompts_and_selector.PILOT2_HYPOTHESES["h3_more_max_turns_40"]
    pair_spec = pilot2_eval.EvalPairSpec(
        global_pair_idx=7,
        task_id="demo_task",
        test_idx=0,
        train_pairs=self.train_pairs,
        test_input=self.test_input,
        ground_truth_output=[[0, 0, 2, 0], [0, 2, 2, 2]],
    )
    sb = sandbox_verifier.Pilot2AttemptSandbox(
        train_pairs=self.train_pairs,
        test_input=self.test_input,
        use_arc_helpers=cfg.use_arc_helpers,
        auto_check_on_solve=cfg.auto_check_on_solve,
        run_equivariance_audit=cfg.use_equivariance_audit,
    )
    st = pilot2_eval.AttemptStreamState(
        pair_spec=pair_spec,
        attempt_idx=0,
        cfg=cfg,
        sandbox=sb,
        system_prompt="sys",
        user_prompt="usr",
        transcript="<|im_start|>assistant\n",
        seed=10007,
    )
    assistant_msg = (
        '<think> Replace 1 with 2. </think>\n'
        '<tool_call>{"name": "submit_program", "arguments": '
        '{"code": "def solve(grid):\\n    return np.where(grid == 1, 2, grid)"}}</tool_call>'
    )
    resp, done, reason = pilot2_eval.execute_tool_for_attempt(st, assistant_msg, 320)
    self.assertTrue(done)
    self.assertEqual(reason, "verified_submit_program")
    self.assertIn("Verification PASSED", resp)
    att_sum = pilot2_eval.build_attempt_candidate_summary(st)
    self.assertTrue(att_sum.train_pass)
    self.assertEqual(att_sum.verified_grid, [[0, 0, 2, 0], [0, 2, 2, 2]])

    with tempfile.TemporaryDirectory() as tmpdir:
      pairs_file = os.path.join(tmpdir, "pairs_shard_00_of_06.jsonl")
      pilot2_eval._safe_cns_append(
          pairs_file,
          json.dumps({
              "global_pair_idx": 7,
              "task_id": "demo_task",
              "test_idx": 0,
              "pass1": True,
              "pass2": True,
              "sel_oracle": True,
              "diag_oracle": True,
              "any_verified": True,
              "attempt_1": [[0, 0, 2, 0], [0, 2, 2, 2]],
              "attempt_2": [[0, 0, 2, 0], [0, 2, 2, 2]],
          })
          + "\n",
      )
      done_set, p1, p2, so, do, ver, sub = pilot2_eval.load_existing_completed_pairs(pairs_file)
      self.assertEqual(done_set, {7})
      self.assertEqual(p1, 1)
      self.assertEqual(p2, 1)
      self.assertEqual(so, 1)
      self.assertEqual(do, 1)
      self.assertEqual(ver, 1)
      self.assertIn("demo_task", sub)


if __name__ == "__main__":
  googletest.main()
