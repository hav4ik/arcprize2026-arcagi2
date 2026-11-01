"""End-to-end unit tests for Top-10K Pass@1 DeepSeek-V4.1 rollout generation (`v8`), 3-level verification, and Qwen SFT compatibility."""

from __future__ import annotations

import json
import os
import tempfile

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_arc2_eval as ds_eval
from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_top10k_rollouts as top10k
from google3.experimental.users.arazdai.kaggle.sft import data_pipeline
from google3.testing.pybase import googletest


class FakeQwenTokenizer:
  """Deterministic byte-level fake tokenizer with exact special tokens for unit testing."""

  IM_START_ID = 248044
  IM_END_ID = 248045
  ASST_ID = 77091
  USER_ID = 872
  SYS_ID = 8948
  NEWLINE_ID = 198

  SPECIAL_MAP = {
      "<|im_start|>": IM_START_ID,
      "<|im_end|>": IM_END_ID,
      "assistant": ASST_ID,
      "user": USER_ID,
      "system": SYS_ID,
  }

  def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
    del add_special_tokens
    out: list[int] = []
    i = 0
    n = len(text)
    while i < n:
      matched = False
      for tok_str in ("<|im_start|>", "<|im_end|>"):
        if text.startswith(tok_str, i):
          out.append(self.SPECIAL_MAP[tok_str])
          i += len(tok_str)
          for role_str in ("assistant", "user", "system"):
            if text.startswith(role_str + "\n", i):
              out.append(self.SPECIAL_MAP[role_str])
              out.append(self.NEWLINE_ID)
              i += len(role_str) + 1
              break
          matched = True
          break
      if matched:
        continue
      ch = text[i]
      if ch == "\n":
        out.append(self.NEWLINE_ID)
      else:
        out.append(1000 + ord(ch))
      i += 1
    return out


class Top10kRolloutsTest(googletest.TestCase):

  def test_live_session_3level_verification_and_sft_pipeline_compatibility(self):
    # Candidate puzzle: invert 0 <-> 1 on a 2x2 grid
    train_pairs = [
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
    ]
    test_input = [[1, 1], [0, 1]]
    target_output = [[0, 0], [1, 0]]
    extra_pairs = [
        {"input": [[1, 0], [1, 0]], "output": [[0, 1], [0, 1]]},
        {"input": [[0, 1], [0, 1]], "output": [[1, 0], [1, 0]]},
    ]
    cand_meta = {
        "sample_id": "NVARC_000001",
        "dataset": "NVARC",
        "original_task_id": "syn_invert_01",
        "deepseek_task_id": "syn_invert_01",
        "source_group": "nvarc_synthetic",
        "family_id": "fam_invert",
        "seed_task_id": "seed_invert",
        "learning_signal_score": 88.5,
        "learning_tier": "Tier1_Gold_MultiStep",
        "rank": 1,
        "train": train_pairs,
        "test": [{"input": test_input, "output": target_output}],
        "extra_verification_pairs": extra_pairs,
    }

    init_msgs, eff_val = ds_eval.build_havu_initial_messages(
        task=cand_meta,
        test_input=test_input,
        attempt_idx=0,
        compact_grids=False,
    )
    self.assertEqual(eff_val, 0)

    session = ds_eval.HavuSubprocessSession(
        train=train_pairs,
        test_input=test_input,
        timeout_s=10.0,
        style="ipython",
    )
    st = top10k.RolloutAttemptState(
        task_id="syn_invert_01",
        pair_idx=0,
        attempt_idx=0,
        soft_target=40000,
        hard_budget=80000,
    )
    rt = top10k.Top10kRolloutRuntime(
        state=st,
        messages=init_msgs,
        reasoning_effort=eff_val,
        session=session,
        base_seed=12345,
        candidate_meta=cand_meta,
        extra_verification_pairs=extra_pairs,
    )

    try:
      # Simulate Turn 1: define helper + solve(g) in IPython session via run_python
      code_t1 = (
          "def helper_flip(v):\n"
          "    return 1 - v\n"
          "def solve(g):\n"
          "    return [[helper_flip(x) for x in row] for row in g]\n"
          "print('train_ok:', all(solve(p['input']) == p['output'] for p in train))"
      )
      status, tool_out1 = rt.session.run_python(code_t1)
      self.assertEqual(status, "completed")
      self.assertIn("train_ok: True", tool_out1)

      st.turns += 1
      st.tool_calls += 1
      st.output_tokens += 50
      st.prompt_tokens = 900
      st.max_prompt_tokens = 900
      rt.uncompacted_messages.append({
          "role": "assistant",
          "content": "",
          "reasoning_content": "Let's define `solve(g)` that flips 0 and 1 and test on `train`.",
          "tool_calls": [{
              "id": "call_1_0",
              "type": "function",
              "function": {
                  "name": "run_python",
                  "arguments": json.dumps({"code": code_t1}),
              },
          }],
      })
      rt.uncompacted_messages.append({
          "role": "tool",
          "tool_call_id": "call_1_0",
          "content": tool_out1,
      })
      rt.events.append({
          "turn": 1,
          "output_tokens": 50,
          "thinking_tokens": 20,
          "tool_use_tokens": 30,
          "answer_tokens": 0,
          "finish_reason": "stop",
          "num_tool_calls": 1,
          "is_canonical": True,
      })

      # Simulate Turn 2: call submit_answer(code="solve(test_input)")
      tool_out2, grid = rt.session.submit_answer("solve(test_input)")
      self.assertEqual(grid, target_output)
      rt.accepted_grid = grid
      rt.submit_code = "solve(test_input)"
      rt.answer_source = "submit"
      rt.stop_reason = "submitted"
      st.turns += 1
      st.tool_calls += 1
      st.output_tokens += 30
      rt.uncompacted_messages.append({
          "role": "assistant",
          "content": "",
          "reasoning_content": "All train pairs passed. Submitting `solve(test_input)`.",
          "tool_calls": [{
              "id": "call_2_0",
              "type": "function",
              "function": {
                  "name": "submit_answer",
                  "arguments": json.dumps({"code": "solve(test_input)"}),
              },
          }],
      })
      rt.uncompacted_messages.append({
          "role": "tool",
          "tool_call_id": "call_2_0",
          "content": tool_out2,
      })
      rt.events.append({
          "turn": 2,
          "output_tokens": 30,
          "thinking_tokens": 15,
          "tool_use_tokens": 15,
          "answer_tokens": 0,
          "finish_reason": "stop",
          "num_tool_calls": 1,
          "is_canonical": True,
      })

      # Run 3-level verification inside live session
      rt.train_labels = top10k.run_3level_session_check(
          session=rt.session,
          pred=rt.accepted_grid,
          code=rt.submit_code,
          extra_pairs=extra_pairs,
      )
      self.assertIs(rt.train_labels["train_pass"], True)
      self.assertEqual(rt.train_labels["train_frac"], 1.0)
      self.assertIs(rt.train_labels["level3_pass"], True)
      self.assertEqual(rt.train_labels["level3_frac"], 1.0)
      self.assertEqual(rt.train_labels["train_trace"], "expr")

      # Now test that a rule that overfits train + test_input but fails extra_pairs gets level3_pass=False
      bad_extra_pairs = [
          {"input": [[1, 0], [1, 0]], "output": [[9, 9], [9, 9]]},
      ]
      overfit_labels = top10k.run_3level_session_check(
          session=rt.session,
          pred=rt.accepted_grid,
          code=rt.submit_code,
          extra_pairs=bad_extra_pairs,
      )
      self.assertIs(overfit_labels["train_pass"], True)
      self.assertIs(overfit_labels["level3_pass"], False)
      self.assertEqual(overfit_labels["level3_frac"], 0.0)
    finally:
      rt.session.close()

    # Build E031 row and verify all 24 core E031 columns + Qwen SFT pipeline compatibility
    row, is_verified_clean, bad_reasons = top10k.build_e031_rollout_row(
        rt=rt,
        target_output=target_output,
        active_model_name="DeepSeek-V4.1-Flash-1017 (Full-Precision BF16->FP32)",
        active_backend="ds41_full_precision",
        eval_task_ids={"89565ca0"},
    )
    self.assertTrue(is_verified_clean)
    self.assertEqual(bad_reasons, [])
    for col in top10k.E031_CORE_COLUMNS:
      self.assertIn(col, row)
    self.assertIs(row["correct"], True)
    self.assertIs(row["train_pass"], True)
    self.assertIs(row["all_3_levels_verified"], True)

    with tempfile.TemporaryDirectory(prefix="top10k_e2e_") as tmp_dir:
      pq_path = os.path.join(tmp_dir, "rollouts_shard_000of064.parquet")
      self.assertTrue(top10k.write_parquet_snapshot([row], pq_path))
      stats, recs = data_pipeline.audit_rollout_file(
          pq_path, {"eval_task_ids": {"89565ca0"}}
      )
      self.assertEqual(stats["verified_clean_count"], 1)
      self.assertEqual(len(recs), 1)
      segments = data_pipeline.build_segments(recs[0]["messages"], max_turns=40)
      tok = FakeQwenTokenizer()
      input_ids, loss_mask = data_pipeline.tokenize_and_verify_segments(
          segments=segments,
          tokenizer=tok,
          im_start_id=tok.IM_START_ID,
          im_end_id=tok.IM_END_ID,
          asst_role_id=tok.ASST_ID,
          newline_id=tok.NEWLINE_ID,
      )
      self.assertGreater(int(loss_mask.sum()), 0)


from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_1.setB.tpu_vllm import deepseek_v41_e065_correct_and_rewrite as e065_cr


class E065CorrectAndRewriteTest(googletest.TestCase):

  def test_e065_iterative_correction_and_prefix_free_clean_rewrite(self):
    train_pairs = [
        {"input": [[0, 2], [2, 0]], "output": [[2, 0], [0, 2]]},
        {"input": [[0, 0], [2, 2]], "output": [[2, 2], [0, 0]]},
    ]
    test_pairs = [
        {"input": [[2, 2], [0, 2]], "output": [[0, 0], [2, 0]]},
        {"input": [[0, 2], [0, 0]], "output": [[2, 0], [2, 2]]},
    ]
    failure_rec = {
        "task_id": "syn_swap2_01",
        "parent": "syn_swap2",
        "source": "nvarc_b1",
        "train_pass": True,
        "train_frac": 1.0,
        "failure_category": "demo_overfit_train_pass_test_fail",
        "turns": 4,
        "stop_reason": "submitted",
        "last_reasoning_excerpt": "I assumed the grid was always rotated 180 degrees.",
        "last_submit_code": "def solve(g): return [r[::-1] for r in g[::-1]]",
        "pair_checks": [
            {
                "pair": 0,
                "diff_summary": {
                    "status": "cell_mismatch",
                    "gt_shape": [2, 2],
                    "pred_shape": [2, 2],
                    "cell_accuracy": 0.5,
                    "mismatched_cells": 2,
                },
            }
        ],
        "train": train_pairs,
        "test": test_pairs,
    }
    prior_attempts = [
        {
            "attempt_round": 1,
            "train_pass": False,
            "train_frac": 0.5,
            "correct_all": False,
            "stop_reason": "submitted",
            "test_diag": "pair0_cell_mismatch_2_cells",
            "code": "def solve(g): return g[::-1]",
        }
    ]
    corr_msgs, _ = e065_cr.build_correction_initial_messages(
        task={"task_id": "syn_swap2_01", "train": train_pairs, "test": test_pairs},
        test_input=test_pairs[0]["input"],
        failure_rec=failure_rec,
        prior_attempts=prior_attempts,
        attempt_round=2,
        difficulty_info={"reasoning_bottleneck": "Confused 180-deg rotation with 0<->2 color inversion."},
    )
    self.assertEqual(len(corr_msgs), 2)
    self.assertIn("Prior Correction Retry #1", corr_msgs[1]["content"])
    self.assertIn("Confused 180-deg rotation", corr_msgs[1]["content"])

    # Verify multi-pair evaluation in live session
    session = ds_eval.HavuSubprocessSession(
        train=train_pairs,
        test_input=test_pairs[0]["input"],
        timeout_s=10.0,
        style="ipython",
    )
    try:
      good_code = "def solve(g):\n    return [[2 - x for x in r] for r in g]\n"
      session.run_python(good_code)
      _, pred0 = session.submit_answer("solve(test_input)")
      multi_res = e065_cr.evaluate_all_test_pairs_in_session(
          session=session,
          pair0_pred=pred0,
          submit_code="solve(test_input)",
          test_pairs=test_pairs,
      )
      self.assertTrue(multi_res["correct_all"])
      self.assertEqual(multi_res["preds_by_pair"], [test_pairs[0]["output"], test_pairs[1]["output"]])
    finally:
      session.close()

    # Now test clean rewrite synthesis + E065 parquet row assembly (preserving inline base64 PNG images)
    asst_msgs, evs, _, tr_labels = e065_cr.synthesize_verified_clean_trajectory_in_sandbox(
        train_pairs=train_pairs,
        test_input=test_pairs[0]["input"],
        verified_solve_code="def solve(g):\n    return [[2 - x for x in r] for r in g]",
        submit_code="solve(test_input)",
        reasoning_summary=(
            "Previous attempt failed because of rotation.\n"
            "Inspecting the training grids shows every 0 becomes 2 and every 2 becomes 0."
        ),
    )
    self.assertFalse(e065_cr.has_forbidden_meta_language(asst_msgs))
    self.assertIs(tr_labels["train_pass"], True)

    orig_e065_messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Solve puzzle syn_swap2_01."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    },
                },
            ],
        },
        {"role": "assistant", "content": "bad old attempt", "reasoning_content": "old wrong thought"},
    ]
    orig_group_rows = [
        {
            "task_id": "syn_swap2_01",
            "pair": p_idx,
            "attempt": 0,
            "source": "nvarc_b1",
            "parent": "syn_swap2",
            "correct": False,
            "pred": None,
            "answer_source": "none",
            "stop_reason": "max_turns",
            "turns": 25 if p_idx == 0 else 0,
            "tool_calls": 25 if p_idx == 0 else 0,
            "prompt_tokens": 1200 if p_idx == 0 else 0,
            "completion_tokens": 30000 if p_idx == 0 else 0,
            "reasoning_tokens": 25000 if p_idx == 0 else 0,
            "tool_use_tokens": 5000 if p_idx == 0 else 0,
            "answer_tokens": 0,
            "elapsed_s": 120.0 if p_idx == 0 else 0.0,
            "messages": json.dumps(orig_e065_messages),
            "events": "[]",
            "error": None,
            "train_pass": False,
            "train_frac": 0.0,
            "train_trace": "untraceable",
            "train_paths": "[]",
            "verified_code": "",
            "train_check_s": 0.01,
        }
        for p_idx in range(2)
    ]

    img_rows, txt_rows = e065_cr.assemble_e065_rewritten_parquet_rows(
        orig_group_rows=orig_group_rows,
        clean_asst_and_tool_messages=asst_msgs,
        clean_events=evs,
        preds_by_pair=multi_res["preds_by_pair"],
        train_labels=tr_labels,
        solved_on_attempt=2,
        rewrite_source="sandbox_verified_clean_synthesis",
        elapsed_s=4.5,
    )
    self.assertEqual(len(img_rows), 2)
    self.assertEqual(len(txt_rows), 2)
    parsed_img_msgs = json.loads(img_rows[0]["messages"])
    # First user message must still have the inline base64 PNG image part and NO old assistant turns
    self.assertIsInstance(parsed_img_msgs[1]["content"], list)
    self.assertEqual(parsed_img_msgs[1]["content"][1]["type"], "image_url")
    self.assertEqual(len(parsed_img_msgs), 2 + len(asst_msgs))

    with tempfile.TemporaryDirectory(prefix="e065_rw_") as tmp_dir:
      pq_path = os.path.join(tmp_dir, "e065_images_corrected_rewritten_shard_000of001.parquet")
      self.assertTrue(e065_cr._write_parquet_rows_to_cns(img_rows, pq_path))
      stats, recs = data_pipeline.audit_e065_rollout_file(
          pq_path, {"eval_task_ids": {"89565ca0"}}
      )
      self.assertEqual(stats["verified_clean_count"], 1)
      self.assertEqual(len(recs), 1)
      self.assertEqual(recs[0]["n_images"], 1)


if __name__ == "__main__":
  googletest.main()

