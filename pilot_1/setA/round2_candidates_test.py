"""Unit tests for round-2 plumbing: views, sandbox candidates, isolation, selector."""

import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import prompts_and_voting as pv
from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import round2_candidates as r2
from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import sandbox
from google3.testing.pybase import googletest


def _task():
  # Rule: output = input with every 1 replaced by 2 (shape preserving).
  def f(g):
    return [[2 if v == 1 else v for v in row] for row in g]

  ins = [
      [[0, 1, 0, 0], [1, 1, 0, 3]],
      [[1, 0], [0, 1], [3, 3]],
      [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
  ]
  test_in = [[1, 0, 3], [0, 1, 0]]
  return {
      "train": [{"input": i, "output": f(i)} for i in ins],
      "test": [{"input": test_in, "output": f(test_in)}],
  }


def _crop_task():
  # Size-changing rule: crop to bounding box of non-zero cells.
  def f(g):
    arr = np.asarray(g)
    rs, cs = np.where(arr != 0)
    return arr[rs.min() : rs.max() + 1, cs.min() : cs.max() + 1].tolist()

  ins = [
      [[0, 0, 0, 0, 0], [0, 2, 2, 0, 0], [0, 2, 3, 0, 0], [0, 0, 0, 0, 0]],
      [[0, 0, 0], [0, 0, 0], [4, 5, 6], [0, 0, 0]],
  ]
  test_in = [[0, 0, 0, 0], [0, 0, 7, 8], [0, 0, 9, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
  return {
      "train": [{"input": i, "output": f(i)} for i in ins],
      "test": [{"input": test_in, "output": f(test_in)}],
  }, f


class ViewTest(googletest.TestCase):

  def test_inversion_rectangular(self):
    rng = np.random.default_rng(0)
    for _ in range(50):
      h, w = int(rng.integers(1, 12)), int(rng.integers(1, 12))
      g = rng.integers(0, 10, size=(h, w)).tolist()
      for view in r2.VIEW_NAMES:
        t = r2.apply_view(g, view)
        if view == "rot90":
          self.assertEqual((len(t), len(t[0])), (w, h))
        self.assertEqual(r2.invert_view(t, view), g)

  def test_size_changing_rule_roundtrip(self):
    task, f = _crop_task()
    solver_task = r2.strip_test_outputs(task)
    test_in = task["test"][0]["input"]
    target = task["test"][0]["output"]
    for view in ("rot90", "fliplr"):
      t_task, t_test = r2.transform_task_view(solver_task, test_in, view)
      # A program that is correct *in the transformed frame* is the conjugate
      # of f. Applying it to the transformed test input and inverting must
      # reproduce the original-frame answer, including the new output shape.
      pred_view = f(t_test)  # crop is equivariant, so f itself works in-frame
      self.assertEqual(r2.invert_view(pred_view, view), target)
      # Transformed demos must be internally consistent with the rule.
      for p in t_task["train"]:
        self.assertEqual(f(p["input"]), p["output"])
      self.assertNotIn("output", t_task["test"][0])

  def test_strip_test_outputs(self):
    task = _task()
    st = r2.strip_test_outputs(task)
    self.assertEqual(list(st["test"][0].keys()), ["input"])


class SandboxCandidateTest(googletest.TestCase):

  def _sb(self, auto_check=False, task=None):
    task = task or _task()
    return sandbox.ArcPythonSandbox(
        task=r2.strip_test_outputs(task),
        test_input=task["test"][0]["input"],
        freeze_first_verified=True,
        auto_check=auto_check,
        structured_feedback=False,
        auto_check_style="concise",
        snapshot_defs=True,
        compute_invariance=False,
    )

  def test_silent_snapshot_and_submit_preservation(self):
    sb = self._sb(auto_check=False)
    sb.ctx = {"turn": 2, "gen_tokens": 1234, "t": 3.0}
    out = sb.execute_turn("run_python", "def solve(grid):\n  return grid\nprint('hi')")
    self.assertEqual(out.strip(), "hi")  # no auto feedback in baseline mode
    self.assertLen(sb.candidates, 1)
    self.assertEqual(sb.candidates[0]["source"], "solve_def")
    self.assertEqual(sb.candidates[0]["pool"], "diagnostic")
    self.assertIsNone(sb.candidates[0]["grid"])
    self.assertEqual(sb.candidates[0]["gen_tokens_at_creation"], 1234)
    self.assertEqual(sb.candidates[0]["turn"], 2)
    # First (wrong) submission, then correct one, then a wrong revision.
    sb.ctx = {"turn": 3, "gen_tokens": 2000, "t": 4.0}
    msg = sb.execute_turn("submit_program", "def solve(grid):\n  return grid")
    self.assertIn("REJECTED", msg)
    sb.ctx = {"turn": 4, "gen_tokens": 3000, "t": 5.0}
    good = "def solve(grid):\n  return [[2 if v == 1 else v for v in r] for r in grid]"
    msg = sb.execute_turn("submit_program", good)
    self.assertIn("ACCEPTED", msg)
    self.assertTrue(sb.train_passed)
    sb.ctx = {"turn": 5, "gen_tokens": 4000, "t": 6.0}
    bad = "def solve(grid):\n  return [[9 for v in r] for r in grid]"
    sb.execute_turn("submit_program", bad)
    subs = [c for c in sb.candidates if c["source"] == "submit_program"]
    self.assertLen(subs, 3)
    primary = [c for c in subs if c["is_primary"]]
    self.assertLen(primary, 1)
    self.assertEqual(primary[0]["code"], good)
    self.assertEqual(primary[0]["gen_tokens_at_creation"], 3000)
    self.assertLen(sb.cell_history, 4)
    self.assertEqual(subs[0]["demo_pass_count"], 0)
    self.assertEqual(subs[1]["demo_pass_count"], 3)
    self.assertEqual(subs[0]["demo_shape_ok_count"], 3)

  def test_concise_auto_check_feedback(self):
    sb = self._sb(auto_check=True)
    out = sb.execute_turn("run_python", "def solve(grid):\n  return grid")
    self.assertIn("[auto-check] solve() vs 3 demonstrations: 0/3 passed exactly.", out)
    self.assertIn("shape OK (2x4); 3/8 cells differ", out)
    self.assertIn("(r0,c1) exp 2 got 1", out)
    # Same source again -> suppressed.
    out2 = sb.execute_turn("run_python", "def solve(grid):\n  return grid")
    self.assertIn("feedback not repeated", out2)
    # Wrong dims + exception + truncation note.
    out3 = sb.execute_turn("run_python", "def solve(grid):\n  return [[0]*40 for _ in range(40)]")
    self.assertIn("exception", out3)  # 40x40 is not a valid grid -> error
    out4 = sb.execute_turn("run_python", "def solve(grid):\n  return [grid[0]]")
    self.assertIn("wrong output dimensions", out4)
    big = {"train": [{"input": [[1] * 10 for _ in range(10)], "output": [[2] * 10 for _ in range(10)]}],
           "test": [{"input": [[1] * 10 for _ in range(10)], "output": [[2] * 10 for _ in range(10)]}]}
    sb2 = self._sb(auto_check=True, task=big)
    out5 = sb2.execute_turn("run_python", "def solve(grid):\n  return grid")
    self.assertIn("[showing first 20 of 100 mismatches]", out5)
    # Verified auto-check candidate enters the selection pool.
    good = "def solve(grid):\n  return [[2 if v == 1 else v for v in r] for r in grid]"
    out6 = sb.execute_turn("run_python", good)
    self.assertIn("3/3 passed exactly", out6)
    ac = [c for c in sb.candidates if c["source"] == "auto_check"]
    self.assertEqual(ac[-1]["pool"], "selection")
    self.assertTrue(ac[-1]["train_pass"])
    self.assertEqual(ac[0]["pool"], "diagnostic")
    self.assertFalse(sb.train_passed)  # auto-check never ends the attempt

  def test_isolated_reevaluation_with_helpers_and_state(self):
    task = _task()
    sb = self._sb(auto_check=False, task=task)
    sb.execute_turn("run_python", "def recolor(v):\n  return 2 if v == 1 else v\nmemo = {}\nmemo['x'] = 1")
    sb.execute_turn("run_python", "def solve(grid):\n  return [[recolor(v) for v in r] for r in grid]")
    # State-dependent program: relies on a mutable global populated by a call.
    sb.execute_turn("run_python", "STATE = []\nSTATE.append(2)")
    sb.execute_turn("submit_program", "def solve(grid):\n  return [[STATE[0] if v == 1 else v for v in r] for r in grid]")
    cands = r2.finalize_attempt_candidates(
        sb.candidates, sb.cell_history, sb.task, sb.test_input, "identity", attempt_idx=0
    )
    by_src = {}
    for c in cands:
      by_src.setdefault(c["source"], []).append(c)
    helper_cand = by_src["solve_def"][0]
    self.assertEqual(helper_cand["isolation_mode"], "with_helpers")
    self.assertTrue(helper_cand["train_pass"])
    self.assertEqual(helper_cand["grid"], task["test"][0]["output"])
    self.assertEqual(helper_cand["pool"], "diagnostic")
    state_cand = by_src["submit_program"][0]
    # Live run passed (STATE populated), isolated run must NOT (list stays empty).
    self.assertTrue(state_cand["live"]["train_pass"])
    self.assertEqual(state_cand["isolation_mode"], "with_helpers")
    self.assertFalse(state_cand["train_pass"])
    self.assertEqual(state_cand["pool"], "selection")

  def test_isolated_timeout_does_not_crash(self):
    task = _task()
    sb = self._sb(auto_check=False, task=task)
    sb.execute_turn("submit_program", "def solve(grid):\n  while True:\n    pass")
    cands = r2.finalize_attempt_candidates(
        sb.candidates, sb.cell_history, sb.task, sb.test_input, "identity", attempt_idx=0, timeout_s=1.0
    )
    self.assertLen(cands, 1)
    self.assertIsNone(cands[0]["grid"])


class SelectorTest(googletest.TestCase):

  def _c(self, grid, att, tp=True, tf=1.0, cf=1.0, prim=False, toks=1000, turn=3, suspect=False, pool="selection", source="submit_program"):
    return {
        "grid": grid, "attempt_idx": att, "train_pass": tp, "train_frac": tf, "cell_frac": cf,
        "is_primary": prim, "gen_tokens_at_creation": toks, "turn": turn, "suspect": suspect,
        "pool": pool, "source": source,
    }

  def test_votes_and_tiers(self):
    a, b, c = [[1]], [[2]], [[3]]
    cands = [
        self._c(a, 0, prim=True), self._c(a, 0),  # revision: still one vote
        self._c(b, 1, prim=True), self._c(b, 2, prim=True),
        self._c(c, 3, tp=False, tf=0.5, prim=True),
    ]
    top1, top2, ranked = r2.select_top_two(cands, [[0]])
    self.assertEqual(top1, b)
    self.assertEqual(top2, a)
    self.assertEqual(ranked[0]["verified_votes"], 2)
    self.assertEqual(ranked[1]["verified_votes"], 1)
    self.assertEqual(ranked[1]["total_votes"], 1)
    self.assertEqual(ranked[2]["best_tier"], 2)

  def test_new_tiebreak_tokens_then_turns_then_attempt(self):
    a, b, c = [[1]], [[2]], [[3]]
    cands = [
        self._c(a, 0, prim=True, toks=9000, turn=5),
        self._c(b, 1, prim=True, toks=3000, turn=9),
        self._c(c, 2, prim=True, toks=3000, turn=4),
    ]
    top1, top2, ranked = r2.select_top_two(cands, [[0]])
    self.assertEqual(top1, c)  # fewest tokens tie -> fewer turns
    self.assertEqual(top2, b)
    self.assertEqual([e["min_attempt_idx"] for e in ranked], [2, 1, 0])
    # Attempt index as final deterministic tie-break.
    cands = [self._c(a, 3, prim=True, toks=1, turn=1), self._c(b, 1, prim=True, toks=1, turn=1)]
    top1, _, _ = r2.select_top_two(cands, [[0]])
    self.assertEqual(top1, b)

  def test_score_pair_pools(self):
    target = [[7]]
    rows = [
        {"candidates": [self._c([[7]], 0, pool="diagnostic", source="solve_def"), self._c([[1]], 0, prim=True)]},
        {"candidates": [self._c([[1]], 1, prim=True)]},
    ]
    s = r2.score_pair(rows, fallback_grid=[[0]], target=target)
    self.assertFalse(s["top1_exact"])
    self.assertFalse(s["top2_exact"])
    self.assertFalse(s["selection_oracle"])
    self.assertTrue(s["diagnostic_oracle"])
    self.assertEqual(s["n_distinct_selection_grids"], 1)
    self.assertTrue(rows[0]["candidates"][0]["exact"])

  def test_fallback(self):
    s = r2.score_pair([{"candidates": []}], fallback_grid=[[5]], target=[[1]])
    self.assertEqual(s["top1"], [[5]])


class PromptTest(googletest.TestCase):

  def test_transcript_has_no_test_output(self):
    task = _task()
    st = r2.strip_test_outputs(task)
    t = pv.build_initial_transcript(task=st, test_input=task["test"][0]["input"], use_tools=True, soft_target=24000, hard_budget=48000)
    self.assertTrue(t.endswith("<|im_start|>assistant\n<think>\n"))
    self.assertIn("Test Input", t)
    self.assertEqual(r2.stable_seed("x", 0, 1), r2.stable_seed("x", 0, 1))
    self.assertNotEqual(r2.stable_seed("x", 0, 1), r2.stable_seed("x", 0, 2))


if __name__ == "__main__":
  googletest.main()
