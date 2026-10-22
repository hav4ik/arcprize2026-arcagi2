"""Standalone Python sandbox, program verifier, and equivariance auditor for pilot_2.

Written from scratch for pilot_2 with zero dependencies on pilot_1.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import io
import itertools
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np
import scipy
from scipy import ndimage

from google3.experimental.users.arazdai.kaggle.arc_agi_2.pilot_2 import arc_dsl_and_summary


Grid = list[list[int]]


def is_valid_grid(obj: Any) -> bool:
  """Return True if `obj` can be converted to a rectangular 2D int grid in 0..9."""
  try:
    arr = np.asarray(obj)
  except Exception:
    return False
  if arr.ndim != 2:
    return False
  h, w = arr.shape
  if not (1 <= h <= 30 and 1 <= w <= 30):
    return False
  if arr.dtype.kind not in ("i", "u", "b"):
    try:
      arr_int = arr.astype(int)
      if not np.array_equal(arr, arr_int):
        return False
      arr = arr_int
    except Exception:
      return False
  return bool(np.all((arr >= 0) & (arr <= 9)))


def to_grid(obj: Any) -> Grid:
  """Convert a valid 2D array-like into a list[list[int]]."""
  arr = np.asarray(obj, dtype=int)
  return [[int(v) for v in row] for row in arr]


def grid_to_canonical_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
  """Convert a 2D list[list[int]] into a hashable tuple of tuples."""
  return tuple(tuple(int(v) for v in row) for row in grid)


def format_grid_compact(grid: Grid | np.ndarray) -> str:
  """Format a 2D grid as compact space-separated rows."""
  arr = np.asarray(grid, dtype=int)
  return "\n".join(" ".join(str(int(v)) for v in row) for row in arr)


def format_pair_diff(got: np.ndarray, exp: np.ndarray, pair_idx: int) -> str:
  """Format rich compact feedback comparing `got` vs `exp` on training pair `pair_idx`."""
  if got.shape != exp.shape:
    return (
        f"Pair {pair_idx}: FAIL (shape mismatch: got {got.shape[0]}x{got.shape[1]}, "
        f"expected {exp.shape[0]}x{exp.shape[1]})"
    )
  diff_mask = got != exp
  n_diff = int(diff_mask.sum())
  total = int(exp.size)
  if n_diff == 0:
    return f"Pair {pair_idx}: PASS ({exp.shape[0]}x{exp.shape[1]})"
  acc_pct = 100.0 * (total - n_diff) / max(1, total)
  rs, cs = np.where(diff_mask)
  if n_diff <= 14:
    coords_str = ", ".join(
        f"({int(r)},{int(c)}) {int(got[r, c])}>{int(exp[r, c])}"
        for r, c in zip(rs, cs)
    )
    return (
        f"Pair {pair_idx}: FAIL ({n_diff}/{total} cells wrong, {acc_pct:.1f}% match) "
        f"| diffs (r,c) got>exp: {coords_str}"
    )
  bad_rows = sorted(set(int(r) for r in rs))[:6]
  row_snippets = []
  for r in bad_rows:
    g_row = "".join(str(int(v)) for v in got[r])
    e_row = "".join(str(int(v)) for v in exp[r])
    row_snippets.append(f"r{r}: got={g_row} vs exp={e_row}")
  more = (
      f" (+{len(set(int(r) for r in rs)) - 6} more rows)"
      if len(set(int(r) for r in rs)) > 6
      else ""
  )
  return (
      f"Pair {pair_idx}: FAIL ({n_diff}/{total} cells wrong, {acc_pct:.1f}% match) "
      f"| " + " ; ".join(row_snippets) + more
  )


def sanity_check_prediction(
    pred: Grid,
    train_pairs: list[dict[str, Any]],
    test_input: Grid,
) -> tuple[bool, str]:
  """Check that a predicted test grid is structurally non-degenerate."""
  if not is_valid_grid(pred):
    return False, "invalid_grid"
  pred_arr = np.asarray(pred, dtype=int)
  test_in_arr = np.asarray(test_input, dtype=int)
  train_in_arrs = [np.asarray(p["input"], dtype=int) for p in train_pairs]
  train_out_arrs = [np.asarray(p["output"], dtype=int) for p in train_pairs]

  if len(np.unique(pred_arr)) == 1:
    if not any(len(np.unique(o)) == 1 for o in train_out_arrs):
      return False, "degenerate_all_single_colour"

  if pred_arr.shape == test_in_arr.shape and np.array_equal(pred_arr, test_in_arr):
    if not any(
        i.shape == o.shape and np.array_equal(i, o)
        for i, o in zip(train_in_arrs, train_out_arrs)
    ):
      return False, "degenerate_unchanged_test_input"

  return True, "ok"


def ast_constant_lint_penalty(
    code: str,
    train_pairs: list[dict[str, Any]],
    test_input: Grid,
) -> float:
  """Return a small penalty in [0.0, 0.5] if `solve` hardcodes training dimensions when test differs."""
  try:
    tree = ast.parse(code)
  except Exception:
    return 0.0
  test_h, test_w = np.asarray(test_input).shape
  train_dims: set[int] = set()
  for p in train_pairs:
    ih, iw = np.asarray(p["input"]).shape
    oh, ow = np.asarray(p["output"]).shape
    for d in (ih, iw, oh, ow):
      if d >= 5 and d not in (test_h, test_w):
        train_dims.add(int(d))
  if not train_dims:
    return 0.0

  solve_node = None
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "solve":
      solve_node = node
  target = solve_node if solve_node is not None else tree
  found_hardcoded = 0
  for sub in ast.walk(target):
    if isinstance(sub, ast.Constant) and isinstance(sub.value, int):
      if int(sub.value) in train_dims:
        found_hardcoded += 1
  return min(0.5, 0.25 * found_hardcoded)


@dataclasses.dataclass
class CodeTurnTrace:
  """Complete trace of a single tool invocation for CNS archival."""

  turn_idx: int
  tool_name: str
  code_or_payload: str
  stdout: str
  error: str | None
  defined_solve: bool
  train_pass: bool
  train_passed_count: int
  train_total_count: int
  train_frac: float
  mean_cell_acc: float
  pair_reports: list[str]
  test_pred_shape: tuple[int, int] | None
  sanity_ok: bool
  sanity_reason: str
  equivariance_bonus: float
  cum_gen_tokens: int


@dataclasses.dataclass
class VerifiedVariant:
  """Record of a verified (`train_pass=True`) program produced during an attempt."""

  grid: Grid
  code: str
  turn_idx: int
  cum_gen_tokens: int
  equivariance_bonus: float


class Pilot2AttemptSandbox:
  """Thread-safe, fork-free stateful per-attempt Python sandbox with trace logging."""

  def __init__(
      self,
      train_pairs: list[dict[str, Any]],
      test_input: Grid,
      use_arc_helpers: bool = False,
      auto_check_on_solve: bool = False,
      run_equivariance_audit: bool = False,
      max_output_chars: int = 3500,
  ) -> None:
    self.train_pairs = train_pairs
    self.test_input = test_input
    self.use_arc_helpers = use_arc_helpers
    self.auto_check_on_solve = auto_check_on_solve
    self.run_equivariance_audit = run_equivariance_audit
    self.max_output_chars = max_output_chars

    self._train_np = [
        {
            "input": np.asarray(p["input"], dtype=int),
            "output": np.asarray(p["output"], dtype=int),
        }
        for p in train_pairs
    ]
    self._test_np = np.asarray(test_input, dtype=int)
    self._active_stdout_buf: io.StringIO = io.StringIO()

    self.helper_cells: list[str] = []
    self.code_traces: list[CodeTurnTrace] = []
    self.verified_variants: list[VerifiedVariant] = []

    self.first_tp_pred: Grid | None = None
    self.first_tp_code: str | None = None
    self.first_tp_turn: int | None = None
    self.first_tp_gen_tokens: int | None = None
    self.first_tp_equivariance_bonus: float = 0.0

    self.latest_tp_pred: Grid | None = None
    self.latest_tp_code: str | None = None
    self.best_partial_pred: Grid | None = None
    self.best_partial_code: str | None = None
    self.best_partial_pair_reports: list[str] = []
    self.best_partial_train_frac: float = 0.0
    self.best_partial_cell_acc: float = 0.0
    self.best_partial_gen_tokens: int = 0
    self.best_partial_turn: int = 0
    self.has_defined_solve: bool = False

    self.submitted_answer_grid: Grid | None = None
    self.submitted_answer_turn: int | None = None
    self.submitted_answer_gen_tokens: int | None = None

    self.env: dict[str, Any] = {
        "__builtins__": __builtins__,
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "ndimage": ndimage,
        "Counter": collections.Counter,
        "defaultdict": collections.defaultdict,
        "deque": collections.deque,
        "combinations": itertools.combinations,
        "permutations": itertools.permutations,
        "product": itertools.product,
        "print": self._sandbox_print,
        "check": self._check_fn,
    }
    if self.use_arc_helpers:
      exec(arc_dsl_and_summary.ARC_HELPERS_SOURCE, self.env)  # pylint: disable=exec-used
      self.env["print"] = self._sandbox_print
      self.env["check"] = self._check_fn
    self._refresh_task_globals()

  def _sandbox_print(self, *args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end
    if self._active_stdout_buf.tell() < 32000:
      self._active_stdout_buf.write(text)

  def _refresh_task_globals(self) -> None:
    self.env["train"] = [
        {"input": p["input"].copy(), "output": p["output"].copy()}
        for p in self._train_np
    ]
    self.env["test_input"] = self._test_np.copy()
    self.env["print"] = self._sandbox_print
    self.env["check"] = self._check_fn

  def _check_fn(self, fn: Any = None) -> bool:
    target_fn = fn if fn is not None else self.env.get("solve")
    if not callable(target_fn):
      self._sandbox_print("check(): No callable `solve(grid)` function found.")
      return False
    saved_train = self.env.pop("train", None)
    saved_test = self.env.pop("test_input", None)
    try:
      all_ok = True
      reports = []
      for idx, pair in enumerate(self._train_np):
        try:
          pred = target_fn(pair["input"].copy())
          if not is_valid_grid(pred):
            all_ok = False
            reports.append(
                f"Pair {idx}: FAIL (returned invalid grid of type {type(pred).__name__})"
            )
            continue
          pred_arr = np.asarray(pred, dtype=int)
          rep = format_pair_diff(pred_arr, pair["output"], idx)
          if not (
              pred_arr.shape == pair["output"].shape
              and np.array_equal(pred_arr, pair["output"])
          ):
            all_ok = False
          reports.append(rep)
        except Exception as exc:
          all_ok = False
          reports.append(f"Pair {idx}: ERROR ({type(exc).__name__}: {exc})")
      self._sandbox_print("\n".join(reports))
      return all_ok
    finally:
      if saved_train is not None:
        self.env["train"] = saved_train
      if saved_test is not None:
        self.env["test_input"] = saved_test

  def _run_in_subprocess(
      self,
      code: str,
      require_solve: bool = False,
      timeout_s: float = 6.0,
  ) -> dict[str, Any]:
    """Execute `code` in-process inside a traced daemon thread with zero os.fork()."""
    stdout_buf = io.StringIO()
    self._active_stdout_buf = stdout_buf
    self._refresh_task_globals()
    prev_solve = self.env.get("solve")

    result: dict[str, Any] = {
        "stdout": "",
        "error": None,
        "defined_solve": False,
        "train_pass": False,
        "train_passed_count": 0,
        "train_total_count": len(self.train_pairs),
        "train_frac": 0.0,
        "mean_cell_acc": 0.0,
        "pair_reports": [],
        "test_pred": None,
        "test_error": None,
        "sanity_ok": False,
        "sanity_reason": "not_evaluated",
        "equivariance_bonus": 0.0,
    }

    deadline = time.monotonic() + timeout_s

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
      del frame, event, arg
      if time.monotonic() > deadline:
        raise TimeoutError("Execution timed out (limit: 6.0s).")
      return _tracer

    def _worker() -> None:
      sys.settrace(_tracer)
      try:
        exec(code, self.env)  # pylint: disable=exec-used
        solve_fn = self.env.get("solve")
        should_eval_solve = callable(solve_fn) and (
            require_solve or (solve_fn is not prev_solve) or ("def solve" in code)
        )
        if should_eval_solve:
          result["defined_solve"] = True
          saved_train = self.env.pop("train", None)
          saved_test = self.env.pop("test_input", None)
          try:
            passed_count = 0
            cell_accs: list[float] = []
            pair_reports: list[str] = []
            for idx, pair in enumerate(self._train_np):
              try:
                got_raw = solve_fn(pair["input"].copy())
                if not is_valid_grid(got_raw):
                  pair_reports.append(
                      f"Pair {idx}: FAIL (returned invalid grid: {type(got_raw).__name__})"
                  )
                  cell_accs.append(0.0)
                  continue
                got_arr = np.asarray(got_raw, dtype=int)
                exp_arr = pair["output"]
                rep = format_pair_diff(got_arr, exp_arr, idx)
                pair_reports.append(rep)
                if got_arr.shape == exp_arr.shape and np.array_equal(
                    got_arr, exp_arr
                ):
                  passed_count += 1
                  cell_accs.append(1.0)
                elif got_arr.shape == exp_arr.shape:
                  cell_accs.append(float(np.mean(got_arr == exp_arr)))
                else:
                  cell_accs.append(0.0)
              except TimeoutError:
                raise
              except Exception as exc:
                pair_reports.append(
                    f"Pair {idx}: ERROR ({type(exc).__name__}: {exc})"
                )
                cell_accs.append(0.0)

            total_pairs = max(1, len(self._train_np))
            result["train_passed_count"] = passed_count
            result["train_frac"] = passed_count / total_pairs
            result["mean_cell_acc"] = (
                float(np.mean(cell_accs)) if cell_accs else 0.0
            )
            result["pair_reports"] = pair_reports
            result["train_pass"] = passed_count == len(self._train_np)

            try:
              test_raw = solve_fn(self._test_np.copy())
              if is_valid_grid(test_raw):
                pred_grid = to_grid(test_raw)
                result["test_pred"] = pred_grid
                s_ok, s_reason = sanity_check_prediction(
                    pred_grid, self.train_pairs, self.test_input
                )
                result["sanity_ok"] = s_ok
                result["sanity_reason"] = s_reason
              else:
                result["test_error"] = (
                    f"solve(test_input) returned invalid grid of type {type(test_raw).__name__}"
                )
            except TimeoutError:
              raise
            except Exception as exc:
              result["test_error"] = f"{type(exc).__name__}: {exc}"

            if (
                result["train_pass"]
                and result["sanity_ok"]
                and self.run_equivariance_audit
            ):
              eq_score = 0.0
              d8_ops = [
                  lambda g: np.rot90(g, 1),
                  lambda g: np.rot90(g, 2),
                  lambda g: np.rot90(g, 3),
                  lambda g: np.fliplr(g),
                  lambda g: np.flipud(g),
              ]
              d8_passed = 0
              d8_tested = 0
              for op in d8_ops:
                all_op_ok = True
                for pair in self._train_np:
                  try:
                    t_in = op(pair["input"].copy())
                    t_out_exp = op(pair["output"].copy())
                    t_got = solve_fn(t_in)
                    if not is_valid_grid(t_got):
                      all_op_ok = False
                      break
                    t_got_arr = np.asarray(t_got, dtype=int)
                    if t_got_arr.shape != t_out_exp.shape or not np.array_equal(
                        t_got_arr, t_out_exp
                    ):
                      all_op_ok = False
                      break
                  except TimeoutError:
                    raise
                  except Exception:
                    all_op_ok = False
                    break
                d8_tested += 1
                if all_op_ok:
                  d8_passed += 1
              if d8_tested > 0:
                eq_score += 1.5 * (d8_passed / d8_tested)

              bg_c = int(
                  arc_dsl_and_summary._bg_colour_host(self._train_np[0]["input"])
              )
              fg_cols = sorted(
                  {
                      int(c)
                      for pair in self._train_np
                      for c in np.unique(pair["input"])
                      if int(c) != bg_c
                  }
              )
              if len(fg_cols) >= 2:
                c1, c2 = fg_cols[0], fg_cols[1]

                def _swap(arr: np.ndarray) -> np.ndarray:
                  out = arr.copy()
                  m1 = arr == c1
                  m2 = arr == c2
                  out[m1] = c2
                  out[m2] = c1
                  return out

                swap_ok = True
                for pair in self._train_np:
                  try:
                    s_got = solve_fn(_swap(pair["input"]))
                    if not is_valid_grid(s_got):
                      swap_ok = False
                      break
                    s_got_arr = np.asarray(s_got, dtype=int)
                    s_exp = _swap(pair["output"])
                    if s_got_arr.shape != s_exp.shape or not np.array_equal(
                        s_got_arr, s_exp
                    ):
                      swap_ok = False
                      break
                  except TimeoutError:
                    raise
                  except Exception:
                    swap_ok = False
                    break
                if swap_ok:
                  eq_score += 1.0

              full_source = "\n".join(self.helper_cells + [code])
              penalty = ast_constant_lint_penalty(
                  full_source, self.train_pairs, self.test_input
              )
              result["equivariance_bonus"] = max(
                  0.0, round(eq_score - penalty, 4)
              )
          finally:
            if saved_train is not None:
              self.env["train"] = saved_train
            if saved_test is not None:
              self.env["test_input"] = saved_test
      except Exception as exc:
        result["error"] = (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        )
      finally:
        sys.settrace(None)

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=timeout_s + 0.5)
    if worker_thread.is_alive():
      return {
          "stdout": stdout_buf.getvalue(),
          "error": "TimeoutError: Python cell execution exceeded hard 6.0s limit.",
          "defined_solve": False,
          "train_pass": False,
          "train_passed_count": 0,
          "train_total_count": len(self.train_pairs),
          "train_frac": 0.0,
          "mean_cell_acc": 0.0,
          "pair_reports": [],
          "test_pred": None,
          "test_error": None,
          "sanity_ok": False,
          "sanity_reason": "timeout",
          "equivariance_bonus": 0.0,
      }
    result["stdout"] = stdout_buf.getvalue()
    return result

  def _truncate(self, text: str) -> str:
    if len(text) <= self.max_output_chars:
      return text
    half = self.max_output_chars // 2 - 40
    return (
        text[:half]
        + f"\n... [truncated {len(text) - self.max_output_chars} chars] ...\n"
        + text[-half:]
    )

  def _record_solve_outcome(
      self,
      res: dict[str, Any],
      code: str,
      turn_idx: int,
      cum_gen_tokens: int,
  ) -> None:
    self.has_defined_solve = True
    pred: Grid | None = res.get("test_pred")
    if pred is None:
      return
    train_pass = bool(res.get("train_pass"))
    sanity_ok = bool(res.get("sanity_ok"))
    eq_bonus = float(res.get("equivariance_bonus", 0.0))

    if train_pass and sanity_ok:
      self.latest_tp_pred = pred
      self.latest_tp_code = code
      self.verified_variants.append(
          VerifiedVariant(
              grid=pred,
              code=code,
              turn_idx=turn_idx,
              cum_gen_tokens=cum_gen_tokens,
              equivariance_bonus=eq_bonus,
          )
      )
      if self.first_tp_pred is None:
        self.first_tp_pred = pred
        self.first_tp_code = code
        self.first_tp_turn = turn_idx
        self.first_tp_gen_tokens = cum_gen_tokens
        self.first_tp_equivariance_bonus = eq_bonus
    elif sanity_ok:
      t_frac = float(res.get("train_frac", 0.0))
      c_acc = float(res.get("mean_cell_acc", 0.0))
      if (t_frac, c_acc) >= (self.best_partial_train_frac, self.best_partial_cell_acc):
        self.best_partial_pred = pred
        self.best_partial_code = code
        self.best_partial_pair_reports = list(res.get("pair_reports", []))
        self.best_partial_train_frac = t_frac
        self.best_partial_cell_acc = c_acc
        self.best_partial_gen_tokens = cum_gen_tokens
        self.best_partial_turn = turn_idx

  def execute_run_python(
      self,
      code: str,
      turn_idx: int,
      cum_gen_tokens: int,
  ) -> tuple[str, bool]:
    """Execute a `run_python` tool call and return `(tool_response_text, achieved_train_pass)`."""
    res = self._run_in_subprocess(code)
    if res.get("error") is None:
      self.helper_cells.append(code)

    defined_solve = bool(res.get("defined_solve"))
    if defined_solve:
      self._record_solve_outcome(res, code, turn_idx, cum_gen_tokens)

    pred = res.get("test_pred")
    self.code_traces.append(
        CodeTurnTrace(
            turn_idx=turn_idx,
            tool_name="run_python",
            code_or_payload=code,
            stdout=res.get("stdout", ""),
            error=res.get("error"),
            defined_solve=defined_solve,
            train_pass=bool(res.get("train_pass")),
            train_passed_count=int(res.get("train_passed_count", 0)),
            train_total_count=int(res.get("train_total_count", len(self.train_pairs))),
            train_frac=float(res.get("train_frac", 0.0)),
            mean_cell_acc=float(res.get("mean_cell_acc", 0.0)),
            pair_reports=list(res.get("pair_reports", [])),
            test_pred_shape=(
                (len(pred), len(pred[0])) if pred is not None and pred else None
            ),
            sanity_ok=bool(res.get("sanity_ok")),
            sanity_reason=str(res.get("sanity_reason", "")),
            equivariance_bonus=float(res.get("equivariance_bonus", 0.0)),
            cum_gen_tokens=cum_gen_tokens,
        )
    )

    out_parts: list[str] = []
    stdout_str = (res.get("stdout") or "").strip()
    if stdout_str:
      out_parts.append(self._truncate(stdout_str))
    if res.get("error"):
      out_parts.append(f"ERROR:\n{self._truncate(res['error'])}")
    if not out_parts:
      out_parts.append("(executed with no stdout)")

    achieved_tp = False
    if self.auto_check_on_solve and defined_solve and res.get("error") is None:
      passed = int(res.get("train_passed_count", 0))
      total = int(res.get("train_total_count", len(self.train_pairs)))
      reports = "\n".join(res.get("pair_reports", []))
      if res.get("train_pass") and res.get("sanity_ok"):
        achieved_tp = True
        pred_arr = np.asarray(pred, dtype=int)
        out_parts.append(
            f"\n[Auto-Check `solve(grid)`: PASS {passed}/{total} train pairs! "
            f"test_output shape={pred_arr.shape[0]}x{pred_arr.shape[1]}. "
            f"Call `submit_program` with `def solve(grid):` to finalize.]\n{reports}"
        )
      else:
        reason = (
            f" (test_output issue: {res.get('test_error') or res.get('sanity_reason')})"
            if res.get("train_pass") and not res.get("sanity_ok")
            else ""
        )
        out_parts.append(
            f"\n[Auto-Check `solve(grid)`: {passed}/{total} train pairs passed{reason}]\n{reports}"
        )

    return "\n".join(out_parts), achieved_tp

  def execute_submit_program(
      self,
      code: str,
      turn_idx: int,
      cum_gen_tokens: int,
  ) -> tuple[str, bool]:
    """Execute a `submit_program` tool call and return `(tool_response_text, should_terminate)`."""
    res = self._run_in_subprocess(code, require_solve=True)
    if res.get("error") is None:
      self.helper_cells.append(code)

    defined_solve = bool(res.get("defined_solve"))
    if defined_solve:
      self._record_solve_outcome(res, code, turn_idx, cum_gen_tokens)

    pred = res.get("test_pred")
    self.code_traces.append(
        CodeTurnTrace(
            turn_idx=turn_idx,
            tool_name="submit_program",
            code_or_payload=code,
            stdout=res.get("stdout", ""),
            error=res.get("error"),
            defined_solve=defined_solve,
            train_pass=bool(res.get("train_pass")),
            train_passed_count=int(res.get("train_passed_count", 0)),
            train_total_count=int(res.get("train_total_count", len(self.train_pairs))),
            train_frac=float(res.get("train_frac", 0.0)),
            mean_cell_acc=float(res.get("mean_cell_acc", 0.0)),
            pair_reports=list(res.get("pair_reports", [])),
            test_pred_shape=(
                (len(pred), len(pred[0])) if pred is not None and pred else None
            ),
            sanity_ok=bool(res.get("sanity_ok")),
            sanity_reason=str(res.get("sanity_reason", "")),
            equivariance_bonus=float(res.get("equivariance_bonus", 0.0)),
            cum_gen_tokens=cum_gen_tokens,
        )
    )

    if res.get("error"):
      return (
          f"submit_program ERROR:\n{self._truncate(res['error'])}\n"
          "Fix the error and call `submit_program` again.",
          False,
      )
    if not defined_solve:
      return (
          "submit_program ERROR: Your code did not define `def solve(grid):`. "
          "Define `def solve(grid):` returning a 2D grid and call `submit_program` again.",
          False,
      )

    passed = int(res.get("train_passed_count", 0))
    total = int(res.get("train_total_count", len(self.train_pairs)))
    reports = "\n".join(res.get("pair_reports", []))

    if res.get("train_pass") and res.get("sanity_ok") and pred is not None:
      pred_arr = np.asarray(pred, dtype=int)
      return (
          f"Verification PASSED ({passed}/{total} training pairs exact match).\n"
          f"{reports}\n"
          f"Generated test output ({pred_arr.shape[0]}x{pred_arr.shape[1]}):\n"
          f"{format_grid_compact(pred)}",
          True,
      )

    if res.get("train_pass") and not res.get("sanity_ok"):
      err_detail = res.get("test_error") or res.get("sanity_reason")
      return (
          f"Verification passed {passed}/{total} train pairs, capability check on "
          f"`test_input` FAILED ({err_detail}). Inspect `test_input` and fix `solve(grid)`.",
          False,
      )

    return (
        f"Verification FAILED ({passed}/{total} training pairs passed).\n"
        f"{reports}\n"
        "Inspect the mismatched coordinates above using `run_python`, refine your rule, "
        "and call `submit_program` again.",
        False,
    )

  def execute_submit_answer(
      self,
      grid_obj: Any,
      turn_idx: int,
      cum_gen_tokens: int,
  ) -> tuple[str, bool]:
    """Validate and record a direct `submit_answer` tool call."""
    valid = is_valid_grid(grid_obj)
    pred_grid = to_grid(grid_obj) if valid else None
    s_ok = False
    s_reason = "invalid_grid"
    if pred_grid is not None:
      s_ok, s_reason = sanity_check_prediction(
          pred_grid, self.train_pairs, self.test_input
      )
      if s_ok:
        self.submitted_answer_grid = pred_grid
        self.submitted_answer_turn = turn_idx
        self.submitted_answer_gen_tokens = cum_gen_tokens

    self.code_traces.append(
        CodeTurnTrace(
            turn_idx=turn_idx,
            tool_name="submit_answer",
            code_or_payload=str(grid_obj)[:1000],
            stdout="",
            error=None if valid else "Invalid 2D grid in submit_answer",
            defined_solve=False,
            train_pass=False,
            train_passed_count=0,
            train_total_count=len(self.train_pairs),
            train_frac=0.0,
            mean_cell_acc=0.0,
            pair_reports=[],
            test_pred_shape=(
                (len(pred_grid), len(pred_grid[0])) if pred_grid else None
            ),
            sanity_ok=s_ok,
            sanity_reason=s_reason,
            equivariance_bonus=0.0,
            cum_gen_tokens=cum_gen_tokens,
        )
    )
    if valid and s_ok:
      return (
          f"Direct grid answer recorded ({len(pred_grid)}x{len(pred_grid[0])}).",
          True,
      )
    return (
        f"submit_answer rejected ({s_reason}). Prefer writing `def solve(grid):` and "
        "calling `submit_program` so your rule is verified on all training pairs.",
        False,
    )
