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
      use_e031_ipython_mode: bool = False,
  ) -> None:
    self.train_pairs = train_pairs
    self.test_input = test_input
    self.use_arc_helpers = use_arc_helpers
    self.auto_check_on_solve = auto_check_on_solve
    self.run_equivariance_audit = run_equivariance_audit
    self.use_e031_ipython_mode = use_e031_ipython_mode
    self.max_output_chars = 16000 if use_e031_ipython_mode else max_output_chars

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
    self._rejected_submit_codes: set[str] = set()

    self.env: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
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
    if self.use_e031_ipython_mode:
      train_lists = [
          {"input": to_grid(p["input"]), "output": to_grid(p["output"])}
          for p in self.train_pairs
      ]
      test_list = to_grid(self.test_input)
      self.env["train"] = train_lists
      self.env["test_input"] = test_list
      self.env["test_in"] = test_list
      self.env["task"] = {
          "train": train_lists,
          "test": [{"input": test_list}],
      }
    else:
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
          arg = (
              to_grid(pair["input"])
              if self.use_e031_ipython_mode
              else pair["input"].copy()
          )
          pred = target_fn(arg)
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
    if self.use_e031_ipython_mode and timeout_s == 6.0:
      timeout_s = 15.0
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

    class _SandboxTimeout(BaseException):
      pass

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
      del frame, event, arg
      if time.monotonic() > deadline:
        raise _SandboxTimeout(f"Execution timed out (limit: {timeout_s:g}s).")
      return _tracer

    def _worker() -> None:
      sys.settrace(_tracer)
      try:
        if self.use_e031_ipython_mode:
          import pprint  # pylint: disable=g-import-not-at-top
          tree = ast.parse(code, "<ipython-input>", "exec")
          last_expr = None
          if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_stmt = tree.body.pop()
            if isinstance(last_stmt, ast.Expr):
              last_expr = ast.Expression(last_stmt.value)
          exec(compile(tree, "<ipython-input>", "exec"), self.env)  # pylint: disable=exec-used
          if last_expr is not None:
            val = eval(compile(last_expr, "<ipython-input>", "eval"), self.env)  # pylint: disable=eval-used
            if val is not None:
              if isinstance(val, (list, tuple, dict, set, frozenset)):
                txt = pprint.pformat(val, width=79, sort_dicts=False)
              else:
                txt = repr(val)
              self._sandbox_print("\n" + txt if "\n" in txt else txt)
        else:
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
                inp_arg = (
                    to_grid(pair["input"])
                    if self.use_e031_ipython_mode
                    else pair["input"].copy()
                )
                try:
                  got_raw = solve_fn(inp_arg)
                except Exception:
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
              except (TimeoutError, _SandboxTimeout):
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
              test_arg = (
                  to_grid(self._test_np)
                  if self.use_e031_ipython_mode
                  else self._test_np.copy()
              )
              try:
                test_raw = solve_fn(test_arg)
              except Exception:
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
            except (TimeoutError, _SandboxTimeout):
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
                  except (TimeoutError, _SandboxTimeout):
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
                  except (TimeoutError, _SandboxTimeout):
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
      except _SandboxTimeout as exc:
        result["error"] = f"TimeoutError: {exc}"
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
          "error": f"TimeoutError: Execution timed out (limit: {timeout_s:g}s).",
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

    if self.use_e031_ipython_mode:
      stdout_raw = res.get("stdout") or ""
      err_raw = res.get("error")
      if err_raw:
        status = "timeout" if "TimeoutError" in err_raw else "error"
        parts = [f"[{status}]"]
        if stdout_raw:
          parts.append(stdout_raw.rstrip("\n"))
        parts.append(err_raw.rstrip("\n"))
      else:
        parts = [
            "[completed]",
            stdout_raw.rstrip("\n") if stdout_raw else "[no output; use print()]",
        ]
      text = "\n".join(parts)
      if len(text) > 16000:
        text = text[:16000] + f"\n... [truncated, {len(text)} total chars]"
      return text, bool(res.get("train_pass") and res.get("sanity_ok"))

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

  def _check_e031_train_reproduction(
      self,
      code: str,
  ) -> tuple[bool, int, float]:
    """Check whether `code` (or `solve` in session) reproduces all `train` pairs (E031 `check_answer_on_train`)."""
    import copy  # pylint: disable=g-import-not-at-top
    import types  # pylint: disable=g-import-not-at-top

    total = max(1, len(self._train_np))
    test_names = ("test_in", "test_input")

    def _eval_candidate_fn(cand_fn: Any) -> tuple[bool, int, float]:
      passed = 0
      cell_accs: list[float] = []
      for pair in self._train_np:
        try:
          try:
            got_raw = cand_fn(copy.deepcopy(to_grid(pair["input"])))
          except Exception:
            got_raw = cand_fn(pair["input"].copy())
          if not is_valid_grid(got_raw):
            cell_accs.append(0.0)
            continue
          got_arr = np.asarray(got_raw, dtype=int)
          exp_arr = pair["output"]
          if got_arr.shape == exp_arr.shape and np.array_equal(got_arr, exp_arr):
            passed += 1
            cell_accs.append(1.0)
          elif got_arr.shape == exp_arr.shape:
            cell_accs.append(float(np.mean(got_arr == exp_arr)))
          else:
            cell_accs.append(0.0)
        except Exception:
          cell_accs.append(0.0)
      mean_acc = float(np.mean(cell_accs)) if cell_accs else 0.0
      return (passed == len(self._train_np)), passed, mean_acc

    best_res: tuple[bool, int, float] = (False, 0, 0.0)

    # 1. If `code` is an expression referencing `test_input` or `test_in`, evaluate it directly on each train pair.
    stripped_code = code.strip()
    if any(tn in stripped_code for tn in test_names):
      try:
        comp_expr = compile(stripped_code, "<train_check_expr>", "eval")

        def _expr_runner(inp_grid: Any) -> Any:
          local_env = dict(self.env)
          local_env["test_input"] = inp_grid
          local_env["test_in"] = inp_grid
          return eval(comp_expr, local_env)  # pylint: disable=eval-used

        tp, p_cnt, m_acc = _eval_candidate_fn(_expr_runner)
        if tp:
          return True, p_cnt, m_acc
        if (p_cnt, m_acc) > (best_res[1], best_res[2]):
          best_res = (tp, p_cnt, m_acc)
      except SyntaxError:
        pass

    # 2. Check explicit single-arg call AST or `solve` or recent 1-arg user functions in session.
    candidate_fns: list[Any] = []
    try:
      tree = ast.parse(stripped_code, mode="eval")
      if (
          isinstance(tree.body, ast.Call)
          and len(tree.body.args) == 1
          and not tree.body.keywords
          and isinstance(tree.body.args[0], ast.Name)
          and tree.body.args[0].id in test_names
          and isinstance(tree.body.func, ast.Name)
      ):
        cand = self.env.get(tree.body.func.id)
        if callable(cand):
          candidate_fns.append(cand)
    except SyntaxError:
      pass

    if callable(self.env.get("solve")):
      candidate_fns.append(self.env.get("solve"))

    user_fns = [
        v
        for k, v in self.env.items()
        if isinstance(v, types.FunctionType)
        and not k.startswith("_")
        and getattr(v, "__code__", None) is not None
        and v.__code__.co_argcount >= 1
    ]
    for u_fn in reversed(user_fns[-8:]):
      if u_fn not in candidate_fns:
        candidate_fns.append(u_fn)

    for cand_fn in candidate_fns:
      tp, p_cnt, m_acc = _eval_candidate_fn(cand_fn)
      if tp:
        return True, p_cnt, m_acc
      if (p_cnt, m_acc) > (best_res[1], best_res[2]):
        best_res = (tp, p_cnt, m_acc)

    return best_res

  def _run_with_hard_timeout(
      self,
      fn_body: Any,
      timeout_s: float = 15.0,
  ) -> tuple[Any, str | None]:
    """Run `fn_body()` inside a traced daemon thread with a strict `BaseException` timeout."""
    deadline = time.monotonic() + timeout_s

    class _SandboxTimeout(BaseException):
      pass

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
      del frame, event, arg
      if time.monotonic() > deadline:
        raise _SandboxTimeout(f"Execution timed out (limit: {timeout_s:g}s).")
      return _tracer

    box: dict[str, Any] = {"val": None, "err": None}

    def _worker() -> None:
      sys.settrace(_tracer)
      try:
        box["val"] = fn_body()
      except _SandboxTimeout as exc:
        box["err"] = f"TimeoutError: {exc}"
      except Exception as exc:
        box["err"] = f"{type(exc).__name__}: {exc}"
      finally:
        sys.settrace(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s + 0.5)
    if t.is_alive():
      return None, f"TimeoutError: Execution timed out (limit: {timeout_s:g}s)."
    return box["val"], box["err"]

  def execute_submit_answer_code(
      self,
      code: str,
      turn_idx: int,
      cum_gen_tokens: int,
  ) -> tuple[str, bool]:
    """Execute E031 `submit_answer(code)` in the persistent IPython session and return `(tool_response_text, should_terminate)`."""
    self._refresh_task_globals()

    def _eval_and_check() -> tuple[Any, Grid | None, Grid | None, bool, int, float]:
      try:
        val = eval(compile(code, "<submit_answer>", "eval"), self.env)  # pylint: disable=eval-used
      except SyntaxError:
        exec(compile(code, "<submit_answer>", "exec"), self.env)  # pylint: disable=exec-used
        val = self.env.get("answer")

      p_grid: Grid | None = None
      s_grid: Grid | None = None
      if is_valid_grid(val):
        p_grid = to_grid(val)
      elif (
          isinstance(val, (list, tuple))
          and len(val) == 2
          and is_valid_grid(val[0])
          and is_valid_grid(val[1])
      ):
        p_grid = to_grid(val[0])
        s_grid = to_grid(val[1])

      if p_grid is None:
        return val, None, None, False, 0, 0.0

      tp, p_cnt, m_acc = self._check_e031_train_reproduction(code)
      return val, p_grid, s_grid, tp, p_cnt, m_acc

    out_tuple, eval_err = self._run_with_hard_timeout(_eval_and_check, timeout_s=15.0)
    if eval_err is not None or out_tuple is None:
      return (
          f"[invalid answer: could not evaluate code ({eval_err}). "
          "Fix and call submit_answer again.]",
          False,
      )

    _, primary_grid, secondary_grid, train_pass, passed_cnt, mean_acc = out_tuple

    if primary_grid is None:
      return (
          "[invalid answer: expected 2D list[list[int]] with values 0..9, shape 1..30x1..30 "
          "(or [grid_1, grid_2] for 2 attempts). Fix and call submit_answer again.]",
          False,
      )
    self.submitted_answer_grid = primary_grid
    self.submitted_answer_turn = turn_idx
    self.submitted_answer_gen_tokens = cum_gen_tokens

    if train_pass:
      self.latest_tp_pred = primary_grid
      self.latest_tp_code = code
      self.verified_variants.append(
          VerifiedVariant(
              grid=primary_grid,
              code=code,
              turn_idx=turn_idx,
              cum_gen_tokens=cum_gen_tokens,
              equivariance_bonus=0.0,
          )
      )
      if self.first_tp_pred is None:
        self.first_tp_pred = primary_grid
        self.first_tp_code = code
        self.first_tp_turn = turn_idx
        self.first_tp_gen_tokens = cum_gen_tokens
        self.first_tp_equivariance_bonus = 0.0
      if secondary_grid is not None:
        self.verified_variants.append(
            VerifiedVariant(
                grid=secondary_grid,
                code=code + " # [grid_2]",
                turn_idx=turn_idx,
                cum_gen_tokens=cum_gen_tokens,
                equivariance_bonus=0.0,
            )
        )
    else:
      t_frac = passed_cnt / max(1, len(self._train_np))
      if (t_frac, mean_acc) >= (
          self.best_partial_train_frac,
          self.best_partial_cell_acc,
      ):
        self.best_partial_pred = primary_grid
        self.best_partial_code = code
        self.best_partial_train_frac = t_frac
        self.best_partial_cell_acc = mean_acc
        self.best_partial_gen_tokens = cum_gen_tokens
        self.best_partial_turn = turn_idx

    self.code_traces.append(
        CodeTurnTrace(
            turn_idx=turn_idx,
            tool_name="submit_answer",
            code_or_payload=code[:1000],
            stdout="",
            error=None,
            defined_solve=callable(self.env.get("solve")),
            train_pass=train_pass,
            train_passed_count=passed_cnt,
            train_total_count=len(self.train_pairs),
            train_frac=passed_cnt / max(1, len(self.train_pairs)),
            mean_cell_acc=mean_acc,
            pair_reports=[],
            test_pred_shape=(len(primary_grid), len(primary_grid[0])),
            sanity_ok=True,
            sanity_reason="ok",
            equivariance_bonus=0.0,
            cum_gen_tokens=cum_gen_tokens,
        )
    )

    # E031 `verify_submit`: if a candidate rule/function in the session failed to reproduce all
    # training outputs, reject the submission ONCE so the model can inspect the failure and fix it,
    # while keeping `submitted_answer_grid` and `best_partial_pred` saved as fallback.
    norm_code = code.strip()
    has_testable_fn = (
        any(tn in norm_code for tn in ("test_in", "test_input"))
        or callable(self.env.get("solve"))
        or passed_cnt > 0
        or mean_acc > 0.0
    )
    if (
        not train_pass
        and has_testable_fn
        and norm_code not in self._rejected_submit_codes
    ):
      self._rejected_submit_codes.add(norm_code)
      total_tr = len(self.train_pairs)
      return (
          f"[submission rejected: your rule only matched {passed_cnt}/{total_tr} training pairs "
          f"(mean cell accuracy {100.0 * mean_acc:.1f}%). Inspect the failing training pairs in "
          "`run_python`, fix your rule so it reproduces ALL training outputs, and call `submit_answer` again.]",
          False,
      )

    if secondary_grid is not None:
      return (
          f"[2 answers accepted: {len(primary_grid)}x{len(primary_grid[0])} and "
          f"{len(secondary_grid)}x{len(secondary_grid[0])} grids]",
          True,
      )
    return (
        f"[answer accepted: {len(primary_grid)}x{len(primary_grid[0])} grid]",
        True,
    )

  def finalize_e031_session_fallback(
      self,
      turn_idx: int,
      cum_gen_tokens: int,
  ) -> None:
    """Run E031's 3-tier session fallback (`answer` -> `solve(test_input)` -> `salvage_answer`) if no `submit_answer` succeeded."""
    if self.submitted_answer_grid is not None or self.first_tp_pred is not None:
      return
    import copy  # pylint: disable=g-import-not-at-top
    import types  # pylint: disable=g-import-not-at-top

    self._refresh_task_globals()
    # Tier 1: `answer` in session
    ans_val = self.env.get("answer")
    if is_valid_grid(ans_val):
      self.execute_submit_answer_code("answer", turn_idx, cum_gen_tokens)
      return
    if (
        isinstance(ans_val, (list, tuple))
        and len(ans_val) == 2
        and is_valid_grid(ans_val[0])
        and is_valid_grid(ans_val[1])
    ):
      self.execute_submit_answer_code("answer", turn_idx, cum_gen_tokens)
      return

    # Tier 2: `solve(test_input)` in session
    if callable(self.env.get("solve")):
      _, ok = self.execute_submit_answer_code(
          "solve(test_input)", turn_idx, cum_gen_tokens
      )
      if ok:
        return

    # Tier 3: `salvage_answer` across user functions in session (bounded by 15s timeout)
    def _salvage() -> str:
      skip_names = {"exit", "quit", "get_ipython", "open"}
      best_score = (0, 0.0, -1)
      best_grid: Grid | None = None
      best_fn_name = ""
      idx = 0
      for name, fn in list(self.env.items()):
        if name.startswith("_") or name in skip_names:
          continue
        if not isinstance(fn, types.FunctionType):
          continue
        code_obj = getattr(fn, "__code__", None)
        if code_obj is None:
          continue
        n_defaults = len(fn.__defaults__) if fn.__defaults__ else 0
        n_req = code_obj.co_argcount - n_defaults
        if not (n_req <= 1 <= code_obj.co_argcount):
          continue
        idx += 1
        n_exact = 0
        sim_sum = 0.0
        ok_fn = True
        for pair in self._train_np:
          try:
            try:
              got_raw = fn(copy.deepcopy(to_grid(pair["input"])))
            except Exception:
              got_raw = fn(pair["input"].copy())
          except Exception:
            ok_fn = False
            break
          if not is_valid_grid(got_raw):
            ok_fn = False
            break
          got_arr = np.asarray(got_raw, dtype=int)
          exp_arr = pair["output"]
          if got_arr.shape == exp_arr.shape:
            if np.array_equal(got_arr, exp_arr):
              n_exact += 1
              sim_sum += 1.0
            else:
              sim_sum += float(np.mean(got_arr == exp_arr))
        if not ok_fn:
          continue
        score = (n_exact, sim_sum / max(1, len(self._train_np)), idx)
        if score > best_score:
          try:
            try:
              test_raw = fn(copy.deepcopy(to_grid(self._test_np)))
            except Exception:
              test_raw = fn(self._test_np.copy())
            if is_valid_grid(test_raw):
              best_score = score
              best_grid = to_grid(test_raw)
              best_fn_name = name
          except Exception:
            pass
      return best_fn_name if best_grid is not None else ""

    best_fn_name, _ = self._run_with_hard_timeout(_salvage, timeout_s=15.0)
    if best_fn_name:
      self.execute_submit_answer_code(
          f"{best_fn_name}(test_input)", turn_idx, cum_gen_tokens
      )

