"""Python REPL sandbox, helper DSL, and training-pair verification for ARC-AGI-2."""

import ast
import builtins
from collections import Counter, deque
import copy
import io
import math
import sys
import threading
import time
import traceback
from typing import Any
import numpy as np

Grid = list[list[int]]


def normalize_grid(val: Any) -> Grid | None:
  """Converts a numpy array or nested list into a validated 2D rectangular Grid (0..9)."""
  if isinstance(val, np.ndarray):
    val = val.tolist()
  if not isinstance(val, (list, tuple)) or not val:
    return None
  out: Grid = []
  w: int | None = None
  for r in val:
    if isinstance(r, np.ndarray):
      r = r.tolist()
    if not isinstance(r, (list, tuple)) or not r:
      return None
    if w is None:
      w = len(r)
      if w > 30:
        return None
    elif len(r) != w:
      return None
    row_ints: list[int] = []
    for x in r:
      if isinstance(x, bool):
        return None
      if isinstance(x, (int, np.integer)):
        iv = int(x)
      elif isinstance(x, (float, np.floating)) and float(x).is_integer():
        iv = int(x)
      else:
        return None
      if not 0 <= iv <= 9:
        return None
      row_ints.append(iv)
    out.append(row_ints)
  if len(out) > 30:
    return None
  return out


def grid_key(g: Grid | None) -> str:
  if g is None:
    return ""
  return ";".join("".join(str(v) for v in row) for row in g)


def bg_colour(g: Any) -> int:
  arr = np.asarray(g, dtype=int)
  vals, counts = np.unique(arr, return_counts=True)
  return int(vals[int(np.argmax(counts))])


def bbox(mask_or_grid: Any, colour: int | None = None) -> tuple[int, int, int, int] | None:
  arr = np.asarray(mask_or_grid)
  mask = (arr != 0) if colour is None else (arr == colour)
  if not np.any(mask):
    return None
  rs, cs = np.where(mask)
  return int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1


def crop(g: Any, box: tuple[int, int, int, int] | None = None, bg: int = 0) -> np.ndarray:
  arr = np.asarray(g, dtype=int)
  if box is None:
    box = bbox(arr != bg)
    if box is None:
      return arr.copy()
  r0, c0, r1, c1 = box
  return arr[r0:r1, c0:c1].copy()


def find_objects(
    grid: Any,
    connectivity: int = 4,
    bg: int | None = None,
    multicolour: bool = False,
    diag: bool = False,
) -> list[dict[str, Any]]:
  """Connected components helper preloaded into the REPL as both `find_objects` and `objects`."""
  arr = np.asarray(grid, dtype=int)
  if bg is None:
    bg = bg_colour(arr)
  if diag:
    connectivity = 8
  h, w = arr.shape
  seen = np.zeros((h, w), dtype=bool)
  nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  if connectivity == 8:
    nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
  out: list[dict[str, Any]] = []
  for r in range(h):
    for c in range(w):
      v = int(arr[r, c])
      if seen[r, c] or v == bg:
        continue
      q = deque([(r, c)])
      seen[r, c] = True
      pixels: list[tuple[int, int]] = []
      while q:
        cr, cc = q.popleft()
        pixels.append((cr, cc))
        for dr, dc in nbrs:
          nr, nc = cr + dr, cc + dc
          if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
            nv = int(arr[nr, nc])
            if (multicolour and nv != bg) or (not multicolour and nv == v):
              seen[nr, nc] = True
              q.append((nr, nc))
      rs = [p[0] for p in pixels]
      cs = [p[1] for p in pixels]
      r0, c0, r1, c1 = min(rs), min(cs), max(rs) + 1, max(cs) + 1
      mask = np.zeros((h, w), dtype=bool)
      for pr, pc in pixels:
        mask[pr, pc] = True
      sub = np.where(mask[r0:r1, c0:c1], arr[r0:r1, c0:c1], bg)
      cols = {int(arr[pr, pc]) for pr, pc in pixels}
      out.append({
          "colour": v if len(cols) == 1 else -1,
          "color": v if len(cols) == 1 else -1,
          "colours": cols,
          "size": len(pixels),
          "bbox": (r0, c0, r1, c1),
          "shape": (r1 - r0, c1 - c0),
          "mask": mask,
          "subgrid": sub,
          "pixels": pixels,
          "cells": pixels,
      })
  out.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
  return out


def d8(g: Any) -> list[np.ndarray]:
  arr = np.asarray(g, dtype=int)
  out = [np.rot90(arr, k) for k in range(4)]
  f = np.fliplr(arr)
  out += [np.rot90(f, k) for k in range(4)]
  return out


def format_plain_diff(idx: int, expected: Grid, got: Grid | None, err: str | None) -> str:
  if err:
    return f"  train[{idx}]: FAIL ({err})"
  if got is None:
    return f"  train[{idx}]: FAIL (did not return a valid 2D integer grid 0..9)"
  eh, ew = len(expected), len(expected[0])
  gh, gw = len(got), len(got[0])
  if (eh, ew) != (gh, gw):
    exp_cnt = Counter(v for r in expected for v in r)
    got_cnt = Counter(v for r in got for v in r)
    return (
        f"  train[{idx}]: FAIL (shape {gh}x{gw}, expected {eh}x{ew}; "
        f"got colors={dict(sorted(got_cnt.items()))} vs exp colors={dict(sorted(exp_cnt.items()))})"
    )
  mismatches: list[str] = []
  wrong_coords: list[tuple[int, int]] = []
  for r in range(eh):
    for c in range(ew):
      if expected[r][c] != got[r][c]:
        wrong_coords.append((r, c))
        if len(mismatches) < 12:
          mismatches.append(f"({r},{c}):got={got[r][c]}!=exp={expected[r][c]}")
  wrong = len(wrong_coords)
  if wrong == 0:
    return f"  train[{idx}]: PASS ({eh}x{ew})"

  total = eh * ew
  matched = total - wrong
  acc_pct = 100.0 * matched / max(1, total)
  sample_str = ", ".join(mismatches)
  if wrong > len(mismatches):
    sample_str += f", ... (+{wrong - len(mismatches)} more)"

  rs = [p[0] for p in wrong_coords]
  cs = [p[1] for p in wrong_coords]
  r0, r1 = min(rs), max(rs)
  c0, c1 = min(cs), max(cs)

  exp_cnt = Counter(v for r in expected for v in r)
  got_cnt = Counter(v for r in got for v in r)
  all_cols = sorted(set(exp_cnt.keys()) | set(got_cnt.keys()))
  deltas = [
      f"color {k}: got {got_cnt.get(k, 0)} vs exp {exp_cnt.get(k, 0)} ({got_cnt.get(k, 0) - exp_cnt.get(k, 0):+d})"
      for k in all_cols
      if got_cnt.get(k, 0) != exp_cnt.get(k, 0)
  ]
  delta_line = f"\n    color count diff: {'; '.join(deltas)}" if deltas else ""

  # Render compact visual diff over the mismatch bounding box (padded by 1 cell, max 14x14)
  vr0 = max(0, r0 - 1)
  vr1 = min(eh - 1, r1 + 1)
  vc0 = max(0, c0 - 1)
  vc1 = min(ew - 1, c1 + 1)
  vis_lines: list[str] = []
  if (vr1 - vr0 + 1) <= 14 and (vc1 - vc0 + 1) <= 14:
    col_hdr = " ".join(f"{c:>3d}" for c in range(vc0, vc1 + 1))
    vis_lines.append(f"    visual diff (rows {vr0}..{vr1}, cols {vc0}..{vc1}; '.v'=match, 'g>e'=got>exp):")
    vis_lines.append(f"         {col_hdr}")
    for r in range(vr0, vr1 + 1):
      cells = []
      for c in range(vc0, vc1 + 1):
        gv, ev = got[r][c], expected[r][c]
        cells.append(f" .{ev}" if gv == ev else f"{gv}>{ev}")
      vis_lines.append(f"    r{r:02d}: " + " ".join(cells))
  vis_block = ("\n" + "\n".join(vis_lines)) if vis_lines else ""

  return (
      f"  train[{idx}]: FAIL ({matched}/{total} cells match = {acc_pct:.1f}%,"
      f" {wrong} wrong in bbox r={r0}..{r1}, c={c0}..{c1}: {sample_str})"
      f"{delta_line}{vis_block}"
  )


_FORBIDDEN_SOLVE_GLOBALS = frozenset({
    "train",
    "train_inputs",
    "train_outputs",
    "test_input",
    "test_input_arr",
    "all_test_inputs",
})


def _find_forbidden_globals_in_code_obj(code_obj: Any) -> list[str]:
  bad: set[str] = set()
  for name in getattr(code_obj, "co_names", ()):
    if name in _FORBIDDEN_SOLVE_GLOBALS:
      bad.add(name)
  for const in getattr(code_obj, "co_consts", ()):
    if hasattr(const, "co_names"):
      bad.update(_find_forbidden_globals_in_code_obj(const))
  return sorted(bad)


def audit_test_grid(task: dict[str, Any], test_input: Grid, test_out: Grid) -> list[str]:
  """Lightweight zero-token sanity audit on test_out (checks invented colors / broken size rules / no-op)."""
  flags_out: list[str] = []
  train = task["train"]
  if not train:
    return flags_out
  th, tw = len(test_out), len(test_out[0])
  ih, iw = len(test_input), len(test_input[0])

  if all(
      (len(p["input"]), len(p["input"][0]))
      == (len(p["output"]), len(p["output"][0]))
      for p in train
  ):
    if (th, tw) != (ih, iw):
      flags_out.append(
          f"all training pairs preserve input shape, but test output is {th}x{tw} vs test input {ih}x{iw}"
      )
  elif len(train) >= 2:
    out_shapes = {(len(p["output"]), len(p["output"][0])) for p in train}
    in_shapes = {(len(p["input"]), len(p["input"][0])) for p in train}
    if len(out_shapes) == 1 and len(in_shapes) > 1:
      exp_h, exp_w = next(iter(out_shapes))
      if (th, tw) != (exp_h, exp_w):
        flags_out.append(
            f"all training outputs have constant shape {exp_h}x{exp_w} across varying input sizes, but test output is {th}x{tw}"
        )

  if test_out == test_input and all(p["input"] != p["output"] for p in train):
    flags_out.append(
        "test output is identical to test_input (every training pair modified its input)"
    )

  test_in_cols = {v for r in test_input for v in r}
  test_out_cols = {v for r in test_out for v in r}
  if len(test_out_cols) <= 1 and all(
      len({v for r in p["output"] for v in r}) >= 2 for p in train
  ):
    flags_out.append(
        "test output collapsed to a single uniform color (all background), whereas every training output has >=2 colors"
    )

  new_test_cols = test_out_cols - test_in_cols
  if new_test_cols:
    train_added = [
        {v for r in p["output"] for v in r} - {v for r in p["input"] for v in r}
        for p in train
    ]
    ever_added = set().union(*train_added) if train_added else set()
    invented = new_test_cols - ever_added
    if invented:
      flags_out.append(
          f"test output introduces color(s) {sorted(invented)} absent from test_input and never added in train"
      )
  return flags_out


class ArcPythonSandbox:
  """Per-attempt stateful Python REPL + train-pair verifier + candidate harvester."""

  def __init__(
      self,
      task: dict[str, Any],
      test_input: Grid,
      freeze_first_verified: bool = False,
      auto_check: bool = True,
  ):
    self.task = task
    self.test_input = copy.deepcopy(test_input)
    # Non-destructive audit: once a verified (train-pass & audit-clean) program
    # exists, later verified submissions are appended as candidates but never
    # replace the primary answer of this attempt.
    self.freeze_first_verified = freeze_first_verified
    # Pilot P1: when True, every `run_python` turn that (re)defines `solve`
    # without calling `check`/`submit_program` gets an automatic harness
    # verification appended to the tool response.
    self.auto_check = auto_check
    self.train_passed = False
    self.best_train_frac = 0.0
    self.best_cell_frac = 0.0
    self.primary_grid: Grid | None = None
    self.primary_train_pass: bool | None = None
    self.primary_train_frac: float | None = None
    self.primary_suspect: bool = False
    self.first_verified_grid: Grid | None = None
    self.last_verified_grid: Grid | None = None
    self.num_verified_submissions = 0
    self.frozen_overrides = 0  # verified submissions that would have replaced the primary
    # Diagnostics for the commit-pacing pilot (P2) and offline analysis.
    self.ever_defined_solve = False
    self.num_verifications = 0  # harness verification events (check/submit/auto)
    self.num_auto_checks = 0
    # All distinct candidate grids discovered during the attempt:
    # list of dicts: {grid, source, train_pass, train_frac, cell_frac, suspect}
    self.candidates: list[dict[str, Any]] = []
    self._out_buf = io.StringIO()
    self._check_called_in_turn = False

    self.ns: dict[str, Any] = {
        "__builtins__": builtins.__dict__,
        "np": np,
        "math": math,
        "copy": copy,
        "Counter": Counter,
        "deque": deque,
        "objects": find_objects,
        "find_objects": find_objects,
        "bbox": bbox,
        "crop": crop,
        "bg_colour": bg_colour,
        "most_common_color": bg_colour,
        "d8": d8,
        "print": self._custom_print,
        "check": self._check_fn,
        "submit_program": self._submit_program_fn,
        "submit_answer": self._submit_answer_fn,
    }
    self._restore_task_globals()
    try:
      import scipy  # pylint: disable=g-import-not-at-top
      import scipy.ndimage  # pylint: disable=g-import-not-at-top

      self.ns["scipy"] = scipy
      self.ns["ndimage"] = scipy.ndimage
      self.ns["label"] = scipy.ndimage.label
      self.ns["binary_fill_holes"] = scipy.ndimage.binary_fill_holes
    except Exception:
      pass

  def _restore_task_globals(self) -> None:
    """Restores uncorrupted copies of `train` and `test_input` before each turn."""
    self.ns["train"] = [
        {"input": copy.deepcopy(p["input"]), "output": copy.deepcopy(p["output"])}
        for p in self.task["train"]
    ]
    self.ns["train_inputs"] = [
        np.array(p["input"], dtype=int) for p in self.task["train"]
    ]
    self.ns["train_outputs"] = [
        np.array(p["output"], dtype=int) for p in self.task["train"]
    ]
    self.ns["test_input"] = copy.deepcopy(self.test_input)
    self.ns["test_input_arr"] = np.array(self.test_input, dtype=int)
    self.ns["all_test_inputs"] = [
        copy.deepcopy(t["input"])
        for t in self.task.get("test", [{"input": self.test_input}])
    ]

  def _custom_print(
      self, *args: Any, sep: str = " ", end: str = "\n", **kw: Any
  ) -> None:
    del kw
    s = sep.join(str(a) for a in args) + end
    if self._out_buf.tell() < 12000:
      self._out_buf.write(s[:4000])

  def _record_candidate(
      self,
      grid: Grid,
      source: str,
      train_pass: bool | None,
      train_frac: float | None,
      cell_frac: float = 0.0,
      suspect: bool = False,
      is_primary: bool = False,
  ) -> None:
    verified = train_pass is True and not suspect
    wants_primary = is_primary
    if verified:
      self.num_verified_submissions += 1
      if self.first_verified_grid is None:
        self.first_verified_grid = grid
      self.last_verified_grid = grid
    if is_primary and self.freeze_first_verified and self.train_passed:
      # A verified primary already exists: keep it and record this grid as a
      # non-primary candidate (it can still become attempt_2 through voting).
      if grid != self.primary_grid:
        self.frozen_overrides += 1
      is_primary = False
    if is_primary:
      for prev_c in self.candidates:
        prev_c["is_primary"] = False
      self.primary_grid = grid
      self.primary_train_pass = train_pass
      self.primary_train_frac = train_frac
      self.primary_suspect = suspect
      if verified:
        self.train_passed = True
    if train_frac is not None and train_frac > self.best_train_frac:
      self.best_train_frac = train_frac
    if cell_frac > self.best_cell_frac:
      self.best_cell_frac = cell_frac
    self.candidates.append({
        "grid": grid,
        "source": source,
        "train_pass": train_pass,
        "train_frac": train_frac,
        "cell_frac": cell_frac,
        "suspect": suspect,
        "is_primary": is_primary,
        "wants_primary": wants_primary,
        "seq": len(self.candidates),
    })

  def _eval_callable_on_task(
      self, fn: Any
  ) -> tuple[bool, float, float, Grid | None, list[str], str | None]:
    """Evaluates `fn(grid)` on all training inputs and `test_input` with `train`/`test_input` isolated."""
    if hasattr(fn, "__code__"):
      bad_refs = _find_forbidden_globals_in_code_obj(fn.__code__)
      if bad_refs:
        cheat_err = (
            f"RuntimeError: solve(grid) must be a pure function of `grid` and"
            f" cannot reference global {bad_refs} (do not hardcode or look up"
            f" train/test_input inside solve)"
        )
        lines = [
            f"  train[{i}]: FAIL ({cheat_err})"
            for i in range(len(self.task["train"]))
        ]
        return False, 0.0, 0.0, None, lines, cheat_err

    saved_globals: dict[str, Any] = {}
    fn_globals = getattr(fn, "__globals__", None)
    if isinstance(fn_globals, dict):
      for k in _FORBIDDEN_SOLVE_GLOBALS:
        if k in fn_globals:
          saved_globals[k] = fn_globals.pop(k)

    try:
      passed = 0
      cell_scores: list[float] = []
      lines: list[str] = []
      n_train = max(1, len(self.task["train"]))
      for i, pair in enumerate(self.task["train"]):
        inp = copy.deepcopy(pair["input"])
        exp = pair["output"]
        got_norm: Grid | None = None
        err_str: str | None = None
        try:
          try:
            raw = fn(copy.deepcopy(inp))
          except Exception:
            raw = fn(np.array(inp, dtype=int))
          got_norm = normalize_grid(raw)
        except Exception as e:
          err_str = f"{type(e).__name__}: {e}"
        if got_norm == exp:
          passed += 1
          cell_scores.append(1.0)
        elif got_norm is not None and (len(got_norm), len(got_norm[0])) == (
            len(exp),
            len(exp[0]),
        ):
          eh, ew = len(exp), len(exp[0])
          m = sum(
              1
              for r in range(eh)
              for c in range(ew)
              if exp[r][c] == got_norm[r][c]
          )
          cell_scores.append(m / max(1, eh * ew))
        else:
          cell_scores.append(0.0)
        lines.append(format_plain_diff(i, exp, got_norm, err_str))

      test_norm: Grid | None = None
      test_err: str | None = None
      try:
        try:
          raw_t = fn(copy.deepcopy(self.test_input))
        except Exception:
          raw_t = fn(np.array(self.test_input, dtype=int))
        test_norm = normalize_grid(raw_t)
      except Exception as e:
        test_err = f"{type(e).__name__}: {e}"
    finally:
      if isinstance(fn_globals, dict) and saved_globals:
        fn_globals.update(saved_globals)

    train_frac = passed / n_train
    cell_frac = sum(cell_scores) / len(cell_scores) if cell_scores else 0.0
    all_ok = (passed == n_train) and (test_norm is not None)
    return all_ok, train_frac, cell_frac, test_norm, lines, test_err

  def _check_fn(self, fn: Any) -> bool:
    self._check_called_in_turn = True
    self.num_verifications += 1
    all_ok, train_frac, cell_frac, test_norm, lines, test_err = (
        self._eval_callable_on_task(fn)
    )
    n_train = len(self.task["train"])
    passed = int(round(train_frac * n_train))
    self._custom_print(
        f"check: {passed}/{n_train} training pairs passed"
        f" (mean cell accuracy={cell_frac*100:.1f}%)"
    )
    for l in lines:
      self._custom_print(l)
    if test_norm is not None:
      audit_flags = audit_test_grid(self.task, self.test_input, test_norm)
      self._record_candidate(
          grid=test_norm,
          source="check",
          train_pass=(passed == n_train),
          train_frac=train_frac,
          cell_frac=cell_frac,
          suspect=bool(audit_flags),
          is_primary=False,
      )
      if passed == n_train and not audit_flags:
        self._custom_print(
            "  All training pairs passed! Call `submit_program` with `def"
            " solve(grid):` now to record your verified solution."
        )
    elif test_err:
      self._custom_print(f"  test_input error: {test_err}")
    return all_ok

  def _submit_program_fn(self, fn_or_code: Any) -> str:
    self._check_called_in_turn = True
    self.num_verifications += 1
    fn = fn_or_code
    if isinstance(fn_or_code, str):
      exec(fn_or_code, self.ns)  # pylint: disable=exec-used
      fn = self.ns.get("solve")
    if callable(fn):
      self.ever_defined_solve = True
    if not callable(fn):
      msg = "REJECTED: submit_program expects a callable `solve(grid)` or code defining `def solve(grid):`."
      self._custom_print(msg)
      return msg

    all_ok, train_frac, cell_frac, test_norm, lines, test_err = (
        self._eval_callable_on_task(fn)
    )
    n_train = len(self.task["train"])
    passed = int(round(train_frac * n_train))

    if test_norm is not None:
      audit_flags = audit_test_grid(self.task, self.test_input, test_norm)
      suspect = bool(audit_flags)
      self._record_candidate(
          grid=test_norm,
          source="submit_program",
          train_pass=(passed == n_train),
          train_frac=train_frac,
          cell_frac=cell_frac,
          suspect=suspect,
          is_primary=(passed == n_train and not suspect)
          or (self.primary_grid is None),
      )
      if all_ok and not suspect:
        grid_preview = "\n".join(
            "[" + ", ".join(str(v) for v in row) + "]" for row in test_norm
        )
        train_in_cols = {
            v for p in self.task["train"] for r in p["input"] for v in r
        }
        test_in_cols = {v for r in self.test_input for v in r}
        test_out_cols = {v for r in test_norm for v in r}
        novel_in = sorted(test_in_cols - train_in_cols)
        color_note = (
            f"\nNote: test_input contains color(s) {novel_in} not seen in"
            " train inputs."
            if novel_in
            else ""
        )
        msg = (
            f"ACCEPTED: solve(grid) passed all {n_train}/{n_train} training"
            f" pairs. Test output recorded ({len(test_norm)}x{len(test_norm[0])},"
            f" colors={sorted(test_out_cols)}):{color_note}\n{grid_preview}"
        )
        self._custom_print(msg)
        return msg
      if all_ok and suspect:
        msg = (
            f"WARNING (AUDIT): solve(grid) passed {n_train}/{n_train} training"
            " pairs, but test_input output looks suspect:\n  - "
            + "\n  - ".join(audit_flags)
            + "\nCandidate recorded as fallback, but please inspect test_input"
            " and generalize solve(grid) if needed."
        )
        self._custom_print(msg)
        return msg

    extra = f"\n  test_input error: {test_err}" if test_err else ""
    msg = (
        f"REJECTED: solve(grid) passed {passed}/{n_train} training pairs"
        f" (cell accuracy={cell_frac*100:.1f}%):\n"
        + "\n".join(lines)
        + extra
        + "\nFix solve(grid) using the visual diff above and resubmit."
    )
    self._custom_print(msg)
    return msg

  def _submit_answer_fn(self, val_or_code: Any) -> str:
    self._check_called_in_turn = True
    val = val_or_code
    expr_str: str | None = None
    if isinstance(val_or_code, str):
      expr_str = val_or_code.strip()
      try:
        val = eval(expr_str, self.ns)  # pylint: disable=eval-used
      except Exception:
        exec(expr_str, self.ns)  # pylint: disable=exec-used
        val = self.ns.get("answer")
        if val is None and callable(self.ns.get("solve")):
          val = self.ns["solve"](copy.deepcopy(self.test_input))

    norm = normalize_grid(val)
    if norm is None:
      msg = "REJECTED: submit_answer did not produce a valid 2D integer grid (0..9)."
      self._custom_print(msg)
      return msg

    # Check if either `expr_str` or `solve` in namespace can be verified on train
    train_pass: bool | None = None
    train_frac: float | None = None
    cell_frac = 0.0
    if callable(self.ns.get("solve")):
      _, tf, cf, test_from_solve, _, _ = self._eval_callable_on_task(
          self.ns["solve"]
      )
      if test_from_solve == norm:
        train_frac = tf
        cell_frac = cf
        train_pass = tf == 1.0
    if train_pass is None and expr_str:
      tp, tf = self._check_expr_on_train(expr_str)
      if tp is not None:
        train_pass = tp
        train_frac = tf

    audit_flags = audit_test_grid(self.task, self.test_input, norm)
    self._record_candidate(
        grid=norm,
        source="submit_answer",
        train_pass=train_pass,
        train_frac=train_frac,
        cell_frac=cell_frac,
        suspect=bool(audit_flags),
        is_primary=True,
    )
    if train_pass is not False and not audit_flags:
      self.train_passed = True
    msg = f"ACCEPTED: {len(norm)}x{len(norm[0])} answer grid recorded."
    self._custom_print(msg)
    return msg

  def _check_expr_on_train(self, expr: str) -> tuple[bool | None, float | None]:
    """Re-evaluates submitted expression with `test_input` bound to each training input."""
    try:
      tree = ast.parse(expr)
    except SyntaxError:
      return None, None
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if "test_input" not in names:
      return None, None
    passed = 0
    n_train = max(1, len(self.task["train"]))
    for pair in self.task["train"]:
      env = dict(self.ns)
      env["test_input"] = copy.deepcopy(pair["input"])
      try:
        try:
          got = eval(expr, env)  # pylint: disable=eval-used
        except SyntaxError:
          exec(expr, env)  # pylint: disable=exec-used
          got = env.get("answer")
        norm = normalize_grid(got)
      except Exception:
        return False, passed / n_train
      if norm is None:
        return False, passed / n_train
      if norm == pair["output"]:
        passed += 1
    return (passed == n_train), (passed / n_train)

  def execute_turn(
      self, tool_name: str, code: str, timeout_s: float = 5.0
  ) -> str:
    """Executes `run_python`, `submit_program`, or `submit_answer` with a hard thread-local trace timeout."""
    import sys  # pylint: disable=g-import-not-at-top

    self._restore_task_globals()
    self._out_buf = io.StringIO()
    self._check_called_in_turn = False
    prev_solve = self.ns.get("solve")
    err_holder: list[str] = []
    ret_msg: list[str] = []
    auto_check_out: list[str] = []
    deadline = time.monotonic() + timeout_s

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
      del frame, event, arg
      if time.monotonic() > deadline:
        raise TimeoutError(f"Execution timed out after {timeout_s:.0f}s.")
      return _tracer

    def _runner() -> None:
      sys.settrace(_tracer)
      try:
        if tool_name == "submit_program":
          m = self._submit_program_fn(code)
          ret_msg.append(m)
        elif tool_name == "submit_answer":
          m = self._submit_answer_fn(code)
          ret_msg.append(m)
        else:
          # Default `run_python` / `python`: try eval on last expression or exec
          exec(code, self.ns)  # pylint: disable=exec-used
          new_solve = self.ns.get("solve")
          if callable(new_solve):
            self.ever_defined_solve = True
          if (
              self.auto_check
              and callable(new_solve)
              and new_solve is not prev_solve
              and not self._check_called_in_turn
          ):
            # Run the harness verification into a SEPARATE buffer so that the
            # feedback is never swallowed by the 12k-char stdout cap or the
            # 6k-char response truncation of the model's own prints.
            model_buf = self._out_buf
            self._out_buf = io.StringIO()
            try:
              self.num_auto_checks += 1
              self._check_fn(new_solve)
              auto_check_out.append(self._out_buf.getvalue().strip())
            finally:
              self._out_buf = model_buf
      except TimeoutError as te:
        err_holder.append(f"ERROR: {te}")
      except Exception:
        err_holder.append(traceback.format_exc(limit=4))
      finally:
        sys.settrace(None)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s + 0.5)
    if t.is_alive():
      return f"ERROR: Execution timed out after {timeout_s:.0f}s."

    out = self._out_buf.getvalue().strip()
    if not out and ret_msg:
      out = "\n".join(ret_msg).strip()
    if err_holder:
      tb = "\n".join(err_holder).strip()
      out = f"{out}\n{tb}".strip() if out else tb
    if len(out) > 6000:
      out = out[:3000] + "\n...[truncated]...\n" + out[-2000:]
    if auto_check_out and auto_check_out[0]:
      chk = auto_check_out[0]
      if len(chk) > 3500:
        chk = chk[:3500] + "\n...[check output truncated]..."
      header = "[Auto-running check(solve) on all training pairs]:"
      out = f"{out}\n\n{header}\n{chk}" if out else f"{header}\n{chk}"
    return out or "(ok)"

  def harvest_workspace_candidates(
      self, timeout_s: float = 5.0
  ) -> list[dict[str, Any]]:
    """Zero-token salvage of `solve(test_input)`, `answer`, and user 1-arg functions with a hard timeout."""
    import sys  # pylint: disable=g-import-not-at-top

    deadline = time.monotonic() + timeout_s

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
      del frame, event, arg
      if time.monotonic() > deadline:
        raise TimeoutError("Harvest timed out")
      return _tracer

    def _harvest() -> None:
      sys.settrace(_tracer)
      try:
        # 1. Check `solve` in namespace
        if callable(self.ns.get("solve")):
          try:
            _, tf, cf, test_norm, _, _ = self._eval_callable_on_task(
                self.ns["solve"]
            )
            if test_norm is not None:
              audit_flags = audit_test_grid(
                  self.task, self.test_input, test_norm
              )
              self._record_candidate(
                  grid=test_norm,
                  source="workspace:solve",
                  train_pass=(tf == 1.0),
                  train_frac=tf,
                  cell_frac=cf,
                  suspect=bool(audit_flags),
                  is_primary=(self.primary_grid is None),
              )
          except TimeoutError:
            return
          except Exception:
            pass

        # 2. Check explicit grid variables in namespace
        for var_name in ("answer", "test_output", "pred", "output_grid", "out"):
          if var_name in self.ns:
            norm = normalize_grid(self.ns[var_name])
            if norm is not None and norm != self.test_input:
              self._record_candidate(
                  grid=norm,
                  source=f"workspace:{var_name}",
                  train_pass=None,
                  train_frac=None,
                  cell_frac=0.0,
                  suspect=False,
                  is_primary=(self.primary_grid is None),
              )

        # 3. Scan any other 1-arg user functions in namespace (`salvage_answer`)
        skip = {
            "objects",
            "find_objects",
            "bbox",
            "crop",
            "bg_colour",
            "most_common_color",
            "d8",
            "print",
            "check",
            "submit_program",
            "submit_answer",
            "solve",
        }
        for name, obj in list(self.ns.items()):
          if name.startswith("_") or name in skip or not callable(obj):
            continue
          if getattr(obj, "__module__", None) not in (
              None,
              "__main__",
              "builtins",
          ):
            continue
          try:
            _, tf, cf, test_norm, _, _ = self._eval_callable_on_task(obj)
            if test_norm is not None and tf > 0.0:
              audit_flags = audit_test_grid(
                  self.task, self.test_input, test_norm
              )
              self._record_candidate(
                  grid=test_norm,
                  source=f"salvage:{name}",
                  train_pass=(tf == 1.0),
                  train_frac=tf,
                  cell_frac=cf,
                  suspect=bool(audit_flags),
                  is_primary=(self.primary_grid is None),
              )
          except TimeoutError:
            return
          except Exception:
            pass
      finally:
        sys.settrace(None)

    t = threading.Thread(target=_harvest, daemon=True)
    t.start()
    t.join(timeout=timeout_s + 0.5)
    return list(self.candidates)


