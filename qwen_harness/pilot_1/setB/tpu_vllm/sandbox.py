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


def _compact_grid_lines(g: Grid, max_dim: int = 18) -> list[str]:
  h = len(g)
  w = len(g[0]) if h else 0
  if h == 0 or w == 0 or h > max_dim or w > max_dim:
    return []
  return ["      [" + ", ".join(str(v) for v in row) + "]" for row in g]


def compute_code_simplicity(fn_or_code: Any) -> dict[str, Any]:
  """Computes zero-token program simplicity & generality metrics (AST nodes, literals, LOC)."""
  src = ""
  if isinstance(fn_or_code, str):
    src = fn_or_code
  elif hasattr(fn_or_code, "_source_code"):
    src = str(getattr(fn_or_code, "_source_code", ""))
  if not src and hasattr(fn_or_code, "__code__"):
    co = fn_or_code.__code__
    n_consts = len(getattr(co, "co_consts", ()))
    n_bytes = len(getattr(co, "co_code", b""))
    score = 1.0 / (1.0 + 0.005 * n_bytes + 0.02 * n_consts)
    return {"ast_nodes": n_bytes // 2, "loc": 0, "int_literals": n_consts, "simplicity_score": round(score, 4)}
  lines = [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]
  loc = len(lines)
  try:
    tree = ast.parse(src)
    ast_nodes = sum(1 for _ in ast.walk(tree))
    # Count integer constants outside 0..9 (potential hardcoded coordinates/sizes)
    large_ints = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
        and (n.value < 0 or n.value > 9)
    )
  except Exception:
    ast_nodes = loc * 6
    large_ints = 0
  score = 1.0 / (1.0 + 0.004 * ast_nodes + 0.08 * large_ints)
  return {
      "ast_nodes": ast_nodes,
      "loc": loc,
      "large_int_literals": large_ints,
      "simplicity_score": round(score, 4),
  }


def format_plain_diff(
    idx: int,
    expected: Grid,
    got: Grid | None,
    err: str | None,
    inp: Grid | None = None,
    structured_feedback: bool = True,
) -> str:
  if err:
    return f"  train[{idx}]: FAIL (Exception: {err})"
  if got is None:
    return f"  train[{idx}]: FAIL (did not return a valid 2D integer grid 0..9)"
  eh, ew = len(expected), len(expected[0])
  gh, gw = len(got), len(got[0])
  if (eh, ew) != (gh, gw):
    exp_cnt = Counter(v for r in expected for v in r)
    got_cnt = Counter(v for r in got for v in r)
    grid_block = ""
    if structured_feedback:
      exp_lines = _compact_grid_lines(expected, max_dim=16)
      got_lines = _compact_grid_lines(got, max_dim=16)
      if exp_lines:
        grid_block += "\n    expected grid:\n" + "\n".join(exp_lines)
      if got_lines:
        grid_block += "\n    predicted grid:\n" + "\n".join(got_lines)
    return (
        f"  train[{idx}]: FAIL (predicted shape {gh}x{gw} != expected shape {eh}x{ew}; "
        f"got colors={dict(sorted(got_cnt.items()))} vs exp colors={dict(sorted(exp_cnt.items()))})"
        f"{grid_block}"
    )
  mismatches: list[str] = []
  wrong_coords: list[tuple[int, int]] = []
  for r in range(eh):
    for c in range(ew):
      if expected[r][c] != got[r][c]:
        wrong_coords.append((r, c))
        if len(mismatches) < 16:
          mismatches.append(f"({r},{c}):exp={expected[r][c]},got={got[r][c]}")
  wrong = len(wrong_coords)
  if wrong == 0:
    return f"  train[{idx}]: PASS (shape {eh}x{ew})"

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

  full_grids_block = ""
  if structured_feedback and eh <= 16 and ew <= 16:
    exp_lines = _compact_grid_lines(expected, max_dim=16)
    got_lines = _compact_grid_lines(got, max_dim=16)
    if exp_lines and got_lines:
      full_grids_block = (
          "\n    expected grid:\n"
          + "\n".join(exp_lines)
          + "\n    predicted grid:\n"
          + "\n".join(got_lines)
      )

  return (
      f"  train[{idx}]: FAIL (shape {gh}x{gw} vs expected {eh}x{ew}; "
      f"{matched}/{total} cells match = {acc_pct:.1f}%, "
      f"{wrong} mismatched at zero-indexed (r,c) in bbox r={r0}..{r1}, c={c0}..{c1}: {sample_str})"
      f"{delta_line}{vis_block}{full_grids_block}"
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
      structured_feedback: bool = True,
      auto_check_style: str = "legacy",
      snapshot_defs: bool = False,
      compute_invariance: bool = True,
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
    self.structured_feedback = structured_feedback
    # Round 2: "legacy" reuses `_check_fn` output; "concise" emits the compact
    # structured feedback (pass count, dims, <=20 cell diffs, exceptions) and
    # suppresses repeated feedback for unchanged `solve` source.
    self.auto_check_style = auto_check_style
    # Round 2: when True and auto_check is False, every `run_python` cell that
    # (re)defines a callable `solve` is recorded silently (code only, no
    # evaluation, no model-visible output) as a diagnostic-pool candidate.
    self.snapshot_defs = snapshot_defs
    # Round 2: skip the (unpopulated / unused) d8-invariance + simplicity
    # scoring to save CPU.
    self.compute_invariance = compute_invariance
    # Round 2: creation context set by the runner before each `execute_turn`
    # (model turn index, cumulative generated tokens, wall seconds).
    self.ctx: dict[str, Any] = {"turn": 0, "gen_tokens": 0, "t": 0.0}
    # Round 2: every executed cell (for isolated re-evaluation / replay).
    self.cell_history: list[dict[str, Any]] = []
    self._last_auto_hash: str | None = None
    self._last_auto_summary: str = ""
    self._last_eval_detail: list[dict[str, Any]] = []
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
    self.repair_rounds_used = 0
    self.failed_submission_seen = False
    self.repaired_to_pass = False
    self.last_solve_source = ""
    # All distinct candidate grids discovered during the attempt:
    # list of dicts: {grid, source, train_pass, train_frac, cell_frac, suspect, d8_invariance, simplicity_score}
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

  def _eval_invariance_on_test_and_train(
      self, fn: Any, base_test_norm: Grid | None
  ) -> float:
    """Computes zero-token behavioral consistency of `fn` under simple transforms (rot180, fliplr, flipud, transpose)."""
    if base_test_norm is None:
      return 0.0
    saved_globals: dict[str, Any] = {}
    fn_globals = getattr(fn, "__globals__", None)
    if isinstance(fn_globals, dict):
      for k in _FORBIDDEN_SOLVE_GLOBALS:
        if k in fn_globals:
          saved_globals[k] = fn_globals.pop(k)
    try:
      arr_in = np.asarray(self.test_input, dtype=int)
      arr_out = np.asarray(base_test_norm, dtype=int)
      transforms = [
          (lambda a: np.rot90(a, 2), lambda a: np.rot90(a, -2)),
          (np.fliplr, np.fliplr),
          (np.flipud, np.flipud),
          (lambda a: a.T, lambda a: a.T),
      ]
      consistent = 0
      total = 0
      for fwd, inv in transforms:
        total += 1
        t_in = fwd(arr_in).tolist()
        try:
          try:
            raw = fn(copy.deepcopy(t_in))
          except Exception:
            raw = fn(np.array(t_in, dtype=int))
          norm = normalize_grid(raw)
          if norm is not None:
            inv_out = inv(np.asarray(norm, dtype=int)).tolist()
            if inv_out == arr_out.tolist():
              consistent += 1
        except Exception:
          pass
      return round(consistent / max(1, total), 4)
    finally:
      if isinstance(fn_globals, dict) and saved_globals:
        fn_globals.update(saved_globals)

  def _record_candidate(
      self,
      grid: Grid | None,
      source: str,
      train_pass: bool | None,
      train_frac: float | None,
      cell_frac: float = 0.0,
      suspect: bool = False,
      is_primary: bool = False,
      d8_invariance: float = 0.0,
      simplicity_score: float = 0.5,
      code: str | None = None,
      detail: list[dict[str, Any]] | None = None,
      test_err: str | None = None,
  ) -> None:
    verified = train_pass is True and not suspect
    wants_primary = is_primary
    if grid is None:
      # Silent snapshot (round 2 diagnostic pool): no live evaluation, so it
      # can never be primary or count as a verified submission.
      is_primary = False
      wants_primary = False
      verified = False
    if verified:
      self.num_verified_submissions += 1
      if self.failed_submission_seen:
        self.repaired_to_pass = True
      if self.first_verified_grid is None:
        self.first_verified_grid = grid
      self.last_verified_grid = grid
    elif train_pass is False:
      if self.failed_submission_seen:
        self.repair_rounds_used += 1
      self.failed_submission_seen = True
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
    # Round 2 pool assignment: explicit submissions / visible answers are
    # selection-pool candidates; auto-checked REPL programs join the selection
    # pool only when they pass every demonstration; everything else
    # (explicit `check`, silent solve snapshots, workspace salvage) is
    # diagnostic-only.
    if source in ("submit_program", "submit_answer", "visible_json", "no_tools_json"):
      pool = "selection"
    elif source == "auto_check" and train_pass is True:
      pool = "selection"
    else:
      pool = "diagnostic"
    if detail is None:
      detail = list(self._last_eval_detail) if grid is not None else []
    n_demo = len(self.task.get("train", []))
    self.candidates.append({
        "grid": grid,
        "source": source,
        "pool": pool,
        "train_pass": train_pass,
        "train_frac": train_frac,
        "cell_frac": cell_frac,
        "suspect": suspect,
        "is_primary": is_primary,
        "wants_primary": wants_primary,
        "d8_invariance": d8_invariance,
        "simplicity_score": simplicity_score,
        "seq": len(self.candidates),
        "code": code if code is not None else (self.last_solve_source or None),
        "cell_idx": len(self.cell_history),
        "turn": int(self.ctx.get("turn", 0)),
        "gen_tokens_at_creation": int(self.ctx.get("gen_tokens", 0)),
        "created_at_s": round(float(self.ctx.get("t", 0.0)), 1),
        "demo_total": n_demo,
        "demo_pass_count": sum(1 for d in detail if d.get("pass")),
        "demo_shape_ok_count": sum(1 for d in detail if d.get("shape_ok")),
        "demo_mismatch_counts": [d.get("n_mismatch") for d in detail],
        "demo_errors": [d.get("error") for d in detail],
        "test_err": test_err,
    })

  def _eval_callable_detailed(
      self, fn: Any
  ) -> tuple[bool, float, float, Grid | None, list[str], str | None]:
    """Alias kept for readability: evaluation with per-demo detail recorded."""
    return self._eval_callable_on_task(fn)


  def _eval_callable_on_task(
      self, fn: Any
  ) -> tuple[bool, float, float, Grid | None, list[str], str | None]:
    """Evaluates `fn(grid)` on all training inputs and `test_input` with `train`/`test_input` isolated."""
    self._last_eval_detail = []
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
        self._last_eval_detail = [
            {
                "pass": False,
                "shape_ok": False,
                "exp_shape": (len(p["output"]), len(p["output"][0])),
                "got_shape": None,
                "n_mismatch": None,
                "mismatches": [],
                "error": cheat_err,
            }
            for p in self.task["train"]
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
      details: list[dict[str, Any]] = []
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
        eh, ew = len(exp), len(exp[0])
        det: dict[str, Any] = {
            "pass": False,
            "shape_ok": False,
            "exp_shape": (eh, ew),
            "got_shape": None,
            "n_mismatch": None,
            "mismatches": [],
            "error": err_str,
        }
        if got_norm == exp:
          passed += 1
          cell_scores.append(1.0)
          det.update({"pass": True, "shape_ok": True, "got_shape": (eh, ew), "n_mismatch": 0})
        elif got_norm is not None and (len(got_norm), len(got_norm[0])) == (
            eh,
            ew,
        ):
          mism: list[tuple[int, int, int, int]] = []
          n_mism = 0
          for r in range(eh):
            for c in range(ew):
              if exp[r][c] != got_norm[r][c]:
                n_mism += 1
                if len(mism) < 20:
                  mism.append((r, c, exp[r][c], got_norm[r][c]))
          cell_scores.append((eh * ew - n_mism) / max(1, eh * ew))
          det.update(
              shape_ok=True, got_shape=(eh, ew), n_mismatch=n_mism, mismatches=mism
          )
        else:
          cell_scores.append(0.0)
          if got_norm is not None:
            det.update(got_shape=(len(got_norm), len(got_norm[0])))
          elif err_str is None:
            det.update(error="did not return a valid 2D integer grid 0..9")
        details.append(det)
        lines.append(
            format_plain_diff(
                i,
                exp,
                got_norm,
                err_str,
                inp=inp,
                structured_feedback=self.structured_feedback,
            )
        )

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

    self._last_eval_detail = details
    train_frac = passed / n_train
    cell_frac = sum(cell_scores) / len(cell_scores) if cell_scores else 0.0
    all_ok = (passed == n_train) and (test_norm is not None)
    return all_ok, train_frac, cell_frac, test_norm, lines, test_err


  def _score_extras(self, fn: Any, test_norm: Grid | None, passed: int, n_train: int) -> tuple[float, float]:
    """Returns (d8_invariance, simplicity_score); zeros when disabled (round 2)."""
    if not self.compute_invariance:
      return 0.0, 0.5
    simp = compute_code_simplicity(self.last_solve_source or fn)["simplicity_score"]
    d8_inv = (
        self._eval_invariance_on_test_and_train(fn, test_norm)
        if passed == n_train and test_norm is not None
        else 0.0
    )
    return d8_inv, simp

  def _check_fn(self, fn: Any) -> bool:
    self._check_called_in_turn = True
    self.num_verifications += 1
    all_ok, train_frac, cell_frac, test_norm, lines, test_err = (
        self._eval_callable_on_task(fn)
    )
    n_train = len(self.task["train"])
    passed = int(round(train_frac * n_train))
    self._custom_print(
        f"check: {passed}/{n_train} demonstrations passed"
        f" (mean cell accuracy={cell_frac*100:.1f}%)"
    )
    for l in lines:
      self._custom_print(l)
    if test_norm is not None:
      audit_flags = audit_test_grid(self.task, self.test_input, test_norm)
      d8_inv, simp = self._score_extras(fn, test_norm, passed, n_train)
      self._record_candidate(
          grid=test_norm,
          source="check",
          train_pass=(passed == n_train),
          train_frac=train_frac,
          cell_frac=cell_frac,
          suspect=bool(audit_flags),
          is_primary=False,
          d8_invariance=d8_inv,
          simplicity_score=simp,
          test_err=test_err,
      )
      if passed == n_train and not audit_flags:
        self._custom_print(
            "  All training pairs passed! Call `submit_program` with `def"
            " solve(grid):` now to record your verified solution."
        )
      elif passed < n_train:
        self._custom_print(
            "  REPAIR NEEDED: Reconsider both your implementation and your"
            " underlying rule using all demonstrations above. Revise `def"
            " solve(grid):` and re-check against all demonstrations."
        )
    elif test_err:
      self._custom_print(f"  test_input error: {test_err}")
      if self.snapshot_defs:
        self._record_candidate(
            grid=None, source="check", train_pass=(passed == n_train),
            train_frac=train_frac, cell_frac=cell_frac, test_err=test_err,
            detail=list(self._last_eval_detail),
        )
    return all_ok

  def _concise_auto_check(self, fn: Any, code: str) -> str:
    """Round-2 EXP-2 feedback: compact structured demonstration results.

    Returns the text to append to the tool response (never dumps full grids,
    and does not repeat identical feedback for unchanged `solve` source).
    """
    import hashlib  # pylint: disable=g-import-not-at-top

    src = self.last_solve_source or code or ""
    h = hashlib.sha1(src.encode("utf-8", "ignore")).hexdigest()
    self.num_verifications += 1
    all_ok, train_frac, cell_frac, test_norm, _, test_err = (
        self._eval_callable_on_task(fn)
    )
    del all_ok
    n_train = len(self.task["train"])
    passed = int(round(train_frac * n_train))
    details = list(self._last_eval_detail)
    suspect = False
    if test_norm is not None:
      audit_flags = audit_test_grid(self.task, self.test_input, test_norm)
      suspect = bool(audit_flags)
    self._record_candidate(
        grid=test_norm,
        source="auto_check",
        train_pass=(passed == n_train),
        train_frac=train_frac,
        cell_frac=cell_frac,
        suspect=suspect,
        is_primary=False,
        code=src,
        detail=details,
        test_err=test_err,
    )
    if h == self._last_auto_hash:
      return (
          f"[auto-check] `solve` source unchanged since the last check;"
          f" feedback not repeated ({passed}/{n_train} demonstrations passed)."
      )
    self._last_auto_hash = h
    out: list[str] = [
        f"[auto-check] solve() vs {n_train} demonstrations:"
        f" {passed}/{n_train} passed exactly."
    ]
    for i, d in enumerate(details):
      if d.get("pass"):
        continue
      eh, ew = d["exp_shape"]
      if d.get("error") and d.get("got_shape") is None:
        err = str(d["error"]).strip().splitlines()[-1][:300]
        out.append(f"  demo {i + 1}: FAIL - exception: {err}")
      elif not d.get("shape_ok"):
        gs = d.get("got_shape")
        got_txt = f"{gs[0]}x{gs[1]}" if gs else "no valid grid"
        out.append(
            f"  demo {i + 1}: FAIL - output {got_txt}, expected {eh}x{ew}"
            " (wrong output dimensions)"
        )
      else:
        n_mism = int(d.get("n_mismatch") or 0)
        cells = ", ".join(
            f"(r{r},c{c}) exp {e} got {g}" for r, c, e, g in d.get("mismatches", [])
        )
        trunc = (
            f" [showing first {len(d.get('mismatches', []))} of {n_mism} mismatches]"
            if n_mism > len(d.get("mismatches", []))
            else ""
        )
        out.append(
            f"  demo {i + 1}: FAIL - shape OK ({eh}x{ew});"
            f" {n_mism}/{eh * ew} cells differ: {cells}{trunc}"
        )
    if test_err:
      out.append(f"  test_input: exception: {str(test_err)[:300]}")
    if passed == n_train and not suspect:
      out.append(
          "  All training pairs passed! Call `submit_program` with `def"
          " solve(grid):` now to record your verified solution."
      )
    elif passed == n_train and suspect:
      out.append(
          "  All demonstrations passed, but the test output looks suspect"
          " (shape/color heuristics); inspect `test_input` before submitting."
      )
    return "\n".join(out)


  def _submit_program_fn(self, fn_or_code: Any) -> str:
    self._check_called_in_turn = True
    self.num_verifications += 1
    fn = fn_or_code
    if isinstance(fn_or_code, str):
      self.last_solve_source = fn_or_code
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

    if test_norm is None and self.snapshot_defs:
      # Preserve the submitted program even though solve(test_input) failed.
      self._record_candidate(
          grid=None, source="submit_program", train_pass=(passed == n_train),
          train_frac=train_frac, cell_frac=cell_frac, test_err=test_err,
          detail=list(self._last_eval_detail),
      )

    if test_norm is not None:
      audit_flags = audit_test_grid(self.task, self.test_input, test_norm)
      suspect = bool(audit_flags)
      d8_inv, simp = self._score_extras(fn, test_norm, passed, n_train)
      self._record_candidate(
          grid=test_norm,
          source="submit_program",
          train_pass=(passed == n_train),
          train_frac=train_frac,
          cell_frac=cell_frac,
          suspect=suspect,
          is_primary=(passed == n_train and not suspect)
          or (self.primary_grid is None),
          d8_invariance=d8_inv,
          simplicity_score=simp,
          test_err=test_err,
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
        f"REJECTED: solve(grid) passed {passed}/{n_train} demonstrations"
        f" (cell accuracy={cell_frac*100:.1f}%):\n"
        + "\n".join(lines)
        + extra
        + "\nREPAIR INSTRUCTION: Reconsider both your implementation and your"
        " underlying rule using all demonstrations above. Check whether your"
        " rule explains every training input->output transition, then revise"
        " `def solve(grid):` and resubmit."
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
    from_solve = False
    if callable(self.ns.get("solve")):
      _, tf, cf, test_from_solve, _, _ = self._eval_callable_on_task(
          self.ns["solve"]
      )
      if test_from_solve == norm:
        train_frac = tf
        cell_frac = cf
        train_pass = tf == 1.0
        from_solve = True
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
        code=(self.last_solve_source if from_solve else ""),
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
    concise_out: list[str] = []
    deadline = time.monotonic() + timeout_s
    cell_rec: dict[str, Any] = {
        "idx": len(self.cell_history),
        "tool": tool_name,
        "code": code,
        "turn": int(self.ctx.get("turn", 0)),
        "gen_tokens": int(self.ctx.get("gen_tokens", 0)),
        "ok": True,
        "defined_solve": False,
    }

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
      del frame, event, arg
      if time.monotonic() > deadline:
        raise TimeoutError(f"Execution timed out after {timeout_s:.0f}s.")
      return _tracer

    n_cands_before = len(self.candidates)

    def _record_failed_submission(reason: str) -> None:
      # "Preserve every submitted program": a submit_program whose evaluation
      # timed out / crashed still gets a (grid-less) candidate record.
      if tool_name == "submit_program" and len(self.candidates) == n_cands_before:
        try:
          self._record_candidate(
              grid=None, source="submit_program", train_pass=None,
              train_frac=None, cell_frac=0.0, code=code, detail=[],
              test_err=reason,
          )
        except Exception:  # pylint: disable=broad-exception-caught
          pass

    def _runner() -> None:
      sys.settrace(_tracer)
      try:
        if tool_name == "submit_program":
          m = self._submit_program_fn(code)
          ret_msg.append(m)
          cell_rec["defined_solve"] = callable(self.ns.get("solve"))
        elif tool_name == "submit_answer":
          m = self._submit_answer_fn(code)
          ret_msg.append(m)
        else:
          # Default `run_python` / `python`: try eval on last expression or exec
          if "def solve" in code:
            self.last_solve_source = code
          exec(code, self.ns)  # pylint: disable=exec-used
          new_solve = self.ns.get("solve")
          if callable(new_solve):
            self.ever_defined_solve = True
          changed = callable(new_solve) and new_solve is not prev_solve
          cell_rec["defined_solve"] = bool(changed)
          if changed and self.auto_check and not self._check_called_in_turn:
            # Run the harness verification into a SEPARATE buffer so that the
            # feedback is never swallowed by the 12k-char stdout cap or the
            # 6k-char response truncation of the model's own prints.
            model_buf = self._out_buf
            self._out_buf = io.StringIO()
            try:
              self.num_auto_checks += 1
              if self.auto_check_style == "concise":
                concise_out.append(self._concise_auto_check(new_solve, code))
              else:
                self._check_fn(new_solve)
                auto_check_out.append(self._out_buf.getvalue().strip())
            finally:
              self._out_buf = model_buf
          elif changed and not self.auto_check and self.snapshot_defs:
            # Round 2: silent diagnostic snapshot (code only; evaluated later in
            # isolation). No model-visible output, no live evaluation.
            self._record_candidate(
                grid=None,
                source="solve_def",
                train_pass=None,
                train_frac=None,
                cell_frac=0.0,
                code=code,
                detail=[],
            )
      except TimeoutError as te:
        sys.settrace(None)
        cell_rec["ok"] = False
        err_holder.append(f"ERROR: {te}")
        _record_failed_submission(f"TimeoutError: {te}")
      except Exception:
        sys.settrace(None)
        cell_rec["ok"] = False
        tb = traceback.format_exc(limit=4)
        err_holder.append(tb)
        _record_failed_submission(tb.strip().splitlines()[-1][:300] if tb.strip() else "error")
      finally:
        sys.settrace(None)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout_s + 0.5)
    if t.is_alive():
      cell_rec["ok"] = False
      cell_rec["timed_out"] = True
      self.cell_history.append(cell_rec)
      _record_failed_submission(f"TimeoutError: execution timed out after {timeout_s:.0f}s")
      return f"ERROR: Execution timed out after {timeout_s:.0f}s."
    self.cell_history.append(cell_rec)


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
    if concise_out and concise_out[0]:
      chk = concise_out[0]
      if len(chk) > 3500:
        chk = chk[:3500] + "\n...[auto-check output truncated]..."
      out = f"{out}\n\n{chk}" if out else chk
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
              simp = compute_code_simplicity(
                  self.last_solve_source or self.ns["solve"]
              )["simplicity_score"]
              d8_inv = (
                  self._eval_invariance_on_test_and_train(
                      self.ns["solve"], test_norm
                  )
                  if tf == 1.0
                  else 0.0
              )
              self._record_candidate(
                  grid=test_norm,
                  source="workspace:solve",
                  train_pass=(tf == 1.0),
                  train_frac=tf,
                  cell_frac=cf,
                  suspect=bool(audit_flags),
                  is_primary=(self.primary_grid is None),
                  d8_invariance=d8_inv,
                  simplicity_score=simp,
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


# ==============================================================================
# Round 2: isolated candidate re-evaluation (authoritative for selection)
# ==============================================================================


def _noop(*args: Any, **kwargs: Any) -> str:
  del args, kwargs
  return ""


def extract_helper_source(prior_cells: list[str]) -> str:
  """Collects top-level import / def / class / literal-assign statements.

  Expression statements, loops, calls and any assignment whose value is not a
  pure literal are skipped, so the replay reconstructs helper definitions but
  no leftover mutable state. `def solve` definitions are skipped as well (the
  candidate cell provides its own `solve`).
  """
  pieces: list[str] = []
  for cell in prior_cells:
    try:
      tree = ast.parse(cell)
    except Exception:
      continue
    for node in tree.body:
      keep = False
      if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
        keep = True
      elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        keep = node.name != "solve"
      elif isinstance(node, ast.Assign):
        bad = any(
            isinstance(n, (ast.Call, ast.Name, ast.Attribute, ast.Subscript))
            for n in ast.walk(node.value)
        )
        keep = not bad
      if keep:
        try:
          pieces.append(ast.unparse(node))
        except Exception:
          pass
  return "\n\n".join(pieces)


def run_traced(fn: Any, timeout_s: float) -> tuple[bool, Any]:
  """Runs `fn()` in a daemon thread with a trace-based deadline.

  Returns `(ok, result_or_error_string)`.
  """
  import sys  # pylint: disable=g-import-not-at-top

  deadline = time.monotonic() + timeout_s
  box: list[Any] = []
  err: list[str] = []

  def _tracer(frame: Any, event: str, arg: Any) -> Any:
    del frame, event, arg
    if time.monotonic() > deadline:
      raise TimeoutError(f"timed out after {timeout_s:.0f}s")
    return _tracer

  def _runner() -> None:
    sys.settrace(_tracer)
    try:
      box.append(fn())
    except BaseException as e:  # pylint: disable=broad-exception-caught
      err.append(f"{type(e).__name__}: {e}")
    finally:
      sys.settrace(None)

  t = threading.Thread(target=_runner, daemon=True)
  t.start()
  t.join(timeout=timeout_s + 0.5)
  if t.is_alive():
    return False, f"TimeoutError: timed out after {timeout_s:.0f}s"
  if err:
    return False, err[0]
  return True, (box[0] if box else None)


def _is_name_error(s: str | None) -> bool:
  return bool(s) and str(s).startswith("NameError")


def evaluate_candidate_isolated(
    code: str | None,
    prior_cells: list[str],
    task: dict[str, Any],
    test_input: Grid,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
  """Re-evaluates a candidate program in a fresh namespace.

  Stage A: the candidate cell alone (preloaded helpers only). Stage B (only if
  Stage A hits a NameError): replay helper definitions from earlier cells
  first. Returns a dict with `mode` in {self_contained, with_helpers, failed,
  no_code} plus the demonstration / test results.
  """
  out: dict[str, Any] = {
      "mode": "no_code",
      "train_pass": None,
      "train_frac": None,
      "cell_frac": 0.0,
      "grid": None,
      "test_err": None,
      "demo_pass_count": 0,
      "demo_shape_ok_count": 0,
      "demo_mismatch_counts": [],
      "demo_errors": [],
      "error": None,
  }
  if not code or "solve" not in code:
    return out

  def _attempt(helper_src: str | None) -> dict[str, Any]:
    sb = ArcPythonSandbox(
        task=task,
        test_input=test_input,
        auto_check=False,
        snapshot_defs=False,
        compute_invariance=False,
    )
    sb.ns["check"] = _noop
    sb.ns["submit_program"] = _noop
    sb.ns["submit_answer"] = _noop
    sb.ns["print"] = _noop
    res: dict[str, Any] = dict(out)
    if helper_src:
      def _exec_helpers() -> None:
        for stmt in helper_src.split("\n\n"):
          try:
            exec(stmt, sb.ns)  # pylint: disable=exec-used
          except Exception:
            pass
      run_traced(_exec_helpers, timeout_s)
    ok, err = run_traced(lambda: exec(code, sb.ns), timeout_s)  # pylint: disable=exec-used
    if not ok:
      res["error"] = str(err)
    fn = sb.ns.get("solve")
    if not callable(fn):
      res["error"] = res["error"] or "solve not defined"
      return res
    ok2, ev = run_traced(lambda: sb._eval_callable_on_task(fn), timeout_s)  # pylint: disable=protected-access
    if not ok2:
      res["error"] = str(ev)
      return res
    _, tf, cf, test_norm, _, test_err = ev
    n = max(1, len(task["train"]))
    det = list(sb._last_eval_detail)  # pylint: disable=protected-access
    res.update(
        train_pass=(int(round(tf * n)) == n),
        train_frac=tf,
        cell_frac=cf,
        grid=test_norm,
        test_err=test_err,
        demo_pass_count=sum(1 for d in det if d.get("pass")),
        demo_shape_ok_count=sum(1 for d in det if d.get("shape_ok")),
        demo_mismatch_counts=[d.get("n_mismatch") for d in det],
        demo_errors=[d.get("error") for d in det],
    )
    return res

  res_a = _attempt(None)
  needs_helpers = _is_name_error(res_a.get("error")) or any(
      _is_name_error(e) for e in res_a.get("demo_errors", [])
  ) or _is_name_error(res_a.get("test_err"))
  if not needs_helpers:
    res_a["mode"] = "self_contained" if res_a.get("grid") is not None or res_a.get("train_frac") is not None else "failed"
    return res_a
  helper_src = extract_helper_source(prior_cells)
  if not helper_src:
    res_a["mode"] = "failed"
    return res_a
  res_b = _attempt(helper_src)
  res_b["mode"] = "with_helpers" if (res_b.get("grid") is not None or res_b.get("train_frac") is not None) else "failed"
  return res_b


