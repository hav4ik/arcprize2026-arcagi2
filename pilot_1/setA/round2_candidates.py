"""Round-2 candidate pools, frozen selector, view transforms and evaluator.

Everything in this module is model-free and runs on CPU; it is unit-tested in
`round2_candidates_test.py`. The ranking tuple below is pre-registered and
must not be changed after results are observed.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

from google3.experimental.users.arazdai.kaggle.arc_agi_2.tpu_vllm import sandbox

Grid = list[list[int]]

CONFIG_VERSION = "r2-v1"

# Pre-registered ranking tuple (descending sort). Documented in round2_plan.md.
RANKING_TUPLE = (
    "-best_tier",
    "verified_votes",
    "primary_votes",
    "total_votes",
    "best_train_frac",
    "best_cell_frac",
    "has_primary",
    "-min_gen_tokens_at_creation",
    "-min_turn",
    "-min_attempt_idx",
)

# ==============================================================================
# 1. View transforms (EXP-5)
# ==============================================================================

VIEW_NAMES = ("identity", "rot90", "fliplr")


def apply_view(g: Grid, view: str) -> Grid:
  """Forward transform of a grid into the solver's coordinate frame."""
  arr = np.asarray(g, dtype=int)
  if view == "identity":
    return arr.tolist()
  if view == "rot90":
    return np.rot90(arr, 1).tolist()
  if view == "fliplr":
    return np.fliplr(arr).tolist()
  raise ValueError(f"unknown view {view!r}")


def invert_view(g: Grid | None, view: str) -> Grid | None:
  """Inverse transform back to the original frame (works for any HxW)."""
  if g is None:
    return None
  arr = np.asarray(g, dtype=int)
  if arr.ndim != 2 or arr.size == 0:
    return None
  if view == "identity":
    return arr.tolist()
  if view == "rot90":
    return np.rot90(arr, -1).tolist()
  if view == "fliplr":
    return np.fliplr(arr).tolist()
  raise ValueError(f"unknown view {view!r}")


def transform_task_view(
    solver_task: dict[str, Any], test_input: Grid, view: str
) -> tuple[dict[str, Any], Grid]:
  """Transforms every demonstration input/output and the test input(s)."""
  t_task = {
      "train": [
          {"input": apply_view(p["input"], view), "output": apply_view(p["output"], view)}
          for p in solver_task["train"]
      ],
      "test": [{"input": apply_view(p["input"], view)} for p in solver_task.get("test", [])],
  }
  return t_task, apply_view(test_input, view)


def strip_test_outputs(task: dict[str, Any]) -> dict[str, Any]:
  """Solver-facing task: demonstrations + test inputs only (no test outputs)."""
  return {
      "train": copy.deepcopy(task["train"]),
      "test": [{"input": copy.deepcopy(p["input"])} for p in task.get("test", [])],
  }


# ==============================================================================
# 2. Seeds / ids
# ==============================================================================


def stable_seed(task_id: str, pair_idx: int, attempt_idx: int) -> int:
  h = hashlib.sha256(f"{task_id}:{pair_idx}:{attempt_idx}".encode()).hexdigest()
  return int(h[:8], 16) % (2**31 - 1)


# ==============================================================================
# 3. Candidate finalization (isolated re-evaluation + restoration)
# ==============================================================================


def finalize_attempt_candidates(
    raw_cands: list[dict[str, Any]],
    cell_history: list[dict[str, Any]],
    task_view: dict[str, Any],
    test_input_view: Grid,
    view: str,
    attempt_idx: int,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
  """Re-evaluates every candidate in isolation and restores grids to the original frame.

  The isolated result is authoritative for `train_pass`/`grid` when the
  candidate has code; grid-only candidates (visible JSON / submit_answer
  expressions) keep their live values with `isolation_mode='n/a'`.
  """
  out: list[dict[str, Any]] = []
  cell_codes = [c.get("code", "") for c in cell_history]
  for c in raw_cands:
    cand = dict(c)
    cand["attempt_idx"] = attempt_idx
    cand["view"] = view
    cand["grid_view"] = c.get("grid")
    cand["live"] = {
        "train_pass": c.get("train_pass"),
        "train_frac": c.get("train_frac"),
        "cell_frac": c.get("cell_frac"),
        "demo_pass_count": c.get("demo_pass_count"),
        "demo_shape_ok_count": c.get("demo_shape_ok_count"),
        "demo_mismatch_counts": c.get("demo_mismatch_counts"),
        "test_err": c.get("test_err"),
    }
    code = c.get("code")
    if code and "solve" in code:
      prior = cell_codes[: max(0, int(c.get("cell_idx", len(cell_codes))))]
      iso = sandbox.evaluate_candidate_isolated(
          code, prior, task_view, test_input_view, timeout_s=timeout_s
      )
      cand["isolated"] = {k: v for k, v in iso.items() if k != "grid"}
      cand["isolation_mode"] = iso["mode"]
      if iso["mode"] in ("self_contained", "with_helpers"):
        iso_grid = iso.get("grid")
        cand["live_isolated_grid_match"] = (iso_grid == c.get("grid")) if (iso_grid is not None and c.get("grid") is not None) else None
        cand["grid_view"] = iso_grid
        cand["train_pass"] = iso.get("train_pass")
        cand["train_frac"] = iso.get("train_frac")
        cand["cell_frac"] = iso.get("cell_frac") or 0.0
        cand["demo_pass_count"] = iso.get("demo_pass_count")
        cand["demo_shape_ok_count"] = iso.get("demo_shape_ok_count")
        cand["demo_mismatch_counts"] = iso.get("demo_mismatch_counts")
        cand["test_err"] = iso.get("test_err")
        if iso_grid is not None:
          flags = sandbox.audit_test_grid(task_view, test_input_view, iso_grid)
          cand["suspect"] = bool(flags)
          cand["audit_flags"] = flags
      else:
        # Isolated execution failed outright: keep live values but mark it.
        cand["isolation_failed"] = True
        cand["live_isolated_grid_match"] = None
    else:
      cand["isolation_mode"] = "n/a"
      cand["isolated"] = None
    gv = cand.get("grid_view")
    cand["grid"] = invert_view(gv, view) if gv is not None else None
    # A verified primary must still be verified after isolation; otherwise it is
    # demoted to a partial candidate (it keeps `is_primary` for voting purposes
    # but at its new tier).
    cand["verified"] = bool(cand.get("train_pass") is True and not cand.get("suspect"))
    # Re-derive pool after isolation (auto-check candidates must still pass).
    src = cand.get("source", "")
    if src in ("submit_program", "submit_answer", "visible_json", "no_tools_json"):
      cand["pool"] = "selection"
    elif src == "auto_check" and cand.get("train_pass") is True:
      cand["pool"] = "selection"
    else:
      cand["pool"] = "diagnostic"
    out.append(cand)
  return out


# ==============================================================================
# 4. Frozen selector
# ==============================================================================


def _tier(c: dict[str, Any]) -> int:
  tp = c.get("train_pass")
  suspect = bool(c.get("suspect", False))
  is_prim = bool(c.get("is_primary", False))
  if tp is True and not suspect:
    return 0
  if (tp is True and suspect) or (is_prim and tp is None):
    return 1
  return 2


def rank_candidates(
    cands: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Groups candidates by identical restored grid and ranks groups.

  `cands` must carry `attempt_idx`, `grid`, `train_pass`, `train_frac`,
  `cell_frac`, `suspect`, `is_primary`, `gen_tokens_at_creation`, `turn`.
  Returns the ranked list of grid-group feature dicts (best first).
  """
  by_key: dict[str, dict[str, Any]] = {}
  for c in cands:
    g = c.get("grid")
    if g is None:
      continue
    k = sandbox.grid_key(g)
    if not k:
      continue
    att = int(c.get("attempt_idx", 0))
    tier = _tier(c)
    tf = float(c.get("train_frac") or 0.0)
    cf = float(c.get("cell_frac") or 0.0)
    is_prim = bool(c.get("is_primary", False))
    toks = int(c.get("gen_tokens_at_creation") or 0)
    turn = int(c.get("turn") or 0)
    e = by_key.get(k)
    if e is None:
      e = {
          "grid": g,
          "key": k,
          "best_tier": tier,
          "verified_attempts": set(),
          "primary_attempts": set(),
          "attempts": set(),
          "best_train_frac": tf,
          "best_cell_frac": cf,
          "has_primary": is_prim,
          "min_gen_tokens_at_creation": toks,
          "min_turn": turn,
          "min_attempt_idx": att,
          "sources": [],
          "n_candidates": 0,
      }
      by_key[k] = e
    e["best_tier"] = min(e["best_tier"], tier)
    e["best_train_frac"] = max(e["best_train_frac"], tf)
    e["best_cell_frac"] = max(e["best_cell_frac"], cf)
    e["has_primary"] = e["has_primary"] or is_prim
    e["min_gen_tokens_at_creation"] = min(e["min_gen_tokens_at_creation"], toks)
    e["min_turn"] = min(e["min_turn"], turn)
    e["min_attempt_idx"] = min(e["min_attempt_idx"], att)
    e["attempts"].add(att)
    if tier == 0:
      e["verified_attempts"].add(att)
    if is_prim:
      e["primary_attempts"].add(att)
    e["sources"].append(f"a{att}:{c.get('source', '')}")
    e["n_candidates"] += 1

  groups = list(by_key.values())
  for e in groups:
    e["verified_votes"] = len(e["verified_attempts"])
    e["primary_votes"] = len(e["primary_attempts"])
    e["total_votes"] = len(e["attempts"])
  groups.sort(
      key=lambda e: (
          -e["best_tier"],
          e["verified_votes"],
          e["primary_votes"],
          e["total_votes"],
          e["best_train_frac"],
          e["best_cell_frac"],
          1 if e["has_primary"] else 0,
          -e["min_gen_tokens_at_creation"],
          -e["min_turn"],
          -e["min_attempt_idx"],
      ),
      reverse=True,
  )
  for rank, e in enumerate(groups):
    e["rank"] = rank
    e["verified_attempts"] = sorted(e["verified_attempts"])
    e["primary_attempts"] = sorted(e["primary_attempts"])
    e["attempts"] = sorted(e["attempts"])
  return groups


def select_top_two(
    cands: list[dict[str, Any]], fallback_grid: Grid
) -> tuple[Grid, Grid, list[dict[str, Any]]]:
  """Returns (top1, top2, ranked_groups). top2 == top1 when only one grid exists."""
  ranked = rank_candidates(cands)
  if not ranked:
    fb = copy.deepcopy(fallback_grid)
    return fb, fb, []
  top1 = ranked[0]["grid"]
  top2 = ranked[1]["grid"] if len(ranked) >= 2 else top1
  return top1, top2, ranked


# ==============================================================================
# 5. Evaluator (runs after solving; the only place that sees test outputs)
# ==============================================================================


def score_pair(
    attempt_rows: list[dict[str, Any]],
    fallback_grid: Grid,
    target: Grid,
) -> dict[str, Any]:
  """Builds pools from finalized attempt rows, selects top-two and scores."""
  sel_cands: list[dict[str, Any]] = []
  all_cands: list[dict[str, Any]] = []
  for row in attempt_rows:
    for c in row.get("candidates", []):
      if c.get("grid") is None:
        continue
      all_cands.append(c)
      if c.get("pool") == "selection":
        sel_cands.append(c)
  top1, top2, ranked = select_top_two(sel_cands, fallback_grid)
  top1_exact = top1 == target
  top2_exact = top1_exact or (top2 == target)
  sel_oracle = any(c["grid"] == target for c in sel_cands)
  diag_oracle = any(c["grid"] == target for c in all_cands)
  # Per-candidate exactness (added after solving).
  for row in attempt_rows:
    for c in row.get("candidates", []):
      c["exact"] = (c.get("grid") == target) if c.get("grid") is not None else None
  th, tw = len(target), len(target[0])
  shape_ok = len(top1) == th and len(top1[0]) == tw
  cell_acc = 0.0
  if shape_ok:
    cell_acc = sum(
        1 for r in range(th) for c in range(tw) if top1[r][c] == target[r][c]
    ) / max(1, th * tw)
  ranked_compact = [
      {
          k: v
          for k, v in e.items()
          if k != "grid"
      }
      | {"exact": e["grid"] == target}
      for e in ranked[:8]
  ]
  return {
      "top1": top1,
      "top2": top2,
      "top1_exact": top1_exact,
      "top2_exact": top2_exact,
      "top1_shape_ok": shape_ok,
      "top1_cell_acc": round(cell_acc, 4),
      "selection_oracle": sel_oracle,
      "diagnostic_oracle": diag_oracle,
      "n_selection_candidates": len(sel_cands),
      "n_all_candidates": len(all_cands),
      "n_distinct_selection_grids": len(ranked),
      "ranking": ranked_compact,
      "ranking_tuple": list(RANKING_TUPLE),
  }
