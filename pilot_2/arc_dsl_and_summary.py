"""Standalone ARC-AGI-2 DSL helpers and deterministic perception summary for pilot_2.

Written from scratch for pilot_2 with zero dependencies on pilot_1.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import ndimage


ARC_HELPERS_SOURCE = r'''
def bg_colour(grid):
    """Return the most frequent colour in `grid` (standard ARC background heuristic)."""
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def bbox(mask_or_coords):
    """Return (r0, c0, r1, c1) inclusive bounding box of a boolean mask or (N, 2) coords."""
    arr = np.asarray(mask_or_coords)
    if arr.ndim == 2 and arr.dtype == bool:
        rs, cs = np.where(arr)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        rs, cs = arr[:, 0], arr[:, 1]
    else:
        rs, cs = np.where(arr != 0)
    if len(rs) == 0:
        return (0, 0, -1, -1)
    return (int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max()))


def crop(grid, box=None, bg=0):
    """Crop `grid` to `box=(r0, c0, r1, c1)` inclusive, or to non-`bg` bounding box."""
    g = np.asarray(grid)
    if box is None:
        box = bbox(g != bg)
    r0, c0, r1, c1 = box
    if r1 < r0 or c1 < c0:
        return g[:0, :0].copy()
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def d8(patch):
    """Return all 8 dihedral (rotation + reflection) transforms of a 2D array."""
    p = np.asarray(patch)
    out = []
    for k in range(4):
        r = np.rot90(p, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def match_d8(a, b, ignore_colour=False):
    """Return True if 2D patch `a` matches `b` under any of the 8 dihedral transforms."""
    pa = np.asarray(a) != 0 if ignore_colour else np.asarray(a)
    pb = np.asarray(b) != 0 if ignore_colour else np.asarray(b)
    for variant in d8(pa):
        if variant.shape == pb.shape and np.array_equal(variant, pb):
            return True
    return False


def objects(grid, bg=None, connectivity=8, multicolour=False):
    """Extract connected components from `grid`, sorted top-to-bottom, left-to-right."""
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = bg_colour(g)
    struct = (
        np.ones((3, 3), dtype=int)
        if connectivity == 8
        else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    )
    comps = []
    if multicolour:
        fg_mask = g != bg
        labeled, num = ndimage.label(fg_mask, structure=struct)
        for idx in range(1, num + 1):
            m = labeled == idx
            r0, c0, r1, c1 = bbox(m)
            sub = np.where(m[r0 : r1 + 1, c0 : c1 + 1], g[r0 : r1 + 1, c0 : c1 + 1], bg)
            coords = np.argwhere(m)
            comps.append({
                "colour": int(g[m][0]) if len(np.unique(g[m])) == 1 else -1,
                "size": int(m.sum()),
                "bbox": (r0, c0, r1, c1),
                "patch": sub,
                "mask": m,
                "coords": coords,
            })
    else:
        for col in sorted(int(c) for c in np.unique(g) if int(c) != bg):
            col_mask = g == col
            labeled, num = ndimage.label(col_mask, structure=struct)
            for idx in range(1, num + 1):
                m = labeled == idx
                r0, c0, r1, c1 = bbox(m)
                sub = np.where(m[r0 : r1 + 1, c0 : c1 + 1], col, bg)
                coords = np.argwhere(m)
                comps.append({
                    "colour": col,
                    "size": int(m.sum()),
                    "bbox": (r0, c0, r1, c1),
                    "patch": sub,
                    "mask": m,
                    "coords": coords,
                })
    comps.sort(key=lambda o: (o["bbox"][0], o["bbox"][1], -o["size"]))
    return comps


def panels(grid):
    """Detect full-row or full-column uniform separator lines and split `grid` into sub-panels."""
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = bg_colour(g)
    sep_rows = [
        r for r in range(H) if len(np.unique(g[r, :])) == 1 and int(g[r, 0]) != bg
    ]
    sep_cols = [
        c for c in range(W) if len(np.unique(g[:, c])) == 1 and int(g[0, c]) != bg
    ]

    def _ranges(length, seps):
        bounds = [-1] + list(seps) + [length]
        return [
            (bounds[i] + 1, bounds[i + 1])
            for i in range(len(bounds) - 1)
            if bounds[i] + 1 < bounds[i + 1]
        ]

    r_ranges = _ranges(H, sep_rows)
    c_ranges = _ranges(W, sep_cols)
    return [
        [g[r0:r1, c0:c1].copy() for (c0, c1) in c_ranges]
        for (r0, r1) in r_ranges
    ]


def stamp(canvas, patch, r0, c0, transparent=None):
    """Paste `patch` onto a copy of `canvas` at top-left `(r0, c0)`, clipping to bounds."""
    out = np.asarray(canvas, dtype=int).copy()
    p = np.asarray(patch, dtype=int)
    H, W = out.shape
    ph, pw = p.shape
    for dr in range(ph):
        rr = r0 + dr
        if rr < 0 or rr >= H:
            continue
        for dc in range(pw):
            cc = c0 + dc
            if cc < 0 or cc >= W:
                continue
            val = int(p[dr, dc])
            if transparent is not None and val == transparent:
                continue
            out[rr, cc] = val
    return out


def ray(grid, r, c, dr, dc, colour=None, stop_at=None):
    """Cast a ray from `(r+dr, c+dc)` in direction `(dr, dc)` until boundary or `stop_at` colour."""
    out = np.asarray(grid, dtype=int).copy()
    H, W = out.shape
    stop_set = set() if stop_at is None else (
        {int(stop_at)} if isinstance(stop_at, (int, np.integer)) else {int(x) for x in stop_at}
    )
    rr, cc = r + dr, c + dc
    coords = []
    while 0 <= rr < H and 0 <= cc < W:
        if int(out[rr, cc]) in stop_set:
            break
        coords.append((rr, cc))
        if colour is not None:
            out[rr, cc] = int(colour)
        rr += dr
        cc += dc
    return out if colour is not None else coords
'''


def _canonical_d8_key(mask_patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
  b = np.asarray(mask_patch, dtype=np.uint8)
  candidates: list[tuple[tuple[int, int], bytes]] = []
  for k in range(4):
    rot = np.rot90(b, k)
    candidates.append((rot.shape, rot.tobytes()))
    flp = np.fliplr(rot)
    candidates.append((flp.shape, flp.tobytes()))
  return min(candidates)


def _extract_components_host(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
  g = np.asarray(grid, dtype=int)
  struct = np.ones((3, 3), dtype=int)
  comps: list[dict[str, Any]] = []
  for col in sorted(int(c) for c in np.unique(g) if int(c) != bg):
    mask_col = g == col
    labeled, num = ndimage.label(mask_col, structure=struct)
    for idx in range(1, num + 1):
      m = labeled == idx
      rs, cs = np.where(m)
      r0, c0, r1, c1 = int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())
      sub_mask = m[r0 : r1 + 1, c0 : c1 + 1]
      comps.append({
          "colour": col,
          "size": int(m.sum()),
          "bbox": (r0, c0, r1, c1),
          "hw": (r1 - r0 + 1, c1 - c0 + 1),
          "mask_patch": sub_mask,
          "d8_key": _canonical_d8_key(sub_mask),
      })
  comps.sort(key=lambda o: (o["bbox"][0], o["bbox"][1], -o["size"]))
  return comps


def _bg_colour_host(grid: np.ndarray) -> int:
  vals, counts = np.unique(grid, return_counts=True)
  return int(vals[int(np.argmax(counts))])


def _index_to_letters(idx: int) -> str:
  letters = []
  n = idx
  while True:
    letters.append(chr(ord("A") + (n % 26)))
    n = n // 26 - 1
    if n < 0:
      break
  return "".join(reversed(letters))


def build_task_summary(
    train_pairs: list[dict[str, Any]],
    test_input: list[list[int]] | np.ndarray,
    max_objs_per_grid: int = 18,
) -> str:
  """Build Vu-style deterministic perception & test-novelty summary for an ARC task."""
  train_in_arrays = [np.asarray(p["input"], dtype=int) for p in train_pairs]
  train_out_arrays = [np.asarray(p["output"], dtype=int) for p in train_pairs]
  test_arr = np.asarray(test_input, dtype=int)

  shape_registry: dict[tuple[tuple[int, int], bytes], str] = {}

  def get_shape_id(d8_key: tuple[tuple[int, int], bytes]) -> str:
    if d8_key not in shape_registry:
      shape_registry[d8_key] = _index_to_letters(len(shape_registry))
    return shape_registry[d8_key]

  lines: list[str] = [
      "### Host-Computed Perception & Structural Summary (8-connected components, shapes up to D8 symmetry)"
  ]

  train_h_list: list[int] = []
  train_w_list: list[int] = []
  train_colors_union: set[int] = set()
  train_obj_counts: list[int] = []
  same_io_shape_all = True

  for idx, (inp, out) in enumerate(zip(train_in_arrays, train_out_arrays), start=1):
    ih, iw = inp.shape
    oh, ow = out.shape
    train_h_list.append(ih)
    train_w_list.append(iw)
    if (ih, iw) != (oh, ow):
      same_io_shape_all = False

    in_bg = _bg_colour_host(inp)
    out_bg = _bg_colour_host(out)
    in_comps = _extract_components_host(inp, in_bg)
    out_comps = _extract_components_host(out, out_bg)
    train_obj_counts.append(len(in_comps))

    in_vals, in_cnts = np.unique(inp, return_counts=True)
    out_vals, out_cnts = np.unique(out, return_counts=True)
    for c in in_vals:
      train_colors_union.add(int(c))

    in_hist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(in_vals, in_cnts))
    out_hist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(out_vals, out_cnts))
    lines.append(
        f"- Example {idx}: in={ih}x{iw} (bg={in_bg}, counts=[{in_hist}], objs={len(in_comps)}) "
        f"-> out={oh}x{ow} (bg={out_bg}, counts=[{out_hist}], objs={len(out_comps)})"
    )

    if len(in_comps) <= max_objs_per_grid and in_comps:
      in_desc = []
      for o in in_comps:
        sid = get_shape_id(o["d8_key"])
        r0, c0, r1, c1 = o["bbox"]
        in_desc.append(
            f"c{o['colour']}:sz{o['size']}:sh{sid}@({r0},{c0})-({r1},{c1})"
        )
      lines.append("  in_objs : " + "; ".join(in_desc))

    if len(out_comps) <= max_objs_per_grid and out_comps:
      out_desc = []
      for o in out_comps:
        sid = get_shape_id(o["d8_key"])
        r0, c0, r1, c1 = o["bbox"]
        out_desc.append(
            f"c{o['colour']}:sz{o['size']}:sh{sid}@({r0},{c0})-({r1},{c1})"
        )
      lines.append("  out_objs: " + "; ".join(out_desc))

    if 0 < len(in_comps) <= max_objs_per_grid and 0 < len(out_comps) <= max_objs_per_grid:
      unchanged = 0
      translated = 0
      recoloured = 0
      d8_transformed = 0
      used_out: set[int] = set()
      for ic in in_comps:
        matched = False
        for j, oc in enumerate(out_comps):
          if j in used_out:
            continue
          if (
              ic["colour"] == oc["colour"]
              and ic["bbox"] == oc["bbox"]
              and np.array_equal(ic["mask_patch"], oc["mask_patch"])
          ):
            unchanged += 1
            used_out.add(j)
            matched = True
            break
        if matched:
          continue
        for j, oc in enumerate(out_comps):
          if j in used_out:
            continue
          if ic["d8_key"] == oc["d8_key"]:
            used_out.add(j)
            if (
                ic["colour"] == oc["colour"]
                and ic["mask_patch"].shape == oc["mask_patch"].shape
                and np.array_equal(ic["mask_patch"], oc["mask_patch"])
            ):
              translated += 1
            elif (
                ic["colour"] != oc["colour"]
                and ic["mask_patch"].shape == oc["mask_patch"].shape
                and np.array_equal(ic["mask_patch"], oc["mask_patch"])
            ):
              recoloured += 1
            else:
              d8_transformed += 1
            break
      appeared = len(out_comps) - len(used_out)
      disappeared = len(in_comps) - len(used_out)
      lines.append(
          f"  obj_diff: unchanged={unchanged}, moved={translated}, "
          f"recoloured={recoloured}, d8_rot_flip={d8_transformed}, "
          f"appeared={appeared}, disappeared={disappeared}"
      )

  th, tw = test_arr.shape
  t_bg = _bg_colour_host(test_arr)
  t_comps = _extract_components_host(test_arr, t_bg)
  t_vals, t_cnts = np.unique(test_arr, return_counts=True)
  t_hist = ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(t_vals, t_cnts))
  lines.append(
      f"- Test Input: shape={th}x{tw} (bg={t_bg}, counts=[{t_hist}], objs={len(t_comps)})"
  )
  if len(t_comps) <= max_objs_per_grid and t_comps:
    t_desc = []
    for o in t_comps:
      sid = get_shape_id(o["d8_key"])
      r0, c0, r1, c1 = o["bbox"]
      t_desc.append(
          f"c{o['colour']}:sz{o['size']}:sh{sid}@({r0},{c0})-({r1},{c1})"
      )
    lines.append("  test_objs: " + "; ".join(t_desc))

  warnings: list[str] = []
  min_h, max_h = min(train_h_list), max(train_h_list)
  min_w, max_w = min(train_w_list), max(train_w_list)
  if not (min_h <= th <= max_h and min_w <= tw <= max_w):
    warnings.append(
        f"Test input dimensions ({th}x{tw}) are OUTSIDE train range "
        f"(H in [{min_h},{max_h}], W in [{min_w},{max_w}]) — NEVER hardcode grid sizes!"
    )
  new_colors = sorted(int(c) for c in t_vals if int(c) not in train_colors_union)
  if new_colors:
    warnings.append(
        f"Test input contains NEW colour(s) {new_colors} not seen in train inputs — "
        f"write colour-agnostic logic!"
    )
  if train_obj_counts:
    min_oc, max_oc = min(train_obj_counts), max(train_obj_counts)
    if not (min_oc <= len(t_comps) <= max_oc):
      warnings.append(
          f"Test input component count ({len(t_comps)}) is outside train range "
          f"[{min_oc},{max_oc}] — iterate dynamically over all objects!"
      )
  if same_io_shape_all:
    lines.append(
        "- Dimension rule hint: Every training pair preserves input shape (out.shape == in.shape)."
    )
  if warnings:
    lines.append("### IMPORTANT TEST NOVELTY WARNINGS (Outside train range):")
    for w in warnings:
      lines.append(f"- WARNING: {w}")

  return "\n".join(lines)
