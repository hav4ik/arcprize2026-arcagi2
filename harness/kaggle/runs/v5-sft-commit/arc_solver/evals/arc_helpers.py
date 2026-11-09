"""E017 P3: numpy grid helpers. Used on the host by task_summary.py and, with --helpers, preloaded into the model's
sandbox session (the harness executes this file's source there, so it must stay plain numpy + stdlib).

Grids are lists of lists or numpy arrays of ints 0-9. Coordinates are (row, col); boxes are inclusive
(r0, c0, r1, c1).
"""
import copy as _copy
import hashlib as _hashlib
from collections import Counter as _Counter

import numpy as np

D8_NAMES = ("id", "rot90", "rot180", "rot270", "flipud", "fliplr", "transpose", "antitranspose")


def _arr(g):
    return np.array(g, dtype=int)


def bg_colour(g):
    """Most frequent colour (ties: the smaller colour)."""
    vals, counts = np.unique(_arr(g), return_counts=True)
    return int(vals[counts == counts.max()].min())


def bbox(mask):
    """(r0, c0, r1, c1) inclusive box of the True cells of `mask`, or None if empty."""
    rows, cols = np.nonzero(np.asarray(mask))
    if not len(rows):
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def crop(g, box):
    r0, c0, r1, c1 = box
    return _arr(g)[r0:r1 + 1, c0:c1 + 1]


def _d8_op(a, name):
    return {"id": lambda x: x, "rot90": lambda x: np.rot90(x, 1), "rot180": lambda x: np.rot90(x, 2),
            "rot270": lambda x: np.rot90(x, 3), "flipud": np.flipud, "fliplr": np.fliplr,
            "transpose": lambda x: x.T, "antitranspose": lambda x: np.rot90(x, 2).T}[name](a)


def d8(g):
    """{name: array} for the 8 rotations/reflections (rot90 is numpy's counter-clockwise rotation)."""
    a = _arr(g)
    return {name: _d8_op(a, name) for name in D8_NAMES}


def match_d8(a, b):
    """Name of the first D8 transform t with t(a) == b, else None."""
    a, b = _arr(a), _arr(b)
    for name in D8_NAMES:
        t = _d8_op(a, name)
        if t.shape == b.shape and (t == b).all():
            return name
    return None


def _components(mask, diag=True):
    """Connected components of a boolean mask as lists of (r, c), ordered by their first cell (reading order)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] if diag else \
        [(-1, 0), (0, -1), (0, 1), (1, 0)]
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack, cells = [(r, c)], []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in steps:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append(sorted(cells))
    return out


def _local_mask(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    m = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
    for r, c in cells:
        m[r - r0, c - c0] = True
    return m


def _holes(local):
    """Number of 4-connected regions of non-object cells fully enclosed by the object."""
    padded = np.pad(local, 1, constant_values=False)
    regions = _components(~padded, diag=False)
    h, w = padded.shape
    return sum(not any(r in (0, h - 1) or c in (0, w - 1) for r, c in reg) for reg in regions)


def shape_key(cells_or_mask):
    """8-hex-digit id of a cell set's shape, the same for all rotations/reflections (and positions)."""
    m = np.asarray(cells_or_mask)
    if m.dtype != bool:
        m = _local_mask([tuple(x) for x in cells_or_mask])
    else:
        box = bbox(m)
        m = crop(m.astype(int), box).astype(bool)
    forms = []
    for name in D8_NAMES:
        t = _d8_op(m, name)
        forms.append((t.shape, np.packbits(t).tobytes()))
    shape, packed = min(forms)
    return _hashlib.sha1(repr((shape, packed)).encode()).hexdigest()[:8]


def objects(g, by="colour", diag=True, bg=None):
    """Objects of a grid, in reading order of their first cell.

    by='colour': connected cells of one colour (excluding bg); by='nonbg': connected non-background cells of any
    colours. Each object: {colour (most common), colours, cells [(r, c)], bbox (r0, c0, r1, c1), size, holes,
    touches_border, shape_key}."""
    a = _arr(g)
    if bg is None:
        bg = bg_colour(a)
    h, w = a.shape
    comps = []
    if by == "colour":
        for colour in sorted(set(np.unique(a).tolist()) - {bg}):
            comps.extend(_components(a == colour, diag))
        comps.sort()
    elif by == "nonbg":
        comps = _components(a != bg, diag)
    else:
        raise ValueError("by must be 'colour' or 'nonbg'")
    out = []
    for cells in comps:
        counts = _Counter(int(a[r, c]) for r, c in cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        box = (min(rs), min(cs), max(rs), max(cs))
        local = _local_mask(cells)
        out.append({"colour": max(sorted(counts), key=lambda k: counts[k]), "colours": sorted(counts),
                    "cells": cells, "bbox": box, "size": len(cells), "holes": _holes(local),
                    "touches_border": box[0] == 0 or box[1] == 0 or box[2] == h - 1 or box[3] == w - 1,
                    "shape_key": shape_key(local)})
    return out


def separators(g, bg=None):
    """Full rows/columns of one non-background colour that split the grid into panels.

    Returns {colour, rows, cols} (line indices) for the colour with the most such lines, or None."""
    a = _arr(g)
    if bg is None:
        bg = bg_colour(a)
    best = None
    for colour in sorted(set(np.unique(a).tolist()) - {bg}):
        rows = [r for r in range(a.shape[0]) if (a[r] == colour).all()]
        cols = [c for c in range(a.shape[1]) if (a[:, c] == colour).all()]
        if rows == list(range(a.shape[0])) or cols == list(range(a.shape[1])):
            continue
        if len(_spans(a.shape[0], rows)) * len(_spans(a.shape[1], cols)) < 2:
            continue
        if best is None or len(rows) + len(cols) > len(best["rows"]) + len(best["cols"]):
            best = {"colour": colour, "rows": rows, "cols": cols}
    return best


def _spans(n, lines):
    """Maximal runs of indices in range(n) not in `lines`."""
    spans, start = [], None
    for i in range(n + 1):
        inside = i < n and i not in lines
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            spans.append((start, i - 1))
            start = None
    return spans


def panels(g, bg=None):
    """Sub-grids split by full separator lines: list of {'bbox', 'grid', 'pos': (i, j)} in reading order
    (empty list if the grid has no separator lines)."""
    a = _arr(g)
    sep = separators(a, bg)
    if sep is None:
        return []
    out = []
    for i, (r0, r1) in enumerate(_spans(a.shape[0], sep["rows"])):
        for j, (c0, c1) in enumerate(_spans(a.shape[1], sep["cols"])):
            out.append({"bbox": (r0, c0, r1, c1), "grid": a[r0:r1 + 1, c0:c1 + 1], "pos": (i, j)})
    return out


def stamp(g, pattern, at, clip=True, transparent=None):
    """Copy of g with `pattern` pasted with its top-left cell at `at` (may be negative or overhang);
    cells outside the grid are dropped (clip=True) or raise; cells equal to `transparent` are skipped."""
    a = _arr(g).copy()
    p = _arr(pattern)
    r0, c0 = at
    for (dr, dc), v in np.ndenumerate(p):
        if transparent is not None and v == transparent:
            continue
        r, c = r0 + dr, c0 + dc
        if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            a[r, c] = v
        elif not clip:
            raise IndexError(f"cell {(r, c)} outside the grid")
    return a


def ray(g, start, d, stop=None, bg=None):
    """Cells visited from `start` (exclusive) stepping by d=(dr, dc) until the grid edge or the first cell whose
    value satisfies stop(value) (that cell is not included). Default stop: any non-background value."""
    a = _arr(g)
    if stop is None:
        b = bg_colour(a) if bg is None else bg
        stop = lambda v: v != b  # noqa: E731
    r, c = start
    out = []
    while True:
        r, c = r + d[0], c + d[1]
        if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or stop(int(a[r, c])):
            return out
        out.append((r, c))


def show(g):
    """Print one string of digits per row."""
    for row in _arr(g):
        print("".join(str(int(v)) for v in row))


def check(fn, pairs=None, max_diffs=8):
    """Run fn on every training input (a copy) and print one line per pair: OK, the shape mismatch, an error, or
    up to `max_diffs` differing cells as (r,c) got/exp; then ALL PASS or k/n pass. Returns True if all pass."""
    if pairs is None:
        pairs = globals()["train"]
    ok = 0
    for i, pair in enumerate(pairs):
        exp = _arr(pair["output"])
        try:
            got = fn(_copy.deepcopy(pair["input"]))
            got = _arr(got.tolist() if hasattr(got, "tolist") else got)
        except Exception as exc:  # report and continue with the next pair
            print(f"pair {i}: error {type(exc).__name__}: {str(exc)[:200]}")
            continue
        if got.shape != exp.shape:
            print(f"pair {i}: shape {got.shape[0]}x{got.shape[1] if got.ndim > 1 else 0} "
                  f"vs expected {exp.shape[0]}x{exp.shape[1]}")
            continue
        diff = list(zip(*np.nonzero(got != exp)))
        if not diff:
            ok += 1
            print(f"pair {i}: OK")
            continue
        cells = ", ".join(f"({int(r)},{int(c)}) {int(got[r, c])}/{int(exp[r, c])}" for r, c in diff[:max_diffs])
        more = f", +{len(diff) - max_diffs} more" if len(diff) > max_diffs else ""
        print(f"pair {i}: {len(diff)} cells differ (r,c got/exp): {cells}{more}")
    print("ALL PASS" if ok == len(pairs) else f"{ok}/{len(pairs)} pass")
    return ok == len(pairs)


HELPER_NAMES = ("bg_colour", "bbox", "crop", "d8", "match_d8", "shape_key", "objects", "separators", "panels",
                "stamp", "ray", "show", "check")
for _name in HELPER_NAMES:
    globals()[_name]._arc_helper = True  # the train-check labeller skips these when probing session functions
del _name
