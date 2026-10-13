"""E017 P1 (--task-summary): a host-computed perception pack appended to the first user message.

Deterministic and LLM-free. For each training pair: a header line (sizes, background, changed cells, colour
counts, simple whole-grid relations), the input's object table (8-connected same-colour components; multi-colour
components when they exist; separator panels), and an input -> output object matching. For the test input: the
same table plus an explicit "Outside train range" line (size, colours, object counts, border contact, object
sizes, unseen shapes; the features of experiments/E017_reasoning_efficiency/scripts/test_novelty.py).

Shapes get letter ids shared across the whole task: equal letters mean the same shape up to rotation/reflection.
The text is capped at about 3k tokens (`max_chars`) by showing fewer object rows; the test section is kept whole.
"""
from collections import Counter

import numpy as np

from arc_helpers import _arr, _local_mask, bg_colour, match_d8, objects, separators

INTRO = ("A computed summary of the grids follows; trust it over reading the grids by hand, and use code for any "
         "count.")
LEGEND = ("Objects = 8-connected cells of one colour, background excluded, numbered in reading order; "
          "bbox=(r0,c0)-(r1,c1) inclusive; holes = enclosed regions; border = touches the grid edge; "
          "shape letters are shared by all grids (same letter = same shape up to rotation/reflection; "
          "'dot' = 1 cell).")
MAX_CHARS = 10000  # ~3k tokens of this digit-heavy text
ROW_STEPS = (12, 8, 5, 3, 0)


def _counts(a):
    vals, cnt = np.unique(a, return_counts=True)
    return "{" + ",".join(f"{int(v)}:{int(n)}" for v, n in zip(vals, cnt)) + "}"


def _box(b):
    return f"({b[0]},{b[1]})-({b[2]},{b[3]})"


def _ids(nums, limit=10):
    shown = " ".join(f"#{n}" for n in nums[:limit])
    return shown + (f" +{len(nums) - limit} more" if len(nums) > limit else "")


class _Shapes:
    """Task-wide letter ids for shape keys, in order of first appearance."""

    def __init__(self):
        self.names = {}

    def name(self, obj):
        if obj["size"] == 1:
            return "dot"
        key = obj["shape_key"]
        if key not in self.names:
            n, s = len(self.names), ""
            while True:
                s = chr(65 + n % 26) + s
                n = n // 26 - 1
                if n < 0:
                    break
            self.names[key] = s
        return self.names[key]


def _grid_relation(a, b):
    """Short description of a simple whole-grid relation from input a to a differently sized output b."""
    name = match_d8(a, b)
    if name:
        return f"out = in {name}"
    (h, w), (H, W) = a.shape, b.shape
    if H <= h and W <= w:
        for r in range(h - H + 1):
            for c in range(w - W + 1):
                if (a[r:r + H, c:c + W] == b).all():
                    return f"out = crop of in at ({r},{c})"
    if H % h == 0 and W % w == 0:
        ky, kx = H // h, W // w
        if (np.kron(a, np.ones((ky, kx), dtype=int)) == b).all():
            return f"out = in scaled x{ky}" if ky == kx else f"out = in scaled {ky}x{kx}"
        if (np.tile(a, (ky, kx)) == b).all():
            return f"out = in tiled {ky}x{kx}"
    return None


def _object_rows(objs, shapes, rows):
    """Object table lines: the `rows` largest objects in reading order, the rest summarised per colour."""
    keep = sorted(sorted(range(len(objs)), key=lambda i: (-objs[i]["size"], i))[:rows])
    lines = [f"  #{i + 1} c{objs[i]['colour']} size={objs[i]['size']} bbox={_box(objs[i]['bbox'])} "
             f"holes={objs[i]['holes']} border={'yes' if objs[i]['touches_border'] else 'no'} "
             f"shape={shapes.name(objs[i])}" for i in keep]
    rest = [objs[i] for i in range(len(objs)) if i not in set(keep)]
    if rest:
        by = {}
        for o in rest:
            by.setdefault(o["colour"], []).append(o["size"])
        lines.append(f"  +{len(rest)} more: " + ", ".join(
            f"c{c} x{len(s)} (sizes {min(s)}-{max(s)})" for c, s in sorted(by.items())))
    return lines


def _grid_block(label, a, objs, shapes, rows):
    lines = [f" {label} objects ({len(objs)}{'' if len(objs) <= rows else f'; {rows} largest shown'}):"
             if objs else f" {label} objects: none"]
    if objs:
        lines += _object_rows(objs, shapes, rows)
    multi = [o for o in objects(a, by="nonbg") if len(o["colours"]) > 1]
    if multi:
        shown = "; ".join(f"colours {','.join(map(str, o['colours']))} size={o['size']} bbox={_box(o['bbox'])}"
                          for o in multi[:min(4, max(rows, 1))])
        lines.append(f" {label} multi-colour objects ({len(multi)}): {shown}"
                     + (f"; +{len(multi) - 4} more" if len(multi) > 4 and rows >= 4 else ""))
    sep = separators(a)
    if sep:
        nr, nc = len(_runs(a.shape[0], sep["rows"])), len(_runs(a.shape[1], sep["cols"]))
        where = "; ".join(x for x in (f"rows {','.join(map(str, sep['rows']))}" if sep["rows"] else "",
                                       f"cols {','.join(map(str, sep['cols']))}" if sep["cols"] else "") if x)
        lines.append(f" {label} panels: {nr}x{nc} split by colour {sep['colour']} at {where}")
    return lines


def _runs(n, lines):
    from arc_helpers import _spans
    return _spans(n, lines)


def _match(a, b, in_objs, out_objs, out_bg):
    """Input -> output object matching for same-sized grids. Returns (summary line, unmatched output objects)."""
    out_cells = {tuple(o["cells"]): j for j, o in enumerate(out_objs)}
    claimed, groups = set(), {}

    def add(kind, text):
        groups.setdefault(kind, []).append(text)

    pending = []
    for i, o in enumerate(in_objs):
        j = out_cells.get(tuple(o["cells"]))
        if j is not None and j not in claimed:
            claimed.add(j)
            if out_objs[j]["colour"] == o["colour"]:
                add("unchanged", i + 1)
            else:
                add(f"recoloured {o['colour']}->{out_objs[j]['colour']}", i + 1)
        else:
            pending.append(i)
    for i in pending:
        o = in_objs[i]
        local = _local_mask(o["cells"])
        best = None
        for j, q in enumerate(out_objs):
            if j in claimed or q["colour"] != o["colour"] or q["shape_key"] != o["shape_key"]:
                continue
            dist = abs(q["bbox"][0] - o["bbox"][0]) + abs(q["bbox"][1] - o["bbox"][1])
            if best is None or dist < best[0]:
                best = (dist, j)
        if best is not None:
            j = best[1]
            claimed.add(j)
            q = out_objs[j]
            op = match_d8(local, _local_mask(q["cells"]))
            dy, dx = q["bbox"][0] - o["bbox"][0], q["bbox"][1] - o["bbox"][1]
            if op == "id":
                add("moved", f"#{i + 1} ({dy:+d},{dx:+d})")
            else:
                add("transformed", f"#{i + 1} {op} to {_box(q['bbox'])}")
            continue
        vals = [int(b[r, c]) for r, c in o["cells"]]
        if all(v == out_bg for v in vals):
            add("removed", i + 1)
        else:
            kept = sum(v == o["colour"] for v in vals)
            add("changed", f"#{i + 1} ({kept}/{o['size']} cells keep c{o['colour']})")
    parts = []
    order = ["unchanged", "moved", "transformed", "removed", "changed"]
    for kind in sorted(groups, key=lambda k: (order.index(k) if k in order else 1.5, k)):
        items = groups[kind]
        if items and isinstance(items[0], int):
            parts.append(f"{kind} {_ids(items)}")
        else:
            parts.append(f"{kind} " + ", ".join(items[:8]) + (f", +{len(items) - 8} more" if len(items) > 8 else ""))
    unmatched = [q for j, q in enumerate(out_objs) if j not in claimed]
    return " in->out: " + ("; ".join(parts) if parts else "no input objects"), unmatched


def _match_resized(in_objs, out_objs):
    """For differently sized grids: which input objects reappear (same colour and shape) in the output."""
    found, used = [], set()
    for i, o in enumerate(in_objs):
        for j, q in enumerate(out_objs):
            if j not in used and q["colour"] == o["colour"] and q["shape_key"] == o["shape_key"] and o["size"] > 1:
                op = match_d8(_local_mask(o["cells"]), _local_mask(q["cells"]))
                found.append(f"#{i + 1} at {_box(q['bbox'])}" + ("" if op == "id" else f" ({op})"))
                used.add(j)
                break
    return " in objects found in out (same colour+shape): " + (", ".join(found[:10]) + (
        f", +{len(found) - 10} more" if len(found) > 10 else "") if found else "none")


def _feats(a, objs):
    return {"h": a.shape[0], "w": a.shape[1], "colours": set(np.unique(a).tolist()), "bg": bg_colour(a),
            "ncol": len(np.unique(a)), "comps": len(objs), "touch": sum(o["touches_border"] for o in objs),
            "maxsize": max((o["size"] for o in objs), default=0), "minsize": min((o["size"] for o in objs), default=0)}


def _range(vals):
    lo, hi = min(vals), max(vals)
    return f"train: {lo}" if lo == hi else f"train {lo}-{hi}"


def novelty_line(train_inputs, train_objs, test, test_objs, shapes):
    """The 'Outside train range' line for the test input."""
    tr = [_feats(a, o) for a, o in zip(train_inputs, train_objs)]
    te = _feats(test, test_objs)
    flags = []

    def out(k):
        return not min(f[k] for f in tr) <= te[k] <= max(f[k] for f in tr)

    if out("h") or out("w"):
        flags.append(f"size {te['h']}x{te['w']} (train heights {_range([f['h'] for f in tr])[6:].strip(': ')}, "
                     f"widths {_range([f['w'] for f in tr])[6:].strip(': ')})")
    new = sorted(te["colours"] - set().union(*[f["colours"] for f in tr]))
    if new:
        flags.append(("new colour " if len(new) == 1 else "new colours ") + ",".join(map(str, new)))
    always = sorted(set.intersection(*[f["colours"] for f in tr]) - te["colours"])
    if always:
        flags.append(("colour " if len(always) == 1 else "colours ") + ",".join(map(str, always))
                     + " absent (present in every train input)")
    labels = {"ncol": "number of colours", "comps": "object count", "touch": "objects touching the border",
              "maxsize": "largest object size", "minsize": "smallest object size"}
    for k, text in labels.items():
        if out(k):
            flags.append(f"{text} {te[k]} ({_range([f[k] for f in tr])})")
    seen = {o["shape_key"] for objs in train_objs for o in objs}
    unseen = []
    for i, o in enumerate(test_objs):
        if o["shape_key"] not in seen and o["size"] > 1:
            name = shapes.name(o)
            if name not in [u[0] for u in unseen]:
                unseen.append((name, i + 1))
    if unseen:
        flags.append("shapes not in any train input: " + ", ".join(f"{n} (#{i})" for n, i in unseen[:8])
                     + (f", +{len(unseen) - 8} more" if len(unseen) > 8 else ""))
    if te["bg"] not in {f["bg"] for f in tr}:
        flags.append(f"background {te['bg']} (train: {','.join(map(str, sorted({f['bg'] for f in tr})))})")
    if not flags:
        return (" Outside train range: nothing (size, colours, object count, border contact and object sizes are "
                "within the train inputs' ranges; every shape appears in train).")
    return " Outside train range: " + "; ".join(flags)


def _render(train, test_input, rows):
    shapes = _Shapes()
    ins = [_arr(p["input"]) for p in train]
    outs = [_arr(p["output"]) for p in train]
    in_objs = [objects(a) for a in ins]
    out_objs = [objects(b) for b in outs]
    test = _arr(test_input)
    test_objs = objects(test)
    # assign shape letters in a fixed order: pair by pair (input, then output), then the test input
    for i in range(len(train)):
        for o in in_objs[i] + out_objs[i]:
            shapes.name(o)
    for o in test_objs:
        shapes.name(o)
    pair_blocks = []
    for i, (a, b) in enumerate(zip(ins, outs)):
        abg, bbg = bg_colour(a), bg_colour(b)
        head = f"Ex{i}: in {a.shape[0]}x{a.shape[1]} bg={abg} -> out {b.shape[0]}x{b.shape[1]} bg={bbg}"
        if a.shape == b.shape:
            diff = a != b
            trans = Counter((int(x), int(y)) for x, y in zip(a[diff], b[diff]))
            top = ", ".join(f"{x}->{y} x{n}" for (x, y), n in sorted(trans.items(), key=lambda t: (-t[1], t[0]))[:6])
            more = f", +{len(trans) - 6} kinds" if len(trans) > 6 else ""
            head += f" | {int(diff.sum())} cells changed" + (f" ({top}{more})" if trans else "")
            op = match_d8(a, b)
            if op not in (None, "id"):
                head += f" | out = in {op}"
        else:
            rel = _grid_relation(a, b)
            if rel:
                head += f" | {rel}"
        head += f" | colours in {_counts(a)} out {_counts(b)}"
        lines = [head] + _grid_block("in", a, in_objs[i], shapes, rows)
        if a.shape == b.shape:
            match, unmatched = _match(a, b, in_objs[i], out_objs[i], bbg)
            lines.append(match)
            if unmatched:
                lines.append(f" out objects not matching an input object ({len(unmatched)}): " + "; ".join(
                    f"c{q['colour']} size={q['size']} bbox={_box(q['bbox'])} shape={shapes.name(q)}"
                    for q in unmatched[:max(rows // 2, 2)])
                    + (f"; +{len(unmatched) - max(rows // 2, 2)} more" if len(unmatched) > max(rows // 2, 2) else ""))
        else:
            lines += _grid_block("out", b, out_objs[i], shapes, max(rows // 2, min(rows, 3)))
            lines.append(_match_resized(in_objs[i], out_objs[i]))
        pair_blocks.append("\n".join(lines))
    tl = [f"Test: in {test.shape[0]}x{test.shape[1]} bg={bg_colour(test)} | colours {_counts(test)}"]
    tl += _grid_block("test", test, test_objs, shapes, max(rows, 5))
    tl.append(novelty_line(ins, in_objs, test, test_objs, shapes))
    return pair_blocks, "\n".join(tl)


def task_summary(train, test_input, max_chars=MAX_CHARS):
    """The perception pack text (starts with a blank line), at most about `max_chars` characters."""
    for rows in ROW_STEPS:
        pairs, test = _render(train, test_input, rows)
        text = "\n\n" + "\n".join([INTRO, LEGEND, ""] + pairs + [test])
        if len(text) <= max_chars:
            return text
    # still too long at 0 rows: keep the test section, cut the pair blocks
    head = "\n\n" + "\n".join([INTRO, LEGEND, ""])
    budget = max_chars - len(head) - len(test) - 40
    body = "\n".join(pairs)[:max(budget, 0)]
    return head + body + "\n[summary truncated]\n" + test
