"""Linux CPU-only child for REPL-TIR tool calls; never import a model here.

One process per python/submit call. stdin carries one JSON request holding the public task (train
pairs plus test inputs, never test outputs), the trajectory's earlier cells, and one operation:
["cell", code] runs a new cell and ["eval", function_name] runs a function on every input.
["probe", function_name] evaluates one reserved input with the same visible namespace; the
reserved answer stays in the controller and is never sent to this process. The
child preloads the libraries cells may import, installs resource limits and the fail-closed
seccomp filter from qwen38_tir_worker.py (no opens, sockets, exec or clone), replays the earlier
cells silently and runs the operation. It writes one JSON line to the original stdout and exits.
"""

import ast
import contextlib
import dis
import hashlib
import importlib.util
import inspect
import io
import json
import linecache
import os
import resource
import signal
import sys
import time
import traceback
import types
from pathlib import Path

# The child is launched with -P and a minimal environment: load only this trusted
# sibling explicitly, before seccomp, without adding a model-writable sys.path.
_audit_spec = importlib.util.spec_from_file_location(
    "_arc_input_audit", Path(__file__).with_name("qwen38_repl_audit.py"))
_audit_module = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(_audit_module)
normalize_measurement = _audit_module.normalize_measurement
_execution_spec = importlib.util.spec_from_file_location(
    "_arc_execution_audit", Path(__file__).with_name("qwen38_repl_execution.py"))
_execution_module = importlib.util.module_from_spec(_execution_spec)
_execution_spec.loader.exec_module(_execution_module)
_memorization_spec = importlib.util.spec_from_file_location(
    "_arc_memorization_guard", Path(__file__).with_name("qwen38_repl_memorization.py"))
_memorization_module = importlib.util.module_from_spec(_memorization_spec)
_memorization_spec.loader.exec_module(_memorization_module)

MAX_REQUEST = 20_000_000
MAX_OUTPUT = 4000
# Imported before seccomp so cells can import them; anything else fails with ModuleNotFoundError.
PRELOAD = (
    "numpy numpy.random numpy.fft numpy.linalg scipy.ndimage scipy.optimize scipy.sparse "
    "scipy.sparse.csgraph collections itertools math cmath functools fractions decimal numbers "
    "inspect copy types heapq random bisect operator string statistics typing dataclasses enum "
    "json re array pprint textwrap"
).split()
OPTIONAL = ("scipy.signal", "scipy.spatial")


class TimeLimit(BaseException):  # not Exception: model code often has bare "except Exception"
    pass


def _alarm(signum, frame):
    raise TimeLimit("time limit exceeded")


class Sink(io.StringIO):
    """Keeps the head of the output plus the tail, so long prints don't eat memory or context."""

    def __init__(self, limit=MAX_OUTPUT):
        super().__init__()
        self.limit, self.dropped, self.tail = limit, 0, ""

    def write(self, text):
        room = self.limit * 2 // 3 - self.tell()
        if room > 0:
            super().write(text[:room])
            text = text[room:]
        if text:
            self.dropped += len(text)
            self.tail = (self.tail + text)[-(self.limit // 3):]
        return len(text)

    def text(self):
        head = self.getvalue()
        if not self.dropped:
            return head
        cut = self.dropped - len(self.tail)
        return f"{head}\n... [{cut} characters cut] ...\n{self.tail}" if cut > 0 else head + self.tail


class Null(io.TextIOBase):
    def write(self, text):
        return len(text)


def preload():
    import importlib
    for name in PRELOAD:
        importlib.import_module(name)
    for name in OPTIONAL:
        with contextlib.suppress(ImportError):
            importlib.import_module(name)
    import numpy as np
    import scipy.ndimage as ndi
    from collections import Counter
    # NumPy 2.x and the stdlib import some modules lazily on first use; warm the common grid
    # operations here instead of permitting file access after model code starts.
    probe = np.array([[0, 1], [1, 1]], dtype=np.int64)
    np.unique(probe, return_counts=True)
    probe.max(), probe.sum(), probe.mean(), np.argwhere(probe), np.rot90(probe), np.pad(probe, 1)
    repr(probe), str(probe), repr(probe.astype(float)), repr(probe.astype(bool)), repr(np.int64(3))
    ndi.label(probe), ndi.find_objects(ndi.label(probe)[0]), ndi.binary_fill_holes(probe)
    Counter([0, 1, 1]).most_common(1)
    try:
        np.set_printoptions(linewidth=250, threshold=2000, legacy="1.25")  # scalars print as 3
    except ValueError:  # NumPy 1.x prints scalars plainly already
        np.set_printoptions(linewidth=250, threshold=2000)
    return np, ndi


def restrict(extra=("setitimer", "getitimer")):
    spec = importlib.util.spec_from_file_location("_arc_tir_sandbox", Path(__file__).with_name("qwen38_tir_worker.py"))
    sandbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox)
    sandbox.restrict_system_calls(extra=extra)


# ----- helpers visible to the model ------------------------------------------------------


def to_grid(value):
    """A valid ARC grid as a list of int lists, or ValueError."""
    a = np.asarray(value)
    if a.dtype == object or a.ndim != 2:
        raise ValueError(f"must return a 2D grid, got shape {a.shape} dtype {a.dtype}")
    if not (1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30):
        raise ValueError(f"grid shape {a.shape} is outside 1..30")
    if a.dtype != bool and not np.issubdtype(a.dtype, np.integer):
        if not np.issubdtype(a.dtype, np.number) or not np.all(a == np.round(a)):
            raise ValueError(f"grid values must be integers, got dtype {a.dtype}")
    a = a.astype(int)
    if a.min() < 0 or a.max() > 9:
        raise ValueError("grid values must be in 0..9")
    return a.tolist()


def fmt(grid):
    return "\n".join(GRID_SEP.join(str(int(v)) for v in row) for row in np.asarray(grid))


def training_grid_reference(value):
    """Reference an unchanged complete training grid, including copies/aliases.

    This is a display/efficiency guard, not a sandbox boundary. It does not change array values,
    solver evaluation or check() diagnostics, and does not inspect hidden test outputs.
    """
    if not isinstance(value, np.ndarray) and type(value) not in (list, tuple):
        return None
    try:
        a = np.asarray(value)
        if a.ndim != 2 or a.dtype.kind not in "biuf":
            return None
        for label, grid in TRAIN_ARRAYS:
            if a.shape == grid.shape and np.array_equal(a, grid):
                return f"[unchanged {label}, {a.shape[0]}x{a.shape[1]}; already in the original prompt]"
    except (TypeError, ValueError, RecursionError):
        pass
    return None


def display_value(value, seen=None):
    reference = training_grid_reference(value)
    if reference:
        DISPLAY_STATE["suppressed"] += 1
        return reference
    # Handle print(train), pair dictionaries and trailing expressions without changing session data.
    if type(value) not in (list, tuple, dict):
        return value
    seen = set() if seen is None else seen
    if id(value) in seen:
        return value
    seen.add(id(value))
    try:
        if type(value) is dict:
            return {k: display_value(v, seen) for k, v in value.items()}
        return type(value)(display_value(v, seen) for v in value)
    finally:
        seen.remove(id(value))


def display_print(*values, **kwargs):
    print(*(display_value(value) for value in values), **kwargs)


def show(*grids, reason=None):
    if reason is not None and (len(grids) != 1 or not isinstance(reason, str) or len(reason.strip()) < 12):
        raise ValueError("A targeted full-grid review needs one grid and a specific reason (12+ characters).")
    for grid in grids:
        a = np.asarray(grid)
        reference = training_grid_reference(a)
        if reference:
            if reason and not DISPLAY_STATE["reviewed"]:
                DISPLAY_STATE["reviewed"] = True
                print(f"[targeted review: {reason.strip()}]")
            else:
                DISPLAY_STATE["suppressed"] += 1
                print(reference)
                continue
        print(f"({a.shape[0]}x{a.shape[1]})\n{fmt(a)}\n" if a.ndim == 2 else repr(grid))


def view(*grids, reason):
    """Return only validated numeric grids; the controller, not model code, draws the PNG."""
    if not isinstance(reason, str) or not 12 <= len(reason.strip()) <= 400:
        raise ValueError("view needs a specific reason of 12-400 characters")
    if not 1 <= len(grids) <= 3 or len(VISUALS) >= 2:
        raise ValueError("view accepts 1-3 grids and at most two calls per cell")
    VISUALS.append({"reason": reason.strip(), "grids": [to_grid(g) for g in grids]})
    print(f"[view {len(VISUALS)} queued: {reason.strip()}]")


def objects(grid, bg=None, diag=False, multicolor=False):
    """Connected components of non-background cells, sorted by top-left corner."""
    from collections import Counter
    g = np.asarray(grid)
    if bg is None:
        bg = Counter(g.ravel().tolist()).most_common(1)[0][0]
    structure = np.ones((3, 3), int) if diag else None
    masks = [(None, g != bg)] if multicolor else [(int(c), g == c) for c in np.unique(g) if c != bg]
    found = []
    for color, m in masks:
        labels, n = ndi.label(m, structure=structure)
        for k in range(1, n + 1):
            rr, cc = np.nonzero(labels == k)
            r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
            mask = labels[r0:r1 + 1, c0:c1 + 1] == k
            found.append(dict(
                color=color if color is not None else Counter(g[rr, cc].tolist()).most_common(1)[0][0],
                size=len(rr), cells=list(zip(rr.tolist(), cc.tolist())),
                bbox=(int(r0), int(c0), int(r1), int(c1)), mask=mask,
                grid=np.where(mask, g[r0:r1 + 1, c0:c1 + 1], bg)))
    found.sort(key=lambda o: o["bbox"][:2])
    return found


def _d4(a):
    for flip in (False, True):
        for k in range(4):
            yield ("reflect_lr+" if flip else "") + f"rot{k * 90}", np.rot90(np.fliplr(a) if flip else a, k)


def _shape_id(mask):
    # Canonical occupied-cell shape, independent of color/orientation, not an object identity.
    canonical = min(f"{a.shape[0]}x{a.shape[1]}:" + "".join(str(int(v)) for v in a.ravel())
                    for _, a in _d4(mask))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _grid(grid):
    return np.asarray(to_grid(grid), dtype=int)


def _background(g, bg):
    if bg is not None:
        if type(bg) is not int or not 0 <= bg <= 9:
            raise ValueError("bg must be an integer in 0..9 or None")
        return bg
    from collections import Counter
    return Counter(g.ravel().tolist()).most_common(1)[0][0]


def _bounded_int(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")


def describe_grid(grid, bg=None, diag=False, multicolor=False, max_objects=12):
    """Compact measurements; connectivity/background are choices, not inferred ARC semantics."""
    _bounded_int(max_objects, "max_objects", 100)
    g = _grid(grid)
    background = _background(g, bg)
    comps = objects(g, background, diag, multicolor)
    colors, counts = np.unique(g, return_counts=True)
    selected, owners = [], {}
    for i, obj in enumerate(comps[:max_objects]):
        r0, c0, r1, c1 = obj["bbox"]
        mask = obj["mask"]
        filled = ndi.binary_fill_holes(mask)
        selected.append(dict(id=i, color=obj["color"], colors=sorted(set(int(g[r, c]) for r, c in obj["cells"])),
                             size=obj["size"], bbox=list(obj["bbox"]), shape_id_d4=_shape_id(mask),
                             holes=int(ndi.label(filled & ~mask)[1]),
                             border_contacts=[n for n, yes in (("top", r0 == 0), ("left", c0 == 0),
                                               ("bottom", r1 == len(g) - 1), ("right", c1 == len(g[0]) - 1)) if yes],
                             mask_symmetries=[name for name, a in _d4(mask)
                                              if name != "rot0" and np.array_equal(a, mask)]))
        owners.update((tuple(cell), i) for cell in obj["cells"])
    adjacency = set()
    for (r, c), i in owners.items():
        for pos in ((r + 1, c), (r, c + 1)):
            j = owners.get(pos)
            if j is not None and i != j:
                adjacency.add(tuple(sorted((i, j))))
    contains = []
    for a in selected:
        for b in selected:
            ar, ac, az, aw = a["bbox"]
            br, bc, bz, bw = b["bbox"]
            if a["id"] != b["id"] and a["bbox"] != b["bbox"] and ar <= br <= bz <= az and ac <= bc <= bw <= aw:
                contains.append([a["id"], b["id"]])
    return dict(shape=list(g.shape), area=int(g.size), background=background,
                background_source="most_common_guess" if bg is None else "explicit",
                colors={int(c): int(n) for c, n in zip(colors, counts)},
                symmetries=[name for name, a in _d4(g) if name != "rot0" and np.array_equal(a, g)],
                object_count=len(comps), objects=selected, objects_omitted=max(0, len(comps) - len(selected)),
                adjacency_4=[list(pair) for pair in sorted(adjacency)], bbox_contains=contains)


def grid_diff(before, after, bg=None, diag=False, multicolor=False, max_matches=12):
    """Compare known grids; all matching components are candidates, never asserted identities."""
    _bounded_int(max_matches, "max_matches", 100)
    a, b = _grid(before), _grid(after)
    abg, bbg = _background(a, bg), _background(b, bg)
    ac, bc = objects(a, abg, diag, multicolor), objects(b, bbg, diag, multicolor)
    ak, bk = [_shape_id(o["mask"]) for o in ac], [_shape_id(o["mask"]) for o in bc]
    candidates, total = [], 0
    for i, left in enumerate(ac):
        for j, right in enumerate(bc):
            if ak[i] != bk[j]:
                continue
            transforms = [name for name, m in _d4(left["mask"]) if np.array_equal(m, right["mask"])]
            unchanged = (left["cells"] == right["cells"] and
                         all(int(a[r, c]) == int(b[r, c]) for r, c in left["cells"]))
            if unchanged:
                continue
            total += 1
            if len(candidates) < max_matches:
                candidates.append(dict(before_id=i, after_id=j, shape_id_d4=ak[i],
                    before_bbox=list(left["bbox"]), after_bbox=list(right["bbox"]),
                    before_color=left["color"], after_color=right["color"], transforms=transforms,
                    bbox_offset=[right["bbox"][k] - left["bbox"][k] for k in (0, 1)],
                    ambiguous=ak.count(ak[i]) > 1 or bk.count(ak[i]) > 1))
    aligned = a.shape == b.shape
    transitions = []
    if aligned:
        for old in np.unique(a):
            for new in np.unique(b):
                n = int(((a == old) & (b == new)).sum())
                if n and old != new:
                    transitions.append(dict(before=int(old), after=int(new), cells=n))
    return dict(before_shape=list(a.shape), after_shape=list(b.shape),
                backgrounds=[abg, bbg], changed_cells=int((a != b).sum()) if aligned else None,
                color_count_delta={c: int((b == c).sum() - (a == c).sum()) for c in range(10)
                                   if int((b == c).sum()) != int((a == c).sum())},
                aligned_color_changes=transitions, object_counts=[len(ac), len(bc)],
                possible_matches=candidates, matches_omitted=total - len(candidates),
                matching="D4 occupied-shape only; colors may differ; not proof of identity or a transformation rule")


def search_candidates(candidates, max_candidates=256, max_results=5):
    """Bounded program/parameter search using visible demonstrations ONLY; no test access/scoring."""
    _bounded_int(max_candidates, "max_candidates", 4096)
    _bounded_int(max_results, "max_results", 100)
    import itertools
    source = iter(candidates.items() if isinstance(candidates, dict) else candidates)
    rows = []
    for name, fn in itertools.islice(source, max_candidates):
        if not isinstance(name, str) or not callable(fn):
            raise ValueError("candidates must contain (string name, callable) pairs")
        exact, cell_scores, errors = 0, [], []
        for pair in TRAIN:
            try:
                got, want = _grid(fn(np.array(pair["input"]))), np.array(pair["output"])
                score = float((got == want).mean()) if got.shape == want.shape else 0.0
                exact += int(got.shape == want.shape and np.array_equal(got, want))
                cell_scores.append(score)
            except Exception as exc:
                cell_scores.append(0.0)
                errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
        rows.append(dict(name=name, train_passed=exact, train_total=len(TRAIN),
                         mean_cell_accuracy=sum(cell_scores) / len(TRAIN) if TRAIN else 0.0,
                         errors=errors[:2]))
    truncated = next(source, None) is not None
    rows.sort(key=lambda row: (-row["train_passed"], -row["mean_cell_accuracy"]))
    return dict(tested=len(rows), truncated=truncated, exact_fits=sum(r["train_passed"] == len(TRAIN) for r in rows),
                best=rows[:max_results], scoring="visible training pairs only; train fit is not test correctness")


def mismatch(got, want, detail=True):
    """One-line summary of how got differs from want (lists of lists), plus maps when detail."""
    if len(got) != len(want) or len(got[0]) != len(want[0]):
        return f"shape {len(got)}x{len(got[0])}, expected {len(want)}x{len(want[0])}"
    wrong = [(r, c, got[r][c], want[r][c]) for r in range(len(want)) for c in range(len(want[0]))
             if got[r][c] != want[r][c]]
    text = f"{len(wrong)} wrong cells: " + ", ".join(f"({r},{c}) got {g} want {w}" for r, c, g, w in wrong[:8])
    if len(wrong) > 8:
        text += ", ..."
    if detail:
        marks = "\n".join(GRID_SEP.join("." if g == w else str(w) for g, w in zip(gr, wr)) for gr, wr in zip(got, want))
        text += f"\nyour output:\n{fmt(got)}\nexpected values at wrong cells ('.' = correct):\n{marks}"
    return text


def check(fn):
    """Run fn on every training input; print PASS or the mismatch per pair. True if all pass."""
    report = _memorization_module.inspect_solver(fn, TRAIN, np=np)
    if report["blocked"]:
        MEMORIZATION_REJECTIONS.append(report)
        raise ValueError(_memorization_module.MESSAGE)
    passed, detailed = 0, False
    for i, pair in enumerate(TRAIN):
        try:
            got = to_grid(fn(np.array(pair["input"])))
        except Exception as exc:  # noqa: BLE001 - feedback for the model
            print(f"pair {i}: ERROR {type(exc).__name__}: {exc}")
            continue
        if got == pair["output"]:
            passed += 1
            print(f"pair {i}: PASS")
        else:
            print(f"pair {i}: FAIL {mismatch(got, pair['output'], detail=not detailed)}")
            if TRAIN_MISMATCH_IMAGES:
                training_mismatch_view(i, pair, got)
            detailed = True
    print(f"{passed}/{len(TRAIN)} training pairs pass")
    # Current-cell telemetry, not a model-written claim or a source-text heuristic.
    # Hold bounded references until the cell ends so a later redefinition of solve
    # cannot be mistaken for the function that was actually checked.
    CHECKS.append((fn, passed))
    del CHECKS[:-32]
    return passed == len(TRAIN)


TRAIN = []  # pristine copies (lists) of the training pairs, set per job
TRAIN_ARRAYS = []
DISPLAY_STATE = {"suppressed": 0, "reviewed": False}
VISUALS = []
TRAIN_MISMATCH_IMAGES = False
GRID_SEP = ""  # " " when the controller writes spaced grids (--tir-repl-grid-text spaced)
CHECKS = []
MEMORIZATION_REJECTIONS = []


def training_mismatch_view(index, pair, got):
    """One bounded training-only diagnostic per cell. Never inspects a test answer."""
    if len(VISUALS) >= 2 or any(v.get("automatic_training_mismatch") for v in VISUALS):
        return
    want = pair["output"]
    same = (len(got), len(got[0])) == (len(want), len(want[0]))
    if same:
        wrong = [(r, c) for r in range(len(want)) for c in range(len(want[0])) if got[r][c] != want[r][c]]
        if not wrong:
            return
        r0 = max(0, min(r for r, _ in wrong) - 2)
        c0 = max(0, min(c for _, c in wrong) - 2)
        r1, c1 = min(len(want), r0 + 16), min(len(want[0]), c0 + 16)
        crop = lambda grid: [row[c0:c1] for row in grid[r0:r1]]
        panels = [crop(got), crop(want)]
        order = "prediction, expected training output"
        if (len(pair["input"]), len(pair["input"][0])) == (len(want), len(want[0])):
            panels.insert(0, crop(pair["input"]))
            order = "training input, prediction, expected training output"
        reason = (f"Visible training pair {index}, crop rows [{r0},{r1}), columns [{c0},{c1}); "
                  f"panels left to right: {order}. {len(wrong)} wrong cells in the full output.")
    else:
        panels = [pair["input"], got, want]
        reason = (f"Visible training pair {index}, shape mismatch; full panels left to right: "
                  "training input, prediction, expected training output.")
    VISUALS.append(dict(reason=reason, grids=panels, automatic_training_mismatch=True))
    print("[automatic training mismatch image queued: " + reason + "]")


def namespace(task, structure_tools=False, visual_tools=False, train_mismatch_images=False, grid_sep=""):
    global TRAIN_MISMATCH_IMAGES, GRID_SEP
    TRAIN_MISMATCH_IMAGES = bool(visual_tools and train_mismatch_images)
    GRID_SEP = grid_sep
    import functools
    import itertools
    import math
    from collections import Counter, defaultdict, deque
    TRAIN[:] = [{"input": p["input"], "output": p["output"]} for p in task["train"]]
    TRAIN_ARRAYS[:] = [(f"train[{i}]['{kind}']", np.array(p[kind]))
                      for i, p in enumerate(task["train"]) for kind in ("input", "output")]
    ns = {
        "__name__": "__main__", "np": np, "numpy": np, "ndi": ndi, "math": math, "itertools": itertools,
        "functools": functools, "Counter": Counter, "defaultdict": defaultdict, "deque": deque,
        "show": show, "print": display_print, "objects": objects, "check": check,
        "train": [{k: np.array(p[k]) for k in ("input", "output")} for p in task["train"]],
        "test_inputs": [np.array(p["input"]) for p in task["test"]],
    }
    if structure_tools:
        ns.update(describe_grid=describe_grid, grid_diff=grid_diff, search_candidates=search_candidates)
    if visual_tools:
        ns["view"] = view
    return ns


def restore_task_data(ns, task):
    """Reserved input bindings survive accidental rebinding/mutation; user scratch state does not reset.

    Apply at every cell boundary, including journal replay, so the restored state seen by the model
    is the state reconstructed on its next call. Only public task data is used here.
    """
    def same_grid(value, grid):
        return (type(value) is np.ndarray and np.issubdtype(value.dtype, np.integer)
                and np.array_equal(value, grid))

    def same_train(value):
        return (type(value) is list and len(value) == len(task["train"])
                and all(type(pair) is dict and set(pair) == {"input", "output"}
                        and all(same_grid(pair[k], want[k]) for k in ("input", "output"))
                        for pair, want in zip(value, task["train"])))

    def same_tests(value):
        return (type(value) is list and len(value) == len(task["test"])
                and all(same_grid(grid, pair["input"]) for grid, pair in zip(value, task["test"])))

    restored = []
    for key, matches in (("train", same_train), ("test_inputs", same_tests)):
        try:
            unchanged = matches(ns.get(key))
        except Exception:  # malformed user data is replaced, not trusted
            unchanged = False
        if not unchanged:
            ns[key] = ([{k: np.array(p[k]) for k in ("input", "output")} for p in task["train"]]
                       if key == "train" else [np.array(p["input"]) for p in task["test"]])
            restored.append(key)
    return restored


# ----- execution -------------------------------------------------------------------------


def error_text(exc):
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    kept = [line for line in lines if Path(__file__).name not in line]
    text = "".join(kept).strip()
    return text if len(text) <= 2500 else "...\n" + text[-2500:]


def run_cell(code, ns, number, timeout):
    """Exec a cell, displaying the value of a trailing expression. Raises like the code does."""
    CHECKS.clear()  # Also clear on a syntax error, before the journal's last check can leak.
    MEMORIZATION_REJECTIONS.clear()
    name = f"<cell {number}>"
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)
    tree = ast.parse(code, filename=name)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    DISPLAY_STATE.update(suppressed=0, reviewed=False)
    VISUALS.clear()
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        exec(compile(tree, name, "exec"), ns)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), name, "eval"), ns)
            if value is not None:
                print(repr(display_value(value)))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def candidate_identity(fn, journal):
    """Conservative identity of code plus referenced helper/data state, never just predictions.

    This is a no-op optimization, not a general Python equivalence test. Dynamic namespace
    access, custom objects, module mutations and oversized state disable reuse. Line numbers
    and unrelated new functions do not change a solver's identity. Unknown means re-audit.
    """
    dynamic = {"globals", "locals", "vars", "eval", "exec", "compile", "getattr", "setattr",
               "delattr", "__import__"}
    libraries = {"builtins", "numpy", "scipy", "math", "cmath", "itertools", "functools",
                 "collections", "operator", "heapq", "bisect", "copy", "statistics",
                 "fractions", "decimal", "re"}
    active, references, count = set(), {}, [0]

    def unsupported(reason):
        raise ValueError(reason)

    def code_names(code):
        names = set(code.co_names)
        for const in code.co_consts:
            if type(const) is types.CodeType:
                names.update(code_names(const))
        return names

    def global_names(code):
        names = {i.argval for i in dis.get_instructions(code)
                 if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
        for const in code.co_consts:
            if type(const) is types.CodeType:
                names.update(global_names(const))
        return names

    def library(value):
        module, name = value.__module__, value.__qualname__
        if module.split(".")[0] not in libraries or "<" in name:
            unsupported("unknown external callable")
        return [module, name]

    def constant(value):
        # Compiler interning of co_consts is not user-data aliasing. Otherwise an
        # identical definition in a later cell can get a different fingerprint.
        if type(value) is tuple:
            return ["tuple", [constant(v) for v in value]]
        if type(value) is frozenset:
            return ["frozenset", sorted([constant(v) for v in value], key=repr)]
        return freeze(value)

    def freeze(value, attributes=frozenset()):
        count[0] += 1
        if count[0] > 20000:
            unsupported("dependency graph too large")
        typ = type(value)
        if typ in (str, bytes, list, tuple, dict, set, frozenset) and len(value) > 100_000:
            unsupported("dependency value too large")
        if value is None or typ in (bool, int, str):
            return [typ.__name__, value]
        if typ is float:
            return ["float", value.hex()]
        if typ is complex:
            return ["complex", value.real.hex(), value.imag.hex()]
        if typ is bytes:
            return ["bytes", value.hex()]
        if value is Ellipsis:
            return ["ellipsis"]
        if typ is types.CodeType:
            return ["code", value.co_code.hex(), value.co_argcount, value.co_posonlyargcount,
                    value.co_kwonlyargcount, value.co_flags, list(value.co_names),
                    list(value.co_varnames), list(value.co_freevars), list(value.co_cellvars),
                    [constant(c) for c in value.co_consts], value.co_exceptiontable.hex()]
        if id(value) in active:
            unsupported("cyclic dependency")
        if typ in (list, tuple, dict, set, frozenset, np.ndarray):
            if id(value) in references:
                return ["reference", references[id(value)]]
            references[id(value)] = len(references)
        active.add(id(value))
        try:
            if typ in (tuple, list):
                return [typ.__name__, [freeze(v, attributes) for v in value]]
            if typ is dict:
                return ["dict", [[freeze(k), freeze(v, attributes)] for k, v in value.items()]]
            if typ in (set, frozenset):
                return [typ.__name__, sorted([freeze(v, attributes) for v in value], key=repr)]
            if typ is np.ndarray:
                if (value.dtype.hasobject or value.dtype.metadata or value.nbytes > 1_000_000
                        or value.base is not None):
                    unsupported("unsupported array state")
                return ["array", value.dtype.str, freeze(value.dtype.descr), list(value.shape),
                        list(value.strides), bool(value.flags.writeable),
                        hashlib.sha256(value.tobytes()).hexdigest()]
            if isinstance(value, np.generic):
                return ["numpy-scalar", value.dtype.str, freeze(value.item())]
            if typ is types.ModuleType:
                name = value.__name__
                if name.split(".")[0] not in libraries or "random" in name.split("."):
                    unsupported("stateful or unknown module")
                # Only statically referenced attributes; dynamic lookup is disallowed below.
                return ["module", name, [[k, freeze(vars(value)[k], attributes)]
                        for k in sorted(attributes) if k in vars(value) and not k.startswith("__")]]
            if typ is types.FunctionType:
                if value.__code__.co_filename == __file__:
                    return ["runtime-helper", value.__name__, freeze(value.__code__)]
                if not value.__code__.co_filename.startswith("<cell "):
                    return ["library-function", library(value), freeze(value.__code__)]
                names = code_names(value.__code__)
                if names & dynamic or any(n.startswith("__") for n in names):
                    unsupported("dynamic namespace access")
                dependencies = []
                for name in sorted(global_names(value.__code__)):
                    scope = value.__globals__ if name in value.__globals__ else value.__builtins__
                    if name not in scope:
                        unsupported("unresolved global")
                    dependencies.append([name, freeze(scope[name], names)])
                codes = [value.__code__]
                for code in codes:
                    codes.extend(c for c in code.co_consts if type(c) is types.CodeType)
                    for inst in dis.get_instructions(code):
                        if inst.opname == "IMPORT_NAME":
                            if inst.argval not in sys.modules:
                                unsupported("unloaded import")
                            dependencies.append(["import:" + inst.argval,
                                                 freeze(sys.modules[inst.argval], names)])
                return ["function", freeze(value.__code__), freeze(value.__defaults__, names),
                        freeze(value.__kwdefaults__, names),
                        [freeze(c.cell_contents, names) for c in (value.__closure__ or ())], dependencies]
            if typ in (types.BuiltinFunctionType, types.BuiltinMethodType):
                owner = value.__self__
                if owner is not None and type(owner) is not types.ModuleType:
                    unsupported("bound mutable callable")
                return ["builtin", library(value)]
            if typ is type:
                return ["library-type", library(value)]
            if typ is np.ufunc:
                return ["numpy-ufunc", value.__name__]
            if typ is type(np.array_equal) and typ.__module__.startswith("numpy"):
                return ["numpy-dispatcher", library(value)]
            unsupported("unsupported dependency type: " + typ.__module__ + "." + typ.__name__)
        finally:
            active.remove(id(value))

    try:
        for cell in journal:
            try:
                tree = ast.parse(cell)
            except SyntaxError:
                continue  # A syntax-error cell did not run.
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and
                        (isinstance(node.ctx, (ast.Store, ast.Del)) or node.attr.startswith("__"))):
                    unsupported("journal may mutate external state")
                if isinstance(node, ast.Name) and (node.id in dynamic or node.id.startswith("__")):
                    unsupported("journal uses dynamic namespace access")
                if (isinstance(node, ast.Attribute)
                        and node.attr in {"seterr", "seterrcall", "set_numeric_ops"}):
                    unsupported("journal changes library semantics")
        data = json.dumps(freeze(fn), separators=(",", ":"), ensure_ascii=True).encode()
        if len(data) > 2_000_000:
            unsupported("serialized dependency graph too large")
        return {"digest": hashlib.sha256(data).hexdigest(), "reason": None, "version": 1}
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        return {"digest": None, "reason": str(exc)[:160], "version": 1}


def evaluate(fn_name, ns, task, timeout, journal=()):
    fn = ns.get(fn_name)
    if not callable(fn):
        return {"error": f"{fn_name} is not defined as a function in the python session"}
    # TRAIN is the pristine visible task, including for a holdout-input probe.
    # Never use hidden test labels, and never execute a known lookup shortcut.
    guard = _memorization_module.inspect_solver(fn, TRAIN, np=np)
    if guard["blocked"]:
        return {"error": _memorization_module.MESSAGE, "memorization_guard": guard}
    out = {"error": None}
    try:
        out["source"] = inspect.getsource(fn)
    except (OSError, TypeError):
        out["source"] = None
    before = candidate_identity(fn, journal)
    for split, grids in (("train", [p["input"] for p in task["train"]]), ("test", [p["input"] for p in task["test"]])):
        cases = out[split] = []
        for grid in grids:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                cases.append({"grid": to_grid(fn(np.array(grid))), "error": None})
            except BaseException as exc:  # noqa: BLE001
                cases.append({"grid": None, "error": f"{type(exc).__name__}: {exc}"[:600]})
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    # Stateful solvers must not reuse an audit merely because one invocation matched.
    after = candidate_identity(fn, journal)
    out["identity"] = (before if before == after else
                       {"digest": None, "reason": "evaluation mutated dependencies", "version": 1})
    return out


def measure_inputs(fn_name, ns, task, timeout):
    fn = ns.get(fn_name)
    if not callable(fn):
        return {"error": f"{fn_name} is not defined as a function"}
    rows = []

    def plain(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    for split in ("train", "test"):
        for index, pair in enumerate(task[split]):
            row = dict(split=split, index=index, values=None, error=None)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                row["values"] = normalize_measurement(plain(fn(np.array(pair["input"]))))
            except BaseException as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:600]
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                restore_task_data(ns, task)
            rows.append(row)
    return dict(error=None, measurements=rows)


def checked_alternatives(ns, task, timeout, preferred_name):
    """Bounded current-cell capture, never trust a printed PASS or a rebound check."""
    preferred = ns.get(preferred_name)
    # Keep object identity: check(alt) followed by redefining alt is not a check of new alt.
    candidates, seen = [], set()
    for fn, passed in list(CHECKS):
        if passed != len(task["train"]) or fn is preferred or id(fn) in seen:
            continue
        names = [name for name, value in ns.items() if value is fn and name.isidentifier()
                 and not name.startswith("_") and name != preferred_name]
        if names:
            seen.add(id(fn))
            candidates.append(names[0])
    results = []
    for name in candidates[:2]:
        # Re-run all training and test inputs, independent of check's reported pass count.
        restore_task_data(ns, task)
        results.append(dict(function=name, **evaluate(name, ns, task, timeout)))
    return results


def run_job(task, journal, op, timeout, probe_input=None, structure_tools=False, visual_tools=False,
            train_mismatch_images=False, capture_checked_alternatives=None, execution_trace=None,
            measurement_candidate=None, journal_test_counts=None, max_output=MAX_OUTPUT, auto_repair=False,
            grid_sep=""):
    """Replay visible-task cells, then execute a cell, evaluate, or probe one input without its answer.

    auto_repair: a failed cell also reports the public names it started from, for the parent's typo repair.
    """
    import random
    random.seed(0)  # replays must rebuild the same state the model saw
    np.random.seed(0)
    def replay_task(number):
        return (dict(train=task["train"], test=task["test"][:journal_test_counts[number]])
                if journal_test_counts is not None else task)

    # Old cells must see exactly the test-input prefix available when executed.
    # Otherwise revealing an input would silently rewrite cached helper state.
    ns = namespace(replay_task(0) if journal else task, structure_tools, visual_tools, train_mismatch_images,
                   grid_sep)
    start = time.monotonic()
    with contextlib.redirect_stdout(Null()), contextlib.redirect_stderr(Null()):
        for number, code in enumerate(journal):
            visible = replay_task(number)
            if journal_test_counts is not None:
                restore_task_data(ns, visible)
            try:
                run_cell(code, ns, number, timeout)
            except BaseException:  # noqa: BLE001 - the model already saw this error
                pass
            finally:
                restore_task_data(ns, visible)
    if journal_test_counts is not None:
        restore_task_data(ns, task)
    replay = time.monotonic() - start
    start = time.monotonic()
    if op[0] in ("cell", "check"):
        sink = Sink(max_output)
        names = sorted(k for k in ns if not k.startswith("_")) if auto_repair and op[0] == "cell" else None
        result = {"error": None, "timeout": False}
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                if op[0] == "cell":
                    run_cell(op[1], ns, len(journal), timeout)
                else:
                    # Controller-owned training-only check. A model cell may have rebound
                    # ns['check']; do not call that binding or count journal-replay checks.
                    CHECKS.clear()
                    MEMORIZATION_REJECTIONS.clear()
                    VISUALS.clear()
                    DISPLAY_STATE.update(suppressed=0, reviewed=False)
                    solver = ns.get(op[1])
                    if not callable(solver):
                        raise ValueError(f"{op[1]} is not defined as a function")
                    signal.setitimer(signal.ITIMER_REAL, timeout)
                    try:
                        check(solver)
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
            except TimeLimit:
                result["timeout"] = True
            except BaseException as exc:  # noqa: BLE001
                result["error"] = error_text(exc)
            finally:
                result["restored_inputs"] = restore_task_data(ns, task)
        result["output"] = sink.text()
        if names is not None and result["error"]:
            result["names"] = names
        solver = ns.get("solve")
        result["solver_check"] = {
            "defined": callable(solver),
            "checked": any(fn is solver for fn, _ in CHECKS),
            "train_passed": next((passed for fn, passed in reversed(CHECKS) if fn is solver), None),
            "train_total": len(task["train"]),
        }
        if MEMORIZATION_REJECTIONS:
            result["memorization_rejections"] = list(MEMORIZATION_REJECTIONS)
            result["solver_check"]["memorization_guard"] = MEMORIZATION_REJECTIONS[-1]
        if visual_tools:
            result["visuals"] = list(VISUALS)  # Current cell only; replay images are not resent.
        result["suppressed_training_grids"] = DISPLAY_STATE["suppressed"]
        if DISPLAY_STATE["suppressed"]:
            result["output"] += ("\n[display guard] Full unchanged training grids were replaced with references; "
                                 "the original prompt still contains them. Measure a property, inspect a crop "
                                 "or use check(solve) for mismatches. If necessary, show(grid, reason='specific "
                                 "question') permits one targeted full-grid review per cell.\n")
        if capture_checked_alternatives and not result["error"] and not result["timeout"]:
            with contextlib.redirect_stdout(Null()), contextlib.redirect_stderr(Null()):
                result["checked_alternatives"] = checked_alternatives(
                    ns, task, timeout, capture_checked_alternatives)
    else:
        with contextlib.redirect_stdout(Null()), contextlib.redirect_stderr(Null()):
            # Journal and helpers see ONLY the original visible task. A probe is an additional
            # isolated function invocation, not a replacement task or a new test_inputs entry.
            evaluation_task = {"train": [], "test": [{"input": probe_input}]} if op[0] == "probe" else task
            if op[0] == "trace":
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    result = _execution_module.trace_execution(
                        op[1], ns, task, execution_trace, np=np, to_grid=to_grid)
                    result["identity"] = candidate_identity(ns.get(op[1]), journal)
                except BaseException as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"[:600], "complete": False}
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            elif op[0] == "measure":
                before = candidate_identity(ns.get(measurement_candidate), journal) if measurement_candidate else None
                result = measure_inputs(op[1], ns, evaluation_task, timeout)
                if measurement_candidate:
                    after = candidate_identity(ns.get(measurement_candidate), journal)
                    result["candidate_identity"] = (before if before == after else {
                        "digest": None, "reason": "measurement mutated solver dependencies", "version": 1})
            else:
                result = evaluate(op[1], ns, evaluation_task, timeout, journal)
    result.update(replay_seconds=round(replay, 3), seconds=round(time.monotonic() - start, 3))
    return result


def check_request(request):
    if (not isinstance(request, dict)
            or set(request) - {"cpus", "probe_input", "structure_tools", "visual_tools", "train_mismatch_images",
                               "capture_checked_alternatives", "execution_trace", "measurement_candidate",
                               "journal_test_counts", "max_output", "auto_repair", "grid_sep"}
            != {"task", "journal", "op", "timeout", "memory_mib", "cpu_seconds"}):
        raise ValueError("Runtime accepts task, journal, op and limits only")
    if type(request.get("structure_tools", False)) is not bool:
        raise ValueError("structure_tools must be boolean")
    if type(request.get("visual_tools", False)) is not bool:
        raise ValueError("visual_tools must be boolean")
    if type(request.get("train_mismatch_images", False)) is not bool:
        raise ValueError("train_mismatch_images must be boolean")
    if request.get("train_mismatch_images") and not request.get("visual_tools"):
        raise ValueError("train_mismatch_images requires visual_tools")
    if type(request.get("auto_repair", False)) is not bool:
        raise ValueError("auto_repair must be boolean")
    if request.get("grid_sep", "") not in ("", " "):
        raise ValueError("grid_sep must be '' or ' '")
    max_output = request.get("max_output", MAX_OUTPUT)
    if type(max_output) is not int or not 200 <= max_output <= 200_000:
        raise ValueError("max_output must be an integer in 200..200000")
    cpus = request.get("cpus", [])
    if not isinstance(cpus, list) or not all(type(c) is int and 0 <= c < 4096 for c in cpus):
        raise ValueError("cpus must be a list of CPU numbers")
    task, journal, op = request["task"], request["journal"], request["op"]
    if not isinstance(task, dict) or set(task) != {"train", "test"}:
        raise ValueError("Task must hold train and test only")
    if any(set(p) != {"input", "output"} for p in task["train"]) or any(set(p) != {"input"} for p in task["test"]):
        raise ValueError("Train pairs hold input and output; test pairs hold input only")
    if not isinstance(journal, list) or not all(isinstance(code, str) for code in journal):
        raise ValueError("Journal must be a list of cells")
    counts = request.get("journal_test_counts")
    if "journal_test_counts" in request and (
            not isinstance(counts, list) or len(counts) != len(journal)
            or any(type(n) is not int or not 1 <= n <= len(task["test"]) for n in counts)
            or counts != sorted(counts)):
        raise ValueError("journal_test_counts must be monotone visible-input counts, one per cell")
    if (not isinstance(op, list) or len(op) != 2 or op[0] not in ("cell", "check", "eval", "probe", "measure", "trace")
            or not isinstance(op[1], str)):
        raise ValueError("op must be cell, check, eval, probe, measure or trace with a code/name string")
    if (op[0] == "trace") != ("execution_trace" in request):
        raise ValueError("execution_trace is required for trace and forbidden for other operations")
    if op[0] == "trace":
        _execution_module.validate_spec(request["execution_trace"], task)
    if "measurement_candidate" in request and (op[0] != "measure"
            or not isinstance(request["measurement_candidate"], str)
            or not request["measurement_candidate"].isidentifier()):
        raise ValueError("measurement_candidate requires a function name and a measure operation")
    capture = request.get("capture_checked_alternatives")
    if capture is not None and (op[0] != "cell" or not isinstance(capture, str) or not capture.isidentifier()):
        raise ValueError("capture_checked_alternatives requires a preferred function name and a cell operation")
    if op[0] == "check" and op[1] != "solve":
        raise ValueError("The provisional training check accepts solve only")
    if (op[0] == "probe") != ("probe_input" in request):
        raise ValueError("probe_input is required for probe and forbidden for other operations")
    if "probe_input" in request:
        grid = request["probe_input"]
        if (not isinstance(grid, list) or not 1 <= len(grid) <= 30
                or not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30
                or any(not isinstance(row, list) or len(row) != len(grid[0])
                       or any(type(v) is not int or not 0 <= v <= 9 for v in row) for row in grid)):
            raise ValueError("probe_input must be an ARC input grid, never a pair or answer")
    for key, low, high in (("timeout", 0.1, 600), ("memory_mib", 256, 65536), ("cpu_seconds", 1, 7200)):
        value = request[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"Invalid {key}")


def main():
    report = {"sandbox_enabled": False, "fatal": None}
    out = 1
    try:
        request = json.loads(sys.stdin.buffer.read(MAX_REQUEST))
        check_request(request)
        if request.get("cpus"):
            # Keep cells off the cores the model server is pinned to.
            os.sched_setaffinity(0, request["cpus"])
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        cpu = int(request["cpu_seconds"])
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        global np, ndi
        np, ndi = preload()
        if request.get("auto_repair"):
            sys.modules["scipy"].ndi = ndi  # `from scipy import ndi`: our preloaded alias, written as an import
        # The result goes to a private copy of stdout; fds 1 and 2 now point at /dev/null so
        # stray low-level writes from cells cannot corrupt the JSON line.
        out = os.dup(1)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        os.close(null)
        memory = int(request["memory_mib"]) * 1024 ** 2
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        signal.signal(signal.SIGALRM, _alarm)
        restrict()
        report["sandbox_enabled"] = True
        report.update(run_job(request["task"], request["journal"], request["op"], float(request["timeout"]),
                              request.get("probe_input"), request.get("structure_tools", False),
                              request.get("visual_tools", False), request.get("train_mismatch_images", False),
                              request.get("capture_checked_alternatives"), request.get("execution_trace"),
                              request.get("measurement_candidate"), request.get("journal_test_counts"),
                              request.get("max_output", MAX_OUTPUT), request.get("auto_repair", False),
                              request.get("grid_sep", "")))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, never to the model
        report["fatal"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
    try:
        data = (json.dumps(report, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        data = (json.dumps({"sandbox_enabled": report["sandbox_enabled"], "fatal": f"unserialisable result: {exc}"}) + "\n").encode()
    view = memoryview(data)
    while view:
        view = view[os.write(out, view):]
    os._exit(0)


np = ndi = None

if __name__ == "__main__":
    main()
