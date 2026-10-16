"""The `python` tool: run an exploratory snippet with the task preloaded.

Each call is a fresh subprocess (nothing persists between calls, as in the
Tufa duck harness), so a crash or an infinite loop costs one call. The task is
injected as numpy arrays plus a few ARC helpers; stdout is captured and capped.

Optional function retention (dfranzen's ARC-AGI-3 fork of the duck harness): with a
store, plain top-level functions a call defines are kept as SOURCE and replayed into
later calls; variables never persist. Old transcripts show why: 30-38% of all
function code the model wrote re-defined a function it had already written, and
19 of 20 ab3q36 submits re-typed functions first written in a python call.
"""
import json
import os
import subprocess
import sys

HELPERS = r'''
import numpy as np
from collections import Counter, defaultdict, deque
import itertools, math, functools, copy, re

def show(g):
    """Digit rows, one line per row."""
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(g))

def diff(a, b):
    """Cells where a and b differ, as (row, col, a_value, b_value); size mismatch -> string."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape mismatch {a.shape} vs {b.shape}"
    return [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in zip(*np.nonzero(a != b))]

def objects(g, bg=None, diagonal=False, multicolor=False):
    """Connected components. bg defaults to the most common colour.
    Returns dicts: color (or colors), cells [(r,c)], bbox (r0,c0,r1,c1), size, shape (cropped mask)."""
    g = np.asarray(g)
    if bg is None:
        vals, counts = np.unique(g, return_counts=True)
        bg = int(vals[np.argmax(counts)])
    H, W = g.shape
    seen = np.zeros_like(g, dtype=bool)
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)] + ([(-1,-1),(-1,1),(1,-1),(1,1)] if diagonal else [])
    out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            col = g[r, c]
            q, cells = deque([(r, c)]), []
            seen[r, c] = True
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] != bg \
                            and (multicolor or g[ny, nx] == col):
                        seen[ny, nx] = True; q.append((ny, nx))
            ys, xs = zip(*cells)
            r0, c0, r1, c1 = min(ys), min(xs), max(ys), max(xs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            for y, x in cells:
                mask[y - r0, x - c0] = g[y, x]
            d = {"cells": cells, "bbox": (r0, c0, r1, c1), "size": len(cells), "shape": mask}
            if multicolor:
                d["colors"] = sorted({int(g[y, x]) for y, x in cells})
            else:
                d["color"] = int(col)
            out.append(d)
    return out
'''

RUNNER = r'''
import sys, json, io, contextlib, traceback
job = json.loads(sys.stdin.read())
ns = {"__name__": "__arc_repl__"}
exec(job["helpers"], ns)
import numpy as np
ns["train_inputs"] = [np.array(g) for g in job["train_inputs"]]
ns["train_outputs"] = [np.array(g) for g in job["train_outputs"]]
ns["test_inputs"] = [np.array(g) for g in job["test_inputs"]]
ns["train"] = list(zip(ns["train_inputs"], ns["train_outputs"]))
keep = job.get("retained")                # None: retention off (each call starts fresh)
preloaded = set(ns)
restore_failed = []
if keep is not None:
    # Replay what earlier calls kept: their imports, then their function definitions.
    # Source only -- no variable or array is ever carried over.
    for src in keep["imports"]:
        try:
            exec(compile(src, "<kept>", "exec"), ns)
        except Exception:
            pass
    for name, src in keep["defs"].items():
        try:
            exec(compile(src, "<kept>", "exec"), ns)
        except Exception:
            restore_failed.append(name)
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(compile(job["code"], "<snippet>", "exec"), ns)
        if "result" in ns and ns["result"] is not None:
            print(repr(ns["result"]))
except Exception:
    tb = traceback.format_exc().strip().splitlines()
    buf.write("\n" + "\n".join(tb[-4:]))
sys.stdout.write("@@REPL@@" + json.dumps(buf.getvalue()) + "\n")
if keep is not None:
    import ast, builtins, dis, types
    defs, imports, rejected = {}, [], {}
    try:
        tree = ast.parse(job["code"])
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    bound = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not getattr(node, "level", 0):
            src = ast.get_source_segment(job["code"], node)
            if src:
                imports.append(src)
                bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
    new = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    available = (set(dir(builtins)) | preloaded | set(keep["defs"]) | bound
                 | {n.name for n in new} | set(keep.get("bound", [])))
    for node in new:
        fn = ns.get(node.name)
        # only a def that actually ran and still holds the name (not reassigned later)
        if not (isinstance(fn, types.FunctionType) and fn.__code__.co_filename == "<snippet>"
                and fn.__code__.co_firstlineno == node.lineno):
            continue
        reads, todo = set(), [fn.__code__]
        while todo:
            co = todo.pop()
            reads |= {i.argval for i in dis.get_instructions(co)
                      if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
            todo += [c for c in co.co_consts if isinstance(c, types.CodeType)]
        missing = sorted(r for r in reads if isinstance(r, str) and r not in available)
        if missing:
            rejected[node.name] = missing
        else:
            defs[node.name] = ast.get_source_segment(job["code"], node)
    sys.stdout.write("@@KEPT@@" + json.dumps({"defs": defs, "imports": imports, "bound": sorted(bound),
                                             "rejected": rejected,
                                             "restore_failed": restore_failed}) + "\n")
'''

PREAMBLE_DOC = (
    "Preloaded in every `python` call (each call starts fresh; nothing persists): "
    "`np`, `train_inputs`, `train_outputs`, `test_inputs` (lists of numpy int arrays), "
    "`train` (list of (input, output) pairs), and helpers `show(g)` (digit rows), "
    "`diff(a, b)` (differing cells as (row, col, a, b)), "
    "`objects(g, bg=None, diagonal=False, multicolor=False)` (connected components with "
    "color, cells, bbox, size, shape). Print compact summaries, never whole large grids."
)


RETAIN_DOC = (
    "Variables do not persist between calls, but every function you define with a plain "
    "top-level `def` is kept: later `python` calls and `submit` can call it without "
    "redefining it. A kept function may use builtins, the preloaded names, imports and "
    "other kept functions; pass anything else in as an argument.")


def new_store():
    """Per-attempt record of what python calls have kept (source only)."""
    return {"defs": {}, "imports": [], "bound": []}


def run_snippet(code, task, timeout=30.0, max_chars=4000, mem_bytes=2 << 30, store=None):
    """Run one snippet. With `store` (from new_store()), functions defined by earlier calls
    are replayed first and the ones this call defines are added to the store; a line at
    the end of the output says what was kept."""
    job = {"code": code, "helpers": HELPERS,
           "train_inputs": [p["input"] for p in task["train"]],
           "train_outputs": [p["output"] for p in task["train"]],
           "test_inputs": [t["input"] for t in task["test"]]}
    if store is not None:
        job["retained"] = store
    pre = None
    if os.name == "posix":
        def pre():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except Exception:
                pass
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps(job),
                           capture_output=True, text=True, timeout=timeout, preexec_fn=pre,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout:.0f}s]"
    out, kept = None, None
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("@@KEPT@@") and kept is None:
            kept = json.loads(line[len("@@KEPT@@"):])
        if line.startswith("@@REPL@@"):
            out = json.loads(line[len("@@REPL@@"):])
            break
    if out is None:
        out = "[snippet crashed] " + " | ".join((p.stderr or "").strip().splitlines()[-2:])
    out = out.strip() or "[no output -- print() what you want to see]"
    if len(out) > max_chars:
        out = out[:max_chars] + f"\n... [truncated {len(out) - max_chars} chars]"
    if store is not None and kept is not None:
        out += _absorb(store, kept)
    return out


def _absorb(store, kept):
    """Merge one call's kept definitions into the store; return the note for the model."""
    for name in kept["restore_failed"]:
        store["defs"].pop(name, None)
    for src in kept["imports"]:
        if src not in store["imports"]:
            store["imports"].append(src)
    store["bound"] = sorted(set(store["bound"]) | set(kept["bound"]))
    notes = []
    for name, src in kept["defs"].items():
        if store["defs"].get(name) != src:
            store["defs"].pop(name, None)            # re-insert: newest definition last
            store["defs"][name] = src
    if kept["defs"]:
        notes.append("kept for later calls: " + ", ".join(sorted(kept["defs"])))
    for name, missing in kept["rejected"].items():
        notes.append(f"not kept: {name} (uses {', '.join(missing[:4])} from this snippet -- "
                     "pass it as an argument to keep the function)")
    if kept["restore_failed"]:
        notes.append("dropped (failed to reload): " + ", ".join(kept["restore_failed"]))
    return ("\n[" + "; ".join(notes) + "]") if notes else ""


def with_kept(code, store):
    """A submitted program plus the kept functions (and imports) it relies on.

    Returns (full_source, names_added). The program's own definitions win; kept
    functions are added only if the program uses them, transitively, and transform()
    itself is taken from the store if the program does not define one."""
    import ast
    if not store or not store["defs"]:
        return code, []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, []
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}

    def used(src):
        try:
            return {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
        except SyntaxError:
            return set()

    need = (used(code) & set(store["defs"])) - defined
    if "transform" not in defined and "transform" in store["defs"]:
        need.add("transform")
    todo = list(need)
    while todo:
        for n in (used(store["defs"][todo.pop()]) & set(store["defs"])) - defined - need:
            need.add(n)
            todo.append(n)
    if not need:
        return code, []
    # Kept functions were written against the python tool's namespace (np, objects(), ...),
    # so the helper preamble goes in first; the program's own definitions still win.
    parts = ([HELPERS.strip()] + list(store["imports"])
             + [src for n, src in store["defs"].items() if n in need])
    return "\n".join(parts) + "\n\n" + code, sorted(need)
