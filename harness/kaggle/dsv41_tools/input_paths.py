"""Bounded input lookup for the Kaggle notebooks, and the check that enforces the rule (Manh, 2026-10-30): notebook,
cell and solver code never scans the input root (/kaggle/input, ARC_INPUT_ROOT on the pods) recursively. With his
datasets attached (CUDA 13 debs, vLLM wheels, sources), one recursive scan walks thousands of files: ~5 min per call.

- HELPER: cell 0's find_one / find_input / find_inputs. Exact candidates first, then the attached sources' mount dirs
  (INPUT_ROOTS, from kernel-metadata.json at build time), then fixed-depth globs (INPUT_DEPTHS) only.
- input_roots(meta): those mount dirs for a kernel-metadata.json dict.
- rewrite(text): the recursive lookups of earlier builds (Version 5's cells 0 and 2; cells 1b, 0b and 2 and the
  load-test line as copied into Manh's notebooks) replaced by the bounded helper; everything else is left as is.
- input_scans(text): the recursive scans of the input root in a piece of Python source (a cell, a solver, a builder).
- finish(nb, meta): every builder's last step on a notebook: rewrite, INPUT_ROOTS from the final metadata, and a
  build-time check that no code cell scans the input root.
"""
import ast
import json
import re

# Kaggle's mounts (smoke logs 2026-10-24..29): datasets/<owner>/<slug>/..., competitions/<slug>/...,
# notebooks/<owner>/<slug>/... (kernel outputs, e.g. the fork wheel set in .../wheels/),
# models/<owner>/<model>/<framework, lower-cased>/<variation>/<version>/...; Manh's wheels sit two levels down in
# datasets/<owner>/vllm-wheels/wheels/wheels/. Older mounts and the pods (ARC_INPUT_ROOT) put <slug>/ at the top.
INPUT_DEPTHS = ("", "*", "*/*", "competitions/*", "datasets/*/*", "datasets/*/*/*", "notebooks/*/*", "notebooks/*/*/*",
                "models/*/*/*/*/*")

HELPER = '''\
# Inputs are found at exact paths and fixed depths only, never by a recursive scan of INPUT (with Manh's datasets
# attached one scan walks thousands of files, ~5 min on Kaggle; input_paths.py). find_input(name): the exact
# candidates given, then the attached sources' mount dirs (INPUT_ROOTS, from kernel-metadata.json at build time),
# then INPUT_DEPTHS: Kaggle mounts datasets/<owner>/<slug>, competitions/<slug>, notebooks/<owner>/<slug> and
# models/<owner>/<model>/<framework>/<variation>/<version>; older mounts and the pods use <slug>/ at the top.
INPUT_ROOTS = []
INPUT_DEPTHS = __DEPTHS__

def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None

def input_patterns(name, exact=()):
    dirs = [f"{INPUT}/{d}" for d in INPUT_ROOTS] + [f"{INPUT}/{d}" if d else INPUT for d in INPUT_DEPTHS]
    return list(dict.fromkeys(list(exact) + [f"{d}/{name}" for d in dirs]))

def find_input(name, *exact):
    """The first input file called `name` (glob characters only in the name), or None."""
    for pattern in input_patterns(name, exact):
        hit = find_one(pattern)
        if hit:
            return hit
    return None

def find_inputs(name, *exact):
    """Every input file called `name` at those places, sorted."""
    return sorted({p for pattern in input_patterns(name, exact) for p in glob.glob(pattern)})
'''.replace("__DEPTHS__", repr(INPUT_DEPTHS))

ROOTS_LINE = re.compile(r"^INPUT_ROOTS = \[.*\]$", re.M)

# Manh's vLLM wheel set (his cell 3 installs from exactly these two dirs)
MANH_WHEELS = ('f"{INPUT}/datasets/*/vllm-wheels/wheels/wheels/vllm-*.whl"',
               'f"{INPUT}/vllm-wheels/wheels/wheels/vllm-*.whl"')
# a cell that runs before cell 0 (settings cell 00) has no INPUT or helper: fixed depths under /kaggle/input
PRE_CELL0_DIRS = '("", "*/", "*/*/", "datasets/*/*/")'

LEGACY_FIND_ONE = ('def find_one(pattern):\n'
                   '    hits = sorted(glob.glob(pattern, recursive=True))\n'
                   '    return hits[0] if hits else None\n')
LEGACY = [
    # cell 0 (Version 5 / build.py before 2026-10-30, and its copies in Manh's notebooks)
    (LEGACY_FIND_ONE, HELPER),
    # cell 0b from a pins dataset
    ("find_one(f\"{INPUT}/**/{PINS['archive']}\")", 'find_input(PINS["archive"])'),
    # cell 2 (B32-M) as copied into Manh's notebooks
    ('        ENV_RECORD["vllm_wheels_in_inputs"] = sorted(os.path.basename(p) for p in\n'
     '                                                     glob.glob(f"{INPUT}/**/vllm-*.whl", recursive=True))\n',
     '        ENV_RECORD["vllm_wheels_in_inputs"] = sorted(os.path.basename(p) for p in find_inputs(\n'
     f'            "vllm-*.whl", {MANH_WHEELS[0]},\n'
     f'            {MANH_WHEELS[1]}))\n'),
    # the B32-M load test in cell 00 (runs before cell 0)
    ('    _lf = sorted(_lg.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True))\n',
     f'    _lf = sorted({{_p for _d in {PRE_CELL0_DIRS}\n'
     '                  for _p in _lg.glob(f"/kaggle/input/{_d}arc-agi_evaluation_challenges.json")})\n'),
]
# find_one(f"{INPUT}/**/<name>") (cells 0, 1b and 2) -> find_input("<name>")
LEGACY_CALL = re.compile(r'find_one\(f"\{INPUT\}/\*\*/([^"{}/]+)"\)')


def rewrite(text):
    """`text` with every legacy recursive input lookup replaced by the bounded helper (a cell 0 gets HELPER)."""
    for old, new in LEGACY:
        text = text.replace(old, new)
    return LEGACY_CALL.sub(lambda m: f'find_input("{m.group(1)}")', text)


def input_roots(meta):
    """The attached sources' mount dirs (relative to the input root) for a kernel-metadata.json dict."""
    roots = [f"competitions/{c}" for c in meta.get("competition_sources", [])]
    for key, kind in (("dataset_sources", "datasets"), ("kernel_sources", "notebooks")):
        roots += ["{}/{}".format(kind, "/".join(s.split("/")[:2])) for s in meta.get(key, [])]
    for m in meta.get("model_sources", []):  # owner/model/framework/variation[/version]; Kaggle lower-cases framework
        p = m.split("/")[:5]
        if len(p) > 2:
            p[2] = p[2].lower()
        roots.append("models/" + "/".join(p + ["*"] * (5 - len(p))))
    return list(dict.fromkeys(roots))


def with_roots(text, roots):
    """A cell 0 with its INPUT_ROOTS line set to `roots` (unchanged when it has none)."""
    return ROOTS_LINE.sub(lambda _m: "INPUT_ROOTS = " + json.dumps(roots), text, count=1)


_ROOT = re.compile(r"kaggle/input|\bINPUT\b|ARC_INPUT_ROOT")
# string literals and templates: the input root followed by ** (find_one(f"{INPUT}/**/x"), "/kaggle/input/**/x", ...)
_LITERAL = re.compile(r"""(?:/kaggle/input|\{(?:\w+\.)?INPUT\})[^"'\n]*\*\*""")
# Path("/kaggle/input").rglob(...), Path(INPUT).rglob(...), os.walk("/kaggle/input"), os.walk(INPUT)
_CALL = re.compile(r"""(?:Path\(\s*(?:["']/kaggle/input/?["']|(?:\w+\.)?INPUT)\s*\)\s*\.(?:rglob|walk)\(|"""
                   r"""os\.walk\(\s*(?:["']/kaggle/input/?["']|(?:\w+\.)?INPUT\b))""")
# shell: find / ls -R / du / tree over the input root or a whole mount category
_SHELL = re.compile(r"""\b(?:find|du|tree|ls(?:\s+-\w+)*\s+-\w*R\w*)(?:\s+-\w+)*\s+["']?"""
                    r"""(?:/kaggle/input|\$\{?INPUT\}?|\{INPUT\})(?:/(?:datasets|models|notebooks|competitions))?/?"""
                    r"""(?=["'\s;)|&]|$)""", re.M)


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append(_dotted(node.func) + "()")
    return ".".join(reversed(parts))


def _strings(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _ast_scans(text):
    """Recursive scans in parsed code: glob/iglob with recursive=True or a ** pattern, rglob, os.walk / Path.walk, and
    any helper call given a ** pattern, whose arguments or receiver name the input root directly or through a
    variable assigned from it (to a fixed point)."""
    src = "\n".join(re.sub(r"^(\s*)[!%].*$", r"\1pass", ln) for ln in text.split("\n"))  # IPython ! and % lines
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    lines = [ln.encode() for ln in src.split("\n")]  # ast offsets count UTF-8 bytes

    def segment(node):  # ast.get_source_segment re-splits the whole source on every call
        if getattr(node, "end_lineno", None) is None:
            return ""
        if node.lineno == node.end_lineno:
            seg = lines[node.lineno - 1][node.col_offset:node.end_col_offset]
        else:
            seg = b"\n".join([lines[node.lineno - 1][node.col_offset:]] + lines[node.lineno:node.end_lineno - 1]
                             + [lines[node.end_lineno - 1][:node.end_col_offset]])
        return seg.decode(errors="replace")

    # names assigned a value that names the root (one level: names are not scoped here, so no transitive closure)
    tainted = {"INPUT"}
    for n in ast.walk(tree):
        targets = (n.targets if isinstance(n, ast.Assign) else
                   [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) else [])
        if targets and getattr(n, "value", None) is not None and _ROOT.search(segment(n.value)):
            tainted |= {x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = _dotted(n.func)
        last = name.rsplit(".", 1)[-1]
        args = list(n.args) + [k.value for k in n.keywords]
        star = any("/**" in s or "**/" in s for a in n.args for s in _strings(a))
        rec = any(k.arg == "recursive" and not (isinstance(k.value, ast.Constant) and not k.value.value)
                  for k in n.keywords)
        if last in ("glob", "iglob"):
            scan = rec or star
        elif last == "rglob" or name in ("os.walk", "walk") or (last == "walk" and "Path" in name):
            scan = True
        else:
            scan = star
        if not scan:
            continue
        parts = args + ([n.func.value] if isinstance(n.func, ast.Attribute) else [])
        seg = " ".join(segment(p) for p in parts)
        if _ROOT.search(seg) or any(isinstance(x, ast.Name) and x.id in tainted for p in parts for x in ast.walk(p)):
            found.append(segment(n) or name)
    return found


def input_scans(text):
    """The recursive scans of the input root in `text` (Python source, a notebook cell or a builder with cell templates
    in string literals). Comment lines are ignored."""
    code = "\n".join(ln for ln in text.split("\n") if not ln.lstrip().startswith("#"))
    found = [m.group(0) for rx in (_LITERAL, _CALL, _SHELL) for m in rx.finditer(code)]
    return found + [s for s in _ast_scans(text) if not any(s in f or f in s for f in found)]


def notebook_scans(nb):
    """{code cell index: scans} for a notebook dict."""
    out = {}
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code":
            s = input_scans("".join(c["source"]))
            if s:
                out[i] = s
    return out


def finish(nb, meta):
    """A builder's last step: legacy lookups rewritten in every code cell, INPUT_ROOTS from the final metadata, and no
    code cell left that scans the input root (raises AssertionError)."""
    roots = input_roots(meta)
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            text = with_roots(rewrite("".join(c["source"])), roots)
            c["source"] = text.splitlines(keepends=True)
    bad = notebook_scans(nb)
    assert not bad, f"code cells scan the input root recursively (input_paths.py): {bad}"
    return nb
