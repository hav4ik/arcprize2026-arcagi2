"""Retry-once auto-repair of identifier typos in run_python cells (E043 typo analysis).

Expert-pruned DeepSeek-V4.1 models misspell multi-token names (`coms` for `comps`, `scpy` for `scipy`):
experiments/E043_dsv41_real_pruning/analysis/typos/REPORT.md. On an unknown-name error the harness can rewrite the
name to its unique close match and re-run the cell once. Replayed on recorded errors, the retried cell completed in
61% / 52% / 45% of identifier errors for E128 / E112 / E96. The rewrite matched the model's own next fix in 89-93%.

    from auto_repair import propose
    fix = propose(code, error_text, session_names)
    if fix:
        new_code, note = fix          # re-run new_code once; show note + "\n" + its output to the model

`propose` returns None unless all of these hold:
- **The error is an unknown name.** It is one of:
  - NameError;
  - AttributeError of a module or a known type (numpy.ndarray, list, dict, set, str, deque, Counter, ...);
  - ImportError / ModuleNotFoundError;
  - TypeError "unexpected keyword argument" for a known function.
- **There is a unique close match.** It must be within OSA edit distance 2 (1 for names of at most 4
  characters).
  - Names shorter than 4 characters are not repaired (`min_len`), except keyword arguments.
  - A positional `dbool` / `dint` becomes `dtype=bool` / `dtype=int`: the model restarted `dtype=` after the comma.
  - NameError candidates: `session_names`, the names the cell binds, and builtins.
  - Other errors: the members of the named module or type, from `members` if given, else this process's own import
    of the module, plus the bundled table `auto_repair_names.json`.
  - Ties are broken by: deletion of the candidate first, then the longest common prefix, then names used in the
    cell. A remaining tie means no repair.
- **The typo is in this cell.** Only Python NAME tokens are rewritten, never strings or comments.
- **The part of the cell that already ran has no side effects outside the cell.** Checked statically:
  - the top-level statements up to and including the failing one, found from the traceback;
  - no mutation of an object the cell did not create first (`x.append`, `x[i] = ...`, `x += ...`, `del x[...]`,
    `heapq.heappush(x, ...)`, `random.shuffle(x)`, ...);
  - no rebinding of a session name from its own value (`x = x + 1`);
  - no file or OS calls;
  - no call of a function from this cell that mutates a name it does not own.
  - Functions defined in earlier cells are assumed pure, unless `strict=True`, which refuses once any of them ran.

Checked against the replay study on 3,410 recorded identifier errors
(experiments/E043_dsv41_real_pruning/analysis/typos/validate_auto_repair.py):
- With min_len=1 and dtype_rule=False (the replay's own rules), it proposes the same rewrite, or none, in 99%, and
  never a different name.
- With the defaults it keeps 241 of the 250 E128 repairs that completed in the replay. It skips 13 E128 names
  shorter than 4 characters, and the side-effect check refuses 7.
- The rewrite differs from the model's own next-call fix in 1 of 76 labelled E128 cases (E112 9 of 147, E96 18 of
  185): `analysis/typos/repair_disagreements.py`.
Tests: evals/tests/test_auto_repair.py.
"""
import ast
import builtins
import importlib
import io
import json
import os
import re
import sys
import tokenize
import warnings

NOTE = "[auto-corrected: {} -> {}]"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_repair_names.json")) as _f:
    _NAMES = json.load(_f)
BUILTINS = set(_NAMES["builtins"]) | {n for n in dir(builtins) if not n.startswith("_")}
TOP_MODULES = (set(_NAMES["stdlib_modules"]) | set(_NAMES["third_party_modules"])
               | {m for m in getattr(sys, "stdlib_module_names", ()) if not m.startswith("_")})
# importable packages whose members this process may introspect (others: bundled table only)
_INTROSPECT = ("numpy", "scipy", "skimage", "collections", "itertools", "functools", "math", "copy", "re", "operator",
               "heapq", "random", "string", "json", "statistics", "bisect")
TYPES = {"numpy.ndarray": "ndarray", "ndarray": "ndarray", "list": "list", "dict": "dict", "set": "set",
         "frozenset": "frozenset", "tuple": "tuple", "str": "str", "int": "int", "collections.deque": "deque",
         "deque": "deque", "Counter": "Counter", "collections.Counter": "Counter", "defaultdict": "defaultdict",
         "collections.defaultdict": "defaultdict", "numpy.int64": "generic", "numpy.int32": "generic",
         "numpy.float64": "generic", "numpy.bool_": "generic", "numpy.bool": "generic", "numpy.uint8": "generic"}
_TYPE_OBJ = {"list": list, "dict": dict, "set": set, "frozenset": frozenset, "tuple": tuple, "str": str, "int": int}
KWARGS = {"zeros": ["shape", "dtype", "order", "like"], "ones": ["shape", "dtype", "order", "like"],
          "empty": ["shape", "dtype", "order", "like"], "full": ["shape", "fill_value", "dtype", "order", "like"],
          "zeros_like": ["a", "dtype", "order", "subok", "shape"], "ones_like": ["a", "dtype", "order", "subok", "shape"],
          "full_like": ["a", "fill_value", "dtype", "order", "subok", "shape"],
          "array": ["object", "dtype", "copy", "order", "subok", "ndmin", "like"], "asarray": ["a", "dtype", "order", "like"],
          "arange": ["start", "stop", "step", "dtype", "like"], "sort": ["key", "reverse"], "sorted": ["key", "reverse"],
          "label": ["input", "structure", "output"], "max": ["key", "default"], "min": ["key", "default"],
          "unique": ["ar", "return_index", "return_inverse", "return_counts", "axis", "equal_nan"],
          "argwhere": ["a"], "where": ["condition", "x", "y"]}
MUTATORS = {"append", "appendleft", "extend", "extendleft", "insert", "pop", "popleft", "popitem", "remove", "clear",
            "update", "add", "discard", "setdefault", "sort", "reverse", "fill", "put", "itemset", "resize", "rotate",
            "subtract", "intersection_update", "difference_update", "symmetric_difference_update", "write",
            "writelines", "__setitem__", "__delitem__"}
# module functions that mutate their first argument
ARG_MUTATORS = {"heappush", "heappop", "heapify", "heapreplace", "heappushpop", "shuffle", "copyto", "put", "place",
                "putmask", "fill_diagonal", "insort", "insort_left", "insort_right"}
UNSAFE_CALLS = {"exec", "eval", "setattr", "delattr", "globals", "__import__"}
UNSAFE_MODULES = {"os", "shutil", "subprocess", "sys", "pathlib", "pickle", "socket"}
# calls whose result may alias their argument (views, iterators, container access)
ALIASING_CALLS = {"iter", "reversed", "enumerate", "zip", "items", "values", "keys", "get", "setdefault",
                  "pop", "popleft", "asarray", "view", "ravel", "reshape", "squeeze", "transpose", "swapaxes",
                  "next"}
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))(?::\s?(.*))?$")


def parse_error(error_text):
    """(typo, kind, extra) for an unknown-name error in the tool output, else None. kind: name, attr, import,
    module, kwarg; extra: ("module", name) / ("type", name) for attr, the module for import, the function for kwarg."""
    exc = msg = None
    for line in reversed((error_text or "").splitlines()):
        m = _EXC.match(line.strip())
        if m:
            exc, msg = m.group(1).rsplit(".", 1)[-1], m.group(2) or ""
            break
    if exc == "NameError":
        m = re.match(r"name '([^']+)' is not defined", msg) or re.match(r"free variable '([^']+)'", msg)
        return (m.group(1), "name", None) if m else None
    if exc == "AttributeError":
        m = re.match(r"[Mm]odule '([^']+)' has no attribute '([^']+)'", msg)
        if m:
            return m.group(2), "attr", ("module", m.group(1))
        m = re.match(r"'([^']+)' object has no attribute '([^']+)'", msg)
        return (m.group(2), "attr", ("type", m.group(1))) if m else None
    if exc == "ModuleNotFoundError":
        m = re.match(r"No module named '([^']+)'", msg)
        return (m.group(1), "module", None) if m else None
    if exc == "ImportError":
        m = re.match(r"cannot import name '([^']+)' from '([^']+)'", msg)
        return (m.group(1), "import", m.group(2)) if m else None
    if exc == "TypeError":
        m = re.search(r"(\w+)\(\) got an unexpected keyword argument '([^']+)'", msg)
        if m:
            return m.group(2), "kwarg", m.group(1)
        m = re.match(r"'([^']+)' is an invalid keyword argument for (\w+)\(\)", msg)
        return (m.group(1), "kwarg", m.group(2)) if m else None
    return None


def osa(a, b):
    """Optimal-string-alignment (restricted Damerau-Levenshtein) distance."""
    if a == b:
        return 0
    d = [[i + j if not i * j else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def unique_close(word, cands, cell_names=(), prefix_ok=False):
    """The unique best candidate (see the module docstring), or None. prefix_ok: a unique candidate that starts with
    the word also qualifies (keyword arguments: d -> dtype)."""
    maxd = 1 if len(word) <= 4 else 2
    scored = []
    for c in set(cands):
        if c == word or len(c) < 2:
            continue
        if prefix_ok and c.startswith(word):
            scored.append((0, 0, -len(word), 0, c))
            continue
        if abs(len(c) - len(word)) > maxd:
            continue
        dist = osa(word, c)
        if dist <= maxd:
            cp = 0
            while cp < min(len(c), len(word)) and c[cp] == word[cp]:
                cp += 1
            scored.append((dist, 0 if len(c) > len(word) else 1, -cp, 0 if c in cell_names else 1, c))
    if not scored:
        return None
    scored.sort()
    ties = [s for s in scored if s[:4] == scored[0][:4]]
    return scored[0][4] if len(ties) == 1 else None


def _introspect(qualname):
    if not qualname.split(".")[0] in _INTROSPECT:
        return set()
    try:
        return {n for n in dir(importlib.import_module(qualname)) if not n.startswith("__")}
    except Exception:
        return set()


def default_members(qualname):
    """Members of a module (dotted name, `np` = numpy) or of a known type name from an error message."""
    if qualname in TYPES:
        t = TYPES[qualname]
        names = set(_NAMES["type_attrs"].get(t, ()))
        if t in _TYPE_OBJ:
            names |= {n for n in dir(_TYPE_OBJ[t]) if not n.startswith("__")}
        elif t in ("deque", "Counter", "defaultdict"):
            names |= {n for n in dir(getattr(importlib.import_module("collections"), t)) if not n.startswith("__")}
        elif t in ("ndarray", "generic"):
            try:
                names |= {n for n in dir(getattr(importlib.import_module("numpy"), t)) if not n.startswith("__")}
            except Exception:
                pass
        return names
    qualname = re.sub(r"^np(?=\.|$)", "numpy", qualname)
    return set(_NAMES["module_attrs"].get(qualname, ())) | _introspect(qualname)


def submodules(package):
    names = {m.split(".")[len(package.split(".")):][0] for m in _NAMES["module_attrs"] if m.startswith(package + ".")}
    if package == "scipy":
        names |= {"ndimage", "signal", "spatial", "sparse", "optimize", "stats", "special", "linalg", "interpolate",
                  "cluster", "fft", "io", "integrate"}
    try:
        import pkgutil
        mod = importlib.import_module(package) if package.split(".")[0] in _INTROSPECT else None
        if mod is not None and hasattr(mod, "__path__"):
            names |= {m.name for m in pkgutil.iter_modules(mod.__path__) if not m.name.startswith("_")}
    except Exception:
        pass
    return names


def _parse(code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(code)


def _bound_names(tree):
    """Every name the cell binds anywhere (assignments, loop/with/except targets, defs, imports, comprehensions)."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            out |= {a.arg for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs} if not isinstance(
                n, ast.ClassDef) else set()
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names if a.name != "*"}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


DTYPE_TYPO = re.compile(r"d+(?:type)?_?(bool|int|float|complex|object|str|uint|u?int(?:8|16|32|64)|float(?:16|32|64))")


def suggest(code, error_text, session_names=(), members=None, min_len=4, dtype_rule=True):
    """(typo, replacement, kind) without the side-effect check, or None.

    min_len: misspelled names shorter than this are left alone (keyword arguments excepted). In the replay, rewrites
    of names of at most 3 characters matched the model's own fix only about half the time (deq -> dq vs deque).
    dtype_rule: a positional argument `dbool` / `dint` (the model's restart of `dtype=bool`) becomes `dtype=bool`.
    """
    err = parse_error(error_text)
    if err is None:
        return None
    typo, kind, extra = err
    if kind != "kwarg" and len(typo.rpartition(".")[2]) < min_len:
        return None
    if kind == "name" and dtype_rule:
        m = DTYPE_TYPO.fullmatch(typo)
        if m and rewrite(code, typo, "dtype=" + m.group(1), "dtype") is not None:
            return typo, "dtype=" + m.group(1), "dtype"
    members = members or default_members
    try:
        tree = _parse(code)
    except SyntaxError:
        return None
    cell = set(re.findall(r"[A-Za-z_]\w*", code))
    if kind == "name":
        cands = set(session_names) | _bound_names(tree) | BUILTINS
    elif kind == "attr":
        cands = members(extra[1])
    elif kind == "import":
        cands = members(extra) | (submodules(extra) if "." not in extra else set())
    elif kind == "module":
        base, _, typo = typo.rpartition(".")
        cands = submodules(base) if base else TOP_MODULES
    else:  # kwarg
        r = unique_close(typo, KWARGS.get(extra, ()), prefix_ok=True)
        return (typo, r, kind) if r else None
    r = unique_close(typo, cands, cell_names=cell)
    return (typo, r, kind) if r else None


def rewrite(code, typo, repl, kind):
    """Replace the typo's NAME tokens (by kind: plain names, `.attr`, imported names, `name=` keywords)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None
    starts = [0]
    for line in code.splitlines(True):
        starts.append(starts[-1] + len(line))
    sig = [t for t in toks if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT,
                                             tokenize.DEDENT)]
    spans = []
    for i, t in enumerate(sig):
        if t.type != tokenize.NAME or t.string != typo:
            continue
        after_dot = i > 0 and sig[i - 1].string == "."
        before_eq = i + 1 < len(sig) and sig[i + 1].string == "="
        after_comma = i > 0 and sig[i - 1].string == ","
        ok = {"name": not after_dot, "attr": after_dot, "import": not after_dot, "module": True,
              "kwarg": before_eq and not after_dot, "dtype": after_comma and not before_eq}[kind]
        if ok:
            spans.append((starts[t.start[0] - 1] + t.start[1], starts[t.end[0] - 1] + t.end[1]))
    if not spans:
        return None
    for s, e in reversed(spans):
        code = code[:s] + repl + code[e:]
    return code


def _traceback_line(code, error_text):
    """1-based line of the cell where the failing top-level statement is, or None if it cannot be located."""
    lines = (error_text or "").splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.startswith("Traceback (most recent call last)"))
    except StopIteration:
        return None
    for l in lines[i + 1:]:
        m = re.match(r'\s*File "(<cell>|<ipython-input-[^"]*>|<string>)", line (\d+)', l)
        if m:
            return int(m.group(2))
        if l.startswith("  File ") or not l.strip():
            continue  # a library frame first: not expected
        text = l.strip()
        hits = [k + 1 for k, c in enumerate(code.splitlines()) if c.strip() == text]
        return max(hits) if hits else None
    return None


def _base(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _Scan:
    """Sequential scan of executed statements, collecting base names of state changes outside the cell.

    bound: names bound so far in this scope. owned: the subset that refers to objects created here (a literal, a call
    result, arithmetic), as opposed to aliases of session objects (`g = train[0]`, loop variables over session data).
    """

    def __init__(self, funcs, modules, session, strict, globals_=(), undefined=()):
        self.funcs, self.modules, self.session, self.strict = funcs, modules, session, strict
        self.globals = set(globals_)
        self.undefined = set(undefined)  # the misspelled name: it never existed, so nothing was done to it
        self.bad = set()

    def fresh(self, v, owned):
        """Does the value refer only to objects created in this scope (or new objects)?"""
        if isinstance(v, ast.Name):
            return v.id in owned or v.id in self.undefined
        if isinstance(v, (ast.Attribute, ast.Subscript, ast.Starred)):
            b = _base(v)
            return b is None or b in owned
        if isinstance(v, ast.NamedExpr):
            return self.fresh(v.value, owned)
        if isinstance(v, ast.IfExp):
            if isinstance(v.test, ast.Constant):  # the models' inline patches: `old if False else new`
                return self.fresh(v.body if v.test.value else v.orelse, owned)
            return self.fresh(v.body, owned) and self.fresh(v.orelse, owned)
        if isinstance(v, (ast.Tuple, ast.List)):
            return all(self.fresh(e, owned) for e in v.elts)
        if isinstance(v, ast.Call):
            f = v.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in ALIASING_CALLS:
                return (all(self.fresh(a, owned) for a in v.args)
                        and (not isinstance(f, ast.Attribute) or _base(f.value) in self.modules
                             or self.fresh(f.value, owned)))
            return True
        return True  # constants, arithmetic, comprehensions, ... build new objects

    def iter_owned(self, it, owned):
        return all(n.id in owned or n.id in BUILTINS or n.id in self.modules
                   for n in ast.walk(it) if isinstance(n, ast.Name))

    def flag(self, name):
        if name not in self.undefined:
            self.bad.add(name or "<expr>")

    def expr(self, node, bound, owned):
        bound, owned = set(bound), set(owned)
        for n in ast.walk(node):
            if isinstance(n, ast.comprehension):
                names = {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
                bound |= names
                if self.iter_owned(n.iter, owned):
                    owned |= names
            elif isinstance(n, ast.NamedExpr):
                bound.add(n.target.id)
                if self.fresh(n.value, owned):
                    owned.add(n.target.id)
            elif isinstance(n, ast.Lambda):
                args = {a.arg for a in n.args.args}
                bound |= args
                owned |= args
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                self.call(n, bound, owned)

    def call(self, n, bound, owned):
        f = n.func
        if isinstance(f, ast.Name):
            if f.id in UNSAFE_CALLS:
                self.flag("<" + f.id + ">")
            if f.id == "open":
                modes = n.args[1:2] + [k.value for k in n.keywords if k.arg == "mode"]
                if any(not isinstance(m, ast.Constant) or set(str(m.value)) & set("wax+") for m in modes):
                    self.flag("<open>")
            fn = self.funcs.get(f.id)
            if fn is not None and f.id in bound:
                params, mparams, mfree = fn
                for b in mfree:
                    if b.startswith("<") or b not in owned:
                        self.flag(b)
                if mparams:
                    for i, a in enumerate(n.args):
                        if isinstance(a, ast.Starred) or (i < len(params) and params[i] in mparams
                                                          and not self.fresh(a, owned)):
                            self.flag(_base(a))
                    for k in n.keywords:
                        if (k.arg is None or k.arg in mparams) and not self.fresh(k.value, owned):
                            self.flag(_base(k.value))
            elif self.strict and f.id in self.session and f.id not in bound and f.id not in BUILTINS:
                self.flag(f.id)
        elif isinstance(f, ast.Attribute):
            b = _base(f.value)
            if b in UNSAFE_MODULES and b not in bound:
                self.flag("<" + b + ">")
            if f.attr in MUTATORS and b not in self.modules and not self.fresh(f.value, owned):
                self.flag(b)
            if f.attr in ARG_MUTATORS and n.args and not self.fresh(n.args[0], owned):
                self.flag(_base(n.args[0]))

    def target(self, t, bound, owned, value=None):
        if isinstance(t, (ast.Tuple, ast.List)):
            vals = value.elts if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(t.elts) else None
            for k, e in enumerate(t.elts):
                self.target(e, bound, owned, vals[k] if vals else value)
            return
        if isinstance(t, ast.Starred):
            return self.target(t.value, bound, owned, value)
        if isinstance(t, (ast.Attribute, ast.Subscript)):
            if not self.fresh(t, owned):
                self.flag(_base(t))
            self.expr(t, bound, owned)
            return
        if isinstance(t, ast.Name):
            if t.id in self.globals:
                self.flag(t.id)
            if value is not None and t.id not in bound and any(
                    isinstance(n, ast.Name) and n.id == t.id for n in ast.walk(value)):
                self.flag(t.id)  # x = x + 1 on a session x: not idempotent
            bound.add(t.id)
            if value is None or self.fresh(value, owned):
                owned.add(t.id)
            else:
                owned.discard(t.id)

    def body(self, stmts, bound, owned):
        for s in stmts:
            self.stmt(s, bound, owned)

    def function(self, s):
        """(params, params it mutates, other names it mutates) of a function defined in the cell."""
        a = s.args
        params = [x.arg for x in a.posonlyargs + a.args]
        allp = set(params) | {x.arg for x in a.kwonlyargs} | {x.arg for x in (a.vararg, a.kwarg) if x}
        glob = {n for g in ast.walk(s) if isinstance(g, (ast.Global, ast.Nonlocal)) for n in g.names}
        inner = _Scan(self.funcs, self.modules, (), False, glob, self.undefined)
        local = _bound_names(ast.Module(body=s.body, type_ignores=[])) - glob
        inner.body(s.body, set(allp), set())
        mparams = {b for b in inner.bad if b in allp or (b in local and b not in allp)}
        mfree = {b for b in inner.bad if b not in allp and b not in local}
        if any(b in local and b not in allp for b in inner.bad):  # a local alias of something: assume a param
            mparams |= allp
        return params, mparams, mfree

    def stmt(self, s, bound, owned):
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in s.decorator_list:
                self.expr(d, bound, owned)
            self.funcs[s.name] = self.function(s)
            bound.add(s.name)
            owned.add(s.name)
        elif isinstance(s, ast.ClassDef):
            bound.add(s.name)
            owned.add(s.name)
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names:
                name = (a.asname or a.name).split(".")[0]
                bound.add(name)
                owned.add(name)
                self.modules.add(name)
        elif isinstance(s, ast.Assign):
            self.expr(s.value, bound, owned)
            for t in s.targets:
                self.target(t, bound, owned, s.value)
        elif isinstance(s, ast.AnnAssign):
            if s.value is not None:
                self.expr(s.value, bound, owned)
                self.target(s.target, bound, owned, s.value)
        elif isinstance(s, ast.AugAssign):
            self.expr(s.value, bound, owned)
            if isinstance(s.target, ast.Name):
                if s.target.id not in owned or s.target.id in self.globals:
                    self.flag(s.target.id)  # session value rebound from itself, or in-place on a session object
            elif not self.fresh(s.target, owned):
                self.flag(_base(s.target))
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    if t.id not in bound or t.id in self.globals:
                        self.flag(t.id)
                    bound.discard(t.id)
                    owned.discard(t.id)
                elif not self.fresh(t, owned):
                    self.flag(_base(t))
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            self.expr(s.iter, bound, owned)
            names = {x.id for x in ast.walk(s.target) if isinstance(x, ast.Name)}
            for x in names:
                if x in self.globals:
                    self.flag(x)
            bound |= names
            if self.iter_owned(s.iter, owned):
                owned |= names
            else:
                owned -= names
            self.body(s.body, bound, owned)
            self.body(s.orelse, bound, owned)
        elif isinstance(s, (ast.While, ast.If)):
            self.expr(s.test, bound, owned)
            self.body(s.body, bound, owned)
            self.body(s.orelse, bound, owned)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                self.expr(it.context_expr, bound, owned)
                if it.optional_vars is not None:
                    self.target(it.optional_vars, bound, owned, it.context_expr)
            self.body(s.body, bound, owned)
        elif isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
            self.body(s.body, bound, owned)
            for h in s.handlers:
                if h.name:
                    bound.add(h.name)
                    owned.add(h.name)
                self.body(h.body, bound, owned)
            self.body(s.orelse, bound, owned)
            self.body(s.finalbody, bound, owned)
        elif isinstance(s, (ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue)):
            pass
        else:  # Expr, Return, Raise, Assert, Match, ...
            for c in ast.iter_child_nodes(s):
                if isinstance(c, ast.expr):
                    self.expr(c, bound, owned)
                elif isinstance(c, ast.stmt):
                    self.stmt(c, bound, owned)
                elif hasattr(c, "body"):  # match_case
                    self.body(c.body, bound, owned)


MODULE_ALIASES = {"np", "numpy", "scipy", "ndimage", "math", "json", "itertools", "collections", "functools", "heapq",
                  "random", "re", "statistics", "string", "operator", "copy", "bisect"}


def ran_side_effects(code, error_text, session_names=(), strict=False, undefined=()):
    """True if the part of the cell that ran before the error may have changed state outside the cell. `undefined`:
    names known not to exist (the misspelled name of a NameError)."""
    tree = _parse(code)
    line = _traceback_line(code, error_text)
    body = tree.body
    if line is not None:
        idx = [k for k, s in enumerate(body) if s.lineno <= line <= (s.end_lineno or s.lineno)]
        body = body[: idx[-1] + 1] if idx else body
    scan = _Scan({}, set(MODULE_ALIASES), set(session_names), strict, undefined=undefined)
    scan.body(body, set(), set())
    return bool(scan.bad)


def propose(code, error_text, session_names=(), members=None, strict=False, min_len=4):
    """(new_code, note) for a retry-once repair, or None (see the module docstring)."""
    s = suggest(code, error_text, session_names, members, min_len)
    if s is None:
        return None
    typo, repl, kind = s
    new = rewrite(code, typo, repl, kind)
    if new is None or new == code:
        return None
    try:
        if ran_side_effects(code, error_text, session_names, strict, {typo} if kind in ("name", "dtype") else ()):
            return None
    except SyntaxError:
        return None
    return new, NOTE.format(typo, repl)
