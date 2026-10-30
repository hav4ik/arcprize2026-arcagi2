"""Retry-once auto-repair of identifier typos in run_python cells (ported from Havu E043 / Candidate B32).

On an unknown-name error (`NameError`, `AttributeError`, `ModuleNotFoundError`, `ImportError`, or
`TypeError` unexpected keyword argument), `propose(code, error_text, session_names)` finds a unique
close match within OSA edit distance <= 2 (<= 1 for len <= 4, min_len=4 except keyword arguments,
plus positional `dbool`/`dint` -> `dtype=bool`/`dtype=int`), statically verifies via AST dataflow
analysis (`_Scan`) that statements executed up to the failing line had zero side effects on external
session state, and rewrites only Python `NAME` tokens.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import io
import json
import os
import pkgutil
import re
import sys
import tokenize
from typing import Any, Callable
import warnings


NOTE = "[auto-corrected: {} -> {}]"

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_repair_names.json"),
    "r",
    encoding="utf-8",
) as _f:
  _NAMES: dict[str, Any] = json.load(_f)

BUILTINS = set(_NAMES["builtins"]) | {
    n for n in dir(builtins) if not n.startswith("_")
}
TOP_MODULES = (
    set(_NAMES["stdlib_modules"])
    | set(_NAMES["third_party_modules"])
    | {m for m in getattr(sys, "stdlib_module_names", ()) if not m.startswith("_")}
)
_INTROSPECT = (
    "numpy",
    "scipy",
    "skimage",
    "collections",
    "itertools",
    "functools",
    "math",
    "copy",
    "re",
    "operator",
    "heapq",
    "random",
    "string",
    "json",
    "statistics",
    "bisect",
)
TYPES = {
    "numpy.ndarray": "ndarray",
    "ndarray": "ndarray",
    "list": "list",
    "dict": "dict",
    "set": "set",
    "frozenset": "frozenset",
    "tuple": "tuple",
    "str": "str",
    "int": "int",
    "collections.deque": "deque",
    "deque": "deque",
    "Counter": "Counter",
    "collections.Counter": "Counter",
    "defaultdict": "defaultdict",
    "collections.defaultdict": "defaultdict",
    "numpy.int64": "generic",
    "numpy.int32": "generic",
    "numpy.float64": "generic",
    "numpy.bool_": "generic",
    "numpy.bool": "generic",
    "numpy.uint8": "generic",
}
_TYPE_OBJ = {
    "list": list,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "tuple": tuple,
    "str": str,
    "int": int,
}
KWARGS = {
    "zeros": ["shape", "dtype", "order", "like"],
    "ones": ["shape", "dtype", "order", "like"],
    "empty": ["shape", "dtype", "order", "like"],
    "full": ["shape", "fill_value", "dtype", "order", "like"],
    "zeros_like": ["a", "dtype", "order", "subok", "shape"],
    "ones_like": ["a", "dtype", "order", "subok", "shape"],
    "full_like": ["a", "fill_value", "dtype", "order", "subok", "shape"],
    "array": ["object", "dtype", "copy", "order", "subok", "ndmin", "like"],
    "asarray": ["a", "dtype", "order", "like"],
    "arange": ["start", "stop", "step", "dtype", "like"],
    "sort": ["key", "reverse"],
    "sorted": ["key", "reverse"],
    "label": ["input", "structure", "output"],
    "max": ["key", "default"],
    "min": ["key", "default"],
    "unique": ["ar", "return_index", "return_inverse", "return_counts", "axis", "equal_nan"],
    "argwhere": ["a"],
    "where": ["condition", "x", "y"],
}
MUTATORS = {
    "append",
    "appendleft",
    "extend",
    "extendleft",
    "insert",
    "pop",
    "popleft",
    "popitem",
    "remove",
    "clear",
    "update",
    "add",
    "discard",
    "setdefault",
    "sort",
    "reverse",
    "fill",
    "put",
    "itemset",
    "resize",
    "rotate",
    "subtract",
    "intersection_update",
    "difference_update",
    "symmetric_difference_update",
    "write",
    "writelines",
    "__setitem__",
    "__delitem__",
}
ARG_MUTATORS = {
    "heappush",
    "heappop",
    "heapify",
    "heapreplace",
    "heappushpop",
    "shuffle",
    "copyto",
    "put",
    "place",
    "putmask",
    "fill_diagonal",
    "insort",
    "insort_left",
    "insort_right",
}
UNSAFE_CALLS = {"exec", "eval", "setattr", "delattr", "globals", "__import__"}
UNSAFE_MODULES = {"os", "shutil", "subprocess", "sys", "pathlib", "pickle", "socket"}
ALIASING_CALLS = {
    "iter",
    "reversed",
    "enumerate",
    "zip",
    "items",
    "values",
    "keys",
    "get",
    "setdefault",
    "pop",
    "popleft",
    "asarray",
    "view",
    "ravel",
    "reshape",
    "squeeze",
    "transpose",
    "swapaxes",
    "next",
}
MODULE_ALIASES = {
    "np",
    "numpy",
    "scipy",
    "ndimage",
    "math",
    "json",
    "itertools",
    "collections",
    "functools",
    "heapq",
    "random",
    "re",
    "statistics",
    "string",
    "operator",
    "copy",
    "bisect",
}
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))(?::\s?(.*))?$")
DTYPE_TYPO = re.compile(
    r"d+(?:type)?_?(bool|int|float|complex|object|str|uint|u?int(?:8|16|32|64)|float(?:16|32|64))"
)


def parse_error(error_text: str) -> tuple[str, str, Any] | None:
  """Extract `(typo, kind, extra)` for an unknown-name error in tool output, else None."""
  exc = msg = None
  for line in reversed((error_text or "").splitlines()):
    m = _EXC.match(line.strip())
    if m:
      exc, msg = m.group(1).rsplit(".", 1)[-1], m.group(2) or ""
      break
  if exc == "NameError" and msg is not None:
    m = re.match(r"name '([^']+)' is not defined", msg) or re.match(
        r"free variable '([^']+)'", msg
    )
    return (m.group(1), "name", None) if m else None
  if exc == "AttributeError" and msg is not None:
    m = re.match(r"[Mm]odule '([^']+)' has no attribute '([^']+)'", msg)
    if m:
      return m.group(2), "attr", ("module", m.group(1))
    m = re.match(r"'([^']+)' object has no attribute '([^']+)'", msg)
    return (m.group(2), "attr", ("type", m.group(1))) if m else None
  if exc == "ModuleNotFoundError" and msg is not None:
    m = re.match(r"No module named '([^']+)'", msg)
    return (m.group(1), "module", None) if m else None
  if exc == "ImportError" and msg is not None:
    m = re.match(r"cannot import name '([^']+)' from '([^']+)'", msg)
    return (m.group(1), "import", m.group(2)) if m else None
  if exc == "TypeError" and msg is not None:
    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '([^']+)'", msg)
    if m:
      return m.group(2), "kwarg", m.group(1)
    m = re.match(r"'([^']+)' is an invalid keyword argument for (\w+)\(\)", msg)
    return (m.group(1), "kwarg", m.group(2)) if m else None
  return None


def osa(a: str, b: str) -> int:
  """Optimal-string-alignment (restricted Damerau-Levenshtein) distance."""
  if a == b:
    return 0
  d = [
      [i + j if not i * j else 0 for j in range(len(b) + 1)]
      for i in range(len(a) + 1)
  ]
  for i in range(1, len(a) + 1):
    for j in range(1, len(b) + 1):
      d[i][j] = min(
          d[i - 1][j] + 1,
          d[i][j - 1] + 1,
          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
      )
      if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
        d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
  return d[len(a)][len(b)]


def unique_close(
    word: str,
    cands: Any,
    cell_names: Any = (),
    prefix_ok: bool = False,
) -> str | None:
  """Return the unique closest candidate within edit distance threshold, or None."""
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
      scored.append((
          dist,
          0 if len(c) > len(word) else 1,
          -cp,
          0 if c in cell_names else 1,
          c,
      ))
  if not scored:
    return None
  scored.sort()
  ties = [s for s in scored if s[:4] == scored[0][:4]]
  return scored[0][4] if len(ties) == 1 else None


def _introspect(qualname: str) -> set[str]:
  if qualname.split(".")[0] not in _INTROSPECT:
    return set()
  try:
    return {
        n
        for n in dir(importlib.import_module(qualname))
        if not n.startswith("__")
    }
  except Exception:  # pylint: disable=broad-except
    return set()


def default_members(qualname: str) -> set[str]:
  """Members of a module (`np` = `numpy`) or known container/ndarray type."""
  if qualname in TYPES:
    t = TYPES[qualname]
    names = set(_NAMES["type_attrs"].get(t, ()))
    if t in _TYPE_OBJ:
      names |= {n for n in dir(_TYPE_OBJ[t]) if not n.startswith("__")}
    elif t in ("deque", "Counter", "defaultdict"):
      names |= {
          n
          for n in dir(getattr(importlib.import_module("collections"), t))
          if not n.startswith("__")
      }
    elif t in ("ndarray", "generic"):
      try:
        names |= {
            n
            for n in dir(getattr(importlib.import_module("numpy"), t))
            if not n.startswith("__")
        }
      except Exception:  # pylint: disable=broad-except
        pass
    return names
  qualname = re.sub(r"^np(?=\.|$)", "numpy", qualname)
  return set(_NAMES["module_attrs"].get(qualname, ())) | _introspect(qualname)


def submodules(package: str) -> set[str]:
  """Submodule names of a package."""
  names = {
      m.split(".")[len(package.split(".")) :][0]
      for m in _NAMES["module_attrs"]
      if m.startswith(package + ".")
  }
  if package == "scipy":
    names |= {
        "ndimage",
        "signal",
        "spatial",
        "sparse",
        "optimize",
        "stats",
        "special",
        "linalg",
        "interpolate",
        "cluster",
        "fft",
        "io",
        "integrate",
    }
  try:
    mod = (
        importlib.import_module(package)
        if package.split(".")[0] in _INTROSPECT
        else None
    )
    if mod is not None and hasattr(mod, "__path__"):
      names |= {
          m.name
          for m in pkgutil.iter_modules(mod.__path__)
          if not m.name.startswith("_")
      }
  except Exception:  # pylint: disable=broad-except
    pass
  return names


def _parse(code: str) -> ast.Module:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return ast.parse(code)


def _bound_names(tree: ast.AST) -> set[str]:
  """Every name bound anywhere in the AST."""
  out: set[str] = set()
  for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
      out.add(n.id)
    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      out.add(n.name)
      if not isinstance(n, ast.ClassDef):
        out |= {
            a.arg for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs
        }
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
      out |= {
          (a.asname or a.name).split(".")[0]
          for a in n.names
          if a.name != "*"
      }
    elif isinstance(n, ast.ExceptHandler) and n.name:
      out.add(n.name)
  return out


def rewrite(code: str, typo: str, repl: str, kind: str) -> str | None:
  """Replace matching Python NAME tokens without touching strings or comments."""
  try:
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
  except (tokenize.TokenError, SyntaxError, IndentationError):
    return None
  starts = [0]
  for line in code.splitlines(True):
    starts.append(starts[-1] + len(line))
  sig = [
      t
      for t in toks
      if t.type
      not in (
          tokenize.NL,
          tokenize.NEWLINE,
          tokenize.COMMENT,
          tokenize.INDENT,
          tokenize.DEDENT,
      )
  ]
  spans = []
  for i, t in enumerate(sig):
    if t.type != tokenize.NAME or t.string != typo:
      continue
    after_dot = i > 0 and sig[i - 1].string == "."
    before_eq = i + 1 < len(sig) and sig[i + 1].string == "="
    after_comma = i > 0 and sig[i - 1].string == ","
    ok = {
        "name": not after_dot,
        "attr": after_dot,
        "import": not after_dot,
        "module": True,
        "kwarg": before_eq and not after_dot,
        "dtype": after_comma and not before_eq,
    }[kind]
    if ok:
      spans.append((
          starts[t.start[0] - 1] + t.start[1],
          starts[t.end[0] - 1] + t.end[1],
      ))
  if not spans:
    return None
  for s, e in reversed(spans):
    code = code[:s] + repl + code[e:]
  return code


def suggest(
    code: str,
    error_text: str,
    session_names: Any = (),
    members: Callable[[str], set[str]] | None = None,
    min_len: int = 4,
    dtype_rule: bool = True,
) -> tuple[str, str, str] | None:
  """Return `(typo, replacement, kind)` without side-effect check, or None."""
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


def _traceback_line(code: str, error_text: str) -> int | None:
  """Return 1-based line number of the failing top-level statement, or None."""
  lines = (error_text or "").splitlines()
  try:
    i = next(
        k
        for k, l in enumerate(lines)
        if l.startswith("Traceback (most recent call last)")
    )
  except StopIteration:
    return None
  for l in lines[i + 1 :]:
    m = re.match(
        r'\s*File "(<cell>|<ipython-input-[^"]*>|<string>)", line (\d+)', l
    )
    if m:
      return int(m.group(2))
    if l.startswith("  File ") or not l.strip():
      continue
    text = l.strip()
    hits = [k + 1 for k, c in enumerate(code.splitlines()) if c.strip() == text]
    return max(hits) if hits else None
  return None


def _base(node: ast.AST) -> str | None:
  while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
    node = node.value
  return node.id if isinstance(node, ast.Name) else None


class _Scan:
  """Sequential AST scan detecting mutations to state outside the current cell."""

  def __init__(
      self,
      funcs: dict[str, Any],
      modules: set[str],
      session: set[str],
      strict: bool,
      globals_: Any = (),
      undefined: Any = (),
  ) -> None:
    self.funcs = funcs
    self.modules = modules
    self.session = session
    self.strict = strict
    self.globals = set(globals_)
    self.undefined = set(undefined)
    self.bad: set[str] = set()

  def fresh(self, v: ast.AST, owned: set[str]) -> bool:
    if isinstance(v, ast.Name):
      return v.id in owned or v.id in self.undefined
    if isinstance(v, (ast.Attribute, ast.Subscript, ast.Starred)):
      b = _base(v)
      return b is None or b in owned
    if isinstance(v, ast.NamedExpr):
      return self.fresh(v.value, owned)
    if isinstance(v, ast.IfExp):
      if isinstance(v.test, ast.Constant):
        return self.fresh(v.body if v.test.value else v.orelse, owned)
      return self.fresh(v.body, owned) and self.fresh(v.orelse, owned)
    if isinstance(v, (ast.Tuple, ast.List)):
      return all(self.fresh(e, owned) for e in v.elts)
    if isinstance(v, ast.Call):
      f = v.func
      name = (
          f.id
          if isinstance(f, ast.Name)
          else f.attr if isinstance(f, ast.Attribute) else None
      )
      if name in ALIASING_CALLS:
        return all(self.fresh(a, owned) for a in v.args) and (
            not isinstance(f, ast.Attribute)
            or _base(f.value) in self.modules
            or self.fresh(f.value, owned)
        )
      return True
    return True

  def iter_owned(self, it: ast.AST, owned: set[str]) -> bool:
    return all(
        n.id in owned or n.id in BUILTINS or n.id in self.modules
        for n in ast.walk(it)
        if isinstance(n, ast.Name)
    )

  def flag(self, name: str | None) -> None:
    if name not in self.undefined:
      self.bad.add(name or "<expr>")

  def expr(self, node: ast.AST, bound: set[str], owned: set[str]) -> None:
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

  def call(self, n: ast.Call, bound: set[str], owned: set[str]) -> None:
    f = n.func
    if isinstance(f, ast.Name):
      if f.id in UNSAFE_CALLS:
        self.flag("<" + f.id + ">")
      if f.id == "open":
        modes = n.args[1:2] + [k.value for k in n.keywords if k.arg == "mode"]
        if any(
            not isinstance(m, ast.Constant) or set(str(m.value)) & set("wax+")
            for m in modes
        ):
          self.flag("<open>")
      fn = self.funcs.get(f.id)
      if fn is not None and f.id in bound:
        params, mparams, mfree = fn
        for b in mfree:
          if b.startswith("<") or b not in owned:
            self.flag(b)
        if mparams:
          for i, a in enumerate(n.args):
            if isinstance(a, ast.Starred) or (
                i < len(params)
                and params[i] in mparams
                and not self.fresh(a, owned)
            ):
              self.flag(_base(a))
          for k in n.keywords:
            if (k.arg is None or k.arg in mparams) and not self.fresh(
                k.value, owned
            ):
              self.flag(_base(k.value))
      elif (
          self.strict
          and f.id in self.session
          and f.id not in bound
          and f.id not in BUILTINS
      ):
        self.flag(f.id)
    elif isinstance(f, ast.Attribute):
      b = _base(f.value)
      if b in UNSAFE_MODULES and b not in bound:
        self.flag("<" + b + ">")
      if (
          f.attr in MUTATORS
          and b not in self.modules
          and not self.fresh(f.value, owned)
      ):
        self.flag(b)
      if f.attr in ARG_MUTATORS and n.args and not self.fresh(n.args[0], owned):
        self.flag(_base(n.args[0]))

  def target(
      self,
      t: ast.AST,
      bound: set[str],
      owned: set[str],
      value: ast.AST | None = None,
  ) -> None:
    if isinstance(t, (ast.Tuple, ast.List)):
      vals = (
          value.elts
          if isinstance(value, (ast.Tuple, ast.List))
          and len(value.elts) == len(t.elts)
          else None
      )
      for k, e in enumerate(t.elts):
        self.target(e, bound, owned, vals[k] if vals else value)
      return
    if isinstance(t, ast.Starred):
      self.target(t.value, bound, owned, value)
      return
    if isinstance(t, (ast.Attribute, ast.Subscript)):
      if not self.fresh(t, owned):
        self.flag(_base(t))
      self.expr(t, bound, owned)
      return
    if isinstance(t, ast.Name):
      if t.id in self.globals:
        self.flag(t.id)
      if (
          value is not None
          and t.id not in bound
          and any(
              isinstance(n, ast.Name) and n.id == t.id for n in ast.walk(value)
          )
      ):
        self.flag(t.id)
      bound.add(t.id)
      if value is None or self.fresh(value, owned):
        owned.add(t.id)
      else:
        owned.discard(t.id)

  def body(
      self, stmts: list[ast.stmt], bound: set[str], owned: set[str]
  ) -> None:
    for s in stmts:
      self.stmt(s, bound, owned)

  def function(
      self, s: ast.FunctionDef | ast.AsyncFunctionDef
  ) -> tuple[list[str], set[str], set[str]]:
    a = s.args
    params = [x.arg for x in a.posonlyargs + a.args]
    allp = (
        set(params)
        | {x.arg for x in a.kwonlyargs}
        | {x.arg for x in (a.vararg, a.kwarg) if x}
    )
    glob = {
        n
        for g in ast.walk(s)
        if isinstance(g, (ast.Global, ast.Nonlocal))
        for n in g.names
    }
    inner = _Scan(self.funcs, self.modules, set(), False, glob, self.undefined)
    local = _bound_names(ast.Module(body=s.body, type_ignores=[])) - glob
    inner.body(s.body, set(allp), set())
    mparams = {
        b for b in inner.bad if b in allp or (b in local and b not in allp)
    }
    mfree = {b for b in inner.bad if b not in allp and b not in local}
    if any(b in local and b not in allp for b in inner.bad):
      mparams |= allp
    return params, mparams, mfree

  def stmt(self, s: ast.stmt, bound: set[str], owned: set[str]) -> None:
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
          self.flag(s.target.id)
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
    elif isinstance(
        s, (ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue)
    ):
      pass
    else:
      for c in ast.iter_child_nodes(s):
        if isinstance(c, ast.expr):
          self.expr(c, bound, owned)
        elif isinstance(c, ast.stmt):
          self.stmt(c, bound, owned)
        elif hasattr(c, "body"):
          self.body(c.body, bound, owned)


def ran_side_effects(
    code: str,
    error_text: str,
    session_names: Any = (),
    strict: bool = False,
    undefined: Any = (),
) -> bool:
  """Return True if statements executed prior to the error may have mutated external session state."""
  tree = _parse(code)
  line = _traceback_line(code, error_text)
  body = tree.body
  if line is not None:
    idx = [
        k
        for k, s in enumerate(body)
        if s.lineno <= line <= (s.end_lineno or s.lineno)
    ]
    body = body[: idx[-1] + 1] if idx else body
  scan = _Scan(
      {}, set(MODULE_ALIASES), set(session_names), strict, undefined=undefined
  )
  scan.body(body, set(), set())
  return bool(scan.bad)


def propose(
    code: str,
    error_text: str,
    session_names: Any = (),
    members: Callable[[str], set[str]] | None = None,
    strict: bool = False,
    min_len: int = 4,
) -> tuple[str, str] | None:
  """Return `(new_code, note)` for a safe retry-once repair, or None."""
  s = suggest(code, error_text, session_names, members, min_len)
  if s is None:
    return None
  typo, repl, kind = s
  new = rewrite(code, typo, repl, kind)
  if new is None or new == code:
    return None
  try:
    if ran_side_effects(
        code,
        error_text,
        session_names,
        strict,
        {typo} if kind in ("name", "dtype") else (),
    ):
      return None
  except SyntaxError:
    return None
  return new, NOTE.format(typo, repl)
