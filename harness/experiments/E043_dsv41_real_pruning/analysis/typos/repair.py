#!/usr/bin/env python3
"""Step 4: static auto-repair simulation ("rewrite unknown names to a unique close match before executing").

For every run_python call, in order within the attempt:
  known = preloaded (train, test_input, np, json) + builtins + names bound by earlier calls + names bound anywhere in
          this call (all scopes, static) + import aliases
  1. bare names loaded but not known  -> unique candidate in `known` with OSA distance <= 2 (<= 1 for names <= 4 chars)
  2. module attributes  np.X / itertools.X / collections.X / ndimage.X / scipy.X not in dir(module)
                                       -> unique candidate in dir(module), distance <= 2
  3. method names .X not an attribute of any common type/module and never assigned/defined in the code
                                       -> unique candidate among common-type methods, distance <= 2
  4. import X / from X import Y with X not an importable module / Y not in dir(X)
                                       -> unique candidate, distance <= 2
Ties (two candidates at the minimal distance) are not repaired.

Outputs out/repair_<model>.pkl: one row per call with the rewrites proposed, the actual status/exception, and for
identifier-error calls whether the rewrite matches the intended name recovered in ident.py.
"""
import io
import os
import pickle
import re
import sys
import tokenize as pytok
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from ident import (ALL_TYPE_ATTRS, BUILTINS, MODULE_ATTRS, MODULE_NAMES, PRELOADED, UNAVAILABLE,  # noqa: E402
                   loose_bound)
from textutil import KW, ident_occurrences, osa  # noqa: E402

MOD_ALIAS = {"np": "numpy", "numpy": "numpy", "itertools": "itertools", "collections": "collections",
             "ndimage": "scipy.ndimage", "ndi": "scipy.ndimage", "scipy": "scipy", "math": "math",
             "heapq": "heapq", "functools": "functools", "operator": "operator", "random": "random",
             "copy": "copy", "re": "re"}
METHODS = ALL_TYPE_ATTRS | set().union(*MODULE_ATTRS.values())
TOP_MODULES = {m for m in MODULE_NAMES if "." not in m}


def unique_close(word, cands, maxd=None):
    if maxd is None:
        maxd = 1 if len(word) <= 4 else 2
    best, bd = [], 99
    for c in cands:
        if c == word or len(c) < 2 or abs(len(c) - len(word)) > maxd:
            continue
        d = osa(word, c)
        if d < bd:
            best, bd = [c], d
        elif d == bd:
            best.append(c)
    if bd <= maxd and len(best) == 1:
        return best[0], bd
    return None, (bd if bd <= maxd else None)


def attrs_assigned(code):
    return set(re.findall(r"\.([A-Za-z_]\w*)\s*=(?!=)", code)) | set(re.findall(r"\bdef\s+([A-Za-z_]\w*)", code))


def precheck(code, session_names, session_attrs, aliases):
    """[(kind, typo, repair)] proposed for one call."""
    out = []
    try:
        occ = ident_occurrences(code)
    except Exception:
        return out
    bound = loose_bound(code)
    known = session_names | bound | BUILTINS | PRELOADED | KW
    code_attrs = attrs_assigned(code) | session_attrs
    # import statements
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w\s,]+)|import\s+([\w.]+))", code, re.M):
        mod = m.group(1) or m.group(3)
        if mod and mod not in MODULE_NAMES and mod.split(".")[0] not in UNAVAILABLE:
            parts = mod.split(".")
            for i in range(len(parts)):
                sub = ".".join(parts[:i + 1])
                if sub not in MODULE_NAMES:
                    pool = TOP_MODULES if i == 0 else {x.split(".")[-1] for x in MODULE_NAMES
                                                       if x.startswith(".".join(parts[:i]) + ".")}
                    r, _ = unique_close(parts[i], pool)
                    out.append(("module", parts[i], r))
                    break
        elif m.group(1) and m.group(1) in MODULE_ATTRS:
            names = [x.strip() for x in m.group(2).split(",") if x.strip()]
            for nm in names:
                nm = nm.split()[0]
                pool = MODULE_ATTRS[m.group(1)] | ({"ndimage", "signal", "spatial", "sparse", "optimize", "stats"}
                                                   if m.group(1) == "scipy" else set())
                if nm not in pool:
                    r, _ = unique_close(nm, pool)
                    out.append(("import", nm, r))
    seen = set()
    for i, (s, e, name, kind) in enumerate(occ):
        if (kind, name) in seen:
            continue
        if kind == "name" and name not in known:
            seen.add((kind, name))
            r, _ = unique_close(name, known - KW)
            out.append(("name", name, r))
        elif kind == "attr":
            # what precedes the dot?
            j = s - 1
            while j >= 0 and code[j] == " ":
                j -= 1
            k = j - 1
            while k >= 0 and (code[k].isalnum() or code[k] == "_"):
                k -= 1
            base = code[k + 1:j]
            if base in aliases:
                mod = aliases[base]
            elif base in MOD_ALIAS and base not in bound:
                mod = MOD_ALIAS[base]
            else:
                mod = None
            if mod in MODULE_ATTRS:
                if name not in MODULE_ATTRS[mod] and not (mod == "scipy"):
                    seen.add((kind, name))
                    r, _ = unique_close(name, MODULE_ATTRS[mod])
                    out.append(("modattr", name, r))
            elif name not in METHODS and name not in code_attrs and name not in known:
                seen.add((kind, name))
                r, _ = unique_close(name, METHODS)
                out.append(("method", name, r))
    return out


def import_aliases(code):
    al = {}
    for m in re.finditer(r"\bimport\s+([\w.]+)\s+as\s+(\w+)", code):
        al[m.group(2)] = m.group(1)
    for m in re.finditer(r"\bfrom\s+([\w.]+)\s+import\s+(\w+)\s+as\s+(\w+)", code):
        al[m.group(3)] = f"{m.group(1)}.{m.group(2)}"
    for m in re.finditer(r"\bfrom\s+(scipy)\s+import\s+([\w\s,]+)", code):
        for x in m.group(2).split(","):
            x = x.strip().split(" ")[0]
            if x:
                al[x] = f"scipy.{x}"
    return al


def run_attempt(a):
    rows = []
    session, sattrs, aliases = set(), set(), {}
    seq = [(ti, c) for ti, t in enumerate(a["turns"]) for c in t["calls"]]
    for qi, (ti, c) in enumerate(seq):
        if c["name"] != "run_python":
            continue
        code = c["code"] or ""
        aliases.update(import_aliases(code))
        props = precheck(code, session, sattrs, aliases)
        rows.append({"key": (a["task_id"], a["pair"], a["attempt"]), "call_seq": qi, "turn": ti,
                     "status": c["status"], "exc_type": c["exc_type"], "exc_msg": (c["exc_msg"] or "")[:160],
                     "props": props})
        session |= loose_bound(code)
        sattrs |= attrs_assigned(code)
    return rows


def run(model):
    A = pickle.load(open(f"{OUT}/attempts_{model}.pkl", "rb"))
    with ProcessPoolExecutor(20) as ex:
        res = [r for rr in ex.map(run_attempt, A, chunksize=8) for r in rr]
    for r in res:
        r["model"] = model
    pickle.dump(res, open(f"{OUT}/repair_{model}.pkl", "wb"))
    trig = sum(1 for r in res if any(p[2] for p in r["props"]))
    print(model, len(res), "calls; calls with >=1 proposed rewrite:", trig, flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        run(m)
