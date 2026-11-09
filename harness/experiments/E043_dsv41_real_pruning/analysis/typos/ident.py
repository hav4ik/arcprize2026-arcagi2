#!/usr/bin/env python3
"""Step 2: identifier-level analysis of every run_python call.

For each model:
  out/ident_errors_<model>.pkl : one row per error call (all exception types), with, for identifier errors, the typo,
                                 the recovered intended identifier (model's own fix in the next call, else nearest
                                 vocabulary name), edit and token-level classification, history of the intended name.
  out/occ_<model>.pkl          : one row per identifier occurrence in code (denominators): tokens in context,
                                 previous occurrence distance, token-identical copy or not, lead char, kind.
"""
import ast
import builtins
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from textutil import IDENT, KW, edit_type, encode, first_diff, ident_occurrences, osa, span_tokens  # noqa: E402

API = json.load(open(f"{OUT}/api_names.json"))
BUILTINS = set(API["builtins"])
PRELOADED = {"train", "test_input", "np", "json"}
MODULE_ATTRS = {k: set(v) for k, v in API["module_attrs"].items()}
TYPE_ATTRS = {k: set(v) for k, v in API["type_attrs"].items()}
ALL_TYPE_ATTRS = set().union(*TYPE_ATTRS.values())
SCIPY_SUBMODULES = {"ndimage", "signal", "spatial", "sparse", "optimize", "stats", "special", "linalg", "interpolate",
                    "cluster", "fft", "io", "integrate", "misc"}
MODULE_NAMES = set(API["stdlib_modules"]) | {"numpy", "scipy"} | {f"scipy.{s}" for s in SCIPY_SUBMODULES} | {
    "scipy.sparse.csgraph", "scipy.spatial.distance"}
UNAVAILABLE = {"sklearn", "skimage", "cv2", "networkx", "pandas", "PIL", "sympy", "torch", "matplotlib"}
API_NAMES = set().union(*MODULE_ATTRS.values()) | set(MODULE_ATTRS) | {"deque", "Counter", "defaultdict", "ndimage",
                                                                      "itertools", "collections", "scipy", "math",
                                                                      "label", "binary_fill_holes", "heapq",
                                                                      "permutations", "combinations", "product"}
TYPE_MAP = {"numpy.ndarray": "ndarray", "collections.deque": "deque", "Counter": "Counter", "set": "set",
            "list": "list", "dict": "dict", "tuple": "tuple", "str": "str", "int": "int", "frozenset": "frozenset",
            "collections.defaultdict": "defaultdict", "defaultdict": "defaultdict", "numpy.int64": "generic",
            "numpy.bool_": "generic", "numpy.bool": "generic", "numpy.float64": "generic"}
KWARGS = ["dtype", "axis", "return_counts", "return_index", "return_inverse", "structure", "key", "reverse",
          "keepdims", "out", "order", "fill_value", "k", "axes", "minlength", "default", "end", "sep", "flush",
          "linewidth", "threshold", "edgeitems", "maxlen", "connectivity", "copy", "start", "repeat", "r", "initial"]


def bound_names(code):
    """Names bound anywhere in code (assign/for/with/def/class/import/args/comprehension/walrus); None if no parse."""
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out.update(n.names)
    return out


def loose_bound(code):
    b = bound_names(code)
    if b is not None:
        return b
    # regex fallback: `name =`, `def name`, `for a, b in`, `import x as y`
    out = set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=(?!=)", code, re.M))
    out |= set(re.findall(r"\bdef\s+([A-Za-z_]\w*)", code))
    for m in re.finditer(r"\bfor\s+([\w\s,()]+?)\s+in\b", code):
        out |= set(IDENT.findall(m.group(1)))
    out |= set(re.findall(r"\bimport\s+[\w.]+\s+as\s+(\w+)", code))
    out |= set(re.findall(r"\bimport\s+(\w+)", code))
    return out


def parse_ident_error(exc_type, msg):
    """(typo, kind, extra) or None."""
    msg = msg or ""
    if exc_type == "NameError":
        m = re.match(r"name '([^']+)' is not defined", msg) or re.match(r"free variable '([^']+)'", msg)
        return (m.group(1), "name", None) if m else None
    if exc_type == "AttributeError":
        m = re.match(r"[Mm]odule '([^']+)' has no attribute '([^']+)'", msg)
        if m:
            return (m.group(2), "attr", ("module", m.group(1)))
        m = re.match(r"'([^']+)' object has no attribute '([^']+)'", msg)
        if m:
            return (m.group(2), "attr", ("type", m.group(1)))
        return None
    if exc_type == "ModuleNotFoundError":
        m = re.match(r"No module named '([^']+)'", msg)
        return (m.group(1), "module", None) if m else None
    if exc_type == "ImportError":
        m = re.match(r"cannot import name '([^']+)' from '([^']+)'", msg)
        return (m.group(1), "import", ("module", m.group(2))) if m else None
    if exc_type == "TypeError":
        m = re.search(r"got an unexpected keyword argument '([^']+)'", msg) or \
            re.match(r"'([^']+)' is an invalid keyword argument", msg)
        return (m.group(1), "kwarg", None) if m else None
    return None


def locate(code, typo, kind):
    """Char span of the first occurrence of the typo (last dotted component for modules)."""
    occ = ident_occurrences(code)
    if kind == "module":
        parts = typo.split(".")
        m = re.search(r"(?<![\w.])" + re.escape(typo) + r"(?![\w])", code)
        if not m:
            return None
        s = m.end() - len(parts[-1])
        return s, m.end(), parts[-1]
    want = {"name": ("name", "kwarg", "import"), "attr": ("attr",), "import": ("import", "name"),
            "kwarg": ("kwarg", "name")}[kind]
    for s, e, n, k in occ:
        if n == typo and k in want:
            return s, e, n
    for s, e, n, k in occ:
        if n == typo:
            return s, e, n
    m = re.search(r"(?<![\w])" + re.escape(typo) + r"(?![\w])", code)
    return (m.start(), m.end(), typo) if m else None


def map_span(L, L2, ss, ee):
    sm = SequenceMatcher(None, L, L2, autojunk=False)
    js = je = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if js is None and i1 <= ss < i2:
            js = j1 + (ss - i1) if tag == "equal" else j1
        if je is None and i1 < ee <= i2:
            je = j1 + (ee - i1) if tag == "equal" else j2
    if js is None or je is None or je < js:
        return None
    while js > 0 and (L2[js - 1].isalnum() or L2[js - 1] == "_"):
        js -= 1
    while je < len(L2) and (L2[je].isalnum() or L2[je] == "_"):
        je += 1
    return L2[js:je]


def fix_from_next(code, s, e, typo, next_codes):
    """The model's own correction: align the typo's line with the most similar line of the next call(s)."""
    ls = code.rfind("\n", 0, s) + 1
    le = code.find("\n", e)
    le = len(code) if le < 0 else le
    L = code[ls:le]
    for nc in next_codes:
        best = None
        for L2 in nc.splitlines():
            if not L2.strip() or re.search(r"(?<![\w.])" + re.escape(typo) + r"(?!\w)", L2):
                continue
            r = SequenceMatcher(None, L, L2, autojunk=False).ratio()
            if r >= 0.6 and (best is None or r > best[0]):
                best = (r, L2)
        if best:
            fx = map_span(L, best[1], s - ls, e - ls)
            if fx and fx != typo:
                return fx, best[0]
    return None, None


def vocab_for(kind, extra, typo, prev_names, cur_names):
    if kind == "kwarg":
        return set(KWARGS)
    if kind == "name":
        return set(prev_names) | set(cur_names) | PRELOADED | BUILTINS | API_NAMES
    if kind == "attr":
        if extra and extra[0] == "module":
            mod = {"np": "numpy"}.get(extra[1], extra[1])
            return MODULE_ATTRS.get(mod, set()) | (MODULE_ATTRS["numpy"] if mod == "numpy" else set())
        t = TYPE_MAP.get(extra[1]) if extra else None
        return TYPE_ATTRS.get(t, set()) | (ALL_TYPE_ATTRS if t is None else set())
    if kind == "import":
        mod = extra[1] if extra else ""
        return MODULE_ATTRS.get(mod, set()) | ({f"{s}" for s in SCIPY_SUBMODULES} if mod == "scipy" else set())
    if kind == "module":
        if "." in typo:
            base = typo.rsplit(".", 1)[0]
            return {m.split(".")[-1] for m in MODULE_NAMES if m.startswith(base + ".")} | (
                SCIPY_SUBMODULES if base == "scipy" else set())
        return {m for m in MODULE_NAMES if "." not in m} | UNAVAILABLE
    return set()


def nearest(typo, vocab, prefer=()):
    best, bd = [], 99
    for v in vocab:
        if v == typo or abs(len(v) - len(typo)) > 6 or (v.startswith("_") and not typo.startswith("_")):
            continue
        d = osa(typo, v)
        if d < bd:
            best, bd = [v], d
        elif d == bd:
            best.append(v)
    if not best:
        return None, None, 0
    pref = [b for b in best if b in prefer]
    pick = sorted(pref or best, key=lambda x: (-len(x), x))[0]
    return pick, bd, len(best)


def tokens_at(code, s, e):
    ids, offs, toks = encode(code)
    ks = span_tokens(ids, offs, toks, s, e)
    return [toks[k] for k in ks], [ids[k] for k in ks], [offs[k] for k in ks]


def token_detail(code, s, e, typo, intended):
    """Canonical BPE tokens of the typo (as written) and of the intended name substituted at the same place."""
    t_toks, t_ids, t_offs = tokens_at(code, s, e)
    code2 = code[:s] + intended + code[e:]
    i_toks, i_ids, i_offs = tokens_at(code2, s, s + len(intended))
    d = first_diff(typo, intended)
    # token index (within the intended's tokens) containing char s+d (or the end, if typo is a prefix)
    pos = s + min(d, len(intended) - 1)
    k_div = next((k for k, (a, b) in enumerate(i_offs) if a <= pos < b), len(i_offs) - 1)
    at_boundary = d > 0 and any(a == s + d for a, b in i_offs[1:]) if d < len(intended) else True
    # common token prefix
    ctp = 0
    while ctp < min(len(t_ids), len(i_ids)) and t_ids[ctp] == i_ids[ctp]:
        ctp += 1
    # boundaries of the intended inside the word (char offsets relative to s)
    bounds = [max(a, s) - s for a, b in i_offs[1:]]
    btype = None
    if at_boundary and 0 < d < len(intended):
        btype = "mid" if intended[d - 1].isalnum() and intended[d].isalnum() else "sym"
    elif 0 < d < len(intended):
        btype = "inside-token"
    elif d == 0:
        btype = "first-char"
    else:
        btype = "end"
    n_mid = sum(1 for b in bounds if 0 < b < len(intended) and intended[b - 1].isalnum() and intended[b].isalnum())
    return {"btype": btype, "i_n_mid": n_mid, "i_first_merged": bool(i_offs) and i_offs[0][0] < s,
            "t_toks": t_toks, "i_toks": i_toks, "i_ids": i_ids, "n_tok_i": len(i_toks), "n_tok_t": len(t_toks),
            "first_diff_char": d, "div_tok": k_div, "at_boundary": at_boundary, "common_tok_prefix": ctp,
            "i_bounds": bounds, "lead": code[s - 1] if s > 0 else "^"}


def lead_class(ch):
    if ch in " ":
        return "space"
    if ch in "\n\t^":
        return "linestart"
    if ch == ".":
        return "dot"
    if ch in "([{":
        return "open"
    if ch in ",":
        return "comma"
    if ch in "=<>+-*/%!&|^~:":
        return "op"
    return "other"


def analyze_attempt(a):
    """Returns (error_rows, occ_rows) for one attempt."""
    seq = []  # (turn_idx, call_idx, call, turn)
    for ti, t in enumerate(a["turns"]):
        for ci, c in enumerate(t["calls"]):
            seq.append((ti, ci, c, t))
    errors, occs = [], []
    hist = defaultdict(list)  # name -> [(gpos, tok_ids tuple, src, turn)]
    reason_words = defaultdict(list)  # word -> [turn] mentions in reasoning
    gpos = 0
    prev_bound = set()
    prev_names = set()
    seen_turns = set()
    for qi, (ti, ci, c, t) in enumerate(seq):
        if ti not in seen_turns:
            seen_turns.add(ti)
            rtxt = t["reasoning"] + "\n" + t["content"]
            r_ids, r_offs, _ = encode(rtxt)
            for m in IDENT.finditer(rtxt):
                ks = span_tokens(r_ids, r_offs, None, m.start(), m.end())
                hist[m.group()].append((gpos + (ks[0] if ks else 0), tuple(r_ids[k] for k in ks), "reason", ti))
                reason_words[m.group()].append(ti)
            gpos += len(r_ids)
        if c["name"] != "run_python" and c["name"] != "submit_answer":
            continue
        code = c["code"] or ""
        ids, offs, toks = encode(code)
        occ = ident_occurrences(code)
        cur_bound = loose_bound(code)
        ie = parse_ident_error(c["exc_type"], c["exc_msg"]) if c["status"] == "error" else None
        loc = locate(code, ie[0], ie[1]) if ie else None
        err = None
        if c["status"] == "error":
            err = {"model": a["model"], "task_id": a["task_id"], "pair": a["pair"], "attempt": a["attempt"],
                   "turn": ti, "call_seq": qi, "n_calls": len(seq), "exc_type": c["exc_type"],
                   "exc_msg": (c["exc_msg"] or "")[:200], "prompt_tokens": t["prompt_tokens"],
                   "completion_tokens": t["completion_tokens"], "code_tokens": len(ids)}
        typo_done = False
        for (s, e, name, kind) in occ:
            ks = span_tokens(ids, offs, toks, s, e)
            tid = tuple(ids[k] for k in ks)
            h = hist.get(name, [])
            mypos = gpos + (ks[0] if ks else 0)
            prev = h[-1] if h else None
            bnd = [offs[k][0] - s for k in ks[1:] if s < offs[k][0] < e]
            n_mid = sum(1 for b in bnd if name[b - 1].isalnum() and name[b].isalnum())
            row = (name, kind, len(name), len(ks), lead_class(code[s - 1] if s > 0 else "^"), n_mid, len(bnd),
                   (ks[0] < len(offs) and offs[ks[0]][0] < s) if ks else False,
                   prev is not None, (mypos - prev[0]) if prev else -1, prev[2] if prev else "",
                   (prev[3] == ti) if prev else False, any(x[1] == tid for x in h), (prev[1] == tid) if prev else False,
                   name in PRELOADED or name in BUILTINS, ti, t["prompt_tokens"] or 0, c["status"])
            is_typo_here = (loc is not None and not typo_done and s == loc[0] and kind != "module")
            if loc is not None and ie[1] == "module" and not typo_done and s <= loc[0] < e:
                is_typo_here = True
            if is_typo_here and err is not None:
                typo_done = True
                err["_hist_snapshot"] = (mypos, {k: list(v) for k, v in hist.items()})
            occs.append(row)
            hist[name].append((mypos, tid, "code", ti))
        if err is not None:
            if ie:
                typo, kind, extra = ie
                err.update({"typo": typo, "kind": kind, "extra": extra})
                nexts = [x[2]["code"] for x in seq[qi + 1: qi + 3] if x[2]["name"] == "run_python"]
                if loc is None:
                    err["located"] = False
                else:
                    s, e, tword = loc
                    err["located"] = True
                    err["line"] = code[code.rfind("\n", 0, s) + 1: (code.find("\n", e) if code.find("\n", e) >= 0
                                                                     else len(code))][:300]
                    fx, fr = fix_from_next(code, s, e, tword, nexts)
                    err["fix"], err["fix_ratio"] = fx, fr
                    cur_names = {n for (_, _, n, _) in occ if n != tword}
                    vocab = vocab_for(kind, extra, typo, prev_names, cur_names)
                    near, nd, nties = nearest(tword, vocab, prefer=prev_names | cur_names)
                    err["near"], err["near_dist"], err["near_ties"] = near, nd, nties
                    # classification
                    if kind == "name" and (tword in prev_bound or tword in PRELOADED):
                        cls = "state-or-scope"
                    elif kind == "name" and tword in API_NAMES and tword not in cur_bound:
                        cls = "missing-import"
                    elif kind == "module" and tword in UNAVAILABLE:
                        cls = "unavailable-module"
                    elif kind == "module" and (tword in ("Counter", "deque", "defaultdict")):
                        cls = "wrong-api"
                    elif kind == "attr" and extra and extra[0] == "type" and tword in ALL_TYPE_ATTRS:
                        cls = "wrong-api"
                    elif kind == "name" and tword in cur_bound and tword not in {n for (_, _, n, k) in occ if k != "name"}:
                        cls = "use-before-def"
                    else:
                        cls = "typo"
                    # intended: model's fix if it is a close rewrite, else nearest vocab name
                    intended, src = None, None
                    fx_ok = bool(fx and re.fullmatch(r"[\w.]+", fx))
                    fx_valid = fx_ok and (fx in vocab or fx.split(".")[-1] in vocab)
                    fx_span = bool(fx and re.fullmatch(r"\w+=\w+", fx) and fx[:1] == tword[:1]
                                   and edit_type(tword, fx) in ("chunk-deletion", "deletion", "tail-replaced"))
                    if fx_ok and near == fx:
                        intended, src = fx, "fix=vocab"
                    elif fx_span:
                        intended, src = fx, "fix-span"  # e.g. dbool -> dtype=bool
                    elif near and nd == 1 and nties == 1 and len(near) >= 3:
                        intended, src = near, "vocab"
                    elif fx_valid and osa(tword, fx) <= 2 and len(fx) >= 2:
                        intended, src = fx, "fix"
                    elif near and nd is not None and nd <= 2 and nties == 1 and len(near) >= 3 and near[0] == tword[0]:
                        intended, src = near, "vocab"
                    elif len(tword) >= 2 and len([v for v in vocab if v.startswith(tword) and v != tword
                                                  and not v.startswith("_")]) == 1:
                        intended = [v for v in vocab if v.startswith(tword) and v != tword and not v.startswith("_")][0]
                        src = "vocab-prefix"
                    elif fx_valid and osa(tword, fx) <= max(2, len(fx) // 2):
                        intended, src = fx, "fix-far"
                    elif near and nd is not None and nd <= 2 and len(near) >= 3 and near[0] == tword[0]:
                        intended, src = near, "vocab-tie"
                    if cls == "typo" and intended is None:
                        cls = "unresolved"
                    err["cls"], err["intended"], err["intended_src"] = cls, intended, src
                    if intended:
                        err["edit"] = edit_type(tword, intended)
                        err["dist"] = osa(tword, intended)
                        err.update(token_detail(code, s, e, tword, intended))
                        err["lead_cls"] = lead_class(err["lead"])
                        # history of the intended name at the typo position
                        snap = err.pop("_hist_snapshot", None)
                        hh = (snap[1].get(intended, []) if snap else hist.get(intended, []))
                        mypos = snap[0] if snap else gpos
                        hh = [x for x in hh if x[0] < mypos]
                        err["i_prev"] = len(hh) > 0
                        err["i_prev_code"] = any(x[2] == "code" for x in hh)
                        err["i_prev_reason"] = any(x[2] == "reason" for x in hh)
                        err["i_prev_dist"] = (mypos - hh[-1][0]) if hh else -1
                        err["i_prev_same_turn"] = (hh[-1][3] == ti) if hh else False
                        err["i_prev_src"] = hh[-1][2] if hh else ""
                        err["i_prev_turns_back"] = (ti - hh[-1][3]) if hh else -1
                        err["i_tok_match_any"] = any(x[1] == tuple(err["i_ids"]) for x in hh)
                        err["i_tok_match_last"] = (hh[-1][1] == tuple(err["i_ids"])) if hh else False
                        err["i_preloaded"] = intended in PRELOADED or intended in BUILTINS or intended in API_NAMES
                        err["typo_in_reason_same_turn"] = ti in reason_words.get(tword, [])
                        err["typo_in_reason_any"] = any(x <= ti for x in reason_words.get(tword, []))
                        err["typo_in_prev_code"] = tword in prev_names
                    # next-call outcome
                    nx = [x for x in seq[qi + 1:] if x[2]["name"] == "run_python"]
                    if nx:
                        n0 = nx[0][2]
                        err["next_status"] = n0["status"]
                        err["next_exc"] = n0["exc_type"]
                        err["next_has_typo"] = bool(re.search(r"(?<![\w])" + re.escape(tword) + r"(?!\w)", n0["code"]))
                        err["next_same_err"] = (n0["exc_type"] == c["exc_type"] and (n0["exc_msg"] or "") == (c["exc_msg"] or ""))
                        err["next_sim"] = SequenceMatcher(None, code, n0["code"], autojunk=False).quick_ratio()
                        err["next_turn_tokens"] = nx[0][3]["completion_tokens"]
                        err["next_turn_gap"] = nx[0][0] - ti
                    err["did_you_mean"] = "Did you mean" in c["output"]
            err.pop("_hist_snapshot", None)
            errors.append(err)
        prev_bound |= cur_bound
        prev_names |= {n for (_, _, n, _) in occ}
        gpos += len(ids)
    return errors, occs


OCC_COLS = ["name", "kind", "len", "n_tok", "lead", "n_mid", "n_bnd", "first_merged", "has_prev", "prev_dist", "prev_src", "prev_same_turn",
            "tok_match_any", "tok_match_last", "preloaded", "turn", "prompt_tokens", "status"]


def run_model(model):
    A = pickle.load(open(f"{OUT}/attempts_{model}.pkl", "rb"))
    with ProcessPoolExecutor(20) as ex:
        res = list(ex.map(analyze_attempt, A, chunksize=4))
    errors = [e for r in res for e in r[0]]
    occs = []
    for a, r in zip(A, res):
        key = (a["task_id"], a["pair"], a["attempt"])
        occs.extend(key + o for o in r[1])
    pickle.dump(errors, open(f"{OUT}/ident_errors_{model}.pkl", "wb"))
    pickle.dump(occs, open(f"{OUT}/occ_{model}.pkl", "wb"))
    print(model, len(errors), "errors", len(occs), "occurrences", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        run_model(m)
