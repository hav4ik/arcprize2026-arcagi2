"""E020 host-side checks of a train-passing program (no model tokens; never uses the test output).

test_audit(train, test_input, test_output, runtimes): flags from the program's own test output
  - colours in the test output that appear in no training output and not in test_input;
  - an output size that breaks a size rule every training pair follows (same as input, constant, fixed ratio);
  - an output identical to test_input (when no training output equals its input), or near-empty;
  - test-input features outside the training range (the task summary's "Outside train range" line);
  - a test run much slower than the training runs.
lint(code, train): hardcoded constants in the program source (AST)
  - literal tables (dict / list / set literals with >= 6 int-ish entries);
  - literal coordinates: subscripts with two int literals, one of them >= 2 in size, e.g. g[3][5] or g[3, 5];
  - (lists of 8 neighbour offsets in -1..1 are not flagged)
  - literal grids: nested list literals equal to a training grid, or comparisons against nested list literals;
  - branches on exact sizes: comparisons of len(...) / .shape with an int literal > 3.
"""
import ast
import re

SLOW_FACTOR, SLOW_MIN_S = 5.0, 2.0


def _colours(g):
    return {v for row in g for v in row}


def _shape(g):
    return len(g), len(g[0])


def _bg(g):
    counts = {}
    for row in g:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda v: counts[v])


def size_rule_flag(train, test_input, out):
    ins = [_shape(p["input"]) for p in train]
    outs = [_shape(p["output"]) for p in train]
    ti, to = _shape(test_input), _shape(out)
    if all(i == o for i, o in zip(ins, outs)):
        if to != ti:
            return f"every training output has its input's size, but your test output is {to[0]}x{to[1]} for a {ti[0]}x{ti[1]} input"
        return None
    if len(set(outs)) == 1:
        if to != outs[0]:
            return f"every training output is {outs[0][0]}x{outs[0][1]}, but your test output is {to[0]}x{to[1]}"
        return None
    for d, name in ((0, "height"), (1, "width")):
        ratios = {o[d] / i[d] for i, o in zip(ins, outs)}
        if len(ratios) == 1:
            r = ratios.pop()
            if abs(to[d] - r * ti[d]) > 1e-9:
                return (f"in every training pair the output {name} is {r:g} x the input {name}, but your test output "
                        f"{name} is {to[d]} for an input {name} of {ti[d]}")
    return None


def test_audit(train, test_input, out, runtimes=None, summary_text=None):
    flags = []
    train_out_cols = set().union(*[_colours(p["output"]) for p in train])
    new = sorted(_colours(out) - train_out_cols - _colours(test_input))
    if new:  # colours invented by the program (in no training output and not in test_input)
        flags.append(f"colour(s) {','.join(map(str, new))} appear in your test output but in no training output "
                     "and not in test_input")
    flag = size_rule_flag(train, test_input, out)
    if flag:
        flags.append(flag)
    if out == test_input and not any(p["output"] == p["input"] for p in train):
        flags.append("your test output is identical to test_input, but no training output equals its input")
    bg = _bg(out)
    frac = sum(v != bg for row in out for v in row) / (len(out) * len(out[0]))
    train_fracs = []
    for p in train:
        b = _bg(p["output"])
        o = p["output"]
        train_fracs.append(sum(v != b for row in o for v in row) / (len(o) * len(o[0])))
    if len(_colours(out)) == 1 and all(len(_colours(p["output"])) > 1 for p in train):
        flags.append("your test output is a single colour, unlike every training output")
    elif frac < 0.5 * min(train_fracs):
        flags.append(f"your test output is nearly empty: {frac:.0%} non-background cells (training outputs: "
                     f"{min(train_fracs):.0%}-{max(train_fracs):.0%})")
    if summary_text:
        m = re.search(r"Outside train range: (.*)", summary_text)
        if m and not m.group(1).startswith("nothing"):
            flags.append("test_input is outside the training range: " + m.group(1).strip()[:600])
    if runtimes and len(runtimes) > 1:
        tr, te = max(runtimes[:-1]), runtimes[-1]
        if te > SLOW_MIN_S and te > SLOW_FACTOR * max(tr, 0.05):
            flags.append(f"solve took {te:.1f}s on test_input against at most {tr:.1f}s on the training inputs")
    return flags


def _int_like(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _int_like(node.operand)
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(_int_like(e) for e in node.elts)
    return False


def _int_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        return -_int_value(node.operand)
    return None


def _nested_list(node):
    return isinstance(node, ast.List) and node.elts and all(isinstance(e, ast.List) for e in node.elts)


def lint(code, train=None):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    grids = [p[k] for p in (train or []) for k in ("input", "output")]
    flags, seen = [], set()

    def add(kind, node, text):
        key = (kind, getattr(node, "lineno", 0))
        if key not in seen:
            seen.add(key)
            flags.append(f"line {getattr(node, 'lineno', '?')}: {text}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and len(node.keys) >= 6 and sum(
                _int_like(k) for k in node.keys if k is not None) + sum(_int_like(v) for v in node.values) >= 6:
            add("table", node, f"a literal table with {len(node.keys)} entries (a lookup fitted to the training "
                               "pairs will not generalise; derive the mapping from the grid)")
        elif isinstance(node, (ast.List, ast.Set, ast.Tuple)) and len(node.elts) >= 6 and all(
                isinstance(e, (ast.Tuple, ast.List)) and _int_like(e) for e in node.elts) and not all(
                _safe(e) and all(abs(v) <= 1 for v in _flat(ast.literal_eval(e))) for e in node.elts):  # not offsets
            if _nested_list(node) and any(ast.literal_eval(node) == g for g in grids if _safe(node)):
                add("grid", node, "a literal copy of a training grid")
            else:
                add("table", node, f"a literal list of {len(node.elts)} coordinate/value tuples")
        if _nested_list(node) and _safe(node) and any(ast.literal_eval(node) == g for g in grids):
            add("grid", node, "a literal copy of a training grid")
        if isinstance(node, ast.Subscript):
            idx = node.slice
            if isinstance(idx, ast.Tuple) and len(idx.elts) == 2 and all(_int_like(e) for e in idx.elts):
                vals = [_int_value(e) for e in idx.elts]
                if all(isinstance(v, int) for v in vals) and max(abs(v) for v in vals) >= 2:
                    add("coord", node, f"a hardcoded cell index [{vals[0]}, {vals[1]}]")
            elif (isinstance(node.value, ast.Subscript) and _int_like(idx) and _int_like(node.value.slice)
                  and not isinstance(node.value.slice, ast.Tuple)):
                vals = [_int_value(node.value.slice), _int_value(idx)]
                if all(isinstance(v, int) for v in vals) and max(abs(v) for v in vals) >= 2:
                    add("coord", node, f"a hardcoded cell index [{vals[0]}][{vals[1]}]")
        if isinstance(node, ast.Compare):
            parts = [node.left] + node.comparators
            if any(_nested_list(p) for p in parts):
                add("cmp", node, "a comparison with a literal grid (special-casing particular inputs)")
            sized = any((isinstance(p, ast.Call) and getattr(p.func, "id", "") == "len") or
                        (isinstance(p, ast.Attribute) and p.attr == "shape") or
                        (isinstance(p, ast.Subscript) and isinstance(p.value, ast.Attribute)
                         and p.value.attr == "shape") for p in parts)
            lits = [_int_value(p) for p in parts if isinstance(p, ast.Constant) and type(p.value) is int]
            if sized and any(v > 3 for v in lits) and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in node.ops):
                add("size", node, f"a branch on an exact grid size ({max(lits)})")
    return flags[:8]


def _flat(x):
    return [v for e in x for v in _flat(e)] if isinstance(x, (list, tuple)) else [x]


def _safe(node):
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, SyntaxError, TypeError):
        return False


def audit_report(n_train, out, flags, lint_flags, digits):
    lines = [f"[program passes all {n_train} training pairs. Before it becomes your answer, review this host-side "
             f"audit. Your output for test_input ({len(out)}x{len(out[0])}):", digits]
    if flags:
        lines.append("Audit flags (things that differ from every training pair):")
        lines += [f"- {f}" for f in flags]
    if lint_flags:
        lines.append("Hardcoded constants in your program (often fit the training pairs only):")
        lines += [f"- {f}" for f in lint_flags]
    if not flags and not lint_flags:
        lines.append("No audit flags.")
    lines.append("Check the output against your rule and against what is different about test_input. If it is right, "
                 "call submit_program again with the same code: that makes it your answer. Otherwise fix solve and "
                 "submit the fixed program.]")
    return "\n".join(lines)

