"""Narrow, label-blind guard for whole-demonstration answer dispatch.

Recognize full-input dictionary lookups and exact-training-input return branches
in ordinary Python solver/helper source. Do not ban training-derived colors,
dimensions, object templates, parameters, or unrelated notebook variables.
This is bounded best-effort inspection, NOT a proof of generalization or a
security boundary. Obfuscated/dynamic Python and unknown dependencies can escape
inspection. Only a positively recognized shortcut is rejected.
"""
import ast
import inspect
import json
import textwrap
import types


MESSAGE = (
    "Whole-training-input answer lookup is not a general solver: the submitted "
    "function dispatches a complete input to a stored demonstration output. "
    "Remove that shortcut and run the SAME transformation on training and test "
    "inputs, then check again. Learning colors, object templates or rule parameters "
    "from training examples is allowed. No test answers were used by this guard."
)
UNKNOWN = object()


class Value:
    def __init__(self, kind=None, value=UNKNOWN):
        self.kind, self.value = kind, value


def inspect_solver(fn, train, *, np):
    """Inspect only fn's reachable helpers/data; never execute model code here."""
    findings, visited, returned = [], set(), {}
    incomplete = [False]
    remaining = [12000]
    table_cache = {}

    def grid(value):
        # No arbitrary __array__/iteration/equality hooks during inspection.
        if type(value) is np.ndarray:
            if value.ndim != 2 or value.size > 900 or value.dtype.kind not in "biu":
                return None
            value = value.tolist()
        if type(value) not in (tuple, list) or not 1 <= len(value) <= 30:
            return None
        if any(type(r) not in (tuple, list) or not 1 <= len(r) <= 30 for r in value):
            return None
        if len({len(r) for r in value}) != 1:
            return None
        if any(type(x) not in (int, bool) for r in value for x in r):
            return None
        return tuple(tuple(int(x) for x in r) for r in value)

    pairs = [(grid(p["input"]), grid(p["output"])) for p in train]
    pairs = [(i, o) for i, o in pairs if i is not None and o is not None]
    known_inputs, known_outputs = {i for i, _ in pairs}, {o for _, o in pairs}
    fingerprints = {}
    for inp, out in pairs:
        shape, flat = (len(inp), len(inp[0])), tuple(x for r in inp for x in r)
        array = np.array(inp)
        keys = [inp, (shape, inp), flat, (shape, flat), str([list(r) for r in inp]),
                repr(inp), str(array), json.dumps(inp), json.dumps(inp, separators=(",", ":"))]
        keys += [np.array(inp, dtype=dtype).tobytes() for dtype in
                 (np.int64, np.int32, np.int16, np.int8, np.uint8)]
        for key in keys:
            fingerprints.setdefault(key, set()).add(out)

    def plain_key(value, depth=0):
        if depth > 4:
            return False
        if type(value) in (str, bytes, int, bool):
            return not isinstance(value, (str, bytes)) or len(value) <= 10000
        return (type(value) is tuple and len(value) <= 1000
                and all(plain_key(x, depth+1) for x in value))

    def answer_table(value):
        if type(value) is not dict:
            return False
        if id(value) in table_cache:
            return table_cache[id(value)]
        if len(value) > 4096:
            incomplete[0] = True
            return False
        found = any(plain_key(k) and k in fingerprints and grid(v) in fingerprints[k]
                    for k, v in value.items())
        table_cache[id(value)] = found
        return found

    def training_pairs(value):
        return (type(value) in (list, tuple) and len(value) == len(pairs) and bool(pairs)
                and all(type(p) is dict and grid(p.get("input")) == inp
                        and grid(p.get("output")) == out for p, (inp, out) in zip(value, pairs)))

    def classify(value):
        if answer_table(value):
            return Value("answer_table", value)
        if training_pairs(value):
            return Value("train", value)
        return Value(value=value)

    def add(function, node, kind):
        row = dict(function=function, line=getattr(node, "lineno", None), kind=kind)
        if row not in findings and len(findings) < 8:
            findings.append(row)

    def analyze(function, arguments=(), keywords=None, depth=0):
        if type(function) is not types.FunctionType:
            return Value()
        # Source for notebook cells is in linecache, so this works inside seccomp.
        if not function.__code__.co_filename.startswith("<cell "):
            return Value()
        keywords = keywords or {}
        def signature(v):
            return v.kind, id(v.value)
        key = (id(function), tuple(signature(v) for v in arguments),
               tuple((k, signature(v)) for k, v in sorted(keywords.items())))
        if key in visited:
            return returned.get(key, Value())
        if depth > 8 or len(visited) >= 32:
            incomplete[0] = True
            return Value()
        visited.add(key)
        try:
            source = inspect.getsource(function)
            if len(source) > 50000:
                incomplete[0] = True
                return Value()
            tree = ast.parse(textwrap.dedent(source))
        except (OSError, TypeError, SyntaxError, IndentationError):
            incomplete[0] = True
            return Value()
        definition = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if definition is None:
            incomplete[0] = True
            return Value()
        scope = dict(function.__globals__)
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            try:
                scope[name] = cell.cell_contents
            except ValueError:
                pass
        parameters = definition.args.posonlyargs + definition.args.args
        env = {p.arg: Value() for p in parameters + definition.args.kwonlyargs}
        defaults = function.__defaults__ or ()
        for p, value in zip(parameters[len(parameters)-len(defaults):], defaults):
            env[p.arg] = classify(value)
        env.update({name: classify(value) for name, value in (function.__kwdefaults__ or {}).items()})
        env.update({p.arg: v for p, v in zip(parameters, arguments)})
        env.update({k: v for k, v in keywords.items() if k in env})
        return_values = []

        def tick():
            remaining[0] -= 1
            if remaining[0] < 0:
                incomplete[0] = True
                raise StopIteration

        def expr(node, local):
            tick()
            if node is None:
                return Value()
            if isinstance(node, ast.Name):
                return local[node.id] if node.id in local else classify(scope.get(node.id, UNKNOWN))
            if isinstance(node, ast.Constant):
                return Value(value=node.value)
            if isinstance(node, (ast.Tuple, ast.List)):
                values = [expr(n, local) for n in node.elts]
                value = (tuple(v.value for v in values) if isinstance(node, ast.Tuple)
                         else [v.value for v in values])
                kind = next((k for k in ("whole", "train_input") if any(v.kind == k for v in values)), None)
                return Value(kind, value)
            if isinstance(node, ast.Subscript):
                obj, index = expr(node.value, local), expr(node.slice, local)
                if obj.kind == "answer_table" and index.kind == "whole":
                    add(function.__name__, node, "whole_input_answer_table")
                if obj.kind == "pair" and type(index.value) is str and index.value in ("input", "output"):
                    return Value("train_" + index.value)
                if obj.kind == "train":
                    return Value("pair")
                # A crop, cell, object or feature derived from the argument is not
                # the complete input. Do not propagate the whole-input tag here.
                if type(obj.value) is dict and plain_key(index.value):
                    return classify(obj.value.get(index.value, UNKNOWN))
                return Value()
            if isinstance(node, ast.Attribute):
                expr(node.value, local)
                return Value()
            if isinstance(node, ast.Call):
                positional = [expr(a, local) for a in node.args]
                keywords = {k.arg: expr(k.value, local) for k in node.keywords if k.arg is not None}
                args = positional + list(keywords.values())
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                obj = expr(node.func.value, local) if isinstance(node.func, ast.Attribute) else Value()
                if name in ("get", "__getitem__") and obj.kind == "answer_table" and args and args[0].kind == "whole":
                    add(function.__name__, node, "whole_input_answer_table")
                if name in ("array_equal", "equal") and len(args) >= 2:
                    if exact_match(args[0], args[1]):
                        return Value("exact_match")
                if name in ("all", "bool") and (obj.kind == "exact_match" or any(v.kind == "exact_match" for v in args)):
                    return Value("exact_match")
                # Only content-preserving encodings, never arbitrary arithmetic,
                # shape-only keys or object extraction, carry this tag.
                if name in ("array", "asarray", "copy", "deepcopy", "tuple", "list", "str", "repr",
                            "bytes", "tolist", "tobytes", "flatten", "ravel", "reshape", "astype", "dumps", "map"):
                    vals = [obj] + args
                    for kind in ("whole", "train_input", "train_output"):
                        if any(v.kind == kind for v in vals):
                            return Value(kind)
                if isinstance(node.func, ast.Name):
                    callee = expr(node.func, local)
                    if type(callee.value) is types.FunctionType:
                        return analyze(callee.value, positional, keywords, depth+1)
                return Value()
            if isinstance(node, ast.Compare):
                values = [expr(n, local) for n in [node.left] + node.comparators]
                if (len(values) == 2 and isinstance(node.ops[0], ast.Eq)
                        and exact_match(values[0], values[1])):
                    return Value("exact_match")
                return Value()
            if isinstance(node, ast.BoolOp):
                values = [expr(n, local) for n in node.values]
                return Value("exact_match" if isinstance(node.op, ast.And)
                             and any(v.kind == "exact_match" for v in values) else None)
            if isinstance(node, ast.DictComp) and len(node.generators) == 1:
                gen = node.generators[0]
                copied = dict(local)
                if expr(gen.iter, local).kind == "train" and isinstance(gen.target, ast.Name):
                    copied[gen.target.id] = Value("pair")
                k, v = expr(node.key, copied), expr(node.value, copied)
                if k.kind == "train_input" and v.kind == "train_output":
                    return Value("answer_table")
                return Value()
            if isinstance(node, ast.IfExp):
                condition = expr(node.test, local)
                yes = expr(node.body, local)
                expr(node.orelse, local)
                if condition.kind == "exact_match" and is_output(yes):
                    add(function.__name__, node, "exact_training_input_return")
                return Value()
            # Visit nested calls for evidence; don't infer their semantics.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    expr(child, local)
            return Value()

        def exact_match(a, b):
            return ((a.kind == "whole" and (b.kind == "train_input" or grid(b.value) in known_inputs))
                    or (b.kind == "whole" and (a.kind == "train_input" or grid(a.value) in known_inputs)))

        def is_output(value):
            return value.kind == "train_output" or grid(value.value) in known_outputs

        def statements(nodes, local, matched=False):
            for node in nodes:
                tick()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = expr(node.value, local)
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            local[target.id] = value
                        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                            key = expr(target.slice, local)
                            if key.kind == "train_input" and value.kind == "train_output":
                                local[target.value.id] = Value("answer_table")
                elif isinstance(node, ast.If):
                    condition = expr(node.test, local)
                    statements(node.body, dict(local), matched or condition.kind == "exact_match")
                    statements(node.orelse, dict(local), matched)
                elif isinstance(node, ast.For):
                    iterable = expr(node.iter, local)
                    if isinstance(node.target, ast.Name):
                        local[node.target.id] = Value("pair" if iterable.kind == "train" else None)
                    statements(node.body, local, matched)
                    statements(node.orelse, local, matched)
                elif isinstance(node, ast.Return):
                    value = expr(node.value, local)
                    return_values.append(value)
                    if matched and is_output(value):
                        add(function.__name__, node, "exact_training_input_return")
                else:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, ast.expr):
                            expr(child, local)
                        elif isinstance(child, ast.stmt):
                            statements([child], dict(local), matched)
        try:
            statements(definition.body, env)
        except StopIteration:
            pass
        result = Value()
        if return_values and all(v.kind == return_values[0].kind for v in return_values):
            result = Value(return_values[0].kind)
        returned[key] = result
        return result

    analyze(fn, [Value("whole")])
    return dict(version=1, blocked=bool(findings), findings=findings,
                incomplete=incomplete[0], functions_inspected=len(visited),
                scope="known whole-training-input answer shortcuts; not a generalization proof")
