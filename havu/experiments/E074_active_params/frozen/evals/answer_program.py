"""E047 --prior-mode verifycode: the program behind an attempt's answer, rebuilt from its transcript (no thinking)."""
import ast, json


def _calls(m):
    out = []
    for x in m:
        for c in x.get("tool_calls") or []:
            if isinstance(c, str):
                continue
            f = c.get("function", {})
            try:
                a = json.loads(f.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            code = a.get("code") or a.get("expr") or ""
            if code:
                out.append((f.get("name"), code))
    return out


def _free(node):
    """Names a definition or statement reads from outside: loads minus its own stores and parameters."""
    loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        stores = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        stores |= {a.arg for a in ast.walk(node) if isinstance(a, ast.arg)}
        stores |= {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n is not node}
        return loads - stores
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)} if node.value else set()
    return loads


def answer_program(messages, max_chars=4000):
    """E047: the program behind an attempt's answer, from its transcript: the last top-level definition of every
    function and the imports/assignments it needs, reachable from the last submit_answer expression (or the last
    cell that sets `answer`). No thinking, no printed output. None when nothing is traceable."""
    calls = _calls(messages)
    defs, assigns, imports, pos = {}, {}, {}, {}
    for name, code in calls:
        if name != "run_python":
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            seg = ast.get_source_segment(code, node)
            if seg is None:
                continue
            pos[seg] = len(pos)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defs[node.name] = (seg, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    imports[(a.asname or a.name).split(".")[0]] = seg
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            assigns[n.id] = (seg, node)
    subs = [c for n, c in calls if n == "submit_answer"]
    root = subs[-1] if subs else "answer"
    try:
        root_tree = ast.parse(root)
    except SyntaxError:
        return None
    need, seen, order = [n.id for n in ast.walk(root_tree) if isinstance(n, ast.Name)], set(), []
    while need:
        k = need.pop()
        if k in seen or k in ("test_input", "test_inputs", "train", "np"):
            continue
        seen.add(k)
        src = defs.get(k) or assigns.get(k)
        if src:
            order.append(src[0])
            need += sorted(_free(src[1]))
        elif k in imports:
            order.append(imports[k])
    if not any(o.lstrip().startswith(("def ", "class ")) or "=" in o for o in order):
        return None
    body = "\n\n".join(sorted(dict.fromkeys(order), key=lambda seg: (not seg.startswith(("import", "from")),
                                                                     pos.get(seg, 0))))
    prog = body + f"\n\n# submitted: {root}"
    return prog if len(prog) <= max_chars else prog[:max_chars] + "\n# [truncated]"
