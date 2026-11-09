"""Q3 (part 1): extract the final Python program from correct DeepSeek tool attempts (E009/E010/E011 lean2sub
full runs), pick one program per solved eval task, compute automatic features, and choose the samples
for manual review.

    .venv/bin/python -I experiments/E015_dsl_analysis/extract_programs.py
Writes programs/<task>.py (one per solved task), programs.jsonl, sample_solved.txt, sample_hard.txt.

Program = the submitted expression (or, for `submit_answer("answer")`, the last top-level statement that
assigned that variable) plus the transitive closure of the session's top-level function definitions it
references (last definition before submission wins). Imports are dropped from the line counts.
"""
import ast
import glob
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARC2_EVAL_DIR, HERE, ROOT  # noqa: E402

RESULTS = f"{ROOT}/evals/results"
RUN_GLOB = "tools_persistent_grid_lean2sub*_full_*.jsonl"

# Regex feature detectors (lower-case source). Coarse; the manual review is the real judgement.
FEATURES = {
    "connected_components": r"ndimage\.label|\blabel\(|def (cc|components|get_comp|comps|flood|bfs|find_obj|"
                            r"get_obj|objects|connected)|deque\(|stack\.pop\(|visited|seen",
    "bounding_box": r"\.min\(\)|\.max\(\)|min\(r|max\(r|min\(c|max\(c|bbox|argwhere|np\.where\(",
    "flood_fill": r"flood|fill_region|binary_fill_holes|enclosed|reach",
    "symmetry": r"rot90|fliplr|flipud|\[::-1\]|\.t\b|transpose|mirror|symmetr",
    "counting": r"counter\(|\.count\(|bincount|most_common|return_counts",
    "recolouring": r"==\s*\d+\]\s*=|\]\s*=\s*\d|recolor|colour_map|color_map|cmap|mapping",
    "tiling_pattern": r"np\.tile|np\.kron|% *period|period|repeat|tile",
    "gravity_movement": r"gravity|shift|move|slide|while .*(==|!=) *0|drop|fall",
    "line_drawing": r"while 0 ?<=|\bray|extend|draw_line|shoot",
    "conditional_rules": r"\bif\b.*\bif\b|elif",
    "shape_matching": r"np\.array_equal|shape ==|template|pattern|match",
    "crop_subgrid": r"\[r0:|\[\w+:\w+\+1, *\w+:\w+\+1\]|crop|subgrid",
}


def load_messages(path):
    try:
        return json.load(gzip.open(path))["messages"]
    except Exception:  # noqa: BLE001
        return None


def tool_calls(msgs):
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:  # noqa: BLE001
                continue
            yield tc["function"]["name"], args.get("code") or ""


def safe_parse(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
           {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def extract(msgs):
    """Return (program_source, entry_kind) or (None, reason)."""
    defs = {}          # name -> source of last top-level def/class
    assigns = defaultdict(list)  # var -> list of (source, names)
    last_cell = {}     # var -> whole cell that last assigned it (fallback for script-style answers)
    submit = None
    for name, code in tool_calls(msgs):
        if name == "submit_answer":
            submit = code
            continue
        if name != "run_python":
            continue
        tree = safe_parse(code)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defs[node.name] = ast.get_source_segment(code, node)
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            assigns[n.id].append(ast.get_source_segment(code, node))
                            last_cell[n.id] = code
            elif isinstance(node, (ast.For, ast.While, ast.If, ast.With)):
                # top-level loops that build the answer in place
                src = ast.get_source_segment(code, node)
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        assigns[n.id].append(src)
                    if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
                        assigns[n.value.id].append(src)
    if submit is None:
        return None, "no_submit"
    st = safe_parse(submit.strip())
    if st is None:
        return None, "submit_unparsable"
    parts = [submit.strip()]
    todo = list(names_in(st))
    seen = set()
    # Expression is a bare variable: pull in the statements that built it (last assignment + in-place loops).
    if isinstance(st.body[-1], ast.Expr) and isinstance(st.body[-1].value, (ast.Name, ast.Attribute, ast.Subscript)):
        var = next(iter(n.id for n in ast.walk(st.body[-1].value) if isinstance(n, ast.Name)), None)
        if var in assigns:
            srcs = assigns[var]
            # last plain assignment and anything after it
            parts = list(dict.fromkeys(srcs[-3:])) + parts
            for s in srcs[-3:]:
                t = safe_parse(s)
                if t is not None:
                    todo += list(names_in(t))
    kind = "function" if any(isinstance(n, ast.Call) for n in ast.walk(st)) else "variable"
    body = []
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in defs:
            body.append(defs[n])
            t = safe_parse(defs[n])
            if t is not None:
                todo += list(names_in(t))
    if body:
        kind = "function"
    else:
        var = next(iter(n.id for n in ast.walk(st) if isinstance(n, ast.Name)), None)
        if var in last_cell:
            parts = [last_cell[var].strip()] + [submit.strip()]
        kind = "script"  # no function: answer built by a top-level script (often hand-entered cells)
    return "\n\n".join(reversed(body)) + "\n\n# submitted:\n" + "\n".join(parts), kind


def loc(src):
    return sum(1 for line in src.splitlines() if line.strip() and not line.strip().startswith(("#", "import ",
                                                                                                "from ")))


def features(src):
    s = src.lower()
    return sorted(k for k, rx in FEATURES.items() if re.search(rx, s))


def main():
    ev = {os.path.basename(f)[:-5] for f in glob.glob(f"{ARC2_EVAL_DIR}/*.json")}
    progs = defaultdict(list)
    attempts = Counter()
    correct = Counter()
    for f in sorted(glob.glob(f"{RESULTS}/{RUN_GLOB}")):
        run = os.path.basename(f)[:-6]
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task_id") not in ev:
                continue
            attempts[r["task_id"]] += 1
            if str(r.get("correct")) != "True":
                continue
            correct[r["task_id"]] += 1
            msgs = load_messages(r.get("trace_path", ""))
            if not msgs:
                continue
            src, kind = extract(msgs)
            if src is None:
                continue
            progs[r["task_id"]].append({"task": r["task_id"], "pair": r.get("pair"), "attempt": r.get("attempt"),
                                        "run": run, "kind": kind, "loc": loc(src), "src": src,
                                        "n_defs": src.count("\ndef ") + src.startswith("def ")})
    os.makedirs(f"{HERE}/programs", exist_ok=True)
    chosen = {}
    for t, ps in progs.items():
        # Prefer function-based programs; among them the median-length one (robust to hard-coded short answers and
        # to bloated exploratory sessions).
        fn = [p for p in ps if p["kind"] == "function" and p["n_defs"] > 0] or ps
        fn.sort(key=lambda p: p["loc"])
        p = fn[len(fn) // 2]
        p["features"] = features(p["src"])
        p["n_correct_programs"] = len(ps)
        p["loc_min"] = fn[0]["loc"]
        chosen[t] = p
        with open(f"{HERE}/programs/{t}.py", "w") as fh:
            fh.write(f"# task {t} pair {p['pair']} attempt {p['attempt']} run {p['run']} (median-length of "
                     f"{len(fn)} function programs)\n" + p["src"] + "\n")
    with open(f"{HERE}/programs.jsonl", "w") as fh:
        for t in sorted(chosen):
            p = chosen[t]
            fh.write(json.dumps({k: v for k, v in p.items() if k != "src"}) + "\n")
    rng = random.Random(15)
    sample = sorted(rng.sample(sorted(chosen), 40))
    open(f"{HERE}/sample_solved.txt", "w").write("\n".join(sample) + "\n")
    # Hard tasks: lowest share of correct attempts in these runs (never-solved tasks first).
    rate = {t: correct[t] / attempts[t] for t in ev if attempts[t]}
    # Pairs never solved by ANY run in evals/results (all models/configs) go first.
    solved_pairs = set()
    for f in glob.glob(f"{RESULTS}/*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task_id") in ev and str(r.get("correct")) == "True":
                solved_pairs.add((r["task_id"], str(r.get("pair"))))
    never = {t for t in ev for i in range(len(json.load(open(f"{ARC2_EVAL_DIR}/{t}.json"))["test"]))
             if (t, str(i)) not in solved_pairs}
    hard = sorted(ev, key=lambda t: (t not in never, rate.get(t, 0), t))[:20]
    open(f"{HERE}/sample_hard.txt", "w").write("\n".join(
        f"{t}\t{correct[t]}/{attempts[t]}\t{'has_never_solved_pair' if t in never else ''}" for t in hard) + "\n")
    locs = sorted(p["loc"] for p in chosen.values())
    print("solved tasks with programs:", len(chosen), "of", len({t for t in correct}))
    print("kinds:", Counter(p["kind"] for p in chosen.values()))
    print("loc median/mean/p90/max:", locs[len(locs) // 2], round(sum(locs) / len(locs), 1),
          locs[int(0.9 * len(locs))], locs[-1])
    print("features:", Counter(f for p in chosen.values() for f in p["features"]).most_common())
    print("hard:", [(t, f"{correct[t]}/{attempts[t]}") for t in hard])


if __name__ == "__main__":
    main()
