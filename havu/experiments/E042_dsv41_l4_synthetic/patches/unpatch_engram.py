"""Undo E042's synthetic-fold edits to engram.py (pairs read from apply_patches.py). Usage: <apply_patches.py> <engram.py>"""
import ast
import sys

src = open(sys.argv[1]).read()
pairs = None
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "edit" and len(node.args) == 2:
        if "common/engram.py" in ast.unparse(node.args[0]):
            pairs = ast.literal_eval(node.args[1])
assert pairs, "engram pairs not found"
text = open(sys.argv[2]).read()
for old, new in reversed(pairs):
    assert text.count(new) == 1, (text.count(new), new[:80])
    text = text.replace(new, old)
assert "E042" not in text and "phys_rows" not in text
open(sys.argv[2], "w").write(text)
print(f"reverted {len(pairs)} edits: {sys.argv[2]}")
