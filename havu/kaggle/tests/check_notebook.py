"""Check the built v2 notebook without a GPU.

    .venv/bin/python kaggle/tests/check_notebook.py [--extract-to DIR] [--skip-fake]

1. nbformat validation and `jupyter nbconvert --to script`.
2. Extract every `%%writefile /kaggle/working/arc_solver/...` cell and the prompt cell into DIR (default: a temp dir)
   with the same layout; check they equal the repo sources (solver.py = build_notebook.solver_source()).
3. Import each extracted module.
4. Run fake_server_test.py and fake_tools_test.py against the extracted modules (ARC_TEST_SRC=DIR).
"""
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NBS = {"v2": ROOT / "kaggle/nb_v2/arc2-qwen38-tools-submit.ipynb",
       "v3": ROOT / "kaggle/nb_v3/arc2-qwen38-tools-submit.ipynb",
       "v4": ROOT / "kaggle/nb_v4/arc2-qwen38-tools-submit.ipynb"}  # v4 = the v3 build with v4_settings.json
DEST = "/kaggle/working/arc_solver"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-to")
    ap.add_argument("--skip-fake", action="store_true")
    ap.add_argument("--variant", default="v2", choices=sorted(NBS))
    a = ap.parse_args()
    NB = NBS[a.variant]
    import nbformat
    nb = nbformat.read(str(NB), as_version=4)
    nbformat.validate(nb)
    print("nbformat: valid,", len(nb.cells), "cells")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([str(ROOT / ".venv/bin/jupyter"), "nbconvert", "--to", "script", "--output-dir", td,
                            str(NB)], capture_output=True, text=True)
        print("nbconvert --to script: rc", r.returncode, r.stderr.strip().splitlines()[-1:] if r.stderr else "")
        assert r.returncode == 0
    out = Path(a.extract_to or tempfile.mkdtemp())
    written = {}
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        m = re.match(r"%%writefile (\S+)\n", c.source)
        if m:
            rel = os.path.relpath(m.group(1), DEST)
            written[rel] = c.source[m.end():]
        elif c.source.startswith("ARC_PROMPT = "):
            ns = {}
            exec(c.source.split("\nopen(")[0], ns)
            written["notes/prompts/arcprize_official_system_prompt.txt"] = ns["ARC_PROMPT"]
    for rel, text in written.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        # IPython's %%writefile writes the cell body plus a trailing newline
        p.write_text(text if rel.startswith("notes/prompts/") else text + "\n")
    print("extracted", len(written), "files to", out)
    sys.path[:0] = [str(ROOT / "kaggle")]
    import build_notebook
    assert written["solver.py"] == build_notebook.solver_source(), "embedded solver differs from solver_source()"
    modules = build_notebook.V3_MODULES if a.variant in ("v3", "v4") else build_notebook.V2_MODULES
    for repo, rel in modules:
        assert written[rel] == (ROOT / repo).read_text(), f"{rel} differs from the repo"
    assert written["notes/prompts/arcprize_official_system_prompt.txt"] == \
        (ROOT / "notes/prompts/arcprize_official_system_prompt.txt").read_text()
    print("embedded sources equal the repo sources")
    env = dict(os.environ, ARC_WORK=tempfile.mkdtemp())
    mods = [m[1][len("evals/"):-3].replace("/", ".") for m in modules
            if m[1].endswith(".py") and not m[1].endswith("__init__.py")] + ["solver"]
    code = (f"import sys, importlib; sys.path[:0] = [{str(out)!r}, {str(out / 'evals')!r}]\n"
            f"for m in {mods!r}:\n"
            "    mod = importlib.import_module(m); print('import ok', m, getattr(mod, 'HARNESS_VERSION', ''))\n"
            "import arc_harness.arcprize as A; assert A.TEMPLATE.endswith('Your response:')\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, cwd=str(out))
    print(r.stdout.strip())
    assert r.returncode == 0, r.stderr[-2000:]
    if not a.skip_fake:
        tests = ["fake_server_test.py", "fake_tools_test.py"] + (["fake_v3_test.py"] if a.variant in ("v3", "v4") else [])
        for t in tests:
            r = subprocess.run([sys.executable, str(ROOT / "kaggle/tests" / t)], capture_output=True, text=True,
                               env=dict(os.environ, ARC_TEST_SRC=str(out)), timeout=600)
            print(t, "against extracted modules:", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-500:])
            assert r.returncode == 0 and "ALL OK" in r.stdout
    print("NOTEBOOK CHECK OK")


if __name__ == "__main__":
    main()
