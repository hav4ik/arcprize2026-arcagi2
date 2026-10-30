"""Rule (Manh, 2026-10-30; AGENTS.md): notebook, cell and solver code never scans the input root (/kaggle/input, or
ARC_INPUT_ROOT on the pods) recursively. With his CUDA 13 debs and vLLM wheels attached, one recursive scan walks
thousands of files: ~5 min per call on Kaggle. Inputs are found by cell 0's bounded helper (../input_paths.py).

- the scanner (input_paths.input_scans) catches every banned form and passes scans of our own small trees;
- every cell file, cell template, builder, solver and other live code under kaggle/ is free of input scans (built
  outputs, downloaded runs, review copies and the frozen pinned snapshots of submitted candidates excepted);
- every builder's notebook, built into a temp dir, has no input scan in any code cell, takes cell 0's INPUT_ROOTS
  from its own kernel-metadata.json and passes ARC_MODEL_DIR to the solver; Manh's notebooks as pulled are rewritten;
- find_input finds what the old recursive lookup found on Kaggle's and the pods' mount layouts, and nothing below the
  fixed depths; the cells that run before cell 0 and the solvers' model lookup do the same.
No GPU, no network, no Kaggle.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_input_scans.py'"""
import ast
import glob
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = D.parents[1]
KAGGLE = ROOT / "kaggle"
sys.path[:0] = [str(D), str(HERE)]
import input_paths as IP  # noqa: E402
import test_img as TI  # noqa: E402  (its helpers only; unittest collects classes, not modules)
from test_b32m import manh_stub  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="input_scans_"))
# frozen, sha256-pinned copies of submitted candidates (B32 77fe80a3, B32.1, V6, the unused E064 variant): left
# byte-identical; their only input scan sits behind ARC_MODEL_DIR, which every notebook's solver cell passes
FROZEN = ("pinned_candb", "pinned_candb1", "pinned_mtf_v6", "pinned_manh")
# built outputs, downloaded runs, review copies, tests, vendored vLLM sources, data
SKIP_DIRS = {"out", "runs", "reviews", "tests", "tests_mtf_v6", "vllm_overlay_src", "datasets", "__pycache__"} | set(FROZEN)
MANH_PULL = Path("/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/"
                 "manh_nb_pull")  # his older notebook (build_manh_stack.py), as pulled


def live_sources():
    """Every .py / .ipynb / .sh under kaggle/ that can still be built or run (input_paths.py itself holds the legacy
    patterns it rewrites)."""
    out = []
    for p in sorted(KAGGLE.rglob("*")):
        rel = p.relative_to(KAGGLE).parts
        if (p.suffix in (".py", ".ipynb", ".sh") and p.is_file() and p != D / "input_paths.py"
                and not set(rel[:-1]) & SKIP_DIRS and not rel[0].startswith("nb_")):
            out.append(p)
    return out


def load(path):
    return json.loads(Path(path).read_text())


def scans_of(path):
    if path.suffix == ".ipynb":
        return IP.notebook_scans(load(path))
    return IP.input_scans(path.read_text())


def run(argv):
    r = subprocess.run([sys.executable] + [str(a) for a in argv], capture_output=True, text=True, cwd=str(D))
    assert r.returncode == 0, " ".join(map(str, argv)) + "\n" + r.stdout[-3000:] + r.stderr[-3000:]


def helper_ns(inp, roots=()):
    ns = {"glob": glob, "INPUT": str(inp)}
    exec(compile(IP.HELPER, "cell0_lookup", "exec"), ns)
    ns["INPUT_ROOTS"] = list(roots)
    return ns


def old_lookup(inp, name):
    """The lookup every cell used before 2026-10-30: the sorted first hit of a recursive scan."""
    hits = sorted(glob.glob(f"{inp}/**/{name}", recursive=True))
    return hits[0] if hits else None


def touch(root, *rels):
    for rel in rels:
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    return Path(root)


V5_META = load(D / "out/nb_submit_v4/kernel-metadata.json")
KAGGLE_TREE = ("competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
               "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json",
               "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json",
               "datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz", "datasets/chankhavu/dsv41-tools-kit/MANIFEST.json",
               "datasets/chankhavu/dsv41-pins-x/dsv41_pins.tgz",
               "notebooks/chankhavu/dsv41-sm89-fork-py312-cu130/wheels/vllm-0.28.1rc1+g0-cp312-linux_x86_64.whl",
               "notebooks/chankhavu/dsv41-sm89-fork-py312-cu130/wheels/requirements.lock",
               "models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1/e043_manifest.json",
               "models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1/config.json",
               "models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1/gptq/layer0.json",
               "datasets/nguyennguyen599/vllm-wheels/wheels/wheels/vllm-0.32.1rc1.dev0+gf506afa210-cp313-linux_x86_64.whl",
               "datasets/nguyennguyen599/vllm-wheels/wheels/wheels/torch-2.14.0-cp313-linux_x86_64.whl",
               "datasets/nguyennguyen599/cu13-install/cuda-keyring_1.1-1_all.deb",
               "datasets/nguyennguyen599/arc2-2026-src/third_party/a/b/c/d/setup.py")
LAYOUTS = {  # name: (files under the input root, INPUT_ROOTS)
    "kaggle": (KAGGLE_TREE, IP.input_roots(V5_META)),
    "kaggle_no_roots": (KAGGLE_TREE, []),
    "kaggle_old": (("arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
                    "arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json", "dsv41-tools-kit/dsv41_kit.tgz",
                    "vllm-wheels/wheels/wheels/vllm-0.32.1rc1.dev0+g5e1bccd07c-cp313-linux_x86_64.whl",
                    "dsv41-sm89-fork-py312-cu130/wheels/vllm-0.28.1rc1+g0-cp312-linux_x86_64.whl"), []),
    "pod_e2e": (("arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json", "dsv41-tools-kit/dsv41_kit.tgz",
                 "models/e128/e043_manifest.json"), []),
    "pod_flat": (("dsv41_kit.tgz", "arc-agi_test_challenges.json"), []),
    "test_fixture": (("competitions/arc/arc-agi_test_challenges.json",
                      "kernels/wheels/vllm-0.28.1rc1-cp312-linux_x86_64.whl", "datasets/chankhavu/p/dsv41_pins.tgz"), []),
}
NAMES = ("arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json", "dsv41_kit.tgz", "e043_manifest.json",
         "vllm-0.28.1rc1*.whl", "dsv41_pins.tgz", "MANIFEST.json", "requirements.lock")


class Scanner(unittest.TestCase):
    BANNED = {
        "glob **": 'import glob\nx = glob.glob(f"{INPUT}/**/a.json", recursive=True)\n',
        "find_one **": 'p = find_one(f"{INPUT}/**/x.tgz")\n',
        "literal root **": '_lf = sorted(_lg.glob("/kaggle/input/**/e.json", recursive=True))\n',
        "rglob": 'from pathlib import Path\nm = next(Path("/kaggle/input").rglob("e.json"))\n',
        "os.walk via variable": 'import os\nroot = os.environ.get("ARC_INPUT_ROOT", "/kaggle/input")\n'
                                'for r, d, f in os.walk(root):\n    pass\n',
        "pattern variable": 'import glob\npat = INPUT + "/**/x"\nglob.glob(pat, recursive=True)\n',
        "shell find": '!find /kaggle/input -name "*.whl"\n',
        "shell du": 'import subprocess\nsubprocess.run("du -sh /kaggle/input/datasets", shell=True)\n',
        "shell ls -R": '!ls -R /kaggle/input\n',
        "template": "CELL = r'''\nWHEELS = find_one(f\"{INPUT}/**/vllm-*.whl\")\n'''\n",
        "module INPUT": 'p = S.find_one(f"{S.INPUT}/**/r.json")\n',
    }
    ALLOWED = {
        "vllm install dir": 'import glob\nfiles = glob.glob(f"{_vroot}/models/deepseek_v41/**/*.py", recursive=True)\n',
        "located model dir": 'import os\nfor r, d, f in os.walk(model_dir):\n    pass\n',
        "our kit": 'm = {p: 1 for p in sorted(kit.rglob("*"))}\n',
        "bounded": 'p = find_input("x.tgz")\nq = glob.glob(f"{INPUT}/*/*/x")\n',
        "listing": '!ls /kaggle/input\n!ls -la /kaggle/input/datasets/x\n!cp -r /kaggle/input/datasets/o/s/patches /tmp/\n',
        "working dir": "import glob\nglob.glob('/kaggle/working/cache/**', recursive=True)\n",
        "comment": '# never find_one(f"{INPUT}/**/x")\nx = 1\n',
    }

    def test_banned_forms_are_caught(self):
        for name, text in self.BANNED.items():
            self.assertTrue(IP.input_scans(text), name)

    def test_own_trees_pass(self):
        for name, text in self.ALLOWED.items():
            self.assertEqual(IP.input_scans(text), [], name)

    def test_legacy_rewrite_of_version5_and_manh(self):
        # the notebooks the builders start from (Version 5's, Manh's v12 as pulled) scan the input root; rewrite() and
        # finish() leave none
        for nb_path, meta_path in ((D / "out/nb_submit_v4/dsv41-tools-submit.ipynb", None),
                                   (D / "out/manh_v12_363638159/dsv41-tools-submit-b32m-external-server.ipynb",
                                    D / "out/manh_v12_363638159/kernel-metadata.json")):
            nb = load(nb_path)
            self.assertTrue(IP.notebook_scans(nb), nb_path.name)
            meta = load(meta_path) if meta_path else V5_META
            IP.finish(nb, meta)
            self.assertEqual(IP.notebook_scans(nb), {}, nb_path.name)
            cell0 = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("# Cell 0: clock"))
            self.assertIn(f"INPUT_ROOTS = {json.dumps(IP.input_roots(meta))}\n", cell0)
            self.assertIn('find_input("dsv41_kit.tgz")', cell0)
            self.assertIn('find_input("e043_manifest.json")', cell0)

    def test_input_roots_from_metadata(self):
        self.assertEqual(IP.input_roots(V5_META), [
            "competitions/arc-prize-2026-arc-agi-2", "datasets/chankhavu/dsv41-tools-kit",
            "notebooks/chankhavu/dsv41-sm89-fork-py312-cu130",
            "models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1"])  # Kaggle lower-cases PyTorch
        self.assertEqual(IP.input_roots({"dataset_sources": ["nguyennguyen599/vllm-wheels/10"],
                                         "model_sources": ["o/m/PyTorch/default"]}),
                         ["datasets/nguyennguyen599/vllm-wheels", "models/o/m/pytorch/default/*"])


class LiveSources(unittest.TestCase):
    def test_no_input_scans(self):
        files = live_sources()
        names = {str(p.relative_to(KAGGLE)) for p in files}
        for must in ("dsv41_tools/build.py", "dsv41_tools/build_frozen_mt.py", "dsv41_tools/build_img.py",
                     "dsv41_tools/build_b32m.py", "dsv41_tools/build_img_manh.py", "dsv41_tools/solver_dsv41.py",
                     "dsv41_tools/pinned_img/solver_dsv41.py", "dsv41_tools/stage_model.py",
                     "dsv41_tools/cells/cell0b_pins.py", "dsv41_tools/cells/cell1b_integrity.py",
                     "dsv41_tools/cells/b32m_env.py", "dsv41_tools/cells/manh_m1_setup.py", "src/solver.py",
                     "build_notebook.py", "dsv41_tools/fail_tests.sh"):
            self.assertIn(must, names)
        bad = {str(p.relative_to(KAGGLE)): s for p in files for s in [scans_of(p)] if s}
        self.assertEqual(bad, {})

    def test_frozen_snapshots_scan_only_behind_arc_model_dir(self):
        for d in FROZEN:
            text = (D / d / "solver_dsv41.py").read_text()
            scans = IP.input_scans(text)
            self.assertEqual(scans, ['Path("/kaggle/input").rglob('], d)
            line = next(ln for ln in text.splitlines() if "rglob(" in ln)
            self.assertTrue(line.strip().startswith('model = E("ARC_MODEL_DIR") or next('), d)


class Built(unittest.TestCase):
    """Every builder's notebook, built into a temp dir: no code cell scans the input root, cell 0's INPUT_ROOTS come
    from the notebook's own kernel-metadata.json, and the solver gets ARC_MODEL_DIR."""
    @classmethod
    def setUpClass(cls):
        cls.built = {}  # name -> (notebook dict, metadata dict)
        manh = SCRATCH / "manh_stub"
        manh_stub(manh)
        ext = SCRATCH / "manh_ext"
        manh_stub(ext)
        nb = load(ext / "manh.ipynb")
        nb["cells"][-1]["source"] = TI.MANH_SERVER.splitlines(keepends=True)
        (ext / "manh.ipynb").write_text(json.dumps(nb))
        b = SCRATCH / "build"
        run([D / "build.py", "--out", b])
        for d in sorted(b.glob("nb_*")):
            cls.add(d.name, d)
        pinned = D / "pinned_candb"
        frozen = ["--workers", "56", "--budget", "0.5", "--gate", "0.85", "--retention", "512", "--auto-repair",
                  "--admit-max-waiting", "1", "--harness", pinned / "run_deepseek_tools.py", "--train-check",
                  pinned / "train_check.py"]
        run([D / "build_frozen_mt.py", "--slug", "t-frozen", "--out", SCRATCH / "frozen"] + frozen)
        cls.add("frozen", SCRATCH / "frozen")
        run([D / "build_frozen_mt.py", "--slug", "t-frozen-smoke", "--out", SCRATCH / "frozen_smoke"] + frozen
            + ["--smoke-minutes", "37", "--smoke-stress", "--smoke-hang", "--stall-restart", "600", "--smoke-integrity",
               ROOT / "experiments/E043_dsv41_real_pruning/results/kaggle_e128_manifest.json"])
        cls.add("frozen_smoke", SCRATCH / "frozen_smoke")
        for name, extra in (("b32m", []), ("b32m_load", ["--load-test"])):
            run([D / "build_b32m.py", "--manh-nb", manh, "--out", SCRATCH / name] + extra)
            cls.add(name, SCRATCH / name)
        for name, src, extra in (("img", manh, []), ("img_load", manh, ["--load-test"]),
                                 ("img_ext", ext, ["--external", "--workers", "120"])):
            run([D / "build_img.py", "--manh-nb", src, "--model-source", TI.MODEL, "--out", SCRATCH / name] + extra)
            cls.add(name, SCRATCH / name)
        if (TI.MANH_V4 / "kernel-metadata.json").exists():  # Manh's notebook as pulled (his v6)
            for name, extra in (("img_manh", ["--model-source", TI.MODEL, "--model-manifest", TI.E72_MANIFEST]),
                                ("img_manh_text", ["--text"]),
                                ("img_manh_load", ["--text", "--commit", "load"])):
                run([D / "build_img_manh.py", "--manh-nb", TI.MANH_V4, "--out", SCRATCH / name] + extra)
                cls.add(name, SCRATCH / name)
        if (MANH_PULL / "kernel-metadata.json").exists():
            run([D / "build_manh_stack.py", "--manh-nb", MANH_PULL, "--out", SCRATCH / "manh_stack"])
            cls.add("manh_stack", SCRATCH / "manh_stack")
        cls.local_run = (b / "local_run.py").read_text()

    @classmethod
    def add(cls, name, d):
        meta = load(d / "kernel-metadata.json")
        cls.built[name] = (load(d / meta["code_file"]), meta)

    def test_every_builder_was_built(self):
        want = {"nb_submit", "nb_smoke", "nb_bench", "nb_evalbench-w48", "nb_evalbench-w64", "frozen", "frozen_smoke",
                "b32m", "b32m_load", "img", "img_load", "img_ext"}
        self.assertLessEqual(want, set(self.built))

    def test_no_code_cell_scans_the_input_root(self):
        bad = {name: IP.notebook_scans(nb) for name, (nb, _m) in self.built.items() if IP.notebook_scans(nb)}
        self.assertEqual(bad, {})
        self.assertEqual(IP.input_scans(self.local_run), [])

    def test_input_roots_follow_the_metadata(self):
        for name, (nb, meta) in self.built.items():
            lines = [ln for c in nb["cells"] for ln in "".join(c["source"]).splitlines() if ln.startswith("INPUT_ROOTS = ")]
            if name == "manh_stack":  # its own cell M1 (exact paths), no cell 0
                self.assertEqual(lines, [])
                continue
            self.assertEqual(lines, [f"INPUT_ROOTS = {json.dumps(IP.input_roots(meta))}"], name)
            self.assertTrue(IP.input_roots(meta), name)

    def test_lookups_are_the_bounded_helper(self):
        for name, (nb, _m) in self.built.items():
            text = "".join("".join(c["source"]) for c in nb["cells"])
            if name != "manh_stack":
                self.assertIn('find_input("dsv41_kit.tgz")', text, name)
                self.assertIn('find_input("e043_manifest.json")', text, name)
                self.assertNotIn("recursive=True))\n    return hits[0]", text, name)
            if name.endswith("_load"):
                self.assertIn('_lg.glob(f"/kaggle/input/{_d}arc-agi_evaluation_challenges.json")', text, name)

    def test_rewritten_cells_compile(self):
        for name, (nb, _m) in self.built.items():
            for i, c in enumerate(nb["cells"]):
                text = "".join(c["source"])
                if c["cell_type"] == "code" and ("find_input" in text or "_lf = sorted(" in text):
                    compile(re.sub(r"(?m)^(\s*)[!%].*$", r"\1pass", text), f"{name}:{i}", "exec")

    def test_cell0_finds_the_inputs(self):
        # cell 0 as built, executed on Kaggle's mount layout and on the pods' flat one (ARC_INPUT_ROOT)
        nb, meta = self.built["frozen"]
        cell0 = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("# Cell 0: clock"))
        model = "models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1"
        for layout, kit, files in (("kaggle", "datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz",
                                    KAGGLE_TREE[:3] + (f"{model}/e043_manifest.json", "datasets/o/deep/a/b/c/x.json")),
                                   ("pod_flat", "dsv41_kit.tgz", ("arc-agi_test_challenges.json", "m/e043_manifest.json"))):
            root = Path(tempfile.mkdtemp(prefix=f"cell0_{layout}_", dir=SCRATCH))
            inp = touch(root / "input", *files)
            (inp / kit).parent.mkdir(parents=True, exist_ok=True)
            (inp / kit).symlink_to(D / "kaggle_kit_v5/dsv41_kit.tgz")
            for f in files:
                if f.endswith("challenges.json"):
                    (inp / f).write_text(json.dumps({"t1": {"train": [], "test": [{"input": [[1]]}]}}))
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "KAGGLE_"))}
            env.update(ARC_INPUT_ROOT=str(inp), ARC_OUT=str(root / "work"), ARC_SCRATCH=str(root / "scratch"),
                       ARC_MODE="submit")
            ns, out = {}, io.StringIO()
            with patch.dict(os.environ, env, clear=True), redirect_stdout(out):
                exec(compile(cell0, "cell0", "exec"), ns)
            self.assertNotEqual(ns["MODE"], "fallback", out.getvalue())
            self.assertEqual(ns["CHALLENGES"], old_lookup(inp, "arc-agi_test_challenges.json"))
            self.assertEqual(ns["MODEL_SRC"], str(Path(old_lookup(inp, "e043_manifest.json")).parent))
            self.assertIn(f"kit unpacked from {inp}/{kit}", out.getvalue())
            self.assertEqual(ns["INPUT_ROOTS"], IP.input_roots(meta))

    def test_solver_gets_arc_model_dir(self):
        # the solvers' own lookup is the fallback only: every notebook's solver launch passes ARC_MODEL_DIR
        for name, (nb, _m) in self.built.items():
            srcs = ["".join(c["source"]) for c in nb["cells"]]
            launch = [i for i, s in enumerate(srcs) if "solver_dsv41.py" in s and "Popen(" in s]
            self.assertTrue(launch, name)
            for i in launch:
                self.assertTrue(any("ARC_MODEL_DIR" in s for s in srcs[:i + 1]), (name, i))


class Lookup(unittest.TestCase):
    """find_input / find_inputs on Kaggle's and the pods' mount layouts against the old recursive lookup."""

    def tree(self, name):
        files, roots = LAYOUTS[name]
        inp = touch(Path(tempfile.mkdtemp(prefix=f"{name}_", dir=SCRATCH)) / "input", *files)
        return inp, helper_ns(inp, roots)

    def test_same_hits_as_the_recursive_lookup(self):
        for layout in LAYOUTS:
            inp, ns = self.tree(layout)
            for name in NAMES:
                self.assertEqual(ns["find_input"](name), old_lookup(inp, name), (layout, name))

    def test_manh_wheel_record_and_exact_candidates(self):
        for layout in ("kaggle", "kaggle_old"):
            inp, ns = self.tree(layout)
            got = ns["find_inputs"]("vllm-*.whl", f"{inp}/datasets/*/vllm-wheels/wheels/wheels/vllm-*.whl",
                                    f"{inp}/vllm-wheels/wheels/wheels/vllm-*.whl")
            self.assertEqual(got, sorted(glob.glob(f"{inp}/**/vllm-*.whl", recursive=True)), layout)
        inp, ns = self.tree("kaggle")
        self.assertEqual(ns["find_input"]("dsv41_kit.tgz", f"{inp}/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz"),
                         f"{inp}/datasets/chankhavu/dsv41-tools-kit/dsv41_kit.tgz")
        self.assertIsNone(ns["find_input"]("no_such_file.json"))

    def test_attached_mounts_first_then_fixed_depths_only(self):
        inp, ns = self.tree("kaggle")
        pats = ns["input_patterns"]("x.json")
        self.assertEqual(pats[:4], [f"{inp}/{r}/x.json" for r in IP.input_roots(V5_META)])
        self.assertNotIn("**", "".join(pats))
        # nothing below the fixed depths: a deep file the old recursive lookup found is not looked for
        self.assertEqual(old_lookup(inp, "setup.py"), f"{inp}/datasets/nguyennguyen599/arc2-2026-src/third_party/a/b/c/d/setup.py")
        self.assertIsNone(ns["find_input"]("setup.py"))

    def test_cells_before_cell_0(self):
        # cell 00's load test and build.py's evalbench cell run before cell 0 (fixed depths under /kaggle/input)
        build_src = (D / "build.py").read_text()
        hits_line = next(ln for ln in build_src.splitlines() if ln.startswith("_hits = lambda p:"))
        load = IP.LEGACY[3][1]
        m1 = (D / "cells/manh_m1_setup.py").read_text()
        m1_block = m1[m1.index("    _C = sorted("):m1.index("CHALLENGES = ")]
        final = (KAGGLE / "build_notebook.py").read_text()
        final_block = final[final.index("test_path = ("):final.index("tasks = json.load(open(test_path))")]
        for layout in ("kaggle", "kaggle_old"):
            inp, _ns = self.tree(layout)
            ns = {"glob": glob, "_lg": glob}
            exec(hits_line.replace("/kaggle/input", str(inp)), ns)
            self.assertEqual(ns["_hits"]("arc-agi_evaluation_challenges.json")[:1],
                             [old_lookup(inp, "arc-agi_evaluation_challenges.json")])
            exec(load.strip().replace("/kaggle/input", str(inp)), ns)
            self.assertEqual(ns["_lf"][:1], [old_lookup(inp, "arc-agi_evaluation_challenges.json")])
            exec("import glob\n" + "if True:\n" + m1_block.replace("/kaggle/input", str(inp)), ns)
            self.assertEqual(ns["_C"][:1], [old_lookup(inp, "arc-agi_test_challenges.json")])
            exec(final_block.replace("/kaggle/input", str(inp)), ns)
            self.assertEqual(ns["test_path"], old_lookup(inp, "arc-agi_test_challenges.json"))


def solver_function(path, name):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"Path": Path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class SolverModelDir(unittest.TestCase):
    SOLVERS = (D / "solver_dsv41.py", D / "pinned_img/solver_dsv41.py")

    def test_arc_model_dir_first(self):
        for p in self.SOLVERS:
            self.assertIn('    model = E("ARC_MODEL_DIR") or find_model_dir()\n', p.read_text())

    def test_fixed_depth_lookup(self):
        for p in self.SOLVERS:
            find = solver_function(p, "find_model_dir")
            for layout in ("kaggle", "pod_e2e"):
                files, _r = LAYOUTS[layout]
                inp = touch(Path(tempfile.mkdtemp(dir=SCRATCH)) / "input", *files)
                old = old_lookup(inp, "e043_manifest.json")
                self.assertEqual(find(str(inp)), str(Path(old).parent), (p.parent.name, layout))
            deep = touch(Path(tempfile.mkdtemp(dir=SCRATCH)) / "input", "datasets/o/s/a/b/c/e043_manifest.json")
            self.assertEqual(find(str(deep)), "")
            self.assertEqual(find(str(SCRATCH / "nonexistent")), "")


class QwenSolver(unittest.TestCase):
    """The Qwen pipeline's solver (kaggle/src/solver.py): the same files as its old recursive lookups."""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse((KAGGLE / "src/solver.py").read_text())
        keep = {"INPUT_DEPTHS", "find_one", "find_input", "find_below", "comp_file", "official_model_dir", "source_dir"}
        body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in keep)
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in keep for t in n.targets))]
        cls.code = compile(ast.Module(body=body, type_ignores=[]), "solver_paths", "exec")

    def ns(self, inp):
        ns = {"glob": glob, "os": os, "INPUT": str(inp)}
        exec(self.code, ns)
        return ns

    def test_kaggle_and_old_layouts(self):
        files = ("competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json",
                 "models/mikedan7/qwen3-8-27b-fp8-official/pytorch/hf-fp8/1/config.json",
                 "models/chankhavu/qwen3-8-27b-e032-sft/transformers/fp8block/1/config.json",
                 "models/chankhavu/qwen3-8-27b-e032-sft/transformers/fp8block/1/tokenizer.json")
        old = ("arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json", "qwen3-8-27b-fp8-official/config.json",
               "local/arc-agi_local_challenges.json")
        for tree in (files, old):
            inp = touch(Path(tempfile.mkdtemp(dir=SCRATCH)) / "input", *tree)
            ns = self.ns(inp)
            self.assertEqual(ns["comp_file"]("arc-agi_test_challenges.json"),
                             old_lookup(inp, "arc-agi_test_challenges.json"))
            self.assertEqual(ns["official_model_dir"](),
                             str(Path(old_lookup(inp, "qwen3-8-27b-fp8*/**/config.json")).parent))
        inp = touch(Path(tempfile.mkdtemp(dir=SCRATCH)) / "input", *files)
        ns = self.ns(inp)
        self.assertEqual(ns["source_dir"]("chankhavu/qwen3-8-27b-e032-sft", "tokenizer.json"),
                         f"{inp}/models/chankhavu/qwen3-8-27b-e032-sft/transformers/fp8block/1")
        inp = touch(Path(tempfile.mkdtemp(dir=SCRATCH)) / "input", *old)
        self.assertEqual(self.ns(inp)["find_input"]("arc-agi_local_challenges.json"),
                         old_lookup(inp, "arc-agi_local_challenges.json"))


if __name__ == "__main__":
    unittest.main()
