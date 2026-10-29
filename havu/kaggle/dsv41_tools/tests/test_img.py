"""E066 image candidate (build_img.py): B32-M + the E058 image harness + a vision-tower checkpoint. Checks the built
notebook pins pinned_img/ (image solver switch, frozen live harness, current grid_image.py), sets the image switch and
the server's per-prompt image limit, takes the E066 Kaggle Model, and applies both sha256-guarded vLLM patches (real
diffs against the wheel's 5e1bccd07c files when the fork clone is available); and that the solver's ARC_GRID_FORMAT
switch builds the image harness args (no --token-exact) and keeps the vision encoder, while the default is unchanged.
No GPU, no network, no Kaggle.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_img.py'"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
P = D / "pinned_img"
KIT_TGZ = D / "kaggle_kit_v5/dsv41_kit.tgz"
CLONE = Path(os.environ.get("E062_FORK_CLONE", "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/"
                            "81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/e062_port/manh_vllm"))
MODEL = "chankhavu/dsv41-flash-1017-arc-tool-img-e72-152-avg120-2bit/PyTorch/default/1"
sys.path.insert(0, str(HERE))
from test_b32m import manh_stub  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="img_tests_"))


def build(out, *extra):
    manh = SCRATCH / "manh"
    if not manh.exists():
        manh_stub(manh)
    r = subprocess.run([sys.executable, str(D / "build_img.py"), "--manh-nb", str(manh), "--out", str(out),
                        "--model-source", MODEL, *extra], capture_output=True, text=True, cwd=str(D))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    nb = json.load(open(out / "dsv41-tools-submit-img.ipynb"))
    return ["".join(c["source"]) for c in nb["cells"]], json.load(open(out / "kernel-metadata.json"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Built(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells, cls.meta = build(SCRATCH / "nb")

    def test_pins_are_pinned_img(self):
        cell0b = next(c for c in self.cells if c.startswith("# Cell 0b"))
        pins = json.loads(cell0b.split("_MT_SHA = ", 1)[1].split("\n", 1)[0])
        want = {"solver_dsv41.py": P / "solver_dsv41.py", "repo/evals/run_deepseek_tools.py": P / "run_deepseek_tools.py",
                "repo/evals/arc_harness/grid_image.py": P / "arc_harness/grid_image.py",
                "repo/evals/train_check.py": P / "train_check.py", "repo/evals/auto_repair.py": P / "auto_repair.py",
                "repo/evals/auto_repair_names.json": P / "auto_repair_names.json"}
        self.assertEqual(pins, {k: sha(v) for k, v in want.items()})
        # B32's train_check and auto-repair are unchanged
        self.assertEqual(pins["repo/evals/train_check.py"][:8], "8fe36175")
        self.assertEqual(pins["repo/evals/auto_repair.py"][:8], "2292369c")
        expect = json.loads(cell0b.split("_MT_EXPECT = ", 1)[1].split("\n", 1)[0])
        self.assertEqual(expect["ARC_GRID_FORMAT"], "text+image")
        need = json.loads(cell0b.split("_MT_NEED = ", 1)[1].split("\n", 1)[0])
        self.assertIn("def chat_context_cap", need["repo/evals/run_deepseek_tools.py"])
        self.assertIn("def aligned_cell", need["repo/evals/arc_harness/grid_image.py"])

    def test_settings_and_model(self):
        s0 = self.cells[0]
        self.assertIn('("ARC_GRID_FORMAT", "text+image")', s0)
        self.assertIn("--limit-mm-per-prompt", s0)
        self.assertEqual(self.meta["model_sources"], [MODEL])
        self.assertEqual(self.meta["docker_image_pinning_type"], "latest")

    def test_patch_cell_replaces_rowchunk_cell(self):
        cell = next(c for c in self.cells if c.startswith("# Cell 2r (E066"))
        self.assertNotIn("__", re.sub(r"__pycache__|__file__", "", cell.split("IMG_PATCHES = ", 1)[1].split("]\n", 1)[0]))
        self.assertFalse(any(c.startswith("# Cell 2r (B32-M plan B)") for c in self.cells))
        for rel in ("vllm/models/deepseek_v41/nvidia/model.py", "vllm/model_executor/layers/fused_moe/router/dsv4_topk.py"):
            self.assertIn(rel, cell)


def patch_ns():
    cell = (D / "cells/img_patches.py").read_text()
    body = cell[cell.index("def apply_unified"):cell.index("IMG_PATCH_RECORDS = []")]
    g = {"os": os, "re": re, "hashlib": hashlib, "glob": __import__("glob")}
    exec(body, g)
    return g


MANH_SERVER = ("import json\nimport os\nos.environ.update({\n    \"VLLM_HOST_ALLREDUCE\": \"1\",\n"
               "    \"VLLM_HOST_ALLREDUCE_MODE\": \"60\",\n})\nimport time\n"
               "server_root = f\"/kaggle/working/dsv41_server_{time.time_ns()}\"\n"
               "!python -u /kaggle/working/dsv41_vllm.py \\\n  --qwen38-variant dsv4-vllm \\\n"
               "  --dsv41-vllm-model /kaggle/input/models/chankhavu/x/pytorch/default/1 \\\n"
               "  --output-root {server_root} \\\n  --qwen38-port 18084 \\\n  --context 128000 \\\n"
               "  --concurrency 64 \\\n  --vision-max-pixels 2097152 \\\n  --vllm-arg=--offload-backend=uva \\\n"
               "  --vllm-arg=--watermark=0.1\n")


class External(unittest.TestCase):
    """--external: Manh's server cell (his launcher and flags) serves; the solver attaches (ARC_EXTERNAL_SERVER)."""
    @classmethod
    def setUpClass(cls):
        manh = SCRATCH / "manh_ext"
        manh_stub(manh)
        nb = json.load(open(manh / "manh.ipynb"))
        nb["cells"][-1] = {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": MANH_SERVER.splitlines(keepends=True)}  # his server cell (the stub's is minimal)
        json.dump(nb, open(manh / "manh.ipynb", "w"))
        out = SCRATCH / "nb_ext"
        r = subprocess.run([sys.executable, str(D / "build_img.py"), "--manh-nb", str(manh), "--out", str(out),
                            "--model-source", MODEL, "--external", "--workers", "120"],
                           capture_output=True, text=True, cwd=str(D))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        cls.cells = ["".join(c["source"]) for c in json.load(open(out / "dsv41-tools-submit-img.ipynb"))["cells"]]

    def test_server_cell_edits_only(self):
        srv = next(c for c in self.cells if c.startswith("# Manh's server cell"))
        for want in ("--dsv41-vllm-model {E066_MODEL}", "--context 163840", "--concurrency 120", "--qwen38-port 18084", "--vision-max-images 32 \\\n",
                     '"VLLM_HOST_ALLREDUCE_MODE": "556"', "--vllm-arg=--watermark=0.1 \\\n",
                     "--vllm-arg=--tool-call-parser=deepseek_v41", "--vllm-arg=--enable-auto-tool-choice",
                     'os.environ["ARC_EXTERNAL_SERVER_LOG"] = f"{server_root}/server/server.log"'):
            self.assertIn(want, srv)
        self.assertNotIn("--context 128000", srv)
        i = next(i for i, c in enumerate(self.cells) if c.startswith("# Manh's server cell"))
        self.assertTrue(self.cells[i - 1].startswith("# Cell 2s (E066 --external)"))
        self.assertLess(i, next(j for j, c in enumerate(self.cells) if c.startswith("# Cell 3: the solver")))
        self.assertGreater(i, next(j for j, c in enumerate(self.cells) if c.startswith("# Cell 2r (E066")))
        self.assertFalse(any(c.startswith("# Cell 2m (B32-M)") for c in self.cells))

    def test_settings_attach_the_solver(self):
        s0 = self.cells[0]
        self.assertIn('("ARC_EXTERNAL_SERVER", "http://127.0.0.1:18084/v1")', s0)
        self.assertIn('("ARC_PORT", "18084")', s0)
        self.assertIn('("ARC_WORKERS", "120")', s0)
        cell0b = next(c for c in self.cells if c.startswith("# Cell 0b"))
        expect = json.loads(cell0b.split("_MT_EXPECT = ", 1)[1].split("\n", 1)[0])
        self.assertEqual(expect["ARC_EXTERNAL_SERVER"], "http://127.0.0.1:18084/v1")


class Patches(unittest.TestCase):
    def test_synthetic_variants_fixed_unknown(self):
        g = patch_ns()
        h = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
        base1, fixed1 = "a\nb\nc\nd\n", "a\nb\nX\nc\nd\n"
        base2, fixed2 = "a\nb\nq\nc\nd\n", "a\nb\nq\nX\nc\nd\n"
        variants = [{"wheel": "w1", "base": h(base1), "fixed": h(fixed1), "diff": "@@ -2,2 +2,3 @@\n b\n+X\n c\n"},
                    {"wheel": "w2", "base": h(base2), "fixed": h(fixed2), "diff": "@@ -3,2 +3,3 @@\n q\n+X\n c\n"}]
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        p = d / "m.py"
        for base, fixed, wheel in ((base1, fixed1, "w1"), (base2, fixed2, "w2")):
            p.write_text(base)
            rec = g["guarded_patch"](p, False, variants)
            self.assertEqual((rec["action"], rec["wheel"], p.read_text()), ("patched", wheel, fixed))
            self.assertEqual(g["guarded_patch"](p, False, variants)["action"], "already_fixed")
        p.write_text("other\n")
        with self.assertRaises(RuntimeError):
            g["guarded_patch"](p, False, variants)
        self.assertEqual(g["guarded_patch"](p, True, variants)["action"], "rerun_left_as_is")
        self.assertEqual(p.read_text(), "other\n")

    @unittest.skipUnless((CLONE / ".git").exists(), "fork clone not available")
    def test_real_diffs_on_the_wheels_files(self):
        g = patch_ns()
        cell = next(c for c in build(SCRATCH / "nb2")[0] if c.startswith("# Cell 2r (E066"))
        ns = {}
        exec("import base64\n" + cell[cell.index("IMG_PATCHES = "):cell.index("def apply_unified")], ns)
        commits = {"g5e1bccd07c": "5e1bccd07c", "gf506afa210 / g71dca8bc24": "f506afa210",
                   "g5e1bccd07c .. g71dca8bc24": "5e1bccd07c"}
        n = 0
        for pt in ns["IMG_PATCHES"]:
            for v in pt["variants"]:
                base = subprocess.run(["git", "-C", str(CLONE), "show", f"{commits[v['wheel']]}:{pt['rel']}"],
                                      capture_output=True, check=True).stdout
                self.assertEqual(hashlib.sha256(base).hexdigest(), v["base"], (pt["rel"], v["wheel"]))
                self.assertEqual(hashlib.sha256(v["diff"].encode()).hexdigest(), v["diff_sha"])
                d = Path(tempfile.mkdtemp(dir=SCRATCH))
                p = d / Path(pt["rel"]).name
                p.write_bytes(base)
                rec = g["guarded_patch"](p, False, pt["variants"])
                self.assertEqual(rec["action"], "patched", rec)
                self.assertEqual(sha(p), v["fixed"])
                n += 1
        self.assertEqual(n, 3)
        # the f506 target is exactly the dynamic-count-moe branch's model.py
        want = subprocess.run(["git", "-C", str(CLONE), "show", "dynamic-count-moe:vllm/models/deepseek_v41/nvidia/model.py"],
                              capture_output=True, check=True).stdout
        self.assertEqual(hashlib.sha256(want).hexdigest(), ns["IMG_PATCHES"][0]["variants"][1]["fixed"])


SOLVER_PROBE = r'''
import json, os, sys, importlib.util
spec = importlib.util.spec_from_file_location("solver_img", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path[:0] = [str(m.KIT / "repo" / "evals")]
m.STATE["model"] = "/m"
a = m.make_args()
print(json.dumps({"images": m.IMAGES, "grid_format": a.grid_format, "image_cell": a.image_cell,
                  "token_exact": a.token_exact, "context_limit": a.context_limit, "multi_test": a.multi_test,
                  "lmo": "--language-model-only" in m.server_cmd(), "extra": m.server_cmd()[-2:]}))
'''


def probe(env):
    kit = SCRATCH / f"kit_{abs(hash(json.dumps(env, sort_keys=True)))}"
    if not kit.exists():
        kit.mkdir()
        with tarfile.open(KIT_TGZ) as tf:
            tf.extractall(kit, filter="data")
        for src, arc in ((P / "solver_dsv41.py", "solver_dsv41.py"), (P / "run_deepseek_tools.py", "repo/evals/run_deepseek_tools.py"),
                         (P / "arc_harness/grid_image.py", "repo/evals/arc_harness/grid_image.py"),
                         (P / "train_check.py", "repo/evals/train_check.py")):
            (kit / arc).write_bytes(src.read_bytes())
    clean = {k: v for k, v in os.environ.items() if not k.startswith("ARC_")}
    run_env = {**clean, "ARC_KIT": str(kit), "ARC_OUT": str(kit / "out"), "ARC_MULTI_TEST": "1",
               "ARC_CONTEXT_LIMIT": "163840", "ARC_SERVER_EXTRA_ARGS": json.dumps(["--limit-mm-per-prompt", '{"image": 64}']),
               **env}
    r = subprocess.run([sys.executable, "-c", SOLVER_PROBE, str(kit / "solver_dsv41.py")], capture_output=True,
                       text=True, env=run_env, cwd=str(kit))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


class SolverSwitch(unittest.TestCase):
    def test_image_switch(self):
        out = probe({"ARC_GRID_FORMAT": "text+image"})
        self.assertEqual((out["images"], out["grid_format"], out["image_cell"], out["token_exact"]),
                         (True, "text+image", 0, False))
        self.assertEqual((out["context_limit"], out["multi_test"], out["lmo"]), (163840, True, False))
        self.assertEqual(out["extra"], ["--limit-mm-per-prompt", '{"image": 64}'])

    def test_default_is_b32s_text_harness(self):
        out = probe({})
        self.assertEqual((out["images"], out["grid_format"], out["token_exact"], out["lmo"]), (False, "json", True, True))

    def test_bad_value_refused(self):
        with self.assertRaises(AssertionError):
            probe({"ARC_GRID_FORMAT": "image"})


if __name__ == "__main__":
    unittest.main()


MANH_V4 = Path("/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/"
               "manh_ext_nb/v6")


E72_MANIFEST = D.parents[1] / "experiments/E066_dsv41_image_pruning/results/e72c152_gptq_sha256_manifest.json"


@unittest.skipUnless((MANH_V4 / "kernel-metadata.json").exists(), "Manh's v4 notebook not pulled")
class ManhBase(unittest.TestCase):
    """build_img_manh.py: Manh's own notebook (his serving) + only our image / E72-152 cells."""
    @classmethod
    def setUpClass(cls):
        out = SCRATCH / "nb_manh"
        r = subprocess.run([sys.executable, str(D / "build_img_manh.py"), "--manh-nb", str(MANH_V4), "--out", str(out),
                            "--model-source", MODEL, "--model-manifest", str(E72_MANIFEST)],
                           capture_output=True, text=True, cwd=str(D))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        cls.cells = ["".join(c["source"]) for c in json.load(open(out / "dsv41-tools-submit-img-manh.ipynb"))["cells"]]
        cls.meta = json.load(open(out / "kernel-metadata.json"))
        meta = json.load(open(MANH_V4 / "kernel-metadata.json"))
        cls.his = ["".join(c["source"]) for c in json.load(open(MANH_V4 / meta["code_file"]))["cells"]]

    def test_only_our_cells_change(self):
        his = set(self.his)
        ours = [c.splitlines()[0][:14] for c in self.cells if c not in his]
        self.assertEqual(ours, ["# E066 image c", "# Cell 0b (E04", "# Cell 1b (smo", "# Cell 2r (E06", "# Cell 2c (B32",
                                "# Cell 3w (E06", "# Cell 4b (B32"])
        w = next(i for i, c in enumerate(self.cells) if c.startswith("# Cell 3w (E066"))
        self.assertTrue(self.cells[w - 1].startswith("# Cell 3: the solver"))
        self.assertTrue(self.cells[w + 1].startswith("# Cell 4: final check"))
        # every other cell is his, in his order
        self.assertEqual([c for c in self.cells if c in his], [c for c in self.his if c.splitlines()[0][:14] not in
                                                               ("# B32-M + MANH", "# Cell 0b (E04", "# Cell 1b (smo",
                                                                "# Cell 2c (B32", "# Cell 4b (B32")])
        i = next(i for i, c in enumerate(self.cells) if c.startswith("# Cell 2r (E066"))
        self.assertTrue(self.cells[i - 1].startswith("# Manh's cell 5"))
        self.assertTrue(self.cells[i + 1].startswith("# Cell 2 (B32-M)"))
        server = next(c for c in self.cells if c.startswith("# Cell 2m (B32-M + Manh's server)"))
        self.assertIn(server, self.his)  # his production server cell, byte-identical
        self.assertEqual(self.meta["model_sources"], [MODEL])
        self.assertEqual(self.meta["docker_image_pinning_type"], "latest")
        self.assertNotIn("docker_image", self.meta)

    def test_settings_and_pins_cell_run(self):
        s0 = next(c for c in self.cells if c.startswith("# E066 image candidate"))
        cell0b = next(c for c in self.cells if c.startswith("# Cell 0b"))
        for rerun in (True, False):
            kit = Path(tempfile.mkdtemp(dir=SCRATCH)) / "kit"
            kit.mkdir()
            with tarfile.open(KIT_TGZ) as tf:
                tf.extractall(kit, filter="data")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "KAGGLE_", "MANH_"))}
            env["ARC_SMOKE_ENV_LOG"] = str(kit / "smoke_env.jsonl")
            if rerun:
                env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
            code_ = f"""
import os, sys, json
os.environ.clear(); os.environ.update({env!r})
ns = {{}}
exec(compile({s0!r}, "cell00", "exec"), ns)
ns.update(KIT={str(kit)!r}, MODE="submit", RERUN={rerun!r})
exec(compile({cell0b!r}, "cell0b", "exec"), ns)
print("ENV", json.dumps({{k: os.environ.get(k) for k in ("ARC_GRID_FORMAT", "MANH_VISION_MAX_IMAGES", "ARC_WORKERS",
                                                         "MANH_VISION")}}))
print("LIGHTENV", json.dumps({{k: os.environ.get(k) for k in ("ARC_TASK_IDS", "E066_LIGHT_SMOKE", "ARC_STOP_NEW_S",
                                                           "ARC_FORCE_AT_S", "ARC_HARD_STOP_S", "ARC_KILL_AT_S",
                                                           "B32M_LOAD_TEST")}}))
"""
            script = kit.parent / "run_cells.py"
            script.write_text(code_)
            r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            self.assertIn("E046 CHECK OK", r.stdout, r.stdout[-1500:])
            got = json.loads(r.stdout.split("ENV ", 1)[1].splitlines()[0])
            self.assertEqual(got, {"ARC_GRID_FORMAT": "text+image", "MANH_VISION_MAX_IMAGES": "32", "ARC_WORKERS": "80",
                                   "MANH_VISION": "1"})
            light = json.loads(r.stdout.split("LIGHTENV ", 1)[1].splitlines()[0])
            if rerun:  # the scored rerun: none of the light smoke's settings
                self.assertEqual(light, dict.fromkeys(light))
            else:
                self.assertEqual(light, {"ARC_TASK_IDS": "00576224,25ff71a9", "E066_LIGHT_SMOKE": "1", "ARC_STOP_NEW_S": "4200",
                                         "ARC_FORCE_AT_S": "4320", "ARC_HARD_STOP_S": "4380", "ARC_KILL_AT_S": "4470",
                                         "B32M_LOAD_TEST": None})
            self.assertEqual(sha(kit / "solver_dsv41.py"), sha(P / "solver_dsv41.py"))
            self.assertEqual(sha(kit / "repo/evals/arc_harness/grid_image.py"), sha(P / "arc_harness/grid_image.py"))

    def test_contract_has_image_checks(self):
        c2 = next(c for c in self.cells if c.startswith("# Cell 2c "))
        self.assertIn('if os.environ.get("ARC_GRID_FORMAT") == "text+image":', c2)
        self.assertIn("import re, sys, threading, urllib.request", c2)
        compile(c2, "cell2c", "exec")


    def test_integrity_is_the_e66_manifest_and_report_fixes(self):
        c1b = next(c for c in self.cells if c.startswith("# Cell 1b "))
        m = json.load(open(E72_MANIFEST))
        data = json.loads(c1b.split('"""', 2)[1]) if '"""{' in c1b else None
        self.assertIn(m["entries"][0]["sha256"], c1b)
        self.assertNotIn("model-00001-of-00002", c1b)
        rep = next(c for c in self.cells if c.startswith("# Cell 4b "))
        self.assertIn(r'server up after (\d+)s(?: \(external\))?; (.*)', rep)
        import re as _re
        rx = _re.search(r'_re_up = None|re.findall\(r"(server up after[^"]+)"', rep).group(1)
        self.assertTrue(_re.findall(rx, "[x] server up after 412s (external); GPU KV cache size: 1,336,475 tokens"))
        self.assertTrue(_re.findall(rx, "[x] server up after 412s; GPU KV cache size: 1 tokens"))
        self.assertIn('globals().get("MANH_SERVER_LOG")', rep)
        self.assertIn('(("contract",) if os.environ.get("ARC_GRID_FORMAT") == "text+image" else ())', rep)
        compile(rep, "cell4b", "exec")


def build_manh():
    out = Path(tempfile.mkdtemp(dir=SCRATCH)) / "nb"
    r = subprocess.run([sys.executable, str(D / "build_img_manh.py"), "--manh-nb", str(MANH_V4), "--out", str(out),
                        "--model-source", MODEL, "--model-manifest", str(E72_MANIFEST)], capture_output=True, text=True,
                       cwd=str(D))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return ["".join(c["source"]) for c in json.load(open(out / "dsv41-tools-submit-img-manh.ipynb"))["cells"]]


class LightSmokeChecks(unittest.TestCase):
    """Audit 2026-10-29: the light smoke's verdict needs real answers and no crash, records cell 2r's outcome and the
    model files' names and sizes, and cell 3w keeps the server up until the contract is done."""
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(D))
        import build_img_manh as bim
        cls.bim = bim
        cells = build_manh()
        cls.rep = next(c for c in cells if c.startswith("# Cell 4b "))
        cls.wait = next(c for c in cells if c.startswith("# Cell 3w "))

    def run_checks(self, w, sl):
        ns = {"_w": str(w), "_sl": sl, "_b32m_bad": [], "json": json, "os": os}
        for snip in (self.bim.E066_REPORT_CHECKS, self.bim.E066_LIGHT_CHECKS):
            self.assertIn(snip, self.rep)
            exec("if True:\n" + snip, ns)
        return ns["_b32m_bad"]

    def setUp(self):
        self._old = os.environ.get("E066_LIGHT_SMOKE")
        os.environ["E066_LIGHT_SMOKE"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("E066_LIGHT_SMOKE", None)
        else:
            os.environ["E066_LIGHT_SMOKE"] = self._old

    def test_good_run_passes(self):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        (w / "dsv41_logs").mkdir()
        (w / "e066_vllm_patch.json").write_text(json.dumps([{"file": "/v/model.py", "action": "patched"},
                                                            {"file": "/v/dsv4_topk.py", "action": "already_fixed"}]))
        (w / "smoke_integrity.json").write_text(json.dumps({"model_stat": {"files": 137, "missing": [], "size_mismatch": []}}))
        (w / "dsv41_logs/attempts.jsonl").write_text(json.dumps({"task_id": "a", "pred": [[1]], "correct": True}) + "\n")
        (w / "submission.json").write_text(json.dumps({"a": [{"attempt_1": [[1]], "attempt_2": [[1]]}]}))
        self.assertEqual(self.run_checks(w, "server up after 9s (external); submission written (final)"), [])

    def test_bad_run_fails_on_each_problem(self):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        (w / "dsv41_logs").mkdir()
        (w / "e066_vllm_patch.json").write_text(json.dumps([{"file": "/v/model.py", "action": "failed"}]))
        (w / "smoke_integrity.json").write_text(json.dumps({"model_stat": {"files": 137, "missing": ["config.json"],
                                                                           "size_mismatch": []}}))
        (w / "dsv41_logs/attempts.jsonl").write_text(json.dumps({"task_id": "a", "pred": None, "error": "x"}) + "\n")
        bad = self.run_checks(w, "server died (exit 1)")
        self.assertEqual(len(bad), 5, bad)
        for frag in ("E066 vLLM patch model.py: failed", "model files", "solver.log: server died", "no attempt produced",
                     "no valid submission.json"):
            self.assertTrue(any(frag in b for b in bad), (frag, bad))
        self.assertTrue(all(b.startswith("environment") for b in bad[:2]), bad)

    def test_missing_records_are_failures(self):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        bad = self.run_checks(w, "")
        self.assertEqual(len(bad), 4, bad)  # patch record, integrity file, attempts.jsonl, submission.json

    def test_wait_cell_joins_the_contract_thread(self):
        import io, contextlib, threading, time
        t = threading.Thread(target=time.sleep, args=(0.3,))
        t.start()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(self.wait, {"RERUN": False, "_ct_thread": t})
        self.assertFalse(t.is_alive())
        self.assertIn("contract thread finished", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(self.wait, {"RERUN": True})  # the scored rerun: nothing
        self.assertEqual(out.getvalue(), "")


PROBE_TEST = r'''
import importlib.util, json, socket, sys
spec = importlib.util.spec_from_file_location("solver_img", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.EXTERNAL_PROBE_S = 0
seq = []
def fake(addr, timeout=5):
    e = seq.pop(0)
    if e is not None:
        raise e
    class C:
        def close(self): pass
    return C()
socket.create_connection = fake
out = {}
s = m.ExternalServer()
seq[:] = [TimeoutError(), TimeoutError(), None, TimeoutError(), TimeoutError()]
out["two_timeouts_then_ok"] = [s.poll() for _ in range(5)]
seq[:] = [TimeoutError()]
out["third_in_a_row"] = s.poll()
s = m.ExternalServer()
seq[:] = [ConnectionRefusedError()]
out["refused"] = s.poll()
print(json.dumps(out))
'''


class ExternalProbe(unittest.TestCase):
    def test_timeouts_need_three_in_a_row_refused_is_dead_at_once(self):
        kit = SCRATCH / "kit_probe"
        if not kit.exists():
            kit.mkdir()
            with tarfile.open(KIT_TGZ) as tf:
                tf.extractall(kit, filter="data")
            (kit / "solver_dsv41.py").write_bytes((P / "solver_dsv41.py").read_bytes())
        env = {k: v for k, v in os.environ.items() if not k.startswith("ARC_")}
        env.update(ARC_KIT=str(kit), ARC_OUT=str(kit / "out"), ARC_EXTERNAL_SERVER="http://127.0.0.1:1/v1")
        r = subprocess.run([sys.executable, "-c", PROBE_TEST, str(kit / "solver_dsv41.py")], capture_output=True,
                           text=True, env=env, cwd=str(kit))
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(out["two_timeouts_then_ok"], [None, None, None, None, None])
        self.assertEqual(out["third_in_a_row"], 1)
        self.assertEqual(out["refused"], 1)
