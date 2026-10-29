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
import time
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
        self.assertEqual(ours, ["# E066 image c", "# Cell 0b (E04", "# Cell 1b (smo", "# Cell 2r (E06", "# Cell 2m (B32",
                                "# Cell 2s (E06", "# Cell 2c (B32", "# Cell 3: the ", "# Cell 3s (E06", "# Cell 3w (E06",
                                "# Cell 4b (B32"])
        w = next(i for i, c in enumerate(self.cells) if c.startswith("# Cell 3w (E066"))
        self.assertTrue(self.cells[w - 1].startswith("# Cell 3s (E066)"))
        self.assertTrue(self.cells[w + 1].startswith("# Cell 4: final check"))
        # every other cell is his, in his order
        self.assertEqual([c for c in self.cells if c in his], [c for c in self.his if c.splitlines()[0][:14] not in
                                                               ("# B32-M + MANH", "# Cell 0b (E04", "# Cell 1b (smo",
                                                                "# Cell 2m (B32", "# Cell 2c (B32", "# Cell 3: the ",
                                                                "# Cell 4b (B32")])
        i = next(i for i, c in enumerate(self.cells) if c.startswith("# Cell 2r (E066"))
        self.assertTrue(self.cells[i - 1].startswith("# Manh's cell 5"))
        self.assertTrue(self.cells[i + 1].startswith("# Cell 2 (B32-M)"))
        server = next(c for c in self.cells if c.startswith("# Cell 2m (B32-M + Manh's server)"))
        his_server = next(c for c in self.his if c.startswith("# Cell 2m (B32-M + Manh's server)"))
        diff = [(a, b) for a, b in zip(his_server.splitlines(), server.splitlines()) if a != b]
        self.assertEqual(len(server.splitlines()), len(his_server.splitlines()))
        self.assertEqual(diff, [("MANH_CONTEXT, MANH_SEQS = 163840, 80", "MANH_CONTEXT, MANH_SEQS = 163840, 64  # E066 build: "
                                 "Manh's 163840, 80 -> 163840, 64 (E066 build margins)")])  # his server cell, 1 line
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
            self.assertEqual(got, {"ARC_GRID_FORMAT": "text+image", "MANH_VISION_MAX_IMAGES": "32", "ARC_WORKERS": "64",
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
        self.assertIn(r'server up after (\d+)s(?: \([^)]*\))?; (.*)', rep)
        import re as _re
        rx = _re.search(r'_re_up = None|re.findall\(r"(server up after[^"]+)"', rep).group(1)
        self.assertTrue(_re.findall(rx, "[x] server up after 412s (external); GPU KV cache size: 1,336,475 tokens"))
        self.assertTrue(_re.findall(rx, "[x] server up after 412s; GPU KV cache size: 1 tokens"))
        self.assertTrue(_re.findall(rx, "[x] server up after 0s (external, start 0); GPU KV cache size: 1 tokens"))
        s0 = self.cells[1]
        self.assertIn('("ARC_CONN_TRIES", "12")', s0)
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


class ServerFlagOptions(unittest.TestCase):
    def test_offload_and_util_edit_only_those_tokens(self):
        out = Path(tempfile.mkdtemp(dir=SCRATCH)) / "nb"
        r = subprocess.run([sys.executable, str(D / "build_img_manh.py"), "--manh-nb", str(MANH_V4), "--out", str(out),
                            "--model-source", MODEL, "--model-manifest", str(E72_MANIFEST), "--kv-offload-gib", "6",
                            "--gpu-util", "0.93"], capture_output=True, text=True, cwd=str(D))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        cells = ["".join(c["source"]) for c in json.load(open(out / "dsv41-tools-submit-img-manh.ipynb"))["cells"]]
        meta = json.load(open(MANH_V4 / "kernel-metadata.json"))
        his = ["".join(c["source"]) for c in json.load(open(MANH_V4 / meta["code_file"]))["cells"]]
        a = next(c for c in his if c.startswith("# Cell 2m (B32-M + Manh's server)")).splitlines()
        b = next(c for c in cells if c.startswith("# Cell 2m (B32-M + Manh's server)")).splitlines()
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(diff), 2, diff)  # MANH_CONTEXT/MANH_SEQS + the one line holding both flags
        self.assertIn('"--gpu-memory-utilization", "0.93", "--kv-offload-gib-per-rank", "6"', diff[1][1])
        self.assertEqual(diff[1][0].replace('"0.94"', '"0.93"').replace('"12"', '"6"'), diff[1][1])


class ContextOption(unittest.TestCase):
    def test_context_moves_his_line_our_limit_and_the_0b_expectation(self):
        out = Path(tempfile.mkdtemp(dir=SCRATCH)) / "nb"
        r = subprocess.run([sys.executable, str(D / "build_img_manh.py"), "--manh-nb", str(MANH_V4), "--out", str(out),
                            "--model-source", MODEL, "--model-manifest", str(E72_MANIFEST), "--context", "131072"],
                           capture_output=True, text=True, cwd=str(D))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        cells = ["".join(c["source"]) for c in json.load(open(out / "dsv41-tools-submit-img-manh.ipynb"))["cells"]]
        s0 = next(c for c in cells if c.startswith("# E066 image candidate"))
        self.assertIn('("ARC_CONTEXT_LIMIT", "131072"),', s0)
        self.assertNotIn('"163840"', s0.split("ARC_CONTEXT_LIMIT", 2)[1][:30])
        c2m = next(c for c in cells if c.startswith("# Cell 2m (B32-M + Manh's server)"))
        self.assertIn("MANH_CONTEXT, MANH_SEQS = 131072, 64", c2m)
        b = next(c for c in cells if c.startswith("# Cell 0b"))
        self.assertIn('"ARC_CONTEXT_LIMIT": "131072"', b)


FAKE_SERVER = r"""
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            body = json.dumps({"data": [{"id": "dsv41", "max_model_len": 131072}]}).encode()
        else:
            body = b'vllm:generation_tokens_total{engine="0"} 5.0\nvllm:num_requests_running{engine="0"} 0.0\n'
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""
FAKE_LAUNCHER = r"""
import json, os, subprocess, sys
a = sys.argv
root, port = a[a.index("--output-root") + 1], a[a.index("--port") + 1]
os.makedirs(f"{root}/server")
p = subprocess.Popen([sys.executable, a[a.index("--server") + 1], port], start_new_session=True,
                     stdout=open(f"{root}/server/server.log", "w"), stderr=subprocess.STDOUT)
json.dump({"pid": p.pid}, open(f"{root}/server/launch.json", "w"))
"""


class Supervisor(unittest.TestCase):
    """Cell 2s: a dead server is relaunched with cell 2m's command and a fresh --output-root; the status file follows
    S26's format; after E066_MAX_RESTARTS it says failed."""
    def test_relaunch_then_give_up(self):
        import signal, socket, threading
        d = Path(tempfile.mkdtemp(dir=SCRATCH))
        (d / "srv.py").write_text(FAKE_SERVER)
        (d / "launch.py").write_text(FAKE_LAUNCHER)
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        cmd = [sys.executable, str(d / "launch.py"), "--output-root", str(d / "dsv41_logs"), "--port", str(port),
               "--server", str(d / "srv.py")]
        subprocess.run(cmd, check=True)
        pid0 = json.load(open(d / "dsv41_logs/server/launch.json"))["pid"]

        def alive(pid):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    return f.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
            except OSError:
                return False
        for _ in range(100):
            try:
                urllib_get(f"http://127.0.0.1:{port}/v1/models")
                break
            except Exception:
                time.sleep(0.1)
        env_keys = ("E066_MAX_RESTARTS", "E066_RESTART_MIN_LEFT_S", "E066_STALL_S", "ARC_STOP_NEW_S")
        old = {k: os.environ.get(k) for k in env_keys}
        os.environ.update(E066_MAX_RESTARTS="1", E066_RESTART_MIN_LEFT_S="0", E066_STALL_S="0", ARC_STOP_NEW_S="99999")
        ns = {"T0": time.time(), "WORK": str(d), "MODE": "submit", "_cmd": cmd, "_env": dict(os.environ),
              "MANH_BASE_URL": f"http://127.0.0.1:{port}/v1", "MANH_SERVER_PID": pid0,
              "MANH_SERVER_LOG": str(d / "dsv41_logs/server/server.log"), "_manh_alive": alive, "_t": time.time()}
        try:
            code_ = (D / "cells/e066_supervisor.py").read_text()
            exec(compile(code_, "cell2s", "exec"), ns)
            ns["_sup_gpu_apps"] = lambda: []  # this box's own GPU processes are not the fake server's
            status = lambda: json.load(open(ns["SUPERVISOR_STATUS"]))
            self.assertEqual((status()["start"], status()["state"], status()["pid"]), (0, "up", pid0))
            os.killpg(pid0, signal.SIGKILL)
            t = time.time()
            while time.time() - t < 60 and not (status()["start"] == 1 and status()["state"] == "up"):
                time.sleep(0.5)
            st = status()
            self.assertEqual((st["start"], st["state"]), (1, "up"), st)
            self.assertNotEqual(st["pid"], pid0)
            self.assertEqual(ns["MANH_SERVER_PID"], st["pid"])
            self.assertTrue(ns["MANH_SERVER_LOG"].endswith("dsv41_logs/r1/server/server.log"))
            self.assertIn("dsv41", urllib_get(f"http://127.0.0.1:{port}/v1/models"))
            os.killpg(st["pid"], signal.SIGKILL)  # second death: no restart left
            t = time.time()
            while time.time() - t < 30 and status()["state"] != "failed":
                time.sleep(0.5)
            self.assertEqual(status()["state"], "failed", status())
            self.assertTrue(any("no restart" in e for e in ns["SUPERVISOR"]["events"]), ns["SUPERVISOR"]["events"])
        finally:
            ns.get("SUPERVISOR", {})["stop"] = True
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for pid in {pid0, ns.get("MANH_SERVER_PID")}:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass

    def test_cells_wired(self):
        cells = build_manh()
        heads = [c.splitlines()[0] for c in cells]
        at = lambda p: next(i for i, h in enumerate(heads) if h.startswith(p))
        self.assertEqual(at("# Cell 2s (E066"), at("# Cell 2m (B32-M") + 1)
        self.assertEqual(at("# Cell 3s (E066)"), at("# Cell 3: the solver") + 1)
        c3 = next(c for c in cells if c.startswith("# Cell 3: the solver"))
        self.assertIn('ARC_EXTERNAL_SERVER_STATUS=globals().get("SUPERVISOR_STATUS") or ""', c3)
        self.assertIn('ARC_MAX_SERVER_RESTARTS="3" if globals().get("SUPERVISOR_STATUS") else "0"', c3)
        compile(c3, "cell3", "exec")


def urllib_get(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


class TextBuild(unittest.TestCase):
    """--text (B32-M fix): his E128 model and cell 1b, B32's harness, MANH_VISION=0, no cell 2r; supervisor kept."""
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(dir=SCRATCH)) / "nb"
        r = subprocess.run([sys.executable, str(D / "build_img_manh.py"), "--text", "--manh-nb", str(MANH_V4), "--out",
                            str(cls.out), "--slug", "t", "--gpu-util", "0.90", "--kv-offload-gib", "10",
                            "--server-env", "VLLM_DSA_DECODE_INDEXER_ROW_SHARD=0"], capture_output=True, text=True,
                           cwd=str(D))
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        cls.cells = ["".join(c["source"]) for c in json.load(open(cls.out / "t.ipynb"))["cells"]]
        cls.meta = json.load(open(cls.out / "kernel-metadata.json"))
        his_meta = json.load(open(MANH_V4 / "kernel-metadata.json"))
        cls.his = ["".join(c["source"]) for c in json.load(open(MANH_V4 / his_meta["code_file"]))["cells"]]
        cls.his_meta = his_meta

    def test_cells(self):
        heads = [c.splitlines()[0] for c in self.cells]
        self.assertFalse(any(h.startswith("# Cell 2r") for h in heads))
        self.assertTrue(any(h.startswith("# Cell 2s (E066") for h in heads))
        s0 = self.cells[1]
        self.assertIn('("MANH_VISION", "0")', s0)
        self.assertNotIn("ARC_GRID_FORMAT", s0)
        b = next(c for c in self.cells if c.startswith("# Cell 0b"))
        self.assertIn('"MANH_VISION": "0"', b)
        self.assertNotIn('"ARC_GRID_FORMAT"', b)
        c1b = next(c for c in self.cells if c.startswith("# Cell 1b"))
        his1b = next(c for c in self.his if c.startswith("# Cell 1b"))
        self.assertIn('state["model_stat"] = stat', c1b)
        self.assertEqual(c1b.split("INTEGRITY = ", 1)[1][:200], his1b.split("INTEGRITY = ", 1)[1][:200])  # his manifest
        rep = next(c for c in self.cells if c.startswith("# Cell 4b"))
        self.assertNotIn("cell 2r's patch outcome", rep)
        self.assertIn("cell 1b's names-and-sizes pass", rep)
        c2m = next(c for c in self.cells if c.startswith("# Cell 2m"))
        self.assertIn('"VLLM_DSA_DECODE_INDEXER_ROW_SHARD": "0",  # E066 build (was "1")', c2m)
        self.assertIn('"--gpu-memory-utilization", "0.90", "--kv-offload-gib-per-rank", "10"', c2m)
        self.assertEqual(self.meta["model_sources"], self.his_meta["model_sources"])

    def test_pins_are_b32s_harness_and_our_solver(self):
        b = next(c for c in self.cells if c.startswith("# Cell 0b"))
        for f, h in (("solver_dsv41.py", sha(P / "solver_dsv41.py")),
                     ("repo/evals/run_deepseek_tools.py", sha(D / "pinned_candb/run_deepseek_tools.py"))):
            self.assertIn(h, b, f)
        self.assertNotIn("grid_image.py", b)


class TextOnV7(unittest.TestCase):
    """--text --no-supervisor on his v7: his supervisor and cell 3 wiring, his cell 1b typo fixed, every cell parses."""
    V7 = Path("/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/manh_ext_nb/v7")

    @unittest.skipUnless(V7.exists(), "his v7 pull not present")
    def test_build(self):
        import ast
        out = Path(tempfile.mkdtemp(dir=SCRATCH)) / "nb"
        r = subprocess.run([sys.executable, str(D / "build_img_manh.py"), "--text", "--no-supervisor", "--manh-nb",
                            str(self.V7), "--out", str(out), "--slug", "t", "--gpu-util", "0.90", "--exec-timeout-s", "900"],
                           capture_output=True, text=True, cwd=str(D))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        cells = ["".join(c["source"]) for c in json.load(open(out / "t.ipynb"))["cells"]]
        meta = json.load(open(self.V7 / "kernel-metadata.json"))
        his = ["".join(c["source"]) for c in json.load(open(self.V7 / meta["code_file"]))["cells"]]
        self.assertFalse(any(c.startswith(("# Cell 2s", "# Cell 3s", "# Cell 2r")) for c in cells))
        self.assertIn(next(c for c in his if c.startswith("# Cell 3: the solver")), cells)  # his cell 3, untouched
        self.assertIn(next(c for c in his if c.startswith("# Cell 4: final check")), cells)
        for c in cells:
            if c.startswith(("# Cell 1b", "# Cell 0b", "# Cell 2m", "# Cell 3", "# Cell 4b", "# B32-M fix")):
                ast.parse(c)
        self.assertNotIn("len(json.load(f))f", next(c for c in cells if c.startswith("# Cell 1b")))


class SolverConnErrors(unittest.TestCase):
    def test_dying_engine_500_is_a_connection_error(self):
        src_ = (P / "solver_dsv41.py").read_text()
        block = src_[src_.index("CONN_ERRORS = ("):src_.index("TASK_LIMIT = ")]
        self.assertIn('"The server had an error while processing your request"', block)  # E066 kill test


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
        (w / "smoke_env.jsonl").write_text(json.dumps({"memory.max": str(173 * 2**30),
                                                       "stat": {"anon": 119 * 2**30, "shmem": 7 * 2**30}}) + "\n")
        self.assertEqual(self.run_checks(w, "server up after 9s (external); submission written (final)"), [])

    def test_bad_run_fails_on_each_problem(self):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        (w / "dsv41_logs").mkdir()
        (w / "e066_vllm_patch.json").write_text(json.dumps([{"file": "/v/model.py", "action": "failed"}]))
        (w / "smoke_integrity.json").write_text(json.dumps({"model_stat": {"files": 137, "missing": ["config.json"],
                                                                           "size_mismatch": []}}))
        (w / "dsv41_logs/attempts.jsonl").write_text(json.dumps({"task_id": "a", "pred": None, "error": "x"}) + "\n")
        bad = self.run_checks(w, "server died (exit 1)")
        self.assertEqual(len(bad), 6, bad)  # + no RAM samples
        for frag in ("E066 vLLM patch model.py: failed", "model files", "solver.log: server died", "no attempt produced",
                     "no valid submission.json"):
            self.assertTrue(any(frag in b for b in bad), (frag, bad))
        self.assertTrue(all(b.startswith("environment") for b in bad[:2]), bad)

    def test_missing_records_are_failures(self):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        bad = self.run_checks(w, "")
        self.assertEqual(len(bad), 5, bad)  # patch record, integrity file, attempts.jsonl, submission.json, no RAM file

    def test_ram_check_without_cgroup_v2_warns(self):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        (w / "dsv41_logs").mkdir()
        (w / "e066_vllm_patch.json").write_text(json.dumps([{"file": "/v/model.py", "action": "patched"}]))
        (w / "smoke_integrity.json").write_text(json.dumps({"model_stat": {"files": 1, "missing": [], "size_mismatch": []}}))
        (w / "dsv41_logs/attempts.jsonl").write_text(json.dumps({"task_id": "a", "pred": [[1]]}) + "\n")
        (w / "submission.json").write_text(json.dumps({"a": [{"attempt_1": [[1]]}]}))
        (w / "smoke_env.jsonl").write_text(json.dumps({"memory.max": None, "stat": {}}) + "\n")  # cgroup v1 pod
        self.assertEqual(self.run_checks(w, "submission written (final)"), [])

    def ram_bad(self, anon_gib, shmem_gib, workers="64"):
        w = Path(tempfile.mkdtemp(dir=SCRATCH))
        (w / "dsv41_logs").mkdir()
        (w / "e066_vllm_patch.json").write_text(json.dumps([{"file": "/v/model.py", "action": "patched"}]))
        (w / "smoke_integrity.json").write_text(json.dumps({"model_stat": {"files": 1, "missing": [], "size_mismatch": []}}))
        (w / "dsv41_logs/attempts.jsonl").write_text(json.dumps({"task_id": "a", "pred": [[1]]}) + "\n")
        (w / "submission.json").write_text(json.dumps({"a": []}) or "{}")
        (w / "submission.json").write_text(json.dumps({"a": [{"attempt_1": [[1]]}]}))
        rows = [{"memory.max": str(173 * 2**30), "stat": {"anon": int(anon_gib * 2**30), "shmem": int(shmem_gib * 2**30)}},
                {"memory.max": str(173 * 2**30), "stat": {"anon": int((anon_gib - 10) * 2**30), "shmem": int(shmem_gib * 2**30)}}]
        (w / "smoke_env.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        old_env = os.environ.get("ARC_SMOKE_ENV_LOG"), os.environ.get("ARC_WORKERS")
        os.environ["ARC_SMOKE_ENV_LOG"] = str(w / "smoke_env.jsonl")
        os.environ["ARC_WORKERS"] = workers
        try:
            return self.run_checks(w, "submission written (final)")
        finally:
            for k, v in zip(("ARC_SMOKE_ENV_LOG", "ARC_WORKERS"), old_env):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_ram_headroom_guard(self):
        # ~147 anon + 7 shmem at smoke peak -> projected 170.4 of 173 (2.6 GiB headroom): fails
        self.assertTrue(any("host RAM: projected" in b for b in self.ram_bad(147, 7)))
        # offload 6 GiB/rank: ~119 + 7 -> projected 142.4: passes
        self.assertEqual(self.ram_bad(119, 7), [])

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
