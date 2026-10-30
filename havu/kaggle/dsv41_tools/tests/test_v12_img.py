"""V12 + images (build_v12_img.py): Manh's V12 notebook (363638159) with the E058 image switch and nothing else changed.
- the built notebook: only the listed cells change, with exactly the approved lines; the overlay is pinned_v12img/
  (V12's train_check / auto-repair byte-identical); settings, model input, cell 2m flag, new commit-only cells;
- the solver switch: text mode builds exactly V12's harness argv (and Namespace); image mode = V12's minus
  --token-exact plus --grid-format text+image --image-cell 0, --context-limit kept; a bad value exits; MM counters
  only in image mode; the solver diff 11bfaaaa -> new is exactly the approved one;
- the harness: the diff 882c817f -> new is exactly the approved one; the image code is 1d1a281e's byte for byte; on
  V12's text path (token-exact, multi-test, context limit, auto-repair) solve() sends the same requests and returns
  the same records as 882c817f; on the image path the same as 1d1a281e (images, caps, context-full fallback);
- the harness's own image prompts on eval tasks: images decode to the grids at the aligned geometry, <= 32 per prompt;
- cell 2i's helpers (bidirectional source check on the fork's files, probe scoring) and cell 4i's verdict.
No GPU, no network, no Kaggle.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_v12_img.py'"""
import ast
import base64
import copy
import difflib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = D.parents[1]
sys.path[:0] = [str(D), str(ROOT / "evals" / "tests")]
import build_v12_img as B  # noqa: E402
import input_paths as IP  # noqa: E402

P = D / "pinned_v12img"
IMG = D / "pinned_img"
KIT_TGZ = D / "kaggle_kit_v5/dsv41_kit.tgz"
CLONE = Path(os.environ.get("E062_FORK_CLONE", "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/"
                            "81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/e062_port/manh_vllm"))
SCRATCH = Path(tempfile.mkdtemp(prefix="v12img_tests_"))
EVAL = ROOT / "data/arc-agi_evaluation_challenges.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def cell(cells, prefix):
    hits = [src(c) for c in cells if src(c).startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def overlay(cell0b):
    b64 = cell0b.split('_MT = "', 1)[1].split('"', 1)[0]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64))) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


def v12_cells():
    meta = json.loads((B.V12_DIR / "kernel-metadata.json").read_text())
    return json.loads((B.V12_DIR / meta["code_file"]).read_text())["cells"]


V12_FILES = overlay(cell(v12_cells(), "# Cell 0b"))  # V12's own pinned files (11bfaaaa, 882c817f, ...)


def added_removed(a, b):
    """(removed lines, added lines) of the unified diff a -> b, in order."""
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0))
    return ([x[1:] for x in d if x.startswith("-") and not x.startswith("---")],
            [x[1:] for x in d if x.startswith("+") and not x.startswith("+++")])


class Built(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb, cls.meta, cls.v12, cls.v12meta = B.build(integrity=B.MANIFEST, causal=B.CAUSAL_MANIFEST)
        cls.cells = [src(c) for c in cls.nb["cells"]]
        cls.keep = [src(c) for c in B.build(integrity=B.MANIFEST, causal=B.CAUSAL_MANIFEST,
                                            keep_v12_lookups=True)[0]["cells"]]
        cls.old = [src(c) for c in cls.v12["cells"]]

    def test_only_these_cells_change(self):
        # keep-lookups build: V12's cells, in V12's order, with 2i after 2c and 4i last; changed: markdown, 00, 0b, 1b, 2m
        self.assertEqual(len(self.keep), len(self.old) + 2)
        new = [c for c in self.keep if not c.startswith(("# Cell 2i (V12 + images", "# Cell 4i (V12 + images"))]
        changed = [i for i, (a, b) in enumerate(zip(self.old, new)) if a != b]
        self.assertEqual(changed, [0, 1, 4, 6, 13])
        self.assertTrue(self.keep[16].startswith("# Cell 2i") and self.keep[15].startswith("# Cell 2c"))
        self.assertTrue(self.keep[-1].startswith("# Cell 4i") and self.keep[-2].startswith("# B32-M smoke verdict"))
        # default build: plus the bounded lookups (input_paths.finish) in cells 00, 0, 1b, 2, and nothing scans
        nb = copy.deepcopy(B.build(integrity=B.MANIFEST, causal=B.CAUSAL_MANIFEST, keep_v12_lookups=True)[0])
        IP.finish(nb, self.meta)
        self.assertEqual([src(c) for c in nb["cells"]], self.cells)
        self.assertEqual(IP.notebook_scans(self.nb), {})
        self.assertTrue(IP.notebook_scans({"cells": [{"cell_type": "code", "source": c} for c in self.keep]}))

    def test_cell00_lines(self):
        rem, add = added_removed(self.old[1], self.keep[1])
        self.assertEqual(rem, ['               ("ARC_MAX_SERVER_RESTARTS", "3")):'])
        self.assertEqual(add, B.CELL00_HEAD.splitlines() + B.CELL00_SETTINGS_NEW.splitlines()
                         + B.CELL00_MODEL.replace("__MODEL_DIR__", '""').splitlines() + B.CELL00_WHICH.splitlines()
                         + B.CELL00_PRINT.splitlines())

    def test_cell00_settings_run(self):
        for model_dir, want in (("", None), ("/root/v12img_model", "/root/v12img_model")):
            c00 = src(B.build(model_dir=model_dir, keep_v12_lookups=True)[0]["cells"][1])
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_", "KAGGLE_"))}
            env["KAGGLE_IS_COMPETITION_RERUN"] = "1"  # the scored rerun: no smoke settings, no sampler thread
            code = f"import json, os\nexec(compile({c00!r}, 'c00', 'exec'))\nprint('ENV', json.dumps(dict(os.environ)))"
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            got = json.loads(r.stdout.split("ENV ", 1)[1].splitlines()[0])
            self.assertEqual((got["ARC_GRID_FORMAT"], got["MANH_VISION_MAX_IMAGES"], got.get("ARC_MODEL_SRC")),
                             ("text+image", "32", want))
            self.assertEqual(got["ARC_CONTEXT_LIMIT"], "163840")
            self.assertEqual(got["ARC_WORKERS"], "64")

    def test_cell0b_pins_and_overlay(self):
        c = cell(self.nb["cells"], "# Cell 0b")
        files = overlay(c)
        self.assertEqual(files, {arc: p.read_bytes() for arc, p in B.FILES.items()})
        pins = json.loads(c.split("_MT_SHA = ", 1)[1].split("\n", 1)[0])
        self.assertEqual(pins, {arc: sha(b) for arc, b in files.items()})
        for arc in ("repo/evals/train_check.py", "repo/evals/auto_repair.py", "repo/evals/auto_repair_names.json"):
            self.assertEqual(files[arc], V12_FILES[arc], arc)  # V12's, byte for byte
        self.assertEqual(sha(files["repo/evals/arc_harness/grid_image.py"]), sha((IMG / "arc_harness/grid_image.py").read_bytes()))
        self.assertTrue(pins["repo/evals/arc_harness/grid_image.py"].startswith("18ed78fb"))
        expect = json.loads(c.split("_MT_EXPECT = ", 1)[1].split("\n", 1)[0])
        self.assertEqual({k: expect[k] for k in ("ARC_GRID_FORMAT", "MANH_VISION_MAX_IMAGES", "ARC_CONTEXT_LIMIT")},
                         {"ARC_GRID_FORMAT": "text+image", "MANH_VISION_MAX_IMAGES": "32", "ARC_CONTEXT_LIMIT": "163840"})
        need = json.loads(c.split("_MT_NEED = ", 1)[1].split("\n", 1)[0])
        for f, needs in need.items():
            for n in needs:
                self.assertIn(n, files[f].decode(), (f, n))
        rem, add = added_removed(self.old[4], self.keep[4])
        self.assertEqual([x.split(" = ", 1)[0] for x in rem], ["_MT", "_MT_SHA", "_MT_EXPECT", "_MT_NEED"])
        self.assertEqual([x.split(" = ", 1)[0] for x in add if not x.startswith("#")], ["_MT", "_MT_SHA", "_MT_EXPECT", "_MT_NEED"])

    def test_cell0b_runs_and_checks(self):
        # cell 0b on a kit as cell 0 unpacks it: the overlay installs, the hash / feature / settings checks pass
        c00, c0b = self.cells[1], cell(self.nb["cells"], "# Cell 0b")
        for rerun in (False, True):
            kit = SCRATCH / f"kit0b_{rerun}"
            kit.mkdir()
            with tarfile.open(KIT_TGZ) as tf:
                tf.extractall(kit, filter="data")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_", "KAGGLE_"))}
            env["ARC_SMOKE_ENV_LOG"] = str(kit / "smoke_env.jsonl")
            if rerun:
                env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
            code = (f"import os\nns = {{}}\nexec(compile({c00!r}, 'c00', 'exec'), ns)\n"
                    f"ns.update(KIT={str(kit)!r}, MODE='submit', RERUN={rerun!r})\nexec(compile({c0b!r}, 'c0b', 'exec'), ns)\n")
            script = SCRATCH / f"run0b_{rerun}.py"  # cell 0b holds the ~117 kB overlay: too long for argv
            script.write_text(code)
            r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            self.assertIn("E046 CHECK OK", r.stdout, r.stdout[-1500:])
            for arc, p in B.FILES.items():
                self.assertEqual((kit / arc).read_bytes(), p.read_bytes(), arc)

    def test_cell1b_manifest(self):
        c = cell(self.nb["cells"], "# Cell 1b")
        ns = {"json": json, "os": os}
        line = next(i for i, ln in enumerate(c.splitlines()) if ln.startswith("INTEGRITY_BY_MODEL = "))
        code = "\n".join(c.splitlines()[line:line + 3])
        for which, path in (("c1", B.MANIFEST), ("causal", B.CAUSAL_MANIFEST)):
            with patch.dict(os.environ, {"V12IMG_MODEL": which}):
                exec(code, ns)
            self.assertEqual(ns["INTEGRITY"]["model"], json.loads(path.read_text())["model"], which)
        with patch.dict(os.environ, {}):
            os.environ.pop("V12IMG_MODEL", None)
            exec(code, ns)
        man = ns["INTEGRITY"]["model"]  # default: c1
        want = json.loads(B.MANIFEST.read_text())["model"]
        self.assertEqual(man, want)
        c1 = json.loads((ROOT / "experiments/E066_dsv41_image_pruning/results/c1_gptq_sha256_manifest.json").read_text())
        self.assertEqual(man, {e["path"]: [e["size"], e["sha256"]] for e in c1["entries"]})
        self.assertEqual(len(man), 137)
        v12 = json.loads(self.old[6].split('INTEGRITY = json.loads(r"""', 1)[1].split('""")', 1)[0])["model"]
        eng = {k: v for k, v in man.items() if k.startswith("model-engram-")}
        self.assertEqual(eng, {k: v for k, v in v12.items() if k.startswith("model-engram-")})  # E128's 16 shards
        self.assertEqual(len(eng), 16)
        for f in ("config.json", "e043_manifest.json", "model.safetensors.index.json", "model-00001.safetensors",
                  "gptq/layer_00.safetensors", "gptq/layer_39.safetensors", "tokenizer.json"):
            self.assertIn(f, man)
        self.assertFalse([k for k in man if k.startswith(".") or "/." in k])
        rem, add = added_removed(self.old[6], self.keep[6])
        self.assertEqual((len(rem), len(add)), (1, 3))
        self.assertTrue(rem[0].startswith("INTEGRITY = ") and add[0].startswith("INTEGRITY_BY_MODEL = "))
        self.assertTrue(add[2].startswith('INTEGRITY = INTEGRITY_BY_MODEL[os.environ.get("V12IMG_MODEL", "c1")]'))

    def test_cell2m_flag_only_with_images(self):
        rem, add = added_removed(self.old[13], self.keep[13])
        wm = [ln for ln in self.old[13].splitlines() if B.CELL2M_WATERMARK in ln]
        self.assertEqual(len(wm), 1)
        self.assertEqual(rem, wm)  # Manh's --watermark-reload-only, inserted right after --watermark=0.1 (one line)
        self.assertEqual(add, [wm[0].replace(B.CELL2M_WATERMARK, B.CELL2M_WATERMARK_NEW)] + B.CELL2M_ADD.splitlines())
        self.assertIn('"--vllm-arg=--watermark=0.1", "--vllm-arg=--watermark-reload-only", ', self.cells[13])
        self.assertIn('"VLLM_GPU_DRIVER_RESERVE_MB": "256",', self.old[13])  # already V12's, unchanged
        self.assertIn('"VLLM_GPU_DRIVER_RESERVE_MB": "256",', self.cells[13])
        c = self.keep[13]
        i = c.index(B.CELL2M_ANCHOR) + len(B.CELL2M_ANCHOR)
        snippet = "\n".join(ln[8:] for ln in c[i:].splitlines()[:2])
        for fmt, want in (("text+image", ["--vllm-arg=--disable-chunked-mm-input"]), ("json", []), (None, [])):
            ns = {"os": os, "_cmd": []}
            with patch.dict(os.environ, {} if fmt is None else {"ARC_GRID_FORMAT": fmt}):
                if fmt is None:
                    os.environ.pop("ARC_GRID_FORMAT", None)
                exec(snippet, ns)
            self.assertEqual(ns["_cmd"], want, fmt)
        # his launcher turns MANH_VISION_MAX_IMAGES into --vision-max-images (-> --limit-mm-per-prompt) unchanged
        self.assertIn('MANH_VISION_MAX_IMAGES = int(os.environ.get("MANH_VISION_MAX_IMAGES", "16"))', c)
        self.assertIn('"--vision-max-images", str(MANH_VISION_MAX_IMAGES)', c)

    def test_new_cells_commit_only_and_compile(self):
        for prefix in ("# Cell 2i", "# Cell 4i"):
            c = cell(self.nb["cells"], prefix)
            compile(c, prefix, "exec")
            self.assertTrue("RERUN" in c or "KAGGLE_IS_COMPETITION_RERUN" in c)
        self.assertEqual(cell(self.nb["cells"], "# Cell 2i"), (D / "cells/v12img_smoke.py").read_text())
        self.assertEqual(cell(self.nb["cells"], "# Cell 4i"), (D / "cells/v12img_verdict.py").read_text())

    def test_metadata(self):
        m, v = self.meta, self.v12meta
        self.assertEqual(m["model_sources"], [B.MODEL_SOURCE])
        for k in ("dataset_sources", "competition_sources", "kernel_sources", "docker_image", "machine_shape",
                  "enable_gpu", "language", "kernel_type"):
            self.assertEqual(m[k], v[k], k)
        self.assertEqual((m["is_private"], m["enable_internet"], m["id"]), (True, False, "chankhavu/dsv41-tools-v12-img"))
        self.assertNotIn("id_no", m)
        line = next(ln for ln in src(self.nb["cells"][2]).splitlines() if ln.startswith("INPUT_ROOTS = "))
        self.assertIn('"models/arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-2bit/pytorch/default/1"', line)

    def test_cli_build(self):
        out = SCRATCH / "cli"
        r = subprocess.run([sys.executable, str(D / "build_v12_img.py"), "--out", str(out)], capture_output=True,
                           text=True, cwd=str(D))
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        nb = json.loads((out / "dsv41-tools-v12-img.ipynb").read_text())
        self.assertEqual([src(c) for c in nb["cells"]], self.cells)


# ---------------------------------------------------------------- solver switch
SOLVER_PROBE = r'''
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("solver_probe", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path[:0] = [str(m.KIT / "repo" / "evals")]
import run_deepseek_tools as h
cap = {}
orig = h.parser
def parser():
    p = orig()
    pa = p.parse_args
    def parse_args(argv=None, *a, **k):
        cap["argv"] = list(argv)
        return pa(argv, *a, **k)
    p.parse_args = parse_args
    return p
h.parser = parser
m.STATE["model"] = "/m"
a = m.make_args()
print("PROBE " + json.dumps({"argv": cap["argv"], "ns": {k: str(v) for k, v in sorted(vars(a).items())},
                             "images": getattr(m, "IMAGES", None), "metric_keys": list(m.METRIC_KEYS),
                             "replay": getattr(h, "REPLAY", None) is not None,
                             "server_cmd": m.server_cmd()}))
'''
V12_ENV = {"ARC_WORKERS": "64", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_CONTEXT_LIMIT": "163840",
           "ARC_KV_ADMIT_MAX": "0.85", "ARC_KV_ADMIT_MIN_WORKERS": "16", "ARC_AUTO_REPAIR": "1",
           "ARC_ADMIT_MAX_WAITING": "1", "ARC_PORT": "18084", "ARC_MAX_SERVER_RESTARTS": "3",
           "ARC_EXTERNAL_SERVER": "http://127.0.0.1:18084/v1", "ARC_MODEL_DIR": "/m"}


def kit_with(files, name):
    kit = SCRATCH / name
    if not kit.exists():
        kit.mkdir()
        with tarfile.open(KIT_TGZ) as tf:
            tf.extractall(kit, filter="data")
        for arc, data in files.items():
            (kit / arc).parent.mkdir(parents=True, exist_ok=True)
            (kit / arc).write_bytes(data)
    return kit


def probe(files, name, env):
    kit = kit_with(files, name)
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_"))}
    run_env = {**clean, "ARC_KIT": str(kit), "ARC_OUT": str(SCRATCH / f"out_{name}"), **V12_ENV, **env}
    r = subprocess.run([sys.executable, "-c", SOLVER_PROBE, str(kit / "solver_dsv41.py")], capture_output=True,
                       text=True, env=run_env, cwd=str(kit), timeout=120)
    if r.returncode != 0:
        return {"returncode": r.returncode, "stderr": r.stderr[-1500:]}
    out = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("PROBE "))[6:])
    out["ns"] = {k: v.replace(str(kit), "<kit>") for k, v in out["ns"].items()}
    return out


NEW_FILES = {arc: p.read_bytes() for arc, p in B.FILES.items()}


class SolverSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v12 = probe(V12_FILES, "kit_v12", {})
        cls.text = probe(NEW_FILES, "kit_new", {})
        cls.json = probe(NEW_FILES, "kit_new", {"ARC_GRID_FORMAT": "json"})
        cls.img = probe(NEW_FILES, "kit_new", {"ARC_GRID_FORMAT": "text+image"})

    def test_text_mode_is_v12_exactly(self):
        for got in (self.text, self.json):
            self.assertEqual(got["argv"], self.v12["argv"])
            self.assertEqual(got["ns"], self.v12["ns"])
            self.assertEqual(got["metric_keys"], self.v12["metric_keys"])
            self.assertEqual(got["server_cmd"], self.v12["server_cmd"])
            self.assertFalse(got["images"])
        self.assertIn("--token-exact", self.v12["argv"])

    def test_image_mode_args(self):
        v12 = self.v12["argv"]
        i = v12.index("--multi-test")  # the end of HARNESS
        want = [x for x in v12[:i] if x != "--token-exact"] + ["--grid-format", "text+image", "--image-cell", "0"] + v12[i:]
        self.assertEqual(self.img["argv"], want)
        ns = self.img["ns"]
        self.assertEqual((ns["token_exact"], ns["grid_format"], ns["image_cell"], ns["context_limit"], ns["multi_test"],
                          ns["multi_budget"], ns["auto_repair"]), ("False", "text+image", "0", "163840", "True", "0.5", "True"))
        diff = {k for k in ns if ns[k] != self.v12["ns"][k]}
        self.assertEqual(diff, {"token_exact", "grid_format", "image_cell"})
        self.assertTrue(self.img["images"])
        self.assertEqual(self.img["metric_keys"], self.v12["metric_keys"] + ["mm_cache_queries_total", "mm_cache_hits_total"])
        self.assertNotIn("--language-model-only", self.img["server_cmd"])  # only when the solver starts its own server

    def test_bad_value_exits(self):
        got = probe(NEW_FILES, "kit_new", {"ARC_GRID_FORMAT": "image"})
        self.assertNotEqual(got.get("returncode", 0), 0)
        self.assertIn("ARC_GRID_FORMAT must be json or text+image", got["stderr"])

    def test_solver_diff_is_the_approved_one(self):
        rem, add = added_removed(V12_FILES["solver_dsv41.py"].decode(), (P / "solver_dsv41.py").read_text())
        img = (IMG / "solver_dsv41.py").read_text()
        self.assertEqual(rem, [  # the image switch's 2 lines, then R1 / R2's
            '    return [VLLM, "serve", STATE["model"], "--served-model-name", "dsv41", "--host", '
            '"127.0.0.1", "--port", str(PORT),',
            '    return h.parser().parse_args(HARNESS + (["--multi-test"] if MULTI_TEST else [])',
            '                        stop_server()',  # R2: inside try / finally (STATE["restarting"])
            '                        start_server()',
            '                    if not queue or elapsed() >= STOP_NEW_S or STATE["forced"] or STATE.get("admission_closed"):',
            '                    time.sleep(5)',
            '                finished, _ = wait(list(running), timeout=10, return_when=FIRST_COMPLETED)',
            '                        requeue = None',
            '                            if any(c in rec["error"] for c in CONN_ERRORS) and conn_retries.get(key, 0) < CONN_TRIES:',
            '                                # Lost to the server (died, restarting, or dropped the stream): re-queue without using',
            '                                # a try.',
            '                                requeue = ((t, p, a, tries),',
            '                                           f"re-queue {t}.{p}#{a} after a server-side error ({conn_retries[key]})")',
            '                            queue.insert(0, requeue[0])',
            "                            f\"train_pass={rec.get('train_pass')} cuts={rec.get('cuts')} {rec.get('wall_s')}s\")"])
        ported = [x for x in add if x in img.splitlines()]  # verbatim from pinned_img (64ef0a0f)
        new = [x for x in add if x not in img.splitlines()]
        sv = (P / "solver_dsv41.py").read_text()
        for start, end in (("# E066 (opt-in): the E058 image harness", 'IMAGES = GRID_FORMAT == "text+image"\n'),
                           ("    if IMAGES:  # E066: keep the vision encoder", "    return cmd\n"),
                           ('    harness = ([x for x in HARNESS if x != "--token-exact"]', "    return h.parser().parse_args(harness")):
            block = img[img.index(start):img.index(end, img.index(start)) + len(end)]
            self.assertIn(block, sv)  # the image switch, verbatim from pinned_img (64ef0a0f): 15 lines
        self.assertIn('    cmd = [VLLM, "serve", STATE["model"]', sv)
        del ported
        mm = [
            "# images: vLLM's multimodal processor-cache counters (vllm:mm_cache_queries / _hits: images re-sent on later turns",
            "# served from the cache instead of re-processed); not read in text mode",
            'MM_KEYS = ("mm_cache_queries_total", "mm_cache_hits_total")', "if IMAGES:", "    METRIC_KEYS += MM_KEYS",
            '    if IMAGES and "mm_cache_queries_total" in cur:  # images: MM processor-cache hit rate over the window',
            '        mq = d("mm_cache_queries_total")',
            '        tp["mm_queries"], tp["mm_hit"] = round(mq), (round(d("mm_cache_hits_total") / mq, 3) if mq > 0 else None)',
            '                    + (f", MM cache hits {tp[\'mm_hit\']} of {tp[\'mm_queries\']} images" if "mm_queries" in tp else "")']
        self.assertEqual([x for x in new if x in mm], mm)
        r1 = [x for x in new if x not in mm]  # R1 / R2 (tests/test_v12_img_recovery.py, test_v12_img_replay.py)
        for want in ('SERVER_ERRORS = CONN_ERRORS + ("Error code: 500", "Internal Server Error", "APITimeoutError", '
                     '"Request timed out")', "def server_start():", "def server_answers():",
                     "def server_back(since, status, answers):", "import urllib.error", "class Replay:",
                     '    h.REPLAY = Replay() if REPLAY_ON else None  # R2: turn replay after server-side failures'):
            self.assertIn(want, r1)
        for not_ported in ("The server had an error while processing", "def find_model_dir",  # (R1 has its own 500 / timeout)
                           'status.get("state") != "up"'):
            self.assertIn(not_ported, img)
            self.assertNotIn(not_ported, (P / "solver_dsv41.py").read_text(), not_ported)


# ---------------------------------------------------------------- harness
def load_harness(data, name):
    if not (SCRATCH / "arc_helpers.py").exists():  # the harness reads its sibling arc_helpers.py at import (--helpers)
        (SCRATCH / "arc_helpers.py").write_bytes((ROOT / "evals/arc_helpers.py").read_bytes())
    path = SCRATCH / f"{name}.py"
    path.write_bytes(data)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import dsv41_fakes as C  # noqa: E402  (live evals/ on sys.path: deps shared by the three harness versions)

H_V12 = load_harness(V12_FILES["repo/evals/run_deepseek_tools.py"], "rdt_v12_882c")
H_NEW = load_harness((P / "run_deepseek_tools.py").read_bytes(), "rdt_v12img_new")
H_IMG = load_harness((IMG / "run_deepseek_tools.py").read_bytes(), "rdt_img_1d1a")
TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]}, {"input": [[5, 6, 7]], "output": [[7, 6, 5]]}]
T0, T1, T2 = [[1, 0], [0, 2]], [[3, 4, 5]], [[6], [7]]
FLIP = "def solve(g):\n    return np.fliplr(np.array(g))\n"
TASK1 = {"train": TRAIN, "test": [{"input": T0}]}
TASK3 = {"train": TRAIN, "test": [{"input": T0}, {"input": T1}, {"input": T2}]}
VOLATILE = ("seconds",)


def solver_argv(images, extra=(), max_turns=40):
    """The solver's harness argv (from the probe of V12's solver) with a fake base URL."""
    argv = list(SolverSwitch.v12["argv"] if hasattr(SolverSwitch, "v12") else probe(V12_FILES, "kit_v12", {})["argv"])
    if images:
        i = argv.index("--multi-test")
        argv = [x for x in argv[:i] if x != "--token-exact"] + ["--grid-format", "text+image", "--image-cell", "0"] + argv[i:]
    argv[argv.index("--base-url") + 1] = "http://fake:1/v1"
    argv[argv.index("--max-turns") + 1] = str(max_turns)
    return argv + list(extra)


def strip(result):
    return {k: v for k, v in result.items() if k not in VOLATILE}


def canon(x):
    """x with the harness's random tool-call ids renamed in order of appearance (call_<hex>)."""
    text, ids = json.dumps(x), {}
    for m in re.finditer(r"call_[0-9a-f]{16,}", text):
        ids.setdefault(m.group(0), f"call_{len(ids)}")
    for k, v in ids.items():
        text = text.replace(k, v)
    return json.loads(text)


def run_text(mod, turns, task, test_inputs=None, pad=0, extra=()):
    args = mod.parser().parse_args(solver_argv(False, extra))
    srv = C.FakeTokenServer(turns, pad=pad)
    with patch("httpx.post", srv.post):
        trace, result = mod.solve(srv, args, task, task["test"][0]["input"], session_factory=C.local_factory,
                                  **({"test_inputs": test_inputs} if test_inputs else {}))
    return srv.requests, canon(trace["messages"]), canon(strip(result))


class CapturingChat(C.FakeChatServer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sent = []

    def create(self, model, messages, tools, tool_choice, max_tokens, stream, stream_options, extra_body):
        self.sent.append(copy.deepcopy({"messages": messages, "tools": tools, "tool_choice": tool_choice,
                                        "max_tokens": max_tokens, "extra_body": extra_body}))
        return super().create(model, messages, tools, tool_choice, max_tokens, stream, stream_options, extra_body)


def run_image(mod, turns, task, limit, test_inputs=None, pad=0, image_tokens=900, extra=()):
    argv = solver_argv(True, extra)
    if limit is None:
        i = argv.index("--context-limit")
        argv = argv[:i] + argv[i + 2:]
    else:
        argv[argv.index("--context-limit") + 1] = str(limit)
    args = mod.parser().parse_args(argv)
    srv = CapturingChat(turns, limit or 10 ** 9, pad=pad, image_tokens=image_tokens)
    trace, result = mod.solve(srv, args, task, task["test"][0]["input"], session_factory=C.local_factory,
                              **({"test_inputs": test_inputs} if test_inputs else {}))
    return srv.requests, srv.sent, strip(result)


class Harness(unittest.TestCase):
    def test_diff_is_the_approved_one(self):
        rem, add = added_removed(V12_FILES["repo/evals/run_deepseek_tools.py"].decode(), (P / "run_deepseek_tools.py").read_text())
        self.assertEqual(rem, [
            "from arc_harness.grid_image import build_image_prompt",
            "                        msg, usage, finish = turn(client, args, messages, cap, final, think, tok)",  # R2
            '    ap.add_argument("--grid-format", default="json", choices=["json", "image"],',
            '                         "lines) instead of JSON; needs a vision model behind --base-url")',
            '                    help="E033: pixels per grid cell (Qwen3.8: 32 = one LLM visual token per cell, 16 = one ViT patch)")',
            '                    help="E046 (--token-exact): the server\'s max-model-len; each turn\'s max_tokens is capped to the room "',
            '                         "left, and an attempt with less than 1024 tokens of room ends like an output-budget stop "',
            '                         "(session-answer fallback) instead of a context-length error")'])
        img = (IMG / "run_deepseek_tools.py").read_text().splitlines()
        new = (P / "run_deepseek_tools.py").read_text()
        r2 = new[new.index("# R2 (V12 + images"):new.index("def turn(client")].splitlines()  # REPLAY + turn_replayable
        own = [x for x in add if x not in img and x.strip() and x not in r2]  # every other added line is 1d1a281e's
        self.assertEqual(own, [
            '                                    getattr(args, "image_align", "deepseek"))',
            '    replay_state = {"replays": 0}  # R2: turn replays after server-side failures (REPLAY)',
            "                        if tok is None:  # E058 images: the chat path (turn_token_exact caps its own)",
            "                        msg, usage, finish = turn_replayable(client, args, messages, cap, final, think, tok,",
            "                                                             replay_state)",
            '    if replay_state["replays"]:  # R2: only when a turn was replayed (the record is V12\'s otherwise)',
            '        result["replays"], result["replay_errors"] = replay_state["replays"], replay_state["errors"]'])
        self.assertIn("REPLAY = None", r2)  # off unless the solver sets it

    def test_image_code_is_1d1a281e_verbatim(self):
        def defs(path):
            tree = ast.parse(Path(path).read_text())
            text = Path(path).read_text()
            return {n.name if hasattr(n, "name") else ast.unparse(n.targets[0]): ast.get_source_segment(text, n)
                    for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))}
        new, img = defs(P / "run_deepseek_tools.py"), defs(IMG / "run_deepseek_tools.py")
        for name in ("message_token_bound", "chat_context_cap", "text_image_content", "IMAGE_NOTE",
                     "(CHAT_MESSAGE_TOKENS, CHAT_IMAGE_TOKENS, CHAT_CONTEXT_MARGIN, CHAT_TEMPLATE_TOKENS)"):
            self.assertEqual(new[name], img[name], name)
        self.assertEqual(H_NEW.IMAGE_NOTE, H_IMG.IMAGE_NOTES[""])
        self.assertEqual((P / "arc_harness/grid_image.py").read_bytes(), (IMG / "arc_harness/grid_image.py").read_bytes())

    def test_parser_same_namespace_for_v12_argv(self):
        argv = solver_argv(False)
        self.assertEqual(vars(H_NEW.parser().parse_args(argv)), vars(H_V12.parser().parse_args(argv)))

    def test_text_path_same_as_v12(self):
        # V12's text path: --token-exact, multi-test with --multi-budget, --context-limit, --auto-repair
        flip_all = FLIP + "print(solve(test_inputs[0]))"
        scripts = [
            (TASK1, None, [C.ds_call_text(FLIP + "print(solve(test_input))"),
                           C.ds_call_text("solve(test_input)", name="submit_answer")], 0),
            (TASK3, [T0, T1, T2], [C.ds_call_text(flip_all), C.ds_call_text("[solve(t) for t in test_inputs]",
                                                                            name="submit_answer")], 0),
            (TASK1, None, [C.ds_call_text("print(undefined_nmae)"), C.ds_call_text(FLIP + "answer = solve(test_input)"),
                           C.ds_text_answer("done")], 0),  # auto-repair + session answer
            (TASK3, [T0, T1, T2], [C.ds_call_text(FLIP + "answer = [solve(t) for t in test_inputs]")], 163000),  # ContextFull
        ]
        for task, tests, turns, pad in scripts:
            a = run_text(H_V12, list(turns), task, tests, pad=pad)
            b = run_text(H_NEW, list(turns), task, tests, pad=pad)
            self.assertEqual(a[0], b[0])  # (prompt ids, max_tokens) per request
            self.assertEqual(a[1], b[1])  # the transcript
            self.assertEqual(a[2], b[2])  # the record
        self.assertEqual(a[2]["stop_reason"], "context")

    def test_image_path_same_as_1d1a281e(self):
        long = FLIP + "answer = solve(test_input)\n" + "#" * 6000
        n1 = run_image(H_IMG, [("submit_answer", "solve(test_input)")], TASK1, None, pad=500)[0][0][0]
        n1b = run_image(H_IMG, [("submit_answer", "solve(test_input)")], TASK1, None, pad=500, image_tokens=2600)[0][0][0]
        cases = [  # (turns, task, tests, limit, pad, image tokens)
            ([("run_python", FLIP + "print(solve(test_input))"), ("submit_answer", "solve(test_input)")], TASK1, None,
             163840, 0, 900),
            ([("run_python", FLIP), ("submit_answer", "[solve(t) for t in test_inputs]")], TASK3, [T0, T1, T2], 163840, 0,
             900),
            ([("run_python", long), ("submit_answer", "solve(test_input)")], TASK1, None, n1 + 64000 + 3000, 500, 900),
            ([("run_python", FLIP + "answer = solve(test_input)\nprint('#' * 6000)")], TASK1, None, n1b + 5000, 500, 2600),
        ]
        for turns, task, tests, limit, pad, it in cases:
            a = run_image(H_IMG, list(turns), task, limit, tests, pad=pad, image_tokens=it)
            b = run_image(H_NEW, list(turns), task, limit, tests, pad=pad, image_tokens=it)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[1], b[1])  # every request body: messages with the images, tools, max_tokens
            self.assertEqual(a[2], b[2])
            self.assertTrue(all(n + m <= limit for n, m in b[0]), b[0])
            imgs = [p for p in b[1][0]["messages"][-1]["content"] if p["type"] == "image_url"]
            self.assertEqual(len(imgs), 2 * len(task["train"]) + len(tests or [1]))
        self.assertLess(b[0][0][1], 64000)  # the last case: capped by the chat bound


# ---------------------------------------------------------------- image prompts on eval tasks
@unittest.skipUnless(EVAL.exists(), "no local eval challenges")
class EvalPrompts(unittest.TestCase):
    LIMIT = 32

    def test_images_render_and_stay_under_the_limit(self):
        from PIL import Image
        from arc_harness.grid_image import PALETTE, aligned_cell
        tasks = json.load(open(EVAL))
        counts = {}
        for t, task in tasks.items():
            tests = [x["input"] for x in task["test"]]
            parts = H_NEW.text_image_content("P", task["train"], tests if len(tests) > 1 else tests[:1], 0)
            counts[t] = sum(p["type"] == "image_url" for p in parts)
        self.assertLessEqual(max(counts.values()), self.LIMIT)
        self.assertEqual(max(counts.values()), 14)  # 36a08778: 6 train pairs + 2 test inputs
        # through solve() (the solver's image argv), first request of a few tasks: decode every image
        big = max(tasks, key=lambda t: max(len(g) * len(g[0]) for p in tasks[t]["train"] for g in (p["input"], p["output"])))
        for t in ("36a08778", "b6f77b65", big):
            task = tasks[t]
            tests = [x["input"] for x in task["test"]] if len(task["test"]) > 1 else None
            _, sent, _ = run_image(H_NEW, [("submit_answer", "[[0]]" if not tests else "[[[0]]] * %d" % len(tests))],
                                   task, 163840, tests)
            parts = sent[0]["messages"][-1]["content"]
            grids = [g for p in task["train"] for g in (p["input"], p["output"])] + (tests or [task["test"][0]["input"]])
            imgs = [p for p in parts if p["type"] == "image_url"]
            self.assertEqual(len(imgs), len(grids), t)
            self.assertLessEqual(len(imgs), self.LIMIT)
            self.assertTrue(parts[0]["text"].endswith(H_NEW.IMAGE_NOTE))
            for g, p in zip(grids, imgs):
                im = Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))).convert("RGB")
                cell = aligned_cell(g)
                self.assertEqual(im.size, (len(g[0]) * cell, len(g) * cell), t)
                for r in range(len(g)):
                    for c in range(len(g[0])):
                        self.assertEqual(im.getpixel((c * cell + cell // 2, r * cell + cell // 2)), PALETTE[g[r][c]])


# ---------------------------------------------------------------- cells 2i / 4i
def smoke_ns():
    text = (D / "cells/v12img_smoke.py").read_text()
    body = text[:text.index("IMG_SMOKE = {}")]
    ns = {"WORK": str(SCRATCH), "os": os, "json": json, "time": __import__("time"), "subprocess": subprocess,
          "sys": sys}
    exec(compile(body, "cell2i", "exec"), ns)
    return ns


class SmokeCells(unittest.TestCase):
    def test_probe_scoring(self):
        ns = smoke_ns()
        g = [[1, 2], [3, 4]]
        ok = '```json\n{"rows": 2, "cols": 2, "colors": [1, 2, 3, 4], "corners": [1, 2, 3, 4], "grid": [[1, 2], [3, 4]]}\n```'
        self.assertEqual(ns["_vi_score"](g, ns["_vi_parse"](ok)),
                         {"parsed": True, "size": True, "colors": True, "corners": True, "exact": True})
        bad = ns["_vi_score"](g, ns["_vi_parse"]('{"rows": 2, "cols": 3, "grid": [[1, 2]]}'))
        self.assertEqual((bad["size"], bad["exact"]), (False, False))
        self.assertEqual(ns["_vi_score"](g, ns["_vi_parse"]("no json"))["parsed"], False)
        self.assertEqual(len(ns["IMG_PROBE_GRIDS"]), 15)
        self.assertTrue(all(len(x) * len(x[0]) <= 36 for x in ns["IMG_PROBE_GRIDS"].values()))

    @unittest.skipUnless((CLONE / ".git").exists(), "fork clone not available")
    def test_source_check_on_the_fork(self):
        ns = smoke_ns()
        rel = "vllm/transformers_utils/configs/deepseek_v41.py"
        for commit, bad in (("9f9e1dac26^", True), ("9f9e1dac26", False), ("22df163d2d", False), ("8cf0c0664b", False)):
            text = subprocess.run(["git", "-C", str(CLONE), "show", f"{commit}:{rel}"], capture_output=True,
                                  check=True).stdout
            root = SCRATCH / f"vroot_{commit.replace('^', 'p')}"
            (root / "transformers_utils/configs").mkdir(parents=True, exist_ok=True)
            (root / "transformers_utils/configs/deepseek_v41.py").write_bytes(text)
            self.assertEqual(bool(ns["_vi_source_check"](str(root))), bad, commit)
        self.assertTrue(ns["_vi_source_check"](str(SCRATCH / "no_such_vllm")))

    def test_verdict_cell(self):
        text = (D / "cells/v12img_verdict.py").read_text().replace("for _ in range(60)", "for _ in range(1)")
        for content, raises in (({"hard_done": True, "done": True, "fail": [], "warn": ["w"], "probe": {"exact": 9}}, False),
                                ({"hard_done": True, "done": False, "fail": [], "warn": []}, False),
                                ({"hard_done": True, "done": True, "fail": ["bidirectional"], "warn": []}, True),
                                ({"hard_done": False}, True), (None, True)):
            out = SCRATCH / f"verdict_{abs(hash(json.dumps(content)))}"
            out.mkdir(exist_ok=True)
            if content is not None:
                (out / "smoke_image.json").write_text(json.dumps(content))
            env = {"ARC_OUT": str(out), "ARC_GRID_FORMAT": "text+image"}
            with patch.dict(os.environ, env), patch("time.sleep"):
                os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
                if raises:
                    with self.assertRaises(RuntimeError):
                        exec(compile(text, "cell4i", "exec"), {})
                else:
                    exec(compile(text, "cell4i", "exec"), {})
        with patch.dict(os.environ, {"KAGGLE_IS_COMPETITION_RERUN": "1", "ARC_GRID_FORMAT": "text+image",
                                     "ARC_OUT": str(SCRATCH / "nothing")}):
            exec(compile(text, "cell4i", "exec"), {})  # the scored rerun: nothing checked, nothing raised


if __name__ == "__main__":
    unittest.main()
