"""B32-M (build_b32m.py): B32's notebook with Manh's environment. Checks the built notebook keeps B32's pinned code
byte-identical and swaps only the environment layer, that Manh's extra flags are OFF by default (and what ON adds),
and the background API contract check against a fake server. No GPU, no network, no Kaggle.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_b32m.py'"""
import hashlib
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import common as C

D = C.DSV41
MANH_NB = Path(tempfile.gettempdir())  # replaced in setUpClass by a minimal copy of his notebook's cells


def manh_stub(folder):
    """A stand-in for `kaggle kernels pull` of his notebook: his five install cells' shapes + clock + server cell."""
    cells = ["import time\nglobal_end_time = time.time() + 12 * 3600 - 600\n",
             "import subprocess\nproc = subprocess.Popen('sudo dpkg -i /kaggle/input/datasets/nguyennguyen599/cu13-install/x.deb', shell=True)\n",
             "!rm -r /kaggle/working/*\n!cp /kaggle/input/datasets/nguyennguyen599/arc2-2026-src/* /kaggle/working/\n",
             "%env UV_OFFLINE=1\nimport os\nos.environ['UV_FIND_LINKS'] = '/dev/shm/wheels'\ndef copy_item(x):\n    pass\n",
             "!uv pip install --no-build-isolation vllm torch torchvision -U\nproc.wait()\n\nif proc.returncode != 0:\n"
             "    raise RuntimeError(\"CUDA installation failed\")\n\nprint(\"CUDA installation finished.\")\n",
             "!uv pip uninstall tensorflow jax\n!rm -r /dev/shm/wheels\n",
             "!python -u /kaggle/working/dsv41_vllm.py --qwen38-port 18084\n"]
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": c.splitlines(keepends=True)} for c in cells], "metadata": {}, "nbformat": 4,
          "nbformat_minor": 5}
    folder.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(folder / "manh.ipynb", "w"))
    json.dump({"id": "nguyennguyen599/arc2-2026-harness-deepseek", "code_file": "manh.ipynb",
               "docker_image": "gcr.io/kaggle-images/python@sha256:" + "a" * 64}, open(folder / "kernel-metadata.json", "w"))


def build(out, *extra):
    r = subprocess.run([sys.executable, str(D / "build_b32m.py"), "--manh-nb", str(MANH_NB), "--out", str(out), *extra],
                       capture_output=True, text=True, cwd=str(D))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    nb = json.load(open(out / "dsv41-tools-submit-b32m.ipynb"))
    return ["".join(c["source"]) for c in nb["cells"]], json.load(open(out / "kernel-metadata.json"))


class Built(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global MANH_NB
        tmp = Path(tempfile.mkdtemp(prefix="b32m_", dir=C.SCRATCH))
        MANH_NB = tmp / "manh"
        manh_stub(MANH_NB)
        cls.cells, cls.meta = build(tmp / "off")
        cls.cells_on, _ = build(tmp / "on", "--manh-flags")

    def test_pinned_code_is_b32s(self):
        cell0b = next(c for c in self.cells if c.startswith("# Cell 0b"))
        sha = json.loads(cell0b.split("_MT_SHA = ", 1)[1].split("\n", 1)[0])
        want = {"solver_dsv41.py": "solver_dsv41.py", "repo/evals/run_deepseek_tools.py": "run_deepseek_tools.py",
                "repo/evals/train_check.py": "train_check.py", "repo/evals/auto_repair.py": "auto_repair.py",
                "repo/evals/auto_repair_names.json": "auto_repair_names.json"}
        self.assertEqual(sha, {k: hashlib.sha256((D / "pinned_candb" / v).read_bytes()).hexdigest()
                               for k, v in want.items()})
        self.assertEqual({k: v[:8] for k, v in sha.items()}["solver_dsv41.py"], "77fe80a3")

    def test_environment_layer_swapped(self):
        text = "\n".join(self.cells)
        self.assertNotIn("# Cell 2: offline install", text)  # B32's fork wheels / overlay / venv
        self.assertNotIn("vllm_overlay_rw", text)
        self.assertNotIn("!rm -r /kaggle/working/*", [l.strip() for l in text.splitlines()])
        self.assertIn("if not RERUN:  # B32-M", text)
        self.assertIn("uv pip install --no-build-isolation vllm", text)
        self.assertIn('-m vllm.entrypoints.cli.main "$@"', text)
        self.assertEqual(self.meta["kernel_sources"], [])
        self.assertNotIn("docker_image", self.meta)
        self.assertEqual(self.meta["docker_image_pinning_type"], "latest")
        self.assertTrue(self.meta["enable_gpu"])
        self.assertEqual(self.meta["machine_shape"], "NvidiaL4")
        self.assertEqual(self.meta["dataset_sources"][0], "chankhavu/dsv41-tools-kit")
        self.assertIn("nguyennguyen599/vllm-wheels", self.meta["dataset_sources"])
        order = [next(i for i, c in enumerate(self.cells) if c.startswith(p)) for p in (
            "# Cell 0: clock", "# Cell 0a (B32-M GPU gate)", "# Manh's cell 1", "# Manh's cell 5", "# Cell 2r (B32-M plan B)", "# Cell 2 (B32-M)",
            "# Cell 2m", "# Cell 2b", "# Cell 2c (B32-M", "# Cell 3: the solver", "# Cell 4b (B32-M",
            "# Cell 5 (smoke report", "# B32-M smoke verdict")]
        self.assertEqual(order, sorted(order))

    def test_settings_are_the_l4_replicas(self):
        s0 = self.cells[0]
        for needle in ('("ARC_WORKERS", "80")', '("ARC_MAX_NUM_SEQS", "96")', '("ARC_KV_ADMIT_MAX", "0.85")',
                       '("ARC_ADMIT_MAX_WAITING", "1")', '"--decode-context-parallel-size", "4"',
                       '"--cp-kv-cache-interleave-size", "64"', '\\"offload_prompt_only\\":false',
                       'VLLM_HUMMING_MOE_PAD_MULTIPLE', r'max_cudagraph_capture_size\": 96'):
            self.assertIn(needle, s0, needle)

    def test_manh_flags_off_by_default(self):
        off = next(c for c in self.cells if c.startswith("# Cell 2m"))
        on = next(c for c in self.cells_on if c.startswith("# Cell 2m"))
        self.assertIn("MANH_FLAGS = False", off)
        self.assertIn("MANH_FLAGS = True", on)
        env = {"ARC_SERVER_ENV_JSON": json.dumps({"VLLM_PLUGINS": ""}), "ARC_SERVER_EXTRA_ARGS": "[]"}
        g = {"json": json, "BASE_ENV": dict(env)}
        exec(off, g)
        self.assertEqual(g["BASE_ENV"], env)
        g = {"json": json, "BASE_ENV": dict(env)}
        exec(on, g)
        se = json.loads(g["BASE_ENV"]["ARC_SERVER_ENV_JSON"])
        self.assertEqual(se["VLLM_HOST_ALLREDUCE_MODE"], "60")
        self.assertEqual(se["VLLM_PLUGINS"], "")
        self.assertIn("1152x5120", json.loads(se["VLLM_HUMMING_MOE_TUNING_CONFIG"]))


class FakeServer:
    def __init__(self, token_ids=True, max_len=163840):
        outer = self
        self.token_ids, self.max_len = token_ids, max_len

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def reply(self, obj, text=False):
                b = (obj if text else json.dumps(obj)).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.reply({"data": [{"id": "dsv41", "max_model_len": outer.max_len}]})
                else:
                    self.reply("vllm:kv_cache_usage_perc 0.1\nvllm:num_requests_running 0\n", text=True)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    text = json.dumps(body["messages"])
                    self.reply({"tokens": [ord(ch) for ch in text if "reasoning\"" in text or ch], "count": len(text)})
                elif self.path == "/detokenize":
                    self.reply({"prompt": "".join(chr(t) for t in body["tokens"])})
                elif self.path == "/v1/completions":
                    self.reply({"choices": [{"text": "x", "token_ids": [1, 2] if outer.token_ids else None}]})
                else:
                    self.reply({"choices": [{"message": {"tool_calls": [{"function": {"name": "run_python"}}]}}]})
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def contract_ns():
    g = {"json": json, "os": __import__("os"), "time": __import__("time"), "T0": 0.0, "WORK": tempfile.mkdtemp(
        dir=C.SCRATCH), "RERUN": True, "MODE": "submit"}
    exec((D / "cells/b32m_contract.py").read_text(), g)
    return g


class Contract(unittest.TestCase):
    def test_passes_on_a_good_server(self):
        srv = FakeServer()
        try:
            out = contract_ns()["contract_run"](srv.base, wait_s=5)
        finally:
            srv.close()
        self.assertEqual(out["fail"], [])
        self.assertTrue(out["thinking_via_reasoning_content"])
        self.assertEqual(out["completion_token_ids"], 2)
        self.assertTrue(out["metrics"]["kv_cache_usage_perc"])

    def test_fails_without_token_ids_or_short_context(self):
        srv = FakeServer(token_ids=False, max_len=128000)
        try:
            out = contract_ns()["contract_run"](srv.base, wait_s=5)
        finally:
            srv.close()
        self.assertEqual(len(out["fail"]), 2, out["fail"])

    def test_no_server(self):
        out = contract_ns()["contract_run"]("http://127.0.0.1:9", wait_s=1)
        self.assertTrue(out["fail"] and "never answered" in out["fail"][0])


def rowchunk_ns(rerun=False):
    g = {"os": __import__("os"), "MODE": "fallback", "RERUN": rerun}
    text = (D / "cells/b32m_rowchunk.py").read_text()
    diff = (D / "b32m_patches/a6c3ee8c2b_engram_row_chunks.diff").read_bytes()
    import base64
    text = text.replace("__DIFF_B64__", base64.b64encode(diff).decode()).replace(
        "__DIFF_SHA__", hashlib.sha256(diff).hexdigest())
    exec(text, g)
    return g


SCR = Path("/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad")


class RowChunkPatch(unittest.TestCase):
    def tmpfile(self, text):
        d = Path(tempfile.mkdtemp(dir=C.SCRATCH))
        (d / "__pycache__").mkdir()
        (d / "__pycache__" / "model.cpython-313.pyc").write_bytes(b"x")
        (d / "model.py").write_text(text)
        return d / "model.py"

    def test_synthetic_base_fixed_unknown(self):
        g = rowchunk_ns()
        base = "a\nb\nc\nd\n"
        diff = "--- a/m.py\n+++ b/m.py\n@@ -2,2 +2,3 @@\n b\n+X\n c\n"
        fixed = "a\nb\nX\nc\nd\n"
        h = lambda t: hashlib.sha256(t.encode()).hexdigest()
        p = self.tmpfile(base)
        rec = g["rowchunk_patch"](p, False, h(base), h(fixed), diff)
        self.assertEqual((rec["action"], p.read_text(), rec["pyc_removed"]), ("patched", fixed, ["model.cpython-313.pyc"]))
        self.assertEqual(g["rowchunk_patch"](p, False, h(base), h(fixed), diff)["action"], "already_fixed")
        q = self.tmpfile("other\n")
        with self.assertRaises(RuntimeError):
            g["rowchunk_patch"](q, False, h(base), h(fixed), diff)
        rec = g["rowchunk_patch"](q, True, h(base), h(fixed), diff)
        self.assertEqual((rec["action"], q.read_text()), ("rerun_left_as_is", "other\n"))

    @unittest.skipUnless((SCR / "rowchunk_base_model.py").exists(), "5e1bccd07c model.py not available")
    def test_real_diff_turns_5e1bccd_into_a6c3ee8(self):
        g = rowchunk_ns()
        p = self.tmpfile((SCR / "rowchunk_base_model.py").read_text())
        rec = g["rowchunk_patch"](p, False)
        self.assertEqual(rec["action"], "patched")
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), g["ROWCHUNK_FIXED"])
        self.assertEqual(p.read_text(), (SCR / "rowchunk_fixed_model.py").read_text())
        self.assertEqual(g["rowchunk_patch"](p, False)["action"], "already_fixed")



class GpuGate(unittest.TestCase):
    def run_gate(self, smi, rerun, env=None):
        d = Path(tempfile.mkdtemp(dir=C.SCRATCH))
        (d / "nvidia-smi").write_text("#!/bin/sh\n" + smi)
        (d / "nvidia-smi").chmod(0o755)
        g = {"T0": __import__("time").time(), "WORK": str(d), "RERUN": rerun, "MODE": "submit"}
        import os
        old = {k: os.environ.get(k) for k in ("PATH", "ARC_GPU_GATE_S", "ARC_GPU_GATE_PY")}
        os.environ.update({"PATH": f"{d}:" + os.environ["PATH"], "ARC_GPU_GATE_S": "1",
                           "ARC_GPU_GATE_PY": ".".join(map(str, sys.version_info[:2])), **(env or {})})
        try:
            exec((D / "cells/b32m_gpugate.py").read_text(), g)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return g, json.load(open(d / "b32m_gpu_gate.json"))

    def test_four_gpus_pass(self):
        g, rec = self.run_gate("for i in 0 1 2 3; do echo \"GPU $i: NVIDIA L4 (UUID: x)\"; done\n", False)
        self.assertEqual((rec["gpus"], rec["fail"], g["MODE"]), (4, [], "submit"))

    def test_no_gpu_raises_in_commit_and_falls_back_in_rerun(self):
        smi = "echo \"NVIDIA-SMI couldn't find libnvidia-ml.so library in your system.\"; exit 9\n"
        with self.assertRaises(RuntimeError):
            self.run_gate(smi, False)
        g, rec = self.run_gate(smi, True)
        self.assertEqual(g["MODE"], "fallback")
        self.assertEqual(rec["gpus"], 0)
        self.assertTrue(rec["tries"])

    def test_wrong_python_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_gate("for i in 0 1 2 3; do echo \"GPU $i: L4\"; done\n", False, {"ARC_GPU_GATE_PY": "3.99"})


if __name__ == "__main__":
    unittest.main()
