"""E056: the candidate-B overlay pair installs through the notebook's own cells (scripts/notebook_install.py) into a
fake site-packages, CPU only. Builds a scratch notebook with kaggle/dsv41_tools/build_frozen_mt.py --overlay kit/ and
the candidate-B --server-extra (the builder is run, not edited). Cases:
  commit run, settings ok   -> cell 0b OK, all 16 members installed and sha-checked (4 of them new to the kit,
                               replacing stand-ins for the wheel's originals), offload flag kept, alloc conf stripped
  commit run, bad setting   -> cell 0b raises (the commit errors), nothing is installed
  rerun, bad setting        -> fallback: the kit's own 12-file overlay is installed, the 4 wheel files are untouched,
                               ARC_SERVER_EXTRA_ARGS loses --kv-transfer-config

    .venv/bin/python -m unittest discover -s experiments/E056_candidate_b/tests
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

E056 = Path(__file__).resolve().parents[1]
ROOT = E056.parents[1]
KVT = json.dumps({"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {
    "cpu_bytes_to_use": 34359738368, "offload_prompt_only": False, "swa_store_policy": "checkpoint",
    "shared_region": False, "eviction_policy": "lru"}}, separators=(",", ":"))
E051_FILES = ["vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
              "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
              "vllm/v1/kv_offload/cpu/manager.py", "vllm/v1/kv_offload/cpu/spec.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class NotebookInstall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e056_nb_"))
        subprocess.run([sys.executable, str(ROOT / "kaggle/dsv41_tools/build_frozen_mt.py"), "--slug", "e056-test",
                        "--workers", "36", "--overlay", str(E056 / "kit"), "--retention", "512",
                        "--server-extra", json.dumps(["--kv-transfer-config", KVT]), "--out", str(cls.tmp / "nb")],
                       check=True, capture_output=True)
        cls.nb = cls.tmp / "nb/e056-test.ipynb"
        cls.inp = cls.tmp / "input"
        cls.inp.mkdir()
        shutil.copy2(ROOT / "kaggle/dsv41_tools/kaggle_kit_v5/dsv41_kit.tgz", cls.inp / "dsv41_kit.tgz")
        (cls.inp / "arc-agi_test_challenges.json").write_text(json.dumps(
            {"00000000": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]]}]}}))

    def run_nb(self, name, rerun=False, env=()):
        sp = self.tmp / name / "site-packages"
        for rel in E051_FILES:  # stand-ins for the fork wheel's originals
            (sp / rel).parent.mkdir(parents=True, exist_ok=True)
            (sp / rel).write_text(f"# wheel original {rel}\n")
        cmd = [sys.executable, "-I", str(E056 / "scripts/notebook_install.py"), "--notebook", str(self.nb),
               "--input", str(self.inp), "--scratch", str(self.tmp / name / "scratch"), "--site-packages", str(sp)]
        cmd += ["--rerun"] if rerun else []
        for kv in env:
            cmd += ["--env", kv]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("E056 INSTALL SUMMARY ")]
        self.assertTrue(line, r.stdout + r.stderr)
        return r.returncode, json.loads(line[-1].split(" ", 3)[3]), r.stdout, sp

    def test_commit_ok(self):
        rc, s, out, sp = self.run_nb("ok")
        self.assertEqual(rc, 0, out)
        self.assertTrue(s["cell0b_ok"], out)
        self.assertFalse(s["fallback"])
        self.assertEqual(s["kit_overlay_tgz"], sha(E056 / "kit/vllm_overlay_rw.tgz"))
        self.assertEqual(s["files"], 16)
        self.assertEqual(s["files_bad"], [])
        self.assertTrue(s["server_extra_has_kv_transfer"])
        self.assertEqual(s["alloc_conf"], "")
        rows = dict(reversed(ln.split()) for ln in (E056 / "kit/vllm_overlay_rw.sha256").read_text().splitlines())
        for rel in E051_FILES + ["vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py"]:
            self.assertEqual(sha(sp / rel), rows[rel], rel)
        self.assertTrue(rows["vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py"].startswith("9f13485b"))

    def test_commit_bad_setting_raises(self):
        rc, s, out, sp = self.run_nb("bad_commit", env=["ARC_WORKERS=7"])
        self.assertNotEqual(rc, 0)
        self.assertIn("E046 CHECK FAILED (commit run)", s.get("error", ""))
        for rel in E051_FILES:
            self.assertTrue((sp / rel).read_text().startswith("# wheel original"))

    def test_rerun_bad_setting_falls_back_to_kit_overlay(self):
        rc, s, out, sp = self.run_nb("bad_rerun", rerun=True, env=["ARC_WORKERS=7"])
        self.assertEqual(rc, 0, out)
        self.assertTrue(s["fallback"], out)
        self.assertEqual(s["kit_overlay_tgz"], "377d83e0942ef32133545beb8ff7d945039ea5fd061954546b4b046be84fea6e")
        self.assertEqual(s["files"], 12)
        self.assertEqual(s["files_bad"], [])
        self.assertFalse(s["server_extra_has_kv_transfer"])
        for rel in E051_FILES:
            self.assertTrue((sp / rel).read_text().startswith("# wheel original"))
        self.assertTrue(sha(sp / "vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py").startswith("3da076e1"))


if __name__ == "__main__":
    unittest.main()
