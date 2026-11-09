"""The settings cell (cell 00) and cell 0b of dsv41-tools-submit-mtf Version 6, executed against a copy of the Kaggle
kit (v3, 62144e4a / 1ad5ee24) in the four commit / rerun x pass / fail cases (no Kaggle, no GPU)."""
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import common as C

NB = Path("/home/vu/Workspace/ARC-AGI-2/kaggle/dsv41_tools/out/nb_dsv41-tools-submit-mtf/dsv41-tools-submit-mtf.ipynb")
CELLS = json.load(open(NB))["cells"]
SETTINGS, CELL0B = "".join(CELLS[0]["source"]), "".join(CELLS[2]["source"])
V3_SHA = {"solver_dsv41.py": "62144e4a", "repo/evals/run_deepseek_tools.py": "1ad5ee24"}
PIN_SHA = {"solver_dsv41.py": "4b344011", "repo/evals/run_deepseek_tools.py": "a8a5b665"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


def run_cells(rerun, extra_env=None, mode="submit", break_kit=None):
    kit = Path(tempfile.mkdtemp(prefix="kit_")) / "kit"
    shutil.copytree(C.V3, kit)
    if break_kit:
        break_kit(kit)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "KAGGLE_"))}
    if rerun:
        env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    env.update(extra_env or {})
    ns = {}
    out = io.StringIO()
    err = None
    with patch.dict(os.environ, env, clear=True), redirect_stdout(out):
        exec(compile(SETTINGS, "cell00", "exec"), ns)
        ns.update(KIT=str(kit), MODE=mode, RERUN=rerun)
        try:
            exec(compile(CELL0B, "cell0b", "exec"), ns)
        except RuntimeError as exc:
            err = exc
        final_env = {k: v for k, v in os.environ.items() if k.startswith("ARC_")}
    return kit, out.getvalue(), err, final_env


class Cells(unittest.TestCase):
    def test_settings_match_expectations(self):
        self.assertIn('_MT_EXPECT = {"ARC_WORKERS": "36", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", '
                      '"ARC_CONTEXT_LIMIT": "163840", "ARC_KV_ADMIT_MAX": "0.6"}', CELL0B)
        self.assertIn('"solver_dsv41.py": "4b344011', CELL0B)
        self.assertIn('"repo/evals/run_deepseek_tools.py": "a8a5b665', CELL0B)

    def test_rerun_ok(self):
        kit, out, err, env = run_cells(rerun=True)
        self.assertIsNone(err)
        self.assertIn("E046 CHECK OK", out)
        self.assertEqual({f: sha(kit / f) for f in PIN_SHA}, PIN_SHA)
        self.assertEqual({f: sha(str(kit / f) + ".v3orig") for f in V3_SHA}, V3_SHA)
        self.assertNotIn("ARC_TASK_IDS", env)
        self.assertNotIn("ARC_COMMIT_MODE", env)
        self.assertEqual((env["ARC_WORKERS"], env["ARC_MULTI_TEST"], env["ARC_KV_ADMIT_MIN_WORKERS"]), ("36", "1", "16"))
        # the installed solver imports and builds its harness argv with these settings
        with patch.dict(os.environ, env):
            s = C.load_solver(env, name="nb_rerun_solver")
        a = s.make_args()
        self.assertEqual((a.multi_test, a.multi_budget, a.context_limit), (True, 0.5, 163840))

    def test_commit_ok_smoke_settings(self):
        kit, out, err, env = run_cells(rerun=False, mode="smoke")
        self.assertIsNone(err)
        self.assertIn("E046 CHECK OK", out)
        self.assertEqual(env["ARC_TASK_IDS"], "25ff71a9,27a28665,00576224")
        self.assertEqual(env["ARC_COMMIT_MODE"], "smoke")

    def test_commit_wrong_setting_raises(self):
        kit, out, err, env = run_cells(rerun=False, extra_env={"ARC_WORKERS": "44"})
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("E046 CHECK FAILED", out)
        self.assertIn("ARC_WORKERS='44'", out)

    def test_rerun_wrong_setting_falls_back_to_v3(self):
        kit, out, err, env = run_cells(rerun=True, extra_env={"ARC_MULTI_BUDGET": "0.25"})
        self.assertIsNone(err)
        self.assertIn("E046 FALLBACK", out)
        self.assertEqual({f: sha(kit / f) for f in V3_SHA}, V3_SHA)
        for k in ("ARC_MULTI_TEST", "ARC_MULTI_BUDGET", "ARC_KV_ADMIT_MAX", "ARC_CONTEXT_LIMIT"):
            self.assertNotIn(k, env)
        self.assertEqual(env["ARC_WORKERS"], "36")

    def test_rerun_broken_kit(self):
        # FINDING (minor): `_bak[_f]` is recorded before shutil.copy2 makes the backup, and the rerun's restore loop is
        # outside the try. If a backup copy fails (here: the kit file is missing; on Kaggle cell 0's MANIFEST check
        # makes this unreachable, a disk error would not be), the restore raises and the cell errors in the rerun.
        def drop(kit):
            (kit / "repo/evals/run_deepseek_tools.py").unlink()
        with self.assertRaises(FileNotFoundError):
            run_cells(rerun=True, break_kit=drop)

    def test_fallback_mode_skips(self):
        kit, out, err, env = run_cells(rerun=True, mode="fallback")
        self.assertIsNone(err)
        self.assertEqual(out.count("E046"), 1)  # only the settings print
        self.assertEqual({f: sha(kit / f) for f in V3_SHA}, V3_SHA)


if __name__ == "__main__":
    unittest.main()
