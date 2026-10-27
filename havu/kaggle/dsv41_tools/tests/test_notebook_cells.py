"""The settings cell (cell 00) and cell 0b of the next frozen-input candidate, built by ../build_frozen_mt.py from the
live solver and harness, executed against a copy of the Kaggle kit of Version 5 (v3 code: solver 62144e4a, harness
1ad5ee24) in the commit / rerun x pass / fail cases (no Kaggle, no GPU). Ported from the 2026-10-26 deep audit
(../tests_mtf_v6/test_notebook_cells.py, which read the built V6 notebook), plus the S10 regression tests."""
import argparse
import hashlib
import importlib.util
import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import common as C

_spec = importlib.util.spec_from_file_location("build_frozen_mt", C.DSV41 / "build_frozen_mt.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
A = argparse.Namespace(workers=36, budget=0.5, gate=0.6)
EXPECT = {"ARC_WORKERS": "36", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_CONTEXT_LIMIT": "163840",
          "ARC_KV_ADMIT_MAX": "0.6"}
FILES = {"solver_dsv41.py": C.SOLVER, "repo/evals/run_deepseek_tools.py": C.HARNESS}
SETTINGS = B.settings_cell(A)
CELL0B, _ = B.code_cell(FILES, EXPECT)
V3_SHA = {"solver_dsv41.py": "62144e4a", "repo/evals/run_deepseek_tools.py": "1ad5ee24"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:8]


PIN_SHA = {f: sha(src) for f, src in FILES.items()}


def run_cells(rerun, extra_env=None, mode="submit", break_kit=None, cell0b=CELL0B):
    kit = Path(tempfile.mkdtemp(prefix="kit_", dir=C.SCRATCH)) / "kit"
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
            exec(compile(cell0b, "cell0b", "exec"), ns)
        except RuntimeError as exc:
            err = exc
        final_env = {k: v for k, v in os.environ.items() if k.startswith("ARC_")}
    return kit, out.getvalue(), err, final_env


class Cells(unittest.TestCase):
    def test_settings_match_expectations(self):
        self.assertIn('_MT_EXPECT = {"ARC_WORKERS": "36", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", '
                      '"ARC_CONTEXT_LIMIT": "163840", "ARC_KV_ADMIT_MAX": "0.6"}', CELL0B)
        for f, h in PIN_SHA.items():
            self.assertIn(f'"{f}": "{h}', CELL0B)

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
        s = C.load_solver(env, name="nb_rerun_solver", path=kit / "solver_dsv41.py")
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

    def test_commit_old_solver_lacks_fixes(self):
        # the feature check rejects an overlay built from the V6 solver (no restart_allowed / shutdown_socket)
        cell, _ = B.code_cell({**FILES, "solver_dsv41.py": C.DSV41 / "pinned_mtf_v6/solver_dsv41.py"}, EXPECT)
        kit, out, err, env = run_cells(rerun=False, cell0b=cell)
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("solver lacks restart_allowed", out)

    def test_rerun_wrong_setting_falls_back_to_v3(self):
        kit, out, err, env = run_cells(rerun=True, extra_env={"ARC_MULTI_BUDGET": "0.25"})
        self.assertIsNone(err)
        self.assertIn("E046 FALLBACK: the kit's v3 code, per pair, 36 workers\n", out)
        self.assertEqual({f: sha(kit / f) for f in V3_SHA}, V3_SHA)
        for k in ("ARC_MULTI_TEST", "ARC_MULTI_BUDGET", "ARC_KV_ADMIT_MAX", "ARC_CONTEXT_LIMIT"):
            self.assertNotIn(k, env)
        self.assertEqual(env["ARC_WORKERS"], "36")

    def test_s10_rerun_broken_kit_still_runs(self):
        # S10 (fails on V6: `_bak[_f]` was recorded before the backup existed and the restore loop was outside the
        # try, so the rerun raised FileNotFoundError and the notebook failed)
        def drop(kit):
            (kit / "repo/evals/run_deepseek_tools.py").unlink()
        kit, out, err, env = run_cells(rerun=True, break_kit=drop)
        self.assertIsNone(err)
        self.assertIn("E046 CHECK FAILED: FileNotFoundError", out)
        self.assertIn("E046 FALLBACK: the kit's v3 code, per pair, 36 workers; not restored, kept as is: "
                      "repo/evals/run_deepseek_tools.py (KeyError)", out)
        self.assertEqual(sha(kit / "solver_dsv41.py"), V3_SHA["solver_dsv41.py"])
        self.assertEqual(env["ARC_WORKERS"], "36")
        self.assertNotIn("ARC_MULTI_TEST", env)

    def test_s10_restore_failure_keeps_pinned_file(self):
        # S10 (fails on V6: the restore raised IsADirectoryError): a backup that cannot be copied back leaves the
        # pinned file in place; the other file is restored and the settings fall back
        def block(kit):
            (kit / "solver_dsv41.py.v3orig").mkdir()  # copy2 puts the backup inside; copying it back fails
        opt_in = {"ARC_AUTO_REPAIR": "1", "ARC_BEST_GUESS_AT": "0.8", "ARC_BEST_GUESS_TURNS": "3",
                  "ARC_POST_PASS_NOTE": "1"}
        kit, out, err, env = run_cells(rerun=True, extra_env={"ARC_MULTI_BUDGET": "0.25", **opt_in}, break_kit=block)
        self.assertIsNone(err)
        self.assertIn("not restored, kept as is: solver_dsv41.py (IsADirectoryError)", out)
        self.assertEqual(sha(kit / "solver_dsv41.py"), PIN_SHA["solver_dsv41.py"])
        self.assertEqual(sha(kit / "repo/evals/run_deepseek_tools.py"), V3_SHA["repo/evals/run_deepseek_tools.py"])
        self.assertEqual(env["ARC_WORKERS"], "36")
        # the pinned solver still runs per pair with the v3 harness, without the opt-in harness flags (I5, I7, I8),
        # which the v3 harness's parser would reject with SystemExit
        self.assertFalse(set(opt_in) & set(env))
        s = C.load_solver(env, name="nb_kept_solver", path=kit / "solver_dsv41.py")
        self.assertFalse(s.MULTI_TEST)
        self.assertEqual((s.AUTO_REPAIR, s.BEST_GUESS_AT, s.BEST_GUESS_TURNS, s.POST_PASS_NOTE), (False, "", "", False))
        self.assertNotIn("--post-pass-note", s.HARNESS)

    def test_fallback_mode_skips(self):
        kit, out, err, env = run_cells(rerun=True, mode="fallback")
        self.assertIsNone(err)
        self.assertEqual(out.count("E046"), 1)  # only the settings print
        self.assertEqual({f: sha(kit / f) for f in V3_SHA}, V3_SHA)


if __name__ == "__main__":
    unittest.main()


A_NEXT = argparse.Namespace(workers=44, budget=0.5, gate=0.0, retention=512, effort=25, auto_repair=True,
                            best_guess_at=0.8, best_guess_turns=3, post_pass_note=True)
NEXT_FILES = dict(FILES, **{f"repo/evals/{n}": C.ROOT / "evals" / n for n in ("auto_repair.py", "auto_repair_names.json")})
NEXT_NEW = ["repo/evals/auto_repair.py", "repo/evals/auto_repair_names.json"]
NEXT_EXPECT = {"ARC_WORKERS": "44", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_CONTEXT_LIMIT": "163840",
               **dict(B.opt_env(A_NEXT))}
NEXT_NEED = {"repo/evals/run_deepseek_tools.py": ["--auto-repair", "--best-guess-at", "--post-pass-note"],
             "solver_dsv41.py": ["ARC_POST_PASS_NOTE", "ARC_REASONING_EFFORT"]}


class NextCandidate(unittest.TestCase):
    """Files new to the kit (auto-repair) and the opt-in levers: settings cell, checks, rerun fallback."""

    def run_next(self, rerun, **kw):
        cell0b, _ = B.code_cell(NEXT_FILES, NEXT_EXPECT, NEXT_NEW, NEXT_NEED)
        with patch.object(B, "settings_cell", B.settings_cell):
            global SETTINGS
            saved, SETTINGS = SETTINGS, B.settings_cell(A_NEXT)
            try:
                return run_cells(rerun, cell0b=cell0b, **kw)
            finally:
                SETTINGS = saved

    def test_settings_cell(self):
        s = B.settings_cell(A_NEXT)
        for k, v in NEXT_EXPECT.items():
            self.assertIn(f'("{k}", "{v}")', s)
        self.assertEqual(NEXT_EXPECT["ARC_POST_PASS_NOTE"], "1")
        self.assertIn('"ARC_POST_PASS_NOTE",', s.split("print(")[-1])  # in the settings printout too
        self.assertIn('"--prefix-cache-retention-interval", "512"', s)
        self.assertNotIn("ARC_KV_ADMIT_MAX", s.split("import json, os")[1].split("ARC_SERVER_EXTRA_ARGS")[0])

    def test_commit_ok_installs_new_files(self):
        kit, out, err, env = self.run_next(rerun=False)
        self.assertIsNone(err, out)
        self.assertIn("E046 CHECK OK", out)
        for f in NEXT_NEW:
            self.assertTrue((kit / f).exists(), f)
        self.assertEqual(env.get("ARC_REASONING_EFFORT"), "25")
        self.assertIn("512", env.get("ARC_SERVER_EXTRA_ARGS", ""))
        # I8: the installed solver passes --post-pass-note to the installed harness
        a = C.load_solver(env, name="nb_next_solver", path=kit / "solver_dsv41.py").make_args()
        self.assertTrue(a.post_pass_note)

    def test_opt_need(self):
        # the builder's feature check for the opt-in levers (I8: the harness flag and the solver's env variable)
        self.assertEqual(B.opt_need(A_NEXT), NEXT_NEED)
        self.assertEqual(B.opt_need(A), {})
        self.assertEqual(B.opt_need(argparse.Namespace(post_pass_note=True)),
                         {"repo/evals/run_deepseek_tools.py": ["--post-pass-note"],
                          "solver_dsv41.py": ["ARC_POST_PASS_NOTE"]})
        self.assertEqual(B.opt_env(argparse.Namespace(post_pass_note=False)), [])

    def test_commit_harness_without_post_pass_raises(self):
        # a pinned harness without --post-pass-note fails cell 0b's feature check in the commit run
        old = Path(tempfile.mkdtemp(prefix="old_harness_", dir=C.SCRATCH)) / "run_deepseek_tools.py"
        old.write_text(C.HARNESS.read_text().replace("--post-pass-note", "--post-pass-xxxx"))
        cell0b, _ = B.code_cell({**NEXT_FILES, "repo/evals/run_deepseek_tools.py": old}, NEXT_EXPECT, NEXT_NEW,
                                NEXT_NEED)
        global SETTINGS
        saved, SETTINGS = SETTINGS, B.settings_cell(A_NEXT)
        try:
            kit, out, err, env = run_cells(False, cell0b=cell0b)
        finally:
            SETTINGS = saved
        self.assertIsInstance(err, RuntimeError)
        self.assertIn("repo/evals/run_deepseek_tools.py lacks --post-pass-note", out)

    def test_rerun_fallback_restores_v3_and_drops_levers(self):
        # a wrong setting fails the check; the rerun restores the v3 solver/harness and drops every opt-in lever
        kit, out, err, env = self.run_next(rerun=True, extra_env={"ARC_WORKERS": "7"})
        self.assertIsNone(err)
        self.assertIn("E046 FALLBACK", out)
        self.assertEqual({f: sha(kit / f) for f in V3_SHA}, V3_SHA)
        for k in ("ARC_REASONING_EFFORT", "ARC_AUTO_REPAIR", "ARC_BEST_GUESS_AT", "ARC_BEST_GUESS_TURNS",
                  "ARC_POST_PASS_NOTE", "ARC_MULTI_TEST"):
            self.assertNotIn(k, env)
        self.assertEqual(env["ARC_WORKERS"], "36")


class OverlayAndServerExtra(unittest.TestCase):
    """--overlay ships a replacement kit overlay pair; --server-extra adds server flags; the rerun fallback restores the
    kit's overlay and Version 5's server flags (so the stock connector never sees --kv-transfer-config)."""
    E054 = C.ROOT / "experiments/E054_exact_cherrypicks/kit"

    def test_server_extra_in_settings(self):
        a = argparse.Namespace(**vars(A_NEXT), server_extra='["--kv-transfer-config", "{}"]')
        self.assertIn('"--kv-transfer-config", "{}"', B.settings_cell(a))

    def test_overlay_commit_and_rerun_fallback(self):
        if not (self.E054 / "vllm_overlay_rw.tgz").exists():
            self.skipTest("E054 overlay not built")
        files = dict(FILES, **{n: self.E054 / n for n in ("vllm_overlay_rw.tgz", "vllm_overlay_rw.sha256")})
        cell0b, _ = B.code_cell(files, EXPECT)
        kit, out, err, env = run_cells(rerun=False, cell0b=cell0b)
        self.assertIsNone(err, out)
        self.assertEqual(sha(kit / "vllm_overlay_rw.tgz"), sha(self.E054 / "vllm_overlay_rw.tgz"))
        orig = sha(C.V3 / "vllm_overlay_rw.tgz")
        kit, out, err, env = run_cells(rerun=True, cell0b=cell0b, extra_env={"ARC_WORKERS": "7",
                                       "ARC_SERVER_EXTRA_ARGS": '["--kv-transfer-config", "{}"]'})
        self.assertIn("E046 FALLBACK", out)
        self.assertEqual(sha(kit / "vllm_overlay_rw.tgz"), orig)
        self.assertNotIn("kv-transfer-config", env["ARC_SERVER_EXTRA_ARGS"])
        self.assertEqual(env["ARC_SERVER_EXTRA_ARGS"], B.V5_SERVER_EXTRA)


class AllocConf(unittest.TestCase):
    def test_offload_candidate_strips_expandable_segments(self):
        a = argparse.Namespace(**vars(A_NEXT), server_extra='["--kv-transfer-config", "{}"]')
        ns = {}
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512"}, clear=True):
            exec(compile(B.settings_cell(a), "cell00", "exec"), ns)
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")

    def test_other_candidates_leave_it(self):
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", B.settings_cell(A_NEXT))
