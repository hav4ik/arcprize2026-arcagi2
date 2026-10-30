"""run_notebook.py: the /kaggle mapping (symlinks only, never real data), working dir, commit/rerun environment,
hooks, dry run; and a live run of a tiny notebook when a Python with nbclient + ipykernel is available
(RELI_NB_PYTHON=<python>, else this interpreter if it has both)."""
import io
import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from kit_helpers import KIT, TmpCase, read
import run_notebook as rn

RUNNER = os.path.join(KIT, "run_notebook.py")


def nb_python():
    p = os.environ.get("RELI_NB_PYTHON")
    if p:
        return p
    try:
        import ipykernel  # noqa: F401
        import nbclient  # noqa: F401
        return sys.executable
    except ImportError:
        return None


def write_nb(path, sources):
    cells = []
    for s in sources:
        if s.startswith("# md "):
            cells.append({"cell_type": "markdown", "metadata": {}, "source": s[5:]})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                      "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w") as f:
        json.dump(nb, f)


class Plan(TmpCase):
    def setUp(self):
        super().setUp()
        self.k = self.path("kaggle")
        os.makedirs(self.k)
        self.model = self.path("models", "e128")
        os.makedirs(self.model)
        self.comp = self.path("comp")
        os.makedirs(self.comp)

    def test_parse_map(self):
        m = rn.parse_map(["%s=models/u/m/pytorch/default/1" % self.model, "%s=/competitions/c/" % self.comp])
        self.assertEqual(m, [("models/u/m/pytorch/default/1", self.model), ("competitions/c", self.comp)])
        for bad in (["%s=../x" % self.model], ["%s=" % self.model], ["/nonexistent=x"], ["nodest"],
                    ["%s=a" % self.model, "%s=a/b" % self.comp], ["%s=a" % self.model, "%s=a" % self.comp]):
            with self.assertRaises(rn.Refused, msg=bad):
                rn.parse_map(bad)

    def test_inputs_link_keep_relink_prune(self):
        m = rn.parse_map(["%s=models/u/m/1" % self.model])
        acts = rn.plan_inputs(self.k, m, prune=True)
        self.assertEqual([x[0] for x in acts], ["link"])
        rn.apply_inputs(acts)
        dest = os.path.join(self.k, "input", "models/u/m/1")
        self.assertEqual(os.readlink(dest), self.model)
        self.assertEqual([x[0] for x in rn.plan_inputs(self.k, m, True)], ["keep"])
        old = os.path.join(self.k, "input", "models/u/old/1")
        os.makedirs(os.path.dirname(old))
        os.symlink(self.comp, old)
        real = os.path.join(self.k, "input", "datasets", "real.txt")
        os.makedirs(os.path.dirname(real))
        with open(real, "w") as f:
            f.write("data")
        acts = rn.plan_inputs(self.k, rn.parse_map(["%s=models/u/m/1" % self.comp]), prune=True)
        self.assertEqual(sorted(x[0] for x in acts), ["prune", "relink"])
        rn.apply_inputs(acts)
        self.assertFalse(os.path.lexists(old))
        self.assertTrue(os.path.exists(real))  # real files are never pruned
        self.assertEqual(os.readlink(dest), self.comp)

    def test_never_replaces_real_data(self):
        os.makedirs(os.path.join(self.k, "input", "models", "u", "m", "1"))
        with self.assertRaises(rn.Refused):
            rn.plan_inputs(self.k, rn.parse_map(["%s=models/u/m/1" % self.model]), False)

    def test_working(self):
        w = self.path("kwork_r1")
        cwd, acts = rn.plan_working(self.k, w, None)
        rn.apply_working(acts)
        self.assertEqual(cwd, w)
        self.assertEqual(os.readlink(os.path.join(self.k, "working")), w)
        with open(os.path.join(w, "out.txt"), "w") as f:
            f.write("x")
        with self.assertRaises(rn.Refused):  # the same working dir again: not empty
            rn.plan_working(self.k, w, None)
        with self.assertRaises(rn.Refused):  # /kaggle/working -> a non-empty dir
            rn.plan_working(self.k, None, None)
        w2 = self.path("kwork_r2")
        cwd, acts = rn.plan_working(self.k, w2, None)
        rn.apply_working(acts)
        self.assertEqual(os.readlink(os.path.join(self.k, "working")), w2)
        self.assertTrue(os.path.exists(os.path.join(w, "out.txt")))  # the old run's files stay

    def test_working_real_dir_needs_archive(self):
        kw = os.path.join(self.k, "working")
        os.makedirs(kw)
        with open(os.path.join(kw, "old.txt"), "w") as f:
            f.write("x")
        with self.assertRaises(rn.Refused):
            rn.plan_working(self.k, None, None)
        arch = self.path("archive", "run1_working")
        cwd, acts = rn.plan_working(self.k, None, arch)
        rn.apply_working(acts)
        self.assertTrue(os.path.exists(os.path.join(arch, "old.txt")))
        self.assertEqual(os.listdir(kw), [])

    def test_env(self):
        base = {"HOME": "/root", "PATH": "/usr/bin", rn.RERUN_VAR: "1", "SECRET_THING": "x"}
        e = rn.build_env(base, "commit")
        self.assertNotIn(rn.RERUN_VAR, e)
        self.assertEqual(e["SECRET_THING"], "x")
        e = rn.build_env(base, "rerun", clean=True, venv="/root/kenv", extra=["ARC_SERVER_ENV_JSON={\"a\": 1}"])
        self.assertEqual(e[rn.RERUN_VAR], "1")
        self.assertNotIn("SECRET_THING", e)
        self.assertTrue(e["PATH"].startswith("/root/kenv/bin:/usr/local/cuda/bin:"))
        self.assertEqual(e["VIRTUAL_ENV"], "/root/kenv")
        self.assertEqual(e["ARC_SERVER_ENV_JSON"], '{"a": 1}')
        with self.assertRaises(rn.Refused):
            rn.build_env(base, "commit", extra=[rn.RERUN_VAR + "=1"])

    def test_hooks(self):
        self.assertEqual(rn.parse_hooks(["3:echo a", "3:echo b:c"]), {3: ["echo a", "echo b:c"]})
        for bad in (["x:echo"], ["3:"], ["echo"]):
            with self.assertRaises(rn.Refused):
                rn.parse_hooks(bad)

    def test_dry_run_changes_nothing(self):
        nb = self.path("nb.ipynb")
        write_nb(nb, ["# md title", "import os", "print(1)"])
        args = [nb, "--out", self.path("run", "nb"), "--mode", "rerun", "--kaggle-root", self.k,
                "--map", "%s=models/u/m/1" % self.model, "--working", self.path("w"), "--dry-run",
                "--on-cell-start", "2:echo hi"]
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(rn.main(args), 0)
        self.assertIn('"KAGGLE_IS_COMPETITION_RERUN": "1"', out.getvalue())
        self.assertIn("2 code     print(1)   <- hook", out.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.k, "input")))
        self.assertFalse(os.path.exists(self.path("w")))

    def test_refusals(self):
        nb = self.path("nb.ipynb")
        write_nb(nb, ["print(1)"])
        base = [nb, "--out", self.path("run", "nb"), "--mode", "commit", "--kaggle-root", self.k]
        with redirect_stderr(io.StringIO()) as err:
            with mock.patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}):
                self.assertEqual(rn.main(base + ["--dry-run"]), 2)
            self.assertEqual(rn.main(base + ["--on-cell-start", "5:echo", "--dry-run"]), 2)
        self.assertIn("Kaggle kernel", err.getvalue())
        self.assertIn("out of range", err.getvalue())


@unittest.skipUnless(nb_python(), "needs a Python with nbclient + ipykernel (set RELI_NB_PYTHON)")
class Live(TmpCase):
    def run_nb(self, sources, mode, extra=(), env=None):
        nb = self.path("nb.ipynb")
        write_nb(nb, sources)
        k = self.path("kaggle")
        src = self.path("comp")
        os.makedirs(src, exist_ok=True)
        with open(os.path.join(src, "arc-agi_test_challenges.json"), "w") as f:
            f.write("{}")
        out = self.path("run_%s" % mode, "nb")
        work = self.path("kwork_%s" % mode)
        cmd = [nb_python(), "-I", RUNNER, nb, "--out", out, "--mode", mode, "--kaggle-root", k,
               "--map", "%s=competitions/arc-prize-2026-arc-agi-2" % src, "--working", work] + list(extra)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env or dict(os.environ))
        return p, out, work, k

    def test_rerun_clean_env_and_hook(self):
        marker = self.path("hook_marker")
        p, out, work, k = self.run_nb(
            ["# md # notebook", "import os, glob\nprint('RERUN=', os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))\n"
             "print('CWD=', os.getcwd())\nprint('SECRET=', os.environ.get('RELI_TEST_SECRET'))",
             "print('TEST=', glob.glob('/nonexistent') or os.listdir(os.environ['KROOT'] + "
             "'/input/competitions/arc-prize-2026-arc-agi-2'))\nopen('marker.txt', 'w').write('x')"],
            "rerun", ["--clean-env", "--env", "KROOT=" + self.path("kaggle"), "--on-cell-start",
                      "2:echo hooked > %s" % marker],
            env=dict(os.environ, RELI_TEST_SECRET="leak"))
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        tl = read(out + ".timeline")
        self.assertIn("NOTEBOOK_DONE ok", tl)
        self.assertIn("START cell 1", tl)
        self.assertNotIn("START cell 0", tl)  # markdown
        self.assertIn("HOOK cell 2", tl)
        log = read(out + ".log")
        self.assertIn("RERUN= 1", log)
        self.assertIn("CWD= %s" % work, log)
        self.assertIn("SECRET= None", log)  # --clean-env drops the runner's variables
        self.assertIn("TEST= ['arc-agi_test_challenges.json']", log)
        self.assertTrue(os.path.exists(os.path.join(work, "marker.txt")))
        self.assertEqual(read(marker).strip(), "hooked")
        man = json.loads(read(out + ".manifest.json"))
        self.assertEqual(man["mode"], "rerun")
        nb = json.loads(read(out + ".ipynb"))
        self.assertIn("RERUN= 1", "".join(nb["cells"][1]["outputs"][0]["text"]))

    def test_commit_drops_an_inherited_rerun_flag_and_stops_at_errors(self):
        p, out, work, k = self.run_nb(
            ["import os\nprint('RERUN=', os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))", "raise ValueError('boom')",
             "print('never')"], "commit", env=dict(os.environ, KAGGLE_IS_COMPETITION_RERUN="1"))
        self.assertEqual(p.returncode, 1)
        tl = read(out + ".timeline")
        self.assertIn("NOTEBOOK_DONE error", tl)
        self.assertIn("boom", tl)
        self.assertNotIn("START cell 2", tl)
        log = read(out + ".log")
        self.assertIn("RERUN= None", log)
        self.assertIn("ERROR ValueError: boom", log)
        nb = json.loads(read(out + ".ipynb"))  # the partial notebook is saved
        self.assertEqual(nb["cells"][1]["outputs"][0]["ename"], "ValueError")


if __name__ == "__main__":
    unittest.main()
