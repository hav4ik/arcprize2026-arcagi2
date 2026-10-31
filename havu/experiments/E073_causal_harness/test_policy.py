import copy
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import build
import analyze

HERE = Path(__file__).resolve().parent


class Policy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build.main()
        with patch.dict(os.environ, {'ARC_ATTEMPT_FORMATS': 'json,json,text+image'}, clear=True):
            sp = importlib.util.spec_from_file_location('e073_test_solver', HERE/'frozen/solver_causal.py')
            cls.solver = importlib.util.module_from_spec(sp); sp.loader.exec_module(cls.solver)

    def test_format_is_per_attempt_without_shared_mutation(self):
        args = SimpleNamespace(grid_format='text+image', token_exact=True, image_cell=8, x=9)
        before = vars(args).copy()
        self.assertEqual([self.solver.attempt_args(args, a).grid_format for a in range(3)],
                         ['json', 'json', 'text+image'])
        self.assertEqual(vars(args), before)
        self.assertTrue(all(self.solver.attempt_args(args, a).token_exact for a in range(3)))

    def test_extra_waits_for_both_and_uses_no_answers_from_key(self):
        r0 = {'attempt': 0, 'train_pass': False, 'pred': [[1]]}
        r1 = {'attempt': 1, 'train_pass': None, 'pred': None}
        self.assertFalse(self.solver.needs_extra([r0]))
        self.assertTrue(self.solver.needs_extra([r0, r1]))
        self.assertFalse(self.solver.needs_extra([r0, dict(r1, train_pass=True, pred=[[2]])]))
        self.assertFalse(self.solver.needs_extra([r0, dict(r1, provisional=True)]))

    def test_gold_keeps_original_harness(self):
        gold = json.loads((HERE/'frozen/gold.ipynb').read_text())
        meta = json.loads((build.D/'out/manh_v12_363638159/kernel-metadata.json').read_text())
        old = json.loads((build.D/'out/manh_v12_363638159'/meta['code_file']).read_text())
        a, b = build.overlay(gold), build.overlay(old)
        self.assertEqual(a['repo/evals/run_deepseek_tools.py'], b['repo/evals/run_deepseek_tools.py'])
        self.assertEqual(a['solver_dsv41.py'].decode(), b['solver_dsv41.py'].decode().replace(
            'Path("/kaggle/input").rglob("e043_manifest.json")',
            'Path("/kaggle/input").glob("models/*/*/*/*/*/e043_manifest.json")'))
        self.assertEqual(build.src(build.cell(gold, '# Cell 2m')), build.src(build.cell(old, '# Cell 2m')))

    def test_candidate_pins_and_serving(self):
        nb = json.loads((HERE/'frozen/causal.ipynb').read_text())
        files = build.overlay(nb)
        self.assertEqual(files['solver_dsv41.py'], (HERE/'frozen/solver_causal.py').read_bytes())
        te = json.loads((build.D/'out/nb_v12_img_te/dsv41-tools-v12-img.ipynb').read_text())
        self.assertEqual(files['repo/evals/run_deepseek_tools.py'], build.overlay(te)['repo/evals/run_deepseek_tools.py'])
        s = build.src(build.cell(nb, '# Cell 2m'))
        self.assertNotIn('watermark-reload-only', s)
        self.assertNotIn('decoder_replay_cudagraph_capture_sizes', s)
        self.assertIn('--disable-chunked-mm-input', s)
        self.assertFalse(build.IP.notebook_scans(nb))

    def test_three_draws_never_become_three_guesses(self):
        rs = [dict(task_id='x', pair=0, attempt=a, pred=[[a]], train_pass=False) for a in range(3)]
        sol = {'x': [[[2]]]}
        self.assertEqual(analyze.values(analyze.records_for(rs, 'retry'), sol)['x'], 0)
        self.assertEqual(analyze.values(analyze.records_for(rs, 'TI'), sol)['x'], 1)
        rs[2]['train_pass'] = True
        self.assertEqual(analyze.values(analyze.records_for(rs, 'retry'), sol)['x'], 1)

    def test_train_pass_suppresses_retry_without_test_key(self):
        rs = [dict(task_id='x', pair=0, attempt=a, pred=[[a]], train_pass=(a==0)) for a in range(3)]
        self.assertEqual([r['attempt'] for r in analyze.records_for(rs, 'retry')], [0, 1])

    def test_multitest_branch_preserves_pairs(self):
        rs = [dict(task_id='x', pair=0, attempt=a, pred=[[[a]], [[a+1]]], multi_test=2,
                   train_pass=False) for a in range(3)]
        sol = {'x': [[[2]], [[3]]]}
        self.assertEqual(analyze.values(analyze.records_for(rs, 'TI'), sol)['x'], 1)

    def test_real_overlay_checks_and_no_silent_fallback(self):
        nb = json.loads((HERE/'frozen/causal.ipynb').read_text())
        for mismatch in (False, True):
            with tempfile.TemporaryDirectory(prefix='e073_pins_') as td, patch.dict(os.environ, {}, clear=True):
                with tarfile.open(build.D/'kaggle_kit_v5/dsv41_kit.tgz') as tf:
                    tf.extractall(td, filter='data')
                ns = {}
                exec(build.src(nb['cells'][1]), ns)
                ns.update(KIT=td, MODE='submit', RERUN=True)
                if mismatch:
                    os.environ['ARC_WORKERS'] = '63'
                    with self.assertRaisesRegex(RuntimeError, 'CHECK FAILED'):
                        exec(build.src(build.cell(nb, '# Cell 0b')), ns)
                else:
                    exec(build.src(build.cell(nb, '# Cell 0b')), ns)
                self.assertEqual((Path(td)/'solver_dsv41.py').read_bytes(), build.overlay(nb)['solver_dsv41.py'])


if __name__ == '__main__':
    unittest.main()
