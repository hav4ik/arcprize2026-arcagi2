import copy
import json
from pathlib import Path
import unittest

from validate_training_data import validate_calibration_data, validate_rows, validate_training_diagnostic


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


class TrainingBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.allowed = set(json.loads((EXP/'configs/training_data_policy.json').read_text())['allowed_batches'])
        cls.train = set(json.loads((ROOT/'data/arc-agi_training_challenges.json').read_text()))
        cls.evaluation = set(json.loads((ROOT/'data/arc-agi_evaluation_challenges.json').read_text()))
        cls.rows = [json.loads(s) for s in (EXP/'data/screen/img/index.jsonl').read_text().splitlines()]
        cls.nvarc = [r for r in cls.rows if '_nvarc_' in r['batch']]
        cls.original = [r for r in cls.rows if r['batch'].endswith('_arc2_train')]

    def check_rows(self, rows):
        return validate_rows(rows, self.allowed, self.train, self.evaluation)

    def test_real_nvarc_training_rollouts_pass(self):
        self.assertGreater(self.check_rows(self.nvarc), 0)

    def test_original_training_rollouts_pass(self):
        self.assertGreater(self.check_rows(self.original), 0)

    def test_historical_mixed_pools_pass_with_disjoint_families(self):
        self.assertEqual(validate_calibration_data(EXP, ['screen', 'select', 'audit']),
                         {'screen_img': 16, 'screen_txt': 16,
                          'select_img': 24, 'select_txt': 24,
                          'audit_img': 32, 'audit_txt': 32})

    def test_original_training_diagnostic_passes(self):
        self.assertEqual(validate_training_diagnostic(EXP), 12)

    def test_evaluation_derivative_rejected_despite_allowed_batch(self):
        r = copy.deepcopy(self.nvarc[0])
        r['parent'] = sorted(self.evaluation)[0]
        r['task_id'] = r['parent'] + '_synthetic'
        with self.assertRaisesRegex(ValueError, 'Evaluation-derived'):
            self.check_rows([r])

    def test_disguised_source_and_full_subset_rejected(self):
        r = copy.deepcopy(self.nvarc[0])
        r['task_id'] = r['parent']
        with self.assertRaisesRegex(ValueError, 'NVARC child'):
            self.check_rows([r])
        r = copy.deepcopy(self.nvarc[0])
        r['batch'] = 'e065_arc2_train'
        with self.assertRaisesRegex(ValueError, 'original training task'):
            self.check_rows([r])
        r = copy.deepcopy(self.nvarc[0])
        r['batch'] = 'e065_nvarc_full'
        with self.assertRaisesRegex(ValueError, 'Non-allowlisted'):
            self.check_rows([r])

    def test_evaluation_original_rejected_despite_allowed_batch(self):
        r = copy.deepcopy(self.original[0])
        r['parent'] = r['task_id'] = sorted(self.evaluation)[0]
        with self.assertRaisesRegex(ValueError, 'Evaluation-derived'):
            self.check_rows([r])

    def test_empty_and_unknown_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Empty'):
            self.check_rows([])
        r = copy.deepcopy(self.nvarc[0])
        r['parent'], r['task_id'] = 'unknown', 'unknown_child'
        with self.assertRaisesRegex(ValueError, 'training registry'):
            self.check_rows([r])


if __name__ == '__main__':
    unittest.main()
