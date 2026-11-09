"""T1 / T6 of ../next_candidate_fixes.md: experiments/E042_dsv41_l4_synthetic/scripts/bench_accuracy.py (pairs matched by
test input, reference by mode, no-answers wrong on both sides, cut attempts apart) and deadline_sim.py (the clean E112
pool). Synthetic data, plus the real result files when they are present."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import common as C

sys.path.insert(0, str(C.ROOT / "experiments/E042_dsv41_l4_synthetic/scripts"))
import bench_accuracy as BA  # noqa: E402
import deadline_sim as DS  # noqa: E402

A_IN, A_OUT, B_IN, B_OUT = [[1]], [[2]], [[3]], [[4]]


class BenchAccuracy(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="bench_", dir=C.SCRATCH))
        gh = self.d / "github"
        gh.mkdir()
        # the reference set (GitHub eval dir, outputs inline): t1 with pairs A, B; t3 multi-test
        (gh / "t1.json").write_text(json.dumps({"train": [], "test": [{"input": A_IN, "output": A_OUT},
                                                                     {"input": B_IN, "output": B_OUT}]}))
        recs = [{"task_id": "t1", "pair": 0, "attempt": a, "correct": a == 0, "config": {"challenges": str(gh)}}
                for a in (0, 1)]  # pair A: 1 of 2 correct
        recs += [{"task_id": "t1", "pair": 1, "attempt": a, "correct": True, "config": {"challenges": str(gh)}}
                 for a in (0, 1)]  # pair B: always correct
        recs.append({"task_id": "t1", "pair": 1, "attempt": 2, "error": "x", "config": {"challenges": str(gh)}})
        (self.d / "ref_pair.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        # the bench's set (Kaggle json): t1's pairs in the other order, and t2 the reference never ran
        self.chal = {"t1": {"train": [], "test": [{"input": B_IN}, {"input": A_IN}]},
                     "t2": {"train": [], "test": [{"input": [[5]]}]}}
        self.sol = {"t1": [B_OUT, A_OUT], "t2": [[[6]]]}

    def bench(self, records):
        p = self.d / f"att_{len(list(self.d.glob('att_*')))}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        return BA.load_bench(p, BA.load_set(self.write_chal(), self.sol), self.sol)

    def write_chal(self):
        p = self.d / "challenges.json"
        p.write_text(json.dumps(self.chal))
        return p

    def test_t1_matching_modes_noanswers_and_cuts(self):
        ref = BA.load_reference(["pair"], prefix=str(self.d / "ref_"))
        keys = BA.load_set(self.write_chal(), self.sol)
        self.assertEqual(sorted(len(v) for v in ref.values()), [2, 2])  # the errored reference record is left out
        self.assertEqual(ref[keys[("t1", 0)]], [True, True])  # bench pair 0 is the reference's pair 1 (B)
        bench = self.bench([
            {"task_id": "t1", "pair": 0, "attempt": 0, "pred": B_OUT, "cuts": {}},  # correct
            {"task_id": "t1", "pair": 1, "attempt": 0, "pred": None, "cuts": {}},  # no answer: wrong
            {"task_id": "t1", "pair": 1, "attempt": 1, "pred": A_OUT, "cuts": {"deadline": 1}},  # cut: apart
            {"task_id": "t1", "pair": 0, "attempt": 1, "pred": None, "error": "TurnError: x"},  # error: apart
            {"task_id": "t2", "pair": 0, "attempt": 0, "pred": [[6]], "cuts": {}}])  # no reference
        s = BA.compare(bench, ref)
        self.assertEqual((s["attempts"], s["pairs"], s["correct"], s["cut"], s["cut_correct"], s["errors"],
                          s["no_ref"]), (2, 2, 1, 1, 1, 1, 1))
        self.assertAlmostEqual(s["ref_acc"], (1.0 + 0.5) / 2)  # by (task, index) the old script used 0.75 vs 1.0
        self.assertFalse(bench["multi"])

    def test_t1_multi_records_split_per_pair(self):
        bench = self.bench([{"task_id": "t1", "pair": 0, "attempt": 0, "multi_test": 2, "pred": [B_OUT, [[0]]],
                             "cuts": {}},
                            {"task_id": "t1", "pair": 0, "attempt": 1, "multi_test": 2, "pred": [B_OUT, None]}])
        self.assertTrue(bench["multi"])
        keys = BA.load_set(self.write_chal(), self.sol)
        self.assertEqual(bench["att"], {(keys[("t1", 0)], 0): True, (keys[("t1", 1)], 0): False,
                                        (keys[("t1", 0)], 1): True, (keys[("t1", 1)], 1): False})
        self.assertEqual(set(BA.REFERENCES), {"pair", "multi"})
        self.assertIn("e128t_final_all", BA.REFERENCES["multi"])

    @unittest.skipUnless((C.ROOT / "experiments/E046_multi_test_rollout/kaggle_bench/dsv41_logs/attempts.jsonl").exists()
                         and Path(f"{BA.PREFIX}e128t_final_all.jsonl").exists(), "local result files")
    def test_t1_kaggle_bench(self):
        sol = json.loads((C.ROOT / "data/arc-agi_evaluation_solutions.json").read_text())
        keys = BA.load_set(C.ROOT / "data/arc-agi_evaluation_challenges.json", sol)
        bench = BA.load_bench(C.ROOT / "experiments/E046_multi_test_rollout/kaggle_bench/dsv41_logs/attempts.jsonl",
                              keys, sol)
        s = BA.compare(bench, BA.load_reference(BA.REFERENCES["multi"]))
        self.assertEqual((s["attempts"], s["correct"], s["cut"]), (87, 64, 17))
        self.assertLess(abs(s["acc"] - s["ref_acc"]), 0.02)  # no L4 gap


class DeadlineSim(unittest.TestCase):
    def test_t6_clean_e112_pool(self):
        self.assertEqual(DS.POOLS["E112"], [("e112t_w10_u2_gptq_eng64_sub80", {0}),
                                            ("e112t_w10_u2_gptq_eng64_sub80_r2", {0})])
        d = Path(tempfile.mkdtemp(prefix="sim_", dir=C.SCRATCH))
        rows = [{"task_id": "t", "pair": 0, "attempt": a, "correct": a == 0, "usage": {"completion": 10}, "turns": 2}
                for a in (0, 1)]
        (d / "Xrun.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        old = DS.RES, DS.TAG
        DS.RES, DS.TAG = d, "X"
        try:
            self.assertEqual(len(DS.load(["run"])[("t", 0)]), 2)  # a tag alone: every attempt (as before)
            pool = DS.load([("run", {0})])
            self.assertEqual([r["correct"] for r in pool[("t", 0)]], [True])
        finally:
            DS.RES, DS.TAG = old

    @unittest.skipUnless(Path(f"{DS.RES}/{DS.TAG}e112t_w10_u2_gptq_eng64_sub80_r2.jsonl").exists(), "local results")
    def test_t6_real_pool_counts(self):
        pool = DS.load(DS.POOLS["E112"])
        recs = [r for v in pool.values() for r in v]
        self.assertEqual((len(pool), len(recs), sum(r["correct"] for r in recs)), (167, 334, 230))


if __name__ == "__main__":
    unittest.main()
