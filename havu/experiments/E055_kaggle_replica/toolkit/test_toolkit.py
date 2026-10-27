#!/usr/bin/env python3
"""Tests of the E055 replica toolkit (CPU only): .venv/bin/python -m unittest experiments/E055_kaggle_replica/toolkit/test_toolkit.py

- make_workload: D8 / colour inverses, whole-task round trips, validation of the shipped workload, reproducibility,
  prompt divergence;
- replica_report: Kaggle task-weighted vs per-pair scoring, multi-test record splitting, final vs re-queued errors,
  the Kaggle-clock mapping, and agreement with the solver's own "local score" lines on archived runs.
"""
import argparse
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import make_workload as mw  # noqa: E402
import replica_report as rr  # noqa: E402

RUNS = ROOT / "experiments/E046_multi_test_rollout/pods/runs"


def rand_grid(rng, h=None, w=None):
    h, w = h or rng.randint(1, 30), w or rng.randint(1, 30)
    return [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]


class WorkloadTest(unittest.TestCase):
    def test_d8_inverses_and_shapes(self):
        rng = random.Random(1)
        for _ in range(200):
            g = rand_grid(rng)
            for k in mw.D8:
                t = mw.D8[k](g)
                self.assertEqual(mw.D8[mw.D8_INV[k]](t), g, k)
                want = (len(g[0]), len(g)) if k in mw.SWAPS_SHAPE else (len(g), len(g[0]))
                self.assertEqual((len(t), len(t[0])), want, k)

    def test_d8_group_elements_distinct(self):
        g = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.assertEqual(len({json.dumps(mw.D8[k](g)) for k in mw.D8}), 8)
        self.assertEqual(mw.D8["r90"](g), [[7, 4, 1], [8, 5, 2], [0, 6, 3]])  # clockwise

    def test_colour_permutation_keeps_background(self):
        rng = random.Random(2)
        for _ in range(100):
            d8, perm = mw.draw(rng, [1, 2, 3])
            self.assertEqual(perm[0], 0)
            self.assertEqual(sorted(perm), list(range(10)))
            self.assertTrue(all(perm[c] != c for c in (1, 2, 3)))
            self.assertNotEqual(d8, "id")
            g = rand_grid(rng)
            self.assertEqual(mw.invert(mw.transform(g, d8, perm), d8, perm), g)
            zeros = sum(v == 0 for r in g for v in r)
            self.assertEqual(sum(v == 0 for r in mw.transform(g, d8, perm) for v in r), zeros)

    def test_task_round_trip(self):
        rng = random.Random(3)
        task = {"train": [{"input": rand_grid(rng, 4, 7), "output": rand_grid(rng, 2, 3)} for _ in range(3)],
                "test": [{"input": rand_grid(rng, 5, 6)}, {"input": rand_grid(rng, 9, 1)}]}
        sols = [rand_grid(rng, 3, 3), rand_grid(rng, 1, 8)]
        for k in mw.NON_ID:
            perm = [0] + rng.sample(range(1, 10), 9)
            ct, cs = mw.transform_task(task, sols, k, perm)
            for a, b in zip(ct["train"], task["train"]):
                self.assertEqual(mw.invert(a["input"], k, perm), b["input"])
                self.assertEqual(mw.invert(a["output"], k, perm), b["output"])
            for a, b in zip(ct["test"], task["test"]):
                self.assertEqual(mw.invert(a["input"], k, perm), b["input"])
            for a, b in zip(cs, sols):
                self.assertEqual(mw.invert(a, k, perm), b)

    def test_shipped_workload_validates(self):
        name = "evalx2t_s2026"
        d = HERE / "workloads"
        if not (d / f"{name}_challenges.json").exists():
            self.skipTest("workload not built")
        chal, sols, idmap = (json.loads((d / f"{name}_{x}.json").read_text()) for x in ("challenges", "solutions", "idmap"))
        src_c = json.loads((ROOT / "data/arc-agi_evaluation_challenges.json").read_text())
        src_s = json.loads((ROOT / "data/arc-agi_evaluation_solutions.json").read_text())
        n = mw.validate(chal, sols, idmap, src_c, src_s, mw.prompt_builder()[0])
        self.assertEqual((n["tasks"], n["pairs"], n["copies"]), (240, 344, 120))
        man = json.loads((d / f"{name}_manifest.json").read_text())
        for f, h in man["outputs"].items():
            self.assertEqual(mw.sha256(d / f), h, f)
        # every copy diverges from its original within the first grid, almost always within its first row
        copies = [m for m in idmap.values() if m["kind"] == "copy"]
        self.assertGreaterEqual(sum(m["shared_chars"] < m["first_row_chars"] for m in copies), 110)
        self.assertLess(max(m["shared_chars"] for m in copies), 300)

    def test_build_is_reproducible(self):
        args = argparse.Namespace(challenges=str(ROOT / "data/arc-agi_evaluation_challenges.json"),
                                  solutions=str(ROOT / "data/arc-agi_evaluation_solutions.json"),
                                  seed=7, copies=1, no_original=False)
        a, b = mw.build(args)[:3], mw.build(args)[:3]
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        args.seed = 8
        c = mw.build(args)[:3]
        self.assertNotEqual(json.dumps(a[2], sort_keys=True), json.dumps(c[2], sort_keys=True))

    def test_divergence_inside_first_grid(self):
        bp = mw.prompt_builder()[0]
        g = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]  # all-zero border: no D8 image changes the first row
        task = {"train": [{"input": g, "output": g}], "test": [{"input": g}]}
        ct, _ = mw.transform_task(task, [g], "r90", [0, 2, 3, 4, 5, 6, 7, 8, 9, 1])
        shared, row, start = mw.divergence(bp, task, ct)
        self.assertEqual(shared, len("[[0, 0, 0], [0, "))  # diverges at the recoloured 5
        self.assertGreater(start, 100)  # the instruction text before the first grid is shared by every task


def fake_run(tmp, solutions, records, sub, lines, metrics=(), env=None):
    d = Path(tmp)
    (d / "out/dsv41_logs").mkdir(parents=True)
    (d / "env.txt").write_text("\n".join(f"{k}={v}" for k, v in (env or {}).items()) + "\n")
    (d / "out/dsv41_logs/attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (d / "out/dsv41_logs/metrics.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
    (d / "out/dsv41_logs/solver.log").write_text("\n".join(lines) + "\n")
    (d / "out/submission.json").write_text(json.dumps(sub))
    sp = d / "sol_solutions.json"
    sp.write_text(json.dumps(solutions))
    return d, sp


def ns(**kw):
    base = dict(solutions="", challenges="", idmap="", baseline="", no_traces=True, out="")
    base.update(kw)
    return argparse.Namespace(**base)


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_task_weighted_vs_per_pair(self):
        A, B, C = [[1]], [[2]], [[3]]
        sols = {"t1": [A, B, C], "t2": [A]}  # t1: 3 pairs, t2: 1 pair
        sub = {"t1": [{"attempt_1": A, "attempt_2": C}, {"attempt_1": C, "attempt_2": C}, {"attempt_1": A, "attempt_2": A}],
               "t2": [{"attempt_1": C, "attempt_2": A}]}
        recs = [{"task_id": "t1", "pair": 0, "attempt": 0, "multi_test": 3, "pred": [A, C, A], "t_end_h": 1.0,
                 "wall_s": 600, "usage": {"completion": 5000}, "train_pass": True, "turns": 5, "tool_calls": 5},
                {"task_id": "t2", "pair": 0, "attempt": 0, "pred": C, "t_end_h": 1.2, "wall_s": 600,
                 "usage": {"completion": 3000}, "turns": 3},
                {"task_id": "t2", "pair": 0, "attempt": 1, "pred": A, "t_end_h": 1.5, "wall_s": 600,
                 "usage": {"completion": 1000000004, "completion_reported_cut": True}, "cuts": {"deadline": 1}}]
        lines = ["[10:00:00 +0.00h] fallback submission written (2 tasks); attempting 2 tasks, 4 pairs; deadlines: "
                 "stop-new 10.83h, force 11.17h, hard stop 11.50h",
                 "[10:05:00 +0.08h] server up after 280s; GPU KV cache size: 1,556,850 tokens",
                 "[11:30:00 +1.50h] submission written (final)", "[11:30:00 +1.50h] local score: 2/4 pairs",
                 "[11:30:01 +1.50h] done"]
        d, sp = fake_run(self.tmp, sols, recs, sub, lines, env={"ARC_MULTI_TEST": "1", "ARC_WORKERS": "4"})
        run = rr.Run(d, ns(solutions=str(sp)))
        R = rr.analyse(run, ns())
        self.assertAlmostEqual(R["score_pair"], 50.0)  # 2 of 4 pairs
        self.assertAlmostEqual(R["score_task"], 100 * (1 / 3 + 1) / 2)  # Kaggle: mean of per-task fractions
        self.assertEqual(R["solver_local_score"], (2, 4))
        self.assertEqual(R["acc"]["attempt 0"], (1, 4))  # t1 multi split into 3 pair rows (1 right) + t2 wrong
        self.assertEqual(R["acc"]["attempt 1"], (1, 1))
        self.assertEqual(rr.completion(recs[2]), 4)  # the BIG cut marker is removed
        self.assertEqual(len(R["jobs"]), 2)  # multi-test: one job per task
        self.assertEqual(R["cut_deadline"], 1)
        self.assertEqual(R["answer_path_lost"], 0)

    def test_final_vs_requeued_errors(self):
        A = [[1]]
        sols = {"t1": [A], "t2": [A]}
        sub = {t: [{"attempt_1": [[0]], "attempt_2": [[0]]}] for t in sols}
        err = lambda t, h: {"task_id": t, "pair": 0, "attempt": 0, "pred": None, "error": "TurnError: APIError",
                            "t_end_h": h, "wall_s": 60, "usage": {"completion": 100}}
        recs = [err("t1", 0.3), err("t2", 0.3), err("t2", 0.6)]
        lines = ["[10:00:00 +0.00h] attempting 2 tasks, 2 pairs; deadlines: stop-new 10.83h, force 11.17h, hard stop 11.50h",
                 "[10:18:00 +0.30h] retry t1.0#0: TurnError: APIError", "[10:18:00 +0.30h] retry t2.0#0: TurnError: APIError"]
        d, sp = fake_run(self.tmp, sols, recs, sub, lines)
        run = rr.Run(d, ns(solutions=str(sp)))
        R = rr.analyse(run, ns())
        self.assertEqual(R["final_err_keys"], {"t2.0#0"})  # t2 errored twice with one retry: lost for good
        self.assertEqual(R["no_a0_err"], [("t2", 0)])

    def test_kaggle_clock_mapping(self):
        A = [[1]]
        sols = {"t1": [A]}
        lines = ["[10:00:00 +0.00h] attempting 1 tasks, 1 pairs; deadlines: stop-new 10.83h, force 11.17h, hard stop 11.50h",
                 "[10:05:00 +0.08h] server start (first): model m", "[10:10:00 +0.17h] server up after 300s"]
        d, sp = fake_run(self.tmp, sols, [], {"t1": [{"attempt_1": A, "attempt_2": A}]}, lines)
        run = rr.Run(d, ns(solutions=str(sp)))
        R = rr.analyse(run, ns())
        t_up = 0.17 * 3600
        # 1 h of 10-s server rows with prefill share 0.2 everywhere
        R["stall"] = [({"t": t_up + 10 * i}, 0.2) for i in range(1, 361)]
        f = rr.kaggle_clock(run, R, 1.5, 0.95)
        self.assertAlmostEqual(f(100), 100 + rr.KAGGLE_SHIFT_S)
        self.assertAlmostEqual(f(t_up + 3600) - f(t_up), 3600 * (1 + 0.5 * 0.2) / 0.95, delta=1)
        xs = [f(t) for t in range(0, 8000, 50)]
        self.assertEqual(xs, sorted(xs))

    def test_archived_runs_match_solver_scores(self):
        cases = [(RUNS / "c36_deadline", (33, 172)), (ROOT / "experiments/E050_effort_ab/runs/eff50", (120, 172)),
                 (ROOT / "experiments/E046_multi_test_rollout/kaggle_bench", (65, 172))]
        n = 0
        for d, want in cases:
            if not d.exists():
                continue
            run = rr.Run(d, ns())
            R = rr.analyse(run, ns())
            self.assertEqual(R["solver_local_score"], want, d.name)
            self.assertEqual(sum(R["pair_ok"].values()), want[0], d.name)
            self.assertEqual(R["answer_path_lost"], 0, d.name)
            n += 1
        if not n:
            self.skipTest("archived runs not present")

    def test_rebuilt_submission_matches_solver(self):
        """The record-based rebuild (live mirrors, Kaggle-clock cut) selects the same answers as the solver did."""
        dirs = [RUNS / x for x in ("c36_deadline", "c36_restart", "mt44_g70_ri512_u970", "mt44_g70_ri512_new")]
        dirs += [ROOT / "experiments/E050_effort_ab/runs/eff50", ROOT / "experiments/E046_multi_test_rollout/kaggle_bench"]
        n = 0
        for d in dirs:
            if not d.exists():
                continue
            run = rr.Run(d, ns())
            _, last = rr.records_by_key(run)
            sub = rr.reconstruct_submission(run, last)
            for t, sol in run.solutions.items():
                for q in range(len(sol)):
                    got = sol[q] in (sub[t][q]["attempt_1"], sub[t][q]["attempt_2"])
                    want = sol[q] in (run.submission[t][q]["attempt_1"], run.submission[t][q]["attempt_2"])
                    self.assertEqual(got, want, f"{d.name} {t}.{q}")
            n += 1
        if not n:
            self.skipTest("archived runs not present")

    def test_report_renders_on_archived_run(self):
        d = RUNS / "mt44_g70_ri512_new"
        if not d.exists():
            self.skipTest("archived run not present")
        run = rr.Run(d, ns())
        R = rr.analyse(run, ns())
        text = rr.report(run, R, ns())
        for sec in ("## (a) Score", "## (b) Coverage", "## (c) Throughput", "## (d) Time decomposition",
                    "## (e) Memory", "## (f) Bottleneck verdict"):
            self.assertIn(sec, text)
        self.assertEqual(R["gpu_kind"], "L4")
        self.assertGreater(len(R["srv_rows"]), 1000)
        self.assertTrue(0.05 < R["prefill_share"] < 0.3)


if __name__ == "__main__":
    unittest.main()
