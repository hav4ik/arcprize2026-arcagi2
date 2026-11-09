"""report.py: verdict wording, warm-up exclusion, restarts, unplanned deaths, per-hour dynamics, JSON."""
import io
import json
import unittest
from contextlib import redirect_stdout

from kit_helpers import TmpCase, read, record, write_records
import report

T = 1793310000.0
T0 = T - 1800  # first sample at Kaggle h 0.5


def series(hours, f, step=60, **kw):
    return [record(T + i * step, t0=T0, startup_t=T, **dict(kw, **f(i * step / 3600.0)))
            for i in range(int(hours * 3600 / step))]


class Report(TmpCase):
    def build(self, recs, extra=()):
        p = self.path("mon.jsonl")
        write_records(p, recs)
        return report.build(report.build_args(["--samples", p] + list(extra)))

    def test_growing_toward_the_cap_with_eta(self):
        # 6 h: RAM flat at 134 GiB; cap - peak falls 0.2 GiB/h from 2.0 (0.8 at the end) -> ETA ~2.7 h at h 6.5
        out, text = self.build(series(6, lambda h: {"keq": 134.0, "cmp": 2.0 - 0.2 * h, "peak": 19.0 + 0.2 * h}))
        v = out["verdict"]["line"]
        self.assertTrue(v.startswith("VERDICT: RISK: worst cap - peak shrinking toward 0.26 GiB, ETA ~2.7 h"), v)
        self.assertIn("Kaggle-eq RAM stable", v)
        self.assertAlmostEqual(out["trends"]["cap_peak"]["slope_per_h"], -0.2, places=3)
        self.assertAlmostEqual(out["trends"]["peak"]["slope_per_h"], 0.2, places=3)
        self.assertEqual(out["clock"], "Kaggle h")
        self.assertIn("## Per-hour dynamics", text)
        self.assertIn("| 1-2 | 134.00 | 0.00 | 19.30 | 0.20 |", text)

    def test_stable(self):
        out, _ = self.build(series(4, lambda h: {"keq": 134.0 + 0.01 * h, "cmp": 1.7}))
        self.assertEqual(out["verdict"]["status"], "OK")
        self.assertIn("stable", out["verdict"]["line"])

    def test_warmup_is_excluded(self):
        # the first 20 min ramp RAM 100 -> 140 (server load); afterwards flat: the slope must be ~0
        out, _ = self.build(series(3, lambda h: {"keq": 100 + 120 * h if h < 1 / 3 else 140.0, "cmp": 1.7}),
                            ["--window-h", "5"])
        self.assertAlmostEqual(out["trends"]["ram"]["slope_per_h"], 0.0, places=6)
        self.assertEqual(out["trends"]["ram"]["status"], "OK")

    def test_oom_blows_up(self):
        recs = series(5, lambda h: {"cmp": max(0.05, 1.5 - 0.3 * h), "ooms": 1 if h >= 4 else 0})
        out, text = self.build(recs)
        self.assertEqual(out["verdict"]["status"], "FAIL")
        self.assertIn("GPU memory (first OOM) blew up at 4.50 h", out["verdict"]["line"])  # Kaggle h = 4 + 0.5
        self.assertIn("first GPU OOM", text)

    def test_ram_over_the_limit(self):
        out, _ = self.build(series(3, lambda h: {"keq": 150 + 10 * h}))
        self.assertIn("host RAM (Kaggle's limit) blew up at", out["verdict"]["line"])

    def test_restart_segments_and_unplanned_death(self):
        a = series(2, lambda h: {"cmp": 1.0 - 0.3 * h, "key": "s#0"})
        b = [record(T + 7200 + i * 60, t0=T0, startup_t=T + 7200, key="s#1", cmp=1.6, died=1)
             for i in range(120)]
        out, text = self.build(a + b)
        self.assertEqual(out["verdict"]["status"], "FAIL")
        self.assertIn("server died at 2.50 h without a fault injection", out["verdict"]["line"])
        self.assertEqual(out["trends"]["cap_peak"]["now"], 1.6)  # the current server only
        self.assertEqual(out["trends"]["cap_peak"]["status"], "OK")
        self.assertIn("server restart", text)

    def test_planned_death_is_not_a_failure(self):
        recs = [record(T + i * 60, t0=T0, startup_t=T, died=0 if i < 60 else 1, cmp=1.6) for i in range(180)]
        fl = self.path("fault.jsonl")
        with open(fl, "w") as f:
            f.write(json.dumps({"phase": "signaled", "action": "kill-tp", "t": T + 59 * 60 + 30, "pid": 20828,
                                "target": "VLLM::Worker_TP3"}) + "\n")
        out, text = self.build(recs, ["--fault-log", fl])
        self.assertNotIn("without a fault injection", out["verdict"]["line"])
        self.assertIn("server died (fault injection)", text)
        self.assertIn("## Fault injections", text)

    def test_json_and_meminfo_source_warning(self):
        p = self.path("mon.jsonl")
        write_records(p, series(1, lambda h: {"src": "meminfo_hostwide"}))
        jp = self.path("r.json")
        args = report.build_args(["--samples", p, "--json", jp])
        out, text = report.build(args)
        self.assertIn("WARNING: no cgroup accounting", text)
        with redirect_stdout(io.StringIO()):
            report.main(["--samples", p, "--json", jp])
        self.assertIn("verdict", json.loads(read(jp)))

    def test_without_solver_clock(self):
        out, text = self.build([record(T + i * 60, t0=None, startup_t=T) for i in range(60)])
        self.assertEqual(out["clock"], "h since first sample")

    def test_empty(self):
        out, text = self.build([])
        self.assertIn("no samples", text)


if __name__ == "__main__":
    unittest.main()
