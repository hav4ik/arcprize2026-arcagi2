"""watcher.py: level crossings fire once, state survives --once polls, events, trend warnings, exit codes."""
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

from kit_helpers import TmpCase, read, record, write_records
import watcher

T = 1793310000.0
T0 = T - 600


def codes(alerts):
    return [(a[1], a[2]) for a in alerts]


class Watcher(TmpCase):
    def run_once(self, recs, extra=(), state=None, now=None):
        p = self.path("mon.jsonl")
        write_records(p, recs)
        a = watcher.build_args(["--samples", p, "--no-stale"] + list(extra))
        w = watcher.Watcher(a, state)
        alerts, _ = w.run_once(now)
        return w, alerts

    def test_ram_levels_fire_once(self):
        recs = [record(T + 60 * i, keq=k, t0=T0) for i, k in enumerate((150, 156, 157, 166, 160, 174))]
        w, alerts = self.run_once(recs)
        ram = [a for a in alerts if a[2] == "ram"]
        self.assertEqual([a[1] for a in ram], ["WARN", "ALERT", "CRIT"])
        self.assertIn("Kaggle would OOM-kill", ram[-1][3])
        self.assertIn("[Kaggle h", ram[0][3])

    def test_state_survives_polls(self):
        p = self.path("mon.jsonl")
        st = self.path("st.json")
        args = ["--samples", p, "--state", st, "--alerts", self.path("al.log"), "--no-stale", "--once"]
        write_records(p, [record(T, keq=156, t0=T0)])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(watcher.main(args), 2)  # WARN
            self.assertEqual(watcher.main(args), 0)  # nothing new
        write_records(p, [record(T, keq=156, t0=T0), record(T + 60, keq=157, t0=T0), record(T + 120, keq=166, t0=T0)])
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(watcher.main(args), 3)  # ALERT only
        self.assertEqual(out.getvalue().count(" ram: "), 1)
        self.assertEqual(read(self.path("al.log")).count("\n"), 2)

    def test_vram_worst_rank_retries_and_ooms(self):
        recs = [record(T, cmp=0.8, t0=T0), record(T + 60, cmp=0.45, t0=T0), record(T + 120, cmp=0.2, retries=2, t0=T0),
                record(T + 180, cmp=0.2, retries=2, t0=T0), record(T + 240, cmp=0.1, retries=3, ooms=1, t0=T0)]
        _, alerts = self.run_once(recs)
        c = codes(alerts)
        self.assertEqual([x for x in c if x[1] == "cap_peak"], [("WARN", "cap_peak"), ("ALERT", "cap_peak")])
        self.assertEqual(c.count(("ALERT", "retries")), 2)  # 0 -> 2, 2 -> 3; one line each, all ranks listed
        self.assertIn(("CRIT", "vllm_oom"), c)
        self.assertIn("worst rank TP", [a for a in alerts if a[2] == "cap_peak"][0][3])

    def test_new_server_rearms_cap_alerts(self):
        recs = [record(T, cmp=0.2, key="a#0", t0=T0), record(T + 60, cmp=0.2, key="b#0", t0=T0)]
        _, alerts = self.run_once(recs)
        self.assertEqual(codes(alerts).count(("ALERT", "cap_peak")), 2)

    def test_events(self):
        recs = [record(T, t0=T0),
                record(T + 60, engines=2, died=1, t0=T0, sup={"state": "restarting", "start": 1}),
                record(T + 120, engines=1, died=1, stop_error=1, t0=T0, sup={"state": "restarting", "start": 1}),
                record(T + 480, engines=1, died=1, stop_error=1, t0=T0, sup={"state": "restarting", "start": 1}),
                record(T + 540, engines=1, died=1, stop_error=1, t0=T0, sup={"state": "up", "start": 1},
                       nb_end="NOTEBOOK_DONE ok +43000s")]
        recs[0]["supervisor"] = {"state": "up", "start": 0}
        recs[1]["vllm"]["counts"]["dead"] = 3
        _, alerts = self.run_once(recs)
        c = codes(alerts)
        for want in (("CRIT", "two_engines"), ("ALERT", "server_died"), ("ALERT", "supervisor_restart"),
                     ("ALERT", "engine_dead"), ("ALERT", "stop_error"), ("ALERT", "not_up"),
                     ("INFO", "notebook_end")):
            self.assertIn(want, c)
        self.assertEqual(c.count(("ALERT", "server_died")), 1)

    def test_error_burst(self):
        recs = [record(T + 60 * i, errors=e, t0=T0) for i, e in enumerate((0, 1, 2, 7, 8))]
        _, alerts = self.run_once(recs)
        self.assertEqual(codes(alerts).count(("WARN", "error_burst")), 1)

    def test_stale_and_source(self):
        p = self.path("mon.jsonl")
        write_records(p, [record(T, t0=T0, src="meminfo_hostwide")])
        w = watcher.Watcher(watcher.build_args(["--samples", p]))
        alerts, _ = w.run_once(now=T + 1000)
        self.assertIn(("ALERT", "stale"), codes(alerts))
        self.assertIn(("WARN", "ram_source"), codes(alerts))
        self.assertEqual(w.run_once(now=T + 1100)[0], [])

    def test_trend_warning(self):
        # cap - peak falls 0.3 GiB/h from 2.0: at 3 h it is ~1.1, ETA to 0.26 ~2.8 h, before the 11.5 h end
        recs = [record(T + 60 * i, cmp=2.0 - 0.3 * i / 60, t0=T0, startup_t=T) for i in range(180)]
        _, alerts = self.run_once(recs)
        tr = [a for a in alerts if a[2] == "trend"]
        self.assertTrue(tr)
        self.assertIn("worst cap - peak shrinking toward 0.26 GiB", tr[-1][3])

    def test_stable_run_is_quiet(self):
        recs = [record(T + 60 * i, keq=134.0 + 0.001 * i, cmp=1.7, t0=T0, startup_t=T) for i in range(300)]
        _, alerts = self.run_once(recs)
        self.assertEqual(alerts, [])

    def test_env_threshold_and_smi_option(self):
        with mock.patch.dict(os.environ, {"RELI_RAM_WARN": "100", "RELI_SMI_FREE_WARN_MIB": "2000"}):
            _, alerts = self.run_once([record(T, keq=140, t0=T0)])
        self.assertIn(("WARN", "ram"), codes(alerts))
        self.assertIn(("INFO", "smi_free"), codes(alerts))

    def test_follow_exit_on(self):
        p = self.path("mon.jsonl")
        write_records(p, [record(T, t0=T0), record(T + 60, keq=166, t0=T0)])
        with redirect_stdout(io.StringIO()) as out:
            rc_ = watcher.main(["--samples", p, "--follow", "--exit-on", "ALERT", "--no-stale", "--poll-s", "0.01"])
        self.assertEqual(rc_, 3)
        self.assertIn("FLAG ALERT ram", out.getvalue())
        self.assertIn("status at", out.getvalue())

    def test_until_epoch(self):
        now = 1793310000.0  # 2026-10-29 21:40 UTC
        self.assertEqual(watcher.until_epoch("22:00", now) - now, 20 * 60)
        self.assertGreater(watcher.until_epoch("21:00", now), now)


if __name__ == "__main__":
    unittest.main()
