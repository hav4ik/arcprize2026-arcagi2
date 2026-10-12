"""Offline tests for kaggle/monitor_submissions.py (fake API responses, no network).

    .venv/bin/python kaggle/tests/test_monitor_submissions.py
"""
import datetime as dt
import enum
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor_submissions as ms  # noqa: E402

UTC = dt.timezone.utc


class SubmissionStatus(enum.Enum):  # mirrors kagglesdk's enum
    PENDING = 0
    COMPLETE = 1
    ERROR = 2


def fake_sub(ref, status, score="", err="", date=dt.datetime(2026, 10, 11, 23, 51, 41, 847000)):
    return SimpleNamespace(ref=ref, date=date, description="Qwen3.8 + tools", status=status,
                           public_score=score, private_score="", error_description=err, file_name="submission.json")


class FakeApi:
    def __init__(self, pages):
        self.pages, self.calls = pages, 0

    def competition_submissions(self, competition, page_size=20):
        p = self.pages[min(self.calls, len(self.pages) - 1)]
        self.calls += 1
        if isinstance(p, Exception):
            raise p
        return p


def t(h):  # hours after submission time
    return dt.datetime(2026, 10, 11, 23, 51, 41, 847000, tzinfo=UTC) + dt.timedelta(hours=h)


class ParseTest(unittest.TestCase):
    def test_status_forms(self):
        for v in (SubmissionStatus.PENDING, "SubmissionStatus.PENDING", "pending", 0):
            self.assertEqual(ms.parse_status(v), "PENDING")
        self.assertEqual(ms.parse_status(SubmissionStatus.ERROR), "ERROR")
        self.assertEqual(ms.parse_status("COMPLETE"), "COMPLETE")

    def test_normalize_object_and_csv_dict(self):
        n = ms.normalize(fake_sub(57088016, SubmissionStatus.COMPLETE, score="0.125"))
        self.assertEqual(n["id"], "57088016")
        self.assertEqual(n["submitted_at"], "2026-10-11T23:51:41.847000+00:00")
        self.assertEqual(n["status"], "COMPLETE")
        self.assertEqual(n["public_score"], 0.125)
        self.assertIsNone(n["private_score"])
        # CSV row from `kaggle competitions submissions --csv`
        row = {"ref": "57088016", "date": "2026-10-11 23:51:41.847000", "description": "x",
               "status": "SubmissionStatus.PENDING", "publicScore": "", "privateScore": ""}
        n2 = ms.normalize(row)
        self.assertEqual(n2["submitted_at"], n["submitted_at"])
        self.assertEqual(n2["status"], "PENDING")
        self.assertIsNone(n2["public_score"])

    def test_fetch_skips_none(self):
        api = FakeApi([[fake_sub(1, SubmissionStatus.PENDING), None]])
        self.assertEqual([s["id"] for s in ms.fetch(api, "c")], ["1"])

    def test_table(self):
        subs = [ms.normalize(fake_sub(1, SubmissionStatus.PENDING)),
                ms.normalize(fake_sub(2, SubmissionStatus.ERROR, err="Notebook Timeout"))]
        out = ms.table(subs, t(3.42))
        self.assertIn("3.42 h", out)
        self.assertIn("[error: Notebook Timeout]", out)


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "watch.jsonl"
        self.out = io.StringIO()

    def tearDown(self):
        self.dir.cleanup()

    def watch(self):
        return ms.Watch(self.path, interval=300, warn_hours=11.0, guard_hours=11.75, out=self.out)

    def poll(self, w, status, h, **kw):
        return w.update([ms.normalize(fake_sub(57088016, status, **kw))], t(h))

    def records(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def test_lifecycle_and_wall_time(self):
        w = self.watch()
        self.assertEqual(len(self.poll(w, SubmissionStatus.PENDING, 0.2)), 1)
        self.assertEqual(self.poll(w, SubmissionStatus.PENDING, 5.0), [])      # no change -> no record
        self.assertEqual(self.poll(w, SubmissionStatus.PENDING, 9.5), [])
        ch = self.poll(w, SubmissionStatus.COMPLETE, 9.5 + 5 / 60, score="0.0833")
        self.assertEqual(len(ch), 1)
        r = ch[0]
        self.assertEqual((r["status"], r["prev_status"], r["public_score"]), ("COMPLETE", "PENDING", 0.0833))
        self.assertAlmostEqual(r["wall_h_lower"], 9.5, places=3)               # last PENDING poll
        self.assertAlmostEqual(r["wall_h_upper"], 9.5 + 5 / 60, places=3)      # first non-pending poll
        self.assertEqual([x["status"] for x in self.records()], ["PENDING", "COMPLETE"])
        log = self.out.getvalue()
        self.assertIn("NEW 57088016 PENDING 0.20 h", log)
        self.assertIn("DONE 57088016 COMPLETE public=0.0833", log)
        self.assertIn("wall 9.50-9.58 h", log)

    def test_resume_from_state_file(self):
        w = self.watch()
        self.poll(w, SubmissionStatus.PENDING, 0.2)
        w2 = self.watch()                                                       # restart
        self.assertEqual(self.poll(w2, SubmissionStatus.PENDING, 4.0), [])      # already recorded
        r = self.poll(w2, SubmissionStatus.ERROR, 6.0, err="Notebook Threw Exception")[0]
        self.assertEqual(r["error"], "Notebook Threw Exception")
        self.assertAlmostEqual(r["wall_h_lower"], 4.0, places=3)
        self.assertEqual(len(self.records()), 2)

    def test_already_complete_when_first_seen(self):
        w = self.watch()
        r = self.poll(w, SubmissionStatus.COMPLETE, 30, score="0.1")[0]
        self.assertNotIn("wall_h_upper", r)
        self.assertIn("SEEN 57088016 COMPLETE", self.out.getvalue())

    def test_warning_once_and_heartbeat(self):
        w = self.watch()
        self.poll(w, SubmissionStatus.PENDING, 10.9)
        self.assertNotIn("WARNING", self.out.getvalue())
        self.poll(w, SubmissionStatus.PENDING, 11.1)
        self.poll(w, SubmissionStatus.PENDING, 11.2)
        self.assertEqual(self.out.getvalue().count("WARNING"), 1)
        w.heartbeat([ms.normalize(fake_sub(57088016, SubmissionStatus.PENDING))], t(3.42))
        self.assertIn("pending 3.42 h; notebook guard at ~11.75 h", self.out.getvalue())


class RetryTest(unittest.TestCase):
    def test_backoff_then_success(self):
        api = FakeApi([RuntimeError("429 Too Many Requests"), [fake_sub(1, SubmissionStatus.PENDING)]])
        msgs, sleeps = [], []
        orig = ms.time.sleep
        ms.time.sleep = sleeps.append
        try:
            subs = ms.fetch_with_retry(lambda: api, {}, "c", msgs.append)
        finally:
            ms.time.sleep = orig
        self.assertEqual(subs[0]["id"], "1")
        self.assertEqual(sleeps, [30.0])
        self.assertIn("429", msgs[0])


if __name__ == "__main__":
    unittest.main()
