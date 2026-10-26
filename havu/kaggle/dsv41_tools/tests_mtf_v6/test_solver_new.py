"""Deep audit tests of the pinned solver 4b344011 (new E046/E042 paths): env parsing, split_multi, the per-task
queue, pick, the KV gate, and a mini end-to-end of main() with fake attempts (no server, no GPU)."""
import json
import os
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import common as C

NB_ENV = {"ARC_WORKERS": "36", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_CONTEXT_LIMIT": "163840",
          "ARC_MAX_NUM_SEQS": "64", "ARC_GPU_UTIL": "0.945", "ARC_KV_ADMIT_MAX": "0.6", "ARC_KV_ADMIT_MIN_WORKERS": "16"}


class EnvParsing(unittest.TestCase):
    def args(self, env):
        return C.load_solver(env).make_args()

    def test_notebook_settings(self):
        s = C.load_solver(NB_ENV)
        self.assertEqual((s.WORKERS, s.MULTI_TEST, s.KV_ADMIT_MAX, s.KV_ADMIT_MIN_WORKERS), (36, True, 0.6, 16))
        a = s.make_args()
        self.assertEqual((a.multi_test, a.multi_budget, a.context_limit), (True, 0.5, 163840))
        self.assertEqual((a.max_total_output, a.budget_soft, a.max_turns, a.max_tokens), (80000, 40000, 40, 64000))
        self.assertTrue(a.token_exact and a.answer_tool and a.fallback_solve and a.train_check and a.rescue_code)
        self.assertEqual(a.final_mode, "session")

    def test_defaults_absent_empty_zero(self):
        a = self.args({})
        self.assertEqual((a.multi_test, a.multi_budget, a.context_limit), (False, 0.0, 0))
        for env, want in (({"ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": ""}, (True, 0.0, 0)),
                          ({"ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0"}, (True, 0.0, 0)),
                          ({"ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": " 0.5 "}, (True, 0.5, 0)),
                          ({"ARC_MULTI_TEST": "0", "ARC_MULTI_BUDGET": "0.5"}, (False, 0.0, 0)),
                          ({"ARC_MULTI_TEST": "", "ARC_CONTEXT_LIMIT": "163840"}, (False, 0.0, 163840)),
                          ({"ARC_MULTI_TEST": "true"}, (False, 0.0, 0)),  # only "1" turns it on
                          ({"ARC_CONTEXT_LIMIT": "0"}, (False, 0.0, 0)),
                          ({"ARC_CONTEXT_LIMIT": ""}, (False, 0.0, 0))):
            a = self.args(env)
            self.assertEqual((a.multi_test, a.multi_budget, a.context_limit), want, env)
        s = C.load_solver({})
        self.assertEqual((s.KV_ADMIT_MAX, s.KV_ADMIT_MIN_WORKERS, s.TASK_IDS), (0.0, 16, []))
        self.assertEqual(C.load_solver({"ARC_TASK_IDS": "a,b,,c"}).TASK_IDS, ["a", "b", "c"])

    def test_bad_values(self):
        with self.assertRaises(ValueError):  # at import (module level), before the try in __main__
            C.load_solver({"ARC_KV_ADMIT_MAX": ""})
        s = C.load_solver({"ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "half"})
        with self.assertRaises(SystemExit):  # argparse error; SystemExit is not caught by main's `except Exception`
            s.make_args()
        s = C.load_solver({"ARC_CONTEXT_LIMIT": "163840.0"})
        with self.assertRaises(SystemExit):
            s.make_args()


class SplitAndPick(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = C.load_solver(NB_ENV)

    def test_split_multi(self):
        task = {"test": [{"input": [[1]]}] * 3}
        g = [[[1]], [[2]], [[3]]]
        rec = {"task_id": "t", "pair": 0, "attempt": 1, "multi_test": 3, "pred": g, "train_pass": True,
               "usage": {"completion": 100}}
        out = self.s.split_multi(rec, task)
        self.assertEqual([(r["pair"], r["pred"], r.get("multi_share"), r["train_pass"], r["attempt"]) for r in out],
                         [(0, [[1]], None, True, 1), (1, [[2]], True, True, 1), (2, [[3]], True, True, 1)])
        self.assertEqual(out[1]["usage"], {"completion": 100})  # usage copied (harness pair_records zeroes it)
        self.assertEqual(rec["pred"], g)  # the attempt record is not modified
        for bad in (None, [[[1]], [[2]]], "x"):
            out = self.s.split_multi(dict(rec, pred=bad), task)
            self.assertEqual([r["pred"] for r in out], [None] * 3)
        out = self.s.split_multi(dict(rec, pred=None, error="TurnError: x"), task)
        self.assertTrue(all(r["error"] for r in out))
        single = {"task_id": "u", "pair": 0, "attempt": 0, "pred": [[1]]}
        self.assertEqual(self.s.split_multi(single, {"test": [{}]}), [single])
        driver = {"task_id": "t", "pair": 0, "attempt": 0, "error": "driver: X", "pred": None}  # no multi_test key
        self.assertEqual(self.s.split_multi(driver, task), [driver])

    def test_pick_multi(self):
        s = self.s
        s.RECORDS.clear()
        s.RECORDS[("a", 0)] = [{"attempt": 0, "train_pass": True, "pred": [[1]]}]
        s.RECORDS[("b", 0)] = [{"attempt": 0, "train_pass": False, "pred": [[1]]}]
        queue = [("a", 0, 1, 0), ("c", 0, 1, 0), ("b", 0, 1, 0)]
        self.assertEqual(s.pick(queue), 2)  # b: attempt 0 failed the train check -> first
        self.assertEqual(s.pick([("a", 0, 1, 0), ("c", 0, 1, 0)]), 1)  # c still running -> before a
        self.assertEqual(s.pick([("a", 0, 1, 0), ("x", 0, 0, 0)]), 1)  # attempt-0 jobs always first
        s.RECORDS.clear()


class KVGate(unittest.TestCase):
    def setUp(self):
        self.s = C.load_solver(NB_ENV)
        self.now = [1000.0]
        self.reads = []
        self.kv = [0.3]

        def metrics():
            self.reads.append(self.now[0])
            v = self.kv[0]
            return None if v == "down" else ({} if v == "nokey" else {"kv_cache_usage_perc": v, "t": self.now[0]})
        self.p1 = patch.object(self.s, "server_metrics", metrics)
        self.p2 = patch.object(self.s.time, "time", lambda: self.now[0])
        self.p1.start()
        self.p2.start()

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()

    def test_below_min_workers_never_reads(self):
        self.kv[0] = 0.99
        self.assertTrue(all(self.s.kv_admits(n) for n in range(16)))
        self.assertEqual(self.reads, [])
        self.assertFalse(self.s.kv_admits(16))

    def test_low_kv_admits_many_per_reading(self):
        self.kv[0] = 0.3
        self.assertTrue(all(self.s.kv_admits(20 + i) for i in range(16)))
        self.assertEqual(len(self.reads), 1)

    def test_near_threshold_one_launch_per_reading(self):
        self.kv[0] = 0.55
        self.assertTrue(self.s.kv_admits(20))
        self.assertFalse(self.s.kv_admits(21))
        self.now[0] += 9.9
        self.assertFalse(self.s.kv_admits(21))
        self.now[0] += 0.1
        self.assertTrue(self.s.kv_admits(21))  # fresh reading
        self.assertEqual(len(self.reads), 2)

    def test_above_threshold_holds_and_logs_every_300s(self):
        self.kv[0] = 0.6
        logs = []
        with patch.object(self.s, "log", logs.append):
            for _ in range(5):
                self.assertFalse(self.s.kv_admits(30))
                self.now[0] += 60
            self.assertEqual(len(logs), 1)
            self.assertIn("KV gate: usage 0.600 (limit 0.6); holding new attempts at 30 running", logs[0])
            self.now[0] += 300
            self.kv[0] = 0.4
            self.assertTrue(self.s.kv_admits(30))

    def test_metrics_missing_fail_open(self):
        for v in ("down", "nokey"):  # unreachable /metrics, or the metric renamed: the gate admits (fail-open)
            self.s.STATE.pop("kv_t", None)
            self.kv[0] = v
            self.assertTrue(self.s.kv_admits(30))
            self.assertTrue(self.s.kv_admits(31))


class FakeProc:
    pid = 999999999

    def poll(self):
        return None


def run_main(env, tasks, attempt_fn, metrics=None, timeout=60):
    """Run the pinned solver's main() with a fake server and fake attempts; returns (solver, launches, sub)."""
    work = Path(tempfile.mkdtemp(prefix="solver_main_"))
    (work / "model").mkdir()
    (work / "model" / "config.json").write_text("{}")
    (work / "challenges.json").write_text(json.dumps(tasks))
    full_env = {**env, "ARC_CHALLENGES": str(work / "challenges.json"), "ARC_MODEL_DIR": str(work / "model"),
                "ARC_OUT": str(work / "out")}
    s = C.load_solver(full_env)
    launches, lock = [], threading.Lock()

    def start_server():
        s.STATE["server"], s.STATE["server_ok"] = FakeProc(), True
        return True

    def run_attempt(args, task, t, p, a):
        with lock:
            launches.append((t, p, a))
        return attempt_fn(args, task, t, p, a)

    class Exit(BaseException):
        pass

    def fake_exit(code):
        raise Exit(code)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with patch.object(s, "start_server", start_server), patch.object(s, "stop_server", lambda: None), \
                patch.object(s, "run_attempt", run_attempt), patch.object(s, "server_metrics", metrics or (lambda: None)), \
                patch.object(s.os, "_exit", fake_exit), patch.dict(os.environ, full_env):
            try:
                s.main()
            except Exit:
                pass
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
    sub = json.loads((work / "out" / "submission.json").read_text())
    return s, launches, sub, work


def grid(v, h=2, w=2):
    return [[v] * w for _ in range(h)]


TASKS = {"aaaa0001": {"train": [{"input": grid(1), "output": grid(2)}], "test": [{"input": grid(1)}]},
         "bbbb0002": {"train": [{"input": grid(1, 3, 3), "output": grid(2)}],
                      "test": [{"input": grid(3)}, {"input": grid(4)}]},
         "cccc0003": {"train": [{"input": grid(1, 5, 5), "output": grid(2)}],
                      "test": [{"input": grid(5)}, {"input": grid(6)}, {"input": grid(7)}]}}


def harness_like(args, task, t, p, a):
    """What run_attempt returns for an answered attempt: multi tasks a list of n grids (colour = 10 * a + pair)."""
    n = len(task["test"])
    multi = args.multi_test and n > 1
    preds = [grid((p + q + a) % 10) for q in range(n)]
    rec = {"pred": preds if multi else preds[0], "stop_reason": "submitted", "usage": {"completion": 1000},
           "turns": 3, "train_pass": a == 0, "task_id": t, "pair": p, "attempt": a, "wall_s": 1.0, "cuts": {}}
    if multi:
        rec["multi_test"] = n
    return rec


class MainLoop(unittest.TestCase):
    def test_multi_queue_and_submission(self):
        s, launches, sub, work = run_main({**NB_ENV, "ARC_WORKERS": "1"}, TASKS, harness_like)
        # one job per task and attempt; pair 0 only; attempt 0 for all tasks (smallest first), then attempt 1
        self.assertEqual(launches, [("aaaa0001", 0, 0), ("bbbb0002", 0, 0), ("cccc0003", 0, 0),
                                    ("aaaa0001", 0, 1), ("bbbb0002", 0, 1), ("cccc0003", 0, 1)])
        self.assertEqual([len(sub[t]) for t in sorted(TASKS)], [1, 2, 3])
        for t, task in TASKS.items():
            for q in range(len(task["test"])):
                self.assertEqual(sub[t][q], {"attempt_1": grid(q % 10), "attempt_2": grid((q + 1) % 10)}, (t, q))
        lines = (work / "out" / "dsv41_logs" / "attempts.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(sorted(s.RECORDS), [("aaaa0001", 0), ("bbbb0002", 0), ("bbbb0002", 1), ("cccc0003", 0),
                                             ("cccc0003", 1), ("cccc0003", 2)])
        self.assertTrue(all(len(v) == 2 for v in s.RECORDS.values()))

    def test_per_pair_queue_when_off(self):
        env = {k: v for k, v in NB_ENV.items() if k not in ("ARC_MULTI_TEST", "ARC_MULTI_BUDGET")}
        s, launches, sub, work = run_main({**env, "ARC_WORKERS": "1"}, TASKS, harness_like)
        self.assertEqual(len(launches), 12)
        self.assertEqual(sub["cccc0003"][2], {"attempt_1": grid(2), "attempt_2": grid(3)})

    def test_multi_error_retry_and_driver_crash(self):
        calls = {}

        def flaky(args, task, t, p, a):
            calls[(t, a)] = calls.get((t, a), 0) + 1
            if t == "bbbb0002" and a == 0 and calls[(t, a)] == 1:  # harness error, no answer: retried once
                return {"pred": None, "error": "RuntimeError: boom", "multi_test": 2, "usage": {"completion": 0},
                        "task_id": t, "pair": p, "attempt": a, "cuts": {}}
            if t == "cccc0003" and a == 1:
                raise RuntimeError("driver crash")  # not retried twice: JOB_TRIES 2 -> one retry
            return harness_like(args, task, t, p, a)
        s, launches, sub, work = run_main({**NB_ENV, "ARC_WORKERS": "1"}, TASKS, flaky)
        self.assertEqual(calls[("bbbb0002", 0)], 2)
        self.assertEqual(calls[("cccc0003", 1)], 2)
        self.assertEqual(sub["bbbb0002"][1]["attempt_1"], grid(1))
        # cccc0003 attempt 1 never answered: attempt 0's grid + the fallback (test input) for every pair
        for q in range(3):
            self.assertEqual(sub["cccc0003"][q], {"attempt_1": grid(q), "attempt_2": TASKS["cccc0003"]["test"][q]["input"]})

    def test_task_ids_and_full_submission(self):
        s, launches, sub, work = run_main({**NB_ENV, "ARC_WORKERS": "2", "ARC_TASK_IDS": "cccc0003,zzzz9999"},
                                          TASKS, harness_like)
        self.assertEqual(sorted(set(t for t, _, _ in launches)), ["cccc0003"])
        self.assertEqual(set(sub), set(TASKS))  # the submission still covers every task
        self.assertEqual(sub["aaaa0001"][0]["attempt_1"], TASKS["aaaa0001"]["test"][0]["input"])

    def test_gate_holds_with_fake_metrics(self):
        running = {"n": 0}
        lock = threading.Lock()
        release = threading.Event()

        def slow(args, task, t, p, a):
            with lock:
                running["n"] += 1
            release.wait(5)
            with lock:
                running["n"] -= 1
            return harness_like(args, task, t, p, a)
        many = {f"t{i:07d}": TASKS["aaaa0001"] for i in range(12)}
        peak = {"n": 0}

        def metrics():
            with lock:
                peak["n"] = max(peak["n"], running["n"])
            return {"kv_cache_usage_perc": 0.7, "t": time.time()}
        timer = threading.Timer(3.0, release.set)
        timer.start()
        env = {**NB_ENV, "ARC_WORKERS": "10", "ARC_KV_ADMIT_MIN_WORKERS": "4"}
        s, launches, sub, work = run_main(env, many, slow, metrics=metrics)
        timer.cancel()
        self.assertEqual(len(launches), 24)  # everything ran in the end
        self.assertLessEqual(peak["n"], 4)  # while KV read 0.7, no launch above 4 running


if __name__ == "__main__":
    unittest.main()
