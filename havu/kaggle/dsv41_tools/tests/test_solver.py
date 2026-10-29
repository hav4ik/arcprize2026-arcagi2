"""Tests of the live solver (kaggle/dsv41_tools/solver_dsv41.py) on the Kaggle runtime of the next candidate: env
parsing, split_multi, the per-task queue, pick, the KV gate, and main() with a fake server and fake attempts (no GPU).
Ported from the 2026-10-26 deep audit (../tests_mtf_v6/test_solver_new.py), plus one regression test per scheduling
fix of ../next_candidate_fixes.md (S1, S5, S6, S7, S12, S13, S14, T3); each fails on the V6 solver 4b344011.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests"""
import http.server
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import common as C

NB_ENV = {"ARC_WORKERS": "36", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_CONTEXT_LIMIT": "163840",
          "ARC_MAX_NUM_SEQS": "64", "ARC_GPU_UTIL": "0.945", "ARC_KV_ADMIT_MAX": "0.6", "ARC_KV_ADMIT_MIN_WORKERS": "16"}


def free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


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
        self.assertEqual(s.ENV_WARNINGS, [])
        s.STATE["model"] = "/m"
        cmd = s.server_cmd()
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "64")
        self.assertEqual(cmd[cmd.index("--gpu-memory-utilization") + 1], "0.945")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "163840")

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
        self.assertEqual((s.MAX_RESTARTS, s.WORKERS, s.PORT, s.FORCE_AT_S), (3, 48, 8100, 11 * 3600 + 10 * 60))
        self.assertEqual(C.load_solver({"ARC_TASK_IDS": "a,b,,c"}).TASK_IDS, ["a", "b", "c"])

    def test_s12_bad_values_give_defaults_and_warnings(self):
        # S12 (fails on V6: ValueError at import, or argparse SystemExit escaping main's `except Exception`)
        s = C.load_solver({"ARC_KV_ADMIT_MAX": "", "ARC_WORKERS": "lots", "ARC_T0": "", "ARC_SERVER_EXTRA_ARGS": "[oops",
                           "ARC_SERVER_ENV_JSON": "[]", "ARC_GPU_UTIL": "nan"})
        self.assertEqual((s.KV_ADMIT_MAX, s.WORKERS, s.SERVER_EXTRA_ARGS, s.SERVER_ENV_JSON, s.GPU_UTIL),
                         (0.0, 48, [], {}, "0.93"))
        self.assertLess(abs(s.T0 - time.time()), 60)
        self.assertEqual(len(s.ENV_WARNINGS), 6, s.ENV_WARNINGS)
        self.assertTrue(any("ARC_WORKERS='lots'" in w for w in s.ENV_WARNINGS), s.ENV_WARNINGS)
        a = C.load_solver({"ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "half"}).make_args()
        self.assertEqual((a.multi_test, a.multi_budget), (True, 0.0))
        self.assertEqual(C.load_solver({"ARC_CONTEXT_LIMIT": "163840.0"}).make_args().context_limit, 163840)
        self.assertEqual(C.load_solver({"ARC_CONTEXT_LIMIT": "-5"}).make_args().context_limit, 0)
        self.assertEqual(C.load_solver({"ARC_WORKERS": "36.0"}).WORKERS, 36)

    def test_server_cmd_json_redacts_credentials(self):
        # E050: a box image's HF_TOKEN reached server_cmd.json; credential-like keys are now redacted
        import json, tempfile
        from unittest.mock import patch
        s = C.load_solver(NB_ENV)
        s.STATE["model"] = "/m"
        d = Path(tempfile.mkdtemp())
        env = {"HF_TOKEN": "hf_" + "x" * 34, "HF_HUB_OFFLINE": "1", "VLLM_API_KEY": "k", "CUDA_HOME": "/c"}
        with patch.object(s, "LOG", d), patch.object(s, "server_env", lambda: env), \
                patch.object(s, "server_alive", lambda: False), patch.object(s.subprocess, "Popen", side_effect=OSError):
            try:
                s.start_server()
            except Exception:
                pass
        got = json.loads((d / "server_cmd.json").read_text())["env"]
        self.assertEqual(got["HF_TOKEN"], "<redacted>")
        self.assertEqual(got["VLLM_API_KEY"], "<redacted>")
        self.assertEqual((got["HF_HUB_OFFLINE"], got["CUDA_HOME"]), ("1", "/c"))
        self.assertNotIn("hf_xxx", (d / "server_cmd.json").read_text())

    def test_e050_reasoning_effort_env(self):
        # ARC_REASONING_EFFORT sets the numeric V4.1 effort; unset, empty or invalid keeps V6's 50
        effort = lambda env: C.load_solver(env).EXTRA_BODY["chat_template_kwargs"]["reasoning_effort"]
        self.assertEqual(effort(NB_ENV), 50)
        self.assertEqual(effort({"ARC_REASONING_EFFORT": "25"}), 25)
        self.assertEqual(effort({"ARC_REASONING_EFFORT": " 100 "}), 100)
        for bad in ("", "0", "101", "low", "nan"):
            self.assertEqual(effort({"ARC_REASONING_EFFORT": bad}), 50, bad)

    def test_i7_best_guess_env(self):
        # I7: ARC_BEST_GUESS_AT / ARC_BEST_GUESS_TURNS pass --best-guess-at / --best-guess-turns through
        a = self.args({"ARC_BEST_GUESS_AT": "0.8", "ARC_BEST_GUESS_TURNS": "3"})
        self.assertEqual((a.best_guess_at, a.best_guess_turns), (0.8, 3))
        self.assertEqual(self.args(NB_ENV).best_guess_at, 0.0)  # off unless set
        self.assertEqual((self.args({"ARC_BEST_GUESS_TURNS": " 2 "}).best_guess_turns,
                          self.args({"ARC_BEST_GUESS_AT": "0.75"}).best_guess_at), (2, 0.75))
        # defensive parsing (S12): empty / zero = off quietly; not a number or out of range = off + a warning
        for env, n_warn in (({"ARC_BEST_GUESS_AT": "", "ARC_BEST_GUESS_TURNS": ""}, 0),
                            ({"ARC_BEST_GUESS_AT": "0", "ARC_BEST_GUESS_TURNS": "0"}, 0),
                            ({"ARC_BEST_GUESS_AT": "high", "ARC_BEST_GUESS_TURNS": "x"}, 2),
                            ({"ARC_BEST_GUESS_AT": "1.0", "ARC_BEST_GUESS_TURNS": "40"}, 2),
                            ({"ARC_BEST_GUESS_AT": "-0.5", "ARC_BEST_GUESS_TURNS": "-3"}, 2),
                            ({"ARC_BEST_GUESS_AT": "nan", "ARC_BEST_GUESS_TURNS": "inf"}, 2)):
            s = C.load_solver(env)
            a = s.make_args()
            self.assertEqual((a.best_guess_at, a.best_guess_turns), (0.0, 0), env)
            self.assertEqual(len(s.ENV_WARNINGS), n_warn, (env, s.ENV_WARNINGS))
        # the turn bound follows ARC_MAX_TURNS
        a = self.args({"ARC_MAX_TURNS": "60", "ARC_BEST_GUESS_TURNS": "40"})
        self.assertEqual((a.max_turns, a.best_guess_turns), (60, 40))

    def test_i8_post_pass_env(self):
        # I8: ARC_POST_PASS_NOTE=1 passes --post-pass-note; anything else (unset, empty, 0, true) leaves it off
        s = C.load_solver({**NB_ENV, "ARC_POST_PASS_NOTE": "1"})
        a = s.make_args()
        self.assertTrue(s.POST_PASS_NOTE and a.post_pass_note)
        self.assertEqual((a.multi_test, a.answer_tool, s.ENV_WARNINGS), (True, True, []))
        for env in (NB_ENV, {"ARC_POST_PASS_NOTE": ""}, {"ARC_POST_PASS_NOTE": "0"}, {"ARC_POST_PASS_NOTE": "true"}):
            s = C.load_solver(env)
            self.assertFalse(s.POST_PASS_NOTE or s.make_args().post_pass_note, env)
        # with I7 on too, both reach the harness argv
        a = self.args({"ARC_POST_PASS_NOTE": "1", "ARC_BEST_GUESS_AT": "0.8"})
        self.assertEqual((a.post_pass_note, a.best_guess_at), (True, 0.8))


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
        out = self.s.split_multi(dict(rec, pred=[[[1]], None, [[3]]]), task)  # A1: a partial multi answer
        self.assertEqual([r["pred"] for r in out], [[[1]], None, [[3]]])
        out = self.s.split_multi(dict(rec, pred=None, error="TurnError: x"), task)
        self.assertTrue(all(r["error"] for r in out))
        single = {"task_id": "u", "pair": 0, "attempt": 0, "pred": [[1]]}
        self.assertEqual(self.s.split_multi(single, {"test": [{}]}), [single])
        driver = {"task_id": "t", "pair": 0, "attempt": 0, "error": "driver: X", "pred": None}  # no multi_test key
        self.assertEqual(self.s.split_multi(driver, task), [driver])

    def test_s6_provisional_records(self):
        # S6: a harvested answer counts until a later record of the same attempt brings a grid for that pair
        s = self.s
        s.RECORDS.clear()
        task = {"test": [{"input": [[1]]}] * 2}
        rec = {"task_id": "t", "pair": 0, "attempt": 0, "multi_test": 2, "pred": [[[5]], [[6]]], "error": "x"}
        s.keep(rec, task, provisional=True)
        s.keep({"task_id": "t", "pair": 0, "attempt": 0, "multi_test": 2, "pred": [None, [[8]]]}, task)
        self.assertEqual([(r["pred"], r.get("provisional")) for r in s.RECORDS[("t", 0)]], [([[5]], True), (None, None)])
        self.assertEqual([(r["pred"], r.get("provisional")) for r in s.RECORDS[("t", 1)]], [([[8]], None)])
        sub = s.build_submission({"t": {"test": [{"input": [[1]]}, {"input": [[2]]}]}})
        self.assertEqual(sub["t"], [{"attempt_1": [[5]], "attempt_2": [[1]]}, {"attempt_1": [[8]], "attempt_2": [[2]]}])
        s.RECORDS.clear()

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


class MetricsServer:
    """A local /metrics endpoint serving `body` (str)."""

    def __init__(self, body):
        self.body = body
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                b = outer.body.encode("latin-1")
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


GARBLED = "vllm:kv_cache_usage_perc{x} not_a_number\nvllm:kv_cache_usage_perc\ngarbage {{{\n"


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

    def test_s1_metrics_missing_fails_safe(self):
        # S1 (fails on V6: the gate admitted any number of attempts without a KV reading): brief gaps admit, after
        # more than 2 readings without KV usage the gate holds at 32 running, and a reading reopens it
        for v in ("down", "nokey"):
            self.s.STATE.clear()
            self.kv[0] = v
            logs = []
            with patch.object(self.s, "log", logs.append):
                for _ in range(2):
                    self.assertTrue(self.s.kv_admits(40))  # readings 1-2: admit as before
                    self.now[0] += 10
                self.assertFalse(self.s.kv_admits(40))  # reading 3: fail safe above 32 running
                self.assertTrue(self.s.kv_admits(31))
                self.assertFalse(self.s.kv_admits(32))
            self.assertEqual(len(logs), 1)
            self.assertIn("WARNING KV gate: no KV usage from /metrics in 3 readings", logs[0])
            self.now[0] += 10
            self.kv[0] = 0.2
            self.assertTrue(self.s.kv_admits(40))  # a reading again: the normal gate
            self.assertEqual(self.s.STATE["kv_missing"], 0)


class WaitingGate(unittest.TestCase):
    def test_s17_server_queue_holds_the_gate(self):
        # S17 (fails before: after preemptions freed KV, a low reading admitted attempts while the server queued requests)
        for limit, want in (("0", True), ("1", False)):
            s = C.load_solver({**NB_ENV, "ARC_ADMIT_MAX_WAITING": limit})
            now, m = [1000.0], {"kv_cache_usage_perc": 0.3, "num_requests_waiting": 12.0}
            logs = []
            with patch.object(s, "server_metrics", lambda: dict(m, t=now[0])), \
                    patch.object(s.time, "time", lambda: now[0]), patch.object(s, "log", logs.append):
                self.assertEqual(s.kv_admits(30), want)
                if limit == "1":
                    self.assertIn("KV gate: server has 12 waiting requests (limit 1), KV 0.300; holding new attempts "
                                  "at 30 running", logs[0])
                    self.assertTrue(s.kv_admits(15))  # below ARC_KV_ADMIT_MIN_WORKERS: never gated
                    now[0] += 10
                    m["num_requests_waiting"] = 0.0
                    self.assertTrue(s.kv_admits(30))  # the queue drained: the KV gate decides again


class MetricsParsing(unittest.TestCase):
    def test_s13_garbled_metrics_never_raise(self):
        # S13 (fails on V6: IndexError from server_metrics, raised through kv_admits into the main loop)
        bodies = {"valid": ('vllm:kv_cache_usage_perc{engine="0"} 0.3\nvllm:num_requests_running{engine="0"} 4.0\n',
                            {"kv_cache_usage_perc": 0.3, "num_requests_running": 4.0}),
                  "garbled": (GARBLED, {}),
                  "nan": ('vllm:kv_cache_usage_perc{engine="0"} NaN\n', {}),
                  "inf": ('vllm:kv_cache_usage_perc{engine="0"} +Inf\nvllm:num_requests_running 2\n',
                          {"num_requests_running": 2.0}),
                  "binary": ("\x00\xff\xfe vllm:\nvllm:\n", {})}
        srv = MetricsServer("")
        try:
            s = C.load_solver({**NB_ENV, "ARC_PORT": str(srv.port), "ARC_KV_ADMIT_MIN_WORKERS": "0"})
            for name, (body, want) in bodies.items():
                srv.body = body
                m = s.server_metrics()
                self.assertEqual({k: v for k, v in m.items() if k != "t"}, want, name)
                s.STATE.pop("kv_t", None)
                self.assertIn(s.kv_admits(20), (True, False))
        finally:
            srv.close()


class OffloadMetrics(unittest.TestCase):
    def test_s16_host_restored_tokens_are_not_uncached(self):
        # S16 (fails before: tokens restored from host by the KV offload connector counted as uncached)
        s = C.load_solver(NB_ENV)
        base = {"generation_tokens_total": 0.0, "prefix_cache_hits_total": 0.0, "prefix_cache_queries_total": 0.0,
                "kv_cache_usage_perc": 0.5, "num_requests_running": 4.0, "num_requests_waiting": 0.0,
                "num_preemptions_total": 0.0}
        for ext, want_uncached, want_host in ((None, 300.0, None), (0.0, 300.0, 0.0), (200.0, 100.0, 200.0)):
            prev, cur = dict(base, t=100.0), dict(base, t=110.0, generation_tokens_total=500.0,
                                                  prefix_cache_hits_total=7000.0, prefix_cache_queries_total=10000.0)
            if ext is not None:
                prev["external_prefix_cache_hits_total"], cur["external_prefix_cache_hits_total"] = 0.0, ext * 10
            s.STATE["metrics"] = prev
            with patch.object(s, "server_metrics", lambda: dict(cur)):
                tp = s.throughput()
            self.assertEqual(tp["uncached_tok_s"], want_uncached)
            self.assertEqual(tp.get("host_tok_s"), want_host)
            self.assertEqual(tp["prefix_hit"], 0.7)


class FakeProc:
    pid = 999999999

    def __init__(self):
        self.dead = False
        self.returncode = None

    def poll(self):
        if self.dead:
            self.returncode = 1
        return self.returncode


class Exit(BaseException):
    pass


def run_main(env, tasks, attempt_fn, metrics=None, patch_metrics=True, fake_server=True, kit=None, setup=None,
             path=C.SOLVER):
    """Run the live solver's main() (or the solver at `path`) with fake attempts (and by default a fake server); ->
    (solver, launches, sub, work dir, starts). `setup(solver)` runs before main (extra patches)."""
    work = Path(tempfile.mkdtemp(prefix="solver_main_", dir=C.SCRATCH))
    (work / "model").mkdir()
    (work / "model" / "config.json").write_text("{}")
    (work / "challenges.json").write_text(json.dumps(tasks))
    full_env = {**env, "ARC_CHALLENGES": str(work / "challenges.json"), "ARC_MODEL_DIR": str(work / "model"),
                "ARC_OUT": str(work / "out"), **({"ARC_KIT": str(kit)} if kit else {})}
    s = C.load_solver(full_env, path=path)
    launches, lock, starts = [], threading.Lock(), []

    def start_server():
        starts.append(s.STATE["restarts"])
        s.STATE["server"], s.STATE["server_ok"] = FakeProc(), True
        return True

    def run_attempt(args, task, t, p, a):
        with lock:
            launches.append((t, p, a))
        return attempt_fn(args, task, t, p, a)

    def fake_exit(code):
        raise Exit(code)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    patches = [patch.object(s, "run_attempt", run_attempt), patch.object(s.os, "_exit", fake_exit),
               patch.dict(os.environ, full_env)]
    if fake_server:
        patches += [patch.object(s, "start_server", start_server), patch.object(s, "stop_server", lambda: None)]
    if patch_metrics:
        patches.append(patch.object(s, "server_metrics", metrics or (lambda: None)))
    try:
        for p in patches:
            p.start()
        if setup:
            setup(s)
        try:
            s.main()
        except Exit:
            pass
    finally:
        for p in reversed(patches):
            p.stop()
        for sig, h in old.items():
            signal.signal(sig, h)
    sub = json.loads((work / "out" / "submission.json").read_text())
    return s, launches, sub, work, starts


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


def log_text(work):
    return (work / "out" / "dsv41_logs" / "solver.log").read_text()


class MainLoop(unittest.TestCase):
    def test_multi_queue_and_submission(self):
        s, launches, sub, work, _ = run_main({**NB_ENV, "ARC_WORKERS": "1"}, TASKS, harness_like)
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
        s, launches, sub, work, _ = run_main({**env, "ARC_WORKERS": "1"}, TASKS, harness_like)
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
        s, launches, sub, work, _ = run_main({**NB_ENV, "ARC_WORKERS": "1"}, TASKS, flaky)
        self.assertEqual(calls[("bbbb0002", 0)], 2)
        self.assertEqual(calls[("cccc0003", 1)], 2)
        self.assertEqual(sub["bbbb0002"][1]["attempt_1"], grid(1))
        # cccc0003 attempt 1 never answered: attempt 0's grid + the fallback (test input) for every pair
        for q in range(3):
            self.assertEqual(sub["cccc0003"][q], {"attempt_1": grid(q), "attempt_2": TASKS["cccc0003"]["test"][q]["input"]})

    def test_task_ids_and_full_submission(self):
        s, launches, sub, work, _ = run_main({**NB_ENV, "ARC_WORKERS": "2", "ARC_TASK_IDS": "cccc0003,zzzz9999"},
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
        s, launches, sub, work, _ = run_main(env, many, slow, metrics=metrics)
        timer.cancel()
        self.assertEqual(len(launches), 24)  # everything ran in the end
        self.assertLessEqual(peak["n"], 4)  # while KV read 0.7, no launch above 4 running

    def test_s13_loop_exception_does_not_end_the_run(self):
        # S13 (fails on V6: the exception reached main's `finally`, which wrote the fallback-only submission and
        # exited before the crash handler, with 0 attempts and no traceback)
        state = {"n": 0}

        def setup(s):
            real = s.pick

            def pick(queue):
                state["n"] += 1
                if state["n"] <= 2:
                    raise IndexError("list index out of range (injected)")
                return real(queue)
            patch.object(s, "pick", pick).start()
        s, launches, sub, work, _ = run_main({**NB_ENV, "ARC_WORKERS": "1"}, TASKS, harness_like, setup=setup)
        patch.stopall()
        self.assertEqual(len(launches), 6)
        self.assertEqual(sub["cccc0003"][2], {"attempt_1": grid(2), "attempt_2": grid(3)})
        text = log_text(work)
        self.assertIn("ERROR main loop iteration failed (1): IndexError: list index out of range (injected)", text)
        self.assertIn("ERROR main loop iteration failed (2)", text)

    def test_s13_garbled_metrics_end_to_end(self):
        # S13 + S1 (fails on V6: "server up" -> "submission written (final)" with 0 attempts): a real /metrics
        # endpoint serving garbled lines; the gate reads it on every launch
        srv = MetricsServer(GARBLED)
        try:
            env = {**NB_ENV, "ARC_WORKERS": "2", "ARC_KV_ADMIT_MIN_WORKERS": "0", "ARC_PORT": str(srv.port)}
            s, launches, sub, work, _ = run_main(env, TASKS, harness_like, patch_metrics=False)
        finally:
            srv.close()
        self.assertEqual(len(launches), 6)
        self.assertNotIn("ERROR main loop", log_text(work))

    def test_s6_two_crashes_restart_twice_and_keep_answers(self):
        # S6 (fails on V6: one restart per run, so the second crash ended the run with jobs unstarted; and an attempt
        # with a harvested answer was not re-queued)
        calls = {}
        holder = {}

        def crashing(args, task, t, p, a):
            calls[(t, a)] = calls.get((t, a), 0) + 1
            if (t, a) in (("aaaa0001", 0), ("bbbb0002", 0)) and calls[(t, a)] == 1:
                holder["s"].STATE["server"].dead = True  # the server dies under this attempt
                time.sleep(0.2)
                rec = {"pred": grid(7) if t == "aaaa0001" else None, "error": "TurnError: Connection refused",
                       "stop_reason": "error", "usage": {"completion": 10}, "task_id": t, "pair": p, "attempt": a,
                       "cuts": {}, "train_pass": None}
                if t == "bbbb0002":
                    rec["multi_test"] = 2
                return rec
            if (t, a) == ("aaaa0001", 0):  # the re-run ends without an answer: the harvested one stays
                return {"pred": None, "stop_reason": "output_budget", "usage": {"completion": 10}, "task_id": t,
                        "pair": p, "attempt": a, "cuts": {}, "train_pass": None}
            return harness_like(args, task, t, p, a)

        def setup(s):
            holder["s"] = s
        s, launches, sub, work, starts = run_main({**NB_ENV, "ARC_WORKERS": "1"}, TASKS, crashing, setup=setup)
        self.assertEqual(starts, [0, 1, 2])  # first start + 2 restarts
        self.assertEqual(launches.count(("aaaa0001", 0, 0)), 2)
        self.assertEqual(launches.count(("bbbb0002", 0, 0)), 2)
        self.assertEqual(len(launches), 8)
        self.assertEqual(sub["aaaa0001"][0], {"attempt_1": grid(7), "attempt_2": grid(1)})  # harvested + attempt 1
        self.assertIn("re-queue aaaa0001.0#0 after a server-side error (1): TurnError: Connection refused; its session "
                      "answer is kept", log_text(work))
        self.assertTrue(all(len(json.loads(line)) for line in
                            (work / "out" / "dsv41_logs" / "attempts.jsonl").read_text().splitlines()))

    def test_e055_engine_death_is_a_connection_error(self):
        # E055 (fails before: an engine death's APIError/500 used up a job try, so with ARC_JOB_TRIES=1 the job was lost)
        s = C.load_solver(NB_ENV)
        for err in ("TurnError: APIError: EngineCore encountered an issue. See stack trace (above) for the root cause.",
                    "TurnError: InternalServerError: Error code: 500 - {'error': {'message': 'EngineCore encountered an issue.",
                    "EngineDeadError: EngineCore encountered an issue"):
            self.assertTrue(any(c in err for c in s.CONN_ERRORS), err)
        calls, holder = {}, {}

        def dying(args, task, t, p, a):
            calls[(t, a)] = calls.get((t, a), 0) + 1
            if (t, a) == ("aaaa0001", 0) and calls[(t, a)] == 1:
                holder["s"].STATE["server"].dead = True
                time.sleep(0.2)
                return {"pred": None, "error": "TurnError: APIError: EngineCore encountered an issue. See stack trace",
                        "stop_reason": "error", "usage": {"completion": 10}, "task_id": t, "pair": p, "attempt": a,
                        "cuts": {}, "train_pass": None}
            return harness_like(args, task, t, p, a)

        def setup(s):
            holder["s"] = s
        s, launches, sub, work, starts = run_main({**NB_ENV, "ARC_WORKERS": "1", "ARC_JOB_TRIES": "1"}, TASKS, dying,
                                                  setup=setup)
        self.assertEqual(launches.count(("aaaa0001", 0, 0)), 2)  # re-queued after the death, not dropped
        self.assertIn("re-queue aaaa0001.0#0 after a server-side error", log_text(work))

    def test_s20_hung_server_is_restarted(self):
        # S20 (fails before: a server whose process stays alive but generates nothing was never restarted)
        holder, calls = {}, {}

        def hung_metrics():  # flat until the watchdog restarts the server, then generating again
            s = holder["s"]
            gen = 100.0 if s.STATE["restarts"] == 0 else 100.0 + 50 * time.time()
            return {"generation_tokens_total": gen, "num_requests_running": 3.0, "t": time.time()}

        def attempt(args, task, t, p, a):
            calls[(t, a)] = calls.get((t, a), 0) + 1
            s = holder["s"]
            if s.STATE["restarts"] == 0:  # blocked on the hung server until the watchdog restarts it
                for _ in range(600):
                    if s.STATE["restarts"] > 0:
                        break
                    time.sleep(0.1)
                return {"pred": None, "error": "TurnError: Connection refused", "stop_reason": "error",
                        "usage": {"completion": 0}, "task_id": t, "pair": p, "attempt": a, "cuts": {}}
            return harness_like(args, task, t, p, a)

        def setup(s):
            holder["s"] = s
        s, launches, sub, work, starts = run_main({**NB_ENV, "ARC_WORKERS": "2", "ARC_STALL_RESTART_S": "2"}, TASKS,
                                                  attempt, metrics=hung_metrics, setup=setup)
        self.assertEqual(starts[:2], [0, 1])
        self.assertIn("server stalled: no generated tokens for", log_text(work))
        self.assertIn("re-queue aaaa0001.0#0 after a server-side error", log_text(work))
        self.assertEqual(launches.count(("aaaa0001", 0, 0)), 2)  # re-run on the restarted server
        self.assertEqual(log_text(work).count("server stalled"), 1)

    def test_s20_off_by_default(self):
        s = C.load_solver(NB_ENV)
        self.assertEqual(s.STALL_RESTART_S, 0)

    def test_s21_conn_hold_until_server_answers(self):
        s = C.load_solver({**NB_ENV, "ARC_CONN_HOLD": "1"})
        now, alive = [1000.0], [False]
        with patch.object(s.time, "time", lambda: now[0]), patch.object(s, "server_alive", lambda: alive[0]):
            self.assertFalse(s.conn_hold())  # no server-side error yet
            s.STATE["conn_error_t"] = now[0]
            self.assertTrue(s.conn_hold())  # server not answering: hold
            alive[0] = True
            now[0] += 2
            self.assertTrue(s.conn_hold())  # re-probed at most every 5 s
            now[0] += 5
            self.assertFalse(s.conn_hold())  # answers again: launches resume
            self.assertIsNone(s.STATE["conn_error_t"])
        self.assertFalse(C.load_solver(NB_ENV).conn_hold())  # off by default

    def test_s22_restart_ramp(self):
        s = C.load_solver({**NB_ENV, "ARC_WORKERS": "56", "ARC_RESTART_RAMP": "16,8"})
        self.assertEqual(s.RESTART_RAMP, (16, 8))
        t = [100.0]
        with patch.object(s, "elapsed", lambda: t[0]):
            self.assertEqual(s.worker_cap(), 56)  # no restart yet
            s.STATE["ramp_t0"] = t[0]
            self.assertEqual(s.worker_cap(), 16)
            t[0] += 125
            self.assertEqual(s.worker_cap(), 32)
            t[0] += 300
            self.assertEqual(s.worker_cap(), 56)
            self.assertIsNone(s.STATE["ramp_t0"])
        self.assertIsNone(C.load_solver(NB_ENV).RESTART_RAMP)

    def test_s20_s21_s22_together(self):
        holder = {}

        def hung_metrics():
            s = holder["s"]
            gen = 100.0 if s.STATE["restarts"] == 0 else 100.0 + 50 * time.time()
            return {"generation_tokens_total": gen, "num_requests_running": 2.0, "t": time.time()}

        def attempt(args, task, t, p, a):
            s = holder["s"]
            if s.STATE["restarts"] == 0:
                for _ in range(600):
                    if s.STATE["restarts"] > 0:
                        break
                    time.sleep(0.1)
                return {"pred": None, "error": "TurnError: Connection refused", "stop_reason": "error",
                        "usage": {"completion": 0}, "task_id": t, "pair": p, "attempt": a, "cuts": {}}
            return harness_like(args, task, t, p, a)

        def setup(s):
            holder["s"] = s
            s.server_alive = lambda: True
        s, launches, sub, work, starts = run_main({**NB_ENV, "ARC_WORKERS": "2", "ARC_STALL_RESTART_S": "2",
                                                   "ARC_CONN_HOLD": "1", "ARC_RESTART_RAMP": "1,60"}, TASKS, attempt,
                                                  metrics=hung_metrics, setup=setup)
        self.assertEqual(starts[:2], [0, 1])
        self.assertEqual(log_text(work).count("server stalled"), 1)
        self.assertTrue(all(len(sub[t]) == len(TASKS[t]["test"]) for t in TASKS))
        self.assertIn("submission written (final)", log_text(work))

    def test_s6_restart_time_rule(self):
        s = C.load_solver({"ARC_STOP_NEW_S": "3600", "ARC_HARD_STOP_S": "7200", "ARC_RESTART_MIN_LEFT_S": "1800"})
        self.assertTrue(s.restart_allowed(0))
        self.assertFalse(s.restart_allowed(3600 - 599))  # the server could not be up before stop-new
        s.STATE["up_s"] = 1200
        self.assertFalse(s.restart_allowed(3600 - 1100))
        self.assertTrue(s.restart_allowed(3600 - 1300))
        s.STATE["restarts"] = 3
        self.assertFalse(s.restart_allowed(0))

    def test_s7_preflight_retried_then_run(self):
        # S7 (fails on V6: "fail preflight" ended the run before the server started)
        kit = Path(tempfile.mkdtemp(prefix="kit_s7_", dir=C.SCRATCH))
        (kit / "repo").symlink_to(C.KIT / "repo")
        count = kit / "count"
        (kit / "stage_model.py").write_text(
            "from pathlib import Path\n"
            f"C = Path({str(count)!r})\n"
            "def preflight(d):\n"
            "    n = int(C.read_text()) + 1 if C.exists() else 1\n"
            "    C.write_text(str(n))\n"
            "    return None if n > 2 else 'model-00001-of-00031.safetensors unreadable (OSError)'\n")
        marker = kit / "stage_done"
        marker.write_text("fail preflight: 1 of 31 files bad: model-00001-of-00031.safetensors unreadable (OSError)\n")

        def setup(s):
            s.PREFLIGHT_WAIT_S = 0.05
        s, launches, sub, work, starts = run_main({**NB_ENV, "ARC_WORKERS": "1", "ARC_STAGE_DONE": str(marker)},
                                                  TASKS, harness_like, kit=kit, setup=setup)
        self.assertEqual(count.read_text(), "3")
        self.assertEqual(starts, [0])
        self.assertEqual(len(launches), 6)
        self.assertIn("checkpoint pre-flight passed on retry", log_text(work))
        # a pre-flight that never passes: the server starts anyway once the retry time is spent
        count.write_text("-100")

        def setup2(s):
            s.PREFLIGHT_WAIT_S, s.PREFLIGHT_RETRY_S = 0.05, 0.3
        s, launches, sub, work, starts = run_main({**NB_ENV, "ARC_WORKERS": "1", "ARC_STAGE_DONE": str(marker)},
                                                  TASKS, harness_like, kit=kit, setup=setup2)
        self.assertEqual((starts, len(launches)), ([0], 6))
        self.assertIn("WARNING checkpoint pre-flight still failing", log_text(work))


FAKE_VLLM = r'''#!{python}
"""Fake `vllm serve`: start 1 leaves a child in its process group (holding "GPU memory") and exits during startup;
a later start refuses to come up while that child lives (like a vLLM start on GPUs still in use), else serves."""
import http.server, os, subprocess, sys, time
from pathlib import Path
state = Path({state!r})
port = int(sys.argv[sys.argv.index("--port") + 1])
n = len(list(state.glob("start_*")))
(state / f"start_{{n}}").write_text(str(os.getpid()))
if n == 0:
    child = subprocess.Popen(["sleep", "300"])
    (state / "child").write_text(str(child.pid))
    time.sleep(0.5)
    sys.exit(3)
pid = int((state / "child").read_text())
try:
    os.kill(pid, 0)
    alive = Path(f"/proc/{{pid}}/stat").read_text().split()[2] != "Z"
except OSError:
    alive = False
if alive:
    print("CUDA error: out of memory (GPUs held by a leftover worker)", flush=True)
    sys.exit(4)

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{{}}")
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


class ServerStart(unittest.TestCase):
    def test_s5_failed_start_is_stopped_before_the_retry(self):
        # S5 (fails on V6: the failed server's leftover child kept the GPUs, so the retry failed too: "server failed
        # twice" and fallback rows only)
        state = Path(tempfile.mkdtemp(prefix="fakevllm_", dir=C.SCRATCH))
        vllm = state / "vllm"
        vllm.write_text(FAKE_VLLM.format(python=sys.executable, state=str(state)))
        vllm.chmod(0o755)
        env = {**NB_ENV, "ARC_WORKERS": "1", "ARC_VLLM": str(vllm), "ARC_PORT": str(free_port()),
               "ARC_SCRATCH": str(state), "PATH": f"{C.fake_bin()}:{os.environ['PATH']}"}
        sleeps = []

        def setup(s):
            real_sleep = s.time.sleep
            s_mod = s

            def short_sleep(x):  # the 5 s startup / watchdog polls
                sleeps.append(x)
                real_sleep(min(x, 0.2))
            patch.object(s_mod.time, "sleep", short_sleep).start()
        try:
            s, launches, sub, work, _ = run_main(env, TASKS, harness_like, fake_server=False, setup=setup)
        finally:
            patch.stopall()
        text = log_text(work)
        self.assertIn("server exited during startup (code 3)", text)
        self.assertIn("server up after", text)
        self.assertEqual(len(launches), 6)
        child = int((state / "child").read_text())
        self.assertFalse(Path(f"/proc/{child}").exists() and
                         Path(f"/proc/{child}/stat").read_text().split()[2] != "Z")
        self.assertEqual(len(list(state.glob("start_*"))), 2)


PINNED_EXT = C.DSV41 / "pinned_candb_ext" / "solver_dsv41.py"  # B32 + S23 + S24: the manh-server notebook's


class ModelsServer:
    """S23: a stand-in for the notebook's vLLM server: /v1/models lists `ids`; close() stops it (its port then refuses
    connections) and reopen() serves again on the same port."""

    def __init__(self, ids=("dsv41",)):
        self.ids, self.port = list(ids), 0
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                b = json.dumps({"object": "list", "data": [{"id": i, "object": "model"} for i in outer.ids]}).encode()
                self.send_response(200 if self.path == "/v1/models" else 404)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        self.handler = Handler
        self.reopen()

    def reopen(self):
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), self.handler)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def short_sleeps(s):
    """run_main setup: the solver's 5 s polls take 0.05 s (undone by patch.stopall())."""
    real_sleep = s.time.sleep
    patch.object(s.time, "sleep", lambda x: real_sleep(min(x, 0.05))).start()


class ExternalServer(unittest.TestCase):
    """S23: ARC_EXTERNAL_SERVER, on the live solver and on pinned_candb_ext (B32 + S23)."""
    SOLVERS = (C.SOLVER, PINNED_EXT)

    def env(self, port, **kw):
        return {**NB_ENV, "ARC_WORKERS": "1", "ARC_EXTERNAL_SERVER": f"http://127.0.0.1:{port}/v1",
                "ARC_VLLM": str(C.SCRATCH / "no_vllm_here"), **kw}

    def test_s23_env_and_url(self):
        for path in self.SOLVERS:
            with self.subTest(path.parent.name):
                s = C.load_solver({**NB_ENV, "ARC_EXTERNAL_SERVER": " http://127.0.0.1:1234/v1/ "}, path=path)
                self.assertEqual((s.URL, s.MAX_RESTARTS), ("http://127.0.0.1:1234", 0))
                self.assertEqual(s.make_args().base_url, "http://127.0.0.1:1234/v1")
                s = C.load_solver({**NB_ENV, "ARC_EXTERNAL_SERVER": "http://10.0.0.2:30000",
                                   "ARC_MAX_SERVER_RESTARTS": "2"}, path=path)
                self.assertEqual((s.URL, s.MAX_RESTARTS), ("http://10.0.0.2:30000", 2))
                s = C.load_solver({**NB_ENV, "ARC_PORT": "8123"}, path=path)
                self.assertEqual((s.EXTERNAL, s.URL, s.MAX_RESTARTS), ("", "http://127.0.0.1:8123", 3))

    def test_s23_uses_the_running_server(self):
        # S23 (fails before: the solver always launched its own vLLM on ARC_PORT; here ARC_VLLM does not exist)
        for path in self.SOLVERS:
            with self.subTest(path.parent.name):
                srv = ModelsServer()
                slog = Path(tempfile.mkdtemp(prefix="extlog_", dir=C.SCRATCH)) / "dsv41_server.log"
                slog.write_text("INFO GPU KV cache size: 1,234,567 tokens\n")
                try:
                    s, launches, sub, work, _ = run_main(self.env(srv.port, ARC_EXTERNAL_SERVER_LOG=str(slog)), TASKS,
                                                         harness_like, fake_server=False, path=path)
                    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/v1/models", timeout=5) as r:
                        self.assertEqual(r.status, 200)  # still up: the solver never stops it
                finally:
                    srv.close()
                text = log_text(work)
                self.assertIn("(external); GPU KV cache size: 1,234,567 tokens", text)
                self.assertNotIn("server start (", text)
                self.assertFalse((work / "out" / "dsv41_logs" / "server_cmd.json").exists())
                self.assertEqual(len(launches), 6)
                for t, task in TASKS.items():
                    for q in range(len(task["test"])):
                        self.assertEqual(sub[t][q], {"attempt_1": grid(q % 10), "attempt_2": grid((q + 1) % 10)})

    def test_s23_wrong_model_or_no_server_keeps_the_fallback(self):
        for path in self.SOLVERS:
            with self.subTest(path.parent.name):
                srv = ModelsServer(ids=["qwen38"])
                try:
                    s, launches, sub, work, _ = run_main(self.env(srv.port), TASKS, harness_like, fake_server=False,
                                                         path=path)
                finally:
                    srv.close()
                text = log_text(work)
                self.assertIn("serves ['qwen38'], not dsv41", text)
                self.assertIn("server failed to start (0 retries); keeping the fallback submission", text)
                self.assertEqual((launches, sub["aaaa0001"][0]), ([], {"attempt_1": grid(1), "attempt_2": grid(1)}))
                try:
                    s, launches, sub, work, _ = run_main(self.env(free_port(), ARC_SERVER_UP_TIMEOUT_S="0.3"), TASKS,
                                                         harness_like, fake_server=False, path=path,
                                                         setup=short_sleeps)
                finally:
                    patch.stopall()
                self.assertIn("not up after", log_text(work))
                self.assertEqual(launches, [])

    def test_s23_server_death(self):
        # S23: a refused connection counts as a dead server at once (as an exited process): no restart by default;
        # with ARC_MAX_SERVER_RESTARTS=1 the solver waits for the server to answer again and re-runs the job
        for path in self.SOLVERS:
            for restarts in ("", "1"):
                with self.subTest(path.parent.name, restarts=restarts):
                    srv, died = ModelsServer(), []

                    def attempt(args, task, t, p, a):
                        if (t, a) == ("aaaa0001", 0) and not died:
                            died.append(1)
                            srv.close()
                            if restarts:
                                threading.Timer(0.5, srv.reopen).start()
                            return {"pred": grid(7), "error": "TurnError: Connection refused", "stop_reason": "error",
                                    "usage": {"completion": 10}, "task_id": t, "pair": p, "attempt": a, "cuts": {},
                                    "train_pass": None}
                        return harness_like(args, task, t, p, a)

                    def setup(s):
                        short_sleeps(s)
                        patch.object(s, "EXTERNAL_PROBE_S", 0.0).start()
                    env = self.env(srv.port, ARC_SERVER_UP_TIMEOUT_S="20",
                                   **({"ARC_MAX_SERVER_RESTARTS": restarts} if restarts else {}))
                    try:
                        s, launches, sub, work, _ = run_main(env, TASKS, attempt, fake_server=False, setup=setup,
                                                             path=path)
                    finally:
                        patch.stopall()
                        srv.close()
                    text = log_text(work)
                    self.assertIn("server died (code 1)", text)
                    if restarts:
                        self.assertIn("(wait 1): started by the notebook", text)
                        self.assertEqual(text.count("server up after"), 2)
                        self.assertEqual((len(launches), launches.count(("aaaa0001", 0, 0))), (7, 2))
                    else:
                        self.assertIn("server down and no restart left; finishing with what we have", text)
                        self.assertEqual(launches, [("aaaa0001", 0, 0)])
                        self.assertEqual(sub["aaaa0001"][0]["attempt_1"], grid(7))  # the harvested answer is kept
                        self.assertEqual(sub["cccc0003"][2], {"attempt_1": grid(7), "attempt_2": grid(7)})


def conn_error(t, p, a):
    return {"pred": grid(7), "error": "TurnError: Connection refused", "stop_reason": "error",
            "usage": {"completion": 10}, "task_id": t, "pair": p, "attempt": a, "cuts": {}, "train_pass": None}


class SupervisedServer(unittest.TestCase):
    """S24: ARC_EXTERNAL_SERVER_STATUS (the status file of Manh's --supervise), on the live solver and on
    pinned_candb_ext."""
    SOLVERS = ExternalServer.SOLVERS

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="supervised_", dir=C.SCRATCH))
        self.status = self.dir / "supervisor.json"

    def write(self, **status):  # as the supervisor does: replaced atomically
        tmp = self.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(status))
        os.replace(tmp, self.status)

    def server_log(self, n, kv):
        path = self.dir / f"server{n}.log"
        path.write_text(f"INFO GPU KV cache size: {kv} tokens\n")
        return str(path)

    def env(self, port, **kw):
        return {**NB_ENV, "ARC_WORKERS": "1", "ARC_EXTERNAL_SERVER": f"http://127.0.0.1:{port}/v1",
                "ARC_EXTERNAL_SERVER_STATUS": str(self.status), "ARC_VLLM": str(C.SCRATCH / "no_vllm_here"), **kw}

    @staticmethod
    def setup(s):
        short_sleeps(s)
        patch.object(s, "EXTERNAL_PROBE_S", 0.0).start()

    def test_s24_env_and_status(self):
        for path in self.SOLVERS:
            with self.subTest(path.parent.name):
                self.status.unlink(missing_ok=True)
                s = C.load_solver({**NB_ENV, "ARC_EXTERNAL_SERVER": "http://127.0.0.1:1234/v1",
                                   "ARC_EXTERNAL_SERVER_STATUS": str(self.status)}, path=path)
                self.assertEqual((s.EXTERNAL_STATUS, s.MAX_RESTARTS), (str(self.status), 3))
                self.assertEqual(s.external_status(), {})  # no file yet
                self.write(start=2, state="up")
                self.assertEqual(s.external_status(), {"start": 2, "state": "up"})
                self.status.write_text("{not json")
                self.assertEqual(s.external_status(), {})

    def test_s24_restart_on_the_same_port(self):
        # S24 (fails before: the port never refused, so the solver never saw the restart: no "server died", and the
        # second start's log was not the one in the status)
        for path in self.SOLVERS:
            with self.subTest(path.parent.name):
                srv, restarted = ModelsServer(), []
                self.write(start=0, state="up", pid=1, log=self.server_log(0, "1,000"), up_s=523.0)

                def attempt(args, task, t, p, a):
                    if (t, a) == ("aaaa0001", 0) and not restarted:
                        restarted.append(1)
                        self.write(start=1, state="up", pid=2, log=self.server_log(1, "2,000"), up_s=401.0)
                        return conn_error(t, p, a)
                    return harness_like(args, task, t, p, a)
                try:
                    s, launches, sub, work, _ = run_main(self.env(srv.port), TASKS, attempt, fake_server=False,
                                                         setup=self.setup, path=path)
                finally:
                    patch.stopall()
                    srv.close()
                text = log_text(work)
                self.assertIn("(external, start 0); GPU KV cache size: 1,000 tokens", text)
                self.assertIn("server died (code 1)", text)
                self.assertIn("(wait 1): started by the notebook", text)
                self.assertIn("(external, start 1); GPU KV cache size: 2,000 tokens", text)
                self.assertEqual(s.STATE["up_s"], 523.0)  # the first start's, from the status
                self.assertEqual((len(launches), launches.count(("aaaa0001", 0, 0))), (7, 2))

    def test_s24_supervisor_gave_up(self):
        # S24 (fails before: with ARC_MAX_SERVER_RESTARTS=3 the solver waited ARC_SERVER_UP_TIMEOUT_S three times for a
        # server its supervisor no longer restarts)
        for path in self.SOLVERS:
            with self.subTest(path.parent.name):
                srv, died = ModelsServer(), []
                self.write(start=0, state="up", pid=1, log=self.server_log(0, "1,000"), up_s=523.0)

                def attempt(args, task, t, p, a):
                    if (t, a) == ("aaaa0001", 0) and not died:
                        died.append(1)
                        srv.close()
                        self.write(start=0, state="failed", pid=1, up_s=523.0,
                                   error="server exited with code 1; no restart left (0)")
                        return conn_error(t, p, a)
                    return harness_like(args, task, t, p, a)
                env = self.env(srv.port, ARC_MAX_SERVER_RESTARTS="3", ARC_SERVER_UP_TIMEOUT_S="5")
                try:
                    s, launches, sub, work, _ = run_main(env, TASKS, attempt, fake_server=False, setup=self.setup,
                                                         path=path)
                finally:
                    patch.stopall()
                    srv.close()
                text = log_text(work)
                self.assertIn("its supervisor is failed (server exited with code 1; no restart left (0)); no wait left",
                              text)
                self.assertIn("server down and no restart left; finishing with what we have", text)
                self.assertNotIn("not up after", text)
                self.assertEqual(launches, [("aaaa0001", 0, 0)])
                self.assertEqual(sub["aaaa0001"][0]["attempt_1"], grid(7))  # the harvested answer is kept


SSE_DONE = b"data: [DONE]\n\n"


def sse(d):
    return b"data: " + json.dumps(d).encode() + b"\n\n"


def text_chunk(text, finish=None):
    return sse({"id": "cmpl-x", "object": "text_completion", "created": 1, "model": "dsv41",
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish,
                             "token_ids": [ord(c) % 50000 for c in text]}]})


class StallingTokenServer:
    """A token-exact vLLM stand-in: /tokenize (one id per character) and /v1/completions streaming scripted turns;
    the turn "STALL" sends one chunk and then nothing (a stalled stream)."""

    def __init__(self, turns):
        self.turns, self.stalled = list(turns), threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.endswith("/tokenize"):
                    text = "".join(m["role"] + (m.get("content") or "") + (m.get("reasoning_content") or "")
                                   + json.dumps(m.get("tool_calls") or "") for m in body["messages"])
                    data = json.dumps({"tokens": [ord(c) % 50000 for c in text + ("<g>" if body.get(
                        "add_generation_prompt") else "")]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                turn = outer.turns.pop(0)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(b):
                    self.wfile.write(hex(len(b))[2:].encode() + b"\r\n" + b + b"\r\n")
                    self.wfile.flush()
                if turn == "STALL":
                    send(text_chunk("Thinking about the rule. "))
                    outer.stalled.set()
                    time.sleep(60)
                    return
                send(text_chunk(turn))
                send(text_chunk("", "stop"))
                n = len(turn)
                send(sse({"id": "cmpl-x", "object": "text_completion", "created": 1, "model": "dsv41", "choices": [],
                          "usage": {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}}))
                send(SSE_DONE)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def ds_call(code):
    d = "｜DSML｜"
    return (f"thinking</think>\n\n<{d} calls>\n<{d} invoke name=\"run_python\">\n"
            f"<{d} parameter name=\"code\" string=\"true\">{code}</{d} parameter>\n</{d} invoke>\n"
            f"</{d} calls><｜end▁of▁sentence｜>")


class ForceSweep(unittest.TestCase):
    def test_s14_stalled_stream_ends_at_force_with_session_answer(self):
        # S14 (fails on V6: Stream.close() from the sweep did not wake the reader, so the attempt stayed blocked until
        # the 900 s read timeout or the hard stop, and its session answer was lost)
        srv = StallingTokenServer([ds_call("answer = [r[::-1] for r in test_input]"), "STALL"])
        out = {}
        try:
            s = C.load_solver({**NB_ENV, "ARC_PORT": str(srv.port)})
            s.LOG.mkdir(parents=True, exist_ok=True)
            (s.LOG / "traces").mkdir(exist_ok=True)
            args = s.make_args()
            task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4, 5]]}]}

            def run():
                out["rec"] = s.run_attempt(args, task, "t0000001", 0, 0)
            th = threading.Thread(target=run, daemon=True)
            th.start()
            self.assertTrue(srv.stalled.wait(30))
            time.sleep(0.5)
            t = time.time()
            with patch.object(s, "log", lambda m: None):
                s.force_sweep()
            th.join(20)
            self.assertFalse(th.is_alive(), "the stalled attempt did not end after the force sweep")
            self.assertLess(time.time() - t, 15)
        finally:
            srv.close()
        rec = out["rec"]
        self.assertEqual((rec["stop_reason"], rec["pred"], rec["cuts"]), ("output_budget", [[5, 4, 3]], {"deadline": 1}))
        self.assertEqual((rec["answer_source"], rec["train_trace"]), ("session", "skipped_deadline"))


class Traces(unittest.TestCase):
    def test_t3_rerun_keeps_both_traces(self):
        # T3 (fails on V6: the re-run overwrote the killed attempt's trace file)
        s = C.load_solver(NB_ENV)
        s.LOG.mkdir(parents=True, exist_ok=True)
        (s.LOG / "traces").mkdir(exist_ok=True)
        import run_deepseek_tools as h
        n = {"i": 0}

        def fake_solve(client, args, task, test_input, session_factory=None, **kw):
            n["i"] += 1
            return {"messages": [{"run": n["i"]}], "events": []}, {"pred": None, "usage": {"completion": 1}}
        task = {"train": [], "test": [{"input": [[1]]}]}
        with patch.object(h, "solve", fake_solve):
            r1 = s.run_attempt(None, task, "t", 0, 1)
            r2 = s.run_attempt(None, task, "t", 0, 1)
        self.assertEqual((r1["trace_file"], r2["trace_file"]), ("t_p0_a1.json.gz", "t_p0_a1_r1.json.gz"))
        import gzip
        runs = [json.load(gzip.open(s.LOG / "traces" / f, "rt"))["messages"][0]["run"]
                for f in ("t_p0_a1.json.gz", "t_p0_a1_r1.json.gz")]
        self.assertEqual(runs, [1, 2])


if __name__ == "__main__":
    unittest.main()
