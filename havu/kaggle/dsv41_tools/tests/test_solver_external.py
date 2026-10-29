# NOT USED (E064, 2026-10-28: superseded by B32-M; kept for reference)
"""E064: the external-server mode of kaggle/dsv41_tools/pinned_manh/solver_dsv41.py (B32's pinned solver + an opt-in
ARC_EXTERNAL_SERVER for Manh's notebook, which starts vLLM itself). Without the variable the solver must behave as
B32's (pinned_candb, 77fe80a3); with it, it must never start, stop or kill a server, must follow the server's
max_model_len, and must keep writing a valid submission when the server goes away. No GPU, no network.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_solver_external.py'"""
import http.server
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

MANH = C.DSV41 / "pinned_manh/solver_dsv41.py"
B32 = C.DSV41 / "pinned_candb/solver_dsv41.py"
NB_ENV = {"ARC_WORKERS": "2", "ARC_MULTI_TEST": "1", "ARC_MULTI_BUDGET": "0.5", "ARC_CONTEXT_LIMIT": "163840",
          "ARC_KV_ADMIT_MAX": "0.85", "ARC_KV_ADMIT_MIN_WORKERS": "16", "ARC_AUTO_REPAIR": "1"}
TASKS = {"aaaa0001": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1]]}]},
         "bbbb0002": {"train": [{"input": [[1, 1]], "output": [[2, 2]]}], "test": [{"input": [[3]]}, {"input": [[4]]}]}}


class FakeVllm:
    """/v1/models (ids, max_model_len), /health and /metrics, like a vLLM OpenAI server; `up` toggles answering."""

    def __init__(self, ids=("dsv41",), max_len=128000, kv=0.3):
        self.ids, self.max_len, self.kv, self.up, self.gets = list(ids), max_len, kv, True, []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.gets.append(self.path)
                if not outer.up:
                    self.send_response(503)
                    self.end_headers()
                    return
                if self.path == "/v1/models":
                    body = json.dumps({"data": [{"id": i, "max_model_len": outer.max_len} for i in outer.ids]})
                elif self.path == "/metrics":
                    body = (f'vllm:kv_cache_usage_perc{{engine="0"}} {outer.kv}\n'
                            'vllm:num_requests_running{engine="0"} 1.0\nvllm:num_requests_waiting{engine="0"} 0.0\n')
                elif self.path == "/health":
                    body = ""
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                b = body.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class Exit(BaseException):
    pass


def answer(args, task, t, p, a):
    n = len(task["test"])
    multi = args.multi_test and n > 1
    preds = [[[(q + a) % 10]] for q in range(n)]
    rec = {"pred": preds if multi else preds[0], "stop_reason": "submitted", "usage": {"completion": 100},
           "turns": 2, "train_pass": True, "task_id": t, "pair": p, "attempt": a, "wall_s": 1.0, "cuts": {}}
    if multi:
        rec["multi_test"] = n
    return rec


def run_external(env, attempt_fn=answer, model_dir=True, setup=None):
    """main() of the pinned_manh solver with its REAL start_server / stop_server against `env`'s external server;
    any vllm Popen or killpg fails the test. -> (solver, args seen by attempts, submission, work dir, killpg calls)."""
    work = Path(tempfile.mkdtemp(prefix="solver_ext_", dir=C.SCRATCH))
    (work / "challenges.json").write_text(json.dumps(TASKS))
    full = {**env, "ARC_CHALLENGES": str(work / "challenges.json"), "ARC_OUT": str(work / "out")}
    if model_dir:
        (work / "model").mkdir()
        (work / "model" / "config.json").write_text("{}")
        full["ARC_MODEL_DIR"] = str(work / "model")
    else:
        full["ARC_MODEL_DIR"] = str(work / "missing")
    s = C.load_solver(full, path=MANH)
    seen, kills = [], []

    def run_attempt(args, task, t, p, a):
        seen.append(args)
        return attempt_fn(args, task, t, p, a)

    def no_popen(*a, **kw):
        raise AssertionError(f"external mode must not start processes: {a[0] if a else kw}")

    def fake_exit(code):
        raise Exit(code)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    patches = [patch.object(s, "run_attempt", run_attempt), patch.object(s.os, "_exit", fake_exit),
               patch.object(s.subprocess, "Popen", no_popen), patch.object(s.os, "killpg", lambda *a: kills.append(a)),
               patch.dict(os.environ, full)]
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
    return s, seen, json.loads((work / "out" / "submission.json").read_text()), work, kills


class B32Unchanged(unittest.TestCase):
    def test_without_external_server_matches_b32(self):
        b, m = C.load_solver(NB_ENV, path=B32), C.load_solver(NB_ENV, path=MANH)
        self.assertEqual(m.EXTERNAL, "")
        self.assertEqual(m.URL, b.URL)
        self.assertEqual(m.HARNESS, b.HARNESS)
        for mod in (b, m):
            mod.STATE["model"] = "/x/model"
        self.assertEqual(m.server_cmd(), b.server_cmd())
        self.assertEqual(m.make_args(), b.make_args())

    def test_harness_sends_reasoning_next_to_reasoning_content(self):
        text = (C.DSV41 / "pinned_manh/run_deepseek_tools.py").read_text()
        self.assertIn('{**m, "reasoning": m["reasoning_content"]}', text)


class ExternalMode(unittest.TestCase):
    def test_runs_against_external_server_without_starting_or_killing(self):
        srv = FakeVllm(max_len=128000)
        try:
            s, seen, sub, work, kills = run_external({**NB_ENV, "ARC_EXTERNAL_SERVER": srv.url})
        finally:
            srv.close()
        self.assertEqual(s.URL, srv.url.removesuffix("/v1"))
        self.assertEqual(kills, [])
        self.assertTrue(seen)
        self.assertTrue(all(a.context_limit == 128000 and a.model == "dsv41" for a in seen))
        self.assertTrue(all(a.base_url == srv.url for a in seen))
        self.assertEqual(sub["bbbb0002"][1]["attempt_1"], [[1]])
        log = (work / "out/dsv41_logs/solver.log").read_text()
        self.assertIn("external server up after", log)
        self.assertIn("using 128000", log)
        self.assertIn("/v1/models", srv.gets)

    def test_context_limit_kept_when_server_allows_more(self):
        srv = FakeVllm(max_len=200000)
        try:
            _, seen, _, _, _ = run_external({**NB_ENV, "ARC_EXTERNAL_SERVER": srv.url})
        finally:
            srv.close()
        self.assertTrue(all(a.context_limit == 163840 for a in seen))

    def test_served_model_mismatch_uses_server_id(self):
        srv = FakeVllm(ids=("deepseek-e128",))
        try:
            _, seen, _, work, _ = run_external({**NB_ENV, "ARC_EXTERNAL_SERVER": srv.url})
        finally:
            srv.close()
        self.assertTrue(all(a.model == "deepseek-e128" for a in seen))
        self.assertIn("not 'dsv41'", (work / "out/dsv41_logs/solver.log").read_text())

    def test_no_model_dir_still_runs(self):
        srv = FakeVllm()
        try:
            s, seen, sub, work, _ = run_external({**NB_ENV, "ARC_EXTERNAL_SERVER": srv.url}, model_dir=False)
        finally:
            srv.close()
        self.assertTrue(seen)
        self.assertIsNone(s.tokenizer())
        self.assertIn("the external server has the model", (work / "out/dsv41_logs/solver.log").read_text())

    def test_server_never_up_keeps_fallback(self):
        srv = FakeVllm()
        srv.up = False
        try:
            _, seen, sub, work, kills = run_external({**NB_ENV, "ARC_EXTERNAL_SERVER": srv.url,
                                                      "ARC_SERVER_UP_TIMEOUT_S": "3", "ARC_MAX_SERVER_RESTARTS": "0"})
        finally:
            srv.close()
        self.assertEqual(seen, [])
        self.assertEqual(kills, [])
        self.assertEqual(sub["aaaa0001"][0]["attempt_1"], [[1]])  # fallback: the test input
        self.assertIn("not answering", (work / "out/dsv41_logs/solver.log").read_text())

    def test_server_lost_mid_run_ends_with_what_we_have(self):
        srv = FakeVllm()
        first = threading.Event()

        def attempt(args, task, t, p, a):
            if t == "aaaa0001" and a == 0:
                return answer(args, task, t, p, a)
            first.set()
            srv.up = False  # the server goes away while later attempts run
            time.sleep(1.5)
            raise ConnectionError("Connection refused")
        try:
            s, _, sub, work, kills = run_external(
                {**NB_ENV, "ARC_WORKERS": "1", "ARC_EXTERNAL_SERVER": srv.url, "ARC_EXTERNAL_DOWN_S": "1",
                 "ARC_EXTERNAL_REWAIT_S": "2", "ARC_MAX_SERVER_RESTARTS": "1", "ARC_RESTART_MIN_LEFT_S": "0",
                 "ARC_RESTART_UP_S": "1", "ARC_CONN_TRIES": "1", "ARC_JOB_TRIES": "1"},
                attempt_fn=attempt, setup=lambda s: setattr(s.ExternalServer, "poll", fast_poll(s)))
        finally:
            srv.close()
        log = (work / "out/dsv41_logs/solver.log").read_text()
        self.assertTrue(first.is_set())
        self.assertEqual(kills, [])
        self.assertIn("server died", log)
        self.assertIn("reconnect 1", log)
        self.assertEqual(sub["aaaa0001"][0]["attempt_1"], [[0]])  # the answer kept from before the loss
        self.assertEqual(set(sub), set(TASKS))


def fast_poll(s):
    """ExternalServer.poll checking every call (the 15 s throttle would outlast the test)."""
    def poll(self):
        now = time.time()
        if s.server_alive():
            self.ok_t = now
        self.returncode = 1 if now - self.ok_t > s.EXTERNAL_DOWN_S else None
        return self.returncode
    return poll


if __name__ == "__main__":
    unittest.main()
