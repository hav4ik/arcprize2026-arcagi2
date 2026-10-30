"""V12 + images recovery R1 (pinned_v12img/solver_dsv41.py): server-side failures cost no tries and no burst of
re-queues into a dead server. The solver's real main loop runs against a fake supervised server (HTTP + a
supervisor.json the test writes); only run_attempt is replaced, by one real OpenAI-client request per attempt.
- the server dies mid-request and refuses for a while, then a new start comes up;
- the engine answers 500 (/health 503), then the server dies and restarts;
- the server hangs past the client timeout (each attempt's first request);
- the normal path makes exactly V12's requests (no /health probe), with V12's solver (11bfaaaa) as the reference.
Checks: no try spent, at most one free re-queue per attempt per server start, held attempts dispatched only once the
server is up again. No GPU, no network, no Kaggle.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_v12_img_recovery.py'"""
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_v12_img as T  # noqa: E402  (kits with V12's / our files)

SCRATCH = Path(tempfile.mkdtemp(prefix="v12img_recovery_"))
DRIVER = r'''
import importlib.util, json, os, sys, time
spec = importlib.util.spec_from_file_location("solver_rec", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from openai import OpenAI
TIMEOUT = float(os.environ["TEST_CLIENT_TIMEOUT"])

def run_attempt(args, task, t, p, a):
    c = OpenAI(api_key="none", base_url=m.URL + "/v1", max_retries=0, timeout=TIMEOUT)
    t1 = time.time()
    rec = {"task_id": t, "pair": p, "attempt": a, "turns": 1, "wall_s": 0.0, "cuts": {}}
    try:
        c.chat.completions.create(model="dsv41", messages=[{"role": "user", "content": f"{t}.{p}#{a}"}], max_tokens=4)
        rec.update(pred=task["test"][p]["input"], stop_reason="submitted", usage={"completion": 4})
    except Exception as exc:  # as the harness records it
        rec.update(pred=None, stop_reason="error", usage={"completion": 0}, error=f"{type(exc).__name__}: {exc}"[:1000])
    rec["wall_s"] = round(time.time() - t1, 2)
    return rec

m.run_attempt = run_attempt
m.make_args = lambda: None
m.main()
'''


class Fake:
    """A vLLM-like server under a restart supervisor: GET /v1/models, /health, /metrics; POST /v1/chat/completions
    (answers after `delay` s). mode: ok | dying (requests in flight are dropped, new ones refused: the port closes) |
    e500 (completions 500, /health 503) | hang_first (each attempt's first completion hangs past the client timeout)."""

    def __init__(self, status_path, delay=1.0):
        self.status_path, self.delay, self.mode, self.log, self.seen = Path(status_path), delay, "ok", [], set()
        self.lock = threading.Lock()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.status(0, "up")
        self.httpd = None
        self.up()

    def status(self, start, state):
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"start": start, "state": state, "max_restarts": 3}))
        os.replace(tmp, self.status_path)

    def up(self):
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def send(self, code, body):
                b = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                with fake.lock:
                    fake.log.append((time.time(), "GET", self.path, None))
                if self.path == "/v1/models":
                    self.send(200, {"data": [{"id": "dsv41", "max_model_len": 163840}]})
                elif self.path == "/health":
                    self.send(503 if fake.mode == "e500" else 200, {})
                elif self.path == "/metrics":
                    b = b"vllm:kv_cache_usage_perc 0.1\nvllm:num_requests_waiting 0\n"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                else:
                    self.send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                key = body["messages"][0]["content"]
                with fake.lock:
                    fake.log.append((time.time(), "POST", self.path, key))
                    first = key not in fake.seen
                    fake.seen.add(key)
                if fake.mode == "e500":
                    return self.send(500, {"error": {"message": "EngineCore died", "type": "InternalServerError"}})
                if fake.mode == "hang_first" and first:
                    time.sleep(6)  # past the client timeout
                    return self.send(200, {})
                time.sleep(fake.delay)
                if fake.mode == "dying":  # the engine died under this request: the connection drops
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                self.send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": "dsv41",
                                "choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant", "content": "ok"}}],
                                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}})

        class S(http.server.ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.httpd = S(("127.0.0.1", self.port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def down(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def posts(self):
        with self.lock:
            return [x for x in self.log if x[1] == "POST"]


def run_solver(files, kit_name, fake, scenario=None, tasks=6, workers=3, timeout_s=3.0, extra_env=None):
    """The solver's main() on `tasks` one-pair tasks (1 attempt each) against `fake`; `scenario(fake)` runs in a thread
    while it works. -> (solver.log text, attempts.jsonl records, wall seconds)."""
    kit = T.kit_with(files, kit_name)
    work = Path(tempfile.mkdtemp(dir=SCRATCH))
    chal = {f"t{i:02d}": {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[i % 10]]}]}
            for i in range(tasks)}
    (work / "c.json").write_text(json.dumps(chal))
    (work / "model").mkdir()
    (work / "model/config.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_"))}
    env.update(T.V12_ENV)
    env.update(ARC_KIT=str(kit), ARC_OUT=str(work / "out"), ARC_CHALLENGES=str(work / "c.json"),
               ARC_MODEL_DIR=str(work / "model"), ARC_STAGE_DONE="", ARC_WORKERS=str(workers), ARC_ATTEMPTS="1",
               ARC_MULTI_TEST="0", ARC_EXTERNAL_SERVER=f"http://127.0.0.1:{fake.port}/v1",
               ARC_EXTERNAL_SERVER_STATUS=str(fake.status_path), TEST_CLIENT_TIMEOUT=str(timeout_s),
               ARC_HARD_STOP_S="40000", **(extra_env or {}))
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", DRIVER, str(kit / "solver_dsv41.py")], env=env, cwd=str(kit),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    th = threading.Thread(target=scenario, args=(fake,), daemon=True) if scenario else None
    if th:
        th.start()
    out, _ = p.communicate(timeout=240)
    if th:
        th.join(60)
    log = (work / "out/dsv41_logs/solver.log").read_text()
    recs = [json.loads(ln) for ln in open(work / "out/dsv41_logs/attempts.jsonl")]
    return log, recs, time.time() - t0, out


def first_post(fake, n=1, wait_s=30):
    t = time.time()
    while len(fake.posts()) < n and time.time() - t < wait_s:
        time.sleep(0.05)


class Recovery(unittest.TestCase):
    def check(self, log, recs, n=6):
        done = [r for r in recs if not r.get("error")]
        self.assertEqual(sorted(r["task_id"] for r in done), [f"t{i:02d}" for i in range(n)], log[-3000:])
        self.assertNotIn("] retry ", log)  # no try spent
        self.assertNotIn("ERROR main loop", log)
        holds = re.findall(r"hold (\S+) after a server-side error", log)
        return holds

    def test_dies_refuses_then_restarts(self):
        fake = Fake(SCRATCH / "st_die.json")
        t_up = {}

        def scenario(f):
            first_post(f, 3)
            f.mode = "dying"
            f.status(0, "up")
            time.sleep(f.delay + 0.3)  # the in-flight requests are dropped
            f.down()  # the port closes: refused
            time.sleep(0.5)
            f.status(0, "restarting")
            time.sleep(4)
            f.mode = "ok"
            f.status(1, "starting")  # the supervisor's next start, then its server answers, then "up"
            f.up()
            t_up["t"] = time.time()
            f.status(1, "up")
        log, recs, wall, out = run_solver(T.NEW_FILES, "kit_new", fake, scenario)
        holds = self.check(log, recs)
        self.assertTrue(holds, log[-3000:])
        self.assertLessEqual(max(holds.count(k) for k in set(holds)), 1)  # one free re-queue per attempt per start
        # held attempts went out again only once the new start was up
        for k in set(holds):
            again = [x for x in fake.posts() if x[3] == k.replace("#", "#")][1:]
            self.assertTrue(again and all(x[0] >= t_up["t"] for x in again), (k, again, t_up))
        self.assertIn("server up again", log)

    def test_500s_then_dies(self):
        fake = Fake(SCRATCH / "st_500.json")
        t_up = {}

        def scenario(f):
            first_post(f, 1)
            f.mode = "e500"  # the engine is dead; its API server answers 500 and /health 503
            time.sleep(4)
            f.down()
            f.status(0, "restarting")
            time.sleep(3)
            f.mode = "ok"
            f.status(1, "starting")  # the supervisor's next start, then its server answers, then "up"
            f.up()
            t_up["t"] = time.time()
            f.status(1, "up")
        log, recs, wall, out = run_solver(T.NEW_FILES, "kit_new", fake, scenario)
        holds = self.check(log, recs)
        self.assertTrue(holds)
        self.assertLessEqual(max(holds.count(k) for k in set(holds)), 1)
        e500 = [r for r in recs if "Error code: 500" in (r.get("error") or "")]
        self.assertTrue(e500)
        for r in e500:  # each attempt got at most one 500: it was held, not re-sent into the dying engine
            self.assertEqual(sum(1 for x in recs if x["task_id"] == r["task_id"] and x.get("error")), 1)
        health = [x for x in fake.log if x[2] == "/health"]
        self.assertTrue(health)  # held: probed with backoff, no faster than 1 s apart
        gaps = [b[0] - a[0] for a, b in zip(health, health[1:])]
        self.assertTrue(all(g > 0.9 for g in gaps), gaps)

    def test_hang_past_client_timeout(self):
        fake = Fake(SCRATCH / "st_hang.json", delay=0.2)
        fake.mode = "hang_first"
        log, recs, wall, out = run_solver(T.NEW_FILES, "kit_new", fake, timeout_s=2.0)
        holds = self.check(log, recs)
        self.assertEqual(sorted(holds), sorted(f"t{i:02d}.0#0" for i in range(6)))  # each exactly once
        self.assertTrue(all("APITimeoutError" in r["error"] or "Request timed out" in r["error"]
                            for r in recs if r.get("error")))

    def test_second_failure_on_the_same_start_costs_a_try(self):
        # the free re-queue is once per server start: a server that hangs every request leaves the normal tries
        fake = Fake(SCRATCH / "st_hang2.json", delay=0.2)
        fake.mode = "hang_first"
        fake.seen = set()
        orig = fake.posts

        class Always(set):
            def __contains__(self, k):
                return False  # every request is a "first": always hangs
        fake.seen = Always()
        log, recs, wall, out = run_solver(T.NEW_FILES, "kit_new", fake, tasks=2, workers=2, timeout_s=1.0)
        holds = re.findall(r"hold (\S+) after a server-side error", log)
        self.assertEqual(sorted(holds), ["t00.0#0", "t01.0#0"])
        self.assertEqual(len(re.findall(r"\] retry t0\d\.0#0", log)), 2, log[-2000:])  # then JOB_TRIES 2
        self.assertEqual(len(recs), 6)  # 1 + 1 free + 1 retry per attempt, all errors
        del orig

    def test_normal_path_same_requests_as_v12(self):
        runs = {}
        for name, files, kit in (("v12", T.V12_FILES, "kit_v12"), ("new", T.NEW_FILES, "kit_new")):
            fake = Fake(SCRATCH / f"st_ok_{name}.json", delay=0.3)
            log, recs, wall, out = run_solver(files, kit, fake, tasks=8, workers=3)
            self.assertTrue(all(not r.get("error") for r in recs))
            runs[name] = (sorted((x[1], x[2], x[3]) for x in fake.log), log, wall)
        self.assertEqual(runs["new"][0], runs["v12"][0])
        self.assertFalse([x for x in runs["new"][0] if x[1] == "/health"])
        self.assertNotIn("hold ", runs["new"][1])
        strip = lambda s: re.sub(r"^\[[^\]]*\] ", "", s, flags=re.M)  # noqa: E731
        lines = lambda s: sorted(ln for ln in strip(s).splitlines() if not ln.startswith(("[", "server up after")))  # noqa: E731
        self.assertEqual(len(lines(runs["new"][1])), len(lines(runs["v12"][1])))


if __name__ == "__main__":
    unittest.main()
