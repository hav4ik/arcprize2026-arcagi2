"""V12 + images recovery R2, turn replay (pinned_v12img: harness REPLAY hook + solver Replay).
- harness: a server dying mid-stream at turn k: the attempt continues at turn k with the same history (token-exact:
  the same prompt ids; chat path with images: the same messages), the sandbox state intact, the dropped partial turn
  not counted; the replay cap; no replay when wait_for_server() says no (the record is the no-replay one) or for a
  non-server error; the normal path makes exactly the calls of the hook-less harness, in both modes;
- solver: Replay.acquire's concurrency cap and KV gate; wait_for_server against a fake supervised server (a new start
  comes up: True; the supervisor gives up or the force deadline passes: False); ARC_REPLAY=0 turns it off.
No GPU, no network, no Kaggle.
Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_v12_img_replay.py'"""
import ast
import copy
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_v12_img as T  # noqa: E402
import test_v12_img_recovery as R  # noqa: E402

C, H = T.C, T.H_NEW
P = T.P
SCRATCH = Path(tempfile.mkdtemp(prefix="v12img_replay_"))


def solver_errors():
    """SERVER_ERRORS as the pinned solver defines them."""
    tree = ast.parse((P / "solver_dsv41.py").read_text())
    ns = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("CONN_ERRORS", "SERVER_ERRORS"):
            exec(compile(ast.Module(body=[n], type_ignores=[]), "solver", "exec"), ns)
    return ns["SERVER_ERRORS"]


class FakeReplay:
    errors = solver_errors()

    def __init__(self, wait=True, max_per_attempt=3):
        self.wait, self.max_per_attempt, self.calls, self.released = wait, max_per_attempt, [], 0

    def wait_for_server(self):
        self.calls.append("wait")
        return self.wait

    def acquire(self):
        self.calls.append("acquire")
        done = []

        def release():
            if not done:
                done.append(1)
                self.released += 1
        return release


class DyingTokenServer(C.FakeTokenServer):
    """FakeTokenServer whose requests number `die` (0-based) stream a partial completion, then drop the connection."""

    def __init__(self, turns, die=(), **kw):
        super().__init__(turns, **kw)
        self.die, self.n, self.prompts = set(die), 0, []

    @property
    def completions(self):
        server, inner = self, C.FakeTokenServer.completions.fget(self)

        class _C:
            @staticmethod
            def create(model, prompt, max_tokens, stream, stream_options, extra_body):
                i = server.n
                server.n += 1
                server.prompts.append(list(prompt))
                if i in server.die:
                    server.requests.append((len(prompt), max_tokens))
                    ch = SimpleNamespace(text="partial reasoning that is lost", finish_reason=None,
                                         model_extra={"token_ids": [1, 2, 3]}, index=0)
                    return _Dying([SimpleNamespace(choices=[ch], usage=None)])
                return inner.create(model, prompt, max_tokens, stream, stream_options, extra_body)
        return _C()


class _Dying:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        import httpx
        yield from self.chunks
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    def close(self):
        pass


class DyingChatServer(C.FakeChatServer):
    def __init__(self, turns, limit, die=(), **kw):
        super().__init__(turns, limit, **kw)
        self.die, self.n, self.sent = set(die), 0, []

    def create(self, model, messages, tools, tool_choice, max_tokens, stream, stream_options, extra_body):
        i = self.n
        self.n += 1
        self.sent.append(copy.deepcopy({"messages": messages, "max_tokens": max_tokens}))
        if i in self.die:
            d = SimpleNamespace(content="", reasoning_content="lost", tool_calls=None, model_extra={})
            return _Dying([SimpleNamespace(choices=[SimpleNamespace(delta=d, finish_reason=None)], usage=None)])
        return super().create(model, messages, tools, tool_choice, max_tokens, stream, stream_options, extra_body)


TURNS_TEXT = [C.ds_call_text("x = 41\nprint('defined')"), C.ds_call_text("print(x + 1)"),
              C.ds_call_text(T.FLIP + "answer = solve(test_input)"),
              C.ds_call_text("solve(test_input)", name="submit_answer")]
TURNS_CHAT = [("run_python", "x = 41\nprint('defined')"), ("run_python", "print(x + 1)"),
              ("run_python", T.FLIP + "answer = solve(test_input)"), ("submit_answer", "solve(test_input)")]


def run_text(replay, die=(), turns=TURNS_TEXT):
    args = H.parser().parse_args(T.solver_argv(False))
    srv = DyingTokenServer(list(turns), die=die)
    with patch.object(H, "REPLAY", replay), patch("httpx.post", srv.post):
        trace, result = H.solve(srv, args, T.TASK1, T.TASK1["test"][0]["input"], session_factory=C.local_factory)
    return srv, T.canon(trace["messages"]), T.canon(T.strip(result))


def run_chat(replay, die=(), turns=TURNS_CHAT):
    argv = T.solver_argv(True)
    args = H.parser().parse_args(argv)
    srv = DyingChatServer(list(turns), 163840, die=die)
    with patch.object(H, "REPLAY", replay):
        trace, result = H.solve(srv, args, T.TASK1, T.TASK1["test"][0]["input"], session_factory=C.local_factory)
    return srv, T.canon(trace["messages"]), T.canon(T.strip(result))


def without_replay_fields(rec):
    return {k: v for k, v in rec.items() if k not in ("replays", "replay_errors")}


class HarnessReplay(unittest.TestCase):
    def test_token_exact_dies_mid_stream_at_turn_k(self):
        ref_srv, ref_msgs, ref = run_text(None)
        for k in (1, 2):  # the request of turn k + 1 (0-based k) dies once
            rp = FakeReplay()
            srv, msgs, rec = run_text(rp, die={k})
            self.assertEqual(msgs, ref_msgs)  # the same history: nothing of the dying try is kept
            self.assertEqual(without_replay_fields(rec), ref)  # same answer, usage, turns: the partial is not counted
            self.assertEqual((rec["replays"], rp.calls, rp.released), (1, ["wait", "acquire"], 1))
            self.assertIn("RemoteProtocolError", rec["replay_errors"][0])
            self.assertEqual(srv.prompts[k], srv.prompts[k + 1])  # the stored prompt ids, sent again
            self.assertEqual(srv.prompts[:k] + srv.prompts[k + 1:], ref_srv.prompts)
        tool = [m["content"] for m in msgs if m["role"] == "tool"]
        self.assertIn("42", tool[1])  # the session kept x from turn 1 through the replay

    def test_chat_path_dies_mid_stream_at_turn_k(self):
        ref_srv, ref_msgs, ref = run_chat(None)
        rp = FakeReplay()
        srv, msgs, rec = run_chat(rp, die={1})
        self.assertEqual(msgs, ref_msgs)
        self.assertEqual(without_replay_fields(rec), ref)
        self.assertEqual(srv.sent[1], srv.sent[2])  # the same messages (with the images) and max_tokens again
        self.assertEqual(srv.sent[:1] + srv.sent[2:], ref_srv.sent)
        first_user = next(m for m in srv.sent[2]["messages"] if m["role"] == "user")
        self.assertEqual(sum(p["type"] == "image_url" for p in first_user["content"]), 5)
        self.assertEqual(rec["replays"], 1)
        self.assertIn("42", [m["content"] for m in msgs if m["role"] == "tool"][1])

    def test_replay_cap(self):
        rp = FakeReplay(max_per_attempt=3)
        srv, msgs, rec = run_text(rp, die={1, 2, 3, 4, 5})
        self.assertEqual(rec["replays"], 3)
        self.assertEqual(rec["stop_reason"], "error")
        self.assertIn("RemoteProtocolError", rec["error"])
        self.assertEqual(srv.n, 5)  # turn 1, then turn 2 tried 4 times
        self.assertEqual(rp.released, 3)

    def test_no_replay_when_the_server_is_not_coming_back(self):
        _, ref_msgs, ref = run_text(None, die={1})  # V12: the attempt ends with the error
        rp = FakeReplay(wait=False)
        srv, msgs, rec = run_text(rp, die={1})
        self.assertEqual((msgs, rec), (ref_msgs, ref))
        self.assertEqual(rp.calls, ["wait"])
        self.assertNotIn("replays", rec)

    def test_no_replay_for_other_errors(self):
        rp = FakeReplay()
        turns = [C.ds_call_text("print(1)"), ValueError("not a server error")]
        srv, msgs, rec = run_text(rp, turns=turns)
        self.assertEqual(rp.calls, [])
        _, ref_msgs, ref = run_text(None, turns=turns)
        self.assertEqual((msgs, rec), (ref_msgs, ref))

    def test_normal_path_same_calls_both_modes(self):
        for run in (run_text, run_chat):
            rp = FakeReplay()
            a, b = run(None), run(rp)
            self.assertEqual(rp.calls, [])
            self.assertEqual(a[1:], b[1:])
            if run is run_text:
                self.assertEqual((a[0].prompts, a[0].requests), (b[0].prompts, b[0].requests))
            else:
                self.assertEqual(a[0].sent, b[0].sent)

    def test_release_on_first_chunk(self):
        n = []
        s = H._ReleaseOnFirst(iter([1, 2, 3]), lambda: n.append(1))
        it = iter(s)
        self.assertEqual(next(it), 1)
        self.assertEqual(len(n), 1)  # released at the first chunk


# ---------------------------------------------------------------- solver side
def load_solver(env):
    kit = T.kit_with(T.NEW_FILES, "kit_replay")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_"))}
    with patch.dict(os.environ, {**clean, **T.V12_ENV, "ARC_KIT": str(kit), "ARC_OUT": str(SCRATCH / "out"), **env},
                    clear=True):
        spec = importlib.util.spec_from_file_location(f"solver_replay_{abs(hash(json.dumps(env, sort_keys=True)))}",
                                                      kit / "solver_dsv41.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    return m


class SolverReplay(unittest.TestCase):
    def test_concurrency_cap(self):
        m = load_solver({"ARC_KV_ADMIT_MAX": "0"})
        m.STATE["server_ok"] = True
        rp = m.Replay(concurrency=8)
        live, peak, lock = [0], [0], threading.Lock()

        def one():
            release = rp.acquire()
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.2)
            with lock:
                live[0] -= 1
            release()
            release()  # idempotent
        ths = [threading.Thread(target=one) for _ in range(30)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(30)
        self.assertEqual(peak[0], 8)
        self.assertEqual(rp.stats["replays"], 30)

    def test_kv_gate(self):
        m = load_solver({"ARC_KV_ADMIT_MAX": "0.85"})
        m.STATE["server_ok"] = True
        kv = iter([0.95, 0.9, 0.5])
        m.server_metrics = lambda: {"kv_cache_usage_perc": next(kv)}
        with patch.object(m.time, "sleep") as sl:
            release = m.Replay(concurrency=2).acquire()
        self.assertIsNotNone(release)
        self.assertEqual(sl.call_count, 2)  # waited twice for the KV gate
        release()

    def test_wait_for_server_new_start(self):
        fake = R.Fake(SCRATCH / "st_replay.json")
        m = load_solver({"ARC_EXTERNAL_SERVER": f"http://127.0.0.1:{fake.port}/v1",
                         "ARC_EXTERNAL_SERVER_STATUS": str(fake.status_path)})
        m.STATE["server_ok"] = True
        fake.mode = "e500"  # /health 503: the engine is dead
        t_up = {}

        def scenario():
            time.sleep(2)
            fake.down()
            fake.status(0, "restarting")
            m.STATE["server_ok"], m.STATE["restarting"] = False, True
            time.sleep(2)
            fake.mode = "ok"
            fake.status(1, "starting")
            fake.up()
            fake.status(1, "up")
            m.STATE["restarts"] += 1
            m.STATE["server_ok"], m.STATE["restarting"] = True, False
            t_up["t"] = time.time()
        th = threading.Thread(target=scenario)
        th.start()
        rp = m.Replay()
        self.assertTrue(rp.wait_for_server())
        self.assertGreaterEqual(time.time(), t_up["t"])
        th.join()
        health = [x for x in fake.log if x[2] == "/health"]
        self.assertTrue(health)
        self.assertTrue(all(b[0] - a[0] > 0.9 for a, b in zip(health, health[1:])))  # backoff, shared probe
        fake.status(1, "failed")
        self.assertFalse(rp.wait_for_server())  # the supervisor gave up
        fake.status(2, "up")
        m.STATE["forced"] = True
        self.assertFalse(rp.wait_for_server())  # the force deadline passed
        fake.down()

    def test_switch(self):
        for env, want in (({}, True), ({"ARC_REPLAY": "0"}, False)):
            got = T.probe(T.NEW_FILES, "kit_replay", env)
            self.assertEqual(got["replay"], want, got)


if __name__ == "__main__":
    unittest.main()
