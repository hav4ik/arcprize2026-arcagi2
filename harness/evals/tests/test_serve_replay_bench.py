"""evals/serve_replay_bench.py against a fake OpenAI-compatible server (http.server) that streams exactly the tokens it
is asked for, keeps a block prefix cache, and serves /metrics, /tokenize and /v1/models. Covers trace parsing, turn
ordering and prompt growth (tokens / chat / synthetic modes), fixed output lengths, concurrency, seeded order, window
aggregation, Prometheus and server-log parsing, the prep cache and tool auto-detection.

    .venv/bin/python -m unittest evals/tests/test_serve_replay_bench.py
"""
import gzip
import hashlib
import io
import json
import random
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve_replay_bench as B  # noqa: E402

BLOCK = 4


def fake_render(messages, tools=None, gen=False):
    """A prefix-stable 'chat template': one token per character."""
    s = "T" + json.dumps(tools, sort_keys=True) if tools else ""
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        s += f"<{m['role']}>" + (m.get("reasoning_content") or "") + (c or "") + json.dumps(m.get("tool_calls") or "")
        s += "</>"
    if gen:
        s += "<assistant>"
    return [ord(ch) % 900 + 50 for ch in s]


def make_trace(n_turns, seed, tools=None, task=None, finish=None, extra_events=()):
    """A harness-shaped trace whose recorded prompt / completion tokens are what fake_render gives, so a faithful
    replay reproduces the recorded prompt lengths exactly."""
    rng = random.Random(seed)
    task = task or f"task{seed}"
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": f"{task} " + "x" * rng.randint(20, 60)}]
    events = []
    for k in range(n_turns):
        prompt = len(fake_render(msgs, tools, True))
        msgs.append({"role": "assistant", "content": "", "reasoning_content": "r" * rng.randint(5, 30),
                     "tool_calls": [{"id": f"c{k}", "type": "function", "function": {
                         "name": "run_python", "arguments": json.dumps({"code": "print(1)"})}}]})
        comp = len(fake_render(msgs, tools, False)) - prompt
        events.append({"turn": k + 1, "usage": {"prompt_tokens": prompt, "completion_tokens": comp,
                                                "prompt_tokens_details": {"cached_tokens": prompt // 2}},
                       "finish_reason": (finish or {}).get(k, "tool_calls"), "max_tokens": 64000, "final": False})
        msgs.append({"role": "tool", "tool_call_id": f"c{k}", "content": "out" * rng.randint(1, 10)})
        events.append({"turn": k + 1, "tool_call_id": f"c{k}", "status": "completed"})
    events += list(extra_events)
    return {"messages": msgs, "events": events}


class Backlog(ThreadingHTTPServer):
    request_queue_size = 256  # like uvicorn's 2048: 64 simultaneous connects must not be reset


class FakeVLLM:
    def __init__(self, token_delay=0.0, return_ids=True, honor_min_tokens=True, continuous_usage=True, drop_first=0,
                 http_error=False, render=None):
        self.render = render or fake_render
        self.drop_first, self.http_error = drop_first, http_error  # close / reject generation requests
        self.token_delay, self.return_ids = token_delay, return_ids
        self.honor_min_tokens, self.continuous_usage = honor_min_tokens, continuous_usage
        self.lock = threading.Lock()
        self.requests, self.tokenize_calls = [], 0
        self.active = self.max_active = 0
        self.blocks = set()
        self.c = {"generation_tokens_total": 0, "prompt_tokens_total": 0, "prefix_cache_queries_total": 0,
                  "prefix_cache_hits_total": 0, "num_preemptions_total": 0}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json({"data": [{"id": "fake-model"}]})
                if self.path == "/version":
                    return self._json({"version": "0.0.fake"})
                if self.path == "/metrics":
                    data = fake.metrics_text().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    return self.wfile.write(data)
                self._json({"error": "not found"}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    with fake.lock:
                        fake.tokenize_calls += 1
                    ids = fake.render(body["messages"], body.get("tools"), body.get("add_generation_prompt"))
                    return self._json({"tokens": ids, "count": len(ids), "max_model_len": 100000})
                if self.path in ("/v1/completions", "/v1/chat/completions"):
                    with fake.lock:
                        drop = fake.drop_first > 0
                        fake.drop_first -= drop
                    if drop:  # the connection closes without a response
                        self.close_connection = True
                        return
                    if fake.http_error:
                        return self._json({"error": {"message": "prompt too long"}}, 400)
                    try:
                        fake.generate(self, body, chat=self.path.endswith("chat/completions"))
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self._json({"error": "not found"}, 404)

        self.httpd = Backlog(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def metrics_text(self):
        with self.lock:
            c, active = dict(self.c), self.active
        lines = ["# HELP vllm:num_requests_running Number of requests running.",
                 "# TYPE vllm:num_requests_running gauge"]
        for engine, scale in (("0", 1), ("1", 0)):  # a second, idle engine: values must be summed
            lab = f'{{engine="{engine}",model_name="fake-model"}}'
            lines.append(f"vllm:num_requests_running{lab} {float(active * scale)}")
            lines.append(f"vllm:num_requests_waiting{lab} 0.0")
            lines.append(f"vllm:kv_cache_usage_perc{lab} {0.5 * scale}")
            for k, v in c.items():
                lines.append(f"vllm:{k}{lab} {float(v * scale)}")
            lines.append(f"vllm:external_prefix_cache_hits_total{lab} 0.0")
        return "\n".join(lines) + "\n"

    def _hits(self, ids, salt):
        h, hits = salt or "", 0
        for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
            h = hashlib.sha1((h + str(ids[i:i + BLOCK])).encode()).hexdigest()
            if h not in self.blocks:
                break
            hits += BLOCK
        return hits

    def _store(self, ids, salt):
        h = salt or ""
        for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
            h = hashlib.sha1((h + str(ids[i:i + BLOCK])).encode()).hexdigest()
            self.blocks.add(h)

    def generate(self, handler, body, chat):
        prompt = self.render(body["messages"], body.get("tools"), True) if chat else body["prompt"]
        mx = body["max_tokens"]
        forced = body.get("ignore_eos") and body.get("min_tokens") == mx
        n = mx if (forced and self.honor_min_tokens) else max(1, mx // 2)
        base = sum(prompt[-4:])
        gen = [(base + 7 * i) % 900 + 50 for i in range(n)]
        salt = body.get("cache_salt")
        with self.lock:
            hits = self._hits(prompt, salt)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            rec = {"path": handler.path, "body": body, "prompt": prompt, "gen": gen, "hits": hits,
                   "t_start": time.monotonic()}
            self.requests.append(rec)
            self.c["prefix_cache_queries_total"] += len(prompt)
            self.c["prefix_cache_hits_total"] += hits
            self.c["prompt_tokens_total"] += len(prompt)
        cont = self.continuous_usage and (body.get("stream_options") or {}).get("continuous_usage_stats")

        def send(obj):
            handler.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            handler.wfile.flush()

        ok = False
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            if not chat and self.return_ids:  # vLLM echoes the prompt ids first
                send({"choices": [{"index": 0, "text": "", "prompt_token_ids": prompt, "token_ids": []}]})
            for i, tid in enumerate(gen):
                if self.token_delay:
                    time.sleep(self.token_delay)
                ch = ({"index": 0, "delta": {"content": "x"}, "finish_reason": None} if chat
                      else {"index": 0, "text": "x", "finish_reason": None})
                if not chat and self.return_ids:
                    ch["token_ids"] = [tid]
                if i == n - 1:
                    ch["finish_reason"] = "length"
                d = {"choices": [ch]}
                if cont:
                    d["usage"] = {"prompt_tokens": len(prompt), "completion_tokens": i + 1}
                send(d)
                with self.lock:
                    self.c["generation_tokens_total"] += 1
            ok = True
        finally:
            # the request leaves the "running" set (and its blocks are cached) before the client can see the end
            # of the stream and send its next turn
            with self.lock:
                self.active -= 1
                rec["t_end"] = time.monotonic()
                self._store(prompt + gen, salt)
        if ok:
            send({"choices": [], "usage": {"prompt_tokens": len(prompt), "completion_tokens": n,
                                           "prompt_tokens_details": {"cached_tokens": hits}}})
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()


def write_traces(d, traces):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for name, t in traces.items():
        with gzip.open(d / f"{name}.json.gz", "wt") as f:
            json.dump(t, f)
    return d


class ServerCase(unittest.TestCase):
    fake_kwargs = {}

    def setUp(self):
        self.fake = FakeVLLM(**self.fake_kwargs)
        self.tmp = Path(tempfile.mkdtemp(prefix="replay_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def tearDown(self):
        self.fake.close()

    def run_bench(self, traces_dir, *extra, out="out"):
        argv = ["--base-url", self.fake.url + "/v1", "--traces", str(traces_dir), "--out", str(self.tmp / out),
                "--window-s", "0.25", "--warmup-s", "0", "--quiet", "--tools", "none", *extra]
        return B.main(argv)

    def by_attempt(self, out="out"):
        """The bench's request records (one per turn) paired with what the fake server received, in order (only valid
        at --concurrency 1, where requests are sequential)."""
        recs = [json.loads(x) for x in (self.tmp / out / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), len(self.fake.requests))
        return list(zip(recs, self.fake.requests))


class TraceParsing(unittest.TestCase):
    def test_turns_follow_assistant_messages_and_usage(self):
        t = make_trace(3, 1, finish={1: "length"},
                       extra_events=[{"turn": 3, "rescued": "nudge"}])
        # a 4th assistant message without a usage event (a turn that failed mid-stream) is dropped
        t["messages"].append({"role": "assistant", "content": "partial", "reasoning_content": "..."})
        a = B.parse_trace(t, "k", "src")
        self.assertEqual(len(a.turns), 3)
        self.assertEqual(a.dropped_turns, 1)
        asst = [i for i, m in enumerate(t["messages"]) if m["role"] == "assistant"]
        self.assertEqual([x.msg_index for x in a.turns], asst[:3])
        usage = [e["usage"] for e in t["events"] if "usage" in e]
        self.assertEqual([x.completion for x in a.turns], [u["completion_tokens"] for u in usage])
        self.assertEqual([x.prompt_rec for x in a.turns], [u["prompt_tokens"] for u in usage])
        self.assertEqual([x.cached_rec for x in a.turns], [u["prompt_tokens_details"]["cached_tokens"] for u in usage])
        self.assertEqual([x.n_tools for x in a.turns], [1, 1, 1])
        self.assertEqual([x.resync for x in a.turns], [False, True, False])  # a length cut resyncs; a nudge does not

    def test_rescued_code_turn_resyncs(self):
        t = make_trace(2, 2, extra_events=[{"turn": 1, "rescued": "code"}])
        self.assertTrue(B.parse_trace(t, "k", "src").turns[0].resync)

    def test_not_a_trace(self):
        self.assertIsNone(B.parse_trace({"pred": 1}, "k", "s"))
        self.assertIsNone(B.parse_trace({"messages": [], "events": []}, "k", "s"))

    def test_load_dedups_dir_and_tarball(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        traces = {f"a{i}_p0_a0": make_trace(2, i) for i in range(3)}
        write_traces(tmp / "traces", traces)
        (tmp / "status.json").write_text(json.dumps({"phase": "done"}))
        with tarfile.open(tmp / "dsv41_traces.tar.gz", "w:gz") as tf:
            for name, t in list(traces.items())[:2] + [("b_p0_a0", make_trace(2, 9))]:
                data = gzip.compress(json.dumps(t).encode())
                info = tarfile.TarInfo(f"traces/{name}.json.gz")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        atts, stats = B.load_attempts([str(tmp)])
        self.assertEqual(len(atts), 4)
        self.assertEqual(stats["not_traces"], 1)
        # the same set from the tarball + dir in another order gives the same canonical order
        atts2, _ = B.load_attempts([str(tmp / "dsv41_traces.tar.gz"), str(tmp / "traces")])
        self.assertEqual([a.digest for a in atts], [a.digest for a in atts2])

    def test_seeded_order(self):
        atts = [B.parse_trace(make_trace(1 + i % 3, i), f"t{i}.json.gz", "s") for i in range(8)]
        o1 = [a.key for a in B.attempt_order(atts, 1)]
        self.assertEqual(o1, [a.key for a in B.attempt_order(atts, 1)])
        self.assertNotEqual(o1, [a.key for a in B.attempt_order(atts, 2)])
        self.assertNotEqual(o1, [a.key for a in B.attempt_order(atts, 1, pass_index=1)])
        heavy = B.attempt_order(atts, 1, "heavy-first")
        self.assertEqual([a.tokens for a in heavy], sorted((a.tokens for a in atts), reverse=True))


class TokensMode(ServerCase):
    fake_kwargs = {"token_delay": 0.0}

    def test_turn_order_prompt_growth_and_fixed_lengths(self):
        traces = {f"t{i}_p0_a0": make_trace(3 + i, 10 + i) for i in range(3)}
        d = write_traces(self.tmp / "traces", traces)
        s = self.run_bench(d, "--concurrency", "1", "--seed", "3")
        recorded = {f"{k}.json.gz": [e["usage"] for e in t["events"] if "usage" in e] for k, t in traces.items()}
        pairs = self.by_attempt()
        self.assertEqual(len(pairs), sum(len(v) for v in recorded.values()))
        prev = None
        for rec, req in pairs:
            body, u = req["body"], recorded[rec["attempt"]][rec["turn"] - 1]
            self.assertEqual(req["path"], "/v1/completions")
            self.assertEqual(body["max_tokens"], u["completion_tokens"])
            self.assertEqual(body["min_tokens"], u["completion_tokens"])
            self.assertTrue(body["ignore_eos"])
            self.assertEqual(body["temperature"], 0.0)
            self.assertTrue(body["return_token_ids"])
            self.assertTrue(body["cache_salt"].endswith("-p0"))
            self.assertEqual(len(body["prompt"]), u["prompt_tokens"])  # faithful prompt growth
            self.assertEqual(rec["completion_served"], u["completion_tokens"])
            if rec["turn"] > 1:  # turn k+1 = turn k's prompt + the server's generated ids + the new messages
                self.assertEqual(prev[1]["attempt"], rec["attempt"])
                self.assertEqual(prev[1]["turn"], rec["turn"] - 1)
                grown = prev[0]["prompt"] + prev[0]["gen"]
                self.assertEqual(body["prompt"][:len(grown)], grown)
                self.assertGreaterEqual(req["hits"], len(grown) - BLOCK)
            prev = (req, rec)
        # attempts ran one after another in the seeded order, each turn in order
        order = [a.key for a in B.attempt_order(B.load_attempts([str(d)])[0], 3)]
        firsts = [rec["attempt"] for rec, _ in pairs if rec["turn"] == 1]
        self.assertEqual(firsts, order)
        self.assertEqual(s["run"]["attempts_completed"], 3)
        self.assertEqual(s["fidelity"]["plan"]["prompt_len_exact_frac"], 1.0)
        self.assertEqual(s["requests_all"]["prompt_vs_recorded_exact_frac"], 1.0)
        self.assertEqual(s["requests_all"]["output_len_mismatch"], 0)
        self.assertEqual(s["requests_all"]["missing_token_ids"], 0)
        self.assertGreater(s["all"]["prefix_hit"], 0.5)
        ws = [json.loads(x) for x in (self.tmp / "out/windows.jsonl").read_text().splitlines()]
        want = sum(u["completion_tokens"] for v in recorded.values() for u in v)
        self.assertEqual(sum(w["client_gen_tokens"] for w in ws), want)  # client-side window counts
        self.assertEqual(sum(w["gen_tokens"] for w in ws), want)  # /metrics deltas
        self.assertTrue((self.tmp / "out/report.md").read_text().startswith("# Serve replay bench"))

    def test_resync_after_length_cut_sends_full_render(self):
        t = make_trace(3, 5, finish={0: "length"})
        d = write_traces(self.tmp / "traces", {"cut_p0_a0": t})
        self.run_bench(d, "--concurrency", "1")
        reqs = self.fake.requests
        asst = [i for i, m in enumerate(t["messages"]) if m["role"] == "assistant"]
        self.assertEqual(reqs[1]["body"]["prompt"], fake_render(t["messages"][:asst[1]], None, True))
        grown = reqs[1]["prompt"] + reqs[1]["gen"]  # turn 3 continues from turn 2's generated ids again
        self.assertEqual(reqs[2]["body"]["prompt"][:len(grown)], grown)

    def test_unflagged_resync_is_detected_from_recorded_lengths(self):
        # the real run re-rendered turn 2 (non-canonical turn-1 output, a parse fallback): its recorded prompt is the
        # full render, 2 tokens shorter than turn 1's prompt + its sampled ids + the tool result
        t = make_trace(3, 7)
        usage = [e["usage"] for e in t["events"] if "usage" in e]
        usage[0]["completion_tokens"] += 2  # turn 2's recorded prompt stays = len(full render)
        att = B.parse_trace(t, "k", "s")
        plan, pred = B.plan_tokens(att, lambda reqs: [fake_render(m, None, g) for m, g in reqs])
        self.assertEqual([s[0] for s in plan.segs], ["full", "delta"])
        self.assertEqual(pred, [x.prompt_rec for x in att.turns])
        att.turns[1].prompt_rec += 2  # a continued history that matches: delta as usual
        plan, _ = B.plan_tokens(att, lambda reqs: [fake_render(m, None, g) for m, g in reqs])
        self.assertEqual(plan.segs[0][0], "delta")

    def test_same_seed_same_requests(self):
        traces = {f"t{i}_p0_a0": make_trace(2, 30 + i) for i in range(5)}
        d = write_traces(self.tmp / "traces", traces)
        s1 = self.run_bench(d, "--concurrency", "1", "--seed", "7", "--cache-salt", "fixed", out="o1")
        p1 = [r["body"]["prompt"] for r in self.fake.requests]
        self.fake.requests.clear()
        s2 = self.run_bench(d, "--concurrency", "1", "--seed", "7", "--cache-salt", "fixed", out="o2")
        self.assertEqual(p1, [r["body"]["prompt"] for r in self.fake.requests])
        self.assertEqual(s1["launch_order"], s2["launch_order"])
        s3 = self.run_bench(d, "--concurrency", "1", "--seed", "8", out="o3")
        self.assertNotEqual(s1["launch_order"], s3["launch_order"])

    def test_prep_cache_skips_tokenize(self):
        d = write_traces(self.tmp / "traces", {f"t{i}_p0_a0": make_trace(3, 40 + i) for i in range(3)})
        cache = str(self.tmp / "prep.json.gz")
        self.run_bench(d, "--prep-cache", cache, out="o1")
        self.assertGreater(self.fake.tokenize_calls, 0)
        p1 = sorted(len(r["body"]["prompt"]) for r in self.fake.requests)
        self.fake.tokenize_calls = 0
        self.fake.requests.clear()
        s = self.run_bench(d, "--prep-cache", cache, out="o2")
        self.assertEqual(self.fake.tokenize_calls, 1)  # only the check of the cached plans against the server
        self.assertEqual(s["fidelity"]["server_render_check"], "match")
        self.assertEqual(p1, sorted(len(r["body"]["prompt"]) for r in self.fake.requests))

    def test_prepare_only(self):
        d = write_traces(self.tmp / "traces", {"t_p0_a0": make_trace(3, 1)})
        s = self.run_bench(d, "--prepare-only")
        self.assertEqual(self.fake.requests, [])
        self.assertEqual(s["fidelity"]["plan"]["turns"], 3)
        self.assertTrue((self.tmp / "out/prepare.json").exists())


class ToolsAuto(ServerCase):
    def test_auto_picks_the_matching_harness_tool_set(self):
        tools = B.harness_tools(["--answer-tool", "--multi-test"])
        d = write_traces(self.tmp / "traces", {f"t{i}_p0_a0": make_trace(2, 50 + i, tools=tools) for i in range(2)})
        cache = str(self.tmp / "prep.json.gz")
        s = self.run_bench(d, "--tools", "auto", "--prep-cache", cache)
        self.assertEqual(list(s["fidelity"]["tools"].values()), ["auto:--answer-tool --multi-test"])
        self.assertEqual(s["fidelity"]["plan"]["prompt_len_exact_frac"], 1.0)
        self.fake.tokenize_calls = 0
        self.run_bench(d, "--tools", "auto", "--prep-cache", cache, out="o2")
        self.assertEqual(self.fake.tokenize_calls, 1)  # detection and renders come from the cache; 1 server check


TOY_ENCODING = """import json


def encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True,
                    reasoning_effort=None, return_multi_modal_data=False):
    s = "[%s|%s|%s]" % (thinking_mode, reasoning_effort, drop_thinking)
    for m in messages:
        if m.get("tools"):
            s += "T" + json.dumps(m["tools"], sort_keys=True)
        s += "<" + m["role"] + ">" + (m.get("reasoning_content") or "") + (m.get("content") or "")
        s += json.dumps(m.get("tool_calls") or "") + "</>"
    if messages[-1]["role"] != "assistant":
        s += "<assistant>"
    return s
"""


def toy_model_dir(path):
    """A model dir for --local-render: a toy encoding.py and a char-level tokenizer.json."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    path = Path(path)
    (path / "encoding").mkdir(parents=True)
    (path / "encoding/encoding.py").write_text(TOY_ENCODING)
    vocab = {"[UNK]": 0, **{chr(c): c - 31 for c in range(32, 127)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")  # every char, newlines too
    tok.save(str(path / "tokenizer.json"))
    return path


def toy_render(messages, tools=None, gen=False):
    """What LocalRenderer gives for the toy model dir with the default template kwargs."""
    conv = json.loads(json.dumps(messages))
    if tools:
        conv[0]["tools"] = tools
    ns = {}
    exec(TOY_ENCODING, ns)
    text = ns["encode_messages"](conv, "thinking", drop_thinking=True, reasoning_effort=50)
    return [ord(c) - 31 if 32 <= ord(c) < 127 else 0 for c in text]


class LocalRender(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="replay_local_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.model = toy_model_dir(self.tmp / "model")
        tools = B.harness_tools(["--answer-tool"])
        self.traces = {}
        for i in range(3):  # recorded lengths as the toy server renders them
            t = make_trace(3, 150 + i, tools=tools)
            usage = [e["usage"] for e in t["events"] if "usage" in e]
            asst = [j for j, m in enumerate(t["messages"]) if m["role"] == "assistant"]
            for k, u in enumerate(usage):
                u["prompt_tokens"] = len(toy_render(t["messages"][:asst[k]], tools))
                u["completion_tokens"] = (len(toy_render(t["messages"][:asst[k] + 1], tools))
                                          - u["prompt_tokens"])
            self.traces[f"t{i}_p0_a0"] = t
        self.dir = write_traces(self.tmp / "traces", self.traces)

    def test_renderer_follows_the_fork_wrapper(self):
        r = B.LocalRenderer(self.model)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "assistant", "content": "", "reasoning": "why"}]
        text = r.text(msgs, [{"x": 1}], {"thinking": True, "reasoning_effort": 50})
        self.assertTrue(text.startswith('[thinking|50|True]T[{"x": 1}]<system>'))  # tools on a new system message
        self.assertIn("<user>a\n\nb", text)
        self.assertIn("<assistant>why", text)  # reasoning -> reasoning_content
        self.assertTrue(r.text(msgs, None, {}).startswith("[thinking|high|True]<user>"))
        self.assertTrue(r.text(msgs, None, {"reasoning_effort": "none"}).startswith("[chat|high|True]"))
        want = [ord(c) - 31 if 32 <= ord(c) < 127 else 0 for c in r.text(msgs, None, {})]  # "\n" is unknown: 0
        self.assertEqual(r.render_many([(msgs, True)], None, {}), [want])

    def test_offline_prep_then_run_against_the_server(self):
        cache = str(self.tmp / "prep.json.gz")
        s = B.main(["--traces", str(self.dir), "--prepare-only", "--local-render", str(self.model), "--prep-cache",
                    cache, "--out", str(self.tmp / "prep"), "--quiet"])  # no server at all
        self.assertEqual(s["fidelity"]["tools"], {str(self.dir): "auto:--answer-tool"})
        self.assertEqual(s["fidelity"]["plan"]["prompt_len_exact_frac"], 1.0)
        tools_file = self.tmp / "prep/tools.json"
        self.assertEqual(json.loads(tools_file.read_text()), B.harness_tools(["--answer-tool"]))
        fake = FakeVLLM(render=toy_render)
        try:  # the pod side: tools from the file, plans from the cache
            s = B.main(["--base-url", fake.url, "--traces", str(self.dir), "--prep-cache", cache, "--out",
                        str(self.tmp / "run"), "--window-s", "0.25", "--warmup-s", "0", "--quiet",
                        "--tools", str(tools_file)])
            self.assertEqual(fake.tokenize_calls, 1)
            self.assertEqual(s["fidelity"]["server_render_check"], "match")
            self.assertEqual(s["requests_all"]["prompt_vs_recorded_exact_frac"], 1.0)
        finally:
            fake.close()
        fake = FakeVLLM()  # a server with another template: the check says so
        try:
            s = B.main(["--base-url", fake.url, "--traces", str(self.dir), "--prep-cache", cache, "--out",
                        str(self.tmp / "run2"), "--window-s", "0.25", "--warmup-s", "0", "--quiet"])
            self.assertEqual(s["fidelity"]["server_render_check"], "mismatch")
        finally:
            fake.close()


class Concurrency(ServerCase):
    fake_kwargs = {"token_delay": 0.004}

    def test_fixed_attempts_in_flight(self):
        d = write_traces(self.tmp / "traces", {f"t{i}_p0_a0": make_trace(2, 60 + i) for i in range(8)})
        s = self.run_bench(d, "--concurrency", "3")
        self.assertEqual(self.fake.max_active, 3)
        self.assertEqual(s["run"]["attempts_completed"], 8)
        self.assertEqual(s["run"]["attempts_started"], 8)
        recs = [json.loads(x) for x in (self.tmp / "out/requests.jsonl").read_text().splitlines()]
        # per attempt, turn 2 starts only after turn 1 ended
        by = {}
        for r in recs:
            by.setdefault(r["attempt"], {})[r["turn"]] = r
        for turns in by.values():
            self.assertEqual(sorted(turns), [1, 2])
            self.assertGreaterEqual(turns[2]["t_start"], turns[1]["t_end"])
        ws = [json.loads(x) for x in (self.tmp / "out/windows.jsonl").read_text().splitlines()]
        self.assertTrue(any(w["attempts_in_flight"] == 3 for w in ws))

    def test_duration_cycles_passes_and_aborts_at_the_end(self):
        d = write_traces(self.tmp / "traces", {f"t{i}_p0_a0": make_trace(2, 70 + i) for i in range(3)})
        s = self.run_bench(d, "--concurrency", "2", "--duration", "1.5", "--warmup-s", "0.5")
        self.assertGreater(s["run"]["attempts_started"], 3)  # more than one pass
        salts = {r["body"]["cache_salt"] for r in self.fake.requests}
        self.assertGreaterEqual(len({x.rsplit("-p", 1)[1] for x in salts}), 2)  # one salt per pass
        self.assertEqual(len({x.rsplit("-p", 1)[0] for x in salts}), 1)
        ws = [json.loads(x) for x in (self.tmp / "out/windows.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(ws), 4)
        self.assertTrue(ws[0]["warmup"] and not ws[3]["warmup"])
        self.assertTrue(ws[-1]["tail"])
        self.assertTrue(s["steady"] and s["steady"]["t0"] >= 0.5)
        self.assertLess(s["run"]["elapsed_s"], 5)
        self.assertEqual(s["run"]["attempt_errors"], 0)

    def test_think_time_between_turns(self):
        d = write_traces(self.tmp / "traces", {"t_p0_a0": make_trace(2, 80)})
        self.run_bench(d, "--concurrency", "1", "--think-time", "0.3")
        r1, r2 = self.fake.requests
        self.assertGreaterEqual(r2["t_start"] - r1["t_end"], 0.28)


class Retries(ServerCase):
    fake_kwargs = {"drop_first": 2}

    def test_connect_failure_before_response_is_retried(self):
        d = write_traces(self.tmp / "traces", {"t_p0_a0": make_trace(2, 130)})
        s = self.run_bench(d, "--concurrency", "1")
        recs = [json.loads(x) for x in (self.tmp / "out/requests.jsonl").read_text().splitlines()]
        self.assertEqual([r.get("retries") for r in recs], [2, None])
        self.assertEqual((s["run"]["attempts_completed"], s["requests_all"]["connect_retries"]), (1, 2))


class HttpErrors(ServerCase):
    fake_kwargs = {"http_error": True}

    def test_http_errors_end_attempts_and_the_run(self):
        d = write_traces(self.tmp / "traces", {f"t{i}_p0_a0": make_trace(2, 140 + i) for i in range(5)})
        s = self.run_bench(d, "--concurrency", "1", "--max-errors", "1")
        self.assertEqual((s["run"]["attempts_completed"], s["run"]["attempt_errors"]), (0, 2))
        self.assertEqual(len(self.fake.requests), 0)
        recs = [json.loads(x) for x in (self.tmp / "out/requests.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), 2)  # one failed turn per attempt; no retry of an HTTP error
        self.assertTrue(all(r["error"].startswith("HTTP 400") and not r.get("retries") for r in recs))


class ChatMode(ServerCase):
    def test_chat_requests_carry_recorded_messages(self):
        tools = B.harness_tools(["--answer-tool"])
        t = make_trace(3, 90, tools=tools)
        d = write_traces(self.tmp / "traces", {"t_p0_a0": t})
        s = self.run_bench(d, "--mode", "chat", "--concurrency", "1", "--tools", "harness:--answer-tool")
        asst = [i for i, m in enumerate(t["messages"]) if m["role"] == "assistant"]
        usage = [e["usage"] for e in t["events"] if "usage" in e]
        self.assertEqual(len(self.fake.requests), 3)
        for k, req in enumerate(self.fake.requests):
            b = req["body"]
            self.assertEqual(req["path"], "/v1/chat/completions")
            self.assertEqual(b["messages"], t["messages"][:asst[k]])
            self.assertEqual(b["tools"], tools)
            self.assertEqual(b["chat_template_kwargs"], {"thinking": True, "reasoning_effort": 50})
            self.assertEqual((b["max_tokens"], b["min_tokens"]), (usage[k]["completion_tokens"],) * 2)
            self.assertNotIn("prompt", b)
        self.assertEqual(s["requests_all"]["prompt_vs_recorded_exact_frac"], 1.0)
        self.assertEqual(s["requests_all"]["output_len_mismatch"], 0)


class SyntheticMode(ServerCase):
    def test_lengths_without_tokenizer(self):
        traces = {f"t{i}_p0_a0": make_trace(3, 100 + i) for i in range(2)}
        u = json.loads(json.dumps(traces["t0_p0_a0"]))  # attempt 1 of the same task: shares turn 1 with t0
        u["messages"][-1]["content"] += " (other attempt)"
        traces["u_p0_a1"] = u
        d = write_traces(self.tmp / "traces", traces)
        s = self.run_bench(d, "--mode", "synthetic", "--concurrency", "1")
        self.assertEqual(self.fake.tokenize_calls, 0)
        self.assertEqual(s["workload"]["attempts"], 3)
        self.assertEqual(s["requests_all"]["prompt_vs_recorded_exact_frac"], 1.0)
        pairs = self.by_attempt()
        p1 = {rec["attempt"]: req["body"]["prompt"] for rec, req in pairs if rec["turn"] == 1}
        self.assertTrue(all(1000 <= x < 30000 for p in p1.values() for x in p))
        self.assertEqual(p1["t0_p0_a0.json.gz"], p1["u_p0_a1.json.gz"])
        for (r0, q0), (r1, q1) in zip(pairs, pairs[1:]):
            if r1["turn"] == r0["turn"] + 1:
                grown = q0["prompt"] + q0["gen"]
                self.assertEqual(q1["body"]["prompt"][:len(grown)], grown)


class FakeQuirks(ServerCase):
    fake_kwargs = {"return_ids": False, "honor_min_tokens": False, "continuous_usage": False}

    def test_mismatch_and_missing_ids_are_counted(self):
        d = write_traces(self.tmp / "traces", {"t_p0_a0": make_trace(3, 110)})
        s = self.run_bench(d, "--concurrency", "1")
        r = s["requests_all"]
        self.assertEqual(r["output_len_mismatch"], 3)
        self.assertEqual(r["missing_token_ids"], 3)
        self.assertEqual(s["run"]["attempts_completed"], 1)
        # without ids the client counts one token per text chunk; the totals still match what was served
        served = sum(len(q["gen"]) for q in self.fake.requests)
        ws = [json.loads(x) for x in (self.tmp / "out/windows.jsonl").read_text().splitlines()]
        self.assertEqual(sum(w["client_gen_tokens"] for w in ws), served)


class Aggregation(unittest.TestCase):
    M0 = {"generation_tokens_total": 100, "prompt_tokens_total": 1000, "prefix_cache_queries_total": 1000,
          "prefix_cache_hits_total": 500, "external_prefix_cache_hits_total": 100, "num_preemptions_total": 1,
          "external_prefix_cache_queries_total": 400, "kv_cache_usage_perc": 0.2, "kv_cache_usage_max": 0.2}
    M1 = {"generation_tokens_total": 400, "prompt_tokens_total": 3000, "prefix_cache_queries_total": 3000,
          "prefix_cache_hits_total": 2000, "external_prefix_cache_hits_total": 300, "num_preemptions_total": 3,
          "external_prefix_cache_queries_total": 900, "kv_cache_usage_perc": 0.5, "kv_cache_usage_max": 0.6,
          "num_requests_running": 10, "num_requests_waiting": 2}

    def acc(self):
        a = B.new_acc()
        a.update({"gen": 250, "prompt": 2000, "ttft": [1.0, 2.0, 3.0, 4.0], "latency": [5.0, 6.0], "turns": 2,
                  "itl_ms": [10.0, 20.0]})
        return a

    def test_window_rates(self):
        w = B.make_window(10.0, 20.0, self.M0, self.M1, self.acc(), {"mem_headroom_min_gib": 1.5},
                          {"attempts_in_flight": 4}, warmup_s=5, tail_at=None)
        self.assertEqual((w["gen_tok_s"], w["prompt_tok_s"], w["client_gen_tok_s"]), (30.0, 200.0, 25.0))
        self.assertEqual(w["prefix_hit"], 0.75)
        self.assertEqual(w["uncached_tok_s"], 30.0)  # (2000 - 1500 - 200) / 10
        self.assertEqual(w["external_hit_tok_s"], 20.0)
        self.assertEqual(w["preemptions"], 2)
        self.assertEqual((w["kv_usage"], w["kv_usage_max"], w["running"], w["waiting"]), (0.5, 0.6, 10, 2))
        self.assertEqual((w["ttft_p50"], w["latency_p90"], w["itl_ms_p50"]), (2.5, 5.9, 15.0))
        self.assertEqual((w["warmup"], w["tail"], w["attempts_in_flight"], w["mem_headroom_min_gib"]),
                         (False, False, 4, 1.5))
        self.assertTrue(B.make_window(0, 10, self.M0, self.M1, self.acc(), warmup_s=5)["warmup"])
        self.assertTrue(B.make_window(10, 20, self.M0, self.M1, self.acc(), tail_at=15)["tail"])
        # a window that ends at the --duration deadline (a tick a few ms late) is not tail
        self.assertFalse(B.make_window(10, 20.01, self.M0, self.M1, self.acc(), tail_at=20)["tail"])

    def test_counter_reset_and_missing_metrics(self):
        w = B.make_window(0, 10, self.M1, self.M0, self.acc())
        self.assertTrue(w["counter_reset"])
        self.assertNotIn("gen_tok_s", w)
        w = B.make_window(0, 10, None, self.M1, self.acc())
        self.assertNotIn("gen_tok_s", w)
        self.assertEqual(w["kv_usage"], 0.5)
        self.assertEqual(w["client_gen_tok_s"], 25.0)

    def test_summary_sums_raw_counts(self):
        m2 = {k: v * 2 for k, v in self.M1.items()}
        w1 = B.make_window(0, 10, self.M0, self.M1, self.acc())
        w2 = B.make_window(10, 30, self.M1, m2, self.acc(), {"mem_headroom_min_gib": 0.7, "mem_ooms": 1})
        s = B.summarize_windows([w1, w2])
        q = (3000 - 1000) + 3000
        h = (2000 - 500) + 2000
        self.assertEqual(s["prefix_hit"], round(h / q, 4))
        self.assertEqual(s["gen_tok_s"], round((300 + 400) / 30, 1))
        self.assertEqual(s["kv_usage_mean"], round((0.5 * 10 + 1.0 * 20) / 30, 4))
        self.assertEqual((s["mem_headroom_min_gib"], s["mem_ooms"], s["preemptions"]), (0.7, 1, 2 + 3))
        self.assertEqual(B.summarize_windows([]), {})

    def test_percentile(self):
        self.assertEqual(B.pct([3, 1, 2, None], 50), 2)
        self.assertAlmostEqual(B.pct([1, 2, 3, 4], 90), 3.7)
        self.assertIsNone(B.pct([], 50))

    def test_request_summary(self):
        reqs = [{"ttft": 1.0, "latency": 2.0, "itl_ms": 10.0, "completion_served": 5, "completion_requested": 5,
                 "prompt_served": 100, "prompt_recorded": 100, "cached_served": 50, "cached_recorded": 90},
                {"ttft": 3.0, "latency": 4.0, "itl_ms": 30.0, "completion_served": 4, "completion_requested": 5,
                 "prompt_served": 110, "prompt_recorded": 100, "cached_served": 10, "cached_recorded": 10},
                {"error": "HTTP 500", "completion_served": 0, "completion_requested": 5}]
        s = B.summarize_requests(reqs)
        self.assertEqual((s["requests"], s["ok"], s["errors"]), (3, 2, 1))
        self.assertEqual((s["ttft_p50"], s["output_len_mismatch"]), (2.0, 1))
        self.assertEqual(s["prompt_vs_recorded_exact_frac"], 0.5)
        self.assertEqual(s["recorded_prefix_hit"], 0.5)
        self.assertEqual(s["served_prefix_hit"], round(60 / 210, 4))


class Prometheus(unittest.TestCase):
    def test_sums_engines_and_aliases(self):
        text = "\n".join([
            "# HELP vllm:prefix_cache_hits_total x", "# TYPE vllm:prefix_cache_hits_total counter",
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 10.0',
            'vllm:prefix_cache_hits_total{engine="1",model_name="m"} 5.0',
            'vllm:prefix_cache_hits_created{engine="0",model_name="m"} 1.7e9',
            'vllm:gpu_prefix_cache_queries_total{engine="0"} 40',
            'vllm:gpu_cache_usage_perc{engine="0"} 0.2', 'vllm:gpu_cache_usage_perc{engine="1"} 0.6',
            'vllm:num_requests_running{engine="0"} 3', 'vllm:num_requests_running{engine="1"} 4',
            "vllm:num_preemptions_total 2", "vllm:generation_tokens_total NaN", "vllm:prompt_tokens_total garbage",
            'vllm:time_to_first_token_seconds_bucket{le="0.1"} 5', "other_metric 1"])
        m = B.parse_prometheus(text)
        self.assertEqual(m["prefix_cache_hits_total"], 15.0)
        self.assertEqual(m["prefix_cache_queries_total"], 40.0)
        self.assertAlmostEqual(m["kv_cache_usage_perc"], 0.4)
        self.assertEqual(m["kv_cache_usage_max"], 0.6)
        self.assertEqual(m["num_requests_running"], 7.0)
        self.assertEqual(m["num_preemptions_total"], 2.0)
        self.assertNotIn("generation_tokens_total", m)
        self.assertNotIn("prompt_tokens_total", m)

    def test_current_name_wins_over_deprecated(self):
        m = B.parse_prometheus("vllm:kv_cache_usage_perc 0.3\nvllm:gpu_cache_usage_perc 0.9\n")
        self.assertEqual(m["kv_cache_usage_perc"], 0.3)


E = "\x1b"
CAP = (f"{E}[0;36m(Worker_TP{{r}} pid=16530{{r}}){E}[0;0m {E}[32mINFO{E}[0m {E}[90m10-29 23:49:03{E}[0m "
       f"{E}[90m[gpu_worker.py:1171]{E}[0m Capped the PyTorch allocator at {{cap}} GiB (18.94 GiB reserved), keeping "
       "0.25 GiB free for other GPU allocations; later allocations use expandable segments.")
MEM = (f"{E}[0;36m(Worker_TP{{r}} pid=16530{{r}}){E}[0;0m {E}[32mINFO{E}[0m {E}[90m10-30 00:04:04{E}[0m "
       f"{E}[90m[gpu_worker.py:1184]{E}[0m GPU memory: {{a}} GiB allocated (peak {{p}} GiB), 19.39 GiB reserved "
       "(0.29 GiB in split segments), {f} GiB free, {n} allocator retries, {o} OOMs.")
KV = (f"{E}[0;36m(EngineCore pid=165115){E}[0;0m {E}[32mINFO{E}[0m {E}[90m10-29 23:48:34{E}[0m "
      f"{E}[90m[kv_cache_utils.py:1158]{E}[0m GPU KV cache size: 1,617,253 tokens, Maximum concurrency for 163,840 "
      "tokens per request: 9.")


def mem_line(r, a, p, f=2.0, n=0, o=0):
    return MEM.format(r=r, a=a, p=p, f=f, n=n, o=o)


class ServerLog(unittest.TestCase):
    def test_parse_lines(self):
        kind, rank, v = B.parse_log_line(mem_line(2, 18.62, 18.94, f=2.01))
        self.assertEqual((kind, rank), ("mem", "Worker_TP2"))
        self.assertEqual((v["allocated"], v["peak"], v["reserved"], v["split"], v["free"], v["retries"], v["ooms"]),
                         (18.62, 18.94, 19.39, 0.29, 2.01, 0, 0))
        kind, rank, v = B.parse_log_line(CAP.format(r=3, cap=21.15))
        self.assertEqual((kind, rank, v["cap"], v["reserved"]), ("cap", "Worker_TP3", 21.15, 18.94))
        self.assertEqual(B.parse_log_line(KV), ("kv", "EngineCore", 1617253))
        k, _, v = B.parse_log_line("GPU memory: 512 MiB allocated (peak 1024 MiB), 2 GiB reserved, 1 GiB free, "
                                   "3 allocator retries, 1 OOMs.")
        self.assertEqual((k, v["allocated"], v["peak"], v["split"], v["retries"], v["ooms"]),
                         ("mem", 0.5, 1.0, None, 3, 1))
        self.assertIsNone(B.parse_log_line("INFO something else"))

    def test_tail_headroom_over_time(self):
        path = Path(tempfile.mkdtemp()) / "server.log"
        self.addCleanup(shutil.rmtree, path.parent, True)
        with open(path, "w") as f:
            f.write(KV + "\n" + CAP.format(r=0, cap=21.2) + "\n" + CAP.format(r=1, cap=21.0) + "\n")
            f.write(mem_line(0, 18.6, 18.7) + "\n" + mem_line(1, 18.6, 18.7) + "\n")
        tail = B.LogTail(str(path))
        self.assertEqual(tail.caps, {"Worker_TP0": 21.2, "Worker_TP1": 21.0})
        self.assertEqual(tail.kv_tokens, 1617253)
        w0 = tail.window()  # startup state
        self.assertEqual((w0["mem_headroom_min_gib"], w0["mem_lines"]), (2.3, 0))
        with open(path, "a") as f:
            f.write(mem_line(0, 18.7, 19.9, f=0.5, n=2) + "\n" + mem_line(1, 18.6, 19.1) + "\n")
            f.write(mem_line(0, 18.6, 19.0, n=2) + "\n")
            f.write(mem_line(1, 18.6, 19.2)[:40])  # a partial line is kept for the next read
        w1 = tail.window()
        self.assertEqual(w1["mem_lines"], 3)
        self.assertEqual(w1["mem_peak_max_gib"], 19.9)
        self.assertEqual(w1["mem_free_min_gib"], 0.5)
        self.assertEqual(w1["mem_headroom_min_gib"], round(min(21.2 - 19.9, 21.0 - 19.1), 3))
        self.assertEqual((w1["mem_retries"], w1["mem_retries_total"]), (2, 2))
        with open(path, "a") as f:
            f.write(mem_line(1, 18.6, 19.2)[40:] + "\n")
        w2 = tail.window()
        self.assertEqual((w2["mem_lines"], w2["mem_retries"]), (1, 0))
        self.assertEqual(w2["mem_headroom_min_gib"], round(min(21.2 - 19.0, 21.0 - 19.2), 3))

    def test_bench_reports_memory_from_log(self):
        fake = FakeVLLM()
        try:
            tmp = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, tmp, True)
            log_path = tmp / "server.log"
            log_path.write_text(KV + "\n" + CAP.format(r=0, cap=21.2) + "\n" + mem_line(0, 18.6, 18.7) + "\n")
            d = write_traces(tmp / "traces", {"t_p0_a0": make_trace(2, 120)})
            s = B.main(["--base-url", fake.url, "--traces", str(d), "--out", str(tmp / "out"), "--window-s", "0.2",
                        "--warmup-s", "0", "--quiet", "--tools", "none", "--server-log", str(log_path)])
            self.assertEqual(s["server"]["kv_cache_tokens"], 1617253)
            self.assertEqual(s["all"]["mem_headroom_min_gib"], 2.5)
            self.assertIn("Allocator headroom", (tmp / "out/report.md").read_text())
        finally:
            fake.close()


if __name__ == "__main__":
    unittest.main()
