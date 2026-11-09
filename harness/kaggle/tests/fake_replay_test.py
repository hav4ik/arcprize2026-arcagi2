"""Offline test of kaggle/src/replay_bench.py (E014) against a fake streaming server (no GPU).

    .venv/bin/python kaggle/tests/fake_replay_test.py

The fake server implements /v1/completions (streaming, ignore_eos, exact max_tokens, continuous usage, a toy
prefix cache over earlier prompt+output strings that reports usage.prompt_tokens_details.cached_tokens), /v1/models,
/metrics (vLLM names) and /v1/loads (SGLang shape). Checks: the replay builds prompt_{t+1} = prompt_t + generated +
recorded suffix (so continuation requests hit the cache at ~100% of ideal), levels ramp, per-level summaries have
the expected fields, both server flavours of server_stats() parse, and results JSON is written.
"""
import asyncio
import gzip
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kaggle"))
import build_notebook  # noqa: E402

PORT = 18124
tmp = Path(tempfile.mkdtemp())
(tmp / "input/replay").mkdir(parents=True)
(tmp / "work").mkdir()
data = json.load(gzip.open(ROOT / "kaggle/datasets/agentic_replay/upload/agentic_replay.json.gz", "rt"))[:12]
for x in data:  # shrink: short generations so the test is fast
    for t in x["turns"]:
        t["gen"] = min(t["gen"], 40)
with open(tmp / "input/replay/agentic_replay.json", "w") as f:  # Kaggle decompresses uploaded .gz files
    json.dump(data, f)
(tmp / "solver.py").write_text(build_notebook.solver_source().replace("PORT = 8000", f"PORT = {PORT}"))
(tmp / "replay_bench.py").write_text((ROOT / "kaggle/src/replay_bench.py").read_text())

CACHE, STATE = [], {"prompt": 0, "hits": 0, "requests": 0, "running": 0}


def ntok(s):
    return max(1, len(s) // 4)


def cached_prefix(prompt):
    best = 0
    for c in CACHE:
        n = len(os.path.commonprefix([c, prompt]))
        best = max(best, n)
    return ntok(prompt[:best]) if best else 0


async def completions(req):
    body = await req.json()
    assert body["ignore_eos"] is True and body["stream"] and body["top_k"] == 20
    prompt, mt = body["prompt"], body["max_tokens"]
    pt, ct = ntok(prompt), cached_prefix(prompt)
    STATE["prompt"] += pt
    STATE["hits"] += ct
    STATE["requests"] += 1
    STATE["running"] += 1
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    await resp.prepare(req)
    out = []
    try:
        for i in range(mt):
            piece = f"w{i % 10}  "[:4]
            out.append(piece)
            u = {"prompt_tokens": pt, "completion_tokens": i + 1, "total_tokens": pt + i + 1}
            await resp.write(f"data: {json.dumps({'choices': [{'text': piece, 'index': 0}], 'usage': u})}\n\n".encode())
            await asyncio.sleep(0.001)
        u = {"prompt_tokens": pt, "completion_tokens": mt, "prompt_tokens_details": {"cached_tokens": ct}}
        await resp.write(f"data: {json.dumps({'choices': [{'text': '', 'finish_reason': 'length'}], 'usage': u})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    finally:
        STATE["running"] -= 1
        CACHE.append(prompt + "".join(out))
    return resp


async def models(req):
    return web.json_response({"data": [{"id": "qwen3.8-27b"}]})


async def metrics(req):
    txt = (f"# HELP x\nvllm:prefix_cache_queries_total{{engine=\"0\"}} {STATE['prompt']}\n"
           f"vllm:prefix_cache_hits_total{{engine=\"0\"}} {STATE['hits']}\nvllm:num_preemptions_total 0\n"
           f"vllm:kv_cache_usage_perc 0.25\nvllm:num_requests_running {STATE['running']}\nvllm:num_requests_waiting 0\n")
    return web.Response(text=txt)


async def loads(req):
    return web.json_response({"loads": [{"num_running_reqs": STATE["running"], "num_waiting_reqs": 0, "token_usage": 0.3,
                                         "num_used_tokens": 1000, "total_prefill_uncached_tokens":
                                         STATE["prompt"] - STATE["hits"], "queues": {"retracted": 0}}]})


def serve():
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/v1/loads", loads)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", PORT).start())
    loop.run_forever()


threading.Thread(target=serve, daemon=True).start()
os.environ.update(ARC_WORK=str(tmp / "work"), ARC_INPUT=str(tmp / "input"), ARC_BENCH_STOP_MIN="30",
                  ARC_REPLAY_TOOL_DELAY="0.05", ARC_SERVER="vllm")
sys.path.insert(0, str(tmp))
import replay_bench as R  # noqa: E402

S = R.S


class P:
    pid = 0

    def poll(self):
        return None


S.launch_server = lambda mdir: (P(), os.path.join(S.LOG_DIR, "fake.log"))
S.wait_ready = lambda proc, path, timeout: True
S.kill_server = lambda proc=None: None
S.server_capacity = lambda path: (100000, 64)
S.model_dir = lambda: "/nonexistent"
os.makedirs(S.LOG_DIR, exist_ok=True)
open(os.path.join(S.LOG_DIR, "fake.log"), "w").write("Engine 000: fake\n")
R.SAMPLE_S = 0.5
plan = [{"name": "fake_vllm", "env": {"ARC_SERVER_EXTRA": "[]"}, "max_seqs": 8, "levels": [[2, 5], [4, 6]]},
        {"name": "fake_inherit", "base": "fake_vllm", "extra_env": {"ARC_SGL_HICACHE": "2"}, "levels": [[2, 3]]},
        {"name": "fake_nobase", "base": "missing", "levels": [[2, 3]]}]
asyncio.run(R.main_async(plan))
res = json.load(open(R.OUT))["configs"]["fake_vllm"]
lv = res["levels"]
print(json.dumps(lv, indent=1))
assert [x["agents"] for x in lv] == [2, 4], lv
for x in lv:
    for k in ("decode_tok_s", "client_hit_rate", "cont_hit_vs_ideal", "server_hit_rate", "running_mean", "ttft_p50"):
        assert x.get(k) is not None, (k, x)
    assert x["cont_hit_vs_ideal"] > 0.95, x  # prompt_{t+1} extends prompt_t + generated text
assert res["errors"] == 0 and len(res["requests"]) > 10
allc = json.load(open(R.OUT))["configs"]
assert allc["fake_inherit"]["ok_env"] == {"ARC_SERVER_EXTRA": "[]", "ARC_SGL_HICACHE": "2"}, allc["fake_inherit"]
assert allc["fake_inherit"]["levels"][0]["agents"] == 2 and "skipped" in allc["fake_nobase"]
# SGLang flavour of server_stats / cache_line
S.SERVER = "sglang"
st = S.server_stats()
assert st and "uncached_prefill" in st, st
st["_client_prompt"] = STATE["prompt"]
print(S.cache_line(st, None))
S.SERVER = "vllm"
st2 = S.server_stats()
print(S.cache_line(st2, None))
assert "cache hit" in S.cache_line(st2, None)
# warm start: prompt equals recorded transcript prefix and context <= WARM_MAX
import random  # noqa: E402
k, pr = R.warm_start(json.load(gzip.open(ROOT / "kaggle/datasets/agentic_replay/upload/agentic_replay.json.gz", "rt"))[0],
                     random.Random(1))
print("warm start turn", k, len(pr), "chars")
print("fake_replay_test OK", flush=True)
sys.stdout.flush()
os._exit(0)
