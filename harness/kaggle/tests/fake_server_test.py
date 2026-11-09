"""Local tests of kaggle/src/solver.py's client, scheduler and submission logic against a fake vLLM server.

    .venv/bin/python kaggle/tests/fake_server_test.py

The fake server streams 'thinking' tokens; depending on the prompt hash it closes </think> and answers with the
expected output grid (or the test input), emits EOS mid-thought, or thinks until max_tokens. Scenarios:

1. normal:   fast server, small budgets -> budget / low-yield-EOS forcing, attempt 2 scheduling, scoring.
2. deadline: slow server, force deadline 6 s away -> in-flight attempts cut by the deadline and forced to answer;
             hard stop respected.
3. down:     no server at all -> every attempt errors, submission stays the valid fallback.
4. submission: build() semantics (fallbacks, distinct attempt_2, invalid grids rejected) and check_submission.
"""
import asyncio
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kaggle"))
import build_notebook  # noqa: E402

work = tempfile.mkdtemp()
ns = {"__name__": "solver_test"}
SRC_DIR = os.environ.get("ARC_TEST_SRC")  # modules extracted from the notebook (check_notebook.py), else repo
if SRC_DIR:
    sys.path.insert(0, os.path.join(SRC_DIR, "evals"))
src = (open(os.path.join(SRC_DIR, "solver.py")).read() if SRC_DIR else build_notebook.solver_source()).replace('WORK = "/kaggle/working"', f'WORK = {work!r}')
src = src.replace('PORT = 8000', 'PORT = 18123')
exec(compile(src, "solver.py", "exec"), ns)
ns["AGENT"] = "plain"  # v1 no-tool path (tools: fake_tools_test.py)

CH = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))
SOL = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
STATE = {"delay": 0.002, "requests": 0}


def expected_for(prompt):
    ti = prompt.split("--Test Input--")[-1].split("--End of Test Input--")[0].strip()
    for tid, t in CH.items():
        for p, pair in enumerate(t["test"]):
            if json.dumps(pair["input"]) == ti:
                return json.dumps(SOL[tid][p])
    return ti


async def completions(req):
    STATE["requests"] += 1
    body = await req.json()
    prompt, mt = body["prompt"], body["max_tokens"]
    assert body["stream"] and body["temperature"] == 1.0 and body["top_k"] == 20
    rnd = random.Random(hash(prompt[:3000]) % 1000)
    kind = rnd.choice(["answer", "answer", "eos", "forever"])
    forced = prompt.endswith(ns["FORCE_TEXT"])
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    await resp.prepare(req)
    ans = expected_for(prompt) if rnd.random() < 0.5 else "[[1, 2], [3, 4]]"
    finish = None
    if forced:
        pieces = [ans]
    elif kind == "answer" and mt > 50:
        pieces = ["thinking "] * min(mt - 10, 30) + ["</think>", "\n\n", ans]
    elif kind == "eos":
        pieces, finish = ["hmm "] * min(mt, 5), "stop"
    else:
        pieces = ["think "] * mt
    n = 0
    try:
        for p in pieces:
            if n >= mt:
                finish = "length"
                break
            n += 1
            d = {"choices": [{"text": p, "finish_reason": None}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": n}}
            await resp.write(f"data: {json.dumps(d)}\n\n".encode())
            await asyncio.sleep(STATE["delay"])
        else:
            finish = finish or "stop"
        d = {"choices": [{"text": "", "finish_reason": finish or "length"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": n}}
        await resp.write(f"data: {json.dumps(d)}\n\ndata: [DONE]\n\n".encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass  # client disconnected (deadline): like vLLM aborting the request
    return resp


async def tokenize(req):
    body = await req.json()
    return web.json_response({"count": len(body["prompt"]) // 3})


def summary(sch):
    recs = sch.records
    return {"records": len(recs), "answered": sum(r["answered"] for r in recs),
            "attempt2": sum(r["attempt"] == 1 for r in recs),
            "force": {str(k): sum(r["force_reason"] == k for r in recs)
                      for k in ["budget", "deadline", "low_yield_eos", None]},
            "errors": sum("error" in r for r in recs)}


async def run_sched(tasks, force_in, hard_in, conc, b1, b2min, b2max):
    sub = ns["Submission"](tasks)
    sub.write()
    ns.update(B_MAX1=b1, B_MIN1=b1, B_MIN2=b2min, B_MAX2=b2max, RATE_PRIOR=2000.0)
    client = ns["Client"]()
    now = time.time()
    sch = ns["Scheduler"](tasks, sub, client, now + force_in, now + hard_in, conc, attempts=2, tag="t")
    sch.launch_margin = 1
    await sch.run(ns["order_jobs"](tasks))
    sub.write()
    ns["check_submission"](ns["OUT_SUB"], tasks)
    return sch, json.load(open(ns["OUT_SUB"])), time.time() - now


async def main():
    os.makedirs(ns["LOG_DIR"], exist_ok=True)
    tasks = {t: CH[t] for t in sorted(CH)[:20]}

    # 4. submission semantics
    sub = ns["Submission"](tasks)
    t0, t1 = sorted(tasks)[:2]
    g = [[1, 2], [3, 4]]
    sub.answers[(t0, 0)] = {0: g, 1: g}          # duplicate -> attempt_2 = fallback
    sub.answers[(t1, 0)] = {1: [[1, 2], [3]]}    # jagged only -> both fallback
    b = sub.build()
    assert b[t0][0]["attempt_1"] == g and b[t0][0]["attempt_2"] == tasks[t0]["test"][0]["input"]
    assert b[t1][0]["attempt_1"] == tasks[t1]["test"][0]["input"] == b[t1][0]["attempt_2"]
    sub.write()
    assert ns["check_submission"](ns["OUT_SUB"], tasks)
    print("submission semantics OK")

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/tokenize", tokenize)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18123).start()

    # 1. normal
    sch, s, el = await run_sched(tasks, 25, 40, 8, 400, 10, 300)
    sm = summary(sch)
    print("normal:", sm, "score", ns["score"](s, SOL, tasks), f"{el:.1f}s")
    assert sm["errors"] == 0 and sm["attempt2"] > 0 and sm["force"]["budget"] > 0

    # 2. deadline: 50 tok/s per stream, 2000-token budgets, force after 6 s, hard stop after 15 s
    STATE["delay"] = 0.02
    sch, s, el = await run_sched(tasks, 6, 15, 8, 2000, 10, 300)
    sm = summary(sch)
    print("deadline:", sm, f"{el:.1f}s")
    assert sm["force"]["deadline"] > 0 and el < 16
    STATE["delay"] = 0.002

    await runner.cleanup()

    # 3. server down
    ns["Client"]().http  # noqa
    sch, s, el = await run_sched(tasks, 10, 15, 8, 400, 10, 300)
    sm = summary(sch)
    print("down:", sm, f"{el:.1f}s")
    assert sm["answered"] == 0 and all(s[t][p]["attempt_1"] == tasks[t]["test"][p]["input"]
                                       for t in tasks for p in range(len(tasks[t]["test"])))

    # 5. crash + restart: server absent for the first 3 s, scheduler told it is down; jobs must be requeued
    down = {"v": True}
    sub = ns["Submission"](tasks)
    ns.update(B_MAX1=400, B_MIN1=400, B_MIN2=10, B_MAX2=300)
    now = time.time()
    sch = ns["Scheduler"](tasks, sub, ns["Client"](), now + 25, now + 40, 8, attempts=1, tag="crash")
    sch.launch_margin = 1
    sch.server_down = lambda: down["v"]

    async def restart():
        await asyncio.sleep(3)
        await runner2.setup()
        await web.TCPSite(runner2, "127.0.0.1", 18123).start()
        down["v"] = False
    app2 = web.Application()
    app2.router.add_post("/v1/completions", completions)
    app2.router.add_post("/tokenize", tokenize)
    runner2 = web.AppRunner(app2)
    rt = asyncio.create_task(restart())
    await sch.run(ns["order_jobs"](tasks))
    await rt
    sm = summary(sch)
    print("crash+restart:", sm)
    assert sm["errors"] == 0 and sm["records"] == sum(len(t["test"]) for t in tasks.values())
    await runner2.cleanup()
    print("ALL OK")


asyncio.run(main())
