"""Local end-to-end test of the v2 tool-agent path (solver.Scheduler.attempt_tools + E012 agent_qwen + LocalExecutor)
against a scripted fake streaming server. No GPU.

    .venv/bin/python kaggle/tests/fake_tools_test.py

The fake model picks a behaviour per task (hash of the prompt start):
- submit:   run_python(print(len(train))) -> submit_answer(<solution literal or test_input>)
- crash:    run_python(os._exit(1)) (executor restart) -> run_python(while True: pass) (timeout) -> submit_answer
- forever:  thinks until the cap -> forced submit_answer (stop=["</parameter>"]) with `test_input`
- badcall:  malformed tool call -> answers in text with a JSON grid (parse source)
Scenarios: normal, deadline (slow stream; force after a few seconds), server down. Checks submission validity and
the answer sources / counters reported per attempt.
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
sys.path.insert(0, str(ROOT / "evals"))
import build_notebook  # noqa: E402

work = tempfile.mkdtemp()
SRC_DIR = os.environ.get("ARC_TEST_SRC")  # modules extracted from the notebook (check_notebook.py), else repo
if SRC_DIR:
    sys.path.insert(0, os.path.join(SRC_DIR, "evals"))
src = (open(os.path.join(SRC_DIR, "solver.py")).read() if SRC_DIR else build_notebook.solver_source()).replace('WORK = "/kaggle/working"', f'WORK = {work!r}')
src = src.replace('PORT = 8000', 'PORT = 18124')
os.environ["ARC_TOOL_TIMEOUT"] = "2"
ns = {"__name__": "solver_test"}
exec(compile(src, "solver.py", "exec"), ns)
ns["AGENT"] = "tools"

CH = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))
SOL = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
STATE = {"delay": 0.001, "requests": 0, "stops": 0}


def task_of(prompt):
    ti = prompt.split("--Test Input--")[-1].split("--End of Test Input--")[0].strip()
    for tid, t in CH.items():
        for p, pair in enumerate(t["test"]):
            if json.dumps(pair["input"]) == ti:
                return tid, p
    return None, None


def call(name, code):
    return f"<tool_call>\n<function={name}>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"


def script(prompt, stop):
    tid, p = task_of(prompt)
    kind = random.Random(tid).choice(["submit", "submit", "crash", "forever", "badcall"])
    if stop:  # forced submit expression
        STATE["stops"] += 1
        return ["answer if 'answer' in dir() else test_input"], "stop"
    n_resp = prompt.count("<tool_response>")
    gen = prompt.split("<|im_start|>assistant\n<think>\n")[-1]
    if kind == "forever":
        return ["hmm "] * 100000, "length"
    if kind == "badcall":
        if n_resp == 0:
            return ["ok</think>\n\n", "<tool_call>\n<function=run_python>\nno params here"], "stop"
        return ["fine</think>\n\nThe answer is ", json.dumps(SOL[tid][p])], "stop"
    if kind == "crash" and n_resp == 0:
        return ["let me crash</think>\n\n", call("run_python", "import os\nos._exit(1)")], "stop"
    if kind == "crash" and n_resp == 1:
        return ["now hang</think>\n\n", call("run_python", "while True:\n    pass")], "stop"
    if n_resp == (0 if kind == "submit" else 2) and "answer =" not in gen:
        return ["look</think>\n\n", call("run_python", f"print(len(train))\nanswer = {json.dumps(SOL[tid][p])}")], "stop"
    return ["submit</think>\n\n", call("submit_answer", "answer")], "stop"


async def completions(req):
    STATE["requests"] += 1
    body = await req.json()
    prompt, mt, stop = body["prompt"], body["max_tokens"], body.get("stop")
    pieces, finish = script(prompt, stop)
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    await resp.prepare(req)
    n = 0
    try:
        for piece in pieces:
            if n >= mt:
                finish = "length"
                break
            n += 1
            d = {"choices": [{"text": piece, "finish_reason": None}],
                 "usage": {"prompt_tokens": len(prompt) // 3, "completion_tokens": n}}
            await resp.write(f"data: {json.dumps(d)}\n\n".encode())
            await asyncio.sleep(STATE["delay"])
        d = {"choices": [{"text": "", "finish_reason": finish}],
             "usage": {"prompt_tokens": len(prompt) // 3, "completion_tokens": n}}
        await resp.write(f"data: {json.dumps(d)}\n\ndata: [DONE]\n\n".encode())
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def tokenize(req):
    body = await req.json()
    return web.json_response({"count": len(body["prompt"]) // 3})


async def run(tasks, force_in, hard_in, conc, cap):
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(conc + 8))
    sub = ns["Submission"](tasks)
    sub.write()
    ns.update(B_MAX1=cap, B_MIN1=cap, B_MIN2=10 ** 9, RATE_PRIOR=5000.0, TOOLS_CAP=cap)
    now = time.time()
    sch = ns["Scheduler"](tasks, sub, ns["Client"](), now + force_in, now + hard_in, conc, attempts=1, tag="tools")
    sch.launch_margin = 1
    await sch.run(ns["order_jobs"](tasks))
    sub.write()
    ns["check_submission"](ns["OUT_SUB"], tasks)
    recs = sch.records
    s = json.load(open(ns["OUT_SUB"]))
    summ = {"attempts": len(recs), "answered": sum(r["answered"] for r in recs),
            "sources": {k: sum(r.get("answer_source") == k for r in recs)
                        for k in ("submit", "forced", "session", "parse", "lenient", None)},
            "tool_calls": sum(r.get("tool_calls") or 0 for r in recs),
            "tool_errors": sum(r.get("tool_errors") or 0 for r in recs),
            "parse_failures": sum(r.get("parse_failures") or 0 for r in recs),
            "deadline_cut": sum((r.get("deadline_cuts") or 0) > 0 for r in recs),
            "errors": [r.get("error") for r in recs if r.get("error")][:3],
            "score": ns["score"](s, SOL, tasks), "seconds": round(time.time() - now, 1)}
    return summ, recs


async def main():
    os.makedirs(ns["LOG_DIR"], exist_ok=True)
    tasks = {t: CH[t] for t in sorted(CH)[:12]}
    n_pairs = sum(len(t["test"]) for t in tasks.values())
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/tokenize", tokenize)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18124).start()

    summ, recs = await run(tasks, 60, 80, 8, 3000)
    print("tools normal:", json.dumps(summ))
    assert summ["attempts"] == n_pairs and not summ["errors"]
    assert summ["sources"]["submit"] > 0 and summ["sources"]["forced"] > 0 and summ["sources"]["parse"] > 0
    assert summ["tool_errors"] > 0 and summ["parse_failures"] > 0 and STATE["stops"] > 0
    assert summ["score"][0] > 0

    STATE["delay"] = 0.01  # 100 tok/s per stream; "forever" attempts get cut by the deadline
    summ, recs = await run(tasks, 5, 20, 8, 100000)
    print("tools deadline:", json.dumps(summ))
    assert summ["deadline_cut"] > 0 and summ["seconds"] < 21 and not summ["errors"]
    STATE["delay"] = 0.001

    await runner.cleanup()
    summ, recs = await run(tasks, 5, 8, 8, 3000)
    print("tools server down:", json.dumps(summ))
    assert summ["answered"] == 0
    print("ALL OK")


asyncio.run(main())
