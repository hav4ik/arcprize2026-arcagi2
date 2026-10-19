"""v3 (E021) tests of kaggle/src/solver.py without a GPU: answer selection, free attempt_2, gated second attempt
(scheduling, priority, deadlines), memory caps, think cap / program mode / helpers / summary wiring, and
submission.json validity in the failure paths (emergency writer, final repair cell).

    .venv/bin/python kaggle/tests/fake_v3_test.py          # ARC_TEST_SRC=<extracted notebook dir> to test the bundle

Synthetic tasks with a known rule (flip left-right) so the host-side train checks give real outcomes. The fake
model's behaviour comes from the task id prefix (attempt index >= 1 always behaves like "good"):
- good:    defines solve = fliplr, submits solve(test_input)            -> train pass
- bad:     solve = identity plus alt = fliplr, submits solve(test_input) -> train fail, alternative alt passes
- bad2:    solve = identity only, submits solve(test_input)              -> train fail, no passing alternative
- unknown: answer = test_input, submits `answer`                        -> no train check possible
- none:    thinks until the cap; forced expression fails                 -> no answer
"""
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kaggle"))
import build_notebook  # noqa: E402

SRC_DIR = os.environ.get("ARC_TEST_SRC")
sys.path.insert(0, os.path.join(SRC_DIR, "evals") if SRC_DIR else str(ROOT / "evals"))
work = tempfile.mkdtemp()
SOLVER_SRC = open(os.path.join(SRC_DIR, "solver.py")).read() if SRC_DIR else build_notebook.solver_source()
PORT = 18125
src = SOLVER_SRC.replace('WORK = "/kaggle/working"', f'WORK = {work!r}').replace("PORT = 8000", f"PORT = {PORT}")
os.environ["ARC_TOOL_TIMEOUT"] = "5"
ns = {"__name__": "solver_v3_test"}
exec(compile(src, "solver.py", "exec"), ns)
from arc_harness import agent_qwen as AQ  # noqa: E402  (the same module objects the solver imports)
from arc_harness import selection as SEL  # noqa: E402

KINDS = ["good", "bad", "bad2", "unknown", "none"]


def make_tasks(per_kind=2, seed=0):
    rnd = random.Random(seed)
    tasks, sols = {}, {}

    def grid():
        h, w = rnd.randint(2, 4), rnd.randint(3, 5)
        return [[rnd.randint(0, 9) for _ in range(w)] for _ in range(h)]
    for k in KINDS:
        for i in range(per_kind):
            tr = [grid() for _ in range(3)]
            te = grid()
            while te == [r[::-1] for r in te]:
                te = grid()
            tid = f"{k}_{i}"
            tasks[tid] = {"train": [{"input": g, "output": [r[::-1] for r in g]} for g in tr], "test": [{"input": te}]}
            sols[tid] = [[r[::-1] for r in te]]
    return tasks, sols


TASKS, SOLS = make_tasks()
STATE = {"delay": 0.001, "starts": {}, "requests": 0}
GEN = "<|im_start|>assistant\n<think>\n"


def task_of(prompt):
    ti = prompt.split("--Test Input--")[-1].split("--End of Test Input--")[0].strip()
    for tid, t in TASKS.items():
        if json.dumps(t["test"][0]["input"]) == ti:
            return tid
    raise KeyError("unknown task")


def call(name, code):
    return f"<tool_call>\n<function={name}>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"


FLIP = "def solve(grid):\n    return [row[::-1] for row in grid]"
IDENT = "def solve(grid):\n    return [list(row) for row in grid]"


def script(prompt, stop):
    tid = task_of(prompt)
    program = '"name": "submit_program"' in prompt
    gen = prompt.split(GEN)[-1]
    n_resp = prompt.count("<tool_response>")
    if n_resp == 0 and gen == "" and stop is None:
        STATE["starts"][tid] = STATE["starts"].get(tid, 0) + 1
    kind = tid.split("_")[0] if STATE["starts"].get(tid, 1) == 1 else "good"
    if stop:  # forced submit_answer expression
        return ["nothing_here" if kind == "none" else "solve(test_input)"], "stop"
    if prompt.endswith("<parameter=code>\n"):  # think cap closed the thinking into a run_python call
        return ["print('acting')\n</parameter>\n</function>\n</tool_call>"], "stop"
    if kind == "none":
        return ["hmm "] * 100000, "length"
    if program:
        code = FLIP if kind == "good" else IDENT
        if "d8(" in prompt.split("<|im_start|>user")[0] and kind == "good":  # helpers in the program namespace
            code = "def solve(grid):\n    return d8(grid)['fliplr'].tolist()"
        if n_resp == 0:
            return ["ok</think>\n\n", call("submit_program", code)], "stop"
        return ["hmm "] * 100000, "length"
    first = {"good": FLIP, "bad": IDENT + "\ndef alt(grid):\n    return [row[::-1] for row in grid]",
             "bad2": IDENT, "unknown": "answer = test_input"}[kind]
    if n_resp == 0:
        return ["look</think>\n\n", call("run_python", first + "\nprint(len(train))")], "stop"
    expr = "answer" if kind == "unknown" else "solve(test_input)"
    return ["submit</think>\n\n", call("submit_answer", expr)], "stop"


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


def settings(**kw):
    base = dict(AGENT="tools", ATTEMPT2="legacy", FREE_ATTEMPT2=0, ATTEMPT2_UNKNOWN=1, TOOLS_ANSWER_MODE="submit",
                TOOLS_PROMPT="", HELPERS=0, TASK_SUMMARY=0, TURN_THINK_CAP=0, TURN_THINK_GRACE=2048,
                FEEDBACK="basic", SESSION_MEM_MB=2048, TOOL_TIMEOUT=5.0, RATE_PRIOR=5000.0)
    base.update(kw)
    ns.update(base)


async def run(tasks, force_in, hard_in, conc, cap, b_min2=1000, tag="v3"):
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(conc + 8))
    STATE["starts"] = {}
    sub = ns["Submission"](tasks)
    sub.write()
    ns.update(B_MAX1=cap, B_MIN1=cap, B_MIN2=b_min2, B_MAX2=cap, TOOLS_CAP=cap)
    now = time.time()
    sch = ns["Scheduler"](tasks, sub, ns["Client"](), now + force_in, now + hard_in, conc, attempts=2, tag=tag)
    sch.launch_margin = 1
    starts = []
    orig = sch.attempt

    async def traced(job, budget):
        starts.append((time.time(), job, budget))
        return await orig(job, budget)
    sch.attempt = traced
    await sch.run(ns["order_jobs"](tasks))
    sub.write()
    ns["check_submission"](ns["OUT_SUB"], tasks)
    return sch, sub, json.load(open(ns["OUT_SUB"])), starts, time.time() - now


def correct(final, tid, slot):
    return final[tid][0][slot] == SOLS[tid][0]


# ---------------------------------------------------------------- 0. never kill by name off Kaggle (E021 incident)

def test_no_name_kill_off_kaggle():
    """Static: every pkill in the solver sits under `if ON_KAGGLE`, and the notebook guard cell's under an isdir
    check of /kaggle/input. Dynamic (in a child process, so nothing of this test is at risk): kill_children() and
    kill_server() issue no pkill off Kaggle and do kill their own descendant."""
    lines = SOLVER_SRC.splitlines()
    for i, ln in enumerate(lines):
        if "pkill" in ln and not ln.strip().startswith(("#", '"""')) and "name-based" not in ln:
            ctx = "\n".join(lines[max(0, i - 3):i])
            assert "if ON_KAGGLE" in ctx, f"unguarded pkill at solver line {i + 1}: {ln}"
    g = build_notebook.GUARD_CELL.splitlines()
    for i, ln in enumerate(g):
        if "pkill" in ln:
            assert "isdir('/kaggle/input')" in g[i - 1], f"unguarded pkill in GUARD_CELL: {ln}"
    assert ns["ON_KAGGLE"] is False
    solver_file = Path(tempfile.mkdtemp()) / "solver.py"
    solver_file.write_text(SOLVER_SRC)
    child = (
        "import os, subprocess, sys, time\n"
        f"ns = {{'__name__': 'kill_check'}}\nexec(compile(open({str(solver_file)!r}).read(), 'solver.py', 'exec'), ns)\n"
        "cmds = []\nreal_run = subprocess.run\n"
        "subprocess.run = lambda c, *a, **k: cmds.append(c) or real_run('true', shell=True)\n"
        "ns['sh'] = lambda c, timeout=None: (cmds.append(c), (0, '100'))[1]\n"
        "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(0.3)\nassert kid.pid in ns['descendants'](), 'descendants'\n"
        "ns['kill_children']()\nkid.wait(timeout=10)\n"
        "class P:\n    pid = 2 ** 22 + 12345\n"
        "ns['kill_server'](P())\n"
        "assert not any('pkill' in str(c) for c in cmds), cmds\nprint('OK', ns['ON_KAGGLE'], kid.returncode)\n")
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=180,
                       env=dict(os.environ, ARC_INPUT="/nonexistent"))
    assert r.returncode == 0 and r.stdout.strip() == "OK False -9", (r.stdout, r.stderr[-1500:])
    print("no name-based kill off Kaggle OK")


# ---------------------------------------------------------------- 1. selection (pure)

def test_selection():
    g = lambda v: [[v]]  # noqa: E731
    # v2 rule without metadata: attempt answers in attempt order
    top = SEL.select_two(SEL.candidates({0: {"pred": g(1)}, 1: {"pred": g(2)}}))
    assert [c["grid"] for c in top] == [g(1), g(2)]
    # a train-passing alternative goes first; the attempt's own unchecked answer is kept second
    top = SEL.select_two(SEL.candidates({0: {"pred": g(1), "train_pass": None, "alternatives": [
        {"src": "salvage:f", "pred": g(3), "train_frac": 1.0}, {"src": "answer", "pred": g(4), "train_frac": None}]}}))
    assert [c["grid"] for c in top] == [g(3), g(1)]
    # answer known to fail train: alternatives ranked by train fraction compete with it
    top = SEL.select_two(SEL.candidates({0: {"pred": g(1), "train_pass": False, "train_frac": 0.0, "alternatives": [
        {"src": "a", "pred": g(5), "train_frac": 0.33}, {"src": "b", "pred": g(6), "train_frac": 0.67}]}}))
    assert [c["grid"] for c in top] == [g(6), g(5)]
    # second attempt's answer beats first attempt's alternatives; duplicates collapse
    top = SEL.select_two(SEL.candidates({0: {"pred": g(1), "alternatives": [{"src": "a", "pred": g(1)},
                                                                         {"src": "b", "pred": g(7)}]},
                                         1: {"pred": g(8)}}))
    assert [c["grid"] for c in top] == [g(1), g(8)]
    # invalid grids are ignored
    assert SEL.select_two(SEL.candidates({0: {"pred": [[1, 2], [3]], "alternatives": [{"pred": "x"}, 5]}})) == []
    # retry gating
    assert SEL.retry_category({"pred": None}) == 2
    assert SEL.retry_category({"pred": g(1), "train_pass": True}) is None
    assert SEL.retry_category({"pred": g(1), "train_pass": False, "answer_source": "submit"}) == 0
    assert SEL.retry_category({"pred": g(1), "train_pass": None, "answer_source": "forced"}) == 2
    assert SEL.retry_category({"pred": g(1), "train_pass": False, "answer_source": "program_best_fail"}) == 2
    assert SEL.retry_category({"pred": g(1), "train_pass": None, "answer_source": "submit"}) == 1
    assert SEL.retry_category({"pred": g(1), "train_pass": None, "answer_source": "submit"}, False) is None
    assert SEL.retry_category({"pred": g(1), "train_pass": False,
                               "alternatives": [{"pred": g(2), "train_frac": 1.0}]}) is None
    # program mode (v4/K3): eligibility = no train-passing grid
    assert SEL.retry_category({"pred": g(1), "train_pass": True, "answer_source": "program_pass"}, False) is None
    assert SEL.retry_category({"pred": g(1), "train_pass": False, "train_frac": 0.5,
                               "answer_source": "session_solve"}, False) == 2
    assert SEL.retry_category({"pred": g(1), "train_pass": True, "train_frac": 1.0,
                               "answer_source": "session_solve"}, False) is None  # reproduces every train pair
    assert SEL.retry_category({"pred": g(1), "train_pass": False, "answer_source": "program_best_fail"}, False) == 2
    assert SEL.retry_category({"pred": None, "answer_source": None}, False) == 2
    assert SEL.retry_category({"pred": None, "alternatives": [{"src": "submission_0", "pred": g(3),
                                                               "train_frac": 1.0}]}, False) is None
    keys = sorted([("none", SEL.retry_priority({"pred": None, "tokens": 100})),
                   ("unknown", SEL.retry_priority({"pred": g(1), "answer_source": "submit", "tokens": 5})),
                   ("fail_lo", SEL.retry_priority({"pred": g(1), "train_pass": False, "train_frac": 0.2, "tokens": 9})),
                   ("fail_hi", SEL.retry_priority({"pred": g(1), "train_pass": False, "train_frac": 0.8, "tokens": 50}))],
                  key=lambda x: x[1])
    assert [k for k, _ in keys] == ["fail_hi", "fail_lo", "unknown", "none"]
    # Submission.build: sources, fallback, robustness to malformed metadata (falls back to the v2 rule)
    tasks = {"t": {"test": [{"input": [[0, 1]]}], "train": []}}
    sub = ns["Submission"](tasks)
    sub.answers[("t", 0)] = {0: [[5]]}
    sub.meta[("t", 0)] = {0: {"train_pass": False, "alternatives": [{"src": "s", "pred": [[6]], "train_frac": 1.0}]}}
    b = sub.build()
    assert b["t"][0] == {"attempt_1": [[6]], "attempt_2": [[5]]}, b
    sub.meta[("t", 0)] = {0: {"alternatives": object()}}  # garbage -> v2 rule, never an exception
    b = sub.build()
    assert b["t"][0] == {"attempt_1": [[5]], "attempt_2": [[0, 1]]}, b
    sub.write()
    assert ns["check_submission"](ns["OUT_SUB"], tasks)
    print("selection OK")


# ---------------------------------------------------------------- 2. gated scheduler unit checks (no server)

def test_gated_unit():
    settings(ATTEMPT2="gated")
    ns.update(B_MIN2=10000, B_MAX2=48000, TOOLS_CAP=48000)
    now = time.time()
    sch = ns["Scheduler"](TASKS, ns["Submission"](TASKS), None, now + 3600, now + 3700, 4, attempts=2)
    sch.gen_start = now - 10 ** 4
    sch.rate = lambda: 100.0  # 100 tok/s total, 4 slots -> 25 tok/s per attempt (no finished records)
    # priority order: failing answers (higher train fraction first), then no answer
    for j, key in [(("a", 0, 1), (1, 0, 5)), (("b", 0, 1), (0, -0.2, 9)), (("c", 0, 1), (0, -0.9, 99))]:
        sch.priority[j] = key
        sch.pending2.append(j)
    order = []
    while True:
        nj = sch.next_job()
        if nj is None:
            break
        order.append(nj[0][0])
        assert 10000 <= nj[1] <= 48000, nj
        sch.inflight[nj[0]] = [nj[1], 0]
    assert order[:2] == ["c", "b"], order  # capacity 100*3600 = 360k: all three fit unless capacity runs out
    # first attempts always first
    sch.pending1 = [("x", 0, 0)]
    sch.pending2 = [("d", 0, 1)]
    assert sch.next_job()[0] == ("x", 0, 0)
    # capacity exhausted (in-flight budgets) -> wait, job kept
    sch.inflight = {("y", 0, 0): [10 ** 7, 0]}
    assert sch.next_job() is None and sch.pending2 == [("d", 0, 1)]
    assert sch.may_get_work()  # a first attempt is in flight: workers wait instead of exiting
    # no time left for any second attempt -> queue dropped, workers can exit
    sch.inflight = {}
    sch.force_at = time.time() + 60  # 25 tok/s * 59 s * 0.85 < B_MIN2
    assert sch.next_job() is None and sch.pending2 == [] and sch.gated["dropped_time"] == 1
    assert not sch.may_get_work()
    # off / legacy
    settings(ATTEMPT2="off")
    sch.pending2 = [("d", 0, 1)]
    assert sch.next_job() is None
    print("gated scheduler unit OK")


# ---------------------------------------------------------------- 3. memory caps and feature self-test

def test_memory_and_selftest():
    from arc_harness import executor as EX
    settings(TOOLS_ANSWER_MODE="program", HELPERS=1, TASK_SUMMARY=1, TURN_THINK_CAP=100, FREE_ATTEMPT2=1)
    saved = AQ.PROGRAM_MEMORY_MB
    ns["v3_selftest"]()
    st = ns["PHASES"]["v3_selftest"]
    assert st == {"session_mem": True, "helpers": True, "program": True, "task_summary": True}, st
    assert AQ.PROGRAM_MEMORY_MB == 2048
    cfg = ns["PHASES"]["agent_config"]
    assert cfg["answer_mode"] == "program" and cfg["helpers"] and cfg["task_summary"] and cfg["turn_think_cap"] == 100
    assert cfg["record_alternatives"] and cfg["harness_version"] == "qwen-tools-5", cfg
    # session cap: a 3 GB allocation fails with MemoryError, the session keeps its state; 500 MB still works
    with EX.LocalExecutor([{"input": [[1]], "output": [[1]]}], [[2]], timeout=10,
                          prelude_extra=ns["session_prelude"](2048, False)) as ex:
        r1 = ex.execute("keep = 41\nx = bytearray(3 * 2 ** 30)")
        r2 = ex.execute("y = bytearray(500 * 2 ** 20)\nprint(keep + 1, len(y) > 0)")
    assert "MemoryError" in r1["stderr"] and r2["stdout"].strip() == "42 True", (r1, r2)
    # a broken feature is switched off (v2 behaviour), not fatal: unknown prompt + bad answer mode
    settings(TOOLS_ANSWER_MODE="submit", TOOLS_PROMPT="lean2progroles", HELPERS=0, TASK_SUMMARY=0)
    ns["v3_selftest"]()
    assert ns["TOOLS_PROMPT"] == "", "program prompt with submit mode must fall back to the default"
    settings(SESSION_MEM_MB=0)
    assert ns["session_prelude"]() == ""
    AQ.PROGRAM_MEMORY_MB = saved
    settings()
    print("memory caps + self-test OK")


def test_e031_mode_wiring():
    """E032: the e031-mode settings reach the AgentConfig; notes keep the 20k / 60k training scale under a lower cap."""
    e031 = dict(TOOLS_PROMPT="lean2sub", TOOLS_DESC="e031", EXEC_STYLE="ipython", TOOLS_SOFT=20000, TOOLS_SOFT_FRAC=0.0,
                NOTE_CAP=60000, SOFT_WORDING="strong", EFFORT_HEADER="minimal", TURNS_NOTE=5, TOOLS_MAX_TURNS=40, TOOLS_TIER="xhigh")
    saved = {k: ns[k] for k in e031}
    try:
        settings(**e031)
        ns["v3_selftest"]()
        assert not [d for d in ns["DROPPED"] if d.split(":")[0] in ("tool_desc", "effort_header", "turns_note",
                                                                     "budget_note_cap", "exec_style")], ns["DROPPED"]
        cfg = ns["PHASES"]["agent_config"]
        want = {k: v for k, v in AQ.E031_CONFIG.items() if k not in ("max_total_output", "helpers", "task_summary", "tool_timeout")}
        assert {k: cfg.get(k, getattr(AQ.AgentConfig(), k)) for k in want} == want, cfg
        small = ns["build_agent_config"](AQ, 30000)
        assert (small.max_total_output, small.budget_soft, small.budget_note_cap) == (30000, 20000, 60000)
        assert AQ.budget_note(small, 21000).startswith("\n[budget: 21k output tokens used; aim to finish within 20k "
                                                       "(hard limit 60k)] Past the target")
        p = AQ.initial_prompt([{"input": [[1]], "output": [[2]]}], [[1]], small)
        assert "</IMPORTANT>\n\n" + AQ.EFFORT_HEADERS["minimal"] + "<|im_end|>" in p
    finally:
        settings(**saved)
    ns["v3_selftest"]()  # back to the defaults
    print("e031 mode wiring OK")


# ---------------------------------------------------------------- 4. end to end against the fake server

async def test_e2e():
    n_pairs = len(TASKS)
    # (a) v2 defaults: legacy, no alternatives -> attempt_2 = test input copy for single-answer pairs
    settings()
    sch, sub, final, starts, el = await run(TASKS, 60, 80, 6, 3000, b_min2=10 ** 9)
    recs0 = [r for r in sch.records if r["attempt"] == 0]
    assert len(recs0) == n_pairs and not any(r.get("alternatives") for r in sch.records)
    assert all(final[t][0]["attempt_2"] == TASKS[t]["test"][0]["input"] for t in TASKS if t.startswith("good"))
    v2_score = ns["score"](final, SOLS, TASKS)[0]
    print("v2 defaults:", v2_score, f"{el:.1f}s")

    # (b) free attempt_2 only: "bad" pairs are rescued by the train-passing alternative, no extra attempts
    settings(FREE_ATTEMPT2=1, ATTEMPT2="off")
    sch, sub, final, starts, el = await run(TASKS, 60, 80, 6, 3000)
    assert all(r["attempt"] == 0 for r in sch.records)
    bad = [r for r in sch.records if r["task_id"].startswith("bad_")]
    assert all(r["train_pass"] is False and any(a["train_frac"] == 1.0 for a in r["alternatives"]) for r in bad), bad
    assert all(correct(final, t, "attempt_1") or correct(final, t, "attempt_2") for t in TASKS if t.startswith("bad_"))
    assert all(r["train_pass"] is True for r in sch.records if r["task_id"].startswith("good"))
    free_score = ns["score"](final, SOLS, TASKS)[0]
    assert free_score > v2_score, (free_score, v2_score)
    print("free attempt_2:", free_score, {k: v for k, v in sub.sources.items() if k[0].startswith("bad")})

    # (c) gated: second attempts only for bad2 / none, after every first attempt started
    settings(FREE_ATTEMPT2=1, ATTEMPT2="gated")
    sch, sub, final, starts, el = await run(TASKS, 60, 80, 4, 3000)
    a1 = [s for s in starts if s[1][2] == 1]
    assert {j[0].split("_")[0] for _, j, _ in a1} == {"bad2", "none", "unknown"}, [j for _, j, _ in a1]
    assert max(t for t, j, _ in starts if j[2] == 0) <= min(t for t, _, _ in a1)
    assert sch.gated["eligible"] == 6 and sch.gated["started"] == 6, sch.gated
    gated_score = ns["score"](final, SOLS, TASKS)[0]
    assert gated_score == len(TASKS), gated_score  # every pair ends with the right grid
    # ARC_ATTEMPT2_UNKNOWN=0: the unchecked "unknown" answers are not retried
    settings(FREE_ATTEMPT2=1, ATTEMPT2="gated", ATTEMPT2_UNKNOWN=0)
    sch, sub, final, starts, el = await run(TASKS, 60, 80, 4, 3000)
    assert {j[0].split("_")[0] for _, j, _ in starts if j[2] == 1} == {"bad2", "none"}
    assert ns["score"](final, SOLS, TASKS)[0] == len(TASKS) - 2
    print("gated:", gated_score, sch.gated, f"{el:.1f}s")

    # (d) gated + deadline: slow stream; second attempts must not start once they cannot finish, everything
    # ends by the hard stop, and the file stays valid
    STATE["delay"] = 0.01
    settings(FREE_ATTEMPT2=1, ATTEMPT2="gated")
    sch, sub, final, starts, el = await run(TASKS, 8, 14, 4, 100000, b_min2=200)
    assert el < 15.5, el
    assert all(s[0] <= sch.force_at - sch.launch_margin for s in starts)
    print("gated deadline:", len(starts), "starts,", sch.gated, f"{el:.1f}s",
          "deadline cuts", sum((r.get("deadline_cuts") or 0) > 0 for r in sch.records))
    STATE["delay"] = 0.001

    # (e) program mode + helpers + summary + think cap
    settings(TOOLS_ANSWER_MODE="program", HELPERS=1, TASK_SUMMARY=1, TURN_THINK_CAP=40, TURN_THINK_GRACE=20,
             FREE_ATTEMPT2=1, ATTEMPT2="gated")
    ns["v3_selftest"]()
    sch, sub, final, starts, el = await run(TASKS, 60, 80, 6, 1500)
    srcs = {r["task_id"]: r.get("answer_source") for r in sch.records if r["attempt"] == 0}
    assert all(srcs[t] == "program_pass" for t in TASKS if t.startswith("good")), srcs
    nudged = [r for r in sch.records if r["task_id"].startswith("none") and r["attempt"] == 0]
    assert all((r.get("think_nudges") or 0) > 0 and (r.get("think_forced") or 0) > 0 for r in nudged), nudged
    assert all(r.get("harness_version") == "qwen-tools-5" for r in sch.records)
    print("program+helpers+summary+thinkcap:", ns["score"](final, SOLS, TASKS)[0], sch.gated,
          {k: sum(v == k for v in srcs.values()) for k in set(srcs.values())})
    settings()
    AQ.PROGRAM_MEMORY_MB = 3072


# ---------------------------------------------------------------- 5. failure paths of submission.json

def test_emergency_and_final_cell():
    """Solver subprocess against a server that never answers; submission.json is corrupted mid-run; the emergency
    timer must leave a valid file and exit 0 well before the hard stop. Then the notebook's final cell repairs a
    file with broken slots."""
    td = Path(tempfile.mkdtemp())
    inp, wk = td / "input", td / "working"
    (inp / "arc-prize-2026-arc-agi-2").mkdir(parents=True)
    wk.mkdir()
    test_file = inp / "arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    test_file.write_text(json.dumps(TASKS))  # ids differ from the public file -> rerun path (mode submit)
    import shutil
    pkg = wk / "arc_solver"  # the notebook's layout: evals/ + notes/prompts/ next to solver.py
    srcroot = Path(SRC_DIR) if SRC_DIR else ROOT
    shutil.copytree(srcroot / "evals/arc_harness", pkg / "evals/arc_harness",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("arc_helpers.py", "task_summary.py"):
        shutil.copy(srcroot / "evals" / f, pkg / "evals" / f)
    (pkg / "notes/prompts").mkdir(parents=True)
    shutil.copy(srcroot / "notes/prompts/arcprize_official_system_prompt.txt", pkg / "notes/prompts/")
    solver = pkg / "solver.py"
    solver.write_text(SOLVER_SRC)
    hang = subprocess.Popen([sys.executable, "-c", (  # /v1/models answers; completions and /tokenize hang
        "import http.server, json, time\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a):\n        pass\n"
        "    def do_GET(self):\n"
        "        b = json.dumps({'data': [{'id': 'x'}]}).encode()\n"
        "        self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers()\n"
        "        self.wfile.write(b)\n"
        "    def do_POST(self):\n        time.sleep(3600)\n"
        "http.server.ThreadingHTTPServer(('127.0.0.1', 18126), H).serve_forever()\n")])
    time.sleep(1)
    env = dict(os.environ, ARC_T0=str(time.time()), ARC_WORK=str(wk), ARC_INPUT=str(inp), ARC_MODE="local",
               ARC_EXTERNAL_SERVER="http://127.0.0.1:18126/v1", ARC_FORCE_AT_S="300", ARC_HARD_STOP_S="400",
               ARC_EMERGENCY_S="25", ARC_CONCURRENCY="4", ARC_ATTEMPT2="gated", ARC_FREE_ATTEMPT2="1",
               PYTHONPATH="")
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "sglang", "vllm.entrypoints",
                              "VLLM::decoy"])  # another user's server/eval: must survive the emergency path
    p = subprocess.Popen([sys.executable, "-u", str(solver)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=env, cwd=str(wk))
    try:
        t = time.time()
        while not (wk / "submission.json").exists() and time.time() - t < 20:
            time.sleep(0.2)
        time.sleep(8)
        (wk / "submission.json").write_text("{")  # corrupt it while attempts hang
        out, _ = p.communicate(timeout=120)
    finally:
        hang.kill()
        p.kill()
        decoy_alive = decoy.poll() is None
        decoy.kill()
    assert decoy_alive, "the emergency path killed an unrelated process matching vllm/sglang names"
    el = time.time() - float(env["ARC_T0"])
    assert "EMERGENCY timer fired" in out and "rerun=True" in out and "agent=tools" in out, out[-3000:]
    assert "falling back to the plain" not in out and "v3 self-test" in out, out[-3000:]
    assert p.returncode == 0 and el < 60, (p.returncode, el)
    assert ns["check_submission"](str(wk / "submission.json"), TASKS)
    left = subprocess.run(["pgrep", "-f", str(solver)], capture_output=True, text=True).stdout.strip()
    assert not left, f"solver processes left: {left}"
    print(f"emergency writer OK (exit {p.returncode} at {el:.0f}s, file valid)")

    # notebook final cell: repair broken slots, keep valid ones
    sub = {t: [{"attempt_1": SOLS[t][0], "attempt_2": [[1, 2], [3]]}] for t in TASKS}
    del sub["good_0"]
    (wk / "submission.json").write_text(json.dumps(sub))
    cell = (build_notebook.FINAL_CELL.replace("/kaggle/working", str(wk)).replace("/kaggle/input", str(inp)))
    cell = cell.split("# Commit-run gate")[0]
    r = subprocess.run([sys.executable, "-c", cell], capture_output=True, text=True, cwd=str(td))
    assert r.returncode == 0 and "slots repaired: 11" in r.stdout, (r.stdout, r.stderr)
    fixed = json.load(open(wk / "submission.json"))
    assert ns["check_submission"](str(wk / "submission.json"), TASKS)
    assert fixed["bad_0"][0]["attempt_1"] == SOLS["bad_0"][0]
    print("final repair cell OK:", r.stdout.strip())


async def main():
    os.makedirs(ns["LOG_DIR"], exist_ok=True)
    test_no_name_kill_off_kaggle()
    test_selection()
    test_gated_unit()
    test_memory_and_selftest()
    test_e031_mode_wiring()
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/tokenize", tokenize)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()
    await test_e2e()
    await runner.cleanup()
    test_emergency_and_final_cell()
    print("ALL OK")


asyncio.run(main())
