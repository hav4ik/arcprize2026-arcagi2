"""Fake vLLM (DeepSeek-V4.1 token-exact) server for adversarial end-to-end tests of solver_dsv41.py.

Started by the solver as `$ARC_VLLM serve <model> ... --port P` (our fake `vllm` wrapper execs this file).
Tokenizer: id = ord(char) (prefix-stable by construction). /tokenize renders a DeepSeek-like template (earlier
reasoning dropped before a later user message, like V4.1). /v1/completions streams canonical or deliberately broken
V4.1 DSML output, chosen per attempt by a seeded behaviour mix.

Env: FAKE_CHALLENGES (default ARC_CHALLENGES), FAKE_LOG, FAKE_STATE, FAKE_SEED, FAKE_MIX (name), FAKE_STARTUP_S,
FAKE_DIE_AFTER (completion requests; once per FAKE_STATE), FAKE_NEVER_UP, FAKE_METRICS (normal|404|hang|nan|nokv|high|
dynamic), FAKE_STALL_S, FAKE_CHUNK_DELAY, FAKE_CTX_PAD.
"""
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time

from aiohttp import web

E = os.environ.get
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
USER, ASSIST, TOOL, TOOL_END = "<｜User｜>", "<｜Assistant｜>", "<｜Tool｜>", "</｜Tool｜>"
DSML = "｜DSML｜"
CHAL = json.load(open(E("FAKE_CHALLENGES") or E("ARC_CHALLENGES")))
LOGF = E("FAKE_LOG", "/dev/null")
STATE_DIR = E("FAKE_STATE", "/tmp")
SEED = E("FAKE_SEED", "0")
MIX = E("FAKE_MIX", "all")
STALL_S = float(E("FAKE_STALL_S", "3600"))
CHUNK_DELAY = float(E("FAKE_CHUNK_DELAY", "0.002"))
CTX_PAD = int(E("FAKE_CTX_PAD", "60000"))
METRICS = E("FAKE_METRICS", "normal")
DIE_AFTER = int(E("FAKE_DIE_AFTER", "0"))
DIE_TIMES = int(E("FAKE_DIE_TIMES", "1"))  # next-candidate e2e: die this many times (once per server process)
ST = {"requests": 0, "gen": 0, "starts": {}, "task_cache": {}, "active": {}, "max_active": 0, "inflight": 0,
      "max_inflight": 0}

SINGLE = {  # behaviour -> weight (single test input or per-pair attempt)
    "good": 20, "good_ud": 8, "oracle": 8, "numpy": 8, "badshape": 6, "answer_only": 6, "never": 4, "budget": 4,
    "garbage": 8, "loop": 3, "ctx": 3, "err500": 3, "drop": 3, "nousage": 2, "abort": 2, "slow": 3, "tokerr": 2,
    "tool_hang": 2, "shadow": 2, "stall": 0,
}
MULTI = {  # several test inputs in one attempt
    "mgood": 20, "oracle": 6, "mnumpy": 8, "mnumpy3d": 3, "mrows": 4, "mtuple": 3, "mwrongn": 8, "mwrongn_never": 6,
    "mnosolve": 4, "mbadgrid": 4, "never": 3, "budget": 3, "garbage": 6, "loop": 2, "ctx": 2, "err500": 2,
    "drop": 2, "nousage": 1, "abort": 1, "slow": 2, "tool_hang": 1, "stall": 0,
}
if MIX == "stall":  # force-deadline scenario: many attempts hang inside a stream
    SINGLE.update(stall=40)
    MULTI.update(stall=40)
if MIX == "tokerr":
    SINGLE = {"tokerr": 1}
    MULTI = {"tokerr": 1}
if MIX == "clean":
    SINGLE = {"good": 5, "oracle": 3, "numpy": 2, "good_ud": 2}
    MULTI = {"mgood": 5, "oracle": 3, "mnumpy": 2}


def log(**kw):
    kw["t"] = round(time.time(), 3)
    with open(LOGF, "a") as f:
        f.write(json.dumps(kw) + "\n")


# ---------------------------------------------------------------- template / tokenizer

def dsml_calls(calls):
    out = f"\n\n<{DSML} calls>\n"
    for name, params in calls:
        out += f'<{DSML} invoke name="{name}">\n'
        for k, v in params:
            if isinstance(v, str):
                out += f'<{DSML} parameter name="{k}" string="true">{v}</{DSML} parameter>\n'
            else:
                out += f'<{DSML} parameter name="{k}" string="false">{json.dumps(v)}</{DSML} parameter>\n'
        out += f"</{DSML} invoke>\n"
    return out + f"</{DSML} calls>"


def render(messages, gen, pad_tools=False):
    last_user = max([i for i, m in enumerate(messages) if m.get("role") == "user"] or [-1])
    parts = [BOS]
    for i, m in enumerate(messages):
        r = m.get("role")
        if r == "system":
            parts.append(m.get("content") or "")
        elif r == "user":
            parts.append(USER + (m.get("content") or ""))
        elif r == "assistant":
            reasoning = (m.get("reasoning_content") or "") if i > last_user else ""  # V4.1 drops old thinking
            s = ASSIST + "<think>" + reasoning + "</think>" + (m.get("content") or "")
            calls = []
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"]["arguments"])
                except Exception:
                    args = {}
                calls.append((c["function"]["name"], list(args.items()) if isinstance(args, dict) else []))
            if calls:
                s += dsml_calls(calls)
            parts.append(s + EOS)
        elif r == "tool":
            parts.append(TOOL + (m.get("content") or "") + TOOL_END + (" " * CTX_PAD if pad_tools else ""))
    if gen:
        parts.append(ASSIST + "<think>")
    return "".join(parts)


MARK = re.compile(r"\[fake:(\w+):(\d+):(\d+):([^\]]*)\]")


def identify(user_text):
    h = hashlib.sha1(user_text.encode("utf-8", "surrogatepass")).hexdigest()
    if h in ST["task_cache"]:
        return ST["task_cache"][h]
    found = None
    multi = "--Test input 0--" in user_text
    tail = user_text.split("--Test Input--")[-1] if "--Test Input--" in user_text else user_text
    for t, task in CHAL.items():
        if json.dumps(task["train"][0]["input"]) not in user_text or json.dumps(task["train"][0]["output"]) not in user_text:
            continue
        if multi:
            if all(json.dumps(x["input"]) in user_text for x in task["test"]):
                found = (t, -1, len(task["test"]))
                break
        else:
            body = tail.split("--End of Test Input--")[0].strip()
            for p, x in enumerate(task["test"]):
                if json.dumps(x["input"]) == body:
                    found = (t, p, 1)
                    break
            if found:
                break
    ST["task_cache"][h] = found
    return found


def choose(task, pair, n, start):
    rng = random.Random(f"{SEED}:{task}:{pair}:{start}")
    mix = MULTI if n > 1 else SINGLE
    names = [k for k, w in mix.items() if w > 0]
    return rng.choices(names, [mix[k] for k in names])[0]


# ---------------------------------------------------------------- behaviours: (reasoning, content, calls) per turn

FLIP = "def solve(g):\n    return [list(r)[::-1] for r in g]\n"
FLIPUD = "def solve(g):\n    return [list(r) for r in g[::-1]]\n"
ORACLE = ("def solve(g):\n    for p in train:\n        if p['input'] == g:\n            return p['output']\n"
          "    return [list(r)[::-1] for r in g]\n")
NPFLIP = "def solve(g):\n    return np.fliplr(np.array(g))\n"


def ans(n):
    return "[solve(t) for t in test_inputs]" if n > 1 else "solve(test_input)"


def setans(n):
    return f"answer = {ans(n)}\nprint('ok', len(train))"


def rp(code):
    return [("run_python", [("code", code)])]


def sa(code):
    return [("submit_answer", [("code", code)])]


def plan(beh, k, n):
    """-> dict(kind=..., reasoning, content, calls) for turn k (0-based). kind: canon | raw | special."""
    def canon(calls, reasoning="Brief note.", content=""):
        return {"kind": "canon", "reasoning": reasoning, "content": content, "calls": calls}
    maxt = 40 * (1 + 0.5 * (n - 1)) if n > 1 else 40
    if beh in ("good", "mgood", "slow", "drop", "err500", "nousage", "abort", "tokerr"):
        if k == 0:
            return canon(rp(FLIP + setans(n)))
        if k == 1 and beh in ("drop", "err500", "nousage", "abort"):
            return {"kind": beh}
        if k == 1 and beh == "slow":
            return {"kind": "slow", "then": canon(sa(ans(n)))}
        return canon(sa(ans(n)))
    if beh == "stall":
        if k == 0:
            return canon(rp(FLIP + setans(n)))
        return {"kind": "stall"}
    if beh == "good_ud":
        return canon(rp(FLIPUD + setans(n))) if k == 0 else canon(sa(ans(n)))
    if beh == "oracle":
        return canon(rp(ORACLE + setans(n))) if k == 0 else canon(sa(ans(n)))
    if beh == "numpy":
        return canon(rp(NPFLIP + "print(solve(test_input).shape)")) if k == 0 else canon(sa("solve(test_input)"))
    if beh == "mnumpy":
        return canon(rp(NPFLIP + "print(1)")) if k == 0 else canon(sa("[solve(t) for t in test_inputs]"))
    if beh == "mnumpy3d":  # one numpy array of all outputs if shapes agree, else list of arrays
        code = ("_o = [solve(t) for t in test_inputs]\n"
                "answer = np.array(_o) if len({x.shape for x in _o}) == 1 else _o")
        return canon(rp(NPFLIP + code)) if k == 0 else canon(sa("answer"))
    if beh == "mrows":
        return canon(rp(FLIP)) if k == 0 else canon(sa("[[np.array(r) for r in solve(t)] for t in test_inputs]"))
    if beh == "mtuple":
        return canon(rp(FLIP)) if k == 0 else canon(sa("tuple(np.array(solve(t)) for t in test_inputs)"))
    if beh == "mwrongn":
        seq = [rp(FLIP + "print(1)"), sa("[solve(t) for t in test_inputs[:-1]]"), sa("solve(test_input)"),
               sa("[solve(t) for t in test_inputs] + [[[1]]]"), sa("[solve(t) for t in test_inputs]")]
        return canon(seq[min(k, len(seq) - 1)])
    if beh == "mwrongn_never":
        if k == 0:
            return canon(rp(FLIP + "answer = solve(test_input)"))
        return canon(sa("[solve(t) for t in test_inputs[:1]]") if k % 2 else rp("print(len(test_inputs))"))
    if beh == "mnosolve":
        return canon(rp("answer = [[[0]]]\nprint(1)")) if k == 0 else canon(rp("print('thinking')"))
    if beh == "mbadgrid":
        if k == 0:
            return canon(rp(FLIP + "answer = [solve(t) for t in test_inputs]\nanswer[-1] = [[1, 2], [3]]"))
        if k < 4:
            return canon(sa("answer"))
        return canon(rp("print(2)"))
    if beh == "badshape":
        bad = ["[[1, 2], [3]]", "[[10]]", "[[True]]", "[[1.0]]", "np.zeros((31, 31), int)", "[]", "None", "[[]]",
               "{'a': 1}", "'[[1]]'", "[[-1]]", "np.zeros((2, 2, 2), int)", "[[1]] * 0"]
        if k == 0:
            return canon(rp(FLIP + "print(1)"))
        if k <= len(bad):
            return canon(sa(bad[k - 1]))
        return canon(sa("solve(test_input)"))
    if beh == "answer_only":
        return canon(rp(FLIP + setans(n))) if k == 0 else canon(rp(f"print({k})"))
    if beh == "never":
        return canon(rp(f"print({k})"))
    if beh == "budget":
        return canon(rp(FLIP + setans(n))) if k == 0 else {"kind": "budget"}
    if beh == "loop":
        return canon(rp(FLIP + setans(n))) if k == 0 else {"kind": "loop"}
    if beh == "ctx":
        return canon(rp(FLIP + setans(n) + "\nprint('x' * 3000)"))
    if beh == "tool_hang":
        if k == 0:
            return canon(rp(FLIP + "print(1)"))
        if k == 1:
            return canon(rp("while True:\n    pass"))
        return canon(sa(ans(n)))
    if beh == "shadow":
        if k == 0:
            return canon(rp(NPFLIP + "def eval(*a):\n    return 0\ndef compile(*a, **k):\n    raise SyntaxError\n"
                            "json = None\ncontextlib = None\nio = None\nprint(1)"))
        return canon(sa("solve(test_input)"))
    if beh == "garbage":
        g = ["no_think", "fence", "badname", "eos_mid", "badjson", "nocode", "ctrl", "two_calls", "huge_print",
             "think_calls", "unclosed_call", "empty"]
        if k == 0:
            return canon(rp("print('start')"))
        if k <= len(g):
            return {"kind": "garbage", "g": g[k - 1], "n": n}
        return canon(sa(ans(n)))
    return canon(rp(f"print({k})"))


def garbage_text(g, n):
    fence = "```python\n" + FLIP + setans(n) + "\n```"
    if g == "no_think":
        return "I keep thinking about this without ever closing my thoughts. " * 3 + EOS
    if g == "fence":
        return "Let me run code.</think>Here is code:\n" + fence + EOS
    if g == "badname":
        return "x</think>" + dsml_calls([("run_ppython", [("code", "print('badname')")])]) + EOS
    if g == "eos_mid":
        return "a" + EOS + "b</think>text" + EOS
    if g == "badjson":
        return ("x</think>\n\n<" + DSML + ' calls>\n<' + DSML + ' invoke name="run_python">\n<' + DSML
                + ' parameter name="code" string="false">print("badjson")</' + DSML + ' parameter>\n</' + DSML
                + ' invoke>\n</' + DSML + " calls>" + EOS)
    if g == "nocode":
        return "x</think>" + dsml_calls([("run_python", [("cmd", "print(1)")])]) + EOS
    if g == "ctrl":
        return "ctl \x00\x01\x1b[31m \ufffd \u202e rtl</think>" + dsml_calls([("run_python", [("code", "print('\\x00ok')")])]) + EOS
    if g == "two_calls":
        return "x</think>" + dsml_calls([("run_python", [("code", FLIP + "print(2)")]),
                                         ("submit_answer", [("code", "[[1, 2], [3]]")])]) + EOS
    if g == "huge_print":
        return "x</think>" + dsml_calls([("run_python", [("code", "print('y' * 3_000_000)")])]) + EOS
    if g == "think_calls":  # a calls block opened inside the thinking (no </think>)
        return "thinking " + dsml_calls([("run_python", [("code", "print('in think')")])]) + EOS
    if g == "unclosed_call":
        return ("x</think>\n\n<" + DSML + ' calls>\n<' + DSML + ' invoke name="run_python">\n<' + DSML
                + ' parameter name="code" string="true">print("unclosed")')
    if g == "empty":
        return ""
    return "?"


def canon_text(p):
    calls = dsml_calls(p["calls"]) if p["calls"] else ""
    return p["reasoning"] + "</think>" + p["content"] + calls + EOS


# ---------------------------------------------------------------- HTTP

async def models(req):
    return web.json_response({"object": "list", "data": [{"id": "dsv41", "object": "model"}]})


async def tokenize(req):
    body = await req.json()
    msgs = body.get("messages") or []
    text = render(msgs, bool(body.get("add_generation_prompt")))
    m = None
    for x in msgs:
        if x.get("role") == "assistant":
            m = MARK.search(x.get("content") or "") or m
            if m:
                break
    if m and m.group(1) == "tokerr" and sum(x.get("role") == "assistant" for x in msgs) >= 1:
        return web.json_response({"error": "tokenize failed (fake)"}, status=500)
    if m and m.group(1) == "ctx":
        text = render(msgs, bool(body.get("add_generation_prompt")), pad_tools=True)
    return web.json_response({"count": len(text), "max_model_len": 163840, "tokens": [ord(c) for c in text]})


def chunk(text, finish=None, usage=None):
    d = {"id": "cmpl-fake", "object": "text_completion", "created": int(time.time()), "model": "dsv41",
         "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish, "stop_reason": None,
                      "token_ids": [ord(c) for c in text]}], "usage": None}
    if usage is not None:
        d = {"id": "cmpl-fake", "object": "text_completion", "created": int(time.time()), "model": "dsv41",
             "choices": [], "usage": usage}
    return f"data: {json.dumps(d)}\n\n".encode()


async def completions(req):
    ST["requests"] += 1
    deaths = len([f for f in os.listdir(STATE_DIR) if f.startswith("died_")])
    if DIE_AFTER and ST["requests"] >= DIE_AFTER and deaths < DIE_TIMES:
        open(f"{STATE_DIR}/died_{deaths}", "w").write(str(time.time()))
        log(event="die", requests=ST["requests"])
        os._exit(17)
    body = await req.json()
    ids = body["prompt"]
    max_tokens = int(body.get("max_tokens") or 1000)
    text = "".join(map(chr, ids))
    k = text.count(ASSIST) - 1
    m = MARK.search(text)
    user = text.split(USER, 1)[1].split(ASSIST, 1)[0] if USER in text else ""
    if k == 0 or not m:
        info = identify(user)
        if info is None:
            log(event="unknown_task", k=k)
            return web.json_response({"error": "unknown task"}, status=400)
        t, p, n = info
        key = f"{t}:{p}"
        start = ST["starts"].get(key, 0)
        ST["starts"][key] = start + 1
        beh = choose(t, p, n, start)
        if k > 0:
            log(event="marker_lost", task=t, k=k)
        marker = f"[fake:{beh}:{start}:{n}:{t}#{p}]"
    else:
        beh, start, n, tp = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        marker = None
        t = tp
    attempt_key = f"{t}:{start}" if marker is None else f"{t}:{start}"
    now = time.time()
    ST["active"][attempt_key] = now
    for a in [a for a, ts in ST["active"].items() if now - ts > 20]:
        del ST["active"][a]
    ST["max_active"] = max(ST["max_active"], len(ST["active"]))
    p = plan(beh, k, n)
    kind = p["kind"]
    if marker is not None and kind == "canon":
        p["content"] = marker + p["content"]
    log(event="turn", beh=beh, k=k, n=n, kind=kind if kind != "garbage" else "g:" + p["g"], task=t,
        prompt=len(ids), max_tokens=max_tokens, active=len(ST["active"]), inflight=ST["inflight"])
    if kind == "err500":
        return web.json_response({"error": {"message": "fake internal error", "type": "InternalServerError"}},
                                 status=500)
    if len(ids) + max_tokens > int(E("FAKE_MAX_MODEL_LEN", "163840")):  # vLLM: 400 past max-model-len
        log(event="ctx400", task=t, beh=beh, k=k, prompt=len(ids), max_tokens=max_tokens)
        return web.json_response({"error": {"message": f"maximum context length is 163840 tokens; requested "
                                            f"{len(ids) + max_tokens}", "type": "BadRequestError"}}, status=400)
    resp = web.StreamResponse()
    resp.content_type = "text/event-stream"
    await resp.prepare(req)
    ST["inflight"] += 1
    ST["max_inflight"] = max(ST["max_inflight"], ST["inflight"])
    finish = "stop"
    try:
        if kind == "canon":
            out = canon_text(p)
        elif kind == "slow":
            out = canon_text(p["then"])
        elif kind == "garbage":
            out = garbage_text(p["g"], p["n"])
        elif kind == "loop":
            out = "I will check the grid again. " * 600
        elif kind in ("drop", "nousage", "abort", "stall", "budget"):
            out = "Thinking about the rule. " * 20
        else:
            out = "?" + EOS
        if len(out) > max_tokens:
            out, finish = out[:max_tokens], "length"
        if kind == "loop":
            finish = "length"
        n_sent = 0
        for i in range(0, max(len(out), 1), 64):
            piece = out[i:i + 64]
            await resp.write(chunk(piece))
            n_sent += len(piece)
            if CHUNK_DELAY:
                await asyncio.sleep(CHUNK_DELAY)
            if kind == "slow" and i == 64:
                await asyncio.sleep(15)
            if kind == "stall" and i == 128:
                log(event="stall_begin", task=t)
                await asyncio.sleep(STALL_S)
            if kind == "drop" and i >= 128:
                log(event="drop", task=t)
                req.transport.abort()
                return resp
        if kind == "abort":
            finish = "abort"
        comp = n_sent
        if kind == "budget":
            comp, finish = max_tokens, "length"
        await resp.write(chunk("", finish=finish))
        ST["gen"] += comp
        if kind != "nousage":
            await resp.write(chunk("", usage={"prompt_tokens": len(ids), "completion_tokens": comp,
                                               "total_tokens": len(ids) + comp,
                                               "prompt_tokens_details": {"cached_tokens": max(0, len(ids) - 500)}}))
        await resp.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, asyncio.CancelledError, RuntimeError) as exc:
        log(event="client_gone", task=t, beh=beh, k=k, err=type(exc).__name__)
    finally:
        ST["inflight"] -= 1
    return resp


async def metrics(req):
    if METRICS == "404":
        return web.Response(status=404, text="not found")
    if METRICS == "hang":
        await asyncio.sleep(3600)
    if METRICS == "dynamic":
        kv = min(1.0, len([a for a, ts in ST["active"].items() if time.time() - ts < 20]) * 0.03)
    elif METRICS == "high":
        kv = 0.97
    else:
        kv = 0.25
    lines = [f'vllm:generation_tokens_total{{engine="0",model_name="dsv41"}} {float(ST["gen"])}',
             f'vllm:prefix_cache_hits_total{{engine="0",model_name="dsv41"}} 100.0',
             f'vllm:prefix_cache_queries_total{{engine="0",model_name="dsv41"}} 120.0',
             f'vllm:num_requests_running{{engine="0",model_name="dsv41"}} {ST["inflight"]}.0',
             f'vllm:num_requests_waiting{{engine="0",model_name="dsv41"}} 0.0',
             f'vllm:num_preemptions_total{{engine="0",model_name="dsv41"}} 0.0']
    if METRICS == "nan":
        lines.append('vllm:kv_cache_usage_perc{engine="0",model_name="dsv41"} NaN')
    elif METRICS != "nokv":
        lines.append(f'vllm:kv_cache_usage_perc{{engine="0",model_name="dsv41"}} {kv}')
    if METRICS == "garbled":
        lines = ["vllm:kv_cache_usage_perc{x} not_a_number", "vllm:kv_cache_usage_perc", "garbage {{{"]
    log(event="metrics", mode=METRICS, max_active=ST["max_active"], max_inflight=ST["max_inflight"])
    return web.Response(text="\n".join(lines) + "\n")


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8100
    startup = float(E("FAKE_STARTUP_S", "2"))
    log(event="start", pid=os.getpid(), port=port, argv=sys.argv[1:4], mix=MIX, metrics=METRICS,
        restart=os.path.exists(f"{STATE_DIR}/died_0"))
    if os.path.exists(f"{STATE_DIR}/died_0") and E("FAKE_RESTART_NEVER_UP"):
        os.environ["FAKE_NEVER_UP"] = E("FAKE_RESTART_NEVER_UP")
    if E("FAKE_NEVER_UP") == "exit":
        time.sleep(1)
        sys.exit(3)
    if E("FAKE_NEVER_UP") == "hang":
        time.sleep(10 ** 6)
    time.sleep(startup)
    app = web.Application(client_max_size=1 << 30)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    web.run_app(app, host="127.0.0.1", port=port, print=None, access_log=None)


if __name__ == "__main__":
    main()
