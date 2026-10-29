#!/usr/bin/env python3
"""E062 R1: greedy probe on one real ~40k-token tool transcript (a B32 replica trace), cold, R repeats.

  run:      probe_long.py --url http://127.0.0.1:8100 --messages probe_long.json --repeats 3 --json out.json
            POST /reset_prefix_cache before each request (VLLM_SERVER_DEV_MODE=1), /v1/chat/completions with the
            harness's two tools, temperature 0, seed 0, max_tokens 256, top-5 logprobs.
  compare:  probe_long.py --compare ref.json other.json
            within-reference pairs (the noise floor), within-other, across: first divergence of the generated token
            sequence, |d logprob| of token 0, mean |d logprob| over the common prefix.
The stack is not run-to-run deterministic (E054), so the gate is the noise floor, as in E054's logprob_probe.py.
"""

import argparse
import itertools
import json
import statistics
import time
import urllib.request

RUN_PYTHON = ("Execute a Python cell in your persistent IPython session. Variables/functions persist. Preloaded: train "
              "(list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. test_inputs: list of every test "
              "input grid of the task (test_input is test_inputs[0]). scipy is available. No internet or host files. "
              "Timeout 30s. Use print() to inspect results. Check transformations on all training pairs.")
SUBMIT = ("End this attempt with your test output. `code` is a Python expression evaluated in your session, e.g. "
          "solve(test_input) or answer, giving a rectangular grid of ints 0-9 (list or numpy array); statements that set "
          "a variable `answer` also work. The harness reads the grid directly: do not retype it. Invalid grids return an "
          "error and the attempt continues. When the task has several test inputs, the expression must give a list with "
          "one grid per test input, in order, e.g. [solve(t) for t in test_inputs].")
TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}}
    for n, d in (("run_python", RUN_PYTHON), ("submit_answer", SUBMIT))]


def post(url, path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def run(a):
    msgs = json.load(open(a.messages))["messages"]
    out = {"url": a.url, "messages": a.messages, "runs": [], "t": time.time()}
    for rep in range(a.repeats):
        post(a.url, "/reset_prefix_cache")
        t0 = time.time()
        r = post(a.url, "/v1/chat/completions", {
            "model": a.model, "messages": msgs, "tools": TOOLS, "max_tokens": a.max_tokens, "temperature": 0,
            "seed": 0, "logprobs": True, "top_logprobs": 5})
        ch = r["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        rec = {"rep": rep, "s": round(time.time() - t0, 1), "prompt_tokens": r["usage"]["prompt_tokens"],
               "tokens": [x["token"] for x in lp], "lps": [x["logprob"] for x in lp],
               "top0": {t["token"]: t["logprob"] for t in (lp[0]["top_logprobs"] if lp else [])}}
        out["runs"].append(rec)
        print(json.dumps({"rep": rep, "n": len(rec["tokens"]), "prompt_tokens": rec["prompt_tokens"], "s": rec["s"],
                          "head": "".join(rec["tokens"][:12])}), flush=True)
    json.dump(out, open(a.json, "w"))


def pair(x, y):
    n = min(len(x["tokens"]), len(y["tokens"]))
    div = next((i for i in range(n) if x["tokens"][i] != y["tokens"][i]), n)
    t0 = set(x["top0"]) | set(y["top0"])
    d0 = max((abs(x["top0"].get(t, -30) - y["top0"].get(t, -30)) for t in t0), default=0.0)
    dl = [abs(x["lps"][i] - y["lps"][i]) for i in range(div)]
    return {"div": div, "dlp0": d0, "mean_dlp": statistics.mean(dl) if dl else 0.0,
            "argmax0_same": bool(x["tokens"] and y["tokens"] and x["tokens"][0] == y["tokens"][0])}


def summ(ps):
    if not ps:
        return {}
    return {"n": len(ps), "div_median": statistics.median(p["div"] for p in ps), "div_min": min(p["div"] for p in ps),
            "dlp0_max": max(p["dlp0"] for p in ps), "mean_dlp_median": statistics.median(p["mean_dlp"] for p in ps),
            "argmax0_same": sum(p["argmax0_same"] for p in ps)}


def compare(files):
    ref = json.load(open(files[0]))["runs"]
    print(json.dumps({"ref": files[0], "within_ref": summ([pair(x, y) for x, y in itertools.combinations(ref, 2)])}))
    for f in files[1:]:
        o = json.load(open(f))["runs"]
        print(json.dumps({"arm": f, "within": summ([pair(x, y) for x, y in itertools.combinations(o, 2)]),
                          "across": summ([pair(x, y) for x in ref for y in o])}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--messages")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--json")
    p.add_argument("--compare", nargs="+")
    a = p.parse_args()
    compare(a.compare) if a.compare else run(a)
