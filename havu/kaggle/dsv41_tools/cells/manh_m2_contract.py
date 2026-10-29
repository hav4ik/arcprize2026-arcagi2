# NOT USED (E064, 2026-10-28: superseded by B32-M; kept for reference)
# Cell M2 (E064): what B32's harness needs from the server (E062 l4/api_check.py), checked once it answers.
# Required: /v1/models (id, max_model_len), /tokenize renders previous-turn thinking (the harness sends it as both
# `reasoning_content` and `reasoning`), token-id /v1/completions returns token ids (--token-exact), /metrics has
# vllm:kv_cache_usage_perc (KV gate). Reported: tool-call parsing on the chat path, the server's max_num_seqs.
# Commit run: a failed requirement RAISES. Rerun: printed, the solver runs anyway.
import re, urllib.request


def _get(path, timeout=10):
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def _post(path, body, timeout=600):
    req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


CONTRACT = {}
if MODE != "fallback":
    _wait = float(os.environ.get("ARC_CONTRACT_WAIT_S", "5400"))
    _t = time.time()
    while time.time() - _t < _wait:
        try:
            _models = json.loads(_get("/v1/models"))["data"]
            break
        except Exception:
            time.sleep(10)
    else:
        _models = None
    CONTRACT["server_up_s"] = round(time.time() - _t)
    _fail = []
    if not _models:
        _fail.append(f"no /v1/models answer within {_wait:.0f}s")
    else:
        MODEL = BASE_ENV["ARC_SERVED_MODEL"]
        CONTRACT["models"] = [(m.get("id"), m.get("max_model_len")) for m in _models]
        if MODEL not in [m.get("id") for m in _models]:
            print(f"warning: the server serves {CONTRACT['models']}, not {MODEL!r}; the solver uses its first id")
            MODEL = _models[0].get("id")
        _kw = {"thinking": True, "reasoning_effort": 50}
        _tools = [{"type": "function", "function": {
            "name": "run_python", "description": "Run Python code in a persistent session and return stdout.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
        _mark = "ZEBRA-THINKING-MARKER-42"
        try:
            for _field in ("reasoning_content", "reasoning"):
                _msgs = [{"role": "user", "content": "What is 2+3? Use the tool."},
                         {"role": "assistant", "content": "", _field: f"I should compute it. {_mark}",
                          "tool_calls": [{"id": "c1", "type": "function", "function": {
                              "name": "run_python", "arguments": json.dumps({"code": "print(2+3)"})}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": "5"}]
                _tk = _post("/tokenize", {"model": MODEL, "messages": _msgs, "tools": _tools,
                                          "chat_template_kwargs": _kw, "add_generation_prompt": True})
                _dt = _post("/detokenize", {"model": MODEL, "tokens": _tk["tokens"]})
                CONTRACT[f"thinking_via_{_field}"] = _mark in _dt["prompt"]
            if not (CONTRACT["thinking_via_reasoning_content"] or CONTRACT["thinking_via_reasoning"]):
                _fail.append("/tokenize drops previous-turn thinking under both keys")
        except Exception as exc:
            _fail.append(f"/tokenize or /detokenize failed: {type(exc).__name__}: {exc}"[:300])
        try:
            _msgs = [{"role": "system", "content": "You solve tasks with the run_python tool."},
                     {"role": "user", "content": "Compute the sum of the squares of 1..10 with run_python."}]
            _tk = _post("/tokenize", {"model": MODEL, "messages": _msgs, "tools": _tools, "chat_template_kwargs": _kw,
                                      "add_generation_prompt": True})
            _r = _post("/v1/completions", {"model": MODEL, "prompt": _tk["tokens"], "max_tokens": 64, "temperature": 0,
                                           "return_token_ids": True, "cache_salt": "e064-contract"})
            CONTRACT["completion_token_ids"] = bool(_r["choices"][0].get("token_ids"))
            if not CONTRACT["completion_token_ids"]:
                _fail.append("token-id /v1/completions returns no token_ids")
        except Exception as exc:
            _fail.append(f"token-id /v1/completions failed: {type(exc).__name__}: {exc}"[:300])
        try:
            _m = _get("/metrics")
            CONTRACT["metrics"] = {k: bool(re.search(rf"^vllm:{k}\b", _m, re.M)) for k in (
                "kv_cache_usage_perc", "num_requests_running", "num_requests_waiting", "generation_tokens_total",
                "prefix_cache_hits_total", "num_preemptions_total")}
            if not CONTRACT["metrics"]["kv_cache_usage_perc"]:
                _fail.append("/metrics has no vllm:kv_cache_usage_perc (KV gate)")
        except Exception as exc:
            _fail.append(f"/metrics failed: {type(exc).__name__}: {exc}"[:300])
        try:  # informative only: the harness parses tool calls itself on the token-exact path
            _c = _post("/v1/chat/completions", {"model": MODEL, "messages": _msgs, "tools": _tools, "temperature": 0,
                                                "max_tokens": 1500, "chat_template_kwargs": _kw})
            CONTRACT["chat_tool_calls"] = len(_c["choices"][0]["message"].get("tool_calls") or [])
        except Exception as exc:
            CONTRACT["chat_tool_calls"] = f"{type(exc).__name__}"
        try:  # Manh's --concurrency must cover our workers
            _log = open(f"{server_root}/server/server.log", errors="replace").read()  # noqa: F821 (Manh's cell)
            _seqs = re.findall(r"max_num_seqs[=:'\" ]+(\d+)", _log)
            CONTRACT["server_max_num_seqs"] = int(_seqs[-1]) if _seqs else None
            if _seqs and int(_seqs[-1]) < int(BASE_ENV["ARC_WORKERS"]):
                print(f"warning: server max_num_seqs {_seqs[-1]} < ARC_WORKERS {BASE_ENV['ARC_WORKERS']}", flush=True)
        except Exception as exc:
            CONTRACT["server_max_num_seqs"] = f"unknown ({type(exc).__name__})"
    CONTRACT["fail"] = _fail
    json.dump(CONTRACT, open(f"{WORK}/e064_contract.json", "w"), indent=1)
    if _fail:
        print("=" * 100, f"E064 CONTRACT FAILED: {_fail}", CONTRACT, "=" * 100, sep="\n", flush=True)
        if not RERUN:
            raise RuntimeError(f"E064 CONTRACT FAILED (commit run): {_fail}")
    else:
        print("E064 CONTRACT OK:", CONTRACT, flush=True)
