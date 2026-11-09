# Cell 2c (B32-M smoke, commit runs only): the API contract of the solver's server (his vLLM), checked in a
# background thread once it answers (E062 l4/api_check.py), so it costs no wall time. Required (the report cell
# fails the commit run otherwise): /v1/models serves dsv41 with max_model_len >= ARC_CONTEXT_LIMIT; /tokenize +
# /detokenize render a tool transcript; token-id /v1/completions returns token_ids (--token-exact); /metrics has
# vllm:kv_cache_usage_perc (KV gate). Recorded only: previous-turn thinking via `reasoning_content` (B32's harness
# key; upstream reads `reasoning`: only the rare full re-renders, ~0.1% of turns, lose it) and via `reasoning`, and
# a chat tool call (the harness parses tool calls itself on the token-exact path). Writes smoke_contract.json.
import re, sys, threading, urllib.request

CONTRACT_OUT = f"{WORK}/smoke_contract.json"


def contract_run(base, model="dsv41", wait_s=3600):
    res, fail = {}, []

    def post(path, body, timeout=600):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    t = time.time()
    models = None
    while time.time() - t < wait_s:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
                models = json.loads(r.read())["data"]
            break
        except Exception:
            time.sleep(10)
    res["server_up_at_s"] = round(time.time() - T0)
    if not models:
        return {**res, "fail": [f"server never answered /v1/models within {wait_s}s"], "done": True}
    res["models"] = [(m.get("id"), m.get("max_model_len")) for m in models]
    ctx = int(os.environ.get("ARC_CONTEXT_LIMIT", "163840"))
    if not any(m.get("id") == model and (m.get("max_model_len") or 0) >= ctx for m in models):
        fail.append(f"/v1/models {res['models']}: need {model} with max_model_len >= {ctx}")
    kw = {"thinking": True, "reasoning_effort": 50}
    tools = [{"type": "function", "function": {
        "name": "run_python", "description": "Run Python code in a persistent session and return stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
    mark = "ZEBRA-THINKING-MARKER-42"
    try:
        for field in ("reasoning_content", "reasoning"):
            msgs = [{"role": "user", "content": "What is 2+3? Use the tool."},
                    {"role": "assistant", "content": "", field: f"I should compute it. {mark}",
                     "tool_calls": [{"id": "c1", "type": "function", "function": {
                         "name": "run_python", "arguments": json.dumps({"code": "print(2+3)"})}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "5"}]
            tk = post("/tokenize", {"model": model, "messages": msgs, "tools": tools, "chat_template_kwargs": kw,
                                    "add_generation_prompt": True})
            dt = post("/detokenize", {"model": model, "tokens": tk["tokens"]})
            res[f"thinking_via_{field}"] = mark in dt["prompt"]
            res[f"tokenize_{field}_n"] = tk.get("count")
    except Exception as exc:
        fail.append(f"/tokenize or /detokenize: {type(exc).__name__}: {exc}"[:300])
    try:
        msgs = [{"role": "system", "content": "You solve tasks with the run_python tool."},
                {"role": "user", "content": "Compute the sum of the squares of 1..10 with run_python."}]
        tk = post("/tokenize", {"model": model, "messages": msgs, "tools": tools, "chat_template_kwargs": kw,
                                "add_generation_prompt": True})
        r = post("/v1/completions", {"model": model, "prompt": tk["tokens"], "max_tokens": 64, "temperature": 0,
                                     "return_token_ids": True, "skip_special_tokens": False,
                                     "cache_salt": "b32m-contract"})
        res["completion_token_ids"] = len(r["choices"][0].get("token_ids") or [])
        if not res["completion_token_ids"]:
            fail.append("token-id /v1/completions returned no token_ids")
    except Exception as exc:
        fail.append(f"token-id /v1/completions: {type(exc).__name__}: {exc}"[:300])
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            m = r.read().decode(errors="replace")
        res["metrics"] = {k: bool(re.search(rf"^vllm:{k}\b", m, re.M)) for k in (
            "kv_cache_usage_perc", "num_requests_running", "num_requests_waiting", "generation_tokens_total",
            "prefix_cache_hits_total", "num_preemptions_total")}
        if not res["metrics"]["kv_cache_usage_perc"]:
            fail.append("/metrics lacks vllm:kv_cache_usage_perc (KV gate)")
    except Exception as exc:
        fail.append(f"/metrics: {type(exc).__name__}: {exc}"[:300])
    try:
        c = post("/v1/chat/completions", {"model": model, "messages": msgs, "tools": tools, "temperature": 0,
                                          "max_tokens": 1500, "chat_template_kwargs": kw})
        res["chat_tool_calls"] = len(c["choices"][0]["message"].get("tool_calls") or [])
    except Exception as exc:
        res["chat_tool_calls"] = f"{type(exc).__name__}"
    if os.environ.get("ARC_GRID_FORMAT") == "text+image":  # E066: the image harness runs on the chat path
        if not (isinstance(res["chat_tool_calls"], int) and res["chat_tool_calls"] >= 1):
            fail.append(f"chat tool calls not parsed ({res['chat_tool_calls']}): the image harness needs the "
                        "server's --tool-call-parser deepseek_v41 --enable-auto-tool-choice")
        try:  # one token-aligned grid image next to its text, as the harness sends it
            sys.path.insert(0, f"{KIT}/repo/evals")
            from arc_harness.grid_image import aligned_cell, data_uri
            g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            text = {"type": "text", "text": "Describe this 3x3 grid in one sentence."}
            img = {"type": "image_url", "image_url": {"url": data_uri(g, aligned_cell(g, "deepseek"))}}
            n = []
            for content in ([text], [text, img]):
                c = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": content}],
                                                  "temperature": 0, "max_tokens": 16, "chat_template_kwargs": kw})
                n.append(c["usage"]["prompt_tokens"])
            res["image_prompt_tokens"] = n[1] - n[0]
            if res["image_prompt_tokens"] <= 0:
                fail.append(f"an image added no prompt tokens ({n}): the server dropped it")
        except Exception as exc:
            fail.append(f"image chat request: {type(exc).__name__}: {exc}"[:300])
    return {**res, "fail": fail, "done": True}


def contract_thread():
    try:
        out = contract_run(f"http://127.0.0.1:{os.environ.get('ARC_PORT', '8100')}")
    except Exception as exc:
        out = {"fail": [f"contract check crashed: {type(exc).__name__}: {exc}"], "done": True}
    with open(CONTRACT_OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("B32-M contract:", out, flush=True)


if not RERUN and MODE != "fallback":
    threading.Thread(target=contract_thread, daemon=True).start()
    print("B32-M contract check started (background)", flush=True)
