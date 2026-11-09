"""E062 L4: API checks of a running server before the replica (stdlib only; run on the pod).

1. /tokenize of a 2-turn transcript whose assistant turn carries reasoning_content: the thinking text must appear in
   the rendered prompt (patch 02), and the same with the field named `reasoning`.
2. A greedy chat completion with B32's tools: a tool call must parse (deepseek_v41 tool parser) and the reasoning
   must come back separately.
3. Token-id /v1/completions (the harness's --token-exact path) on the rendered prompt ids.
Usage: python -I api_check.py [base_url]
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


TOOLS = [{"type": "function", "function": {
    "name": "run_python", "description": "Run Python code in a persistent session and return stdout.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
MARK = "ZEBRA-THINKING-MARKER-42"
res = {}
for field in ("reasoning_content", "reasoning"):
    msgs = [{"role": "user", "content": "What is 2+3? Use the tool."},
            {"role": "assistant", "content": "", field: f"I should compute it. {MARK}",
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_python", "arguments": json.dumps({"code": "print(2+3)"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "5"}]
    t = post("/tokenize", {"model": "dsv41", "messages": msgs, "tools": TOOLS, "add_generation_prompt": True})
    d = post("/detokenize", {"model": "dsv41", "tokens": t["tokens"]})
    res[f"tokenize_{field}_has_thinking"] = MARK in d["prompt"]
    res[f"tokenize_{field}_n"] = t["count"]
print(json.dumps(res))

msgs = [{"role": "system", "content": "You solve tasks with the run_python tool."},
        {"role": "user", "content": "Compute the sum of the squares of 1..10 with run_python, then report it."}]
c = post("/v1/chat/completions", {"model": "dsv41", "messages": msgs, "tools": TOOLS, "temperature": 0, "max_tokens": 1500})
m = c["choices"][0]["message"]
print(json.dumps({"finish": c["choices"][0]["finish_reason"], "tool_calls": [tc["function"] for tc in (m.get("tool_calls") or [])],
                  "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
                  "content": (m.get("content") or "")[:200], "usage": c.get("usage")}))

t = post("/tokenize", {"model": "dsv41", "messages": msgs, "tools": TOOLS, "add_generation_prompt": True})
r = post("/v1/completions", {"model": "dsv41", "prompt": t["tokens"], "max_tokens": 64, "temperature": 0,
                             "return_token_ids": True})
ch = r["choices"][0]
print(json.dumps({"completions_ids_ok": bool(ch.get("token_ids")), "text_head": ch["text"][:160], "prompt_n": t["count"]}))
