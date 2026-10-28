#!/usr/bin/env python3
"""E060: verify on the server's rendered prompt (POST /tokenize with messages, then /detokenize) that earlier turns'
reasoning is kept, for a request body like the harness sends (tools, an assistant turn with reasoning_content and a
tool call, its tool result, a second assistant turn, a tool result). Prints, per chat_template_kwargs variant, whether
each turn's reasoning marker appears in the rendered prompt, the reasoning-effort system line, and the token count.

    python check_render.py --base-url http://127.0.0.1:18321 [--trace TRACE.messages.json.gz]
With --trace, renders the first assistant turns of a real harness trace instead of the synthetic one.
"""
import argparse
import gzip
import json
import urllib.request

TOOLS = [{"type": "function", "function": {"name": "run_python", "description": "Run Python.",
                                           "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                          "required": ["code"]}}}]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def synthetic():
    call = lambda i: {"id": f"call_{i}", "type": "function",
                      "function": {"name": "run_python", "arguments": json.dumps({"code": f"print({i})"})}}
    return [
        {"role": "user", "content": "Solve the task."},
        {"role": "assistant", "content": "", "reasoning_content": "MARKER_REASONING_TURN_1 think think",
         "tool_calls": [call(1)]},
        {"role": "tool", "tool_call_id": "call_1", "content": "1"},
        {"role": "assistant", "content": "", "reasoning_content": "MARKER_REASONING_TURN_2 more thinking",
         "tool_calls": [call(2)]},
        {"role": "tool", "tool_call_id": "call_2", "content": "2"},
    ], ["MARKER_REASONING_TURN_1", "MARKER_REASONING_TURN_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18321")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--trace")
    args = ap.parse_args()
    if args.trace:
        msgs = json.load(gzip.open(args.trace))["messages"]
        cut = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][:3]
        msgs = msgs[:cut[-1] + 2]
        markers = [m["reasoning_content"][:60] for m in msgs if m["role"] == "assistant" and m.get("reasoning_content")]
    else:
        msgs, markers = synthetic()
    for name, kw in (("default (no kwargs)", None), ("preserve_thinking=true", {"preserve_thinking": True}),
                     ("preserve_thinking=false", {"preserve_thinking": False})):
        body = {"model": args.model, "messages": msgs, "tools": TOOLS, "add_generation_prompt": True}
        if kw is not None:
            body["chat_template_kwargs"] = kw
        tok = post(args.base_url + "/tokenize", body)
        text = post(args.base_url + "/detokenize", {"model": args.model, "tokens": tok["tokens"]})["prompt"]
        kept = [m[:30] in text for m in markers]
        effort = [ln for ln in text.splitlines() if "Reasoning effort" in ln][:1]
        print(f"{name}: {tok['count']} tokens; reasoning kept per turn {kept}; effort line {effort}; "
              f"ends with {text[-40:]!r}")


if __name__ == "__main__":
    main()
