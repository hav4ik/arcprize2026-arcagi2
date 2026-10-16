"""Minimal OpenAI-compatible chat client for a local vLLM server (stdlib only).

Returns a normalised dict: content, reasoning, tool_calls [{id, name, arguments}],
finish_reason, prompt_tokens, completion_tokens. If the server's tool parser
misses a call that the model wrote as qwen3_coder markup in its text, the call
is recovered from the markup, as the Tufa harness does.
"""
import json
import re
import urllib.error
import urllib.request

TOOL_BLOCK = re.compile(r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([\w\-]+)>\n?(.*?)\n?</parameter>", re.S)


def recover_markup(text):
    calls = []
    for name, body in TOOL_BLOCK.findall(text or ""):
        args = {k: v for k, v in PARAM.findall(body)}
        calls.append({"id": f"markup-{len(calls) + 1}", "name": name, "arguments": args})
    return calls


def strip_markup(text):
    return TOOL_BLOCK.sub("", text or "").strip()


class ChatClient:
    def __init__(self, base_url, model, temperature=0.6, top_p=0.95, top_k=20,
                 max_tokens=16384, thinking=True, timeout=1800, reasoning_effort=None,
                 thinking_token_budget=None, template_kwargs=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        # top_k=None leaves it to the server (gpt-oss wants no top-k).
        self.sampling = {k: v for k, v in {"temperature": temperature, "top_p": top_p,
                                           "top_k": top_k}.items() if v is not None}
        self.reasoning_effort = reasoning_effort
        # vLLM >= 0.19 forces </think> after this many reasoning tokens (needs the server
        # flag --reasoning-config). Qwen3.8 otherwise spends whole 16k turns thinking.
        self.thinking_token_budget = thinking_token_budget
        # Extra chat-template kwargs, e.g. Qwen3.8's reasoning_effort (xhigh default /
        # medium / low) -- the template itself turns it into a system instruction.
        self.template_kwargs = template_kwargs or {}
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.timeout = timeout

    def chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None,
             tool_choice=None, template_kwargs=None):
        # template_kwargs: per-request overrides on top of the client's own, e.g. the
        # reasoning-effort ladder stepping one turn down to "medium".
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens or self.max_tokens,
                   "chat_template_kwargs": {"enable_thinking": self.thinking, **self.template_kwargs,
                                            **(template_kwargs or {})},
                   **self.sampling}
        if seed is not None:
            payload["seed"] = seed
        if self.reasoning_effort:          # gpt-oss: low / medium / high
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking_token_budget:
            payload["thinking_token_budget"] = self.thinking_token_budget
        if tools:
            payload["tools"] = tools
            # A named tool forces that call (vLLM guided decoding); default lets it choose.
            payload["tool_choice"] = ({"type": "function", "function": {"name": tool_choice}}
                                      if tool_choice else "auto")
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                body = json.loads(r.read().decode())
            return normalise(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            overflow = "maximum context length" in detail or "context length" in detail
            if (e.code == 400 and not overflow and "thinking_token_budget" in payload
                    and any(w in detail.lower() for w in ("thinking", "budget", "reasoning"))):
                # A server started without --reasoning-config (the last rungs of the startup
                # ladder) may refuse the budget: drop it for every later call, not fail them all.
                self.thinking_token_budget = None
                return self.chat(messages, tools, max_tokens, seed, timeout, tool_choice, template_kwargs)
            return {"error": f"HTTP {e.code}: {detail}", "context_overflow": overflow}
        except Exception as e:
            # probe24 lost every result to one uncaught socket TimeoutError.
            text = f"{type(e).__name__}: {e}"
            return {"error": text[:300], "context_overflow": False,
                    "timeout": isinstance(e, TimeoutError) or "timed out" in text}


def normalise(body):
    ch = body["choices"][0]
    msg = ch.get("message", {}) or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}
        calls.append({"id": tc.get("id") or f"call-{len(calls) + 1}",
                      "name": fn.get("name"), "arguments": args})
    if not calls:
        calls = recover_markup(content) or recover_markup(reasoning)
        content = strip_markup(content)
    usage = body.get("usage") or {}
    return {"content": content, "reasoning": reasoning, "tool_calls": calls,
            "finish_reason": ch.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)}
