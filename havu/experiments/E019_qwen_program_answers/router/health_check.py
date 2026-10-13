"""Health-check SGLang workers (or the router) listed in ENDPOINTS.yaml. Tokens are never printed.

    .venv/bin/python experiments/E019_qwen_program_answers/router/health_check.py vastai-1x6000-au vastai-1x6000-cn ...

Per endpoint: /health RTT (median of 5), /v1/models, server info (context length, max running requests,
radix cache, KV dtype, max_total_num_tokens), and one short chat completion with a tool call.
"""
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def endpoint(name):
    for e in yaml.safe_load((ROOT / "ENDPOINTS.yaml").read_text()):
        if e.get("name") == name:
            return f"http://{e['addr']}", e.get("token") or ""
    raise KeyError(name)


def call(base, token, path, body=None, timeout=60):
    hdr = {"Content-Type": "application/json"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=hdr)
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    dt = time.time() - t
    try:
        return json.loads(raw), dt
    except json.JSONDecodeError:
        return raw, dt


TOOLS = [{"type": "function", "function": {
    "name": "python", "description": "Run Python code and return stdout.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]


def check(name):
    base, token = endpoint(name)
    out = {"name": name}
    rtts = []
    for _ in range(5):
        try:
            _, dt = call(base, token, "/health", timeout=20)
            rtts.append(dt)
        except Exception as e:  # noqa: BLE001
            out.setdefault("health_errors", []).append(type(e).__name__)
    out["health_rtt_ms_median"] = round(1000 * statistics.median(rtts)) if rtts else None
    models, _ = call(base, token, "/v1/models")
    out["models"] = [m["id"] for m in models["data"]]
    out["max_model_len"] = models["data"][0].get("max_model_len")
    info = None
    for p in ("/get_server_info", "/server_info"):
        try:
            info, _ = call(base, token, p)
            break
        except Exception:  # noqa: BLE001
            continue
    if isinstance(info, dict):
        st = (info.get("internal_states") or [{}])[0]
        out["server"] = {k: info.get(k) for k in (
            "model_path", "quantization", "kv_cache_dtype", "tp_size", "context_length", "max_running_requests",
            "disable_radix_cache", "chunked_prefill_size", "mem_fraction_static", "schedule_policy",
            "attention_backend", "reasoning_parser", "tool_call_parser", "version", "enable_metrics",
            "page_size", "max_prefill_tokens")}
        out["server"]["max_total_num_tokens"] = st.get("max_total_num_tokens", info.get("max_total_num_tokens"))
        out["server"]["effective_max_running_requests"] = st.get("max_running_requests")
    body = {"model": out["models"][0], "max_tokens": 1500, "temperature": 0.6, "tools": TOOLS,
            "messages": [{"role": "user", "content": "Use the python tool to compute 1234*5678. Call the tool."}],
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    t = time.time()
    r, dt = call(base, token, "/v1/chat/completions", body, timeout=300)
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    out["chat_tool"] = {"s": round(dt, 1), "finish": r["choices"][0]["finish_reason"],
                        "tool_calls": [(c["function"]["name"], c["function"]["arguments"][:80]) for c in tc],
                        "completion_tokens": r["usage"]["completion_tokens"],
                        "tok_s": round(r["usage"]["completion_tokens"] / dt, 1)}
    return out


if __name__ == "__main__":
    for n in sys.argv[1:]:
        try:
            print(json.dumps(check(n), indent=1), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"name": n, "error": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
