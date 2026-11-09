"""Tiny fake of the vLLM endpoints parity_test.py uses (plumbing check only)."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
STATE = {"gpu": set(), "host": set(), "ext": 0.0, "load": 0.0, "store": 0.0}
def hit(prompt, cache):
    n = (len(prompt) - 1) // 128 * 128
    while n > 0 and tuple(prompt[:n]) not in cache:
        n -= 128
    return max(n, 0)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, ctype="application/json"):
        b = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        txt = (f"vllm:external_prefix_cache_hits_total {STATE['ext']}\n"
               f"vllm:kv_offload_load_bytes_total{{transfer_type=\"CPU_to_GPU\"}} {STATE['load']}\n"
               f"vllm:kv_offload_store_bytes_total{{transfer_type=\"GPU_to_CPU\"}} {STATE['store']}\n")
        self._send(txt, "text/plain")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/reset_prefix_cache"):
            STATE["gpu"].clear()
            if "reset_external=true" in self.path: STATE["host"].clear()
            return self._send({"ok": True})
        prompt = body["prompt"]; mt = body["max_tokens"]
        g = hit(prompt, STATE["gpu"]); h = hit(prompt, STATE["host"])
        cached = max(g, h)
        if h > g: STATE["ext"] += h - g; STATE["load"] += 1e6
        out = [(sum(prompt) + i) % 50000 for i in range(mt)]
        full = prompt + out
        for k in range(128, len(full), 128):
            STATE["gpu"].add(tuple(full[:k])); STATE["host"].add(tuple(full[:k])); STATE["store"] += 1
        self._send({"choices": [{"token_ids": out, "text": "", "logprobs": {"token_logprobs": [-0.5] * mt, "top_logprobs": [{"a": -0.5}] * mt}}],
                    "usage": {"prompt_tokens_details": {"cached_tokens": cached}}})
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
