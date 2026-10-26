"""Call the pinned solver's server_metrics() / kv_admits() against /metrics bodies served locally."""
import http.server, importlib.util, os, sys, threading, traceback
BODIES = {
    "valid": 'vllm:kv_cache_usage_perc{engine="0"} 0.3\nvllm:num_requests_running{engine="0"} 4.0\n',
    "name_only_line": 'vllm:kv_cache_usage_perc{engine="0"} 0.3\nvllm:num_requests_waiting\n',
    "value_not_number": 'vllm:kv_cache_usage_perc{engine="0"} abc\n',
    "nan": 'vllm:kv_cache_usage_perc{engine="0"} NaN\n',
    "inf": 'vllm:kv_cache_usage_perc{engine="0"} +Inf\n',
    "empty": '',
    "binary": '\x00\xff\xfe vllm:\n',
}
cur = {"b": ""}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = cur["b"].encode("latin-1"); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ.update(ARC_PORT=str(srv.server_address[1]), ARC_OUT=sys.argv[2], ARC_KV_ADMIT_MAX="0.6", ARC_KV_ADMIT_MIN_WORKERS="16")
spec = importlib.util.spec_from_file_location("solver_pinned", sys.argv[1]); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
S.LOG.mkdir(parents=True, exist_ok=True)
for name, body in BODIES.items():
    cur["b"] = body; S.STATE.pop("kv_t", None)
    try:
        m = S.server_metrics(); ok = S.kv_admits(20)
        print(f"{name:18s} metrics={ {k: v for k, v in (m or {}).items() if k != 't'} } kv_admits(20)={ok}")
    except Exception as exc:
        print(f"{name:18s} RAISES {type(exc).__name__}: {exc}  <- from {traceback.extract_tb(exc.__traceback__)[-1].line}")
