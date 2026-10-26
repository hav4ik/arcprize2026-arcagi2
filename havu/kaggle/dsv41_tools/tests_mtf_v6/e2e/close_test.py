"""Does Stream.close() from another thread wake a thread blocked reading a stalled SSE stream (as force_sweep does)?"""
import socket, threading, time, http.server, sys
import httpx
from openai import OpenAI

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        d = b'data: {"id":"x","object":"text_completion","created":1,"model":"m","choices":[{"index":0,"text":"a","finish_reason":null,"logprobs":null}]}\n\n'
        self.wfile.write(hex(len(d))[2:].encode() + b"\r\n" + d + b"\r\n"); self.wfile.flush()
        time.sleep(30)  # stall
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
mode = sys.argv[1]
c = OpenAI(api_key="x", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0, timeout=httpx.Timeout(7200, connect=30, read=900))
s = c.completions.create(model="m", prompt=[1, 2, 3], max_tokens=5, stream=True)
out = {}
def reader():
    t = time.time()
    try:
        for ch in s:
            pass
        out["r"] = f"ended normally after {time.time() - t:.1f}s"
    except Exception as exc:
        out["r"] = f"raised {type(exc).__name__} after {time.time() - t:.1f}s"
th = threading.Thread(target=reader, daemon=True); th.start()
time.sleep(1)
t = time.time()
if mode == "close":
    s.close()  # what GuardedStream.close() / force_sweep does
elif mode == "shutdown":
    stream = s.response.extensions["network_stream"]
    stream.get_extra_info("socket").shutdown(socket.SHUT_RDWR)
    s.close()
th.join(10)
print(f"{mode}: reader {'still blocked 10 s after close' if th.is_alive() else out['r']} (close took {time.time() - t:.2f}s incl. join)")
