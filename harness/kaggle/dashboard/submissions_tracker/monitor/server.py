#!/usr/bin/env python3
"""Local ARC monitor: atomic API snapshots, background pod polling, no remote mutations."""
import argparse
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

from collector import Collector
from model import DASH, read_json

HERE = Path(__file__).resolve().parent


def handler_for(collector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Request paths and query strings are not logged.
            pass

        def reply(self, data, content_type, code=200, etag=None, legacy=False):
            if etag and self.headers.get('If-None-Match') == etag:
                self.send_response(304); self.send_header('ETag', etag); self.end_headers(); return
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            policy = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self'" if legacy else
                      "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'self'")
            self.send_header('Content-Security-Policy', policy)
            if etag:
                self.send_header('ETag', etag)
            if 'gzip' in self.headers.get('Accept-Encoding', '') and len(data) > 1000:
                data = gzip.compress(data, compresslevel=3)
                self.send_header('Content-Encoding', 'gzip'); self.send_header('Vary', 'Accept-Encoding')
            self.send_header('Content-Length', str(len(data))); self.end_headers()
            self.wfile.write(data)

        def json(self, data, code=200, etag=None):
            self.reply(json.dumps(data, separators=(',', ':'), allow_nan=False).encode(),
                       'application/json; charset=utf-8', code, etag)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/snapshot':
                data = collector.get()
                tag = '"'+str(data.get('generated_at'))+'"'
                return self.json(data, etag=tag)
            if path == '/api/health':
                return self.json(collector.health())
            if path == '/favicon.ico':
                return self.reply(b'', 'image/x-icon', 204)
            assets = {'/': 'dashboard.html', '/local.html': 'dashboard.html', '/index.html': 'dashboard.html',
                      '/dashboard.html': 'dashboard.html', '/dashboard.js': 'dashboard.js', '/dashboard.css': 'dashboard.css'}
            if path in assets:
                p = HERE/assets[path]
                types = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css'}
                return self.reply(p.read_bytes(), types[p.suffix]+'; charset=utf-8')
            archive = read_json(HERE/'archive.json').get('directory', '')
            if archive and path.startswith('/'+archive+'/'):
                p = (DASH/path.lstrip('/')).resolve()
                if (DASH/archive).resolve() in p.parents and p.is_file() and p.suffix in ('.html', '.json'):
                    return self.reply(p.read_bytes(), ('text/html' if p.suffix == '.html' else 'application/json')+'; charset=utf-8', legacy=True)
            return self.json({'error': 'not found'}, 404)

        def do_POST(self):
            if urlsplit(self.path).path != '/api/refresh':
                return self.json({'error': 'not found'}, 404)
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self.json({'error': 'same-origin request required'}, 403)
            collector.request.set()
            return self.json({'accepted': True}, 202)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8765); ap.add_argument('--bind', default='0.0.0.0')
    ap.add_argument('--interval', type=int, default=60); ap.add_argument('--state-dir', type=Path, default=HERE/'state')
    ap.add_argument('--local-only', action='store_true', help='Do not make Runpod or SSH requests')
    ap.add_argument('--once', action='store_true', help='Collect once and print a short summary')
    a = ap.parse_args()
    c = Collector(a.state_dir, a.interval, not a.local_only)
    if a.once:
        s = c.collect()
        print(json.dumps({'generated_at': s['generated_at'], 'pods': [
            {k: p.get(k) for k in ('id', 'name', 'connected', 'observed_at', 'error')} for p in s['pods']],
            'runs': [{k: r.get(k) for k in ('name', 'state', 'tasks', 'telemetry_at')} for r in s['runs']],
            'errors': s['errors']}, indent=2))
        return
    server = ThreadingHTTPServer((a.bind, a.port), handler_for(c))
    c.start()
    a.state_dir.mkdir(parents=True, exist_ok=True)
    import os
    (a.state_dir/'server.pid').write_text(str(os.getpid())+'\n')
    def stop(signum, frame):
        c.stop.set(); c.request.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(f'ARC monitor listening on {a.bind}:{a.port}; pod poll {a.interval}s', flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        (a.state_dir/'server.pid').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
