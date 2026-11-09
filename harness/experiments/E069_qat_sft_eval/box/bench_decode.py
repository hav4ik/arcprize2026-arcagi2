#!/usr/bin/env python3
"""E068: quick aggregate decode throughput at N concurrent streams (thinking on, effort 50, T=1, distinct prompts so
no prefix sharing beyond the template). Usage: bench_decode.py [--url http://127.0.0.1:18000] [--n 32] [--max-tokens 2048]"""
import argparse
import json
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:18000")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=2048)
a = ap.parse_args()
res = []


def one(i):
    body = {"model": "dsv41", "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 1.0, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": 50}, "cache_salt": f"bench{time.time()}_{i}",
            "messages": [{"role": "user", "content": f"Request {i}: write a long, careful step-by-step analysis of how "
                          f"to detect symmetric objects in a 30x30 grid of digits 0-9, with Python code."}]}
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        u = json.load(r)["usage"]
    res.append((u["completion_tokens"], time.time() - t))


t0 = time.time()
th = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
for x in th:
    x.start()
for x in th:
    x.join()
dt = time.time() - t0
tok = sum(r[0] for r in res)
print(f"{a.n} streams x {a.max_tokens} max: {tok} tokens in {dt:.1f}s = {tok / dt:.0f} tok/s aggregate, "
      f"{tok / dt / a.n:.1f} tok/s per stream")
