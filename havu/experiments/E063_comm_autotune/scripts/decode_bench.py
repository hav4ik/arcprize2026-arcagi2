"""E063: decode-throughput window against a running vLLM server.

Usage: python decode_bench.py PORT CONCURRENCY PROMPT_TOKENS GEN_TOKENS OUT.json
Sends CONCURRENCY requests at once to /v1/completions, each a distinct random
token-id prompt (no prefix hits), greedy, ignore_eos, max_tokens GEN_TOKENS.
Samples vllm:generation_tokens_total and num_requests_running every 2 s; the
steady window is the span while all requests run (prefill done). Reports
gen tok/s over it, and the mean decode step time = window / tokens per stream.
"""
import json, random, sys, threading, time, urllib.request

port, conc, plen, glen, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
base = f"http://127.0.0.1:{port}"


def metrics():
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    gen = run = 0.0
    for line in txt.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            gen += float(line.split()[-1])
        elif line.startswith("vllm:num_requests_running"):
            run += float(line.split()[-1])
    return gen, run


def one(i, res):
    rng = random.Random(1000 + i)
    body = {"model": "dsv41", "prompt": [rng.randrange(1000, 100000) for _ in range(plen)],
            "max_tokens": glen, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    res[i] = (time.time() - t, r["usage"]["completion_tokens"])


res = {}
threads = [threading.Thread(target=one, args=(i, res)) for i in range(conc)]
t0 = time.time()
for th in threads:
    th.start()
samples = []
while any(th.is_alive() for th in threads):
    g, r = metrics()
    samples.append((time.time() - t0, g, r))
    time.sleep(2)
wall = time.time() - t0
steady = [s for s in samples if s[2] >= conc]
win = (steady[-1][0] - steady[0][0], steady[-1][1] - steady[0][1]) if len(steady) > 2 else (0, 0)
rate = win[1] / win[0] if win[0] else 0.0
rep = {"concurrency": conc, "prompt_tokens": plen, "gen_tokens": glen, "wall_s": wall,
       "total_gen": sum(v[1] for v in res.values()), "overall_tok_s": sum(v[1] for v in res.values()) / wall,
       "steady_window_s": win[0], "steady_tok_s": rate,
       "steady_step_ms": 1000 * conc / rate if rate else None, "samples": samples}
json.dump(rep, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "samples"}))
