"""E034 cache diagnosis on the E032 server (vLLM 0.23, --enable-prompt-tokens-details): is the low prefix-cache hit rate
caused by (1) generated tokens not being reusable, (2) KV pressure/eviction, or (3) a token-level prompt mismatch?
Uses real E032 first prompts from /root/arc/runs/e032_probe48 traces. Usage: python cache_diag.py <idle|load> <out.json>"""
import json, sys, glob, re, time, threading, urllib.request
URL = "http://127.0.0.1:8001/v1/completions"
mode, out_path = sys.argv[1], sys.argv[2]
traces = sorted(glob.glob("/root/arc/e032_traces/trace_*.txt"))
prompts = []
for f in traces:
    t = open(f).read()
    i = t.find("<|im_start|>assistant\n")
    if i > 0: prompts.append(t[:i + len("<|im_start|>assistant\n")])
def comp(prompt, n, temp=1.0):
    body = json.dumps({"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": n, "temperature": temp, "top_p": 0.95, "top_k": 20,
                       "skip_special_tokens": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}), timeout=3600))
    u = r["usage"]; d = u.get("prompt_tokens_details") or {}
    return r["choices"][0]["text"], u["prompt_tokens"], d.get("cached_tokens"), u["completion_tokens"]
def metrics():
    m = urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=30).read().decode()
    g = lambda k: sum(float(x) for x in re.findall(rf"^{k}(?:{{[^}}]*}})? ([0-9.e+]+)$", m, re.M))
    return {k: g(k) for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:num_preemptions_total",
                             "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc", "vllm:num_requests_running")}
stop = threading.Event()
def background(i):  # load: long generations on other prompts, like 24 concurrent attempts
    while not stop.is_set():
        try: comp(prompts[(i + 7) % len(prompts)] + f"\n<!-- bg {i} {time.time()} -->", 6000)
        except Exception: time.sleep(5)
bg = []
if mode == "load":
    for i in range(23):
        th = threading.Thread(target=background, args=(i,), daemon=True); th.start(); bg.append(th)
    time.sleep(240)  # let the load reach steady state (KV fills)
res = {"mode": mode, "before": metrics(), "probes": []}
TOOL = "\n</think>\n\n<tool_call>\n<function=run_python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n1\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
for k, P in enumerate(prompts[:4]):
    for n in (500, 2000, 8000):
        P0 = P + f"<think>\nprobe {k}-{n}\n"  # unique prefix tail so earlier probes don't pre-warm this one
        _, pt0, c0, _ = comp(P0, 1)                       # 1) cold prompt
        _, pt0b, c0b, _ = comp(P0, 1)                      # 2) the same prompt again: attention + DeltaNet state reuse
        gen, _, _, ng = comp(P0, n)                        # 3) generate n tokens
        _, pt1, c1, _ = comp(P0 + gen + TOOL, 1)           # 4) next turn = prompt + generated + tool output
        res["probes"].append(dict(prompt=k, n=n, prompt_tokens=pt0, cached_cold=c0, cached_repeat=c0b, generated=ng,
                                  next_prompt_tokens=pt1, cached_next=c1,
                                  reuse_of_prev_prompt=round((c1 or 0) / pt0, 3), share_of_next=round((c1 or 0) / pt1, 3)))
        print(json.dumps(res["probes"][-1]), flush=True)
res["after"] = metrics()
stop.set()
json.dump(res, open(out_path, "w"), indent=1)
print("DIAG_DONE", mode)
