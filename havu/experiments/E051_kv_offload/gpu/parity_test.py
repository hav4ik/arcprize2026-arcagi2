#!/usr/bin/env python3
"""GPU parity test for [dsv41-offload]: a prefix restored from host RAM must give
bit-identical greedy tokens and logprobs to the same prefix hit in the GPU cache.

Needs a server from gpu/serve_dsv41_offload.sh (VLLM_SERVER_DEV_MODE=1 for
/reset_prefix_cache) and NO other traffic: batch composition changes numerics.
Stdlib only.

    python gpu/parity_test.py --url http://127.0.0.1:8100 --lengths 3000 20000 60000

Per prompt length L (random token ids, fixed seed):
  1. reset GPU + host caches; cold run of P (stores P to host);
  2. GPU-hit run of P                         -> out_gpu, cached_gpu;
  3. reset GPU cache only; host-restore run   -> out_host, cached_host;
     require cached_host == cached_gpu == round_down(L - 1, 128), an external
     hit in /metrics, and out_host == out_gpu (token ids AND logprobs, exact);
  4. decode-resume case: P2 = P + sampled ids + 200 "tool" ids. Host path:
     cold P, reset GPU, run P2 (restores the finished request's rolling
     checkpoint). GPU path: cold P again, run P2 (GPU hit at the same point).
     Same checks; the cold run must also be deterministic.
Exit status 0 only if every check passes. Writes a JSON report with --json.
"""

import argparse
import json
import random
import sys
import time
import urllib.request

ALIGN = 128


def http(url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if body is not None or url.endswith(("reset_prefix_cache", "true")) else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def metrics(url):
    out = {}
    for line in http(f"{url}/metrics").splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name.startswith(("vllm:kv_offload", "vllm:external_prefix_cache",
                            "vllm:prefix_cache", "vllm:num_preemptions")):
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def wait_stores_drained(url, quiet_s=3.0, timeout=120):
    key = next((k for k in metrics(url) if k.startswith("vllm:kv_offload_store_bytes")), None)
    if key is None:
        time.sleep(quiet_s)
        return
    last, t_last, t0 = None, time.time(), time.time()
    while time.time() - t0 < timeout:
        v = metrics(url).get(key)
        if v != last:
            last, t_last = v, time.time()
        elif time.time() - t_last >= quiet_s:
            return
        time.sleep(0.5)


def complete(a, prompt_ids, max_tokens):
    body = {"model": a.model, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0.0, "logprobs": a.top_logprobs, "ignore_eos": True,
            "return_token_ids": True}
    r = http(f"{a.url}/v1/completions", body)
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    usage = r.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "ids": ch.get("token_ids"),
        "text": ch.get("text"),
        "token_logprobs": lp.get("token_logprobs"),
        "top_logprobs": lp.get("top_logprobs"),
        "cached": cached,
    }


def reset(a, external):
    q = "?reset_external=true" if external else ""
    http(f"{a.url}/reset_prefix_cache{q}", body={})
    time.sleep(1.0)


def same(x, y):
    return (x["ids"] == y["ids"] and x["token_logprobs"] == y["token_logprobs"]
            and x["top_logprobs"] == y["top_logprobs"])


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x["token_logprobs"] or [], y["token_logprobs"] or [])):
        if p != q or x["ids"][i] != y["ids"][i]:
            return {"pos": i, "gpu": [x["ids"][i], p], "host": [y["ids"][i], q]}
    return None


def record(a, name, prompt, expect, out_gpu, out_host, m0, m1, report):
    ext_hits = sum(m1.get(k, 0) - m0.get(k, 0) for k in m1
                   if k.startswith("vllm:external_prefix_cache_hits"))
    load_bytes = sum(m1.get(k, 0) - m0.get(k, 0) for k in m1
                     if k.startswith("vllm:kv_offload_load_bytes"))
    ok = {
        "cached_gpu==expected": out_gpu["cached"] == expect,
        "cached_host==expected": out_host["cached"] == expect,
        "external_hit": ext_hits > 0,
        "load_bytes>0": load_bytes > 0,
        "identical": same(out_gpu, out_host),
    }
    report.append({"case": name, "len": len(prompt), "expected_hit": expect,
                   "cached_gpu": out_gpu["cached"], "cached_host": out_host["cached"],
                   "external_hits": ext_hits, "load_bytes": load_bytes, "checks": ok,
                   "first_diff": None if ok["identical"] else first_diff(out_gpu, out_host)})
    print(json.dumps(report[-1]))
    return all(ok.values())


def prompt_case(a, prompt, report):
    """Resume at the prompt-end checkpoint: round_down(len(P) - 1, 128)."""
    reset(a, external=True)
    cold = complete(a, prompt, a.max_tokens)
    wait_stores_drained(a.url)
    out_gpu = complete(a, prompt, a.max_tokens)  # GPU prefix-cache hit
    reset(a, external=False)
    m0 = metrics(a.url)
    out_host = complete(a, prompt, a.max_tokens)  # host restore
    m1 = metrics(a.url)
    expect = (len(prompt) - 1) // ALIGN * ALIGN
    ok = record(a, f"prompt{len(prompt)}", prompt, expect, out_gpu, out_host, m0, m1, report)
    return ok, cold


def resume_case(a, prompt, cold, rng, report):
    """Next turn = P + sampled ids + tool ids; resume at the rolling checkpoint
    of the finished request: round_down(len(P) + max_tokens - 1, 128)."""
    p2 = prompt + cold["ids"] + [rng.randrange(1000, 100000) for _ in range(200)]
    expect = (len(prompt) + a.max_tokens - 1) // ALIGN * ALIGN
    # Host path: only the finished request's checkpoints are on the host.
    reset(a, external=True)
    again = complete(a, prompt, a.max_tokens)
    deterministic = again["ids"] == cold["ids"]
    wait_stores_drained(a.url)
    reset(a, external=False)
    m0 = metrics(a.url)
    out_host = complete(a, p2, a.max_tokens)
    m1 = metrics(a.url)
    # GPU path: the same finished request, still resident, then the next turn.
    reset(a, external=True)
    complete(a, prompt, a.max_tokens)
    out_gpu = complete(a, p2, a.max_tokens)
    ok = record(a, f"resume{len(prompt)}", p2, expect, out_gpu, out_host, m0, m1, report)
    report[-1]["cold_deterministic"] = deterministic
    return ok and deterministic


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--lengths", type=int, nargs="+", default=[3000, 20000, 60000])
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--top-logprobs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    a = p.parse_args()
    rng = random.Random(a.seed)
    report, all_ok = [], True
    for n in a.lengths:
        prompt = [rng.randrange(1000, 100000) for _ in range(n)]
        ok, cold = prompt_case(a, prompt, report)
        all_ok &= ok
        all_ok &= resume_case(a, prompt, cold, rng, report)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    print("PARITY", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
