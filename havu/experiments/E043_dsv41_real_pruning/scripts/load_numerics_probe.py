#!/usr/bin/env python3
"""E043: does heavy decode load change the model's numbers? (E112 attempt-1 drop under ~130 streams per server)

  load_numerics_probe.py decode --server URL --n 16 --tokens 256 --out dec_load.jsonl   # under load, then at idle
  load_numerics_probe.py score  --server URL --inp dec_load.jsonl --out score.jsonl      # at idle

decode: for N tool dev prompts (rendered by the server's /tokenize, harness tools, effort 50), greedy-decode T tokens
via /v1/completions with logprobs=0, keeping each sampled token's logprob as computed on the decode path (in whatever
batch the server is running at that moment).
score: teacher-force prompt + generated ids in one prefill (prompt_logprobs=0, fresh cache_salt so nothing is reused)
and compare each generated token's prefill logprob with its decode logprob. With correct kernels the two agree to
numerical noise; a load-dependent decode bug shows up as a larger gap under load than in the idle control.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(__file__))
from tokenize_rollouts import harness_tools  # noqa: E402

KW = {"thinking": True, "reasoning_effort": 50}


def running(server):
    try:
        t = requests.get(f"{server}/metrics", timeout=10).text
        return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith("vllm:num_requests_running"))
    except Exception:
        return -1


def decode(a):
    rows = pq.read_table(a.parquet, columns=["messages"]).to_pylist()
    tools = harness_tools()

    def one(k):
        msgs = json.loads(rows[k % len(rows)]["messages"])[:2]
        ids = requests.post(f"{a.server}/tokenize", json={"model": "dsv41", "messages": msgs, "tools": tools,
                                                          "chat_template_kwargs": KW, "add_generation_prompt": True},
                            timeout=600).json()["tokens"]
        load0 = running(a.server)
        r = requests.post(f"{a.server}/v1/completions", json={
            "model": "dsv41", "prompt": ids, "max_tokens": a.tokens, "temperature": 0.0, "logprobs": 0,
            "return_token_ids": True, "skip_special_tokens": False, "cache_salt": f"lnp-dec-{time.time()}-{k}"},
            timeout=3600).json()
        ch = r["choices"][0]
        return {"probe": k, "prompt_ids": ids, "gen_ids": ch["token_ids"],
                "decode_lp": ch["logprobs"]["token_logprobs"], "running_at_start": load0,
                "running_at_end": running(a.server)}

    with ThreadPoolExecutor(a.n) as pool:
        outs = list(pool.map(one, range(a.n)))
    with open(a.out, "w") as f:
        for o in outs:
            f.write(json.dumps(o) + "\n")
    print(f"decoded {len(outs)} x {a.tokens} tokens; server running requests at start "
          f"{[o['running_at_start'] for o in outs][:4]}..., at end {[o['running_at_end'] for o in outs][:4]}...")


def score(a):
    rows = [json.loads(l) for l in open(a.inp)]
    gaps, outs = [], []
    for r in rows:
        ids = r["prompt_ids"] + r["gen_ids"]
        res = requests.post(f"{a.server}/v1/completions", json={
            "model": "dsv41", "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0,
            "cache_salt": f"lnp-score-{time.time()}"}, timeout=3600).json()
        plp = (res["choices"][0].get("prompt_logprobs") or res.get("prompt_logprobs"))[len(r["prompt_ids"]):]
        pre = [next(iter(v["logprob"] for k, v in d.items() if int(k) == t), None) if d else None
               for d, t in zip(plp, r["gen_ids"])]
        g = [abs(x - y) for x, y in zip(r["decode_lp"], pre) if x is not None and y is not None]
        gaps += g
        outs.append({"probe": r["probe"], "mean_abs_gap": sum(g) / max(1, len(g)), "max_gap": max(g or [0]),
                     "running_at_start": r["running_at_start"]})
    with open(a.out, "w") as f:
        for o in outs:
            f.write(json.dumps(o) + "\n")
    gaps.sort()
    print(f"{a.inp}: {len(gaps)} tokens, mean |decode - prefill| logprob {sum(gaps) / len(gaps):.4f}, "
          f"p50 {gaps[len(gaps) // 2]:.4f}, p99 {gaps[int(0.99 * len(gaps))]:.4f}, max {gaps[-1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["decode", "score"])
    ap.add_argument("--server", default="http://127.0.0.1:18001")
    ap.add_argument("--parquet", default="/workspace/dsv4/data/tool_dev_probe.parquet")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--inp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    decode(a) if a.mode == "decode" else score(a)


if __name__ == "__main__":
    main()
