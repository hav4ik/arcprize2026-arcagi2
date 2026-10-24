#!/usr/bin/env python3
"""E042 P4: prefix-caching probe against the running server.

parity: P (3000 random tokens), P again, then P+S (S = 300 more tokens); greedy 24 tokens each with logprobs.
  Prints token ids, logprobs and prefix-cache counter deltas per request, for comparing a cache-on server
  (warm P+S reuses P's blocks, including the Engram n-gram lookback across the boundary) with a cache-off one.
gencache: P, then P + its greedy output + 16 tokens; counts cache hits to see whether generated-token blocks are reused.
replay: C attempts x T turns shaped like tool use: every turn re-sends the whole history (base prompt, every
  earlier output, a random "tool result" per turn), so a cache should serve all but the new tail.
  Reports per-turn latency, total wall time, generated tok/s and the prefix-cache hit rate over the replay.
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def rand_tokens(rng, n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


async def complete(session, a, prompt, max_tokens, logprobs=False):
    body = {"model": a.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "return_token_ids": True, "logprobs": 1, "return_tokens_as_token_ids": True}
    async with session.post(f"{a.url}/v1/completions", json=body) as r:
        r.raise_for_status()
        d = await r.json()
    c = d["choices"][0]
    ids = c.get("token_ids") or [int(t.split(":")[1]) for t in c["logprobs"]["tokens"]]
    return ids, c["logprobs"]["token_logprobs"], d["usage"]


async def cache_counters(session, a):
    async with session.get(f"{a.url}/metrics") as r:
        txt = await r.text()
    q = h = 0.0
    for line in txt.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total"):
            q += float(line.split()[-1])
        elif line.startswith("vllm:prefix_cache_hits_total"):
            h += float(line.split()[-1])
    return q, h


async def parity(session, a):
    rng = random.Random(a.seed)
    p = rand_tokens(rng, 3000)
    s = rand_tokens(rng, 300)
    out = []
    for name, prompt in (("P_cold", p), ("P_again", p), ("P+S", p + s)):
        q0, h0 = await cache_counters(session, a)
        t0 = time.perf_counter()
        ids, lps, usage = await complete(session, a, prompt, 24)
        dt = time.perf_counter() - t0
        q1, h1 = await cache_counters(session, a)
        out.append({"req": name, "prompt_len": len(prompt), "s": round(dt, 2), "cache_queries": q1 - q0,
                    "cache_hits": h1 - h0, "token_ids": ids, "logprobs": [round(x, 4) for x in lps]})
        print(json.dumps({k: v for k, v in out[-1].items() if k not in ("token_ids", "logprobs")}), flush=True)
    same = out[0]["token_ids"] == out[1]["token_ids"]
    dlp = max(abs(x - y) for x, y in zip(out[0]["logprobs"], out[1]["logprobs"]))
    print(json.dumps({"P_again_same_tokens": same, "P_again_max_abs_dlogprob": round(dlp, 4)}), flush=True)
    return {"mode": "parity", "requests": out, "P_again_same_tokens": same, "P_again_max_abs_dlogprob": dlp}


async def gencache(session, a):
    """Are the KV blocks of generated tokens reusable? P, then P + its greedy output + a short tail."""
    rng = random.Random(a.seed)
    out = []
    for gen in (512, 1024):
        p = rand_tokens(rng, 3000)
        ids, _, _ = await complete(session, a, p, gen)
        q0, h0 = await cache_counters(session, a)
        await complete(session, a, p + ids + rand_tokens(rng, 16), 1)
        q1, h1 = await cache_counters(session, a)
        r = {"prompt": 3000, "generated": len(ids), "resend_len": 3000 + len(ids) + 16, "hits": h1 - h0,
             "hits_if_prompt_only": (3000 // 128) * 128, "hits_if_output_cached": ((3000 + len(ids)) // 128) * 128}
        out.append(r)
        print(json.dumps(r), flush=True)
    return {"mode": "gencache", "rows": out}


async def attempt(session, a, i, rec):
    rng = random.Random(a.seed * 1000 + i)
    hist = rand_tokens(rng, a.base)
    for t in range(a.turns):
        t0 = time.perf_counter()
        ids, _, _ = await complete(session, a, hist, a.gen)
        rec.append({"attempt": i, "turn": t, "prompt_len": len(hist), "s": time.perf_counter() - t0})
        hist = hist + ids + rand_tokens(rng, a.tool)


async def replay(session, a):
    q0, h0 = await cache_counters(session, a)
    rec = []
    t0 = time.perf_counter()
    await asyncio.gather(*(attempt(session, a, i, rec) for i in range(a.conc)))
    wall = time.perf_counter() - t0
    q1, h1 = await cache_counters(session, a)
    prompt_tokens = sum(r["prompt_len"] for r in rec)
    new_tokens = a.conc * (a.base + (a.turns - 1) * (a.gen + a.tool))
    per_turn = {t: round(statistics.mean(r["s"] for r in rec if r["turn"] == t), 2) for t in range(a.turns)}
    out = {"mode": "replay", "conc": a.conc, "turns": a.turns, "base": a.base, "gen": a.gen, "tool": a.tool,
           "wall_s": round(wall, 1), "gen_tok_s": round(a.conc * a.turns * a.gen / wall, 1),
           "prompt_tokens": prompt_tokens, "reusable_frac": round(1 - new_tokens / prompt_tokens, 3),
           "cache_queries": q1 - q0, "cache_hits": h1 - h0,
           "hit_rate": round((h1 - h0) / (q1 - q0), 3) if q1 > q0 else None,
           "turn_latency_mean_s": per_turn}
    print(json.dumps(out), flush=True)
    return out


async def run(a):
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        res = await {"parity": parity, "replay": replay, "gencache": gencache}[a.mode](session, a)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["parity", "replay", "gencache"])
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41-e64")
    p.add_argument("--conc", type=int, default=16)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--base", type=int, default=2048)
    p.add_argument("--gen", type=int, default=192)
    p.add_argument("--tool", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
