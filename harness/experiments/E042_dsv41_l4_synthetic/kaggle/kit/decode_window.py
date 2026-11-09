#!/usr/bin/env python3
"""E042: steady-state decode throughput at a given context length and batch size.

Sends C streaming completions at once, each with a distinct random-token prompt of
`ctx` tokens, temperature 0, ignore_eos and a max_tokens budget large enough that
no request finishes during the window. After the last request has produced its
first token (all C are decoding, no prefill left), it skips `warm` steps, then
counts every token produced by all C requests for `window` seconds. Then it closes
the streams, which aborts the requests on the server.

Reports: prefill (time until all C first tokens, aggregate prompt tok/s) and steady
decode (aggregate tok/s, per-stream median inter-token gap) in that window.
Bench-serve TPOT is contaminated by other requests' chunked prefill; this is not.
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


async def one(session, url, model, prompt, max_tokens, t0, rec):
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
        "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    async with session.post(f"{url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or {}
            n = usage.get("completion_tokens")
            if n is not None:
                rec.append((time.perf_counter() - t0, n))


async def profile(session, a):
    t = time.perf_counter()
    try:
        async with session.post(f"{a.url}/start_profile") as r:
            r.raise_for_status()
        await asyncio.sleep(a.profile_s)
        async with session.post(f"{a.url}/stop_profile", timeout=aiohttp.ClientTimeout(total=900)) as r:
            r.raise_for_status()
        return {"ok": True, "s": round(time.perf_counter() - t, 1)}
    except Exception as exc:  # a failed capture must not cost the decode measurement
        return {"ok": False, "error": repr(exc)[:300]}


async def run(a):
    rng = random.Random(a.seed)
    # --ctx-file: one context length per stream (a steady-state snapshot of trace-shaped load, trace_shapes.py).
    ctxs = json.load(open(a.ctx_file)) if a.ctx_file else [a.ctx] * a.conc
    a.conc = len(ctxs)
    prompts = [[rng.randrange(1000, 100000) for _ in range(n)] for n in ctxs]
    max_tokens = [min(a.max_tokens, a.max_model_len - n - 8) for n in ctxs]
    recs = [[] for _ in range(a.conc)]
    # A stream whose prompt is scheduled last gets no bytes until the whole batch is prefilled (long-context points).
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.sock_read)
    # limit=0: aiohttp caps a session at 100 connections by default, which would hold back streams beyond 100.
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        t0, u0 = time.perf_counter(), time.time()
        tasks = [asyncio.create_task(one(session, a.url, a.model, p, m, t0, r))
                 for p, m, r in zip(prompts, max_tokens, recs)]
        # Wait for every stream's first token, then warm-up steps, then the window.
        while not all(recs) and not any(t.done() for t in tasks):
            await asyncio.sleep(0.05)
        if not all(recs):
            for t in tasks:
                if t.done() and t.exception():
                    raise t.exception()
        t_all_first = max(r[0][0] for r in recs)
        target = [r[-1][1] + a.warm for r in recs]
        while any(r[-1][1] < tgt for r, tgt in zip(recs, target)):
            await asyncio.sleep(0.02)
        prof = None
        if a.profile_s:
            # Capture steady decode at this context (server started with --profiler-config), then warm up again
            # so the measured window excludes the capture and the trace write.
            prof = await profile(session, a)
            target = [r[-1][1] + a.warm for r in recs]
            while any(r[-1][1] < tgt for r, tgt in zip(recs, target)):
                await asyncio.sleep(0.02)
        w0 = time.perf_counter() - t0
        n0 = [r[-1][1] for r in recs]
        await asyncio.sleep(a.window)
        w1 = time.perf_counter() - t0
        n1 = [r[-1][1] for r in recs]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    tokens = sum(n1) - sum(n0)
    gaps = []
    for r in recs:
        pts = [(t, n) for t, n in r if w0 <= t <= w1]
        gaps += [(t2 - t1) / max(1, n2 - n1) for (t1, n1), (t2, n2) in zip(pts, pts[1:]) if n2 > n1]
    out = {
        "ctx": a.ctx if not a.ctx_file else round(statistics.mean(ctxs)), "conc": a.conc, "warm": a.warm,
        "window_s": round(w1 - w0, 2),
        # Unix start and window offset: lines the prefill [0, prefill_all_first_s] and the window up with gpu.csv clocks.
        "t0_unix": round(u0, 1), "window_start_s": round(w0, 2),
        "prefill_all_first_s": round(t_all_first, 2),
        "prefill_tok_s": round(sum(ctxs) / t_all_first, 1),
        "first_token_s": sorted(round(r[0][0], 2) for r in recs),
        "decode_tokens": tokens, "decode_tok_s": round(tokens / (w1 - w0), 1),
        "per_stream_tok_s": round(tokens / (w1 - w0) / a.conc, 2),
        "itl_median_ms": round(1000 * statistics.median(gaps), 1) if gaps else None,
        "context_at_window_end": max(n + k for n, k in zip(ctxs, n1)),
    }
    if a.ctx_file:
        q = sorted(ctxs)
        out["ctx_file"] = a.ctx_file
        out["ctx_p50_p90_max"] = [q[len(q) // 2], q[int(0.9 * len(q))], q[-1]]
    if prof is not None:
        out["profile"] = prof
    print(json.dumps(out))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41-e64")
    p.add_argument("--ctx", type=int, default=0)
    p.add_argument("--conc", type=int, default=0)
    p.add_argument("--ctx-file", help="JSON list of per-stream context lengths (overrides --ctx/--conc)")
    p.add_argument("--warm", type=int, default=16)
    p.add_argument("--window", type=float, default=30.0)
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sock-read", type=float, default=3600)
    p.add_argument("--profile-s", type=float, default=0, help="capture a torch profile for this many seconds before the window")
    p.add_argument("--json")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
