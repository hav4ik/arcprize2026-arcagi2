#!/usr/bin/env python3
"""E042 R2: tool-mode load shaped like the real E030 attempts (trace_shapes.py tools file), in steady state.

Each of C slots runs attempts back to back. Slots start mid-attempt, as a closed loop would be in steady state: an
attempt drawn in proportion to its generated tokens, a turn drawn in proportion to its completion, a uniform point
inside that completion, with the history up to that point sent cold; the first turn generates only the rest of the
completion. Later attempts are uniform draws from turn 0.
A turn sends the whole history as token ids (random ids for prompt and tool results), generates exactly the recorded
completion tokens (ignore_eos), then appends the generated ids (return_token_ids) plus the recorded tool-result tokens as
random ids and waits --tool-delay (code execution). Sending back the model's own ids lets the next turn hit the
generated-token blocks when the server caches them (--prefix-cache-retention-interval > 0; P7/P11). Run with
--enable-prefix-caching.
The window opens --warm-s after every slot's first token (the cold histories are prefilled by then) and after the
turns queued behind the cold prefills have their first token too, and lasts --window-s. Reports generated tok/s, prefix-cache hit rate and uncached prompt tokens in the window, turn latency.
--max-wall-s caps the whole run (setup + warm-up + window) for time-boxed runs; the window shrinks to fit.
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


async def metrics(session, a):
    async with session.get(f"{a.url}/metrics") as r:
        txt = await r.text()
    out = {}
    for line in txt.splitlines():
        for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:generation_tokens_total",
                  "vllm:num_preemptions_total"):
            if line.startswith(k):
                out[k] = out.get(k, 0.0) + float(line.split()[-1])
    return out


async def turn(session, a, prompt, max_tokens, t0, ev, on_first=None):
    """Stream one completion; ev gets (time, new tokens) per chunk, on_first() runs at the first token.
    Returns (ttft, duration, generated ids)."""
    body = {"model": a.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "return_token_ids": True,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    ts = time.perf_counter()
    ttft, last, ids = None, 0, []
    async with session.post(f"{a.url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        # Split SSE lines ourselves: the first chunk echoes the prompt ids (return_token_ids), over aiohttp's 512 KB
        # readline limit for prompts past ~85K tokens.
        buf = b""
        async for data in resp.content.iter_any():
            buf += data
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                line = raw.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                for ch in d.get("choices") or []:
                    ids.extend(ch.get("token_ids") or [])
                n = (d.get("usage") or {}).get("completion_tokens")
                if n is not None and n > last:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - ts
                        if on_first:
                            on_first()
                    ev.append((now - t0, n - last))
                    last = n
    return ttft, time.perf_counter() - ts, ids


async def slot(session, a, i, shapes, t0, st):
    rng = random.Random(a.seed * 1000 + i)
    w_att = [sum(c for c, _ in s["turns"]) for s in shapes]
    first = True

    def started():
        st["first"].setdefault(i, time.perf_counter() - t0)

    while not st["stop"]:
        if first:
            att = rng.choices(shapes, weights=w_att)[0]
            k = rng.choices(range(len(att["turns"])), weights=[c for c, _ in att["turns"]])[0]
            u = rng.randrange(att["turns"][k][0])  # tokens of the current completion already generated
            hist = rand_tokens(rng, att["p0"] + sum(c + r for c, r in att["turns"][:k]) + u)
        else:
            att, k, u = rng.choice(shapes), 0, 0
            hist = rand_tokens(rng, att["p0"])
        for c, r in att["turns"][k:]:
            c, u = c - u, 0
            if len(hist) + c > a.max_model_len - 8:
                break
            key = object()
            st["pending"].add(key)
            ttft, dur, ids = await turn(session, a, hist, c, t0, st["ev"],
                                        lambda: (st["pending"].discard(key), first and started()))
            st["pending"].discard(key)
            st["turns"].append({"slot": i, "t_end": time.perf_counter() - t0, "prompt": len(hist), "gen": c,
                                "ids_back": len(ids), "ttft": ttft, "s": dur})
            first = False
            if st["stop"]:
                return
            hist = hist + ids[:c] + rand_tokens(rng, c - len(ids[:c]) + r)
            if r:
                await asyncio.sleep(a.tool_delay)
        first = False
        st["attempts_done"] += 1


async def run(a):
    shapes = json.load(open(a.shapes))
    st = {"stop": False, "ev": [], "turns": [], "first": {}, "attempts_done": 0, "pending": set()}
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.sock_read)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(slot(session, a, i, shapes, t0, st)) for i in range(a.conc)]
        setup_deadline = a.max_wall_s - a.warm_s - 600 if a.max_wall_s else float("inf")
        while len(st["first"]) < a.conc and time.perf_counter() - t0 < setup_deadline:
            # Setup ends when every slot has its first token (its cold history is prefilled), not its first turn:
            # length-biased first turns run up to ~36K tokens, hours at a few tok/s per stream.
            for t in tasks:
                if t.done() and t.exception():
                    raise t.exception()
            await asyncio.sleep(0.5)
        t_setup, started = time.perf_counter() - t0, len(st["first"])
        # Turns that slots began during setup waited behind the cold prefills, and the prefill churn evicted their
        # cached prefixes (box E48 x 128: 18 min of 0% hits after setup). Let them clear before the warm-up.
        queued = set(st["pending"])
        while queued & st["pending"] and time.perf_counter() - t0 < setup_deadline:
            await asyncio.sleep(0.5)
        t_drain = time.perf_counter() - t0
        await asyncio.sleep(a.warm_s)
        w0 = time.perf_counter() - t0
        m0 = await metrics(session, a)
        n_turns0, att0 = len(st["turns"]), st["attempts_done"]
        window = min(a.window_s, a.max_wall_s - w0) if a.max_wall_s else a.window_s
        await asyncio.sleep(max(60, window))
        w1 = time.perf_counter() - t0
        m1 = await metrics(session, a)
        st["stop"] = True
        turns_w = st["turns"][n_turns0:]
        att_w = st["attempts_done"] - att0
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    gen = sum(n for t, n in st["ev"] if w0 <= t <= w1)
    q, h = d.get("vllm:prefix_cache_queries_total", 0), d.get("vllm:prefix_cache_hits_total", 0)
    out = {
        "mode": "tool_replay", "shapes": a.shapes, "conc": a.conc, "tool_delay_s": a.tool_delay,
        "setup_s": round(t_setup, 1), "slots_started_at_setup": started, "drain_s": round(t_drain - t_setup, 1), "window_s": round(w1 - w0, 1),
        "gen_tok_s": round(gen / (w1 - w0), 1),
        "gen_tok_s_metrics": round(d.get("vllm:generation_tokens_total", 0) / (w1 - w0), 1),
        "prompt_tokens_queried": int(q), "prompt_tokens_uncached": int(q - h),
        "uncached_prefill_tok_s": round((q - h) / (w1 - w0), 1),
        "hit_rate": round(h / q, 3) if q else None,
        "preemptions": int(d.get("vllm:num_preemptions_total", 0)),
        "turns_done_in_window": len(turns_w), "attempts_done_in_window": att_w,
        "ids_back_ok": round(sum(x["ids_back"] >= x["gen"] for x in turns_w) / len(turns_w), 3) if turns_w else None,
        "turn_prompt_p50_p90": ([sorted(x["prompt"] for x in turns_w)[len(turns_w) // 2],
                                 sorted(x["prompt"] for x in turns_w)[int(0.9 * len(turns_w))]] if turns_w else None),
        "ttft_median_s": round(statistics.median(x["ttft"] for x in turns_w if x["ttft"]), 2) if turns_w else None,
        "turn_s_median": round(statistics.median(x["s"] for x in turns_w), 1) if turns_w else None,
    }
    print(json.dumps(out), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({**out, "turns": st["turns"]}, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41-e64")
    p.add_argument("--conc", type=int, default=64)
    p.add_argument("--warm-s", type=float, default=300)
    p.add_argument("--window-s", type=float, default=2400)
    p.add_argument("--tool-delay", type=float, default=1.0)
    p.add_argument("--max-model-len", type=int, default=163840)
    p.add_argument("--sock-read", type=float, default=14400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-wall-s", type=float, default=0, help="cap on setup + warm-up + window (0: none)")
    p.add_argument("--json")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
