#!/usr/bin/env python3
"""Multi-turn agentic load generator for vLLM KV-cache experiments (E067 kvcache lab).

N closed-loop worker slots. Each slot runs "attempts" back to back. An attempt is a
conversation of T turns over token ids (exact prefix reuse, like --token-exact):

    prompt_0 = shared_sys + unique_task
    turn t:  send prompt_t (completions, ignore_eos, max_tokens = reply_t, stream)
             prompt_{t+1} = prompt_t + sampled_ids_t + tool_ids_t   (tool output: random ids)
             sleep think_t   (sandbox time)

Optional admission gate (new attempts only; turns of running attempts are never gated):
    start a new attempt only if the server's kv_cache_usage_perc < --gate-kv
    or fewer than --gate-min attempts are active.

Outputs: <out>.turns.jsonl (one row per request), <out>.metrics.jsonl (1 Hz /metrics
samples plus client state and host memory), <out>.config.json.
"""
import argparse
import asyncio
import json
import os
import random
import re
import time

import aiohttp
import numpy as np

METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)')
KEEP_PREFIXES = (
    "vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:prefix_cache_", "vllm:external_prefix_cache_", "vllm:num_preemptions",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:prompt_tokens_by_source",
    "vllm:prompt_tokens_cached", "vllm:iteration_tokens_total_sum", "vllm:iteration_tokens_total_count",
    "vllm:kv_offload", "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count",
    "vllm:request_success_total", "vllm:kv_block_",
)


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if not name.startswith(KEEP_PREFIXES):
            continue
        # keep only the label that matters (reason / source / transfer_type), drop model/engine
        lab = ""
        for key in ("reason", "source", "transfer_type", "le"):
            mm = re.search(key + r'="([^"]*)"', labels)
            if mm:
                lab += "{" + mm.group(1) + "}"
        try:
            out[name + lab] = out.get(name + lab, 0.0) + float(val)
        except ValueError:
            pass
    return out


def proc_tree_rss_mb(pid):
    """RSS (MB) of pid and all descendants, from /proc. Pinned host memory shows up here."""
    if not pid:
        return None
    children = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            children.setdefault(ppid, []).append(int(p))
        except Exception:
            pass
    total, stack = 0, [pid]
    while stack:
        q = stack.pop()
        try:
            with open(f"/proc/{q}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
        except Exception:
            pass
        stack.extend(children.get(q, []))
    return total / 1024


def mem_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return None


class State:
    def __init__(self):
        self.active_attempts = 0
        self.inflight = 0
        self.gate_blocked = 0
        self.kv_usage = 0.0
        self.stop = False
        self.t0 = time.time()


async def sampler(session, args, st, fout):
    while not st.stop:
        t = time.time()
        rec = {"t": round(t - st.t0, 3), "active": st.active_attempts, "inflight": st.inflight,
               "gate_blocked": st.gate_blocked}
        try:
            async with session.get(args.url + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                m = parse_metrics(await r.text())
            rec.update(m)
            st.kv_usage = m.get("vllm:kv_cache_usage_perc", st.kv_usage)
        except Exception as e:  # noqa: BLE001
            rec["metrics_error"] = str(e)[:100]
        rec["server_rss_mb"] = proc_tree_rss_mb(args.server_pid)
        rec["mem_avail_mb"] = mem_available_mb()
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        await asyncio.sleep(max(0.0, args.sample_interval - (time.time() - t)))


async def one_request(session, args, prompt_ids, max_tokens):
    body = {
        "model": args.model, "prompt": prompt_ids, "max_tokens": max_tokens,
        "min_tokens": max_tokens, "ignore_eos": True, "temperature": 1.0, "top_p": 1.0,
        "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
    }
    t_send = time.time()
    t_first = None
    ids, usage = [], None
    async with session.post(args.url + "/v1/completions", json=body,
                            timeout=aiohttp.ClientTimeout(total=args.request_timeout)) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        buf = b""
        done = False
        async for piece in r.content.iter_any():  # manual line split: chunk 1 echoes prompt ids (>128 KB)
            buf += piece
            while b"\n" in buf and not done:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    tids = ch.get("token_ids")
                    if tids:
                        if t_first is None:
                            t_first = time.time()
                        ids.extend(tids)
            if done:
                break
    t_done = time.time()
    return ids, usage, t_send, t_first or t_done, t_done


async def worker(wid, session, args, st, fout, sys_ids):
    rng = random.Random(args.seed * 1000 + wid)
    nrng = np.random.default_rng(args.seed * 1000 + wid)
    await asyncio.sleep(rng.uniform(0, args.stagger))
    attempt_idx = 0
    while not st.stop and time.time() - st.t0 < args.duration:
        # admission gate for NEW attempts
        t_gate = time.time()
        if args.gate_kv < 1.0:
            st.gate_blocked += 1
            while (not st.stop and time.time() - st.t0 < args.duration
                   and st.active_attempts >= args.gate_min and st.kv_usage >= args.gate_kv):
                await asyncio.sleep(0.25)
            st.gate_blocked -= 1
            if st.stop or time.time() - st.t0 >= args.duration:
                break
        gate_wait = time.time() - t_gate
        st.active_attempts += 1
        aid = f"w{wid}a{attempt_idx}"
        attempt_idx += 1
        n_turns = rng.randint(args.turns_min, args.turns_max)
        ids = list(sys_ids) + nrng.integers(100, 150000, args.init_tokens - len(sys_ids)).tolist()
        t_att = time.time()
        turns_done = 0
        try:
            for turn in range(n_turns):
                reply = rng.randint(args.reply_min, args.reply_max)
                if len(ids) + reply > args.max_ctx:
                    break
                if st.stop or time.time() - st.t0 >= args.duration + args.drain:
                    break
                st.inflight += 1
                err = None
                try:
                    out_ids, usage, ts, tf, td = await one_request(session, args, ids, reply)
                except Exception as e:  # noqa: BLE001
                    err = repr(e)[:200]
                    out_ids, usage, ts, tf, td = [], None, time.time(), None, time.time()
                finally:
                    st.inflight -= 1
                cached = None
                if usage and usage.get("prompt_tokens_details"):
                    cached = usage["prompt_tokens_details"].get("cached_tokens")
                rec = {"aid": aid, "w": wid, "turn": turn, "prompt": len(ids), "out": len(out_ids),
                       "cached": cached, "t_send": round(ts - st.t0, 3),
                       "ttft": None if tf is None else round(tf - ts, 3),
                       "lat": round(td - ts, 3), "err": err, "gate_wait": round(gate_wait, 2) if turn == 0 else None}
                fout.write(json.dumps(rec) + "\n")
                if err:
                    break
                turns_done += 1
                tool = rng.randint(args.tool_min, args.tool_max)
                ids = ids + out_ids + nrng.integers(100, 150000, tool).tolist()
                if time.time() - st.t0 >= args.duration:
                    break
                await asyncio.sleep(rng.uniform(args.think_min, args.think_max))
        finally:
            st.active_attempts -= 1
        fout.write(json.dumps({"aid": aid, "w": wid, "attempt_done": turns_done == n_turns,
                               "turns_done": turns_done, "n_turns": n_turns,
                               "dur": round(time.time() - t_att, 2), "final_ctx": len(ids)}) + "\n")
        fout.flush()


async def main_async(args):
    st = State()
    sys_ids = np.random.default_rng(12345).integers(100, 150000, args.sys_tokens).tolist()
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        with open(args.out + ".turns.jsonl", "w") as ft, open(args.out + ".metrics.jsonl", "w") as fm:
            samp = asyncio.create_task(sampler(session, args, st, fm))
            workers = [asyncio.create_task(worker(i, session, args, st, ft, sys_ids))
                       for i in range(args.concurrency)]
            try:
                await asyncio.wait_for(asyncio.gather(*workers), timeout=args.duration + args.drain)
            except asyncio.TimeoutError:
                pass
            st.stop = True
            for w in workers:
                w.cancel()
            await asyncio.sleep(args.sample_interval + 0.2)
            samp.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8067")
    ap.add_argument("--model", default="m")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--drain", type=float, default=30, help="extra seconds for in-flight turns")
    ap.add_argument("--stagger", type=float, default=30)
    ap.add_argument("--turns-min", type=int, default=9)
    ap.add_argument("--turns-max", type=int, default=15)
    ap.add_argument("--sys-tokens", type=int, default=400)
    ap.add_argument("--init-tokens", type=int, default=3000)
    ap.add_argument("--tool-min", type=int, default=200)
    ap.add_argument("--tool-max", type=int, default=2000)
    ap.add_argument("--reply-min", type=int, default=200)
    ap.add_argument("--reply-max", type=int, default=800)
    ap.add_argument("--think-min", type=float, default=1.0)
    ap.add_argument("--think-max", type=float, default=4.0)
    ap.add_argument("--max-ctx", type=int, default=31000)
    ap.add_argument("--gate-kv", type=float, default=1.0, help="admit new attempts only if KV usage < this")
    ap.add_argument("--gate-min", type=int, default=4, help="always allow this many active attempts")
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--server-pid", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out + ".config.json", "w") as f:
        json.dump(vars(args), f, indent=1)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
