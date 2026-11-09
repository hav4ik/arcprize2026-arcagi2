"""Routing-key stickiness probe for the SGLang Model Gateway (manual policy, E019). No tokens involved.

    .venv/bin/python experiments/E019_qwen_program_answers/router/routing_probe.py sticky --convs 3 --turns 4
    .venv/bin/python experiments/E019_qwen_program_answers/router/routing_probe.py spread --keys 21
    .venv/bin/python experiments/E019_qwen_program_answers/router/routing_probe.py turn --key K --state F

sticky: N conversations, each with its own `x-smg-routing-key`, T turns each, sent one request at a time and
        interleaved across conversations (c0t0, c1t0, c2t0, c0t1, ...). Every turn re-sends the whole transcript
        (ARC-task prefix + previous turns) via /generate, like the Qwen harness does via /v1/completions. The
        serving worker of each request is read from the gateway's per-worker Prometheus counter
        smg_worker_cb_outcomes_total{worker=...} (delta of 1), and cached_tokens from meta_info.
spread: K new keys at once (one short request each, unique prefixes); prints the per-worker counter deltas.
turn:   one more turn of a saved conversation (--state file from `sticky --save`), to test reassignment after
        its worker is removed.
"""
import argparse
import asyncio
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
from arc_harness.arcprize import TASKS, build_prompt  # noqa: E402

LINE = re.compile(r'^smg_worker_cb_outcomes_total\{worker="([^"]+)",outcome="success"\}\s+([0-9.]+)')


def counters(prom):
    out = {}
    for line in urllib.request.urlopen(prom + "/metrics", timeout=10).read().decode().splitlines():
        m = LINE.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def delta(a, b):
    return {w: b[w] - a.get(w, 0) for w in b if b[w] - a.get(w, 0)}


def prompts(n, offset=0):
    out = []
    for f in sorted(TASKS.glob("*.json"))[offset:offset + n]:
        t = json.loads(f.read_text())
        user = build_prompt(t["train"], t["test"][0]["input"]).strip()
        out.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n")
    return out


def turn_suffix(k):
    return ("\n</think>\n\n<tool_call>\n<function=python>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n"
            f"<tool_response>\nturn {k} ok\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")


async def gen(client, gw, key, text, n):
    r = await client.post(gw + "/generate", headers={"x-smg-routing-key": key}, json={
        "text": text, "sampling_params": {"max_new_tokens": n, "temperature": 1.0, "top_p": 0.95, "top_k": 20}})
    r.raise_for_status()
    j = r.json()
    return j["text"], j["meta_info"]


async def sticky(args):
    run = args.tag or time.strftime("%H%M%S")
    convs = [{"key": f"probe-{run}-{i}", "text": p, "turns": []} for i, p in enumerate(prompts(args.convs))]
    async with httpx.AsyncClient(timeout=600) as client:
        for k in range(args.turns):
            for c in convs:
                before = counters(args.prom)
                t = time.time()
                out, m = await gen(client, args.gateway, c["key"], c["text"], args.max_tokens)
                d = delta(before, counters(args.prom))
                c["turns"].append({"turn": k, "worker": ",".join(d) or "?", "prompt": m["prompt_tokens"],
                                   "cached": m["cached_tokens"], "s": round(time.time() - t, 2)})
                c["text"] += out + turn_suffix(k)
    ok = True
    for c in convs:
        ws = {t["worker"] for t in c["turns"]}
        hit = [round(t["cached"] / t["prompt"], 3) for t in c["turns"]]
        ok &= len(ws) == 1
        print(f"{c['key']}: workers={sorted(ws)} cached_frac_by_turn={hit} "
              f"prompt_tokens={[t['prompt'] for t in c['turns']]} s={[t['s'] for t in c['turns']]}")
    print("STICKY" if ok else "NOT STICKY")
    if args.save:
        Path(args.save).write_text(json.dumps(convs))


async def spread(args):
    run = args.tag or time.strftime("%H%M%S")
    pool = prompts(args.keys, offset=40)
    before = counters(args.prom)
    async with httpx.AsyncClient(timeout=600) as client:
        res = await asyncio.gather(*[gen(client, args.gateway, f"spread-{run}-{i}", p, 16)
                                     for i, p in enumerate(pool)], return_exceptions=True)
    errs = [repr(r)[:160] for r in res if isinstance(r, Exception)]
    d = delta(before, counters(args.prom))
    print(f"{args.keys} new keys -> per-worker requests {d}; errors {len(errs)} {errs[:3]}")


async def one_turn(args):
    convs = json.loads(Path(args.state).read_text())
    c = next(c for c in convs if c["key"] == args.key)
    async with httpx.AsyncClient(timeout=600) as client:
        before = counters(args.prom)
        try:
            out, m = await gen(client, args.gateway, c["key"], c["text"], args.max_tokens)
            print(f"{c['key']}: served by {list(delta(before, counters(args.prom)))} "
                  f"prompt={m['prompt_tokens']} cached={m['cached_tokens']} (previous worker "
                  f"{c['turns'][-1]['worker']})")
        except httpx.HTTPStatusError as e:
            print(f"{c['key']}: HTTP {e.response.status_code} {e.response.text[:200]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sticky", "spread", "turn"])
    ap.add_argument("--gateway", default="http://127.0.0.1:30100")
    ap.add_argument("--prom", default="http://127.0.0.1:29100")
    ap.add_argument("--convs", type=int, default=3)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--keys", type=int, default=21)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--tag")
    ap.add_argument("--save")
    ap.add_argument("--state")
    ap.add_argument("--key")
    args = ap.parse_args()
    asyncio.run({"sticky": sticky, "spread": spread, "turn": one_turn}[args.mode](args))


if __name__ == "__main__":
    main()
