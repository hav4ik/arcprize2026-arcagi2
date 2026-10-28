#!/usr/bin/env python3
"""E060 microbenchmark for a self-hosted OpenAI-compatible vLLM server (repeatable on an L4 box for the
4090-to-L4 conversion factor).

decode C:S   holds C long generations open for S seconds (real ARC eval prompts as raw completions in the Qwen chat
             format with an open <think>, ignore_eos, T 0.7 / top-p 0.95 / top-k 20) and reports generation tok/s
             from the server's vllm:generation_tokens_total counter over the steady-state windows (the first window,
             which holds the prefills, is reported but excluded from the mean).
prefill N    sends 3 unique prompts of about N tokens one at a time with max_tokens 1 (no prefix-cache reuse) and
             reports input tok/s = N / time to first token (median of the 3).

    python bench_flashnext.py --base-url http://127.0.0.1:18321/v1 decode 1:120 16:180 32:180 56:180 prefill 8000 32000
"""
import argparse
import asyncio
import json
import random
import re
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for cand in (ROOT / "evals", ROOT.parent / "evals", ROOT.parents[1] / "evals"):
    if (cand / "arc_harness").exists():
        sys.path.insert(0, str(cand))
        break
from arc_harness.arcprize import TASKS, build_prompt  # noqa: E402


def prompts():
    out = []
    for f in sorted(TASKS.glob("*.json")):
        t = json.loads(f.read_text())
        user = build_prompt(t["train"], t["test"][0]["input"]).strip()
        out.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n")
    return out


def gen_tokens(metrics_url):
    text = urllib.request.urlopen(metrics_url, timeout=10).read().decode()
    return sum(float(m.group(1)) for m in re.finditer(r"^vllm:generation_tokens_total(?:\{[^}]*\})?\s+(\S+)$", text, re.M))


async def one(client, model, prompt):
    stream = await client.completions.create(
        model=model, prompt=prompt, max_tokens=60000, stream=True, temperature=0.7, top_p=0.95,
        extra_body={"top_k": 20, "ignore_eos": True})
    async for _ in stream:
        pass


async def decode(client, model, pool, metrics_url, conc, seconds, window=30):
    tasks = [asyncio.create_task(one(client, model, pool[(i * 7) % len(pool)])) for i in range(conc)]
    t0, last, rates = time.time(), gen_tokens(metrics_url), []
    while time.time() - t0 < seconds:
        await asyncio.sleep(window)
        n = gen_tokens(metrics_url)
        rates.append((n - last) / window)
        last = n
        print(f"  decode C={conc} t={time.time() - t0:4.0f}s gen tok/s={rates[-1]:7.1f}", flush=True)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(15)
    steady = rates[1:] or rates
    res = {"kind": "decode", "concurrency": conc, "seconds": seconds, "windows": rates,
           "gen_tps_steady": round(st.mean(steady), 1), "per_stream": round(st.mean(steady) / conc, 2)}
    print(json.dumps(res), flush=True)
    return res


async def prefill(client, model, n_tokens):
    times = []
    for _ in range(3):
        words = " ".join(str(random.randrange(10 ** 6)) for _ in range(n_tokens // 4))  # ~4 tokens per 6-digit number + space
        prompt = f"<|im_start|>user\n{words}<|im_end|>\n<|im_start|>assistant\n"
        t0 = time.time()
        r = await client.completions.create(model=model, prompt=prompt, max_tokens=1, temperature=0.0)
        dt = time.time() - t0
        times.append((r.usage.prompt_tokens, dt))
    tps = st.median(p / d for p, d in times)
    res = {"kind": "prefill", "target_tokens": n_tokens, "runs": [(p, round(d, 2)) for p, d in times],
           "input_tps_median": round(tps, 1)}
    print(json.dumps(res), flush=True)
    return res


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:18321/v1")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--out", help="append JSON results here")
    ap.add_argument("plan", nargs="+", help="decode C:S [C:S ...] prefill N [N ...]")
    args = ap.parse_args()
    from openai import AsyncOpenAI
    metrics_url = args.base_url.rstrip("/").removesuffix("/v1") + "/metrics"
    pool, mode, results = prompts(), None, []
    async with AsyncOpenAI(base_url=args.base_url, api_key="none", timeout=7200, max_retries=0) as client:
        for item in args.plan:
            if item in ("decode", "prefill"):
                mode = item
            elif mode == "decode":
                c, s = map(int, item.split(":"))
                results.append(await decode(client, args.model, pool, metrics_url, c, s))
            elif mode == "prefill":
                results.append(await prefill(client, args.model, int(item)))
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps({**r, "t": time.strftime("%FT%TZ", time.gmtime())}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
