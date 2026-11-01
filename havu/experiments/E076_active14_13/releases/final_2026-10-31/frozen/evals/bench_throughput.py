"""Decode-throughput sweep for a self-hosted OpenAI-compatible server on ARC-shaped work.

For each CONCURRENCY:SECONDS spec, holds that many long generations open (real ARC eval prompts, raw
completions, ignore_eos so nothing finishes early), reports streamed tokens/s every 30 s, then cancels.
The first windows include prefill; read the steady-state rows. Sequences grow by ~tok/s/C per second,
so a long spec also shows how throughput falls as contexts get long.

    .venv/bin/python evals/bench_throughput.py --endpoint vastai-4xl40s-qwen27b-fp8 \
        --model Qwen/Qwen3.8-27B-FP8 1:90 8:210 16:210 32:240 48:240 64:900
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from arc_harness.arcprize import TASKS, build_prompt  # noqa: E402


def endpoint(name):
    import yaml
    for e in yaml.safe_load((ROOT / "ENDPOINTS.yaml").read_text()):
        if e.get("name") == name:
            return f"http://{e['addr']}/v1", e["token"]
    raise KeyError(name)


def prompts():
    out = []
    for f in sorted(TASKS.glob("*.json")):
        t = json.loads(f.read_text())
        user = build_prompt(t["train"], t["test"][0]["input"]).strip()
        out.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n")
    return out


async def one(client, model, prompt, counter):
    stream = await client.completions.create(
        model=model, prompt=prompt, max_tokens=60000, stream=True, temperature=1.0, top_p=0.95,
        extra_body={"top_k": 20, "ignore_eos": True})
    async for _ in stream:
        counter[0] += 1


async def point(client, model, pool, conc, seconds, window=30):
    counter, t0, last = [0], time.time(), 0
    print(f"=== C={conc} start {time.strftime('%H:%M:%S', time.gmtime())} UTC", flush=True)
    tasks = [asyncio.create_task(one(client, model, pool[(i * 7) % len(pool)], counter)) for i in range(conc)]
    while time.time() - t0 < seconds:
        await asyncio.sleep(window)
        n = counter[0]
        print(f"  C={conc} t={time.time() - t0:4.0f}s tok/s={(n - last) / window:7.1f}", flush=True)
        last = n
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(20)  # let the server drop the cancelled requests


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="entry name in ENDPOINTS.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("specs", nargs="+", help="CONCURRENCY:SECONDS")
    args = ap.parse_args()
    from openai import AsyncOpenAI
    base, token = endpoint(args.endpoint)
    pool = prompts()
    async with AsyncOpenAI(base_url=base, api_key=token, timeout=7200, max_retries=0) as client:
        for spec in args.specs:
            conc, seconds = map(int, spec.split(":"))
            await point(client, args.model, pool, conc, seconds)


if __name__ == "__main__":
    asyncio.run(main())
