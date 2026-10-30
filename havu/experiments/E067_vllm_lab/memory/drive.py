"""Load driver for the E067 memory lab (OpenAI completions API, token-id prompts).

Prompts are random token ids (unique per request, so no prefix-cache hits
unless --shared-prefix is given). Outputs are forced with ignore_eos.

  drive.py ladder --lens 2048,4096,... [--out-tokens 16] [--repeat 1]
  drive.py mix --conc 16 --seconds 300 --lo 1024 --hi 24576 --olo 64 --ohi 512
  drive.py burst --conc 8 --len 30000 --rounds 3    # C long prefills at once

Every request is logged as one JSON line to --log (epoch times).
"""

import argparse
import asyncio
import json
import math
import random
import time

import aiohttp

URL = "http://127.0.0.1:{port}/v1/completions"


async def one(session, args, plen, olen, rng, tag, logf):
    prompt = [rng.randrange(1000, 150000) for _ in range(plen)]
    body = {"model": "m", "prompt": prompt, "max_tokens": olen, "temperature": 0.8,
            "ignore_eos": True}
    t0 = time.time()
    status, err, usage = None, None, None
    try:
        async with session.post(URL.format(port=args.port), json=body,
                                timeout=aiohttp.ClientTimeout(total=3600)) as r:
            status = r.status
            js = await r.json(content_type=None)
            if r.status == 200:
                usage = js.get("usage")
            else:
                err = str(js)[:300]
    except Exception as e:  # server died etc.
        err = repr(e)[:300]
    rec = {"tag": tag, "plen": plen, "olen": olen, "t0": t0, "t1": time.time(),
           "status": status, "err": err, "usage": usage}
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ladder", "mix", "burst"])
    ap.add_argument("--port", type=int, default=18467)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lens", default="2048,4096,8192,16384,24576,32768,49152,61440")
    ap.add_argument("--out-tokens", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--lo", type=int, default=1024)
    ap.add_argument("--hi", type=int, default=24576)
    ap.add_argument("--olo", type=int, default=64)
    ap.add_argument("--ohi", type=int, default=512)
    ap.add_argument("--len", type=int, default=30000)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    logf = open(args.log, "a")
    conn = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=conn) as s:
        if args.mode == "ladder":
            for rep in range(args.repeat):
                for L in [int(x) for x in args.lens.split(",")]:
                    r = await one(s, args, L, args.out_tokens, rng, f"ladder{rep}", logf)
                    print(L, r["status"], round(r["t1"] - r["t0"], 2), r["err"] or "", flush=True)
                    if r["status"] != 200:
                        return
        elif args.mode == "burst":
            for k in range(args.rounds):
                rs = await asyncio.gather(*[one(s, args, args.len, args.out_tokens, rng,
                                                f"burst{k}", logf) for _ in range(args.conc)])
                print("round", k, [r["status"] for r in rs], flush=True)
                if any(r["status"] != 200 for r in rs):
                    return
        else:
            stop = time.time() + args.seconds
            done = {"n": 0, "bad": 0}

            async def worker(i):
                wr = random.Random(args.seed * 1000 + i)
                while time.time() < stop:
                    plen = int(math.exp(wr.uniform(math.log(args.lo), math.log(args.hi))))
                    olen = wr.randint(args.olo, args.ohi)
                    r = await one(s, args, plen, olen, wr, "mix", logf)
                    done["n"] += 1
                    if r["status"] != 200:
                        done["bad"] += 1
                        if r["status"] is None:  # connection error: server gone
                            return

            await asyncio.gather(*[worker(i) for i in range(args.conc)])
            print("mix done", done, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
