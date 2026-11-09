#!/usr/bin/env python3
"""E054 (c3): server-level sanity probe, statistical (the stack is not run-to-run deterministic).

The E051 session measured on a 4x4090 box that two runs of the same prompt on the SAME server differ already at
token 0 (logprob -4.436 vs -4.495), also with VLLM_BATCH_INVARIANT=1. So exact greedy parity cannot be a server
gate. The exactness gates are the kernel / layer A/Bs on saved inputs (indexer_ab.py, engram_wkv_ab.py). This
probe only checks that an arm stays inside the OLD arm's own run-to-run noise:

  run:      logprob_probe.py --url http://127.0.0.1:8100 --repeats 3 --json runs/<arm>/probe.json
            5 fixed prompts x R repeats, sequential (one request in flight), greedy, 128 tokens, top-5 logprobs,
            POST /reset_prefix_cache before each request (cold; needs VLLM_SERVER_DEV_MODE=1, serve_e054.sh sets it)
  compare:  logprob_probe.py --compare runs/old/probe.json runs/both/probe.json [runs/old2/probe.json ...]
            the first file is the reference arm; pairs of runs of the same prompt are grouped as
            within-reference (the noise floor), within-other, across (reference vs other)
            per pair: first token-id divergence, |d logprob| of token 0 (top-5 union), mean |d logprob| of the
            chosen tokens over the common prefix, top-5 overlap at token 0

Pass (pre-registered, per other arm): across medians inside the noise floor with slack, i.e.
  median first divergence (across) >= 0.5 x median first divergence (within reference), or >= 64 tokens;
  mean |d lp| over the prefix (across, median) <= 2 x the within-reference median + 0.01;
  token 0's argmax identical across arms on >= 4/5 prompts; no across |d lp(token 0)| > 0.5 nats.
A broken change (e.g. wkv shards out of order) fails all of these at once; numerical noise fails none.
"""

import argparse
import itertools
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    "Solve step by step: a train leaves at 3:40 pm and travels 210 km at 84 km/h. When does it arrive?",
    "Here is an ARC grid task.\nInput:\n0 0 1\n0 1 0\n1 0 0\nOutput:\n1 0 0\n0 1 0\n0 0 1\n"
    "Describe the transformation and apply it to:\n2 0 0\n0 0 3\n0 4 0\nOutput:\n",
    "def flood_fill(grid, r, c, color):\n    \"\"\"Fill the 4-connected region containing (r, c) with color.\"\"\"\n",
    "The three most important ideas in the theory of computation are",
    "Translate to French and explain any idiom: 'It is raining cats and dogs, so we will cross that bridge "
    "when we come to it.'",
]


def post(url, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def run(a):
    out = {"url": a.url, "model": a.model, "max_tokens": a.max_tokens, "repeats": a.repeats, "cold": a.cold,
           "prompts": PROMPTS, "runs": [], "t": time.time()}
    for rep in range(a.repeats):
        for i, prompt in enumerate(PROMPTS):
            if a.cold:
                try:
                    post(a.url, "/reset_prefix_cache")
                except Exception as exc:
                    sys.exit(f"/reset_prefix_cache failed ({exc}); start the server with VLLM_SERVER_DEV_MODE=1 "
                             "or pass --warm")
            r = post(a.url, "/v1/completions", {
                "model": a.model, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": 0, "seed": 0,
                "logprobs": 5, "return_token_ids": True})
            ch = r["choices"][0]
            lp = ch.get("logprobs") or {}
            rec = {"prompt": i, "rep": rep, "ids": ch.get("token_ids"), "text": ch.get("text", "")[:400],
                   "lps": lp.get("token_logprobs"), "top": lp.get("top_logprobs")}
            out["runs"].append(rec)
            print(json.dumps({"prompt": i, "rep": rep, "n": len(rec["ids"] or []),
                              "lp0": rec["lps"][0] if rec["lps"] else None}), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def pair_stats(x, y):
    ids_x, ids_y = x["ids"] or [], y["ids"] or []
    n = min(len(ids_x), len(ids_y))
    div = next((i for i in range(n) if ids_x[i] != ids_y[i]), n)
    tx, ty = (x["top"] or [{}])[0] or {}, (y["top"] or [{}])[0] or {}
    union = set(tx) | set(ty)
    d0 = max((abs(tx.get(t, -30.0) - ty.get(t, -30.0)) for t in union), default=0.0)
    common = range(min(div, len(x["lps"] or []), len(y["lps"] or [])))
    dmean = statistics.mean(abs(x["lps"][i] - y["lps"][i]) for i in common) if len(common) else 0.0
    am_x = max(tx, key=tx.get) if tx else None
    am_y = max(ty, key=ty.get) if ty else None
    return {"first_div": div, "d_lp0_top5": d0, "mean_d_lp_prefix": dmean,
            "top5_overlap0": len(set(tx) & set(ty)) / max(1, len(union)), "argmax0_same": am_x == am_y}


def summarize(pairs):
    if not pairs:
        return None
    col = lambda k: [p[k] for p in pairs]  # noqa: E731
    return {"pairs": len(pairs), "first_div_median": statistics.median(col("first_div")),
            "first_div_min": min(col("first_div")), "d_lp0_max": round(max(col("d_lp0_top5")), 4),
            "d_lp0_median": round(statistics.median(col("d_lp0_top5")), 4),
            "mean_d_lp_median": round(statistics.median(col("mean_d_lp_prefix")), 5),
            "top5_overlap0_min": round(min(col("top5_overlap0")), 3)}


def compare(files):
    arms = [json.load(open(f)) for f in files]
    ref = arms[0]
    by = lambda d, i: [r for r in d["runs"] if r["prompt"] == i]  # noqa: E731
    within_ref = [pair_stats(x, y) for i in range(len(PROMPTS)) for x, y in itertools.combinations(by(ref, i), 2)]
    floor = summarize(within_ref)
    print(json.dumps({"reference": files[0], "within_reference": floor}))
    ok_all = True
    for f, d in zip(files[1:], arms[1:]):
        within = [pair_stats(x, y) for i in range(len(PROMPTS)) for x, y in itertools.combinations(by(d, i), 2)]
        across = [pair_stats(x, y) for i in range(len(PROMPTS)) for x in by(ref, i) for y in by(d, i)]
        s = summarize(across)
        argmax_prompts = sum(all(pair_stats(x, y)["argmax0_same"] for x in by(ref, i) for y in by(d, i))
                             for i in range(len(PROMPTS)))
        checks = {
            "first_div": s["first_div_median"] >= min(64, 0.5 * floor["first_div_median"]),
            "mean_d_lp": s["mean_d_lp_median"] <= 2 * floor["mean_d_lp_median"] + 0.01,
            "argmax0_4of5": argmax_prompts >= 4,
            "d_lp0_le_0.5": s["d_lp0_max"] <= 0.5,
        }
        ok = all(checks.values())
        ok_all &= ok
        print(json.dumps({"arm": f, "within": summarize(within), "across": s, "argmax0_same_prompts": argmax_prompts,
                          "checks": checks, "pass": ok}))
    return ok_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="dsv41")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warm", dest="cold", action="store_false", help="do not reset the prefix cache")
    p.add_argument("--json")
    p.add_argument("--compare", nargs="+", metavar="PROBE_JSON")
    a = p.parse_args()
    if a.compare:
        sys.exit(0 if compare(a.compare) else 1)
    run(a)


if __name__ == "__main__":
    main()
